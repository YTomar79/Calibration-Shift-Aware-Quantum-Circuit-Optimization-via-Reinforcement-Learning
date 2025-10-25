import torch
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator, QasmSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit.circuit import Gate
from functools import lru_cache
import hashlib
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import RemoveResetInZeroState, GateDirection
from qiskit_aer.library import SaveStatevector
import json


# works on Qiskit 1.2.4
torch.set_default_dtype(torch.float32)

class QuantumCircuitEnvExpanded:
    def __init__(self, num_qubits=10, max_time_steps=40, noise_decay_factor=0.97, idle_noise_reduction=0.015,
                 max_steps_per_episode=200, hardware_topology=None, total_episodes=10001, seed=None, debug=False):
        self.device = torch.device('cpu')
        self.num_qubits = num_qubits # to add more variable circuits, i made sure to vary the # of qubits between 3 and 12
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.hardware_topology = hardware_topology
        self.rng = np.random.default_rng(seed)
        self.debug = debug

        #backend setup
        self.backend = Aer.get_backend('qasm_simulator')

        #gPU-compatible state variables
        self.qubit_usage = torch.zeros(self.num_qubits, device=self.device)
        self.qubit_age = torch.zeros(self.num_qubits, device=self.device)
        self.qubit_idle_time = torch.zeros(self.num_qubits, device=self.device)
        self.multi_qubit_usage = torch.zeros((self.num_qubits, self.num_qubits), device=self.device)
        self.qubit_connectivity = self._init_qubit_connectivity()
        self.noise_profile = self._init_noise_profile()

        self.gate_operations = torch.zeros((self.num_qubits, self.max_time_steps), device=self.device)
        self.path_usage_matrix = torch.zeros((self.num_qubits, self.num_qubits), device=self.device)
        self.current_step = 0
        self.error_rates = torch.zeros(self.num_qubits, device=self.device)
        self.coherence_times = torch.zeros(self.num_qubits, device=self.device)
        self._get_backend_properties()
        self.episode_count = 0
        self.current_episode = 0
        self.total_episodes = total_episodes

        #additional state tracking
        self.action_repeat_counter = {'swap': 0, 'gate_insertion': 0, 'multi_qubit_gate': 0}
        self.noise_decay_factor = noise_decay_factor
        self.idle_noise_reduction = idle_noise_reduction
        self.dynamic_depth = False
        self.num_layers = 1
        self.fidelity_reward_multiplier = 10
        self.deactivated_qubits = torch.zeros(self.num_qubits, dtype=torch.bool, device=self.device)
        self.noise_model = self._initialize_noise_model()
        self.fidelity_history = torch.zeros(self.max_time_steps, dtype=torch.float32, device=self.device)
        self.invalid_gate_count = 0
        self.successful_episodes = 0
        self.fidelity_before = 0.0
        self.fidelity_after = 0.0

        self.min_depth_threshold = max(2, int(0.25 * self.num_qubits))  # minimum depth dynamically set based on qubits
        self.consecutive_no_improvement_steps = 0  # tracks stagnation
        self.max_no_improvement_steps = 5  #maximum allowed steps without improvement
        self.previous_depth = None  # tracks the previous depth for stagnation checks


        if self.debug:
            print(f"Initialized environment on device: {self.device}")



    def _high_connectivity_topology(self):
        #create a fully connected topology excluding self-loops
        connectivity = torch.ones((self.num_qubits, self.num_qubits), device=self.device) - torch.eye(self.num_qubits, device=self.device)
        return connectivity


    def _low_connectivity_topology(self):
        #create a ring-like topology
        connectivity = torch.zeros((self.num_qubits, self.num_qubits), device=self.device)
        indices = torch.arange(self.num_qubits - 1, device=self.device)
        connectivity[indices, indices + 1] = 1
        connectivity[indices + 1, indices] = 1
        return connectivity


    def _clustered_topology(self, num_clusters=4):
        connectivity = torch.zeros((self.num_qubits, self.num_qubits), device=self.device)
        cluster_size = self.num_qubits // num_clusters
        for cluster in range(num_clusters):
            start, end = cluster * cluster_size, (cluster + 1) * cluster_size
            #fully connect each cluster
            for i in range(start, end):
                connectivity[i, start:end] = 1
            #connect adjacent clusters
            if cluster < num_clusters - 1:
                connectivity[end - 1, end] = connectivity[end, end - 1] = 1
        return connectivity





    def _initialize_noise_model(self, calibration_file="ibm_torino_calibration.json"):
        # load calibration data from JSON file
        with open(calibration_file, 'r') as f:
            properties = json.load(f)

        noise_model = NoiseModel()

        max_iterations = 5  # add this line
        for qubit_index, qubit_props in enumerate(properties['qubits']):
            t1_time = next((prop['value'] for prop in qubit_props if prop['name'] == 'T1'), None)
            t2_time = next((prop['value'] for prop in qubit_props if prop['name'] == 'T2'), None)
            gate_time = 50e-9  # default gate time for single-qubit gates

            if t1_time and t2_time:

                t1_time *= 0.8  # reduce T1
                t2_time *= 0.7  # reduce T2
                iterations = 0
                while t2_time > 2 * t1_time and iterations < max_iterations:
                    t2_time = min(t2_time, 2 * t1_time * 0.99)  
                    iterations += 1

                if iterations == max_iterations:
                    print(f"Warning: Max iterations reached for qubit {qubit_index}. Final T2: {t2_time}, T1: {t1_time}")

                thermal_error = thermal_relaxation_error(t1_time, t2_time, gate_time)
                noise_model.add_quantum_error(thermal_error, ['u3', 'u2', 'u1'], [qubit_index])            
            readout_err_prob = next((prop['value'] for prop in qubit_props if prop['name'] == 'readout_error'), None)
            if readout_err_prob is not None:
                readout_err_prob = min(1.0, readout_err_prob * 1.6)
                readout_error = ReadoutError([[1 - readout_err_prob, readout_err_prob],
                                              [readout_err_prob, 1 - readout_err_prob]])
                noise_model.add_readout_error(readout_error, [qubit_index])


        for gate in properties['gates']:
            if gate['gate'] == 'cx':  # cx gate (CNOT)
                qubits = gate['qubits']
                cx_error_rate = next((param['value'] for param in gate['parameters'] if param['name'] == 'gate_error'), None)
                if cx_error_rate is not None:
                    # scale up two qubit depolarizing error rate
                    cx_error_rate = min(1.0, cx_error_rate * 1.6)
                    dep_error = depolarizing_error(cx_error_rate, 2)
                    noise_model.add_quantum_error(dep_error, ['cx'], qubits)

        # add idle errors for qubits not actively operated on
        idle_gate_time = 75e-9  # example idle time
        for qubit_index in range(len(properties['qubits'])):
            if t1_time and t2_time:
                idle_error = thermal_relaxation_error(t1_time, t2_time, idle_gate_time)
                noise_model.add_quantum_error(idle_error, ['id'], [qubit_index])

        return noise_model




    def _init_qubit_connectivity(self):
        topology_type = self.rng.choice(['high', 'low', 'clustered'], p=[0.3, 0.5, 0.2])
        if topology_type == 'high':
            return self._high_connectivity_topology()
        elif topology_type == 'low':
            return self._low_connectivity_topology()
        else:
            return self._clustered_topology()

    def _init_noise_profile(self):
        noise_profile = np.random.uniform(0.01, 0.1, size=(self.num_qubits, self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if np.random.rand() < 0.3:
                    noise_profile[i][j] *= 1.5
                    noise_profile[j][i] *= 1.5
        return torch.tensor(noise_profile, device=self.device)

    def _get_backend_properties(self):

        if not hasattr(self.backend, "properties") or self.backend.properties() is None:
            #simulator case: mock the properties
            print("Backend is a simulator or lacks properties; mocking properties.")
            self.error_rates = torch.zeros(self.num_qubits, device=self.device)  # no readout error
            self.coherence_times = torch.full((self.num_qubits,), 100e-6, device=self.device)  # mock T1 times
        else:
            #real backend: fetch properties
            backend_properties = self.backend.properties()

            def get_numeric_value(qubit_property, key):
                value = qubit_property.get(key)
                return float(value[0] if isinstance(value, tuple) else value)

            self.error_rates = torch.tensor(
                [get_numeric_value(backend_properties.qubit_property(q), 'readout_error') for q in range(self.num_qubits)],
                device=self.device
            )
            self.coherence_times = torch.tensor(
                [get_numeric_value(backend_properties.qubit_property(q), 'T1') for q in range(self.num_qubits)],
                device=self.device
            )




    def reset(self, episode=0):

        def validate_tensor(tensor, name, expected_dtype=torch.float32, expected_device=self.device):
            assert tensor.dtype == expected_dtype, f"{name} has incorrect dtype: {tensor.dtype}, expected {expected_dtype}"
            assert tensor.device == expected_device, f"{name} is on {tensor.device}, expected {expected_device}"

        #reset and randomize initial circuit properties
        self.current_circuit = None
        self.depth = 0

        # randomize noise profile (error rates) and coherence times
        self.error_rates = torch.tensor(
            np.random.uniform(0.02, 0.1, size=(self.num_qubits,)), dtype=torch.float32, device=self.device
        )

        self.coherence_times = torch.tensor(
            np.random.uniform(20, 100, size=(self.num_qubits,)), dtype=torch.float32, device=self.device
        )

        # generate a new complex circuit
        self.current_circuit = self._generate_complex_circuit()

        # reset episode metrics
        self.logical_circuit = torch.randint(0, 2, (self.num_qubits, self.max_time_steps), dtype=torch.int32, device=self.device)
        self.gate_operations = torch.zeros((self.num_qubits, self.max_time_steps), device=self.device)
        self.path_usage_matrix = torch.zeros((self.num_qubits, self.num_qubits), device=self.device)
        self.qubit_usage = torch.zeros(self.num_qubits, device=self.device)
        self.multi_qubit_usage = torch.zeros((self.num_qubits, self.num_qubits), device=self.device)
        self.fidelity_history = torch.zeros(self.max_time_steps, dtype=torch.float32, device=self.device)
        self.deactivated_qubits = torch.zeros(self.num_qubits, dtype=torch.bool, device=self.device)

        # initialize/reset other attributes
        self.current_step = 0
        self.current_episode = episode


        self.fidelity_before = 0.0
        self.fidelity_after = 0.0

        self.invalid_gate_count = 0
        self.episode_count += 1
        self.consecutive_no_improvement_steps = 0
        self.qubit_connectivity = torch.tensor(self._init_qubit_connectivity(), device=self.device).clone().detach()
        self.noise_profile = self._init_noise_profile()

        # validate randomization results
        validate_tensor(self.qubit_connectivity, "Qubit Connectivity")
        validate_tensor(self.error_rates, "Error Rates", expected_dtype=torch.float32)
        validate_tensor(self.coherence_times, "Coherence Times", expected_dtype=torch.float32)

        # update metrics based on the generated circuit
        self._update_metrics_from_circuit()

        # debugging!
        if self.debug:
            print(f"Error Rates:\n{self.error_rates}")
            print(f"Coherence Times:\n{self.coherence_times}")
            print(f"Connectivity:\n{self.qubit_connectivity}")

        return self._get_observation()

    def step(self, action):

        try:
            #decode action and apply it
            action_type, details = self._decode_action(action)
            self._apply_action(action_type, details)

            #calculate fidelity before the action
            fidelity_before = self._calculate_fidelity()
            self.fidelity_before = fidelity_before

            #apply dynamic noise and connectivity updates
            self._apply_dynamic_noise_feedback(action_type=action_type, details=details)
            self._evolve_connectivity()


            depth = self.current_circuit.depth() if self.current_circuit else 0

            #calculate fidelity after the action
            fidelity_after = self._calculate_fidelity()
            self.fidelity_after = fidelity_after

            #fidelity improvement
            fidelity_improvement = torch.clamp(torch.tensor(fidelity_after - fidelity_before), min=-0.5, max=0.5)

            #calculate rewards
            reward = self._calculate_action_reward(action_type, details, fidelity_improvement)

            #dynamic plateau threshold based on episode progression
            plateau_threshold = max(0.001, 0.01 * (1 - (self.current_episode / self.total_episodes)))

            #termination criteria
            done = False
            min_depth_threshold = self._calculate_dynamic_min_depth()
            fidelity_threshold = self._calculate_dynamic_fidelity_threshold()
            """
            if self.current_episode > 6500:
                #check if optimization criteria are met
                if fidelity_after >= fidelity_threshold and depth <= min_depth_threshold:

                    print(f"Optimization complete at step {self.current_step}: Fidelity = {fidelity_after:.4f}, Depth = {depth}.")
                    self.successful_episodes += 1

                    #calculate scaling factor based on remaining steps
                    remaining_steps_ratio = max(0.1, (300 - self.current_step) / 300)

                    #adjust reward dynamically based on remaining steps
                    goal_reward_base = 45.0 + 0.4 * ((self.num_qubits + depth) / max(1, self.max_time_steps))
                    scaled_goal_reward = goal_reward_base * 1.5 * (1 + remaining_steps_ratio)  # Add up to 50% boost

                    #ensure the reward remains significant
                    reward += scaled_goal_reward
                    self.num_qubits = random.randint(3, 13)
                    done = True
            """


            # fidelity plateau detection, adding later once the agent understands that this is fidelity driven optimization

            if not done:     # avoid overriding optimization completion termination
                recent_fidelity_gains = self.fidelity_history[max(0, self.current_step - 5):self.current_step]
                if len(recent_fidelity_gains) > 4 and max(recent_fidelity_gains) < plateau_threshold and self.current_step > 10:
                    print(f"Fidelity plateau detected at step {self.current_step}. Ending episode.")

                    done = True
                    self.opti = True


            # dynamic step limit (fallback termination; remove later)
            if self.current_episode > 9000:
                if not done and self.current_step >= 200:
                    print(f"Maximum steps reached at step {self.current_step}. Ending episode.")

                    done = True
            else:
                if not done and self.current_step >= 200:
                    print(f"Maximum steps reached at step {self.current_step}. Ending episode.")

                    done = True


            # debugging information
            if self.debug:
                print(f"Step {self.current_step}: Fidelity = {fidelity_after:.4f}, Depth = {depth}, Reward = {reward:.4f}")
                print(f"Successful episodes: {self.successful_episodes}")

            # update fidelity history
            if len(self.fidelity_history) < self.max_time_steps:
                self.fidelity_history[self.current_step] = fidelity_after

            # update step count and depth
            self.previous_depth = depth
            self.current_step += 1

            return self._get_observation(), reward, done, {}

        except Exception as e:
            print(f"[ERROR] Step execution failed: {e}")
            raise







    def _generate_complex_circuit(self):

        from qiskit import QuantumCircuit
        num_qubits = self.num_qubits
        target_depth = 0

        # adjust target depth based on coherence times
        avg_coherence_time = torch.mean(self.coherence_times).item()
        if avg_coherence_time < 30.0:  # threshold for low coherence times
            target_depth = max(7, target_depth // 2)  # reduce depth to avoid excessive penalties
        else:
            target_depth = max(5, min(30, self.depth))  # ensure a minimum depth of 5


        circuit = QuantumCircuit(num_qubits)

        # dynamically build the circuit
        for layer in range(target_depth):
            self._apply_layer(circuit)
            if layer % 2 == 0:
                self._apply_entanglement(circuit)

        print(f"Generated stress-test circuit: Depth = {target_depth}, Qubits = {num_qubits}")
        self.current_circuit = circuit
        self._update_metrics_from_circuit()
        return circuit




    def _apply_layer(self, qc):
        gate_types = np.random.choice(
            ['h', 'rx', 'rz', 'cx', 'swap', 'ry', 's', 't'], size=self.num_qubits
        )
        for qubit, gate_type in enumerate(gate_types):
            if gate_type == 'h':
                qc.h(qubit)
            elif gate_type == 'rx':
                qc.rx(np.random.uniform(0, np.pi), qubit)
            elif gate_type == 'rz':
                qc.rz(np.random.uniform(0, 2 * np.pi), qubit)
            elif gate_type == 'ry':
                qc.ry(np.random.uniform(0, np.pi / 2), qubit)
            elif gate_type == 's':
                qc.s(qubit)
            elif gate_type == 't':
                qc.t(qubit)
            elif gate_type == 'cx':
                target_qubit = (qubit + 1) % self.num_qubits
                # validate indices before accessing qubit_connectivity
                if qubit < self.qubit_connectivity.size(0) and target_qubit < self.qubit_connectivity.size(1):
                    if self.qubit_connectivity[qubit, target_qubit] > 0.3:  # check connectivity
                        qc.cx(qubit, target_qubit)
                        self.multi_qubit_usage[qubit, target_qubit] += 1
                        self.path_usage_matrix[qubit, target_qubit] += 1
                        self.path_usage_matrix[target_qubit, qubit] += 1
            elif gate_type == 'swap':
                target_qubit = (qubit + 1) % self.num_qubits
                # validate indices before accessing qubit_connectivity
                if qubit < self.qubit_connectivity.size(0) and target_qubit < self.qubit_connectivity.size(1):
                    if self.qubit_connectivity[qubit, target_qubit] > 0.3:  # check connectivity
                        qc.swap(qubit, target_qubit)
                        self.multi_qubit_usage[qubit, target_qubit] += 1
                        self.path_usage_matrix[qubit, target_qubit] += 1
                        self.path_usage_matrix[target_qubit, qubit] += 1



    def _apply_entanglement(self, qc):

        for i in range(self.num_qubits - 1):
            target_qubit = (i + 1) % self.num_qubits
            if self.qubit_connectivity[i, target_qubit] > 0.5:  # prioritize strong connections
                qc.cz(i, target_qubit)


    def _calculate_dynamic_fidelity_threshold(self):

        # baseline fidelity threshold
        base_threshold = 0.96

        # episode progress influence
        progress_factor = 0.02 * (self.current_step / self.max_time_steps)

        # historical fidelity trend
        if self.current_step > 0:
            recent_weight = 0.6
            historical_weight = 0.4
            recent_avg_fidelity = self.fidelity_history[max(0, self.current_step - 50):self.current_step].mean().item()
            avg_fidelity = (recent_weight * recent_avg_fidelity + historical_weight * self.fidelity_history[:self.current_step].mean().item())
        else:
            avg_fidelity = 0.98
        historical_factor = max(0, 0.02 * (avg_fidelity - base_threshold))

        # complexity influence
        complexity_factor = 0.02 * self._calculate_circuit_complexity()

        # adjust complexity factor for coherence trends
        avg_coherence = self.coherence_times.mean().item()
        coherence_factor = 0.01 if avg_coherence < 30 else 0.02
        complexity_factor *= coherence_factor

        # aggregate threshold
        fidelity_threshold = base_threshold + progress_factor + historical_factor + complexity_factor
        fidelity_threshold = min(fidelity_threshold, 0.995)

        if self.debug:
            print(f"Dynamic Fidelity Threshold: {fidelity_threshold:.3f} "
                  f"(Base: {base_threshold}, Progress: {progress_factor:.3f}, "
                  f"History: {historical_factor:.3f}, Complexity: {complexity_factor:.3f})")

        return fidelity_threshold





    def _update_metrics_from_circuit(self):

        if not self.current_circuit:
            raise ValueError("No circuit exists to update metrics from.")

        # update depth
        self.depth = self.current_circuit.depth()

        # reset metrics
        self.multi_qubit_usage.fill_(0)
        self.path_usage_matrix.fill_(0)

        # update metrics based on gate operations in the circuit
        for gate in self.current_circuit.data:
            if gate[0].name in ['cx', 'cz', 'swap']:
                # find indices of the control and target qubits
                control, target = [self.current_circuit.qubits.index(qubit) for qubit in gate[1]]
                self.multi_qubit_usage[control, target] += 1
                self.path_usage_matrix[control, target] += 1

        # debugging output
        if self.debug:
            print(f"[METRICS UPDATED] Depth: {self.depth}")
            print(f"Multi-Qubit Usage:\n{self.multi_qubit_usage}")
            print(f"Path Usage Matrix:\n{self.path_usage_matrix}")



    def _calculate_dynamic_max_depth(self):
        # base max depth proportional to the number of qubits
        base_max_depth = self.num_qubits * 10

        # circuit complexity factor (normalized and capped)
        complexity_factor = min(self._calculate_circuit_complexity(), 3.0) / 3.0
        layer_factor = np.log2(self.num_layers + 1)
        avg_coherence_time = self.coherence_times.mean().item()
        coherence_factor = 0.8 if avg_coherence_time < 30 else 1.0
        progress_ratio = self.current_step / self.max_time_steps
        difficulty_factor = max(0.6, np.exp(-self.episode_count / 5000))
        min_depth_threshold = self._calculate_dynamic_min_depth()


        scaled_max_depth = base_max_depth * complexity_factor * layer_factor * coherence_factor * difficulty_factor


        min_gap = self._calculate_dynamic_min_gap()
        final_max_depth = max(scaled_max_depth, min_depth_threshold + min_gap)

        if self.debug:
            print(f"Dynamic max depth calculated: {final_max_depth:.2f} "
                  f"(Base: {base_max_depth}, Complexity: {complexity_factor:.2f}, "
                  f"Layer Factor: {layer_factor:.2f}, Coherence Factor: {coherence_factor:.2f}, "
                  f"Difficulty: {difficulty_factor:.2f}, Min Depth: {min_depth_threshold})")

        return int(final_max_depth)

    def _calculate_dynamic_min_depth(self):

        # baseline threshold starts at 75% of the number of qubits
        base_min_depth = max(10, int(self.num_qubits * 1.0))

        # complexity factor scales with how intricate the circuit is
        complexity_factor = self._calculate_circuit_complexity()  # circuit complexity factor

        # episode progress ratio
        progress_ratio = self.current_step / self.max_time_steps

        # fidelity trend: how well the agent has maintained fidelity
        fidelity_trend = (
            self.fidelity_history[:self.current_step].mean().item()
            if self.current_step > 0
            else 1.0
        )


        # dynamic scaling based on complexity, fidelity, and progress
        # progress rewards more aggressive depth minimization; fidelity rewards better results.
        # adjusted scaling factor
        scaling_factor = (
            1.0
            + (0.3 * min(progress_ratio, 0.8))  # cap progression impact
            - (0.3 * (1.0 - fidelity_trend))  # sightly stronger fidelity trend influence
            + (0.35 * complexity_factor)  # increase complexity factor's weight
        )



        # ensure a realistic yet challenging depth threshold
        dynamic_min_depth = max(int(base_min_depth * scaling_factor), 10)  # Ensure it's >= 5

        if self.debug:
            print(
                f"Dynamic min depth calculated: {dynamic_min_depth} "
                f"(Base: {base_min_depth}, Complexity: {complexity_factor:.2f}, "
                f"Progress: {progress_ratio:.2f}, Fidelity Trend: {fidelity_trend:.2f})"
            )

        return dynamic_min_depth


    def _calculate_dynamic_min_gap(self):

        base_min_gap = 25  # initial minimum gap
        complexity_factor = self._calculate_circuit_complexity()  # circuit complexity factor
        episode_scaling = max(1.0 - (self.episode_count / 10000), 0.5)  # scale down over episodes


        if self.current_step > 0:
            fidelity_trend = torch.mean(self.fidelity_history[:self.current_step]).item()
        else:
            fidelity_trend = 1.0

        progress_ratio = self.current_step / self.max_time_steps


        dynamic_scaling = 1.0 + (0.4 * complexity_factor) - (0.15 * (1.0 - fidelity_trend)) + (0.6 * progress_ratio)

        scaled_min_gap = base_min_gap * dynamic_scaling * episode_scaling

        # minimum gap of 3
        final_min_gap = max(int(scaled_min_gap), 5)

        if self.debug:
            print(
                f"Dynamic min gap calculated: {final_min_gap} "
                f"(Base: {base_min_gap}, Complexity: {complexity_factor:.2f}, "
                f"Fidelity: {fidelity_trend:.2f}, Progress: {progress_ratio:.2f})"
            )

        return final_min_gap


    def _calculate_dynamic_depth_limit(self):


        qubit_factor = self.num_qubits * 4  #allowing more depth for more qubits
        layer_factor = np.log2(self.num_layers + 1)  # logorithmic scale for increasing layers...

        # base depth limit calculation
        max_depth = int(qubit_factor * layer_factor)


        system_resource_factor = 1.25

        max_depth *= system_resource_factor


        max_depth = min(max_depth, 125)

        return max_depth


    def _calculate_circuit_complexity(self):

        # Base factors for qubits, layers, and gates
        qubit_factor = self.num_qubits * 0.2  # Higher weight for qubits (more qubits = more complexity)
        layer_factor = self.num_layers * 0.3  # Higher weight for layers
        gate_factor = sum(
            count * weight for gate, weight in {"cx": 0.4, "cz": 0.3, "h": 0.1}.items()
            if (count := self.current_circuit.count_ops().get(gate, 0))
        )

        # Connectivity factor: penalizes sparse connectivity
        connectivity_factor = 1.0 - torch.mean(self.qubit_connectivity).item()  # Sparse connectivity adds complexity

        # Aggregate complexity score
        complexity_score = qubit_factor + layer_factor + gate_factor + connectivity_factor

        # Normalize and cap complexity score to prevent excessive scaling
        complexity_score = min(complexity_score, 3.0)  # Cap at 3 for extremely complex circuits

        if self.debug:
            print(f"Complexity Score: {complexity_score:.2f} "
                  f"(Qubits: {qubit_factor:.2f}, Layers: {layer_factor:.2f}, "
                  f"Gates: {gate_factor:.2f}, Connectivity: {connectivity_factor:.2f})")

        return complexity_score




    def _apply_dynamic_noise_feedback(self, action_type=None, details=None):


        for qubit in range(self.num_qubits):
            if self.qubit_usage[qubit] > 0:
                # increase error rates slightly for heavily used qubits
                self.error_rates[qubit] += self.rng.uniform(0.005, 0.025) * self.qubit_usage[qubit]

        # localized penalties for specific actions
        if action_type in ['swap', 'multi_qubit_gate'] and details:
            for qubit in details:
                if qubit < self.num_qubits:  # ensure valid qubit index
                    self.error_rates[qubit] += 0.01  # additional localized penalty

        # calculate path usage penalty
        path_usage_penalty = torch.sum(self.path_usage_matrix ** 2) * 0.02

        # apply decay to coherence times
        noise_decay = torch.tensor(
            self.rng.uniform(0.5, 1.5, size=self.num_qubits),
            dtype=torch.float32,
            device=self.device
        )

        # ensure coherence_times is a PyTorch tensor!
        self.coherence_times = torch.tensor(
            self.coherence_times, dtype=torch.float32, device=self.device
        )

        # apply coherence time decay and path usage penalties
        self.coherence_times -= noise_decay * (self.qubit_usage > 0).float()
        self.coherence_times -= path_usage_penalty  #apply path usage penalty to coherence times

        # ensure error_rates is a PyTorch tensor
        self.error_rates = torch.tensor(
            self.error_rates, dtype=torch.float32, device=self.device
        )

        #ensure error rates and coherence times remain within bounds
        self.error_rates = torch.clamp(self.error_rates, 0.01, 0.1)
        self.coherence_times = torch.clamp(self.coherence_times, 10.0, 100.0)

        if self.debug:
            print(f"Dynamic Noise Feedback Applied:")
            print(f"Error Rates:\n{self.error_rates}")
            print(f"Coherence Times:\n{self.coherence_times}")








    def _get_observation(self):
        # convert all NumPy arrays to PyTorch tensors
        usage_normalized = torch.tensor(self.qubit_usage / self.max_steps_per_episode, device=self.device, dtype=torch.float32)
        error_normalized = torch.tensor(self.error_rates / self.error_rates.max(), device=self.device, dtype=torch.float32)
        coherence_normalized = torch.tensor(self.coherence_times / self.coherence_times.max(), device=self.device, dtype=torch.float32)
        connectivity_flattened = torch.tensor(self.qubit_connectivity.flatten(), device=self.device, dtype=torch.float32)
        noise_profile_flattened = torch.tensor(self.noise_profile.flatten(), device=self.device, dtype=torch.float32)
        path_usage_normalized = torch.tensor(self.path_usage_matrix.flatten(), device=self.device, dtype=torch.float32)

        # include circuit depth as a normalized feature
        if hasattr(self, 'current_circuit') and self.current_circuit is not None:
            depth_normalized = self.current_circuit.depth() / self.max_time_steps
            depth_feature = torch.tensor([depth_normalized], device=self.device, dtype=torch.float32)
        else:
            # if no circuit exists yet, default depth normalization to 0
            depth_feature = torch.tensor([0.0], device=self.device, dtype=torch.float32)

        # concatenate all tensors
        return torch.cat([
            usage_normalized,
            error_normalized,
            coherence_normalized,
            connectivity_flattened,
            noise_profile_flattened,
            path_usage_normalized,
            depth_feature
        ])




    def _apply_idle_noise_reduction(self):
        idle_mask = self.qubit_idle_time > 3
        reduction_factor = 1 - self.idle_noise_reduction
        self.noise_profile[idle_mask, :] *= reduction_factor
        self.noise_profile[:, idle_mask] *= reduction_factor


    def _apply_noise_decay(self, action_type, details):
        self.noise_profile *= self.noise_decay_factor
        if isinstance(details, int):
            details = [details]
        if action_type in self.action_repeat_counter:
            self.action_repeat_counter[action_type] += 1
            if self.action_repeat_counter[action_type] > 5:
                noise_penalty = 1.02
                for qubit in details:
                    self.noise_profile[qubit, :] *= noise_penalty
                    self.noise_profile[:, qubit] *= noise_penalty


    @lru_cache(maxsize=128)

    def _transpile_cached(self, qc_qasm):
        backend_info = str(self.backend.configuration())
        cache_key = hashlib.md5((qc_qasm + backend_info).encode('utf-8')).hexdigest()
        qc = QuantumCircuit.from_qasm_str(qc_qasm)  # reconstruct circuit from QASM

        return self._optimize_transpilation(qc)


    def _optimize_transpilation(self, circuit):
        pass_manager = PassManager()
        pass_manager.append(RemoveResetInZeroState())
        if self.backend.configuration().coupling_map:
            coupling_map = CouplingMap(self.backend.configuration().coupling_map)
            pass_manager.append(GateDirection(coupling_map))
        transpiled_circuit = pass_manager.run(circuit)
        print(f"Circuit depth after optimization: {transpiled_circuit.depth()}")
        return transpiled_circuit


    def _apply_real_hardware_feedback(self):
        circuit = self._create_circuit()
        transpiled_circuit = self._optimize_transpilation(circuit)

        print("Executing circuit on qasm_simulator with noise model...")
        try:
            job = self.backend.run(transpiled_circuit, shots=1024)
            result = job.result()

            if result.get_counts():
                counts = result.get_counts()
                print(f"Simulation complete. Counts: {counts}")
                self._update_error_and_coherence(counts)
            else:
                print("Warning: No counts available for experiment.")
        except Exception as e:
            print(f"Error during simulation: {e}")
            return

        decay_rate = 0.99  # coherence decay factor per step
        fluctuation = torch.tensor(
            self.rng.uniform(-0.005, 0.005, size=self.error_rates.shape), device=self.device
        )

        active_qubits = ~self.deactivated_qubits
        self.coherence_times[active_qubits] *= decay_rate
        self.error_rates[active_qubits] = torch.clamp(
            self.error_rates[active_qubits] + fluctuation[active_qubits], 0.01, 0.1
        )

        deactivate_mask = (self.coherence_times < 30) | (self.error_rates > 0.08)
        self.deactivated_qubits |= deactivate_mask
        self.coherence_times[self.deactivated_qubits] = torch.clamp(
            self.coherence_times[self.deactivated_qubits] + 1, max=100
        )
        reactivation_mask = (self.coherence_times > 60) & self.deactivated_qubits
        self.deactivated_qubits[reactivation_mask] = False

        if self.debug:
            print(f"Updated coherence times: {self.coherence_times}")
            print(f"Updated error rates: {self.error_rates}")




    def _is_logical_swap_effective(self, qubit_a, qubit_b):

        return (self.coherence_times[qubit_a] > self.error_rates[qubit_a]) and \
               (self.coherence_times[qubit_b] > self.error_rates[qubit_b])

    def _apply_rotation_reward(self, details, rotation_type):
        qubit = details
        base_reward = 0.05
        improvement = self._calculate_improvement(qubit, qubit) * (0.1 if rotation_type == 'x' else 0.15)
        return base_reward + improvement


    def _adjust_connectivity_reward(self, details):

        qubit_a, qubit_b = details
        reward = 0
        if self.qubit_connectivity[qubit_a, qubit_b] == 0:
            self.qubit_connectivity[qubit_a, qubit_b] = 1
            self.qubit_connectivity[qubit_b, qubit_a] = 1
            reward += 0.1
        else:
            reward -= 0.1
        reward = torch.tensor(reward, device=self.device)
        return torch.clamp(reward, min=-1.5, max=1.5).item()



    def _calculate_action_reward(self, action_type, details, fidelity_improvement):
        reward = 0.0


        fidelity_term = fidelity_improvement ** 1.5 * 50  

        if fidelity_improvement < 0:
            fidelity_term = fidelity_improvement * 20

        reward += fidelity_term

        progress_ratio = self.current_step / self.max_time_steps
        early_boost_weight = 1.0 - progress_ratio
        if fidelity_improvement > 0:
            reward += early_boost_weight * fidelity_improvement * 5  # gentle bonus

        depth = self.current_circuit.depth() if self.current_circuit else 0
        adjusted_max_depth = self._calculate_dynamic_max_depth()
        # encourage staying under a certain fraction of the dynamic max depth.
        if depth < 0.8 * adjusted_max_depth:
            reward += 0.05  # modest reward for efficiency
        else:
            # penalize exceeding the target.
            reward -= 0.02 * (depth - 0.8 * adjusted_max_depth)



        avg_error_rate = torch.mean(self.error_rates).item()
        avg_coherence_time = torch.mean(self.coherence_times).item()
        noise_penalty = 0.05 * torch.mean(self.error_rates * self.qubit_usage)
        noise_penalty += torch.mean(1 / (self.coherence_times + 1e-5))
        noise_reward = 0.2 * (1 - avg_error_rate)
        noise_reward += 0.1 * (avg_coherence_time / max(1, torch.max(self.coherence_times).item()))
        reward += (noise_reward - noise_penalty)

        action_rewards = {
            'swap': 0.1,
            'logical_swap': 0.2,
            'reorder': 0.1,
            'rotation_z': 0.15,
            'rotation_x': 0.15,
            'error_correction': 0.4,
            'adjust_connectivity': 0.2,
            'gate_insertion': 0.2,
            'multi_qubit_gate': 0.2,
            'reset': 0.1,
            'gate_cancellation': 0.15,  #moderate reward as it directly reduces circuit depth
            'topology_aware_swap': 0.2,  #higher reward due to connectivity optimization
            'adaptive_layer_management': 0.25,  #significant reward for successful layer reduction
            'parameterized_optimization': 0.15,  #moderate reward for parameter tuning
            'noise_aware_scheduling': 0.3
        }
        action_bonus = action_rewards.get(action_type, 0.0)
        #if fidelity is stable or improving, reward the action slightly more.
        if fidelity_improvement >= 0:
            action_bonus *= 1.25

        reward += action_bonus

        # penalize repeated invalid gates. important because it prevent fake optimization.
        invalid_gate_penalty = 0.01 * self.invalid_gate_count
        reward -= invalid_gate_penalty


        reward = max(reward, -2.0)

        if self.debug:
            print(f"Action: {action_type}, Fidelity Improvement: {fidelity_improvement:.4f}")
            print(f"Fidelity Term: {fidelity_term:.4f}")
            print(f"Depth: {depth}, Depth Penalty/Reward Calculation Adjusted: {reward:.4f}")
            print(f"Swaps: {num_swaps}, Multi-Qubit Ops: {num_multi_qubit_ops}")
            print(f"Noise Reward: {noise_reward:.4f}, Noise Penalty: {noise_penalty:.4f}")
            print(f"Action Bonus: {action_bonus:.4f}")
            print(f"Invalid Gate Penalty: {invalid_gate_penalty:.4f}, Total Reward: {reward:.4f}")

        return reward


    def _cancel_gates(self):
        if not self.current_circuit:
            return 0

        original_depth = self.current_circuit.depth()

        #create a new circuit for the optimized version
        optimized_circuit = QuantumCircuit(self.num_qubits)

        #iterate through the gates and cancel where possible
        skip_next = False
        for i, instruction in enumerate(self.current_circuit.data):
            if skip_next:
                skip_next = False
                continue

            if i < len(self.current_circuit.data) - 1:
                next_instruction = self.current_circuit.data[i+1]

                # check for adjacent inverse gates
                if (instruction[0].name == next_instruction[0].name and
                    instruction[1] == next_instruction[1] and
                    instruction[0].name in ['x', 'y', 'z', 'h']):
                    skip_next = True
                    continue

            # if not cancelled, add to the optimized circuit
            optimized_circuit.append(instruction[0], instruction[1])

        # update the current circuit
        self.current_circuit = optimized_circuit

        # calculate the depth reduction
        depth_reduction = original_depth - self.current_circuit.depth()

        return depth_reduction

    def _topology_aware_swap(self, qubit_a, qubit_b):
        if not self.current_circuit:
            return 0

        # check if the swap is necessary based on topology
        if self.qubit_connectivity[qubit_a, qubit_b] > 0.5:
            return 0  # no need to swap, already well-connected

        # find the best-connected neighbor for each qubit
        best_neighbor_a = max(range(self.num_qubits), key=lambda q: self.qubit_connectivity[qubit_a, q])
        best_neighbor_b = max(range(self.num_qubits), key=lambda q: self.qubit_connectivity[qubit_b, q])

        # perform the swap that improves connectivity the most
        if self.qubit_connectivity[qubit_a, best_neighbor_b] > self.qubit_connectivity[qubit_b, best_neighbor_a]:
            if qubit_a != best_neighbor_b:  # add this check
                self.current_circuit.swap(qubit_a, best_neighbor_b)
                return 1
        else:
            if qubit_b != best_neighbor_a:  # add this check
                self.current_circuit.swap(qubit_b, best_neighbor_a)
                return 1

        return 0  # no swap performed

    def _adaptive_layer_management(self):
        if not self.current_circuit or len(self.current_circuit.data) == 0:
            print("Warning: Cannot perform adaptive layer management on an empty circuit.")
            return 0

        original_depth = self.current_circuit.depth()
        fidelity_before = self.fidelity_before

        # only try to remove the last layer if there are enough gates
        if len(self.current_circuit.data) >= self.num_qubits:
            last_layer = self.current_circuit.data[-self.num_qubits:]
            self.current_circuit.data = self.current_circuit.data[:-self.num_qubits]

            fidelity_after = self._calculate_fidelity()

            if fidelity_after < fidelity_before * 0.95:  # allow up to 5% fidelity drop
                # restore the layer if fidelity dropped too much
                self.current_circuit.data.extend(last_layer)
                return 0
            else:
                return original_depth - self.current_circuit.depth()
        else:
            print("Warning: Circuit too shallow for adaptive layer management.")
            return 0



    def _optimize_parameters(self):
        if not self.current_circuit:
            return 0

        fidelity_before = self.fidelity_before
        improvement = 0

        for i, instruction in enumerate(self.current_circuit.data):
            if instruction[0].name in ['rx', 'ry', 'rz']:
                original_param = instruction[0].params[0]

                # try small perturbations
                for delta in [-0.1, 0.1]:
                    new_param = original_param + delta
                    self.current_circuit.data[i][0].params[0] = new_param

                    fidelity_after = self._calculate_fidelity()

                    if fidelity_after > fidelity_before:
                        fidelity_before = fidelity_after
                        improvement += 1
                    else:
                        self.current_circuit.data[i][0].params[0] = original_param

        return improvement
    def _noise_aware_scheduling(self):
        if not self.current_circuit:
            return 0

        original_fidelity = self.fidelity_before

        # sort qubits by their error rates (ascending)
        sorted_qubits = sorted(range(self.num_qubits), key=lambda q: self.error_rates[q])

        # create a new circuit with reordered gates
        new_circuit = QuantumCircuit(self.num_qubits)

        # apply single-qubit gates first, starting with the least noisy qubits
        for qubit in sorted_qubits:
            for instruction in self.current_circuit.data:
                if len(instruction[1]) == 1 and instruction[1][0] == qubit:
                    new_circuit.append(instruction[0], instruction[1])


        for instruction in self.current_circuit.data:
            if len(instruction[1]) == 2:
                new_circuit.append(instruction[0], instruction[1])

        self.current_circuit = new_circuit

        new_fidelity = self._calculate_fidelity()

        return 1 if new_fidelity > original_fidelity else 0


    def _update_circuit_representation(self):

    # create a new QuantumCircuit object
        qc = QuantumCircuit(self.num_qubits)

    # iterate over all qubits and time steps
        for qubit in range(self.num_qubits):
            for time_step in range(self.max_time_steps):
            # fetch the gate type for the specific qubit and time step
                gate_type = self.gate_operations[qubit, time_step].item()

                if gate_type == 1:  # hadamard gate
                    qc.h(qubit)
                elif gate_type == 2:  # rx rotation
                    angle = np.pi / 4
                    qc.rx(angle, qubit)
                elif gate_type == 3:  # RZ rotation
                    angle = np.pi / 4
                    qc.rz(angle, qubit)
                elif gate_type == 4:  # Single qubit Y rotation
                    angle = np.pi / 4
                    qc.ry(angle, qubit)
                elif gate_type == 5:  # Pauli X gate
                    qc.x(qubit)
                elif gate_type == 6:  # Pauli Y gate
                    qc.y(qubit)
                elif gate_type == 7:  # Pauli Z gate
                    qc.z(qubit)
                elif gate_type == 8:  #S gate
                    qc.s(qubit)
                elif gate_type == 9:  #T gate
                    qc.t(qubit)
                elif gate_type == 10:  #reset
                    qc.reset(qubit)

    # handle multi qubit gates and connectivity
        for qubit_a in range(self.num_qubits):
            for qubit_b in range(qubit_a + 1, self.num_qubits):  # avoid duplicate pairs
            # fetch the gate type for qubit pairs
                multi_qubit_gate = self.gate_operations[qubit_a, qubit_b].item()
                if multi_qubit_gate == 11:  # CNOT gate
                    qc.cx(qubit_a, qubit_b)
                elif multi_qubit_gate == 12:  # SWAP gate
                    qc.swap(qubit_a, qubit_b)
                elif multi_qubit_gate == 13:  # CZ gate
                    qc.cz(qubit_a, qubit_b)

    # update the environment's current circuit
        self.current_circuit = qc


        if self.debug:
            print("Updated logical circuit representation:")
            print(qc)


    def _initialize_random_circuit(self):
        return torch.randint(0, 2, (self.num_qubits, self.max_time_steps), device=self.device)

    def _update_circuit_metrics(self):

        if not isinstance(self.current_circuit, QuantumCircuit):
            print(f"Debug: self.current_circuit type: {type(self.current_circuit)}")
            raise ValueError("Current circuit is not initialized or not a QuantumCircuit.")

        # update circuit depth if within bounds
        if self.current_step < self.max_time_steps:
            circuit_depth = self.current_circuit.depth()
            self.gate_operations[:, self.current_step] = circuit_depth

            # update fidelity history
            fidelity = self.fidelity_after
            self.fidelity_history[self.current_step] = fidelity

            if self.debug:
                print(f"Updated circuit metrics at step {self.current_step}:")
                print(f" Circuit Depth: {circuit_depth}")
                print(f" Fidelity: {fidelity}")
        else:
            if self.debug:
                print(f"Step {self.current_step} exceeds max steps. Metrics update skipped.")




    def _evolve_circuit(self):

        # initialize penalty for invalid or skipped gates
        penalty = 0

        # randomly select a qubit and a time step
        random_qubit = self.rng.integers(self.num_qubits)
        random_time_step = self.rng.integers(self.max_time_steps)

        # randomly decide the gate type and (if applicable) a target qubit
        gate_type = self.rng.choice(['swap', 'cx', 'rz', 'rx', 'h'])
        target_qubit = self.rng.choice(self.num_qubits)

        # ensure target qubit is different from the source qubit for multi-qubit gates
        while target_qubit == random_qubit:
            target_qubit = self.rng.choice(self.num_qubits)

        # validate and apply gate updates
        if gate_type in ['swap', 'cx']:
            # check connectivity for multi-qubit gates
            if self.qubit_connectivity[random_qubit, target_qubit] == 1:
                # assign gate markers for valid gates
                gate_marker = 1 if gate_type == 'swap' else 2
                self.logical_circuit[random_qubit, random_time_step] = gate_marker
                self.logical_circuit[target_qubit, random_time_step] = gate_marker
            else:
                # increment penalty for invalid connectivity
                self.invalid_gate_count += 1
                penalty += 0.075
                if self.debug:
                    print(f"Invalid {gate_type} gate skipped: qubits {random_qubit} and {target_qubit} are not connected.")
        else:
            # single-qubit gates do not require connectivity validation
            gate_marker = {'rz': 3, 'rx': 4, 'h': 5}.get(gate_type, 0)
            self.logical_circuit[random_qubit, random_time_step] = gate_marker

        # add penalties for overused qubits
        overuse_threshold = 10
        if self.qubit_usage[random_qubit] > overuse_threshold:
            penalty += 0.075
            if self.debug:
                print(f"Penalty applied for overusing qubit {random_qubit}.")


        if penalty > 0 and self.debug:
            print(f"Penalty applied for invalid gates or overuse: {penalty:.2f}")
        if self.debug:
            print(f"Updated Logical Circuit:\n{self.logical_circuit}")

        return penalty



    def _reward_swap(self, qubit_a, qubit_b):
        reward = 0.1 if self.qubit_connectivity[qubit_a, qubit_b] == 1 else -0.3
        fidelity_improvement = self._calculate_improvement(qubit_a, qubit_b)
        reward += min(fidelity_improvement / 100, 0.5) if self.qubit_connectivity[qubit_a, qubit_b] == 1 else -0.2
        return reward

    def _reward_gate_insertion(self, qubit):
        base_reward = 0.1
        improvement = self._calculate_improvement(qubit, qubit)
        return base_reward + min(improvement / 100, 1.0)

    def _reward_multi_qubit_gate(self, qubit_a, qubit_b):
        base_reward = 0.15
        reward = base_reward + 0.2 if self.qubit_usage[qubit_a] + self.qubit_usage[qubit_b] < 5 else -0.1
        fidelity_impact = self.fidelity_after * 0.1
        return reward + min(fidelity_impact, 0.5)

    def _reward_logical_swap(self, qubit_a, qubit_b):
        return 0.1 + self._reward_swap(qubit_a, qubit_b)

    def _reward_reorder(self, qubit_a, qubit_b):
        self.gate_operations[[qubit_a, qubit_b]] = self.gate_operations[[qubit_b, qubit_a]]
        return 0.1

    def _reward_adjust_connectivity(self, qubit_a, qubit_b):
        if self.qubit_connectivity[qubit_a, qubit_b] == 0:
            self.qubit_connectivity[qubit_a, qubit_b] = 1
            self.qubit_connectivity[qubit_b, qubit_a] = 1
            return 0.1
        else:
            return -0.1

    def _reward_error_correction(self, qubit):
        self.qubit_usage[qubit] = 0
        return 0.3

    def _reward_reset(self):
        self.reset()
        return 0.2



    def _update_noise_profile(self, action_type, details): # only for swap and multi_qubit_gate

        if action_type == 'swap':

            self.noise_profile[details[0], details[1]] *= 1.02

        elif action_type == 'multi_qubit_gate':

            self.noise_profile[details[0], details[1]] *= 1.01

    def _update_error_and_coherence(self, measurement):
        if self.backend.name == 'qasm_simulator':
            fluctuation = torch.tensor(
                self.rng.uniform(-0.01, 0.01, size=self.num_qubits), device=self.device
            )
            coherence_fluctuation = torch.tensor(
                self.rng.uniform(-5, 5, size=self.num_qubits), device=self.device
            )

            active_qubits = self.deactivated_qubits
            self.error_rates[active_qubits] = torch.clamp(
                self.error_rates[active_qubits] + fluctuation[active_qubits], 0.01, 0.1
            )
            self.coherence_times[active_qubits] = torch.clamp(
                self.coherence_times[active_qubits] + coherence_fluctuation[active_qubits], 50, 150
            )

            self.coherence_times[self.deactivated_qubits] += 1
            reactivation_mask = (self.coherence_times > 60) & self.deactivated_qubits
            self.deactivated_qubits[reactivation_mask] = False

            if self.debug:
                print("Updated error rates:", self.error_rates)
                print("Updated coherence times:", self.coherence_times)


    def _calculate_improvement(self, qubit_a, qubit_b):

        improvement = (self.coherence_times[qubit_a] - self.error_rates[qubit_a]) + \
                      (self.coherence_times[qubit_b] - self.error_rates[qubit_b])
        return torch.clamp(improvement, min=0)


    def _create_circuit(self):

        self.dynamic_depth_variation()
        qc = QuantumCircuit(self.num_qubits)

    # apply initial Hadamard gates for superposition base state
        for qubit in range(self.num_qubits):
            qc.h(qubit)

    # generate gate types in bulk for efficient GPU operation
        for layer in range(self.num_layers):
            gate_types = self.rng.choice(
                ['h', 'rx', 'rz', 'cx', 'swap', 'ry', 's', 't'],
                size=self.num_qubits
            )

            for qubit, gate_type in enumerate(gate_types):
                if gate_type == 'h':
                    qc.h(qubit)
                elif gate_type == 'rx':
                    qc.rx(self.rng.uniform(0, np.pi), qubit)
                elif gate_type == 'rz':
                    qc.rz(self.rng.uniform(0, 2 * np.pi), qubit)
                elif gate_type == 'ry':
                    qc.ry(self.rng.uniform(0, np.pi / 2), qubit)
                elif gate_type == 's':
                    qc.s(qubit)
                elif gate_type == 't':
                    qc.t(qubit)
                elif gate_type == 'cx':
                    target_qubit = (qubit + 1) % self.num_qubits
                    if self.qubit_connectivity[qubit, target_qubit] == 1:
                        qc.cx(qubit, target_qubit)
                        self.multi_qubit_usage[qubit, target_qubit] += 1
                elif gate_type == 'swap':
                    target_qubit = (qubit + 1) % self.num_qubits
                    if self.qubit_connectivity[qubit, target_qubit] == 1:
                        qc.swap(qubit, target_qubit)
                        self.multi_qubit_usage[qubit, target_qubit] += 1

        # introduce a multi-qubit entangling operation every few layers
            if layer % 2 == 0:
                for i in range(0, self.num_qubits - 1, 2):
                    target_qubit = (i + 1) % self.num_qubits
                    qc.cz(i, target_qubit)

        qc.measure_all()
        return self._transpile_for_backend(qc)



    def _calculate_fidelity(self):
        if not self.current_circuit or len(self.current_circuit.data) == 0:
            print("Warning: Empty circuit. Returning old fidelity")
            return self.fidelity_after

        try:
            
            if not any(isinstance(instr.operation, SaveStatevector) for instr in self.current_circuit.data):
                self.current_circuit.save_statevector()

            noise_model = self._initialize_noise_model()
            simulator = AerSimulator(method='statevector', noise_model=noise_model)
            transpiled_circuit = transpile(self.current_circuit, simulator)
            job = simulator.run(transpiled_circuit)
            result = job.result()
            actual_state = result.get_statevector(transpiled_circuit)
            ideal_state = self._generate_ideal_state()

            # calculate fidelity between actual (noisy) and ideal (noiseless) states
            fidelity = state_fidelity(ideal_state, actual_state, validate=False)

            # adjust fidelity based on circuit complexity
            circuit_depth = transpiled_circuit.depth()


            return max(0.0, min(1.0, fidelity))  # clamp between [0, 1]

        except Exception as e:
            print(f"Error during fidelity calculation: {e}")
            return 0.0




    def _generate_ideal_state(self):
        try:
            # create a new circuit with the same structure but without noise
            ideal_circuit = QuantumCircuit(self.num_qubits)

            # recreate the circuit structure without noise
            for instruction in self.current_circuit.data:
                if isinstance(instruction[0], Gate):
                    ideal_circuit.append(instruction[0], instruction[1])

            # ensure save_statevector instruction is present
            ideal_circuit.save_statevector()

            # simulate the noiseless circuit
            simulator = AerSimulator(method='statevector')
            transpiled_ideal_circuit = transpile(ideal_circuit, simulator)
            result = simulator.run(transpiled_ideal_circuit).result()

            # return the DensityMatrix of the ideal state
            return DensityMatrix(result.get_statevector(transpiled_ideal_circuit))
        except Exception as e:
            print(f"Error generating ideal state: {e}")
            return self._generate_default_ideal_state()








    def _generate_target_circuit(self):
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(self.num_qubits)

        target_type = "GHZ" if self.rng.uniform() > 0.5 else "W"

        if target_type == "GHZ":
            # GHZ state
            circuit.h(0)  # apply Hadamard gate to the first qubit
            for i in range(self.num_qubits - 1):
                # apply CNOT gates to entangle all qubits
                if self.qubit_connectivity[i, i + 1] > 0.3:  # ensure connectivity is sufficient
                    circuit.cx(i, i + 1)
                else:
                    print(f"Skipping CNOT between {i} and {i+1} due to low connectivity.")
        elif target_type == "W":
            # generate a W-state (simplified version)
            for i in range(self.num_qubits):
                angle = 2 * np.arcsin(1 / np.sqrt(self.num_qubits - i))
                circuit.ry(angle, i)  # apply rotation to distribute amplitude
                if i < self.num_qubits - 1 and self.qubit_connectivity[i, i + 1] > 0.3:
                    circuit.cx(i, i + 1)  # entangle with the next qubit
                else:
                    print(f"Skipping CNOT between {i} and {i+1} due to low connectivity.")

        for qubit in range(self.num_qubits):
            if self.error_rates[qubit] > 0.05:  
                circuit.x(qubit)  # add X-gates to mitigate decoherence effects

        return circuit


    def _generate_default_ideal_state(self):
        num_states = 2 ** self.num_qubits
        amplitude = 1 / np.sqrt(num_states)
        return [amplitude] * num_states





    def _transpile_for_backend(self, qc):

        print(f"Skipping transpilation. Returning raw circuit") # just pass everything out
        return qc

    def build_and_simulate_circuit(self, reset=True):
        try:
            # rebuild or reset the circuit
            if reset or not self.current_circuit:
                qc = QuantumCircuit(self.num_qubits)
                # apply initial Hadamards if circuit is new
                for qubit in range(self.num_qubits):
                    qc.h(qubit)
                self.current_circuit = qc
            else:
                # start with the existing circuit
                qc = self.current_circuit

            # apply layers of gates
            for layer in range(self.num_layers):
                gate_types = self.rng.choice(['h', 'rx', 'rz', 'cx', 'swap', 'ry', 's', 't'], size=self.num_qubits)
                for qubit, gate_type in enumerate(gate_types):
                    target_qubit = (qubit + 1) % self.num_qubits
                    if gate_type == 'h':
                        qc.h(qubit)
                    elif gate_type == 'rx':
                        qc.rx(self.rng.uniform(0, np.pi), qubit)
                    elif gate_type == 'rz':
                        qc.rz(self.rng.uniform(0, 2 * np.pi), qubit)
                    elif gate_type == 'ry':
                        qc.ry(self.rng.uniform(0, np.pi / 2), qubit)
                    elif gate_type in ['cx', 'swap'] and self.qubit_connectivity[qubit, target_qubit] == 1:
                        getattr(qc, gate_type)(qubit, target_qubit)
                        self.multi_qubit_usage[qubit, target_qubit] += 1
                        self.path_usage_matrix[qubit, target_qubit] += 1
                        self.path_usage_matrix[target_qubit, qubit] += 1
                    elif gate_type == 'cz' and layer % 2 == 0:
                        qc.cz(qubit, target_qubit)
                    self.coherence_times[qubit] *= 0.99  # apply coherence decay

            # assign a unique name to the circuit for tracking
            qc.name = f"circuit_step_{self.current_step}"

            # save statevector
            qc.save_statevector()

            # transpile the circuit for simulation
            simulator = AerSimulator(method='statevector')
            transpiled_circuit = transpile(qc, simulator)


            print("Transpiled Circuit:")
            print(transpiled_circuit.draw())

            # run the simulator
            result = simulator.run(transpiled_circuit).result()

            # retrieve statevector using the transpiled circuit
            statevector = result.get_statevector(transpiled_circuit)
            print(f"Statevector: {statevector}")

            # perform measurement simulation
            qasm_simulator = AerSimulator(method='automatic')
            qc_with_measurements = transpiled_circuit.copy()
            qc_with_measurements.measure_all()

            measurement_job = qasm_simulator.run(qc_with_measurements, shots=1024)
            measurement_result = measurement_job.result()
            counts = measurement_result.get_counts()

            if not counts:
                raise ValueError("Simulation with measurements returned no counts.")

            # update the current circuit (flush/reset)
            if reset:
                self.current_circuit = None
            else:
                self.current_circuit = qc

            # debugging information
            if self.debug:
                print(f"[SIMULATION] Statevector: {statevector}, Circuit Depth: {transpiled_circuit.depth()}")
                print(f"[MEASUREMENT SIMULATION] Counts: {counts}")
                print(f"Metrics after Simulation:\n"
                      f"Multi-Qubit Usage: {self.multi_qubit_usage}\n"
                      f"Path Usage Matrix: {self.path_usage_matrix}\n"
                      f"Coherence Times: {self.coherence_times}")

            return transpiled_circuit

        except Exception as e:
            print(f"[ERROR] Simulation failed: {e}")
            self.current_circuit = None
            raise



    def _evolve_connectivity(self):

        decay_rate = 0.99
        fluctuation = self.rng.uniform(-0.05, 0.05, size=self.qubit_connectivity.shape)

        # update connectivity matrix with decay and fluctuation
        self.qubit_connectivity = torch.clamp(
            self.qubit_connectivity * decay_rate + torch.tensor(fluctuation, device=self.device),
            0.0, 1.0
        )

        # threshold to remove weak connections
        threshold = 0.3
        self.qubit_connectivity[self.qubit_connectivity < threshold] = 0

        # identify and validate essential connections
        essential_connections = self._identify_essential_connections()
        for i, j in essential_connections:
            if i >= self.num_qubits or j >= self.num_qubits:
                if self.debug:
                    print(f"Skipping invalid connection: ({i}, {j})")
                continue
            self.qubit_connectivity[i, j] = max(self.qubit_connectivity[i, j], threshold)
            self.qubit_connectivity[j, i] = max(self.qubit_connectivity[j, i], threshold)

        if self.debug:
            print("Connectivity matrix evolved:")
            print(self.qubit_connectivity)





    def dynamic_depth_variation(self):

        max_possible_connections = self.num_qubits * (self.num_qubits - 1)
        complexity_factor = torch.sum(self.qubit_connectivity).item() / max_possible_connections

        high_complexity_threshold = 0.7
        medium_complexity_threshold = 0.4

        if complexity_factor > high_complexity_threshold:
            self.num_layers = self.rng.integers(8, 12)
        elif complexity_factor > medium_complexity_threshold:
            self.num_layers = self.rng.integers(5, 8)
        else:
            self.num_layers = self.rng.integers(2, 5)

        self.dynamic_depth = self.num_layers > 7

        if self.debug:
            print(f"Complexity factor: {complexity_factor:.2f}")
            print(f"Dynamic depth set to: {self.dynamic_depth}, with {self.num_layers} layers")




    def _perform_multi_qubit_gate(self, qubit_a, qubit_b):

    # base reward for performing the gate
        base_reward = 0.2

    # calculate fidelity before performing the gate to measure improvement
        fidelity_before = self.fidelity_before

    #rreward adjustment based on qubit usage: reward for lower usage, penalize for high usage
        usage_sum = self.qubit_usage[qubit_a] + self.qubit_usage[qubit_b]
        usage_reward = 0.2 * torch.clamp(5 - usage_sum, min=0) / 5  # Scaled to give higher reward when usage is low
        reward = base_reward + usage_reward

    # update usage counts
        self.qubit_usage[qubit_a] += 1
        self.qubit_usage[qubit_b] += 1

    # apply noise feedback and fidelity impact
        self._apply_feedback('multi_qubit_gate', [qubit_a, qubit_b])

    # calculate fidelity impact after applying the gate
        fidelity_after = self._calculate_fidelity()
        fidelity_improvement = fidelity_after - fidelity_before
        fidelity_reward = torch.clamp(torch.tensor(fidelity_improvement * 0.05, device=self.device), max=0.5)


    # apply additional penalty if multi-qubit gate is overused
        gate_count = self.action_repeat_counter.get((qubit_a, qubit_b), 0)
        overuse_penalty = -0.1 if gate_count > 2 else 0
        self.action_repeat_counter[(qubit_a, qubit_b)] = gate_count + 1

    #total reward includes base reward, usage adjustment, fidelity improvement, overuse penalty
        reward += fidelity_reward + overuse_penalty
        reward = torch.tensor(reward, device=self.device)
        return torch.clamp(reward, min=-1.5, max=1.5).item()


    def _identify_essential_connections(self):

        essential_connections = set()  # se set to avoid duplicates

    #process each gate in the logical circuit
        for gate in self.logical_circuit:
        #filter out unused qubit slots
            qubits = gate[gate >= 0]  #keep only valid qubit indices
            if len(qubits) < 2:
                continue  # skip single-qubit gates
        # handle all pairs for multi-qubit gates
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    essential_connections.add((qubits[i].item(), qubits[j].item()))

    # return as a sorted list of tuples
        return sorted(essential_connections)




    def _apply_feedback(self, action_type, qubits):

        # define action-specific degradation factors for both error rates and coherence times
        action_degradation_factors = {
            'multi_qubit_gate': {'error_rate': 1.03, 'coherence': 0.98},
            'swap': {'error_rate': 1.02, 'coherence': 0.99},
            'logical_swap': {'error_rate': 1.01, 'coherence': 0.995}
        }
        degradation_factors = action_degradation_factors.get(action_type, {'error_rate': 1.01, 'coherence': 0.99})

        for qubit in qubits:
            # ensure self.error_rates and self.coherence_times are PyTorch tensors
            if not isinstance(self.error_rates, torch.Tensor):
                self.error_rates = torch.tensor(self.error_rates, dtype=torch.float32, device=self.device)
            if not isinstance(self.coherence_times, torch.Tensor):
                self.coherence_times = torch.tensor(self.coherence_times, dtype=torch.float32, device=self.device)

            # apply degradation to the error rate
            self.error_rates[qubit] *= degradation_factors['error_rate']
            self.error_rates[qubit] = torch.min(
                self.error_rates[qubit],
                torch.tensor(0.1, dtype=torch.float32, device=self.device)
            )

            # apply degradation to coherence time
            self.coherence_times[qubit] *= degradation_factors['coherence']
            self.coherence_times[qubit] = torch.max(
                self.coherence_times[qubit],
                torch.tensor(10.0, dtype=torch.float32, device=self.device)
            )

            # update action history for the qubit
            self.action_repeat_counter[qubit] = self.action_repeat_counter.get(qubit, 0) + 1





    def _insert_gate(self, qubit):
        # choose a gate type with consideration for two-qubit gates
        gate_type = self.rng.choice(['H', 'X', 'Y', 'Z', 'CX', 'SWAP'])
        target_time_step = self.rng.integers(self.max_time_steps)  #random time step within bounds

        # insert single-qubit gates directly
        if gate_type in ['H', 'X', 'Y', 'Z']:
            gate_mapping = {'H': 1, 'X': 2, 'Y': 3, 'Z': 4}
            self.gate_operations[qubit, target_time_step] = gate_mapping[gate_type]

        # handle two-qubit gates by choosing a second qubit
        elif gate_type in ['CX', 'SWAP']:
            target_qubit = self.rng.choice([q for q in range(self.num_qubits) if q != qubit])
            gate_mapping = {'CX': 5, 'SWAP': 6}
            self.gate_operations[qubit, target_time_step] = gate_mapping[gate_type]
            self.gate_operations[target_qubit, target_time_step] = gate_mapping[gate_type]

        # base reward with adjustments for multi-qubit gates
        base_reward = 0.1 if gate_type in ['H', 'X', 'Y', 'Z'] else 0.05

        # improvement calculation for both qubits in case of two-qubit gate
        if gate_type in ['CX', 'SWAP']:
            improvement = (self._calculate_improvement(qubit, target_qubit) +
                       self._calculate_improvement(target_qubit, qubit)) / 2
        else:
            improvement = self._calculate_improvement(qubit, qubit)

        normalized_improvement = torch.clamp(improvement / 100, max=1.0)
        reward = base_reward + normalized_improvement.item()
        return reward



    def _perform_swap(self, qubit_a, qubit_b):
        # check if the swap is feasible
        if self.qubit_connectivity[qubit_a, qubit_b] == 1:
            # choose a consistent time step for the swap
            target_time_step = self.rng.integers(self.max_time_steps)
            self.gate_operations[qubit_a, target_time_step] = 5
            self.gate_operations[qubit_b, target_time_step] = 5

        # base reward with slight adjustment based on fidelity improvement
            base_reward = 0.05
            fidelity_improvement = self._calculate_improvement(qubit_a, qubit_b)
            normalized_improvement = torch.clamp(fidelity_improvement / 100, max=0.5)

        # apply a dynamic reward boost if fidelity improvement is significant
            reward = base_reward + normalized_improvement
            if fidelity_improvement > 20:  # arbitrary threshold for higher fidelity improvement
                reward += 0.05  # small bonus for a highly beneficial swap

            # apply additional penalty if this pair of qubits is swapped frequently
            swap_count = self.action_repeat_counter.get((qubit_a, qubit_b), 0)
            if swap_count > 2:  # threshold for frequent swaps on the same pair
                reward -= 0.1
            self.action_repeat_counter[(qubit_a, qubit_b)] = swap_count + 1

        else:
            # infeasible swap
            reward = -0.5

        # general penalty to limit excessive swaps
        reward -= 0.2
        return torch.clamp(torch.tensor(reward, device=self.device), -1.5, 1.5).item()

    def _select_target_qubits(self): # help our agent become more clinical

        # find all pairs of qubits with valid connectivity
        candidates = torch.nonzero(self.qubit_connectivity > 0, as_tuple=False).cpu().numpy()

        if len(candidates) == 0:
            return None, None  # no valid pairs found

        # sort candidates by combined qubit usage (lower usage prioritized)
        sorted_candidates = sorted(
            candidates,
            key=lambda pair: (self.qubit_usage[pair[0]] + self.qubit_usage[pair[1]])
        )

        # select the first pair of qubits with valid connectivity
        for pair in sorted_candidates:
            qubit_a, qubit_b = pair
            if self._is_gate_valid("multi_qubit_gate", qubit_a, qubit_b):  # ensure gate validity
                return int(qubit_a), int(qubit_b)

        # if no valid pair is found, return None
        return None, None



    def _apply_action(self, action_type, details):
        # initialize circuit if not already done
        if not hasattr(self, 'current_circuit') or self.current_circuit is None:
            print("Warning: current_circuit was not initialized. Reinitializing now.")
            self.current_circuit = QuantumCircuit(self.num_qubits)
            self.current_circuit.h(range(self.num_qubits))  # Initial Hadamard gates for meaningful circuit

        # dynamic target selection for multi-qubit actions
        if action_type in ["swap", "multi_qubit_gate", "logical_swap"]:
            if details is None or not isinstance(details, tuple):
                qubit_a, qubit_b = self._select_target_qubits()
                if qubit_a is None or qubit_b is None:
                    print(f"No valid qubit pair found for {action_type}. Skipping action.")
                    return
                details = (qubit_a, qubit_b)
            else:
                qubit_a, qubit_b = details

            # validate the gate before applying
            if not self._is_gate_valid(action_type, qubit_a, qubit_b):
                self.invalid_gate_count += 1
                if self.debug:
                    print(f"Invalid {action_type} action: qubits {qubit_a} and {qubit_b} are not connected.")
                return

        # handle specific actions
        if action_type == "swap":
            qubit_a, qubit_b = details
            self.current_circuit.swap(qubit_a, qubit_b)
            self.multi_qubit_usage[qubit_a, qubit_b] += 1
            if self.debug:
                print(f"Swap action applied between {qubit_a} and {qubit_b}. ")

        elif action_type == "logical_swap":
            qubit_a, qubit_b = details
            self._perform_logical_swap(qubit_a, qubit_b)
            if self.debug:
                print(f"Logical swap performed between {qubit_a} and {qubit_b}. ")

        elif action_type == "reorder":
            qubit_a, qubit_b = details
            self._reorder_qubits(qubit_a, qubit_b)
            if self.debug:
                print(f"Reordered qubits {qubit_a} and {qubit_b}. ")

        elif action_type == "adjust_connectivity":
            qubit_a, qubit_b = details
            self._adjust_connectivity(qubit_a, qubit_b)
            if self.debug:
                print(f"Adjusted connectivity between {qubit_a} and {qubit_b}. ")

        elif action_type == "gate_insertion":
            # validate gate insertion details
            if not isinstance(details, dict) or "qubit" not in details:
                print(f"Invalid gate insertion details: {details}")
                return

            qubit = details["qubit"]
            gate_type = details.get("gate_type", "H")
            angle = details.get("angle", np.pi / 4)

            try:
                # apply specified gate type
                if gate_type == "H":
                    self.current_circuit.h(qubit)
                elif gate_type == "X":
                    self.current_circuit.x(qubit)
                elif gate_type == "Y":
                    self.current_circuit.y(qubit)
                elif gate_type == "Z":
                    self.current_circuit.z(qubit)
                elif gate_type == "RX":
                    self.current_circuit.rx(angle, qubit)
                elif gate_type == "RZ":
                    self.current_circuit.rz(angle, qubit)
                else:
                    print(f"Unsupported gate type: {gate_type}")
                    return
                if self.debug:
                    print(f"Gate insertion action on qubit {qubit} with gate {gate_type}. ")
            except Exception as e:
                print(f"Error applying gate insertion: {e}")

        elif action_type == "multi_qubit_gate":
            qubit_a, qubit_b = details
            self._perform_multi_qubit_gate(qubit_a, qubit_b)
            if self.debug:
                print(f"Multi-qubit gate performed between {qubit_a} and {qubit_b}. ")

        elif action_type == "error_correction":
            # Single-qudit error correction
            if isinstance(details, int):
                self._perform_error_correction(details)
                if self.debug:
                    print(f"Error correction on qudit {details}")
        elif action_type == "gate_cancellation":
            depth_reduction = self._cancel_gates()
            if self.debug:
                print(f"Gate cancellation applied. Depth reduced by {depth_reduction}.")

        elif action_type == "topology_aware_swap":
            qubit_a, qubit_b = details
            swap_performed = self._topology_aware_swap(qubit_a, qubit_b)
            if self.debug:
                print(f"Topology-aware swap {'performed' if swap_performed else 'not needed'} between {qubit_a} and {qubit_b}.")

        elif action_type == "adaptive_layer_management":
            if self.current_circuit is not None:
                depth_reduction = self._adaptive_layer_management()
                if self.debug:
                    print(f"Adaptive layer management applied. Depth reduced by {depth_reduction}.")
            else:
                print("Warning: Cannot perform adaptive layer management on an empty circuit.")

        elif action_type == "parameterized_optimization":
            improvements = self._optimize_parameters()
            if self.debug:
                print(f"Parameterized optimization applied. {improvements} improvements made.")

        elif action_type == "noise_aware_scheduling":
            improvement = self._noise_aware_scheduling()
            if self.debug:
                print(f"Noise-aware scheduling {'improved' if improvement else 'did not improve'} the circuit.")
        elif action_type == "reset":
            # skip reset based on circuit health (optional override logic can be added here)
            if self.fidelity_before > 0.95 and torch.mean(self.coherence_times).item() > 30.0:
                if self.debug:
                    print("Reset action skipped due to acceptable circuit health.")
            return

            # perform reset on all or specific qubits
            if details is None:  # reset all qubits
                for qubit in range(self.num_qubits):
                    self.current_circuit.reset(qubit)
                    self.qubit_usage[qubit] = 0
                    self.qubit_idle_time[qubit] = 0
            else:  # reset specific qubits
                qubits_to_reset = [details] if isinstance(details, int) else details
                for qubit in qubits_to_reset:
                    self.current_circuit.reset(qubit)


        elif action_type == "rotation_x":
            qubit = details

            self.current_circuit.rx(np.pi / 4, qubit)
            if self.debug:
                print(f"Applied RX rotation on qubit {qubit}. Reward: ")

        elif action_type == "rotation_z":
            qubit = details

            self.current_circuit.rz(np.pi / 4, qubit)
            if self.debug:
                print(f"Applied RZ rotation on qubit {qubit}. Reward: ")

        else:
            print(f"Unknown action type: {action_type}")

        # update qubit usage for involved qubits
        if isinstance(details, tuple):
            for qubit in details:
                self.qubit_usage[qubit] += 1
        elif isinstance(details, int):
            self.qubit_usage[details] += 1



    def _is_gate_valid(self, gate_type, qubit_a, qubit_b=None):
        if gate_type in ['cx', 'swap']:
            return self.qubit_connectivity[qubit_a, qubit_b] == 1
        elif gate_type in ['h', 'rz', 'rx', 'x', 'z', 'y']:  # single-qubit gates
            return self.coherence_times[qubit_a] > 10.0  # nsure qubit health
        return True



    def _perform_logical_swap(self, qubit_a, qubit_b):
        # perform a physical swap and capture its reward
        swap_reward = self._perform_swap(qubit_a, qubit_b)

        # calculate coherence or fidelity improvement specific to the logical swap
        fidelity_improvement = self._calculate_improvement(qubit_a, qubit_b)
        coherence_improvement = (self.coherence_times[qubit_a] + self.coherence_times[qubit_b]) / 2

        # set a base reward for logical swaps and adjust for coherence improvement
        base_reward = 0.3
        if coherence_improvement > 50:
            logical_swap_bonus = 0.1  # additional reward for high-coherence logical swaps
        else:
            logical_swap_bonus = 0

        #total reward includes the swap reward, base logical swap reward, and any bonus
        reward = swap_reward + base_reward + logical_swap_bonus
        reward = torch.tensor(reward, device=self.device)
        return torch.clamp(reward, min=-1.5, max=1.5).item()



    def _reorder_qubits(self, qubit_a, qubit_b):
        self.gate_operations[[qubit_a, qubit_b]] = self.gate_operations[[qubit_b, qubit_a]]

        #base reward for reordering
        base_reward = 0.1

        #calculate fidelity improvement due to reordering
        fidelity_improvement = self._calculate_improvement(qubit_a, qubit_b)
        normalized_fidelity_improvement = min(fidelity_improvement / 100, 0.5)

        #check if the new ordering improves connectivity or coherence
        connectivity_bonus = 0.05 if self.qubit_connectivity[qubit_a, qubit_b] == 1 else 0
        coherence_improvement = (self.coherence_times[qubit_a] + self.coherence_times[qubit_b]) / 2
        coherence_bonus = 0.05 if coherence_improvement > 50 else 0

        #track and penalize excessive reordering of the same qubits
        reorder_count = self.action_repeat_counter.get((qubit_a, qubit_b), 0)
        repeat_penalty = -0.05 if reorder_count > 2 else 0
        self.action_repeat_counter[(qubit_a, qubit_b)] = reorder_count + 1

        #calculate the total reward
        reward = base_reward + normalized_fidelity_improvement + connectivity_bonus + coherence_bonus + repeat_penalty
        return torch.clamp(torch.tensor(reward), -1.5, 1.5).item()


    def _adjust_connectivity(self, qubit_a, qubit_b):
        #check if the connectivity needs adjustment
        if self.qubit_connectivity[qubit_a, qubit_b] == 0:
        #enable connectivity in both directions
            self.qubit_connectivity[qubit_a, qubit_b] = 1
            self.qubit_connectivity[qubit_b, qubit_a] = 1

        #base reward for successful adjustment
            base_reward = 0.1

        #calculate fidelity improvement from the connectivity adjustment
            fidelity_improvement = self._calculate_improvement(qubit_a, qubit_b)
            normalized_fidelity_improvement = min(fidelity_improvement / 100, 0.5)
            coherence_improvement = (self.coherence_times[qubit_a] + self.coherence_times[qubit_b]) / 2
            coherence_bonus = 0.05 if coherence_improvement > 50 else 0

        #discourage excessive adjustments by tracking and penalizing frequent adjustments
            adjust_count = self.action_repeat_counter.get((qubit_a, qubit_b), 0)
            repeat_penalty = -0.05 if adjust_count > 2 else 0
            self.action_repeat_counter[(qubit_a, qubit_b)] = adjust_count + 1

            # calculate total reward
            reward = base_reward + normalized_fidelity_improvement + coherence_bonus + repeat_penalty
        else:
            # penalty for attempting an unnecessary connectivity adjustment
            reward = -0.1

        reward = torch.tensor(reward, device=self.device)
        return torch.clamp(reward, min=-1.5, max=1.5).item()



    def _perform_error_correction(self, qubit):
        # reset the idle time as part of error correction
        self.qubit_idle_time[qubit] = 0

        # base reward for performing error correction
        base_reward = 0.2

        # additional reward based on the qubit's current error rate
        error_rate = self.error_rates[qubit]
        error_rate_bonus = min(error_rate * 2, 0.3)  # Higher error rate yields higher bonus

        # calculate fidelity improvement resulting from error correction
        fidelity_improvement = self._calculate_improvement(qubit, qubit)
        normalized_fidelity_improvement = min(fidelity_improvement / 100, 0.5)  # Cap to a max of 0.5

        # penalize frequent error corrections on the same qubit to avoid redundancy
        correction_count = self.action_repeat_counter.get(qubit, 0)
        repeat_penalty = -0.05 if correction_count > 2 else 0
        self.action_repeat_counter[qubit] = correction_count + 1

        # calculate total reward
        reward = base_reward + error_rate_bonus + normalized_fidelity_improvement + repeat_penalty

        # convert reward to a PyTorch tensor and clamp it
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        return torch.clamp(reward_tensor, -1.5, 1.5).item()



    def _reset_circuit(self):
        # call the main reset function to reset circuit state
        self.reset()

        # base reward for resetting the circuit
        base_reward = 0.2

        # additional reward if the circuit is in poor condition
        avg_error_rate = np.mean(self.error_rates)
        avg_coherence = np.mean(self.coherence_times)
        error_rate_bonus = 0.1 if avg_error_rate > 0.05 else 0
        low_coherence_penalty = -0.1 if avg_coherence < 30 else 0

        #penalize frequent resets within an episode to discourage overuse
        reset_count = self.action_repeat_counter.get('reset', 0)
        repeat_penalty = -0.05 if reset_count > 2 else 0
        self.action_repeat_counter['reset'] = reset_count + 1

        # calculate the total reward with bonuses and penalties
        reward = base_reward + error_rate_bonus + low_coherence_penalty + repeat_penalty
        reward = torch.tensor(reward, device=self.device)
        return torch.clamp(reward, min=-1.5, max=1.5).item()



    def _decode_action(self, action):
        action_mapping = {
            0: ('swap', (0, 1)),
            1: ('swap', (1, 2)),
            2: ('logical_swap', (0, 1)),
            3: ('reorder', (0, 1)),
            4: ('adjust_connectivity', (0, 1)),
            5: ('gate_insertion', 0),
            6: ('multi_qubit_gate', (0, 1)),
            7: ('error_correction', 0),
            8: ('reset', None),
            9: ('rotation_x', 0),
            10: ('rotation_z', 1),
            11: ('gate_cancellation', None),
            12: ('topology_aware_swap', (0, 1)),
            13: ('adaptive_layer_management', None),
            14: ('parameterized_optimization', None),
            15: ('noise_aware_scheduling', None),
        }
        if action not in action_mapping:
            raise ValueError(f"Invalid action: {action}")
        return action_mapping[action]



    def run_consistency_test(self, agent, episodes=10):
        all_rewards = []
        for ep in range(episodes):
            state = self.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                state, reward, done, _ = self.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
            print(f"Episode {ep + 1}: Total Reward = {total_reward}")
        avg_reward = np.mean(all_rewards)
        print(f"Average Reward over {episodes} episodes: {avg_reward}")
        return all_rewards

