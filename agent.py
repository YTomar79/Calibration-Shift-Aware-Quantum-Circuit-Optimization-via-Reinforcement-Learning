
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import math

# define a tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'log_prob', 'fidelity_value', 'coherence_value', 'done'))

# PPO Actor-Critic network with LSTM and dual critics
class PPOActorCriticLSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, lstm_hidden_size=128):
        super(PPOActorCriticLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.actor_fc = nn.Linear(lstm_hidden_size, action_size)
        self.fidelity_critic_fc = nn.Linear(lstm_hidden_size, 1)
        self.coherence_critic_fc = nn.Linear(lstm_hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.fc1, self.actor_fc, self.fidelity_critic_fc, self.coherence_critic_fc]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, lstm_hidden_state, action_mask=None):
        x = torch.relu(self.fc1(state))
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add time dimension

        batch_size = x.size(0)
        if lstm_hidden_state[0].size(1) != batch_size:
            lstm_hidden_state = (
                lstm_hidden_state[0].repeat(1, batch_size, 1),
                lstm_hidden_state[1].repeat(1, batch_size, 1),
            )

        x, lstm_hidden_state = self.lstm(x, lstm_hidden_state)

        if x.size(1) == 1:
            x = x.squeeze(1)
            lstm_hidden_state = (
                lstm_hidden_state[0].squeeze(1),
                lstm_hidden_state[1].squeeze(1),
            )

        action_logits = self.actor_fc(x)
        if action_mask is not None:
            action_logits += action_mask
        action_probs = torch.softmax(action_logits, dim=-1)

        fidelity_value = self.fidelity_critic_fc(x).squeeze()
        coherence_value = self.coherence_critic_fc(x).squeeze()

        return action_probs, fidelity_value, coherence_value, lstm_hidden_state

    def init_lstm_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        return (
            torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        )

class PPOAgent:
    def __init__(self, state_size, action_size, lr=None, gamma=None, k_epochs=None, eps_clip=None, gae_lambda=None,
                 entropy_coefficient=None, entropy_decay=None, replay_buffer_size=None, batch_size=None,
                 alpha=None, beta_start=None, beta_frames=None, min_entropy_coeff=None, max_entropy_coeff=None): # Hyperparamters are not shown...
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.entropy_decay = entropy_decay
        self.min_entropy_coeff = min_entropy_coeff
        self.max_entropy_coeff = max_entropy_coeff
        self.entropy_schedule_mode = "adaptive"  # options for me (js testing): "adaptive"/ "decay"
        self.device = torch.device("cuda")
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.policy = PPOActorCriticLSTM(state_size, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-6)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.priorities = deque(maxlen=replay_buffer_size)

        # initialize for adaptive entropy
        self.entropy_moving_avg = None

    def select_action(self, state, lstm_hidden_state, action_mask=None):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs, fidelity_value, coherence_value, lstm_hidden_state = self.policy(state, lstm_hidden_state, action_mask)

        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob, fidelity_value, coherence_value, lstm_hidden_state

    def store_transition(self, transition, td_error=1.0):
        self.replay_buffer.append(transition)
        self.priorities.append(torch.tensor(td_error + 1e-5, device=self.device))


    def sample_replay_buffer(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None

        priorities = np.array([p.cpu().numpy() if torch.is_tensor(p) else p for p in self.priorities]).flatten()

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        samples = [self.replay_buffer[idx] for idx in indices]

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.replay_buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        self.frame += 1

        return samples, weights, indices



    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5

    def adjust_entropy_coefficient(self, dist, writer=None, episode=None, target_episode=2501, train_frequency=10):
        policy_entropy = dist.entropy().mean().item()

        # initialize moving average if first call
        if self.entropy_moving_avg is None:
            self.entropy_moving_avg = policy_entropy
            self.stuck_counter = 0

            self.plateau_counter = 0

        # update moving average with momentum
        self.entropy_moving_avg = 0.8 * self.entropy_moving_avg + 0.2 * policy_entropy


        num_updates = target_episode / train_frequency
        base_decay_rate = math.log(self.min_entropy_coeff / self.entropy_coefficient) / num_updates


        entropy_ratio = policy_entropy / self.entropy_moving_avg



        if entropy_ratio < 0.9:  
            # very aggressive exploration boost
            self.entropy_coefficient = min(self.entropy_coefficient * 2.0, self.max_entropy_coeff)
            adjusted_decay_rate = base_decay_rate * 0.3  # much slower decay

        elif entropy_ratio > 1.1:
            # stronger exploitation
            self.entropy_coefficient = max(self.entropy_coefficient * 0.7, self.min_entropy_coeff)
            adjusted_decay_rate = base_decay_rate * 1.5

        else:
            adjusted_decay_rate = base_decay_rate



        # apply decay after adjustments
        if episode is not None:
            self.entropy_coefficient = max(
                self.entropy_coefficient * math.exp(adjusted_decay_rate),
                self.min_entropy_coeff
            )

        if random.random() < 0.05 and episode < 6000:
            self.entropy_coefficient = 0.5



        if writer and episode is not None:
            writer.add_scalar('Policy/Entropy', policy_entropy, episode)
            writer.add_scalar('Entropy/Coefficient', self.entropy_coefficient, episode)
            writer.add_scalar('Policy/Entropy_Moving_Avg', self.entropy_moving_avg, episode)






    def train(self, writer=None, episode=None):
        samples, weights, indices = self.sample_replay_buffer()
        if samples is None:
            print("Not enough samples in replay buffer to train.")
            return

        batch = Transition(*zip(*samples))
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        fidelity_values = torch.cat([fv.unsqueeze(0) for fv in batch.fidelity_value]).to(self.device)
        coherence_values = torch.cat([cv.unsqueeze(0) for cv in batch.coherence_value]).to(self.device)
        log_probs = torch.cat([lp.unsqueeze(0) for lp in batch.log_prob]).to(self.device)

        fidelity_advantages, fidelity_returns = self._compute_gae(rewards, dones, fidelity_values)
        coherence_advantages, coherence_returns = self._compute_gae(rewards, dones, coherence_values)

        weight_fidelity = 0.7
        weight_coherence = 1 - weight_fidelity
        advantages = (weight_fidelity * fidelity_advantages + weight_coherence * coherence_advantages).detach()
        returns = (fidelity_returns + coherence_returns).detach()

        old_states = torch.stack([s.to(self.device) for s in batch.state]).view(-1, 1, self.state_size)

        old_actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(-1)

        avg_actor_loss = 0
        avg_critic_loss = 0
        avg_total_loss = 0

        for _ in range(self.k_epochs):
            lstm_hidden_state = self.policy.init_lstm_hidden(batch_size=old_states.size(0))
            action_probs, fidelity_values, coherence_values, _ = self.policy(old_states, lstm_hidden_state)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(old_actions.squeeze())
            ratios = torch.exp(new_log_probs - log_probs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -(torch.min(surr1, surr2) * weights).mean()

            # train fidelity critic with true fidelity values
            true_fidelity_values = torch.tensor(
                [env._calculate_fidelity() for _ in batch.state],
                device=self.device
            )

            fidelity_loss = nn.MSELoss()(fidelity_values, true_fidelity_values)
            coherence_loss = nn.MSELoss()(coherence_values, coherence_returns)
            critic_loss = fidelity_loss * 0.7 + coherence_loss * 0.3  # combine losses




            entropy_loss = -dist.entropy().mean()
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy_loss

            # compute TD errors for PER update
            td_errors = torch.abs(fidelity_returns - fidelity_values.detach()) + \
                        torch.abs(coherence_returns - coherence_values.detach())

            avg_actor_loss += actor_loss.item() / self.k_epochs
            avg_critic_loss += critic_loss.item() / self.k_epochs
            avg_total_loss += total_loss.item() / self.k_epochs

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.update_priorities(indices, td_errors.cpu().numpy())
        self.adjust_entropy_coefficient(dist, writer=writer, episode=episode)

        if writer and episode is not None:
            writer.add_scalar('Loss/Actor_Loss', avg_actor_loss, episode)
            writer.add_scalar('Loss/Critic_Loss', avg_critic_loss, episode)
            writer.add_scalar('Loss/Total_Loss', avg_total_loss, episode)

    def _compute_gae(self, rewards, dones, values):
        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)
        running_advantage = 0
        running_return = 0
        prev_value = 0

        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * prev_value * mask - values[t]
            running_advantage = delta + self.gamma * self.gae_lambda * running_advantage * mask
            running_return = rewards[t] + self.gamma * running_return * mask
            advantages[t] = running_advantage
            returns[t] = running_return
            prev_value = values[t]

        return advantages, returns

    def compute_action_mask(self, env):
        mask = torch.ones(self.action_size, dtype=torch.bool, device=self.device)
        for qubit_a in range(env.num_qubits):
            for qubit_b in range(env.num_qubits):
                if env.qubit_connectivity[qubit_a, qubit_b] == 0:
                    mask[0] = 0
                    mask[1] = 0

        return mask

    def update_curriculum(self, avg_episode_reward):
        if avg_episode_reward >= self.curriculum_threshold:
            self.curriculum_stage += 1
            self.curriculum_threshold *= self.curriculum_increment
            print(f"Curriculum level increased to {self.curriculum_stage}. New threshold: {self.curriculum_threshold}")


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        loaded_buffer = checkpoint['replay_buffer']
        loaded_priorities = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in checkpoint['priorities']]

        max_size = max(len(self.replay_buffer), len(loaded_buffer), len(self.priorities), len(loaded_priorities))

        self.replay_buffer = deque(loaded_buffer, maxlen=max_size)
        self.priorities = deque(loaded_priorities, maxlen=max_size)



        env_state = checkpoint.get('environment_state', {})
        if env_state:
            env.qubit_usage = torch.tensor(env_state['qubit_usage'], device=env.device)
            env.noise_profile = torch.tensor(env_state['noise_profile'], device=env.device)
            env.qubit_connectivity = torch.tensor(env_state['qubit_connectivity'], device=env.device)
            env.fidelity_history = torch.tensor(env_state['fidelity_history'], device=env.device)

        self.lstm_hidden_state = checkpoint.get('lstm_hidden_state', None)

        print(f"Checkpoint loaded from {filename}, resuming from episode {checkpoint['episode']}")
        print(f"Loaded replay buffer size: {len(self.replay_buffer)}")
        print(f"Loaded priorities size: {len(self.priorities)}")

        return checkpoint['episode']






# initialize environment and PPO Agent
env = QuantumCircuitEnvExpanded()
state = env.reset()
state_size = state.shape[0]
action_size = 16

ppo_agent = PPOAgent(state_size, action_size)

num_training_episodes = 10001

checkpoint_frequency = 100
train_frequency = 10  # train every 10 episodes
checkpoint_path = "none!"

try:
    start_episode = ppo_agent.load_checkpoint(checkpoint_path)
    print(f"Resuming training from episode {start_episode}")
except FileNotFoundError:
    print("No checkpoint found. Starting training from scratch.")
    start_episode = 0

writer = SummaryWriter(log_dir='final_ppo')

# main loop
for episode in range(0, num_training_episodes + 10000):
    state = env.reset(episode=episode)
    lstm_hidden_state = ppo_agent.policy.init_lstm_hidden()
    episode_reward = 0
    done = False
    step = 0
    if episode > 9000:
        max_episode_length = 200
    else:
        max_episode_length = 200

    episode_step = 0

    while not done:
        action_mask = ppo_agent.compute_action_mask(env)
        action, action_log_prob, fidelity_value, coherence_value, lstm_hidden_state = ppo_agent.select_action(
            state, lstm_hidden_state, action_mask
        )
        next_state, reward, done, _ = env.step(action)

        # recompute actual fidelity after the step
        true_fidelity = env._calculate_fidelity()  # true fidelity from the environment

        # compute current and next state values using critics
        _, fidelity_next_value, coherence_next_value, _ = ppo_agent.policy(
            torch.tensor(next_state, dtype=torch.float32, device=ppo_agent.device).unsqueeze(0),
            lstm_hidden_state
        )

        # combine fidelity and coherence values for a weighted value function
        weight_fidelity = 0.7
        weight_coherence = 1 - weight_fidelity
        next_value = weight_fidelity * fidelity_next_value.item() + weight_coherence * coherence_next_value.item()

        current_value = weight_fidelity * fidelity_value.item() + weight_coherence * coherence_value.item()

        # proper TD error calculation
        td_error = abs(reward + ppo_agent.gamma * next_value - current_value)



        transition = Transition(state, action, reward, action_log_prob, fidelity_value, coherence_value, done)
        ppo_agent.store_transition(transition, td_error)  # store with priority
        state = next_state
        episode_reward += reward
        step += 1

        if done:
            episode_step = step

    episode_reward = episode_reward/episode_step

    # train the agent every 10 episodes
    if (episode + 1) % train_frequency == 0:
        print(f"Training agent at episode {episode + 1}...")
        ppo_agent.train(writer=writer, episode=episode)


    # log metrics to TensorBoard
    writer.add_scalar('Reward/Episode', episode_reward, episode)
    writer.add_scalar('Fidelity/Predicted', fidelity_value.item(), episode)
    writer.add_scalar('Fidelity/Actual', true_fidelity, episode)
    writer.add_scalar('Episode/Length', step, episode)
    writer.add_scalar('Fidelity/Predicted', fidelity_value.item(), episode)
    writer.add_scalar('Fidelity/Actual', true_fidelity, episode)


    print(f"Episode {episode + 1} | Total Reward: {episode_reward:.4f}")

    # save checkpoint every 100 episodes
    if (episode + 1) % checkpoint_frequency == 0:
        checkpoint = {
            'model_state_dict': ppo_agent.policy.state_dict(),
            'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
            'replay_buffer': list(ppo_agent.replay_buffer),
            'priorities': list(ppo_agent.priorities),
            'entropy_coefficient': ppo_agent.entropy_coefficient,
            'episode': episode + 1,
            'environment_state': {
                'qubit_usage': env.qubit_usage.cpu().numpy(),
                'noise_profile': env.noise_profile.cpu().numpy(),
                'qubit_connectivity': env.qubit_connectivity.cpu().numpy(),
                'fidelity_history': env.fidelity_history.cpu().numpy()
            },
            'lstm_hidden_state': lstm_hidden_state
        }


        torch.save(checkpoint, f"ppo_checkpoint_{episode + 1}.pt")
        print(f"Checkpoint saved at episode {episode + 1}")
writer.flush()
writer.close()

