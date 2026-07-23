import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory

# Automatically select device (CUDA, MPS, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Flappy Bird.
    Manages action selection, replay memory, and optimization.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=1e-4,
        gamma=0.99,
        memory_size=50000,
        tau=0.005,  # Soft update parameter
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Policy network (used to select actions and learn)
        self.policy_dqn = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Target network (used to compute stable target Q-values)
        self.target_dqn = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()  # Target network is always in eval mode

        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Experience Replay Memory
        self.memory = ReplayMemory(maxlen=memory_size)

    def select_action(self, state, epsilon=0.0):
        """
        Selects an action using epsilon-greedy exploration.
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action from policy network
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_dqn(state_t)
            return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, terminated):
        """
        Saves experience in replay memory and trains policy net.
        """
        # Append transition to memory
        self.memory.append((state, action, reward, next_state, terminated))

    def optimize(self, batch_size):
        """
        Samples a batch of experiences and performs a gradient descent update.
        """
        if len(self.memory) < batch_size:
            return None

        # Sample a batch of transitions
        batch = self.memory.sample(batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, terminateds = zip(*batch)

        # Convert to PyTorch tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        terminateds_t = torch.FloatTensor(terminateds).to(self.device)

        # Get current Q values: Q(s, a) using the policy network
        current_q = self.policy_dqn(states_t).gather(1, actions_t)

        # Compute next target Q values: max_a Q_target(s', a) using target network
        with torch.no_grad():
            max_next_q = self.target_dqn(next_states_t).max(dim=1)[0]
            # If state is terminal, the target Q value is just the reward
            target_q = rewards_t + (1 - terminateds_t) * self.gamma * max_next_q

        # Compute loss (Huber Loss)
        loss = self.loss_fn(current_q, target_q.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update the target network using a soft update
        self.soft_update()

        return loss.item()

    def soft_update(self):
        """
        Perform soft update of target network weights:
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        policy_state_dict = self.policy_dqn.state_dict()
        target_state_dict = self.target_dqn.state_dict()
        
        for key in policy_state_dict:
            target_state_dict[key] = (
                self.tau * policy_state_dict[key] + 
                (1.0 - self.tau) * target_state_dict[key]
            )
        self.target_dqn.load_state_dict(target_state_dict)

    def save(self, filepath):
        """
        Saves the policy network parameters.
        """
        torch.save(self.policy_dqn.state_dict(), filepath)

    def load(self, filepath):
        """
        Loads the policy network parameters and syncs the target network.
        """
        self.policy_dqn.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())