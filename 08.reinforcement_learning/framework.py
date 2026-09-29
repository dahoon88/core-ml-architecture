import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Pytorch Neural Network to approximate Q-values
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent using PyTorch and Experience Replay.
    Suitable for environments with continuous state spaces and discrete action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Q-Network
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        Select an action using the epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def learn(self):
        """
        Sample a batch from memory and update the Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
            target_q_values = rewards_tensor + (
                self.gamma * max_next_q_values * (1 - dones_tensor)
            )

        # Compute current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
