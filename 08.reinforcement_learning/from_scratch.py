import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning Agent built from scratch using the Bellman Equation.
    Suitable for environments with discrete state and action spaces.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """
        Select an action using the epsilon-greedy policy.
        """

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)

        return int(np.argmax(self.q_table[state, :]))

    def learn(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ):
        """
        Update the Q-table using the Bellman Equation.
        """

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state, :])

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

        if done and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, state: int) -> int:

        return int(np.argmax(self.q_table[state, :]))
