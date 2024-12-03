"""Graph Transformer based Agent class."""
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class TransformerAgent:
    """Graph Transformer based Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None


    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
