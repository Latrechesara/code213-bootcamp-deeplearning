import random
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9  # 0=empty, 1=X, -1=O
        self.current_player = 1
        return tuple(self.board)

    def available_actions(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def step(self, action):
        if self.board[action] != 0:
            return tuple(self.board), -10, True, {}  # invalid move penalty

        self.board[action] = self.current_player
        winner = self.check_winner()

        if winner != 0:
            return tuple(self.board), 1 if winner == 1 else -1, True, {}

        if all(x != 0 for x in self.board):
            return tuple(self.board), 0, True, {}  # draw

        # switch player
        self.current_player *= -1
        return tuple(self.board), 0, False, {}

    def check_winner(self):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for i,j,k in wins:
            if self.board[i] != 0 and self.board[i] == self.board[j] == self.board[k]:
                return self.board[i]
        return 0


class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        qs = [self.get_Q(state, a) for a in actions]
        max_q = max(qs)
        best = [a for a, q in zip(actions, qs) if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_action):
        predict = self.get_Q(state, action)
        target = reward + self.gamma * self.get_Q(next_state, next_action)
        self.Q[(state, action)] = predict + self.alpha * (target - predict)


def run_episode_sarsa(env, agent, training=True):
    state = env.reset()
    done = False
    total_reward = 0

    # first action
    action = agent.choose_action(state, env.available_actions())

    while not done:
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if not done:
            next_action = agent.choose_action(next_state, env.available_actions())
            if training:
                agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
        else:
            if training:
                agent.update(state, action, reward, next_state, None)

    return total_reward, env.check_winner()


def train_sarsa(episodes=500):
    env = TicTacToe()
    agent = SARSAAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

    rewards, wins = [], []

    for ep in range(episodes):
        total_reward, winner = run_episode_sarsa(env, agent, training=True)
        rewards.append(total_reward)
        wins.append(1 if winner == 1 else 0)

    return agent, rewards, wins


def evaluate(agent, games=50):
    env = TicTacToe()
    agent.epsilon = 0.0  # no exploration
    wins = 0
    for _ in range(games):
        _, winner = run_episode_sarsa(env, agent, training=False)
        if winner == 1:
            wins += 1
    return wins


if __name__ == "__main__":
    episodes = 300
    agent, rewards, wins = train_sarsa(episodes)

    # Plot results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(rewards)
    plt.title("SARSA: Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1,2,2)
    win_rate = [sum(wins[:i+1])/(i+1) for i in range(len(wins))]
    plt.plot(win_rate)
    plt.title("SARSA: Win Rate")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")

    plt.show()

    # Evaluation
    test_wins = evaluate(agent, 50)
    print(f"SARSA Agent won {test_wins}/50 evaluation games")
