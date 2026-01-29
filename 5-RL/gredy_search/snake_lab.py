# snake_lab.py
# ------------------------------------
# Snake Game RL Lab: Exploration vs Exploitation
# ------------------------------------

import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Minimal Snake Environment
# ------------------------------
class SnakeEnv:
    GRID_SIZE = 8
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.snake = [(0, 0)]
        self.food = (np.random.randint(0, self.GRID_SIZE), 
                     np.random.randint(0, self.GRID_SIZE))
        self.done = False
        self.score = 0
        return self.get_state()
    
    def get_state(self):
        # Simple state: snake head + food position
        head = self.snake[0]
        return np.array([head[0], head[1], self.food[0], self.food[1]])
    
    def step(self, action):
        if self.done:
            return self.get_state(), 0, self.done
        
        head_x, head_y = self.snake[0]
        if action == 0: head_y -= 1  # up
        if action == 1: head_y += 1  # down
        if action == 2: head_x -= 1  # left
        if action == 3: head_x += 1  # right
        
        new_head = (head_x, head_y)
        
        reward = -0.1
        if new_head == self.food:
            reward = 10
            self.score += 1
            self.food = (np.random.randint(0, self.GRID_SIZE), 
                         np.random.randint(0, self.GRID_SIZE))
            self.snake.insert(0, new_head)
        elif new_head in self.snake or not (0 <= head_x < self.GRID_SIZE and 0 <= head_y < self.GRID_SIZE):
            reward = -10
            self.done = True
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
        
        return self.get_state(), reward, self.done

# ------------------------------
# Simple ε-greedy Agent
# ------------------------------
class Agent:
    def __init__(self, n_actions=4, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
    
    def get_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(q_values)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

# ------------------------------
# Simulated Q-values (for demo only)
# ------------------------------
def random_q_values(state, n_actions=4):
    return np.random.random(n_actions)

# ------------------------------
# Main training loop
# ------------------------------
if __name__ == "__main__":
    env = SnakeEnv()
    agent = Agent()

    scores = []
    epsilons = []

    for episode in range(50):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            q_values = random_q_values(state)
            action = agent.get_action(q_values)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        scores.append(total_reward)
        epsilons.append(agent.epsilon)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Score per Episode")
    plt.plot(epsilons, label="Epsilon (Exploration Rate)")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Snake Game: Exploration vs Exploitation")
    plt.legend()
    plt.show()
