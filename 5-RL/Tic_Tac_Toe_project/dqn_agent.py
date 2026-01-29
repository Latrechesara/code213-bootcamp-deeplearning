# dqn_agent.py

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ------------------ Tic-Tac-Toe Environment ------------------
class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten()
    
    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]
    
    def step(self, action, player):
        if self.board.flatten()[action] != 0:
            return self.board.flatten(), -10, True  # Illegal move penalty
        self.board[action // 3, action % 3] = player
        reward, self.done, self.winner = self.check_game(player)
        return self.board.flatten(), reward, self.done
    
    def check_game(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return 1, True, player
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player or \
           self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return 1, True, player
        if np.all(self.board != 0):
            return 0, True, 0  # Draw
        return 0, False, None

# ------------------ DQN Model ------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ Replay Buffer ------------------
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# ------------------ Training Loop ------------------
def train_dqn(env, episodes=1000, batch_size=64, gamma=0.99, lr=0.001,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    state_size = 9
    action_size = 9
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    
    epsilon = epsilon_start
    all_rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(env.available_actions())
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = int(torch.argmax(q_values).item())
                    if action not in env.available_actions():
                        action = random.choice(env.available_actions())
            
            next_state, reward, done = env.step(action, player=1)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                next_states = torch.FloatTensor(next_states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)
                
                q_values = model(states).gather(1, actions)
                with torch.no_grad():
                    q_next = model(next_states).max(1)[0]
                    q_target = rewards + gamma * q_next * (1 - dones)
                
                loss = criterion(q_values.squeeze(), q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    return model, all_rewards

# ------------------ Plotting ------------------
def plot_rewards(all_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label='Episode Reward')
    window = 50
    if len(all_rewards) >= window:
        moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(all_rewards)), moving_avg, label=f'{window}-Episode Moving Avg', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Rewards Over Episodes')
    plt.legend()
    plt.show()

# ------------------ Evaluation ------------------
def evaluate(env, model, episodes=100):
    wins, draws, losses = 0, 0, 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state))
                action = int(torch.argmax(q_values).item())
                if action not in env.available_actions():
                    action = random.choice(env.available_actions())
            state, reward, done = env.step(action, player=1)
            if not done:
                opp_action = random.choice(env.available_actions())
                state, reward, done = env.step(opp_action, player=-1)
        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

# ------------------ DQN Agent Wrapper ------------------
class DQNAgent:
    def __init__(self, model=None, env=None, episodes=500):
        """
        Wraps the trained DQN for easy integration with Pygame.
        If no model is provided, trains a new one.
        """
        if model is not None:
            self.model = model
        else:
            if env is None:
                env = TicTacToe()
            self.model, _ = train_dqn(env, episodes=episodes)

    def __call__(self, state):
        """
        Input: flattened board
        Output: Q-values tensor
        """
        with torch.no_grad():
            return self.model(torch.FloatTensor(state))

    def select_action(self, state, available_actions):
        """
        Returns an action index, avoiding illegal moves.
        """
        q_values = self(state)
        action = int(torch.argmax(q_values).item())
        if action not in available_actions:
            action = random.choice(available_actions)
        return action

# ------------------ Example Usage ------------------
if __name__ == "__main__":
    env = TicTacToe()
    model, all_rewards = train_dqn(env, episodes=500)
    plot_rewards(all_rewards)
    evaluate(env, model, episodes=100)
