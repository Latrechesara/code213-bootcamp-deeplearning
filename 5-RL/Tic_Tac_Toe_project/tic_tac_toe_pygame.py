# tic_tac_toe_pygame.py

import pygame
import numpy as np
import random
import torch

# Import your agents
from dqn_agent import DQNAgent
from qlearning_agent import QLearningAgent
from sarsa_agent import SARSAAgent

# ---------------------------
# 1 - Settings and Initialization
# ---------------------------
pygame.init()
size = width, height = 300, 300
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tic-Tac-Toe vs RL Agent")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Initialize board: 0-empty, 1-player, -1-agent
board = np.zeros((3,3), dtype=int)

# Choose your agent: "dqn", "qlearning", "sarsa"
AGENT_TYPE = "qlearning"

if AGENT_TYPE == "dqn":
    model = DQNAgent()
elif AGENT_TYPE == "qlearning":
    model = QLearningAgent()
elif AGENT_TYPE == "sarsa":
    model = SARSAAgent()
else:
    raise ValueError("Unknown agent type")

# ---------------------------
# 2 - Draw Board Function
# ---------------------------
def draw_board(board):
    screen.fill(WHITE)
    # Grid lines
    for i in range(1, 3):
        pygame.draw.line(screen, BLACK, (0, i*100), (300, i*100), 3)
        pygame.draw.line(screen, BLACK, (i*100, 0), (i*100, 300), 3)
    # Draw X and O
    for i in range(3):
        for j in range(3):
            if board[i,j] == 1:
                pygame.draw.line(screen, BLUE, (j*100+10,i*100+10), (j*100+90,i*100+90), 5)
                pygame.draw.line(screen, BLUE, (j*100+90,i*100+10), (j*100+10,i*100+90), 5)
            elif board[i,j] == -1:
                pygame.draw.circle(screen, RED, (j*100+50,i*100+50), 40, 5)
    pygame.display.update()

# ---------------------------
# 3 - Player Move
# ---------------------------
def player_move(board, mouse_pos):
    row, col = mouse_pos[1] // 100, mouse_pos[0] // 100
    if board[row,col] == 0:
        board[row,col] = 1
        return True
    return False

# ---------------------------
# 4 - Agent Move
# ---------------------------
def agent_move(board, model):
    state = board.flatten()
    with torch.no_grad():
        q_values = model(torch.FloatTensor(state))
        action = int(torch.argmax(q_values).item())
    # If chosen cell is occupied, pick a random empty cell
    if board[action // 3, action % 3] != 0:
        empty_cells = [i for i in range(9) if board.flatten()[i] == 0]
        if empty_cells:
            action = random.choice(empty_cells)
    board[action // 3, action % 3] = -1

# ---------------------------
# 5 - Check Winner
# ---------------------------
def check_winner(board):
    lines = list(board) + list(board.T) + [board.diagonal(), np.fliplr(board).diagonal()]
    for line in lines:
        if np.all(line == 1):
            return 1
        elif np.all(line == -1):
            return -1
    if not np.any(board == 0):
        return 0  # Draw
    return None  # Game ongoing

# ---------------------------
# 6 - Game Loop
# ---------------------------
running = True
game_over = False

while running:
    draw_board(board)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            if player_move(board, event.pos):
                result = check_winner(board)
                if result is not None:
                    game_over = True
                    print("Game result:", "Player wins!" if result==1 else "Agent wins!" if result==-1 else "Draw!")
                    continue
                agent_move(board, model)
                result = check_winner(board)
                if result is not None:
                    game_over = True
                    print("Game result:", "Player wins!" if result==1 else "Agent wins!" if result==-1 else "Draw!")

pygame.quit()
