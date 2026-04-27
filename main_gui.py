import pygame
import chess
import torch

from model import DQN
from agent import Agent
from utils import encode_board

# INIT
pygame.init()
WIDTH, HEIGHT = 480, 480
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI (RL)")

# COLORS
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# LOAD MODEL
model = DQN()
model.load_state_dict(torch.load("chess_model.pth"))
model.eval()

agent = Agent(model)
agent.epsilon = 0

board = chess.Board()

# FONT (improved)
font = pygame.font.SysFont("Segoe UI Symbol", 40)

def draw_board():
    square_size = WIDTH // 8
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(WIN, color,
                             (col * square_size, row * square_size,
                              square_size, square_size))

def draw_pieces():
    square_size = WIDTH // 8
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8

        color = (0, 0, 0) if piece.color else (255, 255, 255)
        text = font.render(piece.unicode_symbol(), True, color)

        WIN.blit(text, (col * square_size + 10, row * square_size + 5))

running = True
clock = pygame.time.Clock()
move_timer = 0

while running:
    clock.tick(60)
    move_timer += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # AI MOVE
    if move_timer > 60 and not board.is_game_over():
        state = encode_board(board)
        move = agent.choose_action(state, board)
        board.push(move)
        print("AI move:", move)
        move_timer = 0

    draw_board()
    draw_pieces()
    pygame.display.update()

pygame.quit()