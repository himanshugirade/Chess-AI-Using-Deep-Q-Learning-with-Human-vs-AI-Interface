import pygame
import chess
import torch
import random

from model import DQN
from agent import Agent
from utils import encode_board, get_reward

# ---------- INIT ----------
pygame.init()
WIDTH, HEIGHT = 480, 480
SQ = WIDTH // 8

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI - Human vs RL")

# Colors (wood style)
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)

font = pygame.font.SysFont("Segoe UI Symbol", 56)
result_font = pygame.font.SysFont("Arial", 36)

# ---------- LOAD MODEL ----------
model = DQN()
model.load_state_dict(torch.load("chess_model.pth", map_location="cpu"))
model.eval()

agent = Agent(model)
agent.epsilon = 0.2   # thoda randomness (better moves)

board = chess.Board()

selected_square = None
human_color = chess.WHITE

# ---------- DRAW ----------
def draw_board():
    for row in range(8):
        for col in range(8):
            color = LIGHT if (row + col) % 2 == 0 else DARK
            pygame.draw.rect(WIN, color, (col * SQ, row * SQ, SQ, SQ))


def draw_highlight():
    if selected_square is not None:
        row = 7 - (selected_square // 8)
        col = selected_square % 8
        pygame.draw.rect(WIN, HIGHLIGHT,
                         (col * SQ, row * SQ, SQ, SQ), 4)


def draw_pieces():
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8

        color = (0, 0, 0) if piece.color else (255, 255, 255)

        text = font.render(piece.unicode_symbol(), True, color)
        rect = text.get_rect(center=(col * SQ + SQ // 2,
                                    row * SQ + SQ // 2))

        WIN.blit(text, rect)


# ---------- HELPERS ----------
def get_square_from_mouse(pos):
    x, y = pos
    col = x // SQ
    row = y // SQ
    return (7 - row) * 8 + col


def get_move(src, dst):
    move = chess.Move(src, dst)
    if move in board.legal_moves:
        return move

    # auto promotion
    move_q = chess.Move(src, dst, promotion=chess.QUEEN)
    if move_q in board.legal_moves:
        return move_q

    return None


# ---------- MAIN LOOP ----------
clock = pygame.time.Clock()
running = True

while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # -------- HUMAN MOVE --------
        if event.type == pygame.MOUSEBUTTONDOWN and board.turn == human_color:
            square = get_square_from_mouse(pygame.mouse.get_pos())

            if selected_square is None:
                piece = board.piece_at(square)
                if piece and piece.color == human_color:
                    selected_square = square
            else:
                move = get_move(selected_square, square)

                if move:
                    print("Human move:", move)
                    board.push(move)

                selected_square = None

    # -------- AI MOVE --------
    if board.turn != human_color and not board.is_game_over():

        state = encode_board(board)

        # mix RL + random (better gameplay)
        if random.random() < 0.3:
            move = random.choice(list(board.legal_moves))
        else:
            move = agent.choose_action(state, board)

        print("AI move:", move)

        board.push(move)

        reward = get_reward(board)
        print("Reward:", reward)

    # -------- DRAW --------
    WIN.fill((0, 0, 0))
    draw_board()
    draw_highlight()
    draw_pieces()

    # -------- GAME OVER DISPLAY --------
    if board.is_game_over():
        result = board.result()
        text = result_font.render("Game Over: " + result, True, (255, 0, 0))
        WIN.blit(text, (60, HEIGHT // 2))

    pygame.display.update()

pygame.quit()