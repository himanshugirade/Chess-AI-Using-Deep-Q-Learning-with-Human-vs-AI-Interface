import numpy as np
import chess

def encode_board(board):
    state = np.zeros(64)

    for square, piece in board.piece_map().items():
        value = piece.piece_type
        if piece.color == chess.BLACK:
            value = -value
        state[square] = value

    return state


def get_reward(board):
    if board.is_checkmate():
        return 100
    elif board.is_stalemate():
        return 0
    else:
        return 0.1