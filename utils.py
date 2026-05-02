import numpy as np
import chess

# Convert the chess board into a numerical state (input for the model)
def encode_board(board):
    # Create an array of size 64 (one value per square)
    state = np.zeros(64)

    # Loop through all pieces on the board
    for square, piece in board.piece_map().items():
        value = piece.piece_type  # get piece type (pawn=1, knight=2, etc.)

        # Use negative value for black pieces
        if piece.color == chess.BLACK:
            value = -value

        # Store value in corresponding board position
        state[square] = value

    return state


# Define reward function for reinforcement learning
def get_reward(board):
    # If checkmate occurs → give high reward
    if board.is_checkmate():
        return 100

    # If stalemate → neutral reward
    elif board.is_stalemate():
        return 0

    # Otherwise → small reward to encourage progress
    else:
        return 0.1