import random
import torch

# Agent class handles decision making for the AI
class Agent:
    def __init__(self, model):
        self.model = model
        self.epsilon = 1.0   # start with more random moves (exploration)

    def choose_action(self, state, board):
        # -------- EXPLORATION --------
        # with some probability, choose a random move
        if random.random() < self.epsilon:
            return random.choice(list(board.legal_moves))

        # -------- EXPLOITATION --------
        # otherwise, use the model to choose the best move
        
        # convert state into tensor (required for model input)
        state = torch.FloatTensor(state)
        
        # get Q-values (scores for each possible action)
        q_values = self.model(state)

        # get all legal moves from the current board
        moves = list(board.legal_moves)
        
        # pick the move with highest Q-value
        move_index = torch.argmax(q_values).item()

        # use modulo to make sure index fits within moves list
        return moves[move_index % len(moves)]