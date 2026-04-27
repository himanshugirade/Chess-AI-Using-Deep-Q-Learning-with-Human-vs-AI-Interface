import random
import torch

class Agent:
    def __init__(self, model):
        self.model = model
        self.epsilon = 1.0   # exploration (start me random)

    def choose_action(self, state, board):
        # exploration (random move)
        if random.random() < self.epsilon:
            return random.choice(list(board.legal_moves))

        # exploitation (model se decision)
        state = torch.FloatTensor(state)
        q_values = self.model(state)

        moves = list(board.legal_moves)
        move_index = torch.argmax(q_values).item()

        return moves[move_index % len(moves)]