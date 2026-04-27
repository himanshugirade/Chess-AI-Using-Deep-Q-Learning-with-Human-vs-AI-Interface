import chess
import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils import encode_board, get_reward
from agent import Agent
from model import DQN

model = DQN()
agent = Agent(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.9  # discount factor

for episode in range(100):

    board = chess.Board()
    total_reward = 0

    while not board.is_game_over():

        state = encode_board(board)

        # AI move
        move = agent.choose_action(state, board)
        board.push(move)

        if board.is_game_over():
            reward = get_reward(board)
            total_reward += reward
            break

        # Opponent move
        board.push(random.choice(list(board.legal_moves)))

        next_state = encode_board(board)
        reward = get_reward(board)
        total_reward += reward

        # 🔥 TRAINING STEP
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        q_values = model(state_tensor)
        next_q_values = model(next_state_tensor)

        target = reward + gamma * torch.max(next_q_values)
        predicted = torch.max(q_values)

        loss = loss_fn(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # epsilon decrease (less random over time)
    agent.epsilon = max(0.1, agent.epsilon * 0.995)

    print(f"Episode {episode} | Reward: {total_reward} | Epsilon: {agent.epsilon}")
    torch.save(model.state_dict(), "chess_model.pth")
print("Model saved!")