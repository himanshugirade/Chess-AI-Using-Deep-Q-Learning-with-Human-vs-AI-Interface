import chess
import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils import encode_board, get_reward
from agent import Agent
from model import DQN

# Initialize model and agent
model = DQN()
agent = Agent(model)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.9  # discount factor (importance of future rewards)

# Training loop (episodes)
for episode in range(100):

    board = chess.Board()  # start a new game
    total_reward = 0

    # Play until game ends
    while not board.is_game_over():

        # Convert board into numerical state
        state = encode_board(board)

        # -------- AI MOVE --------
        move = agent.choose_action(state, board)
        board.push(move)

        # If game ends after AI move
        if board.is_game_over():
            reward = get_reward(board)
            total_reward += reward
            break

        # -------- OPPONENT MOVE (random) --------
        board.push(random.choice(list(board.legal_moves)))

        # Get next state after opponent move
        next_state = encode_board(board)
        reward = get_reward(board)
        total_reward += reward

        # -------- TRAINING STEP --------
        
        # Convert states to tensors
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        # Get Q-values for current and next state
        q_values = model(state_tensor)
        next_q_values = model(next_state_tensor)

        # Bellman equation (target value)
        target = reward + gamma * torch.max(next_q_values)
        
        # Predicted Q-value
        predicted = torch.max(q_values)

        # Calculate loss
        loss = loss_fn(predicted, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Reduce exploration over time (epsilon decay)
    agent.epsilon = max(0.1, agent.epsilon * 0.995)

    # Print training progress
    print(f"Episode {episode} | Reward: {total_reward} | Epsilon: {agent.epsilon}")

    # Save model after each episode
    torch.save(model.state_dict(), "chess_model.pth")

print("Model saved!")