# Chess AI using Deep Q-Learning

## Overview
This project implements a Chess AI using Reinforcement Learning (RL) and a Deep Q-Network (DQN).  
The AI learns to play chess by interacting with the environment and improving its decisions based on rewards.

---

## Features
- Chess AI using Deep Q-Learning
- Human vs AI gameplay using Pygame
- Epsilon-greedy strategy for action selection
- Board state encoding for neural network input
- Model training with reward-based learning

---

## Technologies Used
- Python
- PyTorch
- Pygame
- NumPy
- python-chess

---

How It Works
The board is converted into a 64-length numerical vector
The DQN model predicts Q-values for all possible moves
The agent selects moves using epsilon-greedy strategy
Rewards guide the learning process
The model improves over multiple training episodes
Reward System
Checkmate → +100
Stalemate → 0
Normal move → +0.1
