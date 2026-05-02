import torch.nn as nn
import torch

# Deep Q-Network (DQN) model for predicting move values
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First fully connected layer (input: 64 features, output: 128)
        self.fc1 = nn.Linear(64, 128)
        
        # Second hidden layer (128 → 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Output layer (128 → 4672 possible moves in chess)
        self.fc3 = nn.Linear(128, 4672)

    def forward(self, x):
        # Pass input through first layer + activation
        x = torch.relu(self.fc1(x))
        
        # Pass through second layer + activation
        x = torch.relu(self.fc2(x))
        
        # Final output (Q-values for all possible moves)
        return self.fc3(x)