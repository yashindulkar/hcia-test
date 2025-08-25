import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mouse import ACTIONS, GRID_SIZE, EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC_CHEESE

class PolicyNetwork(nn.Module):
    """
    CNN-based policy network that processes grid states and outputs action probabilities
    """
    def __init__(self, grid_size=GRID_SIZE, num_actions=len(ACTIONS)):
        super(PolicyNetwork, self).__init__()
        
        self.grid_size = grid_size
        self.num_actions = num_actions
        
        # CNN layers for processing the grid
        # Input: 6 channels (one-hot encoding of each cell type)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 32 * grid_size * grid_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def grid_to_tensor(self, grid):
        """
        Convert grid to one-hot encoded tensor
        Grid elements: EMPTY=0, MOUSE=1, CHEESE=2, TRAP=3, WALL=4, ORGANIC_CHEESE=5
        """
        # Create one-hot encoding for each cell type
        batch_size = grid.shape[0] if len(grid.shape) == 3 else 1
        if len(grid.shape) == 2:
            grid = grid.unsqueeze(0)  # Add batch dimension
            
        one_hot = torch.zeros(batch_size, 6, self.grid_size, self.grid_size)
        
        for i in range(6):  # 6 different cell types
            one_hot[:, i, :, :] = (grid == i).float()
            
        return one_hot
        
    def forward(self, grid):
        """Forward pass through the network"""
        # Convert grid to tensor if it's not already
        if isinstance(grid, np.ndarray):
            grid = torch.FloatTensor(grid)
            
        # Ensure proper dimensions
        if len(grid.shape) == 2:
            grid = grid.unsqueeze(0)  # Add batch dimension
            
        # Convert to one-hot encoding
        x = self.grid_to_tensor(grid)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Output layer with softmax for action probabilities
        x = self.fc3(x)
        action_probs = F.softmax(x, dim=-1)
        
        return action_probs
    
    def select_action(self, grid, deterministic=False):
        """
        Select action based on current policy
        """
        action_probs = self.forward(grid)
        
        if deterministic:
            # Select action with highest probability
            action_idx = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from the probability distribution
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            
        return action_idx.item(), action_probs
    
    def get_action_log_prob(self, grid, action):
        """
        Get log probability of taking a specific action in a given state
        """
        action_probs = self.forward(grid)
        log_probs = torch.log(action_probs + 1e-8)  # Add small epsilon to avoid log(0)
        
        if isinstance(action, int):
            return log_probs[0, action]
        else:
            # Handle batch of actions
            batch_size = action_probs.shape[0]
            return log_probs[torch.arange(batch_size), action]


class RewardNetwork(nn.Module):
    """
    Neural network to learn reward function from human feedback (Bradley-Terry model)
    """
    def __init__(self, grid_size=GRID_SIZE):
        super(RewardNetwork, self).__init__()
        
        self.grid_size = grid_size
        
        # Similar architecture to policy network but outputs scalar reward
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        conv_output_size = 32 * grid_size * grid_size
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output single reward value
        
        self.dropout = nn.Dropout(0.2)
        
    def grid_to_tensor(self, grid):
        """Convert grid to one-hot encoded tensor (same as PolicyNetwork)"""
        batch_size = grid.shape[0] if len(grid.shape) == 3 else 1
        if len(grid.shape) == 2:
            grid = grid.unsqueeze(0)
            
        one_hot = torch.zeros(batch_size, 6, self.grid_size, self.grid_size)
        
        for i in range(6):
            one_hot[:, i, :, :] = (grid == i).float()
            
        return one_hot
        
    def forward(self, grid):
        """Forward pass to get reward for a state"""
        if isinstance(grid, np.ndarray):
            grid = torch.FloatTensor(grid)
            
        if len(grid.shape) == 2:
            grid = grid.unsqueeze(0)
            
        x = self.grid_to_tensor(grid)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Output reward
        reward = self.fc3(x)
        
        return reward
    
    def predict_trajectory_reward(self, trajectory_states):
        """
        Predict total reward for a trajectory
        trajectory_states: list of grid states
        """
        total_reward = 0
        for state in trajectory_states:
            reward = self.forward(state)
            total_reward += reward.item()
        return total_reward