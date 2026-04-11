import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Dataset class
# PyTorch requires our data to be wrapped in a Dataset class:
#   1. __init__ loads our data
#   2. __len__ shows how many examples we have
#   3. __getitem__ give us example number i

class ChessPositionDataset(Dataset):
    
    def __init__(self, positions_path: str, labels_path: str):
        # Load the .npy files we saved in parse_games.py
        positions = np.load(positions_path)  # shape: (5832, 768)
        labels    = np.load(labels_path)     # shape: (5832,)
        
        # Convert numpy arrays to PyTorch tensors
        # float32 for positions (network input)
        # float32 for labels (we're predicting a number not a category)
        self.positions = torch.tensor(positions, dtype=torch.float32)
        self.labels    = torch.tensor(labels,    dtype=torch.float32)
        
    def __len__(self):
        # PyTorch calls this to know how many examples exist
        return len(self.labels)
    
    def __getitem__(self, idx):
        # PyTorch calls this to fetch one example by index
        # Returns a (position_vector, label) pair
        return self.positions[idx], self.labels[idx]