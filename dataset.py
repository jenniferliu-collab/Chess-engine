import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Dataset class
# PyTorch requires our data to be wrapped in a Dataset class.
# We implement three methods:
#   __init__ load our data
#   __len__ how many examples do we have?
#   __getitem__ give us example number i

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


# 2. Create the dataset

dataset = ChessPositionDataset("positions.npy", "labels.npy")

print(f"Total positions: {len(dataset)}")

# Peek at one example
position, label = dataset[0]
print(f"Position tensor shape: {position.shape}")   # should be torch.Size([768])
print(f"Label: {label.item()}")                     # should be 1.0, 0.0, or -1.0


# 3. Split into train and validation sets
# We hold back 20% of data as a validation set.

total      = len(dataset)
train_size = int(0.8 * total)   # 80% for training
val_size   = total - train_size  # 20% for validation

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"\nTrain size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")


# 4. Create DataLoaders
# DataLoader sits on top of a Dataset and handles:
#   - Batching: grouping examples into chunks of batch_size
#   - Shuffling: randomizing order each epoch so the network
#                doesn't memorize the sequence
# batch_size=64 means the network sees 64 positions at once,
# computes predictions for all 64, measures the combined error,
# then updates its weights once. More efficient than one at a time.

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

print(f"\nBatches per epoch (train): {len(train_loader)}")
print(f"Batches per epoch (val):   {len(val_loader)}")


# 5. Peek inside a batch

positions_batch, labels_batch = next(iter(train_loader))

print(f"\nOne batch:")
print(f"  Positions shape: {positions_batch.shape}")  # torch.Size([64, 768])
print(f"  Labels shape:    {labels_batch.shape}")      # torch.Size([64])
print(f"  First 5 labels:  {labels_batch[:5]}")