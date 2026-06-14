import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def create_multi_horizon_sequences(X, y, seq_length):
    """
    Change Time-Series 2D to Sequence 3D for LSTM 
    and pair with Multi-Horizon Targets (5 targets)
    
    Args:
        X (np.array): Features (N, features)
        y (np.array): Targets (N, horizons) 
        seq_length (int): Size of Time Window (15 days)
        
    Returns:
        X_seq (np.array): Shape (N - seq_length, seq_length, features)
        y_seq (np.array): Shape (N - seq_length, horizons)
    """
    Xs, ys = [], []
    
    # Loop slide window 
    for i in range(len(X) - seq_length):
        Xs.append(X[i : i + seq_length])
        # y will be the value of the day after the Sequence 
        ys.append(y[i + seq_length])
        
    return np.array(Xs), np.array(ys)


class QuantSequenceDataset(Dataset):
    """
    Convert NumPy data to PyTorch Dataset
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Because the Output will be used in BCE Loss, it must be float32 instead of Long
        self.y = torch.tensor(y, dtype=torch.float32) 
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train_seq, y_train_seq, X_test_seq, y_test_seq, batch_size=256, seed=42):
    """
    Wrap Sequence Dataset into DataLoader for use in Train loop
    """
    train_dataset = QuantSequenceDataset(X_train_seq, y_train_seq)
    test_dataset = QuantSequenceDataset(X_test_seq, y_test_seq)
    
    # Lock Seed of DataLoader Generator to make the result reproducible
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Train Loader must Shuffle to prevent the model from remembering the data order
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=g,
        drop_last=False
    )
    
    # Test Loader Do Not Shuffle to keep OOS time frame for verification
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader