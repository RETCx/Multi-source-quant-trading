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


def get_dataloaders(
    X_train_seq, y_train_seq,
    X_val_seq=None, y_val_seq=None,
    X_test_seq=None, y_test_seq=None,
    batch_size=256,
    seed=42
):
    """
    Wrap Sequence Dataset into DataLoaders for Train, Validation, and Test.
    
    Supports:
      - 3-Way split (Train, Val, Test):
          get_dataloaders(X_tr, y_tr, X_val, y_val, X_te, y_te, batch_size=...)
          -> returns (train_loader, val_loader, test_loader)
      - Legacy 2-Way split (Train, Test):
          get_dataloaders(X_tr, y_tr, X_te, y_te, batch_size=...)
          -> returns (train_loader, test_loader)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dataset = QuantSequenceDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=g,
        drop_last=False
    )
    
    # Check if this is the legacy 4-argument call (train and test only)
    if X_test_seq is None and y_test_seq is None and X_val_seq is not None:
        test_dataset = QuantSequenceDataset(X_val_seq, y_val_seq)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        return train_loader, test_loader
        
    val_loader = None
    if X_val_seq is not None and y_val_seq is not None:
        val_dataset = QuantSequenceDataset(X_val_seq, y_val_seq)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
    test_loader = None
    if X_test_seq is not None and y_test_seq is not None:
        test_dataset = QuantSequenceDataset(X_test_seq, y_test_seq)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
    return train_loader, val_loader, test_loader
