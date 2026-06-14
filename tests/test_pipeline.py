# tests/test_data_pipeline.py
import pytest
import numpy as np
import torch
from src.data_pipeline.walk_forward import WalkForwardCV
from src.data_pipeline.sequence import create_multi_horizon_sequences, get_dataloaders

@pytest.fixture
def dummy_data():
    """create dummy data (Fixture)"""
    N, FEATURES, HORIZONS = 200, 10, 5
    X = np.random.randn(N, FEATURES)
    y = np.random.randint(0, 2, size=(N, HORIZONS))
    return X, y

def test_walk_forward_expanding_window():
    """test walk forward expanding window"""
    wf = WalkForwardCV(min_train_size=50, test_size=20, gap=5, step_size=20)
    folds = wf.split(n_samples=200)
    
    assert len(folds) > 0, "Folds must not be empty"
    train_idx, test_idx = folds[0]
    
    #  Time-Series: Train  Test   Gap
    assert train_idx[-1] < test_idx[0], "Data Leakage! Train index overlaps Test index."
    assert test_idx[0] - train_idx[-1] - 1 == 5, "Gap size is incorrect."

def test_walk_forward_sliding_window():
    """test walk forward sliding window"""
    wf = WalkForwardCV(min_train_size=50, test_size=20, gap=5, step_size=20, max_train_size=30)
    folds = wf.split(n_samples=200)
    
    train_idx, _ = folds[1] #  Fold  2
    assert len(train_idx) == 30, "Sliding window failed to limit train size."

def test_create_multi_horizon_sequences(dummy_data):
    """test create multi horizon sequences"""
    X, y = dummy_data
    seq_length = 15
    X_seq, y_seq = create_multi_horizon_sequences(X, y, seq_length)
    
    # 
    assert X_seq.ndim == 3, "X_seq must be 3D (Samples, Seq_Len, Features)"
    assert X_seq.shape[1] == seq_length, "Sequence length mismatch"
    assert X_seq.shape[2] == 10, "Feature dimension mismatch"
    assert len(X_seq) == len(X) - seq_length, "Row count mismatch after sequencing"

def test_dataloaders(dummy_data):
    """test pytorch dataloaders"""
    X, y = dummy_data
    seq_length, batch_size = 15, 32
    X_seq, y_seq = create_multi_horizon_sequences(X, y, seq_length)
    
    split = 100
    train_loader, test_loader = get_dataloaders(
        X_seq[:split], y_seq[:split], 
        X_seq[split:], y_seq[split:], 
        batch_size=batch_size
    )
    
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape == (batch_size, seq_length, 10), "Train Loader Batch Shape incorrect"
    assert y_batch.shape == (batch_size, 5), "Label Batch Shape incorrect"