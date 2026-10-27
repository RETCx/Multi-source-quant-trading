# tests/test_data_pipeline.py
import pytest
import numpy as np
import pandas as pd
import torch
from src.data_pipeline.walk_forward import WalkForwardCV
from src.data_pipeline.sequence import create_multi_horizon_sequences, get_dataloaders
from src.features.fundamentals import merge_and_calc_fundamentals

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
    
    assert train_idx[-1] < test_idx[0], "Data Leakage! Train index overlaps Test index."
    assert test_idx[0] - train_idx[-1] - 1 == 5, "Gap size is incorrect."

def test_walk_forward_sliding_window():
    """test walk forward sliding window"""
    wf = WalkForwardCV(min_train_size=50, test_size=20, gap=5, step_size=20, max_train_size=30)
    folds = wf.split(n_samples=200)
    
    train_idx, _ = folds[1]
    assert len(train_idx) == 30, "Sliding window failed to limit train size."

def test_create_multi_horizon_sequences(dummy_data):
    """test create multi horizon sequences"""
    X, y = dummy_data
    seq_length = 15
    X_seq, y_seq = create_multi_horizon_sequences(X, y, seq_length)
    
    assert X_seq.ndim == 3, "X_seq must be 3D (Samples, Seq_Len, Features)"
    assert X_seq.shape[1] == seq_length, "Sequence length mismatch"
    assert X_seq.shape[2] == 10, "Feature dimension mismatch"
    assert len(X_seq) == len(X) - seq_length, "Row count mismatch after sequencing"

def test_dataloaders_legacy_2way(dummy_data):
    """test pytorch dataloaders legacy 2-way split"""
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
    assert test_loader is not None

def test_dataloaders_3way(dummy_data):
    """test pytorch dataloaders 3-way split (Train, Val, Test)"""
    X, y = dummy_data
    seq_length, batch_size = 15, 16
    X_seq, y_seq = create_multi_horizon_sequences(X, y, seq_length)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        X_seq[:80], y_seq[:80],
        X_seq[80:120], y_seq[80:120],
        X_seq[120:], y_seq[120:],
        batch_size=batch_size
    )
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

def test_prepended_test_sequence_alignment():
    """
    Verify test sequence alignment:
    When prepending sequence_length items to a 60-day test set,
    the resulting sequence count MUST be exactly 60 (matching all 60 test days).
    """
    seq_length = 15
    test_size = 60
    total_features = 8
    
    X_raw = np.random.randn(200, total_features)
    y_raw = np.random.randint(0, 2, size=(200, 5))
    
    test_idx = np.arange(100, 100 + test_size)
    prepended_idx = np.arange(test_idx[0] - seq_length, test_idx[0])
    
    X_te = X_raw[test_idx]
    X_prepended = X_raw[prepended_idx]
    X_te_full = np.concatenate([X_prepended, X_te])
    
    y_te = y_raw[test_idx]
    y_prepended = y_raw[prepended_idx]
    y_te_full = np.concatenate([y_prepended, y_te])
    
    X_te_seq, y_te_seq = create_multi_horizon_sequences(X_te_full, y_te_full, seq_length)
    
    # Crucial test: Must match exact test size (60 days)
    assert len(X_te_seq) == test_size, f"Expected {test_size} sequences, got {len(X_te_seq)}"
    assert len(y_te_seq) == test_size, f"Expected {test_size} targets, got {len(y_te_seq)}"

def test_fundamentals_reporting_lag_prevents_leakage():
    """
    Verify that applying reporting_lag_days prevents look-ahead bias:
    An earnings report on 2024-03-31 with 45-day lag must NOT be visible on 2024-04-15!
    """
    price_dates = pd.date_range("2024-01-01", "2024-06-01", freq="B")
    price_df = pd.DataFrame({
        "adjClose": np.random.uniform(100, 150, size=len(price_dates))
    }, index=price_dates)
    
    fund_df = pd.DataFrame({
        "Net_Income": [1_000_000, 2_000_000],
        "Total_Equity": [10_000_000, 12_000_000],
        "Shares_Outstanding": [100_000, 100_000]
    }, index=pd.to_datetime(["2023-12-31", "2024-03-31"]))
    
    # Merge with 45 days lag
    merged_df = merge_and_calc_fundamentals(price_df, fund_df, reporting_lag_days=45)
    
    # On 2024-04-15 (15 days after Q1 ends), Q1 2024 report should NOT be active yet!
    # It should still be the 2023-12-31 value (Net_Income 1,000,000)
    april_val = merged_df.loc["2024-04-15", "Net_Income"]
    assert april_val == 1_000_000, f"Look-ahead bias detected! Net income on 2024-04-15 was {april_val}"
    
    # On 2024-05-20 (after 45 days), Q1 2024 report should now be active (Net_Income 2,000,000)
    may_val = merged_df.loc["2024-05-20", "Net_Income"]
    assert may_val == 2_000_000, f"Expected 2024-03-31 report to be active by 2024-05-20, got {may_val}"
