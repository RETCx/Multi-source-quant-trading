import pytest
import torch
from src.models.architecture import MultiHorizonLSTM

@pytest.fixture
def batch_size():
    return 32

@pytest.fixture
def seq_length():
    return 15

@pytest.fixture
def features():
    return 10

@pytest.fixture
def model(features):
    return MultiHorizonLSTM(input_size=features, hidden_size=64)

@pytest.fixture
def dummy_input(batch_size, seq_length, features):
    # Dummy data (Batch, Seq_Len, Features)
    return torch.randn(batch_size, seq_length, features)

def test_multi_horizon_lstm_shape(model, dummy_input, batch_size):
    outputs = model(dummy_input)
    
    # 1. must have 5 heads
    assert len(outputs) == 5, "Model must return 5 outputs for 5 horizons"
    
    # 2. each head must have shape (Batch, 1) and values between 0-1 (Sigmoid)
    for out in outputs:
        assert out.shape == (batch_size, 1), "Output shape must be (batch_size, 1)"
        assert torch.all((out >= 0) & (out <= 1)), "Outputs must be probabilities (0 to 1)"