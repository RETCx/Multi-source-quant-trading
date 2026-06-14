import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer

@pytest.fixture
def tiny_dataloaders():
    """Create Tiny DataLoader for test (Batch=16, Sequence=15, Features=10)"""
    batch_size, seq_len, features, horizons = 16, 15, 10, 5
    
    # Dummy Data Train and Test
    X_dummy = torch.randn(batch_size * 2, seq_len, features)
    y_dummy = torch.randint(0, 2, size=(batch_size * 2, horizons), dtype=torch.float32)
    
    dataset = TensorDataset(X_dummy, y_dummy)
    
    # Split into Train/Test
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=range(0, batch_size))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=range(batch_size, batch_size * 2))
    
    return train_loader, test_loader, features

def test_trainer_smoke_test(tiny_dataloaders):
    """
    Test Trainer Loop (Forward, Backward, Evaluate)
    """
    train_loader, test_loader, features = tiny_dataloaders
    
    # 1. Initialize Model and Trainer (Set only 2 Epochs for speed)
    model = MultiHorizonLSTM(input_size=features, hidden_size=16)
    trainer = MultiHeadTrainer(model, epochs=2)
    
    # 2. Run Train
    fold_preds, fold_probas, fold_trues, train_accs, best_epoch, stopped_epoch = trainer.train_fold(train_loader, test_loader)
    
    # 3. Check Trainer Return 5 Heads
    assert len(fold_preds) == 5, "Trainer must return predictions for 5 heads"
    assert len(fold_probas) == 5, "Trainer must return probabilities for 5 heads"
    
    # 4. Check Basic Math not broken (Probabilities must be bounded [0, 1])
    for head_idx in range(5):
        probas = fold_probas[head_idx]
        assert all(0.0 <= p <= 1.0 for p in probas), "Probabilities must be bounded [0, 1]"