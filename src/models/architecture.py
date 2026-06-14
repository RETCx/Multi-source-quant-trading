import torch
import torch.nn as nn

class MultiHorizonLSTM(nn.Module):
    """
    1 Input (Sequence) -> LSTM (Shared Features) -> N Output Heads (Binary Classification)
    n_heads is dynamic and driven by config.yaml target_columns
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, n_heads=5):
        super(MultiHorizonLSTM, self).__init__()
        
        # 1. Shared Feature Extractor
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Multi-Heads (n_heads = len(target_columns) from config)
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_heads)
        ])

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        lstm_out, _ = self.lstm(x)
        
        # Last outputs of sequence
        last_out = self.layer_norm(lstm_out[:, -1, :])
        shared_features = self.dropout(last_out)
        
        # n_heads outputs (one per target horizon)
        outputs = [torch.sigmoid(head(shared_features)) for head in self.heads]
        
        return outputs # Return List of Tensor 5 horizons