import torch
import torch.nn as nn
import copy
import numpy as np

class MultiHeadTrainer:
    def __init__(self, model, learning_rate=0.001, patience=15, epochs=100, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.patience = patience
        self.epochs = epochs
        
        # Binary Cross Entropy for Sigmoid output
        self.criterion = nn.BCELoss() 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.n_heads = len(model.heads)  # dynamic: driven by config

    def train_fold(self, train_loader, test_loader):
        """Train 1 Fold and return Test Set Prediction (Out-of-Sample)"""
        best_loss = float('inf')
        patience_counter = 0
        best_epoch_idx = 0
        best_weights = copy.deepcopy(self.model.state_dict())
        
        stopped_epoch = self.epochs - 1
        for epoch in range(self.epochs):
            # --- TRAIN MODE ---
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                
                outs = self.model(xb)
                
                # Sum Loss from all heads
                loss = 0
                for i in range(self.n_heads):
                    loss += self.criterion(outs[i].squeeze(), yb[:, i])
                    
                loss.backward()
                self.optimizer.step()
                
            # --- EVALUATION MODE (Early Stopping) ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    outs = self.model(xb)
                    for i in range(self.n_heads):
                        val_loss += self.criterion(outs[i].squeeze(), yb[:, i]).item()
                        
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch_idx = epoch
                patience_counter = 0
                best_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    stopped_epoch = epoch
                    break # Early stop trigger
                    
        # Load best model weights for this fold
        self.model.load_state_dict(best_weights)
        self.model.eval()

        # --- Calculate Train Accuracy (In-Sample) ---
        train_correct = {i: 0 for i in range(self.n_heads)}
        train_total = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outs = self.model(xb)
                train_total += yb.size(0)
                
                for i in range(self.n_heads):
                    preds = (outs[i].squeeze() > 0.5).int()
                    trues = yb[:, i].int()
                    train_correct[i] += (preds == trues).sum().item()
                    
        train_accs = {i: train_correct[i] / train_total for i in range(self.n_heads)}
        
        # Collect Test Set Predictions (Out-of-sample)
        fold_preds  = {i: [] for i in range(self.n_heads)}
        fold_trues  = {i: [] for i in range(self.n_heads)}
        fold_probas = {i: [] for i in range(self.n_heads)}
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                outs = self.model(xb)
                
                for i in range(self.n_heads):
                    probas = outs[i].squeeze().cpu().numpy()
                    preds  = (probas > 0.5).astype(int)
                    trues  = yb[:, i].cpu().numpy().astype(int)
                    
                    fold_probas[i].extend(probas)
                    fold_preds[i].extend(preds)
                    fold_trues[i].extend(trues)
                    
        return fold_preds, fold_probas, fold_trues, train_accs, best_epoch_idx, stopped_epoch