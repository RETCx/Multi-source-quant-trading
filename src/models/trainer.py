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

    def train_fold(self, train_loader, val_loader, test_loader=None):
        """
        Train 1 Fold with internal Validation early stopping, 
        and evaluate Out-of-Sample on Test Set.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (used for early stopping & checkpoint selection)
            test_loader: (Optional) Test DataLoader (used ONLY for final out-of-sample evaluation).
                         If test_loader is None, val_loader is used as evaluation (legacy mode).
        """
        eval_loader_for_early_stop = val_loader
        eval_loader_for_test = test_loader if test_loader is not None else val_loader
        
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
                
            # --- VALIDATION MODE (Early Stopping on Validation Set) ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in eval_loader_for_early_stop:
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
                    
        train_accs = {i: train_correct[i] / train_total if train_total > 0 else 0.0 for i in range(self.n_heads)}

        # --- Calculate Validation Accuracy ---
        val_accs = {i: 0.0 for i in range(self.n_heads)}
        if eval_loader_for_early_stop is not None:
            val_correct = {i: 0 for i in range(self.n_heads)}
            val_total = 0
            with torch.no_grad():
                for xb, yb in eval_loader_for_early_stop:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    outs = self.model(xb)
                    val_total += yb.size(0)
                    for i in range(self.n_heads):
                        preds = (outs[i].squeeze() > 0.5).int()
                        trues = yb[:, i].int()
                        val_correct[i] += (preds == trues).sum().item()
            val_accs = {i: val_correct[i] / val_total if val_total > 0 else 0.0 for i in range(self.n_heads)}
        
        # --- Collect Test Set Predictions (Out-of-sample) ---
        fold_preds  = {i: [] for i in range(self.n_heads)}
        fold_trues  = {i: [] for i in range(self.n_heads)}
        fold_probas = {i: [] for i in range(self.n_heads)}
        
        with torch.no_grad():
            for xb, yb in eval_loader_for_test:
                xb = xb.to(self.device)
                outs = self.model(xb)
                
                for i in range(self.n_heads):
                    probas = outs[i].squeeze().cpu().numpy()
                    preds  = (probas > 0.5).astype(int)
                    trues  = yb[:, i].cpu().numpy().astype(int)
                    
                    fold_probas[i].extend(probas)
                    fold_preds[i].extend(preds)
                    fold_trues[i].extend(trues)
                    
        return fold_preds, fold_probas, fold_trues, train_accs, val_accs, best_epoch_idx, stopped_epoch

    def fast_train(self, train_loader):
        """Train quickly on the entire dataset without validation/early-stopping (for Live inference)"""
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                
                outs = self.model(xb)
                
                loss = 0
                for i in range(self.n_heads):
                    loss += self.criterion(outs[i].squeeze(), yb[:, i])
                    
                loss.backward()
                self.optimizer.step()
        
        return self.model
