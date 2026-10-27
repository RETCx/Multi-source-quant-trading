import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import torch

from src.data_pipeline.walk_forward import WalkForwardCV
from src.data_pipeline.sequence import create_multi_horizon_sequences, get_dataloaders
from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer
from src.evaluation.ensemble import ensemble_overlapping_predictions
from src.utils import set_seed

def run_walk_forward_pipeline(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    features: list,
    target_cols: list,
    params: dict,
    cv_config: dict,
    device: torch.device,
    seed: int,
    verbose: bool = True
) -> dict:
    """
    Time-Series Walk-Forward Analysis (WFA) with:
    - Internal chronological Train/Validation split for Early Stopping (Zero Test Leakage)
    - Full prepended lookback sequences for seamless Test set alignment (1:1 date match)
    - Multi-head ensemble out-of-sample predictions
    """
    cv_splitter = WalkForwardCV(**cv_config)
    folds = cv_splitter.summary(len(X_raw))
    
    if not folds:
        print("[WARN] Not enough data for folds. Returning None.")
        return None

    n_heads = len(target_cols)
    seq_len = params['sequence_length']
    
    global_oos_indices = []
    global_oos_probas  = {i: [] for i in range(n_heads)}
    global_oos_trues   = {i: [] for i in range(n_heads)}
    global_oos_fold_ids = {i: [] for i in range(n_heads)}
    global_train_accs  = {i: [] for i in range(n_heads)}
    global_val_accs    = {i: [] for i in range(n_heads)}

    set_seed(seed)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        set_seed(seed + fold_i)
        
        # 1. Chronological Train / Validation Split inside Fold
        # train_idx is partitioned into sub_train and internal val
        n_train = len(train_idx)
        val_size = min(max(int(n_train * 0.15), 30), 60)
        val_gap = min(cv_config.get('gap', 20), 20)
        
        # Safety check if train_idx is small
        if n_train - val_size - val_gap < params['batch_size']:
            val_size = max(int(n_train * 0.10), 15)
            val_gap = min(cv_config.get('gap', 10), 10)
            
        val_split_point = n_train - val_size
        sub_train_end = val_split_point - val_gap
        
        sub_train_idx = train_idx[:sub_train_end]
        val_idx = train_idx[val_split_point:]
        
        # 2. Scale Data strictly on sub_train_idx to prevent leakage into val and test
        scaler = RobustScaler()
        X_sub_tr = scaler.fit_transform(X_raw[sub_train_idx])
        X_val = scaler.transform(X_raw[val_idx])
        X_te = scaler.transform(X_raw[test_idx])
        
        # 3. Create Sequences
        # Sub-train sequences
        X_tr_seq, y_tr_seq = create_multi_horizon_sequences(
            X_sub_tr, y_raw[sub_train_idx], seq_len
        )
        
        # Validation sequences: prepend lookback points from gap/sub_train
        prep_val_idx = np.arange(val_idx[0] - seq_len, val_idx[0])
        prep_val_idx = prep_val_idx[prep_val_idx >= 0]
        X_val_prep = scaler.transform(X_raw[prep_val_idx])
        X_val_full = np.concatenate([X_val_prep, X_val])
        y_val_prep = y_raw[prep_val_idx]
        y_val_full = np.concatenate([y_val_prep, y_raw[val_idx]])
        X_val_seq, y_val_seq = create_multi_horizon_sequences(X_val_full, y_val_full, seq_len)
        
        # Test sequences: prepend lookback points from outer gap/train
        prep_test_idx = np.arange(test_idx[0] - seq_len, test_idx[0])
        prep_test_idx = prep_test_idx[prep_test_idx >= 0]
        X_test_prep = scaler.transform(X_raw[prep_test_idx])
        X_te_full = np.concatenate([X_test_prep, X_te])
        y_test_prep = y_raw[prep_test_idx]
        y_te_full = np.concatenate([y_test_prep, y_raw[test_idx]])
        X_te_seq, y_te_seq = create_multi_horizon_sequences(X_te_full, y_te_full, seq_len)
        
        if len(X_tr_seq) < params['batch_size'] or len(X_te_seq) < 5:
            if verbose:
                print(f"  Fold {fold_i+1:02d}/{len(folds):02d} | Skipped: Insufficient data")
            continue
            
        # 4. DataLoaders (Train, Val, Test)
        train_loader, val_loader, test_loader = get_dataloaders(
            X_tr_seq, y_tr_seq,
            X_val_seq, y_val_seq,
            X_te_seq, y_te_seq,
            batch_size=params['batch_size'],
            seed=seed + fold_i
        )
        
        # 5. Initialize Model
        model = MultiHorizonLSTM(
            input_size=len(features),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            n_heads=n_heads,
        )
        trainer = MultiHeadTrainer(
            model, params['learning_rate'], params['patience'], params['epochs'], device
        )
        
        # 6. Train with Validation Early Stopping & Evaluate purely on Test Set
        fold_preds, fold_probas, fold_trues, train_accs, val_accs, best_ep, stop_ep = trainer.train_fold(
            train_loader, val_loader, test_loader
        )
        
        if verbose:
            print(f"  Fold {fold_i+1:02d}/{len(folds):02d} | Best Ep: {best_ep} | Val Acc: ", end="")
            for h_i in range(n_heads):
                print(f"{target_cols[h_i]}: {val_accs[h_i]:.1%} ", end="")
            print(" | Test Acc: ", end="")
            for h_i in range(n_heads):
                test_acc = accuracy_score(fold_trues[h_i], fold_preds[h_i])
                print(f"{target_cols[h_i]}: {test_acc:.1%} ", end="")
            print()
            
        # 7. Record Fold Results
        # Prepended lookback ensures X_te_seq length exactly matches test_idx length!
        actual_test_idx = test_idx
        assert len(actual_test_idx) == len(X_te_seq), (
            f"Index-sequence length mismatch: {len(actual_test_idx)} vs {len(X_te_seq)}"
        )
        global_oos_indices.extend(actual_test_idx)
        
        for h_i in range(n_heads):
            assert len(fold_probas[h_i]) == len(actual_test_idx), (
                f"Mismatch: {len(fold_probas[h_i])} preds vs {len(actual_test_idx)} indices"
            )
            global_oos_probas[h_i].extend(fold_probas[h_i])
            global_oos_trues[h_i].extend(fold_trues[h_i])
            global_train_accs[h_i].append(train_accs[h_i])
            global_val_accs[h_i].append(val_accs[h_i])
            global_oos_fold_ids[h_i].extend([fold_i + 1] * len(fold_preds[h_i]))

    # 8. Ensemble OOS Predictions
    ensembled_oos = {}
    for h_i, t_name in enumerate(target_cols):
        f_preds, f_probas, f_trues, u_indices = ensemble_overlapping_predictions(
            global_oos_indices,
            global_oos_probas[h_i],
            global_oos_trues[h_i],
            global_oos_fold_ids[h_i],
        )
        ensembled_oos[t_name] = {
            'indices': u_indices,
            'preds':   f_preds,
            'probas':  f_probas,
            'trues':   f_trues,
        }
        
    return {
        'ensembled_oos': ensembled_oos,
        'raw_results': {
            'indices': global_oos_indices,
            'probas': global_oos_probas,
            'trues': global_oos_trues,
            'fold_ids': global_oos_fold_ids,
        },
        'train_accs_mean': {target_cols[i]: np.mean(global_train_accs[i]) if global_train_accs[i] else 0.0 for i in range(n_heads)},
        'val_accs_mean': {target_cols[i]: np.mean(global_val_accs[i]) if global_val_accs[i] else 0.0 for i in range(n_heads)}
    }
