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
    ฟังก์ชันกลางสำหรับลูป Walk-Forward Analysis (WFA)
    คืนค่าเป็น Dictionary ที่ประกอบด้วย:
    - ensembled_oos
    - raw_results (เก็บ indices, probas, trues, fold_ids ไว้เผื่อเซฟลง Pickle)
    - train_accs_mean (ค่าเฉลี่ย Train Accuracy ของแต่ละเป้าหมาย)
    """
    cv_splitter = WalkForwardCV(**cv_config)
    folds = cv_splitter.summary(len(X_raw))
    
    if not folds:
        print("[WARN] Not enough data for folds. Returning None.")
        return None

    n_heads = len(target_cols)
    
    global_oos_indices = []
    global_oos_probas  = {i: [] for i in range(n_heads)}
    global_oos_trues   = {i: [] for i in range(n_heads)}
    global_oos_fold_ids = {i: [] for i in range(n_heads)}
    global_train_accs  = {i: [] for i in range(n_heads)}

    set_seed(seed)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        set_seed(seed + fold_i)
        
        # 1. Scale Data
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_raw[train_idx])
        X_te = scaler.transform(X_raw[test_idx])
        
        # 2. Sequence Creation
        X_tr_seq, y_tr_seq = create_multi_horizon_sequences(X_tr, y_raw[train_idx], params['sequence_length'])
        X_te_seq, y_te_seq = create_multi_horizon_sequences(X_te, y_raw[test_idx], params['sequence_length'])
        
        if len(X_tr_seq) < params['batch_size'] or len(X_te_seq) < 5:
            if verbose:
                print(f"  Fold {fold_i+1:02d}/{len(folds):02d} | Skipped: Insufficient data")
            continue
            
        # 3. DataLoaders
        train_loader, test_loader = get_dataloaders(
            X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, params['batch_size'], seed=seed + fold_i
        )
        
        # 4. Initialize Model
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
        
        # 5. Train
        fold_preds, fold_probas, fold_trues, train_accs, best_ep, stop_ep = trainer.train_fold(train_loader, test_loader)
        
        if verbose:
            print(f"  Fold {fold_i+1:02d}/{len(folds):02d} | Best Epoch: {best_ep} | Acc: ", end="")
            for h_i in range(n_heads):
                test_acc = accuracy_score(fold_trues[h_i], fold_preds[h_i])
                print(f"{target_cols[h_i]}: {test_acc:.1%} ", end="")
            print()
            
        # 6. Record Fold Results
        actual_test_idx = test_idx[params['sequence_length']:]
        global_oos_indices.extend(actual_test_idx)
        
        for h_i in range(n_heads):
            global_oos_probas[h_i].extend(fold_probas[h_i])
            global_oos_trues[h_i].extend(fold_trues[h_i])
            global_train_accs[h_i].append(train_accs[h_i])
            global_oos_fold_ids[h_i].extend([fold_i + 1] * len(fold_preds[h_i]))

    # 7. Ensemble OOS Predictions
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
        'train_accs_mean': {target_cols[i]: np.mean(global_train_accs[i]) for i in range(n_heads)}
    }
