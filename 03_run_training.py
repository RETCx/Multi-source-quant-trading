import os
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

from src.data_pipeline.walk_forward import WalkForwardCV
from src.data_pipeline.sequence import create_multi_horizon_sequences, get_dataloaders
from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer
from src.evaluation.ensemble import ensemble_overlapping_predictions
from src.evaluation.metrics import evaluate_and_report, create_model_comparison_df
from src.utils import set_seed

# ==========================================
# 0. LOAD CONFIG (Single Source of Truth)
# ==========================================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SEED = config['random_seed']
set_seed(SEED)

STOCK_SYMBOL = config['data_settings']['selected_tickers'][0]
DATA_PATH = f"{config['data_path']['features']}/{STOCK_SYMBOL}_with_indicators.csv"

TARGET_COLS = config['target_columns']
EXCLUDE_COLS = config.get('exclude_columns', [])
PARAMS = config['model_params']

CV_CONFIG = {
    'min_train_size': config['walk_forward']['min_train_size'],
    'test_size': config['walk_forward']['test_size'],
    'gap': config['walk_forward']['gap'],
    'step_size': config['walk_forward']['step_size'],
}
max_train = config['walk_forward'].get('max_train_size')
if max_train is not None:
    CV_CONFIG['max_train_size'] = max_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Seed: {SEED}")

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
print(f"\n[{STOCK_SYMBOL}] Loading data and preparing features...")
df = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
df = df.dropna(subset=TARGET_COLS)

# Filter features: exclude targets + excluded columns from config
features = [col for col in df.columns if col not in TARGET_COLS + EXCLUDE_COLS]
X_raw = df[features].values
y_raw = df[TARGET_COLS].values 

print(f"Data Shape: {X_raw.shape[0]} rows, {X_raw.shape[1]} features")
print(f"Excluded: {len(EXCLUDE_COLS)} columns | Targets: {len(TARGET_COLS)}")

# ==========================================
# 2. WALK-FORWARD SPLITTER
# ==========================================
cv_splitter = WalkForwardCV(**CV_CONFIG)
folds = cv_splitter.summary(len(X_raw))

if not folds:
    raise ValueError("Not enough data to create Walk-Forward Folds!")

# Global storage for OOS results (for Ensemble across folds)
global_oos_indices = []
global_oos_probas = {i: [] for i in range(len(TARGET_COLS))}
global_oos_trues = {i: [] for i in range(len(TARGET_COLS))}
global_train_accs = {i: [] for i in range(len(TARGET_COLS))}
global_oos_fold_ids = {i: [] for i in range(len(TARGET_COLS))}
# ==========================================
# 3. TRAINING LOOP
# ==========================================
print("\n" + "="*50)
print(f"STARTING MULTI-HEAD LSTM TRAINING")
print("="*50)

n_heads = len(TARGET_COLS)

for fold_i, (train_idx, test_idx) in enumerate(folds):
    print(f"\n--- Fold {fold_i+1}/{len(folds)} | Train: {len(train_idx)} rows, Test: {len(test_idx)} rows ---")
    
    # Reset seed per fold for reproducibility (same as notebook)
    set_seed(SEED + fold_i)
    
    # 3.1 Scaling (Fit on Train, Transform Test)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_raw[train_idx])
    X_test_scaled = scaler.transform(X_raw[test_idx])
    
    # 3.2 Create 3D Sequences
    X_tr_seq, y_tr_seq = create_multi_horizon_sequences(X_train_scaled, y_raw[train_idx], PARAMS['sequence_length'])
    X_te_seq, y_te_seq = create_multi_horizon_sequences(X_test_scaled, y_raw[test_idx], PARAMS['sequence_length'])
    
    if len(X_tr_seq) < PARAMS['batch_size'] or len(X_te_seq) < 5:
        print("Skipped: Insufficient data after sequencing.")
        continue
        
    # 3.3 Create DataLoaders
    train_loader, test_loader = get_dataloaders(
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, 
        PARAMS['batch_size'], seed=SEED + fold_i
    )
    
    # 3.4 Create Model and Trainer
    model = MultiHorizonLSTM(
        input_size=len(features), 
        hidden_size=PARAMS['hidden_size'], 
        num_layers=PARAMS['num_layers'], 
        dropout=PARAMS['dropout']
    )
    
    trainer = MultiHeadTrainer(
        model, 
        learning_rate=PARAMS['learning_rate'], 
        patience=PARAMS['patience'], 
        epochs=PARAMS['epochs'], 
        device=device
    )
    
    # 3.5 Train and collect predictions
    fold_preds, fold_probas, fold_trues, train_accs, best_epoch, stopped_epoch = trainer.train_fold(train_loader, test_loader)
    
    # Calculate Test Acc for this fold
    test_accs = {i: accuracy_score(fold_trues[i], fold_preds[i]) for i in range(n_heads)}
    
    print(f"  -> Best Model at Epoch {best_epoch+1} (Stopped at Epoch {stopped_epoch+1})")
    for head_idx, target_name in enumerate(TARGET_COLS):
        print(f"     {target_name:10s} | Train Acc: {train_accs[head_idx]*100:.1f}% | Test Acc: {test_accs[head_idx]*100:.1f}%")
        
    # 3.6 Store OOS results (adjust index for sequence offset)
    actual_test_idx = test_idx[PARAMS['sequence_length']:]
    
    global_oos_indices.extend(actual_test_idx)
    for head_idx in range(n_heads):
        global_oos_probas[head_idx].extend(fold_probas[head_idx])
        global_oos_trues[head_idx].extend(fold_trues[head_idx])
        global_train_accs[head_idx].append(train_accs[head_idx])
        global_oos_fold_ids[head_idx].extend([fold_i+1] * len(fold_preds[head_idx]))

# ==========================================
# 4. ENSEMBLE & EVALUATION
# ==========================================
print("\n" + "="*50)
print(f"FINAL OUT-OF-SAMPLE EVALUATION")
print("="*50)

results_summary = {}

for head_idx, target_name in enumerate(TARGET_COLS):
    f_preds, f_probas, f_trues, u_indices = ensemble_overlapping_predictions(
        global_oos_indices, 
        global_oos_probas[head_idx], 
        global_oos_trues[head_idx],
        global_oos_fold_ids[head_idx]
    )
    
    acc = evaluate_and_report(f_trues, f_preds, target_name)
    
    results_summary[target_name] = {
        'train_acc': np.mean(global_train_accs[head_idx]),
        'test_acc': acc,
        'avg_confidence': np.mean(f_probas)
    }

# Summary table
df_summary = create_model_comparison_df(results_summary)
df_summary.to_csv(f"model_summary_{STOCK_SYMBOL}.csv", index=False)
print(f"Saved summary report to model_summary_{STOCK_SYMBOL}.csv")