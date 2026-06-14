import os
import yaml
import pandas as pd
import numpy as np
import torch
import shutil
import pickle
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

from src.data_pipeline.walk_forward import WalkForwardCV
from src.data_pipeline.sequence import create_multi_horizon_sequences, get_dataloaders
from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer
from src.evaluation.ensemble import ensemble_overlapping_predictions
from src.evaluation.metrics import evaluate_and_report, create_model_comparison_df
from src.evaluation.tracker import ExperimentManager
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


# ==========================================
# 3. WALK-FORWARD TRAINING & EVALUATION
# ==========================================
print("\n" + "="*50)
print(f"WALK-FORWARD TRAINING | {STOCK_SYMBOL}")
print("="*50)

pipeline_results = run_walk_forward_pipeline(
    X_raw=X_raw,
    y_raw=y_raw,
    features=features,
    target_cols=TARGET_COLS,
    params=PARAMS,
    cv_config=CV_CONFIG,
    device=device,
    seed=SEED,
    verbose=True
)

if not pipeline_results:
    print("[ERROR] Pipeline returned no results.")
    sys.exit(1)

ensembled_oos = pipeline_results['ensembled_oos']
raw_results = pipeline_results['raw_results']
train_accs_mean = pipeline_results['train_accs_mean']

print("\n" + "="*50)
print(f"FINAL OUT-OF-SAMPLE EVALUATION")
print("="*50)

results_summary = {}

for head_idx, target_name in enumerate(TARGET_COLS):
    # evaluate_and_report requires f_trues, f_preds, target_name
    f_trues = ensembled_oos[target_name]['trues']
    f_preds = ensembled_oos[target_name]['preds']
    f_probas = ensembled_oos[target_name]['probas']
    
    acc = evaluate_and_report(f_trues, f_preds, target_name)
    
    results_summary[target_name] = {
        'train_acc': train_accs_mean[target_name],
        'test_acc': acc,
        'avg_confidence': np.mean(f_probas)
    }

# Summary table
df_summary = create_model_comparison_df(results_summary)

# Save results using ExperimentManager
exp = ExperimentManager(base_dir="data/models", prefix="train", ticker=STOCK_SYMBOL)
exp.save_dataframe(df_summary, "model_summary.csv")
exp.save_config('config.yaml')

# Save predictions and actual values as pickle for backtesting/analysis
oos_data = {
    'raw': {
        'indices': raw_results['indices'],
        'probas': raw_results['probas'],
        'trues': raw_results['trues'],
        'fold_ids': raw_results['fold_ids'],
    },
    'ensembled': ensembled_oos,
    'target_cols': TARGET_COLS,
}
exp.save_pickle(oos_data, "oos_results.pkl")

# --- SAVE OOS PREDICTIONS TO CSV FOR MANUAL INSPECTION ---
print("\n[INFO] Saving OOS Predictions to CSV...")
oos_records = []
# Find all possible OOS indices (union of all targets)
all_oos_indices = set()
for t_name in TARGET_COLS:
    if t_name in ensembled_oos:
        all_oos_indices.update(ensembled_oos[t_name]['indices'])
all_oos_indices = sorted(all_oos_indices)

# Create DataFrame from OOS prediction dates
oos_dates = df.index[all_oos_indices]
df_oos_csv = pd.DataFrame(index=oos_dates)
df_oos_csv.index.name = 'Date'

for t_name in TARGET_COLS:
    if t_name in ensembled_oos:
        oos = ensembled_oos[t_name]
        # Find row indices for this target
        t_indices = oos['indices']
        t_dates = df.index[t_indices]
        
        # Create series to map values to df_oos_csv correctly by date
        pred_series = pd.Series(oos['preds'], index=t_dates)
        prob_series = pd.Series(oos['probas'], index=t_dates)
        true_series = pd.Series(oos['trues'], index=t_dates)
        
        df_oos_csv[f"{t_name}_Pred"] = pred_series
        df_oos_csv[f"{t_name}_Prob"] = prob_series.round(4)
        df_oos_csv[f"{t_name}_True"] = true_series

# Sort dates and save
df_oos_csv = df_oos_csv.sort_index()
exp.save_dataframe(df_oos_csv, "oos_predictions.csv")

print(f"\n[DONE] All artifacts saved successfully.")