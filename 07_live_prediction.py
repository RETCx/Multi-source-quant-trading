import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
import subprocess
from sklearn.preprocessing import RobustScaler

# Fix CWD for Task Scheduler compatibility
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer
from src.utils import set_seed

def load_config():
    with open(os.path.join(SCRIPT_DIR, 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)

def run_script(script_name):
    print(f"\n[SYSTEM] Running {script_name}...")
    result = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, script_name)], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed. Cannot proceed with Live Prediction.")
        sys.exit(result.returncode)
    print(f"[SYSTEM] {script_name} completed successfully.")

def get_best_features(stock, default_features):
    path = os.path.join(SCRIPT_DIR, f"data/models/best_features_{stock}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return data.get('features', default_features)
    return default_features

def main():
    config = load_config()
    set_seed(config['random_seed'])
    
    parser = argparse.ArgumentParser(description="Run Live Prediction")
    parser.add_argument('--no-notify', action='store_true', help="Skip sending email/LINE notifications (for testing)")
    args = parser.parse_args()

    print("="*60)
    print("QUANTITATIVE TRADING - LIVE DEPLOYMENT")
    print("="*60)
    
    print("\n[1] Synchronizing Live Market Data (Running Pipeline...)")
    # Execute the data fetching and feature engineering pipeline to get today's data
    # run_script("01_fetch_data.py")
    # run_script("02_build_features.py")
    
    config = load_config()
    stock = config['data_settings']['selected_tickers'][0]
    data_path = os.path.join(SCRIPT_DIR, f"{config['data_path']['features']}/{stock}_with_indicators.csv")
    
    print(f"\n[2] Loading synchronized market data for {stock}...")
    df = pd.read_csv(data_path, index_col='date', parse_dates=True)
    
    # We need Targets to train the model up to yesterday
    target_cols = config.get('target_columns', ['Target_1D', 'Target_3D', 'Target_5D', 'Target_7D', 'Target_10D'])
    
    # Load optimal features from Phase 1
    base_exclude = config.get('exclude_columns', [])
    fallback_features = [c for c in df.columns if c not in target_cols and c not in base_exclude]
    features = get_best_features(stock, fallback_features)
    
    print(f"[3] Using {len(features)} optimal features (Loaded from Auto Feature Selection)...")
    
    # Separate data into Training Set (where targets are known) and Prediction Set (Today -> Predicts Tomorrow)
    # Targets are NaN for the most recent rows because they haven't happened yet.
    df_train = df.dropna(subset=target_cols).copy()
    
    # The last row of the original df is "Today", which we want to predict.
    # But LSTM needs a sequence of `sequence_length` (e.g., 15 days) to predict tomorrow.
    seq_len = config['model_params']['sequence_length']
    df_live = df.iloc[-seq_len:].copy()
    
    if len(df_live) < seq_len:
        print("[ERROR] Not enough recent data to form a sequence!")
        return
        
    print(f"[4] Fast-Training Live Model on {len(df_train)} historical days...")
    
    X_train_raw = df_train[features].values
    y_train_raw = df_train[target_cols].values
    X_live_raw = df_live[features].values
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_live_scaled = scaler.transform(X_live_raw)
    
    # Create Sequences for Training
    from src.data_pipeline.sequence import create_multi_horizon_sequences
    X_tr_seq, y_tr_seq = create_multi_horizon_sequences(X_train_scaled, y_train_raw, seq_len)
    
    # Create DataLoader for Training
    from torch.utils.data import TensorDataset, DataLoader
    train_data = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_data, batch_size=config['model_params']['batch_size'], shuffle=True)
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiHorizonLSTM(
        input_size=len(features),
        hidden_size=config['model_params']['hidden_size'],
        num_layers=config['model_params']['num_layers'],
        dropout=config['model_params']['dropout'],
        n_heads=len(target_cols)
    ).to(device)
    
    # Train
    # We do a fast train on all data (No Validation Split since we just want to overfit to recent regime)
    trainer = MultiHeadTrainer(
        model=model,
        learning_rate=config['model_params']['learning_rate'],
        epochs=100,
        device=device
    )
    model = trainer.fast_train(train_loader)
            
    print("[5] Model Training Complete. Running Live Inference...")
    
    # Live Inference (Predict Tomorrow)
    model.eval()
    live_tensor = torch.FloatTensor(X_live_scaled).unsqueeze(0).to(device) # Shape: (1, seq_len, features)
    
    with torch.no_grad():
        preds = model(live_tensor)
        
    print("\n" + "="*60)
    print(f"LIVE PREDICTIONS FOR: TOMORROW")
    print("="*60)
    
    best_prob = 0
    best_horizon = 0
    best_dir = 0
    prediction_details = {}  # For notification
    
    for h_i, tgt in enumerate(target_cols):
        prob = preds[h_i].item()
        direction = 1 if prob >= 0.5 else 0
        confidence = prob if direction == 1 else 1 - prob
        direction_str = "UP" if direction == 1 else "DOWN"
        
        print(f"  - {tgt}: Confidence {confidence:.2%} -> {direction_str}")
        prediction_details[tgt] = (confidence, direction_str)
        
        if confidence > best_prob:
            best_prob = confidence
            best_horizon = int(tgt.split('_')[1].replace('D',''))
            best_dir = direction
            
    print("\n[6] Execution Strategy (Based on config.yaml):")
    bt_config = config['backtest']
    prob_threshold = config['trading_setup'].get('live_threshold', 0.70)
    
    # Since we are live, we use Fixed Threshold or a generic safe value
    thresh_val = float(prob_threshold)
    
    print(f"  -> Max Confidence: {best_prob:.2%} (Target Horizon: {best_horizon} Days)")
    print(f"  -> Execution Threshold: {thresh_val:.2%}")
    
    if best_prob >= thresh_val:
        if best_dir == 1:
            action = "BUY"
            print("\n[+] ACTION: [BUY/LONG]")
            print(f"   Enter at Open tomorrow. Target hold {best_horizon} days. Stop-Loss: 2x ATR.")
        else:
            action = "SELL"
            print("\n[-] ACTION: [SELL/SHORT]")
            print(f"   Enter Short at Open tomorrow. Target hold {best_horizon} days. Stop-Loss: 2x ATR.")
    else:
        action = "HOLD"
        print("\n[ ] ACTION: [STAY IN CASH]")
        print("   Confidence is too low. Strategic inactivity advised.")
        
    print("="*60)
    
    # ==========================================
    # 7. SEND EMAIL NOTIFICATION
    # ==========================================
    if args.no_notify:
        print("\n[NOTIFY] Notification skipped: '--no-notify' flag was used.")
    else:
        try:
            from src.notifier import send_email, format_signal_message
            body = format_signal_message(
                stock=stock,
                predictions=prediction_details,
                action=action,
                best_horizon=best_horizon,
                best_confidence=best_prob
            )
            subject = f"[{action}] {stock} - Conf {best_prob:.0%} ({best_horizon}D)"
            send_email(subject=subject, body=body)
        except Exception as e:
            print(f"[NOTIFY] Notification skipped: {e}")

if __name__ == "__main__":
    main()

