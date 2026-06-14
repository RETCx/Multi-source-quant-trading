import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import subprocess
from sklearn.preprocessing import RobustScaler

from src.models.architecture import MultiHorizonLSTM
from src.models.trainer import MultiHeadTrainer
from src.utils import set_seed

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_script(script_name):
    print(f"\n[SYSTEM] Running {script_name}...")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed. Cannot proceed with Live Prediction.")
        sys.exit(result.returncode)
    print(f"[SYSTEM] {script_name} completed successfully.")

def get_best_features(stock, default_features):
    path = f"data/models/best_features_{stock}.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return data.get('features', default_features)
    return default_features

def main():
    print("="*60)
    print("QUANTITATIVE TRADING - LIVE DEPLOYMENT")
    print("="*60)
    
    print("\n[1] Synchronizing Live Market Data (Running Pipeline...)")
    # Execute the data fetching and feature engineering pipeline to get today's data
    run_script("01_fetch_data.py")
    run_script("02_build_features.py")
    
    config = load_config()
    stock = config['data_settings']['selected_tickers'][0]
    data_path = f"{config['data_path']['features']}/{stock}_with_indicators.csv"
    
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
    set_seed(config['random_seed'])
    
    model = MultiHorizonLSTM(
        input_size=len(features),
        hidden_size=config['model_params']['hidden_size'],
        num_layers=config['model_params']['num_layers'],
        dropout=config['model_params']['dropout'],
        n_heads=len(target_cols)
    ).to(device)
    
    # Train
    # We do a fast train on all data (No Validation Split since we just want to overfit to recent regime)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model_params']['learning_rate'])
    criterion = torch.nn.BCELoss()
    epochs = 100 # Fast production train
    
    model.train()
    for ep in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = 0
            for h_i in range(len(target_cols)):
                loss += criterion(outputs[h_i], batch_y[:, h_i].unsqueeze(1))
            loss.backward()
            optimizer.step()
            
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
    
    for h_i, tgt in enumerate(target_cols):
        prob = preds[h_i].item()
        direction = 1 if prob >= 0.5 else 0
        confidence = prob if direction == 1 else 1 - prob
        
        print(f"  - {tgt}: Confidence {confidence:.2%} -> {'UP' if direction == 1 else 'DOWN'}")
        
        if confidence > best_prob:
            best_prob = confidence
            best_horizon = int(tgt.split('_')[1].replace('D',''))
            best_dir = direction
            
    print("\n[6] Execution Strategy (Based on config.yaml):")
    bt_config = config['backtest']
    prob_threshold = bt_config.get('prob_threshold', 85)
    
    # Since we are live, we use Fixed Threshold or a generic safe value
    thresh_val = prob_threshold if prob_threshold <= 1.0 else 0.70 # Default to 70% if percentile config is used
    
    print(f"  -> Max Confidence: {best_prob:.2%} (Target Horizon: {best_horizon} Days)")
    print(f"  -> Execution Threshold: {thresh_val:.2%}")
    
    if best_prob >= thresh_val:
        if best_dir == 1:
            print("\n✅ ACTION: [BUY/LONG]")
            print(f"   Enter at Open tomorrow. Target hold {best_horizon} days. Stop-Loss: 2x ATR.")
        else:
            print("\n❌ ACTION: [SELL/SHORT]")
            print(f"   Enter Short at Open tomorrow. Target hold {best_horizon} days. Stop-Loss: 2x ATR.")
    else:
        print("\n⚪ ACTION: [STAY IN CASH]")
        print("   Confidence is too low. Strategic inactivity advised.")
        
    print("="*60)

if __name__ == "__main__":
    main()
