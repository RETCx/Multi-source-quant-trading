import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
import subprocess
import glob
import pickle
import concurrent.futures
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

def live_predict_stock(stock, config):
    try:
        set_seed(config['random_seed'])
        data_path = os.path.join(SCRIPT_DIR, f"{config['data_path']['features']}/{stock}_with_indicators.csv")
        
        if not os.path.exists(data_path):
            return stock, False, f"Data file not found: {data_path}"
            
        df = pd.read_csv(data_path, index_col='date', parse_dates=True)
        target_cols = config.get('target_columns', ['Target_1D', 'Target_3D', 'Target_5D', 'Target_7D', 'Target_10D'])
        
        base_exclude = config.get('exclude_columns', [])
        fallback_features = [c for c in df.columns if c not in target_cols and c not in base_exclude]
        features = get_best_features(stock, fallback_features)
        
        df_train = df.dropna(subset=target_cols).copy()
        seq_len = config['model_params']['sequence_length']
        df_live = df.iloc[-seq_len:].copy()
        
        if len(df_live) < seq_len:
            return stock, False, "Not enough recent data to form a sequence"
            
        X_train_raw = df_train[features].values
        y_train_raw = df_train[target_cols].values
        X_live_raw = df_live[features].values
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_live_scaled = scaler.transform(X_live_raw)
        
        from src.data_pipeline.sequence import create_multi_horizon_sequences
        X_tr_seq, y_tr_seq = create_multi_horizon_sequences(X_train_scaled, y_train_raw, seq_len)
        
        from torch.utils.data import TensorDataset, DataLoader
        train_data = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
        train_loader = DataLoader(train_data, batch_size=config['model_params']['batch_size'], shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = MultiHorizonLSTM(
            input_size=len(features),
            hidden_size=config['model_params']['hidden_size'],
            num_layers=config['model_params']['num_layers'],
            dropout=config['model_params']['dropout'],
            n_heads=len(target_cols)
        ).to(device)
        
        trainer = MultiHeadTrainer(
            model=model,
            learning_rate=config['model_params']['learning_rate'],
            epochs=100,
            device=device
        )
        model = trainer.fast_train(train_loader)
        
        model.eval()
        live_tensor = torch.FloatTensor(X_live_scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(live_tensor)
            
        # Locate latest oos_results.pkl to compute historical baseline
        pattern = os.path.join(SCRIPT_DIR, config['data_path']['models'], f"train_{stock}_*", "oos_results.pkl")
        pkl_files = sorted(glob.glob(pattern))
        
        ensembled_oos = {}
        if pkl_files:
            latest_pkl = pkl_files[-1]
            try:
                with open(latest_pkl, 'rb') as f:
                    oos_data = pickle.load(f)
                    ensembled_oos = oos_data.get('ensembled', {})
            except Exception:
                pass
                
        strength_mode = config.get('backtest', {}).get('strength_mode', 'z_score')
        rolling_win = config.get('target_rolling_window', 252)
        
        best_strength = -999.0
        best_prob = 0
        best_horizon = 0
        best_dir = 0
        prediction_details = {}
        
        for h_i, tgt in enumerate(target_cols):
            prob = preds[h_i].item()
            direction = 1 if prob >= 0.5 else 0
            confidence = prob if direction == 1 else 1 - prob
            direction_str = "UP" if direction == 1 else "DOWN"
            
            h_mean = 0.5
            h_std = 0.05
            if tgt in ensembled_oos:
                probas = [float(p) for p in ensembled_oos[tgt]['probas'] if float(p) > 1e-6]
                if probas:
                    recent_probas = probas[-rolling_win:]
                    h_mean = np.mean(recent_probas)
                    h_std = np.std(recent_probas) if len(recent_probas) > 1 else 0.05
                    
            if strength_mode == 'z_score':
                safe_std = h_std if h_std > 1e-6 else 1e-6
                strength = (prob - h_mean) / safe_std
                strength_str = f"Z-Score: {strength:+.2f}"
            else:
                strength = prob - h_mean
                strength_str = f"Diff: {strength:+.2%}"
                
            prediction_details[tgt] = {
                'conf': confidence,
                'dir': direction_str,
                'mean': h_mean,
                'strength_str': strength_str,
                'strength': strength
            }
            
            abs_strength = abs(strength)
            if abs_strength > best_strength:
                best_strength = abs_strength
                best_prob = confidence
                best_horizon = int(tgt.split('_')[1].replace('D',''))
                best_dir = direction
                
        # Send back features dict for agentic veto
        current_feat_dict = {feat_name: float(X_live_scaled[-1][idx]) for idx, feat_name in enumerate(features)}
        
        result_data = {
            'best_prob': best_prob,
            'best_horizon': best_horizon,
            'best_dir': best_dir,
            'prediction_details': prediction_details,
            'current_feat_dict': current_feat_dict,
            'strength_mode': strength_mode,
            'rolling_win': rolling_win
        }
        return stock, True, result_data
        
    except Exception as e:
        return stock, False, str(e)


def main():
    # Fix for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    config = load_config()
    set_seed(config['random_seed'])
    
    parser = argparse.ArgumentParser(description="Run Live Prediction")
    parser.add_argument('--no-notify', action='store_true', help="Skip sending email/LINE notifications (for testing)")
    args = parser.parse_args()

    print("="*60)
    print("QUANTITATIVE TRADING - LIVE DEPLOYMENT")
    print("="*60)
    
    tickers = config['data_settings']['selected_tickers']
    print(f"\n[1] Starting Parallel Live Predictions for {len(tickers)} stocks...")
    
    max_workers = min(len(tickers), 4)
    results = {}
    
    if len(tickers) == 1:
        stock, success, data = live_predict_stock(tickers[0], config)
        results[stock] = {'success': success, 'data': data}
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(live_predict_stock, ticker, config): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    stock, success, data = future.result()
                    results[stock] = {'success': success, 'data': data}
                except Exception as exc:
                    results[ticker] = {'success': False, 'data': str(exc)}
                    
    print("\n[2] Execution Strategy & Agentic Veto Check (Based on config.yaml):")
    
    prob_threshold = config['trading_setup'].get('live_threshold', 0.70)
    thresh_val = float(prob_threshold)
    
    try:
        from src.models.self_reflection import AgenticJournal
        journal = AgenticJournal(journal_path=os.path.join(SCRIPT_DIR, "data/models/trade_journal.json"), similarity_threshold=0.95)
    except Exception as e:
        print(f"[WARN] Failed to load AgenticJournal: {e}")
        journal = None
        
    final_actions = {}
    
    for stock in tickers:
        res = results.get(stock)
        if not res or not res['success']:
            print(f"\n[-] {stock}: FAILED ({res.get('data') if res else 'Unknown Error'})")
            continue
            
        data = res['data']
        best_prob = data['best_prob']
        best_horizon = data['best_horizon']
        best_dir = data['best_dir']
        prediction_details = data['prediction_details']
        
        print(f"\n{'='*40}\nSTOCK: {stock}\n{'='*40}")
        print(f"Evaluating targets using Strength Mode: '{data['strength_mode']}' (rolling window: {data['rolling_win']})")
        
        for tgt, det in prediction_details.items():
            print(f"  - {tgt}: Conf {det['conf']:.2%} -> {det['dir']} | Mean: {det['mean']:.2%}, {det['strength_str']}")
            
        print(f"\n  -> Max Confidence: {best_prob:.2%} (Target Horizon: {best_horizon} Days)")
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
                
            if journal:
                try:
                    is_vetoed, veto_reason = journal.check_veto(data['current_feat_dict'], action)
                    if is_vetoed:
                        print("\n[!] ===================================== [!]")
                        print("[!] VETO TRIGGERED BY AGENTIC SELF-REFLECTION")
                        print(f"[!] {veto_reason}")
                        print("[!] ===================================== [!]")
                        action = "HOLD (VETOED)"
                except Exception as e:
                    print(f"[WARN] Failed to run Agentic Veto check for {stock}: {e}")
                    
        else:
            action = "HOLD"
            print("\n[ ] ACTION: [STAY IN CASH]")
            print("   Confidence is too low. Strategic inactivity advised.")
            
        final_actions[stock] = {
            'action': action,
            'best_prob': best_prob,
            'best_horizon': best_horizon,
            'prediction_details': prediction_details
        }
        
    print("\n" + "="*60)
    
    # ==========================================
    # 3. SEND EMAIL NOTIFICATION (COMBINED)
    # ==========================================
    if args.no_notify:
        print("\n[NOTIFY] Notification skipped: '--no-notify' flag was used.")
    else:
        if final_actions:
            try:
                from src.notifier import send_email, format_multi_signal_message
                body = format_multi_signal_message(final_actions)
                # Create a concise subject
                actions_list = [f"{stk}:{act['action'][:4]}" for stk, act in final_actions.items() if act['action'] != 'HOLD']
                if actions_list:
                    subject_str = " ".join(actions_list)
                else:
                    subject_str = "NO ACTION (ALL HOLD)"
                subject = f"[QUANT SIGNAL] {subject_str}"
                
                send_email(subject=subject, body=body)
            except Exception as e:
                print(f"[NOTIFY] Notification skipped: {e}")
        else:
            print("[NOTIFY] No successful predictions to notify.")

if __name__ == "__main__":
    main()

