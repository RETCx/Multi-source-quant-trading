import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

stock = config['data_settings']['selected_tickers'][0]
data_path = f"data/features/{stock}_with_indicators.csv"
targets = config['target_columns']
exclude = config.get('exclude_columns', [])

# 2. Load data
df = pd.read_csv(data_path, index_col='date', parse_dates=True)
df = df.dropna(subset=targets)

# 3. Filter features
feature_cols = [c for c in df.columns if c not in targets + exclude]
X = df[feature_cols]

print("="*50)
print(f"[REPORT] Feature Analysis for {stock}")
print("="*50)
print(f"Total rows: {len(X)}")
print(f"Total features: {len(feature_cols)}\n")

# --- A. Data Quality Checks ---
print("--- 1. Data Quality Checks ---")
nan_counts = X.isna().sum()
if nan_counts.sum() > 0:
    print(f"[WARN] Features with NaNs: {nan_counts[nan_counts > 0].to_dict()}")
else:
    print("[OK] No NaNs found in features.")

inf_counts = np.isinf(X).sum()
if inf_counts.sum() > 0:
    print(f"[WARN] Features with Infs: {inf_counts[inf_counts > 0].to_dict()}")
else:
    print("[OK] No Infinite values found.")

zero_var = X.columns[X.var() == 0].tolist()
if zero_var:
    print(f"[WARN] Features with ZERO variance (useless): {zero_var}")
else:
    print("[OK] No zero-variance features found.")

# --- B. Feature Importance via Random Forest ---
print("\n--- 2. Top 20 Feature Importances (Random Forest) ---")
# Using a small tree to quickly gauge linear/non-linear splits
for target in targets:
    y = df[target]
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y) # Fillna 0 just in case
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n[TARGET]: {target}")
    for i in range(20):
        print(f"  {i+1:2d}. {feature_cols[indices[i]]:20s} ({importances[indices[i]]:.4f})")

# --- C. High Correlation Check (Multicollinearity) ---
print("\n--- 3. Highly Correlated Feature Pairs (R > 0.95) ---")
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
if to_drop:
    print(f"[WARN] Found {len(to_drop)} features highly correlated with others.")
    print("Consider adding some of these to 'exclude_columns' in config.yaml to reduce noise:")
    print(to_drop[:20], "..." if len(to_drop) > 20 else "")
else:
    print("[OK] No highly correlated feature pairs found.")
