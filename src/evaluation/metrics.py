import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_and_report(y_true, y_pred, target_name="Target", verbose=True):
    """
    Classification Report and Accuracy calculation
    """
    acc = accuracy_score(y_true, y_pred)
    if verbose:
        print(f"\n{'='*50}")
        print(f"[EVAL] Horizon: {target_name} (OOS Accuracy: {acc*100:.2f}%)")
        print(f"{'='*50}")
        
        # Use zero_division=0 to suppress warnings when predicting only one class
        report = classification_report(
            y_true, y_pred, 
            target_names=['Down/Flat (0)', 'Up (1)'], 
            zero_division=0
        )
        print(report)
        
    return acc

def create_model_comparison_df(results_dict):
    """
    Converts results dictionary to Pandas DataFrame for summary CSV
    
    Args:
        results_dict: dict with key=Target and value=dict of metrics
    """
    comparison_data = []
    
    for target_name, metrics in results_dict.items():
        row = {
            'Target Horizon': target_name,
            'Train Accuracy': metrics.get('train_acc', 0.0),
        }
        if 'val_acc' in metrics:
            row['Val Accuracy'] = metrics.get('val_acc', 0.0)
        row['Test Accuracy'] = metrics.get('test_acc', 0.0)
        row['Overfitting Gap'] = metrics.get('train_acc', 0.0) - metrics.get('test_acc', 0.0)
        row['Avg Confidence'] = metrics.get('avg_confidence', 0.0)
        comparison_data.append(row)
        
    df = pd.DataFrame(comparison_data)
    
    print(f"\n[REPORT] Summary Report")
    print("-" * 65)
    print(df.to_string(index=False))
    print("-" * 65)
    
    return df
