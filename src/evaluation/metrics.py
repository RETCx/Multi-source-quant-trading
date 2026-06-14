import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_and_report(y_true, y_pred, target_name="Target"):
    """
     Classification Report   Accuracy
    """
    acc = accuracy_score(y_true, y_pred)
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
     Dictionary  Pandas DataFrame  CSV
    
    Args:
        results_dict: dict  key  Target  value  dict  metrics
    """
    comparison_data = []
    
    for target_name, metrics in results_dict.items():
        comparison_data.append({
            'Target Horizon': target_name,
            'Train Accuracy': metrics.get('train_acc', 0.0),
            'Test Accuracy': metrics.get('test_acc', 0.0),
            'Overfitting Gap': metrics.get('train_acc', 0.0) - metrics.get('test_acc', 0.0),
            'Avg Confidence': metrics.get('avg_confidence', 0.0)
        })
        
    df = pd.DataFrame(comparison_data)
    
    print(f"\n[REPORT] Summary Report")
    print("-" * 65)
    print(df.to_string(index=False))
    print("-" * 65)
    
    return df