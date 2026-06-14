import numpy as np
from collections import defaultdict

def ensemble_overlapping_predictions(oos_indices, oos_probas, oos_true, oos_folds=None):
    """
    รวมผลการทำนายที่ซ้อนทับกัน (Overlap) จาก Walk-Forward หลายๆ Fold 
    ด้วยการหาค่าเฉลี่ยความน่าจะเป็น (Probability Averaging)
    
    Args:
        oos_indices (list): List ของวันที่/Index ที่ทำนาย
        oos_probas (list): List ของความน่าจะเป็น (0.0 - 1.0)
        oos_true (list): List ของเฉลยจริง
        oos_folds (list, optional): List ของหมายเลข Fold ที่ทำนาย
        
    Returns:
        final_preds, final_probas, final_true, unique_indices
    """
    # 1. จัดกลุ่มความน่าจะเป็นตาม Index
    predictions_by_idx = defaultdict(list)
    for i, idx in enumerate(oos_indices):
        fold_id = oos_folds[i] if oos_folds else "Unknown"
        predictions_by_idx[idx].append({
            'proba': oos_probas[i],
            'true': oos_true[i],
            'fold': fold_id
        })
        
    unique_indices = sorted(predictions_by_idx.keys())
    final_preds, final_probas, final_true = [], [], []
    
    # 2. หาค่าเฉลี่ย (Ensemble)
    overlap_counts = []
    
    for idx in unique_indices:
        preds_list = predictions_by_idx[idx]
        
        # Average probability จากทุกโมเดลที่ทำนายชนกัน
        avg_proba = np.mean([p['proba'] for p in preds_list])
        
        # ตัดสินใจสุดท้าย: ถ้าความมั่นใจเฉลี่ยเกิน 50% ให้เป็น 1 (Up)
        final_pred = 1 if avg_proba > 0.5 else 0
        
        final_preds.append(final_pred)
        final_probas.append(avg_proba)
        final_true.append(preds_list[0]['true']) # เฉลยจริงเหมือนกันหมด
        overlap_counts.append(len(preds_list))
        
    avg_overlap = np.mean(overlap_counts) if overlap_counts else 0
    max_overlap = np.max(overlap_counts) if overlap_counts else 0
    print(f"     Ensemble Overlap: Average {avg_overlap:.1f} folds/day, Max {max_overlap} folds/day")
    
    return np.array(final_preds), np.array(final_probas), np.array(final_true), unique_indices