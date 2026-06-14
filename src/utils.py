import os
import random
import numpy as np
from datetime import datetime

def needs_download(filepath):
    """
    Checks if the file exists and if it was modified today.
    Returns True if it needs to be downloaded, False otherwise.
    """
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        last_mod_date = datetime.fromtimestamp(mtime).date()
        today_date = datetime.today().date()
        
        if last_mod_date == today_date:
            return False
    return True

def set_seed(seed=42):
    """
    Locks down all random number generators for strict reproducibility.
    Must be called before any model creation or training.
    """
    import torch
    
    # 1. Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # 3. cuDNN Backend (deterministic but slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False