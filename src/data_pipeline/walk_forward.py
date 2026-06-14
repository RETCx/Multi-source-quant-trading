import numpy as np

class WalkForwardCV:
    """
    Time-Series Walk-Forward Analysis Engine (Expanding or Sliding Window)
    guaranteed Out-of-Sample (OOS) and Gap to prevent Data Leakage
    """
    def __init__(self, min_train_size=500, test_size=60, gap=22, step_size=60, max_train_size=None):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap
        self.step_size = step_size
        self.max_train_size = max_train_size
    
    def split(self, n_samples):
        """
        Create (train_indices, test_indices) for each Fold
        
        Args:
            n_samples (int): Total number of samples
            
        Returns:
            list of tuples: [(train_idx, test_idx), ...]
        """
        folds = []
        # The starting point of the first Test Fold must add Gap to Train
        test_start = self.min_train_size + self.gap
        
        while test_start + self.test_size <= n_samples:
            # Train End will leave a Gap before the Test Start
            train_end = test_start - self.gap
            
            if self.max_train_size:
                # Sliding Window mode: Limit Train size not to exceed max_train_size
                train_start = max(0, train_end - self.max_train_size)
            else:
                # Expanding Window mode: Always start from 0
                train_start = 0 
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_start + self.test_size)
            
            folds.append((train_idx, test_idx))
            
            # Move to the next round
            test_start += self.step_size
            
        return folds
    
    def summary(self, n_samples):
        """
        Summary CV Folds on Terminal
        """
        folds = self.split(n_samples)
        mode_name = 'Sliding Window' if self.max_train_size else 'Expanding Window'
        
        print(f"\n Walk-Forward CV Summary ({mode_name}):")
        print(f"   Total samples: {n_samples}")
        print(f"   Number of folds: {len(folds)}")
        print(f"   Min train size: {self.min_train_size}")
        print(f"   Test size per fold: {self.test_size}")
        print(f"   Gap (embargo): {self.gap}")
        print(f"   {'─'*65}")
        
        for i, (tr, te) in enumerate(folds):
            print(f"   Fold {i+1:2d}: Train [{tr[0]:4d}:{tr[-1]:4d}] ({len(tr):4d} rows) "
                  f"→ Gap {self.gap}d → Test [{te[0]:4d}:{te[-1]:4d}] ({len(te):3d} rows)")
        
        print(f"   {'─'*65}")
        if folds:
            all_test_idx = np.concatenate([te for _, te in folds])
            unique_oos = len(np.unique(all_test_idx))
            coverage = (unique_oos / n_samples) * 100
            print(f"   OOS Coverage: {unique_oos}/{n_samples} ({coverage:.1f}%)")
        else:
            print("Not enough data to create any folds!")
            
        return folds