import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"STARTING: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Use the same python executable that is running this script
    result = subprocess.run([sys.executable, script_name])
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}")
        print("Pipeline stopped.")
        sys.exit(result.returncode)
        
    print(f"\n✅ COMPLETED: {script_name} (Took {elapsed:.2f} seconds)")

if __name__ == "__main__":
    print("============================================================")
    print("      QUANT TRADING PIPELINE - FULL EXECUTION MASTER        ")
    print("============================================================")
    
    scripts = [
        "01_fetch_data.py",
        "02_build_features.py",
        "03_run_training.py",
        "04_run_backtest.py"
    ]
    
    total_start_time = time.time()
    
    for script in scripts:
        run_script(script)
        
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total execution time: {total_elapsed / 60:.2f} minutes")
    print(f"{'='*60}\n")
