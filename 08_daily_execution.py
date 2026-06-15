"""
08_daily_execution.py
=====================
Master Script สำหรับรัน Pipeline อัตโนมัติทุกวัน
ออกแบบมาให้ถูกเรียกจาก Windows Task Scheduler ผ่าน run_bot.bat

Flow: Fetch Data -> Build Features -> Train Model -> Live Prediction -> Notify LINE
"""
import subprocess
import sys
import os
import time
import logging
from datetime import datetime

# ==========================================================
# FIX PATH: บังคับให้ Working Directory เป็นที่อยู่ของไฟล์นี้เสมอ
# ป้องกัน Error เวลา Windows Task Scheduler รันจาก C:\Windows\System32
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ==========================================================
# LOGGING: บันทึกผลลัพธ์ลงไฟล์ logs/
# ==========================================================
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================================
# LOAD ENV VARIABLES (.env)
# ==========================================================
from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))


def run_step(script_name: str, step_label: str) -> bool:
    """Run a python script and stream its output in real-time."""
    logger.info(f"{'='*50}")
    logger.info(f"STEP: {step_label}")
    logger.info(f"Running: {script_name}")
    logger.info(f"{'='*50}")
    
    start = time.time()
    
    # ใช้ Popen เพื่อให้ดึง output มาโชว์แบบ Real-time ได้
    process = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, script_name)],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # รวบ Error มาโชว์ในช่องทางเดียวกัน
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # อ่านบรรทัดต่อบรรทัดแบบ Real-time
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            logger.info(f"  | {line}")
            
    process.stdout.close()
    returncode = process.wait()
    
    elapsed = time.time() - start
    
    if returncode != 0:
        logger.error(f"FAILED: {script_name} (exit code {returncode}, took {elapsed:.1f}s)")
        return False
    
    logger.info(f"OK: {script_name} ({elapsed:.1f}s)")
    return True


def send_error_notification(failed_step: str):
    """Send LINE notification when pipeline fails."""
    try:
        from src.notifier import send_line_notify
        msg = (
            f"\n🚨 PIPELINE ERROR 🚨"
            f"\nStep: {failed_step}"
            f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            f"\nCheck log: {log_filename}"
        )
        send_line_notify(msg)
    except Exception as e:
        logger.error(f"Could not send error notification: {e}")


def main():
    total_start = time.time()
    
    logger.info("=" * 60)
    logger.info("  QUANT DAILY BOT - AUTOMATED EXECUTION")
    logger.info(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  CWD:  {os.getcwd()}")
    logger.info("=" * 60)
    
    # --- HARDWARE CHECK ---
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  [HARDWARE] CUDA: TRUE | GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("  [HARDWARE] CUDA: FALSE | GPU not found. Falling back to CPU!")
    except ImportError:
        logger.warning("  [HARDWARE] PyTorch not installed. Skipping CUDA check.")
    logger.info("=" * 60)
    
    # Define the daily pipeline steps
    steps = [
        ("01_fetch_data.py",       "1/4 Fetch latest market data"),
        ("02_build_features.py",   "2/4 Build technical indicators"),
        ("03_run_training.py",     "3/4 Train model (rolling update)"),
        ("07_live_prediction.py",  "4/4 Generate live predictions"),
    ]
    
    for script, label in steps:
        success = run_step(script, label)
        if not success:
            logger.error(f"Pipeline ABORTED at: {label}")
            send_error_notification(label)
            sys.exit(1)
    
    total_elapsed = time.time() - total_start
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  DAILY PIPELINE COMPLETED SUCCESSFULLY 🎉")
    logger.info(f"  Total time: {total_elapsed / 60:.1f} minutes")
    logger.info(f"  Log saved: {log_filename}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
