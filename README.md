# Multi-Source Quant Trading Bot

An end-to-end automated quantitative trading pipeline that fetches market data, generates multi-horizon predictions using a PyTorch LSTM Walk-Forward model, and sends daily trading signals via Email.

---

## Deployment Guide (Server Setup)

Since this project uses `.gitignore` to protect sensitive keys and environment-specific configurations, you need to manually set up a few files when cloning to a new server.

### 1. Install Dependencies
Make sure you have Miniconda/Anaconda installed.
```bash
conda create -n quant_env python=3.12
conda activate quant_env
pip install -r requirements.txt
```

### 2. Setup Environment Variables (`.env`)
Create a `.env` file in the root directory and add your API keys:
```env
tiingo_api_key=YOUR_TIINGO_KEY_HERE

# Email Notification Settings (Use Google App Password, NO SPACES)
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password
SMTP_TO_EMAIL=your_email@gmail.com
```

### 3. Setup GCP Service Account
If you use BigQuery for news sentiment, make sure to place your `gcp_key.json` in the root folder. (The `config.yaml` points to it automatically if configured).

### 4. Create the Execution Script (`run_bot.bat`)
Since `run_bot.bat` is ignored by Git (to prevent conflicts across different machines), **create a new file named `run_bot.bat` in the root folder** and paste the following code. 

**Note:** Change `quant_env` to the actual name of your Conda environment on this machine.

```bat
@echo off
REM ============================================================
REM MULTI-SOURCE QUANT TRADING BOT - DAILY PIPELINE LAUNCHER
REM ============================================================

REM Set working directory to where this .bat file lives
cd /d "%~dp0"

REM ============================================================
REM Activate Conda Environment
REM ============================================================
REM เปลี่ยนชื่อ 'quant_env' เป็นชื่อ Conda Env ของคุณ (เช่น base หรือชื่ออื่น)
call conda activate quant_env
IF %ERRORLEVEL% NEQ 0 (
    echo [BOT] Warning: Could not activate Conda environment. Trying to run with default python...
) ELSE (
    echo [BOT] Conda environment activated successfully.
)

REM Run the daily pipeline
echo [BOT] Starting daily pipeline at %date% %time%
python 08_daily_execution.py

echo [BOT] Pipeline finished at %date% %time%
pause
```

### 5. Automate with Windows Task Scheduler
1. Open **Task Scheduler**.
2. Click **Create Basic Task...**
3. Name it `Quant Daily Bot` and trigger it **Daily** (e.g., at 07:00 AM).
4. For Action, select **Start a program**.
5. Browse and select the `run_bot.bat` you just created.
6. (Optional) In task properties, check "Run whether user is logged on or not" and "Hidden" to run it in the background silently.

--
**Happy Trading! **