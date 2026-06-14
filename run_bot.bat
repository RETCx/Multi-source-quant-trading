@echo off
REM ============================================================
REM  run_bot.bat - Quant Trading Daily Bot Launcher
REM  ใช้สำหรับ Windows Task Scheduler
REM ============================================================

REM Set working directory to where this .bat file lives
cd /d "%~dp0"

REM Activate virtual environment (if exists)
IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [BOT] Virtual environment activated.
) ELSE IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [BOT] Virtual environment activated.
) ELSE (
    echo [BOT] No venv found. Using system Python.
)

REM Run the daily pipeline
echo [BOT] Starting daily pipeline at %date% %time%
python 08_daily_execution.py

echo [BOT] Pipeline finished at %date% %time%
