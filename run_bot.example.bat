@echo off
REM ============================================================
REM  run_bot.bat - Quant Trading Daily Bot Launcher
REM  Used for Windows Task Scheduler
REM ============================================================

REM Set working directory to where this .bat file lives
cd /d "%~dp0"

REM ============================================================
REM Activate Conda Environment
REM ============================================================
REM Change 'base' to your Conda Env name (e.g. quant_env)
call conda activate trading_ai
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
