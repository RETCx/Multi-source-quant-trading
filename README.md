# Multi-Source Quant Trading Bot

An end-to-end automated quantitative trading pipeline that fetches market data, generates multi-horizon predictions using a PyTorch LSTM Walk-Forward model, and sends daily trading signals via Email and LINE.

---

## Key Features & Methodologies

Throughout the development of this project, we implemented several advanced techniques to ensure robustness, prevent data leakage, and automate the pipeline for real-world deployment:

### 1. Multi-Source Data Architecture
* **Alpha Vantage (Fundamentals):** Fetches up to 20 years of quarterly income statements and balance sheets to entirely eliminate the look-ahead bias that occurs when using backward filling (`bfill`) on short-history sources like Yahoo Finance.
* **Google BigQuery (GDELT News):** Extracts global news sentiment using a memory-optimized **Low-RAM Chunk Processing** method (`to_dataframe_iterable()`), allowing massive queries without crashing the system.
* **Tiingo & YFinance:** Reliable API integrations for daily OHLCV, SPY, and VIX data.

### 2. PyTorch Multi-Horizon LSTM
* **Multi-Head Architecture:** A single deep learning model that simultaneously predicts multiple holding periods (`1D`, `3D`, `5D`, `7D`, `10D`) sharing the same hidden state representation.
* **Walk-Forward Cross Validation:** Emulates realistic trading by incrementally expanding/sliding the training window forward in time, preventing future data leakage into the training set.
* **Fast-Training for Live Inference:** The `07_live_prediction.py` script automatically retrains the model on the latest data (without validation splits) using `MultiHeadTrainer.fast_train()` to capture the most recent market regime before predicting tomorrow's signal.

### 3. Realistic Backtesting Engine
* **Dynamic Horizons:** Executes trades based on the highest confidence horizon output by the model.
* **Slippage & Transaction Costs:** Strictly accounts for simulated real-world frictions (e.g., 0.1% Tx Cost, 0.05% Slippage per leg) to ensure backtest metrics reflect achievable results.
* **Volatility-Adjusted Stop Loss:** Dynamically calculates Stop Loss levels using Average True Range (ATR) multipliers.

### 4. Automation
* **Unified Notification System:** Supports both **LINE Notify** and **Gmail API (OAuth2)**. Emails are formatted dynamically using HTML and Base64 encoding. Includes a `--no-notify` flag for silent test runs.
* **Hardware Detection:** Automatically checks and switches to PyTorch CUDA processing if an NVIDIA GPU is available, falling back to CPU gracefully.
* **Path Locking:** Explicitly locks the Current Working Directory (`CWD`) in python scripts to ensure it works flawlessly when triggered by Windows Task Scheduler from `C:\Windows\System32`.

---

## 🛠️ Deployment Guide (Server Setup)

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
ALPHAVANTAGE_API_KEY=YOUR_ALPHAVANTAGE_KEY_HERE
LINE_NOTIFY_TOKEN=YOUR_LINE_TOKEN_HERE

# Email Notification Settings (Used as fallback or sender identifier)
SMTP_EMAIL=your_email@gmail.com
SMTP_TO_EMAIL=your_email@gmail.com
```

### 3. Setup API Keys for Google Cloud
To use Google services (BigQuery for news, Gmail for notifications), you must place specific credential files in the root folder:
* **BigQuery (`gcp_key.json`):** Service Account key with BigQuery access.
* **Gmail API (`credentials.json` & `token.json`):** 
  1. Download the OAuth Desktop App Client ID as `credentials.json` from Google Cloud Console and place it in the root folder.
  2. The first time you run the notifier, a browser window will pop up asking you to log in to your Google Account.
  3. Once authorized, the system will automatically create `token.json` to keep you logged in for future daily runs.

### 4. Configure the Execution Script (`run_bot.bat`)
Open the `run_bot.bat` file in the root folder and change `quant_env` to your actual Conda environment name if it is different.

### 5. Automate with Windows Task Scheduler
1. Open **Task Scheduler**.
2. Click **Create Basic Task...**
3. Name it `Quant Daily Bot` and trigger it **Daily** (e.g., at 07:00 AM).
4. For Action, select **Start a program**.
5. Browse and select the `run_bot.bat` you just created.
6. (Optional) In task properties, check "Run whether user is logged on or not" and "Hidden" to run it in the background silently.

---

## Running Tests
If you only want to test the live prediction pipeline without triggering actual emails or LINE messages:
```bash
python 07_live_prediction.py --no-notify
```
