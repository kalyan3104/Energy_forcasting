# Windows Quickstart (PowerShell)

This short guide shows the minimum steps to build and run the Energy Forecast app on Windows using PowerShell. It's focused on a local developer setup (virtualenv). For a more robust Linux-like workflow, consider using WSL2.

Prerequisites
- Windows 10/11
- Python 3.8–3.11 (installer from python.org)
- Git
- (Optional) Visual C++ Build Tools (for building wheels)

Quick steps

1. Clone the repo and open PowerShell

```powershell
git clone <repo-url> jhanavi_project
cd jhanavi_project\energy-forecast
```

2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, run as Admin once:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Install dependencies

```powershell
pip install --upgrade pip setuptools wheel
pip install -r .\backend\requirements.txt
# Add runtime deps used by the demo UI and tests
pip install streamlit uvicorn xgboost tensorflow httpx plotly
```

4. Prepare data (if required)

If `data/processed/sample.parquet` is missing, either regenerate it with the repo scripts or let the live feeder bootstrap data.

```powershell
python .\backend\scripts\train_arima.py --preprocess-only
```

5. (Optional) Start the live feeder to simulate incoming data

```powershell
python .\scripts\live_feeder.py --interval 5
```

6. Run the backend (FastAPI)

```powershell
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

7. Run Streamlit UI (separate window)

```powershell
.\.venv\Scripts\streamlit run .\streamlit_app\app.py --server.port 8501 --server.address 0.0.0.0
```

8. Verify

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{}"
```

Firewall / network notes
- To access the app from other machines, allow inbound ports in Windows Firewall (8000 and 8501):

```powershell
New-NetFirewallRule -DisplayName "Allow 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Allow 8501" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

Troubleshooting
- If `pred_xgb` or `pred_lstm` are null, check `/models/status` to confirm models loaded. Install missing packages (xgboost, tensorflow) into the venv and restart the backend.
- If pydantic/typing errors appear, upgrade `typing_extensions` and `pydantic`:

```powershell
pip install -U typing_extensions pydantic
```

Alternative: WSL2
- For a Linux-like experience on Windows, install WSL2 (Ubuntu) and follow the repository's Linux instructions — often smoother for Python ML tooling.

If you want, I can add these instructions to the project `README.md` or provide a PowerShell helper script that starts backend, Streamlit and the feeder and records PIDs.
