# Energy Forecast (AI-driven) - Repo

Contents:
- backend/: FastAPI app, training scripts
- streamlit_app/: Streamlit dashboard
- data/: raw and processed datasets
- notebooks/: EDA and experiments

Quickstart (local, without Docker):
1. Create venv:
   python -m venv venv
   source venv/bin/activate
2. Install:
   pip install -r backend/requirements.txt
3. Prepare data:
   - Put datasets inside data/raw/
   - Run preprocessing:
     python backend/scripts/train_arima.py --preprocess-only
   (See scripts for details)
4. Train models:
   python backend/scripts/train_xgb.py
   python backend/scripts/train_lstm.py
   python backend/scripts/train_arima.py
5. Run API:
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

Quickstart (Docker):
1. Copy `.env.example` -> `.env` and adjust variables.
2. docker compose up --build
3. Visit Streamlit at http://localhost:8501 and API at http://localhost:8000/docs

Notes:
- Processed data stored in data/processed/sample.parquet
- Models saved under backend/app/models/

Updated todo list



========- Install system prerequisites (Python or Anaconda, build tools if needed).
- Clone repo and open a terminal.
- Create & activate a virtual environment (or conda env).
- Install Python dependencies.
- Prepare data (preprocess/regenerate parquet if needed).
- (Optional) Train models (or use provided models).
- Start backend (uvicorn).
- Start Streamlit UI.
- (Optional) Start live feeder to simulate incoming data.
- Verify endpoints and UI.
- Troubleshooting and tips for running as services/background.

Ordered step-by-step (PowerShell commands, run as a normal or elevated PowerShell where noted)

1) System prerequisites
- Install Python 3.8–3.11 for best compatibility (download from python.org). On Windows pick the installer and check "Add Python to PATH".
- Install Git (git-scm.com).
- If you plan to install packages that need compilation (rare on this repo but possible), install Visual C++ Build Tools (via Visual Studio installer) — helpful if pip tries to build wheels.
- Optional: Install Anaconda/Miniconda if you prefer conda environments.

2) Clone the repository
Open PowerShell, then:
```powershell
cd C:\path\to\where\you\want\projects
git clone <repo-url> jhanavi_project
cd .\jhanavi_project\energy-forecast
```
(Replace <repo-url> with your repo remote.)

3A) Create & activate a venv (recommended)
```powershell
python -m venv .venv
# PowerShell activation:
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error (cannot run scripts):
# run PowerShell as Administrator and:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then re-run activation.
```

3B) (Alternative) Create & activate conda env
```powershell
conda create -n energyforecast python=3.9 -y
conda activate energyforecast
```

4) Install Python dependencies
- There is a `backend/requirements.txt`. Install it from the project root:
```powershell
pip install --upgrade pip setuptools wheel
pip install -r .\backend\requirements.txt
# Also ensure these are installed (if not in requirements):
pip install streamlit uvicorn httpx plotly joblib pandas numpy
```
Notes:
- If you plan to use the LSTM model, install TensorFlow: `pip install tensorflow` (CPU wheel). On Windows this is typically OK; if you need GPU, follow TensorFlow GPU install docs.
- For XGBoost: `pip install xgboost` usually works on Windows.
- If pip fails on a package, consider using conda (`conda install -c conda-forge xgboost tensorflow`), or install pre-built wheels.

5) Prepare data
- If `data/processed/sample.parquet` exists, you can use it. Otherwise regenerate from CSV:
```powershell
# from repo root
python .\backend\scripts\train_arima.py --preprocess-only
# or manually create parquet:
# (Run in Python or use the repo utilities)
```
- Alternatively, the live feeder will bootstrap a series if no processed data exists.

6) (Optional) Train models
- If you want to (or if models are missing), run:
```powershell
python .\backend\scripts\train_xgb.py
python .\backend\scripts\train_lstm.py
python .\backend\scripts\train_arima.py
```
Note: training may be slow and needs dependencies (scikit-learn, xgboost, tensorflow, etc.).

7) Start backend (FastAPI via uvicorn)
From project root (and with venv/conda env active):
```powershell
# Start in foreground (easy for debugging)
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Or start in background via Start-Process:
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --log-level info" -NoNewWindow
```
Verify:
```powershell
# quick check from PowerShell
Invoke-RestMethod http://127.0.0.1:8000/health
# or
curl.exe http://127.0.0.1:8000/health
```

8) Start Streamlit UI
Open another PowerShell with env active and run:
```powershell
# Make Streamlit listen on all interfaces (0.0.0.0) for external access
.\.venv\Scripts\streamlit.exe run .\streamlit_app\app.py --server.port 8501 --server.address 0.0.0.0
```
- Streamlit will print the local URL (http://localhost:8501). If you used 0.0.0.0 you can access via the machine IP from other machines in the same network.

9) (Optional) Start live feeder (keeps the data updating)
```powershell
# Start feeder in a separate shell
.\.venv\Scripts\python .\scripts\live_feeder.py --interval 5
# To run it in background:
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList ".\scripts\live_feeder.py --interval 5"
```
The feeder will append rows to sample.csv and update `sample.parquet`.

10) Verify predictions and model status
- Models status:
```powershell
curl.exe http://127.0.0.1:8000/models/status | jq
```
- Get a prediction (the server will auto-use recent data if you post `{}`):
```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{}" | jq
```
Expect keys like `pred_xgb`, `pred_lstm`, and `ensemble`. If `pred_xgb` or `pred_lstm` are null, check `/models/status` for whether models loaded.

11) Stop processes
- In PowerShell, stop by closing the window or using taskkill:
```powershell
# find PID (example uses port)
netstat -ano | findstr :8000
taskkill /PID <pid> /F
# Or use Stop-Process in PowerShell:
Get-Process -Id <pid> | Stop-Process
```

Troubleshooting (common issues on Windows)
- "Cannot import typing_extensions.TypeIs" or pydantic typing issues:
  - Upgrade typing_extensions and pydantic: `pip install -U typing_extensions pydantic`
- `httpx` missing for tests or TestClient: `pip install httpx`
- Pip build errors for packages:
  - Install Visual C++ Build Tools (via Visual Studio installer).
  - Or use conda: `conda install -c conda-forge xgboost tensorflow`
- TensorFlow/CPU vs GPU:
  - `pip install tensorflow` installs CPU-only wheel by default on many Windows setups. For GPU you need CUDA/cuDNN and cupy/appropriate TF wheel—follow TensorFlow docs.
- Ports not reachable from other machines:
  - Ensure Windows Firewall allows the port or disable for testing:
    - Open Windows Firewall > Inbound rules > Allow port 8000 / 8501 or run PowerShell:
```powershell
New-NetFirewallRule -DisplayName "Allow 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Allow 8501" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```
- If activation fails due to execution-policy for PowerShell, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` (Admin may be required).

Alternative: use WSL2 (recommended if you prefer Linux tooling)
- Install WSL2 and use Ubuntu; then follow the Linux commands that are already in the README — often easier for python tooling and matching CI.

Extra helpful tips
- Keep a requirements/dev-requirements split (if you frequently need to add packages for training).
- Use `uvicorn --reload` during development to auto-restart on code changes.
- If you want persistent services, create Windows Services (nssm) or use Task Scheduler to run scripts on login.

Small checklist you can copy & run (PowerShell)
1. Clone & cd into repo
2. Create venv & activate
3. pip install -r backend/requirements.txt
4. pip install streamlit uvicorn xgboost tensorflow httpx
5. Start backend and Streamlit in separate shells
6. Optional: run live feeder

Commands chunk (copyable):
```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r .\backend\requirements.txt
pip install streamlit uvicorn xgboost tensorflow httpx plotly
# start backend
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
# in another window start streamlit
.\.venv\Scripts\streamlit run .\streamlit_app\app.py --server.port 8501 --server.address 0.0.0.0
# optional feeder
.\.venv\Scripts\python .\scripts\live_feeder.py --interval 5==================
```

