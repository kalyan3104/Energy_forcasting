import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
print("Make models: starting")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "sample.parquet"
MODEL_DIR = ROOT / "backend" / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
if DATA.exists():
    df = pd.read_parquet(DATA)
else:
    csv = ROOT / "data" / "processed" / "sample.csv"
    df = pd.read_csv(csv, parse_dates=["timestamp"])

power = df["power"].astype(float).values

# 1) Dummy ARIMA-like model: object with get_forecast(steps).predicted_mean
class DummyARIMA:
    def __init__(self, last):
        self.last = float(last)
    def get_forecast(self, steps=1):
        class R:
            def __init__(self, arr):
                self.predicted_mean = np.array(arr)
        # simple linear ramp
        arr = [self.last + 0.01*(i+1) for i in range(steps)]
        return R(arr)

arima = DummyARIMA(power[-1])
joblib.dump(arima, MODEL_DIR / "arima_model.pkl")
print("Wrote arima_model.pkl")

# 2) XGBoost model
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    features = ["sin_hour","cos_hour","hour","lag_1","rolling_3"]
    # Ensure features present or create simple features
    if not set(features).issubset(df.columns):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['sin_hour'] = np.sin(2*np.pi*df['hour']/24)
        df['cos_hour'] = np.cos(2*np.pi*df['hour']/24)
        df['lag_1'] = df['power'].shift(1).fillna(method='bfill')
        df['rolling_3'] = df['power'].rolling(3,min_periods=1).mean()
    X = df[features].fillna(method='bfill').values
    y = df['power'].astype(float).values
    # simple train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print("xgb rmse:", mean_squared_error(y_val, preds, squared=False))
    booster = model.get_booster()
    booster.save_model(str(MODEL_DIR / "xgb_model.bst"))
    joblib.dump(features, MODEL_DIR / "xgb_features.joblib")
    print("Wrote xgb_model.bst and xgb_features.joblib")
except Exception as e:
    print("XGBoost training skipped or failed:", e)

# 3) LSTM model (Keras)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    # Prepare sequences
    seq_len = min(10, len(power)-1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(power.reshape(-1,1))
    Xs, ys = [], []
    for i in range(len(scaled)-seq_len):
        Xs.append(scaled[i:i+seq_len,0])
        ys.append(scaled[i+seq_len,0])
    Xs = np.array(Xs)
    ys = np.array(ys)
    Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
    if Xs.shape[0] > 0:
        model = Sequential()
        model.add(LSTM(8, input_shape=(Xs.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(Xs, ys, epochs=10, batch_size=8, verbose=0)
        model.save(str(MODEL_DIR / "lstm_final.h5"))
        joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")
        print("Wrote lstm_final.h5 and lstm_scaler.joblib")
    else:
        print("Not enough data for LSTM training")
except Exception as e:
    print("LSTM training skipped or failed:", e)

print("Make models: done")
