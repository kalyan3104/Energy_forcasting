from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "sample.parquet"
MODEL_DIR = ROOT / "backend" / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print('Retrain LSTM: loading data')
if DATA.exists():
    df = pd.read_parquet(DATA)
else:
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'sample.csv', parse_dates=['timestamp'])

power = df['power'].astype(float).values
seq_len = 60
# If not enough data, pad at left with edge value to have at least seq_len+1
if len(power) < seq_len + 1:
    pad_len = seq_len + 1 - len(power)
    power = np.concatenate((np.full(pad_len, power[0]), power))

scaler = MinMaxScaler()
scaled = scaler.fit_transform(power.reshape(-1,1)).flatten()
Xs, ys = [], []
for i in range(len(scaled) - seq_len):
    Xs.append(scaled[i:i+seq_len])
    ys.append(scaled[i+seq_len])
Xs = np.array(Xs)
ys = np.array(ys)
Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
print('Training samples:', Xs.shape, ys.shape)

model = Sequential()
model.add(LSTM(16, input_shape=(seq_len,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(Xs, ys, epochs=30, batch_size=8, verbose=1)

model.save(str(MODEL_DIR / 'lstm_final.h5'))
joblib.dump(scaler, MODEL_DIR / 'lstm_scaler.joblib')
print('Saved LSTM model and scaler')
