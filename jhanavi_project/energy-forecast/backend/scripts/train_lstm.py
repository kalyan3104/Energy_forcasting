import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
import joblib
from app.utils import load_processed

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def create_sequences(X, y, seq_len=60):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

def train_lstm(processed_file="sample.parquet", seq_len=60):
    df = load_processed(processed_file)
    features = ['power','sin_hour','cos_hour','dow','lag_1','rolling_3']
    arr = df[features].fillna(0).values
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(arr)
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")
    X_all = arr_scaled[:,1:]
    y_all = arr_scaled[:,0]
    X_seq, y_seq = create_sequences(X_all, y_all, seq_len)
    train_size = int(0.7*len(X_seq))
    val_size = int(0.15*len(X_seq))
    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    checkpoint = ModelCheckpoint(str(MODEL_DIR / "lstm_model.h5"), save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, callbacks=[checkpoint, early], verbose=1)
    model.save(MODEL_DIR / "lstm_final.h5")
    print("Saved LSTM model")
    best = model
    y_pred = best.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("LSTM RMSE (scaled):", rmse)
    return model

if __name__ == "__main__":
    train_lstm()
