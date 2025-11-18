import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_raw_csv(path):
    return pd.read_csv(path, parse_dates=['timestamp'])

def resample_to_minute(df, ts_col='timestamp', value_col='power'):
    df = df.copy()
    df = df.set_index(ts_col).sort_index()
    df = df[value_col].resample('1T').mean()
    df = df.interpolate(limit_direction='both')
    df = df.to_frame().reset_index().rename(columns={value_col:'power'})
    return df

def create_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['sin_hour'] = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour'] = np.cos(2*np.pi*df['hour']/24)
    df['lag_1'] = df['power'].shift(1).fillna(method='bfill')
    df['rolling_3'] = df['power'].rolling(window=3, min_periods=1).mean()
    return df

def save_processed(df, fname="sample.parquet"):
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / fname, index=False)
    print("Saved:", processed_dir / fname)

def load_processed(fname="sample.parquet"):
    return pd.read_parquet(DATA_DIR / "processed" / fname)
