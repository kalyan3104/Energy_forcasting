"""
Train ARIMA model (simple pipeline).
Usage: python train_arima.py
"""
import argparse
import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
from app.utils import load_processed, save_processed

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_arima(processed_file="sample.parquet", order=(5,1,0)):
    df = load_processed(processed_file)
    ts = df.set_index('timestamp')['power'].asfreq('T').fillna(method='ffill')
    model = ARIMA(ts, order=order).fit()
    joblib.dump(model, MODEL_DIR / "arima_model.pkl")
    print("Saved ARIMA model to", MODEL_DIR / "arima_model.pkl")
    return model

if __name__ == "__main__":
    train_arima()
