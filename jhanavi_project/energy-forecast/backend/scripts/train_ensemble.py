import joblib
import numpy as np
from xgboost import DMatrix
from pathlib import Path
from app.utils import load_processed
from statsmodels.tsa.arima.model import ARIMAResults
import xgboost as xgb
import json

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models"

def load_models():
    arima = joblib.load(MODEL_DIR / "arima_model.pkl")
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.bst"))
    lstm_model_path = MODEL_DIR / "lstm_final.h5"
    return arima, xgb_model, lstm_model_path

def compute_ensemble_weights(val_y, preds):
    P = np.column_stack([preds[k] for k in preds])
    w, *_ = np.linalg.lstsq(P, val_y, rcond=None)
    w = np.maximum(w, 0)
    if w.sum() == 0:
        w = np.ones_like(w)/len(w)
    else:
        w = w / w.sum()
    return w

if __name__ == "__main__":
    print("Ensemble script placeholder - implement validation-based weight tuning in your experiments.")
