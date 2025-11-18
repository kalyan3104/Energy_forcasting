import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from app.utils import load_processed
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_xgb(processed_file="sample.parquet"):
    df = load_processed(processed_file)
    X = df[['sin_hour','cos_hour','dow','lag_1','rolling_3']].fillna(0)
    y = df['power'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "objective":"reg:squarederror",
        "eval_metric":"rmse",
        "eta":0.1,
        "max_depth":6
    }
    model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest,'test')], early_stopping_rounds=20, verbose_eval=False)
    model.save_model(MODEL_DIR / "xgb_model.bst")
    joblib.dump(list(X.columns), MODEL_DIR / "xgb_features.joblib")
    y_pred = model.predict(dtest)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("XGBoost RMSE:", rmse)
    return model

if __name__ == "__main__":
    train_xgb()
