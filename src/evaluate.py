import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def directional_accuracy(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) > 0) == (np.asarray(y_pred) > 0)))

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "MAPE": float(mape(y_true, y_pred)),
        "DirectionalAccuracy": float(directional_accuracy(y_true, y_pred)),
    }

def rolling_mae(df: pd.DataFrame, window: int = 30):
    out = df.copy()
    out["abs_err"] = (out["y_true"] - out["y_pred"]).abs()
    out["rolling_mae"] = out["abs_err"].rolling(window).mean()
    return out