import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true > 0) == (y_pred > 0)))

def spearman_corr(y_true, y_pred):
    # Spearman = rank correlation, ổn cho noisy returns
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rt = y_true.argsort().argsort().astype(float)
    rp = y_pred.argsort().argsort().astype(float)
    rt = (rt - rt.mean()) / (rt.std() + 1e-8)
    rp = (rp - rp.mean()) / (rp.std() + 1e-8)
    return float(np.mean(rt * rp))

def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    yt = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
    yp = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)
    return float(np.mean(yt * yp))

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "sMAPE": smape(y_true, y_pred),
        "DirectionalAccuracy": directional_accuracy(y_true, y_pred),
        "Spearman": spearman_corr(y_true, y_pred),
        "Pearson": pearson_corr(y_true, y_pred),  # aka IC (information coefficient)
    }