from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def time_split(df: pd.DataFrame, test_size: float = 0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def train_regressor(X_train: pd.DataFrame, y_train: pd.Series):
    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model

def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)