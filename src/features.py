import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    x = df.copy()

    # Returns
    x["ret_1"] = x["Close"].pct_change(1)
    x["ret_3"] = x["Close"].pct_change(3)
    x["ret_7"] = x["Close"].pct_change(7)

    # Rolling stats
    x["ma_5"] = x["Close"].rolling(5).mean()
    x["ma_20"] = x["Close"].rolling(20).mean()
    x["vol_7"] = x["ret_1"].rolling(7).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()

    # Price-vs-MA ratios
    x["close_ma5_ratio"] = x["Close"] / x["ma_5"]
    x["close_ma20_ratio"] = x["Close"] / x["ma_20"]

    # Volume features
    x["vol_chg_1"] = x["Volume"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    x["vol_ma_7"] = x["Volume"].rolling(7).mean()
    x["volume_ratio_7"] = x["Volume"] / x["vol_ma_7"]

    # Target: next-step return
    x["target_return"] = x["Close"].shift(-horizon) / x["Close"] - 1.0
    x["target_direction"] = (x["target_return"] > 0).astype(int)

    x = x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return x

def get_feature_columns(df: pd.DataFrame):
    exclude = {"Date", "target_return", "target_direction"}
    return [c for c in df.columns if c not in exclude]