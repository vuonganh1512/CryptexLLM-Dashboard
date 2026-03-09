import os
import json
import pandas as pd

from src.config import Config
from src.data_loader import fetch_ohlcv, save_raw
from src.features import build_features, get_feature_columns
from src.models import time_split, train_regressor, predict
from src.evaluate import regression_metrics
from src.backtest import simple_backtest, backtest_summary

def main():
    cfg = Config  # đổi coin ở đây

    raw_path = f"{cfg.raw_dir}/{cfg.asset}_{cfg.interval}.parquet"
    proc_path = f"{cfg.processed_dir}/{cfg.asset}_{cfg.interval}_features.parquet"
    os.makedirs(cfg.processed_dir, exist_ok=True)
    os.makedirs(f"{cfg.outputs_dir}/metrics", exist_ok=True)
    os.makedirs(f"{cfg.outputs_dir}/backtests", exist_ok=True)

    # 1) Fetch data
    raw_df = fetch_ohlcv(cfg.asset, cfg.start_date, cfg.interval)
    save_raw(raw_df, raw_path)

    # 2) Build features
    feat_df = build_features(raw_df, horizon=cfg.target_horizon)
    feat_df.to_parquet(proc_path, index=False)

    # 3) Split
    train_df, test_df = time_split(feat_df, test_size=cfg.test_size)
    feature_cols = get_feature_columns(feat_df)

    X_train = train_df[feature_cols]
    y_train = train_df["target_return"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_return"]

    # 4) Train + predict
    model = train_regressor(X_train, y_train)
    y_pred = predict(model, X_test)

    # 5) Metrics
    metrics = regression_metrics(y_test, y_pred)
    metrics_path = f"{cfg.outputs_dir}/metrics/{cfg.asset}_{cfg.interval}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 6) Backtest
    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "actual_return": y_test.values,
        "pred_return": y_pred,
    })
    bt = simple_backtest(pred_df, fee_bps=5.0)
    bt_path = f"{cfg.outputs_dir}/backtests/{cfg.asset}_{cfg.interval}_backtest.parquet"
    bt.to_parquet(bt_path, index=False)

    bt_summary = backtest_summary(bt)
    print("Forecast metrics:", metrics)
    print("Backtest summary:", bt_summary)
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved backtest -> {bt_path}")

if __name__ == "__main__":
    main()