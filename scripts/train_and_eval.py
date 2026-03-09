import os
import json
import argparse
import pandas as pd

from src.config import Config
from src.data_loader import fetch_ohlcv, save_raw
from src.features import build_features, get_feature_columns
from src.models import time_split, train_regressor, predict
from src.evaluate import regression_metrics
from src.backtest import simple_backtest, backtest_summary

def ensure_dirs(cfg: Config):
    os.makedirs(cfg.raw_dir, exist_ok=True)
    os.makedirs(cfg.processed_dir, exist_ok=True)
    os.makedirs(f"{cfg.outputs_dir}/metrics", exist_ok=True)
    os.makedirs(f"{cfg.outputs_dir}/backtests", exist_ok=True)
    os.makedirs(f"{cfg.outputs_dir}/tables", exist_ok=True)

def run_one_asset(asset: str, start_date: str, interval: str, test_size: float, horizon: int,
                  fee_bps: float, threshold: float) -> dict:
    cfg = Config(asset=asset, start_date=start_date, interval=interval, test_size=test_size, target_horizon=horizon)
    ensure_dirs(cfg)

    raw_path = f"{cfg.raw_dir}/{cfg.asset}_{cfg.interval}.parquet"
    proc_path = f"{cfg.processed_dir}/{cfg.asset}_{cfg.interval}_features.parquet"

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

    # 4) Model: XGB (or fallback)
    model = train_regressor(X_train, y_train)
    y_pred = predict(model, X_test)

    # 5) Naive benchmark: predict next return = today's ret_1
    # ret_1 exists in features; align with test rows
    y_naive = test_df["ret_1"].values

    # 6) Metrics
    metrics_model = regression_metrics(y_test.values, y_pred)
    metrics_naive = regression_metrics(y_test.values, y_naive)

    metrics_path = f"{cfg.outputs_dir}/metrics/{cfg.asset}_{cfg.interval}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {"asset": cfg.asset, "interval": cfg.interval, "horizon": cfg.target_horizon,
             "model": metrics_model, "naive": metrics_naive},
            f, indent=2
        )

    # 7) Backtests
    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "actual_return": y_test.values,
        "pred_return": y_pred,
    })
    naive_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "actual_return": y_test.values,
        "pred_return": y_naive,
    })

    bt_model = simple_backtest(pred_df, fee_bps=fee_bps, threshold=threshold)
    bt_naive = simple_backtest(naive_df, fee_bps=fee_bps, threshold=threshold)

    bt_model_path = f"{cfg.outputs_dir}/backtests/{cfg.asset}_{cfg.interval}_backtest_model.parquet"
    bt_naive_path = f"{cfg.outputs_dir}/backtests/{cfg.asset}_{cfg.interval}_backtest_naive.parquet"
    bt_model.to_parquet(bt_model_path, index=False)
    bt_naive.to_parquet(bt_naive_path, index=False)

    summ_model = backtest_summary(bt_model)
    summ_naive = backtest_summary(bt_naive)

    # 8) Return one-row summary (for tables)
    row = {
        "asset": cfg.asset,
        "interval": cfg.interval,
        "horizon": cfg.target_horizon,
        "threshold": threshold,
        "fee_bps": fee_bps,

        "model_MAE": metrics_model["MAE"],
        "model_RMSE": metrics_model["RMSE"],
        "model_sMAPE": metrics_model["sMAPE"],
        "model_DA": metrics_model["DirectionalAccuracy"],
        "model_Spearman": metrics_model["Spearman"],
        "model_Pearson": metrics_model["Pearson"],
        "model_CumReturn": summ_model.get("CumulativeReturn", 0.0),
        "model_MaxDD": summ_model.get("MaxDrawdown", 0.0),
        "model_Trades": summ_model.get("Trades", 0),
        "model_Exposure": summ_model.get("Exposure", 0.0),

        "naive_MAE": metrics_naive["MAE"],
        "naive_RMSE": metrics_naive["RMSE"],
        "naive_sMAPE": metrics_naive["sMAPE"],
        "naive_DA": metrics_naive["DirectionalAccuracy"],
        "naive_Spearman": metrics_naive["Spearman"],
        "naive_Pearson": metrics_naive["Pearson"],
        "naive_CumReturn": summ_naive.get("CumulativeReturn", 0.0),
        "naive_MaxDD": summ_naive.get("MaxDrawdown", 0.0),
        "naive_Trades": summ_naive.get("Trades", 0),
        "naive_Exposure": summ_naive.get("Exposure", 0.0),
    }
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=str, default="BTC-USD",
                        help="Comma-separated list, e.g. BTC-USD,ETH-USD,SOL-USD")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--fee_bps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.0, help="e.g. 0.002 for +0.2%")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    rows = []
    for asset in assets:
        row = run_one_asset(
            asset=asset,
            start_date=args.start,
            interval=args.interval,
            test_size=args.test_size,
            horizon=args.horizon,
            fee_bps=args.fee_bps,
            threshold=args.threshold,
        )
        rows.append(row)
        print(f"[OK] {asset} metrics+backtests saved.")

    table = pd.DataFrame(rows).sort_values(["asset"])
    os.makedirs("outputs/tables", exist_ok=True)
    out_csv = "outputs/tables/summary_phase2.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nSaved summary table -> {out_csv}")
    print(table[["asset","model_DA","model_CumReturn","naive_DA","naive_CumReturn","threshold","fee_bps"]])

if __name__ == "__main__":
    main()