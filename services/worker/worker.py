import os, json
import pandas as pd
import redis

from src.config import Config
from src.data_loader import fetch_ohlcv
from src.features import build_features, get_feature_columns
from src.models import time_split, train_regressor, predict
from src.evaluate import regression_metrics
from src.backtest import simple_backtest, backtest_summary
from src.explain import generate_explanation

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)

def key_metrics(asset: str, interval: str):
    return f"metrics:{asset}:{interval}"

def key_explain(asset: str, interval: str):
    return f"explain:{asset}:{interval}"

def key_summary():
    return "summary:phase2"

def key_bt(asset: str, interval: str, which: str):
    return f"bt:{which}:{asset}:{interval}"

def run_job(job: dict):
    asset = job.get("asset", "BTC-USD")
    interval = job.get("interval", "1d")
    threshold = float(job.get("threshold", 0.002))
    fee_bps = float(job.get("fee_bps", 5.0))
    start = job.get("start_date", "2021-01-01")
    test_size = float(job.get("test_size", 0.2))
    horizon = int(job.get("horizon", 1))

    cfg = Config(asset=asset, start_date=start, interval=interval, test_size=test_size, target_horizon=horizon)

    raw_df = fetch_ohlcv(cfg.asset, cfg.start_date, cfg.interval)
    feat_df = build_features(raw_df, horizon=cfg.target_horizon)
    train_df, test_df = time_split(feat_df, test_size=cfg.test_size)

    feature_cols = get_feature_columns(feat_df)
    X_train, y_train = train_df[feature_cols], train_df["target_return"]
    X_test, y_test = test_df[feature_cols], test_df["target_return"]

    model = train_regressor(X_train, y_train)
    y_pred = predict(model, X_test)

    # Naive benchmark
    y_naive = test_df["ret_1"].values

    # Metrics
    metrics_model = regression_metrics(y_test.values, y_pred)
    metrics_naive = regression_metrics(y_test.values, y_naive)

    # Backtest inputs
    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "actual_return": y_test.values,
        "pred_return": y_pred
    })
    naive_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "actual_return": y_test.values,
        "pred_return": y_naive
    })

    bt_model = simple_backtest(pred_df, fee_bps=fee_bps, threshold=threshold)
    bt_naive = simple_backtest(naive_df, fee_bps=fee_bps, threshold=threshold)

    # ---- Backtest summaries (store once, reuse for metrics + explain) ----
    model_bt_summary = backtest_summary(bt_model)
    naive_bt_summary = backtest_summary(bt_naive)

    # ---- SAVE METRICS FIRST (fixes 404 until worker finishes) ----
    mkey = key_metrics(asset, interval)
    metrics_payload = {
        "asset": asset,
        "interval": interval,
        "horizon": horizon,
        "threshold": threshold,
        "fee_bps": fee_bps,
        "model": metrics_model,
        "naive": metrics_naive,
        "model_bt_summary": model_bt_summary,
        "naive_bt_summary": naive_bt_summary,
    }
    r.set(mkey, json.dumps(metrics_payload))
    print("[worker] wrote metrics key:", mkey)

    # ---- SAVE BACKTESTS (convert Date to ISO string) ----
    bt_model_out = bt_model.tail(2000).copy()
    bt_naive_out = bt_naive.tail(2000).copy()

    if "Date" in bt_model_out.columns:
        bt_model_out["Date"] = pd.to_datetime(bt_model_out["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "Date" in bt_naive_out.columns:
        bt_naive_out["Date"] = pd.to_datetime(bt_naive_out["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    r.set(key_bt(asset, interval, "model"), json.dumps(bt_model_out.to_dict(orient="records")))
    r.set(key_bt(asset, interval, "naive"), json.dumps(bt_naive_out.to_dict(orient="records")))
    print("[worker] wrote backtest keys:", key_bt(asset, interval, "model"), "and", key_bt(asset, interval, "naive"))

    # ---- SAVE EXPLANATION (CryptexLLM Option A) ----
    analysis_pkg = {
        "asset": asset,
        "interval": interval,
        "threshold": threshold,
        "fee_bps": fee_bps,
        "metrics": {"model": metrics_model, "naive": metrics_naive},
        "backtest_summary": {"model": model_bt_summary, "naive": naive_bt_summary},
    }
    explain_text = generate_explanation(analysis_pkg)
    ekey = key_explain(asset, interval)
    r.set(ekey, explain_text)
    print("[worker] wrote explain key:", ekey)

def main():
    print("[worker] started. waiting for jobs on jobs:train ...")
    while True:
        item = r.brpop("jobs:train", timeout=5)
        if not item:
            continue
        _, payload = item
        job = json.loads(payload)
        try:
            print("[worker] running job:", job)
            run_job(job)
            print("[worker] done:", job)
        except Exception as e:
            print("[worker] error:", e)

if __name__ == "__main__":
    main()