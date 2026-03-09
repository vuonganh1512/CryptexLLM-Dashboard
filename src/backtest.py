import pandas as pd
import numpy as np

def simple_backtest(df: pd.DataFrame, fee_bps: float = 0.0) -> pd.DataFrame:
    """
    df needs columns: Date, actual_return, pred_return
    Rule: long if pred_return > 0 else flat
    """
    x = df.copy().sort_values("Date").reset_index(drop=True)
    x["position"] = (x["pred_return"] > 0).astype(int)
    x["gross_strategy_return"] = x["position"] * x["actual_return"]

    # transaction cost when position changes
    x["turnover"] = x["position"].diff().abs().fillna(0)
    x["cost"] = x["turnover"] * (fee_bps / 10000.0)
    x["net_strategy_return"] = x["gross_strategy_return"] - x["cost"]

    x["cum_market"] = (1 + x["actual_return"]).cumprod() - 1
    x["cum_strategy"] = (1 + x["net_strategy_return"]).cumprod() - 1

    # drawdown
    equity = (1 + x["net_strategy_return"]).cumprod()
    rolling_peak = equity.cummax()
    x["drawdown"] = equity / rolling_peak - 1
    return x

def backtest_summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}
    final_return = float(bt["cum_strategy"].iloc[-1])
    max_drawdown = float(bt["drawdown"].min())
    hit_rate = float((bt.loc[bt["position"] == 1, "net_strategy_return"] > 0).mean()) if (bt["position"] == 1).any() else 0.0
    return {
        "CumulativeReturn": final_return,
        "MaxDrawdown": max_drawdown,
        "HitRate": hit_rate,
    }