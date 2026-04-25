# src/explain.py
from __future__ import annotations
from typing import Dict, Any

def generate_explanation_template(pkg: Dict[str, Any]) -> str:
    asset = pkg.get("asset", "N/A")
    interval = pkg.get("interval", "N/A")
    threshold = pkg.get("threshold")
    fee_bps = pkg.get("fee_bps")

    m = pkg.get("metrics", {})
    model = m.get("model", {})
    naive = m.get("naive", {})

    bt = pkg.get("backtest_summary", {})
    bt_model = bt.get("model", {})
    bt_naive = bt.get("naive", {})

    def f(x, nd=4):
        try: return f"{float(x):.{nd}f}"
        except: return "N/A"

    def pct(x):
        try: return f"{float(x)*100:.2f}%"
        except: return "N/A"

    lines = []
    lines.append(f"### CryptexLLM Explanation - {asset} ({interval})")
    lines.append("")
    lines.append(f"**Strategy settings:** threshold={threshold}, fee_bps={fee_bps}")
    lines.append("")
    lines.append("**Model vs Naive (prediction quality):**")
    lines.append(f"- MAE: model={f(model.get('MAE'))}, naive={f(naive.get('MAE'))}")
    lines.append(f"- RMSE: model={f(model.get('RMSE'))}, naive={f(naive.get('RMSE'))}")
    lines.append(f"- DirectionalAccuracy: model={f(model.get('DirectionalAccuracy'))}, naive={f(naive.get('DirectionalAccuracy'))}")
    lines.append("")
    lines.append("**Backtest (trading relevance):**")
    lines.append(f"- CumulativeReturn: model={pct(bt_model.get('CumulativeReturn'))}, naive={pct(bt_naive.get('CumulativeReturn'))}")
    lines.append(f"- MaxDrawdown: model={pct(bt_model.get('MaxDrawdown'))}, naive={pct(bt_naive.get('MaxDrawdown'))}")
    lines.append(f"- HitRate: model={f(bt_model.get('HitRate'))}, naive={f(bt_naive.get('HitRate'))}")
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("- The LLM layer summarizes metrics and backtest outcomes into a short narrative.")
    lines.append("- Threshold long/flat reduces exposure when the signal is weak, which can reduce overtrading and fee drag.")
    lines.append("- Compare CumulativeReturn vs MaxDrawdown to judge return vs risk.")
    return "\n".join(lines)