from __future__ import annotations

import os
import json
from typing import Dict, Any


# -----------------------------
# 1) Template fallback
# -----------------------------
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
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "N/A"

    def pct(x):
        try:
            return f"{float(x)*100:.2f}%"
        except Exception:
            return "N/A"

    mae_m, mae_n = model.get("MAE"), naive.get("MAE")
    rmse_m, rmse_n = model.get("RMSE"), naive.get("RMSE")
    da_m, da_n = model.get("DirectionalAccuracy"), naive.get("DirectionalAccuracy")

    cr_m, cr_n = bt_model.get("CumulativeReturn"), bt_naive.get("CumulativeReturn")
    dd_m, dd_n = bt_model.get("MaxDrawdown"), bt_naive.get("MaxDrawdown")
    hr_m, hr_n = bt_model.get("HitRate"), bt_naive.get("HitRate")
    tr_m, tr_n = bt_model.get("Trades"), bt_naive.get("Trades")
    ex_m, ex_n = bt_model.get("Exposure"), bt_naive.get("Exposure")

    better_pred = []
    try:
        if mae_m is not None and mae_n is not None and float(mae_m) < float(mae_n):
            better_pred.append("lower MAE")
        if rmse_m is not None and rmse_n is not None and float(rmse_m) < float(rmse_n):
            better_pred.append("lower RMSE")
        if da_m is not None and da_n is not None and float(da_m) > float(da_n):
            better_pred.append("higher directional accuracy")
    except Exception:
        pass

    lines = []
    lines.append(f"## CryptexLLM Explanation Report - {asset} ({interval})")
    lines.append("")
    lines.append("### 1) Setup")
    lines.append(f"- Threshold: `{threshold}` (trade only when predicted return exceeds threshold)")
    lines.append(f"- Fee: `{fee_bps}` bps per trade (models with many trades suffer more fee drag)")
    lines.append("")

    lines.append("### 2) Prediction Quality (Model vs Naive)")
    lines.append(f"- MAE: model={f(mae_m)}, naive={f(mae_n)}")
    lines.append(f"- RMSE: model={f(rmse_m)}, naive={f(rmse_n)}")
    lines.append(f"- DirectionalAccuracy: model={f(da_m)}, naive={f(da_n)}")
    lines.append("")

    lines.append("### 3) Backtest Performance (Trading Relevance)")
    lines.append(f"- CumulativeReturn: model={pct(cr_m)}, naive={pct(cr_n)}")
    lines.append(f"- MaxDrawdown: model={pct(dd_m)}, naive={pct(dd_n)}")
    lines.append(f"- HitRate: model={f(hr_m)}, naive={f(hr_n)}")
    lines.append(f"- Trades: model={tr_m if tr_m is not None else 'N/A'}, naive={tr_n if tr_n is not None else 'N/A'}")
    lines.append(f"- Exposure: model={f(ex_m)}, naive={f(ex_n)}")
    lines.append("")

    lines.append("### 4) Key Takeaways")
    if better_pred:
        lines.append(f"- The model shows **{', '.join(better_pred)}** compared to the naive baseline.")
    else:
        lines.append("- The model does not clearly outperform the naive baseline on prediction metrics.")
    lines.append("- **Prediction metrics and trading results can disagree**: a small error improvement may not translate to profitable trades.")
    lines.append("- Thresholding reduces exposure; fewer trades can reduce fees but may miss opportunities.")
    lines.append("")

    lines.append("### 5) Why metrics can look good but P&L can be negative")
    lines.append("- **Returns are near-zero most days**: even low MAE/RMSE can still miss the sign/magnitude needed to trade.")
    lines.append("- **Threshold + fees create a hurdle**: the signal must exceed threshold and overcome transaction costs.")
    lines.append("- **Timing matters**: slightly wrong timing can flip trade outcomes even when average error is lower.")
    lines.append("")

    lines.append("### 6) Risk Notes")
    lines.append("- Focus on **MaxDrawdown**: large drawdown indicates the strategy suffers in certain periods.")
    lines.append("- If Trades is high, consider increasing threshold or reducing churn to lower fee drag.")
    lines.append("")

    lines.append("### 7) Recommended Next Steps")
    lines.append("- Tune `threshold` and `fee_bps` sensitivity (grid-search a few values).")
    lines.append("- Add walk-forward evaluation to avoid overfitting to one split.")
    lines.append("- Compare against another baseline (e.g., moving-average signal) to contextualize performance.")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# 2) Real LLM call (OpenAI) + visible marker + debug
# -----------------------------
def generate_explanation(pkg: Dict[str, Any]) -> str:
    """
    Real LLM explanation using OpenAI. Falls back to template if missing key or errors.

    Controls:
      - OPENAI_API_KEY (required for LLM)
      - OPENAI_MODEL (default: gpt-5.4-mini)
      - EXPLAIN_DEBUG=1 (optional): append error details to output if fallback happens
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
    debug = os.getenv("EXPLAIN_DEBUG", "0").strip() == "1"

    if not api_key:
        msg = "[explain] OPENAI_API_KEY missing -> fallback template"
        print(msg)
        out = generate_explanation_template(pkg)
        if debug:
            out = out + "\n\n---\n**DEBUG:** " + msg
        return out

    # Keep input compact (cheaper + stable)
    compact_pkg = {
        "asset": pkg.get("asset"),
        "interval": pkg.get("interval"),
        "threshold": pkg.get("threshold"),
        "fee_bps": pkg.get("fee_bps"),
        "metrics": pkg.get("metrics", {}),
        "backtest_summary": pkg.get("backtest_summary", {}),
    }

    instructions = (
        "You are CryptexLLM, a technical assistant that explains crypto forecasting results.\n"
        "Given a JSON package with model vs naive metrics and backtest summaries, write a detailed but readable report.\n"
        "Requirements:\n"
        "- Use Markdown with clear section headers.\n"
        "- Explain how to interpret each metric briefly.\n"
        "- Compare model vs naive and highlight trade-offs.\n"
        "- If prediction metrics improve but backtest is negative, explain plausible reasons (fees, thresholding, noise, timing).\n"
        "- Do NOT give financial advice. Only interpret the provided numbers.\n"
        "- Avoid generic boilerplate; reference the actual numeric values.\n"
    )

    user_input = (
        "Generate a natural-language explanation based only on this JSON:\n\n"
        + json.dumps(compact_pkg, ensure_ascii=False)
    )

    try:
        from openai import OpenAI  # requires: openai>=1.0.0
        client = OpenAI(api_key=api_key)

        print(f"[explain] calling OpenAI model={model} asset={compact_pkg.get('asset')} interval={compact_pkg.get('interval')}")
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
        )

        text = getattr(resp, "output_text", None)
        if not text or not str(text).strip():
            msg = "[explain] OpenAI returned empty output_text -> fallback template"
            print(msg)
            out = generate_explanation_template(pkg)
            if debug:
                out = out + "\n\n---\n**DEBUG:** " + msg
            return out

        # Marker to prove it came from OpenAI call
        marker = f"✅ **Generated by OpenAI** (`{model}`)\n\n"
        return marker + str(text).strip()

    except Exception as e:
        msg = f"[explain] OpenAI call failed -> fallback template: {repr(e)}"
        print(msg)
        out = generate_explanation_template(pkg)
        if debug:
            out = out + "\n\n---\n**DEBUG:** " + msg
        return out