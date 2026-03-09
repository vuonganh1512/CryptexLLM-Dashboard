import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CryptexLLM Dashboard (Phase 2)", layout="wide")
st.title("CryptexLLM Dashboard")
st.caption("Phase 2: Multi-asset + Model vs Naive + Threshold backtest")

# Sidebar controls
asset = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0)
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)
which_bt = st.sidebar.radio("Backtest to view", ["model", "naive"], horizontal=True)

# Paths (Phase 2)
raw_path = f"data/raw/{asset}_{interval}.parquet"
metrics_path = f"outputs/metrics/{asset}_{interval}_metrics.json"
bt_model_path = f"outputs/backtests/{asset}_{interval}_backtest_model.parquet"
bt_naive_path = f"outputs/backtests/{asset}_{interval}_backtest_naive.parquet"
summary_path = "outputs/tables/summary_phase2.csv"

tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Forecast Explorer", "Model Comparison", "Backtest"])

# -------------------------
# Tab 1: Market Overview
# -------------------------
with tab1:
    st.subheader("Market Overview")
    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path)
        raw["Date"] = pd.to_datetime(raw["Date"])

        fig_price = px.line(raw, x="Date", y="Close", title=f"{asset} Close Price")
        st.plotly_chart(fig_price, width="stretch")

        if "Volume" in raw.columns:
            fig_vol = px.bar(raw.tail(180), x="Date", y="Volume", title="Volume (last 180 rows)")
            st.plotly_chart(fig_vol, width="stretch")
    else:
        st.info("No raw data found. Run Phase 2 pipeline to generate data/raw files.")

# -------------------------
# Tab 2: Forecast Explorer
# -------------------------
with tab2:
    st.subheader("Forecast Explorer (Actual vs Predicted Returns)")
    bt_path = bt_model_path if which_bt == "model" else bt_naive_path

    if os.path.exists(bt_path):
        bt = pd.read_parquet(bt_path)
        bt["Date"] = pd.to_datetime(bt["Date"])

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(x=bt["Date"], y=bt["actual_return"], mode="lines", name="Actual Return"))
        fig_ret.add_trace(go.Scatter(x=bt["Date"], y=bt["pred_return"], mode="lines", name=f"Pred Return ({which_bt})"))
        fig_ret.update_layout(title=f"{asset} — Actual vs Predicted Returns ({which_bt})")
        st.plotly_chart(fig_ret, width="stretch")

        st.dataframe(bt[["Date", "actual_return", "pred_return"]].tail(30), width="stretch")
    else:
        st.info("No backtest output found. Run: python -m scripts.train_and_eval (Phase 2).")

# -------------------------
# Tab 3: Model Comparison
# -------------------------
with tab3:
    st.subheader("Model Comparison (Model vs Naive)")

    c1, c2 = st.columns(2)

    # Show per-asset metrics.json (model + naive)
    with c1:
        st.markdown("### Metrics for selected asset")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                m = json.load(f)

            # Phase 2 metrics JSON structure: {"asset":..., "interval":..., "horizon":..., "model":{...}, "naive":{...}}
            model_metrics = m.get("model", {})
            naive_metrics = m.get("naive", {})

            dfm = pd.DataFrame([
                {"which": "model", **model_metrics},
                {"which": "naive", **naive_metrics},
            ])
            st.dataframe(dfm, width="stretch")
        else:
            st.info("No metrics file found for this asset yet.")

    # Show global summary table (multi-asset)
    with c2:
        st.markdown("### Phase 2 summary table (all assets)")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            st.dataframe(summary, width="stretch")

            # Quick bar chart: cumulative return model vs naive
            if "model_CumReturn" in summary.columns and "naive_CumReturn" in summary.columns:
                fig_bar = px.bar(
                    summary,
                    x="asset",
                    y=["model_CumReturn", "naive_CumReturn"],
                    barmode="group",
                    title="Cumulative Return: Model vs Naive"
                )
                st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No summary table found. Run Phase 2 pipeline to generate outputs/tables/summary_phase2.csv")

# -------------------------
# Tab 4: Backtest
# -------------------------
with tab4:
    st.subheader(f"Backtest ({which_bt})")
    bt_path = bt_model_path if which_bt == "model" else bt_naive_path

    if os.path.exists(bt_path):
        bt = pd.read_parquet(bt_path)
        bt["Date"] = pd.to_datetime(bt["Date"])

        c1, c2 = st.columns(2)
        with c1:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=bt["Date"], y=bt["cum_strategy"], mode="lines", name=f"{which_bt} strategy"))
            fig_cum.add_trace(go.Scatter(x=bt["Date"], y=bt["cum_market"], mode="lines", name="market"))
            fig_cum.update_layout(title="Cumulative Return")
            st.plotly_chart(fig_cum, width="stretch")

        with c2:
            fig_dd = px.line(bt, x="Date", y="drawdown", title="Drawdown")
            st.plotly_chart(fig_dd, width="stretch")

        st.dataframe(bt.tail(20), width="stretch")
    else:
        st.info("No backtest found. Run Phase 2 pipeline first.")