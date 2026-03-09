import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CryptexLLM Dashboard", layout="wide")
st.title("CryptexLLM Dashboard")
st.caption("Interactive visual analytics for crypto forecasting + market insight (starter version)")

asset = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0)
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)

raw_path = f"data/raw/{asset}_{interval}.parquet"
feat_path = f"data/processed/{asset}_{interval}_features.parquet"
metrics_path = f"outputs/metrics/{asset}_{interval}_metrics.json"
bt_path = f"outputs/backtests/{asset}_{interval}_backtest.parquet"

tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Forecast Explorer", "Model Comparison", "Backtest"])

with tab1:
    st.subheader("Market Overview")
    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path)
        raw["Date"] = pd.to_datetime(raw["Date"])
        fig_price = px.line(raw, x="Date", y="Close", title=f"{asset} Close Price")
        st.plotly_chart(fig_price, use_container_width=True)

        if "Volume" in raw.columns:
            fig_vol = px.bar(raw.tail(180), x="Date", y="Volume", title="Volume (last 180 rows)")
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("No raw data found. Run scripts/train_and_eval.py first.")

with tab2:
    st.subheader("Forecast Explorer")
    if os.path.exists(bt_path):
        bt = pd.read_parquet(bt_path)
        bt["Date"] = pd.to_datetime(bt["Date"])

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(x=bt["Date"], y=bt["actual_return"], mode="lines", name="Actual Return"))
        fig_ret.add_trace(go.Scatter(x=bt["Date"], y=bt["pred_return"], mode="lines", name="Predicted Return"))
        fig_ret.update_layout(title="Actual vs Predicted Returns")
        st.plotly_chart(fig_ret, use_container_width=True)
    else:
        st.info("No prediction/backtest output found.")

with tab3:
    st.subheader("Model Comparison (starter: one model)")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        dfm = pd.DataFrame([metrics])
        st.dataframe(dfm, use_container_width=True)
    else:
        st.info("No metrics found. Run scripts/train_and_eval.py first.")

with tab4:
    st.subheader("Market Analysis (Backtest)")
    if os.path.exists(bt_path):
        bt = pd.read_parquet(bt_path)
        bt["Date"] = pd.to_datetime(bt["Date"])

        c1, c2 = st.columns(2)
        with c1:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=bt["Date"], y=bt["cum_strategy"], mode="lines", name="Strategy"))
            fig_cum.add_trace(go.Scatter(x=bt["Date"], y=bt["cum_market"], mode="lines", name="Market (target returns)"))
            fig_cum.update_layout(title="Cumulative Return")
            st.plotly_chart(fig_cum, use_container_width=True)

        with c2:
            fig_dd = px.line(bt, x="Date", y="drawdown", title="Strategy Drawdown")
            st.plotly_chart(fig_dd, use_container_width=True)

        st.dataframe(bt.tail(20), use_container_width=True)
    else:
        st.info("No backtest found. Run scripts/train_and_eval.py first.")