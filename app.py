import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import httpx

st.set_page_config(page_title="CryptexLLM Dashboard (Phase 3)", layout="wide")
st.title("CryptexLLM Dashboard")
st.caption("Phase 3: Distributed (API + Worker + Redis) — Streamlit reads from API")

API_BASE = st.sidebar.text_input("API Base URL", "http://localhost:8000")

asset = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0)
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)
threshold = st.sidebar.number_input("Threshold (e.g. 0.002 = 0.2%)", value=0.002, step=0.001, format="%.6f")
fee_bps = st.sidebar.number_input("Fee (bps)", value=5.0, step=1.0)
which_bt = st.sidebar.radio("Backtest to view", ["model", "naive"], horizontal=True)

def fetch_json(path: str):
    url = f"{API_BASE}{path}"
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def post_job():
    url = f"{API_BASE}/jobs/train"
    params = {"asset": asset, "interval": interval, "threshold": threshold, "fee_bps": fee_bps}
    r = httpx.post(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

c0, c1, c2 = st.columns([1, 1, 2])
with c0:
    if st.button("Run Job (enqueue)"):
        try:
            resp = post_job()
            st.success(f"Queued: {resp.get('job')}")
        except Exception as e:
            st.error(f"Failed to enqueue job: {e}")

with c1:
    if st.button("Refresh"):
        st.rerun()

with c2:
    st.info("Tip: Run Job → wait a bit → Refresh to see updated metrics/backtest/explanation.")

# ✅ Added new tab: LLM Explanation
tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Backtest", "LLM Explanation", "Raw JSON"])

# -------------------------
# Metrics tab
# -------------------------
with tab1:
    st.subheader(f"Metrics — {asset} ({interval})")
    try:
        metrics = fetch_json(f"/metrics/{asset}?interval={interval}")
        model_m = metrics.get("model", {})
        naive_m = metrics.get("naive", {})

        dfm = pd.DataFrame([
            {"which": "model", **model_m},
            {"which": "naive", **naive_m},
        ])
        st.dataframe(dfm, width="stretch")

        # Backtest summary
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("### Model backtest summary")
            st.json(metrics.get("model_bt_summary", {}))
        with s2:
            st.markdown("### Naive backtest summary")
            st.json(metrics.get("naive_bt_summary", {}))

    except httpx.HTTPStatusError:
        st.warning("No metrics yet. Run Job (enqueue) first.")
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")

# -------------------------
# Backtest tab
# -------------------------
with tab2:
    st.subheader(f"Backtest — {which_bt} — {asset} ({interval})")
    try:
        bt = fetch_json(f"/backtest/{which_bt}/{asset}?interval={interval}")
        bt_df = pd.DataFrame(bt)
        if "Date" in bt_df.columns:
            bt_df["Date"] = pd.to_datetime(bt_df["Date"], errors="coerce")

        cA, cB = st.columns(2)
        with cA:
            fig_cum = go.Figure()
            if "cum_strategy" in bt_df.columns:
                fig_cum.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["cum_strategy"], mode="lines", name="strategy"))
            if "cum_market" in bt_df.columns:
                fig_cum.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["cum_market"], mode="lines", name="market"))
            fig_cum.update_layout(title="Cumulative Return")
            st.plotly_chart(fig_cum, width="stretch")

        with cB:
            if "drawdown" in bt_df.columns:
                fig_dd = px.line(bt_df, x="Date", y="drawdown", title="Drawdown")
                st.plotly_chart(fig_dd, width="stretch")
            else:
                st.info("No drawdown column found in backtest payload.")

        st.dataframe(bt_df.tail(50), width="stretch")

    except httpx.HTTPStatusError:
        st.warning("No backtest yet. Run Job (enqueue) first.")
    except Exception as e:
        st.error(f"Error fetching backtest: {e}")

# -------------------------
# LLM Explanation tab (Option A)
# -------------------------
with tab3:
    st.subheader(f"LLM Explanation — {asset} ({interval})")
    st.caption("This is a natural-language summary generated from metrics + backtest summaries and cached in Redis as explain:*")

    try:
        payload = fetch_json(f"/explain/{asset}?interval={interval}")
        explanation = payload.get("explanation", "")
        if explanation:
            st.markdown(explanation)
        else:
            st.info("Explanation is empty. Run Job and refresh.")
    except httpx.HTTPStatusError:
        st.warning("No explanation yet. Run Job (enqueue) first, wait for the worker, then Refresh.")
    except Exception as e:
        st.error(f"Error fetching explanation: {e}")

# -------------------------
# Raw JSON tab
# -------------------------
with tab4:
    st.subheader("Raw JSON (debug)")

    try:
        st.markdown("### /metrics")
        st.json(fetch_json(f"/metrics/{asset}?interval={interval}"))
    except Exception:
        st.info("Metrics not available yet.")

    try:
        st.markdown("### /backtest")
        st.json(fetch_json(f"/backtest/{which_bt}/{asset}?interval={interval}")[:5])
        st.caption("Showing first 5 rows only.")
    except Exception:
        st.info("Backtest not available yet.")

    try:
        st.markdown("### /explain")
        st.json(fetch_json(f"/explain/{asset}?interval={interval}"))
    except Exception:
        st.info("Explain not available yet.")