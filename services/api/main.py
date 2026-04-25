import os, json
from fastapi import FastAPI, HTTPException
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="CryptexLLM API (Phase 3)")

def key_metrics(asset: str, interval: str): 
    return f"metrics:{asset}:{interval}"

def key_explain(asset: str, interval: str):
    return f"explain:{asset}:{interval}"

def key_summary(): 
    return "summary:phase2"

def key_bt(asset: str, interval: str, which: str): 
    return f"bt:{which}:{asset}:{interval}"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/jobs/train")
def enqueue_train_job(asset: str = "BTC-USD", interval: str = "1d", threshold: float = 0.002, fee_bps: float = 5.0):
    job = {"asset": asset, "interval": interval, "threshold": threshold, "fee_bps": fee_bps}
    r.lpush("jobs:train", json.dumps(job))
    return {"queued": True, "job": job}

@app.get("/metrics/{asset}")
def get_metrics(asset: str, interval: str = "1d"):
    data = r.get(key_metrics(asset, interval))
    if not data:
        raise HTTPException(404, "No metrics yet. Run a job first.")
    return json.loads(data)

@app.get("/explain/{asset}")
def get_explain(asset: str, interval: str = "1d"):
    data = r.get(key_explain(asset, interval))
    if not data:
        raise HTTPException(404, "No explanation yet. Run a job first.")
    # stored as plain text markdown from worker
    return {"asset": asset, "interval": interval, "explanation": data}

@app.get("/summary")
def get_summary():
    data = r.get(key_summary())
    if not data:
        raise HTTPException(404, "No summary yet. Run multi-asset jobs first.")
    return json.loads(data)

@app.get("/backtest/{which}/{asset}")
def get_backtest(which: str, asset: str, interval: str = "1d"):
    if which not in ("model", "naive"):
        raise HTTPException(400, "which must be model|naive")
    data = r.get(key_bt(asset, interval, which))
    if not data:
        raise HTTPException(404, "No backtest yet.")
    return json.loads(data)