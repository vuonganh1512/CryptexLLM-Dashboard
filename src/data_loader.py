import os
import pandas as pd
import yfinance as yf

def fetch_ohlcv(symbol: str, start_date: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, interval=interval, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # Nếu columns là MultiIndex (thỉnh thoảng xảy ra), flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # đưa index thành cột
    df = df.reset_index()

    # Chuẩn hóa tên cột thời gian
    # yfinance có thể trả về: Date / Datetime / index / date
    possible_date_cols = ["Date", "Datetime", "date", "datetime", "index"]
    date_col = None
    for c in possible_date_cols:
        if c in df.columns:
            date_col = c
            break

    # Nếu vẫn chưa thấy, thử lấy cột đầu tiên nếu nó là datetime-like
    if date_col is None and len(df.columns) > 0:
        first_col = df.columns[0]
        date_col = first_col

    df = df.rename(columns={date_col: "Date"})

    # Parse datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)

    # Chuẩn hóa tên cột OHLCV (phòng trường hợp lowercase)
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    # Giữ các cột cần thiết nếu có
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    # Validate tối thiểu
    required = {"Date", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after fetch: {missing}. Columns={list(df.columns)}")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def save_raw(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)