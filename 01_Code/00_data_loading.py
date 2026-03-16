"""
00_data_loading.py
==================
Load and clean the raw CSV data for ICE London Cocoa Futures.

Two datasets:
  - Daily OHLCV  (~40 years, 1974-2025)
  - 5-minute intraday (~1 year, Feb 2025 - Jan 2026)

Output: cleaned DataFrames saved as CSV in 00_Data/processed/
"""

import pandas as pd
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "00_Data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


# ===================================================================
# 1. DAILY DATA
# ===================================================================
def load_daily():
    """
    Load Daily.csv (German locale, semicolon-separated).

    Format quirks:
      - Two header rows (row 0 = instrument, row 1 = column names)
      - Dates: DD.MM.YYYY
      - Volume: has 'K' suffix (e.g. '5.9 K' means 5900)
      - Data is sorted descending (newest first)
    """
    path = os.path.join(RAW_DIR, "Daily.csv")

    # Skip the first header row (instrument name), use row 2 as header
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=1,          # skip the instrument-name row
        encoding="utf-8-sig"  # handle BOM
    )

    # Rename columns to English
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Parse dates (German format: DD.MM.YYYY)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

    # Parse prices: they are integers in the raw file, no decimals needed
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse volume: remove 'K' suffix and multiply
    df["Volume"] = (
        df["Volume"]
        .astype(str)
        .str.strip()
        .str.replace(" K", "e3", regex=False)  # '5.9 K' -> '5.9e3'
        .str.replace(",", ".", regex=False)
    )
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Sort chronologically (oldest first)
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop rows where OHLC is all NaN
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")

    # Drop bad rows where High < Low (corrupt data points)
    bad = df["High"] < df["Low"]
    if bad.any():
        print(f"  Dropping {bad.sum()} rows where High < Low")
        df = df[~bad].reset_index(drop=True)

    print(f"Daily data loaded: {len(df)} rows")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Missing OHLC: {df[['Open','High','Low','Close']].isna().sum().sum()}")

    return df


# ===================================================================
# 2. INTRADAY (5-MINUTE) DATA
# ===================================================================
def load_5min():
    """
    Load 5min.csv (semicolon-separated, English dates).

    Format quirks:
      - Dates: 'DD. Mon YY' (e.g. '12. Jan 26')
      - Times: HH:MM
      - Prices use dots as decimal separators
      - Data is sorted descending (newest first)
      - Some fields may be empty (Net, %Chg)
    """
    path = os.path.join(RAW_DIR, "5min.csv")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # Rename for clarity
    df.columns = [
        "ExchangeDate", "ExchangeTime", "LocalDate", "LocalTime",
        "Close", "Net", "PctChg", "Open", "Low", "High",
        "Volume", "Bid", "Ask"
    ]

    # Parse datetime: combine date + time
    # Date format: '12. Jan 26' or '31. Dez 25' (German month names!)
    GERMAN_MONTHS = {
        "Jan": "01", "Feb": "02", "Mär": "03", "Mrz": "03", "Mar": "03",
        "Apr": "04", "Mai": "05", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Okt": "10",
        "Oct": "10", "Nov": "11", "Dez": "12", "Dec": "12",
    }

    def parse_5min_datetime(row):
        date_str = row["ExchangeDate"].strip()
        time_str = row["ExchangeTime"].strip()
        parts = date_str.split()
        day = parts[0].rstrip(".")   # '12.' -> '12'
        month_str = parts[1]         # 'Jan' or 'Dez'
        year = parts[2]              # '26'
        year_full = "20" + year if int(year) < 50 else "19" + year
        month_num = GERMAN_MONTHS[month_str]
        return pd.to_datetime(f"{year_full}-{month_num}-{day} {time_str}",
                              format="%Y-%m-%d %H:%M")

    df["Datetime"] = df.apply(parse_5min_datetime, axis=1)
    df["Date"] = df["Datetime"].dt.date

    # Parse numeric columns
    for col in ["Close", "Open", "Low", "High", "Volume", "Bid", "Ask"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Keep only essential columns
    df = df[["Datetime", "Date", "Open", "High", "Low", "Close", "Volume"]]

    print(f"5-min data loaded: {len(df)} rows")
    print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"  Trading days: {df['Date'].nunique()}")

    return df


# ===================================================================
# 3. NINO 3.4 INDEX
# ===================================================================
def load_nino34():
    """
    Load Nino 3.4 raw SST data and convert to anomalies.

    The raw file has absolute sea surface temperatures (°C).
    The standard Nino 3.4 index used in Bouri et al. (2021) and
    Bonato et al. (2023) is the ANOMALY: departure from the
    climatological monthly mean (1991-2020 base period).

    Output: monthly time series with columns [Date, nino34_raw, nino34]
    where nino34 is the anomaly (positive = El Nino, negative = La Nina).
    """
    path = os.path.join(RAW_DIR, "Nino34.csv")
    df = pd.read_csv(path)

    # Reshape from wide (Year, Jan, ..., Dec) to long format
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    records = []
    for _, row in df.iterrows():
        year = int(row["Year"])
        for m_idx, m_name in enumerate(months):
            val = row[m_name]
            if val == -99.99:  # missing value marker
                continue
            records.append({
                "Year": year,
                "Month": m_idx + 1,
                "nino34_raw": val,
            })

    nino = pd.DataFrame(records)
    nino["Date"] = pd.to_datetime(
        nino["Year"].astype(str) + "-" + nino["Month"].astype(str) + "-01"
    )

    # Compute anomalies: subtract 1991-2020 climatological monthly mean
    clim = nino[(nino["Year"] >= 1991) & (nino["Year"] <= 2020)]
    monthly_mean = clim.groupby("Month")["nino34_raw"].mean()
    nino["nino34"] = nino.apply(
        lambda r: r["nino34_raw"] - monthly_mean[r["Month"]], axis=1
    )

    nino = nino[["Date", "Year", "Month", "nino34_raw", "nino34"]]
    nino = nino.sort_values("Date").reset_index(drop=True)

    print(f"Nino 3.4 loaded: {len(nino)} monthly observations")
    print(f"  Date range: {nino['Date'].min().date()} to {nino['Date'].max().date()}")
    print(f"  Anomaly range: [{nino['nino34'].min():.2f}, {nino['nino34'].max():.2f}]")

    return nino


# ===================================================================
# 4. MAIN: Load, clean, save
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading Daily Data")
    print("=" * 60)
    daily = load_daily()
    daily.to_csv(os.path.join(PROC_DIR, "daily_clean.csv"), index=False)
    print(f"  Saved to {PROC_DIR}/daily_clean.csv\n")

    print("=" * 60)
    print("Loading 5-Minute Data")
    print("=" * 60)
    intraday = load_5min()
    intraday.to_csv(os.path.join(PROC_DIR, "5min_clean.csv"), index=False)
    print(f"  Saved to {PROC_DIR}/5min_clean.csv\n")

    print("=" * 60)
    print("Loading Nino 3.4 Index")
    print("=" * 60)
    nino = load_nino34()
    nino.to_csv(os.path.join(PROC_DIR, "nino34_clean.csv"), index=False)
    print(f"  Saved to {PROC_DIR}/nino34_clean.csv\n")

    # Quick preview
    print("=" * 60)
    print("Daily Data Preview")
    print("=" * 60)
    print(daily.head(10).to_string(index=False))
    print(f"\n{'=' * 60}")
    print("5-Min Data Preview")
    print("=" * 60)
    print(intraday.head(10).to_string(index=False))
    print(f"\n{'=' * 60}")
    print("Nino 3.4 Preview (anomalies)")
    print("=" * 60)
    print(nino.tail(12).to_string(index=False))
