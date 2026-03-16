"""
01_yang_zhang_volatility.py
===========================
Compute the Yang-Zhang volatility estimator from daily OHLC data.

The Yang-Zhang estimator combines three components:
  - Overnight variance (open-to-previous-close)
  - Close-to-close (Rogers-Satchell) variance using the daily range
  - Open-to-close variance

Formula (from Methodology chapter, Eq. 3.1):
    sigma_YZ^2 = sigma_O^2 + k * sigma_C^2 + (1-k) * sigma_RS^2

where k = 0.34 / (1.34 + (n+1)/(n-1))

Also computes alternative estimators for comparison:
  - Close-to-close (simple)
  - Parkinson (range-based)
  - Garman-Klass

Output: daily_with_volatility.csv in 00_Data/processed/
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "00_Data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "02_Output")


def load_daily():
    """Load the cleaned daily data."""
    path = os.path.join(PROC_DIR, "daily_clean.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===================================================================
# Volatility Estimators (all work on log-prices)
# ===================================================================

def compute_log_returns(df):
    """Add log-price columns and log-returns to the dataframe."""
    df = df.copy()

    # Log prices
    df["log_open"] = np.log(df["Open"])
    df["log_high"] = np.log(df["High"])
    df["log_low"] = np.log(df["Low"])
    df["log_close"] = np.log(df["Close"])

    # Normalized log returns (used by Yang-Zhang components)
    # o_i = log(Open_t / Close_{t-1})  -> overnight return
    # c_i = log(Close_t / Open_t)      -> open-to-close return
    # u_i = log(High_t / Open_t)
    # d_i = log(Low_t / Open_t)
    df["overnight_ret"] = df["log_open"] - df["log_close"].shift(1)
    df["oc_ret"] = df["log_close"] - df["log_open"]
    df["ho_ret"] = df["log_high"] - df["log_open"]
    df["lo_ret"] = df["log_low"] - df["log_open"]

    # Simple close-to-close log return
    df["log_ret"] = df["log_close"] - df["log_close"].shift(1)

    return df


def yang_zhang_volatility(df, window=22):
    """
    Compute Yang-Zhang volatility over a rolling window.

    Parameters
    ----------
    df : DataFrame with overnight_ret, oc_ret, ho_ret, lo_ret columns
    window : int, rolling window size (default 22 = ~1 month)

    Returns
    -------
    Series of annualized Yang-Zhang volatility
    """
    n = window

    # Optimal weighting factor (minimizes variance of estimator)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Component 1: Overnight variance (open vs previous close)
    sigma2_overnight = df["overnight_ret"].rolling(n).var()

    # Component 2: Open-to-close variance
    sigma2_oc = df["oc_ret"].rolling(n).var()

    # Component 3: Rogers-Satchell variance (uses full range, drift-independent)
    rs_daily = (
        df["ho_ret"] * (df["ho_ret"] - df["oc_ret"])
        + df["lo_ret"] * (df["lo_ret"] - df["oc_ret"])
    )
    sigma2_rs = rs_daily.rolling(n).mean()

    # Yang-Zhang combination
    sigma2_yz = sigma2_overnight + k * sigma2_oc + (1 - k) * sigma2_rs

    # Return as daily volatility (standard deviation), not variance
    # Annualize: multiply by sqrt(252)
    sigma_yz_daily = np.sqrt(sigma2_yz)
    sigma_yz_annual = sigma_yz_daily * np.sqrt(252)

    return sigma_yz_daily, sigma_yz_annual


def close_to_close_volatility(df, window=22):
    """Simple close-to-close volatility (standard deviation of log returns)."""
    sigma_cc = df["log_ret"].rolling(window).std()
    return sigma_cc, sigma_cc * np.sqrt(252)


def parkinson_volatility(df, window=22):
    """
    Parkinson (1980) range-based estimator.
    Uses High-Low range, ~5x more efficient than close-to-close.
    """
    hl_sq = (df["log_high"] - df["log_low"]) ** 2
    sigma2_park = hl_sq.rolling(window).mean() / (4 * np.log(2))
    sigma_park = np.sqrt(sigma2_park)
    return sigma_park, sigma_park * np.sqrt(252)


def garman_klass_volatility(df, window=22):
    """
    Garman-Klass (1980) estimator.
    Uses OHLC, ~7x more efficient than close-to-close.
    """
    hl_sq = (df["log_high"] - df["log_low"]) ** 2
    co_sq = (df["log_close"] - df["log_open"]) ** 2
    gk_daily = 0.5 * hl_sq - (2 * np.log(2) - 1) * co_sq
    sigma2_gk = gk_daily.rolling(window).mean()
    sigma_gk = np.sqrt(sigma2_gk.clip(lower=0))  # clip to avoid sqrt of negative
    return sigma_gk, sigma_gk * np.sqrt(252)


# ===================================================================
# Single-day Yang-Zhang proxy (for HAR model)
# ===================================================================

def yang_zhang_daily_proxy(df):
    """
    Compute a single-day volatility proxy using the Yang-Zhang components.

    For the HAR model we need a DAILY volatility value sigma_t, not a
    rolling-window estimate. We use the square root of the single-day
    Yang-Zhang variance components:

        sigma_t^2 = overnight_ret_t^2 + k * oc_ret_t^2 + (1-k) * rs_t

    This is noisier than the rolling window version but gives us a
    daily time series suitable for HAR aggregation.
    """
    # Use a "window" concept but actually just single-day components
    # For single day, k simplification: use k ≈ 0.34 / 1.34 ≈ 0.254
    # (the (n+1)/(n-1) term -> 1 as n->inf, here we use large-sample k)
    k = 0.34 / 1.34

    # Rogers-Satchell for single day
    rs_t = (
        df["ho_ret"] * (df["ho_ret"] - df["oc_ret"])
        + df["lo_ret"] * (df["lo_ret"] - df["oc_ret"])
    )

    # Single-day Yang-Zhang variance proxy
    sigma2_t = df["overnight_ret"] ** 2 + k * df["oc_ret"] ** 2 + (1 - k) * rs_t

    # Floor at zero (can happen with weird data)
    sigma2_t = sigma2_t.clip(lower=0)
    sigma_t = np.sqrt(sigma2_t)

    return sigma_t


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("Loading cleaned daily data...")
    df = load_daily()
    df = compute_log_returns(df)

    # --- Rolling window volatilities (for plots / descriptive analysis) ---
    print("Computing rolling volatilities (window=22)...")
    df["vol_yz_daily"], df["vol_yz_annual"] = yang_zhang_volatility(df, window=22)
    df["vol_cc_daily"], df["vol_cc_annual"] = close_to_close_volatility(df, window=22)
    df["vol_park_daily"], df["vol_park_annual"] = parkinson_volatility(df, window=22)
    df["vol_gk_daily"], df["vol_gk_annual"] = garman_klass_volatility(df, window=22)

    # --- Daily proxy for HAR model ---
    print("Computing daily Yang-Zhang proxy for HAR model...")
    df["sigma_yz"] = yang_zhang_daily_proxy(df)

    # Save
    out_path = os.path.join(PROC_DIR, "daily_with_volatility.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # Quick summary
    print("\nVolatility summary (annualized, rolling 22-day):")
    vol_cols = ["vol_yz_annual", "vol_cc_annual", "vol_park_annual", "vol_gk_annual"]
    print(df[vol_cols].describe().round(4).to_string())
