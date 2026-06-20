from typing import Dict

import numpy as np
import pandas as pd

from config import REQUIRED_PRICE_COLUMNS, VERBOSE


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase and strip spaces.
    """
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def ensure_required_columns(df: pd.DataFrame, required_columns=None) -> None:
    """
    Raise an error if required price columns are missing.
    """
    if required_columns is None:
        required_columns = REQUIRED_PRICE_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sort_and_clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by datetime index, remove duplicated dates, and drop rows
    where all values are missing.
    """
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.dropna(how="all")
    return out


def compute_daily_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an approximate daily VWAP proxy.

    Since Yahoo Finance daily download does not provide intraday VWAP,
    we use the common OHLC average proxy:
        (open + high + low + close) / 4

    This is more appropriate than cumulative price-volume averaging for
    daily-bar data because each row should represent that day's proxy VWAP.
    """
    out = df.copy()
    out["vwap"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    return out


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns are numeric where possible.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where core fields are missing or invalid.
    """
    out = df.copy()

    core_cols = ["open", "high", "low", "close", "volume"]
    out = out.dropna(subset=core_cols)

    # Volume should be non-negative; price columns should be positive
    out = out[out["volume"] >= 0]
    for col in ["open", "high", "low", "close"]:
        out = out[out[col] > 0]

    return out


def preprocess_single_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one symbol.
    """
    out = standardize_columns(df)
    out = sort_and_clean_index(out)
    out = coerce_numeric(out)

    ensure_required_columns(out)

    out = drop_invalid_rows(out)
    out = compute_daily_vwap(out)

    return out


def preprocess_symbol_data(raw_symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess a dict of per-symbol OHLCV DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned symbol data dictionary.
    """
    processed = {}

    for symbol, df in raw_symbol_data.items():
        try:
            clean_df = preprocess_single_symbol(df)
            if not clean_df.empty:
                processed[symbol] = clean_df
        except Exception as exc:
            if VERBOSE:
                print(f"[preprocess] Skipping {symbol}: {exc}")

    if VERBOSE:
        print(f"[preprocess] Successfully processed {len(processed)} symbols.")

    return processed