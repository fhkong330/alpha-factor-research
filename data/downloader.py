from typing import Dict, List, Optional
import os
import time

import pandas as pd
import requests

from config import START_DATE, END_DATE, VERBOSE

try:
    from config import POLYGON_API_KEY
except ImportError:
    POLYGON_API_KEY = None


POLYGON_BASE_URL = "https://api.polygon.io"
REQUEST_SLEEP_SECONDS = 0.15
REQUEST_TIMEOUT = 20


def _get_polygon_api_key() -> str:
    """
    Read Polygon API key from environment variable first,
    then fall back to config.py.
    """
    api_key = os.getenv("POLYGON_API_KEY", POLYGON_API_KEY)

    if not api_key:
        raise ValueError(
            "Polygon API key not found. Please set POLYGON_API_KEY in config.py "
            "or as an environment variable."
        )

    return api_key


def _build_polygon_aggs_url(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Polygon daily aggregates endpoint:
    /v2/aggs/ticker/{stocksTicker}/range/1/day/{from}/{to}
    """
    return (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start_date}/{end_date}"
    )


def _fetch_polygon_daily_aggs(
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted: bool = True,
    sort: str = "asc",
    limit: int = 50000,
) -> pd.DataFrame:
    """
    Fetch one symbol's daily OHLCV aggregates from Polygon.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex
        Columns = Open, High, Low, Close, Volume
    """
    api_key = _get_polygon_api_key()
    url = _build_polygon_aggs_url(symbol, start_date, end_date)

    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": sort,
        "limit": limit,
        "apiKey": api_key,
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        raise ValueError(
            f"Polygon request failed for {symbol}. "
            f"Status={response.status_code}, Response={response.text[:300]}"
        )

    payload = response.json()

    status = payload.get("status")
    results = payload.get("results", [])

    if status != "OK" or not results:
        if VERBOSE:
            print(f"[downloader] No aggregate results returned for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Map Polygon fields to a more standard OHLCV schema
    rename_map = {
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
        "vw": "VWAP",
        "n": "Transactions",
        "t": "Timestamp",
    }
    df = df.rename(columns=rename_map)

    # Polygon timestamps are Unix milliseconds
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.normalize()
    df = df.set_index("Date").sort_index()

    # Keep the columns our downstream pipeline expects
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing_keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_keep_cols].copy()

    # Make sure numeric
    for col in existing_keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_price_data(
    symbols: List[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Download OHLCV data for a list of symbols using Polygon daily aggregates.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns in ticker-first layout:
        level 0 = ticker
        level 1 = field (Open, High, Low, Close, Volume)
    """
    if not symbols:
        raise ValueError("No symbols provided for download.")

    if VERBOSE:
        print(
            f"[downloader] Downloading Polygon daily aggregates for {len(symbols)} symbols "
            f"from {start_date} to {end_date}..."
        )

    symbol_frames: Dict[str, pd.DataFrame] = {}

    for i, symbol in enumerate(symbols, start=1):
        if VERBOSE:
            print(f"[downloader] ({i}/{len(symbols)}) Fetching {symbol}...")

        try:
            df = _fetch_polygon_daily_aggs(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjusted=True,
                sort="asc",
                limit=50000,
            )

            if not df.empty:
                symbol_frames[symbol] = df
            else:
                if VERBOSE:
                    print(f"[downloader] Empty data for {symbol}, skipping.")

        except Exception as exc:
            if VERBOSE:
                print(f"[downloader] Failed on {symbol}: {exc}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    if not symbol_frames:
        raise ValueError("Downloaded data is empty.")

    combined_data = pd.concat(symbol_frames, axis=1).sort_index(axis=1)

    if VERBOSE:
        print(f"[downloader] Download completed. Raw shape: {combined_data.shape}")

    return combined_data


def _is_field_first_layout(data: pd.DataFrame, symbols: List[str]) -> bool:
    """
    Detect whether downloaded data is arranged as:
    level 0 = field, level 1 = ticker
    """
    if not isinstance(data.columns, pd.MultiIndex):
        return False

    level0 = set(map(str, data.columns.get_level_values(0)))
    level1 = set(map(str, data.columns.get_level_values(1)))
    symbol_set = set(symbols)

    return len(level0.intersection(symbol_set)) == 0 and len(level1.intersection(symbol_set)) > 0


def _is_ticker_first_layout(data: pd.DataFrame, symbols: List[str]) -> bool:
    """
    Detect whether downloaded data is arranged as:
    level 0 = ticker, level 1 = field
    """
    if not isinstance(data.columns, pd.MultiIndex):
        return False

    level0 = set(map(str, data.columns.get_level_values(0)))
    symbol_set = set(symbols)

    return len(level0.intersection(symbol_set)) > 0


def split_data_by_symbol(data: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Split downloaded price data into a dict of single-symbol DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "AAPL": DataFrame with columns like Open/High/Low/Close/Volume,
            ...
        }
    """
    if data.empty:
        raise ValueError("Input data is empty.")

    symbol_data = {}

    if isinstance(data.columns, pd.MultiIndex):
        if _is_ticker_first_layout(data, symbols):
            for symbol in symbols:
                if symbol in data.columns.get_level_values(0):
                    df = data[symbol].copy()
                    if not df.empty:
                        symbol_data[symbol] = df

        elif _is_field_first_layout(data, symbols):
            swapped = data.swaplevel(axis=1).sort_index(axis=1)
            for symbol in symbols:
                if symbol in swapped.columns.get_level_values(0):
                    df = swapped[symbol].copy()
                    if not df.empty:
                        symbol_data[symbol] = df
        else:
            raise ValueError("Unable to detect MultiIndex column layout from downloaded data.")

    else:
        if len(symbols) != 1:
            raise ValueError("Single-level columns received for multiple symbols.")
        symbol_data[symbols[0]] = data.copy()

    if VERBOSE:
        print(f"[downloader] Split into {len(symbol_data)} per-symbol DataFrames.")

    return symbol_data