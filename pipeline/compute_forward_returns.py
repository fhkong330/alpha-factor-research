from typing import Dict

import pandas as pd

from config import FORWARD_RETURN_HORIZON, SIGNAL_LAG_DAYS, VERBOSE


def compute_forward_return_series(
    close_series: pd.Series,
    horizon: int = FORWARD_RETURN_HORIZON,
    signal_lag_days: int = SIGNAL_LAG_DAYS,
) -> pd.Series:
    """
    Compute forward average return over the next `horizon` trading days.

    Logic:
    - daily_return_t = close_t / close_{t-1} - 1
    - future label at date t is the average return over the next horizon days
    - shift is applied to align today's signal with future realized returns

    Parameters
    ----------
    close_series : pd.Series
        Daily close price series.
    horizon : int
        Number of future days in the target return window.
    signal_lag_days : int
        Lag applied to avoid look-ahead.

    Returns
    -------
    pd.Series
        Forward return label aligned to the signal date index.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if signal_lag_days < 0:
        raise ValueError("signal_lag_days must be non-negative")

    daily_returns = close_series.pct_change()

    forward_avg_return = (
        daily_returns.shift(-1)
        .rolling(window=horizon)
        .mean()
    )

    aligned_forward_return = forward_avg_return.shift(signal_lag_days + horizon)

    return aligned_forward_return


def extract_close_panel_from_symbol_data(
    symbol_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Extract close price panel from preprocessed symbol data.

    Returns
    -------
    pd.DataFrame
        date x symbol close-price panel
    """
    close_dict = {}

    for symbol, df in symbol_data.items():
        if "close" in df.columns:
            close_dict[symbol] = df["close"]

    if not close_dict:
        raise ValueError("No close series found in symbol_data.")

    close_panel = pd.DataFrame(close_dict).sort_index()

    if VERBOSE:
        print(
            f"[compute_forward_returns] Close panel shape: {close_panel.shape}"
        )

    return close_panel


def compute_forward_returns_from_symbol_data(
    symbol_data: Dict[str, pd.DataFrame],
    horizon: int = FORWARD_RETURN_HORIZON,
    signal_lag_days: int = SIGNAL_LAG_DAYS,
) -> pd.DataFrame:
    """
    Compute forward returns directly from preprocessed symbol data.

    Returns
    -------
    pd.DataFrame
        date x symbol forward return panel
    """
    close_panel = extract_close_panel_from_symbol_data(symbol_data)

    returns_df = close_panel.apply(
        lambda col: compute_forward_return_series(
            close_series=col,
            horizon=horizon,
            signal_lag_days=signal_lag_days,
        )
    )

    if VERBOSE:
        print(
            f"[compute_forward_returns] Forward returns shape: {returns_df.shape}"
        )

    return returns_df


def extract_proxy_close_from_alpha_panel(
    alpha_panel: pd.DataFrame,
    proxy_alpha_name: str = "alpha_101",
) -> pd.DataFrame:
    """
    Extract a proxy series from alpha panel.

    NOTE:
    This is mainly kept for backward compatibility with the user's earlier script,
    where alpha_101 was incorrectly reused as a proxy for price-based series.
    For the upgraded pipeline, forward returns should preferably be computed from
    the original close prices in symbol_data, not from alpha outputs.

    Parameters
    ----------
    alpha_panel : pd.DataFrame
        MultiIndex-column alpha panel
    proxy_alpha_name : str
        Alpha column name to extract across symbols

    Returns
    -------
    pd.DataFrame
        date x symbol panel
    """
    if not isinstance(alpha_panel.columns, pd.MultiIndex):
        raise ValueError("alpha_panel must have MultiIndex columns.")

    level1 = alpha_panel.columns.get_level_values(1)
    if proxy_alpha_name not in level1:
        raise ValueError(f"{proxy_alpha_name} not found in alpha panel.")

    proxy_df = alpha_panel.xs(proxy_alpha_name, level=1, axis=1).sort_index(axis=1)

    if VERBOSE:
        print(
            f"[compute_forward_returns] Extracted proxy panel "
            f"from alpha '{proxy_alpha_name}', shape={proxy_df.shape}"
        )

    return proxy_df


def stack_signal_and_forward_returns(
    signal_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    signal_name: str = "signal",
    return_name: str = "next_return",
) -> pd.DataFrame:
    """
    Stack cross-sectional signal panel and forward return panel into a long-format DataFrame.

    Parameters
    ----------
    signal_df : pd.DataFrame
        date x symbol signal panel
    forward_returns_df : pd.DataFrame
        date x symbol future-return panel

    Returns
    -------
    pd.DataFrame
        MultiIndex index: (date, symbol)
        Columns: [signal_name, return_name]
    """
    signal_long = signal_df.stack().rename(signal_name)
    return_long = forward_returns_df.stack().rename(return_name)

    combo_df = pd.concat([signal_long, return_long], axis=1).dropna()

    if VERBOSE:
        print(
            f"[compute_forward_returns] Stacked signal/return dataset shape: {combo_df.shape}"
        )

    return combo_df

def compute_next_day_return_panel_from_symbol_data(
    symbol_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a date x symbol panel of next-day realized returns.

    Logic:
    - daily_return_t = close_t / close_{t-1} - 1
    - next_day_return_t = daily_return_{t+1}
    - this is used for more realistic daily-rebalanced backtesting

    Returns
    -------
    pd.DataFrame
        date x symbol next-day realized return panel
    """
    close_panel = extract_close_panel_from_symbol_data(symbol_data)

    daily_return_panel = close_panel.pct_change(fill_method=None)
    next_day_return_panel = daily_return_panel.shift(-1)

    if VERBOSE:
        print(
            f"[compute_forward_returns] Next-day return panel shape: {next_day_return_panel.shape}"
        )

    return next_day_return_panel