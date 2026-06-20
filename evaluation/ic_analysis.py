from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import MIN_CROSS_SECTION_SIZE, VERBOSE


def compute_daily_ic(
    signal_panel: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    min_cross_section_size: int = MIN_CROSS_SECTION_SIZE,
) -> pd.Series:
    """
    Compute daily cross-sectional Spearman IC between one alpha signal panel
    and forward returns.

    Parameters
    ----------
    signal_panel : pd.DataFrame
        date x symbol panel for one alpha
    forward_returns_df : pd.DataFrame
        date x symbol forward return panel
    min_cross_section_size : int
        Minimum number of valid stocks required on a date to compute IC

    Returns
    -------
    pd.Series
        Daily IC time series indexed by date
    """
    common_dates = signal_panel.index.intersection(forward_returns_df.index)
    ic_values = {}

    for date in common_dates:
        try:
            scores = signal_panel.loc[date]
            returns = forward_returns_df.loc[date]

            common_symbols = scores.dropna().index.intersection(returns.dropna().index)

            if len(common_symbols) < min_cross_section_size:
                ic_values[date] = np.nan
                continue

            ic, _ = spearmanr(
                scores.loc[common_symbols],
                returns.loc[common_symbols],
            )
            ic_values[date] = ic
        except Exception:
            ic_values[date] = np.nan

    ic_series = pd.Series(ic_values).sort_index()
    return ic_series


def summarize_ic_series(ic_series: pd.Series) -> Dict[str, float]:
    """
    Summarize a daily IC series into research metrics.
    """
    clean_ic = ic_series.dropna()

    if clean_ic.empty:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "ic_ir": np.nan,
            "positive_ic_ratio": np.nan,
            "n_obs": 0,
        }

    mean_ic = clean_ic.mean()
    std_ic = clean_ic.std(ddof=0)
    ic_ir = mean_ic / (std_ic + 1e-12)
    positive_ic_ratio = (clean_ic > 0).mean()

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "positive_ic_ratio": positive_ic_ratio,
        "n_obs": int(clean_ic.shape[0]),
    }


def evaluate_alpha_panel_ic(
    alpha_panel: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    min_cross_section_size: int = MIN_CROSS_SECTION_SIZE,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Evaluate all alpha signals in a multi-index alpha panel.

    Parameters
    ----------
    alpha_panel : pd.DataFrame
        MultiIndex columns:
            level 0 = symbol
            level 1 = alpha name
    forward_returns_df : pd.DataFrame
        date x symbol forward return panel

    Returns
    -------
    ic_summary_df : pd.DataFrame
        Summary metrics for each alpha
    ic_series_dict : dict[str, pd.Series]
        Daily IC time series for each alpha
    """
    if not isinstance(alpha_panel.columns, pd.MultiIndex):
        raise ValueError("alpha_panel must have MultiIndex columns.")

    alpha_names: List[str] = sorted(alpha_panel.columns.get_level_values(1).unique())

    summary_records = {}
    ic_series_dict = {}

    for alpha_name in alpha_names:
        try:
            signal_panel = alpha_panel.xs(alpha_name, level=1, axis=1).sort_index(axis=1)
            signal_panel = signal_panel.reindex(columns=forward_returns_df.columns)

            ic_series = compute_daily_ic(
                signal_panel=signal_panel,
                forward_returns_df=forward_returns_df,
                min_cross_section_size=min_cross_section_size,
            )
            summary_records[alpha_name] = summarize_ic_series(ic_series)
            ic_series_dict[alpha_name] = ic_series

            if VERBOSE:
                stats = summary_records[alpha_name]
                print(
                    f"[ic_analysis] {alpha_name}: "
                    f"mean_ic={stats['mean_ic']:.6f}, "
                    f"ic_ir={stats['ic_ir']:.6f}, "
                    f"n_obs={stats['n_obs']}"
                )
        except Exception as exc:
            if VERBOSE:
                print(f"[ic_analysis] Failed on {alpha_name}: {exc}")
            summary_records[alpha_name] = {
                "mean_ic": np.nan,
                "std_ic": np.nan,
                "ic_ir": np.nan,
                "positive_ic_ratio": np.nan,
                "n_obs": 0,
            }
            ic_series_dict[alpha_name] = pd.Series(dtype=float)

    ic_summary_df = pd.DataFrame(summary_records).T
    ic_summary_df = ic_summary_df.sort_values(by="mean_ic", ascending=False)

    return ic_summary_df, ic_series_dict