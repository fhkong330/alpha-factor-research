from typing import Dict

import pandas as pd

from alphas.alpha_library import calculate_all_alphas
from alphas.alpha_registry import ALPHA_REGISTRY
from config import ALPHA_PANEL_PATH, VERBOSE
from utils.checks import check_alpha_output, validate_single_symbol_input
from utils.io_utils import save_dataframe


def compute_alpha_panel_for_symbol(
    symbol: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all registered alphas for a single symbol.

    Parameters
    ----------
    symbol : str
        Stock ticker.
    df : pd.DataFrame
        Preprocessed OHLCV DataFrame for one symbol.

    Returns
    -------
    pd.DataFrame
        Alpha DataFrame indexed by date with columns = alpha names.
    """
    validate_single_symbol_input(df)

    alpha_df = calculate_all_alphas(df=df, alpha_registry=ALPHA_REGISTRY)
    check_alpha_output(alpha_df)

    if VERBOSE:
        print(
            f"[build_alpha_panel] {symbol}: computed {alpha_df.shape[1]} alphas "
            f"over {alpha_df.shape[0]} dates."
        )

    return alpha_df


def build_alpha_panel(
    symbol_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a multi-symbol alpha panel.

    Parameters
    ----------
    symbol_data : dict[str, pd.DataFrame]
        Dictionary of preprocessed OHLCV data by symbol.

    Returns
    -------
    pd.DataFrame
        MultiIndex-column DataFrame:
            level 0 = ticker
            level 1 = alpha name
    """
    all_alphas = {}

    for symbol, df in symbol_data.items():
        try:
            alpha_df = compute_alpha_panel_for_symbol(symbol=symbol, df=df)
            all_alphas[symbol] = alpha_df
        except Exception as exc:
            if VERBOSE:
                print(f"[build_alpha_panel] Skipping {symbol}: {exc}")

    if not all_alphas:
        raise ValueError("No alpha panels were successfully computed.")

    panel_df = pd.concat(all_alphas, axis=1)
    panel_df = panel_df.sort_index(axis=1)

    if VERBOSE:
        print(
            f"[build_alpha_panel] Final alpha panel shape: {panel_df.shape} "
            f"(rows={panel_df.shape[0]}, columns={panel_df.shape[1]})."
        )

    return panel_df


def build_and_save_alpha_panel(
    symbol_data: Dict[str, pd.DataFrame],
    output_path=ALPHA_PANEL_PATH,
) -> pd.DataFrame:
    """
    Build the alpha panel and save it to disk.
    """
    panel_df = build_alpha_panel(symbol_data=symbol_data)
    save_dataframe(panel_df, output_path)

    if VERBOSE:
        print(f"[build_alpha_panel] Alpha panel saved to: {output_path}")

    return panel_df


def load_alpha_panel(path=ALPHA_PANEL_PATH) -> pd.DataFrame:
    """
    Load a previously saved alpha panel from CSV.
    """
    panel_df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    if VERBOSE:
        print(f"[build_alpha_panel] Loaded alpha panel from: {path}")
        print(f"[build_alpha_panel] Loaded shape: {panel_df.shape}")

    return panel_df