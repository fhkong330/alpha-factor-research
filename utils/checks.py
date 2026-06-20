from typing import Dict

import pandas as pd

from config import MIN_HISTORY_ROWS, REQUIRED_PRICE_COLUMNS


def check_required_columns(df: pd.DataFrame, required_columns=None) -> None:
    """
    Validate that the input DataFrame contains all required columns.
    """
    if required_columns is None:
        required_columns = REQUIRED_PRICE_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_min_history(df: pd.DataFrame, min_rows: int = MIN_HISTORY_ROWS) -> None:
    """
    Validate that a symbol has enough history to support rolling calculations.
    """
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient history: got {len(df)} rows, need at least {min_rows}."
        )


def check_no_empty_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame is not empty.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")


def validate_single_symbol_input(df: pd.DataFrame) -> None:
    """
    Validate a single-symbol OHLCV DataFrame before alpha computation.
    """
    check_no_empty_dataframe(df)
    check_required_columns(df)
    check_min_history(df)


def validate_symbol_data_dict(symbol_data: Dict[str, pd.DataFrame]) -> None:
    """
    Validate the downloaded / processed symbol dictionary.
    """
    if not symbol_data:
        raise ValueError("Symbol data dictionary is empty.")

    valid_count = 0
    for _, df in symbol_data.items():
        try:
            validate_single_symbol_input(df)
            valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        raise ValueError("No valid symbols passed validation checks.")


def check_alpha_output(alpha_df: pd.DataFrame) -> None:
    """
    Validate alpha output panel for a single symbol.
    """
    check_no_empty_dataframe(alpha_df)

    if alpha_df.isna().all().all():
        raise ValueError("All alpha outputs are NaN.")

    if alpha_df.shape[1] == 0:
        raise ValueError("Alpha output has no columns.")