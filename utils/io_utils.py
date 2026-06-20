from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Any) -> Path:
    """
    Ensure the parent directory of a file path exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Any, index: bool = True) -> None:
    """
    Save a DataFrame to CSV.
    """
    path = ensure_parent_dir(path)
    df.to_csv(path, index=index)


def load_dataframe(
    path: Any,
    index_col=0,
    parse_dates=True,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame.
    """
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates, **kwargs)


def save_text(text: str, path: Any) -> None:
    """
    Save plain text to file.
    """
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_text(text: str, path: Any) -> None:
    """
    Append plain text to file.
    """
    path = ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def file_exists(path: Any) -> bool:
    """
    Check whether a file exists.
    """
    return Path(path).exists()