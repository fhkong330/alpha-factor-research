from typing import Dict, List, Optional

import pandas as pd

from config import COMBO_METHOD, VERBOSE


def extract_selected_alpha_panels(
    alpha_panel: pd.DataFrame,
    selected_alpha_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Extract date x symbol signal panel for each selected alpha.
    """
    if not isinstance(alpha_panel.columns, pd.MultiIndex):
        raise ValueError("alpha_panel must have MultiIndex columns.")

    selected_panels = {}

    for alpha_name in selected_alpha_names:
        if alpha_name not in alpha_panel.columns.get_level_values(1):
            raise ValueError(f"{alpha_name} not found in alpha_panel.")
        selected_panels[alpha_name] = (
            alpha_panel.xs(alpha_name, level=1, axis=1)
            .sort_index(axis=1)
        )

    return selected_panels


def build_equal_weight_combo_signal(
    alpha_panel: pd.DataFrame,
    selected_alpha_names: List[str],
    standardization_method: str = "cross_sectional_rank",
) -> pd.DataFrame:
    """
    Build equal-weight composite signal from selected alpha panels.

    Returns
    -------
    pd.DataFrame
        date x symbol combo signal panel
    """
    selected_panels = extract_selected_alpha_panels(
        alpha_panel=alpha_panel,
        selected_alpha_names=selected_alpha_names,
    )

    standardized_panels = []
    for alpha_name, panel in selected_panels.items():
        standardized_panel = standardize_alpha_panel_cross_sectionally(
            panel=panel,
            method=standardization_method,
        )
        standardized_panels.append(standardized_panel)

    combo_signal = sum(standardized_panels) / len(standardized_panels)
    combo_signal = combo_signal.sort_index().sort_index(axis=1)

    if VERBOSE:
        print(
            f"[build_combo_signal] Built equal-weight combo signal from "
            f"{len(selected_alpha_names)} alphas using standardization={standardization_method}. "
            f"Shape={combo_signal.shape}"
        )

    return combo_signal


def build_ic_weighted_combo_signal(
    alpha_panel: pd.DataFrame,
    selected_alpha_names: List[str],
    ic_summary_df: pd.DataFrame,
    weight_col: str = "mean_ic",
    standardization_method: str = "cross_sectional_rank",
) -> pd.DataFrame:
    """
    Optional extension: build IC-weighted combo signal.
    """
    selected_panels = extract_selected_alpha_panels(
        alpha_panel=alpha_panel,
        selected_alpha_names=selected_alpha_names,
    )

    weights = ic_summary_df.loc[selected_alpha_names, weight_col].copy()
    weights = weights.fillna(0.0)

    if weights.abs().sum() == 0:
        raise ValueError("Selected alpha weights sum to zero.")

    normalized_weights = weights / weights.abs().sum()

    combo_signal = None
    for alpha_name, weight in normalized_weights.items():
        panel = standardize_alpha_panel_cross_sectionally(
            panel=selected_panels[alpha_name],
            method=standardization_method,
        )

        if combo_signal is None:
            combo_signal = panel * weight
        else:
            combo_signal = combo_signal + panel * weight

    combo_signal = combo_signal.sort_index().sort_index(axis=1)

    if VERBOSE:
        print(
            f"[build_combo_signal] Built IC-weighted combo signal using {weight_col} "
            f"and standardization={standardization_method}. Shape={combo_signal.shape}"
        )

    return combo_signal


def build_combo_signal(
    alpha_panel: pd.DataFrame,
    selected_alpha_names: List[str],
    method: str = COMBO_METHOD,
    ic_summary_df: Optional[pd.DataFrame] = None,
    standardization_method: str = "cross_sectional_rank",
) -> pd.DataFrame:
    """
    General combo signal interface.
    """
    if len(selected_alpha_names) == 0:
        raise ValueError("selected_alpha_names is empty.")

    if method == "equal_weight":
        return build_equal_weight_combo_signal(
            alpha_panel=alpha_panel,
            selected_alpha_names=selected_alpha_names,
            standardization_method=standardization_method,
        )

    if method == "ic_weighted":
        if ic_summary_df is None:
            raise ValueError("ic_summary_df is required for ic_weighted combo.")
        return build_ic_weighted_combo_signal(
            alpha_panel=alpha_panel,
            selected_alpha_names=selected_alpha_names,
            ic_summary_df=ic_summary_df,
            standardization_method=standardization_method,
        )

    raise ValueError(f"Unsupported combo method: {method}")

def standardize_alpha_panel_cross_sectionally(
    panel: pd.DataFrame,
    method: str = "cross_sectional_rank",
) -> pd.DataFrame:
    """
    Standardize a date x symbol alpha panel cross-sectionally.

    Parameters
    ----------
    panel : pd.DataFrame
        date x symbol alpha panel
    method : str
        - "cross_sectional_rank": convert each row to percentile ranks
        - "none": no standardization

    Returns
    -------
    pd.DataFrame
        standardized panel
    """
    if method == "none":
        return panel

    if method == "cross_sectional_rank":
        return panel.rank(axis=1, pct=True)

    raise ValueError(f"Unsupported standardization method: {method}")