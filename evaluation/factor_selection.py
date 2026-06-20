from typing import List

import pandas as pd

from config import TOP_N_ALPHAS, VERBOSE


def select_top_alphas(
    ic_summary_df: pd.DataFrame,
    top_n: int = TOP_N_ALPHAS,
    ranking_metric: str = "mean_ic",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank and select top alphas based on a chosen metric.

    Parameters
    ----------
    ic_summary_df : pd.DataFrame
        Alpha IC summary table
    top_n : int
        Number of factors to select
    ranking_metric : str
        Column used for ranking, e.g. mean_ic or ic_ir
    ascending : bool
        False means larger metric ranks higher

    Returns
    -------
    pd.DataFrame
        Top alpha summary table
    """
    if ranking_metric not in ic_summary_df.columns:
        raise ValueError(f"{ranking_metric} not found in ic_summary_df.")

    selected_df = (
        ic_summary_df
        .dropna(subset=[ranking_metric])
        .sort_values(by=ranking_metric, ascending=ascending)
        .head(top_n)
        .copy()
    )

    if VERBOSE:
        print(
            f"[factor_selection] Selected top {selected_df.shape[0]} alphas "
            f"by {ranking_metric}: {selected_df.index.tolist()}"
        )

    return selected_df


def get_top_alpha_names(
    selected_alpha_df: pd.DataFrame,
) -> List[str]:
    """
    Extract selected alpha names from the selection result DataFrame.
    """
    return selected_alpha_df.index.tolist()