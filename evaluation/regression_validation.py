from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import VERBOSE
from pipeline.compute_forward_returns import stack_signal_and_forward_returns


def run_predictive_regression(
    signal_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    signal_name: str = "signal",
    return_name: str = "next_return",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run pooled OLS predictive regression:
        next_return ~ const + signal
    """
    regression_df = stack_signal_and_forward_returns(
        signal_df=signal_df,
        forward_returns_df=forward_returns_df,
        signal_name=signal_name,
        return_name=return_name,
    )

    if regression_df.empty:
        raise ValueError("Regression dataset is empty after stacking and dropping NaNs.")

    X = sm.add_constant(regression_df[signal_name])
    y = regression_df[return_name]

    model = sm.OLS(y, X).fit()

    if VERBOSE:
        print(
            f"[regression_validation] Regression complete. "
            f"n_obs={int(model.nobs)}, "
            f"coef={model.params.get(signal_name, np.nan):.6f}, "
            f"t={model.tvalues.get(signal_name, np.nan):.6f}, "
            f"R2={model.rsquared:.6f}"
        )

    return model


def summarize_regression_result(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    signal_name: str = "signal",
) -> Dict[str, float]:
    """
    Extract key regression metrics into a compact dictionary.
    """
    return {
        "n_obs": float(model.nobs),
        "intercept": float(model.params.get("const", np.nan)),
        "signal_coef": float(model.params.get(signal_name, np.nan)),
        "signal_tstat": float(model.tvalues.get(signal_name, np.nan)),
        "signal_pvalue": float(model.pvalues.get(signal_name, np.nan)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
    }


def regression_summary_text(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> str:
    """
    Return the full statsmodels summary as text.
    """
    return model.summary().as_text()