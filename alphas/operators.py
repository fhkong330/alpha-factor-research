import numpy as np
import pandas as pd


def safe_rank(series: pd.Series) -> pd.Series:
    """
    Cross-sectional percentile rank for a Series.
    Keeps NaNs unchanged.
    """
    return series.rank(pct=True)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Time-series rank of the latest observation within a rolling window.
    Output is scaled to (0, 1].
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x),
        raw=False,
    )


def decay_linear(series: pd.Series, period: int) -> pd.Series:
    """
    Linearly decaying weighted moving average.
    More recent observations receive larger weights.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")

    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()

    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weight_sum,
        raw=True,
    )


def rolling_corr(series_x: pd.Series, series_y: pd.Series, window: int) -> pd.Series:
    """
    Rolling correlation wrapper for readability.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    return series_x.rolling(window).corr(series_y)


def rolling_cov(series_x: pd.Series, series_y: pd.Series, window: int) -> pd.Series:
    """
    Rolling covariance wrapper for readability.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    return series_x.rolling(window).cov(series_y)


def signed_power(series: pd.Series, power: float) -> pd.Series:
    """
    sign(x) * |x|^power
    """
    return np.sign(series) * (series.abs() ** power)


def scale(series: pd.Series, a: float = 1.0) -> pd.Series:
    """
    Rescale a series such that sum(abs(x)) = a.
    Mainly useful for future extensions.
    """
    denom = series.abs().sum()
    if denom == 0 or pd.isna(denom):
        return series * np.nan
    return series * (a / denom)


def delay(series: pd.Series, d: int) -> pd.Series:
    """
    Lag operator.
    """
    if d < 0:
        raise ValueError("d must be non-negative")
    return series.shift(d)


def delta(series: pd.Series, d: int) -> pd.Series:
    """
    Difference operator: x_t - x_{t-d}
    """
    if d <= 0:
        raise ValueError("d must be a positive integer")
    return series.diff(d)