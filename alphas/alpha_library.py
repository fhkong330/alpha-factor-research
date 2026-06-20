import numpy as np
import pandas as pd

from alphas.operators import ts_rank, decay_linear


def alpha_2(df: pd.DataFrame) -> pd.Series:
    return -df["volume"].pct_change(2).rolling(6).corr(
        ((df["close"] - df["open"]) / df["open"])
    ).rank(pct=True)


def alpha_3(df: pd.DataFrame) -> pd.Series:
    return -df["open"].rolling(10).corr(df["volume"]).rank(pct=True)


def alpha_6(df: pd.DataFrame) -> pd.Series:
    return -df["open"].rolling(10).corr(df["volume"])


def alpha_12(df: pd.DataFrame) -> pd.Series:
    return np.sign(df["volume"].diff(1)) * (-df["close"].diff(1))


def alpha_19(df: pd.DataFrame) -> pd.Series:
    return -np.sign(
        (df["close"] - df["close"].shift(7)) + df["close"].diff(7)
    ) * (1 + df["close"].pct_change(250).rolling(5).mean())


def alpha_25(df: pd.DataFrame) -> pd.Series:
    return (
        (-1 * df["close"].pct_change()) * df["volume"] * df["vwap"] * (df["high"] - df["close"])
    ).rank(pct=True)


def alpha_27(df: pd.DataFrame) -> pd.Series:
    signal = (
        df["volume"].rolling(6).corr(df["vwap"].rank(pct=True).rolling(6).mean()) / 2.0
    ).rank(pct=True)
    return pd.Series(np.where(signal > 0.5, -1, 1), index=df.index)


def alpha_30(df: pd.DataFrame) -> pd.Series:
    r = (
        np.sign(df["close"].diff(1))
        + np.sign(df["close"].diff(1).shift(1))
        + np.sign(df["close"].diff(1).shift(2))
    )
    return ((1.0 - r.rank(pct=True)) * df["volume"].rolling(5).sum()) / df["volume"].rolling(20).sum()


def alpha_36(df: pd.DataFrame) -> pd.Series:
    return (
        2.21 * df["close"].rolling(15).corr(df["volume"].shift(1)).rank(pct=True)
        + 0.7 * (df["open"] - df["close"]).rank(pct=True)
        + 0.73 * ts_rank(df["close"].shift(6), 5).rank(pct=True)
        + 0.6 * ((df["close"].rolling(200).mean() - df["open"]) * (df["close"] - df["open"])).rank(pct=True)
    )


def alpha_42(df: pd.DataFrame) -> pd.Series:
    return (df["vwap"] - df["close"]).rank(pct=True) / (df["vwap"] + df["close"]).rank(pct=True)


def alpha_48(df: pd.DataFrame) -> pd.Series:
    return df["close"].diff().rolling(250).corr(df["close"].diff().shift(1)).rank(pct=True)


def alpha_53(df: pd.DataFrame) -> pd.Series:
    numerator = (df["close"] - df["low"]) - (df["high"] - df["close"])
    denominator = (df["close"] - df["low"]).diff(9)
    return -(numerator / denominator)


def alpha_55(df: pd.DataFrame) -> pd.Series:
    return -df["close"].rank(pct=True).rolling(12).corr(df["volume"].rank(pct=True)).rank(pct=True)


def alpha_60(df: pd.DataFrame) -> pd.Series:
    return 2 * df["close"].rank(pct=True) - ts_rank(df["close"].rolling(10).max(), 10)


def alpha_61(df: pd.DataFrame) -> pd.Series:
    return (df["vwap"] - df["vwap"].rolling(16).min()).rank(pct=True)


def alpha_62(df: pd.DataFrame) -> pd.Series:
    lhs = df["vwap"].rolling(10).corr(df["volume"].rolling(10).mean()).rank(pct=True)
    rhs = df["open"].rank(pct=True)
    return -((lhs < rhs).astype(int))


def alpha_63(df: pd.DataFrame) -> pd.Series:
    left = decay_linear(df["close"].diff(2), 8).rank(pct=True)
    right = decay_linear(df["close"].rolling(5).corr(df["vwap"]), 12).rank(pct=True)
    return -(left - right)


def alpha_64(df: pd.DataFrame) -> pd.Series:
    lhs = (df["open"] - df["vwap"]).rank(pct=True) + df["low"].rank(pct=True)
    rhs = (df["high"] + df["vwap"]).rank(pct=True)
    return -((lhs < rhs).astype(int))


def alpha_65(df: pd.DataFrame) -> pd.Series:
    lhs = (df["open"] - df["open"].rolling(14).min()).rank(pct=True)
    rhs = df["volume"].rolling(12).corr(df["vwap"]).rank(pct=True)
    return -((lhs < rhs).astype(int))


def alpha_66(df: pd.DataFrame) -> pd.Series:
    base = (df["low"] - df["vwap"]) / (df["open"] - ((df["high"] + df["low"]) / 2))
    return -(decay_linear(df["vwap"].diff(), 7) + ts_rank(decay_linear(base, 11), 6))


def alpha_67(df: pd.DataFrame) -> pd.Series:
    return -(df["high"] - df["high"].rolling(2).min()).rank(pct=True)


def alpha_68(df: pd.DataFrame) -> pd.Series:
    lhs = ts_rank(df["high"].rolling(9).corr(df["volume"].rolling(9).mean()), 13)
    rhs = df["close"].diff().rank(pct=True)
    return -((lhs < rhs).astype(int))


def alpha_69(df: pd.DataFrame) -> pd.Series:
    signal = df["vwap"].rolling(4).corr(df["volume"].rolling(4).mean()).rank(pct=True) * df["close"].rank(pct=True)
    return -(signal.rank(pct=True))


def alpha_70(df: pd.DataFrame) -> pd.Series:
    signal = df["vwap"].diff().rank(pct=True) ** ts_rank(df["close"].rolling(17).corr(df["vwap"]), 17)
    return -(signal.rank(pct=True))


def alpha_71(df: pd.DataFrame) -> pd.Series:
    return ts_rank(decay_linear(df["close"], 10), 5)


def alpha_72(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(10).corr(df["volume"].rolling(10).mean()).rank(pct=True)


def alpha_73(df: pd.DataFrame) -> pd.Series:
    return -(df["vwap"].diff(4).rank(pct=True))


def alpha_74(df: pd.DataFrame) -> pd.Series:
    lhs = df["close"].rolling(15).corr(df["volume"].rank(pct=True))
    rhs = (df["high"] + df["vwap"]).rank(pct=True)
    return -((lhs < rhs).astype(int))


def alpha_75(df: pd.DataFrame) -> pd.Series:
    return df["vwap"].rolling(4).corr(df["volume"]).rank(pct=True)


def alpha_101(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)


def calculate_all_alphas(df: pd.DataFrame, alpha_registry: dict[str, callable]) -> pd.DataFrame:
    """
    Compute all registered alphas on a single-ticker OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns such as open, high, low, close, volume, vwap.
    alpha_registry : dict[str, callable]
        Mapping from alpha name to alpha function.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per alpha.
    """
    result = pd.DataFrame(index=df.index)

    for alpha_name, alpha_func in alpha_registry.items():
        try:
            result[alpha_name] = alpha_func(df)
        except Exception:
            result[alpha_name] = np.nan

    return result