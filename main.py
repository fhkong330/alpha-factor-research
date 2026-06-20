import numpy as np
import pandas as pd

from config import (
    ALPHA_PANEL_PATH,
    COMBO_SIGNAL_PATH,
    IC_SUMMARY_PATH,
    REGRESSION_SUMMARY_PATH,
    TOP_N_ALPHAS,
    VERBOSE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    SIGNAL_STANDARDIZATION_METHOD,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    PORTFOLIO_SUMMARY_PATH,
    PORTFOLIO_RETURN_PATH,
    PORTFOLIO_CUM_RETURN_PATH,
)
from data.universe import select_universe
from data.downloader import download_price_data, split_data_by_symbol
from data.preprocess import preprocess_symbol_data
from evaluation.factor_selection import get_top_alpha_names, select_top_alphas
from evaluation.ic_analysis import evaluate_alpha_panel_ic
from evaluation.regression_validation import (
    regression_summary_text,
    run_predictive_regression,
    summarize_regression_result,
)
from pipeline.build_alpha_panel import build_and_save_alpha_panel
from pipeline.build_combo_signal import build_combo_signal
from pipeline.compute_forward_returns import (
    compute_next_day_return_panel_from_symbol_data,
)
from utils.checks import validate_symbol_data_dict
from utils.io_utils import save_dataframe, save_text

def run_long_short_backtest(
    composite_signal_df: pd.DataFrame,
    realized_return_df: pd.DataFrame,
    long_threshold: float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Run a simple daily-rebalanced long-short backtest using next-day realized returns.
    """
    rank_signal = composite_signal_df.rank(axis=1, pct=True)

    long_mask = rank_signal >= long_threshold
    short_mask = rank_signal <= short_threshold

    long_weights = long_mask.div(long_mask.sum(axis=1), axis=0)
    short_weights = short_mask.div(short_mask.sum(axis=1), axis=0)

    portfolio_return = (
        (long_weights * realized_return_df).sum(axis=1)
        - (short_weights * realized_return_df).sum(axis=1)
    )
    # portfolio_return = -portfolio_return

    valid_days = (long_mask.sum(axis=1) > 0) & (short_mask.sum(axis=1) > 0)
    portfolio_return = portfolio_return[valid_days].dropna()

    cumulative_return = (1 + portfolio_return).cumprod()

    mean_ret = portfolio_return.mean()
    std_ret = portfolio_return.std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret != 0 else np.nan

    stats = {
        "mean_daily_return": float(mean_ret),
        "std_daily_return": float(std_ret),
        "annualized_sharpe": float(sharpe),
        "avg_long_positions": float(long_mask.sum(axis=1).mean()),
        "avg_short_positions": float(short_mask.sum(axis=1).mean()),
    }

    return portfolio_return, cumulative_return, stats


def main() -> None:
    # =========================
    # Step 1: Universe selection
    # =========================
    symbols = select_universe()

    # =========================
    # Step 2: Download raw data
    # =========================
    raw_data = download_price_data(symbols)
    raw_symbol_data = split_data_by_symbol(raw_data, symbols)

    # =========================
    # Step 3: Preprocess data
    # =========================
    symbol_data = preprocess_symbol_data(raw_symbol_data)
    validate_symbol_data_dict(symbol_data)

    # =========================
    # Step 4: Build alpha panel
    # =========================
    alpha_panel = build_and_save_alpha_panel(symbol_data, output_path=ALPHA_PANEL_PATH)

    # =========================
    # =========================
    # Step 5: Compute next-day realized returns (for both IC evaluation and backtest)
    next_day_return_df = compute_next_day_return_panel_from_symbol_data(symbol_data)

    # Step 6: Train / Test split
    train_alpha_panel = alpha_panel.loc[:TRAIN_END_DATE]
    train_next_day_return_df = next_day_return_df.loc[:TRAIN_END_DATE]

    test_alpha_panel = alpha_panel.loc[TEST_START_DATE:]
    test_next_day_return_df = next_day_return_df.loc[TEST_START_DATE:]

    if VERBOSE:
        print("\n[main] Train/Test split:")
        print(f"[main] Train alpha panel shape: {train_alpha_panel.shape}")
        print(f"[main] Train next-day return shape: {train_next_day_return_df.shape}")
        print(f"[main] Test alpha panel shape: {test_alpha_panel.shape}")
        print(f"[main] Test next-day return shape: {test_next_day_return_df.shape}")

    # =========================
    # Step 8: Evaluate IC on TRAIN set only
    # =========================
    ic_summary_df, ic_series_dict = evaluate_alpha_panel_ic(
        alpha_panel=train_alpha_panel,
        forward_returns_df=train_next_day_return_df,
    )
    save_dataframe(ic_summary_df, IC_SUMMARY_PATH)

    # =========================
    # Step 9: Select top alphas on TRAIN set
    # =========================
    top_alpha_df = select_top_alphas(
        ic_summary_df=ic_summary_df,
        top_n=TOP_N_ALPHAS,
        ranking_metric="mean_ic",
        ascending=False,
    )
    selected_alpha_names = get_top_alpha_names(top_alpha_df)

    # =========================
    # Step 10: Build combo signal on TEST set
    # =========================
    combo_signal_df = build_combo_signal(
        alpha_panel=test_alpha_panel,
        selected_alpha_names=selected_alpha_names,
        method="equal_weight",
        ic_summary_df=ic_summary_df,
        standardization_method=SIGNAL_STANDARDIZATION_METHOD,
    )
    save_dataframe(combo_signal_df, COMBO_SIGNAL_PATH)

    # =========================
    # Step 11: Predictive regression on TEST set
    # =========================
    regression_model = run_predictive_regression(
        signal_df=combo_signal_df,
        forward_returns_df=test_next_day_return_df,
        signal_name="signal",
        return_name="next_return",
    )
    regression_stats = summarize_regression_result(regression_model)
    regression_text = regression_summary_text(regression_model)
    save_text(regression_text, REGRESSION_SUMMARY_PATH)

    # =========================
    # Step 12: Long-short backtest on TEST set
    # =========================
    portfolio_return, cumulative_return, portfolio_stats = run_long_short_backtest(
        composite_signal_df=combo_signal_df,
        realized_return_df=test_next_day_return_df,
        long_threshold=LONG_THRESHOLD,
        short_threshold=SHORT_THRESHOLD,
    )

    save_dataframe(portfolio_return.to_frame(name="portfolio_return"), PORTFOLIO_RETURN_PATH)
    save_dataframe(cumulative_return.to_frame(name="cumulative_return"), PORTFOLIO_CUM_RETURN_PATH)

    portfolio_summary_text = "\n".join(
        [f"{k}: {v}" for k, v in portfolio_stats.items()]
    )
    save_text(portfolio_summary_text, PORTFOLIO_SUMMARY_PATH)

    # =========================
    # Step 13: Console summary
    # =========================
    print("\n========== Alpha Research Pipeline Complete ==========")
    print(f"Number of valid symbols: {len(symbol_data)}")
    print(f"Alpha panel saved to: {ALPHA_PANEL_PATH}")
    print(f"IC summary (TRAIN) saved to: {IC_SUMMARY_PATH}")
    print(f"Combo signal (TEST) saved to: {COMBO_SIGNAL_PATH}")
    print(f"Regression summary (TEST) saved to: {REGRESSION_SUMMARY_PATH}")
    print(f"Portfolio return saved to: {PORTFOLIO_RETURN_PATH}")
    print(f"Portfolio cumulative return saved to: {PORTFOLIO_CUM_RETURN_PATH}")
    print(f"Portfolio summary saved to: {PORTFOLIO_SUMMARY_PATH}")

    print("\nTop selected alphas from TRAIN set:")
    print(top_alpha_df)

    print("\nRegression key metrics on TEST set:")
    for key, value in regression_stats.items():
        print(f"{key}: {value}")

    print("\nPortfolio key metrics on TEST set:")
    for key, value in portfolio_stats.items():
        print(f"{key}: {value}")

    if VERBOSE and ic_series_dict:
        sample_alpha = selected_alpha_names[0] if selected_alpha_names else None
        if sample_alpha is not None:
            print(f"\nSample TRAIN daily IC series preview for {sample_alpha}:")
            print(ic_series_dict[sample_alpha].dropna().head())

if __name__ == "__main__":
    main()