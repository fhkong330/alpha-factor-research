from data.universe import select_universe
from data.downloader import download_price_data, split_data_by_symbol
from data.preprocess import preprocess_symbol_data
from utils.checks import validate_symbol_data_dict

from pipeline.build_alpha_panel import build_alpha_panel
from pipeline.compute_forward_returns import compute_forward_returns_from_symbol_data
from evaluation.ic_analysis import evaluate_alpha_panel_ic

import pandas as pd
import numpy as np

# =========================
# Step 1–5: Full pipeline
# =========================
symbols = select_universe()
raw_data = download_price_data(symbols)
raw_symbol_data = split_data_by_symbol(raw_data, symbols)
symbol_data = preprocess_symbol_data(raw_symbol_data)
validate_symbol_data_dict(symbol_data)

alpha_panel = build_alpha_panel(symbol_data)
forward_returns_df = compute_forward_returns_from_symbol_data(symbol_data)

ic_summary_df, _ = evaluate_alpha_panel_ic(alpha_panel, forward_returns_df)

print("\n========== IC Summary ==========")
print(ic_summary_df.head(10))

# =========================
# Step 6A: Select Top 5 alphas
# =========================
top_alphas = ic_summary_df.sort_values("mean_ic", ascending=False).head(5).index.tolist()

print("\nTop 5 alphas:", top_alphas)

# =========================
# Step 6B: Build composite signal
# =========================
# 把每个alpha抽出来 → 做平均
signal_list = []

for alpha_name in top_alphas:
    df = alpha_panel.xs(alpha_name, level=1, axis=1)
    signal_list.append(df)

# equal-weight combine
composite_signal = sum(signal_list) / len(signal_list)

print("\nComposite signal shape:", composite_signal.shape)
print(composite_signal.head())

# =========================
# Step 6C: Cross-sectional ranking
# =========================
rank_signal = composite_signal.rank(axis=1, pct=True)

print("\nRanked signal head:")
print(rank_signal.head())

# =========================
# Step 6D: Construct long-short portfolio
# =========================
long_threshold = 0.8
short_threshold = 0.2

long_mask = rank_signal >= long_threshold
short_mask = rank_signal <= short_threshold

# normalize weights
long_weights = long_mask.div(long_mask.sum(axis=1), axis=0)
short_weights = short_mask.div(short_mask.sum(axis=1), axis=0)

# portfolio return
portfolio_return = (
    (long_weights * forward_returns_df).sum(axis=1)
    - (short_weights * forward_returns_df).sum(axis=1)
)

portfolio_return = portfolio_return.dropna()

# =========================
# Step 6E: Performance metrics
# =========================
cum_return = (1 + portfolio_return).cumprod()

mean_ret = portfolio_return.mean()
std_ret = portfolio_return.std()

sharpe = mean_ret / std_ret * np.sqrt(252)

print("\n========== Portfolio Performance ==========")
print("Mean daily return:", mean_ret)
print("Std:", std_ret)
print("Sharpe (annualized):", sharpe)

print("\nCumulative return tail:")
print(cum_return.tail())

# =========================
# Step 6F: Diagnostics
# =========================
print("\n========== Diagnostics ==========")
print("Average long positions:", long_mask.sum(axis=1).mean())
print("Average short positions:", short_mask.sum(axis=1).mean())

print("\nSample portfolio returns:")
print(portfolio_return.head())