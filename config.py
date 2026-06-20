from pathlib import Path

# =========================
# Project Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cache"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Universe / Data Settings
# =========================
UNIVERSE_SOURCE = "sp500_information_technology"
N_STOCKS = 40
RANDOM_SEED = 42

START_DATE = "2022-01-01"
END_DATE = "2024-01-01"

# =========================
# Train / Test Split
# =========================
TRAIN_END_DATE = "2023-06-30"
TEST_START_DATE = "2023-07-01"

# =========================
# Signal Standardization
# =========================
SIGNAL_STANDARDIZATION_METHOD = "cross_sectional_rank"
# options:
# - "cross_sectional_rank"
# - "none"

# =========================
# Portfolio Construction
# =========================
LONG_THRESHOLD = 0.8
SHORT_THRESHOLD = 0.2
PORTFOLIO_RETURN_METHOD = "next_day"
# options:
# - "next_day"

PORTFOLIO_SUMMARY_PATH = OUTPUT_DIR / "portfolio_performance.txt"
PORTFOLIO_RETURN_PATH = OUTPUT_DIR / "portfolio_return_series.csv"
PORTFOLIO_CUM_RETURN_PATH = OUTPUT_DIR / "portfolio_cumulative_return.csv"
# =========================
# Alpha Research Settings
# =========================
FORWARD_RETURN_HORIZON = 5      # next 5 trading days
SIGNAL_LAG_DAYS = 1             # shift to avoid look-ahead
MIN_CROSS_SECTION_SIZE = 8      # minimum stocks required per date for IC
TOP_N_ALPHAS = 5

# Optional future extension
COMBO_METHOD = "equal_weight"   # other options can be added later

# =========================
# Output Files
# =========================
ALPHA_PANEL_PATH = OUTPUT_DIR / "alpha_scores_panel.csv"
IC_SUMMARY_PATH = OUTPUT_DIR / "alpha_ic_summary.csv"
COMBO_SIGNAL_PATH = OUTPUT_DIR / "combo_alpha_signal.csv"
REGRESSION_SUMMARY_PATH = OUTPUT_DIR / "regression_summary.txt"

# =========================
# Data Requirements
# =========================
REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
MIN_HISTORY_ROWS = 260  # enough buffer for longer rolling windows in current alpha set

# =========================
# Logging
# =========================
VERBOSE = True


POLYGON_API_KEY = "POLYGON_API_KEY"