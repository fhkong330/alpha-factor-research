import random
from typing import List

import pandas as pd
from io import StringIO
from config import N_STOCKS, RANDOM_SEED, VERBOSE


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TARGET_SECTOR = "Information Technology"


import requests

def get_sp500_table() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(SP500_WIKI_URL, headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch S&P500 table, status: {response.status_code}")

    
    tables = pd.read_html(StringIO(response.text))

    if not tables:
        raise ValueError("No tables found on Wikipedia page.")

    return tables[0]


def clean_symbol(symbol: str) -> str:
    """
    Clean ticker symbols for yfinance compatibility.
    Example: BRK.B -> BRK-B
    """
    return str(symbol).strip().replace(".", "-")


def get_technology_symbols() -> List[str]:
    """
    Return all S&P 500 Information Technology symbols.
    """
    sp500_table = get_sp500_table()

    if "GICS Sector" not in sp500_table.columns or "Symbol" not in sp500_table.columns:
        raise ValueError("Expected columns 'GICS Sector' and 'Symbol' not found in S&P 500 table.")

    tech_table = sp500_table[sp500_table["GICS Sector"] == TARGET_SECTOR].copy()
    tech_symbols = tech_table["Symbol"].astype(str).map(clean_symbol).tolist()

    # Deduplicate while preserving order
    seen = set()
    unique_symbols = []
    for symbol in tech_symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)

    if VERBOSE:
        print(f"[universe] Found {len(unique_symbols)} technology symbols in S&P 500.")

    return unique_symbols


def select_universe(n_stocks: int = N_STOCKS, seed: int = RANDOM_SEED) -> List[str]:
    """
    Randomly sample a fixed number of U.S. technology stocks from the
    S&P 500 Information Technology sector.
    """
    tech_symbols = get_technology_symbols()

    if len(tech_symbols) < n_stocks:
        raise ValueError(
            f"Requested {n_stocks} stocks, but only found {len(tech_symbols)} technology symbols."
        )

    random.seed(seed)
    selected = random.sample(tech_symbols, n_stocks)

    if VERBOSE:
        print(f"[universe] Selected {len(selected)} symbols with seed={seed}.")
        print(f"[universe] Sample universe: {selected}")

    return selected