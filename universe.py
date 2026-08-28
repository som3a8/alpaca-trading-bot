"""
universe.py — Dynamic Watchlist Builder
========================================
Builds the trading universe fresh every session from two live data sources
layered on top of a curated, always-included core:

  1. Curated seed list (always included)      — 47 liquid, diversified,
                                                  large/mid-cap names spanning
                                                  every major sector, hand-
                                                  picked so the bot always has
                                                  a solid universe even if
                                                  every external source fails
  2. S&P 500 constituents (Wikipedia)          — large-cap baseline
  3. yfinance screener — top volume / movers   — momentum candidates

A NASDAQ-100 Wikipedia scrape was previously a third source but is no longer
wired in — Wikipedia restructured that page and the constituent table is no
longer in a scrapeable format, so it silently returned zero results. Not a
real loss: the S&P 500 set already includes the large NASDAQ names (AAPL,
MSFT, NVDA, GOOGL, AMZN, META, etc.), and a fragile scrape returning nothing
is worse than not depending on it.

The final list is de-duplicated, validated (has recent price data), and
capped at MAX_UNIVERSE_SIZE to keep API costs sane.
"""

import time
import random
import requests
import warnings
import yfinance as yf
import pandas as pd
from io import StringIO

warnings.filterwarnings("ignore")

MAX_UNIVERSE_SIZE = 120   # Upper bound — feel free to raise for paid API tiers
VALIDATE_TIMEOUT  = 2     # Seconds to wait per ticker validation request

# Tickers that are ALWAYS in the universe regardless of screener output.
# Curated for liquidity + sector diversity + long, clean trading history
# (the same set validated in backtest.py's 2.2-year walk-forward run).
SEED_TICKERS = [
    # Tech / mega-cap growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "ADBE", "CRM", "ORCL", "AVGO",
    # Financials
    "JPM", "BAC", "GS", "V", "MA", "AXP",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK",
    # Consumer
    "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX",
    # Industrials
    "BA", "CAT", "GE", "HON", "UPS",
    # Energy
    "XOM", "CVX", "COP",
    # Communication
    "DIS", "NFLX", "CMCSA",
    # Utilities
    "NEE", "DUK",
    # Real estate
    "AMT",
    # Semis
    "AMD", "INTC", "TXN", "QCOM",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _fetch_sp500() -> list[str]:
    """Scrapes S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text))
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"⚠️  S&P 500 fetch failed: {e}")
        return []


def _fetch_top_movers() -> list[str]:
    """
    Uses yfinance to pull the top gainers and most-active tickers —
    a lightweight proxy for 'what the market is paying attention to today'.

    yf.screen() returns a dict (not a DataFrame) with a "quotes" list of
    per-ticker dicts — a prior version of this function checked for a
    DataFrame-style ".columns" attribute, which a dict never has, so every
    call silently fell through to the except and returned nothing.
    """
    candidates = []
    for screen in ("most_actives", "day_gainers", "day_losers"):
        try:
            result = yf.screen(screen, count=25)
            quotes = result.get("quotes", []) if isinstance(result, dict) else []
            candidates.extend(q["symbol"] for q in quotes if q.get("symbol"))
        except Exception as e:
            print(f"⚠️  yfinance screener ({screen}) failed: {e}")
    return candidates


def _validate_ticker(symbol: str) -> bool:
    """
    Quick sanity check: does yfinance return a non-empty price for this symbol?
    Filters out delisted, ETFs-you-don't-want, and data-unavailable tickers.
    """
    try:
        info = yf.Ticker(symbol).fast_info
        price = getattr(info, "last_price", None)
        return price is not None and price > 1.0  # Skip penny stocks
    except Exception:
        return False


# ── Public API ─────────────────────────────────────────────────────────────

def build_universe(verbose: bool = True) -> list[str]:
    """
    Assembles and returns the dynamic trading universe.
    Prints a summary when verbose=True.
    """
    if verbose:
        print("\n🌐 Building dynamic trading universe...")

    raw: list[str] = list(SEED_TICKERS)  # Always start with seeds

    sp500    = _fetch_sp500()
    movers   = _fetch_top_movers()

    if verbose:
        print(f"   └─ Seeds: {len(SEED_TICKERS)} | S&P 500: {len(sp500)} | Movers: {len(movers)}")

    # Combine and de-duplicate while preserving seed priority
    combined = raw + sp500 + movers
    seen     = set()
    deduped  = []
    for t in combined:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)

    # Remove obvious non-equity symbols
    deduped = [t for t in deduped
               if "." not in t           # ADRs with dots (e.g. BRK.B handled above)
               and len(t) <= 5
               and t.isalpha()]

    # Sample down if over cap before validation (validation is slow)
    seeds_set = set(SEED_TICKERS)
    non_seeds = [t for t in deduped if t not in seeds_set]
    random.shuffle(non_seeds)
    candidate_pool = SEED_TICKERS + non_seeds[: MAX_UNIVERSE_SIZE * 2]

    # Validate (spot-check a subset to keep startup time reasonable)
    if verbose:
        print(f"   └─ Validating up to {len(candidate_pool)} candidates...")

    validated = list(SEED_TICKERS)  # Seeds are always trusted
    for symbol in candidate_pool:
        if symbol in seeds_set:
            continue
        if len(validated) >= MAX_UNIVERSE_SIZE:
            break
        if _validate_ticker(symbol):
            validated.append(symbol)
        time.sleep(0.05)   # Gentle rate-limit on validation calls

    if verbose:
        print(f"✅ Universe ready: {len(validated)} tickers")

    return validated