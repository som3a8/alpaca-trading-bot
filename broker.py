"""
broker.py — Alpaca Execution & Market Data Layer v2
=====================================================
Additions over v1:
  • execute_sell()                  — market-sell an open position (partial or full)
  • close_position_with_pnl()       — closes + records outcome in ledger
  • get_position()                  — single-ticker position lookup
  • get_live_inventory()            — unchanged
  • get_historical_market_data()    — unchanged
  • execute_calculated_trade()      — unchanged buy logic

Options additions (v3):
  • pick_option_contract()          — finds a liquid, near-OTM call/put for a signal
  • execute_option_trade()          — sizes and submits a BUY_TO_OPEN option order
  • parse_occ_symbol()              — decodes an OCC option symbol (root/expiry/right/strike)
  • close_expiring_options()        — force-closes option positions near expiration
"""

import os
import re
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, ContractType, PositionIntent
from alpaca.trading.requests import MarketOrderRequest, GetOptionContractsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError(
        "❌ Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in your .env file!"
    )

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY)
option_data_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)

# ── Equity position sizing (tunable — scaled for a $100k account) ──────────
# At the old 2.5-4%-of-cash sizing, 5 filled slots deployed only ~12% of a
# $100k account: 0.025*100000=2500, then 2.5% of the *shrinking* remaining
# cash each time after — the position count cap was rarely the binding
# constraint, the tiny fraction was. Raised so a fully-invested book deploys
# a meaningful majority of capital while staying diversified across
# MAX_EQUITY_POSITIONS names, each still capped at MAX_POSITION_VALUE.
#
# Sized up further for the competition: judged on one week's P&L ranked
# against other entrants (a tournament, not real investing) with paper
# money (a bad week costs nothing beyond not placing) — so bigger, more
# concentrated bets that increase the chance of a standout week are the
# rational play here, not the smoothest/safest capital curve.
MAX_EQUITY_POSITIONS = 6
MAX_POSITION_VALUE   = 30_000.00   # per-position cap (30% of a $100k account)
BASE_FRACTION        = 0.18        # of *available cash*, at CONF_FLOOR probability
CONF_BOOST_CAP        = 0.07       # additional fraction at CONF_CEIL probability
CONF_FLOOR           = 0.40        # matches main.py's MIN_BUY_PROB/MIN_SELL_PROB
CONF_CEIL            = 0.65        # realistic ceiling for a 3-class prob in this regime


# ── Portfolio helpers ─────────────────────────────────────────────────────

def get_live_inventory() -> dict[str, int]:
    """Returns {symbol: qty} for all open positions."""
    try:
        return {
            p.symbol: int(float(p.qty))
            for p in trading_client.get_all_positions()
        }
    except Exception as e:
        print(f"⚠️  Could not fetch positions: {e}")
        return {}


def get_position(symbol: str):
    """Returns the Alpaca position object for `symbol`, or None."""
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


# ── Market Data ───────────────────────────────────────────────────────────

def get_historical_market_data(symbol: str,
                                lookback_days: int = 300) -> "pd.DataFrame | None":
    """
    Returns a daily OHLCV DataFrame for `symbol` covering the last
    `lookback_days` calendar days.
    """
    try:
        end   = datetime.now()
        start = end - timedelta(days=lookback_days)

        req  = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        df   = bars.df

        if df is None or df.empty:
            return None

        if isinstance(df.index, pd.MultiIndex) and symbol in df.index.get_level_values(0):
            df = df.xs(symbol, level=0)

        return df.sort_index()

    except Exception as e:
        print(f"❌ Data fetch failed for {symbol}: {e}")
        return None


# ── SELL Execution ────────────────────────────────────────────────────────

def execute_sell(symbol: str, reason: str = "signal") -> bool:
    """
    Liquidates the full open position in `symbol` with a market sell order.
    Cancels any open orders first to free held shares.

    Returns True on success, False on failure.
    """
    pos = get_position(symbol)
    if pos is None:
        print(f"📭 {symbol}: No open position to sell.")
        return False

    qty = float(pos.qty)
    if qty <= 0:
        return False

    # Cancel open orders first
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        open_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for order in open_orders:
            try:
                trading_client.cancel_order_by_id(order.id)
            except Exception:
                pass
    except Exception:
        pass

    # Market sell (SELL_TO_CLOSE for options, plain sell for equities)
    try:
        is_option = pos.asset_class == AssetClass.US_OPTION
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            position_intent=PositionIntent.SELL_TO_CLOSE if is_option else None,
        )
        trading_client.submit_order(order)
        pnl = float(pos.unrealized_pl)
        kind = "OPTION" if is_option else "shares"
        print(f"💰 {symbol}: SELL executed ({qty} {kind}) | reason={reason} | UPL=${pnl:+.2f}")
        return True
    except Exception as e:
        print(f"❌ {symbol}: SELL failed — {e}")
        return False


def close_position_with_pnl(symbol: str, reason: str = "signal") -> float:
    """
    Closes a position and returns the realised P&L (from unrealized_pl at
    the time of close — Alpaca paper accounts settle immediately).
    Also records the outcome in the signal ledger.
    """
    import strategy as strat

    pos = get_position(symbol)
    if pos is None:
        return 0.0

    pnl     = float(pos.unrealized_pl)
    success = execute_sell(symbol, reason=reason)

    if success:
        was_correct = pnl > 0
        strat.record_outcome(symbol, "BUY", was_correct=was_correct)
        strat.record_trade_pnl(symbol, pnl)

    return pnl if success else 0.0


# ── BUY Execution ─────────────────────────────────────────────────────────

def execute_calculated_trade(
    symbol: str,
    current_price: float,
    trade_signal: str,
    confidence: float = 0.58,
    atr_pct: float | None = None,
) -> None:
    """
    Handles BUY or SELL signals:
      • BUY  → size + submit market buy (unchanged logic)
      • SELL → close full position via execute_sell()
    """
    # ── SELL path ─────────────────────────────────────────────────────
    if trade_signal == "SELL":
        pos = get_position(symbol)
        if pos is not None:
            close_position_with_pnl(symbol, reason="SELL signal")
        return

    if trade_signal != "BUY":
        return

    # ── BUY path ──────────────────────────────────────────────────────
    live_inventory   = get_live_inventory()
    shares_owned     = live_inventory.get(symbol, 0)
    current_exposure = shares_owned * current_price

    if current_exposure >= MAX_POSITION_VALUE:
        print(f"🛑 {symbol}: already at max exposure (${MAX_POSITION_VALUE:,.0f}).")
        return

    try:
        account = trading_client.get_account()

        # Max position cap — equity positions only (options have their own
        # separate cap in main.py; mixing them here under-counts equity slots)
        open_positions = trading_client.get_all_positions()
        equity_positions = [p for p in open_positions if p.asset_class == AssetClass.US_EQUITY]
        if len(equity_positions) >= MAX_EQUITY_POSITIONS:
            print(f"⏸  {symbol}: Max capacity ({MAX_EQUITY_POSITIONS}/{MAX_EQUITY_POSITIONS} equity positions). Skipping BUY.")
            return

        available_cash    = float(account.cash)
        total_equity      = float(account.equity)
        long_market_value = float(account.long_market_value)

        if long_market_value >= total_equity:
            print(f"🛑 Margin Guard: LMV (${long_market_value:,.2f}) ≥ equity. Skipping.")
            return

        # Fraction of available cash: BASE_FRACTION at the signal's minimum
        # actionable probability (~0.40, see main.py), scaling up to
        # BASE_FRACTION+CONF_BOOST_CAP as that probability approaches
        # CONF_CEIL. `confidence` here is expected to be the probability of
        # whichever class was actually acted on (buy_prob for a BUY), not
        # the model's raw argmax confidence.
        conf_boost = (confidence - CONF_FLOOR) / (CONF_CEIL - CONF_FLOOR) * CONF_BOOST_CAP
        fraction   = BASE_FRACTION + max(0.0, min(conf_boost, CONF_BOOST_CAP))

        cash_budget = (available_cash * fraction) * 0.98
        budget      = min(cash_budget, MAX_POSITION_VALUE - current_exposure)

        if budget > available_cash:
            budget = available_cash * 0.95

        if budget < 1.00:
            if available_cash >= 1.00:
                budget = 1.00
            else:
                print(f"⏸  {symbol}: Budget (${budget:.2f}) below $1.00 minimum. Skipping.")
                return

        qty = round(budget / current_price, 4)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(order)
        print(f"🚀 {symbol}: BUY {qty} shares @ ~${current_price:.2f}")

    except Exception as e:
        print(f"❌ {symbol}: Execution failed — {e}")


# ── Options ──────────────────────────────────────────────────────────────
# A small, separately risk-gated "sleeve" that expresses the same directional
# ML signal (BUY→call, SELL→put) via long, defined-risk options instead of
# (or alongside) equity. Long-only: max loss per trade is the premium paid.

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(symbol: str) -> "dict | None":
    """
    Decodes a standard OCC option symbol, e.g. 'AAPL260911C00325000' ->
    {'root': 'AAPL', 'expiration': date(2026,9,11), 'right': 'C', 'strike': 325.0}
    Returns None if `symbol` isn't an OCC-format option symbol (e.g. a plain equity ticker).
    """
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root, yymmdd, right, strike_raw = m.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "root":       root,
        "expiration": expiration,
        "right":      right,
        "strike":     int(strike_raw) / 1000.0,
    }


def pick_option_contract(underlying: str, right: "ContractType",
                          current_price: float, otm_pct: float = 0.03,
                          dte_min: int = 14, dte_max: int = 45):
    """
    Finds a liquid contract near `otm_pct` out-of-the-money for `underlying`,
    expiring between `dte_min` and `dte_max` days out. Prefers contracts with
    a recent close price (a simple liquidity proxy — Alpaca's screener-free
    contracts endpoint doesn't expose open interest reliably on every tier).
    Returns the contract object, or None if nothing suitable was found.
    """
    target_strike = current_price * (1 + otm_pct if right == ContractType.CALL else 1 - otm_pct)
    strike_band    = current_price * 0.10   # search ±10% around target for candidates

    today = date.today()
    try:
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            type=right,
            expiration_date_gte=today + timedelta(days=dte_min),
            expiration_date_lte=today + timedelta(days=dte_max),
            strike_price_gte=str(round(max(target_strike - strike_band, 0.01), 2)),
            strike_price_lte=str(round(target_strike + strike_band, 2)),
            limit=100,
        )
        contracts = trading_client.get_option_contracts(req).option_contracts
    except Exception as e:
        print(f"⚠️  {underlying}: option chain fetch failed — {e}")
        return None

    if not contracts:
        return None

    liquid = [c for c in contracts if c.close_price is not None]
    pool   = liquid if liquid else contracts

    dte_mid = (dte_min + dte_max) / 2

    def _score(c):
        dte = (c.expiration_date - today).days
        return (abs(float(c.strike_price) - target_strike), abs(dte - dte_mid))

    return min(pool, key=_score)


def get_option_mid_price(symbol: str, fallback: "float | None" = None) -> "float | None":
    """Latest bid/ask midpoint for an option contract; falls back to `fallback` (e.g. prior close)."""
    try:
        quote = option_data_client.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=symbol)
        )[symbol]
        bid, ask = float(quote.bid_price), float(quote.ask_price)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
    except Exception:
        pass
    return fallback


def execute_option_trade(underlying: str, current_price: float, right: "ContractType",
                          budget: float, otm_pct: float = 0.03,
                          dte_min: int = 14, dte_max: int = 45) -> bool:
    """
    Buys-to-open the nearest liquid ~`otm_pct` OTM call/put on `underlying`,
    sized to fit within `budget` (whole contracts only — a contract controls
    100 shares, so budget must cover at least one). Returns True on success.
    """
    contract = pick_option_contract(underlying, right, current_price, otm_pct, dte_min, dte_max)
    if contract is None:
        print(f"⏸  {underlying}: no suitable {right.value} contract found (options skipped).")
        return False

    price = get_option_mid_price(contract.symbol, fallback=(
        float(contract.close_price) if contract.close_price is not None else None
    ))
    if price is None or price <= 0:
        print(f"⏸  {contract.symbol}: no tradeable quote (illiquid) — options skipped.")
        return False

    contract_cost = price * 100
    qty = int(budget // contract_cost)
    if qty < 1:
        print(f"⏸  {contract.symbol}: 1 contract (${contract_cost:.2f}) exceeds "
              f"options budget (${budget:.2f}) — skipped.")
        return False

    try:
        order = MarketOrderRequest(
            symbol=contract.symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        )
        trading_client.submit_order(order)
        dte = (contract.expiration_date - date.today()).days
        print(f"📜 {underlying}: BUY {qty}x {contract.symbol} "
              f"(strike=${float(contract.strike_price):.2f}, {dte}d to expiry) "
              f"@ ~${price:.2f}/contract (${price*100:.2f} ea)")
        return True
    except Exception as e:
        print(f"❌ {contract.symbol}: option order failed — {e}")
        return False


def get_expiring_option_positions(days_before: int = 3) -> list:
    """
    Returns open option positions expiring within `days_before` days —
    letting a long option ride into expiration risks total premium loss
    (OTM) or an unwanted exercise/assignment (ITM), neither of which this
    bot is built to handle, so these must be closed out early.
    """
    try:
        positions = trading_client.get_all_positions()
    except Exception:
        return []

    today = date.today()
    expiring = []
    for pos in positions:
        if pos.asset_class != AssetClass.US_OPTION:
            continue
        parsed = parse_occ_symbol(pos.symbol)
        if parsed and (parsed["expiration"] - today).days <= days_before:
            expiring.append(pos)
    return expiring