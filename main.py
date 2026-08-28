"""
main.py — Autonomous Trading Bot v3 · Master Execution Loop
=============================================================
New in v3 (all 6 suggested upgrades implemented):
  1. Trailing stop loss     — stop moves up as position profits; never goes down
  2. Time-based exit        — stale positions (no movement in N loops) are closed
  3. Sector / market regime — SPY + QQQ checked every loop; all BUYs suppressed
                              when the broad market is in a confirmed downtrend
  4. Earnings blackout      — yfinance calendar checked; tickers with earnings
                              within EARNINGS_BLACKOUT_DAYS are skipped
  5. Multi-timeframe (MTF)  — 1-hour bars fetched alongside daily; BUY only when
                              hourly trend also bullish (EMA12 > EMA26 on 1h)
  6. Dashboard P&L chart    — session_pnl written to pnl_log.json every loop
                              so dashboard.py can plot a cumulative curve

Original features retained:
  • SELL signal routing + profit-target ($8 UPL)
  • Circuit breaker (-$2 UPL hard close + session ban)
  • Adaptive sleep
  • Confidence gate
"""

import json
import sys
import time
import math
import traceback
import broker
import strategy
import universe as universe_module
import yfinance as yf

from datetime import datetime, timedelta
from pathlib import Path
from alpaca.trading.enums import AssetClass, ContractType


# =====================================================================
# CONFIGURATION
# =====================================================================

CIRCUIT_BREAKER_LOSS_PCT       = -0.05    # -5% of position cost basis: hard stop
PROFIT_TAKE_TARGET_PCT         =  0.12    # +12% of position cost basis: take-profit
# -2.5% was originally backtested and found to be the single biggest drag on
# returns: 177/551 trades (32%) hit that stop, losing -$49,655 total — more
# than every other exit type's gains combined. Widened to -4%, which helped
# (win rate 50.3%→57.7%), then to -5%/+12% here for a different reason: this
# account is judged on ONE week's P&L, ranked against other entrants — a
# tournament payoff, not a real-investing one. Paper money means a bad week
# costs nothing beyond not placing, so a strategy that reliably nets a small
# return doesn't help; one with a real shot at a big week does. That means
# deliberately widening the reward:risk ratio to let winners run much
# further (+6%→+12% target, 1.5%→3% trailing-stop room) rather than tuning
# for the smoothest backtest curve.
MIN_BUY_PROB                   =  0.40
MIN_SELL_PROB                  =  0.40

# ── ATR-scaled equity stops ────────────────────────────────────────────
# CIRCUIT_BREAKER_LOSS_PCT/PROFIT_TAKE_TARGET_PCT above are the same flat
# number for all 47 tickers — a quiet stock (KO) and a volatile one (NVDA)
# get identical stop distance, which isn't principled. Positions now scale
# their own stop/target to their entry-day ATR% instead, clamped to sane
# bounds and preserving the ~2.4:1 reward:risk ratio the flat numbers set.
# Falls back to the flat constants if no ATR was recorded for a position
# (e.g. it predates this feature, or ATR was unavailable at entry).
ATR_STOP_MULT    = 2.0     # stop at -2x entry-day ATR%
ATR_TARGET_MULT  = 4.8     # target at +4.8x — keeps the -5%/+12% ratio
ATR_STOP_FLOOR   = -0.10   # never wider than -10%, regardless of ATR
ATR_STOP_CEIL    = -0.02   # never tighter than -2%, regardless of ATR
ATR_TARGET_FLOOR =  0.05   # never smaller than +5%
ATR_TARGET_CEIL  =  0.25   # never larger than +25%


def _atr_scaled_thresholds(symbol: str) -> tuple[float, float]:
    """Returns (stop_pct, target_pct) for an equity position, scaled to its
    own entry-day ATR% where available, else the flat config constants."""
    meta = position_meta.get(symbol)
    atr  = meta.get("atr_pct") if meta else None
    if not atr or atr <= 0:
        return CIRCUIT_BREAKER_LOSS_PCT, PROFIT_TAKE_TARGET_PCT

    stop   = max(ATR_STOP_FLOOR,   min(ATR_STOP_CEIL,   -atr * ATR_STOP_MULT))
    target = max(ATR_TARGET_FLOOR, min(ATR_TARGET_CEIL,  atr * ATR_TARGET_MULT))
    return stop, target


# ── Trailing stop ────────────────────────────────────────────────────
TRAIL_ACTIVATE_USD   =  3.00   # Start trailing only after $3 unrealised gain
TRAIL_STOP_PCT       =  0.03   # Trail 3% below the running high-water mark — more
                                # room than 1.5% so a real trend isn't cut short

# ── Time-based exit ──────────────────────────────────────────────────
STALE_POSITION_LOOPS = 12      # Close position if held for this many loops
                                # with <1% total move (≈ 1–2 hours at normal cadence)
STALE_MOVE_THRESHOLD =  0.01   # 1% threshold for "meaningful movement"

# ── Market regime ────────────────────────────────────────────────────
REGIME_TICKERS       = ["SPY", "QQQ"]   # Checked each loop
REGIME_EMA_FAST      = 8
REGIME_EMA_SLOW      = 21

# ── Earnings blackout ─────────────────────────────────────────────────
EARNINGS_BLACKOUT_DAYS = 2     # Skip tickers with earnings within this window

# ── Multi-timeframe ───────────────────────────────────────────────────
MTF_ENABLED          = True    # Set False to disable the hourly confirmation
MTF_EMA_FAST         = 12
MTF_EMA_SLOW         = 26

# ── Universe / sleep ─────────────────────────────────────────────────
UNIVERSE_REFRESH_LOOPS = 8
SLEEP_MIN  = 45     # was 120 — check back faster, especially during market hours
SLEEP_MAX  = 150    # was 600 — a full 120-ticker scan already takes several
                     # minutes on its own (800d fetch x TICKER_SLEEP each), so
                     # this mainly trims the *idle* gap between scans, not the
                     # scan itself
TICKER_SLEEP = 1.5

# ── Options sleeve ──────────────────────────────────────────────────────
# Expresses the same directional ML signal (BUY→call, SELL→put) via small,
# defined-risk long options alongside the equity book. Kept deliberately
# small: options are far more volatile than the underlying, so it uses a
# tighter budget, a higher confidence bar, and wider (but still hard) stops.
#
# Tuned for tournament upside, not smooth risk-adjusted returns (see note
# above CIRCUIT_BREAKER_LOSS_PCT): options are naturally convex — capped
# loss (the premium), open-ended payoff shape — which is exactly the risk
# profile that helps in a "rank by P&L" format. Sized up accordingly, with
# further OTM strikes (cheaper premium, more leverage per dollar) and
# shorter expiries (more gamma/price-sensitivity within the ~1-week window,
# rather than paying for time value that extends well past when it matters).
OPTIONS_ENABLED             = True
OPTIONS_MIN_PROB            = 0.46     # Higher bar than equity's 0.40 — decay/leverage risk
OPTIONS_BUDGET_FRACTION     = 0.025    # 2.5% of equity per options trade (was 5% —
                                        # halved after a day where six options
                                        # circuit-breaker exits compounded into a
                                        # ~15% single-day account drawdown)
OPTIONS_MAX_CONCURRENT      = 5        # Cap concurrent option positions
OPTIONS_OTM_PCT             = 0.06     # Target ~6% out-of-the-money — cheaper, more convex
OPTIONS_DTE_MIN             = 10       # Minimum days-to-expiration when selecting
OPTIONS_DTE_MAX             = 30       # Maximum days-to-expiration when selecting
OPTIONS_CLOSE_BEFORE_EXPIRY_DAYS = 3   # Force-close positions this close to expiry
OPTIONS_CIRCUIT_BREAKER_PCT = -0.20    # -20% of premium: hard stop (options are volatile)
OPTIONS_PROFIT_TARGET_PCT   =  1.00    # +100% of premium: let a winning option run further

# ── P&L log path (read by dashboard.py) ──────────────────────────────
PNL_LOG_PATH = Path("pnl_log.json")

# ── Crash resilience ──────────────────────────────────────────────────
# Unattended for a week straight — a single uncaught exception (a network
# blip, an API hiccup) must NOT kill the whole process. The main loop below
# is wrapped so it logs the error and keeps going instead.
LOG_PATH             = Path("logs/bot.log")
CRASH_COOLDOWN_SECS  = 60   # pause after a caught loop-level exception before retrying


class _Tee:
    """Mirrors writes to multiple streams (console + a persistent log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# =====================================================================
# SESSION STATE
# =====================================================================

banned_tickers:    list[str]   = []
signal_history:    list[dict]  = []
loop_counter:      int         = 0
current_watchlist: list[str]   = []
session_pnl:       float       = 0.0

# Trailing stop state: { symbol: { "high_water": float, "stop": float } }
trail_state:       dict        = {}

# Position entry tracking: { symbol: { "loop": int, "entry_price": float } }
position_meta:     dict        = {}

# Earnings cache: { symbol: date_str | None }
earnings_cache:    dict        = {}

# Market regime: True = bullish / sideways (allow buys), False = downtrend (block buys)
market_regime_bullish: bool    = True


# =====================================================================
# P&L LOGGER  (for dashboard cumulative chart)
# =====================================================================

def _append_pnl_log(realised: float, reason: str) -> None:
    """
    Appends a timestamped P&L event to pnl_log.json.
    dashboard.py reads this to draw a cumulative P&L curve.
    """
    try:
        log = []
        if PNL_LOG_PATH.exists():
            log = json.loads(PNL_LOG_PATH.read_text())
        log.append({
            "ts":       datetime.now().isoformat(timespec="seconds"),
            "pnl":      round(realised, 4),
            "reason":   reason,
            "cumulative": round(sum(e["pnl"] for e in log) + realised, 4),
        })
        PNL_LOG_PATH.write_text(json.dumps(log, indent=2))
    except Exception:
        pass


# =====================================================================
# ADAPTIVE SLEEP
# =====================================================================

def compute_adaptive_sleep(recent_signals: list[dict]) -> int:
    if not recent_signals:
        return SLEEP_MAX
    actionable  = [s for s in recent_signals if s.get("signal") in ("BUY", "SELL")]
    action_rate = len(actionable) / max(len(recent_signals), 1)
    avg_conf    = (
        sum(s.get("adjusted", 0) for s in actionable) / len(actionable)
        if actionable else 0.0
    )
    activity = 0.6 * action_rate + 0.4 * avg_conf
    sleep    = int(SLEEP_MAX - activity * (SLEEP_MAX - SLEEP_MIN))
    return max(SLEEP_MIN, min(SLEEP_MAX, sleep))


# =====================================================================
# 1. TRAILING STOP LOSS
# =====================================================================

def update_trailing_stops() -> None:
    """
    For every open position:
      • Once UPL ≥ TRAIL_ACTIVATE_USD, calculate a stop price =
        current_price × (1 - TRAIL_STOP_PCT)
      • If the stop has moved UP (never allowed to move down), save it
      • If current price has fallen BELOW the stop, close the position
    """
    print("\n📈 Trailing stop check...")
    try:
        positions = broker.trading_client.get_all_positions()
    except Exception as e:
        print(f"⚠️  Could not fetch positions: {e}")
        return

    if not positions:
        print("   └─ No open positions.")
        return

    global session_pnl
    for pos in positions:
        symbol = pos.symbol
        if pos.asset_class == AssetClass.US_OPTION:
            continue   # Options use their own profit-target/circuit-breaker gates instead
        try:
            upl           = float(pos.unrealized_pl)
            current_price = float(pos.current_price)
            avg_entry     = float(pos.avg_entry_price)
        except (TypeError, ValueError):
            continue

        # Only start trailing once the position is profitable enough
        if upl < TRAIL_ACTIVATE_USD:
            continue

        # Initialise trail state
        if symbol not in trail_state:
            trail_state[symbol] = {
                "high_water": current_price,
                "stop":       current_price * (1 - TRAIL_STOP_PCT),
            }

        ts = trail_state[symbol]

        # Ratchet the high-water mark up (never down)
        if current_price > ts["high_water"]:
            ts["high_water"] = current_price
            ts["stop"]       = current_price * (1 - TRAIL_STOP_PCT)
            print(f"   └─ 🔺 {symbol}: trail stop raised → ${ts['stop']:.2f} "
                  f"(high=${ts['high_water']:.2f})")

        # Trigger: price dropped below stop
        if current_price <= ts["stop"]:
            print(f"🛑 TRAIL STOP HIT — {symbol} | price=${current_price:.2f} "
                  f"≤ stop=${ts['stop']:.2f} | UPL=${upl:+.2f}")
            realised = broker.close_position_with_pnl(symbol, reason="trailing stop")
            session_pnl += realised
            _append_pnl_log(realised, f"trail stop {symbol}")
            trail_state.pop(symbol, None)
            position_meta.pop(symbol, None)
            print(f"   └─ Session P&L: ${session_pnl:+.2f}")
        else:
            dist_pct = (current_price - ts["stop"]) / current_price * 100
            print(f"   └─ 🟢 {symbol}: price=${current_price:.2f} | "
                  f"stop=${ts['stop']:.2f} ({dist_pct:.1f}% buffer) | UPL=${upl:+.2f}")


# =====================================================================
# 2. TIME-BASED EXIT (stale position cleanup)
# =====================================================================

def check_stale_positions() -> None:
    """
    Closes positions that have been held for ≥ STALE_POSITION_LOOPS loops
    without meaningful price movement (< STALE_MOVE_THRESHOLD from entry).
    These are dead-weight positions locking up capital.
    """
    print("\n⏳ Stale position check...")
    try:
        positions = broker.trading_client.get_all_positions()
    except Exception as e:
        print(f"⚠️  Could not fetch positions: {e}")
        return

    if not positions:
        print("   └─ No open positions.")
        return

    global session_pnl
    for pos in positions:
        symbol = pos.symbol
        if pos.asset_class == AssetClass.US_OPTION:
            continue   # Options are force-closed by check_expiring_options() instead
        meta   = position_meta.get(symbol)
        if meta is None:
            # Register entry if we don't have it yet
            try:
                position_meta[symbol] = {
                    "loop":        loop_counter,
                    "entry_price": float(pos.avg_entry_price),
                }
            except Exception:
                pass
            continue

        loops_held    = loop_counter - meta["loop"]
        entry_price   = meta["entry_price"]
        current_price = float(pos.current_price)
        move_pct      = abs(current_price - entry_price) / entry_price

        if loops_held >= STALE_POSITION_LOOPS and move_pct < STALE_MOVE_THRESHOLD:
            upl = float(pos.unrealized_pl)
            print(f"⏰ STALE EXIT — {symbol} | held {loops_held} loops | "
                  f"move={move_pct:.1%} < {STALE_MOVE_THRESHOLD:.0%} | UPL=${upl:+.2f}")
            realised = broker.close_position_with_pnl(symbol, reason="stale exit")
            session_pnl += realised
            _append_pnl_log(realised, f"stale exit {symbol}")
            trail_state.pop(symbol, None)
            position_meta.pop(symbol, None)
        else:
            remaining = max(0, STALE_POSITION_LOOPS - loops_held)
            print(f"   └─ ⏱  {symbol}: {loops_held} loops held | "
                  f"move={move_pct:.1%} | stale in {remaining} loops")


# =====================================================================
# 3. MARKET REGIME FILTER  (SPY + QQQ)
# =====================================================================

def update_market_regime() -> bool:
    """
    Checks SPY and QQQ using fast/slow EMA on daily closes.
    Returns True (bullish / allow BUYs) or False (downtrend / block BUYs).

    Logic: regime is bearish only when BOTH SPY AND QQQ have their fast
    EMA below their slow EMA.  One index in a downtrend is not enough
    to suppress all trades — it has to be a broad market move.
    """
    global market_regime_bullish
    bearish_count = 0

    for ticker in REGIME_TICKERS:
        try:
            df = broker.get_historical_market_data(ticker, lookback_days=60)
            if df is None or len(df) < REGIME_EMA_SLOW + 5:
                continue
            cl         = df["close"]
            ema_fast   = cl.ewm(span=REGIME_EMA_FAST, adjust=False).mean()
            ema_slow   = cl.ewm(span=REGIME_EMA_SLOW, adjust=False).mean()
            is_bearish = ema_fast.iloc[-1] < ema_slow.iloc[-1]
            icon       = "🔴" if is_bearish else "🟢"
            print(f"   └─ {icon} {ticker}: EMA{REGIME_EMA_FAST}={ema_fast.iloc[-1]:.2f} "
                  f"{'<' if is_bearish else '>'} EMA{REGIME_EMA_SLOW}={ema_slow.iloc[-1]:.2f}")
            if is_bearish:
                bearish_count += 1
        except Exception as e:
            print(f"   └─ ⚠️  {ticker} regime check failed: {e}")

    # Require BOTH to be bearish before suppressing
    market_regime_bullish = bearish_count < len(REGIME_TICKERS)

    status = "🟢 BULLISH (buys allowed)" if market_regime_bullish else "🔴 BEARISH (buys suppressed)"
    print(f"\n🌍 Market regime: {status}")
    return market_regime_bullish


# =====================================================================
# 4. EARNINGS BLACKOUT
# =====================================================================

def _has_earnings_soon(symbol: str) -> bool:
    """
    Returns True if this ticker has an earnings date within
    EARNINGS_BLACKOUT_DAYS calendar days (before OR after today).
    Results are cached per-session to avoid redundant API calls.
    """
    if symbol in earnings_cache:
        cached = earnings_cache[symbol]
        if cached is None:
            return False
        try:
            earn_date = datetime.strptime(cached, "%Y-%m-%d").date()
            delta     = abs((earn_date - datetime.now().date()).days)
            return delta <= EARNINGS_BLACKOUT_DAYS
        except Exception:
            return False

    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            earnings_cache[symbol] = None
            return False

        # yfinance calendar index contains "Earnings Date" as a row
        if "Earnings Date" in cal.index:
            raw = cal.loc["Earnings Date"]
            # May return a single value or a Series
            earn_date = pd.to_datetime(raw.iloc[0] if hasattr(raw, "iloc") else raw).date()
        else:
            earnings_cache[symbol] = None
            return False

        earnings_cache[symbol] = earn_date.isoformat()
        delta = abs((earn_date - datetime.now().date()).days)
        return delta <= EARNINGS_BLACKOUT_DAYS

    except Exception:
        earnings_cache[symbol] = None
        return False


import pandas as pd   # needed for earnings check above


# =====================================================================
# 5. MULTI-TIMEFRAME CONFIRMATION  (hourly EMA trend)
# =====================================================================

def _hourly_trend_bullish(symbol: str) -> bool:
    """
    Fetches ~10 days of 1-hour bars for `symbol` and returns True if
    the fast EMA is above the slow EMA on the most recent hourly bar.

    A BUY signal on the daily timeframe is only acted on when the
    hourly trend is ALSO bullish — this prevents buying into exhaustion
    at the end of daily moves.

    Returns True (allow) on any data / API error so it never silently
    blocks all trades due to a connectivity issue.
    """
    if not MTF_ENABLED:
        return True

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        end   = datetime.now()
        start = end - timedelta(days=10)

        req  = broker.data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
        )
        df = req.df
        if df is None or df.empty:
            return True

        if isinstance(df.index, pd.MultiIndex):
            if symbol in df.index.get_level_values(0):
                df = df.xs(symbol, level=0)
            else:
                return True

        if len(df) < MTF_EMA_SLOW + 5:
            return True

        cl       = df["close"]
        ema_fast = cl.ewm(span=MTF_EMA_FAST, adjust=False).mean()
        ema_slow = cl.ewm(span=MTF_EMA_SLOW, adjust=False).mean()
        return bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])

    except Exception:
        return True   # Fail open — don't block on data errors


# =====================================================================
# PROFIT-TARGET CHECKER
# =====================================================================

def check_profit_targets() -> None:
    print("\n💰 Profit-target check...")
    try:
        positions = broker.trading_client.get_all_positions()
    except Exception as e:
        print(f"⚠️  Could not fetch positions: {e}")
        return

    if not positions:
        print("   └─ No open positions.")
        return

    global session_pnl
    for pos in positions:
        symbol = pos.symbol
        try:
            upl        = float(pos.unrealized_pl)
            cost_basis = float(pos.cost_basis)
        except (TypeError, ValueError):
            continue
        if cost_basis <= 0:
            continue
        upl_pct = upl / cost_basis
        if pos.asset_class == AssetClass.US_OPTION:
            target = OPTIONS_PROFIT_TARGET_PCT
        else:
            _, target = _atr_scaled_thresholds(symbol)

        if upl_pct >= target:
            print(f"🎯 PROFIT TARGET — {symbol} | UPL: ${upl:.2f} ({upl_pct:+.1%})")
            realised = broker.close_position_with_pnl(symbol, reason="profit target")
            session_pnl += realised
            _append_pnl_log(realised, f"profit target {symbol}")
            trail_state.pop(symbol, None)
            position_meta.pop(symbol, None)
            print(f"   └─ Session P&L: ${session_pnl:+.2f}")


# =====================================================================
# OPTIONS EXPIRY CHECK  (time-based exit for the options sleeve)
# =====================================================================

def check_expiring_options() -> None:
    """
    Force-closes any option position within OPTIONS_CLOSE_BEFORE_EXPIRY_DAYS
    of expiration, regardless of P&L — riding a long option into expiry risks
    a total premium loss (if OTM) or an unintended exercise/assignment (if
    ITM), and this bot has no logic to handle either outcome.
    """
    print("\n📅 Options expiry check...")
    expiring = broker.get_expiring_option_positions(OPTIONS_CLOSE_BEFORE_EXPIRY_DAYS)
    if not expiring:
        print("   └─ No option positions nearing expiry.")
        return

    global session_pnl
    for pos in expiring:
        symbol = pos.symbol
        upl    = float(pos.unrealized_pl) if pos.unrealized_pl else 0.0
        print(f"⏰ EXPIRY CLOSE-OUT — {symbol} | UPL: ${upl:+.2f}")
        realised = broker.close_position_with_pnl(symbol, reason="expiry close-out")
        session_pnl += realised
        _append_pnl_log(realised, f"expiry close-out {symbol}")
        print(f"   └─ Session P&L: ${session_pnl:+.2f}")


# =====================================================================
# CIRCUIT BREAKER
# =====================================================================

def audit_positions_and_enforce_circuit_breaker() -> None:
    print("\n🔍 Circuit-breaker audit...")
    try:
        positions = broker.trading_client.get_all_positions()
    except Exception as e:
        print(f"⚠️  Could not fetch positions: {e}")
        return

    if not positions:
        print("   └─ No open positions.")
        return

    global session_pnl
    for pos in positions:
        symbol = pos.symbol
        try:
            upl        = float(pos.unrealized_pl)
            cost_basis = float(pos.cost_basis)
        except (TypeError, ValueError):
            continue
        if cost_basis <= 0:
            continue
        upl_pct = upl / cost_basis
        if pos.asset_class == AssetClass.US_OPTION:
            threshold = OPTIONS_CIRCUIT_BREAKER_PCT
        else:
            threshold, _ = _atr_scaled_thresholds(symbol)

        if upl_pct <= threshold:
            print(f"🚨 CIRCUIT BREAKER — {symbol} | UPL: ${upl:.2f} ({upl_pct:+.1%})")

            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                open_orders = broker.trading_client.get_orders(
                    filter=GetOrdersRequest(
                        status=QueryOrderStatus.OPEN, symbols=[symbol]
                    )
                )
                cancelled = 0
                for order in open_orders:
                    try:
                        broker.trading_client.cancel_order_by_id(order.id)
                        cancelled += 1
                    except Exception:
                        pass
                if cancelled:
                    print(f"   └─ 🗑  Cancelled {cancelled} open order(s)")
                    time.sleep(1.0)
            except Exception as e:
                print(f"   └─ ⚠️  Could not cancel orders: {e}")

            realised = broker.close_position_with_pnl(symbol, reason="circuit breaker")
            session_pnl += realised
            _append_pnl_log(realised, f"circuit breaker {symbol}")
            trail_state.pop(symbol, None)
            position_meta.pop(symbol, None)

            if symbol not in banned_tickers:
                banned_tickers.append(symbol)
                print(f"   └─ 🔒 {symbol} banned for this session")
        else:
            icon = "🟢" if upl >= 0 else "🟡"
            print(f"   └─ {icon} {symbol:<5} UPL: ${upl:>8.2f} ({upl_pct:+.1%})")


# =====================================================================
# MAIN LOOP
# =====================================================================

def run_one_loop() -> None:
    global loop_counter, current_watchlist, signal_history

    loop_counter += 1
    loop_signals: list[dict] = []

    print("\n" + "=" * 65)
    print(f"🔄 Loop #{loop_counter} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Session realised P&L: ${session_pnl:+.2f}")
    print("=" * 65)

    # ── Step 1: Rebuild universe periodically ──────────────────────
    if loop_counter == 1 or loop_counter % UNIVERSE_REFRESH_LOOPS == 0:
        current_watchlist = universe_module.build_universe(verbose=True)
    else:
        remaining = UNIVERSE_REFRESH_LOOPS - (loop_counter % UNIVERSE_REFRESH_LOOPS)
        print(f"📋 Watchlist: {len(current_watchlist)} tickers "
              f"(refreshes in {remaining} loop(s))")

    # ── Step 2: Market regime check ────────────────────────────────
    print("\n🌍 Checking market regime...")
    update_market_regime()

    # ── Step 3: Take profits on winners ────────────────────────────
    check_profit_targets()

    # ── Step 4: Update trailing stops ─────────────────────────────
    update_trailing_stops()

    # ── Step 5: Close stale positions ─────────────────────────────
    check_stale_positions()

    # ── Step 5.5: Close options nearing expiration ─────────────────
    if OPTIONS_ENABLED:
        check_expiring_options()

    # ── Step 6: Enforce circuit breaker ────────────────────────────
    audit_positions_and_enforce_circuit_breaker()

    # ── Step 7: Sync portfolio snapshot ────────────────────────────
    portfolio = broker.get_live_inventory()
    print(f"\n💼 Holdings: {portfolio if portfolio else 'none'}")
    if banned_tickers:
        print(f"🚫 Session bans: {banned_tickers}")

    open_option_positions = 0
    if OPTIONS_ENABLED:
        try:
            open_option_positions = sum(
                1 for p in broker.trading_client.get_all_positions()
                if p.asset_class == AssetClass.US_OPTION
            )
            print(f"📜 Open option positions: {open_option_positions}/{OPTIONS_MAX_CONCURRENT}")
        except Exception:
            open_option_positions = OPTIONS_MAX_CONCURRENT  # fail safe: skip options this loop

    # ── Step 8: Scan every ticker ───────────────────────────────────
    # Warn clearly when running outside market hours — signals are based
    # on yesterday's close so no orders will actually fill until open.
    try:
        import zoneinfo
        et_now = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        mkt_open  = et_now.weekday() < 5 and et_now.hour == 9  and et_now.minute >= 30
        mkt_open  = mkt_open or (et_now.weekday() < 5 and 10 <= et_now.hour <= 15)
        mkt_open  = mkt_open or (et_now.weekday() < 5 and et_now.hour == 16 and et_now.minute == 0)
        if not mkt_open:
            print(f"\n⚠️  US market is CLOSED right now ({et_now.strftime('%H:%M ET %a')}).")
            print("   Volume filters will be relaxed. Orders won't fill until market open.")
    except Exception:
        pass

    print(f"\n📊 Scanning {len(current_watchlist)} tickers...\n")

    buy_count  = 0
    sell_count = 0

    for ticker in current_watchlist:

        if ticker in banned_tickers:
            print(f"🚫 {ticker:<5} — BANNED")
            continue

        try:
            # ── 4. Earnings blackout (equity + options) ─────────────
            # Options were briefly exempted from this on the theory that
            # "defined risk = premium paid" makes an earnings gap fine to
            # hold through. Live experience (a CRM put bought right before
            # its 8/26 earnings gap, -774 at the worst mark) showed the
            # flaw: the model has zero fundamentals edge on earnings
            # surprises, so a position held through one is a coin flip with
            # extra leverage, not informed variance-seeking — it adds noise,
            # not edge. Blocked for both legs now.
            earnings_soon = _has_earnings_soon(ticker)

            # 800d (matches backtest.py) — the 300d default trains the model
            # on ~204 rows instead of ~548, and less training data measurably
            # weakens class separation: a direct live-vs-backtest comparison
            # on the same day showed 300d giving buy/sell probabilities
            # roughly a third of what 800d gave for the same ticker, which
            # meant almost nothing ever cleared the probability gate.
            market_df = broker.get_historical_market_data(ticker, lookback_days=800)
            if market_df is None or len(market_df) < 60:
                bar_count = len(market_df) if market_df is not None else 0
                print(f"⚠️  {ticker:<5} — skipped (only {bar_count} bars)")
                time.sleep(TICKER_SLEEP)
                continue

            live_price   = float(market_df["close"].iloc[-1])
            processed_df = strategy.calculate_indicators(market_df)

            atr_pct = None
            if "ATR_Pct" in processed_df.columns:
                raw_atr = processed_df["ATR_Pct"].iloc[-1]
                if not math.isnan(raw_atr):
                    atr_pct = float(raw_atr)

            result     = strategy.generate_ensemble_signal(
                processed_df, symbol=ticker, loop_number=loop_counter
            )
            signal     = result["signal"]
            confidence = result["confidence"]
            adjusted   = result["adjusted"]
            buy_prob   = result.get("buy_prob",  0.0)
            sell_prob  = result.get("sell_prob", 0.0)
            regime_ok  = result.get("regime_ok", True)

            candle_sc = 0
            if "CANDLE_SCORE" in processed_df.columns:
                v = processed_df["CANDLE_SCORE"].iloc[-1]
                if not math.isnan(v):
                    candle_sc = int(v)

            print(
                f"📡 {ticker:<5} | ${live_price:>8.2f} "
                f"| {signal:<4} | B={buy_prob:.0%} S={sell_prob:.0%} "
                f"| conf={confidence:.0%} adj={adjusted:.0%} "
                f"| 🕯{candle_sc:+d}"
            )

            # ── Model regime block ────────────────────────────────
            if not regime_ok:
                print(f"   └─ 🌫  {result['reason']}")
                time.sleep(TICKER_SLEEP)
                print("-" * 50)
                continue

            # ── Signal from class probabilities ────────────────────
            # Fire BUY/SELL off buy_prob/sell_prob directly (see config
            # comment above) instead of trusting the raw argmax, which is
            # HOLD-dominated by the deadband. This REPLACES `signal`.
            if buy_prob >= MIN_BUY_PROB and buy_prob > sell_prob:
                signal = "BUY"
            elif sell_prob >= MIN_SELL_PROB and sell_prob > buy_prob:
                signal = "SELL"
            else:
                signal = "HOLD"

            # ── 3. Market regime suppression ──────────────────────
            if signal == "BUY" and not market_regime_bullish:
                print(f"   └─ 🔴 BUY suppressed: broad market in downtrend")
                signal = "HOLD"

            # ── 5. Multi-timeframe confirmation ───────────────────
            if signal == "BUY":
                hourly_ok = _hourly_trend_bullish(ticker)
                if not hourly_ok:
                    print(f"   └─ ⏱  BUY blocked: hourly EMA trend is bearish (MTF filter)")
                    signal = "HOLD"
                else:
                    print(f"   └─ ✅ MTF confirmed: hourly trend bullish")

            # Options express the directional view via calls/puts, so a
            # SELL (bearish) view doesn't require already owning shares —
            # capture it here, before the equity-only "must hold" gate below.
            options_signal = signal

            # Only SELL if we hold it
            if signal == "SELL" and ticker not in portfolio:
                print(f"   └─ 📭 SELL ignored: not holding {ticker}")
                signal = "HOLD"

            if earnings_soon and signal != "HOLD":
                print(f"   └─ 📅 {signal} suppressed for equity — earnings within {EARNINGS_BLACKOUT_DAYS}d "
                      f"(options still eligible)")
                signal = "HOLD"

            loop_signals.append({"signal": signal, "adjusted": adjusted})

            if signal == "BUY":
                buy_count += 1
                # Register entry metadata for stale-position tracking and
                # ATR-scaled stops (see CIRCUIT_BREAKER_LOSS_PCT usage below)
                if ticker not in position_meta:
                    position_meta[ticker] = {
                        "loop":        loop_counter,
                        "entry_price": live_price,
                        "atr_pct":     atr_pct,
                    }
            elif signal == "SELL":
                sell_count += 1
                trail_state.pop(ticker, None)
                position_meta.pop(ticker, None)

            # The probability of whichever class was actually acted on —
            # feeds position sizing (see CONF_FLOOR/CONF_CEIL in broker.py)
            action_prob = buy_prob if signal == "BUY" else sell_prob if signal == "SELL" else confidence

            broker.execute_calculated_trade(
                ticker, live_price, signal,
                confidence=action_prob,
                atr_pct=atr_pct,
            )

            # ── Options sleeve: same signal, expressed via calls/puts ──
            if earnings_soon and options_signal != "HOLD":
                print(f"   └─ 📅 {options_signal} suppressed for options too — earnings within "
                      f"{EARNINGS_BLACKOUT_DAYS}d (a defined-risk premium still eats a full loss "
                      f"on a surprise gap, and the model has no fundamentals edge on earnings)")
                options_signal = "HOLD"

            options_action_prob = buy_prob if options_signal == "BUY" else sell_prob
            if (OPTIONS_ENABLED and options_signal in ("BUY", "SELL")
                    and options_action_prob >= OPTIONS_MIN_PROB
                    and open_option_positions < OPTIONS_MAX_CONCURRENT):
                right   = ContractType.CALL if options_signal == "BUY" else ContractType.PUT
                equity  = float(broker.trading_client.get_account().equity)
                o_budget = equity * OPTIONS_BUDGET_FRACTION
                if broker.execute_option_trade(
                    ticker, live_price, right, o_budget,
                    otm_pct=OPTIONS_OTM_PCT,
                    dte_min=OPTIONS_DTE_MIN, dte_max=OPTIONS_DTE_MAX,
                ):
                    open_option_positions += 1

        except Exception as e:
            print(f"❌ {ticker}: {e}")

        time.sleep(TICKER_SLEEP)
        print("-" * 50)

    # ── Step 9: Write P&L snapshot for dashboard ──────────────────
    try:
        snap = {
            "ts":        datetime.now().isoformat(timespec="seconds"),
            "loop":      loop_counter,
            "session_pnl": round(session_pnl, 4),
            "holdings":  list(portfolio.keys()),
        }
        Path("session_snapshot.json").write_text(json.dumps(snap, indent=2))
    except Exception:
        pass

    # ── Step 10: Adaptive sleep ─────────────────────────────────────
    signal_history.extend(loop_signals)
    signal_history = signal_history[-200:]

    sleep_secs = compute_adaptive_sleep(loop_signals)
    print(
        f"\n✅ Scan complete — "
        f"BUY={buy_count} SELL={sell_count} "
        f"| Session P&L: ${session_pnl:+.2f} "
        f"| sleeping {sleep_secs}s"
    )
    time.sleep(sleep_secs)


if __name__ == "__main__":
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    print("🤖 Autonomous Trading Bot v3 initialising...")
    print(f"   Circuit breaker      : ATR-scaled ({ATR_STOP_MULT}x, clamped {ATR_STOP_FLOOR:.0%}..{ATR_STOP_CEIL:.0%}), "
          f"fallback {CIRCUIT_BREAKER_LOSS_PCT:.1%}")
    print(f"   Profit target        : ATR-scaled ({ATR_TARGET_MULT}x, clamped {ATR_TARGET_FLOOR:.0%}..{ATR_TARGET_CEIL:.0%}), "
          f"fallback +{PROFIT_TAKE_TARGET_PCT:.1%}")
    print(f"   Trailing stop        : activates at +${TRAIL_ACTIVATE_USD:.2f}, trails {TRAIL_STOP_PCT:.1%}")
    print(f"   Stale exit           : {STALE_POSITION_LOOPS} loops, <{STALE_MOVE_THRESHOLD:.0%} move")
    print(f"   Regime tickers       : {', '.join(REGIME_TICKERS)}")
    print(f"   Earnings blackout    : ±{EARNINGS_BLACKOUT_DAYS} days (equity + options)")
    print(f"   Multi-timeframe      : {'enabled' if MTF_ENABLED else 'disabled'}")
    print(f"   Buy/Sell prob gate   : {MIN_BUY_PROB:.0%} / {MIN_SELL_PROB:.0%}")
    print(f"   Equity sizing        : {broker.BASE_FRACTION:.0%}-{broker.BASE_FRACTION+broker.CONF_BOOST_CAP:.0%} of cash, "
          f"max {broker.MAX_EQUITY_POSITIONS} positions, ${broker.MAX_POSITION_VALUE:,.0f} cap")
    print(f"   Universe cap         : {universe_module.MAX_UNIVERSE_SIZE} tickers")
    if OPTIONS_ENABLED:
        print(f"   Options sleeve       : {OPTIONS_BUDGET_FRACTION:.1%} equity/trade, "
              f"max {OPTIONS_MAX_CONCURRENT} concurrent, {OPTIONS_MIN_PROB:.0%} prob gate, "
              f"{OPTIONS_OTM_PCT:.0%} OTM, {OPTIONS_DTE_MIN}-{OPTIONS_DTE_MAX}d expiry")
        print(f"   Options risk gates   : {OPTIONS_CIRCUIT_BREAKER_PCT:.0%} stop / "
              f"+{OPTIONS_PROFIT_TARGET_PCT:.0%} target / close {OPTIONS_CLOSE_BEFORE_EXPIRY_DAYS}d before expiry")
    else:
        print("   Options sleeve       : disabled")

    while True:
        try:
            run_one_loop()
        except Exception:
            print("\n\U0001F525 UNCAUGHT LOOP ERROR:\n" + traceback.format_exc())
            print(f"   \u2514\u2500 Cooling down {CRASH_COOLDOWN_SECS}s before retrying...")
            time.sleep(CRASH_COOLDOWN_SECS)
