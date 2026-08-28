"""
backtest.py — Walk-forward equity backtest for the ML signal engine
=====================================================================
Replays historical daily bars through the exact same strategy.py signal
engine (strategy.generate_ensemble_signal) used by the live bot, against a
virtual $100k portfolio, applying the same equity risk gates (circuit
breaker, profit target, trailing stop, stale exit, SELL-signal close) as
main.py. Does NOT touch the live broker — no orders are ever submitted.

This is a validation tool, not a perfect replica of the live bot. A few
things can't be faithfully backtested and are simplified (by design, not
by accident):
  • Daily granularity only. STALE_POSITION_LOOPS means "N simulated
    trading DAYS" here, not "N ~2hr live loops".
  • No multi-timeframe (hourly) confirmation — no historical intraday bars.
  • No earnings blackout — yfinance's calendar only exposes the *next*
    earnings date, not historical ones.
  • No options — options-chain history isn't realistically available, so
    only the equity engine is validated here.
  • Market-regime EMA is computed once over the full series and sliced per
    day (mathematically equivalent for a backward-looking EMA), rather than
    the live bot's approach of re-fetching a fresh 60-day window each loop.
"""

import json
import time
import warnings
import pandas as pd

# strategy.py's own `warnings.filterwarnings("ignore")` doesn't reliably
# suppress the sklearn/joblib UserWarning emitted from inside the RF's
# internal thread pool during ensemble.fit() — filter-list matching across
# threads is unreliable in practice. Neutering warnings.warn() outright is
# the only thing that has actually worked: at ~1 warning per fresh model
# fit x thousands of fits over a full backtest, the unsuppressed version
# writes 500k+ lines to the log and measurably dominates the runtime.
warnings.filterwarnings("ignore")
warnings.warn = lambda *a, **k: None

import broker
import strategy
import main as bot_config          # reuses the live bot's risk-gate constants; safe to
                                    # import — the trading loop is behind __main__ guard
from universe import SEED_TICKERS as TICKERS   # same curated core the live bot always trades

STARTING_CASH       = 100_000.0
LOOKBACK_DAYS        = 800     # ~2.2 calendar years of daily bars
WARMUP_ROWS          = 210     # rows consumed by SMA_200 / ML training before any trading
RETRAIN_EVERY        = 5       # simulated days between model retrains (perf tradeoff)

strategy.CACHE_LOOPS = RETRAIN_EVERY   # this process only — the live bot is a separate process


def fetch_and_prepare(symbol: str):
    raw = broker.get_historical_market_data(symbol, lookback_days=LOOKBACK_DAYS)
    if raw is None or len(raw) < WARMUP_ROWS + 60:
        return None
    return strategy.calculate_indicators(raw)


def regime_series(spy_ind: pd.DataFrame, qqq_ind: pd.DataFrame) -> pd.Series:
    """Bool series indexed like spy_ind: True = bullish/allow buys (mirrors main.py's rule)."""
    def ema_pair(df):
        cl = df["close"]
        fast = cl.ewm(span=bot_config.REGIME_EMA_FAST, adjust=False).mean()
        slow = cl.ewm(span=bot_config.REGIME_EMA_SLOW, adjust=False).mean()
        return fast, slow

    spy_fast, spy_slow = ema_pair(spy_ind)
    qqq_fast, qqq_slow = ema_pair(qqq_ind)
    spy_bear = (spy_fast < spy_slow)
    qqq_bear = (qqq_fast < qqq_slow).reindex(spy_bear.index, method="ffill").fillna(False)
    bearish = spy_bear & qqq_bear   # regime bearish only when BOTH are bearish
    return ~bearish


def run_backtest(tickers=TICKERS, lookback_days=LOOKBACK_DAYS, warmup_rows=WARMUP_ROWS):
    print(f"📥 Fetching {len(tickers)} tickers ({lookback_days}d lookback)...")
    data = {}
    for i, t in enumerate(tickers, 1):
        ind = fetch_and_prepare(t)
        status = f"ok ({len(ind)} rows)" if ind is not None else "skipped (insufficient data)"
        print(f"   [{i}/{len(tickers)}] {t}: {status}")
        if ind is not None:
            data[t] = ind
        time.sleep(0.2)

    print(f"✅ Loaded {len(data)}/{len(tickers)} tickers")

    spy_ind = fetch_and_prepare("SPY")
    qqq_ind = fetch_and_prepare("QQQ")
    bullish = regime_series(spy_ind, qqq_ind)

    sim_days = spy_ind.index[warmup_rows:]
    print(f"🗓  Simulating {len(sim_days)} trading days "
          f"({sim_days[0].date()} → {sim_days[-1].date()})")

    spy_start = float(spy_ind.loc[sim_days[0], "close"])
    spy_end   = float(spy_ind.loc[sim_days[-1], "close"])
    spy_buy_hold_pct = (spy_end - spy_start) / spy_start * 100

    cash          = STARTING_CASH
    positions: dict = {}
    equity_curve   = []
    trade_log      = []

    for day_idx, today in enumerate(sim_days):
        bull = bool(bullish.asof(today))

        for symbol, df in data.items():
            if today not in df.index:
                continue
            window = df.loc[:today]
            if len(window) < warmup_rows:
                continue

            result    = strategy.generate_ensemble_signal(window, symbol=symbol, loop_number=day_idx)
            buy_prob  = result.get("buy_prob", 0.0)
            sell_prob = result.get("sell_prob", 0.0)
            price     = float(window["close"].iloc[-1])

            # Signal from class probabilities directly — mirrors main.py's
            # MIN_BUY_PROB/MIN_SELL_PROB gate, not the model's raw argmax
            # (HOLD-dominated by the deadband; see main.py's config comment).
            if buy_prob >= bot_config.MIN_BUY_PROB and buy_prob > sell_prob:
                signal = "BUY"
            elif sell_prob >= bot_config.MIN_SELL_PROB and sell_prob > buy_prob:
                signal = "SELL"
            else:
                signal = "HOLD"
            action_prob = buy_prob if signal == "BUY" else sell_prob

            if symbol in positions:
                pos        = positions[symbol]
                cost_basis = pos["qty"] * pos["entry_price"]
                upl        = (price - pos["entry_price"]) * pos["qty"]
                upl_pct    = upl / cost_basis if cost_basis else 0.0
                close_reason = None

                if signal == "SELL":
                    close_reason = "SELL signal"
                elif upl_pct >= bot_config.PROFIT_TAKE_TARGET_PCT:
                    close_reason = "profit target"
                elif upl_pct <= bot_config.CIRCUIT_BREAKER_LOSS_PCT:
                    close_reason = "circuit breaker"
                else:
                    if upl >= bot_config.TRAIL_ACTIVATE_USD:
                        if price > pos["high_water"]:
                            pos["high_water"] = price
                            pos["trail_stop"] = price * (1 - bot_config.TRAIL_STOP_PCT)
                        if pos.get("trail_stop") and price <= pos["trail_stop"]:
                            close_reason = "trailing stop"
                    if close_reason is None:
                        days_held = day_idx - pos["entry_day_idx"]
                        move_pct  = abs(price - pos["entry_price"]) / pos["entry_price"]
                        if (days_held >= bot_config.STALE_POSITION_LOOPS
                                and move_pct < bot_config.STALE_MOVE_THRESHOLD):
                            close_reason = "stale exit"

                if close_reason:
                    cash += pos["qty"] * price
                    trade_log.append({
                        "symbol": symbol, "side": "SELL", "day": str(today.date()),
                        "qty": round(pos["qty"], 4), "price": round(price, 2),
                        "pnl": round(upl, 2), "reason": close_reason,
                    })
                    del positions[symbol]

            else:
                if (signal == "BUY" and bull
                        and len(positions) < broker.MAX_EQUITY_POSITIONS):
                    conf_boost = ((action_prob - broker.CONF_FLOOR)
                                  / (broker.CONF_CEIL - broker.CONF_FLOOR) * broker.CONF_BOOST_CAP)
                    fraction   = broker.BASE_FRACTION + max(0.0, min(conf_boost, broker.CONF_BOOST_CAP))
                    budget     = min(cash * fraction * 0.98, broker.MAX_POSITION_VALUE)
                    if 1.0 <= budget <= cash:
                        qty = budget / price
                        cash -= budget
                        positions[symbol] = {
                            "qty": qty, "entry_price": price, "entry_day_idx": day_idx,
                            "high_water": price, "trail_stop": None,
                        }
                        trade_log.append({
                            "symbol": symbol, "side": "BUY", "day": str(today.date()),
                            "qty": round(qty, 4), "price": round(price, 2),
                            "pnl": None, "reason": f"signal (buy_prob={action_prob:.0%})",
                        })

        mtm = cash
        for symbol, pos in positions.items():
            df = data.get(symbol)
            price = float(df.loc[today, "close"]) if (df is not None and today in df.index) else pos["entry_price"]
            mtm += pos["qty"] * price
        equity_curve.append({"day": str(today.date()), "equity": round(mtm, 2)})

        if day_idx % 20 == 0 or day_idx == len(sim_days) - 1:
            print(f"   day {day_idx:>4}/{len(sim_days)} ({today.date()}) | "
                  f"equity=${mtm:>12,.2f} | positions={len(positions)}")

    return equity_curve, trade_log, cash, positions, spy_buy_hold_pct


def summarize(equity_curve, trade_log, cash, positions, spy_buy_hold_pct=None):
    final_equity = equity_curve[-1]["equity"] if equity_curve else STARTING_CASH
    total_return_pct = (final_equity - STARTING_CASH) / STARTING_CASH

    peak = STARTING_CASH
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt["equity"])
        dd = (pt["equity"] - peak) / peak
        max_dd = min(max_dd, dd)

    closed = [t for t in trade_log if t["side"] == "SELL"]
    wins   = [t for t in closed if t["pnl"] > 0]
    win_rate = len(wins) / len(closed) if closed else 0.0

    summary = {
        "starting_cash":    STARTING_CASH,
        "final_equity":     final_equity,
        "total_return_pct": round(total_return_pct * 100, 2),
        "spy_buy_hold_pct": round(spy_buy_hold_pct, 2) if spy_buy_hold_pct is not None else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "closed_trades":    len(closed),
        "win_rate_pct":     round(win_rate * 100, 1),
        "still_open":       len(positions),
        "still_open_symbols": list(positions.keys()),
    }

    print("\n" + "=" * 60)
    print("📊 BACKTEST SUMMARY")
    print("=" * 60)
    print(f"   Starting cash     : ${summary['starting_cash']:,.2f}")
    print(f"   Final equity      : ${summary['final_equity']:,.2f}")
    print(f"   Total return      : {summary['total_return_pct']:+.2f}%")
    if summary["spy_buy_hold_pct"] is not None:
        print(f"   SPY buy & hold    : {summary['spy_buy_hold_pct']:+.2f}%  (same window, benchmark)")
    print(f"   Max drawdown      : {summary['max_drawdown_pct']:.2f}%")
    print(f"   Closed trades     : {summary['closed_trades']}")
    print(f"   Win rate          : {summary['win_rate_pct']:.1f}%")
    print(f"   Still open        : {summary['still_open']} ({summary['still_open_symbols']})")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    equity_curve, trade_log, cash, positions, spy_buy_hold_pct = run_backtest()
    summary = summarize(equity_curve, trade_log, cash, positions, spy_buy_hold_pct)

    out = {
        "summary": summary,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
    }
    with open("backtest_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n💾 Saved backtest_results.json")
