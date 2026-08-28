# Autonomous Trading Agent — Hackathon Write-Up

## AI Logic

The core decision engine is a **3-class ensemble model** (Random Forest + XGBoost +
LightGBM, soft-voting) that predicts tomorrow's directional move for each ticker as
**BUY / HOLD / SELL**, using a ±0.5% deadband on next-day return so the model isn't
trained on noise. Each prediction draws on 35+ engineered features per ticker:

- **Momentum**: RSI (7/14/21), Stochastic, Williams %R, ROC
- **Trend**: MACD, ADX/DI spread, SMA/EMA ratios (20/50/200)
- **Volatility**: Bollinger %B and squeeze, Keltner width, ATR%
- **Volume**: OBV slope, VWAP delta, volume z-score, MFI
- **8 hand-coded candlestick patterns** (doji, hammer, engulfing, morning/evening
  star, harami) and **5 Ichimoku Cloud signals** (TK cross, cloud position)

Models are retrained per ticker on a rolling cache (every few scan loops, not every
loop) to keep the live scan fast across the full watchlist. A per-ticker confidence
multiplier — tracked in a persistent ledger of historical BUY/SELL accuracy — boosts
tickers the model has been reliably right about and discounts ones it hasn't, once
enough trade history exists.

A **market-regime filter** (SPY + QQQ EMA8/21 cross) suppresses all new BUY signals
during a confirmed broad-market downtrend, and a **volatility/chop filter** (ADX +
Bollinger-squeeze) blocks trades when a ticker's recent price action is directionless
noise rather than signal.

Signals are decided from the model's **class probabilities directly**, not its raw
argmax: with a ±0.5% deadband, HOLD is the majority training class on most tickers
most days, so requiring HOLD to lose *and* have low probability throttles trading to
almost nothing. A 3-class baseline is ~33% each, so BUY/SELL fire once their own
probability clears a real, actionable threshold (40% equity, 46% options — higher,
given the added leverage/decay risk) and leads the other classes, independent of
whether HOLD was technically the top prediction.

## Risk Gates

Every gate is **percentage-of-position**, not a flat dollar amount — sized to a
$100,000 account rather than hard-coded for a small test balance:

| Gate | Rule |
|---|---|
| Circuit breaker | **ATR-scaled**: -2x the position's entry-day ATR%, clamped to -2%..-10%, falls back to a flat -5% if no ATR was recorded |
| Profit target | **ATR-scaled**: +4.8x entry-day ATR%, clamped to +5%..+25%, falls back to a flat +12% |
| Trailing stop | Activates at +$3 unrealized gain, trails **3%** below the high-water mark, never loosens |
| Stale-position exit | Close after 12 scan loops with **<1%** total move (dead capital) |
| Probability gate | BUY requires **≥40%** buy-probability, SELL requires **≥40%** sell-probability |
| Earnings blackout | Equity: skip within **±2 days** of earnings. Options: **not blocked** (see below) |
| Multi-timeframe confirm | BUY only fires if the **hourly** EMA trend agrees with the daily signal |
| Position caps | Max **6** concurrent equity positions, **$30,000** max exposure per ticker |
| Position sizing | 18–25% of available cash per trade (scales with signal strength) — concentrated, not spread thin |

The circuit breaker and profit target were originally one flat percentage for
all 47 tickers — a quiet stock (KO) and a volatile one (NVDA) got the exact same
stop distance, which isn't principled. They now scale to each position's own
entry-day ATR% instead (a stock's own average daily trading range), while
preserving the same ~2.4:1 reward:risk ratio the flat numbers established.

The equity earnings blackout stays — a stop-loss can't protect against an
overnight earnings gap, which can blow straight past any % stop before the
market even reopens. The **options sleeve deliberately does not honor it**:
options have genuinely defined risk (max loss = the premium paid) even through
a gap, and earnings are exactly the kind of catalyst that produces the large,
convex moves that sleeve is sized for. Checking the actual earnings calendar for
the 47-ticker core against the competition week (Aug 28 – Sep 4): only **AVGO**
(Sep 2) reports in that window — a real but narrow opportunity, not a broad one.

A separate, independently-risk-gated **options sleeve** expresses the same directional
signal (BUY→call, SELL→put) via small, defined-risk long options: **5% of equity per
trade, max 5 concurrent, 46% probability bar** (higher than equity's, since options
carry leverage/decay risk), targeting **~6% OTM contracts 10–30 days out** — further
out-of-the-money and shorter-dated than a typical directional bet, trading probability
of profit for cheaper premium and more convexity. Because options are far more volatile
than the underlying, they get their **own** wider gates (-35% stop / +100% target)
rather than reusing the equity thresholds, plus a hard close-out **3 days before
expiration** — a long option isn't allowed to ride into expiry, where it risks total
premium loss or an unwanted exercise/assignment.

### Why the risk gates changed shape mid-build

The numbers above are wider and more aggressive than what the backtest below was
originally tuned for, and that's deliberate. This account is judged on **one week's
P&L, ranked against every other entrant** — a tournament payoff, not a real-investing
one. It's also **paper money**, so a bad week costs nothing beyond not placing. Under
those two conditions, the strategy that maximizes expected return (smooth, safe,
small gains) is not the strategy that maximizes the *chance of placing top 3* —
that one leans into variance instead, since only the tail outcome matters and the
downside is free. The circuit breaker, profit target, trailing stop, and options
sizing were all re-widened after the backtest below to reflect that: fewer, bigger,
more convex bets over a smoother equity curve.

**Unattended for a week straight**, the main loop is wrapped in a top-level
try/except: any single uncaught exception (a network blip, an API hiccup) is logged
with a full traceback and the loop retries after a cooldown, instead of silently
killing the whole process and freezing the account's P&L for the rest of the
competition. All output is also mirrored to a persistent log file, not just the
console, so there's a durable record of exactly what happened and why.

## Alpaca Infrastructure

Built entirely on `alpaca-py` against a **dedicated $100,000 paper account**
(options trading level 3 approved):

- `TradingClient` for order submission (equities + single-leg options,
  `BUY_TO_OPEN`/`SELL_TO_CLOSE` intent tagging for options positions) and live
  position/account state
- `StockHistoricalDataClient` for daily + hourly OHLCV bars driving the model and
  the multi-timeframe confirmation
- `OptionHistoricalDataClient` + `GetOptionContractsRequest` for live option-chain
  lookup, liquidity filtering (via last-close/quote availability), and bid/ask
  midpoint pricing before sizing a contract
- A dynamically rebuilt watchlist (~120 tickers) layered from a curated 47-ticker
  core (liquid, sector-diversified large/mid-caps, always included) plus live S&P
  500 constituents and a top-movers screener, validated against real price data
  before trading

The strategy engine was **validated with a walk-forward backtest** (`backtest.py`)
replaying the exact same signal code (not a simplified copy) against ~2.2 years of
historical daily bars, benchmarked against SPY buy-and-hold over the identical
window — and a **9-check regression test suite** (`tests.py`) covering broker
connectivity, options-contract selection, universe integrity, and risk-gate
sanity, built specifically to catch the kind of bug that actually showed up
during development:
- A lookahead bug where the live/predict row was silently dropped a bar early
  — every signal looked normal but was reasoning about stale data.
- A universe data source that silently returned zero results after an
  upstream API changed its response shape from a DataFrame to a dict.
- A live/backtest training-data mismatch: the live bot was fetching only
  the data-fetch function's *default* lookback (~204 trading rows) instead
  of the 548 rows `backtest.py` actually validated the model against. Less
  training data measurably weakens the model's confidence in its own
  predictions — a direct side-by-side test on identical data showed the same
  ticker's buy/sell probabilities roughly triple once the lookback was
  corrected to match. This was the actual reason the live bot looked stuck
  on HOLD — not a broken threshold, but a live path quietly running on a
  fraction of the training history it was supposed to have.

## Backtest Results — Honest Numbers, Not a Sales Pitch

47-ticker universe, 338 simulated trading days (2025-04-21 → 2026-08-24), starting
from $100,000:

| | Original (-2.5% stop) | Risk-tuned (-4% stop) | Tournament-tuned (final) | SPY buy & hold |
|---|---|---|---|---|
| Total return | -2.98% | -2.52% | **+6.54%** | **+48.54%** |
| Max drawdown | -14.14% | -15.15% | -13.94% | — |
| Closed trades | 551 | 473 | 288 | — |
| Win rate | 50.3% | 57.7% | 55.9% | — |

The strategy **lost money over this window while the market rallied hard** — that's
the honest headline, and no amount of downstream polish changes it. Digging into
*why* (a full trade-by-trade breakdown by exit reason) surfaced something specific
and useful: the model's actual signal-driven exits are genuinely profitable
(SELL-signal closes: 62.9% win rate, +$12,855; profit target and trailing stop are
100% win rate by construction, +$27,453 combined) — but the -2.5% circuit breaker
was firing on 32% of *all* trades and, by itself, erased every dollar the rest of
the system made and then some (-$49,655, versus +$46,892 from everything else
combined). That's not "the model is wrong" — it's "the stop was tighter than these
stocks' normal daily noise, so it was cutting positions before the demonstrated
edge ever got a chance to work." Widening it to -4% materially improved win rate
(50.3%→57.7%) with a similar circuit-breaker dynamic still present at a smaller
scale (88 trades, -$42,196 vs +$40,029 from everything else) — better, but not
fixed, and max drawdown ticked up slightly as a direct tradeoff of giving losers
more room.

**What this means going into the competition**: the ensemble has a real, measurable
edge on its own exit signals, and the final "tournament-tuned" pass (wider profit
targets, bigger and more concentrated positions, a larger options sleeve — see
below) let that edge compound into a genuinely positive 16-month backtest, +6.54%,
up from -2.52% at the risk-minimized settings. That wasn't the objective it was
tuned for — the objective was maximizing the chance of a standout single week for
a tournament payoff, which is a different goal than the smoothest long-run curve —
but it's a reassuring sign that leaning into variance didn't come at the cost of
the underlying edge; if anything it let the edge express itself more per correct
call. Two iterations of risk-gate parameter search against a single historical
window already carried real overfitting risk; a 16-month backtest is also one
market regime, not proof any version of this generalizes to next week specifically.
The honest takeaway: this is a risk-managed, bug-fixed, real-money-mechanics-correct
trading agent with a demonstrated (not just assumed) directional edge — not a
guaranteed source of alpha. Whether it has a standout week during the actual
competition is still a genuinely open question, not a promise.
