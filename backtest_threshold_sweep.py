"""
backtest_threshold_sweep.py — one-off comparison of MIN_BUY_PROB/MIN_SELL_PROB
thresholds (0.40 baseline vs 0.35 vs 0.30), reusing backtest.py's exact,
already-validated run_backtest()/summarize() logic unmodified. Monkeypatches
bot_config.MIN_BUY_PROB/MIN_SELL_PROB in-process only — never touches
main.py, so the live bot (a separate process) is unaffected.
"""
import json
import warnings

warnings.filterwarnings("ignore")
warnings.warn = lambda *a, **k: None

import backtest
from backtest import run_backtest, summarize
import main as bot_config

THRESHOLDS = [0.35, 0.30]
results = {}

for thresh in THRESHOLDS:
    print(f"\n{'#'*60}\n# Running backtest at threshold {thresh:.0%}\n{'#'*60}")
    bot_config.MIN_BUY_PROB = thresh
    bot_config.MIN_SELL_PROB = thresh

    equity_curve, trade_log, cash, positions, spy_buy_hold_pct = run_backtest()
    summary = summarize(equity_curve, trade_log, cash, positions, spy_buy_hold_pct)

    out = {"summary": summary, "equity_curve": equity_curve, "trade_log": trade_log}
    fname = f"backtest_results_thresh_{int(thresh*100)}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {fname}")
    results[thresh] = summary

print(f"\n{'='*60}\nSWEEP COMPLETE\n{'='*60}")
for thresh, summary in results.items():
    print(f"  {thresh:.0%}: return={summary['total_return_pct']:+.2f}% "
          f"vs SPY={summary['spy_buy_hold_pct']:+.2f}% | "
          f"trades={summary['closed_trades']} | win_rate={summary['win_rate_pct']:.1f}% | "
          f"max_dd={summary['max_drawdown_pct']:.2f}%")
