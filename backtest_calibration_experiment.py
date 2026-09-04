"""
backtest_calibration_experiment.py — head-to-head backtest of the current
uncalibrated ensemble vs. a calibrated variant (CalibratedClassifierCV with
a chronological 80/20 fit/calibration split — see strategy.CALIBRATE_MODE).

Reuses backtest.py's exact run_backtest()/summarize() logic unmodified.
Never touches main.py — the live bot is unaffected regardless of outcome.
"""
import json
import warnings

warnings.filterwarnings("ignore")
warnings.warn = lambda *a, **k: None

import strategy
from backtest import run_backtest, summarize

results = {}

for label, calibrate in [("uncalibrated", False), ("calibrated", True)]:
    print(f"\n{'#'*60}\n# Running backtest — {label}\n{'#'*60}")
    strategy.CALIBRATE_MODE = calibrate
    strategy._model_cache.clear()
    strategy._model_cache_calibrated.clear()

    equity_curve, trade_log, cash, positions, spy_buy_hold_pct = run_backtest()
    summary = summarize(equity_curve, trade_log, cash, positions, spy_buy_hold_pct)

    out = {"summary": summary, "equity_curve": equity_curve, "trade_log": trade_log}
    fname = f"backtest_results_{label}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {fname}")
    results[label] = summary

print(f"\n{'='*60}\nCALIBRATION EXPERIMENT COMPLETE\n{'='*60}")
for label, summary in results.items():
    print(f"  {label:14}: return={summary['total_return_pct']:+.2f}% "
          f"vs SPY={summary['spy_buy_hold_pct']:+.2f}% | "
          f"trades={summary['closed_trades']} | win_rate={summary['win_rate_pct']:.1f}% | "
          f"max_dd={summary['max_drawdown_pct']:.2f}%")
