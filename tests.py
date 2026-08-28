"""
tests.py — Lightweight regression/smoke tests for the trading bot
=====================================================================
Not a full pytest suite — a set of fast, assert-based checks that catch the
exact kinds of bugs this project has actually had (silent 0%-confidence
signals, a broken universe data source, nonsensical risk-gate constants)
before they reach a live trading loop. Run with:

    python tests.py

Exits non-zero if anything fails. Hits real Alpaca/yfinance endpoints (no
mocking) — this is meant to run against the paper account, never live money.
"""

import sys
import traceback

import broker
import strategy
import universe
import main as bot_config


PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"✅ {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"❌ {name} — {e}")
    except Exception:
        FAIL.append((name, traceback.format_exc()))
        print(f"💥 {name} — unexpected error:\n{traceback.format_exc()}")


# ── Broker / account ─────────────────────────────────────────────────────

def test_broker_connection():
    acct = broker.trading_client.get_account()
    assert float(acct.cash) >= 0, "account cash should be non-negative"
    assert acct.options_approved_level is not None, "options approval level missing"


def test_occ_symbol_parser():
    parsed = broker.parse_occ_symbol("AAPL260911C00325000")
    assert parsed is not None, "should parse a valid OCC symbol"
    assert parsed["root"] == "AAPL"
    assert parsed["right"] == "C"
    assert abs(parsed["strike"] - 325.0) < 1e-9
    assert broker.parse_occ_symbol("AAPL") is None, "plain ticker should not parse as an option"


def test_option_contract_selection():
    from alpaca.trading.enums import ContractType
    df = broker.get_historical_market_data("AAPL", lookback_days=30)
    assert df is not None and not df.empty, "need AAPL price data for this test"
    price = float(df["close"].iloc[-1])
    contract = broker.pick_option_contract("AAPL", ContractType.CALL, price)
    assert contract is not None, "should find a liquid AAPL call contract"
    assert contract.strike_price is not None
    days_out = (contract.expiration_date - __import__("datetime").date.today()).days
    assert 0 < days_out <= 60, f"contract expiry {days_out}d outside a sane window"


# ── Strategy / signal generation ─────────────────────────────────────────

def test_signal_generation_not_degenerate():
    """
    Regression test for the bug where the live-row features got dropped
    along with the NaN-target row, silently producing 0% confidence /
    0% buy_prob / 0% sell_prob on every single ticker. If this ever comes
    back, every signal in the live bot goes dark without an error.
    """
    df = broker.get_historical_market_data("AAPL", lookback_days=400)
    assert df is not None and len(df) > 200, "need enough AAPL history for this test"
    processed = strategy.calculate_indicators(df)
    result = strategy.generate_ensemble_signal(processed, symbol="AAPL", loop_number=1)

    assert result["signal"] in ("BUY", "HOLD", "SELL")
    total_prob = result["buy_prob"] + result["sell_prob"]
    assert total_prob > 0.0, (
        "buy_prob + sell_prob is exactly 0 — this is the stale-feature/"
        "dropped-live-row bug regressing"
    )
    assert 0.0 <= result["buy_prob"] <= 1.0
    assert 0.0 <= result["sell_prob"] <= 1.0


# ── Universe ──────────────────────────────────────────────────────────────

def test_universe_builder():
    watchlist = universe.build_universe(verbose=False)
    assert len(watchlist) <= universe.MAX_UNIVERSE_SIZE
    assert len(watchlist) == len(set(watchlist)), "universe should have no duplicates"
    missing_seeds = set(universe.SEED_TICKERS) - set(watchlist)
    assert not missing_seeds, f"seed tickers dropped from universe: {missing_seeds}"


def test_universe_data_sources_alive():
    """
    Regression test for the bug where _fetch_top_movers() checked for a
    DataFrame '.columns' attribute on what yfinance actually returns as a
    plain dict, so the screener silently returned zero results every time.
    """
    sp500 = universe._fetch_sp500()
    assert len(sp500) > 400, f"S&P 500 scrape looks broken — only got {len(sp500)}"
    movers = universe._fetch_top_movers()
    assert len(movers) > 0, "movers screener returned nothing — likely broken again"


# ── Risk-gate sanity (equity + options) ──────────────────────────────────

def test_risk_gate_constants_sane():
    assert bot_config.CIRCUIT_BREAKER_LOSS_PCT < 0, "circuit breaker must be a negative threshold"
    assert bot_config.PROFIT_TAKE_TARGET_PCT > 0, "profit target must be positive"
    assert bot_config.CIRCUIT_BREAKER_LOSS_PCT < bot_config.PROFIT_TAKE_TARGET_PCT

    assert bot_config.OPTIONS_CIRCUIT_BREAKER_PCT < 0
    assert bot_config.OPTIONS_PROFIT_TARGET_PCT > 0
    # options are more volatile than equity — their gates should be wider, not tighter
    assert bot_config.OPTIONS_CIRCUIT_BREAKER_PCT < bot_config.CIRCUIT_BREAKER_LOSS_PCT
    assert bot_config.OPTIONS_PROFIT_TARGET_PCT > bot_config.PROFIT_TAKE_TARGET_PCT

    assert 0 < bot_config.MIN_BUY_PROB < 1
    assert 0 < bot_config.MIN_SELL_PROB < 1
    assert bot_config.OPTIONS_MIN_PROB > bot_config.MIN_BUY_PROB, (
        "options should require a higher-conviction signal than equity, given the extra leverage/decay risk"
    )


def test_position_sizing_constants_sane():
    assert broker.MAX_EQUITY_POSITIONS > 0
    assert broker.MAX_POSITION_VALUE > 0
    assert 0 < broker.BASE_FRACTION < 1
    assert broker.CONF_BOOST_CAP >= 0
    assert broker.CONF_FLOOR < broker.CONF_CEIL
    # a fully-invested book shouldn't be able to wildly exceed the account —
    # rough sanity check, not a precise simulation
    max_plausible_deployed = broker.MAX_EQUITY_POSITIONS * broker.MAX_POSITION_VALUE
    assert max_plausible_deployed <= 300_000, (
        f"max plausible deployed capital (${max_plausible_deployed:,.0f}) looks "
        f"unreasonably high for a $100k account"
    )


# ── Crash resilience wiring ───────────────────────────────────────────────

def test_main_loop_wired_correctly():
    assert callable(bot_config.run_one_loop), "run_one_loop() must exist and be callable"
    assert bot_config.CRASH_COOLDOWN_SECS > 0
    assert bot_config.LOG_PATH is not None


if __name__ == "__main__":
    tests = [
        test_broker_connection,
        test_occ_symbol_parser,
        test_option_contract_selection,
        test_signal_generation_not_degenerate,
        test_universe_builder,
        test_universe_data_sources_alive,
        test_risk_gate_constants_sane,
        test_position_sizing_constants_sane,
        test_main_loop_wired_correctly,
    ]

    print(f"Running {len(tests)} tests...\n")
    for t in tests:
        check(t.__name__, t)

    print(f"\n{'=' * 50}")
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 50)

    if FAIL:
        sys.exit(1)
