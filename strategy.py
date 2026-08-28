"""
strategy.py — Advanced ML Trading Engine v2
=============================================
Key upgrades over v1:
  • THREE-class model: BUY / HOLD / SELL  (was binary BUY/HOLD only)
  • Candlestick pattern features: doji, hammer, engulfing, shooting star,
    morning/evening star, harami — all encoded as integers
  • Ichimoku Cloud features: tenkan/kijun cross, price vs cloud, chikou span
  • Fixed confidence gate bug: multiplier now BOOSTS strong tickers, never
    blocks them by default
  • Ledger now records SELL outcomes too, not just circuit-breaker losses
  • Model caching: fitted model is reused for CACHE_LOOPS loops per ticker
    so we don't retrain from scratch on every single scan (huge speed win)
  • Regime filter: ADX + BB squeeze detection blocks trades in directionless
    markets where ML signals are noise

Ensemble  : Random Forest + XGBoost + LightGBM soft-voting (3-class)
Features  : 35+ indicators + 8 candlestick patterns + 5 Ichimoku signals
Meta      : signal_ledger.json — per-ticker accuracy, updated on every close
"""

import json
import warnings
import numpy as np
import pandas as pd
import ta

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
# filterwarnings() alone doesn't reliably suppress the sklearn/joblib
# UserWarning raised from inside RandomForest's internal thread pool during
# ensemble.fit() — filter-list matching across threads is unreliable here.
# Neutering warnings.warn() outright is what actually works. Matters for a
# week-long unattended run: unsuppressed, this warning fires on every fresh
# model fit and floods the persistent log file with noise, burying the
# actual signal/trade history the log exists to capture.
warnings.warn = lambda *a, **k: None

LEDGER_PATH   = Path("signal_ledger.json")
CACHE_LOOPS   = 3     # Re-use a fitted model for this many loops before retraining
N_ESTIMATORS  = 200   # Trees per ensemble member — tunable (e.g. lowered by backtest.py for speed)


# =====================================================================
# MODEL CACHE  (in-process, survives loop iterations)
# =====================================================================
# Structure: { symbol: { "model": VotingClassifier, "scaler": StandardScaler,
#                        "features": list, "loop": int } }
_model_cache: dict = {}


# =====================================================================
# CANDLESTICK PATTERN FEATURES
# =====================================================================

def _add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes 8 classic single/multi-bar candlestick patterns as integer
    columns (+1 bullish, -1 bearish, 0 neutral).  These are hand-crafted
    rules that capture what ML on raw OHLC often misses.
    """
    op = df["open"]
    hi = df["high"]
    lo = df["low"]
    cl = df["close"]

    body      = (cl - op).abs()
    full_rng  = (hi - lo).replace(0, np.nan)
    upper_wick = hi - cl.combine(op, max)
    lower_wick = cl.combine(op, min) - lo
    body_pct   = body / full_rng   # 0 = full wick, 1 = full body

    # ── 1. Doji (indecision) ─────────────────────────────────────────
    # Body < 10% of range
    df["CDOJI"] = np.where(body_pct < 0.10, 1, 0)

    # ── 2. Hammer / Inverted Hammer (bullish reversal at lows) ───────
    # Long lower wick (≥2× body), small upper wick
    is_hammer = (lower_wick >= 2 * body) & (upper_wick <= 0.3 * body)
    df["CHAMMER"] = np.where(is_hammer, 1, 0)

    # ── 3. Shooting Star (bearish reversal at highs) ──────────────────
    # Long upper wick (≥2× body), small lower wick
    is_star = (upper_wick >= 2 * body) & (lower_wick <= 0.3 * body)
    df["CSHOOTING_STAR"] = np.where(is_star, -1, 0)

    # ── 4. Bullish Engulfing ──────────────────────────────────────────
    prev_body = body.shift(1)
    prev_bear = (cl.shift(1) < op.shift(1))   # prior bar was red
    curr_bull = cl > op                         # current bar is green
    engulf_bull = curr_bull & prev_bear & (op < cl.shift(1)) & (cl > op.shift(1))
    df["CENGULF_BULL"] = np.where(engulf_bull, 1, 0)

    # ── 5. Bearish Engulfing ──────────────────────────────────────────
    prev_bull = (cl.shift(1) > op.shift(1))
    curr_bear = cl < op
    engulf_bear = curr_bear & prev_bull & (op > cl.shift(1)) & (cl < op.shift(1))
    df["CENGULF_BEAR"] = np.where(engulf_bear, -1, 0)

    # ── 6. Morning Star (3-bar bullish reversal) ──────────────────────
    bar1_bear = cl.shift(2) < op.shift(2)
    bar2_small = body.shift(1) < (body.shift(2) * 0.5)
    bar3_bull  = (cl > op) & (cl > (op.shift(2) + cl.shift(2)) / 2)
    df["CMORNING_STAR"] = np.where(bar1_bear & bar2_small & bar3_bull, 1, 0)

    # ── 7. Evening Star (3-bar bearish reversal) ─────────────────────
    bar1_bull2 = cl.shift(2) > op.shift(2)
    bar3_bear2 = (cl < op) & (cl < (op.shift(2) + cl.shift(2)) / 2)
    df["CEVENING_STAR"] = np.where(bar1_bull2 & bar2_small & bar3_bear2, -1, 0)

    # ── 8. Bullish Harami (inside bar after downtrend) ────────────────
    harami_bull = (
        prev_bear &
        (op > cl.shift(1)) & (cl < op.shift(1)) &   # inside the prior body
        curr_bull
    )
    df["CHARAMI"] = np.where(harami_bull, 1, 0)

    # ── Composite candle score [-4..+4] ──────────────────────────────
    df["CANDLE_SCORE"] = (
        df["CHAMMER"] + df["CSHOOTING_STAR"] + df["CENGULF_BULL"] +
        df["CENGULF_BEAR"] + df["CMORNING_STAR"] + df["CEVENING_STAR"] +
        df["CHARAMI"]
    )

    return df


# =====================================================================
# ICHIMOKU CLOUD FEATURES
# =====================================================================

def _add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 5 Ichimoku-derived binary/continuous features.
    Uses standard periods: 9 / 26 / 52.
    """
    hi = df["high"]
    lo = df["low"]
    cl = df["close"]

    # Tenkan-sen (conversion line) — 9-bar midpoint
    tenkan  = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    # Kijun-sen (base line) — 26-bar midpoint
    kijun   = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    # Senkou Span A — average of tenkan + kijun, shifted 26 forward
    span_a  = ((tenkan + kijun) / 2).shift(26)
    # Senkou Span B — 52-bar midpoint, shifted 26 forward
    span_b  = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)

    df["ICH_Tenkan"]  = tenkan
    df["ICH_Kijun"]   = kijun

    # Price above the cloud?
    cloud_top    = span_a.combine(span_b, max)
    cloud_bottom = span_a.combine(span_b, min)
    df["ICH_Above_Cloud"]  = (cl > cloud_top).astype(int)
    df["ICH_Below_Cloud"]  = (cl < cloud_bottom).astype(int)

    # Tenkan / Kijun cross signal (+1 = golden, -1 = death, 0 = none)
    tk_cross = np.where(
        (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1)),  1,
        np.where(
            (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1)), -1, 0
        )
    )
    df["ICH_TK_Cross"] = tk_cross

    # Normalised distance from kijun (mean-reversion signal)
    df["ICH_Kijun_Dist"] = (cl - kijun) / kijun.replace(0, np.nan)

    return df


# =====================================================================
# INDICATOR CALCULATION
# =====================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 35+ technical features using the `ta` library, plus
    8 candlestick patterns and 5 Ichimoku signals.
    """
    df = df.copy()

    # Ensure we have an 'open' column (Alpaca always provides it)
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])

    hi = df["high"]
    lo = df["low"]
    cl = df["close"]
    vo = df["volume"]

    # ── Momentum ──────────────────────────────────────────────────────
    df["RSI_7"]  = ta.momentum.RSIIndicator(cl, window=7).rsi()
    df["RSI_14"] = ta.momentum.RSIIndicator(cl, window=14).rsi()
    df["RSI_21"] = ta.momentum.RSIIndicator(cl, window=21).rsi()

    stoch = ta.momentum.StochasticOscillator(hi, lo, cl, window=14, smooth_window=3)
    df["STOCHk"] = stoch.stoch()
    df["STOCHd"] = stoch.stoch_signal()

    df["WILLR"]  = ta.momentum.WilliamsRIndicator(hi, lo, cl, lbp=14).williams_r()
    df["ROC_10"] = ta.momentum.ROCIndicator(cl, window=10).roc()
    df["ROC_5"]  = ta.momentum.ROCIndicator(cl, window=5).roc()

    # ── Trend ─────────────────────────────────────────────────────────
    macd_obj     = ta.trend.MACD(cl, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]   = macd_obj.macd()
    df["MACD_H"] = macd_obj.macd_diff()
    df["MACD_S"] = macd_obj.macd_signal()

    adx_obj      = ta.trend.ADXIndicator(hi, lo, cl, window=14)
    df["ADX_14"] = adx_obj.adx()
    df["DI_Pos"] = adx_obj.adx_pos()   # +DI
    df["DI_Neg"] = adx_obj.adx_neg()   # -DI

    df["SMA_20"]  = ta.trend.SMAIndicator(cl, window=20).sma_indicator()
    df["SMA_50"]  = ta.trend.SMAIndicator(cl, window=50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(cl, window=200).sma_indicator()
    df["EMA_12"]  = ta.trend.EMAIndicator(cl, window=12).ema_indicator()
    df["EMA_26"]  = ta.trend.EMAIndicator(cl, window=26).ema_indicator()

    # ── Volatility ────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(cl, window=20, window_dev=2)
    df["BB_Upper"]    = bb.bollinger_hband()
    df["BB_Lower"]    = bb.bollinger_lband()
    df["BB_Mid"]      = bb.bollinger_mavg()
    df["BB_PctB"]     = bb.bollinger_pband()   # %B — 0 at lower, 1 at upper

    df["ATR_14"] = ta.volatility.AverageTrueRange(hi, lo, cl, window=14).average_true_range()

    kc = ta.volatility.KeltnerChannel(hi, lo, cl, window=20)
    df["KC_Upper"] = kc.keltner_channel_hband()
    df["KC_Lower"] = kc.keltner_channel_lband()

    # ── Volume ────────────────────────────────────────────────────────
    df["OBV"]  = ta.volume.OnBalanceVolumeIndicator(cl, vo).on_balance_volume()
    df["VWAP"] = (cl * vo).rolling(14).sum() / vo.rolling(14).sum()

    # MFI — money flow index (volume-weighted RSI)
    df["MFI_14"] = ta.volume.MFIIndicator(hi, lo, cl, vo, window=14).money_flow_index()

    # ── Engineered features ───────────────────────────────────────────
    bb_range           = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Position"]  = (cl - df["BB_Lower"]) / bb_range
    df["BB_Width"]     = bb_range / df["SMA_20"]
    df["KC_Width"]     = (df["KC_Upper"] - df["KC_Lower"]) / cl

    # Bollinger squeeze: BB inside Keltner → low volatility breakout imminent
    df["BB_Squeeze"] = (
        (df["BB_Upper"] < df["KC_Upper"]) & (df["BB_Lower"] > df["KC_Lower"])
    ).astype(int)

    df["ATR_Pct"]    = df["ATR_14"] / cl

    df["Above_SMA20"]  = (cl > df["SMA_20"]).astype(int)
    df["Above_SMA50"]  = (cl > df["SMA_50"]).astype(int)
    df["Above_SMA200"] = (cl > df["SMA_200"]).astype(int)

    df["SMA_20_50_Ratio"] = df["SMA_20"] / df["SMA_50"].replace(0, np.nan)
    df["EMA_12_26_Ratio"] = df["EMA_12"] / df["EMA_26"].replace(0, np.nan)
    df["Close_to_SMA20"]  = cl           / df["SMA_20"].replace(0, np.nan)

    # DI spread: positive = bullish trend, negative = bearish
    df["DI_Spread"] = df["DI_Pos"] - df["DI_Neg"]

    obv_mean        = df["OBV"].abs().rolling(5).mean().replace(0, np.nan)
    df["OBV_Slope"] = df["OBV"].diff(5) / obv_mean
    df["VWAP_Delta"] = (cl - df["VWAP"]) / df["VWAP"].replace(0, np.nan)

    vol_mean            = vo.rolling(20).mean()
    vol_std             = vo.rolling(20).std().replace(0, np.nan)
    df["Volume_ZScore"] = (vo - vol_mean) / vol_std

    df["Gap_Pct"]  = cl.pct_change()
    df["High_Pct"] = (hi - cl) / cl    # distance to daily high
    df["Low_Pct"]  = (cl - lo) / cl    # distance from daily low

    # ── Candlestick patterns ──────────────────────────────────────────
    df = _add_candle_patterns(df)

    # ── Ichimoku Cloud ────────────────────────────────────────────────
    df = _add_ichimoku(df)

    return df


# =====================================================================
# FEATURE COLUMNS
# =====================================================================

FEATURE_COLS = [
    # Momentum
    "RSI_7", "RSI_14", "RSI_21",
    "STOCHk", "STOCHd",
    "WILLR", "ROC_10", "ROC_5",
    # Trend
    "MACD", "MACD_H", "MACD_S",
    "ADX_14", "DI_Spread",
    "Above_SMA20", "Above_SMA50", "Above_SMA200",
    "SMA_20_50_Ratio", "EMA_12_26_Ratio", "Close_to_SMA20",
    # Volatility
    "BB_Position", "BB_Width", "BB_PctB", "KC_Width", "ATR_Pct",
    "BB_Squeeze",
    # Volume
    "OBV_Slope", "VWAP_Delta", "Volume_ZScore", "MFI_14",
    # Price shape
    "Gap_Pct", "High_Pct", "Low_Pct",
    # Candlestick patterns
    "CDOJI", "CHAMMER", "CSHOOTING_STAR",
    "CENGULF_BULL", "CENGULF_BEAR",
    "CMORNING_STAR", "CEVENING_STAR", "CHARAMI",
    "CANDLE_SCORE",
    # Ichimoku
    "ICH_Above_Cloud", "ICH_Below_Cloud",
    "ICH_TK_Cross", "ICH_Kijun_Dist",
]

# Signal class labels for the 3-class model
# 0 = SELL, 1 = HOLD, 2 = BUY
CLASS_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


# =====================================================================
# REGIME FILTER
# =====================================================================

def _is_market_hours() -> bool:
    """
    Returns True if current time (ET) is within regular US market hours
    (9:30 AM – 4:00 PM Mon–Fri).
    """
    from datetime import datetime
    import zoneinfo
    try:
        et   = zoneinfo.ZoneInfo("America/New_York")
        now  = datetime.now(et)
        
        if now.weekday() >= 5:          # Saturday / Sunday
            return False
            
        market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except Exception:
        return True   # Fail open


def _is_tradeable_regime(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Returns (tradeable, reason).
    Blocks signals based on ADX chop and extreme low volume.
    """
    import zoneinfo
    from datetime import datetime
    
    last = df.iloc[-1]

    adx     = last.get("ADX_14",     np.nan)
    squeeze = last.get("BB_Squeeze",   0)
    vol_z   = last.get("Volume_ZScore", np.nan)

    # 1. ADX Filter
    if not np.isnan(adx) and adx < 12 and not squeeze:
        return False, f"Low-trend chop (ADX={adx:.1f}, no squeeze)"

    # 2. Volume Z-Score Filter
    if _is_market_hours():
        # Let's check if we are in the volatile opening 30 minutes (9:30 - 10:00 AM ET)
        et = zoneinfo.ZoneInfo("America/New_York")
        now_et = datetime.now(et)
        is_opening_settlement = (now_et.hour == 9 and now_et.minute < 45) # 9:30 to 9:45
        
        # Only block if volume is dead AND we aren't in the opening stabilization window
        if not is_opening_settlement and not np.isnan(vol_z) and vol_z < -2.5:
            return False, f"Dead volume during market hours (Z={vol_z:.1f})"

    return True, "OK"


# =====================================================================
# SIGNAL LEDGER
# =====================================================================

def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_ledger(ledger: dict) -> None:
    try:
        LEDGER_PATH.write_text(json.dumps(ledger, indent=2))
    except Exception:
        pass


def record_outcome(symbol: str, signal: str, was_correct: bool) -> None:
    """
    Called after a position closes (profit-take, circuit-breaker, or sell signal).
    Updates per-ticker accuracy for ALL signal types (BUY, SELL, HOLD).
    """
    ledger = _load_ledger()
    entry = ledger.setdefault(symbol, {
        "buy_correct": 0,  "buy_total": 0,
        "sell_correct": 0, "sell_total": 0,
        "hold_correct": 0, "hold_total": 0,
        "total_pnl":    0.0,
        "trade_count":  0,
    })
    key = signal.lower()
    if f"{key}_total" in entry:
        entry[f"{key}_total"] += 1
        if was_correct:
            entry[f"{key}_correct"] += 1
    _save_ledger(ledger)


def record_trade_pnl(symbol: str, pnl: float) -> None:
    """Records the dollar P&L of a closed trade for dashboard display."""
    ledger = _load_ledger()
    entry  = ledger.setdefault(symbol, {
        "buy_correct": 0,  "buy_total": 0,
        "sell_correct": 0, "sell_total": 0,
        "hold_correct": 0, "hold_total": 0,
        "total_pnl":    0.0,
        "trade_count":  0,
    })
    entry["total_pnl"]    = round(entry.get("total_pnl", 0.0) + pnl, 4)
    entry["trade_count"]  = entry.get("trade_count", 0) + 1
    _save_ledger(ledger)


def _ticker_confidence_multiplier(symbol: str) -> float:
    """
    Returns a multiplier centred on 1.0 (not 0.85!) so strong tickers
    get boosted while weak ones get penalised.

    Accuracy → multiplier:
      < 40%  → 0.70  (bot is wrong most of the time here)
        50%  → 0.85
        60%  → 1.00  (baseline — random 50/50 market)
        70%  → 1.10
       ≥ 80%  → 1.20  (bot consistently correct here)

    Requires ≥20 trades; defaults to 1.00 until then.
    """
    ledger = _load_ledger()
    entry  = ledger.get(symbol)
    if entry is None:
        return 1.00

    total = entry.get("buy_total", 0)
    if total < 20:
        return 1.00

    accuracy = entry.get("buy_correct", 0) / total
    # Linear map: accuracy [0.4 → 0.8] maps to multiplier [0.70 → 1.20]
    mult = 0.70 + (accuracy - 0.40) * (1.20 - 0.70) / (0.80 - 0.40)
    return float(np.clip(mult, 0.70, 1.20))


# =====================================================================
# PROFIT-TARGET ADVISOR
# =====================================================================

def suggest_sell_price(entry_price: float, atr: float | None,
                       confidence: float = 0.58) -> dict:
    """
    Returns suggested take-profit and stop-loss levels.
    Higher confidence → wider target (let winners run).
    ATR-aware when available; falls back to fixed %.
    """
    if atr and atr > 0:
        # Risk 1× ATR, target 2–3× ATR based on confidence
        reward_mult = 2.0 + (confidence - 0.58) * 5.0   # 2.0×→3.5× as conf rises
        stop_dist   = atr
        tp_dist     = atr * reward_mult
    else:
        stop_dist = entry_price * 0.025   # 2.5%
        tp_dist   = entry_price * (0.05 + (confidence - 0.58) * 0.15)

    return {
        "take_profit": round(entry_price + tp_dist, 4),
        "stop_loss":   round(entry_price - stop_dist, 4),
        "rr_ratio":    round(tp_dist / stop_dist, 2),
    }


# =====================================================================
# ENSEMBLE SIGNAL GENERATOR  (3-class: BUY / HOLD / SELL)
# =====================================================================

def generate_ensemble_signal(df: pd.DataFrame, symbol: str = "",
                              loop_number: int = 0) -> dict:
    """
    Trains (or reuses cached) RF + XGBoost + LightGBM on historical bars
    to predict tomorrow's direction as one of three classes:
      2 = BUY  (price up > +0.5%)
      1 = HOLD (price moves ≤ ±0.5%)
      0 = SELL (price down > -0.5%)

    Returns
    -------
    {
      "signal"     : "BUY" | "HOLD" | "SELL",
      "confidence" : float,   raw probability of the winning class
      "adjusted"   : float,   confidence × per-ticker ledger multiplier
      "sell_prob"  : float,   raw probability of SELL class
      "buy_prob"   : float,   raw probability of BUY class
      "regime_ok"  : bool,    False = blocked by regime filter
      "reason"     : str
    }
    """
    df = df.copy()

    # ── Regime filter ─────────────────────────────────────────────────
    tradeable, regime_reason = _is_tradeable_regime(df)
    if not tradeable:
        return {
            "signal": "HOLD", "confidence": 0.0, "adjusted": 0.0,
            "sell_prob": 0.0, "buy_prob": 0.0, "regime_ok": False,
            "reason": f"Regime blocked: {regime_reason}",
        }

    # ── 3-class target ────────────────────────────────────────────────
    # Tomorrow's return as % relative to today
    fwd_ret = df["close"].shift(-1) / df["close"] - 1

    # Deadband ±0.5%: avoids training on noise
    df["Target"] = np.where(fwd_ret >  0.005, 2,   # BUY
                   np.where(fwd_ret < -0.005, 0,   # SELL
                                              1))  # HOLD

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 10:
        return {
            "signal": "HOLD", "confidence": 0.0, "adjusted": 0.0,
            "sell_prob": 0.0, "buy_prob": 0.0, "regime_ok": True,
            "reason": f"Only {len(available)} features — need ≥10",
        }

    # Live row: the most recent bar with valid features. Its Target is NaN
    # (tomorrow hasn't happened yet) — that's expected, we only need X here.
    df_features = df.dropna(subset=available)
    if df_features.empty:
        return {
            "signal": "HOLD", "confidence": 0.0, "adjusted": 0.0,
            "sell_prob": 0.0, "buy_prob": 0.0, "regime_ok": True,
            "reason": "No bar with complete features",
        }
    live_row = df_features.iloc[[-1]]

    # Training data: historical bars where both features AND the realised
    # outcome are known. This naturally excludes the live row (its Target
    # is NaN), so there's no leakage and no stale-by-one-bar prediction.
    df_train = df.dropna(subset=available + ["Target"])
    if len(df_train) < 60:
        return {
            "signal": "HOLD", "confidence": 0.0, "adjusted": 0.0,
            "sell_prob": 0.0, "buy_prob": 0.0, "regime_ok": True,
            "reason": f"Only {len(df_train)} clean rows — need ≥60",
        }

    X_train = df_train[available]
    y_train = df_train["Target"]
    X_live  = live_row[available]

    if y_train.nunique() < 2:
        return {
            "signal": "HOLD", "confidence": 0.5, "adjusted": 0.5,
            "sell_prob": 0.0, "buy_prob": 0.5, "regime_ok": True,
            "reason": "All training labels same class",
        }

    # ── Model cache ───────────────────────────────────────────────────
    cached = _model_cache.get(symbol)
    use_cache = (
        cached is not None and
        cached.get("features") == available and
        (loop_number - cached.get("loop", 0)) < CACHE_LOOPS
    )

    try:
        if use_cache:
            ensemble = cached["model"]
            scaler   = cached["scaler"]
            X_live_s = scaler.transform(X_live)
        else:
            scaler    = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_live_s  = scaler.transform(X_live)

            rf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=6,
                min_samples_leaf=3, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            xgb = XGBClassifier(
                n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.05,
                eval_metric="mlogloss", verbosity=0,
                use_label_encoder=False, random_state=42, n_jobs=2,
            )
            lgbm = LGBMClassifier(
                n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.05,
                class_weight="balanced", verbosity=-1, random_state=42, n_jobs=2,
            )

            ensemble = VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                voting="soft",
                weights=[1, 1.3, 1.3],
            )
            ensemble.fit(X_train_s, y_train)

            _model_cache[symbol] = {
                "model": ensemble, "scaler": scaler,
                "features": available, "loop": loop_number,
            }

        # ── Predict ───────────────────────────────────────────────────
        proba = ensemble.predict_proba(X_live_s)[0]  # shape: [p_sell, p_hold, p_buy]
        classes = list(ensemble.classes_)

        # Map back to SELL=0, HOLD=1, BUY=2 safely
        p = {cls: proba[i] for i, cls in enumerate(classes)}
        p_sell = p.get(0, 0.0)
        p_hold = p.get(1, 0.0)
        p_buy  = p.get(2, 0.0)

        # Winning class
        best_class = max(p, key=p.get)
        signal     = CLASS_LABELS[best_class]
        confidence = float(p[best_class])

        multiplier = _ticker_confidence_multiplier(symbol)
        adj_conf   = min(confidence * multiplier, 1.0)

        # Candle context for human-readable reason
        last       = live_row.iloc[-1]
        candle_sc  = last.get("CANDLE_SCORE", 0)
        ichi_cross = last.get("ICH_TK_Cross", 0)
        candle_str = f"candle={candle_sc:+.0f}" if candle_sc != 0 else ""
        ichi_str   = ("ichi=golden✚" if ichi_cross == 1 else
                      "ichi=death✖"  if ichi_cross == -1 else "")

        # Top-3 RF feature importances
        importances = ensemble.estimators_[0].feature_importances_
        top_idx     = np.argsort(importances)[-3:][::-1]
        top_feats   = ", ".join(
            f"{available[i]}={importances[i]:.2f}" for i in top_idx
        )

        extras = " | ".join(x for x in [candle_str, ichi_str] if x)
        return {
            "signal":     signal,
            "confidence": round(confidence, 4),
            "adjusted":   round(adj_conf,   4),
            "sell_prob":  round(p_sell, 4),
            "buy_prob":   round(p_buy,  4),
            "regime_ok":  True,
            "reason": (
                f"BUY={p_buy:.0%} SELL={p_sell:.0%} HOLD={p_hold:.0%} "
                f"| adj={adj_conf:.0%} | {extras} | top: {top_feats}"
            ),
        }

    except Exception as e:
        return {
            "signal": "HOLD", "confidence": 0.0, "adjusted": 0.0,
            "sell_prob": 0.0, "buy_prob": 0.0, "regime_ok": True,
            "reason": f"Model error: {e}",
        }