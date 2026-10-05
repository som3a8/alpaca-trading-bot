"""
dashboard.py — every bot at a glance.

Reads all three paper accounts live from Alpaca (BOT1..3 keys in .env), so nothing
has to be pulled from GitHub first. Run it with run_dashboard.bat, or:

    streamlit run dashboard.py
"""

import os
from collections import defaultdict, deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

load_dotenv()

START = float(os.getenv("DASHBOARD_STARTING_BALANCE") or 1000)   # every bot starts here
ET = "America/New_York"

# Line colours validated for light + dark themes and colour-blind separation.
BOTS = [
    dict(id="bot1", label="Bot 1", strategy="ML ensemble",   role="control",
         key="BOT1_API_KEY", secret="BOT1_API_SECRET", color="#2a6fdb"),
    dict(id="bot2", label="Bot 2", strategy="MA crossover",  role="no-ML baseline",
         key="BOT2_API_KEY", secret="BOT2_API_SECRET", color="#d9680b"),
    dict(id="bot3", label="Bot 3", strategy="Mean reversion", role="contrarian",
         key="BOT3_API_KEY", secret="BOT3_API_SECRET", color="#8a4fd3"),
]
GAIN, LOSS = "#2f7a52", "#b3552f"   # status colours, kept apart from the bot colours

PERIODS = {"Today": ("1D", "5Min"), "Week": ("1W", "1H"), "Month": ("1M", "1D"), "All": ("3M", "1D")}


# ── Data ──────────────────────────────────────────────────────────────────

def closed_trades(fills: pd.DataFrame) -> pd.DataFrame:
    """FIFO-match every sell against earlier buys of the same symbol."""
    rows, lots = [], defaultdict(deque)
    for r in fills.sort_values("time").itertuples():
        if r.side == "buy":
            lots[r.symbol].append([r.qty, r.price])
            continue
        remaining, matched, cost, q = r.qty, 0.0, 0.0, lots[r.symbol]
        while remaining > 1e-9 and q:
            take = min(q[0][0], remaining)
            cost += take * q[0][1]
            matched += take
            q[0][0] -= take
            remaining -= take
            if q[0][0] <= 1e-9:
                q.popleft()
        if matched > 0:
            pnl = matched * r.price - cost
            rows.append({"time": r.time, "symbol": r.symbol, "pnl": pnl,
                         "pnl_pct": pnl / cost * 100 if cost else 0.0})
    return pd.DataFrame(rows, columns=["time", "symbol", "pnl", "pnl_pct"])


def equity_history(c: TradingClient, period: str) -> pd.DataFrame:
    per, tf = PERIODS[period]
    hist = c.get_portfolio_history(GetPortfolioHistoryRequest(period=per, timeframe=tf))
    h = pd.DataFrame({"time": pd.to_datetime(hist.timestamp, unit="s", utc=True),
                      "equity": hist.equity}).dropna()
    h = h[h["equity"] > 0].copy()
    h["time"] = h["time"].dt.tz_convert(ET).dt.tz_localize(None)   # naive ET: plotly ignores offsets
    return h


@st.cache_data(ttl=60, show_spinner=False)
def load_bot(key_env: str, secret_env: str, period: str) -> dict:
    key, secret = os.getenv(key_env), os.getenv(secret_env)
    if not key or not secret:
        return {"error": f"{key_env} / {secret_env} missing from .env"}
    try:
        c = TradingClient(key, secret, paper=True)
        acct = c.get_account()
        positions = c.get_all_positions()
        h, note = equity_history(c, period), None
        if len(h) < 2 and period != "Today":
            # Alpaca's multi-day ranges stay empty for a brand-new account until a full
            # session has passed; today's intraday series is there from day one.
            today = equity_history(c, "Today")
            if len(today) >= 2:
                h, note = today, f"Not enough history for '{period}' yet, showing today instead."
        orders = c.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
    except Exception as e:   # bad key, network down, rate limit...
        return {"error": str(e)}

    cols = ["submitted", "time", "symbol", "side", "qty", "price", "status"]
    allo = pd.DataFrame([{
        "submitted": pd.Timestamp(o.submitted_at).tz_convert(ET).tz_localize(None),
        "time": (pd.Timestamp(o.filled_at).tz_convert(ET).tz_localize(None) if o.filled_at else pd.NaT),
        "symbol": o.symbol, "side": o.side.value,
        "qty": float(o.filled_qty or 0), "price": float(o.filled_avg_price or 0),
        "status": o.status.value,
    } for o in orders], columns=cols)
    fills = allo[(allo["status"] == "filled") & (allo["qty"] > 0)]

    pos = pd.DataFrame([{
        "symbol": p.symbol, "qty": float(p.qty), "entry": float(p.avg_entry_price),
        "price": float(p.current_price), "value": float(p.market_value),
        "pnl": float(p.unrealized_pl), "pnl_pct": float(p.unrealized_plpc) * 100,
    } for p in positions], columns=["symbol", "qty", "entry", "price", "value", "pnl", "pnl_pct"])

    return {
        "id": str(acct.id)[:8],
        "equity": float(acct.equity),
        "last_equity": float(acct.last_equity or acct.equity),
        "cash": float(acct.cash),
        "positions": pos, "history": h, "history_note": note, "orders": allo, "fills": fills,
        "trades": closed_trades(fills) if len(fills) else pd.DataFrame(columns=["time", "symbol", "pnl", "pnl_pct"]),
        "last_order": allo["submitted"].max() if len(allo) else None,
    }


def summarize(d: dict) -> dict:
    h, t, p = d["history"], d["trades"], d["positions"]
    closed = len(t)
    return {
        "equity": d["equity"],
        "ret": (d["equity"] / START - 1) * 100,
        "day": d["equity"] - d["last_equity"],
        "dd": float((h["equity"] / h["equity"].cummax() - 1).min() * 100) if len(h) > 1 else 0.0,
        "open": len(p),
        "closed": closed,
        "win": (float((t["pnl"] > 0).mean() * 100) if closed else None),
        "realized": float(t["pnl"].sum()) if closed else 0.0,
        "unrealized": float(p["pnl"].sum()) if len(p) else 0.0,
    }


def ago(ts) -> str:
    if ts is None or pd.isna(ts):
        return "no orders yet"
    now = pd.Timestamp.now(tz=ET).tz_localize(None)
    mins = (now - ts).total_seconds() / 60
    if mins < 2:
        return "just now"
    if mins < 90:
        return f"{int(mins)} min ago"
    if mins < 36 * 60:
        return f"{mins / 60:.0f} h ago"
    return f"{mins / 1440:.0f} d ago"


# ── Charts ────────────────────────────────────────────────────────────────

GRID = "rgba(128,128,128,0.18)"


def equity_chart(data: dict, period: str) -> go.Figure:
    fig = go.Figure()
    ends = []
    for b in BOTS:
        d = data[b["id"]]
        if "error" in d or d["history"].empty:
            continue
        h = d["history"]
        y = (h["equity"] / START - 1) * 100
        fig.add_trace(go.Scatter(
            x=h["time"], y=y, name=f"{b['label']} · {b['strategy']}",
            mode="lines+markers" if len(h) < 15 else "lines",
            line=dict(color=b["color"], width=2.5), marker=dict(size=5),
            hovertemplate="%{y:+.2f}%",
        ))
        ends.append((b, h["time"].iloc[-1], float(y.iloc[-1])))

    # End-of-line labels. If the bots are still bunched together (e.g. all flat at
    # day one) stagger the labels so they don't print on top of each other.
    bunched = ends and (max(e[2] for e in ends) - min(e[2] for e in ends)) < 0.2
    for i, (b, x, y) in enumerate(ends):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=9, color=b["color"], line=dict(width=2, color="rgba(128,128,128,0.35)"))))
        fig.add_annotation(x=x, y=y, text=b["label"], showarrow=False, xanchor="left", xshift=10,
                           yshift=(i - (len(ends) - 1) / 2) * 16 if bunched else 0)

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.6)", line_width=1)
    breaks = [dict(bounds=["sat", "mon"])]
    if period in ("Today", "Week"):
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))   # hide overnight gaps
    fig.update_xaxes(rangebreaks=breaks, gridcolor=GRID, showspikes=True, spikethickness=1)
    fig.update_yaxes(ticksuffix="%", gridcolor=GRID, zeroline=False)
    fig.update_layout(height=380, margin=dict(l=0, r=60, t=34, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def bar_chart(x, y, hover_suffix="") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=[GAIN if v >= 0 else LOSS for v in y],
        text=[f"{v:+.2f}" for v in y], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:+,.2f}" + hover_suffix + "<extra></extra>",
    ))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=16, b=0), bargap=0.45)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="rgba(128,128,128,0.6)")
    return fig


# ── Page ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Trading bots", layout="wide")
st.title("Trading bots")
st.caption("Three paper accounts, $%s each, same loop and same exit rules. Only the entry signal differs." % f"{START:,.0f}")

period = st.radio("Range", list(PERIODS), index=1, horizontal=True, label_visibility="collapsed")


@st.fragment(run_every=60)
def render(period: str):
    data = {b["id"]: load_bot(b["key"], b["secret"], period) for b in BOTS}
    stats = {k: summarize(v) for k, v in data.items() if "error" not in v}

    # Row 1: one card per bot
    for col, b in zip(st.columns(len(BOTS)), BOTS):
        d = data[b["id"]]
        with col, st.container(border=True):
            st.markdown(f"**{b['label']}** · {b['strategy']}  \n:gray[{b['role']}]")
            if "error" in d:
                st.error(d["error"])
                continue
            s = stats[b["id"]]
            st.metric("Equity", f"${s['equity']:,.2f}", f"{s['ret']:+.2f}% since start")
            st.caption(f"Today {s['day']:+,.2f} · {s['open']} open · last order {ago(d['last_order'])} · acct {d['id']}")

    # Row 2: the comparison chart
    st.subheader("Return since start")
    if any("error" not in d and not d["history"].empty for d in data.values()):
        st.plotly_chart(equity_chart(data, period), use_container_width=True, config={"displayModeBar": False})
        for note in {d["history_note"] for d in data.values() if d.get("history_note")}:
            st.caption(note)
    else:
        st.info("No equity history yet. The chart fills in as the bots trade.")

    # Row 3: side-by-side numbers
    if stats:
        st.subheader("Side by side")
        rows = [{
            "Bot": f"{b['label']} · {b['strategy']}", "Equity": s["equity"], "Return %": s["ret"],
            "Today $": s["day"], "Max drawdown %": s["dd"], "Open": s["open"], "Closed trades": s["closed"],
            "Win rate %": s["win"], "Realized $": s["realized"], "Unrealized $": s["unrealized"],
        } for b in BOTS if (s := stats.get(b["id"]))]
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={
                "Equity": st.column_config.NumberColumn(format="$%.2f"),
                "Return %": st.column_config.NumberColumn(format="%+.2f%%"),
                "Today $": st.column_config.NumberColumn(format="%+.2f"),
                "Max drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
                "Win rate %": st.column_config.NumberColumn(format="%.0f%%"),
                "Realized $": st.column_config.NumberColumn(format="%+.2f"),
                "Unrealized $": st.column_config.NumberColumn(format="%+.2f"),
            },
        )
        st.caption(f"Max drawdown is over the selected range. Closed trades and win rate come from every fill "
                   f"(most recent 500 orders per account), FIFO-matched. Updated "
                   f"{pd.Timestamp.now(tz=ET):%H:%M:%S} ET, refreshes every 60 s.")

    # Row 4: one tab per bot
    st.subheader("Per bot")
    for tab, b in zip(st.tabs([f"{b['label']} · {b['strategy']}" for b in BOTS]), BOTS):
        d = data[b["id"]]
        with tab:
            if "error" in d:
                st.error(d["error"])
                continue
            left, right = st.columns(2)
            with left:
                st.markdown("**Open positions: unrealized P&L ($)**")
                p = d["positions"]
                if p.empty:
                    st.info("No open positions.")
                else:
                    st.plotly_chart(bar_chart(p["symbol"], p["pnl"].round(2)), use_container_width=True,
                                    config={"displayModeBar": False}, key=f"pos_{b['id']}")
            with right:
                st.markdown("**Closed trades: realized P&L ($)**")
                t = d["trades"]
                if t.empty:
                    st.info("No closed trades yet.")
                else:
                    last = t.tail(40)
                    st.plotly_chart(bar_chart([f"{r.symbol} {r.time:%m-%d}" for r in last.itertuples()],
                                              last["pnl"].round(2)), use_container_width=True,
                                    config={"displayModeBar": False}, key=f"tr_{b['id']}")
            if not d["positions"].empty:
                st.dataframe(d["positions"], hide_index=True, use_container_width=True, column_config={
                    "entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "pnl": st.column_config.NumberColumn("P&L $", format="%+.2f"),
                    "pnl_pct": st.column_config.NumberColumn("P&L %", format="%+.2f%%"),
                    "qty": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "symbol": "Symbol",
                })
            with st.expander("Recent orders"):
                o = d["orders"].sort_values("submitted", ascending=False).head(25)
                st.dataframe(o[["submitted", "symbol", "side", "qty", "price", "status"]],
                             hide_index=True, use_container_width=True)


render(period)
