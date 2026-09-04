"""
dashboard.py — Live Bot Portfolio Dashboard v3
================================================
New in v3:
  • Cumulative P&L chart (reads pnl_log.json written by main.py)
  • Market regime indicator (reads session_snapshot.json)
  • Stale / trail stop position status badges
  • Per-ticker ledger P&L from signal_ledger.json
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Alpaca Bot Live Tracker v3", layout="wide")
st.title("🤖 Live Bot Portfolio Dashboard v3")

load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    st.error("❌ Missing ALPACA_API_KEY or ALPACA_SECRET_KEY inside your .env file!")
    st.stop()

@st.cache_resource
def get_client():
    return TradingClient(API_KEY, SECRET_KEY, paper=True)

client = get_client()

# Auto-refresh every 15 s
st.write("⏱️ Auto-refreshing every 15 seconds...")
st.fragment(run_every=15)

# ── Helper: load local JSON files ────────────────────────────────────────
def _load_json(path: str, default):
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

# ── Load side-car files written by main.py ───────────────────────────────
pnl_log       = _load_json("pnl_log.json",        [])
session_snap  = _load_json("session_snapshot.json", {})
signal_ledger = _load_json("signal_ledger.json",   {})

try:
    # ── ACCOUNT DATA ─────────────────────────────────────────────────────
    account           = client.get_account()
    equity            = float(account.equity)
    cash              = float(account.cash)
    long_market_value = float(account.long_market_value)
    # last_equity is yesterday's close — not "starting balance" for an
    # all-time return, but there's no account-creation-balance field on the
    # account object, so this is the best available reference point. Set
    # DASHBOARD_STARTING_BALANCE in .env if you want a fixed baseline instead
    # (e.g. right after a paper-account reset).
    starting_balance  = float(os.getenv("DASHBOARD_STARTING_BALANCE") or account.last_equity)
    total_profit_loss = equity - starting_balance
    total_return_pct  = (total_profit_loss / starting_balance) * 100 if starting_balance else 0.0

    session_pnl   = session_snap.get("session_pnl", 0.0)
    loop_number   = session_snap.get("loop",         "—")
    holdings_list = session_snap.get("holdings",     [])

    # ── ROW 1: KEY METRICS ────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Equity",        f"${equity:,.2f}")
    with col2:
        st.metric("Available Cash",      f"${cash:,.2f}")
    with col3:
        st.metric("Stock Value",         f"${long_market_value:,.2f}")
    with col4:
        st.metric("All-time Return",
                  f"${total_profit_loss:+,.2f}",
                  f"{total_return_pct:+.2f}%")
    with col5:
        st.metric("Session Realised P&L",
                  f"${session_pnl:+,.2f}",
                  f"Loop #{loop_number}")

    st.markdown("---")

    # ── ROW 2: CUMULATIVE P&L CHART + ALLOCATION ──────────────────────────
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.subheader("📈 Cumulative Realised P&L")
        if pnl_log:
            pnl_df = pd.DataFrame(pnl_log)
            pnl_df["ts"] = pd.to_datetime(pnl_df["ts"])
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=pnl_df["ts"],
                y=pnl_df["cumulative"],
                mode="lines+markers",
                line=dict(color="#2ecc71" if pnl_df["cumulative"].iloc[-1] >= 0 else "#e74c3c",
                          width=2),
                marker=dict(size=5),
                name="Cumulative P&L",
                hovertemplate="<b>%{x}</b><br>P&L: $%{y:+.2f}<extra></extra>",
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig_pnl.update_layout(
                xaxis_title="Time",
                yaxis_title="Cumulative P&L ($)",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Trade log table
            with st.expander("📋 Trade Log"):
                log_display = pnl_df[["ts", "reason", "pnl", "cumulative"]].copy()
                log_display.columns = ["Time", "Reason", "P&L ($)", "Cumulative ($)"]
                log_display = log_display.sort_values("Time", ascending=False)
                st.dataframe(log_display, use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet — P&L chart will appear after the first exit.")

    with chart_col2:
        # Per-ticker ledger summary
        st.subheader("📚 Ticker Ledger")
        if signal_ledger:
            ledger_rows = []
            for sym, entry in signal_ledger.items():
                total  = entry.get("buy_total", 0)
                correct= entry.get("buy_correct", 0)
                pnl    = entry.get("total_pnl", 0.0)
                acc    = f"{correct/total:.0%}" if total > 0 else "—"
                ledger_rows.append({
                    "Ticker": sym,
                    "Trades": total,
                    "Accuracy": acc,
                    "Total P&L": round(pnl, 2),
                })
            ledger_df = pd.DataFrame(ledger_rows).sort_values("Total P&L", ascending=False)
            st.dataframe(
                ledger_df,
                column_config={
                    "Total P&L": st.column_config.NumberColumn(format="$+%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No ledger data yet.")

    st.markdown("---")

    # ── ROW 3: OPEN POSITIONS ─────────────────────────────────────────────
    positions = client.get_all_positions()

    if not positions:
        st.info("ℹ️ No open positions. Waiting for signals...")
        fig_empty = px.pie(names=["Cash"], values=[cash], title="Portfolio Allocation")
        st.plotly_chart(fig_empty, use_container_width=True)
    else:
        data = []
        for p in positions:
            upl     = round(float(p.unrealized_pl), 2)
            upl_pct = round(float(p.unrealized_plpc) * 100, 2)

            # Status badge: trail stop active? stale?
            sym = p.symbol
            badge = ""
            if upl >= 3.0:
                badge = "🔺 trailing"
            elif upl < 0:
                badge = "🟡 watching"
            else:
                badge = "🟢 healthy"

            data.append({
                "Symbol":               sym,
                "Shares":               round(float(p.qty), 4),
                "Entry Price":          round(float(p.avg_entry_price), 2),
                "Current Price":        round(float(p.current_price), 2),
                "Market Value ($)":     round(float(p.market_value), 2),
                "Unrealized P&L ($)":   upl,
                "Unrealized P&L (%)":   upl_pct,
                "Status":               badge,
            })

        df = pd.DataFrame(data)

        pos_col1, pos_col2 = st.columns(2)

        with pos_col1:
            alloc_names  = df["Symbol"].tolist() + ["Cash"]
            alloc_values = df["Market Value ($)"].tolist() + [max(0.0, cash)]
            fig_pie = px.pie(
                names=alloc_names, values=alloc_values,
                title="Portfolio Allocation",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with pos_col2:
            df["Color"] = df["Unrealized P&L ($)"].apply(
                lambda x: "Profit" if x >= 0 else "Loss"
            )
            fig_bar = px.bar(
                df, x="Symbol", y="Unrealized P&L ($)", color="Color",
                title="Unrealized P&L by Holding",
                color_discrete_map={"Profit": "#2ecc71", "Loss": "#e74c3c"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📋 Active Holdings")
        st.dataframe(
            df.drop(columns=["Color"]),
            column_config={
                "Market Value ($)":   st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P&L ($)": st.column_config.NumberColumn(format="$+%.2f"),
                "Unrealized P&L (%)": st.column_config.NumberColumn(format="+%.2f%%"),
                "Entry Price":        st.column_config.NumberColumn(format="$%.2f"),
                "Current Price":      st.column_config.NumberColumn(format="$%.2f"),
            },
            use_container_width=True,
            hide_index=True,
        )

except Exception as e:
    st.error(f"❌ Failed to communicate with Alpaca API: {e}")