"""
╔══════════════════════════════════════════════════════════════════╗
║  streamlit_app/app.py — 資源法 v5.0 戰情室（含買賣原因顯示）     ║
║                                                                  ║
║  新增功能：                                                      ║
║  - 今日買進原因（結構化顯示）                                     ║
║  - 持股賣出訊號（多層條件）                                       ║
║  - 回測績效圖（Equity Curve）                                     ║
║  - 買賣歷史紀錄（trades.csv）                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import html as _html_escape
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional

# ──────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import os

def _get_secret(key, default=""):
    try: return st.secrets[key]
    except Exception: return os.environ.get(key, default)

GITHUB_OWNER = _get_secret("GITHUB_OWNER", "your-username")
GITHUB_REPO  = _get_secret("GITHUB_REPO",  "stock2026")
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/storage"

st.set_page_config(page_title="資源法 v5.0 戰情室",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root {
    --bg:#f0f4f8; --panel:#ffffff; --border:rgba(59,130,246,0.15);
    --text:#1e293b; --text-mid:#475569; --text-dim:#94a3b8;
    --accent:#2563eb; --green:#059669; --red:#dc2626;
    --amber:#d97706; --purple:#7c3aed; --teal:#0d9488;
}
html,body,.stApp { background:var(--bg)!important; color:var(--text)!important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding:1rem 2rem 3rem!important; max-width:1700px; }

.card { background:var(--panel); border:1px solid var(--border);
        border-radius:10px; padding:16px 18px; margin-bottom:10px; }
.sec-header { display:flex; align-items:center; gap:12px;
              padding:10px 18px; margin:16px 0 10px;
              border-radius:10px; border-left:3px solid;
              background:var(--panel); }
.pill { display:inline-block; padding:2px 9px; border-radius:12px;
        font-size:0.69rem; font-weight:700; font-family:monospace;
        border:1px solid; margin:1px; }
.pill-g { background:rgba(5,150,105,0.08); border-color:rgba(5,150,105,0.3); color:#059669; }
.pill-r { background:rgba(220,38,38,0.08); border-color:rgba(220,38,38,0.3); color:#dc2626; }
.pill-b { background:rgba(37,99,235,0.08); border-color:rgba(37,99,235,0.3); color:#2563eb; }
.pill-a { background:rgba(217,119,6,0.08); border-color:rgba(217,119,6,0.3); color:#d97706; }
.pill-p { background:rgba(124,58,237,0.08); border-color:rgba(124,58,237,0.3); color:#7c3aed; }

.reason-box { background:#f8fafc; border:1px solid rgba(59,130,246,0.15);
              border-left:3px solid #2563eb; border-radius:8px;
              padding:12px 16px; font-size:0.83rem; line-height:1.8;
              color:#1e293b; margin:6px 0; }
.sell-box   { border-left-color:#dc2626; background:#fef2f2; }
.buy-box    { border-left-color:#059669; background:#f0fdf4; }

.data-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.data-table th { background:#e8edf4; color:#94a3b8; font-weight:600;
                 font-family:monospace; font-size:0.67rem; padding:9px 12px;
                 text-align:left; border-bottom:2px solid rgba(59,130,246,0.2); }
.data-table td { padding:9px 12px; border-bottom:1px solid rgba(59,130,246,0.07); }
.data-table tr:hover td { background:rgba(37,99,235,0.03); }
.mono { font-family:monospace; font-weight:700; }
.c-g { color:#059669; } .c-r { color:#dc2626; }
.c-b { color:#2563eb; } .c-a { color:#d97706; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 資料載入
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_url(path_suffix: str) -> Optional[dict]:
    try:
        r = requests.get(f"{BASE_URL}/{path_suffix}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_csv_url(path_suffix: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(f"{BASE_URL}/{path_suffix}", timeout=10)
        if r.status_code == 200:
            from io import StringIO
            return pd.read_csv(StringIO(r.text))
        return None
    except Exception:
        return None


def load_all():
    portfolio  = load_url("portfolio_latest.json")
    v4         = load_url("v4/v4_latest.json")
    v12        = load_url("v12/v12_latest.json")
    regime     = load_url("regime/regime_state.json")
    backtest   = load_url("backtest_result.json")
    trades_df  = load_csv_url("trades.csv")
    return portfolio, v4, v12, regime, backtest, trades_df


# ──────────────────────────────────────────────────────────────
# HTML 渲染工具
# ──────────────────────────────────────────────────────────────

def _render_html(html_body: str, height: int = 400):
    full = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{{background:transparent;font-family:'Noto Sans TC','DM Sans',sans-serif;
      color:#1e293b;margin:0;padding:0;height:{height}px;overflow-y:auto;}}
.data-table{{width:100%;border-collapse:collapse;font-size:0.82rem;}}
.data-table th{{background:#e8edf4;color:#94a3b8;font-weight:600;
                font-family:monospace;font-size:0.67rem;padding:9px 12px;
                text-align:left;border-bottom:2px solid rgba(59,130,246,0.2);
                position:sticky;top:0;}}
.data-table td{{padding:9px 12px;border-bottom:1px solid rgba(59,130,246,0.07);color:#1e293b;}}
.data-table tr:hover td{{background:rgba(37,99,235,0.03);}}
.mono{{font-family:monospace;font-weight:700;}}
.c-g{{color:#059669;}}.c-r{{color:#dc2626;}}
.c-b{{color:#2563eb;}}.c-a{{color:#d97706;}}
.pill{{display:inline-block;padding:2px 9px;border-radius:12px;
       font-size:0.68rem;font-weight:700;font-family:monospace;border:1px solid;}}
.pill-g{{background:rgba(5,150,105,0.08);border-color:rgba(5,150,105,0.3);color:#059669;}}
.pill-r{{background:rgba(220,38,38,0.08);border-color:rgba(220,38,38,0.3);color:#dc2626;}}
.pill-b{{background:rgba(37,99,235,0.08);border-color:rgba(37,99,235,0.3);color:#2563eb;}}
.pill-a{{background:rgba(217,119,6,0.08);border-color:rgba(217,119,6,0.3);color:#d97706;}}
</style></head><body>{html_body}</body></html>"""
    st.html(full)


def _action_pill(action: str) -> str:
    m = {"BUY":("pill-g","▲ 買進"),"SELL_STOP":("pill-r","🛑 停損"),
         "SELL_TP1":("pill-g","🎯 TP1"),"SELL_TP2":("pill-g","🎯 TP2"),
         "SELL_TRAIL":("pill-a","↘ 移停"),"SELL_EV":("pill-a","⚠ EV退"),
         "SELL_REPLACE":("pill-b","🔄 換股")}
    css,lbl = m.get(action,("pill-b",action))
    return f'<span class="pill {css}">{lbl}</span>'


# ──────────────────────────────────────────────────────────────
# Section: 今日買進原因
# ──────────────────────────────────────────────────────────────

def render_buy_reasons(portfolio: dict):
    bought = portfolio.get("bought_today", [])
    if not bought:
        st.info("今日無新進場標的")
        return

    st.markdown(f"""
    <div class="sec-header" style="border-color:#059669;">
      <span style="font-size:1.05rem;font-weight:900;color:#059669;">
        ▲ 今日進場 {len(bought)} 檔 — 買進原因
      </span>
    </div>""", unsafe_allow_html=True)

    for b in bought:
        sym    = b.get("symbol", "—")
        reason = b.get("reason", "—")
        path   = b.get("path", "—")
        ev     = b.get("ev", 0)
        price  = b.get("price", 0)
        shares = b.get("shares", 0)
        ev_pct = ev * 100 if ev < 1 else ev

        # 解析原因各段
        parts = reason.split("｜")
        html_parts = ""
        for p in parts:
            if p.startswith("【V4】"):
                html_parts += f'<div><span class="pill pill-b">V4 訊號</span> {_html_escape.escape(p[5:])}</div>'
            elif p.startswith("【V12】"):
                html_parts += f'<div><span class="pill pill-g">V12 路徑</span> {_html_escape.escape(p[6:])}</div>'
            elif p.startswith("【Regime】"):
                html_parts += f'<div><span class="pill pill-a">Regime</span> {_html_escape.escape(p[9:])}</div>'
            else:
                html_parts += f'<div style="color:#475569;font-size:0.8rem;">{_html_escape.escape(p)}</div>'

        st.markdown(f"""
        <div class="reason-box buy-box">
          <div style="font-weight:700;font-size:0.95rem;margin-bottom:8px;">
            ▲ <b>{sym}</b>
            <span style="font-family:monospace;margin-left:8px;">
              @{price:.1f} × {shares:,}股
            </span>
            <span class="pill pill-g" style="margin-left:8px;">路徑{path}</span>
            <span class="pill pill-b" style="margin-left:4px;">EV={ev_pct:.2f}%</span>
          </div>
          {html_parts}
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Section: 賣出訊號 & 原因
# ──────────────────────────────────────────────────────────────

def render_sell_signals(portfolio: dict):
    positions = portfolio.get("positions", [])
    sold      = portfolio.get("sold_today", [])

    # 顯示已賣出
    if sold:
        st.markdown(f"""
        <div class="sec-header" style="border-color:#dc2626;">
          <span style="font-size:1.05rem;font-weight:900;color:#dc2626;">
            ▼ 今日出場 {len(sold)} 檔 — 賣出原因
          </span>
        </div>""", unsafe_allow_html=True)

        for s in sold:
            sym     = s.get("symbol","—")
            reason  = s.get("reason","—")
            action  = s.get("action","SELL")
            ret     = s.get("ret_pct", 0)
            days    = s.get("days", 0)
            ret_css = "c-g" if ret > 0 else "c-r"
            st.markdown(f"""
            <div class="reason-box sell-box">
              <div style="font-weight:700;font-size:0.95rem;margin-bottom:6px;">
                ▼ <b>{sym}</b>
                {_action_pill(action)}
                <span class="mono {ret_css}" style="margin-left:8px;">{ret:+.2f}%</span>
                <span style="color:#94a3b8;margin-left:8px;">持 {days} 天</span>
              </div>
              <div style="color:#475569;">{_html_escape.escape(reason)}</div>
            </div>""", unsafe_allow_html=True)

    # 顯示現有持倉出場警告
    warn_pos = [p for p in positions if p.get("exit_signal","—") not in ("—","")]
    if warn_pos:
        st.markdown(f"""
        <div class="sec-header" style="border-color:#d97706;">
          <span style="font-size:1.0rem;font-weight:900;color:#d97706;">
            ⚠️ 出場警告：{len(warn_pos)} 檔持倉出現出場訊號
          </span>
        </div>""", unsafe_allow_html=True)

        html = """<table class="data-table"><thead><tr>
            <th>代號</th><th>出場訊號</th><th>現價</th>
            <th>停利①</th><th>停損</th><th>EV現%</th>
            <th>持天</th><th>報酬%</th>
        </tr></thead><tbody>"""
        for p in warn_pos:
            sym   = _html_escape.escape(p.get("symbol","—"))
            sig   = _html_escape.escape(p.get("exit_signal","—"))
            curr  = p.get("curr_price",0)
            tp1   = p.get("tp1_price",0)
            stop  = p.get("stop_price",0)
            ev    = p.get("ev_now",0)
            days  = p.get("days_held",0)
            ret   = p.get("curr_ret_pct",0)
            ret_c = "c-g" if ret>0 else "c-r"
            sig_c = "c-r" if "停損" in sig else "c-a"
            html += f"""<tr>
                <td><b>{sym}</b></td>
                <td><span class="mono {sig_c}">⚠ {sig}</span></td>
                <td class="mono">{curr:.1f}</td>
                <td class="mono c-g">{tp1:.1f}</td>
                <td class="mono c-r">{stop:.1f}</td>
                <td class="mono c-b">{ev:.2f}%</td>
                <td class="mono">{days}</td>
                <td class="mono {ret_c}">{ret:+.2f}%</td>
            </tr>"""
        html += "</tbody></table>"
        _render_html(html, height=min(80 + len(warn_pos)*46, 500))


# ──────────────────────────────────────────────────────────────
# Section: 持倉監控
# ──────────────────────────────────────────────────────────────

def render_positions(portfolio: dict):
    positions = portfolio.get("positions", [])
    n = len(positions)
    avail = portfolio.get("available_cap", 0)
    total = portfolio.get("total_val", 0)

    st.markdown(f"""
    <div class="sec-header" style="border-color:#2563eb;">
      <span style="font-size:1.05rem;font-weight:900;color:#2563eb;">
        📋 目前持倉 {n} 檔
      </span>
      <span style="color:#94a3b8;font-size:0.8rem;margin-left:auto;">
        可用: {avail:,.0f} | 總值: {total:,.0f}
      </span>
    </div>""", unsafe_allow_html=True)

    if not positions:
        st.info("目前無持倉")
        return

    html = """<table class="data-table"><thead><tr>
        <th>代號</th><th>路徑</th><th>進場日</th><th>進場價</th>
        <th>現價</th><th>停利①</th><th>停損</th>
        <th>EV進場%</th><th>EV現%</th><th>持天</th><th>報酬%</th><th>出場訊號</th>
    </tr></thead><tbody>"""

    for p in sorted(positions, key=lambda x: x.get("curr_ret_pct",0), reverse=True):
        sym   = _html_escape.escape(str(p.get("symbol","—")))
        path  = p.get("path","—")
        edate = p.get("entry_date","—")
        epx   = p.get("entry_price",0)
        curr  = p.get("curr_price",0)
        tp1   = p.get("tp1_price",0)
        stop  = p.get("stop_price",0)
        ev_e  = p.get("ev_entry",0)
        ev_n  = p.get("ev_now",0)
        days  = p.get("days_held",0)
        ret   = p.get("curr_ret_pct",0)
        sig   = _html_escape.escape(p.get("exit_signal","—"))
        rc    = "c-g" if ret>0 else "c-r"
        sc    = "c-r" if sig not in ("—","") else "c-b"
        tp1_c = "c-g" if curr >= tp1 else ""
        stop_c= "c-r" if curr <= stop else ""
        watch = "★ " if p.get("is_watchlist") else ""
        path_c= "c-g" if path=="45" else "c-b"
        html += f"""<tr>
            <td><b>{watch}{sym}</b></td>
            <td class="mono {path_c}">{path}</td>
            <td class="mono" style="font-size:0.75rem;">{edate}</td>
            <td class="mono">{epx:.1f}</td>
            <td class="mono">{curr:.1f}</td>
            <td class="mono {tp1_c}">{tp1:.1f}</td>
            <td class="mono {stop_c}">{stop:.1f}</td>
            <td class="mono c-b">{ev_e:.2f}%</td>
            <td class="mono {'c-g' if ev_n>ev_e else 'c-r'}">{ev_n:.2f}%</td>
            <td class="mono">{days}</td>
            <td class="mono {rc}">{ret:+.2f}%</td>
            <td class="mono {sc}">{sig}</td>
        </tr>"""
    html += "</tbody></table>"
    _render_html(html, height=min(80+n*46, 600))


# ──────────────────────────────────────────────────────────────
# Section: 回測績效圖
# ──────────────────────────────────────────────────────────────

def render_backtest(backtest: dict, trades_df):
    st.markdown("""
    <div class="sec-header" style="border-color:#7c3aed;">
      <span style="font-size:1.05rem;font-weight:900;color:#7c3aed;">
        📈 回測績效
      </span>
    </div>""", unsafe_allow_html=True)

    if not backtest:
        st.info("⏳ 尚無回測結果（需執行 backtest_engine.py）")
        return

    # 績效指標
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("總交易筆", backtest.get("total_trades","—"))
    c2.metric("勝率",  f"{backtest.get('win_rate_pct',0):.1f}%")
    c3.metric("總報酬", f"{backtest.get('total_ret_pct',0):+.2f}%")
    c4.metric("年化",  f"{backtest.get('cagr_pct',0):+.2f}%")
    c5.metric("Sharpe", f"{backtest.get('sharpe',0):.2f}")
    c6.metric("最大回撤", f"{backtest.get('max_drawdown_pct',0):.2f}%")

    # Equity Curve
    eq = backtest.get("equity_curve", [])
    if eq:
        df_eq = pd.DataFrame(eq)
        initial = backtest.get("initial_capital", 1_000_000)
        df_eq["ret_pct"] = (df_eq["total_val"] / initial - 1) * 100

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True,
                            subplot_titles=["組合淨值曲線 (報酬%)", "每日部位數"])
        fig.add_scatter(x=df_eq["date"], y=df_eq["ret_pct"],
                        name="報酬%", line=dict(color="#2563eb", width=2),
                        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
                        row=1, col=1)
        fig.add_bar(x=df_eq["date"], y=df_eq["n_pos"],
                    name="持倉數", marker_color="rgba(5,150,105,0.5)",
                    row=2, col=1)
        fig.update_layout(
            height=380, margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(color="#475569", size=11),
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 出場方式分布
    breakdown = backtest.get("exit_breakdown", {})
    if breakdown:
        labels = list(breakdown.keys())
        values = list(breakdown.values())
        color_map = {
            "SELL_STOP":    "#dc2626",
            "SELL_TP1":     "#059669",
            "SELL_TP2":     "#047857",
            "SELL_TRAIL":   "#d97706",
            "SELL_EV":      "#7c3aed",
            "SELL_REPLACE": "#0891b2",
        }
        colors = [color_map.get(l,"#94a3b8") for l in labels]
        fig2 = go.Figure(go.Pie(labels=labels, values=values,
                                marker_colors=colors,
                                textinfo="label+percent",
                                hole=0.4))
        fig2.update_layout(
            title="出場方式分佈", height=280,
            margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#475569"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 交易歷史表格
    if trades_df is not None and not trades_df.empty:
        st.markdown("#### 📋 交易歷史紀錄")
        df_show = trades_df.copy()
        if "ret_pct" in df_show.columns:
            df_show = df_show.sort_values("date", ascending=False)
        st.dataframe(
            df_show[["date","symbol","action","price","shares","ret_pct","reason","path","days_held"]]
            if all(c in df_show.columns for c in ["date","symbol","action","price","shares","ret_pct","reason","path","days_held"])
            else df_show,
            use_container_width=True, height=300,
        )


# ──────────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────────

def main():
    portfolio, v4, v12, regime, backtest, trades_df = load_all()
    portfolio = portfolio or {}

    # Header
    gen_at = portfolio.get("generated_at","—")
    n_pos  = portfolio.get("n_positions", 0)
    regime_label = (regime or {}).get("label","—")
    total  = portfolio.get("total_val", 0)

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;padding:16px 24px;
                background:#ffffff;border:1px solid rgba(59,130,246,0.2);
                border-radius:14px;margin-bottom:12px;
                border-top:3px solid #2563eb;">
      <div>
        <div style="font-size:1.5rem;font-weight:900;color:#1e293b;">
          📊 資源法 AI 戰情室 <span style="font-size:0.9rem;color:#94a3b8;">v5.0</span>
        </div>
        <div style="font-size:0.72rem;color:#94a3b8;font-family:monospace;margin-top:3px;">
          Portfolio Manager · Buy/Sell Reason Display · Backtest Engine
        </div>
      </div>
      <div style="margin-left:auto;display:flex;gap:10px;flex-wrap:wrap;">
        <span class="pill pill-b">更新: {gen_at}</span>
        <span class="pill pill-a">{regime_label}</span>
        <span class="pill pill-g">持倉: {n_pos}</span>
        <span class="pill pill-p">總值: {total:,.0f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 刷新資料"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "▲ 今日買進原因",
        "▼ 賣出訊號",
        "📋 持倉監控",
        "📈 回測績效"
    ])

    with tab1:
        render_buy_reasons(portfolio)
    with tab2:
        render_sell_signals(portfolio)
    with tab3:
        render_positions(portfolio)
    with tab4:
        render_backtest(backtest, trades_df)


if __name__ == "__main__":
    main()
