"""
app.py — 資源法 AI 戰情室
Streamlit 主程式 | 台股/美股雙軌 | 四層數據防火牆
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import time

# ===========================================================================
# 環境變數保密機制
# 優先順序: Streamlit Secrets > .env 檔案 > 使用者手動輸入
# ===========================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()  # 載入本地 .env（開發環境用）
except ImportError:
    pass  # 生產環境不需要 python-dotenv

def _get_secret(key: str, default: str = "") -> str:
    """
    安全讀取金鑰：
    1. Streamlit Cloud Secrets（st.secrets）
    2. 系統環境變數 / .env
    3. 回傳 default（空字串）
    """
    # Streamlit Cloud 部署時使用 st.secrets
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        pass
    # 本地開發：讀 .env 或系統環境變數
    return os.environ.get(key, default)

# 在啟動時預載金鑰（不顯示在介面上）
_ENV_GEMINI_KEY    = _get_secret("GEMINI_API_KEY")
_ENV_FINMIND_TOKEN = _get_secret("FINMIND_TOKEN")

# 本地引擎
from engine_21 import (
    fetch_stock_data,
    stage1_energy_filter,
    stage2_path_filter,
    get_decision,
    get_market_sentiment,
    resolve_symbol,
    _INST_CACHE,          # 投信快取單例
)

# ===========================================================================
# 頁面設定
# ===========================================================================
st.set_page_config(
    page_title="資源法 AI 戰情室",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# 全域 CSS (軍事情報室風格：深色 + 螢光黃/青)
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+TC:wght@400;700;900&display=swap');

:root {
    --bg-dark:    #0a0d14;
    --bg-panel:   #0e1320;
    --bg-card:    #131929;
    --accent:     #f0c040;
    --accent2:    #00e5ff;
    --green:      #00e676;
    --red:        #ff1744;
    --text:       #cdd6f4;
    --text-dim:   #6c7a99;
    --border:     #1e2840;
    --radius:     8px;
}

html, body, .stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
    font-family: 'Noto Sans TC', 'Share Tech Mono', monospace;
}

/* 側欄 */
[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border);
}

/* 標題 */
h1, h2, h3 { color: var(--accent) !important; letter-spacing: 1px; }
h4, h5, h6 { color: var(--accent2) !important; }

/* Metric */
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }

/* 按鈕 */
.stButton > button {
    background: linear-gradient(135deg, #1a2340, #0e1320) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg-dark) !important;
    box-shadow: 0 0 15px rgba(240,192,64,0.4) !important;
}

/* Tabs */
[data-testid="stTab"] button {
    color: var(--text-dim) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* 股票卡片 */
.stock-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'Share Tech Mono', monospace;
    transition: border-color 0.2s;
}
.stock-card:hover { border-left-color: var(--accent); }
.stock-card.bearish { border-left-color: var(--red); }
.stock-card.bullish { border-left-color: var(--green); }

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.ticker-name { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
.price-tag { font-size: 1.2rem; color: var(--text); }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.badge-bull { background: rgba(0,230,118,0.15); color: var(--green); border: 1px solid var(--green); }
.badge-bear { background: rgba(255,23,68,0.15); color: var(--red); border: 1px solid var(--red); }
.badge-neutral { background: rgba(240,192,64,0.15); color: var(--accent); border: 1px solid var(--accent); }

.data-row { display: flex; gap: 20px; margin-top: 6px; flex-wrap: wrap; }
.data-item { font-size: 0.82rem; color: var(--text-dim); }
.data-item span { color: var(--accent2); font-weight: 700; }

.ev-bar {
    margin-top: 8px;
    font-size: 0.85rem;
    color: var(--text);
}

/* 大盤儀表版 */
.market-bar {
    display: flex;
    gap: 16px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 20px;
    margin-bottom: 20px;
    align-items: center;
    font-family: 'Share Tech Mono', monospace;
}
.market-stat { text-align: center; }
.market-stat .val { font-size: 1.3rem; font-weight: 900; }
.market-stat .lbl { font-size: 0.72rem; color: var(--text-dim); }
.bull-val { color: var(--green); }
.bear-val { color: var(--red); }
.neutral-val { color: var(--accent); }

/* AI 摘要框 */
.ai-summary {
    background: linear-gradient(135deg, #0e1320, #111827);
    border: 1px solid var(--accent2);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
}

/* 健康度指標 */
.health-ok  { color: var(--green); }
.health-err { color: var(--red); }
.health-warn{ color: var(--accent); }

/* 數據表格 */
.stDataFrame { background: var(--bg-card) !important; }

/* 輸入框 */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* Info/warning/error boxes */
.stAlert { border-radius: var(--radius) !important; }

/* 系統狀態列 */
.status-bar {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-dim);
    margin-bottom: 16px;
    display: flex;
    gap: 24px;
    align-items: center;
}
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
.dot-ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-err { background: var(--red); }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 常數與預設值
# ===========================================================================
DEFAULT_TW_WATCHLIST = [
    "3037", "6206", "3708", "8096", "3706", "2330", "2317", "2454",
    "2308", "2382", "3711", "2412", "3231", "2379", "3008", "2395",
    "3045", "2327", "2408", "6669", "3034", "2345", "2474", "4938",
    "3443", "2353", "2324", "2603", "2609", "6515", "3661", "3583",
    "6415", "3035", "6231", "1802", "3708", "2313", "2301", "2375", "8358"
]

DEFAULT_US_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL"]

BENCHMARK_TW = "0050.TW"
BENCHMARK_US = "SPY"

LOOKBACK_DAYS = 180
ALPHA_SEEDS_PATH = "alpha_seeds.json"


# ===========================================================================
# Session State 初始化
# ===========================================================================
def init_session():
    defaults = {
        "tw_watchlist": DEFAULT_TW_WATCHLIST.copy(),
        "us_watchlist": DEFAULT_US_WATCHLIST.copy(),
        "last_scan_time": None,
        "scan_results": {},
        "benchmark_tw_df": None,
        "benchmark_us_df": None,
        "market_sentiment_tw": None,
        "market_sentiment_us": None,
        "ai_summary": "",
        "data_health": {},
        "all_warnings": [],
        "selected_stock": None,
        "target_date": datetime.today().strftime("%Y-%m-%d"),
        "gemini_api_key": _ENV_GEMINI_KEY,  # 優先從環境變數讀取
        "active_market": "TW",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ===========================================================================
# 數據獲取 & 快取
# ===========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch(symbol: str, start_str: str, end_str: str) -> dict:
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    return fetch_stock_data(symbol, start, end)


def get_date_range(target_date_str: str, lookback: int = LOOKBACK_DAYS):
    end_dt   = datetime.strptime(target_date_str, "%Y-%m-%d") + timedelta(days=1)
    start_dt = end_dt - timedelta(days=lookback)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ===========================================================================
# Gemini AI 整合
# ===========================================================================
def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ 請在側欄輸入 Gemini API Key 才能啟用 AI 分析。"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemma-3-27b-it")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini 呼叫失敗：{e}"


def build_gemini_prompt(scan_results: dict, market_sentiment: dict,
                        market: str = "TW") -> str:
    top_stocks = []
    for sym, res in scan_results.items():
        if res.get("stage1", {}).get("pass") and res.get("decision"):
            dec = res["decision"]
            s1  = res["stage1"]
            s2  = res.get("stage2", {})
            top_stocks.append(
                f"- {sym}：方向={dec['direction']}，斜率Z={dec['slope_z']:.2f}，"
                f"VRI={dec['vri']:.1f}，PVO={dec['pvo']:.2f}，EV={s2.get('ev','N/A')}%，"
                f"路徑={s2.get('path','N/A')}"
            )

    stocks_text = "\n".join(top_stocks[:15]) or "（本次掃描無通過篩選標的）"

    m_label = market_sentiment.get("label", "不明") if market_sentiment else "不明"
    slope_5d = market_sentiment.get("slope_5d", 0) if market_sentiment else 0

    prompt = f"""你是一位專業的量化交易分析師，請根據以下市場數據給出簡潔的中文綜合決策建議。

【大盤狀態】
市場：{"台股" if market == "TW" else "美股"}
情緒：{m_label}
5日斜率：{slope_5d:.3f}

【通過篩選的潛力標的】（Slope Z > 0.5，VRI 健康，PVO 向上）
{stocks_text}

【你的任務】
1. 全程以以股票分析師與統計學專家 用 2 句話評估當前大盤風險
2. 列出最值得關注的 1-3 檔並說明理由（50字以內/檔）
3. 給出本日整體持倉水位建議（0%-100%）
4. 一句警示（如果大盤偏弱）

請保持簡潔，避免廢話，用數字說話，確認股票名稱。"""
    return prompt


# ===========================================================================
# 掃描核心
# ===========================================================================
def run_scan(watchlist: list, target_date: str, market_type: str, progress_bar=None):
    start_str, end_str = get_date_range(target_date)
    results = {}
    warns_all = []
    health_all = {}

    # 先抓基準
    bm_sym = BENCHMARK_TW if market_type == "TW" else BENCHMARK_US
    bm_res = cached_fetch(bm_sym, start_str, end_str)
    bm_df  = bm_res.get("indicator_df")
    if bm_df is not None and market_type == "TW":
        st.session_state.benchmark_tw_df = bm_df
        st.session_state.market_sentiment_tw = get_market_sentiment(bm_df)
    elif bm_df is not None:
        st.session_state.benchmark_us_df = bm_df
        st.session_state.market_sentiment_us = get_market_sentiment(bm_df)

    bm_close = bm_df['Close'] if bm_df is not None else None

    # ── 台股：批量預載投信資料（FINMIND_TOKEN 有設定才執行）──────────────
    if market_type == "TW" and _ENV_FINMIND_TOKEN:
        # 解析成 yahoo symbol 以便去除後綴
        yahoo_syms = []
        for s in watchlist:
            try:
                ys, _ = resolve_symbol(s)
                yahoo_syms.append(ys)
            except Exception:
                yahoo_syms.append(s)
        _INST_CACHE.batch_init(yahoo_syms)

    total = len(watchlist)
    for i, sym in enumerate(watchlist):
        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"分析 {sym}...")

        res = cached_fetch(sym, start_str, end_str)
        warns_all.extend(res.get("sanity_warns", []))
        health_all[sym] = res.get("health", {"pass": False, "issues": []})

        if res.get("error") or res.get("indicator_df") is None:
            results[sym] = {"error": res.get("error", "無數據"), "symbol": sym}
            continue

        df = res["indicator_df"]
        decision = get_decision(df, market=market_type)
        s1 = stage1_energy_filter(df)
        s2 = stage2_path_filter(sym, s1, ALPHA_SEEDS_PATH)

        # ── 投信買賣超（台股限定）──────────────────────────────────────────
        trust_info = {"trust_net_10d": None, "trust_df": None}
        if market_type == "TW":
            sid = sym.replace(".TWO", "").replace(".TW", "").strip()
            trust_net_10d = _INST_CACHE.get_recent_net(sid, days=10)
            trust_df      = _INST_CACHE.get(sid)
            trust_info = {
                "trust_net_10d": trust_net_10d,   # 近10日累積買賣超（張）
                "trust_df":      trust_df,         # 完整 DataFrame 供圖表用
            }

        results[sym] = {
            "symbol":       sym,
            "yahoo_symbol": res.get("yahoo_symbol", sym),
            "market":       market_type,
            "indicator_df": df,
            "raw_df":       res.get("raw_df"),
            "decision":     decision,
            "stage1":       s1,
            "stage2":       s2,
            "health":       health_all[sym],
            "trust":        trust_info,           # 投信資料
        }

    st.session_state.scan_results    = results
    st.session_state.last_scan_time  = datetime.now().strftime("%H:%M:%S")
    st.session_state.data_health     = health_all
    st.session_state.all_warnings    = warns_all
    return results


# ===========================================================================
# UI 元件
# ===========================================================================
def render_status_bar():
    last_scan = st.session_state.last_scan_time or "尚未掃描"
    total_warns = len(st.session_state.all_warnings)
    health_ok = sum(1 for h in st.session_state.data_health.values() if h.get("pass"))
    health_total = len(st.session_state.data_health)

    st.markdown(f"""
    <div class="status-bar">
        <span><span class="status-dot dot-ok"></span> 系統正常</span>
        <span>📡 最後掃描: <b>{last_scan}</b></span>
        <span>🔬 數據健康: <b class="{'health-ok' if health_ok == health_total else 'health-warn'}">{health_ok}/{health_total}</b></span>
        <span>⚠️ 數據警告: <b class="{'health-err' if total_warns > 0 else 'health-ok'}">{total_warns}</b></span>
    </div>
    """, unsafe_allow_html=True)


def render_market_bar(sentiment: dict, market: str):
    if not sentiment:
        return
    bear    = sentiment.get("bear", 33)
    neutral = sentiment.get("neutral", 34)
    bull    = sentiment.get("bull", 33)
    label   = sentiment.get("label", "震盪")
    s5d     = sentiment.get("slope_5d", 0)
    s20d    = sentiment.get("slope_20d", 0)

    st.markdown(f"""
    <div class="market-bar">
        <div class="market-stat">
            <div class="val neutral-val">{"台股" if market == "TW" else "美股"}</div>
            <div class="lbl">市場</div>
        </div>
        <div class="market-stat">
            <div class="val bear-val">{bear}%</div>
            <div class="lbl">P_熊</div>
        </div>
        <div class="market-stat">
            <div class="val neutral-val">{neutral}%</div>
            <div class="lbl">P_震</div>
        </div>
        <div class="market-stat">
            <div class="val bull-val">{bull}%</div>
            <div class="lbl">P_牛</div>
        </div>
        <div class="market-stat">
            <div class="val neutral-val">{label}</div>
            <div class="lbl">大盤情緒</div>
        </div>
        <div class="market-stat">
            <div class="val {'bull-val' if s5d > 0 else 'bear-val'}">{s5d:+.3f}</div>
            <div class="lbl">5日斜率</div>
        </div>
        <div class="market-stat">
            <div class="val {'bull-val' if s20d > 0 else 'bear-val'}">{s20d:+.3f}</div>
            <div class="lbl">20日斜率</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_card_class(direction: str) -> str:
    if direction == "做多": return "bullish"
    if direction == "做空": return "bearish"
    return ""


def get_badge(direction: str) -> str:
    if direction == "做多":
        return '<span class="badge badge-bull">▲ 做多</span>'
    if direction == "做空":
        return '<span class="badge badge-bear">▼ 做空</span>'
    return '<span class="badge badge-neutral">◆ 觀望</span>'


def render_stock_card(sym: str, res: dict):
    """渲染股票卡片 (HTML)"""
    if res.get("error"):
        st.markdown(f"""
        <div class="stock-card bearish">
            <div class="card-header">
                <span class="ticker-name">{sym}</span>
                <span class="badge badge-bear">❌ 錯誤</span>
            </div>
            <div style="color:#6c7a99;font-size:0.8rem;">{res['error']}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    dec = res.get("decision", {})
    s1  = res.get("stage1", {})
    s2  = res.get("stage2", {})
    health = res.get("health", {})
    market = res.get("market", "TW")
    trust  = res.get("trust", {})

    direction  = dec.get("direction", "觀望")
    action     = dec.get("action", "---")
    close_px   = dec.get("close", 0)
    pvo        = dec.get("pvo", 0)
    vri        = dec.get("vri", 0)
    slope      = dec.get("slope", 0)
    slope_z    = dec.get("slope_z", 0)
    pvo_status = dec.get("pvo_status", "")
    vri_status = dec.get("vri_status", "")
    sig_level  = dec.get("signal_level", "")
    last_action= dec.get("last_action", "---")
    ev         = s2.get("ev", "N/A")
    path       = s2.get("path", "---")
    t_stat     = s2.get("t_stat", None)
    s1_pass    = "✅" if s1.get("pass") else "❌"
    s2_pass    = "✅" if s2.get("pass") else "❌"
    health_icon= "✅" if health.get("pass") else "⚠️"

    # 投信買賣超（台股限定）
    trust_net_10d = trust.get("trust_net_10d", None)
    if trust_net_10d is not None:
        trust_color = "#00e676" if trust_net_10d > 0 else ("#ff1744" if trust_net_10d < 0 else "#6c7a99")
        trust_label = f"近10日投信: <span style='color:{trust_color};font-weight:700;'>{trust_net_10d:+,.0f} 張</span>"
    else:
        trust_label = ""

    card_cls = get_card_class(direction)
    badge    = get_badge(direction)

    # PVO 判讀提示
    pvo_hint  = "（>10 主力層級 / >0 資金流入 / <0 縮量）"
    vri_hint  = "（40-75 健康 / >90 過熱 / <40 冷淡）"
    slope_hint= "（>0.5% 趨勢翻正 / <0 下行）"

    t_stat_str = f"t={t_stat:.1f}" if t_stat is not None else "N/A"
    pvo_color  = "#00e676" if pvo > 10 else ("#f0c040" if pvo > 0 else "#ff1744")
    vri_color  = "#00e676" if 40 <= vri <= 75 else ("#ff1744" if vri > 90 else "#f0c040")
    slope_color= "#00e676" if slope > 0 else "#ff1744"
    ev_color   = "#00e676" if (isinstance(ev, (int, float)) and ev > 3) else "#f0c040"

    mkt_tag = "🇹🇼" if market == "TW" else "🇺🇸"

    st.markdown(f"""
    <div class="stock-card {card_cls}">
        <div class="card-header">
            <span class="ticker-name">{mkt_tag} {sym}</span>
            <span class="price-tag">NT$ {close_px:,.2f}" if market == "TW" else f"$ {close_px:,.2f}</span>
            {badge}
        </div>
        <div style="font-size:0.85rem; color:#00e5ff; margin-bottom:6px;">
            [ AI 判定: {action} ] {sig_level} | 前次: {last_action}
        </div>
        <div class="data-row">
            <div class="data-item">狀態: <span>{pvo_status}</span> / <span>{vri_status}</span></div>
        </div>
        <div class="data-row">
            <div class="data-item">PVO: <span style="color:{pvo_color}">{pvo:+.2f}</span>
                <small style="color:#4a5568">{pvo_hint}</small></div>
        </div>
        <div class="data-row">
            <div class="data-item">VRI: <span style="color:{vri_color}">{vri:.1f}</span>
                <small style="color:#4a5568">{vri_hint}</small></div>
            <div class="data-item">Slope: <span style="color:{slope_color}">{slope:+.3f}%</span>
                <small style="color:#4a5568">{slope_hint}</small></div>
            <div class="data-item">Slope Z: <span>{slope_z:+.2f}</span></div>
        </div>
        <div class="data-row" style="margin-top:8px;">
            <div class="data-item">階段1篩選: <span>{s1_pass}</span></div>
            <div class="data-item">V12.1路徑: <span>{s2_pass} {path} {t_stat_str}</span></div>
            <div class="data-item">數據健康: <span>{health_icon}</span></div>
        </div>
        <div class="ev-bar">
            💰 EV 期望值: <span style="color:{ev_color};font-weight:700;">
                {f'+{ev:.1f}%' if isinstance(ev,(int,float)) else ev}
            </span>
            {"&nbsp;&nbsp;|&nbsp;&nbsp; 🏦 " + trust_label if trust_label else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# K 線圖 (Plotly)
# ===========================================================================
def render_kline_chart(sym: str, res: dict):
    df = res.get("indicator_df")
    if df is None or df.empty:
        st.warning("無法取得圖表數據")
        return

    df = df.tail(90)  # 顯示最近 90 日

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(f"{sym} K線圖", "PVO (量能動能)", "VRI (資金強度)")
    )

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'],  close=df['Close'],
        name="K線",
        increasing_line_color='#00e676',
        decreasing_line_color='#ff1744',
    ), row=1, col=1)

    # 20MA
    ma20 = df['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ma20, name="MA20",
        line=dict(color='#f0c040', width=1, dash='dot')
    ), row=1, col=1)

    # Slope 信號
    bullish_idx = df[df['Slope'] > 0.1]
    fig.add_trace(go.Scatter(
        x=bullish_idx.index, y=bullish_idx['Close'],
        mode='markers', name="Slope>0.1",
        marker=dict(color='#00e5ff', size=5, symbol='circle')
    ), row=1, col=1)

    # PVO
    colors_pvo = ['#00e676' if v >= 0 else '#ff1744' for v in df['PVO']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['PVO'], name="PVO",
        marker_color=colors_pvo, opacity=0.8
    ), row=2, col=1)
    fig.add_hline(y=10,  line_dash="dot", line_color="#f0c040", row=2, col=1)
    fig.add_hline(y=0,   line_dash="dot", line_color="#6c7a99", row=2, col=1)
    fig.add_hline(y=-10, line_dash="dot", line_color="#ff7043", row=2, col=1)

    # VRI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['VRI'], name="VRI",
        line=dict(color='#00e5ff', width=2), fill='tozeroy',
        fillcolor='rgba(0,229,255,0.08)'
    ), row=3, col=1)
    fig.add_hrect(y0=40, y1=75, fillcolor="rgba(0,230,118,0.07)",
                  line_width=0, row=3, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color="#00e676", row=3, col=1)
    fig.add_hline(y=75, line_dash="dot", line_color="#00e676", row=3, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor='#0a0d14',
        plot_bgcolor='#0e1320',
        font=dict(color='#cdd6f4', family='Share Tech Mono'),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font_size=11),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor='#1e2840', showgrid=True)
    fig.update_yaxes(gridcolor='#1e2840', showgrid=True)

    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 數據健康度面板 (Layer 4)
# ===========================================================================
def render_health_panel():
    st.markdown("### 🔬 數據健康度指標（Layer 4 防火牆）")

    if not st.session_state.data_health:
        st.info("請先執行掃描")
        return

    health_data = []
    for sym, h in st.session_state.data_health.items():
        health_data.append({
            "代號": sym,
            "狀態": "✅ 正常" if h.get("pass") else "⚠️ 異常",
            "問題": " | ".join(h.get("issues", [])) or "—"
        })

    df_health = pd.DataFrame(health_data)
    st.dataframe(df_health, use_container_width=True, hide_index=True)

    if st.session_state.all_warnings:
        st.markdown("#### ⚠️ 數據清洗警告")
        for w in st.session_state.all_warnings:
            st.warning(w)


# ===========================================================================
# 側欄設定
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ 資源法 AI 戰情室")
        st.markdown("---")

        # 市場切換
        st.markdown("### 🌐 市場選擇")
        market = st.radio("", ["🇹🇼 台股", "🇺🇸 美股"],
                           horizontal=True,
                           index=0 if st.session_state.active_market == "TW" else 1)
        st.session_state.active_market = "TW" if "台股" in market else "US"

        st.markdown("---")

        # 目標日期
        st.markdown("### 📅 分析日期")
        target = st.date_input("資料截止日",
                                value=datetime.strptime(
                                    st.session_state.target_date, "%Y-%m-%d"),
                                max_value=datetime.today())
        st.session_state.target_date = target.strftime("%Y-%m-%d")

        st.markdown(f"回溯天數: **{LOOKBACK_DAYS}** 日")

        st.markdown("---")

        # 自訂觀察名單
        st.markdown("### 📋 觀察名單")
        if st.session_state.active_market == "TW":
            tw_input = st.text_area(
                "台股代號（逗號分隔）",
                value=", ".join(st.session_state.tw_watchlist),
                height=120
            )
            parsed = [s.strip() for s in tw_input.replace('\n', ',').split(',') if s.strip()]
            if parsed:
                st.session_state.tw_watchlist = parsed
            st.caption(f"共 {len(st.session_state.tw_watchlist)} 檔")
        else:
            us_input = st.text_area(
                "美股代號（逗號分隔）",
                value=", ".join(st.session_state.us_watchlist),
                height=80
            )
            parsed = [s.strip().upper() for s in us_input.replace('\n', ',').split(',') if s.strip()]
            if parsed:
                st.session_state.us_watchlist = parsed
            st.caption(f"共 {len(st.session_state.us_watchlist)} 檔")

        st.markdown("---")

        # Gemini API Key
        st.markdown("### 🤖 Gemini AI 設定")

        if _ENV_GEMINI_KEY:
            # 環境變數已設定，顯示遮罩提示，不讓使用者看到金鑰
            st.success("✅ API Key 已從環境變數自動載入")
            if st.checkbox("手動覆蓋 API Key（選填）", value=False):
                api_key = st.text_input("Gemini API Key（覆蓋用）",
                                         type="password",
                                         placeholder="留空則使用環境變數金鑰")
                if api_key:
                    st.session_state.gemini_api_key = api_key
                else:
                    st.session_state.gemini_api_key = _ENV_GEMINI_KEY
            else:
                st.session_state.gemini_api_key = _ENV_GEMINI_KEY
        else:
            # 環境變數未設定，讓使用者手動輸入
            api_key = st.text_input("Gemini API Key",
                                     type="password",
                                     value=st.session_state.gemini_api_key,
                                     placeholder="AIza...")
            st.session_state.gemini_api_key = api_key
            if api_key:
                st.success("✅ API Key 已設定")
            else:
                st.caption("未設定 → 使用規則引擎模式")

        st.markdown("---")

        # Alpha Seeds 上傳
        st.markdown("### 🗂️ V12.1 Alpha Seeds")
        uploaded = st.file_uploader("上傳 alpha_seeds.json",
                                     type=["json"],
                                     help="由盤後 V12.1 系統產生後上傳")
        if uploaded:
            try:
                data = json.load(uploaded)
                with open(ALPHA_SEEDS_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                st.success(f"✅ 已更新 {len(data)} 筆種子")
            except Exception as e:
                st.error(f"解析失敗: {e}")

        if os.path.exists(ALPHA_SEEDS_PATH):
            with open(ALPHA_SEEDS_PATH, "r") as f:
                seeds = json.load(f)
            st.caption(f"📦 現有種子: {len(seeds)} 檔")

        st.markdown("---")
        st.caption("© 2026 資源法 AI 戰情室 v2.1")


# ===========================================================================
# 主頁面
# ===========================================================================
def main():
    render_sidebar()

    # 頁首
    col_title, col_scan = st.columns([4, 1])
    with col_title:
        st.markdown(f"# ⚡ 資源法 AI 戰情室")
        st.markdown(f"<small style='color:#6c7a99'>分析日期: {st.session_state.target_date} | "
                    f"{'台股模式 🇹🇼' if st.session_state.active_market == 'TW' else '美股模式 🇺🇸'}</small>",
                    unsafe_allow_html=True)
    with col_scan:
        scan_btn = st.button("🔄 執行全盤掃描", use_container_width=True, type="primary")

    # 狀態列
    render_status_bar()

    # 大盤情緒
    sentiment = (st.session_state.market_sentiment_tw
                 if st.session_state.active_market == "TW"
                 else st.session_state.market_sentiment_us)
    render_market_bar(sentiment, st.session_state.active_market)

    # 掃描邏輯
    if scan_btn:
        market = st.session_state.active_market
        wl = (st.session_state.tw_watchlist if market == "TW"
              else st.session_state.us_watchlist)

        with st.spinner("⚙️ 正在擷取數據並計算指標..."):
            pb = st.progress(0, text="初始化...")
            run_scan(wl, st.session_state.target_date, market, progress_bar=pb)
            pb.empty()
        st.rerun()

    # AI 掃描按鈕
    ai_col1, ai_col2 = st.columns([1, 4])
    with ai_col1:
        ai_btn = st.button("🤖 呼叫 Gemini 深度分析", use_container_width=True)
    with ai_col2:
        if st.session_state.ai_summary:
            st.markdown(f"""<div class="ai-summary">{st.session_state.ai_summary}</div>""",
                        unsafe_allow_html=True)

    if ai_btn:
        if not st.session_state.scan_results:
            st.warning("請先執行掃描")
        else:
            market = st.session_state.active_market
            sent   = (st.session_state.market_sentiment_tw if market == "TW"
                      else st.session_state.market_sentiment_us)
            prompt = build_gemini_prompt(
                st.session_state.scan_results, sent, market)
            with st.spinner("🤖 Gemini 分析中..."):
                summary = call_gemini(prompt, st.session_state.gemini_api_key)
            st.session_state.ai_summary = summary
            st.rerun()

    st.markdown("---")

    # ---- 主要 Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 視覺化圖表",
        "📋 數據與排名",
        "🎯 決策戰情室",
        "🔬 數據健康度"
    ])

    results = st.session_state.scan_results
    market  = st.session_state.active_market

    # ================================================================
    # Tab 1: 視覺化圖表
    # ================================================================
    with tab1:
        if not results:
            st.info("👈 請先在左側設定觀察名單並執行「全盤掃描」")
        else:
            # 股票選擇器
            sym_list = [s for s, r in results.items() if not r.get("error")]
            if sym_list:
                selected = st.selectbox(
                    "選擇個股查看 K 線圖",
                    sym_list,
                    index=0
                )
                if selected and selected in results:
                    render_kline_chart(selected, results[selected])

                    # 個股數據摘要
                    res = results[selected]
                    dec = res.get("decision", {})
                    s1  = res.get("stage1", {})
                    s2  = res.get("stage2", {})

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("現價", f"{dec.get('close',0):,.2f}")
                    col2.metric("PVO", f"{dec.get('pvo',0):+.2f}",
                                delta="↑" if dec.get("pvo", 0) > 0 else "↓")
                    col3.metric("VRI", f"{dec.get('vri',0):.1f}")
                    col4.metric("Slope Z", f"{dec.get('slope_z',0):+.2f}")
                    col5.metric("EV 期望值",
                                f"{s2.get('ev','N/A')}%" if isinstance(s2.get('ev'), (int,float))
                                else "N/A")

    # ================================================================
    # Tab 2: 數據與排名
    # ================================================================
    with tab2:
        if not results:
            st.info("請先執行掃描")
        else:
            # 篩選控制
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                show_all = st.checkbox("顯示全部（含未通過）", value=False)
            with col_f2:
                sort_by = st.selectbox("排序依據",
                    ["Slope Z", "VRI", "PVO", "EV期望值", "Score"], index=0)
            with col_f3:
                min_vri = st.slider("VRI 最低門檻", 0, 100, 40)

            # 建立排名表
            table_rows = []
            for sym, res in results.items():
                if res.get("error"):
                    continue
                dec   = res.get("decision", {})
                s1    = res.get("stage1", {})
                s2    = res.get("stage2", {})
                h     = res.get("health", {})
                trust = res.get("trust", {})

                if not show_all and not s1.get("pass"):
                    continue
                if dec.get("vri", 0) < min_vri:
                    continue

                table_rows.append({
                    "代號":    sym,
                    "市場":    "🇹🇼" if res.get("market") == "TW" else "🇺🇸",
                    "現價":    dec.get("close", 0),
                    "方向":    dec.get("direction", "---"),
                    "操作":    dec.get("action", "---"),
                    "PVO":     dec.get("pvo", 0),
                    "VRI":     dec.get("vri", 0),
                    "Slope":   dec.get("slope", 0),
                    "Slope Z": dec.get("slope_z", 0),
                    "Score":   dec.get("score", 0),
                    "EV%":     s2.get("ev", None),
                    "路徑":    s2.get("path", "N/A"),
                    "t值":     s2.get("t_stat", None),
                    "投信10日": trust.get("trust_net_10d", None),  # 近10日投信買賣超（張）
                    "S1":      "✅" if s1.get("pass") else "❌",
                    "S2":      "✅" if s2.get("pass") else "❌",
                    "健康":    "✅" if h.get("pass") else "⚠️",
                    "PVO狀態": dec.get("pvo_status", ""),
                    "VRI狀態": dec.get("vri_status", ""),
                    "日期":    dec.get("date", ""),
                })

            if table_rows:
                sort_col_map = {
                    "Slope Z": "Slope Z",
                    "VRI": "VRI",
                    "PVO": "PVO",
                    "EV期望值": "EV%",
                    "Score": "Score"
                }
                df_table = pd.DataFrame(table_rows)
                sort_col = sort_col_map.get(sort_by, "Slope Z")
                if sort_col in df_table.columns:
                    df_table = df_table.sort_values(sort_col, ascending=False, na_position='last')

                st.markdown(f"**共 {len(df_table)} 檔符合條件**")
                st.dataframe(
                    df_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "現價":    st.column_config.NumberColumn(format="%.2f"),
                        "PVO":     st.column_config.NumberColumn(format="%.2f"),
                        "VRI":     st.column_config.NumberColumn(format="%.1f"),
                        "Slope Z": st.column_config.NumberColumn(format="%.2f"),
                        "EV%":     st.column_config.NumberColumn(format="%.1f"),
                        "投信10日": st.column_config.NumberColumn(
                            label="投信10日(張)",
                            format="%+,.0f",
                            help="近10個交易日投信累積買賣超（正=買超/負=賣超）"
                        ),
                    }
                )

                # 下載 CSV
                csv = df_table.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ 下載 CSV",
                    csv,
                    file_name=f"scan_{st.session_state.target_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("無符合條件的標的，嘗試調整篩選條件")

    # ================================================================
    # Tab 3: 決策戰情室
    # ================================================================
    with tab3:
        if not results:
            st.info("請先執行掃描")
        else:
            # 分類顯示
            bull_stocks    = [(s, r) for s, r in results.items()
                               if r.get("decision", {}).get("direction") == "做多"
                               and r.get("stage1", {}).get("pass")]
            neutral_stocks = [(s, r) for s, r in results.items()
                               if r.get("decision", {}).get("direction") == "觀望"]
            bear_stocks    = [(s, r) for s, r in results.items()
                               if r.get("decision", {}).get("direction") == "做空"]

            st.markdown(f"### 🟢 做多訊號 ({len(bull_stocks)} 檔通過雙層篩選)")
            if bull_stocks:
                # 按 Slope Z 排序
                bull_stocks.sort(
                    key=lambda x: x[1].get("decision", {}).get("slope_z", 0),
                    reverse=True
                )
                for sym, res in bull_stocks:
                    render_stock_card(sym, res)
                    # 點擊查看詳情
                    if st.button(f"📈 查看 {sym} 詳細圖表", key=f"detail_{sym}"):
                        st.session_state.selected_stock = sym
                        # 在下方渲染
            else:
                st.info("目前無做多訊號通過篩選")

            st.markdown(f"### 🔴 做空/警示 ({len(bear_stocks)} 檔)")
            for sym, res in bear_stocks[:5]:  # 最多顯示5
                render_stock_card(sym, res)

            st.markdown(f"### ⚪ 觀望 ({len(neutral_stocks)} 檔)")
            with st.expander("展開觀望清單"):
                for sym, res in neutral_stocks[:20]:
                    dec = res.get("decision", {})
                    st.markdown(
                        f"**{sym}** — {dec.get('pvo_status','')} | "
                        f"Slope Z: {dec.get('slope_z',0):+.2f} | "
                        f"VRI: {dec.get('vri',0):.1f} | "
                        f"PVO: {dec.get('pvo',0):+.2f}"
                    )

            # 選中股票的詳細圖表
            if st.session_state.selected_stock in results:
                sym = st.session_state.selected_stock
                st.markdown(f"---\n### 📊 {sym} 詳細分析")
                render_kline_chart(sym, results[sym])

    # ================================================================
    # Tab 4: 數據健康度
    # ================================================================
    with tab4:
        render_health_panel()

        # Alpha Seeds 預覽
        st.markdown("---")
        st.markdown("### 🗂️ V12.1 Alpha Seeds 預覽")
        if os.path.exists(ALPHA_SEEDS_PATH):
            with open(ALPHA_SEEDS_PATH, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            df_seeds = pd.DataFrame(seeds)
            st.dataframe(df_seeds, use_container_width=True, hide_index=True)

            # 路徑分布餅圖
            if "Path" in df_seeds.columns:
                path_counts = df_seeds["Path"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=path_counts.index,
                    values=path_counts.values,
                    hole=0.4,
                    marker_colors=['#00e676','#f0c040','#ff1744','#00e5ff'],
                ))
                fig_pie.update_layout(
                    title="路徑分布",
                    template="plotly_dark",
                    paper_bgcolor='#0a0d14',
                    font=dict(color='#cdd6f4'),
                    height=300,
                    margin=dict(t=40, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("尚未上傳 alpha_seeds.json，請在側欄上傳")

        # 數據警告
        st.markdown("---")
        st.markdown("### ⚠️ 數據清洗日誌")
        if st.session_state.all_warnings:
            for w in st.session_state.all_warnings:
                st.warning(w)
        else:
            st.success("✅ 無異常數據警告")


if __name__ == "__main__":
    main()
