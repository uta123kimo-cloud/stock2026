"""
app.py — 資源法 AI 戰情室-0409-2
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
# ===========================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        pass
    return os.environ.get(key, default)

_ENV_GEMINI_KEY    = _get_secret("GEMINI_API_KEY")
_ENV_FINMIND_TOKEN = _get_secret("FINMIND_TOKEN")

from engine_21 import (
    fetch_stock_data,
    stage1_energy_filter,
    stage2_path_filter,
    get_decision,
    get_market_sentiment,
    resolve_symbol,
    _INST_CACHE,
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
# 全域 CSS — 淺色系精緻風格
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+TC:wght@400;600;700;900&display=swap');

:root {
    --bg-main:    #f4f6fb;
    --bg-panel:   #ffffff;
    --bg-card:    #f9fafе;
    --bg-card2:   #eef1f8;
    --accent:     #1a56db;
    --accent2:    #0891b2;
    --green:      #059669;
    --red:        #dc2626;
    --amber:      #d97706;
    --text:       #1e293b;
    --text-dim:   #64748b;
    --border:     #e2e8f0;
    --border2:    #cbd5e1;
    --radius:     10px;
    --shadow:     0 1px 4px rgba(30,41,59,0.08), 0 4px 16px rgba(30,41,59,0.04);
    --shadow-lg:  0 4px 20px rgba(30,41,59,0.12);
}

html, body, .stApp {
    background-color: var(--bg-main) !important;
    color: var(--text) !important;
    font-family: 'Noto Sans TC', 'Share Tech Mono', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border);
    box-shadow: 2px 0 8px rgba(30,41,59,0.06);
}

h1, h2, h3 { color: var(--accent) !important; letter-spacing: 0.5px; }
h4, h5, h6 { color: var(--accent2) !important; }

[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.82rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Noto Sans TC', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(26,86,219,0.25) !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    box-shadow: 0 4px 16px rgba(26,86,219,0.4) !important;
    transform: translateY(-1px);
}

[data-testid="stTab"] button,
[data-testid="stTab"] button p,
[data-testid="stTab"] button span {
    color: #1e293b !important;
    font-family: 'Noto Sans TC', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stTab"] button[aria-selected="true"],
[data-testid="stTab"] button[aria-selected="true"] p,
[data-testid="stTab"] button[aria-selected="true"] span {
    color: #000000 !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 700 !important;
}
/* Radio 按鈕（台股/美股）字色黑色 */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] p {
    color: #1e293b !important;
    font-weight: 500 !important;
}

/* 股票卡片 */
.stock-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, border-left-color 0.2s;
}
.stock-card:hover { box-shadow: var(--shadow-lg); }
.stock-card.bearish { border-left-color: var(--red); }
.stock-card.bullish { border-left-color: var(--green); }
.stock-card.final-pick {
    border-left-color: var(--accent);
    border: 2px solid var(--accent);
    background: linear-gradient(135deg, #eff6ff, #ffffff);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.ticker-name { font-size: 1.1rem; font-weight: 800; color: var(--accent); }
.price-tag { font-size: 1.1rem; color: var(--text); font-weight: 600; font-family: 'Share Tech Mono', monospace; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-bull { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.badge-bear { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.badge-neutral { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.badge-final { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }

.data-row { display: flex; gap: 20px; margin-top: 6px; flex-wrap: wrap; }
.data-item { font-size: 0.82rem; color: var(--text-dim); }
.data-item span { color: var(--accent2); font-weight: 700; }

.ev-bar { margin-top: 8px; font-size: 0.85rem; color: var(--text); }

/* 勝率標籤 */
.pattern-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 4px 2px;
}
.pattern-a { background: #d1fae5; color: #065f46; border: 1px solid #34d399; }
.pattern-b { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
.pattern-c { background: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
.pattern-final { background: #dbeafe; color: #1e40af; border: 1px solid #60a5fa; font-size: 0.85rem; padding: 5px 14px; }

/* 大盤儀表 */
.market-bar {
    display: flex;
    gap: 12px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    margin-bottom: 20px;
    align-items: center;
    box-shadow: var(--shadow);
    flex-wrap: wrap;
}
.market-stat { text-align: center; min-width: 60px; }
.market-stat .val { font-size: 1.2rem; font-weight: 900; font-family: 'Share Tech Mono', monospace; }
.market-stat .lbl { font-size: 0.7rem; color: var(--text-dim); margin-top: 2px; }
.bull-val { color: var(--green); }
.bear-val { color: var(--red); }
.neutral-val { color: var(--accent); }

/* AI 摘要 */
.ai-summary {
    background: linear-gradient(135deg, #eff6ff, #f8faff);
    border: 1px solid #bfdbfe;
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--text);
    box-shadow: var(--shadow);
}

/* 狀態列 */
.status-bar {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 16px;
    display: flex;
    gap: 24px;
    align-items: center;
    box-shadow: var(--shadow);
    flex-wrap: wrap;
}
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
.dot-ok { background: var(--green); box-shadow: 0 0 6px rgba(5,150,105,0.4); }
.dot-err { background: var(--red); }

.health-ok  { color: var(--green); font-weight: 700; }
.health-err { color: var(--red); font-weight: 700; }
.health-warn{ color: var(--amber); font-weight: 700; }

/* 決策戰情室標題區塊 */
.decision-header {
    background: linear-gradient(135deg, #1a56db08, #0891b208);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 18px;
    margin: 16px 0 8px 0;
}

/* 最終決策標記 */
.final-decision-box {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
    border: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 18px;
    margin: 8px 0;
    box-shadow: 0 4px 16px rgba(26,86,219,0.12);
}

/* 輸入框 */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    border-color: var(--border2) !important;
    border-radius: var(--radius) !important;
}

.stDataFrame { background: var(--bg-panel) !important; border-radius: var(--radius) !important; }
.stAlert { border-radius: var(--radius) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 常數
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
# 勝率型態辨識
# ===========================================================================
def classify_pattern(dec: dict) -> dict:
    """
    辨識三大高勝率型態：
    A: 資金流入 + 情緒整理 + 強力買進  → 勝率10%: 52.6%, 20%: 37.6%
    B: 主力點火 + 擁擠過熱 + 強力買進  → 勝率10%: 52.4%
    C: VRI擁擠過熱 + 強力買進          → 勝率10%: 59.0%, 20%: 42.6%
    """
    action     = dec.get("action", "")
    pvo_status = dec.get("pvo_status", "")
    vri_status = dec.get("vri_status", "")
    vri        = dec.get("vri", 0)
    pvo        = dec.get("pvo", 0)
    pvo_delta  = dec.get("pvo_delta", 0)

    is_strong_buy = (action == "強力買進")
    is_money_in   = ("資金流入" in pvo_status)        # PVO > 0 且 delta < 10
    is_fire       = ("主力點火" in pvo_status)         # pvo_delta > 10
    is_hot        = ("擁擠過熱" in vri_status)         # VRI > 90
    is_cool       = ("情緒整理" in vri_status)         # VRI < 40

    patterns = []

    # Pattern A: 資金流入 + 情緒整理 + 強力買進
    if is_strong_buy and is_money_in and is_cool:
        patterns.append({
            "code": "A",
            "label": "📈 資金流入＋情緒整理＋強力買進",
            "win10": 52.6, "win20": 37.6,
            "css": "pattern-a",
            "desc": "穩 + 爆發兼具｜主升段最佳"
        })

    # Pattern B: 主力點火 + 擁擠過熱 + 強力買進
    if is_strong_buy and is_fire and is_hot:
        patterns.append({
            "code": "B",
            "label": "🔥 主力點火＋擁擠過熱＋強力買進",
            "win10": 52.4, "win20": None,
            "css": "pattern-b",
            "desc": "超短線爆發最強（風險偏高）"
        })

    # Pattern C: VRI擁擠過熱 + 強力買進（最廣義，VRI>85 也納入）
    if is_strong_buy and (is_hot or vri > 85):
        patterns.append({
            "code": "C",
            "label": "🌡️ VRI擁擠過熱＋強力買進",
            "win10": 59.0, "win20": 42.6,
            "css": "pattern-c",
            "desc": "最高勝率組合"
        })

    return {
        "patterns": patterns,
        "is_key_pattern": len(patterns) > 0,
        "best_win10": max([p["win10"] for p in patterns], default=0),
    }


def translate_path(path_str: str) -> str:
    """將英文路徑狀態翻譯為中文"""
    mapping = {
        "Alive Stage1-Only": "存活(僅Stage1)",
        "Alive":             "存活",
        "Stage1-Only":       "僅Stage1",
        "Dead":              "淘汰",
        "N/A":               "未知",
    }
    if not path_str or path_str == "---":
        return path_str
    for eng, chn in mapping.items():
        if eng in str(path_str):
            return str(path_str).replace(eng, chn)
    return path_str


def calc_vri_ratio(df) -> float:
    """VRI波動 = 近20天內 VRI > 40（健康水溫下限，代表有效能量）的天數 / 20"""
    if df is None or df.empty or "VRI" not in df.columns:
        return 0.0
    recent = df["VRI"].tail(20)
    return round((recent > 40).sum() / min(len(recent), 20), 2)


def calc_pvo_ratio(df) -> float:
    """PVO波動 = 近20天內 PVO > 0（資金流入或主力點火）的天數 / 20"""
    if df is None or df.empty or "PVO" not in df.columns:
        return 0.0
    recent = df["PVO"].tail(20)
    return round((recent > 0).sum() / min(len(recent), 20), 2)


def is_final_candidate(dec: dict, s2: dict) -> bool:
    """判斷是否為最終決策候選（三大型態之一 + Stage2通過）"""
    pat = classify_pattern(dec)
    return pat["is_key_pattern"] and s2.get("pass", False)


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
        "gemini_api_key": _ENV_GEMINI_KEY,
        "active_market": "TW",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ===========================================================================
# 數據獲取
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
# Gemini AI — 精簡版 Prompt（Gemma 4 31B 為主）
# ===========================================================================
def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ 請在側欄輸入 Gemini API Key 才能啟用 AI 分析。"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # 優先 gemma-3-27b-it，fallback gemini-2.0-flash
        for model_name in ["gemma-3-27b-it", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception:
                continue
        return "❌ 所有模型均無法回應，請稍後重試"
    except Exception as e:
        return f"❌ Gemini 呼叫失敗：{e}"


def build_gemini_prompt(scan_results: dict, market_sentiment: dict,
                        market: str = "TW") -> str:
    """
    精簡版 Prompt — 只送三大高勝率型態的標的，減少雜訊。
    Gemma 4 31B 專責量化決策分析。
    """
    final_picks = []
    for sym, res in scan_results.items():
        dec = res.get("decision", {})
        s2  = res.get("stage2", {})
        pat = classify_pattern(dec)
        if not pat["is_key_pattern"]:
            continue
        pat_codes = "+".join([p["code"] for p in pat["patterns"]])
        final_picks.append(
            f"{sym}: 型態={pat_codes}, Slope Z={dec.get('slope_z',0):.2f}, "
            f"VRI={dec.get('vri',0):.1f}, PVO={dec.get('pvo',0):+.2f}, "
            f"EV={s2.get('ev','N/A')}%, 最高勝率10%={pat['best_win10']}%"
        )

    picks_text = "\n".join(final_picks[:10]) or "（無符合三大型態的標的）"
    label = market_sentiment.get("label", "不明") if market_sentiment else "不明"
    slope_5d = market_sentiment.get("slope_5d", 0) if market_sentiment else 0

    return f"""角色：量化交易分析師（統計學 + 技術分析專家）
市場：{"台股" if market == "TW" else "美股"}｜情緒：{label}｜5日斜率：{slope_5d:.3f}

三大高勝率型態候選標的：
{picks_text}

請直接回答（中文，不超過200字）：
1. 大盤風險評估（1句，含具體數字）
2. 最值得操作的1-2檔及理由（各30字）
3. 今日建議持倉水位（百分比）
4. 風險警示（1句）

格式要求：條列清晰，數字說話，禁止廢話。"""


# ===========================================================================
# 掃描核心
# ===========================================================================
def run_scan(watchlist: list, target_date: str, market_type: str, progress_bar=None):
    start_str, end_str = get_date_range(target_date)
    results = {}
    warns_all = []
    health_all = {}

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

    if market_type == "TW" and _ENV_FINMIND_TOKEN:
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

        trust_info = {"trust_net_10d": None, "trust_df": None}
        if market_type == "TW":
            sid = sym.replace(".TWO", "").replace(".TW", "").strip()
            trust_net_10d = _INST_CACHE.get_recent_net(sid, days=10)
            trust_df      = _INST_CACHE.get(sid)
            trust_info = {"trust_net_10d": trust_net_10d, "trust_df": trust_df}

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
            "trust":        trust_info,
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


def render_pattern_badges(patterns: list) -> str:
    """渲染勝率型態徽章 HTML"""
    if not patterns:
        return ""
    html = '<div style="margin:6px 0;">'
    for p in patterns:
        win20_str = f" / 勝率20%: {p['win20']}%" if p.get("win20") else ""
        html += f'<span class="{p["css"]} pattern-tag" title="{p["desc"]}">{p["label"]} 勝率10%: {p["win10"]}%{win20_str}</span>'
    html += "</div>"
    return html


def render_stock_card(sym: str, res: dict, show_final_badge: bool = False):
    """渲染股票卡片"""
    if res.get("error"):
        st.markdown(f"""
        <div class="stock-card bearish">
            <div class="card-header">
                <span class="ticker-name">{sym}</span>
                <span class="badge badge-bear">❌ 錯誤</span>
            </div>
            <div style="color:#94a3b8;font-size:0.8rem;">{res['error']}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    dec    = res.get("decision", {})
    s1     = res.get("stage1", {})
    s2     = res.get("stage2", {})
    health = res.get("health", {})
    market = res.get("market", "TW")
    trust  = res.get("trust", {})
    pat    = classify_pattern(dec)

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

    trust_net_10d = trust.get("trust_net_10d", None)
    if trust_net_10d is not None:
        trust_color = "#059669" if trust_net_10d > 0 else ("#dc2626" if trust_net_10d < 0 else "#64748b")
        trust_label = f"近10日投信: <span style='color:{trust_color};font-weight:700;'>{trust_net_10d:+,.0f} 張</span>"
    else:
        trust_label = ""

    card_extra = "final-pick" if (show_final_badge and pat["is_key_pattern"]) else get_card_class(direction)
    badge      = get_badge(direction)
    pattern_html = render_pattern_badges(pat["patterns"])

    pvo_color  = "#059669" if pvo > 10 else ("#0891b2" if pvo > 0 else "#dc2626")
    vri_color  = "#059669" if 40 <= vri <= 75 else ("#dc2626" if vri > 90 else "#d97706")
    slope_color= "#059669" if slope > 0 else "#dc2626"
    ev_color   = "#059669" if (isinstance(ev, (int, float)) and ev > 3) else "#d97706"
    mkt_tag    = "🇹🇼" if market == "TW" else "🇺🇸"
    t_stat_str = f"t={t_stat:.1f}" if t_stat is not None else "N/A"
    path = translate_path(str(path))
    df_ind = res.get("indicator_df")
    vri_ratio = calc_vri_ratio(df_ind)
    pvo_ratio = calc_pvo_ratio(df_ind)
    import html as _html_mod
    # 用 escape 防止 emoji / 特殊字元破壞 HTML 結構
    pvo_status_s = _html_mod.escape(str(pvo_status))
    vri_status_s = _html_mod.escape(str(vri_status))
    sig_level_s  = _html_mod.escape(str(sig_level))
    action_s     = _html_mod.escape(str(action))
    last_action_s= _html_mod.escape(str(last_action))

    final_badge_html = ""
    if show_final_badge and pat["is_key_pattern"] and s2.get("pass"):
        final_badge_html = f'<span class="badge badge-final">⭐ 最終決策候選</span>'

    ev_str = f"+{ev:.1f}%" if isinstance(ev, (int, float)) else str(ev)
    trust_html = f"&nbsp;&nbsp;|&nbsp;&nbsp; 🏦 {trust_label}" if trust_label else ""
    st.markdown(f"""
    <div class="stock-card {card_extra}">
        <div class="card-header">
            <span class="ticker-name">{mkt_tag} {sym}</span>
            <span class="price-tag">{"NT$" if market == "TW" else "$"} {close_px:,.2f}</span>
            {badge} {final_badge_html}
        </div>
        {pattern_html}
        <div style="font-size:0.84rem;color:#0891b2;margin-bottom:4px;font-weight:600;">
            AI判定: <b>{action_s}</b> {sig_level_s} &nbsp;|&nbsp; 前次: {last_action_s}
        </div>
        <div class="data-row">
            <div class="data-item">PVO狀態: <span>{pvo_status_s}</span></div>
            <div class="data-item">VRI狀態: <span>{vri_status_s}</span></div>
        </div>
        <div class="data-row">
            <div class="data-item">PVO: <span style="color:{pvo_color};font-weight:700;">{pvo:+.2f}</span></div>
            <div class="data-item">VRI: <span style="color:{vri_color};font-weight:700;">{vri:.1f}</span></div>
            <div class="data-item">Slope: <span style="color:{slope_color};">{slope:+.3f}%</span></div>
            <div class="data-item">Slope Z: <span style="color:#1a56db;font-weight:700;">{slope_z:+.2f}</span></div>
        </div>
        <div class="data-row" style="margin-top:6px;">
            <div class="data-item">VRI波動: <span style="color:#0891b2;font-weight:700;">{vri_ratio:.0%}</span>
                <small style="color:#94a3b8">（20日&gt;40天數/20）</small></div>
            <div class="data-item">PVO波動: <span style="color:#059669;font-weight:700;">{pvo_ratio:.0%}</span>
                <small style="color:#94a3b8">（20日&gt;0天數/20）</small></div>
        </div>
        <div class="data-row" style="margin-top:6px;">
            <div class="data-item">S1: <span>{s1_pass}</span></div>
            <div class="data-item">路徑: <span>{s2_pass} {path} {t_stat_str}</span></div>
            <div class="data-item">健康: <span>{health_icon}</span></div>
            <div class="data-item">💰 EV: <span style="color:{ev_color};font-weight:700;">{ev_str}</span>
                {trust_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# K 線圖
# ===========================================================================
def render_kline_chart(sym: str, res: dict):
    df = res.get("indicator_df")
    if df is None or df.empty:
        st.warning("無法取得圖表數據")
        return

    df = df.tail(90)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(f"{sym} K線圖", "PVO (量能動能)", "VRI (資金強度)")
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'],  close=df['Close'],
        name="K線",
        increasing_line_color='#059669',
        decreasing_line_color='#dc2626',
    ), row=1, col=1)

    ma20 = df['Close'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ma20, name="MA20",
        line=dict(color='#d97706', width=1.5, dash='dot')
    ), row=1, col=1)

    bullish_idx = df[df['Slope'] > 0.1]
    fig.add_trace(go.Scatter(
        x=bullish_idx.index, y=bullish_idx['Close'],
        mode='markers', name="Slope>0.1",
        marker=dict(color='#0891b2', size=5, symbol='circle')
    ), row=1, col=1)

    colors_pvo = ['#059669' if v >= 0 else '#dc2626' for v in df['PVO']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['PVO'], name="PVO",
        marker_color=colors_pvo, opacity=0.75
    ), row=2, col=1)
    fig.add_hline(y=10,  line_dash="dot", line_color="#d97706", row=2, col=1)
    fig.add_hline(y=0,   line_dash="dot", line_color="#94a3b8", row=2, col=1)
    fig.add_hline(y=-10, line_dash="dot", line_color="#f87171", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['VRI'], name="VRI",
        line=dict(color='#0891b2', width=2), fill='tozeroy',
        fillcolor='rgba(8,145,178,0.08)'
    ), row=3, col=1)
    fig.add_hrect(y0=40, y1=75, fillcolor="rgba(5,150,105,0.06)",
                  line_width=0, row=3, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color="#059669", row=3, col=1)
    fig.add_hline(y=75, line_dash="dot", line_color="#059669", row=3, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        font=dict(color='#1e293b', family='Noto Sans TC'),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font_size=11),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor='#e2e8f0', showgrid=True)
    fig.update_yaxes(gridcolor='#e2e8f0', showgrid=True)

    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 數據健康度面板
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
# 側欄
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ 資源法 AI 戰情室")
        st.markdown("---")

        st.markdown("### 🌐 市場選擇")
        market = st.radio("", ["🇹🇼 台股", "🇺🇸 美股"],
                           horizontal=True,
                           index=0 if st.session_state.active_market == "TW" else 1)
        st.session_state.active_market = "TW" if "台股" in market else "US"

        st.markdown("---")
        st.markdown("### 📅 分析日期")
        target = st.date_input("資料截止日",
                                value=datetime.strptime(st.session_state.target_date, "%Y-%m-%d"),
                                max_value=datetime.today())
        st.session_state.target_date = target.strftime("%Y-%m-%d")
        st.markdown(f"回溯天數: **{LOOKBACK_DAYS}** 日")

        st.markdown("---")
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
        st.markdown("### 🤖 Gemini AI 設定")
        if _ENV_GEMINI_KEY:
            st.success("✅ API Key 已從環境變數自動載入")
            if st.checkbox("手動覆蓋 API Key（選填）", value=False):
                api_key = st.text_input("Gemini API Key（覆蓋用）",
                                         type="password",
                                         placeholder="留空則使用環境變數金鑰")
                st.session_state.gemini_api_key = api_key if api_key else _ENV_GEMINI_KEY
            else:
                st.session_state.gemini_api_key = _ENV_GEMINI_KEY
        else:
            api_key = st.text_input("Gemini API Key", type="password",
                                     value=st.session_state.gemini_api_key, placeholder="AIza...")
            st.session_state.gemini_api_key = api_key
            if api_key:
                st.success("✅ API Key 已設定")
            else:
                st.caption("未設定 → 使用規則引擎模式")

        st.markdown("---")
        st.markdown("### 🗂️ V12.1 Alpha Seeds")
        st.caption(
            "📌 **用途**：Alpha Seeds 是 V12.1 路徑篩選（Stage2）的歷史回測種子庫，"
            "記錄哪些路徑組合具統計顯著性（t值/EV）。"
            "上傳後系統會以此進行 Stage2 比對，提升決策候選準確率。"
            "未上傳時退回規則引擎模式（Stage1 alone）。"
        )
        uploaded = st.file_uploader("上傳 alpha_seeds.json", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                with open(ALPHA_SEEDS_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                st.cache_data.clear()   # 清除快取，確保下次掃描套用新種子
                st.success(f"✅ 已更新 {len(data)} 筆種子，快取已重置，請重新執行掃描")
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

    col_title, col_scan = st.columns([4, 1])
    with col_title:
        st.markdown(f"# ⚡ 資源法 AI 戰情室")
        st.markdown(f"<small style='color:#64748b'>分析日期: {st.session_state.target_date} | "
                    f"{'台股模式 🇹🇼' if st.session_state.active_market == 'TW' else '美股模式 🇺🇸'}</small>",
                    unsafe_allow_html=True)
    with col_scan:
        scan_btn = st.button("🔄 執行全盤掃描", use_container_width=True, type="primary")

    render_status_bar()

    sentiment = (st.session_state.market_sentiment_tw
                 if st.session_state.active_market == "TW"
                 else st.session_state.market_sentiment_us)
    render_market_bar(sentiment, st.session_state.active_market)

    if scan_btn:
        market = st.session_state.active_market
        wl = (st.session_state.tw_watchlist if market == "TW" else st.session_state.us_watchlist)
        with st.spinner("⚙️ 正在擷取數據並計算指標..."):
            pb = st.progress(0, text="初始化...")
            run_scan(wl, st.session_state.target_date, market, progress_bar=pb)
            pb.empty()
        st.rerun()

    ai_col1, ai_col2 = st.columns([1, 4])
    with ai_col1:
        ai_btn = st.button("🤖 Gemini 深度分析", use_container_width=True)
    with ai_col2:
        if st.session_state.ai_summary:
            # 使用 st.markdown 原生渲染，避免 HTML 雜訊
            import html as _html
            _safe = _html.escape(st.session_state.ai_summary).replace('\n', '<br>')
            st.markdown(f"""<div class="ai-summary">{_safe}</div>""",
                        unsafe_allow_html=True)

    if ai_btn:
        if not st.session_state.scan_results:
            st.warning("請先執行掃描")
        else:
            market = st.session_state.active_market
            sent   = (st.session_state.market_sentiment_tw if market == "TW"
                      else st.session_state.market_sentiment_us)
            prompt = build_gemini_prompt(st.session_state.scan_results, sent, market)
            with st.spinner("🤖 Gemma 4 31B 分析中..."):
                summary = call_gemini(prompt, st.session_state.gemini_api_key)
            st.session_state.ai_summary = summary
            st.rerun()

    st.markdown("---")

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
            sym_list = [s for s, r in results.items() if not r.get("error")]
            if sym_list:
                selected = st.selectbox("選擇個股查看 K 線圖", sym_list, index=0)
                if selected and selected in results:
                    render_kline_chart(selected, results[selected])
                    res = results[selected]
                    dec = res.get("decision", {})
                    s2  = res.get("stage2", {})
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("現價", f"{dec.get('close',0):,.2f}")
                    col2.metric("PVO", f"{dec.get('pvo',0):+.2f}",
                                delta="↑" if dec.get("pvo", 0) > 0 else "↓")
                    col3.metric("VRI", f"{dec.get('vri',0):.1f}")
                    col4.metric("Slope Z", f"{dec.get('slope_z',0):+.2f}")
                    col5.metric("EV 期望值",
                                f"{s2.get('ev','N/A')}%" if isinstance(s2.get('ev'), (int,float)) else "N/A")

    # ================================================================
    # Tab 2: 數據與排名（加入勝率型態判斷）
    # ================================================================
    with tab2:
        if not results:
            st.info("請先執行掃描")
        else:
            # 勝率型態說明
            st.markdown("""
            <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:12px 18px;margin-bottom:16px;">
            <b style="color:#0369a1;">📊 高勝率型態識別</b>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;">
                <span class="pattern-a pattern-tag">📈 A: 資金流入＋情緒整理＋強力買進 &nbsp;勝率10%: 52.6% / 20%: 37.6%&nbsp; 主升段最佳</span>
                <span class="pattern-b pattern-tag">🔥 B: 主力點火＋擁擠過熱＋強力買進 &nbsp;勝率10%: 52.4%&nbsp; 超短線爆發</span>
                <span class="pattern-c pattern-tag">🌡️ C: VRI擁擠過熱＋強力買進 &nbsp;勝率10%: 59.0% / 20%: 42.6%&nbsp; 最高勝率</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                show_all = st.checkbox("顯示全部（含未通過）", value=False)
            with col_f2:
                sort_by = st.selectbox("排序依據",
                    ["最佳勝率10%", "Slope Z", "VRI", "PVO", "EV期望值", "Score"], index=0)
            with col_f3:
                min_vri = st.slider("VRI 最低門檻", 0, 100, 40)

            table_rows = []
            for sym, res in results.items():
                if res.get("error"):
                    continue
                dec   = res.get("decision", {})
                s1    = res.get("stage1", {})
                s2    = res.get("stage2", {})
                h     = res.get("health", {})
                trust = res.get("trust", {})
                pat   = classify_pattern(dec)

                if not show_all and not s1.get("pass"):
                    continue
                if dec.get("vri", 0) < min_vri:
                    continue

                # 型態標籤（最多顯示所有命中型態）
                pat_codes = " + ".join([p["code"] for p in pat["patterns"]]) if pat["patterns"] else "—"
                pat_desc  = " | ".join([p["label"].split(" ", 1)[-1] for p in pat["patterns"]]) if pat["patterns"] else "—"

                df_ind_tab = res.get("indicator_df")
                table_rows.append({
                    "代號":        sym,
                    "市場":        "🇹🇼" if res.get("market") == "TW" else "🇺🇸",
                    "現價":        dec.get("close", 0),
                    "方向":        dec.get("direction", "---"),
                    "操作":        dec.get("action", "---"),
                    "高勝率型態":  pat_codes,
                    "最佳勝率10%": pat["best_win10"] if pat["best_win10"] > 0 else None,
                    "型態說明":    pat_desc,
                    "PVO":         dec.get("pvo", 0),
                    "VRI":         dec.get("vri", 0),
                    "VRI波動":     calc_vri_ratio(df_ind_tab),
                    "PVO波動":     calc_pvo_ratio(df_ind_tab),
                    "Slope Z":     dec.get("slope_z", 0),
                    "Score":       dec.get("score", 0),
                    "EV%":         s2.get("ev", None),
                    "路徑":        translate_path(str(s2.get("path", "N/A"))),
                    "t值":         s2.get("t_stat", None),
                    "投信10日":    trust.get("trust_net_10d", None),
                    "S1":          "✅" if s1.get("pass") else "❌",
                    "S2":          "✅" if s2.get("pass") else "❌",
                    "健康":        "✅" if h.get("pass") else "⚠️",
                    "PVO狀態":     dec.get("pvo_status", ""),
                    "VRI狀態":     dec.get("vri_status", ""),
                    "日期":        dec.get("date", ""),
                })

            if table_rows:
                sort_col_map = {
                    "最佳勝率10%": "最佳勝率10%",
                    "Slope Z": "Slope Z", "VRI": "VRI", "PVO": "PVO",
                    "EV期望值": "EV%", "Score": "Score",
                }
                df_table = pd.DataFrame(table_rows)
                sort_col = sort_col_map.get(sort_by, "最佳勝率10%")
                # 命中型態優先，次要排序為所選欄位
                df_table["_has_pattern"] = df_table["高勝率型態"].ne("—").astype(int)
                secondary = sort_col if sort_col in df_table.columns else "Slope Z"
                df_table = df_table.sort_values(
                    ["_has_pattern", secondary],
                    ascending=[False, False],
                    na_position='last'
                ).drop(columns=["_has_pattern"])

                st.markdown(f"**共 {len(df_table)} 檔符合條件**　｜　"
                            f"🏆 命中高勝率型態: **{df_table['高勝率型態'].ne('—').sum()}** 檔")
                st.dataframe(
                    df_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "現價":        st.column_config.NumberColumn(format="%.2f"),
                        "PVO":         st.column_config.NumberColumn(format="%.2f"),
                        "VRI":         st.column_config.NumberColumn(format="%.1f"),
                        "VRI波動":     st.column_config.NumberColumn(format="%.0%", help="近20日VRI>40（有效能量）天數/20"),
                        "PVO波動":     st.column_config.NumberColumn(format="%.0%", help="近20日PVO>0(資金流入或點火)天數/20"),
                        "Slope Z":     st.column_config.NumberColumn(format="%.2f"),
                        "EV%":         st.column_config.NumberColumn(format="%.1f"),
                        "最佳勝率10%": st.column_config.NumberColumn(format="%.1f%%", help="命中型態的最高10%勝率"),
                        "投信10日":    st.column_config.NumberColumn(
                            label="投信10日(張)", format="%+,.0f",
                            help="近10個交易日投信累積買賣超（正=買超/負=賣超）"
                        ),
                        "高勝率型態":  st.column_config.TextColumn(help="A=資金流入+整理+強買 | B=點火+過熱+強買 | C=VRI過熱+強買"),
                    }
                )

                csv = df_table.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ 下載 CSV", csv,
                    file_name=f"scan_{st.session_state.target_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("無符合條件的標的，嘗試調整篩選條件")

    # ================================================================
    # Tab 3: 決策戰情室（以三大型態為核心）
    # ================================================================
    with tab3:
        if not results:
            st.info("請先執行掃描")
        else:
            # ── 區塊一：最終決策候選（三大型態 + Stage2通過）────────────
            final_picks = [
                (s, r) for s, r in results.items()
                if not r.get("error")
                and is_final_candidate(r.get("decision", {}), r.get("stage2", {}))
            ]
            # 按最佳勝率10%排序
            final_picks.sort(
                key=lambda x: classify_pattern(x[1].get("decision", {}))["best_win10"],
                reverse=True
            )

            st.markdown("""
            <div class="final-decision-box">
                <b style="color:#1a56db;font-size:1.05rem;">⭐ 最終決策候選</b>
                <span style="color:#64748b;font-size:0.85rem;margin-left:12px;">
                    條件：命中三大高勝率型態 + V12.1 Stage2 通過
                </span>
            </div>
            """, unsafe_allow_html=True)

            if final_picks:
                for sym, res in final_picks:
                    dec = res.get("decision", {})
                    s2  = res.get("stage2", {})
                    df_ind = res.get("indicator_df")
                    pat = classify_pattern(dec)
                    pat_labels = " ｜ ".join([p["label"] for p in pat["patterns"]])
                    win_str = f"最高勝率10%: {pat['best_win10']:.1f}%"
                    vri_ratio = calc_vri_ratio(df_ind)
                    pvo_ratio = calc_pvo_ratio(df_ind)
                    path_zh = translate_path(str(s2.get("path","N/A")))

                    # 緊湊摘要列（唯一顯示，移除重複卡片）
                    st.markdown(f"""
                    <div style="background:#eff6ff;border:1px solid #93c5fd;border-left:4px solid #1a56db;
                         border-radius:8px;padding:10px 16px;margin-bottom:6px;">
                        <b style="color:#1e40af;font-size:1rem;">🇹🇼 {sym}</b>
                        <span style="color:#64748b;font-size:0.82rem;margin-left:8px;">{pat_labels}</span>
                        <span class="pattern-c pattern-tag" style="float:right;">{win_str}</span>
                        <br>
                        <span style="font-size:0.83rem;color:#475569;">
                            Slope Z: <b style="color:#1a56db">{dec.get('slope_z',0):+.2f}</b> ｜
                            VRI: <b>{dec.get('vri',0):.1f}</b> ｜
                            PVO: <b>{dec.get('pvo',0):+.2f}</b> ｜
                            EV: <b style="color:#059669">{s2.get('ev','N/A')}%</b> ｜
                            路徑: {path_zh} ｜
                            VRI波動: <b style="color:#0891b2">{vri_ratio:.0%}</b> ｜
                            PVO波動: <b style="color:#059669">{pvo_ratio:.0%}</b>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🔍 目前無標的同時符合三大型態 + Stage2 條件，請查看下方做多訊號")

            st.markdown("---")

            # ── 區塊二：做多訊號（Stage1通過，含型態標籤）──────────────
            bull_stocks = [
                (s, r) for s, r in results.items()
                if not r.get("error")
                and r.get("decision", {}).get("direction") == "做多"
                and r.get("stage1", {}).get("pass")
            ]
            # 三大型態優先排序
            bull_stocks.sort(
                key=lambda x: (
                    classify_pattern(x[1].get("decision", {}))["best_win10"],
                    x[1].get("decision", {}).get("slope_z", 0)
                ),
                reverse=True
            )

            key_count = sum(1 for s, r in bull_stocks
                           if classify_pattern(r.get("decision", {}))["is_key_pattern"])

            st.markdown(f"""
            <div class="decision-header">
                <b>🟢 做多訊號</b>
                <span style="color:#64748b;font-size:0.85rem;margin-left:8px;">
                    共 {len(bull_stocks)} 檔通過 Stage1 篩選
                    ｜ 其中 <b style="color:#059669">{key_count}</b> 檔命中高勝率型態
                </span>
            </div>
            """, unsafe_allow_html=True)

            if bull_stocks:
                for sym, res in bull_stocks:
                    render_stock_card(sym, res, show_final_badge=True)
                    if st.button(f"📈 查看 {sym} 詳細圖表", key=f"detail_{sym}"):
                        st.session_state.selected_stock = sym
            else:
                st.info("目前無做多訊號通過篩選")

            # ── 區塊三：做空 / 觀望 ──────────────────────────────────
            bear_stocks = [(s, r) for s, r in results.items()
                           if not r.get("error")
                           and r.get("decision", {}).get("direction") == "做空"]
            neutral_stocks = [(s, r) for s, r in results.items()
                               if not r.get("error")
                               and r.get("decision", {}).get("direction") == "觀望"]

            if bear_stocks:
                st.markdown(f"### 🔴 做空/警示 ({len(bear_stocks)} 檔)")
                for sym, res in bear_stocks[:5]:
                    render_stock_card(sym, res)

            st.markdown(f"### ⚪ 觀望 ({len(neutral_stocks)} 檔)")
            with st.expander("展開觀望清單"):
                for sym, res in neutral_stocks[:20]:
                    dec = res.get("decision", {})
                    pat = classify_pattern(dec)
                    pat_tag = f" 型態{'/ '.join([p['code'] for p in pat['patterns']])}" if pat["patterns"] else ""
                    pvo_st = dec.get("pvo_status","")
                    st.markdown(
                        f"**{sym}**{pat_tag} — {pvo_st} ｜ "
                        f"Slope Z: `{dec.get('slope_z',0):+.2f}` ｜ "
                        f"VRI: `{dec.get('vri',0):.1f}` ｜ "
                        f"PVO: `{dec.get('pvo',0):+.2f}`"
                    )

            # 選中個股詳細圖
            if st.session_state.selected_stock in results:
                sym = st.session_state.selected_stock
                st.markdown(f"---\n### 📊 {sym} 詳細分析")
                render_kline_chart(sym, results[sym])

    # ================================================================
    # Tab 4: 數據健康度
    # ================================================================
    with tab4:
        render_health_panel()

        st.markdown("---")
        st.markdown("### 🗂️ V12.1 Alpha Seeds 預覽")
        if os.path.exists(ALPHA_SEEDS_PATH):
            with open(ALPHA_SEEDS_PATH, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            df_seeds = pd.DataFrame(seeds)
            st.dataframe(df_seeds, use_container_width=True, hide_index=True)

            if "Path" in df_seeds.columns:
                path_counts = df_seeds["Path"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=path_counts.index,
                    values=path_counts.values,
                    hole=0.4,
                    marker_colors=['#059669','#d97706','#dc2626','#0891b2'],
                ))
                fig_pie.update_layout(
                    title="路徑分布",
                    template="plotly_white",
                    paper_bgcolor='#ffffff',
                    font=dict(color='#1e293b'),
                    height=300,
                    margin=dict(t=40, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("尚未上傳 alpha_seeds.json，請在側欄上傳")

        st.markdown("---")
        st.markdown("### ⚠️ 數據清洗日誌")
        if st.session_state.all_warnings:
            for w in st.session_state.all_warnings:
                st.warning(w)
        else:
            st.success("✅ 無異常數據警告")


if __name__ == "__main__":
    main()
