"""
╔══════════════════════════════════════════════════════════════╗
║   資源法 AI 戰情室  v3.0  —  純展示層 (Display-Only)         ║
║   架構：Precompute + GitHub Storage + Streamlit Render       ║
║   資料來源：GitHub Raw JSON / Parquet (每日自動更新)           ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────
# 環境設定
# ──────────────────────────────────────────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

_ENV_GEMINI_KEY = _get_secret("GEMINI_API_KEY")

# ──────────────────────────────────────────────────────────────
# GitHub 資料源設定（修改為你的 repo）
# ──────────────────────────────────────────────────────────────
GITHUB_RAW = "https://raw.githubusercontent.com/{owner}/{repo}/main/storage"
GITHUB_OWNER = _get_secret("GITHUB_OWNER", "your-username")
GITHUB_REPO  = _get_secret("GITHUB_REPO",  "quant-storage")
BASE_URL     = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/storage"

# ──────────────────────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="資源法 AI 戰情室 v3.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# 全域樣式
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Noto+Sans+TC:wght@300;400;600;900&display=swap');

:root {
    --bg:        #0b0f1a;
    --bg2:       #111827;
    --bg3:       #1a2235;
    --panel:     #161d2e;
    --border:    rgba(99,179,237,0.15);
    --border2:   rgba(99,179,237,0.30);
    --accent:    #3b82f6;
    --accent2:   #06b6d4;
    --green:     #10b981;
    --red:       #ef4444;
    --amber:     #f59e0b;
    --purple:    #8b5cf6;
    --text:      #e2e8f0;
    --text-dim:  #64748b;
    --text-muted:#374151;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Noto Sans TC', sans-serif;
    --glow-b:    0 0 20px rgba(59,130,246,0.25);
    --glow-g:    0 0 20px rgba(16,185,129,0.25);
    --glow-r:    0 0 20px rgba(239,68,68,0.25);
    --radius:    10px;
    --radius-lg: 16px;
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}
.stApp { min-height: 100vh; }
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }

/* 隱藏 Streamlit 預設元素 */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 3rem !important; max-width: 1600px; }

/* 標題列 */
.hq-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 28px; margin-bottom: 8px;
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.05));
    border: 1px solid var(--border2);
    border-radius: var(--radius-lg);
    position: relative; overflow: hidden;
}
.hq-header::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--green));
}
.hq-title { font-size: 1.6rem; font-weight: 900; color: #fff; letter-spacing: -0.5px; }
.hq-sub   { font-size: 0.78rem; color: var(--text-dim); font-family: var(--mono); margin-top: 3px; }
.hq-badge {
    margin-left: auto; padding: 6px 14px; border-radius: 20px;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    color: var(--green); font-size: 0.72rem; font-weight: 700;
    font-family: var(--mono); letter-spacing: 1px;
}

/* 狀態列 */
.status-row {
    display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap;
}
.status-chip {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px; font-size: 0.73rem;
    font-family: var(--mono); border: 1px solid;
}
.chip-ok  { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.3); color: var(--green); }
.chip-err { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.3);  color: var(--red); }
.chip-info{ background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.3); color: var(--accent); }
.chip-warn{ background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); color: var(--amber); }
.dot { width:7px; height:7px; border-radius:50%; animation: pulse 2s infinite; }
.dot-g { background:var(--green); box-shadow:0 0 6px var(--green); }
.dot-r { background:var(--red); }
.dot-b { background:var(--accent); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Section 標題 */
.sec-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px; margin: 20px 0 14px 0;
    border-radius: var(--radius); border-left: 3px solid;
    background: var(--panel);
}
.sec-v4    { border-color: var(--accent); }
.sec-v12   { border-color: var(--green); }
.sec-regime{ border-color: var(--amber); }
.sec-ai    { border-color: var(--purple); }
.sec-label { font-size: 0.68rem; font-family: var(--mono); opacity: 0.6; margin-left: auto; }

/* 卡片 */
.card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 18px; margin-bottom: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: var(--border2); box-shadow: var(--glow-b); }
.card-bull  { border-left: 3px solid var(--green); }
.card-bear  { border-left: 3px solid var(--red); }
.card-neutral{ border-left: 3px solid var(--text-dim); }
.card-star  { border-left: 3px solid var(--accent); box-shadow: var(--glow-b); }

/* 數字顯示 */
.mono-num { font-family: var(--mono); font-weight: 700; }
.c-green  { color: var(--green) !important; }
.c-red    { color: var(--red) !important; }
.c-amber  { color: var(--amber) !important; }
.c-blue   { color: var(--accent) !important; }
.c-cyan   { color: var(--accent2) !important; }
.c-purple { color: var(--purple) !important; }
.c-dim    { color: var(--text-dim) !important; }

/* 指標 pill */
.pill {
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 700; font-family: var(--mono);
    border: 1px solid; margin: 2px;
}
.pill-g  { background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.35); color: var(--green); }
.pill-r  { background: rgba(239,68,68,0.1);   border-color: rgba(239,68,68,0.35);  color: var(--red); }
.pill-b  { background: rgba(59,130,246,0.1);  border-color: rgba(59,130,246,0.35); color: var(--accent); }
.pill-a  { background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.35); color: var(--amber); }
.pill-p  { background: rgba(139,92,246,0.1);  border-color: rgba(139,92,246,0.35); color: var(--purple); }
.pill-c  { background: rgba(6,182,212,0.1);   border-color: rgba(6,182,212,0.35);  color: var(--accent2); }

/* Regime Bar */
.regime-bar {
    display: flex; align-items: stretch; border-radius: var(--radius);
    overflow: hidden; height: 38px; margin: 10px 0;
    border: 1px solid var(--border);
}
.regime-seg {
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; font-family: var(--mono);
    transition: all 0.3s;
}
.seg-bear  { background: rgba(239,68,68,0.18); color: var(--red); }
.seg-range { background: rgba(245,158,11,0.18); color: var(--amber); }
.seg-bull  { background: rgba(16,185,129,0.18); color: var(--green); }

/* 表格 */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.data-table th {
    background: var(--bg3); color: var(--text-dim); font-weight: 600;
    font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.5px;
    padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border);
}
.data-table td { padding: 9px 12px; border-bottom: 1px solid rgba(99,179,237,0.06); }
.data-table tr:hover td { background: rgba(59,130,246,0.04); }

/* Rank badge */
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    font-size: 0.72rem; font-weight: 900; font-family: var(--mono);
}
.rank-1 { background: linear-gradient(135deg,#f59e0b,#d97706); color:#000; }
.rank-2 { background: linear-gradient(135deg,#94a3b8,#64748b); color:#fff; }
.rank-3 { background: linear-gradient(135deg,#cd7c3a,#b45309); color:#fff; }
.rank-n { background: var(--bg3); color: var(--text-dim); border: 1px solid var(--border); }

/* AI 分析框 */
.ai-box {
    background: linear-gradient(135deg, rgba(139,92,246,0.05), rgba(59,130,246,0.05));
    border: 1px solid rgba(139,92,246,0.2); border-left: 3px solid var(--purple);
    border-radius: var(--radius); padding: 18px 22px;
    font-size: 0.92rem; line-height: 1.85; color: var(--text);
}

/* 大盤指標卡 */
.mkt-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 10px; margin: 10px 0;
}
.mkt-cell {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 12px 10px; text-align: center;
}
.mkt-val  { font-size: 1.2rem; font-weight: 900; font-family: var(--mono); }
.mkt-lbl  { font-size: 0.65rem; color: var(--text-dim); margin-top: 3px; letter-spacing: 0.5px; }

/* 路徑標籤 */
.path-tag {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-family: var(--mono); font-size: 0.72rem; font-weight: 700;
}
.path-45  { background: rgba(16,185,129,0.15);  color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
.path-423 { background: rgba(59,130,246,0.15);  color: var(--accent); border: 1px solid rgba(59,130,246,0.3); }
.path-na  { background: rgba(100,116,139,0.15); color: var(--text-dim); border: 1px solid var(--border); }

/* Tabs 樣式 */
[data-testid="stTab"] button { color: #94a3b8 !important; font-family: var(--sans) !important; }
[data-testid="stTab"] button[aria-selected="true"] {
    color: #ffffff !important; font-weight: 700 !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTab"] { background: transparent !important; }

/* Streamlit 元件深色化 */
.stTextInput input, .stSelectbox select {
    background: var(--bg3) !important; color: var(--text) !important;
    border-color: var(--border) !important; border-radius: var(--radius) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #2563eb) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius) !important; font-weight: 700 !important;
    box-shadow: var(--glow-b) !important;
}
.stButton > button:hover { transform: translateY(-1px); filter: brightness(1.1); }
[data-testid="stMetricValue"] { color: var(--accent2) !important; font-family: var(--mono) !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }
[data-testid="stMetricDelta"] > div { font-family: var(--mono) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 資料載入層（GitHub 讀取 + cache）
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_json(path_suffix: str) -> dict | list | None:
    """從 GitHub 讀取 JSON 快照"""
    url = f"{BASE_URL}/{path_suffix}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_parquet_url(path_suffix: str) -> pd.DataFrame | None:
    """從 GitHub 讀取 Parquet 檔"""
    url = f"{BASE_URL}/{path_suffix}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            from io import BytesIO
            return pd.read_parquet(BytesIO(r.content))
        return None
    except Exception:
        return None


def load_v4_snapshot() -> dict | None:
    """V4 快照：TOP20 + 市場強度"""
    return load_json("v4/v4_latest.json")


def load_v12_snapshot() -> dict | None:
    """V12.1 快照：路徑 + EV + action + exit"""
    return load_json("v12/v12_latest.json")


def load_regime_snapshot() -> dict | None:
    """Regime 快照：bull/bear/range 機率"""
    return load_json("regime/regime_state.json")


def load_market_snapshot() -> dict | None:
    """大盤快照"""
    return load_json("market/market_snapshot.json")


def load_trade_history() -> list | None:
    """交易歷史紀錄"""
    return load_json("logs/trade_history.json")


# 模擬資料（GitHub 未連線時 fallback）
def _mock_v4() -> dict:
    import random
    syms = ["2330","2317","2454","2308","2382","3711","2412","6669","3008","2395",
            "2379","3034","2345","3443","3661","6415","3035","2408","3131","5274"]
    rows = []
    for i, s in enumerate(syms):
        score = round(85 - i * 1.8 + random.uniform(-2, 2), 2)
        rows.append({
            "rank": i+1, "symbol": s, "score": score,
            "pvo": round(random.uniform(-5, 20), 2),
            "vri": round(random.uniform(35, 95), 1),
            "slope_z": round(random.uniform(-0.5, 2.5), 2),
            "action": random.choice(["強力買進","買進","觀察","賣出"]),
            "signal": random.choice(["三合一(ABC)","二合一(AB)","二合一(BC)","單一(A)","單一(C)"]),
            "close": round(random.uniform(100, 900), 1),
            "regime": random.choice(["trend","range","recovery"]),
        })
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market": "TW", "top20": rows,
        "pool_mu": 62.3, "pool_sigma": 11.5, "win_rate": 57.1,
    }

def _mock_v12() -> dict:
    import random
    syms = ["2330","2454","3711","6669","3008","2379","3443","6415","3035","2408"]
    rows = []
    for s in syms:
        ev = round(random.uniform(2.0, 9.5), 2)
        rows.append({
            "symbol": s, "path": random.choice(["45","423"]),
            "ev": ev, "action": random.choice(["持有","進場","觀察","出場"]),
            "exit_signal": random.choice(["無","EV衰退","量能枯竭","時間衰減","—"]),
            "quality": random.choice(["Pure","Flicker"]),
            "ev_tier": "⭐核心" if ev>5 else ("🔥主力" if ev>3 else "📌補位"),
            "regime": random.choice(["bull","range","bear"]),
            "days_held": random.randint(1,18),
            "curr_ret_pct": round(random.uniform(-5,15), 2),
            "tp1_price": round(random.uniform(150,900), 1),
            "stop_price": round(random.uniform(100,800), 1),
        })
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "positions": rows,
        "stats": {
            "total_trades": 112, "win_rate": 57.1, "avg_ev": 5.29,
            "max_dd": -6.58, "sharpe": 5.36, "t_stat": 4.032,
            "simple_cagr": 96.9, "pl_ratio": 2.31,
        }
    }

def _mock_regime() -> dict:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bear": 0.22, "range": 0.41, "bull": 0.37,
        "label": "偏多震盪", "active_strategy": "range",
        "active_path": "423", "backup_path": "45",
        "slope_5d": 0.0312, "slope_20d": 0.0105,
        "mkt_rsi": 54.3, "adx": 22.1,
        "history": [
            {"month":"2026-01","bear":0.35,"range":0.45,"bull":0.20,"label":"偏空"},
            {"month":"2026-02","bear":0.28,"range":0.48,"bull":0.24,"label":"震盪"},
            {"month":"2026-03","bear":0.20,"range":0.43,"bull":0.37,"label":"偏多震盪"},
            {"month":"2026-04","bear":0.22,"range":0.41,"bull":0.37,"label":"偏多震盪"},
        ]
    }

def _mock_market() -> dict:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "index_close": 20843.5, "index_chg_pct": 0.62,
        "mkt_rsi": 54.3, "mkt_slope_5d": 0.031, "mkt_slope_20d": 0.011,
        "bear_pct": 22, "range_pct": 41, "bull_pct": 37,
        "label": "偏多震盪",
    }

def _mock_history() -> list:
    import random
    logs = []
    for i in range(30):
        ret = round(random.uniform(-8, 15), 2)
        logs.append({
            "date": f"2026-03-{(i%28)+1:02d}",
            "sym": random.choice(["2330","2454","3711","6669","3008"]),
            "action_type": "賣出",
            "exit_type": random.choice(["停利①","停利②","EV衰退","量能枯竭","硬停損"]),
            "ret": round(ret/100, 4),
            "path": random.choice(["45","423"]),
            "year": 2026,
        })
    return logs


# ══════════════════════════════════════════════════════════════
# Gemini AI
# ══════════════════════════════════════════════════════════════

def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ 請在側欄設定 Gemini API Key。"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        for m in ["gemma-3-27b-it", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                model = genai.GenerativeModel(m)
                return model.generate_content(prompt).text
            except Exception:
                continue
        return "❌ 模型無法回應，請稍後再試。"
    except Exception as e:
        return f"❌ Gemini 錯誤：{e}"


def build_dashboard_prompt(v4: dict, v12: dict, regime: dict, market: dict) -> str:
    top5 = v4.get("top20", [])[:5] if v4 else []
    top5_txt = "\n".join([
        f"  #{r['rank']} {r['symbol']} | Score:{r['score']} | PVO:{r.get('pvo',0):+.2f}"
        f" | VRI:{r.get('vri',0):.1f} | SlopeZ:{r.get('slope_z',0):+.2f}"
        f" | 訊號:{r.get('signal','—')}" for r in top5
    ]) if top5 else "（無資料）"

    active_pos = v12.get("positions", []) if v12 else []
    pos_txt = "\n".join([
        f"  {p['symbol']} | 路徑:{p['path']} | EV:{p['ev']:+.2f}%"
        f" | {p.get('ev_tier','—')} | 持:{p.get('days_held',0)}日"
        f" | 報酬:{p.get('curr_ret_pct',0):+.2f}% | 出場信號:{p.get('exit_signal','—')}"
        for p in active_pos[:8]
    ]) if active_pos else "（無部位）"

    r = regime or _mock_regime()
    mk = market or _mock_market()
    s = v12.get("stats", {}) if v12 else {}

    return f"""你是資深量化交易分析師，精通台股統計套利與資金流向研究。
以下是今日系統快照，請進行簡明精準的盤勢解讀。

【市場環境】
大盤: {mk.get('label','—')} | 大盤RSI: {mk.get('mkt_rsi',0):.1f}
熊:{r.get('bear',0)*100:.0f}% | 震:{r.get('range',0)*100:.0f}% | 牛:{r.get('bull',0)*100:.0f}%
5日斜率: {r.get('slope_5d',0):+.4f} | 20日斜率: {r.get('slope_20d',0):+.4f}
當前策略: {r.get('active_strategy','—')} | 主路徑: {r.get('active_path','—')} | 備援:{r.get('backup_path','—')}

【V4 市場強度 TOP5】
{top5_txt}

【V12.1 目前部位】
{pos_txt}

【V12.1 歷史統計】
總筆數:{s.get('total_trades','N/A')} | 勝率:{s.get('win_rate','N/A')}% | 均EV:{s.get('avg_ev','N/A')}%
t值:{s.get('t_stat','N/A')} | MaxDD:{s.get('max_dd','N/A')}% | Sharpe:{s.get('sharpe','N/A')}

【分析任務】（每段100字以內，引用具體數值，禁止泛泛而談）
**一、大盤風險評估**：Regime機率三維解讀 + 斜率含義 + 今日適合操作的持倉水位
**二、V4訊號解讀**：TOP3標的技術與資金面評估，是否值得關注
**三、V12.1部位診斷**：現有部位健康度，哪些出場信號需警惕
**四、操作建議**：今日優先行動清單（進/出/觀察）+ 主要風險因子

所有句子皆由資深股票分析師與統計學專家深思後的回答。格式：條列為主。"""


def build_single_stock_prompt(sym: str, v4_data: dict, v12_data: dict, regime: dict) -> str:
    v4_row = next((r for r in (v4_data or {}).get("top20", []) if r["symbol"] == sym), None)
    v12_row = next((p for p in (v12_data or {}).get("positions", []) if p["symbol"] == sym), None)
    r = regime or _mock_regime()

    v4_txt  = f"Score:{v4_row['score']} | PVO:{v4_row.get('pvo',0):+.2f} | VRI:{v4_row.get('vri',0):.1f} | SlopeZ:{v4_row.get('slope_z',0):+.2f} | 訊號:{v4_row.get('signal','—')}" if v4_row else "（V4無資料）"
    v12_txt = f"路徑:{v12_row['path']} | EV:{v12_row['ev']:+.2f}% | {v12_row.get('ev_tier','—')} | 持:{v12_row.get('days_held',0)}日 | 報酬:{v12_row.get('curr_ret_pct',0):+.2f}% | 出場:{v12_row.get('exit_signal','—')} | 品質:{v12_row.get('quality','—')}" if v12_row else "（V12.1無部位）"

    return f"""請對 {sym} 進行個股深度分析，每段100字以內，引用具體數值。

【個股快照】
{sym} — V4: {v4_txt}
V12.1: {v12_txt}

【大盤環境】
Regime: 熊{r.get('bear',0)*100:.0f}% | 震{r.get('range',0)*100:.0f}% | 牛{r.get('bull',0)*100:.0f}%
當前路徑: {r.get('active_path','—')}

【分析】
**一、技術面**：PVO/VRI/SlopeZ解讀，目前趨勢位階
**二、路徑判斷**：V12.1路徑(45/423)適合當前Regime嗎？EV是否仍有優勢？
**三、操作建議**：進出場條件、停損邏輯、持倉建議
**四、風險提示**：需警惕的反轉信號

所有句子皆由資深股票分析師與統計學專家深思後的回答。"""


# ══════════════════════════════════════════════════════════════
# UI 工具函數
# ══════════════════════════════════════════════════════════════

def _color_num(val, positive_good=True):
    if val is None: return "c-dim"
    if val > 0: return "c-green" if positive_good else "c-red"
    if val < 0: return "c-red" if positive_good else "c-green"
    return "c-dim"

def _action_pill(action: str) -> str:
    mapping = {
        "強力買進": ("pill-g", "▲ 強力買進"),
        "買進":     ("pill-g", "▲ 買進"),
        "持有":     ("pill-b", "◆ 持有"),
        "進場":     ("pill-g", "▲ 進場"),
        "觀察":     ("pill-a", "◇ 觀察"),
        "賣出":     ("pill-r", "▼ 賣出"),
        "出場":     ("pill-r", "▼ 出場"),
        "觀望":     ("pill-a", "◇ 觀望"),
    }
    css, label = mapping.get(action, ("pill-c", action))
    return f'<span class="pill {css}">{label}</span>'

def _path_tag(path: str) -> str:
    mapping = {"45": "path-45", "423": "path-423"}
    css = mapping.get(str(path), "path-na")
    return f'<span class="path-tag {css}">{path}</span>'

def _rank_badge(rank: int) -> str:
    css = {1:"rank-1", 2:"rank-2", 3:"rank-3"}.get(rank, "rank-n")
    return f'<span class="rank-badge {css}">{rank}</span>'

def _quality_pill(q: str) -> str:
    if q == "Pure":
        return '<span class="pill pill-g">✅ Pure</span>'
    return '<span class="pill pill-a">〔F〕Flicker</span>'

def _exit_pill(sig: str) -> str:
    if not sig or sig in ("—", "無"):
        return '<span class="pill pill-b">持倉中</span>'
    if "停利" in sig:
        return f'<span class="pill pill-g">🎯 {sig}</span>'
    if any(x in sig for x in ["衰退","枯竭","衰減","加速","Slope"]):
        return f'<span class="pill pill-a">⚠️ {sig}</span>'
    if "停損" in sig:
        return f'<span class="pill pill-r">🛑 {sig}</span>'
    return f'<span class="pill pill-c">{sig}</span>'

def render_regime_bar(bear: float, range_: float, bull: float):
    b_pct = bear * 100; r_pct = range_ * 100; u_pct = bull * 100
    st.markdown(f"""
    <div class="regime-bar">
        <div class="regime-seg seg-bear"  style="width:{b_pct:.0f}%">
            熊 {b_pct:.0f}%
        </div>
        <div class="regime-seg seg-range" style="width:{r_pct:.0f}%">
            震 {r_pct:.0f}%
        </div>
        <div class="regime-seg seg-bull"  style="width:{u_pct:.0f}%">
            牛 {u_pct:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session 初始化
# ══════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "gemini_key":      _ENV_GEMINI_KEY,
        "ai_summary":      "",
        "single_sym":      "",
        "single_result":   "",
        "use_mock":        False,
        "last_refresh":    None,
        # 資料快照
        "v4_data":     None,
        "v12_data":    None,
        "regime_data": None,
        "market_data": None,
        "history_data":None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ══════════════════════════════════════════════════════════════
# 側欄
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ 資源法 v3.0")
        st.markdown("---")

        st.markdown("### 🔑 Gemini API Key")
        if _ENV_GEMINI_KEY:
            st.success("✅ 已從環境變數載入")
            if st.checkbox("手動覆蓋"):
                k = st.text_input("API Key（覆蓋）", type="password")
                if k: st.session_state.gemini_key = k
        else:
            k = st.text_input("Gemini API Key", type="password",
                              value=st.session_state.gemini_key,
                              placeholder="AIza...")
            st.session_state.gemini_key = k

        st.markdown("---")
        st.markdown("### 🔗 GitHub 設定")
        st.caption(f"Owner: `{GITHUB_OWNER}`")
        st.caption(f"Repo: `{GITHUB_REPO}`")
        st.caption(f"Base: `.../storage/`")

        st.markdown("---")
        st.markdown("### ⚙️ 開發選項")
        use_mock = st.checkbox("使用模擬資料（Demo模式）",
                               value=st.session_state.use_mock)
        st.session_state.use_mock = use_mock
        if use_mock:
            st.warning("⚠️ 目前顯示模擬數據")

        st.markdown("---")
        st.caption("資料更新時程")
        for t in ["09:30 台股開盤快照","12:00 盤中更新","13:30 盤後計算",
                  "14:30 V12.1決策","15:30 收盤報告","18:00 日結存檔"]:
            st.caption(f"• {t}")

        st.markdown("---")
        st.caption("© 2026 資源法 AI 戰情室 v3.0")


# ══════════════════════════════════════════════════════════════
# 載入所有資料
# ══════════════════════════════════════════════════════════════
def load_all_data():
    use_mock = st.session_state.use_mock
    if use_mock:
        return _mock_v4(), _mock_v12(), _mock_regime(), _mock_market(), _mock_history()

    v4     = load_v4_snapshot()     or _mock_v4()
    v12    = load_v12_snapshot()    or _mock_v12()
    regime = load_regime_snapshot() or _mock_regime()
    market = load_market_snapshot() or _mock_market()
    hist   = load_trade_history()   or _mock_history()
    return v4, v12, regime, market, hist


# ══════════════════════════════════════════════════════════════
# Section 1：V4 市場強度觀察
# ══════════════════════════════════════════════════════════════
def render_v4_section(v4: dict):
    gen_at = v4.get("generated_at", "—")
    wr     = v4.get("win_rate", 0)
    mu     = v4.get("pool_mu", 0)
    sigma  = v4.get("pool_sigma", 0)
    top20  = v4.get("top20", [])

    st.markdown(f"""
    <div class="sec-header sec-v4">
        <span style="font-size:1.1rem;font-weight:900;color:#3b82f6;">
            🟦 Section 1 — V4 市場強度快照
        </span>
        <span class="pill pill-b">TOP 20</span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    # 統計摘要
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pool μ（均分）", f"{mu:.2f}", help="評分池均值")
    c2.metric("Pool σ（標準差）", f"{sigma:.2f}", help="評分池標準差")
    c3.metric("歷史勝率", f"{wr:.1f}%", help="V4 回測勝率")
    c4.metric("觀察標的數", len(top20), help="本次快照標的數")

    if not top20:
        st.info("⏳ 等待 GitHub 資料更新（或啟用 Demo 模式）")
        return

    # 篩選 & 排序
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        filter_action = st.multiselect(
            "操作過濾", ["強力買進","買進","觀察","賣出"],
            default=["強力買進","買進"], key="v4_filter_action"
        )
    with col_f2:
        min_vri = st.slider("VRI 最低", 0, 100, 40, key="v4_vri")
    with col_f3:
        sort_by = st.selectbox("排序", ["rank","score","slope_z","vri","pvo"], key="v4_sort")

    filtered = [r for r in top20
                if (not filter_action or r.get("action") in filter_action)
                and r.get("vri", 0) >= min_vri]
    filtered.sort(key=lambda x: x.get(sort_by, 0), reverse=(sort_by != "rank"))

    # 表格
    st.markdown(f"**顯示 {len(filtered)} / {len(top20)} 檔**")
    html = """
    <table class="data-table">
    <thead><tr>
        <th>#</th><th>代號</th><th>Score</th><th>操作</th><th>訊號型態</th>
        <th>PVO</th><th>VRI</th><th>Slope Z</th><th>現價</th><th>Regime</th>
    </tr></thead><tbody>
    """
    for r in filtered:
        rank    = r.get("rank", "—")
        sym     = r.get("symbol", "—")
        score   = r.get("score", 0)
        action  = r.get("action", "—")
        signal  = r.get("signal", "—")
        pvo     = r.get("pvo", 0)
        vri     = r.get("vri", 0)
        slz     = r.get("slope_z", 0)
        close   = r.get("close", 0)
        regime  = r.get("regime", "—")

        rank_css    = {1:"rank-1",2:"rank-2",3:"rank-3"}.get(rank,"rank-n")
        pvo_css     = "c-green" if pvo > 10 else ("c-cyan" if pvo > 0 else "c-red")
        vri_css     = "c-green" if 40<=vri<=75 else ("c-red" if vri>90 else "c-amber")
        slz_css     = "c-green" if slz>1.5 else ("c-cyan" if slz>0 else "c-red")
        score_css   = "c-green" if score >= mu+sigma else ("c-amber" if score >= mu else "c-dim")
        regime_map  = {"trend":"🚀","range":"〰️","crash":"💥","recovery":"🔄"}
        regime_icon = regime_map.get(regime, "◇")

        # 訊號顏色
        if "三合一" in signal:   sig_css = "pill-p"
        elif "二合一" in signal: sig_css = "pill-b"
        elif "單一"  in signal:  sig_css = "pill-a"
        else:                    sig_css = "pill-c"

        html += f"""
        <tr>
            <td><span class="rank-badge {rank_css}">{rank}</span></td>
            <td><b style="color:#e2e8f0;">{sym}</b></td>
            <td><span class="mono-num {score_css}">{score:.2f}</span></td>
            <td>{_action_pill(action)}</td>
            <td><span class="pill {sig_css}">{signal}</span></td>
            <td><span class="mono-num {pvo_css}">{pvo:+.2f}</span></td>
            <td><span class="mono-num {vri_css}">{vri:.1f}</span></td>
            <td><span class="mono-num {slz_css}">{slz:+.2f}</span></td>
            <td class="mono-num" style="color:#94a3b8;">{close:.1f}</td>
            <td style="color:#64748b;">{regime_icon} {regime}</td>
        </tr>"""
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Section 2：V12.1 交易決策
# ══════════════════════════════════════════════════════════════
def render_v12_section(v12: dict):
    gen_at   = v12.get("generated_at", "—")
    positions = v12.get("positions", [])
    stats    = v12.get("stats", {})

    st.markdown(f"""
    <div class="sec-header sec-v12">
        <span style="font-size:1.1rem;font-weight:900;color:#10b981;">
            🟩 Section 2 — V12.1 交易決策系統
        </span>
        <span class="pill pill-g">路徑 45 / 423</span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    # 統計指標列
    s = stats
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("總交易筆數", s.get("total_trades","—"))
    c2.metric("OOS勝率", f"{s.get('win_rate',0):.1f}%")
    c3.metric("平均EV", f"{s.get('avg_ev',0):+.2f}%")
    c4.metric("最大回撤", f"{s.get('max_dd',0):.2f}%")
    c5.metric("Sharpe", f"{s.get('sharpe',0):.2f}")
    c6.metric("t值", f"{s.get('t_stat',0):+.3f}")
    c7.metric("年化單利", f"{s.get('simple_cagr',0):+.1f}%")

    if not positions:
        st.info("⏳ 等待 GitHub V12.1 快照更新")
        return

    st.markdown("#### 📋 目前部位監控")

    html = """
    <table class="data-table">
    <thead><tr>
        <th>代號</th><th>路徑</th><th>EV</th><th>等級</th>
        <th>操作</th><th>出場信號</th><th>純淨度</th>
        <th>持倉天</th><th>當前報酬</th><th>停利①</th><th>停損</th>
    </tr></thead><tbody>
    """
    for p in positions:
        sym      = p.get("symbol","—")
        path     = p.get("path","—")
        ev       = p.get("ev", 0)
        ev_tier  = p.get("ev_tier","—")
        action   = p.get("action","—")
        exit_sig = p.get("exit_signal","—")
        quality  = p.get("quality","Pure")
        days     = p.get("days_held",0)
        curr_ret = p.get("curr_ret_pct",0)
        tp1      = p.get("tp1_price","—")
        stop     = p.get("stop_price","—")

        ev_css   = "c-green" if ev>5 else ("c-cyan" if ev>3 else "c-amber")
        ret_css  = "c-green" if curr_ret>0 else "c-red"

        html += f"""
        <tr>
            <td><b style="color:#e2e8f0;">{sym}</b></td>
            <td>{_path_tag(path)}</td>
            <td><span class="mono-num {ev_css}">{ev:+.2f}%</span></td>
            <td style="font-size:0.78rem;">{ev_tier}</td>
            <td>{_action_pill(action)}</td>
            <td>{_exit_pill(exit_sig)}</td>
            <td>{_quality_pill(quality)}</td>
            <td class="c-dim mono-num">{days}</td>
            <td><span class="mono-num {ret_css}">{curr_ret:+.2f}%</span></td>
            <td class="c-green mono-num" style="font-size:0.78rem;">{tp1}</td>
            <td class="c-red mono-num" style="font-size:0.78rem;">{stop}</td>
        </tr>"""
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

    # 路徑分佈餅圖
    path_counts = {}
    for p in positions:
        path_counts[p.get("path","?")] = path_counts.get(p.get("path","?"), 0) + 1

    if path_counts:
        fig = go.Figure(go.Pie(
            labels=list(path_counts.keys()),
            values=list(path_counts.values()),
            hole=0.5,
            marker_colors=["#10b981","#3b82f6","#f59e0b"],
            textfont_size=12,
        ))
        fig.update_layout(
            height=200, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(font=dict(color="#e2e8f0")),
            title=dict(text="路徑分佈", font=dict(color="#64748b",size=12))
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Section 3：Regime 大盤
# ══════════════════════════════════════════════════════════════
def render_regime_section(regime: dict, market: dict):
    gen_at = regime.get("generated_at", "—")

    st.markdown(f"""
    <div class="sec-header sec-regime">
        <span style="font-size:1.1rem;font-weight:900;color:#f59e0b;">
            🟨 Section 3 — Regime 市場制度 & 策略切換
        </span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    bear  = regime.get("bear", 0.33)
    range_= regime.get("range", 0.34)
    bull  = regime.get("bull", 0.33)
    label = regime.get("label", "震盪")
    strat = regime.get("active_strategy", "range")
    a_path= regime.get("active_path", "—")
    b_path= regime.get("backup_path", "—")
    s5d   = regime.get("slope_5d", 0)
    s20d  = regime.get("slope_20d", 0)

    # Regime 機率條
    render_regime_bar(bear, range_, bull)

    # 指標格
    mk = market or {}
    idx_close  = mk.get("index_close", 0)
    idx_chg    = mk.get("index_chg_pct", 0)
    mkt_rsi    = mk.get("mkt_rsi", 0)
    adx_val    = regime.get("adx", 0)

    chg_css = "c-green" if idx_chg >= 0 else "c-red"
    s5_css  = "c-green" if s5d >= 0 else "c-red"
    s20_css = "c-green" if s20d >= 0 else "c-red"

    st.markdown(f"""
    <div class="mkt-grid">
        <div class="mkt-cell">
            <div class="mkt-val" style="color:#e2e8f0;">{label}</div>
            <div class="mkt-lbl">大盤情緒</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:#f59e0b;">{strat.upper()}</div>
            <div class="mkt-lbl">當前策略</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:#10b981;">{a_path}</div>
            <div class="mkt-lbl">主路徑</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:#3b82f6;">{b_path}</div>
            <div class="mkt-lbl">備援路徑</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val {'c-green' if idx_chg>=0 else 'c-red'}" 
                 style="color:{'#10b981' if idx_chg>=0 else '#ef4444'}">
                {idx_close:,.1f}
            </div>
            <div class="mkt-lbl">指數收盤</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" 
                 style="color:{'#10b981' if idx_chg>=0 else '#ef4444'}">
                {idx_chg:+.2f}%
            </div>
            <div class="mkt-lbl">日漲跌</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:#06b6d4;">{mkt_rsi:.1f}</div>
            <div class="mkt-lbl">大盤 RSI</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:#8b5cf6;">{adx_val:.1f}</div>
            <div class="mkt-lbl">ADX</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:{'#10b981' if s5d>=0 else '#ef4444'}">{s5d:+.4f}</div>
            <div class="mkt-lbl">5日斜率</div>
        </div>
        <div class="mkt-cell">
            <div class="mkt-val" style="color:{'#10b981' if s20d>=0 else '#ef4444'}">{s20d:+.4f}</div>
            <div class="mkt-lbl">20日斜率</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 歷史 Regime 圖
    history = regime.get("history", [])
    if history:
        df_r = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_bar(x=df_r["month"], y=df_r["bull"]*100, name="牛市",
                    marker_color="rgba(16,185,129,0.7)")
        fig.add_bar(x=df_r["month"], y=df_r["range"]*100, name="震盪",
                    marker_color="rgba(245,158,11,0.7)")
        fig.add_bar(x=df_r["month"], y=df_r["bear"]*100, name="熊市",
                    marker_color="rgba(239,68,68,0.7)")
        fig.update_layout(
            barmode="stack", height=200,
            margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", size=11),
            legend=dict(font=dict(color="#94a3b8"), orientation="h",
                       y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#64748b"),
            yaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#64748b",
                      range=[0,100], ticksuffix="%"),
            title=dict(text="月末 Regime 機率歷史", font=dict(color="#64748b",size=11))
        )
        st.plotly_chart(fig, use_container_width=True)

    # 策略切換邏輯說明
    with st.expander("📖 策略切換邏輯說明"):
        st.markdown("""
        | Regime | 主路徑 | 備援 | 最大部位 | EV門檻 |
        |--------|--------|------|----------|--------|
        | **bull** 牛市 | `45` (65%) | `423` (35%) | 4 | ≥ 3% |
        | **range** 震盪 | `423` (65%) | `45` (35%) | 5 | ≥ 3% |
        | **bear** 熊市 | 空手 | 空手 | 2 | ≥ 4% |

        - **路徑 45**：高 EV 快進快出，適合趨勢行情
        - **路徑 423**：多因子共振型，適合震盪行情
        - **Flicker**：因子中斷標的，自動提高 EV 門檻 × 1.2 且提前停利
        """)


# ══════════════════════════════════════════════════════════════
# 交易歷史圖表
# ══════════════════════════════════════════════════════════════
def render_history_tab(hist: list):
    if not hist:
        st.info("⏳ 等待交易歷史資料")
        return

    df = pd.DataFrame(hist)
    df["ret_pct"] = df["ret"] * 100
    df["win"] = df["ret_pct"] > 0

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 📈 累積報酬曲線")
        df_sorted = df.sort_values("date")
        df_sorted["cumret"] = df_sorted["ret_pct"].cumsum()
        fig = go.Figure()
        fig.add_scatter(
            x=df_sorted["date"], y=df_sorted["cumret"],
            fill="tozeroy", name="累積報酬",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.10)"
        )
        fig.update_layout(
            height=260, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b"),
            xaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#64748b"),
            yaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#64748b",
                      ticksuffix="%"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 出場原因分佈")
        if "exit_type" in df.columns:
            exit_counts = df["exit_type"].value_counts().head(8)
            colors = []
            for et in exit_counts.index:
                if "停利" in et: colors.append("#10b981")
                elif "停損" in et or "硬" in et: colors.append("#ef4444")
                elif "衰退" in et or "枯竭" in et: colors.append("#f59e0b")
                else: colors.append("#3b82f6")
            fig2 = go.Figure(go.Bar(
                x=exit_counts.values, y=exit_counts.index,
                orientation="h", marker_color=colors
            ))
            fig2.update_layout(
                height=260, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748b"),
                xaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#64748b"),
                yaxis=dict(color="#94a3b8"),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

    # 路徑績效摘要
    if "path" in df.columns:
        st.markdown("#### 🛤️ 路徑績效摘要")
        path_stats = df.groupby("path")["ret_pct"].agg(
            筆數="count", 勝率=lambda x: (x>0).mean()*100,
            均報酬="mean", 累計="sum"
        ).round(2)
        st.dataframe(path_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 個股 AI 分析
# ══════════════════════════════════════════════════════════════
def render_single_stock_panel(v4: dict, v12: dict, regime: dict):
    st.markdown(f"""
    <div class="sec-header sec-ai">
        <span style="font-size:1.0rem;font-weight:800;color:#8b5cf6;">
            🔍 個股 AI 深度分析
        </span>
        <span style="color:#64748b;font-size:0.8rem;margin-left:10px;">
            輸入代號 → Gemini 個股深度報告
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        sym_input = st.text_input(
            "個股代號", placeholder="例：2330 / NVDA",
            label_visibility="collapsed", key="single_sym_input"
        )
    with col_btn:
        analyze_btn = st.button("🔍 分析", use_container_width=True, key="single_btn")

    # 顯示上次結果
    if st.session_state.single_result and st.session_state.single_sym:
        import html as _h
        _safe = _h.escape(st.session_state.single_result).replace('\n','<br>')
        st.markdown(f"""
        <div class="ai-box" style="border-left-color:#8b5cf6; margin-top:12px;">
            <div style="font-weight:700;color:#8b5cf6;margin-bottom:10px;
                        padding-bottom:8px;border-bottom:1px solid rgba(139,92,246,0.2);">
                🔍 {st.session_state.single_sym} 個股分析報告
            </div>
            {_safe}
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn and sym_input:
        sym_q = sym_input.strip().upper()
        prompt = build_single_stock_prompt(sym_q, v4, v12, regime)
        with st.spinner(f"🤖 分析 {sym_q} 中..."):
            result = call_gemini(prompt, st.session_state.gemini_key)
        st.session_state.single_sym    = sym_q
        st.session_state.single_result = result
        st.rerun()


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    # 載入資料
    with st.spinner("🔄 從 GitHub 讀取最新快照..."):
        v4, v12, regime, market, hist = load_all_data()
        st.session_state.update({
            "v4_data": v4, "v12_data": v12, "regime_data": regime,
            "market_data": market, "history_data": hist,
            "last_refresh": datetime.now().strftime("%H:%M:%S")
        })

    # ── 標題列 ──
    use_mock = st.session_state.use_mock
    mode_label = "DEMO 模式" if use_mock else "LIVE 資料"
    mode_css   = "chip-warn" if use_mock else "chip-ok"
    gen_at     = v4.get("generated_at", "—") if v4 else "—"

    st.markdown(f"""
    <div class="hq-header">
        <div>
            <div class="hq-title">⚡ 資源法 AI 戰情室 <span style="font-size:0.9rem;opacity:0.6;">v3.0</span></div>
            <div class="hq-sub">Precompute + GitHub Storage + Pure Display Layer</div>
        </div>
        <div class="hq-badge">{mode_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 狀態列 ──
    bear  = (regime or {}).get("bear",  0.33)
    range_= (regime or {}).get("range", 0.34)
    bull  = (regime or {}).get("bull",  0.33)
    label = (regime or {}).get("label", "—")
    a_path= (regime or {}).get("active_path", "—")

    bear_pct = bear * 100
    bull_pct = bull * 100
    dom_css  = "chip-ok" if bull > bear else ("chip-err" if bear > bull else "chip-warn")

    st.markdown(f"""
    <div class="status-row">
        <div class="status-chip chip-ok">
            <span class="dot dot-g"></span> 系統正常
        </div>
        <div class="status-chip chip-info">
            📡 資料: {gen_at}
        </div>
        <div class="status-chip chip-info">
            🕐 刷新: {st.session_state.last_refresh}
        </div>
        <div class="status-chip {dom_css}">
            🌡️ Regime: {label}
            &nbsp;|&nbsp; 熊{bear_pct:.0f}% 牛{bull_pct:.0f}%
        </div>
        <div class="status-chip chip-info">
            🛤️ 主路徑: {a_path}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 個股 AI 分析（首頁常駐）──
    render_single_stock_panel(v4, v12, regime)

    # ── 全盤 Gemini AI 分析（首頁常駐）──
    st.markdown(f"""
    <div class="sec-header sec-ai">
        <span style="font-size:1.0rem;font-weight:800;color:#8b5cf6;">
            🤖 Gemini 全盤 AI 分析
        </span>
    </div>
    """, unsafe_allow_html=True)

    ai_col1, ai_col2 = st.columns([1, 5])
    with ai_col1:
        ai_btn = st.button("🤖 執行 AI 分析", use_container_width=True, key="ai_btn")
    with ai_col2:
        st.caption("分析大盤環境 + V4 TOP5訊號 + V12.1部位診斷（需 Gemini API Key）")

    if st.session_state.ai_summary:
        import html as _h2
        _safe2 = _h2.escape(st.session_state.ai_summary).replace('\n','<br>')
        st.markdown(f'<div class="ai-box">{_safe2}</div>', unsafe_allow_html=True)

    if ai_btn:
        prompt = build_dashboard_prompt(v4, v12, regime, market)
        with st.spinner("🤖 Gemini 深度分析中..."):
            summary = call_gemini(prompt, st.session_state.gemini_key)
        st.session_state.ai_summary = summary
        st.rerun()

    st.markdown("---")

    # ── 四大 Tab ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🟦 V4 市場強度",
        "🟩 V12.1 交易決策",
        "🟨 Regime 大盤",
        "📜 交易歷史"
    ])

    with tab1:
        if v4:
            render_v4_section(v4)
        else:
            st.info("⏳ V4 快照尚未就緒，請確認 GitHub 儲存庫設定或啟用 Demo 模式。")

    with tab2:
        if v12:
            render_v12_section(v12)
        else:
            st.info("⏳ V12.1 快照尚未就緒。")

    with tab3:
        if regime and market:
            render_regime_section(regime, market)
        else:
            st.info("⏳ Regime 快照尚未就緒。")

    with tab4:
        render_history_tab(hist or [])


if __name__ == "__main__":
    main()
