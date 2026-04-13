"""
╔══════════════════════════════════════════════════════════════════╗
║  streamlit_app/app.py — 資源法 AI 戰情室 v5.1（整合版）          ║
║                                                                  ║
║  整合自 v4.1 + v5.0：                                            ║
║  [v4.1] 自選股快覽（V4+V12+watchlist 合并）                      ║
║  [v4.1] 個股 AI 深度分析（Gemini）                               ║
║  [v4.1] Gemini 全盤分析                                          ║
║  [v4.1] Mock 模擬資料模式                                        ║
║  [v4.1] 精美狀態列 + 大盤數字列                                  ║
║  [v4.1] V4 停利/停損欄位 + 候選 5 檔                             ║
║  [v5.0] 今日買進原因（結構化顯示）                               ║
║  [v5.0] 持股賣出訊號（多層條件）                                 ║
║  [v5.0] 回測績效圖（Equity Curve）                               ║
║  [v5.0] 買賣歷史紀錄（trades.csv）                               ║
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
import os

try:
    from dotenv import load_dotenv; load_dotenv()
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

GITHUB_OWNER = _get_secret("GITHUB_OWNER", "your-username")
GITHUB_REPO  = _get_secret("GITHUB_REPO",  "stock2026")
BASE_URL  = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/storage"
REPO_RAW  = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main"

# ──────────────────────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="資源法 AI 戰情室 v5.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Noto+Sans+TC:wght@300;400;600;900&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg:       #f0f4f8;
    --bg2:      #e8edf4;
    --bg3:      #dce4ef;
    --panel:    #ffffff;
    --panel2:   #f7f9fc;
    --border:   rgba(59,130,246,0.15);
    --border2:  rgba(59,130,246,0.30);
    --shadow:   0 1px 4px rgba(30,60,110,0.08), 0 4px 16px rgba(30,60,110,0.06);
    --shadow-lg:0 4px 20px rgba(30,60,110,0.12);
    --text:     #1e293b;
    --text-mid: #475569;
    --text-dim: #94a3b8;
    --accent:   #2563eb;
    --accent2:  #0891b2;
    --green:    #059669;
    --red:      #dc2626;
    --amber:    #d97706;
    --purple:   #7c3aed;
    --teal:     #0d9488;
    --mono:'IBM Plex Mono',monospace;
    --sans:'DM Sans','Noto Sans TC',sans-serif;
    --radius:10px; --radius-lg:16px; --radius-sm:6px;
}

html,body,.stApp { background:var(--bg)!important; color:var(--text)!important; font-family:var(--sans); }
[data-testid="stSidebar"] { background:var(--panel)!important; border-right:1px solid var(--border); }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding:1.2rem 2rem 3rem!important; max-width:1700px; }

.hq-header {
    display:flex; align-items:center; gap:16px;
    padding:18px 28px; margin-bottom:10px;
    background:linear-gradient(135deg,#ffffff,#f0f6ff);
    border:1px solid var(--border2); border-radius:var(--radius-lg);
    box-shadow:var(--shadow-lg); position:relative; overflow:hidden;
}
.hq-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,var(--accent),var(--accent2),var(--teal));
}
.hq-title { font-size:1.55rem; font-weight:900; color:var(--text); letter-spacing:-0.5px; }
.hq-sub   { font-size:0.75rem; color:var(--text-dim); font-family:var(--mono); margin-top:3px; }
.hq-badge { margin-left:auto; padding:6px 16px; border-radius:20px; font-size:0.72rem; font-weight:700; font-family:var(--mono); }

.status-row { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
.status-chip { display:flex; align-items:center; gap:6px; padding:5px 12px; border-radius:20px;
               font-size:0.72rem; font-family:var(--mono); border:1px solid; background:var(--panel); }
.chip-ok   { border-color:rgba(5,150,105,0.3);  color:var(--green); }
.chip-err  { border-color:rgba(220,38,38,0.3);  color:var(--red); }
.chip-info { border-color:rgba(37,99,235,0.25); color:var(--accent); }
.chip-warn { border-color:rgba(217,119,6,0.3);  color:var(--amber); }
.dot { width:7px; height:7px; border-radius:50%; }
.dot-g { background:var(--green); animation:pulse 2s infinite; }
.dot-r { background:var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

.sec-header {
    display:flex; align-items:center; gap:12px;
    padding:12px 20px; margin:18px 0 12px;
    border-radius:var(--radius); border-left:3px solid;
    background:var(--panel); box-shadow:var(--shadow);
}
.sec-v4     { border-color:var(--accent); }
.sec-v12    { border-color:var(--green); }
.sec-regime { border-color:var(--amber); }
.sec-ai     { border-color:var(--purple); }
.sec-watch  { border-color:var(--teal); }
.sec-buy    { border-color:var(--green); }
.sec-sell   { border-color:var(--red); }
.sec-bt     { border-color:var(--purple); }
.sec-label  { font-size:0.67rem; font-family:var(--mono); color:var(--text-dim); margin-left:auto; }

.card { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px 18px; margin-bottom:10px; box-shadow:var(--shadow); }
.card:hover { border-color:var(--border2); box-shadow:var(--shadow-lg); }

.mono-num { font-family:var(--mono); font-weight:700; }
.c-green  { color:var(--green)!important; }
.c-red    { color:var(--red)!important; }
.c-amber  { color:var(--amber)!important; }
.c-blue   { color:var(--accent)!important; }
.c-cyan   { color:var(--accent2)!important; }
.c-purple { color:var(--purple)!important; }
.c-teal   { color:var(--teal)!important; }
.c-dim    { color:var(--text-dim)!important; }
.c-mid    { color:var(--text-mid)!important; }
.c-g { color:#059669; } .c-r { color:#dc2626; }
.c-b { color:#2563eb; } .c-a { color:#d97706; }

.pill {
    display:inline-block; padding:2px 9px; border-radius:12px;
    font-size:0.69rem; font-weight:700; font-family:var(--mono); border:1px solid; margin:1px;
}
.pill-g { background:rgba(5,150,105,0.08);  border-color:rgba(5,150,105,0.3);  color:var(--green); }
.pill-r { background:rgba(220,38,38,0.08);  border-color:rgba(220,38,38,0.3);  color:var(--red); }
.pill-b { background:rgba(37,99,235,0.08);  border-color:rgba(37,99,235,0.3);  color:var(--accent); }
.pill-a { background:rgba(217,119,6,0.08);  border-color:rgba(217,119,6,0.3);  color:var(--amber); }
.pill-p { background:rgba(124,58,237,0.08); border-color:rgba(124,58,237,0.3); color:var(--purple); }
.pill-c { background:rgba(8,145,178,0.08);  border-color:rgba(8,145,178,0.3);  color:var(--accent2); }
.pill-t { background:rgba(13,148,136,0.08); border-color:rgba(13,148,136,0.3); color:var(--teal); }
.pill-cand { background:rgba(217,119,6,0.10); border-color:rgba(217,119,6,0.4); color:var(--amber); }

.path-tag { display:inline-block; padding:2px 9px; border-radius:var(--radius-sm); font-family:var(--mono); font-size:0.72rem; font-weight:700; }
.path-45  { background:rgba(5,150,105,0.10);  color:var(--green); border:1px solid rgba(5,150,105,0.3); }
.path-423 { background:rgba(37,99,235,0.10);  color:var(--accent); border:1px solid rgba(37,99,235,0.3); }
.path-na  { background:rgba(148,163,184,0.12); color:var(--text-dim); border:1px solid var(--border); }

.rank-badge { display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:50%; font-size:0.7rem; font-weight:900; font-family:var(--mono); }
.rank-1 { background:linear-gradient(135deg,#f59e0b,#d97706); color:#fff; }
.rank-2 { background:linear-gradient(135deg,#94a3b8,#64748b); color:#fff; }
.rank-3 { background:linear-gradient(135deg,#cd7c3a,#b45309); color:#fff; }
.rank-n { background:var(--bg3); color:var(--text-dim); border:1px solid var(--border); }

.data-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.data-table th { background:var(--bg2); color:var(--text-dim); font-weight:600; font-family:var(--mono);
                 font-size:0.68rem; letter-spacing:0.5px; padding:9px 12px; text-align:left;
                 border-bottom:2px solid var(--border2); position:sticky; top:0; z-index:1; }
.data-table td { padding:9px 12px; border-bottom:1px solid rgba(59,130,246,0.07); color:var(--text); }
.data-table tr:hover td { background:rgba(37,99,235,0.03); }
.data-table .candidate-row td { background:rgba(217,119,6,0.04); }
.data-table .candidate-row:hover td { background:rgba(217,119,6,0.08); }
.candidate-divider td { background:rgba(217,119,6,0.06)!important; border-top:2px dashed rgba(217,119,6,0.4)!important;
                        padding:5px 12px!important; font-size:0.67rem; font-family:var(--mono); color:var(--amber)!important; font-weight:700; }

.mkt-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:10px; margin:12px 0; }
.mkt-cell { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:14px 10px; text-align:center; box-shadow:var(--shadow); }
.mkt-cell:hover { box-shadow:var(--shadow-lg); border-color:var(--border2); }
.mkt-val  { font-size:1.15rem; font-weight:900; font-family:var(--mono); color:var(--text); }
.mkt-lbl  { font-size:0.62rem; color:var(--text-dim); margin-top:4px; letter-spacing:0.5px; }

.ai-box { background:linear-gradient(135deg,rgba(124,58,237,0.04),rgba(37,99,235,0.04));
          border:1px solid rgba(124,58,237,0.2); border-left:3px solid var(--purple);
          border-radius:var(--radius); padding:18px 22px; font-size:0.95rem; line-height:1.9; color:var(--text); }

.reason-box { background:#f8fafc; border:1px solid rgba(59,130,246,0.15);
              border-left:3px solid #2563eb; border-radius:8px;
              padding:12px 16px; font-size:0.83rem; line-height:1.8; color:#1e293b; margin:6px 0; }
.sell-box   { border-left-color:#dc2626; background:#fef2f2; }
.buy-box    { border-left-color:#059669; background:#f0fdf4; }

.watch-star { color:#d97706; font-size:0.9rem; }
.mono { font-family:monospace; font-weight:700; }

[data-testid="stTab"] button { color:var(--text-mid)!important; font-family:var(--sans)!important; background:transparent!important; }
[data-testid="stTab"] button[aria-selected="true"] { color:var(--accent)!important; font-weight:700!important; border-bottom:2px solid var(--accent)!important; }
.stButton>button { background:linear-gradient(135deg,var(--accent),#1d4ed8)!important; color:#fff!important;
                   border:none!important; border-radius:var(--radius)!important; font-weight:700!important; }
[data-testid="stMetricValue"] { color:var(--accent)!important; font-family:var(--mono)!important; font-size:1.3rem!important; font-weight:700!important; }
[data-testid="stMetricLabel"] { color:var(--text-mid)!important; }
div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:12px 16px; box-shadow:var(--shadow); }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session 初始化
# ══════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "gemini_key": _ENV_GEMINI_KEY, "ai_summary": "",
        "single_sym": "", "single_result": "",
        "use_mock": False, "last_refresh": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ══════════════════════════════════════════════════════════════
# 資料載入
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def load_json_url(path_suffix: str) -> Optional[dict]:
    url = f"{BASE_URL}/{path_suffix}"
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_url(path_suffix: str) -> Optional[dict]:
    return load_json_url(path_suffix)

@st.cache_data(ttl=600, show_spinner=False)
def load_stock_set() -> dict:
    try:
        r = requests.get(f"{REPO_RAW}/stock_set.json", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

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

def load_all_snapshots():
    v4     = load_json_url("v4/v4_latest.json")
    v12    = load_json_url("v12/v12_latest.json")
    regime = load_json_url("regime/regime_state.json")
    market = load_json_url("market/market_snapshot.json")
    status = {
        "v4": v4 is not None, "v12": v12 is not None,
        "regime": regime is not None, "market": market is not None,
    }
    return v4, v12, regime, market, status

def load_all_v5():
    portfolio  = load_url("portfolio_latest.json")
    backtest   = load_url("backtest_result.json")
    trades_df  = load_csv_url("trades.csv")
    return portfolio, backtest, trades_df


# ──────────────────────────────────────────────────────────────
# Mock 資料
# ──────────────────────────────────────────────────────────────
def _mock_v4():
    import random
    syms = ["2330","2317","2454","2308","2382","3711","2412","6669","3008","2395",
            "2379","3034","2345","3443","3661","6415","3035","2408","3131","5274"]
    rows = []
    for i, s in enumerate(syms):
        score = round(85 - i*1.8 + random.uniform(-2, 2), 2)
        close = round(random.uniform(100, 900), 1)
        atr_pct = round(random.uniform(0.015, 0.04), 4)
        tp1 = round(close * 1.20, 1)
        stop = round(close * (1 - atr_pct * 1.5), 1)
        rows.append({
            "rank": i+1, "symbol": s, "score": score,
            "pvo": round(random.uniform(-5, 20), 2),
            "vri": round(random.uniform(35, 95), 1),
            "slope_z": round(random.uniform(-0.5, 2.5), 2),
            "action": random.choice(["強力買進","買進","觀察"]),
            "signal": random.choice(["三合一(ABC)","二合一(AB)","單一(A)"]),
            "close": close, "tp1_price": tp1, "stop_price": stop,
            "regime": "range",
            "is_watchlist": s in ["2330","2317","2454","2308","2382"],
        })
    return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market": "TW", "top20": rows, "top30": rows,
            "pool_mu": 62.3, "pool_sigma": 11.5, "win_rate": 57.1,
            "run_mode": "postmarket"}

def _mock_v12():
    import random
    syms = ["2330","2454","3711","6669","3008","2379","3443","6415","3035","2408"]
    rows = []
    for s in syms:
        ev = round(random.uniform(2.0, 9.5), 2)
        close = round(random.uniform(100, 900), 1)
        tp1 = round(close*1.20, 1); stop = round(close*0.90, 1)
        rows.append({
            "symbol": s, "path": random.choice(["45","423"]),
            "ev": ev, "action": random.choice(["持有","進場","觀察"]),
            "exit_signal": random.choice(["無","—","EV衰退"]),
            "quality": "Pure", "ev_tier": "⭐核心" if ev>5 else "🔥主力",
            "regime": "range", "days_held": random.randint(0, 18),
            "curr_ret_pct": round(random.uniform(-5, 15), 2),
            "tp1_price": tp1, "stop_price": stop, "close": close,
            "is_watchlist": s in ["2330","2454"],
        })
    candidates = []
    for s in ["5274","3131","4763","6148","3293"]:
        ev = round(random.uniform(2.0, 5.0), 2)
        close = round(random.uniform(80, 500), 1)
        candidates.append({
            "symbol": s, "path": random.choice(["45","423"]),
            "ev": ev, "action": "候選", "exit_signal": "—", "quality": "Pure",
            "ev_tier": "📌補位", "regime": "range", "days_held": 0,
            "curr_ret_pct": 0.0, "tp1_price": round(close*1.20, 1),
            "stop_price": round(close*0.90, 1), "close": close, "is_watchlist": False,
        })
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "positions": rows, "candidates": candidates, "run_mode": "postmarket",
        "stats": {"total_trades":112,"win_rate":57.1,"avg_ev":5.29,
                  "max_dd":-6.58,"sharpe":5.36,"t_stat":4.032,"simple_cagr":96.9,"pl_ratio":2.31},
    }

def _mock_regime():
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bear":0.00,"range":0.30,"bull":0.70,"label":"牛市",
        "active_strategy":"bull","active_path":"45","backup_path":"423",
        "slope_5d":0.0041,"slope_20d":0.0165,"mkt_rsi":100.0,"adx":24.8,
        "index_close": 21000.0, "index_chg_pct": 0.08,
        "data_source": "FinMind_Y9999_TAIEX",
        "history":[
            {"month":"2026-01","bear":0.35,"range":0.45,"bull":0.20,"label":"偏空","index_close":19800},
            {"month":"2026-02","bear":0.28,"range":0.48,"bull":0.24,"label":"震盪","index_close":20200},
            {"month":"2026-03","bear":0.10,"range":0.43,"bull":0.47,"label":"偏多","index_close":20700},
            {"month":"2026-04","bear":0.00,"range":0.30,"bull":0.70,"label":"牛市","index_close":21000},
        ]
    }

def _mock_market():
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "index_close":21000.0,"index_chg_pct":0.08,
        "mkt_rsi":100.0,"mkt_slope_5d":0.0041,"mkt_slope_20d":0.0165,
        "data_source":"FinMind_Y9999_TAIEX","run_mode":"postmarket",
    }


# ══════════════════════════════════════════════════════════════
# HTML 渲染工具
# ══════════════════════════════════════════════════════════════
def _render_html(html_body: str, height: int = 400):
    full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=DM+Sans:wght@400;600;700&family=Noto+Sans+TC:wght@400;600;900&display=swap');
:root{{
    --bg:#f0f4f8; --bg2:#e8edf4; --panel:#ffffff;
    --border:rgba(59,130,246,0.15); --border2:rgba(59,130,246,0.30);
    --text:#1e293b; --text-mid:#475569; --text-dim:#94a3b8;
    --accent:#2563eb; --accent2:#0891b2; --green:#059669;
    --red:#dc2626; --amber:#d97706; --purple:#7c3aed; --teal:#0d9488;
}}
body{{background:transparent;color:var(--text);font-family:'DM Sans','Noto Sans TC',sans-serif;margin:0;padding:0;height:{height}px;overflow-y:auto;overflow-x:hidden;}}
.data-table{{width:100%;border-collapse:collapse;font-size:0.82rem;}}
.data-table th{{background:var(--bg2);color:var(--text-dim);font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:0.67rem;letter-spacing:0.5px;padding:9px 12px;text-align:left;border-bottom:2px solid var(--border2);position:sticky;top:0;z-index:1;}}
.data-table td{{padding:9px 12px;border-bottom:1px solid rgba(59,130,246,0.07);vertical-align:middle;color:var(--text);}}
.data-table tr:hover td{{background:rgba(37,99,235,0.03);}}
.data-table .candidate-row td{{background:rgba(217,119,6,0.04);}}
.data-table .candidate-row:hover td{{background:rgba(217,119,6,0.08);}}
.candidate-divider td{{background:rgba(217,119,6,0.06)!important;border-top:2px dashed rgba(217,119,6,0.4)!important;padding:5px 12px!important;font-size:0.67rem;font-family:'IBM Plex Mono',monospace;color:var(--amber)!important;font-weight:700;letter-spacing:1px;}}
.mono-num{{font-family:'IBM Plex Mono',monospace;font-weight:700;}}
.c-green{{color:var(--green)!important;}}.c-red{{color:var(--red)!important;}}
.c-amber{{color:var(--amber)!important;}}.c-blue{{color:var(--accent)!important;}}
.c-cyan{{color:var(--accent2)!important;}}.c-purple{{color:var(--purple)!important;}}
.c-teal{{color:var(--teal)!important;}}.c-dim{{color:var(--text-dim)!important;}}
.c-mid{{color:var(--text-mid)!important;}}
.pill{{display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.68rem;font-weight:700;font-family:'IBM Plex Mono',monospace;border:1px solid;margin:1px;white-space:nowrap;}}
.pill-g{{background:rgba(5,150,105,0.08);border-color:rgba(5,150,105,0.3);color:#059669;}}
.pill-r{{background:rgba(220,38,38,0.08);border-color:rgba(220,38,38,0.3);color:#dc2626;}}
.pill-b{{background:rgba(37,99,235,0.08);border-color:rgba(37,99,235,0.3);color:#2563eb;}}
.pill-a{{background:rgba(217,119,6,0.08);border-color:rgba(217,119,6,0.3);color:#d97706;}}
.pill-p{{background:rgba(124,58,237,0.08);border-color:rgba(124,58,237,0.3);color:#7c3aed;}}
.pill-c{{background:rgba(8,145,178,0.08);border-color:rgba(8,145,178,0.3);color:#0891b2;}}
.pill-t{{background:rgba(13,148,136,0.08);border-color:rgba(13,148,136,0.3);color:#0d9488;}}
.pill-cand{{background:rgba(217,119,6,0.10);border-color:rgba(217,119,6,0.4);color:#d97706;}}
.rank-badge{{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font-size:0.7rem;font-weight:900;font-family:'IBM Plex Mono',monospace;}}
.rank-1{{background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff;}}
.rank-2{{background:linear-gradient(135deg,#94a3b8,#64748b);color:#fff;}}
.rank-3{{background:linear-gradient(135deg,#cd7c3a,#b45309);color:#fff;}}
.rank-n{{background:#e8edf4;color:#94a3b8;border:1px solid rgba(59,130,246,0.15);}}
.path-tag{{display:inline-block;padding:2px 9px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;font-weight:700;}}
.path-45{{background:rgba(5,150,105,0.10);color:#059669;border:1px solid rgba(5,150,105,0.3);}}
.path-423{{background:rgba(37,99,235,0.10);color:#2563eb;border:1px solid rgba(37,99,235,0.3);}}
.path-na{{background:rgba(148,163,184,0.10);color:#94a3b8;border:1px solid rgba(59,130,246,0.15);}}
.watch-star{{color:#d97706;font-size:0.9rem;}}
.reason-box{{background:#f8fafc;border:1px solid rgba(59,130,246,0.15);border-left:3px solid #2563eb;border-radius:8px;padding:12px 16px;font-size:0.83rem;line-height:1.8;color:#1e293b;margin:6px 0;}}
.sell-box{{border-left-color:#dc2626;background:#fef2f2;}}
.buy-box{{border-left-color:#059669;background:#f0fdf4;}}
</style></head><body>{html_body}</body></html>"""
    st.html(full_html)


# ──────────────────────────────────────────────────────────────
# 輔助 pill 函式
# ──────────────────────────────────────────────────────────────
def _action_pill(action: str) -> str:
    mapping = {
        "強力買進":("pill-g","▲ 強力買進"), "買進":("pill-g","▲ 買進"),
        "持有":("pill-b","◆ 持有"), "進場":("pill-g","▲ 進場"),
        "觀察":("pill-a","◇ 觀察"), "賣出":("pill-r","▼ 賣出"),
        "出場":("pill-r","▼ 出場"), "候選":("pill-cand","◈ 候選"),
        "BUY":("pill-g","▲ 買進"), "SELL_STOP":("pill-r","🛑 停損"),
        "SELL_TP1":("pill-g","🎯 TP1"), "SELL_TP2":("pill-g","🎯 TP2"),
        "SELL_TRAIL":("pill-a","↘ 移停"), "SELL_EV":("pill-a","⚠ EV退"),
        "SELL_REPLACE":("pill-b","🔄 換股"),
    }
    css, label = mapping.get(action, ("pill-c", action))
    return f'<span class="pill {css}">{label}</span>'

def _path_tag(path: str) -> str:
    css = {"45":"path-45","423":"path-423"}.get(str(path),"path-na")
    return f'<span class="path-tag {css}">{path}</span>'

def _quality_pill(q: str) -> str:
    if q == "Pure":
        return '<span class="pill pill-g">✅ Pure</span>'
    return '<span class="pill pill-a">〔F〕Flicker</span>'

def _exit_pill(sig: str) -> str:
    if not sig or sig in ("—","無"):
        return '<span class="pill pill-b">持倉中</span>'
    if "停利" in sig:
        return f'<span class="pill pill-g">🎯 {sig}</span>'
    if any(x in sig for x in ["衰退","枯竭","衰減","加速","Slope"]):
        return f'<span class="pill pill-a">⚠️ {sig}</span>'
    if "停損" in sig:
        return f'<span class="pill pill-r">🛑 {sig}</span>'
    return f'<span class="pill pill-c">{sig}</span>'

def _watch_star(is_watch: bool) -> str:
    return '<span class="watch-star">★</span>&nbsp;' if is_watch else ""

def render_regime_bar(bear: float, range_: float, bull: float):
    b, r, u = bear*100, range_*100, bull*100
    st.markdown(f"""
    <div style="display:flex;height:40px;border-radius:10px;overflow:hidden;border:1px solid rgba(59,130,246,0.2);box-shadow:0 1px 4px rgba(30,60,110,0.08);">
        <div style="width:{b:.0f}%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;background:rgba(220,38,38,0.12);color:#dc2626;">{'熊 '+f'{b:.0f}%' if b>5 else ''}</div>
        <div style="width:{r:.0f}%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;background:rgba(217,119,6,0.12);color:#d97706;">{'震 '+f'{r:.0f}%' if r>5 else ''}</div>
        <div style="width:{u:.0f}%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;background:rgba(5,150,105,0.12);color:#059669;">{'牛 '+f'{u:.0f}%' if u>5 else ''}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Gemini AI
# ══════════════════════════════════════════════════════════════
def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ 請在側欄設定 Gemini API Key。"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        for m in ["gemma-3-27b-it","gemini-2.0-flash","gemini-1.5-flash"]:
            try:
                model = genai.GenerativeModel(m)
                return model.generate_content(prompt).text
            except Exception:
                continue
        return "❌ 模型無法回應"
    except Exception as e:
        return f"❌ Gemini 錯誤：{e}"

def build_dashboard_prompt(v4, v12, regime, market) -> str:
    top5 = (v4 or {}).get("top20",[])[:5]
    top5_txt = "\n".join([
        f"  #{r['rank']} {r['symbol']} {'★' if r.get('is_watchlist') else ''}"
        f" | Score:{r['score']} | PVO:{r.get('pvo',0):+.2f}"
        f" | VRI:{r.get('vri',0):.1f} | 訊號:{r.get('signal','—')}"
        for r in top5
    ]) if top5 else "（無資料）"

    pos = (v12 or {}).get("positions",[])
    pos_txt = "\n".join([
        f"  {p['symbol']} {'★' if p.get('is_watchlist') else ''}"
        f" | {p['path']} | EV:{p['ev']:+.2f}%"
        f" | 持:{p.get('days_held',0)}日 | 報酬:{p.get('curr_ret_pct',0):+.2f}%"
        f" | 出場:{p.get('exit_signal','—')}"
        for p in pos[:8]
    ]) if pos else "（無部位）"

    r  = regime or _mock_regime()
    mk = market  or _mock_market()
    s  = (v12 or {}).get("stats",{})
    src = mk.get("data_source","—")
    idx = mk.get("index_close","—")

    return f"""你是資深量化交易分析師，精通台股統計套利。以下是今日資源法系統快照。

【大盤環境】（資料來源: {src}）
加權指數: {idx} | RSI:{r.get('mkt_rsi',0):.1f} | 漲跌:{mk.get('index_chg_pct',0):+.2f}%
Regime: 熊{r.get('bear',0)*100:.0f}% 震{r.get('range',0)*100:.0f}% 牛{r.get('bull',0)*100:.0f}%
策略: {r.get('active_strategy','—')} | 主路徑:{r.get('active_path','—')} | 5日斜率:{r.get('slope_5d',0):+.4f}

【V4 TOP5（★=自選股）】
{top5_txt}

【V12.1 部位（★=自選股）】
{pos_txt}

【歷史統計】總筆:{s.get('total_trades','N/A')} 勝率:{s.get('win_rate','N/A')}% t值:{s.get('t_stat','N/A')}

【分析要求】每段100字以內，引用具體數值：
**一、大盤評估**：Regime三維解讀 + 斜率含義 + 建議持倉水位
**二、V4訊號**：TOP3標的技術評估
**三、部位診斷**：現有部位健康度，出場信號警惕
**四、操作清單**：今日優先行動（進/出/觀察）+ 風險因子"""


# ══════════════════════════════════════════════════════════════
# 側欄
# ══════════════════════════════════════════════════════════════
def render_sidebar(stock_set: dict):
    with st.sidebar:
        st.markdown("## 📊 資源法 v5.1")
        st.markdown("---")
        st.markdown("### 🔑 Gemini API Key")
        if _ENV_GEMINI_KEY:
            st.success("✅ 環境變數已載入")
            if st.checkbox("手動覆蓋"):
                k = st.text_input("API Key（覆蓋）", type="password")
                if k: st.session_state.gemini_key = k
        else:
            k = st.text_input("Gemini API Key", type="password",
                              value=st.session_state.gemini_key, placeholder="AIza...")
            st.session_state.gemini_key = k

        st.markdown("---")
        st.markdown("### 📋 自選股清單")
        watchlist = stock_set.get("watchlist", {}).get("symbols", [])
        if watchlist:
            for sym in watchlist:
                st.caption(f"★ {sym}")
        else:
            st.caption("（未設定）")
        st.caption("[編輯 stock_set.json 更新]")

        st.markdown("---")
        st.markdown("### 🔗 GitHub 設定")
        st.caption(f"Owner: `{GITHUB_OWNER}`")
        st.caption(f"Repo: `{GITHUB_REPO}`")

        st.markdown("---")
        st.markdown("### ⚙️ 開發選項")
        use_mock = st.checkbox("使用模擬資料（Demo）", value=st.session_state.use_mock)
        st.session_state.use_mock = use_mock
        if use_mock:
            st.warning("⚠️ 目前顯示模擬數據")

        st.markdown("---")
        st.caption("資料更新時程（台灣時間）")
        for t in [
            "09:35 開盤快照 [盤中]","10:30 盤中更新 [盤中]","11:30 盤中更新 [盤中]",
            "13:25 盤中收盤前 [盤中]","15:30 盤後全量掃描 [盤後]","20:00 日結存檔 [盤後]",
        ]:
            st.caption(f"• {t}")
        st.caption("---")
        st.caption("© 2026 資源法 AI 戰情室 v5.1")


# ══════════════════════════════════════════════════════════════
# Section: 自選股快覽（V4.1-01/02）
# ══════════════════════════════════════════════════════════════
def render_watchlist_section(v4: dict, v12: dict, watchlist: list):
    top20      = (v4 or {}).get("top20", [])
    positions  = (v12 or {}).get("positions", [])

    v4_map  = {r["symbol"]: r for r in top20}
    v12_map = {p["symbol"]: p for p in positions}

    display_set = {}
    for r in top20:
        if r.get("action") in ("強力買進","買進"):
            display_set[r["symbol"]] = "v4"
    for p in positions:
        display_set[p["symbol"]] = "v12"
    for s in (watchlist or []):
        if s not in display_set:
            display_set[s] = "watch"

    if not display_set:
        return

    total = len(display_set)
    st.markdown(f"""
    <div class="sec-header sec-watch">
        <span style="font-size:1.0rem;font-weight:900;color:#0d9488;">⭐ 自選股快覽</span>
        <span style="color:var(--text-mid);font-size:0.8rem;margin-left:8px;">
            V4持倉 + V12持倉 + 指定監控 共 <b>{total}</b> 檔
        </span>
        <span class="pill pill-t" style="margin-left:8px;">含停利/停損</span>
    </div>
    """, unsafe_allow_html=True)

    html = """<table class="data-table"><thead><tr>
        <th>來源</th><th>代號</th><th>V4 Score</th><th>操作</th><th>訊號</th>
        <th>V12路徑</th><th>EV</th><th>出場信號</th>
        <th>持天</th><th>現價</th><th>報酬%</th><th>停利①</th><th>停損</th>
    </tr></thead><tbody>"""

    for sym, src_type in display_set.items():
        r4  = v4_map.get(sym, {})
        r12 = v12_map.get(sym, {})
        sym_e = _html_escape.escape(sym)
        is_w  = sym in (watchlist or [])

        if src_type == "v4":
            src_tag = '<span class="pill pill-b">V4</span>'
        elif src_type == "v12":
            src_tag = '<span class="pill pill-g">V12</span>'
        else:
            src_tag = '<span class="pill pill-t">★自選</span>'

        score  = r4.get("score", None)
        action = r4.get("action", r12.get("action", "—"))
        signal = _html_escape.escape(str(r4.get("signal", "—")))
        path   = r12.get("path", "—")
        ev     = r12.get("ev", None)
        exs    = r12.get("exit_signal", "—")
        days   = r12.get("days_held", None)
        ret    = r12.get("curr_ret_pct", None)
        close  = r4.get("close", r12.get("close", None))
        tp1    = r12.get("tp1_price", r4.get("tp1_price", None))
        stop   = r12.get("stop_price", r4.get("stop_price", None))

        score_str = f"{score:.2f}" if score is not None else "—"
        score_css = "c-green" if score and score>=70 else "c-amber" if score and score>=55 else "c-dim"
        ev_str    = f"{ev:+.2f}%" if ev is not None else "—"
        ev_css    = "c-green" if ev and ev>5 else "c-cyan" if ev and ev>3 else "c-dim"
        ret_str   = f"{ret:+.2f}%" if ret is not None else "—"
        ret_css   = "c-green" if ret and ret>0 else "c-red" if ret and ret<0 else "c-dim"
        close_str = f"{close:.1f}" if isinstance(close, float) else "—"
        days_str  = str(days) if days is not None else "—"
        tp1_str   = f"{tp1:.1f}" if isinstance(tp1, float) else "—"
        stop_str  = f"{stop:.1f}" if isinstance(stop, float) else "—"

        if "三合一" in signal: sig_css = "pill-p"
        elif "二合一" in signal: sig_css = "pill-b"
        elif "單一" in signal: sig_css = "pill-a"
        else: sig_css = "pill-c"

        watch_html = _watch_star(is_w)
        html += f"""<tr>
            <td>{src_tag}</td>
            <td>{watch_html}<b style="color:#1e293b;">{sym_e}</b></td>
            <td><span class="mono-num {score_css}">{score_str}</span></td>
            <td>{_action_pill(action)}</td>
            <td><span class="pill {sig_css}" style="font-size:0.63rem;">{signal}</span></td>
            <td>{_path_tag(path) if path != "—" else '<span class="c-dim">—</span>'}</td>
            <td><span class="mono-num {ev_css}">{ev_str}</span></td>
            <td>{_exit_pill(exs)}</td>
            <td class="c-mid mono-num">{days_str}</td>
            <td class="mono-num c-mid">{close_str}</td>
            <td><span class="mono-num {ret_css}">{ret_str}</span></td>
            <td class="mono-num c-green">{tp1_str}</td>
            <td class="mono-num c-red">{stop_str}</td>
        </tr>"""

    html += "</tbody></table>"
    _render_html(html, height=min(90 + total * 48, 600))


# ══════════════════════════════════════════════════════════════
# Section: 個股 AI 分析
# ══════════════════════════════════════════════════════════════
def render_single_stock_panel(v4: dict, v12: dict, regime: dict):
    st.markdown("""
    <div class="sec-header sec-ai">
        <span style="font-size:1.0rem;font-weight:800;color:#7c3aed;">🔍 個股 AI 深度分析</span>
        <span style="color:var(--text-mid);font-size:0.8rem;margin-left:10px;">輸入代號 → Gemini 個股報告</span>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_btn = st.columns([3,1])
    with col_in:
        sym_input = st.text_input("個股代號", placeholder="例：2330",
                                  label_visibility="collapsed", key="single_sym_input")
    with col_btn:
        analyze_btn = st.button("🔍 分析", use_container_width=True, key="single_btn")

    if st.session_state.single_result and st.session_state.single_sym:
        safe = _html_escape.escape(st.session_state.single_result).replace('\n','<br>')
        st.markdown(f"""
        <div class="ai-box" style="margin-top:12px;">
            <div style="font-weight:700;color:#7c3aed;margin-bottom:10px;
                        padding-bottom:8px;border-bottom:1px solid rgba(124,58,237,0.15);">
                🔍 {st.session_state.single_sym} 個股分析報告
            </div>
            {safe}
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn and sym_input:
        sym_q  = sym_input.strip().upper()
        top20  = (v4 or {}).get("top20",[])
        v4_row = next((r for r in top20 if r["symbol"]==sym_q), None)
        v12pos = (v12 or {}).get("positions",[])
        v12row = next((p for p in v12pos if p["symbol"]==sym_q), None)
        r      = regime or _mock_regime()

        v4_txt  = (f"Score:{v4_row['score']} | PVO:{v4_row.get('pvo',0):+.2f} | VRI:{v4_row.get('vri',0):.1f} | 訊號:{v4_row.get('signal','—')} | 停利:{v4_row.get('tp1_price','—')} | 停損:{v4_row.get('stop_price','—')}" if v4_row else "（V4無資料）")
        v12_txt = (f"路徑:{v12row['path']} | EV:{v12row['ev']:+.2f}% | 持:{v12row.get('days_held',0)}日 | 報酬:{v12row.get('curr_ret_pct',0):+.2f}% | 出場:{v12row.get('exit_signal','—')}" if v12row else "（V12無部位）")
        prompt  = f"""分析 {sym_q}（{'★自選股' if v4_row and v4_row.get('is_watchlist') else '一般股'}），每段100字以內：
V4: {v4_txt}
V12: {v12_txt}
Regime: 熊{r.get('bear',0)*100:.0f}% 震{r.get('range',0)*100:.0f}% 牛{r.get('bull',0)*100:.0f}% | 主路徑:{r.get('active_path','—')}
請分析：**一、技術面** **二、路徑判斷** **三、操作建議** **四、風險提示**"""

        with st.spinner(f"🤖 分析 {sym_q}..."):
            result = call_gemini(prompt, st.session_state.gemini_key)
        st.session_state.single_sym    = sym_q
        st.session_state.single_result = result
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Section: 今日買進原因（v5.0）
# ══════════════════════════════════════════════════════════════
def render_buy_reasons(portfolio: dict):
    bought = portfolio.get("bought_today", [])
    if not bought:
        st.info("今日無新進場標的")
        return

    st.markdown(f"""
    <div class="sec-header sec-buy">
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
                html_parts += f'<div>{_html_escape.escape(p)}</div>'

        html_body = f"""<div style="margin-bottom:12px;">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
  <b style="font-size:1.0rem;color:#1e293b;">{_html_escape.escape(sym)}</b>
  {_path_tag(path)}
  <span class="pill pill-b">EV {ev_pct:.2f}%</span>
  <span class="pill pill-g">進場 {price:.1f} × {shares} 股</span>
</div>
<div class="reason-box buy-box">{html_parts}</div>
</div>"""
        _render_html(html_body, height=160)


# ══════════════════════════════════════════════════════════════
# Section: 賣出訊號（v5.0）
# ══════════════════════════════════════════════════════════════
def render_sell_signals(portfolio: dict):
    positions = portfolio.get("positions", [])
    sell_signals = [p for p in positions if p.get("exit_signal","—") not in ("—","","無")]

    st.markdown(f"""
    <div class="sec-header sec-sell">
      <span style="font-size:1.05rem;font-weight:900;color:#dc2626;">
        ▼ 賣出訊號監控 {len(sell_signals)} 檔
      </span>
    </div>""", unsafe_allow_html=True)

    if not sell_signals:
        st.info("目前無待出場訊號")
        return

    for p in sell_signals:
        sym  = p.get("symbol","—")
        sig  = p.get("exit_signal","—")
        ret  = p.get("curr_ret_pct",0)
        ret_c = "c-g" if ret > 0 else "c-r"
        html_body = f"""<div class="reason-box sell-box" style="margin-bottom:8px;">
<b>{_html_escape.escape(sym)}</b> &nbsp;
{_exit_pill(sig)} &nbsp;
<span class="mono {ret_c}">{ret:+.2f}%</span>
</div>"""
        _render_html(html_body, height=80)


# ══════════════════════════════════════════════════════════════
# Section: 持倉監控（v5.0）
# ══════════════════════════════════════════════════════════════
def render_positions(portfolio: dict):
    positions = portfolio.get("positions", [])
    if not positions:
        st.info("目前無持倉")
        return

    n = len(positions)
    st.markdown(f"""
    <div class="sec-header sec-v12">
      <span style="font-size:1.05rem;font-weight:900;color:#059669;">
        📋 持倉監控 {n} 檔
      </span>
    </div>""", unsafe_allow_html=True)

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


# ══════════════════════════════════════════════════════════════
# Section: V4 市場強度（V4.1-04）
# ══════════════════════════════════════════════════════════════
def render_v4_section(v4: dict):
    gen_at = v4.get("generated_at","—")
    mode   = v4.get("run_mode","—")
    top20  = v4.get("top20",[])
    mu     = v4.get("pool_mu",0)
    sigma  = v4.get("pool_sigma",0)

    st.markdown(f"""
    <div class="sec-header sec-v4">
        <span style="font-size:1.1rem;font-weight:900;color:#2563eb;">🟦 V4 市場強度快照</span>
        <span class="pill pill-b">TOP 30</span>
        <span class="pill pill-{'g' if mode=='postmarket' else 'a'}">{mode}</span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Pool μ", f"{mu:.2f}")
    c2.metric("Pool σ", f"{sigma:.2f}")
    c3.metric("歷史勝率", f"{v4.get('win_rate',0):.1f}%")
    c4.metric("觀察標的", len(top20))

    if not top20:
        st.info("⏳ 等待 GitHub 資料更新（或啟用 Demo 模式）")
        return

    cf1,cf2,cf3 = st.columns([2,2,2])
    with cf1:
        filter_action = st.multiselect("操作過濾",["強力買進","買進","觀察","賣出"],
                                        default=["強力買進","買進"],key="v4_filter")
    with cf2:
        min_vri = st.slider("VRI 最低",0,100,40,key="v4_vri")
    with cf3:
        sort_by = st.selectbox("排序",["rank","score","slope_z","vri","pvo"],key="v4_sort")

    filtered = [r for r in top20
                if (not filter_action or r.get("action") in filter_action)
                and r.get("vri",0) >= min_vri]
    filtered.sort(key=lambda x: x.get(sort_by,0), reverse=(sort_by!="rank"))

    st.markdown(f"**顯示 {len(filtered)} / {len(top20)} 檔**（★=自選股）")

    html = """<table class="data-table"><thead><tr>
        <th>#</th><th>代號</th><th>Score</th><th>操作</th><th>訊號型態</th>
        <th>PVO</th><th>VRI</th><th>Slope Z</th>
        <th>現價</th><th>停利①</th><th>停損</th>
    </tr></thead><tbody>"""

    for r in filtered:
        rank   = r.get("rank","—")
        sym    = _html_escape.escape(str(r.get("symbol","—")))
        score  = r.get("score",0)
        action = r.get("action","—")
        signal = _html_escape.escape(str(r.get("signal","—")))
        pvo    = r.get("pvo",0)
        vri    = r.get("vri",0)
        slz    = r.get("slope_z",0)
        close  = r.get("close",0)
        tp1    = r.get("tp1_price", None)
        stop   = r.get("stop_price", None)
        is_w   = r.get("is_watchlist",False)

        rank_css  = {1:"rank-1",2:"rank-2",3:"rank-3"}.get(rank,"rank-n")
        pvo_css   = "c-green" if pvo>10 else ("c-cyan" if pvo>0 else "c-red")
        vri_css   = "c-green" if 40<=vri<=75 else ("c-red" if vri>90 else "c-amber")
        slz_css   = "c-green" if slz>1.5 else ("c-cyan" if slz>0 else "c-red")
        score_css = "c-green" if score>=mu+sigma else ("c-amber" if score>=mu else "c-dim")
        if "三合一" in signal: sig_css="pill-p"
        elif "二合一" in signal: sig_css="pill-b"
        elif "單一" in signal: sig_css="pill-a"
        else: sig_css="pill-c"

        tp1_str  = f"{tp1:.1f}" if isinstance(tp1, float) else "—"
        stop_str = f"{stop:.1f}" if isinstance(stop, float) else "—"

        html += f"""<tr>
            <td><span class="rank-badge {rank_css}">{rank}</span></td>
            <td>{_watch_star(is_w)}<b style="color:#1e293b;">{sym}</b></td>
            <td><span class="mono-num {score_css}">{score:.2f}</span></td>
            <td>{_action_pill(action)}</td>
            <td><span class="pill {sig_css}">{signal}</span></td>
            <td><span class="mono-num {pvo_css}">{pvo:+.2f}</span></td>
            <td><span class="mono-num {vri_css}">{vri:.1f}</span></td>
            <td><span class="mono-num {slz_css}">{slz:+.2f}</span></td>
            <td class="mono-num c-mid">{close:.1f}</td>
            <td class="mono-num c-green">{tp1_str}</td>
            <td class="mono-num c-red">{stop_str}</td>
        </tr>"""
    html += "</tbody></table>"
    _render_html(html, height=min(70+len(filtered)*46,650))


# ══════════════════════════════════════════════════════════════
# Section: V12.1（V4.1-05）
# ══════════════════════════════════════════════════════════════
def render_v12_section(v12: dict):
    gen_at     = v12.get("generated_at","—")
    mode       = v12.get("run_mode","—")
    positions  = v12.get("positions",[])
    candidates = v12.get("candidates",[])[:5]
    stats      = v12.get("stats",{})

    st.markdown(f"""
    <div class="sec-header sec-v12">
        <span style="font-size:1.1rem;font-weight:900;color:#059669;">🟩 V12.1 交易決策系統</span>
        <span class="pill pill-g">路徑 45 / 423</span>
        <span class="pill pill-{'g' if mode=='postmarket' else 'a'}">{mode}</span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    s = stats
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("總交易筆數", s.get("total_trades","—"))
    c2.metric("OOS勝率", f"{s.get('win_rate',0):.1f}%")
    c3.metric("平均EV", f"{s.get('avg_ev',0):+.2f}%")
    c4.metric("最大回撤", f"{s.get('max_dd',0):.2f}%")
    c5.metric("Sharpe", f"{s.get('sharpe',0):.2f}")
    c6.metric("t值", f"{s.get('t_stat',0):+.3f}")
    c7.metric("年化單利", f"{s.get('simple_cagr',0):+.1f}%")

    if not positions and not candidates:
        st.info("⏳ 等待 V12.1 快照更新")
        return

    n_pos  = len(positions)
    n_cand = len(candidates)
    st.markdown(f"#### 📋 目前部位監控（持倉 {n_pos} 檔）＋ 候選 {n_cand} 檔（★=自選股）")

    html = """<table class="data-table"><thead><tr>
        <th>代號</th><th>路徑</th><th>EV</th><th>等級</th>
        <th>操作</th><th>出場信號</th><th>純淨度</th>
        <th>持倉天</th><th>報酬</th><th>停利①</th><th>停損</th>
    </tr></thead><tbody>"""

    for p in positions:
        sym     = _html_escape.escape(str(p.get("symbol","—")))
        path    = p.get("path","—")
        ev      = p.get("ev",0)
        ev_tier = _html_escape.escape(str(p.get("ev_tier","—")))
        action  = p.get("action","—")
        exs     = p.get("exit_signal","—")
        quality = p.get("quality","Pure")
        days    = p.get("days_held",0)
        ret     = p.get("curr_ret_pct",0)
        tp1     = _html_escape.escape(str(p.get("tp1_price","—")))
        stop    = _html_escape.escape(str(p.get("stop_price","—")))
        is_w    = p.get("is_watchlist",False)
        ev_css  = "c-green" if ev>5 else ("c-cyan" if ev>3 else "c-amber")
        ret_css = "c-green" if ret>0 else "c-red"

        html += f"""<tr>
            <td>{_watch_star(is_w)}<b style="color:#1e293b;">{sym}</b></td>
            <td>{_path_tag(path)}</td>
            <td><span class="mono-num {ev_css}">{ev:+.2f}%</span></td>
            <td style="font-size:0.78rem;">{ev_tier}</td>
            <td>{_action_pill(action)}</td>
            <td>{_exit_pill(exs)}</td>
            <td>{_quality_pill(quality)}</td>
            <td class="c-mid mono-num">{days}</td>
            <td><span class="mono-num {ret_css}">{ret:+.2f}%</span></td>
            <td class="mono-num c-green" style="font-size:0.78rem;">{tp1}</td>
            <td class="mono-num c-red" style="font-size:0.78rem;">{stop}</td>
        </tr>"""

    if candidates:
        html += f"""<tr class="candidate-divider">
            <td colspan="11">▼ 候選觀察股（尚未進場，共 {len(candidates)} 檔）— EV 達標但未觸發進場條件</td>
        </tr>"""
        for p in candidates:
            sym     = _html_escape.escape(str(p.get("symbol","—")))
            path    = p.get("path","—")
            ev      = p.get("ev",0)
            ev_tier = _html_escape.escape(str(p.get("ev_tier","—")))
            quality = p.get("quality","Pure")
            tp1     = _html_escape.escape(str(p.get("tp1_price","—")))
            stop    = _html_escape.escape(str(p.get("stop_price","—")))
            ev_css  = "c-green" if ev>5 else ("c-cyan" if ev>3 else "c-amber")

            html += f"""<tr class="candidate-row">
                <td><b style="color:#475569;">{sym}</b></td>
                <td>{_path_tag(path)}</td>
                <td><span class="mono-num {ev_css}">{ev:+.2f}%</span></td>
                <td style="font-size:0.78rem;">{ev_tier}</td>
                <td>{_action_pill("候選")}</td>
                <td><span class="pill pill-a">⏳ 待確認</span></td>
                <td>{_quality_pill(quality)}</td>
                <td class="c-dim mono-num">0</td>
                <td class="c-dim mono-num">—</td>
                <td class="mono-num" style="color:#d97706;font-size:0.78rem;">{tp1}</td>
                <td class="mono-num" style="color:#dc2626;font-size:0.78rem;">{stop}</td>
            </tr>"""

    html += "</tbody></table>"
    _render_html(html, height=min(70+(n_pos+n_cand+1)*46,700))


# ══════════════════════════════════════════════════════════════
# Section: Regime（V4.1-06）
# ══════════════════════════════════════════════════════════════
def render_regime_section(regime: dict, market: dict):
    gen_at  = regime.get("generated_at","—")
    bear    = regime.get("bear",0.0)
    range_  = regime.get("range",0.30)
    bull    = regime.get("bull",0.70)
    label   = regime.get("label","牛市")
    strat   = regime.get("active_strategy","bull")
    a_path  = regime.get("active_path","45")
    b_path  = regime.get("backup_path","423")
    s5d     = regime.get("slope_5d",0)
    s20d    = regime.get("slope_20d",0)
    adx     = regime.get("adx",0)
    mkt_rsi = regime.get("mkt_rsi",0)
    src     = regime.get("data_source","FinMind_Y9999_TAIEX")
    idx_close = regime.get("index_close", (market or {}).get("index_close","—"))
    idx_chg   = regime.get("index_chg_pct", (market or {}).get("index_chg_pct", None))

    st.markdown(f"""
    <div class="sec-header sec-regime">
        <span style="font-size:1.1rem;font-weight:900;color:#d97706;">🟨 Regime 市場制度 &amp; 策略切換</span>
        <span class="pill pill-a">{src}</span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    render_regime_bar(bear, range_, bull)

    chg_color = "#059669" if (idx_chg is not None and idx_chg >= 0) else "#dc2626"
    idx_str   = f"{idx_close:,.2f}" if isinstance(idx_close, float) else str(idx_close)
    chg_str   = f"{idx_chg:+.2f}%" if idx_chg is not None else "—"

    st.markdown(f"""
    <div class="mkt-grid">
        <div class="mkt-cell"><div class="mkt-val" style="color:#1e293b;">{label}</div><div class="mkt-lbl">大盤情緒</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#d97706;">{strat.upper()}</div><div class="mkt-lbl">當前策略</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#059669;">{a_path}</div><div class="mkt-lbl">主路徑</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#2563eb;">{b_path}</div><div class="mkt-lbl">備援路徑</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{chg_color};">{idx_str}</div><div class="mkt-lbl">TAIEX 加權指數</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{chg_color};">{chg_str}</div><div class="mkt-lbl">日漲跌幅</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#0891b2;">{mkt_rsi:.1f}</div><div class="mkt-lbl">大盤 RSI(14)</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#7c3aed;">{adx:.1f}</div><div class="mkt-lbl">ADX 趨勢強度</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{'#059669' if s5d>=0 else '#dc2626'};">{s5d:+.4f}</div><div class="mkt-lbl">5日線性斜率</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{'#059669' if s20d>=0 else '#dc2626'};">{s20d:+.4f}</div><div class="mkt-lbl">20日線性斜率</div></div>
    </div>
    """, unsafe_allow_html=True)

    history = regime.get("history",[])
    if history:
        df_r = pd.DataFrame(history)
        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.55, 0.45], shared_xaxes=True,
            subplot_titles=["月末 Regime 三態機率分佈（熊/震盪/牛）","TAIEX 加權指數月收盤走勢"],
            vertical_spacing=0.12
        )
        fig.add_bar(x=df_r["month"], y=df_r["bull"]*100, name="牛市機率",
                    marker_color="rgba(5,150,105,0.6)", row=1, col=1)
        fig.add_bar(x=df_r["month"], y=df_r["range"]*100, name="震盪機率",
                    marker_color="rgba(217,119,6,0.6)", row=1, col=1)
        fig.add_bar(x=df_r["month"], y=df_r["bear"]*100, name="熊市機率",
                    marker_color="rgba(220,38,38,0.5)", row=1, col=1)
        if "index_close" in df_r.columns:
            colors_line = ["#dc2626" if i > 0 and df_r["index_close"].iloc[i] < df_r["index_close"].iloc[i-1]
                           else "#059669" for i in range(len(df_r))]
            fig.add_scatter(x=df_r["month"], y=df_r["index_close"],
                mode="lines+markers+text", name="TAIEX",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=7, color=colors_line),
                text=[f"{v:,.0f}" for v in df_r["index_close"]],
                textposition="top center", textfont=dict(size=10, color="#475569"),
                row=2, col=1)
        fig.update_layout(
            barmode="stack", height=340, margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(color="#475569", size=11),
            legend=dict(font=dict(color="#475569"), orientation="h", y=1.08, x=0,
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(59,130,246,0.2)", borderwidth=1),
            yaxis=dict(range=[0,100], ticksuffix="%", title="機率"),
            yaxis2=dict(title="指數點位"),
        )
        for ann in fig.layout.annotations:
            ann.font.color = "#475569"; ann.font.size = 11
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 上圖：月底 Regime 機率；下圖：TAIEX 月收盤走勢")


# ══════════════════════════════════════════════════════════════
# Section: 回測績效（v5.0）
# ══════════════════════════════════════════════════════════════
def render_backtest(backtest: dict, trades_df):
    st.markdown("""
    <div class="sec-header sec-bt">
      <span style="font-size:1.05rem;font-weight:900;color:#7c3aed;">📈 回測績效</span>
    </div>""", unsafe_allow_html=True)

    if not backtest:
        st.info("⏳ 尚無回測結果（需執行 backtest_engine.py）")
        return

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("總交易筆", backtest.get("total_trades","—"))
    c2.metric("勝率",  f"{backtest.get('win_rate_pct',0):.1f}%")
    c3.metric("總報酬", f"{backtest.get('total_ret_pct',0):+.2f}%")
    c4.metric("年化",  f"{backtest.get('cagr_pct',0):+.2f}%")
    c5.metric("Sharpe", f"{backtest.get('sharpe',0):.2f}")
    c6.metric("最大回撤", f"{backtest.get('max_drawdown_pct',0):.2f}%")

    eq = backtest.get("equity_curve", [])
    if eq:
        df_eq = pd.DataFrame(eq)
        initial = backtest.get("initial_capital", 1_000_000)
        df_eq["ret_pct"] = (df_eq["total_val"] / initial - 1) * 100

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True, subplot_titles=["組合淨值曲線 (報酬%)", "每日部位數"])
        fig.add_scatter(x=df_eq["date"], y=df_eq["ret_pct"],
                        name="報酬%", line=dict(color="#2563eb", width=2),
                        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)", row=1, col=1)
        fig.add_bar(x=df_eq["date"], y=df_eq["n_pos"],
                    name="持倉數", marker_color="rgba(5,150,105,0.5)", row=2, col=1)
        fig.update_layout(
            height=380, margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(color="#475569", size=11),
            legend=dict(orientation="h", y=1.05), yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True)

    breakdown = backtest.get("exit_breakdown", {})
    if breakdown:
        labels = list(breakdown.keys())
        values = list(breakdown.values())
        color_map = {"SELL_STOP":"#dc2626","SELL_TP1":"#059669","SELL_TP2":"#047857",
                     "SELL_TRAIL":"#d97706","SELL_EV":"#7c3aed","SELL_REPLACE":"#0891b2"}
        colors = [color_map.get(l,"#94a3b8") for l in labels]
        fig2 = go.Figure(go.Pie(labels=labels, values=values, marker_colors=colors,
                                textinfo="label+percent", hole=0.4))
        fig2.update_layout(title="出場方式分佈", height=280, margin=dict(l=0,r=0,t=40,b=0),
                           paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#475569"))
        st.plotly_chart(fig2, use_container_width=True)

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


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    stock_set = load_stock_set()
    watchlist = stock_set.get("watchlist", {}).get("symbols", [])
    render_sidebar(stock_set)

    use_mock = st.session_state.use_mock

    if use_mock:
        v4, v12, regime, market = _mock_v4(), _mock_v12(), _mock_regime(), _mock_market()
        status = {k: False for k in ["v4","v12","regime","market"]}
        portfolio, backtest, trades_df = {}, None, None
    else:
        with st.spinner("🔄 從 GitHub 讀取最新快照..."):
            v4, v12, regime, market, status = load_all_snapshots()
            portfolio, backtest, trades_df = load_all_v5()
        v4     = v4     or _mock_v4()
        v12    = v12    or _mock_v12()
        regime = regime or _mock_regime()
        market = market or _mock_market()

    portfolio = portfolio or {}
    st.session_state["last_refresh"] = datetime.now().strftime("%H:%M:%S")

    all_live   = all(status.values())
    mode_label = "LIVE 資料" if all_live else ("DEMO 模式" if use_mock else "部分 DEMO")
    if all_live:
        mode_css = "background:rgba(5,150,105,0.1);border:1px solid rgba(5,150,105,0.3);color:#059669;"
    else:
        mode_css = "background:rgba(217,119,6,0.1);border:1px solid rgba(217,119,6,0.3);color:#d97706;"

    run_mode  = v4.get("run_mode","—")
    gen_at    = v4.get("generated_at","—")
    bear      = regime.get("bear",0.0)
    bull      = regime.get("bull",0.70)
    label     = regime.get("label","牛市")
    a_path    = regime.get("active_path","45")
    src       = regime.get("data_source","FinMind_Y9999_TAIEX")
    idx_close = regime.get("index_close", market.get("index_close","—"))
    idx_chg   = regime.get("index_chg_pct", market.get("index_chg_pct", None))
    idx_str   = f"{idx_close:,.2f}" if isinstance(idx_close, float) else str(idx_close)
    chg_str   = f"{idx_chg:+.2f}%" if isinstance(idx_chg, float) else str(idx_chg)
    chg_css   = "chip-ok" if isinstance(idx_chg, float) and idx_chg >= 0 else "chip-err"
    dom_css   = "chip-ok" if bull > bear else ("chip-err" if bear > bull else "chip-warn")

    # Header
    st.markdown(f"""
    <div class="hq-header">
        <div>
            <div class="hq-title">📊 資源法 AI 戰情室 <span style="font-size:0.85rem;color:#94a3b8;">v5.1</span></div>
            <div class="hq-sub">FinMind Y9999 TAIEX · GitHub Storage · Precompute + Display</div>
        </div>
        <div class="hq-badge" style="{mode_css}">{mode_label}</div>
    </div>
    """, unsafe_allow_html=True)

    if not use_mock and not all_live:
        missing = [k.upper() for k,v in status.items() if not v]
        st.warning(f"⚠️ GitHub 資料讀取失敗：{' / '.join(missing)}。顯示模擬資料。")

    # 狀態列
    st.markdown(f"""
    <div class="status-row">
        <div class="status-chip chip-ok"><span class="dot dot-g"></span> 系統正常</div>
        <div class="status-chip chip-info">📡 資料: {gen_at}</div>
        <div class="status-chip chip-info">🕐 刷新: {st.session_state.last_refresh}</div>
        <div class="status-chip chip-info">📋 模式: {run_mode}</div>
        <div class="status-chip {chg_css}">📈 TAIEX: {idx_str} ({chg_str})</div>
        <div class="status-chip {dom_css}">🌡️ {label} | 熊{bear*100:.0f}% 牛{bull*100:.0f}%</div>
        <div class="status-chip chip-info">🛤️ 主路徑: {a_path}</div>
        <div class="status-chip chip-info">🔗 {src}</div>
    </div>
    """, unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 刷新資料"):
            st.cache_data.clear()
            st.rerun()

    # 自選股快覽
    render_watchlist_section(v4, v12, watchlist)

    # 個股 AI 分析
    render_single_stock_panel(v4, v12, regime)

    # Gemini 全盤分析
    st.markdown("""
    <div class="sec-header sec-ai">
        <span style="font-size:1.0rem;font-weight:800;color:#7c3aed;">🤖 Gemini 全盤 AI 分析</span>
    </div>
    """, unsafe_allow_html=True)

    ai1, ai2 = st.columns([1,5])
    with ai1:
        ai_btn = st.button("🤖 執行分析", use_container_width=True, key="ai_btn")
    with ai2:
        st.caption("整合 TAIEX 大盤環境 + V4 TOP5 + V12.1 部位，一鍵生成 Gemini 操作建議（需 API Key）")

    if st.session_state.ai_summary:
        safe2 = _html_escape.escape(st.session_state.ai_summary).replace('\n','<br>')
        st.markdown(f'<div class="ai-box">{safe2}</div>', unsafe_allow_html=True)

    if ai_btn:
        with st.spinner("🤖 Gemini 分析中..."):
            summary = call_gemini(build_dashboard_prompt(v4, v12, regime, market), st.session_state.gemini_key)
        st.session_state.ai_summary = summary
        st.rerun()

    st.markdown("---")

    # Tabs：整合 v4.1 三個 Tab + v5.0 三個 Tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🟦 V4 市場強度",
        "🟩 V12.1 交易決策",
        "🟨 Regime 大盤",
        "▲ 今日買進原因",
        "▼ 賣出訊號",
        "📈 回測績效",
    ])

    with tab1:
        render_v4_section(v4)
    with tab2:
        render_v12_section(v12)
    with tab3:
        render_regime_section(regime, market)
    with tab4:
        render_buy_reasons(portfolio)
    with tab5:
        render_sell_signals(portfolio)
    with tab6:
        render_backtest(backtest, trades_df)


if __name__ == "__main__":
    main()
