"""
╔══════════════════════════════════════════════════════════════╗
║   資源法 AI 戰情室  v4.0  —  純展示層 (Display-Only)         ║
║   架構：Precompute + GitHub Storage + Streamlit Render       ║
║   資料來源：GitHub Raw JSON (每日自動更新)                    ║
║                                                              ║
║   v4.0 修正：                                                ║
║   [FIX-01] 大盤顯示改為 TAIEX 原始指數（非 ETF 合成）         ║
║   [FIX-02] 新增自選股 watchlist 區塊                         ║
║   [FIX-03] 盤中/盤後模式標示                                 ║
║   [FIX-04] st.html() 取代棄用的 components.v1.html()         ║
╚══════════════════════════════════════════════════════════════╝
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
# GitHub 資料源
# ──────────────────────────────────────────────────────────────
GITHUB_OWNER = _get_secret("GITHUB_OWNER", "uta123kimo-cloud")
GITHUB_REPO  = _get_secret("GITHUB_REPO",  "stock2026")
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/storage"
REPO_RAW = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main"

# ──────────────────────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="資源法 AI 戰情室 v4.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Noto+Sans+TC:wght@300;400;600;900&display=swap');

:root {
    --bg:#0b0f1a; --bg2:#111827; --bg3:#1a2235; --panel:#161d2e;
    --border:rgba(99,179,237,0.15); --border2:rgba(99,179,237,0.30);
    --accent:#3b82f6; --accent2:#06b6d4; --green:#10b981; --red:#ef4444;
    --amber:#f59e0b; --purple:#8b5cf6; --text:#e2e8f0; --text-dim:#64748b;
    --mono:'IBM Plex Mono',monospace; --sans:'Noto Sans TC',sans-serif;
    --glow-b:0 0 20px rgba(59,130,246,0.25); --radius:10px; --radius-lg:16px;
}
html,body,.stApp { background:var(--bg)!important; color:var(--text)!important; font-family:var(--sans); }
[data-testid="stSidebar"] { background:var(--bg2)!important; border-right:1px solid var(--border); }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding:1rem 2rem 3rem!important; max-width:1600px; }

.hq-header {
    display:flex; align-items:center; gap:16px; padding:20px 28px; margin-bottom:8px;
    background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(6,182,212,0.05));
    border:1px solid var(--border2); border-radius:var(--radius-lg); position:relative; overflow:hidden;
}
.hq-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2),var(--green));
}
.hq-title { font-size:1.6rem; font-weight:900; color:#fff; letter-spacing:-0.5px; }
.hq-sub   { font-size:0.78rem; color:var(--text-dim); font-family:var(--mono); margin-top:3px; }
.hq-badge {
    margin-left:auto; padding:6px 14px; border-radius:20px; font-size:0.72rem;
    font-weight:700; font-family:var(--mono); letter-spacing:1px;
}

.status-row { display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap; }
.status-chip {
    display:flex; align-items:center; gap:6px; padding:5px 12px;
    border-radius:20px; font-size:0.73rem; font-family:var(--mono); border:1px solid;
}
.chip-ok   { background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.3); color:var(--green); }
.chip-err  { background:rgba(239,68,68,0.08);  border-color:rgba(239,68,68,0.3);  color:var(--red); }
.chip-info { background:rgba(59,130,246,0.08); border-color:rgba(59,130,246,0.3); color:var(--accent); }
.chip-warn { background:rgba(245,158,11,0.08); border-color:rgba(245,158,11,0.3); color:var(--amber); }
.dot { width:7px; height:7px; border-radius:50%; animation:pulse 2s infinite; }
.dot-g { background:var(--green); box-shadow:0 0 6px var(--green); }
.dot-r { background:var(--red); }
.dot-b { background:var(--accent); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.sec-header {
    display:flex; align-items:center; gap:12px; padding:14px 20px;
    margin:20px 0 14px 0; border-radius:var(--radius); border-left:3px solid; background:var(--panel);
}
.sec-v4     { border-color:var(--accent); }
.sec-v12    { border-color:var(--green); }
.sec-regime { border-color:var(--amber); }
.sec-ai     { border-color:var(--purple); }
.sec-watch  { border-color:var(--accent2); }
.sec-label  { font-size:0.68rem; font-family:var(--mono); opacity:0.6; margin-left:auto; }

.card { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px 18px; margin-bottom:10px; }
.card:hover { border-color:var(--border2); box-shadow:var(--glow-b); }

.mono-num { font-family:var(--mono); font-weight:700; }
.c-green  { color:var(--green)!important; }
.c-red    { color:var(--red)!important; }
.c-amber  { color:var(--amber)!important; }
.c-blue   { color:var(--accent)!important; }
.c-cyan   { color:var(--accent2)!important; }
.c-purple { color:var(--purple)!important; }
.c-dim    { color:var(--text-dim)!important; }

.pill {
    display:inline-block; padding:2px 9px; border-radius:12px;
    font-size:0.7rem; font-weight:700; font-family:var(--mono); border:1px solid; margin:2px;
}
.pill-g { background:rgba(16,185,129,0.1);  border-color:rgba(16,185,129,0.35); color:var(--green); }
.pill-r { background:rgba(239,68,68,0.1);   border-color:rgba(239,68,68,0.35);  color:var(--red); }
.pill-b { background:rgba(59,130,246,0.1);  border-color:rgba(59,130,246,0.35); color:var(--accent); }
.pill-a { background:rgba(245,158,11,0.1);  border-color:rgba(245,158,11,0.35); color:var(--amber); }
.pill-p { background:rgba(139,92,246,0.1);  border-color:rgba(139,92,246,0.35); color:var(--purple); }
.pill-c { background:rgba(6,182,212,0.1);   border-color:rgba(6,182,212,0.35);  color:var(--accent2); }

.regime-bar { display:flex; align-items:stretch; border-radius:var(--radius); overflow:hidden; height:38px; margin:10px 0; border:1px solid var(--border); }
.regime-seg { display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; font-family:var(--mono); transition:all 0.3s; }
.seg-bear  { background:rgba(239,68,68,0.18); color:var(--red); }
.seg-range { background:rgba(245,158,11,0.18); color:var(--amber); }
.seg-bull  { background:rgba(16,185,129,0.18); color:var(--green); }

.data-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.data-table th { background:var(--bg3); color:var(--text-dim); font-weight:600; font-family:var(--mono); font-size:0.7rem; letter-spacing:0.5px; padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); }
.data-table td { padding:9px 12px; border-bottom:1px solid rgba(99,179,237,0.06); }
.data-table tr:hover td { background:rgba(59,130,246,0.04); }

.rank-badge { display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:50%; font-size:0.72rem; font-weight:900; font-family:var(--mono); }
.rank-1 { background:linear-gradient(135deg,#f59e0b,#d97706); color:#000; }
.rank-2 { background:linear-gradient(135deg,#94a3b8,#64748b); color:#fff; }
.rank-3 { background:linear-gradient(135deg,#cd7c3a,#b45309); color:#fff; }
.rank-n { background:var(--bg3); color:var(--text-dim); border:1px solid var(--border); }

.ai-box { background:linear-gradient(135deg,rgba(139,92,246,0.05),rgba(59,130,246,0.05)); border:1px solid rgba(139,92,246,0.2); border-left:3px solid var(--purple); border-radius:var(--radius); padding:18px 22px; font-size:1.013rem; line-height:1.95; color:var(--text); }
.mkt-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(100px,1fr)); gap:10px; margin:10px 0; }
.mkt-cell { background:var(--bg3); border:1px solid var(--border); border-radius:var(--radius); padding:12px 10px; text-align:center; }
.mkt-val  { font-size:1.2rem; font-weight:900; font-family:var(--mono); }
.mkt-lbl  { font-size:0.65rem; color:var(--text-dim); margin-top:3px; letter-spacing:0.5px; }
.path-tag { display:inline-block; padding:3px 10px; border-radius:4px; font-family:var(--mono); font-size:0.72rem; font-weight:700; }
.path-45  { background:rgba(16,185,129,0.15); color:var(--green); border:1px solid rgba(16,185,129,0.3); }
.path-423 { background:rgba(59,130,246,0.15); color:var(--accent); border:1px solid rgba(59,130,246,0.3); }
.path-na  { background:rgba(100,116,139,0.15); color:var(--text-dim); border:1px solid var(--border); }

/* 自選股醒目標記 */
.watchlist-star { color:#f59e0b; font-size:0.85rem; }

[data-testid="stTab"] button { color:#94a3b8!important; font-family:var(--sans)!important; }
[data-testid="stTab"] button[aria-selected="true"] { color:#ffffff!important; font-weight:700!important; border-bottom:2px solid var(--accent)!important; }
.stButton>button { background:linear-gradient(135deg,var(--accent),#2563eb)!important; color:#fff!important; border:none!important; border-radius:var(--radius)!important; font-weight:700!important; }
[data-testid="stMetricValue"] { color:var(--accent2)!important; font-family:var(--mono)!important; font-size:1.4rem!important; }
[data-testid="stMetricLabel"] { color:var(--text-dim)!important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--bg3); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 資料載入
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_json_url(path_suffix: str) -> Optional[dict]:
    url = f"{BASE_URL}/{path_suffix}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_stock_set() -> dict:
    """從 GitHub 讀取 stock_set.json"""
    url = f"{REPO_RAW}/stock_set.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception:
        return {}


def load_all_snapshots():
    v4     = load_json_url("v4/v4_latest.json")
    v12    = load_json_url("v12/v12_latest.json")
    regime = load_json_url("regime/regime_state.json")
    market = load_json_url("market/market_snapshot.json")
    hist   = load_json_url("logs/trade_history.json")
    status = {
        "v4": v4 is not None, "v12": v12 is not None,
        "regime": regime is not None, "market": market is not None,
        "hist": hist is not None,
    }
    return v4, v12, regime, market, hist, status


# ──────────────────────────────────────────────────────────────
# Mock 資料
# ──────────────────────────────────────────────────────────────
def _mock_v4():
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
            "action": random.choice(["強力買進","買進","觀察"]),
            "signal": random.choice(["三合一(ABC)","二合一(AB)","單一(A)"]),
            "close": round(random.uniform(100, 900), 1),
            "regime": "range",
            "is_watchlist": s in ["2330","2317","2454","2308","2382"],
        })
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market": "TW", "top20": rows, "top30": rows,
        "pool_mu": 62.3, "pool_sigma": 11.5, "win_rate": 57.1,
        "run_mode": "postmarket",
    }

def _mock_v12():
    import random
    syms = ["2330","2454","3711","6669","3008","2379","3443","6415","3035","2408"]
    rows = []
    for s in syms:
        ev = round(random.uniform(2.0, 9.5), 2)
        rows.append({
            "symbol": s, "path": random.choice(["45","423"]),
            "ev": ev, "action": random.choice(["持有","進場","觀察"]),
            "exit_signal": random.choice(["無","—","EV衰退"]),
            "quality": "Pure",
            "ev_tier": "⭐核心" if ev>5 else "🔥主力",
            "regime": "range",
            "days_held": random.randint(0, 18),
            "curr_ret_pct": round(random.uniform(-5, 15), 2),
            "tp1_price": round(random.uniform(150, 900), 1),
            "stop_price": round(random.uniform(100, 800), 1),
            "is_watchlist": s in ["2330","2454"],
        })
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "positions": rows, "run_mode": "postmarket",
        "stats": {"total_trades":112,"win_rate":57.1,"avg_ev":5.29,"max_dd":-6.58,"sharpe":5.36,"t_stat":4.032,"simple_cagr":96.9,"pl_ratio":2.31},
    }

def _mock_regime():
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bear":0.22,"range":0.41,"bull":0.37,"label":"偏多震盪",
        "active_strategy":"range","active_path":"423","backup_path":"45",
        "slope_5d":0.0312,"slope_20d":0.0105,"mkt_rsi":54.3,"adx":22.1,
        "index_close": 21450.5,
        "data_source": "FinMind_Y9999_TAIEX",
        "history":[
            {"month":"2026-01","bear":0.35,"range":0.45,"bull":0.20,"label":"偏空"},
            {"month":"2026-02","bear":0.28,"range":0.48,"bull":0.24,"label":"震盪"},
            {"month":"2026-03","bear":0.20,"range":0.43,"bull":0.37,"label":"偏多震盪"},
            {"month":"2026-04","bear":0.22,"range":0.41,"bull":0.37,"label":"偏多震盪"},
        ]
    }

def _mock_market():
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "index_close":21450.5,"index_chg_pct":0.62,
        "mkt_rsi":54.3,"mkt_slope_5d":0.031,"mkt_slope_20d":0.011,
        "data_source":"FinMind_Y9999_TAIEX","run_mode":"postmarket",
    }

def _mock_history():
    import random
    logs = []
    for i in range(30):
        ret = round(random.uniform(-8, 15), 2)
        logs.append({
            "date":f"2026-03-{(i%28)+1:02d}",
            "sym": random.choice(["2330","2454","3711"]),
            "action_type":"賣出","exit_type":random.choice(["停利①","EV衰退","硬停損"]),
            "ret":round(ret/100,4),"path":random.choice(["45","423"]),"year":2026,
        })
    return logs


# ══════════════════════════════════════════════════════════════
# UI 工具
# ══════════════════════════════════════════════════════════════

def _render_html(html_body: str, height: int = 400):
    """使用 st.html() 渲染 HTML 表格（Streamlit 1.36+ API）"""
    full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;600;900&display=swap');
body{{background:transparent;color:#e2e8f0;font-family:'Noto Sans TC',sans-serif;margin:0;padding:0;height:{height}px;overflow-y:auto;overflow-x:hidden;}}
.data-table{{width:100%;border-collapse:collapse;font-size:0.82rem;}}
.data-table th{{background:#1a2235;color:#64748b;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:0.5px;padding:8px 12px;text-align:left;border-bottom:1px solid rgba(99,179,237,0.15);}}
.data-table td{{padding:9px 12px;border-bottom:1px solid rgba(99,179,237,0.06);vertical-align:middle;}}
.data-table tr:hover td{{background:rgba(59,130,246,0.04);}}
.mono-num{{font-family:'IBM Plex Mono',monospace;font-weight:700;}}
.c-green{{color:#10b981!important;}}.c-red{{color:#ef4444!important;}}.c-amber{{color:#f59e0b!important;}}
.c-blue{{color:#3b82f6!important;}}.c-cyan{{color:#06b6d4!important;}}.c-purple{{color:#8b5cf6!important;}}.c-dim{{color:#64748b!important;}}
.pill{{display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.7rem;font-weight:700;font-family:'IBM Plex Mono',monospace;border:1px solid;margin:2px;white-space:nowrap;}}
.pill-g{{background:rgba(16,185,129,0.1);border-color:rgba(16,185,129,0.35);color:#10b981;}}
.pill-r{{background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.35);color:#ef4444;}}
.pill-b{{background:rgba(59,130,246,0.1);border-color:rgba(59,130,246,0.35);color:#3b82f6;}}
.pill-a{{background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.35);color:#f59e0b;}}
.pill-p{{background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.35);color:#8b5cf6;}}
.pill-c{{background:rgba(6,182,212,0.1);border-color:rgba(6,182,212,0.35);color:#06b6d4;}}
.rank-badge{{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font-size:0.72rem;font-weight:900;font-family:'IBM Plex Mono',monospace;}}
.rank-1{{background:linear-gradient(135deg,#f59e0b,#d97706);color:#000;}}
.rank-2{{background:linear-gradient(135deg,#94a3b8,#64748b);color:#fff;}}
.rank-3{{background:linear-gradient(135deg,#cd7c3a,#b45309);color:#fff;}}
.rank-n{{background:#1a2235;color:#64748b;border:1px solid rgba(99,179,237,0.15);}}
.path-tag{{display:inline-block;padding:3px 10px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;font-weight:700;}}
.path-45{{background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);}}
.path-423{{background:rgba(59,130,246,0.15);color:#3b82f6;border:1px solid rgba(59,130,246,0.3);}}
.path-na{{background:rgba(100,116,139,0.15);color:#64748b;border:1px solid rgba(99,179,237,0.15);}}
.watch-star{{color:#f59e0b;font-size:0.85rem;}}
</style></head><body>{html_body}</body></html>"""
    st.html(full_html)


def _action_pill(action: str) -> str:
    mapping = {
        "強力買進":("pill-g","▲ 強力買進"), "買進":("pill-g","▲ 買進"),
        "持有":("pill-b","◆ 持有"), "進場":("pill-g","▲ 進場"),
        "觀察":("pill-a","◇ 觀察"), "賣出":("pill-r","▼ 賣出"),
        "出場":("pill-r","▼ 出場"),
    }
    css, label = mapping.get(action, ("pill-c", action))
    return f'<span class="pill {css}">{label}</span>'

def _path_tag(path: str) -> str:
    css = {"45":"path-45","423":"path-423"}.get(str(path),"path-na")
    return f'<span class="path-tag {css}">{path}</span>'

def _rank_badge(rank: int) -> str:
    css = {1:"rank-1",2:"rank-2",3:"rank-3"}.get(rank,"rank-n")
    return f'<span class="rank-badge {css}">{rank}</span>'

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
    return '<span class="watch-star">★</span> ' if is_watch else ""

def render_regime_bar(bear: float, range_: float, bull: float):
    b, r, u = bear*100, range_*100, bull*100
    st.markdown(f"""
    <div style="display:flex;align-items:stretch;border-radius:10px;overflow:hidden;height:38px;margin:10px 0;border:1px solid rgba(99,179,237,0.15);">
        <div style="width:{b:.0f}%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;background:rgba(239,68,68,0.18);color:#ef4444;">熊 {b:.0f}%</div>
        <div style="width:{r:.0f}%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;background:rgba(245,158,11,0.18);color:#f59e0b;">震 {r:.0f}%</div>
        <div style="width:{u:.0f}%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;font-family:'IBM Plex Mono',monospace;background:rgba(16,185,129,0.18);color:#10b981;">牛 {u:.0f}%</div>
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
# 側欄
# ══════════════════════════════════════════════════════════════

def render_sidebar(stock_set: dict):
    with st.sidebar:
        st.markdown("## ⚡ 資源法 v4.0")
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
        st.caption(f"[編輯 stock_set.json 更新]")

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
            "09:35 開盤快照 [盤中]",
            "10:30 盤中更新 [盤中]",
            "11:30 盤中更新 [盤中]",
            "12:30 盤中更新 [盤中]",
            "13:25 盤中收盤前 [盤中]",
            "15:30 盤後全量掃描 [盤後]",
            "20:00 日結存檔 [盤後]",
        ]:
            st.caption(f"• {t}")
        st.caption("---")
        st.caption("© 2026 資源法 AI 戰情室 v4.0")


# ══════════════════════════════════════════════════════════════
# [FIX-02] 自選股快覽區塊
# ══════════════════════════════════════════════════════════════

def render_watchlist_section(v4: dict, v12: dict, watchlist: list):
    if not watchlist:
        return

    st.markdown(f"""
    <div class="sec-header sec-watch">
        <span style="font-size:1.0rem;font-weight:900;color:#06b6d4;">
            ⭐ 自選股快覽
        </span>
        <span style="color:#64748b;font-size:0.8rem;margin-left:8px;">固定監控 {len(watchlist)} 檔</span>
    </div>
    """, unsafe_allow_html=True)

    top20 = (v4 or {}).get("top20", [])
    positions = (v12 or {}).get("positions", [])
    v4_map  = {r["symbol"]: r for r in top20}
    v12_map = {p["symbol"]: p for p in positions}

    html = """<table class="data-table"><thead><tr>
        <th>代號</th><th>V4 Score</th><th>操作</th><th>訊號</th>
        <th>V12 路徑</th><th>EV</th><th>出場信號</th><th>現價</th><th>報酬%</th>
    </tr></thead><tbody>"""

    for sym in watchlist:
        r4  = v4_map.get(sym, {})
        r12 = v12_map.get(sym, {})
        sym_e  = _html_escape.escape(sym)
        score  = r4.get("score", "—")
        action = r4.get("action", "—")
        signal = _html_escape.escape(str(r4.get("signal", "—")))
        path   = r12.get("path", "—")
        ev     = r12.get("ev", None)
        exs    = r12.get("exit_signal", "—")
        close  = r4.get("close", r12.get("close", "—"))
        ret    = r12.get("curr_ret_pct", None)

        score_str = f"{score:.2f}" if isinstance(score, float) else "—"
        score_css = "c-green" if isinstance(score, float) and score >= 70 else "c-amber" if isinstance(score, float) and score >= 55 else "c-dim"
        ev_str = f"{ev:+.2f}%" if ev is not None else "—"
        ev_css = "c-green" if ev is not None and ev > 5 else "c-cyan" if ev is not None and ev > 3 else "c-dim"
        ret_str = f"{ret:+.2f}%" if ret is not None else "—"
        ret_css = "c-green" if ret is not None and ret > 0 else "c-red" if ret is not None and ret < 0 else "c-dim"
        close_str = f"{close:.1f}" if isinstance(close, float) else str(close)

        if "三合一" in signal: sig_css = "pill-p"
        elif "二合一" in signal: sig_css = "pill-b"
        elif "單一" in signal: sig_css = "pill-a"
        else: sig_css = "pill-c"

        html += f"""<tr>
            <td><span class="watch-star">★</span> <b style="color:#e2e8f0;">{sym_e}</b></td>
            <td><span class="mono-num {score_css}">{score_str}</span></td>
            <td>{_action_pill(action)}</td>
            <td><span class="pill {sig_css}" style="font-size:0.65rem;">{signal}</span></td>
            <td>{_path_tag(path) if path != "—" else '<span class="c-dim">—</span>'}</td>
            <td><span class="mono-num {ev_css}">{ev_str}</span></td>
            <td>{_exit_pill(exs)}</td>
            <td class="mono-num" style="color:#94a3b8;">{close_str}</td>
            <td><span class="mono-num {ret_css}">{ret_str}</span></td>
        </tr>"""

    html += "</tbody></table>"
    _render_html(html, height=min(80 + len(watchlist) * 48, 500))


# ══════════════════════════════════════════════════════════════
# Section 1: V4
# ══════════════════════════════════════════════════════════════

def render_v4_section(v4: dict):
    gen_at = v4.get("generated_at","—")
    mode   = v4.get("run_mode","—")
    top20  = v4.get("top20",[])
    mu     = v4.get("pool_mu",0)
    sigma  = v4.get("pool_sigma",0)

    st.markdown(f"""
    <div class="sec-header sec-v4">
        <span style="font-size:1.1rem;font-weight:900;color:#3b82f6;">🟦 V4 市場強度快照</span>
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
        <th>PVO</th><th>VRI</th><th>Slope Z</th><th>現價</th>
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

        html += f"""<tr>
            <td><span class="rank-badge {rank_css}">{rank}</span></td>
            <td>{_watch_star(is_w)}<b style="color:#e2e8f0;">{sym}</b></td>
            <td><span class="mono-num {score_css}">{score:.2f}</span></td>
            <td>{_action_pill(action)}</td>
            <td><span class="pill {sig_css}">{signal}</span></td>
            <td><span class="mono-num {pvo_css}">{pvo:+.2f}</span></td>
            <td><span class="mono-num {vri_css}">{vri:.1f}</span></td>
            <td><span class="mono-num {slz_css}">{slz:+.2f}</span></td>
            <td class="mono-num" style="color:#94a3b8;">{close:.1f}</td>
        </tr>"""
    html += "</tbody></table>"
    _render_html(html, height=min(70+len(filtered)*46,650))


# ══════════════════════════════════════════════════════════════
# Section 2: V12.1
# ══════════════════════════════════════════════════════════════

def render_v12_section(v12: dict):
    gen_at    = v12.get("generated_at","—")
    mode      = v12.get("run_mode","—")
    positions = v12.get("positions",[])
    stats     = v12.get("stats",{})

    st.markdown(f"""
    <div class="sec-header sec-v12">
        <span style="font-size:1.1rem;font-weight:900;color:#10b981;">🟩 V12.1 交易決策系統</span>
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

    if not positions:
        st.info("⏳ 等待 V12.1 快照更新")
        return

    st.markdown("#### 📋 目前部位監控（★=自選股）")
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
            <td>{_watch_star(is_w)}<b style="color:#e2e8f0;">{sym}</b></td>
            <td>{_path_tag(path)}</td>
            <td><span class="mono-num {ev_css}">{ev:+.2f}%</span></td>
            <td style="font-size:0.78rem;">{ev_tier}</td>
            <td>{_action_pill(action)}</td>
            <td>{_exit_pill(exs)}</td>
            <td>{_quality_pill(quality)}</td>
            <td class="c-dim mono-num">{days}</td>
            <td><span class="mono-num {ret_css}">{ret:+.2f}%</span></td>
            <td class="c-green mono-num" style="font-size:0.78rem;">{tp1}</td>
            <td class="c-red mono-num" style="font-size:0.78rem;">{stop}</td>
        </tr>"""
    html += "</tbody></table>"
    _render_html(html, height=min(70+len(positions)*46,650))


# ══════════════════════════════════════════════════════════════
# Section 3: Regime（[NEW-01] 顯示 TAIEX 原始指數）
# ══════════════════════════════════════════════════════════════

def render_regime_section(regime: dict, market: dict):
    gen_at = regime.get("generated_at","—")
    bear   = regime.get("bear",0.33)
    range_ = regime.get("range",0.34)
    bull   = regime.get("bull",0.33)
    label  = regime.get("label","震盪")
    strat  = regime.get("active_strategy","range")
    a_path = regime.get("active_path","—")
    b_path = regime.get("backup_path","—")
    s5d    = regime.get("slope_5d",0)
    s20d   = regime.get("slope_20d",0)
    adx    = regime.get("adx",0)
    # [NEW-01] 直接顯示 TAIEX 原始指數
    idx_close = regime.get("index_close", (market or {}).get("index_close","—"))
    idx_chg   = (market or {}).get("index_chg_pct", None)
    mkt_rsi   = regime.get("mkt_rsi",0)
    src       = regime.get("data_source","—")

    st.markdown(f"""
    <div class="sec-header sec-regime">
        <span style="font-size:1.1rem;font-weight:900;color:#f59e0b;">🟨 Regime 市場制度 & 策略切換</span>
        <span class="pill pill-a">{src}</span>
        <span class="sec-label">更新: {gen_at}</span>
    </div>
    """, unsafe_allow_html=True)

    render_regime_bar(bear, range_, bull)

    chg_color = "#10b981" if (idx_chg is not None and idx_chg >= 0) else "#ef4444"
    idx_str   = f"{idx_close:,.2f}" if isinstance(idx_close, float) else str(idx_close)
    chg_str   = f"{idx_chg:+.2f}%" if idx_chg is not None else "—"

    st.markdown(f"""
    <div class="mkt-grid">
        <div class="mkt-cell"><div class="mkt-val" style="color:#e2e8f0;">{label}</div><div class="mkt-lbl">大盤情緒</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#f59e0b;">{strat.upper()}</div><div class="mkt-lbl">當前策略</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#10b981;">{a_path}</div><div class="mkt-lbl">主路徑</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#3b82f6;">{b_path}</div><div class="mkt-lbl">備援路徑</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{chg_color};">{idx_str}</div><div class="mkt-lbl">TAIEX 收盤</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{chg_color};">{chg_str}</div><div class="mkt-lbl">日漲跌</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#06b6d4;">{mkt_rsi:.1f}</div><div class="mkt-lbl">大盤 RSI</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:#8b5cf6;">{adx:.1f}</div><div class="mkt-lbl">ADX</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{'#10b981' if s5d>=0 else '#ef4444'}">{s5d:+.4f}</div><div class="mkt-lbl">5日斜率</div></div>
        <div class="mkt-cell"><div class="mkt-val" style="color:{'#10b981' if s20d>=0 else '#ef4444'}">{s20d:+.4f}</div><div class="mkt-lbl">20日斜率</div></div>
    </div>
    """, unsafe_allow_html=True)

    history = regime.get("history",[])
    if history:
        df_r = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_bar(x=df_r["month"],y=df_r["bull"]*100,name="牛市",marker_color="rgba(16,185,129,0.7)")
        fig.add_bar(x=df_r["month"],y=df_r["range"]*100,name="震盪",marker_color="rgba(245,158,11,0.7)")
        fig.add_bar(x=df_r["month"],y=df_r["bear"]*100,name="熊市",marker_color="rgba(239,68,68,0.7)")
        fig.update_layout(
            barmode="stack",height=200,margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b",size=11),
            legend=dict(font=dict(color="#94a3b8"),orientation="h",y=1.1,x=0,bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(99,179,237,0.08)",color="#64748b"),
            yaxis=dict(gridcolor="rgba(99,179,237,0.08)",color="#64748b",range=[0,100],ticksuffix="%"),
            title=dict(text="月末 Regime 機率歷史",font=dict(color="#64748b",size=11))
        )
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 歷史分析
# ══════════════════════════════════════════════════════════════

def render_history_tab(hist: list):
    if not hist:
        st.info("⏳ 等待交易歷史資料")
        return
    df = pd.DataFrame(hist)
    df["ret_pct"] = df["ret"] * 100
    col1,col2 = st.columns([3,2])
    with col1:
        st.markdown("#### 📈 累積報酬曲線")
        df_sorted = df.sort_values("date")
        df_sorted["cumret"] = df_sorted["ret_pct"].cumsum()
        fig = go.Figure()
        fig.add_scatter(x=df_sorted["date"],y=df_sorted["cumret"],fill="tozeroy",name="累積報酬",
                        line=dict(color="#3b82f6",width=2),fillcolor="rgba(59,130,246,0.10)")
        fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#64748b"),
                          xaxis=dict(gridcolor="rgba(99,179,237,0.08)",color="#64748b"),
                          yaxis=dict(gridcolor="rgba(99,179,237,0.08)",color="#64748b",ticksuffix="%"),
                          showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("#### 📊 出場原因分佈")
        if "exit_type" in df.columns:
            ec = df["exit_type"].value_counts().head(8)
            colors = ["#10b981" if "停利" in e else "#ef4444" if ("停損" in e or "硬" in e) else "#f59e0b" if ("衰退" in e or "枯竭" in e) else "#3b82f6" for e in ec.index]
            fig2 = go.Figure(go.Bar(x=ec.values,y=ec.index,orientation="h",marker_color=colors))
            fig2.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#64748b"),
                               xaxis=dict(gridcolor="rgba(99,179,237,0.08)",color="#64748b"),
                               yaxis=dict(color="#94a3b8"),showlegend=False)
            st.plotly_chart(fig2,use_container_width=True)
    if "path" in df.columns:
        st.markdown("#### 🛤️ 路徑績效")
        ps = df.groupby("path")["ret_pct"].agg(筆數="count",勝率=lambda x:(x>0).mean()*100,均報酬="mean",累計="sum").round(2)
        st.dataframe(ps,use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 個股 AI 分析
# ══════════════════════════════════════════════════════════════

def render_single_stock_panel(v4: dict, v12: dict, regime: dict):
    st.markdown("""
    <div class="sec-header sec-ai">
        <span style="font-size:1.0rem;font-weight:800;color:#8b5cf6;">🔍 個股 AI 深度分析</span>
        <span style="color:#64748b;font-size:0.8rem;margin-left:10px;">輸入代號 → Gemini 個股報告</span>
    </div>
    """, unsafe_allow_html=True)

    col_in,col_btn = st.columns([3,1])
    with col_in:
        sym_input = st.text_input("個股代號",placeholder="例：2330",label_visibility="collapsed",key="single_sym_input")
    with col_btn:
        analyze_btn = st.button("🔍 分析",use_container_width=True,key="single_btn")

    if st.session_state.single_result and st.session_state.single_sym:
        safe = _html_escape.escape(st.session_state.single_result).replace('\n','<br>')
        st.markdown(f"""
        <div class="ai-box" style="border-left-color:#8b5cf6;margin-top:12px;">
            <div style="font-weight:700;color:#8b5cf6;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(139,92,246,0.2);">
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

        v4_txt  = (f"Score:{v4_row['score']} | PVO:{v4_row.get('pvo',0):+.2f} | VRI:{v4_row.get('vri',0):.1f} | 訊號:{v4_row.get('signal','—')}" if v4_row else "（V4無資料）")
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
# 主程式
# ══════════════════════════════════════════════════════════════

def main():
    stock_set = load_stock_set()
    watchlist = stock_set.get("watchlist", {}).get("symbols", [])
    render_sidebar(stock_set)

    use_mock = st.session_state.use_mock

    if use_mock:
        v4,v12,regime,market,hist = _mock_v4(),_mock_v12(),_mock_regime(),_mock_market(),_mock_history()
        status = {k: False for k in ["v4","v12","regime","market","hist"]}
    else:
        with st.spinner("🔄 從 GitHub 讀取最新快照..."):
            v4,v12,regime,market,hist,status = load_all_snapshots()
        v4     = v4     or _mock_v4()
        v12    = v12    or _mock_v12()
        regime = regime or _mock_regime()
        market = market or _mock_market()
        hist   = hist   or _mock_history()

    st.session_state["last_refresh"] = datetime.now().strftime("%H:%M:%S")

    all_live   = all(status.values())
    mode_label = "LIVE 資料" if all_live else ("DEMO 模式" if use_mock else "部分 DEMO")
    mode_css   = "background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#10b981;" if all_live else "background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.4);color:#f59e0b;"

    run_mode    = v4.get("run_mode","—")
    gen_at      = v4.get("generated_at","—")
    bear        = regime.get("bear",0.33)
    bull        = regime.get("bull",0.33)
    label       = regime.get("label","—")
    a_path      = regime.get("active_path","—")
    # [NEW-01] 顯示 TAIEX 原始指數
    idx_close   = regime.get("index_close", market.get("index_close","—"))
    idx_chg     = market.get("index_chg_pct","—")
    src         = regime.get("data_source","—")

    st.markdown(f"""
    <div class="hq-header">
        <div>
            <div class="hq-title">⚡ 資源法 AI 戰情室 <span style="font-size:0.9rem;opacity:0.6;">v4.0</span></div>
            <div class="hq-sub">FinMind Y9999 TAIEX + GitHub Storage + Pure Display Layer</div>
        </div>
        <div class="hq-badge" style="{mode_css}">{mode_label}</div>
    </div>
    """, unsafe_allow_html=True)

    if not use_mock and not all_live:
        missing = [k.upper() for k,v in status.items() if not v]
        st.warning(f"⚠️ GitHub 資料讀取失敗：{' / '.join(missing)}。顯示模擬資料。")

    idx_str = f"{idx_close:,.2f}" if isinstance(idx_close, float) else str(idx_close)
    chg_str = f"{idx_chg:+.2f}%" if isinstance(idx_chg, float) else str(idx_chg)
    chg_css = "chip-ok" if isinstance(idx_chg, float) and idx_chg >= 0 else "chip-err"
    dom_css = "chip-ok" if bull > bear else ("chip-err" if bear > bull else "chip-warn")

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

    # [FIX-02] 自選股快覽
    render_watchlist_section(v4, v12, watchlist)

    # 個股 AI
    render_single_stock_panel(v4, v12, regime)

    # Gemini 全盤分析
    st.markdown("""<div class="sec-header sec-ai"><span style="font-size:1.0rem;font-weight:800;color:#8b5cf6;">🤖 Gemini 全盤 AI 分析</span></div>""", unsafe_allow_html=True)
    ai1,ai2 = st.columns([1,5])
    with ai1:
        ai_btn = st.button("🤖 執行分析",use_container_width=True,key="ai_btn")
    with ai2:
        st.caption("分析 TAIEX 大盤環境 + V4 TOP5 + V12.1 部位（需 Gemini API Key）")

    if st.session_state.ai_summary:
        safe2 = _html_escape.escape(st.session_state.ai_summary).replace('\n','<br>')
        st.markdown(f'<div class="ai-box">{safe2}</div>', unsafe_allow_html=True)

    if ai_btn:
        with st.spinner("🤖 Gemini 分析中..."):
            summary = call_gemini(build_dashboard_prompt(v4,v12,regime,market), st.session_state.gemini_key)
        st.session_state.ai_summary = summary
        st.rerun()

    st.markdown("---")

    # Tab 區
    tab1,tab2,tab3,tab4 = st.tabs(["🟦 V4 市場強度","🟩 V12.1 交易決策","🟨 Regime 大盤","📜 交易歷史"])
    with tab1:
        render_v4_section(v4)
    with tab2:
        render_v12_section(v12)
    with tab3:
        render_regime_section(regime, market)
    with tab4:
        render_history_tab(hist or [])


if __name__ == "__main__":
    main()
