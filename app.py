"""
0410-V1.py — 資源法 AI 戰情室（融合版）
整合：
  - app.py 原 Streamlit """
app.py — 資源法 AI 戰情室-0410-2
Streamlit 主程式 | 台股/美股雙軌 | 四層數據防火牆
v2.3 新增：個股 Gemini 深度分析 | 上漲10%機率 | 動態停利停損
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
# 全域 CSS
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Sans+TC:wght@400;600;700;900&display=swap');

:is(body, .stApp, [data-testid]) p,
:is(body, .stApp, [data-testid]) span:not([style*="color"]),
:is(body, .stApp, [data-testid]) label,
:is(body, .stApp, [data-testid]) div:not([style*="color"]) {
    color: #1e293b;
}

:root {
    --bg-main:    #f4f6fb;
    --bg-panel:   #ffffff;
    --bg-card:    #f9fafe;
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
[data-testid="stRadio"] label,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] p {
    color: #1e293b !important;
    font-weight: 500 !important;
}

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

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

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

.ai-summary {
    background: linear-gradient(135deg, #eff6ff, #f8faff);
    border: 1px solid #bfdbfe;
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 1.1rem;
    line-height: 1.8;
    color: var(--text);
    box-shadow: var(--shadow);
}

.status-bar {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--red);
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

.decision-header {
    background: linear-gradient(135deg, #1a56db08, #0891b208);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 18px;
    margin: 16px 0 8px 0;
}

.final-decision-box {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
    border: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 18px;
    margin: 8px 0;
    box-shadow: 0 4px 16px rgba(26,86,219,0.12);
}

.single-stock-header {
    background: var(--bg-panel);
    border: 1px solid #e2e8f0;
    border-left: 4px solid #059669;
    border-radius: var(--radius);
    padding: 10px 16px;
    margin: 12px 0 4px 0;
}

.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    border-color: var(--border2) !important;
    border-radius: var(--radius) !important;
}

.stDataFrame { background: var(--bg-panel) !important; border-radius: var(--radius) !important; }
.stAlert { border-radius: var(--radius) !important; }

[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] label *,
[data-testid="stSelectbox"] label, [data-testid="stSelectbox"] label *,
[data-testid="stSlider"] label, [data-testid="stSlider"] label *,
[data-testid="stRadio"] label, [data-testid="stRadio"] label *,
[data-testid="stRadio"] div[role="radiogroup"] label,
[data-testid="stRadio"] div[role="radiogroup"] p { color: #1e293b !important; }
[data-baseweb="radio"] label, [data-baseweb="radio"] label *,
[data-baseweb="checkbox"] label, [data-baseweb="checkbox"] label *,
[data-baseweb="tab"] { color: #1e293b !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #000000 !important; font-weight:700 !important; }
[role="tab"] { color: #1e293b !important; }
[role="tab"][aria-selected="true"] { color: #000000 !important; font-weight:700 !important; }
[role="radio"] + div, [role="radio"] + div * { color: #1e293b !important; }
button[data-baseweb="tab"], button[data-baseweb="tab"] span,
button[data-baseweb="tab"] p { color: #1e293b !important; }
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] span,
button[data-baseweb="tab"][aria-selected="true"] p { color: #000000 !important; }
[data-testid="stCheckbox"] input[type="checkbox"] { accent-color: #1a56db; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([style*="color"]) { color: #1e293b !important; }

[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label * { color: #1e293b !important; }
[data-baseweb="select"] { background-color: #ffffff !important; }
[data-baseweb="select"] * { color: #dc2626 !important; }
[role="listbox"], [role="option"] { background-color: #ffffff !important; color: #dc2626 !important; }
[role="option"]:hover { background-color: #f5f5f5 !important; color: #dc2626 !important; }
[role="option"][aria-selected="true"] {
    background-color: #fef2f2 !important;
    color: #dc2626 !important;
    font-weight: 700 !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 常數
# ===========================================================================
DEFAULT_TW_WATCHLIST = [
    "3030", "3706", "8096", "2313", "4958",
    "2330", "2317", "2454", "2308", "2382", "2303", "3711", "2412", "2357", "3231",
    "2379", "3008", "2395", "3045", "2327", "2408", "2377", "6669", "2301", "3034",
    "2345", "2474", "3037", "4938", "3443", "2353", "2324", "2603", "2609", "1513",
    "3293", "3680", "3529", "3131", "5274", "6223", "6805", "3017", "3324", "6515",
    "3661", "3583", "6139", "3035", "1560", "8299", "3558", "6187", "3406", "3217",
    "6176", "6415", "6206", "8069", "3264", "5269", "2360", "6271", "3189", "6438",
    "8358", "6231", "2449", "3030", "8016", "6679", "3374", "3014", "3211",
    "6213", "2404", "2480", "3596", "6202", "5443", "5347", "5483", "6147",
    "2313", "3037", "8046", "2368", "4958", "2383", "6269", "5469", "5351",
    "4909", "8050", "6153", "6505", "1802", "3708", "8213", "1325",
    "2344", "6239", "3260", "4967", "6414", "2337", "8096",
    "3551", "2436", "2375", "2492", "2456", "3229", "6173", "3533",
    "3491", "6271", "2313", "2367", "6285", "6190",
    "3062", "2419", "2314", "3305", "3105", "2312", "8086",
    "3081", "2455", "6442", "3163", "4979", "3363", "6451",
    "3450", "4908", "4977", "3234", "2360",
    "1711","1727","2404","2489","3060","3374","3498","3535","3580","3587","3665","4749","4989","6187","6217","6290","6418","6443","6470","6542","6546","6706","6831","6861","6877","8028","8111"
]
DEFAULT_US_WATCHLIST = ["ABNB", "ADBE", "AMD", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI", "AAPL", "AMAT", "ASML", "AXON", "TEAM", "ADSK", "ADP", "ARM", "APP", "AZN", "BIIB", "BKNG", "BKR", "AVGO", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CTAS", "CSCO", "CSGP", "CTSH", "CMCSA", "CPRT", "COST", "CRWD", "CSX", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "META", "FTNT", "GEHC", "GFS", "GILD", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KLAC", "KHC", "LRCX", "LIN", "LULU", "MAR", "MDB", "MRVL", "MELI", "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MNST", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PCAR", "PAYX", "PYPL", "PEP", "PDD", "PLTR", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TMUS", "TSLA", "TTWO", "TTD", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS", "AMKR", "COHR", "CRUS", "ENTG", "LSCC", "MPWR", "MTSI", "ONTO", "QRVO", "SWKS", "TER", "TSM", "MMM", "ABT", "ABBV", "ACN", "AES", "AFL", "A", "APD", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "MO", "AMCR", "AEE", "AAL", "AXP", "AIG", "AMT", "AWK", "AMP", "ABCB", "AME", "APH", "AON", "AOS", "APA", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "AZO", "AVB", "AVY", "BAC", "BBWI", "BAX", "BDX", "BRK-B", "BBY", "BG", "BIO", "BLDR", "BLK", "BK", "BA", "BWA", "BXP", "BSX", "BMY", "BR", "BX", "CHRW", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CAT", "CBOE", "CBRE", "CE", "CNC", "CNP", "CF", "CRL", "SCHW", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "C", "CFG", "CLX", "CME", "CMS", "KO", "CL", "CMA", "CAG", "COP", "ED", "STZ", "COO", "GLW", "CTVA", "CCI", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "DVN", "DECK", "DLR", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EMR", "ENPH", "ETR", "EOG", "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EVRG", "ES", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FRT", "FDX", "FICO", "FIS", "FITB", "FE", "FMC", "F", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GNRC", "GD", "GE", "GEV", "GIS", "GM", "GPC", "GL", "GPN", "GS", "GWW", "HAL", "HIG", "HAS", "HCA", "HSIC", "HSY", "HPE", "HLT", "HOLX", "HD", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IEX", "ITW", "ILMN", "INCY", "IR", "ICE", "IBM", "IP", "IPG", "IFF", "IVZ", "INVH", "IQV", "IRM", "JKHY", "J", "JBHT", "JBL", "SJM", "JNJ", "JCI", "JPM", "K", "KEY", "KEYS", "KMB", "KIM", "KMI", "KR", "KVUE", "LHX", "LH", "LW", "LVS", "LEG", "LDOS", "LEN", "LLY", "LYV", "LKQ", "LMT", "L", "LOW", "LYB", "MTB", "MPC", "MKTX", "MMC", "MLM", "MAS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM", "MAA", "MRNA", "MHK", "TAP", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NSC", "NTRS", "NOC", "NCLH", "NOV", "NRG", "NUE", "NVR", "OXY", "OMC", "OKE", "ORCL", "OTIS", "PKG", "PH", "PAYC", "PNR", "PRGO", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "PODD", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PTC", "PEG", "PSA", "PHM", "PWR", "DGX", "RL", "RJF", "RTX", "O", "REG", "RF", "RSG", "RMD", "RHI", "ROK", "ROL", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STLD", "STX", "SRE", "NOW", "SHW", "SMCI", "SPG", "SNA", "SO", "SOLV", "LUV", "SWK", "STT", "STE", "SYK", "SYF", "SYY", "TECH", "TROW", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "UBER", "UDR", "ULTA", "USB", "UNP", "UAL", "UNH", "UPS", "URI", "UHS", "UNM", "VLTO", "VLO", "VTR", "VRSN", "VZ", "VTRS", "V", "VMC", "VST", "WRB", "WAB", "WMT", "WBA", "DIS", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WY", "WMB", "WYNN", "XYL", "YUM", "ZBRA", "ZBH", "ZTS", "PARA"
]
BENCHMARK_TW = "0050.TW"
BENCHMARK_US = "SPY"
LOOKBACK_DAYS = 180
ALPHA_SEEDS_PATH = "alpha_seeds.json"


# ===========================================================================
# 勝率型態辨識
# ===========================================================================
def classify_pattern(dec: dict) -> dict:
    action     = dec.get("action", "")
    pvo_status = dec.get("pvo_status", "")
    vri_status = dec.get("vri_status", "")
    vri        = dec.get("vri", 0)
    pvo        = dec.get("pvo", 0)

    is_strong_buy = (action == "強力買進")
    is_money_in   = ("資金流入" in pvo_status)
    is_fire       = ("主力點火" in pvo_status)
    is_hot        = ("擁擠過熱" in vri_status)
    is_cool       = ("情緒整理" in vri_status)

    patterns = []

    if is_strong_buy and is_money_in and is_cool:
        patterns.append({
            "code": "A",
            "label": "📈 資金流入＋情緒整理＋強力買進",
            "win10": 52.6, "win20": 37.6,
            "css": "pattern-a",
            "desc": "穩 + 爆發兼具｜主升段最佳"
        })

    if is_strong_buy and is_fire and is_hot:
        patterns.append({
            "code": "B",
            "label": "🔥 主力點火＋擁擠過熱＋強力買進",
            "win10": 52.4, "win20": None,
            "css": "pattern-b",
            "desc": "超短線爆發最強（風險偏高）"
        })

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
    if df is None or df.empty or "VRI" not in df.columns:
        return 0.0
    recent = df["VRI"].tail(20)
    return round((recent > 40).sum() / min(len(recent), 20), 2)


def calc_pvo_ratio(df) -> float:
    if df is None or df.empty or "PVO" not in df.columns:
        return 0.0
    recent = df["PVO"].tail(20)
    return round((recent > 0).sum() / min(len(recent), 20), 2)


# ===========================================================================
# ★ 新增：上漲10%機率 / 動態停利停損計算
# ===========================================================================
def calc_upside_10pct_prob(dec: dict, s2: dict, df_ind) -> dict:
    """
    推算個股上漲至10%的機率、停利價、停損價。

    機率模型（多因子加權）：
      base  = 型態勝率 / 100（無型態→ slope_z>1 取0.38，否則0.30）
      adj  += Slope Z × 0.05（上限+0.12）
      adj  += PVO加權（>10:+8%, 0~10:+3%, <0:-10%）
      adj  += VRI加權（健康區40~75:+5%, 過熱>90:-5%）
      adj  += T值≥2.0:+5%
      adj  += EV>5%:+5%, EV>3%:+3%
      prob  = clamp(base+adj, 5%, 95%)

    停損：ATR代理 × VRI動態倍數（1.5~2.5x），最多12%
    停利：EV×1.5 或最低10%，確保風報比≥1.5x
    """
    close = dec.get("close", 0)
    if close <= 0:
        return {
            "prob": None, "stop_loss": None, "take_profit": None,
            "stop_loss_pct": None, "tp_pct": None, "rr_ratio": None
        }

    pat     = classify_pattern(dec)
    pvo     = dec.get("pvo", 0)
    vri     = dec.get("vri", 0)
    slope_z = dec.get("slope_z", 0)
    ev_raw  = s2.get("ev", None)
    t_stat  = s2.get("t_stat", None)

    # 基礎機率
    if pat["best_win10"] > 0:
        base_prob = pat["best_win10"] / 100.0
    else:
        base_prob = 0.38 if slope_z > 1.0 else 0.30

    # 多因子調整
    adj = 0.0
    adj += min(slope_z * 0.05, 0.12)
    if pvo > 10:         adj += 0.08
    elif pvo > 0:        adj += 0.03
    else:                adj -= 0.10
    if 40 <= vri <= 75:  adj += 0.05
    elif vri > 90:       adj -= 0.05
    if t_stat is not None and abs(t_stat) >= 2.0:
        adj += 0.05
    if isinstance(ev_raw, (int, float)):
        if ev_raw > 5:   adj += 0.05
        elif ev_raw > 3: adj += 0.03

    prob = max(0.05, min(0.95, base_prob + adj))

    # ATR 代理停損（近20日 High-Low 均值）
    if (df_ind is not None and not df_ind.empty
            and 'High' in df_ind.columns and 'Low' in df_ind.columns):
        daily_range = (df_ind['High'] - df_ind['Low']).tail(20)
        atr_pct = float(daily_range.mean() / close) if close > 0 else 0.02
    else:
        atr_pct = 0.02

    stop_multiplier   = 1.5 + (vri / 100.0)          # VRI越高越寬 (1.5x ~ 2.5x)
    stop_loss_pct     = min(atr_pct * stop_multiplier, 0.12)
    stop_loss_price   = round(close * (1 - stop_loss_pct), 2)

    # 停利
    if isinstance(ev_raw, (int, float)) and ev_raw > 0:
        take_profit_pct = min(ev_raw / 100.0 * 1.5, 0.20)
    else:
        take_profit_pct = 0.10
    take_profit_pct   = max(take_profit_pct, stop_loss_pct * 1.5)  # 風報比至少1.5x
    take_profit_price = round(close * (1 + take_profit_pct), 2)

    rr = (take_profit_pct / stop_loss_pct) if stop_loss_pct > 0 else None

    return {
        "prob":          round(prob * 100, 1),
        "stop_loss":     stop_loss_price,
        "take_profit":   take_profit_price,
        "stop_loss_pct": round(stop_loss_pct * 100, 1),
        "tp_pct":        round(take_profit_pct * 100, 1),
        "rr_ratio":      round(rr, 2) if rr is not None else None,
    }


def is_final_candidate(dec: dict, df_ind) -> bool:
    pat = classify_pattern(dec)
    if not pat["is_key_pattern"]:
        return False
    pvo_ratio = calc_pvo_ratio(df_ind)
    vri_ratio = calc_vri_ratio(df_ind)
    return pvo_ratio > 0.6 and vri_ratio > 0.6


# ===========================================================================
# Session State 初始化
# ===========================================================================
def init_session():
    defaults = {
        "tw_watchlist":          DEFAULT_TW_WATCHLIST.copy(),
        "us_watchlist":          DEFAULT_US_WATCHLIST.copy(),
        "last_scan_time":        None,
        "scan_results":          {},
        "benchmark_tw_df":       None,
        "benchmark_us_df":       None,
        "market_sentiment_tw":   None,
        "market_sentiment_us":   None,
        "ai_summary":            "",
        "data_health":           {},
        "all_warnings":          [],
        "selected_stock":        None,
        "target_date":           datetime.today().strftime("%Y-%m-%d"),
        "gemini_api_key":        _ENV_GEMINI_KEY,
        "active_market":         "TW",
        "single_stock_result":   "",   # ★ 個股分析結果暫存
        "single_stock_sym":      "",   # ★ 個股分析代號暫存
        "single_stock_upside":   {},   # ★ 個股停利停損暫存
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ===========================================================================
# JS 顏色修正
# ===========================================================================
def inject_color_fix():
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        function fixColors() {
            document.querySelectorAll('button[data-baseweb="tab"]').forEach(el => {
                el.style.color = '#1e293b';
                el.querySelectorAll('*').forEach(c => c.style.color = '#1e293b');
            });
            document.querySelectorAll('button[data-baseweb="tab"][aria-selected="true"]').forEach(el => {
                el.style.color = '#000000';
                el.querySelectorAll('*').forEach(c => c.style.color = '#000000');
            });
            document.querySelectorAll('[data-baseweb="radio"] label, [data-testid="stRadio"] label').forEach(el => {
                el.style.color = '#1e293b';
                el.querySelectorAll('*').forEach(c => {
                    if(!c.style.color || c.style.color === 'rgb(250, 250, 250)')
                        c.style.color = '#1e293b';
                });
            });
            document.querySelectorAll('[data-testid="stCheckbox"] label').forEach(el => {
                el.style.color = '#1e293b';
            });
            document.querySelectorAll('[data-testid="stSidebar"] label, [data-testid="stSidebar"] p').forEach(el => {
                el.style.color = '#1e293b';
            });
        }
        fixColors();
        const obs = new MutationObserver(fixColors);
        obs.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """, height=0, scrolling=False)


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
# Gemini AI
# ===========================================================================
def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ 請在側欄輸入 Gemini API Key 才能啟用 AI 分析。"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
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


# ===========================================================================
# 全盤 Gemini Prompt
# ===========================================================================
def build_gemini_prompt(scan_results: dict, market_sentiment: dict,
                        market: str = "TW") -> str:
    bull_candidates = []
    final_flag_syms = []

    for sym, res in scan_results.items():
        dec    = res.get("decision", {})
        s1     = res.get("stage1", {})
        s2     = res.get("stage2", {})
        df_ind = res.get("indicator_df")

        if dec.get("direction") != "做多":
            continue

        pat       = classify_pattern(dec)
        pvo_ratio = calc_pvo_ratio(df_ind)
        vri_ratio = calc_vri_ratio(df_ind)
        is_final  = is_final_candidate(dec, df_ind)
        upside    = calc_upside_10pct_prob(dec, s2, df_ind)

        if is_final:
            final_flag_syms.append(sym)

        pat_codes = "+".join([p["code"] for p in pat["patterns"]]) if pat["patterns"] else "無"
        pat_desc  = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "無型態"

        trust     = res.get("trust", {})
        trust_net = trust.get("trust_net_10d", None)
        trust_str = f"{trust_net:+,.0f}張" if trust_net is not None else "N/A"

        ev_val = s2.get("ev", None)
        ev_str = f"{ev_val:+.2f}%" if isinstance(ev_val, (int, float)) else "N/A"
        t_stat = s2.get("t_stat", None)
        t_str  = f"{t_stat:.2f}" if t_stat is not None else "N/A"
        path   = translate_path(str(s2.get("path", "N/A")))

        bull_candidates.append({
            "sym":       sym,
            "close":     dec.get("close", 0),
            "action":    dec.get("action", "---"),
            "pvo":       dec.get("pvo", 0),
            "vri":       dec.get("vri", 0),
            "slope":     dec.get("slope", 0),
            "slope_z":   dec.get("slope_z", 0),
            "pvo_status":dec.get("pvo_status", ""),
            "vri_status":dec.get("vri_status", ""),
            "pvo_ratio": pvo_ratio,
            "vri_ratio": vri_ratio,
            "pat_codes": pat_codes,
            "pat_desc":  pat_desc,
            "best_win10":pat["best_win10"],
            "ev":        ev_str,
            "t_stat":    t_str,
            "path":      path,
            "trust":     trust_str,
            "is_final":  is_final,
            "s1_pass":   s1.get("pass", False),
            "score":     dec.get("score", 0),
            "signal_level": dec.get("signal_level", ""),
            "prob10":    upside.get("prob", "N/A"),
            "tp_price":  upside.get("take_profit", "N/A"),
            "sl_price":  upside.get("stop_loss", "N/A"),
            "tp_pct":    upside.get("tp_pct", "N/A"),
            "sl_pct":    upside.get("stop_loss_pct", "N/A"),
            "rr":        upside.get("rr_ratio", "N/A"),
        })

    bull_candidates.sort(
        key=lambda x: (int(x["is_final"]), x["best_win10"], x["slope_z"]),
        reverse=True
    )
    top_candidates = bull_candidates[:20]

    label   = market_sentiment.get("label", "不明") if market_sentiment else "不明"
    slope5  = market_sentiment.get("slope_5d", 0)  if market_sentiment else 0
    slope20 = market_sentiment.get("slope_20d", 0) if market_sentiment else 0
    bear_p  = market_sentiment.get("bear", 33)     if market_sentiment else 33
    bull_p  = market_sentiment.get("bull", 33)     if market_sentiment else 33
    neu_p   = market_sentiment.get("neutral", 34)  if market_sentiment else 34
    mkt_label = "台股" if market == "TW" else "美股"

    candidate_lines = []
    for c in top_candidates:
        final_marker = "⭐最終候選" if c["is_final"] else ""
        line = (
            f"【{c['sym']}】{final_marker}\n"
            f"  現價:{c['close']:.2f} | 操作:{c['action']} {c['signal_level']}\n"
            f"  PVO:{c['pvo']:+.2f}({c['pvo_status']}) | VRI:{c['vri']:.1f}({c['vri_status']})\n"
            f"  Slope:{c['slope']:+.3f}% | SlopeZ:{c['slope_z']:+.2f} | Score:{c['score']}\n"
            f"  PVO波動率:{c['pvo_ratio']:.0%} | VRI波動率:{c['vri_ratio']:.0%}\n"
            f"  高勝率型態:{c['pat_codes']}({c['pat_desc']}) | 最高勝率10%:{c['best_win10']}%\n"
            f"  EV期望值:{c['ev']} | T值:{c['t_stat']} | 路徑:{c['path']}\n"
            f"  投信10日:{c['trust']} | Stage1:{'✅' if c['s1_pass'] else '❌'}\n"
            f"  上漲10%機率:{c['prob10']}% | 停利:{c['tp_price']}(+{c['tp_pct']}%) | 停損:{c['sl_price']}(-{c['sl_pct']}%) | 風報比:{c['rr']}x"
        )
        candidate_lines.append(line)

    candidates_text = "\n\n".join(candidate_lines) if candidate_lines else "（無做多標的）"
    final_list = "、".join(final_flag_syms) if final_flag_syms else "（無）"

    return f"""你是一位資深量化交易分析師，專精台股/美股技術分析、統計套利與資金流向研究。

═══════════════════════════════════════
【市場環境】{mkt_label}
═══════════════════════════════════════
大盤情緒: {label}
熊市機率: {bear_p}% | 震盪機率: {neu_p}% | 牛市機率: {bull_p}%
5日動能斜率: {slope5:+.4f} | 20日趨勢斜率: {slope20:+.4f}

═══════════════════════════════════════
【做多候選標的】（共 {len(top_candidates)} 檔，僅含做多方向）
最終候選：{final_list}
═══════════════════════════════════════
{candidates_text}

═══════════════════════════════════════
【指標說明】
- PVO: 量能動能指標（>10主力點火, 0~10資金流入, <0資金撤退）
- VRI: 資金強度指標（>90擁擠過熱, 40~75健康區間, <40情緒整理）
- SlopeZ: 標準化趨勢強度（>1.5強勢, >2.0極強）
- PVO/VRI波動率: 近20日有效天數佔比（>60%代表持續性強）
- 三大型態: A=資金流入+整理+強買(52.6%), B=點火+過熱+強買(52.4%), C=VRI過熱+強買(59.0%)
- EV期望值: 基於歷史回測的期望報酬率
- T值: 統計顯著性（>2.0具顯著性）
- 上漲10%機率: 多因子加權模型估算（非保證值）

═══════════════════════════════════════
【分析任務】
═══════════════════════════════════════
**所有的句子都是經由資深股票分析師與專業統計學專家深思後的回答
**一、大盤風險評估**（150字以內）
- 當前市場結構判讀：動能、趨勢、情緒三維分析
- 5日與20日斜率背離或共振的含義
- 熊/牛/震盪機率對今日操作的影響
- 適合的持倉水位建議（含理由）

**二、最終候選精選**（從⭐最終候選中選出1~3檔，每檔150字以內）
對每一檔分析：
1. 技術面：Slope Z趨勢強度、突破位判讀
2. 資金面：PVO狀態、投信籌碼動向
3. 統計優勢：型態勝率、EV期望值、T值顯著性
4. 模型數據：上漲10%機率解讀、停利停損合理性、風報比評估
5. 操作建議：進場時機、停損邏輯、目標區間

**三、風險提示**（100字以內）
- 今日主要風險因子
- 需要迴避的情境

**格式**：條列與段落並用，所有論點引用具體指標數值，禁止泛泛而談。"""


# ===========================================================================
# ★ 新增：個股 Gemini Prompt 建構
# ===========================================================================
def build_single_stock_prompt(matched_key: str, res: dict,
                               market_sentiment_tw, market_sentiment_us) -> str:
    dec_s    = res.get("decision", {})
    s1_s     = res.get("stage1", {})
    s2_s     = res.get("stage2", {})
    df_s     = res.get("indicator_df")
    trust_s  = res.get("trust", {})
    mkt_s    = res.get("market", "TW")

    pat_s     = classify_pattern(dec_s)
    pvo_r_s   = calc_pvo_ratio(df_s)
    vri_r_s   = calc_vri_ratio(df_s)
    is_fin_s  = is_final_candidate(dec_s, df_s)
    upside_s  = calc_upside_10pct_prob(dec_s, s2_s, df_s)

    ev_s    = s2_s.get("ev", None)
    t_s     = s2_s.get("t_stat", None)
    trust_n = trust_s.get("trust_net_10d", None)
    pat_codes_s = "+".join([p["code"] for p in pat_s["patterns"]]) if pat_s["patterns"] else "無"
    pat_desc_s  = " | ".join([p["desc"] for p in pat_s["patterns"]]) if pat_s["patterns"] else "無型態"
    path_s      = translate_path(str(s2_s.get("path", "N/A")))
    sent_s      = (market_sentiment_tw if mkt_s == "TW" else market_sentiment_us) or {}

    ev_str  = f"{ev_s:+.2f}%" if isinstance(ev_s, (int, float)) else "N/A"
    t_str   = f"{t_s:.2f}"   if t_s is not None else "N/A"
    t_sig   = "✅ 顯著(≥2.0)" if (t_s is not None and abs(t_s) >= 2.0) else "⚠️ 不顯著"
    trust_str = f"{trust_n:+,.0f}張" if trust_n is not None else "無資料"

    prob_s  = upside_s.get('prob', 'N/A')
    tp_p    = upside_s.get('take_profit', 'N/A')
    sl_p    = upside_s.get('stop_loss', 'N/A')
    tp_pct  = upside_s.get('tp_pct', 'N/A')
    sl_pct  = upside_s.get('stop_loss_pct', 'N/A')
    rr      = upside_s.get('rr_ratio', 'N/A')

    return f"""你是資深量化交易分析師，精通台股/美股技術分析、統計套利與資金流向研究。
請對以下單一股票進行深度專項分析，每段至少150字，所有論點必須引用具體指標數值，禁止泛泛而談。

═══════════════════════════════
【個股基本資料】
═══════════════════════════════
代號: {matched_key}（{'台股' if mkt_s == 'TW' else '美股'}）
現價: {dec_s.get('close', 0):.2f}
操作信號: {dec_s.get('action', '---')} {dec_s.get('signal_level', '')}
方向判定: {dec_s.get('direction', '---')}
最終候選: {'✅ 是' if is_fin_s else '❌ 否'}
前次操作: {dec_s.get('last_action', '---')}

═══════════════════════════════
【量化指標】
═══════════════════════════════
PVO: {dec_s.get('pvo', 0):+.2f}（{dec_s.get('pvo_status', '')}）
VRI: {dec_s.get('vri', 0):.1f}（{dec_s.get('vri_status', '')}）
Slope: {dec_s.get('slope', 0):+.3f}%
Slope Z: {dec_s.get('slope_z', 0):+.2f}（標準化趨勢強度，>1.5強勢，>2.0極強）
Score: {dec_s.get('score', 0)}
PVO波動率(近20日): {pvo_r_s:.0%}（>60%代表量能持續性強）
VRI波動率(近20日): {vri_r_s:.0%}（>60%代表資金持續性強）

═══════════════════════════════
【統計數據】
═══════════════════════════════
EV期望值: {ev_str}（歷史回測期望報酬）
T值: {t_str}（統計顯著性，{t_sig}）
Stage1通過: {'✅' if s1_s.get('pass') else '❌'}
路徑狀態: {path_s}
高勝率型態: {pat_codes_s}（{pat_desc_s}）
最高勝率10%: {pat_s['best_win10']}%

═══════════════════════════════
【模型推算結果】（供參考，非保證）
═══════════════════════════════
上漲至10%機率: {prob_s}%（多因子加權：型態勝率+SlopeZ+PVO+VRI+T值+EV）
停利目標價: {tp_p}（+{tp_pct}%）
停損位置: {sl_p}（-{sl_pct}%，ATR×VRI動態倍數法）
風報比: {rr}x（停利幅度÷停損幅度，≥1.5為合格）

═══════════════════════════════
【籌碼面】
═══════════════════════════════
投信近10日買賣超: {trust_str}

═══════════════════════════════
【大盤環境】
═══════════════════════════════
{'台股' if mkt_s == 'TW' else '美股'}情緒: {sent_s.get('label', 'N/A')}
熊市機率: {sent_s.get('bear', 'N/A')}% | 牛市機率: {sent_s.get('bull', 'N/A')}%
5日斜率: {sent_s.get('slope_5d', 0):+.4f} | 20日斜率: {sent_s.get('slope_20d', 0):+.4f}

═══════════════════════════════
【分析任務】請完成以下五段分析
═══════════════════════════════
**所有的句子都是經由資深股票分析師與專業統計學專家深思後的回答
**一、技術面研判**（150字以內）
- Slope Z={dec_s.get('slope_z', 0):+.2f} 的趨勢位階含義（強/弱/轉折風險）
- PVO={dec_s.get('pvo', 0):+.2f}（{dec_s.get('pvo_status', '')}）的主力行為解讀
- VRI={dec_s.get('vri', 0):.1f}（{dec_s.get('vri_status', '')}）的資金溫度分析
- PVO波動率{pvo_r_s:.0%}、VRI波動率{vri_r_s:.0%} 代表的能量穩定度評估

**二、統計優勢評估**（150字以內）
- EV={ev_str} 的歷史期望報酬解讀（高/低/負含義）
- T值={t_str} 的統計顯著性分析（樣本可信度）
- 命中型態 {pat_codes_s}（{pat_desc_s}）的歷史勝率意義
- 上漲至10%機率 {prob_s}% 的可信度評估與影響因子說明

**三、籌碼面分析**（100字以內）
- 投信買賣超 {trust_str} 的動向解讀
- 法人籌碼對股價潛在影響
- 若無投信資料，說明其他替代觀察指標

**四、具體操作建議**（150字以內）
- 進場時機與確認條件（需等待哪些指標變化）
- 停損邏輯：{sl_p}（-{sl_pct}%）的合理性分析，說明為何此位置是關鍵
- 停利目標：{tp_p}（+{tp_pct}%）的依據與分批出場策略
- 風報比 {rr}x 的評估（是否值得操作）
- 建議持倉比例（佔總資金）與最大持有天數
- 大盤環境對此股操作的影響

**五、主要風險因子**（100字以內）
- 需要警惕的指標反轉信號（PVO/VRI/Slope 什麼情況應出場）
- 市場環境惡化時的應對策略
- 此股特有的風險注意事項

**格式**：條列與段落並用，所有論點必須引用具體數值，結尾附上一句操作總結。"""


# ===========================================================================
# 掃描核心
# ===========================================================================
def run_scan(watchlist: list, target_date: str, market_type: str, progress_bar=None):
    start_str, end_str = get_date_range(target_date)
    results    = {}
    warns_all  = []
    health_all = {}

    bm_sym = BENCHMARK_TW if market_type == "TW" else BENCHMARK_US
    bm_res = cached_fetch(bm_sym, start_str, end_str)
    bm_df  = bm_res.get("indicator_df")
    if bm_df is not None and market_type == "TW":
        st.session_state.benchmark_tw_df    = bm_df
        st.session_state.market_sentiment_tw = get_market_sentiment(bm_df)
    elif bm_df is not None:
        st.session_state.benchmark_us_df    = bm_df
        st.session_state.market_sentiment_us = get_market_sentiment(bm_df)

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

        df       = res["indicator_df"]
        decision = get_decision(df, market=market_type)
        s1       = stage1_energy_filter(df)
        s2       = stage2_path_filter(sym, s1, ALPHA_SEEDS_PATH)

        trust_info = {"trust_net_10d": None, "trust_df": None}
        if market_type == "TW":
            sid           = sym.replace(".TWO", "").replace(".TW", "").strip()
            trust_net_10d = _INST_CACHE.get_recent_net(sid, days=10)
            trust_df      = _INST_CACHE.get(sid)
            trust_info    = {"trust_net_10d": trust_net_10d, "trust_df": trust_df}

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

    st.session_state.scan_results   = results
    st.session_state.last_scan_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.data_health    = health_all
    st.session_state.all_warnings   = warns_all
    return results


# ===========================================================================
# UI 元件
# ===========================================================================
def render_status_bar():
    last_scan   = st.session_state.last_scan_time or "尚未掃描"
    total_warns = len(st.session_state.all_warnings)
    health_ok   = sum(1 for h in st.session_state.data_health.values() if h.get("pass"))
    health_total= len(st.session_state.data_health)
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
    bear  = sentiment.get("bear", 33)
    neu   = sentiment.get("neutral", 34)
    bull  = sentiment.get("bull", 33)
    label = sentiment.get("label", "震盪")
    s5d   = sentiment.get("slope_5d", 0)
    s20d  = sentiment.get("slope_20d", 0)
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
            <div class="val neutral-val">{neu}%</div>
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


def get_badge(direction: str) -> str:
    _b = "display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;"
    if direction == "做多":
        return f'<span style="{_b}background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;">▲ 做多</span>'
    if direction == "做空":
        return f'<span style="{_b}background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;">▼ 做空</span>'
    return f'<span style="{_b}background:#fef3c7;color:#92400e;border:1px solid #fcd34d;">◆ 觀望</span>'


def render_pattern_badges(patterns: list) -> str:
    if not patterns:
        return ""
    _css_map = {
        "pattern-a": "background:#d1fae5;color:#065f46;border:1px solid #34d399;",
        "pattern-b": "background:#fee2e2;color:#991b1b;border:1px solid #f87171;",
        "pattern-c": "background:#fef3c7;color:#92400e;border:1px solid #fbbf24;",
    }
    _base = "display:inline-flex;align-items:center;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;margin:4px 2px;"
    html = '<div style="margin:6px 0;">'
    for p in patterns:
        win20_str = f" / 勝率20%: {p['win20']}%" if p.get("win20") else ""
        _style = _base + _css_map.get(p["css"], "")
        html += f'<span style="{_style}" title="{p["desc"]}">{p["label"]} 勝率10%: {p["win10"]}%{win20_str}</span>'
    html += "</div>"
    return html


def render_stock_card(sym: str, res: dict, show_final_badge: bool = False):
    if res.get("error"):
        st.markdown(f"""
        <div style="background:#fff5f5;border:1px solid #fca5a5;border-left:4px solid #dc2626;
                    border-radius:10px;padding:10px 16px;margin-bottom:10px;">
            <span style="font-weight:800;color:#dc2626;">{sym}</span>
            <span style="display:inline-block;margin-left:8px;padding:2px 8px;border-radius:6px;
                         background:#fee2e2;color:#991b1b;font-size:0.75rem;font-weight:700;">❌ 錯誤</span>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">{res['error']}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    dec    = res.get("decision", {})
    s1     = res.get("stage1", {})
    s2     = res.get("stage2", {})
    health = res.get("health", {})
    market = res.get("market", "TW")
    trust  = res.get("trust", {})
    df_ind = res.get("indicator_df")
    pat    = classify_pattern(dec)
    upside = calc_upside_10pct_prob(dec, s2, df_ind)

    direction   = dec.get("direction", "觀望")
    action      = dec.get("action", "---")
    close_px    = dec.get("close", 0)
    pvo         = dec.get("pvo", 0)
    vri         = dec.get("vri", 0)
    slope       = dec.get("slope", 0)
    slope_z     = dec.get("slope_z", 0)
    pvo_status  = dec.get("pvo_status", "")
    vri_status  = dec.get("vri_status", "")
    sig_level   = dec.get("signal_level", "")
    last_action = dec.get("last_action", "---")
    s1_pass     = "✅" if s1.get("pass") else "❌"
    s2_pass     = "✅" if s2.get("pass") else "❌"
    health_icon = "✅" if health.get("pass") else "⚠️"

    ev_raw = s2.get("ev", None)
    t_stat = s2.get("t_stat", None)
    path   = translate_path(str(s2.get("path", "---")))
    ev_str = f"+{ev_raw:.2f}%" if isinstance(ev_raw, (int, float)) and ev_raw > 0 else (
             f"{ev_raw:.2f}%" if isinstance(ev_raw, (int, float)) else "N/A")
    t_str  = f"{t_stat:.2f}" if t_stat is not None else "N/A"
    t_sig  = "✅顯著" if (t_stat is not None and abs(t_stat) >= 2.0) else ("⚠️" if t_stat is not None else "N/A")

    pat_desc_list = [p["desc"] for p in pat["patterns"]] if pat["patterns"] else []
    pat_desc_html = " ｜ ".join(pat_desc_list) if pat_desc_list else "無型態"

    trust_net_10d = trust.get("trust_net_10d", None)
    if trust_net_10d is not None:
        trust_color = "#059669" if trust_net_10d > 0 else ("#dc2626" if trust_net_10d < 0 else "#64748b")
        trust_label = f"近10日投信: <span style='color:{trust_color};font-weight:700;'>{trust_net_10d:+,.0f} 張</span>"
    else:
        trust_label = ""

    is_final     = is_final_candidate(dec, df_ind)
    badge        = get_badge(direction)
    pattern_html = render_pattern_badges(pat["patterns"])

    pvo_color   = "#059669" if pvo > 10 else ("#0891b2" if pvo > 0 else "#dc2626")
    vri_color   = "#059669" if 40 <= vri <= 75 else ("#dc2626" if vri > 90 else "#d97706")
    slope_color = "#059669" if slope > 0 else "#dc2626"
    ev_color    = "#059669" if (isinstance(ev_raw, (int, float)) and ev_raw > 3) else "#d97706"
    t_color     = "#059669" if (t_stat is not None and abs(t_stat) >= 2.0) else "#d97706"
    mkt_tag     = "🇹🇼" if market == "TW" else "🇺🇸"

    vri_ratio = calc_vri_ratio(df_ind)
    pvo_ratio = calc_pvo_ratio(df_ind)
    vri_ratio_color = "#059669" if vri_ratio > 0.6 else ("#d97706" if vri_ratio > 0.4 else "#dc2626")
    pvo_ratio_color = "#059669" if pvo_ratio > 0.6 else ("#d97706" if pvo_ratio > 0.4 else "#dc2626")

    # 上漲機率顏色
    prob_val = upside.get("prob")
    prob_color = "#059669" if (prob_val and prob_val >= 55) else ("#d97706" if (prob_val and prob_val >= 45) else "#dc2626")
    rr_val   = upside.get("rr_ratio")
    rr_color = "#059669" if (rr_val and rr_val >= 2.0) else ("#d97706" if (rr_val and rr_val >= 1.5) else "#dc2626")

    import html as _html_mod
    pvo_status_s = _html_mod.escape(str(pvo_status))
    vri_status_s = _html_mod.escape(str(vri_status))
    sig_level_s  = _html_mod.escape(str(sig_level))
    action_s     = _html_mod.escape(str(action))
    last_action_s= _html_mod.escape(str(last_action))
    pat_desc_s   = _html_mod.escape(pat_desc_html)

    final_badge_html = ""
    if show_final_badge and is_final:
        final_badge_html = '<span style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;">⭐ 最終決策候選</span>'

    if show_final_badge and is_final:
        _border_color = "#1a56db"; _bg = "linear-gradient(135deg,#eff6ff,#ffffff)"
    elif direction == "做多":
        _border_color = "#059669"; _bg = "#ffffff"
    elif direction == "做空":
        _border_color = "#dc2626"; _bg = "#ffffff"
    else:
        _border_color = "#0891b2"; _bg = "#ffffff"

    _row = "display:flex;gap:16px;flex-wrap:wrap;margin-top:5px;"
    _lbl = "font-size:0.8rem;color:#64748b;"

    st.markdown(f"""
    <div style="background:{_bg};border:1px solid #e2e8f0;border-left:4px solid {_border_color};
                border-radius:10px;padding:12px 16px;margin-bottom:10px;
                box-shadow:0 1px 4px rgba(30,41,59,0.07);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:1.05rem;font-weight:800;color:#1a56db;">{mkt_tag} {sym}</span>
        <span style="font-size:1.05rem;font-weight:600;font-family:monospace;">
            {"NT$" if market == "TW" else "$"} {close_px:,.2f}
        </span>
        <span>{badge} {final_badge_html}</span>
      </div>
      {pattern_html}
      <div style="font-size:0.8rem;color:#7c3aed;font-style:italic;margin:2px 0 6px 0;">
        📝 型態說明: {pat_desc_s}
      </div>
      <div style="font-size:0.82rem;color:#0891b2;font-weight:600;margin-bottom:4px;">
        AI判定: <b>{action_s}</b>&nbsp;{sig_level_s}&nbsp;｜&nbsp;前次: {last_action_s}
      </div>
      <div style="{_row}">
        <span style="{_lbl}">狀態:&nbsp;<b style="color:#1e293b;">{pvo_status_s}</b>&nbsp;/&nbsp;<b style="color:#1e293b;">{vri_status_s}</b></span>
      </div>
      <div style="{_row}">
        <span style="{_lbl}">PVO:&nbsp;<b style="color:{pvo_color};">{pvo:+.2f}</b></span>
        <span style="{_lbl}">VRI:&nbsp;<b style="color:{vri_color};">{vri:.1f}</b></span>
        <span style="{_lbl}">Slope:&nbsp;<b style="color:{slope_color};">{slope:+.3f}%</b></span>
        <span style="{_lbl}">Slope Z:&nbsp;<b style="color:#1a56db;">{slope_z:+.2f}</b></span>
      </div>
      <div style="{_row}">
        <span style="{_lbl}">VRI波動率:&nbsp;<b style="color:{vri_ratio_color};">{vri_ratio:.0%}</b>
            <small style="color:#94a3b8;">（&gt;60%持續性強）</small></span>
        <span style="{_lbl}">PVO波動率:&nbsp;<b style="color:{pvo_ratio_color};">{pvo_ratio:.0%}</b>
            <small style="color:#94a3b8;">（&gt;60%持續性強）</small></span>
      </div>
      <div style="{_row}">
        <span style="{_lbl}">💰 EV期望值:&nbsp;<b style="color:{ev_color};font-size:0.9rem;">{ev_str}</b></span>
        <span style="{_lbl}">📊 T值:&nbsp;<b style="color:{t_color};">{t_str}</b>
          <span style="color:{t_color};font-size:0.75rem;margin-left:2px;">{t_sig}</span></span>
      </div>
      <div style="{_row}">
        <span style="{_lbl}">🎯 上漲10%機率:&nbsp;<b style="color:{prob_color};font-size:0.9rem;">{prob_val if prob_val else 'N/A'}%</b></span>
        <span style="{_lbl}">停利:&nbsp;<b style="color:#059669;">{upside.get('take_profit','N/A')}</b>
          <small style="color:#94a3b8;">(+{upside.get('tp_pct','N/A')}%)</small></span>
        <span style="{_lbl}">停損:&nbsp;<b style="color:#dc2626;">{upside.get('stop_loss','N/A')}</b>
          <small style="color:#94a3b8;">(-{upside.get('stop_loss_pct','N/A')}%)</small></span>
        <span style="{_lbl}">風報比:&nbsp;<b style="color:{rr_color};">{rr_val if rr_val else 'N/A'}x</b></span>
      </div>
      <div style="{_row}">
        <span style="{_lbl}">S1:&nbsp;{s1_pass}</span>
        <span style="{_lbl}">路徑:&nbsp;{path}</span>
        <span style="{_lbl}">健康:&nbsp;{health_icon}</span>
        {f'<span style="{_lbl}">🏦 {trust_label}</span>' if trust_label else ""}
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
        low=df['Low'], close=df['Close'],
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
        target = st.date_input(
            "資料截止日",
            value=datetime.strptime(st.session_state.target_date, "%Y-%m-%d"),
            max_value=datetime.today()
        )
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
                api_key = st.text_input(
                    "Gemini API Key（覆蓋用）",
                    type="password",
                    placeholder="留空則使用環境變數金鑰"
                )
                st.session_state.gemini_api_key = api_key if api_key else _ENV_GEMINI_KEY
            else:
                st.session_state.gemini_api_key = _ENV_GEMINI_KEY
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                placeholder="AIza..."
            )
            st.session_state.gemini_api_key = api_key
            if api_key:
                st.success("✅ API Key 已設定")
            else:
                st.caption("未設定 → 使用規則引擎模式")

        st.markdown("---")
        st.caption("© 2026 資源法 AI 戰情室 v2.3")


# ===========================================================================
# 主程式
# ===========================================================================
def main():
    inject_color_fix()
    render_sidebar()

    # ── 標題列 ──
    col_title, col_scan = st.columns([4, 1])
    with col_title:
        st.markdown("# ⚡ 資源法 AI 戰情室")
        st.markdown(
            f"<small style='color:#64748b'>分析日期: {st.session_state.target_date} | "
            f"{'台股模式 🇹🇼' if st.session_state.active_market == 'TW' else '美股模式 🇺🇸'}</small>",
            unsafe_allow_html=True
        )
    with col_scan:
        scan_btn = st.button("🔄 執行全盤掃描", use_container_width=True, type="primary")

    render_status_bar()

    sentiment = (
        st.session_state.market_sentiment_tw
        if st.session_state.active_market == "TW"
        else st.session_state.market_sentiment_us
    )
    render_market_bar(sentiment, st.session_state.active_market)

    # ── 全盤掃描 ──
    if scan_btn:
        market = st.session_state.active_market
        wl = (st.session_state.tw_watchlist if market == "TW" else st.session_state.us_watchlist)
        with st.spinner("⚙️ 正在擷取數據並計算指標..."):
            pb = st.progress(0, text="初始化...")
            run_scan(wl, st.session_state.target_date, market, progress_bar=pb)
            pb.empty()
        st.rerun()

    # ── 全盤 Gemini 分析 ──
    ai_col1, ai_col2 = st.columns([1, 4])
    with ai_col1:
        ai_btn = st.button("🤖 Gemini 全盤分析", use_container_width=True)
    with ai_col2:
        if st.session_state.ai_summary:
            import html as _html
            _safe = _html.escape(st.session_state.ai_summary).replace('\n', '<br>')
            st.markdown(f'<div class="ai-summary">{_safe}</div>', unsafe_allow_html=True)

    if ai_btn:
        if not st.session_state.scan_results:
            st.warning("請先執行掃描")
        else:
            market = st.session_state.active_market
            sent   = (st.session_state.market_sentiment_tw if market == "TW"
                      else st.session_state.market_sentiment_us)
            prompt = build_gemini_prompt(st.session_state.scan_results, sent, market)
            with st.spinner("🤖 Gemini 全盤深度分析中（請稍候）..."):
                summary = call_gemini(prompt, st.session_state.gemini_api_key)
            st.session_state.ai_summary = summary
            st.rerun()

    # ════════════════════════════════════════════════════════════════
    # ★ 個股 AI 深度分析區塊
    # ════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #059669;
                border-radius:10px;padding:10px 16px;margin:12px 0 6px 0;">
        <b style="color:#059669;font-size:0.95rem;">🔍 個股 AI 深度分析</b>
        <span style="color:#64748b;font-size:0.82rem;margin-left:10px;">
            輸入觀察名單內的代號 → Gemini 個股專項報告（含停利停損建議）
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_sym_in, col_sym_btn = st.columns([3, 1])
    with col_sym_in:
        single_sym = st.text_input(
            "個股代號",
            placeholder="例如: 2330 / NVDA（需先執行全盤掃描）",
            label_visibility="collapsed",
            key="single_sym_input"
        )
    with col_sym_btn:
        single_btn = st.button("🔍 分析此股", use_container_width=True, key="single_analyze_btn")

    # 顯示上次個股分析結果（避免 rerun 消失）
    if st.session_state.single_stock_result and st.session_state.single_stock_sym:
        sym_disp   = st.session_state.single_stock_sym
        up_cached  = st.session_state.single_stock_upside
        import html as _hc
        _safe_c = _hc.escape(st.session_state.single_stock_result).replace('\n', '<br>')
        prob_c   = up_cached.get('prob', 'N/A')
        tp_c     = up_cached.get('take_profit', 'N/A')
        sl_c     = up_cached.get('stop_loss', 'N/A')
        tp_pct_c = up_cached.get('tp_pct', 'N/A')
        sl_pct_c = up_cached.get('stop_loss_pct', 'N/A')
        rr_c     = up_cached.get('rr_ratio', 'N/A')
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0fdf4,#f8faff);
                    border:1px solid #86efac;border-left:4px solid #059669;
                    border-radius:10px;padding:16px 20px;margin:8px 0 16px 0;
                    font-size:0.98rem;line-height:1.85;color:#1e293b;">
            <div style="font-weight:700;color:#059669;font-size:1.05rem;margin-bottom:10px;
                        padding-bottom:8px;border-bottom:1px solid #bbf7d0;">
                🔍 {sym_disp} 個股深度分析報告
                <span style="font-size:0.78rem;font-weight:400;color:#64748b;margin-left:12px;">
                    上漲10%機率: <b style="color:#059669">{prob_c}%</b>
                    ｜ 停利: <b style="color:#059669">{tp_c}</b>(+{tp_pct_c}%)
                    ｜ 停損: <b style="color:#dc2626">{sl_c}</b>(-{sl_pct_c}%)
                    ｜ 風報比: <b>{rr_c}x</b>
                </span>
            </div>
            {_safe_c}
        </div>
        """, unsafe_allow_html=True)

    # 觸發個股分析
    if single_btn and single_sym:
        sym_query   = single_sym.strip().upper()
        results_now = st.session_state.scan_results

        # 模糊匹配（含 .TW / .TWO 後綴）
        matched_key = None
        for k in results_now:
            k_clean = k.replace(".TWO", "").replace(".TW", "").strip().upper()
            if k_clean == sym_query or k.upper() == sym_query:
                matched_key = k
                break

        if not matched_key or results_now.get(matched_key, {}).get("error"):
            st.warning(
                f"⚠️ 沒有資料：{sym_query}\n"
                f"請確認：①已執行全盤掃描 ②該代號在觀察名單中"
            )
        else:
            res_s    = results_now[matched_key]
            s2_s     = res_s.get("stage2", {})
            df_s     = res_s.get("indicator_df")
            dec_s    = res_s.get("decision", {})
            upside_s = calc_upside_10pct_prob(dec_s, s2_s, df_s)

            single_prompt = build_single_stock_prompt(
                matched_key, res_s,
                st.session_state.market_sentiment_tw,
                st.session_state.market_sentiment_us
            )

            with st.spinner(f"🤖 正在深度分析 {matched_key}，請稍候..."):
                single_result = call_gemini(single_prompt, st.session_state.gemini_api_key)

            # 存入 session state 供顯示
            st.session_state.single_stock_result = single_result
            st.session_state.single_stock_sym    = matched_key
            st.session_state.single_stock_upside = upside_s
            st.rerun()

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # 四大 Tab
    # ════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 視覺化圖表",
        "📋 數據與排名",
        "🎯 決策戰情室",
        "🔬 數據健康度"
    ])

    results = st.session_state.scan_results
    market  = st.session_state.active_market

    # ────────────────────────────────────────────────────────────────
    # Tab 1: 視覺化圖表
    # ────────────────────────────────────────────────────────────────
    with tab1:
        if not results:
            st.info("👈 請先在左側設定觀察名單並執行「全盤掃描」")
        else:
            sym_list = [s for s, r in results.items() if not r.get("error")]
            if sym_list:
                selected = st.selectbox("選擇個股查看 K 線圖", sym_list, index=0)
                if selected and selected in results:
                    render_kline_chart(selected, results[selected])
                    res  = results[selected]
                    dec  = res.get("decision", {})
                    s2   = res.get("stage2", {})
                    df_c = res.get("indicator_df")
                    up   = calc_upside_10pct_prob(dec, s2, df_c)

                    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                    col1.metric("現價",    f"{dec.get('close',0):,.2f}")
                    col2.metric("PVO",     f"{dec.get('pvo',0):+.2f}",
                                delta="↑" if dec.get("pvo", 0) > 0 else "↓")
                    col3.metric("VRI",     f"{dec.get('vri',0):.1f}")
                    col4.metric("Slope Z", f"{dec.get('slope_z',0):+.2f}")
                    ev_v = s2.get('ev', None)
                    col5.metric("EV 期望值", f"{ev_v:+.2f}%" if isinstance(ev_v, (int, float)) else "N/A")
                    t_v = s2.get('t_stat', None)
                    col6.metric("T 值", f"{t_v:.2f}" if t_v is not None else "N/A",
                                delta="顯著✅" if (t_v is not None and abs(t_v) >= 2.0) else None)
                    col7.metric("上漲10%機率", f"{up.get('prob','N/A')}%")
                    col8.metric("風報比", f"{up.get('rr_ratio','N/A')}x")

    # ────────────────────────────────────────────────────────────────
    # Tab 2: 數據與排名
    # ────────────────────────────────────────────────────────────────
    with tab2:
        if not results:
            st.info("請先執行掃描")
        else:
            _pt = "display:inline-flex;align-items:center;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;margin:3px 2px;"
            st.markdown(f"""
            <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:12px 18px;margin-bottom:16px;">
            <b style="color:#0369a1;">📊 高勝率型態識別</b>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">
                <span style="{_pt}background:#d1fae5;color:#065f46;border:1px solid #34d399;">📈 A: 資金流入＋情緒整理＋強力買進 &nbsp;勝率10%: 52.6% / 20%: 37.6%</span>
                <span style="{_pt}background:#fee2e2;color:#991b1b;border:1px solid #f87171;">🔥 B: 主力點火＋擁擠過熱＋強力買進 &nbsp;勝率10%: 52.4%</span>
                <span style="{_pt}background:#fef3c7;color:#92400e;border:1px solid #fbbf24;">🌡️ C: VRI擁擠過熱＋強力買進 &nbsp;勝率10%: 59.0% / 20%: 42.6%</span>
            </div>
            <div style="margin-top:8px;font-size:0.8rem;color:#0369a1;">
                ⭐ 最終候選：命中型態 + PVO波動率&gt;60% + VRI波動率&gt;60%
                ｜ 🎯 上漲10%機率 = 多因子加權模型估算（非保證值）
            </div>
            </div>
            """, unsafe_allow_html=True)

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                show_all = st.checkbox("顯示全部（含未通過）", value=False)
            with col_f2:
                sort_by = st.selectbox(
                    "排序依據",
                    ["最佳勝率10%", "上漲10%機率", "Slope Z", "VRI", "PVO", "EV期望值", "風報比", "Score"],
                    index=0
                )
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
                df_ind_tab = res.get("indicator_df")
                pat   = classify_pattern(dec)

                if not show_all and not s1.get("pass"):
                    continue
                if dec.get("vri", 0) < min_vri:
                    continue

                pat_codes    = " + ".join([p["code"] for p in pat["patterns"]]) if pat["patterns"] else "—"
                pat_desc_tab = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "—"

                ev_raw_tab = s2.get("ev", None)
                t_stat_tab = s2.get("t_stat", None)
                is_fin     = is_final_candidate(dec, df_ind_tab)

                # ★ 計算上漲機率與停利停損
                upside = calc_upside_10pct_prob(dec, s2, df_ind_tab)

                table_rows.append({
                    "代號":        sym,
                    "市場":        "🇹🇼" if res.get("market") == "TW" else "🇺🇸",
                    "現價":        dec.get("close", 0),
                    "方向":        dec.get("direction", "---"),
                    "操作":        dec.get("action", "---"),
                    "最終候選":    "⭐" if is_fin else "—",
                    "高勝率型態":  pat_codes,
                    "最佳勝率10%": pat["best_win10"] if pat["best_win10"] > 0 else None,
                    "型態說明":    pat_desc_tab,
                    "PVO":         dec.get("pvo", 0),
                    "VRI":         dec.get("vri", 0),
                    "VRI波動率":   calc_vri_ratio(df_ind_tab),
                    "PVO波動率":   calc_pvo_ratio(df_ind_tab),
                    "Slope Z":     dec.get("slope_z", 0),
                    "Score":       dec.get("score", 0),
                    "EV期望值%":   ev_raw_tab,
                    "T值":         t_stat_tab,
                    # ★ 新增欄位
                    "上漲10%機率": upside.get("prob"),
                    "停利價":      upside.get("take_profit"),
                    "停損價":      upside.get("stop_loss"),
                    "停利%":       upside.get("tp_pct"),
                    "停損%":       upside.get("stop_loss_pct"),
                    "風報比":      upside.get("rr_ratio"),
                    # ──────────
                    "路徑":        translate_path(str(s2.get("path", "N/A"))),
                    "投信10日":    trust.get("trust_net_10d", None),
                    "S1":          "✅" if s1.get("pass") else "❌",
                    "健康":        "✅" if h.get("pass") else "⚠️",
                    "PVO狀態":     dec.get("pvo_status", ""),
                    "VRI狀態":     dec.get("vri_status", ""),
                    "日期":        dec.get("date", ""),
                })

            if table_rows:
                sort_col_map = {
                    "最佳勝率10%": "最佳勝率10%",
                    "上漲10%機率": "上漲10%機率",
                    "Slope Z":    "Slope Z",
                    "VRI":        "VRI",
                    "PVO":        "PVO",
                    "EV期望值":   "EV期望值%",
                    "風報比":     "風報比",
                    "Score":      "Score",
                }
                df_table  = pd.DataFrame(table_rows)
                sort_col  = sort_col_map.get(sort_by, "最佳勝率10%")
                df_table["_has_pattern"] = df_table["高勝率型態"].ne("—").astype(int)
                secondary = sort_col if sort_col in df_table.columns else "Slope Z"
                df_table = df_table.sort_values(
                    ["_has_pattern", secondary],
                    ascending=[False, False],
                    na_position='last'
                ).drop(columns=["_has_pattern"])

                final_count = df_table["最終候選"].eq("⭐").sum()
                st.markdown(
                    f"**共 {len(df_table)} 檔符合條件**　｜　"
                    f"🏆 命中高勝率型態: **{df_table['高勝率型態'].ne('—').sum()}** 檔　｜　"
                    f"⭐ 最終候選: **{final_count}** 檔"
                )
                st.dataframe(
                    df_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "現價":        st.column_config.NumberColumn(format="%.2f"),
                        "PVO":         st.column_config.NumberColumn(format="%.2f"),
                        "VRI":         st.column_config.NumberColumn(format="%.1f"),
                        "VRI波動率":   st.column_config.NumberColumn(
                            format="%.0%",
                            help="近20日VRI>40天數/20，>60%代表持續性強"
                        ),
                        "PVO波動率":   st.column_config.NumberColumn(
                            format="%.0%",
                            help="近20日PVO>0天數/20，>60%代表持續性強"
                        ),
                        "Slope Z":     st.column_config.NumberColumn(format="%.2f"),
                        "EV期望值%":   st.column_config.NumberColumn(
                            format="%.2f",
                            help="歷史回測期望報酬率(%)"
                        ),
                        "T值":         st.column_config.NumberColumn(
                            format="%.2f",
                            help="統計顯著性，≥2.0具顯著性"
                        ),
                        "最佳勝率10%": st.column_config.NumberColumn(
                            format="%.1f%%",
                            help="命中型態的最高10日勝率"
                        ),
                        # ★ 新增欄位格式設定
                        "上漲10%機率": st.column_config.NumberColumn(
                            format="%.1f%%",
                            help="多因子加權模型估算（型態勝率+SlopeZ+PVO+VRI+T值+EV），非保證值"
                        ),
                        "停利價":      st.column_config.NumberColumn(
                            format="%.2f",
                            help="基於EV×1.5或最低10%，確保風報比≥1.5x"
                        ),
                        "停損價":      st.column_config.NumberColumn(
                            format="%.2f",
                            help="ATR×VRI動態倍數（VRI越高越寬），最多12%"
                        ),
                        "停利%":       st.column_config.NumberColumn(format="%.1f%%"),
                        "停損%":       st.column_config.NumberColumn(format="%.1f%%"),
                        "風報比":      st.column_config.NumberColumn(
                            format="%.2fx",
                            help="停利幅度÷停損幅度，≥1.5合格，≥2.0優質"
                        ),
                        # ──────────
                        "投信10日":    st.column_config.NumberColumn(
                            label="投信10日(張)",
                            format="%+,.0f",
                            help="近10個交易日投信累積買賣超（正=買超/負=賣超）"
                        ),
                        "高勝率型態":  st.column_config.TextColumn(
                            help="A=資金流入+整理+強買 | B=點火+過熱+強買 | C=VRI過熱+強買"
                        ),
                        "型態說明":    st.column_config.TextColumn(help="型態操作特性說明"),
                        "最終候選":    st.column_config.TextColumn(
                            help="⭐=命中型態+PVO波動率>60%+VRI波動率>60%"
                        ),
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

    # ────────────────────────────────────────────────────────────────
    # Tab 3: 決策戰情室
    # ────────────────────────────────────────────────────────────────
    with tab3:
        if not results:
            st.info("請先執行掃描")
        else:
            # ── 最終決策候選 ──
            final_picks = [
                (s, r) for s, r in results.items()
                if not r.get("error")
                and is_final_candidate(r.get("decision", {}), r.get("indicator_df"))
            ]
            final_picks.sort(
                key=lambda x: classify_pattern(x[1].get("decision", {}))["best_win10"],
                reverse=True
            )

            st.markdown("""
            <div class="final-decision-box">
                <b style="color:#1a56db;font-size:1.05rem;">⭐ 最終決策候選</b>
                <span style="color:#64748b;font-size:0.85rem;margin-left:12px;">
                    條件：命中三大高勝率型態 + PVO波動率 &gt; 60% + VRI波動率 &gt; 60%
                </span>
            </div>
            """, unsafe_allow_html=True)

            if final_picks:
                for sym, res in final_picks:
                    dec    = res.get("decision", {})
                    s2     = res.get("stage2", {})
                    df_ind = res.get("indicator_df")
                    pat    = classify_pattern(dec)
                    upside = calc_upside_10pct_prob(dec, s2, df_ind)

                    pat_labels = " ｜ ".join([p["label"] for p in pat["patterns"]])
                    win_str    = f"最高勝率10%: {pat['best_win10']:.1f}%"
                    vri_ratio  = calc_vri_ratio(df_ind)
                    pvo_ratio  = calc_pvo_ratio(df_ind)
                    path_zh    = translate_path(str(s2.get("path","N/A")))
                    pat_descs  = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "—"

                    ev_raw    = s2.get("ev", None)
                    t_raw     = s2.get("t_stat", None)
                    ev_display= f"{ev_raw:+.2f}%" if isinstance(ev_raw, (int, float)) else "N/A"
                    t_display = f"{t_raw:.2f}" if t_raw is not None else "N/A"
                    t_sig_str = " ✅顯著" if (t_raw is not None and abs(t_raw) >= 2.0) else ""

                    ev_c  = "#059669" if (isinstance(ev_raw, (int, float)) and ev_raw > 3) else "#d97706"
                    t_c   = "#059669" if (t_raw is not None and abs(t_raw) >= 2.0) else "#d97706"
                    prob_c_val = upside.get('prob', 0) or 0
                    p_c   = "#059669" if prob_c_val >= 55 else ("#d97706" if prob_c_val >= 45 else "#dc2626")
                    rr_v  = upside.get('rr_ratio', 0) or 0
                    rr_c2 = "#059669" if rr_v >= 2.0 else ("#d97706" if rr_v >= 1.5 else "#dc2626")
                    mkt_tag = "🇹🇼" if res.get("market") == "TW" else "🇺🇸"

                    st.markdown(f"""
                    <div style="background:#eff6ff;border:1px solid #93c5fd;border-left:4px solid #1a56db;
                         border-radius:8px;padding:14px 18px;margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <b style="color:#1e40af;font-size:1rem;">{mkt_tag} {sym}</b>
                            <span style="padding:3px 10px;border-radius:6px;font-size:0.78rem;font-weight:700;
                                         background:#fef3c7;color:#92400e;border:1px solid #fbbf24;">{win_str}</span>
                        </div>
                        <div style="font-size:0.82rem;color:#475569;margin-top:4px;">{pat_labels}</div>
                        <div style="font-size:0.79rem;color:#7c3aed;font-style:italic;margin:2px 0;">📝 {pat_descs}</div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:0.83rem;color:#475569;">
                            <span>Slope Z: <b style="color:#1a56db">{dec.get('slope_z',0):+.2f}</b></span>
                            <span>VRI: <b>{dec.get('vri',0):.1f}</b></span>
                            <span>PVO: <b>{dec.get('pvo',0):+.2f}</b></span>
                            <span>💰 EV: <b style="color:{ev_c}">{ev_display}</b></span>
                            <span>📊 T值: <b style="color:{t_c}">{t_display}{t_sig_str}</b></span>
                        </div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;font-size:0.83rem;color:#475569;">
                            <span>VRI波動率: <b style="color:{'#059669' if vri_ratio > 0.6 else '#d97706'}">{vri_ratio:.0%}</b></span>
                            <span>PVO波動率: <b style="color:{'#059669' if pvo_ratio > 0.6 else '#d97706'}">{pvo_ratio:.0%}</b></span>
                            <span>路徑: {path_zh}</span>
                        </div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;font-size:0.83rem;
                                    background:#f0fdf4;border-radius:6px;padding:6px 10px;">
                            <span>🎯 上漲10%機率: <b style="color:{p_c};font-size:0.95rem;">{upside.get('prob','N/A')}%</b></span>
                            <span>停利: <b style="color:#059669">{upside.get('take_profit','N/A')}</b>
                              (+{upside.get('tp_pct','N/A')}%)</span>
                            <span>停損: <b style="color:#dc2626">{upside.get('stop_loss','N/A')}</b>
                              (-{upside.get('stop_loss_pct','N/A')}%)</span>
                            <span>風報比: <b style="color:{rr_c2}">{upside.get('rr_ratio','N/A')}x</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🔍 目前無標的同時符合三大型態 + PVO波動率>60% + VRI波動率>60%，請查看下方做多訊號")

            st.markdown("---")

            # ── 做多訊號 ──
            bull_stocks = [
                (s, r) for s, r in results.items()
                if not r.get("error")
                and r.get("decision", {}).get("direction") == "做多"
                and r.get("stage1", {}).get("pass")
            ]
            bull_stocks.sort(
                key=lambda x: (
                    int(is_final_candidate(x[1].get("decision", {}), x[1].get("indicator_df"))),
                    classify_pattern(x[1].get("decision", {}))["best_win10"],
                    x[1].get("decision", {}).get("slope_z", 0)
                ),
                reverse=True
            )

            key_count   = sum(1 for s, r in bull_stocks if classify_pattern(r.get("decision", {}))["is_key_pattern"])
            final_count = sum(1 for s, r in bull_stocks if is_final_candidate(r.get("decision", {}), r.get("indicator_df")))

            st.markdown(f"""
            <div class="decision-header">
                <b>🟢 做多訊號</b>
                <span style="color:#64748b;font-size:0.85rem;margin-left:8px;">
                    共 {len(bull_stocks)} 檔通過 Stage1 篩選
                    ｜ 命中高勝率型態: <b style="color:#059669">{key_count}</b> 檔
                    ｜ 最終候選: <b style="color:#1a56db">{final_count}</b> 檔
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

            # ── 做空/觀望 ──
            bear_stocks = [
                (s, r) for s, r in results.items()
                if not r.get("error")
                and r.get("decision", {}).get("direction") == "做空"
            ]
            neutral_stocks = [
                (s, r) for s, r in results.items()
                if not r.get("error")
                and r.get("decision", {}).get("direction") == "觀望"
            ]

            if bear_stocks:
                st.markdown(f"### 🔴 做空/警示 ({len(bear_stocks)} 檔)")
                for sym, res in bear_stocks[:5]:
                    render_stock_card(sym, res)

            st.markdown(f"### ⚪ 觀望 ({len(neutral_stocks)} 檔)")
            with st.expander("展開觀望清單"):
                for sym, res in neutral_stocks[:20]:
                    dec = res.get("decision", {})
                    pat = classify_pattern(dec)
                    pat_tag  = f" 型態{'/ '.join([p['code'] for p in pat['patterns']])}" if pat["patterns"] else ""
                    pvo_st   = dec.get("pvo_status","")
                    st.markdown(
                        f"**{sym}**{pat_tag} — {pvo_st} ｜ "
                        f"Slope Z: `{dec.get('slope_z',0):+.2f}` ｜ "
                        f"VRI: `{dec.get('vri',0):.1f}` ｜ "
                        f"PVO: `{dec.get('pvo',0):+.2f}`"
                    )

            if st.session_state.selected_stock in results:
                sym = st.session_state.selected_stock
                st.markdown(f"---\n### 📊 {sym} 詳細分析")
                render_kline_chart(sym, results[sym])

    # ────────────────────────────────────────────────────────────────
    # Tab 4: 數據健康度
    # ────────────────────────────────────────────────────────────────
    with tab4:
        render_health_panel()

        st.markdown("---")
        st.markdown("### ⚠️ 數據清洗日誌")
        if st.session_state.all_warnings:
            for w in st.session_state.all_warnings:
                st.warning(w)
        else:
            st.success("✅ 無異常數據警告")


if __name__ == "__main__":
    main()
四 Tab 架構（全部保留）
  - 新增 Tab5「OOS績效報告」：版本對比、OOS標準統計框、出場分析
  - 新增 Tab6「訊號分類統計」：三合一/二合一/單一/基準 完整統計表
  - 上漲10%機率模型以基準44.5%為錨點重新校準
  - PVO+VRI+ADX+RSI四維評分（取代純ADX+RSI）

執行：
  pip install streamlit yfinance pandas pandas_ta numpy scipy plotly
  streamlit run 0410-V1.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats
import json
import os
import time
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ===========================================================================
# 環境變數保密機制
# ===========================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

_ENV_GEMINI_KEY    = _get_secret("GEMINI_API_KEY")
_ENV_FINMIND_TOKEN = _get_secret("FINMIND_TOKEN")

# ===========================================================================
# 懶加載重型依賴（避免 import 失敗中斷啟動）
# ===========================================================================
def _try_import_engine():
    try:
        from engine_21 import (fetch_stock_data, stage1_energy_filter,
                                stage2_path_filter, get_decision,
                                get_market_sentiment, resolve_symbol, _INST_CACHE)
        return fetch_stock_data, stage1_energy_filter, stage2_path_filter, \
               get_decision, get_market_sentiment, resolve_symbol, _INST_CACHE
    except ImportError:
        return None

_ENGINE = _try_import_engine()

# ===========================================================================
# 頁面設定
# ===========================================================================
st.set_page_config(
    page_title="資源法 AI 戰情室 V1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600;700&display=swap');
:root{--bg:#f4f6fb;--panel:#fff;--accent:#1a56db;--green:#059669;--red:#dc2626;
      --amber:#d97706;--text:#1e293b;--dim:#64748b;--border:#e2e8f0;--radius:10px}
html,body,.stApp{background:var(--bg)!important;color:var(--text)!important;
  font-family:'Noto Sans TC',sans-serif}
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid var(--border)}
h1,h2,h3{color:var(--accent)!important}
.stButton>button{background:linear-gradient(135deg,#1a56db,#1d4ed8)!important;
  color:#fff!important;border:none!important;border-radius:var(--radius)!important;
  font-weight:600!important}
.kpi-card{background:#fff;border:.5px solid var(--border);border-radius:var(--radius);
  padding:.75rem 1rem;text-align:center;margin-bottom:.5rem}
.kpi-val{font-size:1.3rem;font-weight:700;margin-top:4px}
.kpi-lbl{font-size:.75rem;color:var(--dim)}
.sig-row-head{background:#f1f5f9;font-weight:600;font-size:.8rem}
.sig-row{font-size:.8rem;border-bottom:1px solid var(--border)}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:.72rem;font-weight:700}
.bdg-a{background:#d1fae5;color:#065f46}.bdg-b{background:#fee2e2;color:#991b1b}
.bdg-c{background:#fef3c7;color:#92400e}.bdg-base{background:#e6f1fb;color:#185fa5}
.ai-box{background:linear-gradient(135deg,#eff6ff,#f8faff);border:1px solid #bfdbfe;
  border-left:4px solid var(--accent);border-radius:var(--radius);padding:1rem 1.25rem;
  font-size:1rem;line-height:1.8}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# 常數
# ===========================================================================
DEFAULT_TW_WATCHLIST = [
    "3030","3706","8096","2313","4958","2330","2317","2454","2308","2382",
    "2303","3711","2412","2357","3231","2379","3008","2395","3045","2327",
    "2408","2377","6669","2301","3034","2345","2474","3037","4938","3443",
    "2353","2324","2603","2609","1513","3293","3680","3529","3131","5274",
    "6223","6805","3017","3324","6515","3661","3583","6139","3035","1560",
    "8299","3558","6187","3406","3217","6176","6415","6206","8069","3264",
    "5269","2360","6271","3189","6438","8358","6231","2449","8016","6679",
    "3374","3014","3211","6213","2404","2480","3596","6202","5443","5347",
    "5483","6147","8046","2368","2383","6269","5469","5351","4909","8050",
    "6153","6505","1802","3708","8213","1325","2344","6239","3260","4967",
    "6414","2337","3551","2436","2375","2492","2456","3229","6173","3533",
]
DEFAULT_US_WATCHLIST = [
    "NVDA","TSLA","AAPL","MSFT","GOOGL","AMZN","META","AMD","AVGO","QCOM",
    "ASML","TSM","NFLX","CRM","ADBE","INTC","MU","LRCX","KLAC","AMAT",
    "ARM","PLTR","MSTR","MRVL","CRWD","PANW","FTNT","DDOG","ZS","SNOW",
]
BENCHMARK_TW   = "0050.TW"
BENCHMARK_US   = "SPY"
LOOKBACK_DAYS  = 180
ALPHA_SEEDS    = "alpha_seeds.json"

# 訊號分類已知基準值（來自歷史統計文件）
SIGNAL_BASELINES = {
    '基準-強勢': {'win10':44.5,'days10':8.7,'win20':27.4,'days20':11.6,
                  'loss10':26.9,'ld10':9.3,'stars':'★★★','n_ref':3503},
    '基準-持有': {'win10':39.3,'days10':9.9,'win20':20.2,'days20':12.5,
                  'loss10':23.5,'ld10':10.7,'stars':'★★★','n_ref':8528},
}
SIGNAL_KNOWN_PATTERNS = {
    '單一(C)':    {'win10':59.0,'win20':42.6,'stars':'★★★★'},
    '單一(A)':    {'win10':52.6,'win20':37.6,'stars':'★★★'},
    '單一(B)':    {'win10':52.4,'win20':0,   'stars':'★★★'},
    '二合一(AC)': {'win10':0,   'win20':0,   'stars':'★★★★'},
    '二合一(AB)': {'win10':0,   'win20':0,   'stars':'★★★★'},
    '二合一(BC)': {'win10':0,   'win20':0,   'stars':'★★★★'},
    '三合一':     {'win10':0,   'win20':0,   'stars':'★★★★★'},
}

# 版本對比歷史數據
VERSION_BENCHMARKS = [
    ("V10",    121, 52.1, 2.75,  54.9, -12.60, 2.95, 2.368),
    ("V10.2",   92, 57.6, 5.90,  89.0, -11.10, 5.24, 3.600),
    ("V10.3",   95, 52.6, 4.85,  75.5,  -8.92, 4.77, 3.320),
    ("V11.3",  112, 57.1, 5.29,  96.9,  -6.58, 5.36, 4.032),
    ("V12",     91, 56.0, 5.06,  75.2,  -8.62, 4.80, 3.297),
    ("V12.1",   69, 52.2, 4.55,  50.8, -10.19, 4.12, 2.438),
]

# ===========================================================================
# 指標計算輔助（無 engine_21 時的備用）
# ===========================================================================
def _calc_pvo_series(df, fast=5, slow=20):
    if 'Volume' not in df.columns:
        return pd.Series(0.0, index=df.index)
    vf = df['Volume'].ewm(span=fast).mean()
    vs = df['Volume'].ewm(span=slow).mean()
    return (vf - vs) / (vs + 1e-9) * 100

def _calc_vri_series(df, window=20):
    try:
        import pandas_ta as ta
        rsi = ta.rsi(df['Close'], length=14).clip(0, 100)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        atr_mean = atr.rolling(window).mean()
        atr_norm = (atr / (atr_mean + 1e-9) * 50).clip(0, 100)
        return (rsi * 0.6 + atr_norm * 0.4).clip(0, 100)
    except Exception:
        return pd.Series(50.0, index=df.index)

# ===========================================================================
# 型態識別
# ===========================================================================
def classify_pattern(dec: dict) -> dict:
    pvo    = dec.get("pvo", 0)
    vri    = dec.get("vri", 0)
    action = dec.get("action", "")
    slope_z= dec.get("slope_z", 0)

    is_strong_buy = ("強力" in action or slope_z > 1.5)
    is_fire       = pvo >= 10.0
    is_money_in   = 0.0 <= pvo < 10.0
    is_hot        = vri > 75.0
    is_cool       = vri < 40.0

    patterns = []
    if is_strong_buy and is_money_in and is_cool:
        patterns.append({'code':'A','label':'📈 資金流入+情緒整理+強力買進',
                         'win10':52.6,'win20':37.6,'css':'bdg-a',
                         'desc':'穩+爆發兼具｜主升段最佳'})
    if is_strong_buy and is_fire and is_hot:
        patterns.append({'code':'B','label':'🔥 主力點火+擁擠過熱+強力買進',
                         'win10':52.4,'win20':0,'css':'bdg-b',
                         'desc':'超短線爆發最強（風險偏高）'})
    if is_strong_buy and is_hot:
        codes = [p['code'] for p in patterns]
        if 'C' not in codes:
            patterns.append({'code':'C','label':'🌡️ VRI擁擠過熱+強力買進',
                             'win10':59.0,'win20':42.6,'css':'bdg-c',
                             'desc':'最高勝率組合'})

    codes = [p['code'] for p in patterns]
    if len(codes) >= 3 or set(codes) == {'A','B','C'}:
        sig_label = '三合一'
    elif len(codes) == 2:
        sig_label = '二合一(' + ''.join(sorted(codes)) + ')'
    elif len(codes) == 1:
        sig_label = f'單一({codes[0]})'
    elif slope_z > 1.5:
        sig_label = '基準-強勢'
    else:
        sig_label = '基準-持有'

    return {
        'patterns':     patterns,
        'sig_label':    sig_label,
        'is_key':       len(patterns) > 0,
        'best_win10':   max([p['win10'] for p in patterns], default=0),
    }

def calc_pvo_ratio(df) -> float:
    if df is None or df.empty or 'PVO' not in df.columns:
        return 0.0
    return round((df['PVO'].tail(20) > 0).sum() / 20, 2)

def calc_vri_ratio(df) -> float:
    if df is None or df.empty or 'VRI' not in df.columns:
        return 0.0
    return round((df['VRI'].tail(20) > 40).sum() / 20, 2)

def is_final_candidate(dec, df_ind) -> bool:
    pat = classify_pattern(dec)
    if not pat['is_key']:
        return False
    return calc_pvo_ratio(df_ind) > 0.6 and calc_vri_ratio(df_ind) > 0.6

# ===========================================================================
# 上漲10%機率（以基準44.5%為錨點校準）
# ===========================================================================
def calc_upside_10pct_prob(dec: dict, s2: dict, df_ind) -> dict:
    close = dec.get("close", 0)
    if close <= 0:
        return {'prob':None,'stop_loss':None,'take_profit':None,
                'stop_loss_pct':None,'tp_pct':None,'rr_ratio':None}

    pat     = classify_pattern(dec)
    pvo     = dec.get("pvo", 0)
    vri     = dec.get("vri", 0)
    slope_z = dec.get("slope_z", 0)
    ev_raw  = s2.get("ev", None) if s2 else None
    t_stat  = s2.get("t_stat", None) if s2 else None

    # ★ 以實際基準值44.5%為錨點（原版用0.30/0.38，現在校準）
    ANCHOR_STRONG = 0.445   # 基準-強勢 44.5%
    ANCHOR_HOLD   = 0.393   # 基準-持有 39.3%

    if pat['best_win10'] > 0:
        base_prob = pat['best_win10'] / 100.0
    else:
        base_prob = ANCHOR_STRONG if slope_z > 1.5 else ANCHOR_HOLD

    adj = 0.0
    adj += min(slope_z * 0.04, 0.10)
    if pvo > 10:         adj += 0.07
    elif pvo > 0:        adj += 0.02
    else:                adj -= 0.09
    if 40 <= vri <= 75:  adj += 0.04
    elif vri > 90:       adj -= 0.05
    if t_stat is not None and abs(t_stat) >= 2.0:
        adj += 0.04
    if isinstance(ev_raw, (int, float)):
        if ev_raw > 5:   adj += 0.04
        elif ev_raw > 3: adj += 0.02

    prob = max(0.05, min(0.92, base_prob + adj))

    # ATR動態停損
    if df_ind is not None and not df_ind.empty and 'High' in df_ind.columns:
        atr_pct = float(
            (df_ind['High'] - df_ind['Low']).tail(20).mean() / close)
    else:
        atr_pct = 0.02
    stop_mult     = 1.5 + (vri / 100.0)
    stop_loss_pct = min(atr_pct * stop_mult, 0.12)
    stop_loss_px  = round(close * (1 - stop_loss_pct), 2)

    if isinstance(ev_raw, (int, float)) and ev_raw > 0:
        tp_pct = min(ev_raw / 100.0 * 1.5, 0.20)
    else:
        tp_pct = 0.10
    tp_pct  = max(tp_pct, stop_loss_pct * 1.5)
    tp_px   = round(close * (1 + tp_pct), 2)
    rr      = round(tp_pct / stop_loss_pct, 2) if stop_loss_pct > 0 else None

    return {
        'prob':          round(prob * 100, 1),
        'stop_loss':     stop_loss_px,
        'take_profit':   tp_px,
        'stop_loss_pct': round(stop_loss_pct * 100, 1),
        'tp_pct':        round(tp_pct * 100, 1),
        'rr_ratio':      rr,
    }

# ===========================================================================
# 訊號分類統計計算（從 scan_results）
# ===========================================================================
def compute_signal_stats_from_scan(scan_results: dict) -> dict:
    """
    從 scan_results 中統計各訊號分類的樣本數與勝率代理指標。
    因 Streamlit 版無實際回測 trade_log，改用：
      - 每日收盤報酬（Ret5d）作為 10 日報酬代理
      - 各型態股票在掃描期間的平均 Slope Z 作為強度參考
    同時回傳各標的的訊號分類彙總供表格顯示。
    """
    groups = defaultdict(list)
    for sym, res in scan_results.items():
        if res.get('error'):
            continue
        dec    = res.get('decision', {})
        df_ind = res.get('indicator_df')
        pat    = classify_pattern(dec)
        label  = pat['sig_label']

        # 用近期 5 日報酬作為代理報酬（非真實回測值，僅供分布參考）
        if df_ind is not None and not df_ind.empty and 'Ret5d' in df_ind.columns:
            ret5 = float(df_ind['Ret5d'].iloc[-1])
        else:
            ret5 = 0.0

        groups[label].append({
            'sym':    sym,
            'ret5d':  ret5,
            'slope_z':dec.get('slope_z', 0),
            'pvo':    dec.get('pvo', 0),
            'vri':    dec.get('vri', 0),
        })

    stats_out = {}
    for label, items in groups.items():
        n = len(items)
        if n == 0:
            continue
        rets = np.array([i['ret5d'] for i in items])
        stats_out[label] = {
            'n':          n,
            'syms':       [i['sym'] for i in items],
            'avg_ret5d':  round(float(np.mean(rets)) * 100, 2),
            'avg_slope_z':round(float(np.mean([i['slope_z'] for i in items])), 2),
            'avg_pvo':    round(float(np.mean([i['pvo'] for i in items])), 2),
            'avg_vri':    round(float(np.mean([i['vri'] for i in items])), 2),
        }
    return stats_out

# ===========================================================================
# t 統計量
# ===========================================================================
def _ttest_scan(arr):
    if len(arr) < 2:
        return 0.0, 1.0
    t, p = stats.ttest_1samp(arr, 0)
    return float(t), float(p)

# ===========================================================================
# Gemini AI
# ===========================================================================
def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ 請在側欄輸入 Gemini API Key 才能啟用 AI 分析。"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemma-3-27b-it"]:
            try:
                model    = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception:
                continue
        return "❌ 所有模型均無法回應，請稍後重試"
    except Exception as e:
        return f"❌ Gemini 呼叫失敗：{e}"

def translate_path(path_str: str) -> str:
    mapping = {"Alive Stage1-Only":"存活(僅S1)","Alive":"存活",
               "Stage1-Only":"僅S1","Dead":"淘汰","N/A":"未知"}
    for eng, chn in mapping.items():
        if eng in str(path_str):
            return str(path_str).replace(eng, chn)
    return str(path_str)

# ===========================================================================
# Session State 初始化
# ===========================================================================
def init_session():
    defaults = {
        "tw_watchlist":        DEFAULT_TW_WATCHLIST.copy(),
        "us_watchlist":        DEFAULT_US_WATCHLIST.copy(),
        "last_scan_time":      None,
        "scan_results":        {},
        "market_sentiment_tw": None,
        "market_sentiment_us": None,
        "ai_summary":          "",
        "data_health":         {},
        "active_market":       "TW",
        "target_date":         datetime.today().strftime("%Y-%m-%d"),
        "gemini_api_key":      _ENV_GEMINI_KEY,
        "single_stock_result": "",
        "single_stock_sym":    "",
        "single_stock_upside": {},
        # ★ 新增：訊號分類統計暫存
        "signal_stats":        {},
        # ★ 新增：績效摘要暫存（從回測引擎匯入時使用）
        "perf_summary":        {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ===========================================================================
# 數據獲取（有 engine_21 用 engine，否則用 yfinance 直接取）
# ===========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_raw(symbol: str, start_str: str, end_str: str) -> dict:
    """備用：engine_21 不可用時直接用 yfinance + pandas_ta"""
    try:
        import yfinance as yf
        import pandas_ta as ta
        df = yf.download(symbol, start=start_str, end=end_str,
                         progress=False, auto_adjust=True)
        if df.empty:
            return {"error": "無數據"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['MA20']  = ta.sma(df['Close'], 20)
        df['ATR']   = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['RSI']   = ta.rsi(df['Close'], length=14)
        adx_df      = ta.adx(df['High'], df['Low'], df['Close'])
        df          = pd.concat([df, adx_df], axis=1)
        df['Ret5d'] = df['Close'].pct_change(5)
        df['PVO']   = _calc_pvo_series(df)
        df['VRI']   = _calc_vri_series(df)
        df['Slope'] = df['Close'].pct_change(5) * 100
        df = df.dropna()
        # 簡易 decision 組裝
        if df.empty:
            return {"error": "指標計算後無數據"}
        last     = df.iloc[-1]
        pvo      = float(last.get('PVO', 0))
        vri      = float(last.get('VRI', 50))
        slope    = float(last.get('Slope', 0))
        rsi      = float(last.get('RSI', 50))
        adx      = float(last.get('ADX_14', 0))
        slp_vals = df['Slope'].tail(60).values
        slope_z  = ((slope - float(np.mean(slp_vals))) /
                    (float(np.std(slp_vals)) + 1e-9))
        close    = float(last['Close'])
        ma20     = float(last['MA20'])
        if close > ma20 and slope_z > 1.0 and pvo > 0:
            action    = "強力買進"
            direction = "做多"
        elif close < ma20 or slope_z < -1.5:
            action    = "觀望/迴避"
            direction = "做空"
        else:
            action    = "持有觀察"
            direction = "觀望"
        decision = {
            "close":     close,
            "action":    action,
            "direction": direction,
            "pvo":       pvo,
            "vri":       vri,
            "slope":     slope,
            "slope_z":   slope_z,
            "pvo_status":("主力點火" if pvo >= 10 else ("資金流入" if pvo >= 0 else "資金撤退")),
            "vri_status":("擁擠過熱" if vri > 75 else ("健康區間" if vri >= 40 else "情緒整理")),
            "signal_level":"",
            "last_action":"─",
            "score":     round(adx * 0.3 + rsi * 0.4 + slope * 20 + pvo * 0.1, 1),
            "date":      str(df.index[-1].date()),
        }
        return {
            "symbol":       symbol,
            "indicator_df": df,
            "raw_df":       df,
            "decision":     decision,
            "stage1":       {"pass": direction == "做多"},
            "stage2":       {"pass": False, "path": "N/A", "ev": None, "t_stat": None},
            "health":       {"pass": True, "issues": []},
            "trust":        {"trust_net_10d": None},
            "market":       "TW" if ".TW" in symbol or ".TWO" in symbol else "US",
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_engine(symbol: str, start_str: str, end_str: str) -> dict:
    if _ENGINE is None:
        return cached_fetch_raw(symbol, start_str, end_str)
    fetch_stock_data = _ENGINE[0]
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    return fetch_stock_data(symbol, start, end)

def get_date_range(target_date_str, lookback=LOOKBACK_DAYS):
    end_dt   = datetime.strptime(target_date_str, "%Y-%m-%d") + timedelta(days=1)
    start_dt = end_dt - timedelta(days=lookback)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

def run_scan(watchlist, target_date, market_type, progress_bar=None):
    start_str, end_str = get_date_range(target_date)
    results = {}

    # 大盤數據
    bm_sym = BENCHMARK_TW if market_type == "TW" else BENCHMARK_US
    bm_res = cached_fetch_engine(bm_sym, start_str, end_str)
    bm_df  = bm_res.get("indicator_df")

    total = len(watchlist)
    for i, sym_raw in enumerate(watchlist):
        sym = f"{sym_raw}.TW" if (market_type == "TW" and "." not in sym_raw) else sym_raw
        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"分析 {sym}...")

        res = cached_fetch_engine(sym, start_str, end_str)
        if res.get("error") or res.get("indicator_df") is None:
            results[sym] = {"error": res.get("error","無數據"), "symbol": sym}
            continue

        df       = res["indicator_df"]
        decision = res.get("decision") or {}
        s1       = res.get("stage1", {})
        s2       = res.get("stage2", {})

        # 若 engine_21 可用則用其 decision，否則 raw 版本已內建
        if _ENGINE is not None:
            try:
                get_decision = _ENGINE[3]
                stage1_fn    = _ENGINE[1]
                stage2_fn    = _ENGINE[2]
                decision = get_decision(df, market=market_type)
                s1       = stage1_fn(df)
                s2       = stage2_fn(sym, s1, ALPHA_SEEDS)
            except Exception:
                pass

        results[sym] = {
            "symbol":       sym,
            "market":       market_type,
            "indicator_df": df,
            "decision":     decision,
            "stage1":       s1,
            "stage2":       s2,
            "health":       res.get("health", {"pass":True,"issues":[]}),
            "trust":        res.get("trust", {"trust_net_10d":None}),
        }

    st.session_state.scan_results  = results
    st.session_state.last_scan_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.signal_stats  = compute_signal_stats_from_scan(results)
    return results

# ===========================================================================
# UI 元件
# ===========================================================================
def render_market_bar(sentiment, market):
    if not sentiment:
        st.info("尚未取得大盤數據，請先執行全盤掃描")
        return
    bear  = sentiment.get("bear", 33)
    neu   = sentiment.get("neutral", 34)
    bull  = sentiment.get("bull", 33)
    label = sentiment.get("label", "震盪")
    s5d   = sentiment.get("slope_5d", 0)
    s20d  = sentiment.get("slope_20d", 0)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("市場", "台股🇹🇼" if market == "TW" else "美股🇺🇸")
    c2.metric("P_熊", f"{bear}%")
    c3.metric("P_震", f"{neu}%")
    c4.metric("P_牛", f"{bull}%")
    c5.metric("大盤情緒", label)
    c6.metric("5日斜率", f"{s5d:+.3f}")
    c7.metric("20日斜率", f"{s20d:+.3f}")

def get_badge_html(direction):
    s = "display:inline-block;padding:3px 10px;border-radius:6px;font-size:.75rem;font-weight:700;"
    if direction == "做多":
        return f'<span style="{s}background:#d1fae5;color:#065f46;">▲ 做多</span>'
    if direction == "做空":
        return f'<span style="{s}background:#fee2e2;color:#991b1b;">▼ 做空</span>'
    return f'<span style="{s}background:#fef3c7;color:#92400e;">◆ 觀望</span>'

def render_stock_card(sym, res, show_final=False):
    if res.get("error"):
        st.warning(f"**{sym}** — {res['error']}")
        return
    dec    = res.get("decision", {})
    s1     = res.get("stage1", {})
    s2     = res.get("stage2", {})
    df_ind = res.get("indicator_df")
    trust  = res.get("trust", {})
    pat    = classify_pattern(dec)
    up     = calc_upside_10pct_prob(dec, s2, df_ind)
    is_fin = is_final_candidate(dec, df_ind)

    mkt_tag = "🇹🇼" if res.get("market") == "TW" else "🇺🇸"
    fin_tag = ' <span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:6px;font-size:.72rem;font-weight:700;">⭐ 最終候選</span>' if (show_final and is_fin) else ""
    ev_val  = s2.get("ev", None) if s2 else None
    ev_str  = f"{ev_val:+.2f}%" if isinstance(ev_val,(int,float)) else "N/A"
    t_val   = s2.get("t_stat", None) if s2 else None
    t_str   = f"{t_val:.2f}" if t_val is not None else "N/A"
    trust_n = trust.get("trust_net_10d", None)
    trust_s = f"{trust_n:+,.0f}張" if trust_n is not None else "N/A"
    prob_v  = up.get('prob', 'N/A')
    rr_v    = up.get('rr_ratio', 'N/A')
    close   = dec.get('close', 0)

    border = "#1a56db" if (show_final and is_fin) else ("#059669" if dec.get("direction")=="做多" else "#dc2626")
    bg     = "linear-gradient(135deg,#eff6ff,#fff)" if (show_final and is_fin) else "#fff"

    pat_html = ""
    for p in pat['patterns']:
        pat_html += f'<span class="badge {p["css"]}" style="margin-right:4px">{p["label"]} 勝率10%:{p["win10"]}%</span>'

    pvo_c  = "#059669" if dec.get("pvo",0)>10 else ("#0891b2" if dec.get("pvo",0)>0 else "#dc2626")
    vri_c  = "#059669" if 40<=dec.get("vri",50)<=75 else ("#dc2626" if dec.get("vri",50)>90 else "#d97706")
    prob_c = "#059669" if isinstance(prob_v,(int,float)) and prob_v>=55 else "#d97706"
    rr_c   = "#059669" if isinstance(rr_v,(int,float)) and rr_v>=2.0 else "#d97706"

    st.markdown(f"""
    <div style="background:{bg};border:1px solid #e2e8f0;border-left:4px solid {border};
                border-radius:10px;padding:12px 16px;margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b style="color:#1a56db;font-size:1rem">{mkt_tag} {sym}</b>
        <span style="font-family:monospace;font-size:1rem">{close:,.2f}</span>
        <span>{get_badge_html(dec.get("direction","觀望"))}{fin_tag}</span>
      </div>
      <div style="margin:6px 0">{pat_html if pat_html else '<span style="color:#94a3b8;font-size:.8rem">無高勝率型態</span>'}</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:.82rem;color:#475569;margin-top:6px">
        <span>PVO:<b style="color:{pvo_c}">{dec.get('pvo',0):+.1f}</b></span>
        <span>VRI:<b style="color:{vri_c}">{dec.get('vri',50):.1f}</b></span>
        <span>Slope Z:<b style="color:#1a56db">{dec.get('slope_z',0):+.2f}</b></span>
        <span>EV:<b>{ev_str}</b></span>
        <span>T值:<b>{t_str}</b></span>
        <span>投信:<b>{trust_s}</b></span>
      </div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:.82rem;color:#475569;
                  background:#f0fdf4;border-radius:6px;padding:5px 8px;margin-top:6px">
        <span>🎯 上漲10%機率:<b style="color:{prob_c}">{prob_v}%</b></span>
        <span>停利:<b style="color:#059669">{up.get('take_profit','N/A')}(+{up.get('tp_pct','N/A')}%)</b></span>
        <span>停損:<b style="color:#dc2626">{up.get('stop_loss','N/A')}(-{up.get('stop_loss_pct','N/A')}%)</b></span>
        <span>風報比:<b style="color:{rr_c}">{rr_v}x</b></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_kline_chart(sym, res):
    df = res.get("indicator_df")
    if df is None or df.empty:
        st.warning("無法取得圖表數據")
        return
    df = df.tail(90)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.22, 0.23],
                        subplot_titles=(f"{sym} K線", "PVO", "VRI"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="K線",
        increasing_line_color='#059669', decreasing_line_color='#dc2626'), row=1, col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20",
                                  line=dict(color='#d97706',width=1.5,dash='dot')), row=1, col=1)
    if 'PVO' in df.columns:
        colors_pvo = ['#059669' if v >= 0 else '#dc2626' for v in df['PVO']]
        fig.add_trace(go.Bar(x=df.index, y=df['PVO'], name="PVO",
                             marker_color=colors_pvo, opacity=0.75), row=2, col=1)
        fig.add_hline(y=10, line_dash="dot", line_color="#d97706", row=2, col=1)
        fig.add_hline(y=0,  line_dash="dot", line_color="#94a3b8", row=2, col=1)
    if 'VRI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VRI'], name="VRI",
                                  line=dict(color='#0891b2',width=2),
                                  fill='tozeroy', fillcolor='rgba(8,145,178,0.08)'), row=3, col=1)
        fig.add_hrect(y0=40, y1=75, fillcolor="rgba(5,150,105,0.06)", line_width=0, row=3, col=1)
    fig.update_layout(height=580, template="plotly_white", paper_bgcolor='#fff',
                      xaxis_rangeslider_visible=False, showlegend=False,
                      margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# Tab5：OOS 績效報告
# ===========================================================================
def render_tab_perf():
    st.markdown("## 📊 OOS 績效報告（V12.1C 標準框）")
    st.info("""
    此頁顯示歷代版本對比與 OOS 標準統計。
    **如需本次掃描的即時績效**，請先完成「全盤掃描」；
    **如需完整回測績效**，請執行 `資源法2026-V1.py` 並將輸出貼入下方欄位。
    """)

    # 版本對比表
    st.markdown("### 歷代版本對比")
    df_ver = pd.DataFrame(VERSION_BENCHMARKS,
                          columns=["版本","筆數","勝率%","均EV%","年化%","MaxDD%","Sharpe","t值"])
    st.dataframe(df_ver, use_container_width=True, hide_index=True,
                 column_config={
                     "勝率%":  st.column_config.NumberColumn(format="%.1f%%"),
                     "均EV%":  st.column_config.NumberColumn(format="+%.2f%%"),
                     "年化%":  st.column_config.NumberColumn(format="+%.1f%%"),
                     "MaxDD%": st.column_config.NumberColumn(format="%.2f%%"),
                     "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                     "t值":    st.column_config.NumberColumn(format="+%.3f"),
                 })

    # 本次掃描摘要（從 scan_results 計算）
    results = st.session_state.scan_results
    if results:
        st.markdown("### 本次掃描標的分佈")
        total    = len([r for r in results.values() if not r.get('error')])
        bull_n   = len([r for r in results.values() if not r.get('error')
                        and r.get('decision',{}).get('direction')=='做多'])
        final_n  = len([r for r in results.values() if not r.get('error')
                        and is_final_candidate(r.get('decision',{}), r.get('indicator_df'))])
        with_pat = len([r for r in results.values() if not r.get('error')
                        and classify_pattern(r.get('decision',{}))['is_key']])
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("有效標的", total)
        c2.metric("做多訊號", bull_n)
        c3.metric("命中型態", with_pat)
        c4.metric("最終候選⭐", final_n)

        # 各訊號分類統計摘要
        sig_stats = st.session_state.signal_stats
        if sig_stats:
            st.markdown("### 本次掃描訊號分類分佈")
            rows = []
            for lbl, d in sorted(sig_stats.items(), key=lambda x: -x[1]['n']):
                rows.append({
                    "訊號分類":   lbl,
                    "標的數":     d['n'],
                    "平均5日報酬%": d['avg_ret5d'],
                    "平均Slope Z":  d['avg_slope_z'],
                    "平均PVO":      d['avg_pvo'],
                    "平均VRI":      d['avg_vri'],
                })
            df_sig = pd.DataFrame(rows)
            st.dataframe(df_sig, use_container_width=True, hide_index=True)

    # 手動輸入回測報告
    st.markdown("### 貼入回測報告（選填）")
    perf_text = st.text_area(
        "將 資源法2026-V1.py 執行結果貼入此處",
        height=200,
        placeholder="將終端機輸出的績效報告文字貼入此處..."
    )
    if perf_text:
        st.code(perf_text, language="text")

# ===========================================================================
# Tab6：訊號分類統計
# ===========================================================================
def render_tab_signal():
    st.markdown("## 📈 訊號分類統計（三合一 / 二合一 / 單一 vs 基準）")

    # 已知基準值表
    st.markdown("### 已知統計基準值（來自歷史文件）")
    known_rows = []
    for lbl, bd in SIGNAL_BASELINES.items():
        known_rows.append({
            "訊號分類": lbl,
            "參考樣本數": bd['n_ref'],
            "勝率10%": bd['win10'],
            "均天數(10%)": bd['days10'],
            "勝率20%": bd['win20'],
            "均天數(20%)": bd['days20'],
            "虧損率10%": bd['loss10'],
            "均天數(虧)": bd['ld10'],
            "信效度": bd['stars'],
        })
    for lbl, kn in SIGNAL_KNOWN_PATTERNS.items():
        known_rows.append({
            "訊號分類": lbl,
            "參考樣本數": 0,
            "勝率10%": kn['win10'] if kn['win10'] else None,
            "均天數(10%)": None,
            "勝率20%": kn['win20'] if kn['win20'] else None,
            "均天數(20%)": None,
            "虧損率10%": None,
            "均天數(虧)": None,
            "信效度": kn['stars'],
        })
    df_known = pd.DataFrame(known_rows)
    st.dataframe(df_known, use_container_width=True, hide_index=True,
                 column_config={
                     "勝率10%":   st.column_config.NumberColumn(format="%.1f%%"),
                     "勝率20%":   st.column_config.NumberColumn(format="%.1f%%"),
                     "虧損率10%": st.column_config.NumberColumn(format="%.1f%%"),
                 })

    # 本次掃描實際分類
    sig_stats = st.session_state.signal_stats
    if not sig_stats:
        st.info("請先執行全盤掃描以取得本次訊號分類數據")
    else:
        st.markdown("### 本次掃描實際訊號分類")
        baseline_w10 = SIGNAL_BASELINES['基準-強勢']['win10']
        rows = []
        for lbl, d in sorted(sig_stats.items(), key=lambda x: -x[1]['n']):
            known_w10 = (SIGNAL_BASELINES.get(lbl, {}).get('win10') or
                         SIGNAL_KNOWN_PATTERNS.get(lbl, {}).get('win10') or 0)
            diff      = round(known_w10 - baseline_w10, 1) if known_w10 else None
            rows.append({
                "訊號分類":     lbl,
                "本次標的數":   d['n'],
                "已知勝率10%":  known_w10 if known_w10 else "─",
                "vs基準差值":   f"+{diff:.1f}%" if diff and diff >= 0 else (f"{diff:.1f}%" if diff else "─"),
                "平均5日報酬%": d['avg_ret5d'],
                "平均Slope Z":  d['avg_slope_z'],
                "平均PVO":      d['avg_pvo'],
                "平均VRI":      d['avg_vri'],
                "標的列表":     ", ".join([s.replace(".TW","").replace(".TWO","")
                                          for s in d['syms'][:8]]) + ("..." if len(d['syms'])>8 else ""),
            })
        df_now = pd.DataFrame(rows)
        st.dataframe(df_now, use_container_width=True, hide_index=True)

        # 各型態標的 expander
        for lbl, d in sorted(sig_stats.items(), key=lambda x: -x[1]['n']):
            if d['n'] == 0:
                continue
            with st.expander(f"📋 {lbl} — {d['n']} 支標的"):
                syms_clean = [s.replace(".TW","").replace(".TWO","") for s in d['syms']]
                st.write("、".join(syms_clean))

    # 說明
    st.markdown("""
    ---
    **指標說明**
    - 勝率10%：持有期間報酬曾達 +10%（已知基準值來自歷史文件，本次為掃描代理值）
    - 勝率20%：持有期間報酬曾達 +20%
    - 虧損率10%：持有期間報酬曾低於 -10%
    - 平均5日報酬%：本次掃描時點的 5 日實際報酬（非回測預測）
    - 三合一/二合一需 **n≥30** 才具統計信效度
    - 完整回測統計請執行 `資源法2026-V1.py`
    """)

# ===========================================================================
# 側欄
# ===========================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ 資源法 AI 戰情室 V1")
        st.markdown("---")
        st.markdown("### 🌐 市場選擇")
        market = st.radio("", ["🇹🇼 台股","🇺🇸 美股"], horizontal=True,
                          index=0 if st.session_state.active_market=="TW" else 1)
        st.session_state.active_market = "TW" if "台股" in market else "US"

        st.markdown("---")
        st.markdown("### 📅 分析日期")
        target = st.date_input("資料截止日",
                               value=datetime.strptime(st.session_state.target_date, "%Y-%m-%d"),
                               max_value=datetime.today())
        st.session_state.target_date = target.strftime("%Y-%m-%d")

        st.markdown("---")
        st.markdown("### 📋 觀察名單")
        if st.session_state.active_market == "TW":
            tw_in = st.text_area("台股代號（逗號分隔）",
                                 value=", ".join(st.session_state.tw_watchlist[:30]),
                                 height=100)
            parsed = [s.strip() for s in tw_in.replace('\n',',').split(',') if s.strip()]
            if parsed:
                st.session_state.tw_watchlist = parsed
            st.caption(f"共 {len(st.session_state.tw_watchlist)} 檔")
        else:
            us_in = st.text_area("美股代號（逗號分隔）",
                                 value=", ".join(st.session_state.us_watchlist),
                                 height=80)
            parsed = [s.strip().upper() for s in us_in.replace('\n',',').split(',') if s.strip()]
            if parsed:
                st.session_state.us_watchlist = parsed
            st.caption(f"共 {len(st.session_state.us_watchlist)} 檔")

        st.markdown("---")
        st.markdown("### 🤖 Gemini API Key")
        if _ENV_GEMINI_KEY:
            st.success("✅ 已從環境變數載入")
        else:
            api_key = st.text_input("API Key", type="password",
                                    value=st.session_state.gemini_api_key,
                                    placeholder="AIza...")
            st.session_state.gemini_api_key = api_key
        st.markdown("---")
        if _ENGINE is None:
            st.warning("⚠️ engine_21.py 未找到，使用備用數據模式（yfinance直接取數）")
        else:
            st.success("✅ engine_21 已載入")
        st.caption("© 2026 資源法 AI 戰情室 V1")

# ===========================================================================
# 主程式
# ===========================================================================
def main():
    render_sidebar()

    col_title, col_scan = st.columns([4, 1])
    with col_title:
        st.markdown("# ⚡ 資源法 AI 戰情室 V1")
        st.markdown(
            f"<small style='color:#64748b'>分析日期: {st.session_state.target_date} | "
            f"{'台股🇹🇼' if st.session_state.active_market=='TW' else '美股🇺🇸'} | "
            f"最後掃描: {st.session_state.last_scan_time or '尚未掃描'}</small>",
            unsafe_allow_html=True)
    with col_scan:
        scan_btn = st.button("🔄 執行全盤掃描", use_container_width=True, type="primary")

    sentiment = st.session_state.get(
        "market_sentiment_tw" if st.session_state.active_market=="TW"
        else "market_sentiment_us")
    render_market_bar(sentiment, st.session_state.active_market)

    if scan_btn:
        market = st.session_state.active_market
        wl     = (st.session_state.tw_watchlist if market=="TW"
                  else st.session_state.us_watchlist)
        with st.spinner("⚙️ 正在擷取數據並計算指標..."):
            pb = st.progress(0, text="初始化...")
            run_scan(wl, st.session_state.target_date, market, progress_bar=pb)
            pb.empty()
        st.rerun()

    # ── 全盤 Gemini 分析 ──────────────────────────────────────
    ai_col1, ai_col2 = st.columns([1, 4])
    with ai_col1:
        ai_btn = st.button("🤖 Gemini 全盤分析", use_container_width=True)
    with ai_col2:
        if st.session_state.ai_summary:
            import html as _ht
            safe = _ht.escape(st.session_state.ai_summary).replace('\n','<br>')
            st.markdown(f'<div class="ai-box">{safe}</div>', unsafe_allow_html=True)

    if ai_btn:
        if not st.session_state.scan_results:
            st.warning("請先執行掃描")
        else:
            results = st.session_state.scan_results
            candidates = []
            for sym, res in results.items():
                if res.get('error'): continue
                dec = res.get('decision', {})
                if dec.get('direction') != '做多': continue
                pat = classify_pattern(dec)
                up  = calc_upside_10pct_prob(dec, res.get('stage2'), res.get('indicator_df'))
                candidates.append(
                    f"【{sym}】型態:{pat['sig_label']} PVO:{dec.get('pvo',0):+.1f} "
                    f"VRI:{dec.get('vri',50):.1f} SlopeZ:{dec.get('slope_z',0):+.2f} "
                    f"上漲10%機率:{up.get('prob','N/A')}% 停利:{up.get('take_profit','N/A')} "
                    f"停損:{up.get('stop_loss','N/A')}")

            prompt = f"""你是資深量化交易分析師。今日{st.session_state.active_market}市場做多候選：

{chr(10).join(candidates[:15])}

請完成：
1. 大盤風險評估（100字）
2. 最佳3檔精選分析（各100字，含型態/PVO/VRI/停利停損理由）
3. 主要風險提示（80字）
所有論點引用具體數值。"""
            with st.spinner("🤖 Gemini 分析中..."):
                summary = call_gemini(prompt, st.session_state.gemini_api_key)
            st.session_state.ai_summary = summary
            st.rerun()

    # ── 個股深度分析 ──────────────────────────────────────────
    st.markdown("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid #059669;
                border-radius:10px;padding:8px 14px;margin:10px 0 4px">
        <b style="color:#059669">🔍 個股 AI 深度分析</b>
        <span style="color:#64748b;font-size:.82rem;margin-left:8px">輸入觀察名單內代號</span>
    </div>""", unsafe_allow_html=True)

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        single_sym = st.text_input("個股代號", placeholder="例: 2330 / NVDA",
                                   label_visibility="collapsed", key="single_sym_v1")
    with col_btn:
        single_btn = st.button("🔍 分析此股", use_container_width=True, key="single_btn_v1")

    if st.session_state.single_stock_result and st.session_state.single_stock_sym:
        sym_d  = st.session_state.single_stock_sym
        up_c   = st.session_state.single_stock_upside
        import html as _hc
        safe_c = _hc.escape(st.session_state.single_stock_result).replace('\n','<br>')
        st.markdown(f"""
        <div class="ai-box" style="border-left-color:#059669;margin:8px 0 14px">
            <b style="color:#059669">{sym_d} 個股深度分析</b>
            <span style="font-size:.78rem;color:#64748b;margin-left:8px">
            機率:{up_c.get('prob','N/A')}% | 停利:{up_c.get('take_profit','N/A')}(+{up_c.get('tp_pct','N/A')}%)
            | 停損:{up_c.get('stop_loss','N/A')}(-{up_c.get('stop_loss_pct','N/A')}%)
            | 風報比:{up_c.get('rr_ratio','N/A')}x</span><hr>
            {safe_c}
        </div>""", unsafe_allow_html=True)

    if single_btn and single_sym:
        q = single_sym.strip().upper()
        results_now = st.session_state.scan_results
        matched = None
        for k in results_now:
            kc = k.replace(".TWO","").replace(".TW","").strip().upper()
            if kc == q or k.upper() == q:
                matched = k; break
        if not matched or results_now.get(matched,{}).get('error'):
            st.warning(f"⚠️ 找不到 {q}，請確認已掃描且在觀察名單中")
        else:
            res_s  = results_now[matched]
            dec_s  = res_s.get('decision', {})
            up_s   = calc_upside_10pct_prob(dec_s, res_s.get('stage2'), res_s.get('indicator_df'))
            pat_s  = classify_pattern(dec_s)
            prompt_s = f"""你是資深量化交易分析師，對 {matched} 進行深度分析：
PVO:{dec_s.get('pvo',0):+.2f} VRI:{dec_s.get('vri',50):.1f} SlopeZ:{dec_s.get('slope_z',0):+.2f}
型態:{pat_s['sig_label']} 方向:{dec_s.get('direction','─')} 現價:{dec_s.get('close',0):.2f}
上漲10%機率:{up_s.get('prob','N/A')}% 停利:{up_s.get('take_profit','N/A')}(+{up_s.get('tp_pct','N/A')}%)
停損:{up_s.get('stop_loss','N/A')}(-{up_s.get('stop_loss_pct','N/A')}%) 風報比:{up_s.get('rr_ratio','N/A')}x

請完成（各100字，所有論點必須引用具體數值）：
1. 技術面研判（PVO/VRI/SlopeZ含義）
2. 統計優勢（型態勝率意義）
3. 具體操作建議（進場時機/停損邏輯/停利策略）
4. 主要風險因子"""
            with st.spinner(f"🤖 深度分析 {matched} 中..."):
                result_s = call_gemini(prompt_s, st.session_state.gemini_api_key)
            st.session_state.single_stock_result = result_s
            st.session_state.single_stock_sym    = matched
            st.session_state.single_stock_upside = up_s
            st.rerun()

    st.markdown("---")

    # ── 六大 Tab ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 視覺化圖表",
        "📋 數據與排名",
        "🎯 決策戰情室",
        "🔬 數據健康度",
        "📈 OOS績效報告",   # ★ 新增
        "🏷️ 訊號分類統計",  # ★ 新增
    ])

    results = st.session_state.scan_results
    market  = st.session_state.active_market

    # ── Tab1：K線圖 ───────────────────────────────────────────
    with tab1:
        if not results:
            st.info("👈 請先執行「全盤掃描」")
        else:
            sym_list = [s for s, r in results.items() if not r.get("error")]
            if sym_list:
                selected = st.selectbox("選擇個股", sym_list)
                if selected in results:
                    render_kline_chart(selected, results[selected])
                    dec = results[selected].get('decision', {})
                    s2  = results[selected].get('stage2', {})
                    df_c= results[selected].get('indicator_df')
                    up  = calc_upside_10pct_prob(dec, s2, df_c)
                    c1,c2,c3,c4,c5,c6 = st.columns(6)
                    c1.metric("現價",    f"{dec.get('close',0):,.2f}")
                    c2.metric("PVO",     f"{dec.get('pvo',0):+.2f}")
                    c3.metric("VRI",     f"{dec.get('vri',50):.1f}")
                    c4.metric("Slope Z", f"{dec.get('slope_z',0):+.2f}")
                    c5.metric("上漲10%機率", f"{up.get('prob','N/A')}%")
                    c6.metric("風報比",  f"{up.get('rr_ratio','N/A')}x")

    # ── Tab2：數據與排名 ──────────────────────────────────────
    with tab2:
        if not results:
            st.info("請先執行掃描")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                show_all = st.checkbox("顯示全部（含未通過Stage1）", value=False)
            with col_f2:
                sort_by = st.selectbox("排序依據",
                    ["最佳勝率10%","上漲10%機率","Slope Z","VRI","PVO"])

            table_rows = []
            for sym, res in results.items():
                if res.get("error"): continue
                dec    = res.get('decision', {})
                s1     = res.get('stage1', {})
                s2     = res.get('stage2', {}) or {}
                df_ind = res.get('indicator_df')
                pat    = classify_pattern(dec)
                up     = calc_upside_10pct_prob(dec, s2, df_ind)
                if not show_all and not s1.get("pass"): continue
                ev_v   = s2.get('ev', None)
                t_v    = s2.get('t_stat', None)
                trust  = res.get('trust', {})
                table_rows.append({
                    "代號":        sym.replace(".TW","").replace(".TWO",""),
                    "方向":        dec.get("direction","─"),
                    "訊號分類":    pat['sig_label'],
                    "最佳勝率10%": pat['best_win10'] if pat['best_win10'] else None,
                    "現價":        dec.get("close", 0),
                    "PVO":         dec.get("pvo", 0),
                    "VRI":         dec.get("vri", 50),
                    "Slope Z":     dec.get("slope_z", 0),
                    "EV期望值%":   ev_v,
                    "T值":         t_v,
                    "上漲10%機率": up.get("prob"),
                    "停利價":      up.get("take_profit"),
                    "停損價":      up.get("stop_loss"),
                    "停利%":       up.get("tp_pct"),
                    "停損%":       up.get("stop_loss_pct"),
                    "風報比":      up.get("rr_ratio"),
                    "投信10日(張)":trust.get("trust_net_10d"),
                    "Stage1":      "✅" if s1.get("pass") else "❌",
                    "最終候選":    "⭐" if is_final_candidate(dec, df_ind) else "─",
                })

            if table_rows:
                sort_map = {"最佳勝率10%":"最佳勝率10%","上漲10%機率":"上漲10%機率",
                            "Slope Z":"Slope Z","VRI":"VRI","PVO":"PVO"}
                df_t = pd.DataFrame(table_rows)
                df_t["_key"] = df_t[sort_map.get(sort_by,"最佳勝率10%")].fillna(0)
                df_t = df_t.sort_values("_key", ascending=False).drop(columns=["_key"])
                final_n  = df_t["最終候選"].eq("⭐").sum()
                pat_n    = df_t["最佳勝率10%"].notna().sum()
                st.markdown(f"共 **{len(df_t)}** 檔 | 命中型態 **{pat_n}** 檔 | 最終候選 **{final_n}** 檔")
                st.dataframe(df_t, use_container_width=True, hide_index=True,
                             column_config={
                                 "最佳勝率10%":  st.column_config.NumberColumn(format="%.1f%%"),
                                 "上漲10%機率":  st.column_config.NumberColumn(format="%.1f%%"),
                                 "PVO":          st.column_config.NumberColumn(format="+%.2f"),
                                 "Slope Z":      st.column_config.NumberColumn(format="+%.2f"),
                                 "EV期望值%":    st.column_config.NumberColumn(format="+%.2f%%"),
                                 "風報比":       st.column_config.NumberColumn(format="%.2fx"),
                                 "停利%":        st.column_config.NumberColumn(format="+%.1f%%"),
                                 "停損%":        st.column_config.NumberColumn(format="-%.1f%%"),
                                 "投信10日(張)": st.column_config.NumberColumn(format="%+,.0f"),
                             })
                csv = df_t.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("⬇️ 下載 CSV", csv,
                                   file_name=f"scan_{st.session_state.target_date}.csv",
                                   mime="text/csv")
            else:
                st.info("無符合條件標的")

    # ── Tab3：決策戰情室 ──────────────────────────────────────
    with tab3:
        if not results:
            st.info("請先執行掃描")
        else:
            # 最終候選
            final_picks = [(s, r) for s, r in results.items()
                           if not r.get("error") and is_final_candidate(
                               r.get("decision",{}), r.get("indicator_df"))]
            final_picks.sort(
                key=lambda x: classify_pattern(x[1].get("decision",{}))['best_win10'],
                reverse=True)

            st.markdown("""
            <div style="background:linear-gradient(135deg,#dbeafe,#eff6ff);border:2px solid #1a56db;
                border-radius:10px;padding:10px 16px;margin-bottom:12px">
            <b style="color:#1a56db">⭐ 最終決策候選</b>
            <span style="color:#64748b;font-size:.82rem;margin-left:8px">
            命中型態 + PVO波動率>60% + VRI波動率>60%</span>
            </div>""", unsafe_allow_html=True)

            if final_picks:
                for sym, res in final_picks:
                    render_stock_card(sym, res, show_final=True)
            else:
                st.info("目前無最終候選，查看下方做多訊號")

            st.markdown("---")

            # 做多訊號
            bull_stocks = [(s, r) for s, r in results.items()
                           if not r.get("error")
                           and r.get("decision",{}).get("direction")=="做多"
                           and r.get("stage1",{}).get("pass")]
            bull_stocks.sort(
                key=lambda x: (int(is_final_candidate(x[1].get("decision",{}),
                                                       x[1].get("indicator_df"))),
                               classify_pattern(x[1].get("decision",{}))['best_win10'],
                               x[1].get("decision",{}).get("slope_z",0)),
                reverse=True)

            st.markdown(f"#### 🟢 做多訊號（{len(bull_stocks)} 檔通過Stage1）")
            for sym, res in bull_stocks:
                render_stock_card(sym, res, show_final=True)

    # ── Tab4：數據健康度 ──────────────────────────────────────
    with tab4:
        st.markdown("### 🔬 數據健康度")
        if not st.session_state.data_health:
            st.info("請先執行掃描")
        else:
            health_rows = []
            for sym, h in st.session_state.data_health.items():
                health_rows.append({
                    "代號":  sym,
                    "狀態":  "✅ 正常" if h.get("pass") else "⚠️ 異常",
                    "問題":  " | ".join(h.get("issues",[])) or "─"
                })
            st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

    # ── Tab5：OOS績效報告（★ 新增）─────────────────────────────
    with tab5:
        render_tab_perf()

    # ── Tab6：訊號分類統計（★ 新增）────────────────────────────
    with tab6:
        render_tab_signal()


if __name__ == "__main__":
    main()
