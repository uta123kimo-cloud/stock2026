"""
app.py — 資源法 AI 戰情室
Streamlit 主程式 | 台股/美股雙軌 | 四層數據防火牆
修改版本：
  [新增1] 個股 Gemini 分析欄位（Gemini 分析下方）
  [新增2] 數據排名顯示所有個股 EV/T值，並增加「上漲10%機率/停利/停損」欄位
  [新增3] 做多股票顯示條件滿足當天日期（數據排名 + 決策戰情室）
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

/* [新增1] 個股分析區塊樣式 */
.stock-ai-box {
    background: linear-gradient(135deg, #f0fdf4, #f8faff);
    border: 1px solid #bbf7d0;
    border-left: 4px solid #059669;
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 1.0rem;
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

/* [新增3] 條件滿足日期標籤 */
.signal-date-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    background: #fef9c3;
    color: #713f12;
    border: 1px solid #fde047;
    margin-left: 8px;
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
[role="option"][aria-selected="true"] { background-color: #fef2f2 !important; color: #dc2626 !important; font-weight: 700 !important; }

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
DEFAULT_US_WATCHLIST = ["ABNB", "ADBE", "AMD", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI", "AAPL", "AMAT", "ASML", "AXON", "TEAM", "ADSK", "ADP", "ARM", "APP", "AZN", "BIIB", "BKNG", "BKR", "AVGO", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CTAS", "CSCO", "CSGP", "CTSH", "CMCSA", "CPRT", "COST", "CRWD", "CSX", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "META", "FTNT", "GEHC", "GFS", "GILD", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KLAC", "KHC", "LRCX", "LIN", "LULU", "MAR", "MDB", "MRVL", "MELI", "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MNST", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PCAR", "PAYX", "PYPL", "PEP", "PDD", "PLTR", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TMUS", "TSLA", "TTWO", "TTD", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS", "AMKR", "COHR", "CRUS", "ENTG", "LSCC", "MPWR", "MTSI", "ONTO", "QRVO", "SWKS", "TER", "TSM", "MMM", "ABT", "ABBV", "ACN", "AES", "AFL", "A", "APD", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "MO", "AMCR", "AEE", "AAL", "AXP", "AIG", "AMT", "AWK", "AMP", "ABCB", "AME", "APH", "AON", "AOS", "APA", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "AZO", "AVB", "AVY", "BAC", "BBWI", "BAX", "BDX", "BRK-B", "BBY", "BG", "BIO", "BLDR", "BLK", "BK", "BA", "BWA", "BXP", "BSX", "BMY", "BR", "BF-B", "BX", "CHRW", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CAT", "CBOE", "CBRE", "CE", "CNC", "CNP", "CF", "CRL", "SCHW", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "C", "CFG", "CLX", "CME", "CMS", "KO", "CL", "CMA", "CAG", "COP", "ED", "STZ", "COO", "GLW", "CTVA", "CCI", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "DVN", "DECK", "DLR", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EMR", "ENPH", "ETR", "EOG", "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EVRG", "ES", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FRT", "FDX", "FICO", "FIS", "FITB", "FE", "FMC", "F", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GNRC", "GD", "GE", "GEV", "GIS", "GM", "GPC", "GL", "GPN", "GS", "GWW", "HAL", "HBI", "HIG", "HAS", "HCA", "HSIC", "HSY", "HPE", "HLT", "HOLX", "HD", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IEX", "INFO", "ITW", "ILMN", "INCY", "IR", "ICE", "IBM", "IP", "IPG", "IFF", "IVZ", "INVH", "IQV", "IRM", "JKHY", "J", "JBHT", "JBL", "SJM", "JNJ", "JCI", "JPM", "K", "KEY", "KEYS", "KMB", "KIM", "KMI", "KR", "KVUE", "LHX", "LH", "LW", "LVS", "LEG", "LDOS", "LEN", "LLY", "LYV", "LKQ", "LMT", "L", "LOW", "LYB", "MTB", "MPC", "MKTX", "MMC", "MLM", "MAS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM", "MAA", "MRNA", "MHK", "TAP", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NSC", "NTRS", "NOC", "NCLH", "NOV", "NRG", "NUE", "NVR", "OXY", "OMC", "OKE", "ORCL", "OTIS", "PKG", "PH", "PAYC", "PNR", "PRGO", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "PODD", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PTC", "PEG", "PSA", "PHM", "PWR", "DGX", "RL", "RJF", "RTX", "O", "REG", "RF", "RSG", "RMD", "RHI", "ROK", "ROL", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STLD", "STX", "SRE", "NOW", "SHW", "SMCI", "SPG", "SNA", "SO", "SOLV", "LUV", "SWK", "STT", "STE", "SYK", "SYF", "SYY", "TECH", "TROW", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "UBER", "UDR", "ULTA", "USB", "UNP", "UAL", "UNH", "UPS", "URI", "UHS", "UNM", "VLTO", "VLO", "VTR", "VRSN", "VZ", "VTRS", "V", "VMC", "VST", "WRB", "WAB", "WMT", "WBA", "DIS", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WU", "WRK", "WY", "WMB", "WLTW", "WYNN", "XLNX", "XYL", "YUM", "ZBRA", "ZBH", "ZTS", "PARA"
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


def is_final_candidate(dec: dict, df_ind) -> bool:
    pat = classify_pattern(dec)
    if not pat["is_key_pattern"]:
        return False
    pvo_ratio = calc_pvo_ratio(df_ind)
    vri_ratio = calc_vri_ratio(df_ind)
    return pvo_ratio > 0.6 and vri_ratio > 0.6


# ===========================================================================
# [新增2] 推算上漲10%機率、停利、停損
# ===========================================================================
def calc_upside_stats(dec: dict, df_ind, pat: dict) -> dict:
    """
    以統計方式推算：
    - 上漲10%機率: 基於高勝率型態的win10勝率加權 + SlopeZ修正
    - 停利價: 現價 * (1 + 目標報酬率)，目標報酬率以EV期望值為準，最低5%
    - 停損價: 現價 * (1 - 停損比率)，停損比率=max(ATR倍數, 固定3%)
    """
    close = dec.get("close", 0)
    if close <= 0:
        return {"prob10": None, "tp": None, "sl": None}

    slope_z = dec.get("slope_z", 0)
    vri     = dec.get("vri", 0)
    pvo     = dec.get("pvo", 0)
    best_win10 = pat.get("best_win10", 0)

    # 上漲10%機率推算（基礎勝率 + SlopeZ貢獻 + VRI/PVO修正）
    base_prob = best_win10 if best_win10 > 0 else 30.0
    slope_bonus = min(slope_z * 3.0, 15.0) if slope_z > 0 else max(slope_z * 3.0, -15.0)
    vri_bonus = 5.0 if 40 <= vri <= 75 else (3.0 if vri > 75 else -3.0)
    pvo_bonus = 5.0 if pvo > 10 else (2.0 if pvo > 0 else -5.0)
    prob10 = min(max(base_prob + slope_bonus + vri_bonus + pvo_bonus, 5.0), 95.0)

    # ATR 估算（用近20日高低差均值）
    atr_pct = 0.05  # 預設5%
    if df_ind is not None and not df_ind.empty:
        if "High" in df_ind.columns and "Low" in df_ind.columns:
            recent = df_ind.tail(20)
            hl_range = (recent["High"] - recent["Low"]) / recent["Close"].replace(0, np.nan)
            atr_pct = float(hl_range.mean()) if not hl_range.isna().all() else 0.05

    # 停利：以10%為目標，若SlopeZ強則拉高到12%
    tp_pct = 0.12 if slope_z >= 2.0 else (0.10 if slope_z >= 1.0 else 0.08)
    tp = round(close * (1 + tp_pct), 2)

    # 停損：2倍ATR，最低3%，最高8%
    sl_pct = min(max(atr_pct * 2.0, 0.03), 0.08)
    sl = round(close * (1 - sl_pct), 2)

    return {
        "prob10": round(prob10, 1),
        "tp": tp,
        "sl": sl,
        "tp_pct": round(tp_pct * 100, 1),
        "sl_pct": round(sl_pct * 100, 1),
    }


# ===========================================================================
# [新增3] 取得條件滿足日期（訊號觸發日期）
# ===========================================================================
def get_signal_date(dec: dict, df_ind) -> str:
    """
    取得做多條件滿足的最近日期。
    優先從 decision 的 date 欄位取，
    若無則從 indicator_df 取最後一筆有效日期。
    """
    date_str = dec.get("date", "")
    if date_str:
        return str(date_str)[:10]

    if df_ind is not None and not df_ind.empty:
        try:
            last_idx = df_ind.index[-1]
            if hasattr(last_idx, 'strftime'):
                return last_idx.strftime("%Y-%m-%d")
            return str(last_idx)[:10]
        except Exception:
            pass
    return "—"


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
        "stock_ai_result": "",       # [新增1]
        "stock_ai_query": "",        # [新增1]
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
# JS 強制修正文字顏色
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
                el.querySelectorAll('*').forEach(c => { if(!c.style.color || c.style.color === 'rgb(250, 250, 250)') c.style.color = '#1e293b'; });
            });
            document.querySelectorAll('[data-testid="stCheckbox"] label, [data-baseweb="checkbox"] label').forEach(el => {
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


def build_gemini_prompt(scan_results: dict, market_sentiment: dict,
                        market: str = "TW") -> str:
    """大盤 + 做多候選深度分析 Prompt"""
    bull_candidates = []
    final_flag_syms = []

    for sym, res in scan_results.items():
        dec = res.get("decision", {})
        s1  = res.get("stage1", {})
        s2  = res.get("stage2", {})
        df_ind = res.get("indicator_df")

        if dec.get("direction") != "做多":
            continue

        pat = classify_pattern(dec)
        pvo_ratio = calc_pvo_ratio(df_ind)
        vri_ratio = calc_vri_ratio(df_ind)
        is_final  = is_final_candidate(dec, df_ind)

        if is_final:
            final_flag_syms.append(sym)

        pat_codes = "+".join([p["code"] for p in pat["patterns"]]) if pat["patterns"] else "無"
        pat_desc  = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "無型態"

        trust = res.get("trust", {})
        trust_net = trust.get("trust_net_10d", None)
        trust_str = f"{trust_net:+,.0f}張" if trust_net is not None else "N/A"

        ev_val = s2.get("ev", None)
        ev_str = f"{ev_val:+.2f}%" if isinstance(ev_val, (int, float)) else "N/A"
        t_stat = s2.get("t_stat", None)
        t_str  = f"{t_stat:.2f}" if t_stat is not None else "N/A"
        path   = translate_path(str(s2.get("path", "N/A")))

        upside = calc_upside_stats(dec, df_ind, pat)

        bull_candidates.append({
            "sym": sym,
            "close": dec.get("close", 0),
            "action": dec.get("action", "---"),
            "pvo": dec.get("pvo", 0),
            "vri": dec.get("vri", 0),
            "slope": dec.get("slope", 0),
            "slope_z": dec.get("slope_z", 0),
            "pvo_status": dec.get("pvo_status", ""),
            "vri_status": dec.get("vri_status", ""),
            "pvo_ratio": pvo_ratio,
            "vri_ratio": vri_ratio,
            "pat_codes": pat_codes,
            "pat_desc": pat_desc,
            "best_win10": pat["best_win10"],
            "ev": ev_str,
            "t_stat": t_str,
            "path": path,
            "trust": trust_str,
            "is_final": is_final,
            "s1_pass": s1.get("pass", False),
            "score": dec.get("score", 0),
            "signal_level": dec.get("signal_level", ""),
            "prob10": upside.get("prob10"),
            "tp": upside.get("tp"),
            "sl": upside.get("sl"),
            "signal_date": get_signal_date(dec, df_ind),
        })

    bull_candidates.sort(
        key=lambda x: (int(x["is_final"]), x["best_win10"], x["slope_z"]),
        reverse=True
    )
    top_candidates = bull_candidates[:20]

    label   = market_sentiment.get("label", "不明") if market_sentiment else "不明"
    slope5  = market_sentiment.get("slope_5d", 0) if market_sentiment else 0
    slope20 = market_sentiment.get("slope_20d", 0) if market_sentiment else 0
    bear_p  = market_sentiment.get("bear", 33) if market_sentiment else 33
    bull_p  = market_sentiment.get("bull", 33) if market_sentiment else 33
    neu_p   = market_sentiment.get("neutral", 34) if market_sentiment else 34

    mkt_label = "台股" if market == "TW" else "美股"

    candidate_lines = []
    for c in top_candidates:
        final_marker = "⭐最終候選" if c["is_final"] else ""
        prob_str = f"{c['prob10']}%" if c['prob10'] else "N/A"
        tp_str   = f"{c['tp']}" if c['tp'] else "N/A"
        sl_str   = f"{c['sl']}" if c['sl'] else "N/A"
        line = (
            f"【{c['sym']}】{final_marker} 訊號日:{c['signal_date']}\n"
            f"  現價:{c['close']:.2f} | 操作:{c['action']} {c['signal_level']}\n"
            f"  PVO:{c['pvo']:+.2f}({c['pvo_status']}) | VRI:{c['vri']:.1f}({c['vri_status']})\n"
            f"  Slope:{c['slope']:+.3f}% | SlopeZ:{c['slope_z']:+.2f} | Score:{c['score']}\n"
            f"  PVO波動率:{c['pvo_ratio']:.0%} | VRI波動率:{c['vri_ratio']:.0%}\n"
            f"  高勝率型態:{c['pat_codes']}({c['pat_desc']}) | 最高勝率10%:{c['best_win10']}%\n"
            f"  EV期望值:{c['ev']} | T值:{c['t_stat']} | 路徑:{c['path']}\n"
            f"  推算上漲10%機率:{prob_str} | 停利:{tp_str} | 停損:{sl_str}\n"
            f"  投信10日:{c['trust']} | Stage1:{'✅' if c['s1_pass'] else '❌'}"
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
- 三大型態: A=資金流入+整理+強買(52.6%), B=點火+過熱+強買(52.4%), C=VRI過熱+強買(59.0%)
- EV期望值: 基於歷史回測的期望報酬率
- T值: 統計顯著性（>2.0具顯著性）
- 推算上漲10%機率: 基於型態勝率+SlopeZ+VRI/PVO綜合推算

═══════════════════════════════════════
【分析任務】
═══════════════════════════════════════

**一、大盤風險評估**（200字以上）
- 當前市場結構：動能、趨勢、情緒三維分析
- 5日與20日斜率背離或共振的含義
- 熊/牛/震蕩機率對今日操作的影響
- 適合的持倉水位建議（含理由）

**二、最終候選精選**（從⭐最終候選中選出1~3檔，每檔200字以上）
對每一檔請分析：
1. 技術面研判：Slope Z趨勢強度、是否突破關鍵位
2. 資金面研判：PVO狀態解讀、投信籌碼動向
3. 統計優勢：命中型態、歷史勝率、EV期望值、T值顯著性
4. 波動持續性：PVO/VRI波動率代表的能量穩定度
5. 推算上漲10%機率解讀：為何是這個數字、信心區間
6. 操作建議：停利/停損邏輯、進場時機

**三、風險提示**（100-150字）
- 今日主要風險因子（指標面）
- 需要迴避的情境

**格式要求**：條列與段落並用，數字說話，每個論點需引用具體指標數值。"""


# ===========================================================================
# [新增1] 個股 Gemini 分析 Prompt
# ===========================================================================
def build_stock_gemini_prompt(sym: str, res: dict, market_sentiment: dict, market: str) -> str:
    """針對單一個股的深度 Gemini 分析 Prompt，以大盤為分析架構主軸"""
    dec    = res.get("decision", {})
    s1     = res.get("stage1", {})
    s2     = res.get("stage2", {})
    df_ind = res.get("indicator_df")
    trust  = res.get("trust", {})

    pat       = classify_pattern(dec)
    pat_codes = "+".join([p["code"] for p in pat["patterns"]]) if pat["patterns"] else "無"
    pat_desc  = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "無型態"
    pvo_ratio = calc_pvo_ratio(df_ind)
    vri_ratio = calc_vri_ratio(df_ind)
    upside    = calc_upside_stats(dec, df_ind, pat)
    is_final  = is_final_candidate(dec, df_ind)
    signal_dt = get_signal_date(dec, df_ind)

    ev_val = s2.get("ev", None)
    ev_str = f"{ev_val:+.2f}%" if isinstance(ev_val, (int, float)) else "N/A"
    t_stat = s2.get("t_stat", None)
    t_str  = f"{t_stat:.2f}" if t_stat is not None else "N/A"
    path   = translate_path(str(s2.get("path", "N/A")))
    trust_net = trust.get("trust_net_10d", None)
    trust_str = f"{trust_net:+,.0f}張" if trust_net is not None else "N/A"

    # 大盤資訊
    label   = market_sentiment.get("label", "不明") if market_sentiment else "不明"
    slope5  = market_sentiment.get("slope_5d", 0) if market_sentiment else 0
    slope20 = market_sentiment.get("slope_20d", 0) if market_sentiment else 0
    bear_p  = market_sentiment.get("bear", 33) if market_sentiment else 33
    bull_p  = market_sentiment.get("bull", 33) if market_sentiment else 33
    neu_p   = market_sentiment.get("neutral", 34) if market_sentiment else 34
    mkt_label = "台股" if market == "TW" else "美股"

    prob_str = f"{upside.get('prob10')}%" if upside.get('prob10') else "N/A"
    tp_str   = f"{upside.get('tp')}" if upside.get('tp') else "N/A"
    sl_str   = f"{upside.get('sl')}" if upside.get('sl') else "N/A"
    tp_pct   = f"{upside.get('tp_pct')}%" if upside.get('tp_pct') else "N/A"
    sl_pct   = f"{upside.get('sl_pct')}%" if upside.get('sl_pct') else "N/A"

    return f"""你是一位資深量化交易分析師，請以「大盤環境為主軸」分析個股 {sym} 的投資機會。

═══════════════════════════════════════
【大盤環境（分析主軸）】{mkt_label}
═══════════════════════════════════════
大盤情緒: {label}
熊市機率: {bear_p}% | 震盪機率: {neu_p}% | 牛市機率: {bull_p}%
5日動能斜率: {slope5:+.4f} | 20日趨勢斜率: {slope20:+.4f}

═══════════════════════════════════════
【個股數據】{sym}
═══════════════════════════════════════
訊號觸發日: {signal_dt}
現價: {dec.get('close', 0):.2f} | 方向: {dec.get('direction','—')} | 操作: {dec.get('action','—')} {dec.get('signal_level','')}
PVO: {dec.get('pvo', 0):+.2f} ({dec.get('pvo_status','')}) | VRI: {dec.get('vri', 0):.1f} ({dec.get('vri_status','')})
Slope: {dec.get('slope', 0):+.3f}% | SlopeZ: {dec.get('slope_z', 0):+.2f} | Score: {dec.get('score', 0)}
PVO波動率: {pvo_ratio:.0%} | VRI波動率: {vri_ratio:.0%}
高勝率型態: {pat_codes} ({pat_desc})
最高勝率10%: {pat.get('best_win10', 0):.1f}%
EV期望值: {ev_str} | T值: {t_str} | 路徑: {path}
投信10日: {trust_str} | Stage1: {'✅通過' if s1.get('pass') else '❌未通過'}
最終候選: {'⭐是' if is_final else '否'}
推算上漲10%機率: {prob_str}
停利目標: {tp_str} (+{tp_pct}) | 停損: {sl_str} (-{sl_pct})

═══════════════════════════════════════
【分析要求】請完成以下分析（不限字數，越詳細越好）：
═══════════════════════════════════════

**一、大盤環境對 {sym} 的影響**
- 當前大盤情緒（{label}）對此個股的利弊分析
- 5日/20日斜率與個股走勢的共振或背離
- 在此大盤條件下操作 {sym} 的勝算評估

**二、個股技術與資金分析**
- PVO、VRI 指標當前狀態深度解讀
- SlopeZ={dec.get('slope_z', 0):+.2f} 的趨勢強度含義
- 命中型態（{pat_codes}）的歷史統計意義與操作含義
- PVO波動率{pvo_ratio:.0%}、VRI波動率{vri_ratio:.0%}代表的持續性判斷
- 投信籌碼變化含義（若有）

**三、統計優勢評估**
- EV期望值 {ev_str} 解讀（與市場平均比較）
- T值 {t_str} 的統計顯著性判斷
- 推算上漲10%機率 {prob_str} 的依據與合理性

**四、操作策略建議**
- 進場時機建議（何時入場、何種條件確認）
- 停利目標 {tp_str} 的合理性分析（上方壓力區）
- 停損設定 {sl_str} 的合理性分析（下方支撐區）
- 資金配置建議（相對於大盤風險的倉位）
- 持有天數預估（短線/中線）

**五、風險警示**
- 此股目前最大的指標面風險
- 若大盤轉弱，此股的應對策略

**格式要求**：每個論點必須引用具體數字，禁止泛泛而談，分析需專業且可操作。"""


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
    """渲染股票卡片 — 含訊號日期、停利停損、上漲機率"""
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
    s1_pass    = "✅" if s1.get("pass") else "❌"
    s2_pass    = "✅" if s2.get("pass") else "❌"
    health_icon= "✅" if health.get("pass") else "⚠️"

    ev_raw  = s2.get("ev", None)
    t_stat  = s2.get("t_stat", None)
    path    = translate_path(str(s2.get("path", "---")))
    ev_str  = f"+{ev_raw:.2f}%" if isinstance(ev_raw, (int, float)) and ev_raw > 0 else (
              f"{ev_raw:.2f}%" if isinstance(ev_raw, (int, float)) else "N/A")
    t_str   = f"{t_stat:.2f}" if t_stat is not None else "N/A"
    t_sig   = "✅顯著" if (t_stat is not None and abs(t_stat) >= 2.0) else ("⚠️" if t_stat is not None else "N/A")

    pat_desc_list = [p["desc"] for p in pat["patterns"]] if pat["patterns"] else []
    pat_desc_html = " ｜ ".join(pat_desc_list) if pat_desc_list else "無型態"

    trust_net_10d = trust.get("trust_net_10d", None)
    if trust_net_10d is not None:
        trust_color = "#059669" if trust_net_10d > 0 else ("#dc2626" if trust_net_10d < 0 else "#64748b")
        trust_label = f"近10日投信: <span style='color:{trust_color};font-weight:700;'>{trust_net_10d:+,.0f} 張</span>"
    else:
        trust_label = ""

    is_final = is_final_candidate(dec, df_ind)

    # [新增3] 訊號日期
    signal_dt = get_signal_date(dec, df_ind)

    # [新增2] 停利停損機率
    upside = calc_upside_stats(dec, df_ind, pat)
    prob10 = upside.get("prob10")
    tp_val = upside.get("tp")
    sl_val = upside.get("sl")
    tp_pct = upside.get("tp_pct")
    sl_pct = upside.get("sl_pct")
    prob_str = f"{prob10}%" if prob10 else "N/A"
    tp_str_d = f"{tp_val:.2f}" if tp_val else "N/A"
    sl_str_d = f"{sl_val:.2f}" if sl_val else "N/A"

    if show_final_badge and is_final:
        _border_color = "#1a56db"; _bg = "linear-gradient(135deg,#eff6ff,#ffffff)"
    elif direction == "做多":
        _border_color = "#059669"; _bg = "#ffffff"
    elif direction == "做空":
        _border_color = "#dc2626"; _bg = "#ffffff"
    else:
        _border_color = "#0891b2"; _bg = "#ffffff"

    badge      = get_badge(direction)
    pattern_html = render_pattern_badges(pat["patterns"])

    pvo_color  = "#059669" if pvo > 10 else ("#0891b2" if pvo > 0 else "#dc2626")
    vri_color  = "#059669" if 40 <= vri <= 75 else ("#dc2626" if vri > 90 else "#d97706")
    slope_color= "#059669" if slope > 0 else "#dc2626"
    ev_color   = "#059669" if (isinstance(ev_raw, (int, float)) and ev_raw > 3) else "#d97706"
    t_color    = "#059669" if (t_stat is not None and abs(t_stat) >= 2.0) else "#d97706"
    prob_color = "#059669" if (prob10 and prob10 >= 50) else ("#d97706" if (prob10 and prob10 >= 35) else "#dc2626")
    mkt_tag    = "🇹🇼" if market == "TW" else "🇺🇸"

    vri_ratio = calc_vri_ratio(df_ind)
    pvo_ratio = calc_pvo_ratio(df_ind)
    vri_ratio_color = "#059669" if vri_ratio > 0.6 else ("#d97706" if vri_ratio > 0.4 else "#dc2626")
    pvo_ratio_color = "#059669" if pvo_ratio > 0.6 else ("#d97706" if pvo_ratio > 0.4 else "#dc2626")

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

    _row = "display:flex;gap:16px;flex-wrap:wrap;margin-top:5px;"
    _lbl = "font-size:0.8rem;color:#64748b;"

    st.markdown(f"""
    <div style="background:{_bg};border:1px solid #e2e8f0;border-left:4px solid {_border_color};
                border-radius:10px;padding:12px 16px;margin-bottom:10px;
                box-shadow:0 1px 4px rgba(30,41,59,0.07);">
      <!-- 標頭 -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:1.05rem;font-weight:800;color:#1a56db;">{mkt_tag} {sym}</span>
        <span style="font-size:1.05rem;font-weight:600;font-family:monospace;">
            {"NT$" if market == "TW" else "$"} {close_px:,.2f}
        </span>
        <span>{badge} {final_badge_html}</span>
      </div>
      <!-- [新增3] 訊號觸發日期 -->
      <div style="margin-bottom:4px;">
        <span style="display:inline-block;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;background:#fef9c3;color:#713f12;border:1px solid #fde047;">
          📅 條件滿足日期: {signal_dt}
        </span>
      </div>
      <!-- 高勝率型態徽章 -->
      {pattern_html}
      <!-- 型態說明 -->
      <div style="font-size:0.8rem;color:#7c3aed;font-style:italic;margin:2px 0 6px 0;">
        📝 型態說明: {pat_desc_s}
      </div>
      <!-- AI判定 -->
      <div style="font-size:0.82rem;color:#0891b2;font-weight:600;margin-bottom:4px;">
        AI判定: <b>{action_s}</b>&nbsp;{sig_level_s}&nbsp;｜&nbsp;前次: {last_action_s}
      </div>
      <!-- PVO/VRI狀態 -->
      <div style="{_row}">
        <span style="{_lbl}">狀態:&nbsp;<b style="color:#1e293b;">{pvo_status_s}</b>&nbsp;/&nbsp;<b style="color:#1e293b;">{vri_status_s}</b></span>
      </div>
      <!-- 核心指標 -->
      <div style="{_row}">
        <span style="{_lbl}">PVO:&nbsp;<b style="color:{pvo_color};">{pvo:+.2f}</b></span>
        <span style="{_lbl}">VRI:&nbsp;<b style="color:{vri_color};">{vri:.1f}</b></span>
        <span style="{_lbl}">Slope:&nbsp;<b style="color:{slope_color};">{slope:+.3f}%</b></span>
        <span style="{_lbl}">Slope Z:&nbsp;<b style="color:#1a56db;">{slope_z:+.2f}</b></span>
      </div>
      <!-- 波動率 -->
      <div style="{_row}">
        <span style="{_lbl}">VRI波動率:&nbsp;<b style="color:{vri_ratio_color};">{vri_ratio:.0%}</b>
            <small style="color:#94a3b8;">（&gt;60%持續性強）</small></span>
        <span style="{_lbl}">PVO波動率:&nbsp;<b style="color:{pvo_ratio_color};">{pvo_ratio:.0%}</b>
            <small style="color:#94a3b8;">（&gt;60%持續性強）</small></span>
      </div>
      <!-- EV、T值 -->
      <div style="{_row}">
        <span style="{_lbl}">
          💰 EV期望值:&nbsp;<b style="color:{ev_color};font-size:0.9rem;">{ev_str}</b>
        </span>
        <span style="{_lbl}">
          📊 T值:&nbsp;<b style="color:{t_color};">{t_str}</b>
          <span style="color:{t_color};font-size:0.75rem;margin-left:2px;">{t_sig}</span>
        </span>
      </div>
      <!-- [新增2] 上漲機率 + 停利停損 -->
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;margin-top:6px;">
        <div style="{_row}">
          <span style="{_lbl}">
            🎯 上漲10%機率:&nbsp;<b style="color:{prob_color};font-size:0.95rem;">{prob_str}</b>
          </span>
          <span style="{_lbl}">
            🟢 停利目標:&nbsp;<b style="color:#059669;font-size:0.9rem;">{tp_str_d}</b>
            <small style="color:#94a3b8;">(+{tp_pct if tp_pct else '—'}%)</small>
          </span>
          <span style="{_lbl}">
            🔴 停損價位:&nbsp;<b style="color:#dc2626;font-size:0.9rem;">{sl_str_d}</b>
            <small style="color:#94a3b8;">(-{sl_pct if sl_pct else '—'}%)</small>
          </span>
        </div>
      </div>
      <!-- Stage / Health / Path -->
      <div style="{_row};margin-top:6px;">
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
        st.caption("© 2026 資源法 AI 戰情室 v2.3")


# ===========================================================================
# 主頁面
# ===========================================================================
def main():
    inject_color_fix()
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

    # ── 大盤 Gemini 深度分析區 ──
    ai_col1, ai_col2 = st.columns([1, 4])
    with ai_col1:
        ai_btn = st.button("🤖 Gemini 深度分析（大盤）", use_container_width=True)
    with ai_col2:
        if st.session_state.ai_summary:
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
            with st.spinner("🤖 Gemma 深度分析中（請稍候）..."):
                summary = call_gemini(prompt, st.session_state.gemini_api_key)
            st.session_state.ai_summary = summary
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    # [新增1] 個股 Gemini 分析區（大盤 Gemini 按鈕下方）
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 個股 Gemini 深度分析")
    st.markdown(
        "<small style='color:#64748b;'>以大盤環境為分析主軸，對單一個股進行深度量化解讀。"
        "若個股不在掃描名單內，將顯示「沒有資料」。</small>",
        unsafe_allow_html=True
    )

    stock_col1, stock_col2, stock_col3 = st.columns([2, 1, 3])
    with stock_col1:
        stock_query = st.text_input(
            "輸入個股代號",
            value=st.session_state.stock_ai_query,
            placeholder="例：2330 或 NVDA",
            label_visibility="collapsed"
        )
    with stock_col2:
        stock_ai_btn = st.button("🔬 個股分析", use_container_width=True)
    with stock_col3:
        st.markdown(
            "<span style='color:#64748b;font-size:0.82rem;'>📌 輸入代號後點擊「個股分析」，"
            "系統將以大盤環境為主軸進行深度分析</span>",
            unsafe_allow_html=True
        )

    if stock_ai_btn and stock_query:
        query_sym = stock_query.strip().upper()
        st.session_state.stock_ai_query = query_sym

        # 在掃描結果中尋找（含模糊比對：去掉 .TW/.TWO 後綴）
        found_key = None
        results_now = st.session_state.scan_results

        if query_sym in results_now:
            found_key = query_sym
        else:
            # 模糊比對
            for k in results_now.keys():
                k_clean = k.replace(".TW", "").replace(".TWO", "").upper()
                q_clean = query_sym.replace(".TW", "").replace(".TWO", "").upper()
                if k_clean == q_clean:
                    found_key = k
                    break

        if not results_now:
            st.warning("⚠️ 請先執行全盤掃描，才能進行個股分析。")
        elif found_key is None:
            st.markdown(
                f"""<div style="background:#fff7ed;border:1px solid #fdba74;border-left:4px solid #f97316;
                     border-radius:10px;padding:14px 18px;margin:8px 0;">
                  <b style="color:#ea580c;">🚫 沒有資料</b>
                  <div style="color:#78350f;margin-top:4px;font-size:0.9rem;">
                    「{query_sym}」不在目前的掃描名單內。<br>
                    請將該代號加入側欄的觀察名單並重新掃描，或確認代號格式正確（台股請輸入純數字代號，美股請輸入英文代號）。
                  </div>
                </div>""",
                unsafe_allow_html=True
            )
            st.session_state.stock_ai_result = ""
        else:
            res = results_now[found_key]
            if res.get("error"):
                st.error(f"「{found_key}」數據異常：{res.get('error')}")
                st.session_state.stock_ai_result = ""
            else:
                market = st.session_state.active_market
                sent   = (st.session_state.market_sentiment_tw if market == "TW"
                          else st.session_state.market_sentiment_us)
                prompt = build_stock_gemini_prompt(found_key, res, sent, market)
                with st.spinner(f"🤖 正在對 {found_key} 進行深度分析（請稍候）..."):
                    result = call_gemini(prompt, st.session_state.gemini_api_key)
                st.session_state.stock_ai_result = result
                st.session_state.stock_ai_query  = found_key

    # 顯示個股分析結果
    if st.session_state.stock_ai_result:
        import html as _html2
        _safe2 = _html2.escape(st.session_state.stock_ai_result).replace('\n', '<br>')
        query_label = st.session_state.stock_ai_query or "個股"
        st.markdown(
            f"""<div class="stock-ai-box">
              <div style="font-weight:800;color:#059669;margin-bottom:10px;font-size:1.05rem;">
                🔬 {query_label} 個股深度分析報告
              </div>
              {_safe2}
            </div>""",
            unsafe_allow_html=True
        )

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
                    df_ind_t1 = res.get("indicator_df")
                    pat_t1 = classify_pattern(dec)
                    upside_t1 = calc_upside_stats(dec, df_ind_t1, pat_t1)

                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    col1.metric("現價", f"{dec.get('close',0):,.2f}")
                    col2.metric("PVO", f"{dec.get('pvo',0):+.2f}",
                                delta="↑" if dec.get("pvo", 0) > 0 else "↓")
                    col3.metric("VRI", f"{dec.get('vri',0):.1f}")
                    col4.metric("Slope Z", f"{dec.get('slope_z',0):+.2f}")
                    ev_val = s2.get('ev', None)
                    col5.metric("EV 期望值",
                                f"{ev_val:+.2f}%" if isinstance(ev_val, (int, float)) else "N/A")
                    t_val = s2.get('t_stat', None)
                    col6.metric("T 值",
                                f"{t_val:.2f}" if t_val is not None else "N/A",
                                delta="顯著✅" if (t_val is not None and abs(t_val) >= 2.0) else None)
                    prob_t1 = upside_t1.get("prob10")
                    col7.metric("上漲10%機率",
                                f"{prob_t1}%" if prob_t1 else "N/A")

                    # 停利停損 metrics
                    col_tp, col_sl, _ = st.columns([1, 1, 2])
                    tp_t1 = upside_t1.get("tp")
                    sl_t1 = upside_t1.get("sl")
                    col_tp.metric("🟢 停利目標",
                                  f"{tp_t1:.2f}" if tp_t1 else "N/A",
                                  delta=f"+{upside_t1.get('tp_pct')}%" if upside_t1.get('tp_pct') else None)
                    col_sl.metric("🔴 停損價位",
                                  f"{sl_t1:.2f}" if sl_t1 else "N/A",
                                  delta=f"-{upside_t1.get('sl_pct')}%" if upside_t1.get('sl_pct') else None,
                                  delta_color="inverse")

    # ================================================================
    # Tab 2: 數據與排名
    # [新增2] 顯示所有個股 EV/T值 + 上漲10%機率/停利/停損
    # [新增3] 顯示條件滿足日期
    # ================================================================
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
                ⭐ 最終決策候選條件：命中任一型態 + PVO波動率&gt;60% + VRI波動率&gt;60%
                ｜ 📅 條件滿足日期：做多訊號觸發當天日期
            </div>
            </div>
            """, unsafe_allow_html=True)

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                show_all = st.checkbox("顯示全部（含未通過）", value=False)
            with col_f2:
                sort_by = st.selectbox("排序依據",
                    ["最佳勝率10%", "上漲10%機率", "Slope Z", "VRI", "PVO", "EV期望值", "Score"], index=0)
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

                pat_codes = " + ".join([p["code"] for p in pat["patterns"]]) if pat["patterns"] else "—"
                pat_desc_tab = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "—"

                ev_raw_tab = s2.get("ev", None)
                t_stat_tab = s2.get("t_stat", None)

                is_fin = is_final_candidate(dec, df_ind_tab)

                # [新增2] 上漲機率/停利/停損
                upside_tab = calc_upside_stats(dec, df_ind_tab, pat)
                prob10_tab = upside_tab.get("prob10")
                tp_tab     = upside_tab.get("tp")
                sl_tab     = upside_tab.get("sl")
                tp_pct_tab = upside_tab.get("tp_pct")
                sl_pct_tab = upside_tab.get("sl_pct")

                # [新增3] 訊號日期
                sig_date_tab = get_signal_date(dec, df_ind_tab)

                table_rows.append({
                    "代號":        sym,
                    "市場":        "🇹🇼" if res.get("market") == "TW" else "🇺🇸",
                    "訊號日期":    sig_date_tab,          # [新增3]
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
                    "上漲10%機率": prob10_tab,            # [新增2]
                    "停利目標":    tp_tab,                # [新增2]
                    "停利%":       tp_pct_tab,            # [新增2]
                    "停損價位":    sl_tab,                # [新增2]
                    "停損%":       sl_pct_tab,            # [新增2]
                    "路徑":        translate_path(str(s2.get("path", "N/A"))),
                    "投信10日":    trust.get("trust_net_10d", None),
                    "S1":          "✅" if s1.get("pass") else "❌",
                    "健康":        "✅" if h.get("pass") else "⚠️",
                    "PVO狀態":     dec.get("pvo_status", ""),
                    "VRI狀態":     dec.get("vri_status", ""),
                })

            if table_rows:
                sort_col_map = {
                    "最佳勝率10%": "最佳勝率10%",
                    "上漲10%機率": "上漲10%機率",
                    "Slope Z": "Slope Z", "VRI": "VRI", "PVO": "PVO",
                    "EV期望值": "EV期望值%", "Score": "Score",
                }
                df_table = pd.DataFrame(table_rows)
                sort_col = sort_col_map.get(sort_by, "最佳勝率10%")
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
                        "訊號日期":    st.column_config.TextColumn(
                            help="做多條件滿足的日期（訊號觸發日）",
                            width="small"
                        ),
                        "現價":        st.column_config.NumberColumn(format="%.2f"),
                        "PVO":         st.column_config.NumberColumn(format="%.2f"),
                        "VRI":         st.column_config.NumberColumn(format="%.1f"),
                        "VRI波動率":   st.column_config.NumberColumn(format="%.0%", help="近20日VRI>40天數/20，>60%代表持續性強"),
                        "PVO波動率":   st.column_config.NumberColumn(format="%.0%", help="近20日PVO>0天數/20，>60%代表持續性強"),
                        "Slope Z":     st.column_config.NumberColumn(format="%.2f"),
                        "EV期望值%":   st.column_config.NumberColumn(format="%.2f", help="歷史回測期望報酬率(%)"),
                        "T值":         st.column_config.NumberColumn(format="%.2f", help="統計顯著性，≥2.0具顯著性"),
                        "最佳勝率10%": st.column_config.NumberColumn(format="%.1f%%", help="命中型態的最高10日勝率"),
                        "上漲10%機率": st.column_config.NumberColumn(  # [新增2]
                            format="%.1f%%",
                            help="基於高勝率型態+SlopeZ+VRI/PVO綜合推算的上漲10%機率"
                        ),
                        "停利目標":    st.column_config.NumberColumn(  # [新增2]
                            format="%.2f",
                            help="建議停利目標價位"
                        ),
                        "停利%":       st.column_config.NumberColumn(  # [新增2]
                            format="%.1f%%",
                            help="停利目標距現價漲幅"
                        ),
                        "停損價位":    st.column_config.NumberColumn(  # [新增2]
                            format="%.2f",
                            help="建議停損價位（基於ATR×2）"
                        ),
                        "停損%":       st.column_config.NumberColumn(  # [新增2]
                            format="%.1f%%",
                            help="停損幅度"
                        ),
                        "投信10日":    st.column_config.NumberColumn(
                            label="投信10日(張)", format="%+,.0f",
                            help="近10個交易日投信累積買賣超（正=買超/負=賣超）"
                        ),
                        "高勝率型態":  st.column_config.TextColumn(help="A=資金流入+整理+強買 | B=點火+過熱+強買 | C=VRI過熱+強買"),
                        "型態說明":    st.column_config.TextColumn(help="型態操作特性說明"),
                        "最終候選":    st.column_config.TextColumn(help="⭐=命中型態+PVO波動率>60%+VRI波動率>60%"),
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
    # Tab 3: 決策戰情室
    # [新增3] 顯示條件滿足日期
    # ================================================================
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
                    dec = res.get("decision", {})
                    s2  = res.get("stage2", {})
                    df_ind = res.get("indicator_df")
                    pat = classify_pattern(dec)
                    pat_labels = " ｜ ".join([p["label"] for p in pat["patterns"]])
                    win_str = f"最高勝率10%: {pat['best_win10']:.1f}%"
                    vri_ratio = calc_vri_ratio(df_ind)
                    pvo_ratio = calc_pvo_ratio(df_ind)
                    path_zh = translate_path(str(s2.get("path","N/A")))

                    ev_raw = s2.get("ev", None)
                    t_raw  = s2.get("t_stat", None)
                    ev_display = f"{ev_raw:+.2f}%" if isinstance(ev_raw, (int, float)) else "N/A"
                    t_display  = f"{t_raw:.2f}" if t_raw is not None else "N/A"
                    t_sig_str  = " ✅顯著" if (t_raw is not None and abs(t_raw) >= 2.0) else ""
                    pat_descs  = " | ".join([p["desc"] for p in pat["patterns"]]) if pat["patterns"] else "—"

                    ev_c = "#059669" if (isinstance(ev_raw, (int, float)) and ev_raw > 3) else "#d97706"
                    t_c  = "#059669" if (t_raw is not None and abs(t_raw) >= 2.0) else "#d97706"

                    mkt_tag = "🇹🇼" if res.get("market") == "TW" else "🇺🇸"

                    # [新增2] 上漲機率/停利/停損
                    upside_fp = calc_upside_stats(dec, df_ind, pat)
                    prob10_fp = upside_fp.get("prob10")
                    tp_fp     = upside_fp.get("tp")
                    sl_fp     = upside_fp.get("sl")
                    tp_pct_fp = upside_fp.get("tp_pct")
                    sl_pct_fp = upside_fp.get("sl_pct")
                    prob_c    = "#059669" if (prob10_fp and prob10_fp >= 50) else "#d97706"

                    # [新增3] 訊號日期
                    sig_dt_fp = get_signal_date(dec, df_ind)

                    st.markdown(f"""
                    <div style="background:#eff6ff;border:1px solid #93c5fd;border-left:4px solid #1a56db;
                         border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <b style="color:#1e40af;font-size:1rem;">{mkt_tag} {sym}</b>
                            <span style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.78rem;font-weight:700;background:#fef3c7;color:#92400e;border:1px solid #fbbf24;">{win_str}</span>
                        </div>
                        <!-- [新增3] 訊號日期 -->
                        <div style="margin:4px 0;">
                          <span style="display:inline-block;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;background:#fef9c3;color:#713f12;border:1px solid #fde047;">
                            📅 條件滿足日期: {sig_dt_fp}
                          </span>
                        </div>
                        <div style="font-size:0.82rem;color:#475569;margin-top:4px;">{pat_labels}</div>
                        <div style="font-size:0.79rem;color:#7c3aed;font-style:italic;margin:2px 0;">📝 {pat_descs}</div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;font-size:0.83rem;color:#475569;">
                            <span>Slope Z: <b style="color:#1a56db">{dec.get('slope_z',0):+.2f}</b></span>
                            <span>VRI: <b>{dec.get('vri',0):.1f}</b></span>
                            <span>PVO: <b>{dec.get('pvo',0):+.2f}</b></span>
                            <span>💰 EV: <b style="color:{ev_c}">{ev_display}</b></span>
                            <span>📊 T值: <b style="color:{t_c}">{t_display}{t_sig_str}</b></span>
                            <span>路徑: {path_zh}</span>
                        </div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:4px;font-size:0.83rem;color:#475569;">
                            <span>VRI波動率: <b style="color:{'#059669' if vri_ratio > 0.6 else '#d97706'}">{vri_ratio:.0%}</b></span>
                            <span>PVO波動率: <b style="color:{'#059669' if pvo_ratio > 0.6 else '#d97706'}">{pvo_ratio:.0%}</b></span>
                        </div>
                        <!-- [新增2] 上漲機率/停利/停損 -->
                        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:6px 12px;margin-top:6px;display:flex;gap:20px;flex-wrap:wrap;">
                            <span style="font-size:0.83rem;">🎯 上漲10%機率: <b style="color:{prob_c}">{prob10_fp if prob10_fp else 'N/A'}%</b></span>
                            <span style="font-size:0.83rem;">🟢 停利: <b style="color:#059669">{tp_fp if tp_fp else 'N/A'}</b> <small>(+{tp_pct_fp if tp_pct_fp else '—'}%)</small></span>
                            <span style="font-size:0.83rem;">🔴 停損: <b style="color:#dc2626">{sl_fp if sl_fp else 'N/A'}</b> <small>(-{sl_pct_fp if sl_pct_fp else '—'}%)</small></span>
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

            key_count = sum(1 for s, r in bull_stocks
                           if classify_pattern(r.get("decision", {}))["is_key_pattern"])
            final_count = sum(1 for s, r in bull_stocks
                              if is_final_candidate(r.get("decision", {}), r.get("indicator_df")))

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

            # ── 做空 / 觀望 ──
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
        st.markdown("### ⚠️ 數據清洗日誌")
        if st.session_state.all_warnings:
            for w in st.session_state.all_warnings:
                st.warning(w)
        else:
            st.success("✅ 無異常數據警告")


if __name__ == "__main__":
    main()
