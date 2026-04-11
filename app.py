"""
0410-V1.py — 資源法 AI 戰情室（融合版）
整合：
  - app.py 原 Streamlit 四 Tab 架構（全部保留）
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
