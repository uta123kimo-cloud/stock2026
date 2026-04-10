"""
engine_21.py — 資源法 2.1 核心運算引擎
技術指標全部手刻，不依賴 pandas_ta
支援台股 (.TW/.TWO) 與美股雙軌架構
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import logging
import warnings
import json
import os
from datetime import datetime, timedelta

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
import os
import pandas as pd
from datetime import datetime

# ==========================================
# 程式碼 A：讀取 Token (放在最上面)
# ==========================================
token = os.getenv('FINMIND_TOKEN')

# ==========================================
# 程式碼 B：定義「存檔函式」 (放在 main 上面)
# ==========================================
def save_v12_results(df):
    """
    這是一個定義，放在這裡不會立刻執行。
    只有當我們在 main 裡面呼叫它時，它才會運作。
    """
    if not os.path.exists('storage'):
        os.makedirs('storage')
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    file_path = f"storage/signals_{current_time}.parquet"
    
    df.to_parquet(file_path, index=False)
    print(f"✅ 數據已成功存入: {file_path}")

# ==========================================
# 主程式入口 (MAIN)
# ==========================================
if __name__ == "__main__":
    print("🚀 啟動 V12.1 核心運算...")
    
    # 1. 執行你的 V12.1 運算邏輯 (假設結果存成 final_df)
    # final_df = run_v12_logic(token) 
    
    # 2. 運算結束後，呼叫上面的 B 函式來存檔
    # 這裡就是真正執行「程式碼 B」的地方
    if 'final_df' in locals():
        save_v12_results(final_df)
    else:
        print("❌ 運算失敗，沒有產出數據")
# ===========================================================================
# 環境變數讀取（機密金鑰）
# ===========================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 優先序: Streamlit Secrets → .env → 系統環境變數
def _get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

FINMIND_TOKEN: str = _get_secret("FINMIND_TOKEN")

# ===========================================================================
# FinMind API 設定
# ===========================================================================
FINMIND_URL       = "https://api.finmindtrade.com/api/v4/data"
DATA_FETCH_START  = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")


# ===========================================================================
# 投信買賣超快取 — InstCache
# 使用 FINMIND_TOKEN 從 FinMind 拉取台股投信進出數據
# ===========================================================================
class InstCache:
    """
    批量預載投信買賣超資料，避免逐支呼叫 API 產生速率限制。
    用法:
        _INST_CACHE.batch_init(yahoo_symbol_list)
        df = _INST_CACHE.get("3037")   # 回傳 DataFrame(index=date, col=trust_net) 或 None
    """

    def __init__(self):
        self._cache: dict = {}

    def batch_init(self, syms: list):
        """預載多支股票的投信資料（去除 .TW / .TWO 後綴）"""
        print(f"  預載投信（{len(syms)} 支）...", flush=True)
        ok = 0
        for s in syms:
            sid = s.replace(".TWO", "").replace(".TW", "")
            if sid in self._cache:
                ok += 1
                continue
            df = self._fetch(sid)
            if df is not None:
                self._cache[sid] = df
                ok += 1
        print(f"  投信：{ok}/{len(syms)} 支載入成功")

    def _fetch(self, sid: str) -> pd.DataFrame | None:
        """
        呼叫 FinMind API 取得單支股票投信買賣超。
        回傳 DataFrame(index=date, columns=[trust_net]) 或 None。
        trust_net = 投信買超 - 投信賣超（張數，正=買、負=賣）
        """
        if not FINMIND_TOKEN:
            return None
        try:
            r = requests.get(
                FINMIND_URL,
                params={
                    "dataset":   "TaiwanStockInstitutionalInvestorsBuySell",
                    "data_id":   sid,
                    "start_date": DATA_FETCH_START,
                    "token":     FINMIND_TOKEN,
                },
                timeout=15,
            )
            d = r.json()
            if d.get("status") != 200:
                return None

            df = pd.DataFrame(d["data"])
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"])

            # 只取「投信」那行（中英文名稱皆相容）
            filt = df[df["name"].isin(["投信", "Investment_Trust"])].copy()
            if filt.empty:
                return None

            filt["trust_net"] = (
                pd.to_numeric(filt.get("buy",  "0"), errors="coerce").fillna(0)
                - pd.to_numeric(filt.get("sell", "0"), errors="coerce").fillna(0)
            )
            return (
                filt[["date", "trust_net"]]
                .set_index("date")
                .sort_index()
            )
        except Exception:
            return None

    def get(self, sid: str) -> pd.DataFrame | None:
        """取得已快取的投信數據；sid 不含 .TW/.TWO"""
        return self._cache.get(sid, None)

    def get_recent_net(self, sid: str, days: int = 10) -> float:
        """
        回傳最近 N 個交易日的投信累積買賣超（張），
        可用於作為輔助評分因子。
        """
        df = self.get(sid)
        if df is None or df.empty:
            return 0.0
        return float(df["trust_net"].iloc[-days:].sum())


# 全域單例（模組載入時建立，app.py 可直接 import 使用）
_INST_CACHE = InstCache()


# ===========================================================================
# 台股 / 美股 路徑因子權重 (T = Taiwan, A = America)
# ===========================================================================
FACTOR_WEIGHTS = {
    "TW": {"pvo": 0.20, "vri": 0.20, "slope": 0.60},
    "US": {"pvo": 0.25, "vri": 0.15, "slope": 0.60},
}

# ===========================================================================
# 四層數據防火牆 — Layer 1: 合理性過濾常數
# ===========================================================================
PRICE_MIN = 1.0
PRICE_MAX = 100000.0
VOLUME_MIN = 100
SLOPE_Z_CLIP = 5.0
VRI_CLIP = (0, 100)

# ===========================================================================
# 技術指標手刻 (不依賴 pandas_ta)
# ===========================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """指數移動平均"""
    return series.ewm(span=span, adjust=False).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    """簡單移動平均"""
    return series.rolling(window=window).mean()

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真實波動 ATR (美股因子用)"""
    hi, lo, cl = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs()
    ], axis=1).max(axis=1)
    return _sma(tr, window)

def _rs(series: pd.Series, benchmark: pd.Series, window: int = 20) -> pd.Series:
    """相對強度 RS (美股因子用)"""
    stock_ret = series.pct_change(window)
    bench_ret = benchmark.pct_change(window)
    return (stock_ret - bench_ret) * 100

def _slope_poly(series: pd.Series, window: int = 5) -> float:
    """線性回歸斜率百分比"""
    if len(series) < window:
        return 0.0
    y = series.values[-window:]
    x = np.arange(window)
    slope, _ = np.polyfit(x, y, 1)
    base = y[0] if y[0] != 0 else 1.0
    return (slope / base) * 100

def compute_indicators(df: pd.DataFrame, market: str = "TW",
                        benchmark_close: pd.Series = None) -> pd.DataFrame:
    """
    計算所有技術指標
    market: "TW" = 台股, "US" = 美股
    """
    # PVO (Price/Volume Oscillator on Volume)
    ev12 = _ema(df['Volume'], 12)
    ev26 = _ema(df['Volume'], 26)
    df['PVO'] = ((ev12 - ev26) / (ev26 + 1e-6)) * 100

    # VRI (Volume Ratio Index)
    up_vol = df['Volume'].where(df['Close'].diff() > 0, 0)
    df['VRI'] = (_sma(up_vol, 14) / (_sma(df['Volume'], 14) + 1e-6)) * 100
    df['VRI'] = df['VRI'].clip(*VRI_CLIP)

    # Slope (20日線性斜率 Rolling)
    df['Slope'] = df['Close'].rolling(5).apply(lambda x: _slope_poly(pd.Series(x), 5), raw=False)

    # 美股額外因子: ATR、RS
    if market == "US":
        df['ATR'] = _atr(df, 14)
        df['ATR_pct'] = df['ATR'] / df['Close'] * 100
        if benchmark_close is not None:
            bench_aligned = benchmark_close.reindex(df.index).ffill()
            df['RS'] = _rs(df['Close'], bench_aligned, 20)
        else:
            df['RS'] = 0.0

    # Score (加權綜合評分)
    w = FACTOR_WEIGHTS.get(market, FACTOR_WEIGHTS["TW"])
    df['Score'] = (df['PVO'] * w['pvo']) + (df['VRI'] * w['vri']) + (df['Slope'] * w['slope'])

    return df.dropna(subset=['PVO', 'VRI', 'Slope', 'Score'])


# ===========================================================================
# 四層防火牆 — Layer 2: 原始數據合理性過濾
# ===========================================================================
def sanity_check(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    """
    回傳 (cleaned_df, warnings_list)
    """
    warns = []
    original_len = len(df)

    # 價格合理性
    mask = (df['Close'] >= PRICE_MIN) & (df['Close'] <= PRICE_MAX)
    df = df[mask]
    if len(df) < original_len:
        warns.append(f"[{ticker}] 剔除 {original_len - len(df)} 筆異常價格")

    # 成交量合理性
    df = df[df['Volume'] >= VOLUME_MIN]

    # 移除全 NaN 行
    df = df.dropna(subset=['Close', 'Volume', 'Open', 'High', 'Low'])

    # 檢查 OHLC 邏輯 (High >= Low)
    invalid = df['High'] < df['Low']
    if invalid.any():
        warns.append(f"[{ticker}] 剔除 {invalid.sum()} 筆 High<Low 異常")
        df = df[~invalid]

    return df, warns


# ===========================================================================
# 四層防火牆 — Layer 3: 指標計算邏輯單元測試
# ===========================================================================
def indicator_unit_test(df: pd.DataFrame) -> dict:
    """快速自檢，回傳健康度字典"""
    health = {"pass": True, "issues": []}
    if df['PVO'].isna().mean() > 0.3:
        health["issues"].append("PVO NaN 比例過高")
        health["pass"] = False
    if (df['VRI'] < 0).any() or (df['VRI'] > 100).any():
        health["issues"].append("VRI 超出 [0,100] 範圍")
        health["pass"] = False
    if df['Score'].std() < 1e-6:
        health["issues"].append("Score 標準差接近 0，可能數據凍結")
        health["pass"] = False
    return health


# ===========================================================================
# 代號解析 — 台股自動判 .TW / .TWO，美股直通
# ===========================================================================
def resolve_symbol(symbol: str) -> tuple[str, str]:
    """
    回傳 (yahoo_symbol, market)
    market = "TW" or "US"
    """
    s = str(symbol).strip()
    if s.isdigit():
        for suffix in [".TW", ".TWO"]:
            target = f"{s}{suffix}"
            try:
                t = yf.Ticker(target)
                h = t.history(period="2d")
                if not h.empty:
                    return target, "TW"
            except:
                continue
        return f"{s}.TW", "TW"
    return s, "US"


# ===========================================================================
# 主要數據獲取函數
# ===========================================================================
def fetch_stock_data(symbol: str, start_dt: datetime, end_dt: datetime,
                     benchmark_close: pd.Series = None) -> dict:
    """
    回傳 dict:
      raw_df, indicator_df, market, yahoo_symbol, sanity_warns, health
    """
    yahoo_sym, market = resolve_symbol(symbol)
    result = {
        "symbol": symbol,
        "yahoo_symbol": yahoo_sym,
        "market": market,
        "raw_df": None,
        "indicator_df": None,
        "sanity_warns": [],
        "health": {"pass": False, "issues": ["尚未計算"]},
        "error": None
    }

    try:
        df = yf.download(yahoo_sym, start=start_dt, end=end_dt,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            result["error"] = "無法取得數據"
            return result

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip() for c in df.columns]
        df.index = pd.to_datetime(df.index)

        # Layer 2: 合理性過濾
        df, warns = sanity_check(df.copy(), yahoo_sym)
        result["sanity_warns"] = warns
        result["raw_df"] = df.copy()

        if len(df) < 30:
            result["error"] = f"有效數據不足 (僅 {len(df)} 筆)"
            return result

        # 計算指標
        df = compute_indicators(df.copy(), market=market, benchmark_close=benchmark_close)
        result["indicator_df"] = df

        # Layer 3: 單元測試
        result["health"] = indicator_unit_test(df)

    except Exception as e:
        result["error"] = str(e)

    return result


# ===========================================================================
# 篩選階段 1: 資源法 2.1 能量過濾
# ===========================================================================
def stage1_energy_filter(df: pd.DataFrame, window: int = 60) -> dict:
    """
    篩選條件:
    - Slope Z > 0.5
    - VRI 40-75
    - PVO 向上勾起 (diff > 0)
    回傳評估結果字典
    """
    if df is None or len(df) < window + 5:
        return {"pass": False, "reason": "數據不足"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Slope Z-score
    hist_slopes = df['Slope'].iloc[-window:]
    slope_z = (last['Slope'] - hist_slopes.mean()) / (hist_slopes.std() + 1e-6)
    slope_z = float(np.clip(slope_z, -SLOPE_Z_CLIP, SLOPE_Z_CLIP))

    vri_ok = 40 <= last['VRI'] <= 75
    pvo_hook = last['PVO'] > prev['PVO']

    passed = (slope_z > 0.5) and vri_ok and pvo_hook

    reasons = []
    if slope_z <= 0.5:
        reasons.append(f"Slope Z={slope_z:.2f} ≤ 0.5")
    if not vri_ok:
        reasons.append(f"VRI={last['VRI']:.1f} 不在 40-75 健康帶")
    if not pvo_hook:
        reasons.append("PVO 未向上勾起")

    return {
        "pass": passed,
        "slope_z": slope_z,
        "vri": float(last['VRI']),
        "pvo": float(last['PVO']),
        "pvo_hook": pvo_hook,
        "reason": " | ".join(reasons) if reasons else "通過"
    }


# ===========================================================================
# 篩選階段 2: V12.1 路徑過濾 (讀取 alpha_seeds.json)
# ===========================================================================
def stage2_path_filter(symbol: str, stage1_result: dict,
                        alpha_seeds_path: str = "alpha_seeds.json") -> dict:
    """
    讀取 alpha_seeds.json 做路徑驗證
    若無 JSON，僅依 Stage1 結果返回
    """
    if not stage1_result.get("pass", False):
        return {"pass": False, "ev": None, "path": "N/A", "t_stat": None}

    seeds = {}
    if os.path.exists(alpha_seeds_path):
        try:
            with open(alpha_seeds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 支援 list 格式: [{"Ticker":"3037","Path":"Pure","t_stat":3.2,"EV_Threshold":0.05}]
            if isinstance(data, list):
                seeds = {str(item.get("Ticker", "")): item for item in data}
            elif isinstance(data, dict):
                seeds = data
        except:
            pass

    clean_sym = symbol.split('.')[0]
    seed = seeds.get(clean_sym, seeds.get(symbol, None))

    if seed is None:
        # 無 V12.1 數據，僅依 Stage1
        ev_est = stage1_result["slope_z"] * 0.02
        return {
            "pass": True,
            "ev": round(ev_est * 100, 2),
            "path": "Stage1-Only",
            "t_stat": None,
            "flicker": False
        }

    path = seed.get("Path", "Unknown")
    t_stat = seed.get("t_stat", 0)
    ev_thresh = seed.get("EV_Threshold", 0.05)
    flicker = path.lower() == "flicker"

    passed = (not flicker) and (t_stat >= 2.0) and (ev_thresh >= 0.03)

    ev_est = stage1_result["slope_z"] * 0.02 * (1 + t_stat * 0.1)

    return {
        "pass": passed,
        "ev": round(ev_est * 100, 2),
        "path": path,
        "t_stat": t_stat,
        "flicker": flicker,
        "ev_threshold": ev_thresh
    }


# ===========================================================================
# 決策引擎: 四維建議
# ===========================================================================
def get_decision(df: pd.DataFrame, market: str = "TW", window: int = 60) -> dict:
    """
    回傳最終決策字典
    """
    if df is None or len(df) < window + 3:
        return {"action": "數據不足", "direction": "觀望", "last_action": "---",
                "slope_z": 0, "score_z": 0, "detail": ""}

    last_idx = len(df) - 1
    last = df.iloc[last_idx]
    prev = df.iloc[last_idx - 1]

    hist_slopes = df['Slope'].iloc[max(0, last_idx - window):last_idx + 1]
    hist_scores = df['Score'].iloc[max(0, last_idx - window):last_idx + 1]

    slope_z = float((last['Slope'] - hist_slopes.mean()) / (hist_slopes.std() + 1e-6))
    score_z = float((last['Score'] - hist_scores.mean()) / (hist_scores.std() + 1e-6))
    slope_z = float(np.clip(slope_z, -SLOPE_Z_CLIP, SLOPE_Z_CLIP))

    is_uptrend = (last['Slope'] > prev['Slope'] > df.iloc[last_idx - 2]['Slope'])
    pvo_delta = last['PVO'] - prev['PVO']

    # 方向判定
    if slope_z > 0.6 or (is_uptrend and score_z > 0):
        direction = "做多"
    elif slope_z < -1.0 or (not is_uptrend and score_z < -0.8):
        direction = "做空"
    else:
        direction = "觀望"

    # 細化操作
    if slope_z > 1.5 and pvo_delta > 5:
        action = "強力買進"
    elif slope_z > 0.6:
        action = "波段持有"
    elif is_uptrend:
        action = "準備翻多"
    else:
        action = "觀望整理"

    # 前次行動追蹤
    last_action_date = "---"
    for offset in range(1, 150):
        p_idx = last_idx - offset
        if p_idx < window:
            break
        h_sl = df['Slope'].iloc[p_idx - window:p_idx + 1]
        h_sc = df['Score'].iloc[p_idx - window:p_idx + 1]
        p_sz = (df.iloc[p_idx]['Slope'] - h_sl.mean()) / (h_sl.std() + 1e-6)
        p_scz = (df.iloc[p_idx]['Score'] - h_sc.mean()) / (h_sc.std() + 1e-6)
        p_up = (df.iloc[p_idx]['Slope'] > df.iloc[p_idx - 1]['Slope'] >
                df.iloc[p_idx - 2]['Slope'])

        if p_sz > 0.6 or (p_up and p_scz > 0):
            p_dir = "做多"
        elif p_sz < -1.0 or (not p_up and p_scz < -0.8):
            p_dir = "做空"
        else:
            p_dir = "觀望"

        if p_dir == direction:
            last_action_date = f"{df.index[p_idx].strftime('%m/%d')} {direction}"
        else:
            break

    # PVO / VRI 狀態標籤
    pvo_status = (
        "🔥主力點火" if pvo_delta > 10 else
        "📈資金流入" if last['PVO'] > 0 else
        "😴怠速縮量"
    )
    vri_status = (
        "🌡️健康水溫" if 40 <= last['VRI'] <= 70 else
        "🔴擁擠過熱" if last['VRI'] > 90 else
        "❄️情緒整理"
    )

    # 信號強度
    if slope_z > 1.5:
        signal_level = "🔥主力層級"
    elif slope_z > 0.5:
        signal_level = "💎優質信號"
    elif slope_z > 0:
        signal_level = "📊一般"
    else:
        signal_level = "⚠️弱勢"

    return {
        "direction": direction,
        "action": action,
        "last_action": last_action_date,
        "slope_z": round(slope_z, 3),
        "score_z": round(score_z, 3),
        "pvo_status": pvo_status,
        "vri_status": vri_status,
        "signal_level": signal_level,
        "is_uptrend": is_uptrend,
        "pvo_delta": round(float(pvo_delta), 2),
        "close": round(float(last['Close']), 2),
        "pvo": round(float(last['PVO']), 2),
        "vri": round(float(last['VRI']), 2),
        "slope": round(float(last['Slope']), 3),
        "score": round(float(last['Score']), 2),
        "date": df.index[-1].strftime('%Y/%m/%d'),
    }


# ===========================================================================
# 大盤情緒評估 (用於首頁 P_熊/P_震/P_牛)
# ===========================================================================
def get_market_sentiment(benchmark_df: pd.DataFrame) -> dict:
    """
    簡易三態大盤情緒評估
    """
    if benchmark_df is None or len(benchmark_df) < 20:
        return {"bear": 33, "neutral": 34, "bull": 33, "label": "震盪", "slope_5d": 0}

    close = benchmark_df['Close']
    slope_5d = _slope_poly(close, 5)
    slope_20d = _slope_poly(close, 20)

    # 簡單規則
    if slope_5d > 0.3 and slope_20d > 0:
        bear, neutral, bull = 15, 25, 60
        label = "偏多"
    elif slope_5d < -0.3 and slope_20d < 0:
        bear, neutral, bull = 60, 25, 15
        label = "偏空"
    elif slope_5d < -0.1:
        bear, neutral, bull = 40, 35, 25
        label = "轉弱"
    else:
        bear, neutral, bull = 25, 45, 30
        label = "震盪"

    return {
        "bear": bear,
        "neutral": neutral,
        "bull": bull,
        "label": label,
        "slope_5d": round(slope_5d, 3),
        "slope_20d": round(slope_20d, 3),
    }

