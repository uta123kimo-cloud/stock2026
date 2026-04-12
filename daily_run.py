"""
╔══════════════════════════════════════════════════════════════╗
║  daily_run.py v3.4  資源法 Precompute 主控制器               ║
║                                                              ║
║  修正清單 v3.4：                                             ║
║  [FIX-1] _FeatureEngine / _RegimeEngine 加入空 DataFrame     ║
║          防護：下載失敗時不再觸發 iloc out-of-bounds          ║
║  [FIX-2] ^TWII 下載獨立封裝為 fetch_index()，含指數退讓      ║
║          重試（最多 5 次）與本地快取 fallback                 ║
║  [FIX-3] step_regime 縮排修正，log.info 回到函式內           ║
║  [FIX-4] main() 接上 _risk_engine，風險摘要正確輸出          ║
║  [FIX-5] 所有 iloc[-1] 存取前加 len(df) > 0 檢查            ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import math
import time
import logging
import random
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

# ──────────────────────────────────────────────────────────────
# 系統設定與日誌
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("daily_run")

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(ROOT_DIR, "storage")
V4_DIR      = os.path.join(STORAGE_DIR, "v4")
V12_DIR     = os.path.join(STORAGE_DIR, "v12")
REGIME_DIR  = os.path.join(STORAGE_DIR, "regime")
MARKET_DIR  = os.path.join(STORAGE_DIR, "market")
LOGS_DIR    = os.path.join(STORAGE_DIR, "logs")
DATA_ROOT   = os.path.join(ROOT_DIR, "data")
CACHE_DIR   = os.path.join(STORAGE_DIR, "cache")   # ← v3.4 新增快取目錄

TODAY = date.today().strftime("%Y-%m-%d")
TS    = datetime.now().strftime("%Y%m%d_%H%M")

for _d in [V4_DIR, V12_DIR, REGIME_DIR, MARKET_DIR, LOGS_DIR, DATA_ROOT, CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# 建立偽裝瀏覽器的 Session 以降低 Yahoo 429 機率
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})

# ──────────────────────────────────────────────────────────────
# 股票池
# ──────────────────────────────────────────────────────────────
SYMBOLS = list(dict.fromkeys([
    "3030", "3706", "8096", "2313", "4958",
    "2330", "2317", "2454", "2308", "2382", "2303", "3711", "2412", "2357", "3231",
    "2379", "3008", "2395", "3045", "2327", "2408", "2377", "6669", "2301", "3034",
    "2345", "2474", "3037", "4938", "3443", "2353", "2324", "2603", "2609", "1513",
    "3293", "3680", "3529", "3131", "5274", "6223", "6805", "3017", "3324", "6515",
    "3661", "3583", "6139", "3035", "1560", "8299", "3558", "6187", "3406", "3217",
    "6176", "6415", "6206", "8069", "3264", "5269", "2360", "6271", "3189", "6438",
    "8358", "6231", "2449", "3030", "8016", "6679", "3374", "3014", "3211",
    "6213", "2404", "2480", "3596", "6202", "5443", "5347", "5483", "6147",
    "2313", "3037", "8046", "2368", "4958", "2383", "6269", "5469", "5351", #PCB
    "4909", "8050", "6153", "6505", "1802", "3708", "8213", "1325",
    "2344", "6239", "3260", "4967", "6414", "2337", "8096",#記憶體
    "3551", "2436", "2375", "2492", "2456", "3229", "6173", "3533" #被動元件
    "3491", "6271", "2313", "2367", "6285", "6190", #低軌衛星
    "3062", "2419", "2314", "3305", "3105", "2312", "8086",#低軌衛星
    "3081", "2455", "6442", "3163", "4979", "3363", "6451", #光通訊股
    "3450", "4908", "4977", "3234", "2360", #光通訊股
    "1711","1727","2404","2489","3060","3374","3498","3535","3580","3587",
    "3665","4749","4989","6187","6217","6290","6418","6443","6470","6542",
    "6546","6706","6831","6861","6877","8028","8111"

]))

# ══════════════════════════════════════════════════════════════
# [v3.4 新增] ^TWII 指數專用下載，含重試 + 本地快取 fallback
# ══════════════════════════════════════════════════════════════

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """將 MultiIndex 欄位攤平、欄名去空白。"""
    if df is None:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_index(ticker: str = "^TWII", period: str = "180d",
                max_retries: int = 5) -> pd.DataFrame:
    """
    下載指數資料，失敗時：
      1. 指數退讓重試（最多 max_retries 次）
      2. 全部失敗後讀取本地快取（上次成功的 CSV）
    回傳正規化後的 DataFrame；若完全失敗則回傳空 DataFrame。
    """
    if yf is None:
        log.error("yfinance 未安裝")
        return pd.DataFrame()

    cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('^','')}.csv")

    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker, period=period, progress=False,
                auto_adjust=True, timeout=30, session=_SESSION
            )
            df = _normalize_df(df)
            if not df.empty and len(df) >= 10:
                # 成功 → 更新本地快取
                df.to_csv(cache_path)
                log.info(f"  ✅ {ticker} 下載成功 ({len(df)} 筆)")
                return df
            else:
                log.warning(f"  {ticker} 回傳空資料 (第{attempt+1}次)")
        except Exception as e:
            err = str(e).lower()
            if "too many requests" in err or "429" in err or "rate" in err:
                wait = (2 ** attempt) * 5 + random.uniform(3, 8)   # 5,13,29,61,125 秒
                log.warning(
                    f"  ⚠️ {ticker} 429 Rate Limit，"
                    f"退讓 {wait:.1f}s (第{attempt+1}/{max_retries}次)"
                )
                time.sleep(wait)
            else:
                log.warning(f"  {ticker} 下載錯誤: {e}")
                break   # 非 429 錯誤不重試

    # 全部失敗 → 讀快取
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df = _normalize_df(df)
            if not df.empty:
                log.warning(f"  ⚠️ {ticker} 使用快取資料（{cache_path}）")
                return df
        except Exception as e:
            log.error(f"  快取讀取失敗: {e}")

    log.error(f"  ❌ {ticker} 所有嘗試均失敗，無法取得資料")
    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════
# 純 Numpy/Pandas 技術指標（零 pandas_ta）
# ══════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period, adjust=False).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_h = high.shift(1); prev_l = low.shift(1); prev_c = close.shift(1)
    up     = high - prev_h; down = prev_l - low
    pdm    = np.where((up > down) & (up > 0), up, 0.0)
    mdm    = np.where((down > up) & (down > 0), down, 0.0)
    tr     = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs()
    ], axis=1).max(axis=1)
    def _ew(arr):
        return pd.Series(arr, index=close.index).ewm(com=period-1, adjust=False).mean()
    atr14 = _ew(tr.values)
    pdi   = 100 * _ew(pdm) / (atr14 + 1e-9)
    mdi   = 100 * _ew(mdm) / (atr14 + 1e-9)
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return _ew(dx.values)

def _pvo(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    vf = volume.ewm(span=fast, adjust=False).mean()
    vs = volume.ewm(span=slow, adjust=False).mean()
    return (vf - vs) / (vs + 1e-9) * 100

def _vri(rsi: pd.Series, atr: pd.Series, window: int = 20) -> pd.Series:
    rsi_norm = rsi.clip(0, 100)
    atr_mean = atr.rolling(window).mean()
    atr_norm = (atr / (atr_mean + 1e-9) * 50).clip(0, 100)
    return (rsi_norm * 0.6 + atr_norm * 0.4).clip(0, 100)

def _pvo_consec(pvo_series: pd.Series, thr: float = 0.0) -> pd.Series:
    result = pd.Series(0, index=pvo_series.index, dtype=float)
    cnt = 0
    for i, v in enumerate(pvo_series):
        cnt = cnt + 1 if v > thr else 0
        result.iloc[i] = cnt
    return result

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) < 20:
        return df
    df["RSI"]        = _rsi(df["Close"])
    df["ATR"]        = _atr(df["High"], df["Low"], df["Close"])
    df["ADX"]        = _adx(df["High"], df["Low"], df["Close"])
    df["MA20"]       = df["Close"].rolling(20).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["MA60"]       = df["Close"].rolling(60).mean()
    df["PVO"]        = _pvo(df["Volume"])
    df["VRI"]        = _vri(df["RSI"], df["ATR"])
    df["Slope"]      = df["Close"].pct_change(5) * 100
    df["PVO_consec"] = _pvo_consec(df["PVO"])
    df["PVO_accel"]  = df["PVO"] / (df["PVO"].rolling(5).mean().abs() + 1e-9)
    df["VRI_delta"]  = df["VRI"] - df["VRI"].rolling(5).mean()
    df["VolRatio"]   = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    slope_mu  = df["Slope"].rolling(30).mean()
    slope_std = df["Slope"].rolling(30).std() + 1e-9
    df["SlopeZ"] = (df["Slope"] - slope_mu) / slope_std
    return df

# ══════════════════════════════════════════════════════════════
# yfinance 個股資料抓取
# ══════════════════════════════════════════════════════════════

def fetch_tw_ohlcv(sym: str, period: str = "60d", max_retries: int = 4):
    """下載個股 OHLCV，含指數退讓與後綴切換。"""
    if yf is None:
        return None, None

    for suffix in [".TW", ".TWO"]:
        ticker = f"{sym}{suffix}"
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker, period=period, progress=False,
                    auto_adjust=True, timeout=20, session=_SESSION
                )
                df = _normalize_df(df)
                if not df.empty and len(df) >= 20:
                    return df, suffix
                break   # 無資料，換後綴
            except Exception as e:
                err = str(e).lower()
                if "too many requests" in err or "429" in err or "rate" in err:
                    wait = (3 ** attempt) + random.uniform(3, 7)
                    log.warning(
                        f"  ⚠️ {ticker} 429，退讓 {wait:.1f}s "
                        f"(第{attempt+1}/{max_retries}次)"
                    )
                    time.sleep(wait)
                else:
                    log.debug(f"  {ticker} 錯誤: {e}")
                    break

    log.warning(f"⚠️ {sym} .TW/.TWO 均無資料，跳過")
    return None, None


def load_from_csv(sym: str, day_dir: str):
    csv_path = os.path.join(day_dir, f"{sym}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = _normalize_df(df)
        return df if len(df) >= 20 else None
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# JSON 工具
# ══════════════════════════════════════════════════════════════

def save_json(path: str, data):
    if isinstance(data, dict):
        data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        data["date"] = TODAY
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"✅ 儲存: {path}")

def load_json(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

# ══════════════════════════════════════════════════════════════
# Inline Engines: Regime / Feature / Risk
# ══════════════════════════════════════════════════════════════

class _RegimeEngine:
    SMOOTH = 0.7

    def run(self, market_data: dict, today: str) -> dict:
        try:
            # [FIX-1] 使用 fetch_index() 而非裸 yf.download，含重試 + 空值防護
            bm = fetch_index("^TWII", period="180d")
            if bm.empty or len(bm) < 20:
                log.error("❌ RegimeEngine：^TWII 資料不足，無法計算")
                return {}

            # 必要欄位檢查
            for col in ("High", "Low", "Close"):
                if col not in bm.columns:
                    log.error(f"❌ RegimeEngine：缺少欄位 {col}")
                    return {}

            bm["RSI"]  = _rsi(bm["Close"])
            bm["ADX"]  = _adx(bm["High"], bm["Low"], bm["Close"])
            bm["MA60"] = bm["Close"].rolling(60).mean()

            # [FIX-5] 確認 dropna 後仍有足夠資料
            bm_valid = bm.dropna(subset=["RSI", "ADX", "MA60"])
            if bm_valid.empty:
                log.error("❌ RegimeEngine：指標計算後無有效列")
                return {}

            last  = bm_valid.iloc[-1]
            rsi   = float(last["RSI"])
            adx   = float(last["ADX"])
            close = float(last["Close"])
            ma60  = float(last["MA60"])

            c      = bm["Close"].values.astype(float)
            ma60_s = pd.Series(c).rolling(60).mean().values
            s20    = pd.Series(c).rolling(20).std().values

            ma_dev  = (close - ma60) / (s20[-1] + 1e-9)
            mom20   = (c[-1] / c[-21] - 1) if len(c) > 21 else 0.0
            ma60_sl = (
                (ma60_s[-1] - ma60_s[-6]) / (ma60_s[-6] + 1e-9)
                if len(c) > 65 and not math.isnan(ma60_s[-6])
                else 0.0
            )

            bull_sc  =  ma_dev * 0.4 + mom20 * 3.0 + ma60_sl * 5.0
            bear_sc  = -ma_dev * 0.4 - mom20 * 3.0 - ma60_sl * 5.0
            range_sc =  (1.0 - abs(ma_dev) / 2.0) + (1.0 - abs(mom20) * 5.0) * 0.5
            scores   = [bear_sc, range_sc, bull_sc]
            max_s    = max(scores)
            exps     = [math.exp(s - max_s) for s in scores]
            total    = sum(exps)
            bear_r, range_r, bull_r = [e / total for e in exps]

            old = load_json(os.path.join(REGIME_DIR, "regime_state.json"))
            if old:
                sm = self.SMOOTH
                bear_r  = sm * old.get("bear",  bear_r)  + (1 - sm) * bear_r
                range_r = sm * old.get("range", range_r) + (1 - sm) * range_r
                bull_r  = 1 - bear_r - range_r
                t2 = bear_r + range_r + bull_r + 1e-9
                bear_r /= t2; range_r /= t2; bull_r /= t2

            if   bull_r >= 0.55: strat = "bull";  a_p = "45";  b_p = "423"
            elif bear_r >= 0.60: strat = "bear";  a_p = None;  b_p = None
            else:                strat = "range"; a_p = "423"; b_p = "45"

            if   bull_r > 0.5:    label = "牛市"
            elif bear_r > 0.5:    label = "熊市"
            elif bull_r > bear_r: label = "偏多震盪"
            else:                 label = "偏空震盪"

            s5d  = float(bm["Close"].pct_change(5).iloc[-1])
            s20d = float(bm["Close"].pct_change(20).iloc[-1])

            old_hist   = old.get("history", []) if old else []
            curr_month = datetime.now().strftime("%Y-%m")
            if not any(h["month"] == curr_month for h in old_hist):
                old_hist.append({
                    "month": curr_month,
                    "bear": round(bear_r, 4), "range": round(range_r, 4),
                    "bull": round(bull_r, 4), "label": label
                })
                old_hist = old_hist[-24:]

            return {
                "bear": round(bear_r, 4), "range": round(range_r, 4),
                "bull": round(bull_r, 4), "label": label,
                "active_strategy": strat, "active_path": a_p, "backup_path": b_p,
                "slope_5d": round(s5d, 4), "slope_20d": round(s20d, 4),
                "mkt_rsi": round(rsi, 1), "adx": round(adx, 1),
                "history": old_hist,
            }
        except Exception as e:
            log.error(f"❌ RegimeEngine 失敗: {e}")
            return {}


class _FeatureEngine:
    def run(self, today: str) -> dict:
        try:
            # [FIX-1] 使用 fetch_index() 含重試 + 空值防護
            bm = fetch_index("^TWII", period="60d")
            if bm.empty or len(bm) < 2:
                log.error("❌ FeatureEngine：^TWII 資料不足")
                return {}

            for col in ("High", "Low", "Close"):
                if col not in bm.columns:
                    log.error(f"❌ FeatureEngine：缺少欄位 {col}")
                    return {}

            bm["RSI"] = _rsi(bm["Close"])

            # [FIX-5] dropna 後再取 iloc，避免 out-of-bounds
            bm_valid = bm.dropna(subset=["RSI"])
            if len(bm_valid) < 2:
                log.error("❌ FeatureEngine：RSI 有效資料不足")
                return {}

            last = bm_valid.iloc[-1]
            prev = bm_valid.iloc[-2]
            chg  = float((last["Close"] - prev["Close"]) / prev["Close"] * 100)
            rsi  = float(last["RSI"])

            return {
                "index_close":   float(last["Close"]),
                "index_chg_pct": round(chg, 2),
                "mkt_rsi":       rsi if not math.isnan(rsi) else 50.0,
                "mkt_slope_5d":  round(float(bm["Close"].pct_change(5).iloc[-1] * 100), 4),
                "mkt_slope_20d": round(float(bm["Close"].pct_change(20).iloc[-1] * 100), 4),
                "volume":        float(last.get("Volume", 0)),
            }
        except Exception as e:
            log.error(f"❌ FeatureEngine 失敗: {e}")
            return {}


class _RiskEngine:
    def summarize(self, positions: list) -> dict:
        if not positions:
            return {"total_pos": 0, "avg_ret": 0.0, "exit_count": 0, "exit_syms": []}
        rets  = [p.get("curr_ret_pct", 0) for p in positions]
        exits = [p for p in positions if p.get("exit_signal", "—") not in ("—", "無")]
        return {
            "total_pos":  len(positions),
            "avg_ret":    round(sum(rets) / len(rets), 2),
            "exit_count": len(exits),
            "exit_syms":  [p["symbol"] for p in exits],
        }


_regime_engine  = _RegimeEngine()
_feature_engine = _FeatureEngine()
_risk_engine    = _RiskEngine()

# ══════════════════════════════════════════════════════════════
# V4 引擎
# ══════════════════════════════════════════════════════════════

_POS_WEIGHT = {
    "三合一(ABC)": 0.28, "二合一(AB)": 0.23, "二合一(AC)": 0.23,
    "二合一(BC)":  0.23, "單一(A)":    0.18, "單一(B)":   0.18,
    "單一(C)":     0.18, "基準-強勢":  0.15, "基準-持有": 0.13,
}

def _v4_signal(pvo, vri, slope_z, sc, mu, sigma, pvo_c, pvo_a, vri_d, vol_ratio):
    is_strong = sc > (mu + 0.5 * sigma)
    is_fire   = pvo >= 8.0
    is_money  = 0.0 <= pvo < 8.0
    is_hot    = vri > 70.0
    is_cool   = vri < 45.0
    a_ok = (pvo_c >= 1) and (vri_d < 2.0)
    b_ok = (vri_d > 3.0) and (pvo_a >= 1.3)
    c_ok = vri_d > -5.0

    patterns = []
    if is_strong and is_money and is_cool and a_ok:          patterns.append("A")
    if is_strong and is_fire  and is_hot:                    patterns.append("B")
    if is_strong and is_hot and "B" not in patterns and c_ok: patterns.append("C")

    vtag = (
        "+量爆" if vol_ratio >= 2.0 else
        "+放量" if vol_ratio >= 1.2 else
        "-縮量" if vol_ratio < 0.7 else ""
    )

    ps = frozenset(patterns)
    if   len(ps) >= 3: base = "三合一(ABC)"; combo = base
    elif len(ps) == 2: k = "".join(sorted(ps)); base = f"二合一({k})"; combo = base
    elif len(ps) == 1: p = list(ps)[0]; base = f"單一({p})"; combo = base
    else:              base = "基準-強勢" if slope_z > 1.2 else "基準-持有"; combo = base

    sig_q = (
        1.3 if vol_ratio >= 2.0 else
        1.1 if vol_ratio >= 1.2 else
        0.8 if vol_ratio < 0.7  else 1.0
    )
    if "B" in patterns and b_ok:
        sig_q = min(sig_q * 1.1, 1.5)

    return base + vtag, combo, sig_q


def _v4_score(df, mu, sigma, rtype):
    if df is None or len(df) < 20:
        return None
    last = df.iloc[-1]
    def g(col, d=0.0):
        v = last.get(col, d)
        return float(v) if not (isinstance(v, float) and math.isnan(v)) else d

    pvo   = g("PVO"); vri = g("VRI", 50); slope = g("Slope")
    pvo_c = g("PVO_consec"); pvo_a = g("PVO_accel", 1); vri_d = g("VRI_delta")
    vol_r = g("VolRatio", 1); close = g("Close"); atr = g("ATR", close * 0.02)

    sw = df["Slope"].tail(30)
    slope_z = (slope - float(sw.mean())) / (float(sw.std()) + 1e-9)

    score = 50.0
    score += min(slope_z * 8, 20)
    score += min(pvo * 0.5, 15) if pvo > 0 else max(pvo * 0.3, -10)
    score += 8 if 40 <= vri <= 75 else (-5 if vri > 90 else 0)

    signal, combo, sig_q = _v4_signal(
        pvo, vri, slope_z, score, mu, sigma, pvo_c, pvo_a, vri_d, vol_r
    )

    action = (
        "強力買進" if slope_z >= 1.5 and pvo > 5 else
        "買進"     if slope_z >= 0.5 and pvo > 0  else
        "賣出"     if slope_z < -1.0               else "觀察"
    )

    rm = {"trend": 1.1, "range": 1.0, "recovery": 0.8, "crash": 0.5}.get(rtype, 1.0)
    pw = round(min(max(_POS_WEIGHT.get(combo, 0.15) * sig_q * rm, 0.10), 0.30), 4)

    return {
        "score": round(score, 2), "pvo": round(pvo, 2), "vri": round(vri, 1),
        "slope_z": round(slope_z, 2), "slope": round(slope, 3),
        "action": action, "signal": signal, "combo_key": combo,
        "close": round(close, 1), "atr": round(atr, 2), "pos_weight": pw
    }


def run_v4(symbols, regime, today, day_dir) -> dict:
    label = regime.get("label", "震盪")
    rtype = (
        "crash"    if "熊" in label else
        "trend"    if "牛" in label else
        "recovery" if "回升" in label else "range"
    )
    log.info(f"V4 啟動 | Regime:{label}({rtype}) | 池:{len(symbols)}")

    rows = []; skipped = 0
    for sym in symbols:
        time.sleep(random.uniform(0.8, 1.5))
        try:
            df = load_from_csv(sym, day_dir)
            if df is None:
                df, _ = fetch_tw_ohlcv(sym, "60d")
            if df is None:
                skipped += 1; continue
            df_ind = enrich_df(df)
            if df_ind.dropna(subset=["RSI", "ATR"]).shape[0] < 20:
                skipped += 1; continue
            rows.append({"sym": sym, "df": df_ind})
        except Exception as e:
            log.warning(f"  V4 {sym}: {e}"); skipped += 1

    if not rows:
        log.error("❌ V4：無有效資料"); return {}

    mu, sigma = 62.0, 11.5
    result_rows = []
    for item in rows:
        res = _v4_score(item["df"], mu, sigma, rtype)
        if res:
            res["symbol"] = item["sym"]; res["regime"] = label
            result_rows.append(res)

    if not result_rows:
        log.error("❌ V4：評分空"); return {}

    result_rows.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(result_rows):
        r["rank"] = i + 1
    top20 = result_rows[:20]

    scores = [r["score"] for r in result_rows]
    am = round(float(np.mean(scores)), 2)
    as_ = round(float(np.std(scores)), 2)
    log.info(f"V4 完成 ✅ | TOP20:{len(top20)} | 跳過:{skipped} | μ={am} σ={as_}")
    return {
        "market": "TW", "top20": top20, "pool_mu": am, "pool_sigma": as_,
        "win_rate": 57.1, "regime": label,
        "total_scored": len(result_rows), "skipped": skipped
    }

# ══════════════════════════════════════════════════════════════
# V12.1 引擎
# ══════════════════════════════════════════════════════════════

_PATH_DEFS = {
    "423": {
        "tp1": 0.20, "tp2": 0.35,
        "ev_bear": 0.0096, "ev_range": 0.0834, "ev_bull": 0.0406,
        "oos_ev": 3.02, "order": ["Y4","Y2","Y3"]
    },
    "45": {
        "tp1": 0.20, "tp2": 0.28,
        "ev_bear": 0.0022, "ev_range": 0.0485, "ev_bull": 0.0432,
        "oos_ev": 7.36, "order": ["Y4","Y5"]
    },
}
_REGIME_STRATS = {
    "bull":  {"active":"45",  "backup":"423", "ratio":{"45":0.65,"423":0.35}, "ev_min":0.030, "max_pos":4},
    "range": {"active":"423", "backup":"45",  "ratio":{"423":0.65,"45":0.35}, "ev_min":0.030, "max_pos":5},
    "bear":  {"active":None,  "backup":None,  "ratio":{},                     "ev_min":0.040, "max_pos":2},
}
_ALL_Y_BETAS = {
    "Y1": {"bb_width":+.1527,"rsi_14":+.1156,"ma20_slope":-.0902,"price_mom_20":-.0650,
           "rs_vs_mkt_5":+.0518,"mkt_excess_z":-.0490,"inst_cum_10":-.0362,"vol_pvo":+.0276,
           "vol_pvo_sq":-.0271,"persist_count":+.0195,"atr_regime":+.0168},
    "Y2": {"bb_width":+.1428,"inst_cum_10":-.0652,"atr_regime":+.0420,"rs_vs_mkt_5":+.0379,
           "inst_pvo":+.0327,"rsi_14":-.0267,"inst_x_price":-.0239,"slope_5d":+.0224},
    "Y3": {"bb_width":+.1308,"price_mom_20":-.1043,"rsi_14":+.0592,"rs_vs_mkt_5":+.0260,
           "vri":+.0257,"persist_count":+.0210,"inst_cum_10":-.0185,"vol_pvo_sq":-.0168},
    "Y4": {"bb_width":+.2624,"price_mom_60":+.1054,"high52w_dist":-.0609,"inst_cum_10":-.0599,
           "close_ma5_r":+.0519,"inst_pvo":+.0372,"price_mom_5":-.0358,"vri":+.0347,
           "atr_regime":+.0329,"rsi_14":+.0202,"mkt_excess_z":+.0165},
    "Y5": {"bb_width":+.1872,"high52w_dist":-.0472,"price_mom_60":+.0462,"rs_vs_mkt_5":+.0339,
           "ma20_slope":-.0259,"vri":+.0209,"price_mom_5":+.0124,"inst_pvo":+.0098,
           "inst_cum_10":+.0068,"rsi_14":+.0062},
}
_HIST_STATS = {
    "total_trades": 112, "win_rate": 57.1, "avg_ev": 5.29, "max_dd": -6.58,
    "sharpe": 5.36, "t_stat": 4.032, "simple_cagr": 96.9, "pl_ratio": 2.31
}

def _v12_features(df) -> dict:
    if df is None or len(df) < 20:
        return {}
    try:
        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        v = df["Volume"].values.astype(float)
        n = len(c)

        ma5  = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        s20  = pd.Series(c).rolling(20).std().values
        bb_w = np.where(ma20 > 0, 4 * s20 / (ma20 + 1e-9), 0.0)

        delta = pd.Series(c).diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean().values
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean().values
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100.0

        atr_a = np.zeros(n)
        for i in range(1, n):
            atr_a[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr_s   = pd.Series(atr_a).ewm(com=13, adjust=False).mean().values
        atr_pct = np.where(c > 0, atr_s / (c + 1e-9), 0.03)
        atr_m   = pd.Series(atr_pct).rolling(20).mean().values
        atr_reg = np.where(atr_m > 0, atr_pct / (atr_m + 1e-9), 1.0)

        def _mom(lag):
            a = np.zeros(n)
            for i in range(lag, n):
                if c[i-lag] > 0: a[i] = c[i] / c[i-lag] - 1
            return a
        pm5 = _mom(5); pm20 = _mom(20); pm60 = _mom(60)

        ma20sl = np.zeros(n)
        for i in range(5, n):
            if ma20[i-5] > 1e-9:
                ma20sl[i] = (ma20[i] - ma20[i-5]) / ma20[i-5]

        slope_a = np.zeros(n)
        for i in range(4, n):
            y_ = c[i-4:i+1]; x_ = np.arange(5, dtype=float)
            if y_[0] > 1e-9:
                s_, _ = np.polyfit(x_, y_, 1); slope_a[i] = s_ / y_[0]

        vma10 = pd.Series(v * c).rolling(10).mean().values
        vri   = np.where(vma10 > 0, (v * c) / (vma10 + 1e-9), 1.0)
        h52   = pd.Series(h).rolling(252, min_periods=60).max().bfill().values
        h52d  = np.where(h52 > 0, c / (h52 + 1e-9), 0.0)
        cma5r = np.where(ma5 > 0, (c - ma5) / (ma5 + 1e-9), 0.0)

        vma20 = pd.Series(v).rolling(20).mean().values
        vpvo  = np.where(vma20 > 0, (v - vma20) / (vma20 + 1e-9), 0.0)
        vpvoq = vpvo ** 2
        ic10  = pd.Series(vpvo).rolling(10).sum().fillna(0).values
        vma5  = pd.Series(v).rolling(5).mean().values
        ipvo  = np.where(vma20 > 0, (vma5 - vma20) / (vma20 + 1e-9), 0.0)
        ixp   = ic10 * pm5

        mad  = np.where(ma20 > 0, ma5 - ma20, 0.0)
        ymad = np.where(s20 > 0, mad / (s20 + 1e-9), 0.0)
        pcnt = np.zeros(n); ct = 0
        for i in range(n):
            ct = (ct + 1) if ymad[i] > 0 else 0; pcnt[i] = ct
        persist = np.clip(pcnt / 60.0, 0.0, 1.0)

        i = -1
        return {
            "bb_width": float(bb_w[i]),      "rsi_14": float(rsi[i]),
            "ma20_slope": float(ma20sl[i]),  "price_mom_5": float(pm5[i]),
            "price_mom_20": float(pm20[i]),  "price_mom_60": float(pm60[i]),
            "rs_vs_mkt_5": 0.0,              "mkt_excess_z": float(ymad[i]),
            "inst_cum_10": float(ic10[i]),   "vol_pvo": float(vpvo[i]),
            "vol_pvo_sq": float(vpvoq[i]),   "persist_count": float(persist[i]),
            "atr_regime": float(atr_reg[i]), "vri": float(vri[i]),
            "close_ma5_r": float(cma5r[i]),  "inst_pvo": float(ipvo[i]),
            "inst_x_price": float(ixp[i]),   "high52w_dist": float(h52d[i]),
            "slope_5d": float(slope_a[i]),
            "_close": float(c[i]),           "_atr_pct": float(atr_pct[i]),
            "_atr_regime": float(atr_reg[i]),"_slope_5d": float(slope_a[i]),
        }
    except Exception as e:
        log.warning(f"  v12特徵失敗: {e}"); return {}


def _v12_y_pr(features):
    result = {}
    for yn, beta in _ALL_Y_BETAS.items():
        sc = sum(features.get(k, 0.0) * v for k, v in beta.items())
        if yn == "Y5":
            sc = 1 / (1 + math.exp(-max(-30, min(30, sc))))
        result[f"s_{yn}"]  = sc
        result[f"PR_{yn}"] = 95.0 if sc > 0 else 50.0
    return result


def _v12_path(y_prs, Pb, Pr, Pu):
    PR_THR = 90.0
    first  = {yn for yn in ["Y1","Y2","Y3","Y4","Y5"] if y_prs.get(f"PR_{yn}", 0) >= PR_THR}
    if not first:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "quality": "Pure"}

    comp = {}; matched = []
    for pk, pd_ in _PATH_DEFS.items():
        order = pd_.get("order", [])
        done  = sum(1 for yn in order if yn in first)
        comp[pk] = done
        if done == len(order):
            matched.append(pk)

    best = None
    if matched:
        matched.sort(key=lambda k: -_PATH_DEFS[k].get("oos_ev", 0)); best = matched[0]
    elif comp:
        inc = [(k, v) for k, v in comp.items() if v >= 1]
        if inc:
            inc.sort(key=lambda x: (-x[1], -_PATH_DEFS.get(x[0], {}).get("oos_ev", 0)))
            best = inc[0][0]

    batch = comp.get(best, 0) if best else 0
    ev    = 0.0
    if best:
        pd_ = _PATH_DEFS[best]
        ev  = Pb * pd_["ev_bear"] + Pr * pd_["ev_range"] + Pu * pd_["ev_bull"]
    return {"best": best, "batch": batch, "ev_soft": ev, "quality": "Pure"}


def _v12_exit(old, ev_now, slope, days, curr_ret):
    if curr_ret <= -0.10:  return "硬停損"
    if ev_now  < 0.005:    return "EV衰退"
    ev_e = old.get("ev", ev_now * 100) / 100
    if ev_e > 0 and days > 7:
        drop = (ev_e - ev_now) / ev_e
        if drop > 0.20 and slope < -0.01: return "Slope加速出場"
        if drop > 0.35:                   return "時間衰減"
    if days > 3 and old.get("_pvo", 0.0) < -0.30: return "量能枯竭"
    return "—"


def run_v12(symbols, regime, v4_data, today, day_dir) -> dict:
    label = regime.get("label", "震盪")
    rkey  = (
        "bull"  if "牛" in label else
        "bear"  if "熊" in label else "range"
    )
    strat   = _REGIME_STRATS.get(rkey, _REGIME_STRATS["range"])
    a_p     = strat["active"];  b_p     = strat["backup"]
    ev_min  = strat["ev_min"]; max_pos = strat["max_pos"]
    ratio   = strat["ratio"]
    Pb      = regime.get("bear", 0.33)
    Pr      = regime.get("range", 0.34)
    Pu      = regime.get("bull", 0.33)

    log.info(f"V12 啟動 | Regime:{rkey} | 主路徑:{a_p} | max_pos:{max_pos}")
    old_v12 = load_json(os.path.join(V12_DIR, "v12_latest.json"))
    old_pos = {p["symbol"]: p for p in (old_v12 or {}).get("positions", [])}
    top20   = v4_data.get("top20", [])
    cands   = [
        r for r in top20
        if r.get("action") in ("強力買進","買進") and r.get("score", 0) > 55
    ]
    log.info(f"  V4候選: {len(cands)} 檔")

    positions = []; path_counts = {}
    for cand in cands[:12]:
        sym = cand["symbol"]
        time.sleep(random.uniform(0.8, 1.5))
        try:
            df = load_from_csv(sym, day_dir)
            if df is None:
                df, _ = fetch_tw_ohlcv(sym, "90d")
            if df is None or len(df) < 20:
                continue

            feats = _v12_features(df)
            if not feats:                             continue
            if feats.get("_atr_regime", 1.0) > 2.5:  continue
            if feats.get("_slope_5d", 0.0) < 0.0:    continue

            y_prs     = _v12_y_pr(feats)
            path_info = _v12_path(y_prs, Pb, Pr, Pu)
            best_path = path_info.get("best")
            ev_soft   = path_info.get("ev_soft", 0.0)
            quality   = path_info.get("quality", "Pure")

            if best_path not in [a_p, b_p]:
                best_path = a_p or "423"

            ev_thr = ev_min * (1.20 if quality == "Flicker" else 1.0)
            if ev_soft < ev_thr:
                continue

            max_same = max(1, round(max_pos * ratio.get(best_path, 0.50)))
            if path_counts.get(best_path, 0) >= max_same:
                continue

            if len(positions) >= max_pos:
                if positions:
                    min_ev = min(p.get("ev", 0.0) / 100 for p in positions)
                    if ev_soft < min_ev * 1.20:
                        continue

            last    = df.iloc[-1]
            close   = float(last.get("Close", 0))
            atr_raw = feats.get("_atr_pct", 0.02) * close
            tp1_px  = round(close * (1 + _PATH_DEFS.get(best_path, {}).get("tp1", 0.20)), 1)
            tp2_px  = round(close * (1 + _PATH_DEFS.get(best_path, {}).get("tp2", 0.28)), 1)
            stop_px = round(close - atr_raw * 1.5, 1)
            if quality == "Flicker":
                tp1_px = round(close * (1 + _PATH_DEFS.get(best_path, {}).get("tp1", 0.20) * 0.80), 1)

            if sym in old_pos:
                old         = old_pos[sym]
                days        = old.get("days_held", 0) + 1
                entry_price = old.get("entry_price", close)
                curr_ret    = (close - entry_price) / (entry_price + 1e-9)
                stop_px     = old.get("stop_price", stop_px)
                tp1_px      = old.get("tp1_price", tp1_px)
                action      = "持有"
                old["_pvo"] = feats.get("vol_pvo", 0.0)
                exit_sig    = _v12_exit(old, ev_soft, feats.get("_slope_5d", 0.0), days, curr_ret)
            else:
                days        = 0; entry_price = close; curr_ret = 0.0
                action      = "進場"; exit_sig = "—"

            ev_tier = (
                "⭐核心" if ev_soft >= 0.050 else
                "🔥主力" if ev_soft >= 0.030 else
                "📌補位" if ev_soft >= 0.020 else ""
            )

            positions.append({
                "symbol": sym,    "path": best_path,
                "ev": round(ev_soft * 100, 2), "ev_tier": ev_tier,
                "action": action, "exit_signal": exit_sig,
                "quality": quality, "days_held": days,
                "curr_ret_pct": round(curr_ret * 100, 2),
                "entry_price": round(entry_price, 2),
                "tp1_price": tp1_px, "tp2_price": tp2_px, "stop_price": stop_px,
                "regime": rkey,   "close": round(close, 2),
                "batch": path_info.get("batch", 0),
            })
            path_counts[best_path] = path_counts.get(best_path, 0) + 1
            log.info(
                f"  ✅ {sym} | {best_path} | EV:{ev_soft*100:+.2f}% "
                f"| {quality} | {action}"
            )
            if len(positions) >= max_pos:
                log.info(f"  部位已滿({max_pos})"); break

        except Exception as e:
            log.warning(f"  V12 {sym}: {e}")

    log.info(f"V12 完成 ✅ | 部位:{len(positions)} | 路徑:{path_counts}")
    return {
        "market": "TW", "positions": positions, "stats": _HIST_STATS,
        "regime": rkey, "active_path": a_p, "backup_path": b_p,
        "path_counts": path_counts,
    }

# ══════════════════════════════════════════════════════════════
# Pipeline Steps & Main
# ══════════════════════════════════════════════════════════════

def step_market() -> dict:
    log.info("=== Step 1: Market Snapshot ===")
    data = _feature_engine.run(today=TODAY)
    if data:
        save_json(os.path.join(MARKET_DIR, "market_snapshot.json"), data)
        log.info(    # [FIX-3] 縮排修正：回到函式內
            f"大盤: {data.get('index_close', 0):,.1f} "
            f"({data.get('index_chg_pct', 0):+.2f}%)"
        )
    else:
        log.error("❌ Market snapshot 失敗")
    return data or {}


def step_regime(market_data: dict) -> dict:
    log.info("=== Step 2: Regime Definition ===")
    data = _regime_engine.run(market_data, today=TODAY)
    if data:                                          # [FIX-3] 縮排修正
        save_json(os.path.join(REGIME_DIR, "regime_state.json"), data)
        log.info(
            f"環境狀態: {data.get('label', '未知')} "
            f"(牛:{data.get('bull',0):.2f} 熊:{data.get('bear',0):.2f})"
        )
    else:
        log.error("❌ Regime 定義失敗")
    return data or {}


def main():
    log.info(f"🚀 資源法 Daily Compute 啟動 [{TS}]")

    # ── Step 1: Market ──
    market_data = step_market()

    # ── Step 2: Regime ──
    regime_data = step_regime(market_data)
    if not regime_data:
        log.error("⚠️ 無法取得 Regime 資料，終止後續計算。")
        sys.exit(1)

    # ── Step 3: V4 Engine ──
    log.info("=== Step 3: V4 Engine ===")
    v4_data = run_v4(SYMBOLS, regime_data, TODAY, DATA_ROOT)
    if v4_data:
        save_json(os.path.join(V4_DIR, "v4_latest.json"), v4_data)

    # ── Step 4: V12.1 Engine ──
    log.info("=== Step 4: V12.1 Engine ===")
    v12_data = run_v12(SYMBOLS, regime_data, v4_data or {}, TODAY, DATA_ROOT)
    if v12_data:
        save_json(os.path.join(V12_DIR, "v12_latest.json"), v12_data)

        # ── Step 5: Risk Summary ── [FIX-4] 接上 _risk_engine
        if "positions" in v12_data:
            risk_summary = _risk_engine.summarize(v12_data["positions"])
            log.info("=== Step 5: Risk Summary ===")
            log.info(
                f"總部位: {risk_summary['total_pos']} 檔 "
                f"| 平均報酬: {risk_summary['avg_ret']}%"
            )
            if risk_summary["exit_count"] > 0:
                log.info(f"⚠️ 建議出場: {risk_summary['exit_syms']}")

    log.info("🎉 資源法 V12.1 今日運算全部完成！")


if __name__ == "__main__":
    main()
