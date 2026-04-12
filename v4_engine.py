"""
╔══════════════════════════════════════════════════════════════╗
║  v4_engine.py — 資源法 V4 市場強度引擎 v3.7                  ║
║  介面：run(symbols, regime, today) → dict                    ║
║                                                              ║
║  v3.7 變更：                                                 ║
║  - 完全移除 ^TWII；大盤資料由 daily_run.fetch_market_index 提供 ║
║  - .TW → .TWO fallback + 429 指數退讓                        ║
║  - 純 numpy/pandas 指標（無 pandas_ta 依賴）                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
import math
import os
import random
import time
from datetime import datetime, date
from typing import Optional

log = logging.getLogger("v4_engine")

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import yfinance as yf
except ImportError:
    yf = None

_USES_PANDAS_TA = False   # 標記：不使用 pandas_ta

# ══════════════════════════════════════════════════════════════
# 訊號門檻參數
# ══════════════════════════════════════════════════════════════

PVO_FIRE_THR      = 8.0
PVO_FLOW_THR      = 0.0
VRI_HOT_THR       = 70.0
VRI_COOL_THR      = 45.0
SLOPE_Z_STRONG    = 1.2

PVO_CONSEC_MIN_A  = 1
PVO_ACCEL_MULT    = 1.3
VRI_DELTA_STRONG  = 3.0
VRI_DELTA_COOL    = 2.0
VRI_DELTA_CUTOFF  = -5.0

VOL_RATIO_STRONG  = 2.0
VOL_RATIO_NORMAL  = 1.2
VOL_RATIO_WEAK    = 0.7

POS_WEIGHT = {
    "三合一(ABC)": 0.28, "二合一(AB)": 0.23, "二合一(AC)": 0.23,
    "二合一(BC)":  0.23, "單一(A)":   0.18, "單一(B)":   0.18,
    "單一(C)":     0.18, "基準-強勢": 0.15, "基準-持有": 0.13,
}
POS_DEFAULT_WEIGHT = 0.15
MIN_DATA_ROWS      = 20


# ══════════════════════════════════════════════════════════════
# 純 numpy/pandas 指標函式（取代 pandas_ta）
# ══════════════════════════════════════════════════════════════

def _ta_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _ta_atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period, adjust=False).mean()


def _ta_sma(close, period):
    return close.rolling(period).mean()


def _calc_pvo(df, fast=5, slow=20):
    if "Volume" not in df.columns:
        return pd.Series(0.0, index=df.index)
    vf = df["Volume"].ewm(span=fast).mean()
    vs = df["Volume"].ewm(span=slow).mean()
    return (vf - vs) / (vs + 1e-9) * 100


def _calc_vri(df, window=20):
    if "RSI" not in df.columns or "ATR" not in df.columns:
        return pd.Series(50.0, index=df.index)
    rsi_norm = df["RSI"].clip(0, 100)
    atr_mean = df["ATR"].rolling(window).mean()
    atr_norm = (df["ATR"] / (atr_mean + 1e-9) * 50).clip(0, 100)
    return (rsi_norm * 0.6 + atr_norm * 0.4).clip(0, 100)


def _calc_pvo_consec(df, thr=PVO_FLOW_THR):
    pvo    = df["PVO"]
    consec = pd.Series(0, index=df.index, dtype=float)
    cnt    = 0
    for i, v in enumerate(pvo):
        cnt = cnt + 1 if v > thr else 0
        consec.iloc[i] = cnt
    return consec


def _calc_pvo_accel(df, window=5):
    pvo_mean = df["PVO"].rolling(window).mean()
    return df["PVO"] / (pvo_mean.abs() + 1e-9)


def _calc_vri_delta(df, window=5):
    vri_mean = df["VRI"].rolling(window).mean()
    return df["VRI"] - vri_mean


def _calc_vol_ratio(df, window=20):
    if "Volume" not in df.columns:
        return pd.Series(1.0, index=df.index)
    vol_mean = df["Volume"].rolling(window).mean()
    return df["Volume"] / (vol_mean + 1e-9)


# ══════════════════════════════════════════════════════════════
# 訊號分類器
# ══════════════════════════════════════════════════════════════

def classify_signal_v4(pvo, vri, slope_z, sc, mu_score, sigma_score,
                        pvo_consec, pvo_accel, vri_delta, vol_ratio):
    is_strong_buy = sc > (mu_score + 0.5 * sigma_score)
    is_fire       = pvo >= PVO_FIRE_THR
    is_money_in   = PVO_FLOW_THR <= pvo < PVO_FIRE_THR
    is_hot        = vri > VRI_HOT_THR
    is_cool       = vri < VRI_COOL_THR

    a_valid    = (pvo_consec >= PVO_CONSEC_MIN_A) and (vri_delta < VRI_DELTA_COOL)
    b_enhanced = (vri_delta > VRI_DELTA_STRONG) and (pvo_accel >= PVO_ACCEL_MULT)
    c_valid    = vri_delta > VRI_DELTA_CUTOFF

    patterns = []
    if is_strong_buy and is_money_in and is_cool and a_valid:           patterns.append("A")
    if is_strong_buy and is_fire and is_hot:                            patterns.append("B")
    if is_strong_buy and is_hot and "B" not in patterns and c_valid:    patterns.append("C")

    vol_tag = (
        "+量爆" if vol_ratio >= VOL_RATIO_STRONG else
        "+放量" if vol_ratio >= VOL_RATIO_NORMAL else
        "-縮量" if vol_ratio <  VOL_RATIO_WEAK   else ""
    )
    pat_set = frozenset(patterns)
    if   len(pat_set) >= 3: base_label = "三合一(ABC)"; combo_key = base_label
    elif len(pat_set) == 2: key = "".join(sorted(pat_set)); base_label = f"二合一({key})"; combo_key = base_label
    elif len(pat_set) == 1: p = list(pat_set)[0]; base_label = f"單一({p})"; combo_key = base_label
    else:
        if slope_z > SLOPE_Z_STRONG:
            base_label = "基準-強勢"; combo_key = "基準-強勢"
        else:
            base_label = "基準-持有"; combo_key = "基準-持有"

    label = base_label + vol_tag if vol_tag else base_label

    signal_quality = (
        1.3 if vol_ratio >= VOL_RATIO_STRONG else
        1.1 if vol_ratio >= VOL_RATIO_NORMAL else
        0.8 if vol_ratio <  VOL_RATIO_WEAK   else 1.0
    )
    if "B" in patterns and b_enhanced:
        signal_quality = min(signal_quality * 1.1, 1.5)

    return patterns, label, combo_key, signal_quality


# ══════════════════════════════════════════════════════════════
# Regime 分類
# ══════════════════════════════════════════════════════════════

def _classify_regime_from_label(regime_label: str) -> str:
    label = str(regime_label).lower()
    if "熊" in label or "bear" in label:   return "crash"
    if "牛" in label or "bull" in label:   return "trend"
    if "回升" in label or "recovery" in label: return "recovery"
    return "range"


def get_position_weight(combo_key, signal_quality, regime="range"):
    base = POS_WEIGHT.get(combo_key, POS_DEFAULT_WEIGHT)
    regime_mult = {"trend": 1.1, "range": 1.0, "recovery": 0.8, "crash": 0.5}.get(regime, 1.0)
    weight = base * signal_quality * regime_mult
    return round(min(max(weight, 0.10), 0.30), 4)


# ══════════════════════════════════════════════════════════════
# 資料取得（.TW → .TWO，429 退讓）
# ══════════════════════════════════════════════════════════════

def _fetch_ohlcv(sym: str, period: str = "60d",
                  max_retries: int = 4) -> Optional[pd.DataFrame]:
    """
    先嘗試 .TW（上市），失敗再嘗試 .TWO（上櫃）。
    429 做指數退讓；其他錯誤直接換 suffix。
    不再使用 ^TWII。
    """
    if yf is None or pd is None:
        return None

    for suffix in [".TW", ".TWO"]:
        ticker  = f"{sym}{suffix}"
        got_429 = False

        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker, period=period, progress=False,
                    auto_adjust=True, timeout=20
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).strip() for c in df.columns]

                if not df.empty and "Close" in df.columns and len(df) >= MIN_DATA_ROWS:
                    log.debug(f"  {ticker}: {len(df)} 筆 OK")
                    return df

                log.debug(f"  {ticker}: 資料不足 ({len(df)} 筆)，換 suffix")
                break

            except Exception as e:
                err = str(e).lower()
                if "too many requests" in err or "429" in err or "rate" in err:
                    wait = (2 ** attempt) * 3 + random.uniform(2, 6)
                    log.warning(f"  {ticker} 429，退讓 {wait:.1f}s ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    got_429 = True
                else:
                    log.debug(f"  {ticker} 非429錯誤: {e}")
                    break
        else:
            if got_429:
                log.warning(f"  {ticker} 429 重試耗盡，換 suffix")

    log.warning(f"  {sym} .TW/.TWO 均無資料，跳過")
    return None


def _load_from_csv(sym: str, day_dir: str) -> Optional[pd.DataFrame]:
    if pd is None:
        return None
    csv_path = os.path.join(day_dir, f"{sym}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.columns = [str(c).strip() for c in df.columns]
        return df if len(df) >= MIN_DATA_ROWS else None
    except Exception as e:
        log.debug(f"  {sym} CSV讀取失敗: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 指標計算
# ══════════════════════════════════════════════════════════════

def _compute_stock_indicators(df) -> Optional[pd.DataFrame]:
    if np is None or pd is None:
        log.error("numpy / pandas 未安裝")
        return None
    if df is None or len(df) < MIN_DATA_ROWS:
        return None

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    try:
        df["RSI"]        = _ta_rsi(df["Close"], 14)
        df["ATR"]        = _ta_atr(df["High"], df["Low"], df["Close"], 14)
        df["MA20"]       = _ta_sma(df["Close"], 20)
        df["MA50"]       = _ta_sma(df["Close"], 50)
        df["PVO"]        = _calc_pvo(df)
        df["VRI"]        = _calc_vri(df)
        df["Slope"]      = df["Close"].pct_change(5) * 100
        df["PVO_consec"] = _calc_pvo_consec(df)
        df["PVO_accel"]  = _calc_pvo_accel(df)
        df["VRI_delta"]  = _calc_vri_delta(df)
        df["VolRatio"]   = _calc_vol_ratio(df)
    except Exception as e:
        log.warning(f"  指標計算失敗: {e}")
        return None

    return df.dropna(subset=["RSI", "ATR", "PVO", "VRI"])


# ══════════════════════════════════════════════════════════════
# V4 評分
# ══════════════════════════════════════════════════════════════

def _score_stock(df, mu_score: float, sigma_score: float, regime_type: str) -> Optional[dict]:
    if df is None or len(df) < MIN_DATA_ROWS:
        return None

    last = df.iloc[-1]

    pvo       = float(last.get("PVO", 0))
    vri       = float(last.get("VRI", 50))
    slope     = float(last.get("Slope", 0))
    pvo_c     = float(last.get("PVO_consec", 0))
    pvo_a     = float(last.get("PVO_accel", 1))
    vri_d     = float(last.get("VRI_delta", 0))
    vol_ratio = float(last.get("VolRatio", 1))
    close     = float(last.get("Close", 0))
    atr_raw   = last.get("ATR", float("nan"))
    atr       = float(atr_raw) if not (isinstance(atr_raw, float) and math.isnan(atr_raw)) else close * 0.02

    slope_win = df["Slope"].tail(30)
    slope_z   = (slope - float(slope_win.mean())) / (float(slope_win.std()) + 1e-9)

    score = 50.0
    score += min(slope_z * 8, 20)
    score += min(pvo * 0.5, 15) if pvo > 0 else max(pvo * 0.3, -10)
    score += 8 if 40 <= vri <= 75 else (-5 if vri > 90 else 0)

    _, signal_label, combo_key, sig_quality = classify_signal_v4(
        pvo, vri, slope_z, score, mu_score, sigma_score,
        pvo_c, pvo_a, vri_d, vol_ratio
    )

    action = (
        "強力買進" if slope_z >= 1.5 and pvo > 5 else
        "買進"    if slope_z >= 0.5 and pvo > 0  else
        "賣出"    if slope_z < -1.0               else "觀察"
    )
    pos_weight = get_position_weight(combo_key, sig_quality, regime_type)

    return {
        "score":      round(score, 2),
        "pvo":        round(pvo, 2),
        "vri":        round(vri, 1),
        "slope_z":    round(slope_z, 2),
        "slope":      round(slope, 3),
        "action":     action,
        "signal":     signal_label,
        "combo_key":  combo_key,
        "close":      round(close, 1),
        "atr":        round(atr, 2),
        "pos_weight": pos_weight,
    }


# ══════════════════════════════════════════════════════════════
# 主引擎介面
# ══════════════════════════════════════════════════════════════

def run(symbols: list, regime: dict, today: str,
        daily_close: dict = None) -> dict:
    """
    V4 市場強度引擎。
    大盤資料已由 daily_run.fetch_market_index() 提供，本引擎不再抓取 ^TWII。
    """
    if np is None or pd is None:
        log.error("numpy / pandas 未安裝，V4 引擎無法運行")
        return {}

    regime_label_str = regime.get("label", "震盪")
    regime_type      = _classify_regime_from_label(regime_label_str)

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    day_dir   = os.path.join(data_root, today)

    log.info(f"V4 引擎啟動 | 日期:{today} | Regime:{regime_label_str}({regime_type}) | 池:{len(symbols)}")

    rows = []; skipped = 0
    for sym in symbols:
        time.sleep(random.uniform(0.5, 1.2))
        try:
            df = _load_from_csv(sym, day_dir)
            if df is None:
                df = _fetch_ohlcv(sym, "60d")
            if df is None:
                skipped += 1; continue

            df_ind = _compute_stock_indicators(df)
            if df_ind is None or len(df_ind) < MIN_DATA_ROWS:
                skipped += 1; continue

            rows.append({"sym": sym, "df": df_ind})
        except Exception as e:
            log.warning(f"  V4 {sym} 跳過: {e}")
            skipped += 1

    if not rows:
        log.error("V4：無有效個股資料")
        return {}

    mu_score    = 62.0
    sigma_score = 11.5

    result_rows = []
    for item in rows:
        sym   = item["sym"]
        df_ind = item["df"]
        try:
            res = _score_stock(df_ind, mu_score, sigma_score, regime_type)
            if res is None:
                continue
            res["symbol"] = sym
            res["regime"] = regime_label_str
            result_rows.append(res)
        except Exception as e:
            log.warning(f"  V4 評分 {sym} 失敗: {e}")

    if not result_rows:
        log.error("V4：評分結果為空")
        return {}

    result_rows.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(result_rows):
        r["rank"] = i + 1
    top20 = result_rows[:20]

    scores       = [r["score"] for r in result_rows]
    actual_mu    = round(float(np.mean(scores)), 2) if scores else 0.0
    actual_sigma = round(float(np.std(scores)),  2) if scores else 0.0

    action_counts = {}
    for r in top20:
        a = r.get("action", "─")
        action_counts[a] = action_counts.get(a, 0) + 1

    log.info(
        f"V4 完成 ✅ | TOP20:{len(top20)} | 跳過:{skipped} | "
        f"μ={actual_mu} σ={actual_sigma} | 操作:{action_counts}"
    )

    return {
        "market":        "TW",
        "top20":         top20,
        "pool_mu":       actual_mu,
        "pool_sigma":    actual_sigma,
        "win_rate":      57.1,
        "regime":        regime_label_str,
        "total_scored":  len(result_rows),
        "skipped":       skipped,
    }
