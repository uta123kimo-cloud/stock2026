"""
╔══════════════════════════════════════════════════════════════╗
║  v12_engine.py — 資源法 V12.1 路徑交易決策引擎 v3.7          ║
║  介面：run(symbols, regime, v4_snapshot, today) → dict       ║
║                                                              ║
║  v3.7 變更：                                                 ║
║  - 完全移除 ^TWII 所有引用                                    ║
║  - .TW → .TWO fallback + 429 指數退讓                        ║
║  - 修正 _compute_basic_features 的 atr_a 命名錯誤            ║
║  - 純 numpy/pandas，無 pandas_ta 依賴                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
import math
import os
import json
import random
import time
from datetime import datetime, date
from typing import Optional

log = logging.getLogger("v12_engine")

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
# V12.1 核心參數
# ══════════════════════════════════════════════════════════════

PATH_DEFS = {
    "423": {
        "grade": "S", "order": ["Y4", "Y2", "Y3"],
        "tp1": 0.20, "tp2": 0.35,
        "ev_bear": 0.0096, "ev_range": 0.0834, "ev_bull": 0.0406,
        "oos_wr": 0.51, "oos_ev": 3.02,
    },
    "45": {
        "grade": "A", "order": ["Y4", "Y5"],
        "tp1": 0.20, "tp2": 0.28,
        "ev_bear": 0.0022, "ev_range": 0.0485, "ev_bull": 0.0432,
        "oos_wr": 0.55, "oos_ev": 7.36,
    },
}
ACTIVE_PATHS = {"423", "45"}

REGIME_STRATEGIES = {
    "bull":  {"active_path": "45",  "backup_path": "423",
              "path_ratio": {"45": 0.65, "423": 0.35},  "ev_thr_qual": 0.030, "max_pos": 4},
    "range": {"active_path": "423", "backup_path": "45",
              "path_ratio": {"423": 0.65, "45": 0.35},  "ev_thr_qual": 0.030, "max_pos": 5},
    "bear":  {"active_path": None,  "backup_path": None,
              "path_ratio": {},                           "ev_thr_qual": 0.040, "max_pos": 2},
}

EV_TIER_CORE  = 0.050
EV_TIER_MAIN  = 0.030
EV_TIER_FILL  = 0.020

PR_TRIGGER        = 90
FACTOR_DECAY_DAYS = 5
FACTOR_ALIVE_THR  = 0.80
PR_FRESHNESS      = 15
PR_LOOKBACK       = 10

FLICKER_TP1_SCALE = 0.80
FLICKER_EV_SCALE  = 1.20

TIME_DECAY_DAYS = 7
EV_DECAY_PCT    = 0.35
EV_ACCEL_PCT    = 0.20
EV_DECAY_SLOPE  = -0.01

SLOT_REPLACE_EV_MULT = 1.20

ATR_PENALTY_THR  = 2.5
SLOPE_ENTRY_MIN  = 0.0

ALL_Y_BETAS = {
    "Y1": {"bb_width": +.1527, "rsi_14": +.1156, "ma20_slope": -.0902,
           "price_mom_20": -.0650, "rs_vs_mkt_5": +.0518, "mkt_excess_z": -.0490,
           "inst_cum_10": -.0362, "vol_pvo": +.0276, "vol_pvo_sq": -.0271,
           "persist_count": +.0195, "atr_regime": +.0168},
    "Y2": {"bb_width": +.1428, "inst_cum_10": -.0652, "atr_regime": +.0420,
           "rs_vs_mkt_5": +.0379, "inst_pvo": +.0327, "rsi_14": -.0267,
           "inst_x_price": -.0239, "slope_5d": +.0224},
    "Y3": {"bb_width": +.1308, "price_mom_20": -.1043, "rsi_14": +.0592,
           "rs_vs_mkt_5": +.0260, "vri": +.0257, "persist_count": +.0210,
           "inst_cum_10": -.0185, "vol_pvo_sq": -.0168},
    "Y4": {"bb_width": +.2624, "price_mom_60": +.1054, "high52w_dist": -.0609,
           "inst_cum_10": -.0599, "close_ma5_r": +.0519, "inst_pvo": +.0372,
           "price_mom_5": -.0358, "vri": +.0347, "atr_regime": +.0329,
           "rsi_14": +.0202, "mkt_excess_z": +.0165},
    "Y5": {"bb_width": +.1872, "high52w_dist": -.0472, "price_mom_60": +.0462,
           "rs_vs_mkt_5": +.0339, "ma20_slope": -.0259, "vri": +.0209,
           "price_mom_5": +.0124, "inst_pvo": +.0098,
           "inst_cum_10": +.0068, "rsi_14": +.0062},
}

HISTORICAL_STATS = {
    "total_trades": 112, "win_rate": 57.1, "avg_ev": 5.29,
    "max_dd": -6.58, "sharpe": 5.36, "t_stat": 4.032,
    "simple_cagr": 96.9, "pl_ratio": 2.31,
}

MIN_DATA_ROWS = 20


# ══════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════

def _ev_tier_label(ev: float) -> str:
    if ev >= EV_TIER_CORE: return "⭐核心"
    if ev >= EV_TIER_MAIN: return "🔥主力"
    if ev >= EV_TIER_FILL: return "📌補位"
    return ""


def _get_regime_key(regime: dict) -> str:
    label  = str(regime.get("label", "震盪")).lower()
    bull   = regime.get("bull", 0.0)
    bear   = regime.get("bear", 0.0)
    strat  = regime.get("active_strategy", "")

    if strat:
        if "bull" in strat:  return "bull"
        if "bear" in strat:  return "bear"
        return "range"

    if "牛" in label or "bull" in label or bull > 0.55: return "bull"
    if "熊" in label or "bear" in label or bear > 0.60: return "bear"
    return "range"


def _calc_ev_soft(path_key: str, P_bear: float, P_range: float, P_bull: float) -> float:
    d = PATH_DEFS.get(path_key, {})
    if not d:
        return 0.0
    return (P_bear  * d.get("ev_bear",  0.0) +
            P_range * d.get("ev_range", 0.0) +
            P_bull  * d.get("ev_bull",  0.0))


# ══════════════════════════════════════════════════════════════
# 特徵計算（純 numpy/pandas，無 ^TWII 依賴）
# ══════════════════════════════════════════════════════════════

def _compute_basic_features(df) -> dict:
    """
    計算 Y 因子所需的基礎特徵。
    完全不依賴 ^TWII 或外部大盤資料；rs_vs_mkt_5 以 0.0 代替。
    已修正原始碼中 atr_a/atr_s 命名不一致的 bug。
    """
    if df is None or len(df) < MIN_DATA_ROWS or np is None or pd is None:
        return {}

    try:
        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        v = df["Volume"].values.astype(float)
        n = len(c)

        # MA
        ma5  = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        s20  = pd.Series(c).rolling(20).std().values

        # BB 寬度
        bb_width = np.where(ma20 > 0, 4 * s20 / (ma20 + 1e-9), 0.0)

        # RSI
        delta = pd.Series(c).diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).values / 100.0

        # ATR（修正：統一使用 atr_arr/atr_smooth 命名）
        atr_arr = np.zeros(n)
        for i in range(1, n):
            tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
            atr_arr[i] = tr
        atr_smooth = pd.Series(atr_arr).ewm(com=13, adjust=False).mean().values
        atr_pct    = np.where(c > 0, atr_smooth / (c + 1e-9), 0.03)
        atr_mean   = pd.Series(atr_pct).rolling(20).mean().values
        atr_reg    = np.where(atr_mean > 0, atr_pct / (atr_mean + 1e-9), 1.0)

        # 動量
        def _mom(lag):
            a = np.zeros(n)
            for i in range(lag, n):
                if c[i-lag] > 0:
                    a[i] = c[i] / c[i-lag] - 1
            return a
        pm5  = _mom(5)
        pm20 = _mom(20)
        pm60 = _mom(60)

        # MA20 斜率
        ma20sl = np.zeros(n)
        for i in range(5, n):
            if ma20[i-5] > 1e-9:
                ma20sl[i] = (ma20[i] - ma20[i-5]) / ma20[i-5]

        # Slope 5d
        slope_arr = np.zeros(n)
        for i in range(4, n):
            y_ = c[i-4:i+1]; x_ = np.arange(5, dtype=float)
            if y_[0] > 1e-9:
                s_, _ = np.polyfit(x_, y_, 1)
                slope_arr[i] = s_ / y_[0]

        # VRI
        vma10 = pd.Series(v * c).rolling(10).mean().values
        vri   = np.where(vma10 > 0, (v * c) / (vma10 + 1e-9), 1.0)

        # 52週高點
        h52  = pd.Series(h).rolling(252, min_periods=60).max().bfill().values
        h52d = np.where(h52 > 0, c / (h52 + 1e-9), 0.0)

        # close/ma5
        cma5r = np.where(ma5 > 0, (c - ma5) / (ma5 + 1e-9), 0.0)

        # Vol PVO
        vma20 = pd.Series(v).rolling(20).mean().values
        vpvo  = np.where(vma20 > 0, (v - vma20) / (vma20 + 1e-9), 0.0)
        vpvoq = vpvo ** 2
        ic10  = pd.Series(vpvo).rolling(10).sum().fillna(0).values

        vma5 = pd.Series(v).rolling(5).mean().values
        ipvo = np.where(vma20 > 0, (vma5 - vma20) / (vma20 + 1e-9), 0.0)
        ixp  = ic10 * pm5

        # persist_count
        mad  = np.where(ma20 > 0, ma5 - ma20, 0.0)
        ymad = np.where(s20 > 0, mad / (s20 + 1e-9), 0.0)
        pcnt = np.zeros(n); ct = 0
        for i in range(n):
            ct = (ct + 1) if ymad[i] > 0 else 0; pcnt[i] = ct
        persist_count = np.clip(pcnt / 60.0, 0.0, 1.0)

        i = -1   # 最後一列
        return {
            "bb_width":      float(bb_width[i]),
            "rsi_14":        float(rsi[i]),
            "ma20_slope":    float(ma20sl[i]),
            "price_mom_5":   float(pm5[i]),
            "price_mom_20":  float(pm20[i]),
            "price_mom_60":  float(pm60[i]),
            "rs_vs_mkt_5":   0.0,              # 無大盤資料，填 0
            "mkt_excess_z":  float(ymad[i]),
            "inst_cum_10":   float(ic10[i]),
            "vol_pvo":       float(vpvo[i]),
            "vol_pvo_sq":    float(vpvoq[i]),
            "persist_count": float(persist_count[i]),
            "atr_regime":    float(atr_reg[i]),
            "vri":           float(vri[i]),
            "close_ma5_r":   float(cma5r[i]),
            "inst_pvo":      float(ipvo[i]),
            "inst_x_price":  float(ixp[i]),
            "high52w_dist":  float(h52d[i]),
            "slope_5d":      float(slope_arr[i]),
            # 原始值（進場過濾用）
            "_close":        float(c[i]),
            "_atr_pct":      float(atr_pct[i]),
            "_atr_regime":   float(atr_reg[i]),
            "_slope_5d":     float(slope_arr[i]),
            "_ma20":         float(ma20[i]),
        }
    except Exception as e:
        log.warning(f"  特徵計算失敗: {e}")
        return {}


def _predict_y_score(features: dict, beta: dict, is_log=False) -> float:
    sc = sum(features.get(k, 0.0) * v for k, v in beta.items())
    if is_log and np is not None:
        sc = 1.0 / (1.0 + np.exp(-max(-30, min(30, sc))))
    return sc


def _compute_y_pr_single(features: dict) -> dict:
    result = {}
    for yn, beta in ALL_Y_BETAS.items():
        is_log = (yn == "Y5")
        sc = _predict_y_score(features, beta, is_log)
        result[f"s_{yn}"]  = sc
        result[f"PR_{yn}"] = 95.0 if sc > 0 else 50.0
    return result


# ══════════════════════════════════════════════════════════════
# 路徑識別（V12.1 修正①）
# ══════════════════════════════════════════════════════════════

def identify_path(pr_hist: list, P_bear: float, P_range: float, P_bull: float) -> dict:
    if not pr_hist:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "quality": "Pure"}

    recent = list(pr_hist)[-PR_LOOKBACK:]
    n      = len(recent)

    first_trig_raw = {}
    for i, (d, prs) in enumerate(recent):
        for yn in ["Y1", "Y2", "Y3", "Y4", "Y5"]:
            if yn not in first_trig_raw and prs.get(f"PR_{yn}", 0) >= PR_TRIGGER:
                first_trig_raw[yn] = {"idx": i, "date": d}

    if not first_trig_raw:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "quality": "Pure"}

    alive_thr  = PR_TRIGGER * FACTOR_ALIVE_THR
    latest_prs = recent[-1][1] if recent else {}
    ever_lost  = set()
    first_trig = {}

    for yn, trig_info in first_trig_raw.items():
        latest_pr  = latest_prs.get(f"PR_{yn}", 0)
        days_since = (n - 1) - trig_info["idx"]
        still_alive = (latest_pr >= alive_thr)
        is_decayed  = (days_since > FACTOR_DECAY_DAYS and not still_alive)

        had_gap = False
        if trig_info["idx"] < n - 1:
            for i in range(trig_info["idx"] + 1, n):
                pr_i = recent[i][1].get(f"PR_{yn}", 0)
                if pr_i < alive_thr:
                    had_gap = True; break

        if had_gap or is_decayed:
            ever_lost.add(yn)
        if not still_alive or is_decayed:
            continue

        first_trig[yn] = trig_info

    if not first_trig:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "quality": "Pure"}

    quality = "Flicker" if ever_lost else "Pure"

    last_trigger_idx = max(v["idx"] for v in first_trig.values())
    total_span = len(recent) - 1
    if total_span > 0:
        staleness = total_span - last_trigger_idx
        if staleness > PR_FRESHNESS // 3:
            return {"best": None, "batch": 0, "ev_soft": 0.0, "quality": quality}

    comp = {}; matched = []
    for pk in ACTIVE_PATHS:
        order = PATH_DEFS.get(pk, {}).get("order", [])
        done  = sum(1 for yn in order if yn in first_trig)
        comp[pk] = done
        if done == len(order):
            idxs = [first_trig[yn]["idx"] for yn in order]
            if idxs == sorted(idxs):
                matched.append(pk)

    best = None
    if matched:
        matched.sort(key=lambda k: (-PATH_DEFS[k].get("oos_ev", 0)))
        best = matched[0]
    elif comp:
        inc = [(k, v) for k, v in comp.items() if v >= 1]
        if inc:
            inc.sort(key=lambda x: (-x[1], -PATH_DEFS.get(x[0], {}).get("oos_ev", 0)))
            best = inc[0][0]

    batch   = comp.get(best, 0) if best else 0
    ev_soft = _calc_ev_soft(best, P_bear, P_range, P_bull) if best else 0.0

    return {
        "best":    best,
        "batch":   batch,
        "ev_soft": ev_soft,
        "comp":    comp,
        "quality": quality,
    }


# ══════════════════════════════════════════════════════════════
# 資料取得（.TW → .TWO，429 退讓，無 ^TWII）
# ══════════════════════════════════════════════════════════════

def _fetch_ohlcv(sym: str, period: str = "90d",
                  max_retries: int = 4) -> Optional[pd.DataFrame]:
    """
    先嘗試 .TW（上市），失敗再嘗試 .TWO（上櫃）。
    完全不使用 ^TWII。
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
# 出場信號判斷（V12.1 修正②）
# ══════════════════════════════════════════════════════════════

def _check_exit_signal(pos: dict, ev_now: float, slope_now: float,
                        days_held: int, curr_ret: float) -> str:
    ev_entry = pos.get("ev_soft", 0.0)

    if curr_ret <= -0.10:              return "硬停損"
    if pos.get("profit_locked", False) and curr_ret < 0.01:
        return "保本出場"
    if ev_now < 0.005:                 return "EV衰退"

    if days_held > TIME_DECAY_DAYS and ev_entry > 0:
        ev_drop_ratio = (ev_entry - ev_now) / ev_entry
        if ev_drop_ratio > EV_ACCEL_PCT and slope_now < EV_DECAY_SLOPE:
            return "Slope加速出場"
        if ev_drop_ratio > EV_DECAY_PCT:
            return "時間衰減"

    pvo_now = pos.get("pvo_now", 0.0)
    if days_held > 3 and pvo_now < -0.30:
        return "量能枯竭"

    return "—"


# ══════════════════════════════════════════════════════════════
# 主引擎介面
# ══════════════════════════════════════════════════════════════

def run(symbols: list, regime: dict, v4_snapshot: dict, today: str,
        daily_close: dict = None, prev_v12_path: str = None) -> dict:
    """
    V12.1 路徑交易決策引擎。
    大盤資料由 regime dict 提供，引擎本身不再抓取 ^TWII。
    """
    if np is None or pd is None:
        log.error("numpy / pandas 未安裝，V12 引擎無法運行")
        return {}

    regime_key  = _get_regime_key(regime)
    strategy    = REGIME_STRATEGIES.get(regime_key, REGIME_STRATEGIES["range"])
    active_path = strategy["active_path"]
    backup_path = strategy["backup_path"]
    ev_entry_min = strategy["ev_thr_qual"]
    max_pos      = strategy["max_pos"]
    path_ratio   = strategy.get("path_ratio", {})

    P_bear  = regime.get("bear",  0.33)
    P_range = regime.get("range", 0.34)
    P_bull  = regime.get("bull",  0.33)

    log.info(
        f"V12 引擎啟動 | 日期:{today} | 制度:{regime_key} | "
        f"主路徑:{active_path} | 最大部位:{max_pos}"
    )

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    day_dir   = os.path.join(data_root, today)

    # 載入前次部位
    old_positions = {}
    if prev_v12_path and os.path.exists(prev_v12_path):
        try:
            with open(prev_v12_path, encoding="utf-8") as f:
                prev_data = json.load(f)
            old_positions = {p["symbol"]: p for p in prev_data.get("positions", [])}
            log.info(f"  前次部位: {len(old_positions)} 個")
        except Exception as e:
            log.warning(f"  無法載入前次部位: {e}")

    top20     = v4_snapshot.get("top20", [])
    candidates = [
        r for r in top20
        if r.get("action") in ("強力買進", "買進") and r.get("score", 0) > 55
    ]
    log.info(f"  V4 候選股: {len(candidates)} 檔（TOP20={len(top20)}）")

    positions   = []
    path_counts = {}

    for cand in candidates[:12]:
        sym = cand["symbol"]
        time.sleep(random.uniform(0.8, 1.5))
        try:
            df = _load_from_csv(sym, day_dir)
            if df is None:
                df = _fetch_ohlcv(sym, "90d")
            if df is None or len(df) < MIN_DATA_ROWS:
                log.debug(f"  V12 {sym} 資料不足，跳過")
                continue

            features = _compute_basic_features(df)
            if not features:
                continue

            if features.get("_atr_regime", 1.0) > ATR_PENALTY_THR:
                log.debug(f"  {sym} ATR極端，跳過")
                continue
            if features.get("_slope_5d", 0.0) < SLOPE_ENTRY_MIN:
                log.debug(f"  {sym} Slope不足，跳過")
                continue

            y_prs = _compute_y_pr_single(features)
            info  = identify_path([(today, y_prs)], P_bear, P_range, P_bull)

            best_path = info.get("best")
            batch     = info.get("batch", 0)
            ev_soft   = info.get("ev_soft", 0.0)
            quality   = info.get("quality", "Pure")

            if best_path not in [active_path, backup_path]:
                best_path = active_path or "423"
                batch     = 1

            if ev_soft <= 0.0:
                ev_soft = _calc_ev_soft(best_path, P_bear, P_range, P_bull)

            ev_thr = ev_entry_min * (FLICKER_EV_SCALE if quality == "Flicker" else 1.0)
            if ev_soft < ev_thr:
                log.debug(f"  {sym} EV不足 ({ev_soft*100:.1f}% < {ev_thr*100:.1f}%)")
                continue

            max_same   = max(1, round(max_pos * path_ratio.get(best_path, 0.50)))
            same_count = path_counts.get(best_path, 0)
            if same_count >= max_same:
                log.debug(f"  {sym} 路徑 {best_path} 槽位已滿")
                continue

            if len(positions) >= max_pos:
                if positions:
                    min_ev = min(p.get("ev", 0.0) / 100 for p in positions)
                    if ev_soft < min_ev * SLOT_REPLACE_EV_MULT:
                        log.debug(f"  {sym} EV優勢不足，不替換")
                        continue

            last     = df.iloc[-1]
            close    = float(last.get("Close", 0))
            atr_raw  = float(features.get("_atr_pct", 0.02)) * close
            tp1_px   = round(close * (1 + PATH_DEFS.get(best_path, {}).get("tp1", 0.20)), 1)
            tp2_px   = round(close * (1 + PATH_DEFS.get(best_path, {}).get("tp2", 0.28)), 1)
            stop_px  = round(close - atr_raw * 1.5, 1)

            if quality == "Flicker":
                tp1_px = round(close * (1 + PATH_DEFS.get(best_path, {}).get("tp1", 0.20) * FLICKER_TP1_SCALE), 1)

            if sym in old_positions:
                old         = old_positions[sym]
                days_held   = old.get("days_held", 0) + 1
                entry_price = old.get("entry_price", close)
                curr_ret    = (close - entry_price) / (entry_price + 1e-9)
                stop_px     = old.get("stop_price", stop_px)
                tp1_px      = old.get("tp1_price",  tp1_px)
                action      = "持有"
                exit_signal = _check_exit_signal(
                    {
                        "ev_soft":       old.get("ev", ev_soft * 100) / 100,
                        "profit_locked": curr_ret > 0.10,
                        "pvo_now":       features.get("vol_pvo", 0.0),
                    },
                    ev_soft, features.get("_slope_5d", 0.0), days_held, curr_ret
                )
            else:
                days_held   = 0
                entry_price = close
                curr_ret    = 0.0
                action      = "進場"
                exit_signal = "—"

            ev_tier = _ev_tier_label(ev_soft)

            positions.append({
                "symbol":       sym,
                "path":         best_path,
                "ev":           round(ev_soft * 100, 2),
                "ev_tier":      ev_tier,
                "action":       action,
                "exit_signal":  exit_signal,
                "quality":      quality,
                "days_held":    days_held,
                "curr_ret_pct": round(curr_ret * 100, 2),
                "entry_price":  round(entry_price, 2),
                "tp1_price":    tp1_px,
                "tp2_price":    tp2_px,
                "stop_price":   stop_px,
                "regime":       regime_key,
                "close":        round(close, 2),
                "batch":        batch,
            })
            path_counts[best_path] = path_counts.get(best_path, 0) + 1
            log.info(f"  ✅ {sym} | {best_path} | EV:{ev_soft*100:+.2f}% | {quality} | {action}")

            if len(positions) >= max_pos:
                log.info(f"  部位已滿（{max_pos} 個）")
                break

        except Exception as e:
            log.warning(f"  V12 {sym} 跳過: {e}")

    log.info(f"V12 完成 ✅ | 部位:{len(positions)} | 路徑:{path_counts}")

    return {
        "market":      "TW",
        "positions":   positions,
        "stats":       HISTORICAL_STATS,
        "regime":      regime_key,
        "active_path": active_path,
        "backup_path": backup_path,
        "path_counts": path_counts,
    }
