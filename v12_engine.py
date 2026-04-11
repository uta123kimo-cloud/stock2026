"""
╔══════════════════════════════════════════════════════════════╗
║  v12_engine.py — 資源法 V12.1 路徑交易決策引擎                ║
║  來源：5Y-3W-V12.1.py《精煉實戰版》                           ║
║  介面：run(symbols, regime, v4_snapshot, today) → dict       ║
║                                                              ║
║  V12.1 五大精準修正（完整繼承）：                              ║
║    修正① 路徑有效性：因子存活確認 + Decay + Pure/Flicker       ║
║    修正② EV時間衰退：35%單條件出場（恢復V11.3節奏）            ║
║    修正③ 路徑槽位均衡：震盪423=65%/45=35%，牛市45=65%/423=35% ║
║    修正④ 槽位換股門檻：新股EV > 持倉最低EV × 1.20              ║
║    修正⑤ Pure/Flicker差異化風控                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
import math
import os
import json
from collections import deque
from datetime import datetime, date

log = logging.getLogger("v12_engine")

# ── 選用性依賴 ────────────────────────────────────────────────────
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

# [FIX] pandas_ta 已移除 → 使用 v12_engine 內建純 numpy 特徵計算
_USES_PANDAS_TA = False

# ══════════════════════════════════════════════════════════════════
# V12.1 核心參數（完整繼承自原始 V12.1）
# ══════════════════════════════════════════════════════════════════

# ── 路徑定義（OOS 統計結果）────────────────────────────────────────
PATH_DEFS = {
    "423": {
        "grade": "S", "order": ["Y4", "Y2", "Y3"],
        "tp1": 0.20, "tp2": 0.35,
        "ev_bear": 0.0096, "ev_range": 0.0834, "ev_bull": 0.0406,
        "oos_wr": 0.51, "oos_ev": 3.02,
        "desc": "震盪+10%=39.0% Alpha+17.7pp",
    },
    "45": {
        "grade": "A", "order": ["Y4", "Y5"],
        "tp1": 0.20, "tp2": 0.28,
        "ev_bear": 0.0022, "ev_range": 0.0485, "ev_bull": 0.0432,
        "oos_wr": 0.55, "oos_ev": 7.36,
        "desc": "牛市+10%=26.8% Alpha+11.1pp",
    },
}
ACTIVE_PATHS = {"423", "45"}

# ── Regime 策略配置（V12.1 修正③：均衡化路徑槽位）────────────────
REGIME_STRATEGIES = {
    "bull": {
        "active_path": "45",
        "backup_path": "423",
        "path_ratio":  {"45": 0.65, "423": 0.35},   # V12.1：牛市均衡
        "ev_thr_qual": 0.030,
        "max_pos":     4,
    },
    "range": {
        "active_path": "423",
        "backup_path": "45",
        "path_ratio":  {"423": 0.65, "45": 0.35},   # V12.1：震盪均衡
        "ev_thr_qual": 0.030,
        "max_pos":     5,
    },
    "bear": {
        "active_path": None,
        "backup_path": None,
        "path_ratio":  {},
        "ev_thr_qual": 0.040,
        "max_pos":     2,
    },
}

# ── EV 分層門檻（用於標示，非進場條件）──────────────────────────
EV_TIER_CORE  = 0.050   # ⭐核心
EV_TIER_MAIN  = 0.030   # 🔥主力
EV_TIER_FILL  = 0.020   # 📌補位
EV_ENTRY_MIN  = 0.030   # 統一進場門檻 ≥ 3%

# ── V12.1 修正①：路徑有效性參數 ────────────────────────────────
PR_TRIGGER        = 90    # Y因子觸發 PR 門檻
FACTOR_DECAY_DAYS = 5     # 觸發後超過N天且PR不足 → Decay失效
FACTOR_ALIVE_THR  = 0.80  # 存活門檻（容忍小幅回落）
PR_FRESHNESS      = 15    # 最大允許 staleness
PR_LOOKBACK       = 10    # 路徑識別回看天數

# ── V12.1 修正①：Pure/Flicker 差異化風控 ────────────────────────
FLICKER_TP1_SCALE = 0.80  # Flicker 停利①提早鎖利
FLICKER_EV_SCALE  = 1.20  # Flicker 進場EV門檻提高

# ── V12.1 修正②：EV時間衰退單條件 ──────────────────────────────
TIME_DECAY_DAYS = 7       # 持有超過N天啟動EV衰減檢查
EV_DECAY_PCT    = 0.35    # EV下降 > 35% 即出場（V12=40%雙條件）
EV_ACCEL_PCT    = 0.20    # EV下降 > 20% 且 Slope<0 → 加速出場
EV_DECAY_SLOPE  = -0.01   # Slope < -1% 強制出場補充條件

# ── V12.1 修正④：換股門檻 ────────────────────────────────────────
SLOT_REPLACE_EV_MULT = 1.20  # 新股EV > 被替換股EV × 1.20

# ── 路徑信心係數（OOS 實測） ────────────────────────────────────
PATH_CONFIDENCE = {
    "423": 0.65,
    "45":  0.58,
}
BATCH_WEIGHT = {3: 1.00, 2: 0.82, 1: 0.62}

# ── 進場過濾 ────────────────────────────────────────────────────
ATR_PENALTY_THR   = 2.5   # ATR 極端過濾門檻（超過排除）
SLOPE_ENTRY_MIN   = 0.0   # 正斜率門檻
SLOPE_CONSEC_DAYS = 2     # Slope 連續確認天數

# ── Y 因子 Beta 係數（完整繼承自原始 V12.1）──────────────────────
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

# ── 歷史統計（固定參考值，OOS 實測）────────────────────────────
HISTORICAL_STATS = {
    "total_trades": 112,
    "win_rate":     57.1,
    "avg_ev":       5.29,
    "max_dd":      -6.58,
    "sharpe":       5.36,
    "t_stat":       4.032,
    "simple_cagr":  96.9,
    "pl_ratio":     2.31,
}

MIN_DATA_ROWS = 20


# ══════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════

def _ev_tier_label(ev: float) -> str:
    if ev >= EV_TIER_CORE: return "⭐核心"
    if ev >= EV_TIER_MAIN: return "🔥主力"
    if ev >= EV_TIER_FILL: return "📌補位"
    return ""


def _get_regime_key(regime: dict) -> str:
    """從 regime dict 解析出 bull/range/bear"""
    label   = str(regime.get('label', '震盪')).lower()
    bull    = regime.get('bull', 0.0)
    bear    = regime.get('bear', 0.0)
    strat   = regime.get('active_strategy', '')

    if strat:
        if 'bull' in strat: return 'bull'
        if 'bear' in strat: return 'bear'
        return 'range'

    if '牛' in label or 'bull' in label or bull > 0.55:
        return 'bull'
    if '熊' in label or 'bear' in label or bear > 0.60:
        return 'bear'
    return 'range'


def _calc_ev_soft(path_key: str, P_bear: float, P_range: float, P_bull: float) -> float:
    """軟 Regime 加權 EV 計算"""
    d = PATH_DEFS.get(path_key, {})
    if not d:
        return 0.0
    return (P_bear  * d.get("ev_bear",  0.0) +
            P_range * d.get("ev_range", 0.0) +
            P_bull  * d.get("ev_bull",  0.0))


# ══════════════════════════════════════════════════════════════════
# 簡化的 Y 因子 PR 計算（不需投信資料的版本）
# ══════════════════════════════════════════════════════════════════

def _compute_basic_features(df) -> dict:
    """
    計算 Y 因子所需的基礎特徵（不需投信資料）。
    完整版需要 FinMind 投信資料，此處使用代理指標。
    """
    if df is None or len(df) < MIN_DATA_ROWS or np is None or pd is None:
        return {}

    try:
        c = df['Close'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        v = df['Volume'].values.astype(float)
        n = len(c)

        # MA 指標
        ma5  = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        s20  = pd.Series(c).rolling(20).std().values

        # BB 寬度
        bb_width = np.where(ma20 > 0, (ma20 + 2*s20 - (ma20 - 2*s20)) / (ma20 + 1e-9), 0.0)

        # RSI（使用 EWM 近似）
        delta = pd.Series(c).diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).values / 100.0

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

        # ATR（簡化版）
        atr = np.zeros(n)
        for i in range(1, n):
            tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            atr[i] = tr
        atr_s    = pd.Series(atr_a).ewm(com=13, adjust=False).mean().values
        atr_pct  = np.where(c > 0, atr_s / (c + 1e-9), 0.03)
        atr_mean = pd.Series(atr_pct).rolling(20).mean().values
        atr_reg  = np.where(atr_mean > 0, atr_pct / (atr_mean + 1e-9), 1.0)

        # VRI（量能熱度）
        vma = pd.Series(v).rolling(10).mean().values
        vri = np.where(vma > 0, (v * c) / (vma * c + 1e-9), 1.0)

        # 52週高點距離
        h52 = pd.Series(h).rolling(252, min_periods=60).max().bfill().values
        h52d = np.where(h52 > 0, c / (h52 + 1e-9), 0.0)

        # close / ma5
        cma5r = np.where(ma5 > 0, (c - ma5) / (ma5 + 1e-9), 0.0)

        # Slope 5d（線性回歸斜率）
        slope_arr = np.zeros(n)
        for i in range(4, n):
            y_ = c[i-4:i+1]
            x_ = np.arange(5, dtype=float)
            if y_[0] > 1e-9:
                s_, _ = np.polyfit(x_, y_, 1)
                slope_arr[i] = s_ / y_[0]

        # MA20 斜率
        ma20sl = np.zeros(n)
        for i in range(5, n):
            if ma20[i-5] > 1e-9:
                ma20sl[i] = (ma20[i] - ma20[i-5]) / ma20[i-5]

        # Vol PVO
        vma20 = pd.Series(v).rolling(20).mean().values
        vpvo  = np.where(vma20 > 0, (v - vma20) / (vma20 + 1e-9), 0.0)
        vpvo_sq = vpvo ** 2
        ic10  = pd.Series(vpvo).rolling(10).sum().fillna(0).values

        # VMA5
        vma5   = pd.Series(v).rolling(5).mean().values
        ipvo   = np.where(vma20 > 0, (vma5 - vma20) / (vma20 + 1e-9), 0.0)
        ixp    = ic10 * pm5

        # rs_vs_mkt_5（無大盤資料時用0）
        rvs = np.zeros(n)

        # persist_count（連續在MA20上方天數佔比）
        mad = np.where(ma20 > 0, ma5 - ma20, 0.0)
        ymad = np.where(s20 > 0, mad / (s20 + 1e-9), 0.0)
        pcnt = np.zeros(n)
        ct = 0
        for i in range(n):
            ct = (ct + 1) if ymad[i] > 0 else 0
            pcnt[i] = ct
        persist_count = np.clip(pcnt / 60.0, 0.0, 1.0)

        # mkt_excess_z（無大盤資料時用0）
        mkt_excess_z = ymad.copy()  # 用個股自身 zscore 近似

        # inst 相關（無投信資料，用代理）
        inst_cum_10 = ic10      # 用 vol_pvo 累積代理
        inst_pvo    = ipvo
        inst_x_price = ixp

        last = -1   # 取最後一行
        return {
            "bb_width":       float(bb_width[last]),
            "rsi_14":         float(rsi[last]),
            "ma20_slope":     float(ma20sl[last]),
            "price_mom_5":    float(pm5[last]),
            "price_mom_20":   float(pm20[last]),
            "price_mom_60":   float(pm60[last]),
            "rs_vs_mkt_5":    float(rvs[last]),
            "mkt_excess_z":   float(mkt_excess_z[last]),
            "inst_cum_10":    float(inst_cum_10[last]),
            "vol_pvo":        float(vpvo[last]),
            "vol_pvo_sq":     float(vpvo_sq[last]),
            "persist_count":  float(persist_count[last]),
            "atr_regime":     float(atr_reg[last]),
            "vri":            float(vri[last]),
            "close_ma5_r":    float(cma5r[last]),
            "inst_pvo":       float(inst_pvo[last]),
            "inst_x_price":   float(inst_x_price[last]),
            "high52w_dist":   float(h52d[last]),
            "slope_5d":       float(slope_arr[last]),
            # 原始值（進場過濾用）
            "_close":         float(c[last]),
            "_atr_raw":       float(atr_s[last]),
            "_atr_pct":       float(atr_pct[last]),
            "_atr_regime":    float(atr_reg[last]),
            "_slope_5d":      float(slope_arr[last]),
            "_ma20":          float(ma20[last]),
        }
    except Exception as e:
        log.warning(f"  特徵計算失敗: {e}")
        return {}


def _predict_y_score(features: dict, beta: dict, is_log=False) -> float:
    """根據 Beta 係數計算 Y 因子分數"""
    sc = sum(features.get(k, 0.0) * v for k, v in beta.items())
    if is_log:
        sc = 1.0 / (1.0 + np.exp(-max(-30, min(30, sc))))
    return sc


def _compute_y_pr_single(features: dict) -> dict:
    """
    計算單一個股的 Y1~Y5 分數（標準化版本）。
    由於引擎每次只計算當日截面，使用絕對分數代替百分位。
    """
    result = {}
    for yn, beta in ALL_Y_BETAS.items():
        is_log = (yn == "Y5")
        sc = _predict_y_score(features, beta, is_log)
        result[f"s_{yn}"] = sc
        # 使用簡化 PR：以絕對分數 > 0 視為達到門檻
        # 完整版應跨截面計算百分位，此處用相對強弱近似
        result[f"PR_{yn}"] = 95.0 if sc > 0 else 50.0
    return result


# ══════════════════════════════════════════════════════════════════
# 【V12.1 修正①】路徑有效性識別（完整繼承）
# ══════════════════════════════════════════════════════════════════

def identify_path(pr_hist: list, P_bear: float, P_range: float, P_bull: float) -> dict:
    """
    V12.1 路徑識別（完整繼承自原始 V12.1）。

    核心修正①：
    - 因子存活確認：最新PR < PR_TRIGGER × FACTOR_ALIVE_THR → 從first_trig移除
    - Decay失效：觸發後超過FACTOR_DECAY_DAYS天且最新PR不足 → 清除
    - Pure/Flicker標記：全程有效=Pure，曾失效=Flicker
    """
    if not pr_hist:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "hit10_soft": 0.0}

    recent = list(pr_hist)[-PR_LOOKBACK:]
    n = len(recent)

    # 第一輪：記錄首次觸發位置
    first_trig_raw = {}
    for i, (d, prs) in enumerate(recent):
        for yn in ["Y1", "Y2", "Y3", "Y4", "Y5"]:
            if yn not in first_trig_raw and prs.get(f"PR_{yn}", 0) >= PR_TRIGGER:
                first_trig_raw[yn] = {"idx": i, "date": d}

    if not first_trig_raw:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "hit10_soft": 0.0}

    # V12.1 修正①：因子存活確認 + Decay失效篩選
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
                    had_gap = True
                    break

        if had_gap or is_decayed:
            ever_lost.add(yn)

        if not still_alive or is_decayed:
            continue

        first_trig[yn] = trig_info

    if not first_trig:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "hit10_soft": 0.0}

    quality = "Flicker" if ever_lost else "Pure"

    # 過時性檢查
    last_trigger_idx = max(v["idx"] for v in first_trig.values())
    total_span = len(recent) - 1
    if total_span > 0:
        staleness = (total_span - last_trigger_idx)
        if staleness > PR_FRESHNESS // 3:
            return {"best": None, "batch": 0, "ev_soft": 0.0, "hit10_soft": 0.0}

    # Y2 否決邏輯
    y2_veto = False
    if "Y2" in first_trig:
        y2_idx    = first_trig["Y2"]["idx"]
        is_first  = (y2_idx == min(v["idx"] for v in first_trig.values()))
        has_y4_after = (("Y4" in first_trig) and first_trig["Y4"]["idx"] > y2_idx)
        has_y5_after = (("Y5" in first_trig) and first_trig["Y5"]["idx"] > y2_idx)
        if is_first and not has_y4_after and not has_y5_after:
            y2_veto = True

    # 路徑比對（只比對 ACTIVE_PATHS）
    comp    = {}
    matched = []
    for pk in ACTIVE_PATHS:
        ps    = PATH_DEFS.get(pk, {})
        order = ps.get("order", [])
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

    batch    = comp.get(best, 0) if best else 0
    ev_soft  = _calc_ev_soft(best, P_bear, P_range, P_bull) if best else 0.0

    return {
        "best":       best,
        "batch":      batch,
        "ev_soft":    ev_soft,
        "y2_veto":    y2_veto,
        "comp":       comp,
        "first_trig": first_trig,
        "quality":    quality,
        "ever_lost":  ever_lost,
    }


# ══════════════════════════════════════════════════════════════════
# 資料取得
# ══════════════════════════════════════════════════════════════════

def _fetch_ohlcv(sym: str, period: str = '90d'):
    """嘗試 .TW → .TWO 取得個股 OHLCV"""
    if yf is None or pd is None:
        return None
    for suffix in ['.TW', '.TWO']:
        ticker = f"{sym}{suffix}"
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip() for c in df.columns]
            if not df.empty and len(df) >= MIN_DATA_ROWS:
                return df
        except Exception as e:
            log.debug(f"  {ticker} 下載失敗: {e}")
    return None


def _load_from_csv(sym: str, day_dir: str):
    """優先從每日 CSV 讀取"""
    if pd is None:
        return None
    csv_path = os.path.join(day_dir, f"{sym}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.columns = [str(c).strip() for c in df.columns]
        if len(df) >= MIN_DATA_ROWS:
            return df
    except Exception as e:
        log.debug(f"  {sym} CSV讀取失敗: {e}")
    return None


# ══════════════════════════════════════════════════════════════════
# 出場信號判斷（V12.1 完整繼承）
# ══════════════════════════════════════════════════════════════════

def _check_exit_signal(pos: dict, ev_now: float, slope_now: float,
                        days_held: int, curr_ret: float) -> str:
    """
    V12.1 出場信號判斷（簡化版，適用於 daily_run 快照場景）

    優先級：
    1. 硬停損（-10%）
    2. 保本出場（獲利鎖定後）
    3. EV 衰退（降至0.5%以下）
    4. Slope 加速出場（EV↓>20% 且 Slope<0）
    5. 時間衰減（EV↓>35%，V12.1 單條件）
    """
    ev_entry = pos.get("ev_soft", 0.0)

    # 硬停損
    if curr_ret <= -0.10:
        return "硬停損"

    # 保本出場
    if pos.get("profit_locked", False) and curr_ret < 0.01:
        return "保本出場"

    # EV 衰退（跌至極低）
    if ev_now < 0.005:
        return "EV衰退"

    # 時間衰減判斷（V12.1 修正②）
    if days_held > TIME_DECAY_DAYS and ev_entry > 0:
        ev_drop_ratio = (ev_entry - ev_now) / ev_entry

        # 加速出場：EV小幅下降且Slope已轉負
        if ev_drop_ratio > EV_ACCEL_PCT and slope_now < EV_DECAY_SLOPE:
            return f"Slope加速出場"

        # 單條件出場：EV下降超過門檻（不需Slope確認）
        if ev_drop_ratio > EV_DECAY_PCT:
            return "時間衰減"

    # 量能枯竭（簡化判斷）
    pvo_now = pos.get("pvo_now", 0.0)
    if days_held > 3 and pvo_now < -0.30:
        return "量能枯竭"

    return "—"


# ══════════════════════════════════════════════════════════════════
# 主引擎介面
# ══════════════════════════════════════════════════════════════════

def run(symbols: list, regime: dict, v4_snapshot: dict, today: str,
        daily_close: dict = None, prev_v12_path: str = None) -> dict:
    """
    V12.1 路徑交易決策引擎主函式。

    Args:
        symbols:       個股代號清單（不含後綴）
        regime:        regime_engine 輸出的 dict
        v4_snapshot:   v4_engine.run() 回傳的 dict（用於候選股篩選）
        today:         計算日期字串，格式 'YYYY-MM-DD'
        daily_close:   step_save_daily_data() 回傳的 {sym: {close,...}}
        prev_v12_path: 前一次 v12_latest.json 的路徑（用於持倉延續）

    Returns:
        V12.1 快照 dict，包含：
            positions   — 當前部位列表
            stats       — 歷史統計（固定參考值）
            regime      — 當前策略制度
            active_path / backup_path
    """
    if np is None or pd is None:
        log.error("❌ numpy / pandas 未安裝，V12 引擎無法運行")
        return {}

    # ── 解析 Regime ───────────────────────────────────────────────
    regime_key      = _get_regime_key(regime)
    strategy        = REGIME_STRATEGIES.get(regime_key, REGIME_STRATEGIES["range"])
    active_path     = strategy["active_path"]
    backup_path     = strategy["backup_path"]
    ev_entry_min    = strategy["ev_thr_qual"]
    max_pos         = strategy["max_pos"]
    path_ratio      = strategy.get("path_ratio", {})

    P_bear  = regime.get("bear",  0.33)
    P_range = regime.get("range", 0.34)
    P_bull  = regime.get("bull",  0.33)

    log.info(f"V12 引擎啟動 | 日期: {today} | 制度: {regime_key} | 主路徑: {active_path} | 最大部位: {max_pos}")

    # ── 確定每日 CSV 目錄 ─────────────────────────────────────────
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    day_dir   = os.path.join(data_root, today)

    # ── 載入前一次部位（延續持倉狀態）────────────────────────────
    old_positions = {}
    if prev_v12_path and os.path.exists(prev_v12_path):
        try:
            with open(prev_v12_path, encoding='utf-8') as f:
                prev_data = json.load(f)
            old_positions = {p["symbol"]: p for p in prev_data.get("positions", [])}
            log.info(f"  載入前次部位: {len(old_positions)} 個")
        except Exception as e:
            log.warning(f"  無法載入前次部位: {e}")

    # ── 從 V4 篩選候選股 ──────────────────────────────────────────
    top20     = v4_snapshot.get("top20", [])
    v4_scores = {r["symbol"]: r for r in top20}

    # 候選股：V4 強力買進/買進 且分數 > 55
    candidates = [
        r for r in top20
        if r.get("action") in ("強力買進", "買進") and r.get("score", 0) > 55
    ]
    log.info(f"  V4 候選股: {len(candidates)} 檔（from TOP20={len(top20)}）")

    # ── PR 歷史記錄（用於路徑識別）──────────────────────────────
    # 注意：完整版需要累積多日 PR_history。
    # 此處為每次快照版本，使用當日特徵計算 Y 因子分數建立單日歷史。
    pr_history = {}   # {sym: [(date_str, {PR_Y1: ..., PR_Y2: ...})]}

    positions = []
    path_counts = {}   # 各路徑已用槽位數

    for cand in candidates[:12]:   # 最多考慮前12個候選股
        sym = cand["symbol"]
        try:
            # 載入 OHLCV
            df = _load_from_csv(sym, day_dir)
            if df is None:
                df = _fetch_ohlcv(sym, period='90d')
            if df is None or len(df) < MIN_DATA_ROWS:
                log.warning(f"  V12 {sym} 資料不足，跳過")
                continue

            # 計算基礎特徵
            features = _compute_basic_features(df)
            if not features:
                continue

            # ATR 極端過濾（V12 修正F）
            atr_regime = features.get("_atr_regime", 1.0)
            if atr_regime > ATR_PENALTY_THR:
                log.debug(f"  {sym} ATR極端({atr_regime:.1f}x)，跳過")
                continue

            # Slope 門檻過濾
            slope_now = features.get("_slope_5d", 0.0)
            if slope_now < SLOPE_ENTRY_MIN:
                log.debug(f"  {sym} Slope不足({slope_now:.3f})，跳過")
                continue

            # 計算 Y 因子 PR
            y_prs = _compute_y_pr_single(features)
            pr_history[sym] = [(today, y_prs)]

            # 路徑識別（V12.1 修正①）
            info = identify_path(
                [(today, y_prs)],
                P_bear, P_range, P_bull
            )

            best_path = info.get("best")
            batch     = info.get("batch", 0)
            ev_soft   = info.get("ev_soft", 0.0)
            quality   = info.get("quality", "Pure")

            # 路徑過濾：只接受 active 或 backup 路徑
            if best_path not in [active_path, backup_path]:
                # fallback：根據 regime 直接分配路徑
                best_path = active_path or "423"
                batch     = 1

            if info.get("y2_veto", False):
                log.debug(f"  {sym} Y2否決，跳過")
                continue

            # 若無 ev_soft（路徑識別失敗），用統計默認值
            if ev_soft <= 0.0:
                ev_soft = _calc_ev_soft(best_path, P_bear, P_range, P_bull)

            # V12.1 修正①：Flicker 提高 EV 門檻
            ev_thr = ev_entry_min * (FLICKER_EV_SCALE if quality == "Flicker" else 1.0)
            if ev_soft < ev_thr:
                log.debug(f"  {sym} EV不足({ev_soft*100:.1f}% < {ev_thr*100:.1f}%)，跳過")
                continue

            # 槽位控制（V12.1 修正③）
            ratio_for_path = path_ratio.get(best_path, 0.50)
            max_same       = max(1, round(max_pos * ratio_for_path))
            same_count     = path_counts.get(best_path, 0)
            if same_count >= max_same:
                log.debug(f"  {sym} 路徑 {best_path} 槽位已滿({same_count}/{max_same})")
                continue

            # 槽位換股門檻（V12.1 修正④）
            if len(positions) >= max_pos:
                if positions:
                    min_ev = min(p.get("ev", 0.0) / 100 for p in positions)
                    if ev_soft < min_ev * SLOT_REPLACE_EV_MULT:
                        log.debug(f"  {sym} EV優勢不足，不替換現有部位")
                        continue

            # 計算停損/停利價位
            last       = df.iloc[-1]
            close      = float(last.get('Close', 0))
            atr_raw    = float(features.get("_atr_pct", 0.02)) * close
            tp1_px     = round(close * (1 + PATH_DEFS.get(best_path, {}).get("tp1", 0.20)), 1)
            tp2_px     = round(close * (1 + PATH_DEFS.get(best_path, {}).get("tp2", 0.28)), 1)
            stop_px    = round(close - atr_raw * 1.5, 1)

            # 延續前次部位資訊（如有）
            if sym in old_positions:
                old         = old_positions[sym]
                days_held   = old.get("days_held", 0) + 1
                entry_price = old.get("entry_price", close)
                curr_ret    = (close - entry_price) / (entry_price + 1e-9)
                stop_px     = old.get("stop_price",  stop_px)
                tp1_px      = old.get("tp1_price",   tp1_px)
                action      = "持有"
                # 利用舊的 EV 判斷出場信號
                old_ev = old.get("ev", ev_soft * 100) / 100
                pos.update({})  # 佔位
                exit_signal = _check_exit_signal(
                    {
                        "ev_soft":       old_ev,
                        "profit_locked": curr_ret > 0.10,
                        "pvo_now":       features.get("vol_pvo", 0.0),
                    },
                    ev_soft, slope_now, days_held, curr_ret
                )
                # 若有出場信號，標記但仍保留在快照中（讓 App 顯示警示）
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
                "ev":           round(ev_soft * 100, 2),   # 轉為%顯示
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
            log.info(f"  ✅ {sym} | 路徑:{best_path} | EV:{ev_soft*100:+.2f}% | {quality} | {action}")

            if len(positions) >= max_pos:
                log.info(f"  部位已滿（{max_pos} 個），停止新增")
                break

        except Exception as e:
            log.warning(f"  V12 {sym} 跳過: {e}")
            continue

    log.info(f"V12 完成 ✅ | 部位數: {len(positions)} | 路徑分佈: {path_counts}")

    return {
        "market":      "TW",
        "positions":   positions,
        "stats":       HISTORICAL_STATS,
        "regime":      regime_key,
        "active_path": active_path,
        "backup_path": backup_path,
        "path_counts": path_counts,
    }
