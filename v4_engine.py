"""
╔══════════════════════════════════════════════════════════════╗
║  v4_engine.py — 資源法 V4 市場強度引擎                        ║
║  來源：資源法2026-V4《機構穩態升級版》                         ║
║  介面：run(symbols, regime, today) → dict                    ║
║                                                              ║
║  四大核心升級（繼承自原始 V4）：                               ║
║    升級1：多因子共振引擎（A/B/C 型訊號鬆綁門檻）               ║
║    升級2：互斥獨立事件統計框架                                  ║
║    升級3：動態前置止損（取代硬止損）                            ║
║    升級4：Regime 分類器（趨勢/震盪/崩跌/回升）                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
import math
import os
from datetime import datetime, date

log = logging.getLogger("v4_engine")

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

# [FIX] pandas_ta 已移除 → 使用內建純 numpy/pandas 實作
# 設定旗標供 daily_run.py 識別
_USES_PANDAS_TA = False

# ══════════════════════════════════════════════════════════════════
# [FIX] 純 numpy/pandas 指標函式（取代 pandas_ta，零外部依賴）
# ══════════════════════════════════════════════════════════════════

def _ta_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))

def _ta_atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high-low,(high-pc).abs(),(low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period, adjust=False).mean()

def _ta_sma(close, period):
    return close.rolling(period).mean()

# ══════════════════════════════════════════════════════════════════
# 【V4-升級1】訊號門檻參數（完整繼承自原始 V4）
# ══════════════════════════════════════════════════════════════════
PVO_FIRE_THR      = 8.0    # B型點火門檻（V3=10 → V4=8）
PVO_FLOW_THR      = 0.0    # 資金流入門檻
VRI_HOT_THR       = 70.0   # VRI 熱度門檻（V3=75 → V4=70）
VRI_COOL_THR      = 45.0   # VRI 整理冷區（V3=40 → V4=45）
SLOPE_Z_STRONG    = 1.2    # Slope Z-Score 強勢門檻（V3=1.5 → V4=1.2）

PVO_CONSEC_MIN_A  = 1      # A型最低連續天數（V3=2 → V4=1）
PVO_CONSEC_DAYS   = 3      # B型加速確認連續天數
PVO_ACCEL_MULT    = 1.3    # B型加速乘數（V3=1.5 → V4=1.3）

VRI_DELTA_STRONG  = 3.0    # B型VRI加速門檻（V3=5.0 → V4=3.0）
VRI_DELTA_COOL    = 2.0    # A型VRI上限（未過熱）
VRI_DELTA_CUTOFF  = -5.0   # C型排除急退燒（V3=-3.0 → V4=-5.0）

VOL_RATIO_STRONG  = 2.0    # 量爆門檻（V3=2.5 → V4=2.0）
VOL_RATIO_NORMAL  = 1.2    # 放量門檻（V3=1.5 → V4=1.2）
VOL_RATIO_WEAK    = 0.7    # 縮量判斷

# ── 倉位權重（依訊號型態）────────────────────────────────────────
POS_WEIGHT = {
    '三合一(ABC)': 0.28,
    '二合一(AB)':  0.23,
    '二合一(AC)':  0.23,
    '二合一(BC)':  0.23,
    '單一(A)':     0.18,
    '單一(B)':     0.18,
    '單一(C)':     0.18,
    '基準-強勢':   0.15,
    '基準-持有':   0.13,
}
POS_DEFAULT_WEIGHT = 0.15

# ── 評分池滾動視窗 ────────────────────────────────────────────────
LOOKBACK_WINDOW = 30

# ── 個股過濾門檻 ──────────────────────────────────────────────────
MIN_PRICE = 100
MAX_PRICE = 1000
MIN_DATA_ROWS = 20

# ══════════════════════════════════════════════════════════════════
# 指標計算工具函式（完整繼承自原始 V4）
# ══════════════════════════════════════════════════════════════════

def _calc_pvo(df, fast=5, slow=20):
    """量能振盪指標 (Price Volume Oscillator)"""
    if 'Volume' not in df.columns:
        return pd.Series(0.0, index=df.index)
    vol_fast = df['Volume'].ewm(span=fast).mean()
    vol_slow = df['Volume'].ewm(span=slow).mean()
    return (vol_fast - vol_slow) / (vol_slow + 1e-9) * 100


def _calc_vri(df, window=20):
    """
    量能熱度指標 (Volume-RSI Index)
    RSI（方向）× 60% + ATR正規化（幅度）× 40%
    """
    if 'RSI' not in df.columns or 'ATR' not in df.columns:
        return pd.Series(50.0, index=df.index)
    rsi_norm = df['RSI'].clip(0, 100)
    atr_mean = df['ATR'].rolling(window).mean()
    atr_norm = (df['ATR'] / (atr_mean + 1e-9) * 50).clip(0, 100)
    return (rsi_norm * 0.6 + atr_norm * 0.4).clip(0, 100)


def _calc_pvo_consec(df, thr=PVO_FLOW_THR):
    """PVO 連續正值天數"""
    pvo = df['PVO']
    consec = pd.Series(0, index=df.index)
    cnt = 0
    for i, v in enumerate(pvo):
        cnt = cnt + 1 if v > thr else 0
        consec.iloc[i] = cnt
    return consec


def _calc_pvo_accel(df, window=5):
    """PVO 加速度（相對均值）"""
    pvo_mean = df['PVO'].rolling(window).mean()
    return df['PVO'] / (pvo_mean.abs() + 1e-9)


def _calc_vri_delta(df, window=5):
    """VRI 相對均值的偏差（情緒加速/退燒）"""
    vri_mean = df['VRI'].rolling(window).mean()
    return df['VRI'] - vri_mean


def _calc_vol_ratio(df, window=20):
    """量比（當日量 / 20日均量）"""
    if 'Volume' not in df.columns:
        return pd.Series(1.0, index=df.index)
    vol_mean = df['Volume'].rolling(window).mean()
    return df['Volume'] / (vol_mean + 1e-9)


# ══════════════════════════════════════════════════════════════════
# 【V4-升級1】多因子共振訊號分類
# ══════════════════════════════════════════════════════════════════

def classify_signal_v4(pvo, vri, slope_z, sc, mu_score, sigma_score,
                        pvo_consec, pvo_accel, vri_delta, vol_ratio):
    """
    V4 訊號分類器（完整繼承自原始 V4）

    型態 A（資金布局型）：PVO 流入區 + VRI 整理冷區 + 強力買進 + 連續正值
    型態 B（主力點火型）：PVO 火力區 + VRI 熱度區 + 強力買進
    型態 C（量能過濾型）：VRI 熱度區 + 強力買進（非B型補充）

    回傳：(patterns, label, combo_key, signal_quality)
    """
    is_strong_buy = sc > (mu_score + 0.5 * sigma_score)
    is_fire       = pvo >= PVO_FIRE_THR
    is_money_in   = PVO_FLOW_THR <= pvo < PVO_FIRE_THR
    is_hot        = vri > VRI_HOT_THR
    is_cool       = vri < VRI_COOL_THR

    # V4 重新校準條件
    a_valid    = (pvo_consec >= PVO_CONSEC_MIN_A) and (vri_delta < VRI_DELTA_COOL)
    b_enhanced = (vri_delta > VRI_DELTA_STRONG) and (pvo_accel >= PVO_ACCEL_MULT)
    c_valid    = vri_delta > VRI_DELTA_CUTOFF

    patterns = []
    if is_strong_buy and is_money_in and is_cool and a_valid:
        patterns.append('A')
    if is_strong_buy and is_fire and is_hot:
        patterns.append('B')
    if is_strong_buy and is_hot and 'B' not in patterns and c_valid:
        patterns.append('C')

    # 量比標記
    if   vol_ratio >= VOL_RATIO_STRONG: vol_tag = '+量爆'
    elif vol_ratio >= VOL_RATIO_NORMAL: vol_tag = '+放量'
    elif vol_ratio <  VOL_RATIO_WEAK:   vol_tag = '-縮量'
    else:                               vol_tag = ''

    pat_set = frozenset(patterns)
    if   len(pat_set) >= 3: base_label = '三合一(ABC)'; combo_key = '三合一(ABC)'
    elif len(pat_set) == 2:
        key = ''.join(sorted(pat_set))
        base_label = f'二合一({key})'; combo_key = base_label
    elif len(pat_set) == 1:
        p = list(pat_set)[0]
        base_label = f'單一({p})'; combo_key = base_label
    else:
        if slope_z > SLOPE_Z_STRONG:
            base_label = '基準-強勢'; combo_key = '基準-強勢'
        else:
            base_label = '基準-持有'; combo_key = '基準-持有'

    b_flag = 'B' in patterns and b_enhanced
    label  = base_label + vol_tag if vol_tag else base_label

    signal_quality = 1.0
    if   vol_ratio >= VOL_RATIO_STRONG: signal_quality = 1.3
    elif vol_ratio >= VOL_RATIO_NORMAL: signal_quality = 1.1
    elif vol_ratio <  VOL_RATIO_WEAK:   signal_quality = 0.8
    if b_flag:
        signal_quality = min(signal_quality * 1.1, 1.5)

    return patterns, label, combo_key, signal_quality


# ══════════════════════════════════════════════════════════════════
# 【V4-升級4】Regime 分類器
# ══════════════════════════════════════════════════════════════════

def _classify_regime_from_label(regime_label: str) -> str:
    """
    將 regime 字串標籤對應回 V4 的 regime 類型。
    daily_run.py 傳入的 regime dict 包含 label 欄位。
    """
    label = str(regime_label).lower()
    if '熊' in label or 'bear' in label:
        return 'crash'
    if '牛' in label or 'bull' in label:
        return 'trend'
    if '回升' in label or 'recovery' in label:
        return 'recovery'
    return 'range'


def get_position_weight(combo_key, signal_quality, regime='range'):
    """
    V4 倉位計算（含 Regime 調整）
    trend=1.1x, range=1.0x, recovery=0.8x, crash=0.5x
    """
    base = POS_WEIGHT.get(combo_key, POS_DEFAULT_WEIGHT)
    regime_mult = {
        'trend': 1.1, 'range': 1.0, 'recovery': 0.8, 'crash': 0.5
    }.get(regime, 1.0)
    weight = base * signal_quality * regime_mult
    return round(min(max(weight, 0.10), 0.30), 4)


# ══════════════════════════════════════════════════════════════════
# 資料取得
# ══════════════════════════════════════════════════════════════════

def _fetch_ohlcv(sym: str, period: str = '60d'):
    """嘗試 .TW → .TWO 取得個股 OHLCV，回傳 DataFrame 或 None"""
    if yf is None or pd is None:
        return None
    for suffix in ['.TW', '.TWO']:
        ticker = f"{sym}{suffix}"
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= MIN_DATA_ROWS:
                return df
        except Exception as e:
            log.debug(f"  {ticker} 下載失敗: {e}")
    log.warning(f"⚠️  {sym} 在 .TW / .TWO 均無資料，跳過")
    return None


def _load_from_csv(sym: str, day_dir: str):
    """優先從已存的每日 CSV 讀取資料"""
    if pd is None:
        return None
    csv_path = os.path.join(day_dir, f"{sym}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) >= MIN_DATA_ROWS:
            return df
    except Exception as e:
        log.debug(f"  {sym} CSV 讀取失敗: {e}")
    return None


# ══════════════════════════════════════════════════════════════════
# 核心計算：單股指標
# ══════════════════════════════════════════════════════════════════

def _compute_stock_indicators(df):
    """
    計算 V4 所需的全部技術指標。
    回傳含指標欄位的 DataFrame，或 None（資料不足）。
    """
    if np is None or pd is None:
        log.error("numpy / pandas 未安裝，無法計算指標")
        return None
    if df is None or len(df) < MIN_DATA_ROWS:
        return None

    df = df.copy()
    # 確保欄位名稱正確
    df.columns = [str(c).strip() for c in df.columns]

    try:
        # [FIX] 使用內建純 numpy 函式，不需 pandas_ta
        df['RSI']        = _ta_rsi(df['Close'], 14)
        df['ATR']        = _ta_atr(df['High'], df['Low'], df['Close'], 14)
        df['MA50']       = _ta_sma(df['Close'], 50)
        df['PVO']        = _calc_pvo(df)
        df['VRI']        = _calc_vri(df)
        df['Slope']      = df['Close'].pct_change(5) * 100
        df['PVO_consec'] = _calc_pvo_consec(df)
        df['PVO_accel']  = _calc_pvo_accel(df)
        df['VRI_delta']  = _calc_vri_delta(df)
        df['VolRatio']   = _calc_vol_ratio(df)
    except Exception as e:
        log.warning(f"  指標計算失敗: {e}")
        return None

    return df.dropna(subset=['RSI', 'ATR', 'PVO', 'VRI'])


# ══════════════════════════════════════════════════════════════════
# V4 評分計算
# ══════════════════════════════════════════════════════════════════

def _score_stock(df, mu_score: float, sigma_score: float, regime_type: str) -> dict:
    """
    計算單一個股的 V4 評分與訊號分類。

    評分公式（繼承自 V4）：
        score = 50 + slope_z × 8 + PVO貢獻（上限15）+ VRI加分
    """
    if df is None or len(df) < MIN_DATA_ROWS:
        return None

    last = df.iloc[-1]

    pvo       = float(last.get('PVO', 0))
    vri       = float(last.get('VRI', 50))
    slope     = float(last.get('Slope', 0))
    pvo_c     = float(last.get('PVO_consec', 0))
    pvo_a     = float(last.get('PVO_accel', 1))
    vri_d     = float(last.get('VRI_delta', 0))
    vol_ratio = float(last.get('VolRatio', 1))
    close     = float(last.get('Close', 0))
    atr       = float(last.get('ATR', close * 0.02)) if not math.isnan(float(last.get('ATR', float('nan')))) else close * 0.02

    # Slope Z-Score（過去 30 根的標準化）
    slope_win = df['Slope'].tail(30)
    slope_z   = (slope - float(slope_win.mean())) / (float(slope_win.std()) + 1e-9)

    # 評分
    score = 50.0
    score += min(slope_z * 8, 20)
    score += min(pvo * 0.5, 15) if pvo > 0 else max(pvo * 0.3, -10)
    score += 8 if 40 <= vri <= 75 else (-5 if vri > 90 else 0)

    # 訊號分類
    _, signal_label, combo_key, sig_quality = classify_signal_v4(
        pvo, vri, slope_z, score, mu_score, sigma_score,
        pvo_c, pvo_a, vri_d, vol_ratio
    )

    # 操作判定
    if   slope_z >= 1.5 and pvo > 5:  action = '強力買進'
    elif slope_z >= 0.5 and pvo > 0:  action = '買進'
    elif slope_z < -1.0:              action = '賣出'
    else:                             action = '觀察'

    pos_weight = get_position_weight(combo_key, sig_quality, regime_type)

    return {
        'score':      round(score, 2),
        'pvo':        round(pvo, 2),
        'vri':        round(vri, 1),
        'slope_z':    round(slope_z, 2),
        'slope':      round(slope, 3),
        'action':     action,
        'signal':     signal_label,
        'combo_key':  combo_key,
        'close':      round(close, 1),
        'atr':        round(atr, 2),
        'pos_weight': pos_weight,
    }


# ══════════════════════════════════════════════════════════════════
# 主引擎介面
# ══════════════════════════════════════════════════════════════════

def run(symbols: list, regime: dict, today: str,
        daily_close: dict = None) -> dict:
    """
    V4 市場強度引擎主函式。

    Args:
        symbols:      個股代號清單（不含 .TW/.TWO 後綴）
        regime:       regime_engine 輸出的 dict（含 label、active_strategy 等）
        today:        計算日期字串，格式 'YYYY-MM-DD'
        daily_close:  step_save_daily_data() 回傳的 {sym: {close,...}}，
                      有此資料時優先讀 CSV，減少重複下載

    Returns:
        V4 快照 dict，包含：
            top20       — 排名前20的個股資料列表
            pool_mu     — 評分池均值
            pool_sigma  — 評分池標準差
            win_rate    — 歷史勝率（固定參考值）
            market      — 市場代碼
    """
    if np is None or pd is None:
        log.error("❌ numpy / pandas 未安裝，V4 引擎無法運行")
        return {}
    # 解析 Regime
    regime_label_str = regime.get('label', '震盪')
    regime_type      = _classify_regime_from_label(regime_label_str)

    # 確定每日 CSV 目錄
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    day_dir   = os.path.join(data_root, today)

    log.info(f"V4 引擎啟動 | 日期: {today} | Regime: {regime_label_str}({regime_type}) | 股票池: {len(symbols)} 檔")

    rows = []
    skipped = 0

    for sym in symbols:
        try:
            # 優先讀取已存的 CSV（避免重複下載）
            df = _load_from_csv(sym, day_dir)
            if df is None:
                df = _fetch_ohlcv(sym, period='60d')
            if df is None:
                skipped += 1
                continue

            # 計算指標
            df_ind = _compute_stock_indicators(df)
            if df_ind is None or len(df_ind) < MIN_DATA_ROWS:
                skipped += 1
                continue

            # 暫存基礎評分（用於計算 mu/sigma）
            rows.append({'sym': sym, 'df': df_ind})

        except Exception as e:
            log.warning(f"  V4 {sym} 跳過: {e}")
            skipped += 1
            continue

    if not rows:
        log.error("❌ V4：無有效個股資料")
        return {}

    # 第一輪：粗算評分，取得 mu / sigma
    # 使用固定初始值（與原始 V4 的 historical_scores 一致）
    mu_score    = 62.0
    sigma_score = 11.5

    # 第二輪：正式計算各股評分與訊號
    result_rows = []
    for item in rows:
        sym   = item['sym']
        df_ind = item['df']
        try:
            stock_result = _score_stock(df_ind, mu_score, sigma_score, regime_type)
            if stock_result is None:
                continue
            stock_result['symbol'] = sym
            stock_result['regime'] = regime_label_str
            result_rows.append(stock_result)
        except Exception as e:
            log.warning(f"  V4 評分 {sym} 失敗: {e}")
            continue

    if not result_rows:
        log.error("❌ V4：評分結果為空")
        return {}

    # 排序並取 TOP 20
    result_rows.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(result_rows):
        r['rank'] = i + 1
    top20 = result_rows[:20]

    # 統計 mu / sigma（使用實際評分）
    scores = [r['score'] for r in result_rows]
    actual_mu    = round(float(np.mean(scores)), 2) if scores else 0.0
    actual_sigma = round(float(np.std(scores)), 2)  if scores else 0.0

    log.info(f"V4 完成 ✅ | TOP20 共 {len(top20)} 檔 | 跳過 {skipped} 檔 | μ={actual_mu} σ={actual_sigma}")

    # ── 按操作統計摘要 ──
    action_counts = {}
    for r in top20:
        a = r.get('action', '─')
        action_counts[a] = action_counts.get(a, 0) + 1
    log.info(f"V4 操作分佈: {action_counts}")

    return {
        'market':     'TW',
        'top20':      top20,
        'pool_mu':    actual_mu,
        'pool_sigma': actual_sigma,
        'win_rate':   57.1,          # V4 歷史 OOS 勝率（固定參考值）
        'regime':     regime_label_str,
        'total_scored': len(result_rows),
        'skipped':    skipped,
    }
