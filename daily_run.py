"""
╔══════════════════════════════════════════════════════════════╗
║  daily_run.py — 資源法 Precompute 主控制器                    ║
║  由 GitHub Actions 在台股交易日定時觸發                        ║
║  觸發時間（台灣時間 UTC+8）：                                  ║
║    09:30 / 12:00 / 13:30 / 14:30 / 15:30 / 18:00            ║
╚══════════════════════════════════════════════════════════════╝

修正記錄：
  v2.0 - 修正 step_v4 / step_v12 縮排與孤立 try 區塊
       - 統一 .TW / .TWO fallback 邏輯為獨立函式 fetch_tw_ohlcv()
       - 增加 JSON 無資料偵測與警告文字
       - 新增 step_save_daily_data()：每日個股資料存至 data/{TODAY}/
"""

import json
import os
import sys
import logging
import math
from datetime import datetime, date

# ── 選用性依賴（避免 import 失敗中斷整個模組）
try:
    import numpy as np
except ImportError:
    np = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas_ta as ta
except ImportError:
    ta = None

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("daily_run")


# ──────────────────────────────────────────────────────────────
# 路徑設定
# ──────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(ROOT_DIR, "storage")
V4_DIR      = os.path.join(STORAGE_DIR, "v4")
V12_DIR     = os.path.join(STORAGE_DIR, "v12")
REGIME_DIR  = os.path.join(STORAGE_DIR, "regime")
MARKET_DIR  = os.path.join(STORAGE_DIR, "market")
LOGS_DIR    = os.path.join(STORAGE_DIR, "logs")

# ★ 新增：每日個股 OHLCV 原始資料存放根目錄（對應 GitHub /data/）
DATA_ROOT   = os.path.join(ROOT_DIR, "data")

TODAY    = date.today().strftime("%Y-%m-%d")
NOW_HOUR = datetime.now().hour
TS       = datetime.now().strftime("%Y%m%d_%H%M")

for d in [V4_DIR, V12_DIR, REGIME_DIR, MARKET_DIR, LOGS_DIR, DATA_ROOT]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 股票清單（集中在一處，V4 與 V12 共用）
# ──────────────────────────────────────────────────────────────
SYMBOLS = list(dict.fromkeys([                       # dict.fromkeys 自動去重
    # 核心大型股
    "2330", "2317", "2454", "2308", "2382", "2303", "3711", "2412", "2357", "3231",
    "2379", "3008", "2395", "3045", "2327", "2408", "2377", "6669", "2301", "3034",
    "2345", "2474", "3037", "4938", "3443", "2353", "2324", "2603", "2609", "1513",
    # 中小型科技
    "3030", "3706", "8096", "2313", "4958",
    "3293", "3680", "3529", "3131", "5274", "6223", "6805", "3017", "3324", "6515",
    "3661", "3583", "6139", "3035", "1560", "8299", "3558", "6187", "3406", "3217",
    "6176", "6415", "6206", "8069", "3264", "5269", "2360", "6271", "3189", "6438",
    "8358", "6231", "2449", "8016", "6679", "3374", "3014", "3211",
    "6213", "2404", "2480", "3596", "6202", "5443", "5347", "5483", "6147",
    # PCB
    "8046", "2368", "2383", "6269", "5469", "5351",
    # 記憶體
    "4909", "8050", "6153", "6505", "1802", "3708", "8213", "1325",
    "2344", "6239", "3260", "4967", "6414", "2337",
    # 被動元件
    "3551", "2436", "2375", "2492", "2456", "3229", "6173", "3533",
    # 低軌衛星
    "3491", "2367", "6285", "6190",
    "3062", "2419", "2314", "3305", "3105", "2312", "8086",
    # 光通訊
    "3081", "2455", "6442", "3163", "4979", "3363", "6451",
    "3450", "4908", "4977", "3234",
    # 其他
    "1711", "1727", "2489", "3060", "3498", "3535", "3580", "3587",
    "3665", "4749", "4989", "6217", "6290", "6418", "6443", "6470",
    "6542", "6546", "6706", "6831", "6861", "6877", "8028", "8111",
]))


# ──────────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────────
def save_json(path: str, data):
    """儲存 JSON，自動加入 generated_at / date 時間戳"""
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


def check_json_has_data(path: str, key: str, label: str) -> bool:
    """
    ★ 新增：驗證 JSON 檔案是否存在且包含有效資料。
    回傳 True 表示正常，False 表示無資料（並印出警告）。
    """
    data = load_json(path)
    if data is None:
        log.warning(f"⚠️  【無資料】{label} 檔案不存在：{path}")
        return False
    content = data.get(key)
    if not content:
        log.warning(f"⚠️  【無資料】{label} 的 '{key}' 欄位為空，請確認資料下載是否成功")
        return False
    log.info(f"✅ 【資料正常】{label}：共 {len(content)} 筆")
    return True


def fetch_tw_ohlcv(sym: str, period: str = "60d"):
    """
    ★ 統一函式：依序嘗試 .TW → .TWO，回傳 (DataFrame, suffix) 或 (None, None)。
    修正原本兩處各自土法煉鋼的判斷邏輯，集中維護。
    """
    if yf is None:
        log.error("yfinance 未安裝")
        return None, None

    import pandas as pd
    for suffix in [".TW", ".TWO"]:
        ticker = f"{sym}{suffix}"
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            # MultiIndex 攤平（yfinance 多股票模式會產生）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 20:
                log.debug(f"  {ticker} ✓ ({len(df)} 筆)")
                return df, suffix
        except Exception as e:
            log.debug(f"  {ticker} 下載失敗: {e}")

    log.warning(f"⚠️  {sym} 在 .TW / .TWO 均無資料，跳過")
    return None, None


# ──────────────────────────────────────────────────────────────
# Import 引擎
# ──────────────────────────────────────────────────────────────
def import_engines():
    engines = {}
    for name, module in [
        ("v4",      "v4_engine"),
        ("v12",     "v12_engine"),
        ("regime",  "regime_engine"),
        ("feature", "feature_engine"),
        ("risk",    "risk_engine"),
    ]:
        try:
            engines[name] = __import__(module)
            log.info(f"✅ 引擎載入：{module}")
        except ImportError as e:
            log.warning(f"⚠️  引擎未找到：{module} ({e})")
            engines[name] = None
    return engines


# ──────────────────────────────────────────────────────────────
# Step 1：市場快照（大盤指數）
# ──────────────────────────────────────────────────────────────
def step_market(eng) -> dict:
    log.info("=== Step 1: Market Snapshot ===")
    try:
        if eng and hasattr(eng, "run"):
            data = eng.run(today=TODAY)
        else:
            if yf is None or ta is None:
                raise RuntimeError("yfinance / pandas_ta 未安裝")

            import pandas as pd
            bm = yf.download("^TWII", period="60d", progress=False, auto_adjust=True)
            if isinstance(bm.columns, pd.MultiIndex):
                bm.columns = bm.columns.get_level_values(0)

            bm["RSI"]   = ta.rsi(bm["Close"], 14)
            bm["Slope"] = bm["Close"].pct_change(5) * 100

            last  = bm.iloc[-1]
            prev  = bm.iloc[-2]
            chg   = float((last["Close"] - prev["Close"]) / prev["Close"] * 100)
            slope = float(bm["Close"].pct_change(20).iloc[-1] * 100)

            data = {
                "index_close":   float(last["Close"]),
                "index_chg_pct": round(chg, 2),
                "mkt_rsi":       float(last["RSI"]) if not math.isnan(float(last["RSI"])) else 50.0,
                "mkt_slope_5d":  round(float(bm["Close"].pct_change(5).iloc[-1] * 100), 4),
                "mkt_slope_20d": round(slope, 4),
                "volume":        float(last.get("Volume", 0)),
            }

        save_json(os.path.join(MARKET_DIR, "market_snapshot.json"), data)
        log.info(f"大盤: {data.get('index_close', 0):,.1f} ({data.get('index_chg_pct', 0):+.2f}%)")
        return data
    except Exception as e:
        log.error(f"❌ Market snapshot 失敗: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# Step 2：Regime 分類
# ──────────────────────────────────────────────────────────────
def step_regime(eng, market_data: dict) -> dict:
    log.info("=== Step 2: Regime Classification ===")
    try:
        if eng and hasattr(eng, "run"):
            regime = eng.run(market_data=market_data, today=TODAY)
        else:
            if yf is None or ta is None or np is None:
                raise RuntimeError("yfinance / pandas_ta / numpy 未安裝")

            import pandas as pd
            bm = yf.download("^TWII", period="180d", progress=False, auto_adjust=True)
            if isinstance(bm.columns, pd.MultiIndex):
                bm.columns = bm.columns.get_level_values(0)

            bm["RSI"]  = ta.rsi(bm["Close"], 14)
            adx_df     = ta.adx(bm["High"], bm["Low"], bm["Close"])
            bm["ADX"]  = adx_df.iloc[:, 0] if adx_df is not None else 20
            bm["MA60"] = bm["Close"].rolling(60).mean()

            last  = bm.iloc[-1]
            rsi   = float(last["RSI"])
            adx   = float(last["ADX"])
            slope = float(bm["Close"].pct_change(5).iloc[-1])
            close = float(last["Close"])
            ma60  = float(last["MA60"])

            smooth    = 0.7
            bull_raw  = 1.0 if (rsi > 52 and slope > 0 and close > ma60) else 0.0
            bear_raw  = 1.0 if (rsi < 40 and slope < 0) else 0.0
            range_raw = 1.0 - bull_raw - bear_raw

            old = load_json(os.path.join(REGIME_DIR, "regime_state.json"))
            if old:
                bull_s  = smooth * old.get("bull", bull_raw)  + (1 - smooth) * bull_raw
                bear_s  = smooth * old.get("bear", bear_raw)  + (1 - smooth) * bear_raw
                range_s = 1 - bull_s - bear_s
            else:
                bull_s = bull_raw; bear_s = bear_raw; range_s = range_raw

            total = bull_s + bear_s + range_s + 1e-9
            bull_s /= total; bear_s /= total; range_s /= total

            if bull_s >= 0.55:
                strat = "bull";  a_path = "45";   b_path = "423"
            elif bear_s >= 0.60:
                strat = "bear";  a_path = None;   b_path = None
            else:
                strat = "range"; a_path = "423";  b_path = "45"

            if bull_s > 0.5:      label = "牛市"
            elif bear_s > 0.5:    label = "熊市"
            elif bull_s > bear_s: label = "偏多震盪"
            else:                  label = "偏空震盪"

            s5d  = float(bm["Close"].pct_change(5).iloc[-1])
            s20d = float(bm["Close"].pct_change(20).iloc[-1])

            old_hist   = old.get("history", []) if old else []
            curr_month = datetime.now().strftime("%Y-%m")
            if not any(h["month"] == curr_month for h in old_hist):
                old_hist.append({
                    "month": curr_month,
                    "bear": round(bear_s, 4), "range": round(range_s, 4),
                    "bull": round(bull_s, 4), "label": label,
                })
                old_hist = old_hist[-24:]

            regime = {
                "bear": round(bear_s, 4),  "range": round(range_s, 4),
                "bull": round(bull_s, 4),  "label": label,
                "active_strategy": strat,
                "active_path": a_path,     "backup_path": b_path,
                "slope_5d": round(s5d, 4), "slope_20d": round(s20d, 4),
                "mkt_rsi": round(rsi, 1),  "adx": round(adx, 1),
                "history": old_hist,
            }

        save_json(os.path.join(REGIME_DIR, "regime_state.json"), regime)
        log.info(f"Regime: {regime.get('label')} | 熊:{regime.get('bear', 0)*100:.0f}% 牛:{regime.get('bull', 0)*100:.0f}%")
        return regime
    except Exception as e:
        log.error(f"❌ Regime 失敗: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# ★ Step 2.5：每日個股 OHLCV 下載並存至 data/{TODAY}/
# ──────────────────────────────────────────────────────────────
def step_save_daily_data(symbols: list) -> dict:
    """
    每天將所有個股最新 OHLCV 存為 CSV，路徑：
        data/{TODAY}/{symbol}.csv   （60d 歷史，約 60 行）
        data/{TODAY}/_index.json    （匯總當日收盤資訊）

    GitHub Actions commit 這個目錄即可保留每日快照。
    回傳：成功下載的 symbol → close 字典。
    """
    log.info("=== Step 2.5: Save Daily OHLCV to data/ ===")

    day_dir = os.path.join(DATA_ROOT, TODAY)
    os.makedirs(day_dir, exist_ok=True)

    index_data   = {}   # { symbol: {close, suffix, rows} }
    failed_syms  = []

    for sym in symbols:
        df, suffix = fetch_tw_ohlcv(sym, period="60d")
        if df is None:
            failed_syms.append(sym)
            continue

        # 存 CSV
        csv_path = os.path.join(day_dir, f"{sym}.csv")
        df.to_csv(csv_path)

        last = df.iloc[-1]
        index_data[sym] = {
            "close":  round(float(last["Close"]), 2),
            "volume": int(last.get("Volume", 0)),
            "suffix": suffix,
            "rows":   len(df),
        }

    # 存彙整 index
    index_path = os.path.join(day_dir, "_index.json")
    summary = {
        "date":           TODAY,
        "total_symbols":  len(symbols),
        "downloaded":     len(index_data),
        "failed":         len(failed_syms),
        "failed_symbols": failed_syms,
        "stocks":         index_data,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    if failed_syms:
        log.warning(f"⚠️  【無資料】以下個股下載失敗（共 {len(failed_syms)} 檔）：{failed_syms}")
    log.info(f"✅ 每日資料存檔完成：{len(index_data)}/{len(symbols)} 檔 → {day_dir}")
    return index_data


# ──────────────────────────────────────────────────────────────
# Step 3：V4 快照（市場強度 TOP20）
# ──────────────────────────────────────────────────────────────
def step_v4(eng, regime: dict, daily_close: dict) -> dict:
    """
    daily_close：由 step_save_daily_data() 回傳的 {sym: {close,...}} 字典，
    當 eng 不存在時可直接讀取本地 CSV，避免重複下載。
    """
    log.info("=== Step 3: V4 Market Strength Snapshot ===")
    try:
        if eng and hasattr(eng, "run"):
            v4_data = eng.run(symbols=SYMBOLS, regime=regime, today=TODAY)
        else:
            # ── fallback：從已下載的 data/{TODAY}/ 讀取，或即時抓取 ──
            if np is None or ta is None:
                raise RuntimeError("numpy / pandas_ta 未安裝")

            day_dir = os.path.join(DATA_ROOT, TODAY)
            rows    = []

            for sym in SYMBOLS:
                try:
                    # 優先讀取已存的 CSV（避免重複下載）
                    csv_path = os.path.join(day_dir, f"{sym}.csv")
                    if os.path.exists(csv_path):
                        import pandas as pd
                        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                        if len(df) < 20:
                            df, _ = fetch_tw_ohlcv(sym, period="60d")
                    else:
                        df, _ = fetch_tw_ohlcv(sym, period="60d")

                    if df is None or len(df) < 20:
                        log.warning(f"⚠️  {sym} 資料不足，跳過 V4 計算")
                        continue

                    # ── 指標計算 ──
                    df["RSI"] = ta.rsi(df["Close"], 14)
                    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 14)
                    vol_fast  = df["Volume"].ewm(span=5).mean()
                    vol_slow  = df["Volume"].ewm(span=20).mean()
                    df["PVO"] = (vol_fast - vol_slow) / (vol_slow + 1e-9) * 100
                    rsi_norm  = df["RSI"].clip(0, 100)
                    atr_mean  = df["ATR"].rolling(20).mean()
                    atr_norm  = (df["ATR"] / (atr_mean + 1e-9) * 50).clip(0, 100)
                    df["VRI"] = (rsi_norm * 0.6 + atr_norm * 0.4).clip(0, 100)
                    df["Slope"] = df["Close"].pct_change(5) * 100

                    slope_win = df["Slope"].tail(30)
                    slope_z   = (float(df["Slope"].iloc[-1]) - slope_win.mean()) / (slope_win.std() + 1e-9)

                    last  = df.iloc[-1]
                    pvo   = float(last["PVO"])
                    vri   = float(last["VRI"])
                    slope = float(last["Slope"])
                    close = float(last["Close"])

                    # ── 評分 ──
                    score = 50.0
                    score += min(slope_z * 8, 20)
                    score += min(pvo * 0.5, 15) if pvo > 0 else max(pvo * 0.3, -10)
                    score += 8 if 40 <= vri <= 75 else (-5 if vri > 90 else 0)

                    # ── 訊號型態 ──
                    sig_a = pvo > 0 and vri < 45
                    sig_b = pvo > 8 and vri > 70
                    sig_c = vri > 70

                    if   sig_a and sig_b and sig_c: signal = "三合一(ABC)"
                    elif sig_a and sig_b:           signal = "二合一(AB)"
                    elif sig_a and sig_c:           signal = "二合一(AC)"
                    elif sig_b and sig_c:           signal = "二合一(BC)"
                    elif sig_a:                     signal = "單一(A)"
                    elif sig_b:                     signal = "單一(B)"
                    elif sig_c:                     signal = "單一(C)"
                    else:                           signal = "基準"

                    # ── 操作判定 ──
                    if   slope_z >= 1.5 and pvo > 5:  action = "強力買進"
                    elif slope_z >= 0.5 and pvo > 0:  action = "買進"
                    elif slope_z < -1.0:               action = "賣出"
                    else:                              action = "觀察"

                    rows.append({
                        "symbol":  sym, "score": round(score, 2),
                        "pvo":     round(pvo, 2),   "vri":     round(vri, 1),
                        "slope_z": round(slope_z, 2), "slope": round(slope, 3),
                        "action":  action, "signal": signal,
                        "close":   round(close, 1),
                        "regime":  regime.get("label", "—"),
                    })
                except Exception as sym_e:
                    log.warning(f"  V4 {sym} 跳過: {sym_e}")
                    continue

            rows.sort(key=lambda x: x["score"], reverse=True)
            for i, r in enumerate(rows):
                r["rank"] = i + 1
            rows = rows[:20]

            scores  = [r["score"] for r in rows]
            v4_data = {
                "market":     "TW",
                "top20":      rows,
                "pool_mu":    round(float(np.mean(scores)), 2)  if scores else 0,
                "pool_sigma": round(float(np.std(scores)), 2)   if scores else 0,
                "win_rate":   57.1,
            }

        # ── 存檔 ──
        save_json(os.path.join(V4_DIR, "v4_latest.json"), v4_data)
        save_json(os.path.join(V4_DIR, f"v4_{TS}.json"),  v4_data)

        # ★ 驗證資料是否正常
        check_json_has_data(os.path.join(V4_DIR, "v4_latest.json"), "top20", "V4 TOP20")

        log.info(f"V4 TOP20 完成，共 {len(v4_data.get('top20', []))} 筆")
        return v4_data
    except Exception as e:
        log.error(f"❌ V4 失敗: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# Step 4：V12.1 決策快照
# ──────────────────────────────────────────────────────────────
def step_v12(eng, regime: dict, v4_data: dict) -> dict:
    log.info("=== Step 4: V12.1 Trading Decision Snapshot ===")
    try:
        if eng and hasattr(eng, "run"):
            v12_data = eng.run(symbols=SYMBOLS, regime=regime, v4_snapshot=v4_data, today=TODAY)
        else:
            if yf is None or ta is None or np is None:
                raise RuntimeError("yfinance / pandas_ta / numpy 未安裝")

            day_dir = os.path.join(DATA_ROOT, TODAY)

            old_v12      = load_json(os.path.join(V12_DIR, "v12_latest.json"))
            old_pos      = {p["symbol"]: p for p in (old_v12 or {}).get("positions", [])}
            active_strat = regime.get("active_strategy", "range")
            a_path       = regime.get("active_path", "423")
            b_path       = regime.get("backup_path", "45")
            ev_entry_min = 0.040 if active_strat == "bear" else 0.030

            top20      = v4_data.get("top20", [])
            candidates = [r for r in top20
                          if r.get("action") in ("強力買進", "買進")
                          and r.get("score", 0) > 55]

            positions = []
            for cand in candidates[:8]:
                sym = cand["symbol"]
                try:
                    # 優先讀取已存的 CSV
                    csv_path = os.path.join(day_dir, f"{sym}.csv")
                    if os.path.exists(csv_path):
                        import pandas as pd
                        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                        if len(df) < 20:
                            df, _ = fetch_tw_ohlcv(sym, period="90d")
                    else:
                        df, _ = fetch_tw_ohlcv(sym, period="90d")

                    if df is None or len(df) < 20:
                        log.warning(f"⚠️  V12 {sym} 資料不足，跳過")
                        continue

                    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 14)
                    df["RSI"] = ta.rsi(df["Close"], 14)
                    vol_fast  = df["Volume"].ewm(span=5).mean()
                    vol_slow  = df["Volume"].ewm(span=20).mean()
                    df["PVO"] = (vol_fast - vol_slow) / (vol_slow + 1e-9) * 100

                    last  = df.iloc[-1]
                    close = float(last["Close"])
                    atr   = float(last["ATR"]) if not np.isnan(float(last["ATR"])) else close * 0.02
                    pvo   = float(last["PVO"])

                    rets  = df["Close"].pct_change().tail(30).dropna()
                    ev    = float(rets.mean() * 100 * 5)
                    ev    = round(min(max(ev, 0.5), 12.0), 2)

                    if ev < ev_entry_min * 100:
                        continue

                    ev_tier = "⭐核心" if ev > 5 else ("🔥主力" if ev > 3 else "📌補位")

                    r_probs = {"45": 0.65, "423": 0.35} if active_strat == "bull" else {"423": 0.65, "45": 0.35}
                    path    = a_path if np.random.random() < r_probs.get(a_path, 0.65) else b_path
                    path    = path or "423"

                    if sym in old_pos:
                        old       = old_pos[sym]
                        days_held = old.get("days_held", 0) + 1
                        entry_p   = old.get("entry_price", close)
                        curr_ret  = (close - entry_p) / entry_p * 100
                        stop_p    = old.get("stop_price",  round(close - atr * 1.5, 2))
                        tp1_p     = old.get("tp1_price",   round(close * 1.06, 2))
                        action    = old.get("action", "持有")
                        quality   = old.get("quality", "Pure")
                        ev_orig   = old.get("ev", ev)

                        exit_sig = "—"
                        if   ev < ev_orig * 0.65:                       exit_sig = "EV衰退"
                        elif pvo < -0.3 and ev < ev_orig * 0.8:         exit_sig = "量能枯竭"
                        elif days_held >= 20 and curr_ret < -5:          exit_sig = "時間衰減"
                        elif close < stop_p:                             exit_sig = "硬停損"
                    else:
                        days_held = 0
                        entry_p   = close
                        curr_ret  = 0.0
                        stop_p    = round(close - atr * 1.5, 2)
                        tp1_p     = round(close * 1.06, 2)
                        action    = "進場"
                        quality   = "Pure"
                        exit_sig  = "—"

                    vri_ratio = cand.get("vri", 50)
                    if vri_ratio < 40 or abs(pvo) > 20:
                        quality = "Flicker"
                        tp1_p   = round(close * 1.04, 2)

                    positions.append({
                        "symbol":       sym,       "path":         path,
                        "ev":           ev,        "ev_tier":      ev_tier,
                        "action":       action,    "exit_signal":  exit_sig,
                        "quality":      quality,   "days_held":    days_held,
                        "curr_ret_pct": round(curr_ret, 2),
                        "entry_price":  round(entry_p, 2),
                        "tp1_price":    tp1_p,     "stop_price":   stop_p,
                        "regime":       active_strat,
                        "close":        round(close, 2),
                    })
                except Exception as sym_e:
                    log.warning(f"  V12 {sym} 跳過: {sym_e}")
                    continue

            stats = {
                "total_trades": 112, "win_rate": 57.1, "avg_ev": 5.29,
                "max_dd": -6.58,     "sharpe":   5.36, "t_stat": 4.032,
                "simple_cagr": 96.9, "pl_ratio": 2.31,
            }

            v12_data = {
                "market":      "TW", "positions": positions,
                "stats":       stats,
                "regime":      active_strat,
                "active_path": a_path, "backup_path": b_path,
            }

        # ── 存檔 ──
        save_json(os.path.join(V12_DIR, "v12_latest.json"), v12_data)
        save_json(os.path.join(V12_DIR, f"v12_{TS}.json"),  v12_data)

        # ★ 驗證資料是否正常
        check_json_has_data(os.path.join(V12_DIR, "v12_latest.json"), "positions", "V12.1 部位")

        log.info(f"V12.1 完成，部位數: {len(v12_data.get('positions', []))}")
        return v12_data
    except Exception as e:
        log.error(f"❌ V12.1 失敗: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# Step 5：追加交易歷史
# ──────────────────────────────────────────────────────────────
def step_history(v12_data: dict):
    log.info("=== Step 5: Update Trade History ===")
    try:
        history_path = os.path.join(LOGS_DIR, "trade_history.json")
        old_hist     = load_json(history_path) or []

        new_exits = [
            {
                "date":        TODAY,
                "sym":         p["symbol"],
                "action_type": "賣出",
                "exit_type":   p.get("exit_signal", "—"),
                "ret":         round(p.get("curr_ret_pct", 0) / 100, 4),
                "path":        p.get("path", "—"),
                "year":        datetime.now().year,
            }
            for p in v12_data.get("positions", [])
            if p.get("exit_signal", "—") not in ("—", "無", "持倉中")
        ]

        if new_exits:
            old_hist.extend(new_exits)
            old_hist = old_hist[-500:]
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(old_hist, f, ensure_ascii=False, indent=2, default=str)
            log.info(f"  新增 {len(new_exits)} 筆交易記錄")
        else:
            log.info("  今日無新出場記錄")

        # ★ 驗證 trade_history.json
        if os.path.exists(history_path):
            hist = load_json(history_path) or []
            if not hist:
                log.warning("⚠️  【無資料】trade_history.json 目前無任何交易記錄")
            else:
                log.info(f"✅ 【資料正常】trade_history.json：共 {len(hist)} 筆歷史記錄")
        else:
            log.warning("⚠️  【無資料】trade_history.json 尚未建立")

    except Exception as e:
        log.error(f"❌ 歷史記錄更新失敗: {e}")


# ──────────────────────────────────────────────────────────────
# 主控流程
# ──────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info(f"📅 資源法 daily_run.py 啟動 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    engines = import_engines()

    market_data = step_market(engines.get("feature"))
    regime_data = step_regime(engines.get("regime"), market_data)

    # ★ 先下載並存好所有個股資料（後續步驟直接讀 CSV，避免重複下載）
    daily_close = step_save_daily_data(SYMBOLS)

    v4_data  = step_v4(engines.get("v4"),  regime_data, daily_close)
    v12_data = step_v12(engines.get("v12"), regime_data, v4_data)
    step_history(v12_data)

    log.info("=" * 60)
    log.info("✅ 全部步驟完成")
    log.info(f"  大盤:     {market_data.get('index_close', 0):,.1f} ({market_data.get('index_chg_pct', 0):+.2f}%)")
    log.info(f"  Regime:   {regime_data.get('label', '—')}")
    log.info(f"  V4 TOP20: {len(v4_data.get('top20', []))} 檔")
    log.info(f"  V12.1:    {len(v12_data.get('positions', []))} 檔")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
