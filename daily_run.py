"""
╔══════════════════════════════════════════════════════════════╗
║  daily_run.py — 資源法 Precompute 主控制器                    ║
║  由 GitHub Actions 在台股交易日定時觸發                        ║
║  觸發時間（台灣時間 UTC+8）：                                  ║
║    09:30 / 12:00 / 13:30 / 14:30 / 15:30 / 18:00            ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import logging
from datetime import datetime, date

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

for d in [V4_DIR, V12_DIR, REGIME_DIR, MARKET_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

TODAY    = date.today().strftime("%Y-%m-%d")
NOW_HOUR = datetime.now().hour  # 台灣時間（GitHub Actions 需調整 TZ）
TS       = datetime.now().strftime("%Y%m%d_%H%M")


# ──────────────────────────────────────────────────────────────
# JSON 工具
# ──────────────────────────────────────────────────────────────
def save_json(path: str, data):
    """儲存 JSON，含 generated_at 時間戳"""
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


# ──────────────────────────────────────────────────────────────
# Import 引擎
# ──────────────────────────────────────────────────────────────
def import_engines():
    """動態引入各引擎，失敗時記錄錯誤不中斷"""
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
            # fallback：直接用 yfinance
            import yfinance as yf
            import pandas_ta as ta
            bm = yf.download("^TWII", period="60d", progress=False, auto_adjust=True)
            if isinstance(bm.columns, __import__("pandas").MultiIndex):
                bm.columns = bm.columns.get_level_values(0)
            bm["RSI"]   = ta.rsi(bm["Close"], 14)
            bm["Slope"] = bm["Close"].pct_change(5) * 100

            last  = bm.iloc[-1]
            prev  = bm.iloc[-2]
            chg   = float((last["Close"] - prev["Close"]) / prev["Close"] * 100)
            slope = float(bm["Close"].pct_change(20).iloc[-1] * 100)

            data = {
                "index_close":    float(last["Close"]),
                "index_chg_pct":  round(chg, 2),
                "mkt_rsi":        float(last["RSI"]) if not __import__("math").isnan(last["RSI"]) else 50.0,
                "mkt_slope_5d":   round(float(bm["Close"].pct_change(5).iloc[-1] * 100), 4),
                "mkt_slope_20d":  round(slope, 4),
                "volume":         float(last.get("Volume", 0)),
            }

        # 覆蓋 latest + 歷史
        save_json(os.path.join(MARKET_DIR, "market_snapshot.json"), data)
        log.info(f"大盤: {data.get('index_close',0):,.1f} ({data.get('index_chg_pct',0):+.2f}%)")
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
            # fallback：簡易 Regime 計算
            import yfinance as yf
            import pandas_ta as ta
            import numpy as np

            bm = yf.download("^TWII", period="180d", progress=False, auto_adjust=True)
            if isinstance(bm.columns, __import__("pandas").MultiIndex):
                bm.columns = bm.columns.get_level_values(0)
            bm["RSI"]  = ta.rsi(bm["Close"], 14)
            adx_df     = ta.adx(bm["High"], bm["Low"], bm["Close"])
            bm["ADX"]  = adx_df.iloc[:, 0] if adx_df is not None else 20
            bm["MA60"] = bm["Close"].rolling(60).mean()
            bm["Slope"]= bm["Close"].pct_change(5)

            last     = bm.iloc[-1]
            rsi      = float(last["RSI"])
            adx      = float(last["ADX"])
            slope    = float(last["Slope"])
            close    = float(last["Close"])
            ma60     = float(last["MA60"])

            # Soft Regime 計算（V12.1 邏輯）
            smooth    = 0.7
            bull_raw  = 1.0 if (rsi > 52 and slope > 0 and close > ma60) else 0.0
            bear_raw  = 1.0 if (rsi < 40 and slope < 0) else 0.0
            range_raw = 1.0 - bull_raw - bear_raw

            # 讀取舊 regime 做平滑
            old = load_json(os.path.join(REGIME_DIR, "regime_state.json"))
            if old:
                bull_s  = smooth * old.get("bull", bull_raw) + (1-smooth)*bull_raw
                bear_s  = smooth * old.get("bear", bear_raw) + (1-smooth)*bear_raw
                range_s = 1 - bull_s - bear_s
            else:
                bull_s = bull_raw; bear_s = bear_raw; range_s = range_raw

            # Normalize
            total = bull_s + bear_s + range_s + 1e-9
            bull_s /= total; bear_s /= total; range_s /= total

            # Strategy switching
            if bull_s >= 0.55:
                strat = "bull"; a_path = "45"; b_path = "423"
            elif bear_s >= 0.60:
                strat = "bear"; a_path = None; b_path = None
            else:
                strat = "range"; a_path = "423"; b_path = "45"

            # Label
            if bull_s > 0.5:   label = "牛市"
            elif bear_s > 0.5: label = "熊市"
            elif bull_s > bear_s: label = "偏多震盪"
            else:               label = "偏空震盪"

            s5d  = float(bm["Close"].pct_change(5).iloc[-1])
            s20d = float(bm["Close"].pct_change(20).iloc[-1])

            # 歷史 regime（追加月末）
            old_hist = old.get("history", []) if old else []
            curr_month = datetime.now().strftime("%Y-%m")
            if not any(h["month"] == curr_month for h in old_hist):
                old_hist.append({
                    "month": curr_month,
                    "bear": round(bear_s, 4),
                    "range": round(range_s, 4),
                    "bull": round(bull_s, 4),
                    "label": label,
                })
                old_hist = old_hist[-24:]  # 保留近24個月

            regime = {
                "bear": round(bear_s, 4), "range": round(range_s, 4), "bull": round(bull_s, 4),
                "label": label, "active_strategy": strat,
                "active_path": a_path, "backup_path": b_path,
                "slope_5d": round(s5d, 4), "slope_20d": round(s20d, 4),
                "mkt_rsi": round(rsi, 1), "adx": round(adx, 1),
                "history": old_hist,
            }

        save_json(os.path.join(REGIME_DIR, "regime_state.json"), regime)
        log.info(f"Regime: {regime.get('label')} | 熊:{regime.get('bear',0)*100:.0f}% 牛:{regime.get('bull',0)*100:.0f}%")
        return regime
    except Exception as e:
        log.error(f"❌ Regime 失敗: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# Step 3：V4 快照（市場強度 TOP20）
# ──────────────────────────────────────────────────────────────
def step_v4(eng, regime: dict) -> dict:
    log.info("=== Step 3: V4 Market Strength Snapshot ===")

    SYMBOLS = [
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
    "1711","1727","2404","2489","3060","3374","3498","3535","3580","3587","3665","4749","4989","6187","6217","6290","6418","6443","6470","6542","6546","6706","6831","6861","6877","8028","8111"

    ]

    try:
        if eng and hasattr(eng, "run"):
            v4_data = eng.run(symbols=SYMBOLS, regime=regime, today=TODAY)
        else:
            # fallback：批次抓取並計算快照分數
            import yfinance as yf
            import pandas_ta as ta
            import numpy as np

            rows = []
            for sym in SYMBOLS[:30]:  # 限制數量避免超時
                try:
                    ticker = f"{sym}.TW"
                    df = yf.download(ticker, period="60d", progress=False, auto_adjust=True)
                    if df.empty or len(df) < 20:
                        ticker = f"{sym}.TWO"
                        df = yf.download(ticker, period="60d", progress=False, auto_adjust=True)
                    if df.empty or len(df) < 20:
                        continue
                    if isinstance(df.columns, __import__("pandas").MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # 計算指標
                    df["RSI"]  = ta.rsi(df["Close"], 14)
                    df["ATR"]  = ta.atr(df["High"], df["Low"], df["Close"], 14)
                    vol_fast   = df["Volume"].ewm(span=5).mean()
                    vol_slow   = df["Volume"].ewm(span=20).mean()
                    df["PVO"]  = (vol_fast - vol_slow) / (vol_slow + 1e-9) * 100
                    rsi_norm   = df["RSI"].clip(0, 100)
                    atr_mean   = df["ATR"].rolling(20).mean()
                    atr_norm   = (df["ATR"] / (atr_mean + 1e-9) * 50).clip(0, 100)
                    df["VRI"]  = (rsi_norm * 0.6 + atr_norm * 0.4).clip(0, 100)
                    df["Slope"]= df["Close"].pct_change(5) * 100

                    # Slope Z
                    slope_win  = df["Slope"].tail(30)
                    slope_z    = (float(df["Slope"].iloc[-1]) - slope_win.mean()) / (slope_win.std() + 1e-9)

                    last     = df.iloc[-1]
                    pvo      = float(last["PVO"])
                    vri      = float(last["VRI"])
                    slope    = float(last["Slope"])
                    close    = float(last["Close"])

                    # 評分
                    score = 50.0
                    score += min(slope_z * 8, 20)
                    score += min(pvo * 0.5, 15) if pvo > 0 else max(pvo * 0.3, -10)
                    score += 8 if 40 <= vri <= 75 else (-5 if vri > 90 else 0)

                    # 訊號型態（V4 邏輯簡化版）
                    sig_a = pvo > 0 and vri < 45
                    sig_b = pvo > 8 and vri > 70
                    sig_c = vri > 70

                    if sig_a and sig_b and sig_c: signal = "三合一(ABC)"
                    elif sig_a and sig_b:         signal = "二合一(AB)"
                    elif sig_a and sig_c:         signal = "二合一(AC)"
                    elif sig_b and sig_c:         signal = "二合一(BC)"
                    elif sig_a:                   signal = "單一(A)"
                    elif sig_b:                   signal = "單一(B)"
                    elif sig_c:                   signal = "單一(C)"
                    else:                         signal = "基準"

                    # 操作判定
                    if slope_z >= 1.5 and pvo > 5:   action = "強力買進"
                    elif slope_z >= 0.5 and pvo > 0: action = "買進"
                    elif slope_z < -1.0:              action = "賣出"
                    else:                             action = "觀察"

                    rows.append({
                        "symbol": sym, "score": round(score, 2),
                        "pvo": round(pvo, 2), "vri": round(vri, 1),
                        "slope_z": round(slope_z, 2), "slope": round(slope, 3),
                        "action": action, "signal": signal,
                        "close": round(close, 1),
                        "regime": regime.get("label", "—"),
                    })
                except Exception as sym_e:
                    log.warning(f"  {sym} 跳過: {sym_e}")
                    continue

            rows.sort(key=lambda x: x["score"], reverse=True)
            for i, r in enumerate(rows):
                r["rank"] = i + 1
            rows = rows[:20]  # TOP20

            scores = [r["score"] for r in rows]
            import numpy as np
            v4_data = {
                "market": "TW", "top20": rows,
                "pool_mu":    round(float(np.mean(scores)), 2) if scores else 0,
                "pool_sigma": round(float(np.std(scores)), 2) if scores else 0,
                "win_rate":   57.1,  # V4 歷史勝率（固定）
            }

        # 儲存 latest（供 app.py 讀取）+ 歷史快照
        save_json(os.path.join(V4_DIR, "v4_latest.json"), v4_data)
        save_json(os.path.join(V4_DIR, f"v4_{TS}.json"), v4_data)
        log.info(f"V4 TOP20 完成，共 {len(v4_data.get('top20',[]))} 筆")
        return v4_data
    except Exception as e:
        log.error(f"❌ V4 失敗: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# Step 4：V12.1 決策快照
# ──────────────────────────────────────────────────────────────
def step_v12(eng, regime: dict, v4_data: dict) -> dict:
    log.info("=== Step 4: V12.1 Trading Decision Snapshot ===")

    SYMBOLS_V12 = [
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
    "1711","1727","2404","2489","3060","3374","3498","3535","3580","3587","3665","4749","4989","6187","6217","6290","6418","6443","6470","6542","6546","6706","6831","6861","6877","8028","8111"

    ]

    try:
        if eng and hasattr(eng, "run"):
            v12_data = eng.run(symbols=SYMBOLS_V12, regime=regime, v4_snapshot=v4_data, today=TODAY)
        else:
            # fallback：基於 V4 快照 + Regime 產出 V12.1 部位監控
            import yfinance as yf
            import pandas_ta as ta
            import numpy as np
            from scipy import stats as scipy_stats

            # 讀取舊快照（維持持倉狀態）
            old_v12  = load_json(os.path.join(V12_DIR, "v12_latest.json"))
            old_pos  = {p["symbol"]: p for p in (old_v12 or {}).get("positions", [])}

            active_strat = regime.get("active_strategy", "range")
            a_path       = regime.get("active_path", "423")
            b_path       = regime.get("backup_path", "45")
            ev_entry_min = 0.040 if active_strat == "bear" else 0.030

            # 從 V4 TOP20 取得候選
            top20 = v4_data.get("top20", [])
            candidates = [r for r in top20
                          if r.get("action") in ("強力買進","買進")
                          and r.get("score", 0) > 55]

            positions = []
            for cand in candidates[:8]:
                sym = cand["symbol"]
                try:
                    ticker = f"{sym}.TW"
                    df     = yf.download(ticker, period="90d", progress=False, auto_adjust=True)
                    if df.empty: continue
                    if isinstance(df.columns, __import__("pandas").MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 14)
                    df["RSI"] = ta.rsi(df["Close"], 14)
                    vol_fast  = df["Volume"].ewm(span=5).mean()
                    vol_slow  = df["Volume"].ewm(span=20).mean()
                    df["PVO"] = (vol_fast - vol_slow) / (vol_slow + 1e-9) * 100

                    last      = df.iloc[-1]
                    close     = float(last["Close"])
                    atr       = float(last["ATR"]) if not np.isnan(last["ATR"]) else close * 0.02
                    pvo       = float(last["PVO"])

                    # EV 計算（簡化：基於近30日報酬分布）
                    rets      = df["Close"].pct_change().tail(30).dropna()
                    ev        = float(rets.mean() * 100 * 5)  # 持5日期望值
                    ev        = round(min(max(ev, 0.5), 12.0), 2)

                    if ev < ev_entry_min * 100:
                        continue

                    # EV 等級
                    ev_tier = "⭐核心" if ev > 5 else ("🔥主力" if ev > 3 else "📌補位")

                    # 路徑判定（V12.1 槽位邏輯）
                    r_probs = {"45": 0.65, "423": 0.35} if active_strat == "bull" else {"423": 0.65, "45": 0.35}
                    path    = a_path if np.random.random() < r_probs.get(a_path, 0.65) else b_path
                    path    = path or "423"

                    # 持倉繼承
                    if sym in old_pos:
                        old = old_pos[sym]
                        days_held = old.get("days_held", 0) + 1
                        entry_p   = old.get("entry_price", close)
                        curr_ret  = (close - entry_p) / entry_p * 100
                        stop_p    = old.get("stop_price", round(close - atr * 1.5, 2))
                        tp1_p     = old.get("tp1_price", round(close * 1.06, 2))
                        action    = old.get("action", "持有")
                        quality   = old.get("quality", "Pure")

                        # V12.1 EV 衰退出場信號
                        ev_orig = old.get("ev", ev)
                        exit_sig = "—"
                        if ev < ev_orig * 0.65:
                            exit_sig = "EV衰退"
                        elif pvo < -0.3 and ev < ev_orig * 0.8:
                            exit_sig = "量能枯竭"
                        elif days_held >= 20 and curr_ret < -5:
                            exit_sig = "時間衰減"
                        elif close < stop_p:
                            exit_sig = "硬停損"
                    else:
                        days_held = 0
                        entry_p   = close
                        curr_ret  = 0.0
                        stop_p    = round(close - atr * 1.5, 2)
                        tp1_p     = round(close * 1.06, 2)
                        action    = "進場"
                        quality   = "Pure"
                        exit_sig  = "—"

                    # Flicker 判定（V4 VRI/PVO 不穩定）
                    vri_ratio = cand.get("vri", 50)
                    if vri_ratio < 40 or abs(pvo) > 20:
                        quality = "Flicker"
                        tp1_p   = round(close * 1.04, 2)  # 提前停利

                    positions.append({
                        "symbol":     sym,
                        "path":       path,
                        "ev":         ev,
                        "ev_tier":    ev_tier,
                        "action":     action,
                        "exit_signal":exit_sig,
                        "quality":    quality,
                        "days_held":  days_held,
                        "curr_ret_pct": round(curr_ret, 2),
                        "entry_price":round(entry_p, 2),
                        "tp1_price":  tp1_p,
                        "stop_price": stop_p,
                        "regime":     active_strat,
                        "close":      round(close, 2),
                    })
                except Exception as sym_e:
                    log.warning(f"  V12 {sym} 跳過: {sym_e}")
                    continue

            # 歷史統計（固定 V12.1 OOS 結果）
            stats = {
                "total_trades": 112, "win_rate": 57.1, "avg_ev": 5.29,
                "max_dd": -6.58, "sharpe": 5.36, "t_stat": 4.032,
                "simple_cagr": 96.9, "pl_ratio": 2.31,
            }

            v12_data = {
                "market": "TW", "positions": positions,
                "stats": stats, "regime": active_strat,
                "active_path": a_path, "backup_path": b_path,
            }

        save_json(os.path.join(V12_DIR, "v12_latest.json"), v12_data)
        save_json(os.path.join(V12_DIR, f"v12_{TS}.json"), v12_data)
        log.info(f"V12.1 完成，部位數: {len(v12_data.get('positions',[]))}")
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

        # 抓取已出場的部位並追加
        new_exits = [
            {
                "date":        TODAY,
                "sym":         p["symbol"],
                "action_type": "賣出",
                "exit_type":   p.get("exit_signal","—"),
                "ret":         round(p.get("curr_ret_pct", 0) / 100, 4),
                "path":        p.get("path","—"),
                "year":        datetime.now().year,
            }
            for p in v12_data.get("positions", [])
            if p.get("exit_signal", "—") not in ("—", "無", "持倉中")
        ]

        if new_exits:
            old_hist.extend(new_exits)
            old_hist = old_hist[-500:]  # 保留最近500筆
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(old_hist, f, ensure_ascii=False, indent=2, default=str)
            log.info(f"  新增 {len(new_exits)} 筆交易記錄")
        else:
            log.info("  今日無新出場記錄")
    except Exception as e:
        log.error(f"❌ 歷史記錄更新失敗: {e}")


# ──────────────────────────────────────────────────────────────
# 主控流程
# ──────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info(f"📅 資源法 daily_run.py 啟動 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # 1. 載入引擎
    engines = import_engines()

    # 2. 執行各步驟
    market_data = step_market(engines.get("feature"))
    regime_data = step_regime(engines.get("regime"), market_data)
    v4_data     = step_v4(engines.get("v4"), regime_data)
    v12_data    = step_v12(engines.get("v12"), regime_data, v4_data)
    step_history(v12_data)

    # 3. 執行摘要
    log.info("=" * 60)
    log.info("✅ 全部步驟完成")
    log.info(f"  大盤: {market_data.get('index_close',0):,.1f} ({market_data.get('index_chg_pct',0):+.2f}%)")
    log.info(f"  Regime: {regime_data.get('label','—')}")
    log.info(f"  V4 TOP20: {len(v4_data.get('top20',[]))} 檔")
    log.info(f"  V12.1 部位: {len(v12_data.get('positions',[]))} 檔")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
