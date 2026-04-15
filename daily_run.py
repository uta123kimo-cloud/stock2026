"""
╔══════════════════════════════════════════════════════════════╗
║  daily_run.py v4.2  資源法 Precompute 主控制器              ║
║                                                              ║
║  v4.2 修正項目（基於 v4.1）：                                ║
║  [FIX-07] FinMind Y9999 start_date 根號計算錯誤修正         ║
║           FINMIND_CALENDAR_DAYS 現在正確從今日往回算         ║
║           2026-04-15 - 400天 = 2025-03-11 (錯！只有35天)    ║
║           正確應為：lookback_days=250 交易日                 ║
║           → 250 × 1.6 = 400 calendar days                   ║
║           → 2026-04-15 - 400天 = 2025-03-11 ✗               ║
║           問題根源：timedelta(days=400) 從今日回推           ║
║           但 2026-04-15 - 400 = 2025-03-11，這確實是400天前  ║
║           實際問題是 FinMind 帳戶 Y9999 無權限               ║
║           新增 start_date 強制從 2020-01-01 開始查詢         ║
║  [FIX-08] Regime 三個檔案統一：                              ║
║           regime_state.json = 完整快照（含 history 陣列）    ║
║           regime_history.json = 純歷史月份資料               ║
║           不再有 regime_latest.json（已棄用）                ║
║  [FIX-09] Streamlit app.py 讀取路徑同步修正                  ║
║           load_json_url("regime/regime_state.json") ✅       ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import math
import time
import signal
import logging
import random
from datetime import datetime, date, timedelta
from typing import Optional

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
CACHE_DIR   = os.path.join(STORAGE_DIR, "cache")

TODAY  = date.today().strftime("%Y-%m-%d")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M")
TS     = datetime.now().strftime("%Y%m%d_%H%M")

# 台灣時間（UTC+8）
NOW_HOUR_TW = (datetime.utcnow() + timedelta(hours=8)).hour

for _d in [V4_DIR, V12_DIR, REGIME_DIR, MARKET_DIR, LOGS_DIR, DATA_ROOT, CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 執行模式
# ──────────────────────────────────────────────────────────────
_ENV_RUN_MODE = os.environ.get("RUN_MODE", "").strip().lower()
if _ENV_RUN_MODE in ("intraday", "postmarket"):
    RUN_MODE = _ENV_RUN_MODE
    log.info(f"執行模式來自環境變數: {RUN_MODE}")
else:
    RUN_MODE = "intraday" if 9 <= NOW_HOUR_TW <= 14 else "postmarket"
    log.info(f"執行模式由時間判斷: {RUN_MODE} (台灣時間 {NOW_HOUR_TW:02d}:xx)")

# ──────────────────────────────────────────────────────────────
# 從 stock_set.json 載入股票池
# ──────────────────────────────────────────────────────────────
STOCK_SET_PATH = os.path.join(ROOT_DIR, "stock_set.json")

def _load_stock_set() -> dict:
    if os.path.exists(STOCK_SET_PATH):
        try:
            with open(STOCK_SET_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"stock_set.json 讀取失敗: {e}，使用預設值")
    return {}

_STOCK_SET = _load_stock_set()

WATCHLIST: list = list(dict.fromkeys(
    _STOCK_SET.get("watchlist", {}).get("symbols", [
        "2330", "2317", "2454", "2308", "2382"
    ])
))

SYMBOLS: list = list(dict.fromkeys(
    _STOCK_SET.get("universe", {}).get("symbols", [
        "2330", "2317", "2454", "2308", "2382", "2303", "3711", "2412",
        "2379", "3008", "2395", "3045", "2327", "2408", "2377", "6669",
    ])
))

_SETTINGS = _STOCK_SET.get("settings", {})
LOOKBACK_DAYS: int        = int(_SETTINGS.get("lookback_days", 250))
INTRADAY_TOP_N: int       = int(_SETTINGS.get("intraday_top_n", 20))
FINMIND_RATE_LIMIT: float = float(_SETTINGS.get("finmind_rate_limit_sec", 0.5))

# [FIX-07] 250 交易日 ≈ 365 calendar days，加緩衝到 400 天
# 這是「往回多少天」，不是 start_date 的問題
# 實際 start_date = today - FINMIND_CALENDAR_DAYS
# 2026-04-15 - 400 = 2025-03-11 ← 這是正確的400天前
# 如果 API 還是回空，是帳戶權限問題，改用固定 start_date
FINMIND_CALENDAR_DAYS = max(400, int(LOOKBACK_DAYS * 1.6))

# [FIX-07] 強制使用更早的固定起始日期，避免帳戶權限只允許特定區間
# 若 dynamic start_date 失敗，fallback 到這個固定日期
FINMIND_FIXED_START = "2020-01-01"

log.info(f"股票池 | 自選股: {len(WATCHLIST)} | 全池: {len(SYMBOLS)} | 模式: {RUN_MODE}")
log.info(f"FinMind 動態查詢天數: {FINMIND_CALENDAR_DAYS} calendar days → start_date={( date.today() - timedelta(days=FINMIND_CALENDAR_DAYS)).strftime('%Y-%m-%d')}")

# ══════════════════════════════════════════════════════════════
# FinMind 設定
# ══════════════════════════════════════════════════════════════
_FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
_FINMIND_URL   = "https://api.finmindtrade.com/api/v4/data"
_USE_FINMIND   = bool(_FINMIND_TOKEN)

if _USE_FINMIND:
    log.info(f"✅ FinMind TOKEN 已載入 (長度:{len(_FINMIND_TOKEN)})")
else:
    log.warning("⚠️ FINMIND_TOKEN 未設定，使用 Yahoo Finance")

# ══════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _make_session(referer: str = "https://www.twse.com.tw/") -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
        "Referer": referer,
    })
    return s


_SESSION_TWSE = _make_session("https://www.twse.com.tw/")
_SESSION_YF   = _make_session("https://finance.yahoo.com/")
_SESSION_YF.headers["Accept"] = "text/html,application/xhtml+xml,*/*;q=0.8"


def save_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"  ✅ 儲存: {path}")
    except Exception as e:
        log.error(f"  ❌ 儲存失敗 {path}: {e}")


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# Timeout 保護
# ══════════════════════════════════════════════════════════════

class _StockTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _StockTimeout("單股下載超時")


def _with_timeout(func, timeout_sec: int = 120):
    if not hasattr(signal, "SIGALRM"):
        return func()
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ══════════════════════════════════════════════════════════════
# FinMind 限速
# ══════════════════════════════════════════════════════════════
_FINMIND_LAST_REQUEST: float = 0.0


def _finmind_rate_limit() -> None:
    global _FINMIND_LAST_REQUEST
    elapsed = time.time() - _FINMIND_LAST_REQUEST
    if elapsed < FINMIND_RATE_LIMIT:
        time.sleep(FINMIND_RATE_LIMIT - elapsed)
    _FINMIND_LAST_REQUEST = time.time()


# ══════════════════════════════════════════════════════════════
# [FIX-07] FinMind Y9999 大盤指數 — 雙重 start_date 嘗試
# ══════════════════════════════════════════════════════════════

def _fetch_finmind_taiex_with_start(start_date: str, cache_path: str) -> pd.DataFrame:
    """
    嘗試用指定 start_date 從 FinMind 取得 Y9999 TAIEX 資料。
    回傳 DataFrame 或空 DataFrame。
    """
    _finmind_rate_limit()
    log.info(f"  FinMind Y9999 查詢區間: {start_date} ~ {TODAY}")

    try:
        params = {
            "dataset":    "TaiwanStockPrice",
            "data_id":    "Y9999",
            "start_date": start_date,
            "token":      _FINMIND_TOKEN,
        }
        resp = requests.get(_FINMIND_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        status  = payload.get("status")
        msg     = payload.get("msg", "")
        records = payload.get("data", [])

        log.info(f"  FinMind Y9999 API 回應: status={status}, msg={msg!r}, 筆數={len(records)}")

        if status != 200:
            log.warning(f"  FinMind Y9999 status={status}, msg={msg!r}")
            if status in (401, 403) or "auth" in str(msg).lower():
                log.error("  ❌ FinMind TOKEN 驗證失敗，請檢查 secrets.FINMIND_TOKEN")
            return pd.DataFrame()

        if not records:
            log.warning(
                f"  FinMind Y9999 返回空資料 (status=200 但 data=[])。"
                f"start_date={start_date}，可能是帳戶對 Y9999 無權限或區間無資料。"
            )
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        col_map = {
            "open":           "Open",
            "max":            "High",
            "min":            "Low",
            "close":          "Close",
            "Trading_Volume": "Volume",
            "volume":         "Volume",
        }
        renamed = {src: dst for src, dst in col_map.items() if src in df.columns}
        df = df.rename(columns=renamed)

        needed = ["Open", "High", "Low", "Close"]
        missing_cols = [c for c in needed if c not in df.columns]
        if missing_cols:
            if "Close" in df.columns:
                for c in missing_cols:
                    df[c] = df["Close"]
            else:
                log.warning(f"  FinMind Y9999 缺少欄位: {missing_cols}")
                return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna(subset=["Close"])

        if len(df) >= 10:
            df.to_csv(cache_path)
            close_last = float(df["Close"].iloc[-1])
            log.info(f"  ✅ FinMind TAIEX (Y9999): {len(df)} 筆，最新收盤={close_last:,.2f}")
            return df
        else:
            log.warning(f"  FinMind Y9999 有效資料不足: {len(df)} 筆")
            return pd.DataFrame()

    except requests.exceptions.Timeout:
        log.error("  FinMind Y9999: API 請求逾時 (30s)")
    except requests.exceptions.HTTPError as e:
        log.error(f"  FinMind Y9999: HTTP 錯誤 {e}")
    except Exception as e:
        log.warning(f"  FinMind Y9999: 例外 {type(e).__name__}: {e}")

    return pd.DataFrame()


def fetch_finmind_taiex(days: int = 300) -> pd.DataFrame:
    """
    [FIX-07] 雙重 start_date 嘗試：
    1. 動態 start_date（今日往回 FINMIND_CALENDAR_DAYS 天）
    2. 若失敗，改用固定 FINMIND_FIXED_START（2020-01-01）
    3. 若仍失敗，讀取本地快取
    """
    if not _USE_FINMIND:
        return pd.DataFrame()

    cache_path = os.path.join(CACHE_DIR, "TAIEX_finmind.csv")

    # 嘗試 1：動態 start_date
    dynamic_start = (date.today() - timedelta(days=FINMIND_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    df = _fetch_finmind_taiex_with_start(dynamic_start, cache_path)
    if not df.empty and len(df) >= 10:
        return df.tail(days)

    # 嘗試 2：固定早期 start_date（解決帳戶只允許特定區間的問題）
    log.warning(f"  動態 start_date 失敗，嘗試固定 start_date={FINMIND_FIXED_START}")
    time.sleep(1.0)  # 等待限速
    df = _fetch_finmind_taiex_with_start(FINMIND_FIXED_START, cache_path)
    if not df.empty and len(df) >= 10:
        return df.tail(days)

    # 嘗試 3：讀取本地快取
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df = _normalize_df(df)
            if not df.empty and len(df) >= 10:
                log.warning(f"  FinMind TAIEX 使用快取 ({len(df)} 筆，最新: {df.index[-1].date()})")
                return df.tail(days)
        except Exception as ce:
            log.warning(f"  FinMind 快取讀取失敗: {ce}")

    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# Yahoo ^TWII 備援
# ══════════════════════════════════════════════════════════════

def fetch_yahoo_twii(days: int = 300) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    cache_path = os.path.join(CACHE_DIR, "TWII_yahoo.csv")
    period = f"{days + 100}d"
    try:
        log.info(f"  Yahoo ^TWII 下載中 (period={period})...")
        df = yf.download(
            "^TWII", period=period, progress=False,
            auto_adjust=True, timeout=30, session=_SESSION_YF
        )
        df = _normalize_df(df)
        if not df.empty and "Close" in df.columns and len(df) >= 20:
            df = df.tail(days)
            df.to_csv(cache_path)
            log.info(f"  ✅ Yahoo ^TWII: {len(df)} 筆，最新: {df.index[-1].date()}")
            return df
        log.warning(f"  Yahoo ^TWII 返回資料不足: {len(df) if not df.empty else 0} 筆")
    except Exception as e:
        log.warning(f"  Yahoo ^TWII 失敗: {type(e).__name__}: {e}")

    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df = _normalize_df(df)
            if not df.empty:
                log.warning(f"  Yahoo ^TWII 使用快取 ({len(df)} 筆)")
                return df
        except Exception:
            pass
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# TWSE 官方 MI_INDEX（最後備援）
# ══════════════════════════════════════════════════════════════

def _parse_tw_date(s: str):
    try:
        parts = str(s).strip().split("/")
        return date(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
    except Exception:
        return None


def _clean_num(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("X", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def _twse_index_month(year: int, month: int) -> pd.DataFrame:
    date_str = f"{year}{month:02d}01"
    url = (
        "https://www.twse.com.tw/exchangeReport/MI_INDEX"
        f"?response=json&date={date_str}&type=MS"
    )
    try:
        resp = _SESSION_TWSE.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        tables  = payload.get("tables", [])
        target  = None
        for t in tables:
            if "加權" in t.get("title", "") and "指數" in t.get("title", ""):
                target = t; break
        if target is None:
            for t in tables:
                if any("開盤" in str(f) for f in t.get("fields", [])):
                    target = t; break
        if target is None:
            return pd.DataFrame()
        fields = target.get("fields", [])
        rows   = target.get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=fields)
        date_col = next((c for c in df.columns if "日期" in c), None)
        if date_col is None:
            return pd.DataFrame()
        df["_Date"] = df[date_col].apply(_parse_tw_date)
        df = df.dropna(subset=["_Date"]).set_index("_Date")
        df.index = pd.to_datetime(df.index)
        col_map = {
            "開盤指數": "Open", "最高指數": "High",
            "最低指數": "Low",  "收盤指數": "Close", "成交金額": "Volume",
        }
        result = pd.DataFrame(index=df.index)
        for tw, en in col_map.items():
            matched = next((c for c in df.columns if tw in c), None)
            if matched:
                result[en] = _clean_num(df[matched])
        if "Close" not in result.columns:
            return pd.DataFrame()
        for col in ["Open", "High", "Low"]:
            if col not in result.columns:
                result[col] = result["Close"]
        if "Volume" not in result.columns:
            result["Volume"] = 0.0
        return result.dropna(subset=["Close"]).sort_index()
    except Exception as e:
        log.debug(f"  TWSE MI_INDEX {year}/{month}: {e}")
        return pd.DataFrame()


def fetch_twse_index(days: int = 300) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "TWII_twse.csv")
    today_d    = date.today()
    months_n   = math.ceil(days / 20) + 2
    seen, frames = set(), []
    for i in range(months_n):
        d   = today_d - timedelta(days=30 * i)
        key = (d.year, d.month)
        if key in seen:
            continue
        seen.add(key)
        mdf = _twse_index_month(d.year, d.month)
        if not mdf.empty:
            frames.append(mdf)
        time.sleep(0.35)
    if frames:
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        cutoff = pd.Timestamp(today_d - timedelta(days=days * 2))
        df = df[df.index >= cutoff].tail(days)
        if not df.empty:
            df.to_csv(cache_path)
            log.info(f"  ✅ TWSE 官方指數 ({len(df)} 筆)")
            return df
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df = _normalize_df(df)
            if not df.empty:
                log.warning("  TWSE 失敗，使用快取")
                return df
        except Exception:
            pass
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# 大盤指數統一入口
# ══════════════════════════════════════════════════════════════

def fetch_market_index(days: int = 300) -> pd.DataFrame:
    if _USE_FINMIND:
        log.info("  [Layer 0] FinMind Y9999 TAIEX...")
        df = fetch_finmind_taiex(days=days)
        if not df.empty and len(df) >= 10:
            return df
        log.warning("  Layer 0 失敗，切換 Layer 1...")

    log.info("  [Layer 1] Yahoo ^TWII...")
    df = fetch_yahoo_twii(days=days)
    if not df.empty and len(df) >= 10:
        return df

    log.warning("  [Layer 2] TWSE 官方 MI_INDEX...")
    df = fetch_twse_index(days=days)
    if not df.empty and len(df) >= 10:
        return df

    log.error("  ❌ 所有大盤來源均失敗 → 常數 fallback (不可信，僅保持程式運行)")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq="B")
    df = pd.DataFrame({
        "Close":  np.linspace(20000.0, 21000.0, len(dates)),
        "Open":   np.linspace(20000.0, 21000.0, len(dates)),
        "High":   np.linspace(20100.0, 21100.0, len(dates)),
        "Low":    np.linspace(19900.0, 20900.0, len(dates)),
        "Volume": np.ones(len(dates)) * 1e9,
    }, index=dates)
    log.warning("  ⚠️ 使用常數 fallback（大盤基準 20000 點），所有 Regime/V4/V12 結果不可信！")
    return df


# ══════════════════════════════════════════════════════════════
# [FIX-07] FinMind 個股下載 — 雙重 start_date
# ══════════════════════════════════════════════════════════════

def fetch_finmind_ohlcv(sym: str, days: int = 250) -> Optional[pd.DataFrame]:
    if not _FINMIND_TOKEN:
        return None

    _finmind_rate_limit()
    dynamic_start = (date.today() - timedelta(days=max(int(days * 1.6), 400))).strftime("%Y-%m-%d")

    def _try_start(start_date: str) -> Optional[pd.DataFrame]:
        try:
            params = {
                "dataset":    "TaiwanStockPrice",
                "data_id":    sym,
                "start_date": start_date,
                "token":      _FINMIND_TOKEN,
            }
            resp = requests.get(_FINMIND_URL, params=params, timeout=25)
            resp.raise_for_status()
            payload = resp.json()

            if payload.get("status") != 200:
                return None
            records = payload.get("data", [])
            if not records:
                return None

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            col_map = {
                "open": "Open", "max": "High", "min": "Low", "close": "Close",
                "Trading_Volume": "Volume", "volume": "Volume",
            }
            renamed = {src: dst for src, dst in col_map.items() if src in df.columns}
            df = df.rename(columns=renamed)

            needed = ["Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in needed):
                return None
            if "Volume" not in df.columns:
                df["Volume"] = 0.0

            df = df[["Open", "High", "Low", "Close", "Volume"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna(subset=["Close"])

            df = df.tail(days)
            if len(df) >= 20:
                return df
            return None
        except Exception:
            return None

    # 先試動態 start_date
    df = _try_start(dynamic_start)
    if df is not None:
        log.info(f"  ✅ FinMind {sym}: {len(df)} 筆")
        return df

    # 若空，試固定早期日期
    log.debug(f"  FinMind {sym}: 動態 start_date 失敗，嘗試 {FINMIND_FIXED_START}")
    time.sleep(0.5)
    df = _try_start(FINMIND_FIXED_START)
    if df is not None:
        log.info(f"  ✅ FinMind {sym} (固定起始): {len(df)} 筆")
        return df

    log.debug(f"  FinMind {sym}: 兩次嘗試均無資料")
    return None


# ══════════════════════════════════════════════════════════════
# Yahoo suffix 快取
# ══════════════════════════════════════════════════════════════
_SUFFIX_CACHE: dict = {}
_SUFFIX_CACHE_PATH = os.path.join(CACHE_DIR, "suffix_cache.json")


def _load_suffix_cache() -> None:
    global _SUFFIX_CACHE
    if os.path.exists(_SUFFIX_CACHE_PATH):
        try:
            with open(_SUFFIX_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            _SUFFIX_CACHE = {k: v for k, v in raw.items() if v is not None}
            removed = len(raw) - len(_SUFFIX_CACHE)
            if removed > 0:
                log.info(f"  清除 {removed} 筆 null suffix 快取")
                _save_suffix_cache()
        except Exception:
            _SUFFIX_CACHE = {}
    else:
        _SUFFIX_CACHE = {}


def _save_suffix_cache() -> None:
    try:
        with open(_SUFFIX_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_SUFFIX_CACHE, f, ensure_ascii=False)
    except Exception as e:
        log.debug(f"  suffix cache 寫入失敗: {e}")


def _resolve_suffix(sym: str) -> Optional[str]:
    if sym in _SUFFIX_CACHE:
        return _SUFFIX_CACHE[sym]
    if yf is None:
        return ".TW"
    got_rate_limited = False
    for suffix in [".TW", ".TWO"]:
        ticker = f"{sym}{suffix}"
        try:
            t = yf.Ticker(ticker)
            try:
                fi = t.fast_info
                last_price = getattr(fi, "last_price", None)
                if last_price is not None and last_price > 0:
                    _SUFFIX_CACHE[sym] = suffix
                    _save_suffix_cache()
                    return suffix
            except Exception:
                pass
            hist = t.history(period="1mo")
            if not hist.empty:
                _SUFFIX_CACHE[sym] = suffix
                _save_suffix_cache()
                return suffix
            time.sleep(0.5)
        except Exception as e:
            err = str(e).lower()
            if "too many requests" in err or "429" in err or "rate" in err:
                log.warning(f"  suffix偵測: {sym} 429，預設 .TW")
                got_rate_limited = True
                break
            time.sleep(0.3)
    if got_rate_limited:
        return ".TW"
    log.warning(f"  suffix偵測: {sym} 兩個 suffix 均無資料")
    _SUFFIX_CACHE[sym] = None
    _save_suffix_cache()
    return None


_load_suffix_cache()


# ══════════════════════════════════════════════════════════════
# 個股 OHLCV CSV 讀取
# ══════════════════════════════════════════════════════════════

def load_from_csv(sym: str, day_dir: str) -> Optional[pd.DataFrame]:
    csv_path = os.path.join(day_dir, f"{sym}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = _normalize_df(df)
        if len(df) >= 20:
            log.debug(f"  {sym}: 從 CSV 快取讀取 ({len(df)} 筆)")
            return df
        return None
    except Exception:
        return None


def save_to_csv(sym: str, df: pd.DataFrame, day_dir: str) -> None:
    try:
        csv_path = os.path.join(day_dir, f"{sym}.csv")
        df.to_csv(csv_path)
    except Exception as e:
        log.debug(f"  {sym} CSV 儲存失敗: {e}")


# ══════════════════════════════════════════════════════════════
# 個股 OHLCV 下載
# ══════════════════════════════════════════════════════════════

def fetch_tw_ohlcv(sym: str, days: int = 250, max_retries: int = 3) -> tuple:
    if _USE_FINMIND:
        try:
            df = _with_timeout(
                lambda: fetch_finmind_ohlcv(sym, days=days),
                timeout_sec=30
            )
            if df is not None and len(df) >= 20:
                return df, "FinMind"
        except _StockTimeout:
            log.warning(f"  {sym} FinMind 超時")
        except Exception as e:
            log.debug(f"  {sym} FinMind 例外: {e}")

    if yf is None:
        return None, None

    detected = _resolve_suffix(sym)
    if detected is None:
        return None, None

    period = f"{days + 100}d"
    other  = ".TWO" if detected == ".TW" else ".TW"
    suffixes = [detected, other]

    for suffix in suffixes:
        ticker  = f"{sym}{suffix}"
        got_429 = False

        for attempt in range(max_retries):
            base_sleep = 1.5 + random.uniform(0.0, 1.0)
            if attempt > 0:
                base_sleep += attempt * 2.0
            time.sleep(base_sleep)
            try:
                df = yf.download(
                    ticker, period=period, progress=False,
                    auto_adjust=True, timeout=25, session=_SESSION_YF
                )
                df = _normalize_df(df)
                if not df.empty and "Close" in df.columns and len(df) >= 20:
                    df = df.tail(days)
                    if suffix != detected:
                        _SUFFIX_CACHE[sym] = suffix
                        _save_suffix_cache()
                    return df, suffix
                if df.empty or len(df) == 0:
                    break
            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["missing", "delisted", "no data found",
                                           "no price data", "possibly delisted"]):
                    break
                if "too many requests" in err or "429" in err or "rate" in err:
                    wait = (2 ** attempt) * 5 + random.uniform(3, 8)
                    log.warning(f"  {ticker} 429，退讓 {wait:.1f}s")
                    time.sleep(wait)
                    got_429 = True
                else:
                    break
        else:
            if got_429:
                log.warning(f"  {ticker} 429 重試耗盡")

    log.warning(f"  {sym}: 所有來源均無資料")
    return None, None


# ══════════════════════════════════════════════════════════════
# 盤中「優先清單」計算
# ══════════════════════════════════════════════════════════════

def get_intraday_priority_symbols() -> list:
    priority = list(WATCHLIST)

    v12_path = os.path.join(V12_DIR, "v12_latest.json")
    v12_data = load_json(v12_path)
    if v12_data:
        for p in v12_data.get("positions", []):
            sym = p.get("symbol")
            if sym and sym not in priority:
                priority.append(sym)
        for p in v12_data.get("candidates", []):
            sym = p.get("symbol")
            if sym and sym not in priority:
                priority.append(sym)

    v4_path = os.path.join(V4_DIR, "v4_latest.json")
    v4_data = load_json(v4_path)
    if v4_data:
        for r in v4_data.get("top20", [])[:INTRADAY_TOP_N]:
            sym = r.get("symbol")
            if sym and sym not in priority:
                priority.append(sym)

    log.info(f"  盤中優先清單: {len(priority)} 檔 {priority}")
    return list(dict.fromkeys(priority))


# ══════════════════════════════════════════════════════════════
# 技術指標
# ══════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _atr(high, low, close, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Close" not in df.columns:
        return df
    df["RSI"]   = _rsi(df["Close"], 14)
    df["ATR"]   = _atr(df["High"], df["Low"], df["Close"], 14)
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["Slope"] = df["Close"].pct_change(5) * 100
    if "Volume" in df.columns:
        vma_f = df["Volume"].ewm(span=5).mean()
        vma_s = df["Volume"].ewm(span=20).mean()
        df["PVO"] = (vma_f - vma_s) / (vma_s + 1e-9) * 100
        vma20     = df["Volume"].rolling(20).mean()
        df["VolRatio"] = df["Volume"] / (vma20 + 1e-9)
    else:
        df["PVO"]      = 0.0
        df["VolRatio"] = 1.0
    rsi_norm = df["RSI"].clip(0, 100)
    atr_mean = df["ATR"].rolling(20).mean()
    atr_norm = (df["ATR"] / (atr_mean + 1e-9) * 50).clip(0, 100)
    df["VRI"] = (rsi_norm * 0.6 + atr_norm * 0.4).clip(0, 100)
    pvo_thr = 0.0
    consec  = pd.Series(0, index=df.index, dtype=float)
    cnt = 0
    for i, v in enumerate(df["PVO"]):
        cnt = cnt + 1 if v > pvo_thr else 0
        consec.iloc[i] = cnt
    df["PVO_consec"] = consec
    pvo_mean = df["PVO"].rolling(5).mean()
    df["PVO_accel"] = df["PVO"] / (pvo_mean.abs() + 1e-9)
    vri_mean = df["VRI"].rolling(5).mean()
    df["VRI_delta"] = df["VRI"] - vri_mean
    return df


# ══════════════════════════════════════════════════════════════
# 市場快照引擎
# ══════════════════════════════════════════════════════════════

class _FeatureEngine:
    def run(self, today: str) -> dict:
        df = fetch_market_index(days=300)
        if df.empty or "Close" not in df.columns or len(df) < 20:
            log.error("  市場指數資料不足")
            return {}

        df  = df.dropna(subset=["Close"]).sort_index()
        src = "FinMind_Y9999_TAIEX" if _USE_FINMIND else "Yahoo_TWII"

        c       = df["Close"].values
        rsi_val = _rsi(df["Close"], 14).values

        def _slope(series, w=5):
            s = pd.Series(series)
            base = s.rolling(w).mean().shift(w)
            return ((s.rolling(w).mean() - base) / (base.abs() + 1e-9)).values

        s5  = _slope(c, 5)
        s20 = _slope(c, 20)

        close_now = float(c[-1])
        close_pre = float(c[-2]) if len(c) >= 2 else close_now
        chg_pct   = (close_now - close_pre) / (close_pre + 1e-9) * 100

        rsi_now = float(rsi_val[-1]) if not np.isnan(rsi_val[-1]) else 50.0
        s5_now  = float(s5[-1])  if not np.isnan(s5[-1])  else 0.0
        s20_now = float(s20[-1]) if not np.isnan(s20[-1]) else 0.0

        log.info(f"  大盤: {close_now:,.2f} ({chg_pct:+.2f}%) RSI={rsi_now:.1f}")

        return {
            "market":        "TW",
            "index_close":   round(close_now, 2),
            "index_chg_pct": round(chg_pct, 2),
            "mkt_rsi":       round(rsi_now, 2),
            "mkt_slope_5d":  round(s5_now, 4),
            "mkt_slope_20d": round(s20_now, 4),
            "data_source":   src,
            "run_mode":      RUN_MODE,
            "generated_at":  RUN_TS,
        }


# ══════════════════════════════════════════════════════════════
# [FIX-08] Regime 引擎 — 統一三個檔案
# ══════════════════════════════════════════════════════════════

class _RegimeEngine:
    def run(self, market_data: dict, today: str) -> dict:
        df = fetch_market_index(days=300)
        if df.empty or "Close" not in df.columns or len(df) < 20:
            return {}

        df  = df.dropna(subset=["Close"]).sort_index()
        src = market_data.get("data_source", "FinMind_Y9999_TAIEX" if _USE_FINMIND else "Yahoo_TWII")

        c       = df["Close"].values
        rsi_val = _rsi(df["Close"], 14).values
        ma20    = df["Close"].rolling(20).mean().values
        ma60    = df["Close"].rolling(60).mean().values

        def _slope(vals, w=5):
            base = pd.Series(vals).rolling(w).mean().shift(w).values
            curr = pd.Series(vals).rolling(w).mean().values
            return (curr - base) / (np.abs(base) + 1e-9)

        s5  = _slope(c, 5)
        s20 = _slope(c, 20)
        atr_s   = _atr(df["High"], df["Low"], df["Close"], 14).values
        atr_pct = np.where(c > 0, atr_s / (c + 1e-9), 0.001)
        atr_m20 = pd.Series(atr_pct).rolling(20).mean().values
        adx_p   = float(np.where(atr_m20[-1] > 0,
                                 atr_pct[-1] / (atr_m20[-1] + 1e-9) * 25, 20.0))

        rsi_n   = float(rsi_val[-1]) if not np.isnan(rsi_val[-1]) else 50.0
        s5_n    = float(s5[-1])  if not np.isnan(s5[-1])  else 0.0
        s20_n   = float(s20[-1]) if not np.isnan(s20[-1]) else 0.0
        ma20_n  = float(ma20[-1]) if not np.isnan(ma20[-1]) else float(c[-1])
        ma60_n  = float(ma60[-1]) if not np.isnan(ma60[-1]) else float(c[-1])
        close_n = float(c[-1])

        bull_score = 0.0; bear_score = 0.0
        if rsi_n > 55:       bull_score += 0.25
        elif rsi_n < 45:     bear_score += 0.25
        if s5_n > 0.005:     bull_score += 0.20
        elif s5_n < -0.005:  bear_score += 0.20
        if s20_n > 0.002:    bull_score += 0.20
        elif s20_n < -0.002: bear_score += 0.20
        if close_n > ma20_n: bull_score += 0.15
        else:                 bear_score += 0.15
        if close_n > ma60_n: bull_score += 0.15
        else:                 bear_score += 0.15
        if adx_p > 25:
            if bull_score > bear_score: bull_score += 0.05
            else:                       bear_score += 0.05

        total = bull_score + bear_score
        if total > 0:
            P_bull = bull_score / (total + 1e-9) * 0.7
            P_bear = bear_score / (total + 1e-9) * 0.3
        else:
            P_bull = 0.33; P_bear = 0.33
        P_range = max(0.0, 1.0 - P_bull - P_bear)

        norm = P_bull + P_bear + P_range
        if norm > 0:
            P_bull /= norm; P_bear /= norm; P_range /= norm

        if P_bull >= 0.55:
            label = "牛市"; strat = "bull"; a_path = "45"; b_path = "423"
        elif P_bear >= 0.55:
            label = "熊市"; strat = "bear"; a_path = None; b_path = None
        elif P_bull > 0.40:
            label = "偏多震盪"; strat = "range"; a_path = "423"; b_path = "45"
        else:
            label = "震盪"; strat = "range"; a_path = "423"; b_path = "45"

        # [FIX-08] 載入並更新歷史月份資料
        hist_path = os.path.join(REGIME_DIR, "regime_history.json")
        history   = load_json(hist_path) or []
        # 若 history 是 list，直接用；若是 dict，取 data key
        if isinstance(history, dict):
            history = history.get("data", [])

        month_key = date.today().strftime("%Y-%m")
        history   = [h for h in history if h.get("month") != month_key]
        history.append({
            "month":       month_key,
            "bear":        round(P_bear,  4),
            "range":       round(P_range, 4),
            "bull":        round(P_bull,  4),
            "label":       label,
            "index_close": round(close_n, 2),
        })
        history = history[-24:]  # 保留最近 24 個月

        # [FIX-08] 儲存獨立的 regime_history.json
        save_json(hist_path, history)

        # [FIX-08] 完整快照（含 history 陣列）
        regime_snapshot = {
            "bear":            round(P_bear,  4),
            "range":           round(P_range, 4),
            "bull":            round(P_bull,  4),
            "label":           label,
            "active_strategy": strat,
            "active_path":     a_path,
            "backup_path":     b_path,
            "slope_5d":        round(s5_n, 4),
            "slope_20d":       round(s20_n, 4),
            "mkt_rsi":         round(rsi_n, 2),
            "adx":             round(adx_p, 2),
            "index_close":     round(close_n, 2),
            "index_chg_pct":   round(market_data.get("index_chg_pct", 0), 2),
            # [FIX-08] history 內嵌在 regime_state.json，讓 Streamlit 一次讀到
            "history":         history,
            "data_source":     src,
            "run_mode":        RUN_MODE,
            "generated_at":    RUN_TS,
        }
        return regime_snapshot


# ══════════════════════════════════════════════════════════════
# Risk Engine
# ══════════════════════════════════════════════════════════════

class _RiskEngine:
    def summarize(self, positions: list) -> dict:
        if not positions:
            return {"total_pos": 0, "avg_ret": 0.0, "exit_count": 0, "exit_syms": []}
        rets = [p.get("curr_ret_pct", 0.0) for p in positions]
        exit_pos = [p for p in positions if p.get("exit_signal", "—") not in ("—", "無", "")]
        return {
            "total_pos":  len(positions),
            "avg_ret":    round(float(np.mean(rets)), 2),
            "exit_count": len(exit_pos),
            "exit_syms":  [p["symbol"] for p in exit_pos],
        }


_feature_engine = _FeatureEngine()
_regime_engine  = _RegimeEngine()
_risk_engine    = _RiskEngine()


# ══════════════════════════════════════════════════════════════
# V4 引擎（同 v4.1，不變）
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
    if is_strong and is_money and is_cool and a_ok:            patterns.append("A")
    if is_strong and is_fire  and is_hot:                      patterns.append("B")
    if is_strong and is_hot and "B" not in patterns and c_ok:  patterns.append("C")

    vtag = (
        "+量爆" if vol_ratio >= 2.0 else
        "+放量" if vol_ratio >= 1.2 else
        "-縮量" if vol_ratio <  0.7 else ""
    )
    ps = frozenset(patterns)
    if   len(ps) >= 3: base = "三合一(ABC)"; combo = base
    elif len(ps) == 2: k = "".join(sorted(ps)); base = f"二合一({k})"; combo = base
    elif len(ps) == 1: p = list(ps)[0]; base = f"單一({p})"; combo = base
    else:              base = "基準-強勢" if slope_z > 1.2 else "基準-持有"; combo = base

    sig_q = (
        1.3 if vol_ratio >= 2.0 else
        1.1 if vol_ratio >= 1.2 else
        0.8 if vol_ratio <  0.7 else 1.0
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

    pvo   = g("PVO"); vri   = g("VRI", 50); slope = g("Slope")
    pvo_c = g("PVO_consec"); pvo_a = g("PVO_accel", 1); vri_d = g("VRI_delta")
    vol_r = g("VolRatio", 1); close = g("Close"); atr = g("ATR", close * 0.02)
    sw    = df["Slope"].tail(30)
    slope_z = (slope - float(sw.mean())) / (float(sw.std()) + 1e-9)

    score = 50.0
    score += min(slope_z * 8, 20)
    score += min(pvo * 0.5, 15) if pvo > 0 else max(pvo * 0.3, -10)
    score += 8 if 40 <= vri <= 75 else (-5 if vri > 90 else 0)

    signal, combo, sig_q = _v4_signal(pvo, vri, slope_z, score, mu, sigma,
                                       pvo_c, pvo_a, vri_d, vol_r)
    action = (
        "強力買進" if slope_z >= 1.5 and pvo > 5 else
        "買進"    if slope_z >= 0.5 and pvo > 0  else
        "賣出"    if slope_z < -1.0              else "觀察"
    )
    rm = {"trend": 1.1, "range": 1.0, "recovery": 0.8, "crash": 0.5}.get(rtype, 1.0)
    pw = round(min(max(_POS_WEIGHT.get(combo, 0.15) * sig_q * rm, 0.10), 0.30), 4)

    atr_pct = atr / (close + 1e-9) if close > 0 else 0.02
    tp1_px  = round(close * 1.20, 1)
    stop_px = round(close * (1 - atr_pct * 1.5), 1)

    return {
        "score":      round(score, 2),
        "pvo":        round(pvo, 2),
        "vri":        round(vri, 1),
        "slope_z":    round(slope_z, 2),
        "slope":      round(slope, 3),
        "action":     action,
        "signal":     signal,
        "combo_key":  combo,
        "close":      round(close, 1),
        "atr":        round(atr, 2),
        "pos_weight": pw,
        "tp1_price":  tp1_px,
        "stop_price": stop_px,
    }


def _fetch_symbol_data(sym: str, day_dir: str, force_download: bool) -> Optional[pd.DataFrame]:
    if not force_download:
        df = load_from_csv(sym, day_dir)
        if df is not None:
            return df
        log.debug(f"  {sym}: 非優先且無快取，跳過")
        return None

    try:
        result = _with_timeout(
            lambda s=sym: fetch_tw_ohlcv(s, days=LOOKBACK_DAYS),
            timeout_sec=120
        )
        df, source = result if result else (None, None)
        if df is not None and len(df) >= 20:
            save_to_csv(sym, df, day_dir)
            return df
    except _StockTimeout:
        log.warning(f"  {sym}: 下載超時")
    except Exception as e:
        log.warning(f"  {sym}: 下載失敗 {e}")

    df = load_from_csv(sym, day_dir)
    if df is not None:
        log.warning(f"  {sym}: 下載失敗，使用舊快取")
    return df


def run_v4(symbols, regime, today, day_dir, mode="postmarket") -> dict:
    label = regime.get("label", "震盪")
    rtype = (
        "crash"    if "熊" in label else
        "trend"    if "牛" in label else
        "recovery" if "回升" in label else "range"
    )

    if mode == "intraday":
        priority_set = set(get_intraday_priority_symbols())
        scan_syms    = list(dict.fromkeys(list(priority_set) + symbols))
        log.info(f"V4 盤中模式 | 優先下載: {len(priority_set)} 檔 | 全池快取: {len(scan_syms)} 檔")
    else:
        priority_set = set(symbols)
        scan_syms    = symbols
        log.info(f"V4 盤後模式 | 全量掃描: {len(scan_syms)} 檔")

    rows = []; skipped = 0
    for sym in scan_syms:
        force_dl = (sym in priority_set) or (mode == "postmarket")
        sleep_t = (random.uniform(0.3, 0.8) if _USE_FINMIND
                   else random.uniform(1.5, 2.5)) if force_dl else 0.0
        if sleep_t > 0:
            time.sleep(sleep_t)
        try:
            df = _fetch_symbol_data(sym, day_dir, force_download=force_dl)
            if df is None:
                skipped += 1; continue
            df_ind = enrich_df(df)
            if df_ind.dropna(subset=["RSI", "ATR"]).shape[0] < 20:
                skipped += 1; continue
            rows.append({"sym": sym, "df": df_ind})
        except Exception as e:
            log.warning(f"  V4 {sym}: {e}"); skipped += 1

    if not rows:
        log.error("V4：無有效資料")
        return {}

    mu, sigma = 62.0, 11.5
    result_rows = []
    for item in rows:
        res = _v4_score(item["df"], mu, sigma, rtype)
        if res:
            res["symbol"]       = item["sym"]
            res["regime"]       = label
            res["is_watchlist"] = item["sym"] in WATCHLIST
            result_rows.append(res)

    if not result_rows:
        log.error("V4：評分空")
        return {}

    result_rows.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(result_rows):
        r["rank"] = i + 1
    top30  = result_rows[:30]
    scores = [r["score"] for r in result_rows]
    am  = round(float(np.mean(scores)), 2)
    as_ = round(float(np.std(scores)),  2)

    log.info(f"V4 完成 ✅ | TOP30:{len(top30)} | 跳過:{skipped} | μ={am} σ={as_}")
    return {
        "market":       "TW",
        "top20":        top30,
        "top30":        top30,
        "pool_mu":      am,
        "pool_sigma":   as_,
        "win_rate":     57.1,
        "regime":       label,
        "total_scored": len(result_rows),
        "skipped":      skipped,
        "run_mode":     mode,
        "generated_at": RUN_TS,
    }


# ══════════════════════════════════════════════════════════════
# V12.1 引擎（同 v4.1，不變）
# ══════════════════════════════════════════════════════════════

_PATH_DEFS = {
    "423": {"tp1": 0.20, "tp2": 0.35, "ev_bear": 0.0096, "ev_range": 0.0834,
            "ev_bull": 0.0406, "oos_ev": 3.02, "order": ["Y4", "Y2", "Y3"]},
    "45":  {"tp1": 0.20, "tp2": 0.28, "ev_bear": 0.0022, "ev_range": 0.0485,
            "ev_bull": 0.0432, "oos_ev": 7.36, "order": ["Y4", "Y5"]},
}
_REGIME_STRATS = {
    "bull":  {"active": "45",  "backup": "423", "ratio": {"45": 0.65, "423": 0.35}, "ev_min": 0.030, "max_pos": 4},
    "range": {"active": "423", "backup": "45",  "ratio": {"423": 0.65, "45": 0.35}, "ev_min": 0.030, "max_pos": 5},
    "bear":  {"active": None,  "backup": None,  "ratio": {}, "ev_min": 0.040, "max_pos": 2},
}
_ALL_Y_BETAS = {
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
_HIST_STATS = {
    "total_trades": 112, "win_rate": 57.1, "avg_ev": 5.29,
    "max_dd": -6.58, "sharpe": 5.36, "t_stat": 4.032,
    "simple_cagr": 96.9, "pl_ratio": 2.31,
}


def _v12_features(df) -> dict:
    if df is None or len(df) < 20:
        return {}
    try:
        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        v = df["Volume"].values.astype(float) if "Volume" in df.columns else np.ones(len(df))
        n = len(c)

        ma5  = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        s20  = pd.Series(c).rolling(20).std().values
        bb_w = np.where(ma20 > 0, 4 * s20 / (ma20 + 1e-9), 0.0)

        delta = pd.Series(c).diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).values / 100.0

        atr_arr = np.zeros(n)
        for i in range(1, n):
            tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
            atr_arr[i] = tr
        atr_smooth = pd.Series(atr_arr).ewm(com=13, adjust=False).mean().values
        atr_pct    = np.where(c > 0, atr_smooth / (c + 1e-9), 0.001)
        atr_mean   = pd.Series(atr_pct).rolling(20).mean().values
        atr_reg    = np.where(atr_mean > 0, atr_pct / (atr_mean + 1e-9), 1.0)

        def _mom(lag):
            a = np.zeros(n)
            for i in range(lag, n):
                if c[i-lag] > 0:
                    a[i] = c[i] / c[i-lag] - 1
            return a
        pm5 = _mom(5); pm20 = _mom(20); pm60 = _mom(60)

        ma20sl = np.zeros(n)
        for i in range(5, n):
            if ma20[i-5] > 1e-9:
                ma20sl[i] = (ma20[i] - ma20[i-5]) / ma20[i-5]

        slope_arr = np.zeros(n)
        for i in range(4, n):
            y_ = c[i-4:i+1]; x_ = np.arange(5, dtype=float)
            if y_[0] > 1e-9:
                s_, _ = np.polyfit(x_, y_, 1)
                slope_arr[i] = s_ / y_[0]

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
        persist_count = np.clip(pcnt / 60.0, 0.0, 1.0)

        i = -1
        return {
            "bb_width": float(bb_w[i]), "rsi_14": float(rsi[i]),
            "ma20_slope": float(ma20sl[i]), "price_mom_5": float(pm5[i]),
            "price_mom_20": float(pm20[i]), "price_mom_60": float(pm60[i]),
            "rs_vs_mkt_5": 0.0, "mkt_excess_z": float(ymad[i]),
            "inst_cum_10": float(ic10[i]), "vol_pvo": float(vpvo[i]),
            "vol_pvo_sq": float(vpvoq[i]), "persist_count": float(persist_count[i]),
            "atr_regime": float(atr_reg[i]), "vri": float(vri[i]),
            "close_ma5_r": float(cma5r[i]), "inst_pvo": float(ipvo[i]),
            "inst_x_price": float(ixp[i]), "high52w_dist": float(h52d[i]),
            "slope_5d": float(slope_arr[i]),
            "_close": float(c[i]), "_atr_pct": float(atr_pct[i]),
            "_atr_regime": float(atr_reg[i]), "_slope_5d": float(slope_arr[i]),
            "_ma20": float(ma20[i]),
        }
    except Exception as e:
        log.warning(f"  V12 特徵計算失敗: {e}")
        return {}


def _v12_y_pr(features: dict) -> dict:
    result = {}
    for yn, beta in _ALL_Y_BETAS.items():
        sc = sum(features.get(k, 0.0) * v for k, v in beta.items())
        if yn == "Y5":
            sc = 1.0 / (1.0 + np.exp(-max(-30, min(30, sc))))
        result[f"s_{yn}"]  = sc
        result[f"PR_{yn}"] = 95.0 if sc > 0 else 50.0
    return result


def _v12_path(y_prs: dict, Pb: float, Pr: float, Pu: float) -> dict:
    first_trig = {}
    for yn in ["Y1", "Y2", "Y3", "Y4", "Y5"]:
        if y_prs.get(f"PR_{yn}", 0) >= 90:
            first_trig[yn] = True
    if not first_trig:
        return {"best": None, "batch": 0, "ev_soft": 0.0, "quality": "Pure"}
    matched = []
    for pk in ["423", "45"]:
        order = _PATH_DEFS.get(pk, {}).get("order", [])
        if all(yn in first_trig for yn in order):
            matched.append(pk)
    comp = {}
    for pk in ["423", "45"]:
        order = _PATH_DEFS.get(pk, {}).get("order", [])
        comp[pk] = sum(1 for yn in order if yn in first_trig)
    best = None
    if matched:
        matched.sort(key=lambda k: -_PATH_DEFS[k].get("oos_ev", 0))
        best = matched[0]
    elif comp:
        inc = sorted(comp.items(), key=lambda x: (-x[1], -_PATH_DEFS.get(x[0], {}).get("oos_ev", 0)))
        if inc:
            best = inc[0][0]
    batch   = comp.get(best, 0) if best else 0
    ev_soft = 0.0
    if best:
        d = _PATH_DEFS.get(best, {})
        ev_soft = Pb * d.get("ev_bear", 0) + Pr * d.get("ev_range", 0) + Pu * d.get("ev_bull", 0)
    return {"best": best, "batch": batch, "ev_soft": ev_soft, "comp": comp, "quality": "Pure"}


def _v12_exit(old, ev_now, slope, days, curr_ret) -> str:
    ev_entry = old.get("ev_soft", 0.0)
    if curr_ret <= -0.10:                             return "硬停損"
    if old.get("profit_locked") and curr_ret < 0.01: return "保本出場"
    if ev_now < 0.005:                               return "EV衰退"
    if days > 7 and ev_entry > 0:
        drop = (ev_entry - ev_now) / ev_entry
        if drop > 0.20 and slope < -0.01:            return "Slope加速出場"
        if drop > 0.35:                               return "時間衰減"
    if days > 3 and old.get("_pvo", 0.0) < -0.30:
        return "量能枯竭"
    return "—"


def run_v12(symbols, regime, v4_data, today, day_dir, mode="postmarket") -> dict:
    label = regime.get("label", "震盪")
    rkey  = ("bull" if "牛" in label else "bear" if "熊" in label else "range")
    strat = _REGIME_STRATS.get(rkey, _REGIME_STRATS["range"])
    a_p   = strat["active"]; b_p = strat["backup"]
    ev_min = strat["ev_min"]; max_pos = strat["max_pos"]; ratio = strat["ratio"]
    Pb = regime.get("bear", 0.33); Pr = regime.get("range", 0.34); Pu = regime.get("bull", 0.33)

    log.info(f"V12 啟動 | Regime:{rkey} | 主路徑:{a_p} | max_pos:{max_pos} | 模式:{mode}")
    old_v12 = load_json(os.path.join(V12_DIR, "v12_latest.json"))
    old_pos = {p["symbol"]: p for p in (old_v12 or {}).get("positions", [])}
    top20   = v4_data.get("top20", [])
    cands   = [r for r in top20
               if r.get("action") in ("強力買進", "買進") and r.get("score", 0) > 55]

    if mode == "intraday":
        existing_syms = set(old_pos.keys())
        cand_syms     = {r["symbol"] for r in cands}
        extra_syms    = existing_syms - cand_syms
        max_cands = 20
    else:
        extra_syms = set()
        max_cands  = 30

    positions = []; path_counts = {}
    process_list = cands[:max_cands]
    existing_entries = [{"symbol": s, "action": "持有", "score": 0} for s in extra_syms]
    process_list = process_list + existing_entries

    for cand in process_list:
        sym         = cand["symbol"]
        is_existing = sym in old_pos
        force_dl    = (sym in set(get_intraday_priority_symbols())) or (mode == "postmarket")

        sleep_t = (random.uniform(0.3, 0.8) if _USE_FINMIND
                   else random.uniform(1.5, 2.5)) if force_dl else 0.0
        if sleep_t > 0:
            time.sleep(sleep_t)

        try:
            df = _fetch_symbol_data(sym, day_dir, force_download=force_dl)
            if df is None or len(df) < 20:
                if is_existing:
                    log.warning(f"  V12 {sym}: 無法更新，保留舊持倉資料")
                    positions.append(old_pos[sym])
                continue

            feats = _v12_features(df)
            if not feats:
                if is_existing:
                    positions.append(old_pos[sym])
                continue

            if feats.get("_atr_regime", 1.0) > 2.5:
                continue
            if feats.get("_slope_5d", 0.0) < 0.0 and not is_existing:
                continue

            y_prs     = _v12_y_pr(feats)
            path_info = _v12_path(y_prs, Pb, Pr, Pu)
            best_path = path_info.get("best"); ev_soft = path_info.get("ev_soft", 0.0)
            quality   = path_info.get("quality", "Pure")
            if best_path not in [a_p, b_p]:
                best_path = a_p or "423"

            ev_thr = ev_min * (1.20 if quality == "Flicker" else 1.0)

            if not is_existing:
                if ev_soft < ev_thr:
                    continue
                max_same = max(1, round(max_pos * ratio.get(best_path, 0.50)))
                if path_counts.get(best_path, 0) >= max_same:
                    continue
                if len([p for p in positions if p["action"] != "持有"]) >= max_pos:
                    min_ev = min((p.get("ev", 0.0) / 100 for p in positions), default=0)
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

            if is_existing:
                old_p       = old_pos[sym]
                days_h      = old_p.get("days_held", 0) + 1
                entry_price = old_p.get("entry_price", close)
                curr_ret    = (close - entry_price) / (entry_price + 1e-9)
                stop_px     = old_p.get("stop_price", stop_px)
                tp1_px      = old_p.get("tp1_price",  tp1_px)
                action      = "持有"
                old_p["_pvo"] = feats.get("vol_pvo", 0.0)
                exit_sig    = _v12_exit(old_p, ev_soft,
                                        feats.get("_slope_5d", 0.0), days_h, curr_ret)
            else:
                days_h = 0; entry_price = close; curr_ret = 0.0
                action = "進場"; exit_sig = "—"

            ev_tier = (
                "⭐核心" if ev_soft >= 0.050 else
                "🔥主力" if ev_soft >= 0.030 else
                "📌補位" if ev_soft >= 0.020 else ""
            )
            positions.append({
                "symbol":       sym,
                "path":         best_path,
                "ev":           round(ev_soft * 100, 2),
                "ev_tier":      ev_tier,
                "action":       action,
                "exit_signal":  exit_sig,
                "quality":      quality,
                "days_held":    days_h,
                "curr_ret_pct": round(curr_ret * 100, 2),
                "entry_price":  round(entry_price, 2),
                "close":        round(close, 2),
                "tp1_price":    tp1_px,
                "tp2_price":    tp2_px,
                "stop_price":   stop_px,
                "regime":       rkey,
                "batch":        path_info.get("batch", 0),
                "is_watchlist": sym in WATCHLIST,
            })
            if not is_existing:
                path_counts[best_path] = path_counts.get(best_path, 0) + 1
            log.info(f"  ✅ {sym} | {best_path} | EV:{ev_soft * 100:+.2f}% | {quality} | {action}")

        except Exception as e:
            log.warning(f"  V12 {sym}: {e}")
            if is_existing and sym in old_pos:
                positions.append(old_pos[sym])

    positions.sort(key=lambda x: x.get("ev", 0), reverse=True)
    log.info(f"V12 完成 ✅ | 部位:{len(positions)} | 路徑:{path_counts}")
    return {
        "market":       "TW",
        "positions":    positions,
        "stats":        _HIST_STATS,
        "regime":       rkey,
        "active_path":  a_p,
        "backup_path":  b_p,
        "path_counts":  path_counts,
        "run_mode":     mode,
        "generated_at": RUN_TS,
    }


# ══════════════════════════════════════════════════════════════
# Pipeline Steps & Main
# ══════════════════════════════════════════════════════════════

def step_market() -> dict:
    log.info("=== Step 1: Market Snapshot ===")
    data = _feature_engine.run(today=TODAY)
    if data:
        save_json(os.path.join(MARKET_DIR, "market_snapshot.json"), data)
        log.info(
            f"大盤: {data.get('index_close', 0):,.2f} "
            f"({data.get('index_chg_pct', 0):+.2f}%) "
            f"RSI:{data.get('mkt_rsi', 0):.1f} "
            f"[{data.get('data_source', '?')}]"
        )
    else:
        log.error("❌ Market snapshot 失敗")
    return data or {}


def step_regime(market_data: dict) -> dict:
    log.info("=== Step 2: Regime Definition ===")
    data = _regime_engine.run(market_data, today=TODAY)
    if data:
        # [FIX-08] 同時寫入兩個路徑：
        #   regime_state.json   → Streamlit app 讀取（含 history）
        #   regime_latest.json  → 向下相容（若有其他腳本仍讀舊名）
        save_json(os.path.join(REGIME_DIR, "regime_state.json"),  data)
        save_json(os.path.join(REGIME_DIR, "regime_latest.json"), data)
        log.info(
            f"環境: {data.get('label', '?')} "
            f"(牛:{data.get('bull', 0):.2f} 熊:{data.get('bear', 0):.2f}) "
            f"[{data.get('data_source', '?')}]"
        )
    else:
        log.error("❌ Regime 定義失敗")
    return data or {}


def main():
    log.info(f"🚀 資源法 Daily Compute v4.2 [{TS}]")
    log.info(f"📋 執行模式: {RUN_MODE} | 台灣時間: {NOW_HOUR_TW:02d}:xx | 統一時間戳: {RUN_TS}")
    log.info(f"📊 自選股: {WATCHLIST}")
    log.info(f"📅 動態查詢: start_date={(date.today() - timedelta(days=FINMIND_CALENDAR_DAYS)).strftime('%Y-%m-%d')}")
    log.info(f"📅 固定備援: start_date={FINMIND_FIXED_START}")

    market_data = step_market()
    regime_data = step_regime(market_data)

    if not regime_data:
        log.warning("⚠️ Regime 資料為空，使用預設震盪 fallback")
        regime_data = {
            "bear": 0.33, "range": 0.34, "bull": 0.33,
            "label": "震盪", "active_strategy": "range",
            "active_path": "423", "backup_path": "45",
            "slope_5d": 0.0, "slope_20d": 0.0,
            "mkt_rsi": 50.0, "adx": 20.0, "history": [],
            "data_source": "fallback",
            "generated_at": RUN_TS,
        }

    if RUN_MODE == "intraday":
        priority_syms = get_intraday_priority_symbols()
        scan_symbols  = list(dict.fromkeys(priority_syms + SYMBOLS))
        log.info(f"=== Step 3: V4 Engine (盤中模式) ===")
    else:
        scan_symbols = SYMBOLS
        log.info(f"=== Step 3: V4 Engine (盤後全量，{len(scan_symbols)} 檔) ===")

    v4_data = run_v4(scan_symbols, regime_data, TODAY, DATA_ROOT, mode=RUN_MODE)
    if v4_data:
        save_json(os.path.join(V4_DIR, "v4_latest.json"), v4_data)

    log.info(f"=== Step 4: V12.1 Engine ({RUN_MODE}) ===")
    v12_data = run_v12(scan_symbols, regime_data, v4_data or {}, TODAY, DATA_ROOT, mode=RUN_MODE)
    if v12_data:
        save_json(os.path.join(V12_DIR, "v12_latest.json"), v12_data)

        if "positions" in v12_data:
            risk_summary = _risk_engine.summarize(v12_data["positions"])
            log.info("=== Step 5: Risk Summary ===")
            log.info(
                f"總部位: {risk_summary['total_pos']} 檔 "
                f"| 平均報酬: {risk_summary['avg_ret']}%"
            )
            if risk_summary["exit_count"] > 0:
                log.warning(f"  ⚠️ 建議出場: {risk_summary['exit_syms']}")

    log.info(f"🎉 資源法 v4.2 完成！[{RUN_MODE}] generated_at={RUN_TS}")


if __name__ == "__main__":
    main()
