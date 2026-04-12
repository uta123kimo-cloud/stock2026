"""
debug_dashboard.py — V4 個股資料偵測工具 v3.7
新增功能：
- ETF 合成大盤測試（0050.TW + 006208.TW，取代 ^TWII）
- 個股 .TW → .TWO fallback 結果統計
- 429 rate limit 偵測與建議
- 黃色 st.warning 改為表格顯示，減少視覺噪音
"""

import time
import random
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="資源法 Debug Dashboard v3.7", layout="wide")

# ──────────────────────────────────────────────────────────────
# 全域樣式
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;600;900&display=swap');
body, .stApp { background:#0b0f1a !important; color:#e2e8f0 !important; }
[data-testid="stMetricValue"] { color:#06b6d4 !important; font-family:'IBM Plex Mono',monospace !important; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 資源法 Debug Dashboard v3.7")
st.caption("偵測 ETF 合成大盤 + 個股 .TW/.TWO 抓取結果 | 已移除 ^TWII")

# ──────────────────────────────────────────────────────────────
# 輸入區
# ──────────────────────────────────────────────────────────────
def default_symbols():
    return [
        "2330","2317","2454","2308","2382","2303","3711","2412","2357","3231",
        "3030","3706","8096","2313","4958","6669","3008","2379","3443","3661",
        "6415","3035","2408","3131","5274",
    ]

col_inp1, col_inp2 = st.columns([3, 1])
with col_inp1:
    symbols_text = st.text_area(
        "輸入股票代碼（逗號分隔）",
        ",".join(default_symbols()),
        height=80
    )
with col_inp2:
    period_sel  = st.selectbox("抓取期間", ["30d", "60d", "90d"], index=1)
    max_retry   = st.number_input("429 最大重試次數", min_value=1, max_value=6, value=3)

SYMBOLS = [s.strip() for s in symbols_text.split(",") if s.strip()]

# ══════════════════════════════════════════════════════════════
# Section 1：ETF 合成大盤測試（取代 ^TWII）
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🌐 大盤指數來源測試（ETF 合成）")
st.caption("使用 0050.TW（65%）+ 006208.TW（35%）合成大盤代理，已完全取代 ^TWII")

_ETF_WEIGHTS = {"0050.TW": 0.65, "006208.TW": 0.35}

if st.button("🔍 測試 ETF 合成大盤", key="etf_btn"):
    etf_results = []
    frames = {}

    for ticker, weight in _ETF_WEIGHTS.items():
        ok = False
        rows_n = 0
        latest_close = "—"
        err_msg = ""

        for attempt in range(int(max_retry)):
            try:
                df = yf.download(ticker, period=period_sel, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).strip() for c in df.columns]

                if not df.empty and "Close" in df.columns and len(df) >= 10:
                    ok = True
                    rows_n = len(df)
                    latest_close = f"{float(df['Close'].iloc[-1]):.2f}"
                    frames[ticker] = (df, weight)
                    break
                else:
                    err_msg = f"資料不足 ({len(df)} 筆)"
                    break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "too many requests" in err_str:
                    wait = (2 ** attempt) * 3 + random.uniform(1, 3)
                    err_msg = f"429，等待 {wait:.1f}s"
                    time.sleep(wait)
                else:
                    err_msg = str(e)[:60]
                    break

        etf_results.append({
            "ETF 代碼":    ticker,
            "權重":        f"{weight:.0%}",
            "狀態":        "✅ 成功" if ok else "❌ 失敗",
            "資料筆數":    rows_n,
            "最新收盤":    latest_close,
            "備註":        err_msg if not ok else "正常",
        })

    st.dataframe(pd.DataFrame(etf_results), use_container_width=True)

    # 合成計算
    if len(frames) >= 1:
        all_dates = None
        for ticker, (df, _) in frames.items():
            idx = df.index
            all_dates = idx if all_dates is None else all_dates.intersection(idx)

        if all_dates is not None and len(all_dates) >= 5:
            composite_close = pd.Series(0.0, index=all_dates)
            total_w = 0.0
            for ticker, (df, weight) in frames.items():
                sub  = df.loc[df.index.isin(all_dates)].reindex(all_dates)
                base = float(sub["Close"].iloc[0])
                if base > 0:
                    factor = 100.0 / base
                    composite_close += sub["Close"].fillna(method="ffill") * factor * weight
                    total_w += weight

            if total_w > 0:
                composite_close /= total_w
                st.success(
                    f"✅ ETF 合成大盤成功 | "
                    f"共 {len(all_dates)} 個交易日 | "
                    f"最新合成指數: {float(composite_close.iloc[-1]):.4f}"
                )
                df_preview = composite_close.tail(5).reset_index()
                df_preview.columns = ["日期", "合成指數（base=100）"]
                st.dataframe(df_preview, use_container_width=True)
            else:
                st.error("合成計算失敗：權重為 0")
        else:
            st.warning("共同交易日不足，無法合成")
    else:
        st.error("ETF 均下載失敗，無法合成大盤")


# ══════════════════════════════════════════════════════════════
# Section 2：個股 .TW / .TWO 偵測
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 個股 OHLCV 偵測（.TW → .TWO 自動切換）")

if st.button("🚀 開始個股偵測", key="stock_btn"):
    results  = []
    success  = 0
    fail     = 0
    rate429  = 0
    progress = st.progress(0)
    status_txt = st.empty()

    for i, sym in enumerate(SYMBOLS):
        df_final   = None
        used_ticker = None
        used_suffix = None
        got_429    = False
        fail_reason = ""

        for suffix in [".TW", ".TWO"]:
            ticker  = f"{sym}{suffix}"

            for attempt in range(int(max_retry)):
                try:
                    df = yf.download(
                        ticker, period=period_sel,
                        progress=False, auto_adjust=True, timeout=15
                    )
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [str(c).strip() for c in df.columns]

                    if not df.empty and "Close" in df.columns and len(df) >= 20:
                        df_final    = df
                        used_ticker = ticker
                        used_suffix = suffix
                        success    += 1
                        break
                    else:
                        fail_reason = f"{ticker}: 資料不足 ({len(df)} 筆)"
                        break

                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "too many requests" in err_str:
                        wait = (2 ** attempt) * 3 + random.uniform(1, 4)
                        # 不使用 st.warning（減少黃字），改用 status 文字顯示
                        status_txt.info(f"⏳ {ticker} 429，等待 {wait:.1f}s...")
                        time.sleep(wait)
                        got_429 = True
                        rate429 += 1
                    else:
                        fail_reason = f"{ticker}: {str(e)[:50]}"
                        break
            else:
                if got_429:
                    fail_reason = f"{ticker}: 429 重試耗盡"

            if df_final is not None:
                break

        if df_final is None:
            fail += 1

        rows_n = len(df_final) if df_final is not None else 0
        latest = (
            f"{float(df_final['Close'].iloc[-1]):.2f}"
            if df_final is not None and len(df_final) > 0 else "—"
        )
        results.append({
            "代號":       sym,
            "狀態":       "✅ 成功" if df_final is not None else "❌ 失敗",
            "使用 Ticker": used_ticker or "—",
            "後綴":       used_suffix or "—",
            "資料筆數":   rows_n,
            "最新收盤":   latest,
            "429 次數":   rate429,
            "備註":       "" if df_final is not None else fail_reason,
        })
        rate429 = 0  # 重置（per stock）
        progress.progress((i + 1) / len(SYMBOLS))
        status_txt.empty()
        time.sleep(random.uniform(0.3, 0.8))

    # 統計摘要
    st.subheader("📊 統計摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ 成功", success)
    c2.metric("❌ 失敗", fail)
    c3.metric("成功率", f"{success / max(1, len(SYMBOLS)) * 100:.1f}%")
    suffix_counts = {}
    for r in results:
        s = r.get("後綴", "—")
        if s != "—":
            suffix_counts[s] = suffix_counts.get(s, 0) + 1
    tw_cnt  = suffix_counts.get(".TW",  0)
    two_cnt = suffix_counts.get(".TWO", 0)
    c4.metric(".TW / .TWO", f"{tw_cnt} / {two_cnt}")

    # 完整表格
    st.subheader("📋 詳細結果")
    df_result = pd.DataFrame(results)
    st.dataframe(df_result, use_container_width=True)

    # 失敗清單
    fail_list = df_result[df_result["狀態"] == "❌ 失敗"]["代號"].tolist()
    if fail_list:
        st.subheader("⚠️ 抓不到資料的股票")
        st.code(", ".join(fail_list))
        st.caption(
            "可能原因：① 代號已下市 / 改名  ② yfinance rate limit（明日再試）"
            "  ③ 上市/上櫃代號需確認"
        )

    # .TWO 成功清單（上櫃股確認）
    two_list = df_result[df_result["後綴"] == ".TWO"]["代號"].tolist()
    if two_list:
        st.subheader("🔄 已切換為 .TWO 上櫃的股票")
        st.code(", ".join(two_list))

    # 預覽前3筆成功資料
    st.subheader("🔍 成功樣本預覽（前3筆）")
    shown = 0
    for r in results:
        if r["狀態"] == "✅ 成功" and shown < 3:
            ticker = r["使用 Ticker"]
            try:
                df_prev = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
                if isinstance(df_prev.columns, pd.MultiIndex):
                    df_prev.columns = df_prev.columns.get_level_values(0)
                st.markdown(f"**{r['代號']}** ({ticker})")
                st.dataframe(df_prev.tail(3), use_container_width=True)
                shown += 1
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
# Section 3：快速說明
# ══════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📖 Debug 說明"):
    st.markdown("""
    ### 黃色警告（st.warning）說明
    本版已將大部分 `st.warning()` 改為：
    - **狀態文字**（`st.empty()`）：顯示 429 等待進度，結束後自動清除
    - **表格欄位**（備註欄）：顯示失敗原因，不佔用視覺空間
    - **`st.code()`**：失敗代號清單，比黃字更易複製

    ### ETF 合成取代 ^TWII
    | 來源 | 說明 |
    |------|------|
    | `0050.TW` (65%) | 元大台灣50，追蹤台灣50，與TWII相關>0.99 |
    | `006208.TW` (35%) | 富邦台50，流動性佳，同追蹤台灣50 |
    | ~~`^TWII`~~ | **已完全移除**，yfinance 對此代號 429 頻繁 |

    ### .TW / .TWO 處理邏輯
    ```
    for suffix in [".TW", ".TWO"]:
        if 資料 >= 20 筆 → 採用，break
        if 429 → 指數退讓（2^n × 3 秒）後重試
        if 其他錯誤 → 直接換 suffix
    if 兩個都失敗 → 標記為 ❌ 失敗
    ```

    ### 常見問題
    - **429 Too Many Requests**：yfinance 被 rate limit，建議降低股票池大小或增加延遲
    - **資料不足**：該代號可能為上櫃（.TWO），已自動切換
    - **兩者均失敗**：代號可能已下市、更名或不在 Yahoo Finance 資料庫
    """)
