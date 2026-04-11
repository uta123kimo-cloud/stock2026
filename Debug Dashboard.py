import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="V4 Debug Dashboard", layout="wide")

st.title("📊 V4 Market Data Debug Dashboard")

# === Input Symbols ===
def default_symbols():
    return [
        "2330","2317","2454","2308","2382","2303","3711","2412","2357","3231",
        "3030","3706","8096","2313","4958"
    ]

symbols = st.text_area("輸入股票代碼（逗號分隔）", ",".join(default_symbols()))
SYMBOLS = [s.strip() for s in symbols.split(",") if s.strip()]

# === Debug Run ===
if st.button("🚀 開始偵測"):
    results = []
    success = 0
    fail = 0

    progress = st.progress(0)

    for i, sym in enumerate(SYMBOLS):
        df_final = None
        used_ticker = None

        for suffix in [".TW", ".TWO"]:
            ticker = f"{sym}{suffix}"
            try:
                df = yf.download(ticker, period="60d", progress=False, auto_adjust=True)

                if not df.empty and len(df) >= 20:
                    df_final = df
                    used_ticker = ticker
                    success += 1
                    break

            except Exception as e:
                st.warning(f"{ticker} error: {e}")

        if df_final is None:
            fail += 1
            results.append({
                "symbol": sym,
                "status": "❌ Fail",
                "ticker": "-",
                "rows": 0
            })
        else:
            results.append({
                "symbol": sym,
                "status": "✅ Success",
                "ticker": used_ticker,
                "rows": len(df_final)
            })

        progress.progress((i + 1) / len(SYMBOLS))

    # === Summary ===
    st.subheader("📈 統計結果")
    col1, col2 = st.columns(2)
    col1.metric("成功", success)
    col2.metric("失敗", fail)

    # === Table ===
    df_result = pd.DataFrame(results)
    st.subheader("📋 詳細結果")
    st.dataframe(df_result, use_container_width=True)

    # === Fail List ===
    fail_list = df_result[df_result["status"] == "❌ Fail"]["symbol"].tolist()
    if fail_list:
        st.subheader("⚠️ 抓不到資料的股票")
        st.write(fail_list)

    # === Success Preview ===
    st.subheader("🔍 成功樣本預覽")
    for r in results[:3]:
        if r["status"] == "✅ Success":
            st.write(f"{r['symbol']} ({r['ticker']})")
            df = yf.download(r['ticker'], period="10d", progress=False, auto_adjust=True)
            st.dataframe(df.tail(), use_container_width=True)
