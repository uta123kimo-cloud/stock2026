# 資源法 V12.1 完整系統架構
# GitHub Repo 結構說明

```
stock2026/                          ← GitHub 主倉庫
│
├── .github/
│   └── workflows/
│       └── daily.yml               ← ⭐ 自動排程（盤中×5 + 盤後 + 日結）
│
├── core/                           ← 核心引擎（計算邏輯）
│   ├── v4/
│   │   └── 資源法2026-V4.py       ← V4 市場強度引擎（你的原始檔）
│   └── v12/
│       └── 高EV-V12_1.py          ← V12.1 路徑交易引擎（你的原始檔）
│
├── engine/                         ← 新增決策層
│   ├── signal_engine.py            ← 包裝 V4，輸出標準化買進訊號
│   ├── sell_engine.py              ← 賣出訊號（從 portfolio_manager 拆出）
│   └── portfolio_manager.py        ← ⭐ 核心：買賣決策 + 部位管理
│
├── jobs/
│   └── daily_run.py                ← 主排程腳本（你的原始檔 v4.0）
│                                      整合 portfolio_manager 調用
│
├── backtest/
│   └── backtest_engine.py          ← ⭐ 回測引擎
│
├── streamlit_app/
│   └── app.py                      ← ⭐ v5.0 Streamlit（含買賣原因）
│
├── storage/                        ← ⭐ GitHub 存儲（自動 push）
│   ├── v4/
│   │   ├── v4_latest.json          ← 最新 V4 快照
│   │   └── v4_YYYY-MM-DD.json      ← 每日歸檔（回測用）
│   ├── v12/
│   │   ├── v12_latest.json         ← 最新 V12 快照
│   │   └── v12_YYYY-MM-DD.json
│   ├── regime/
│   │   ├── regime_state.json
│   │   └── regime_YYYY-MM-DD.json
│   ├── market/
│   │   └── market_snapshot.json
│   ├── daily_snapshots/
│   │   └── snapshot_YYYY-MM-DD.json ← 每日組合快照（持倉+買賣原因）
│   ├── positions.csv               ← ⭐ 即時持倉（GitHub 存儲）
│   ├── trades.csv                  ← ⭐ 交易歷史（買賣原因記錄）
│   ├── signals.csv                 ← 今日買進訊號清單
│   ├── portfolio_latest.json       ← 最新組合狀態（Streamlit 讀取）
│   └── backtest_result.json        ← 回測結果
│
├── stock_set.json                  ← 自選股 + 全池設定
├── requirements.txt
└── README.md
```

## 資料流說明

### 買進流程
```
daily_run.py
  → fetch_market_index()          # 大盤資料
  → _RegimeEngine.run()           # 判斷牛熊
  → run_v4(symbols, regime)       # V4 評分 TOP30
  → run_v12(symbols, regime, v4)  # V12 路徑識別
  → portfolio_manager.run()       # 買賣決策
  → 寫入 storage/                 # 存儲
  → git push                      # 推上 GitHub
```

### 賣出決策優先序（portfolio_manager.py）
```
S1: 硬停損（curr ≤ stop_price 或 ret ≤ -10%）
S2: TP2 全出（curr ≥ tp2_price）
S3: Trailing Stop（TP1後高點回撤 ≥ 8%）
S4: EV嚴重衰退（ev_now < 0.5%）
S5: 時間衰減+Slope加速（持>7天，EV衰20%+slope負）
S6: 時間衰減（EV衰>35%）
S7: 量能枯竭（PVO < -0.30 持>3天）
S8: 保本出場（TP1後跌回成本）
※TP1: 達+20%時出清50%，其餘繼續持有
```

### Streamlit 讀取路徑
```
GitHub Raw → storage/portfolio_latest.json  → 今日買賣原因 + 持倉
GitHub Raw → storage/trades.csv             → 歷史交易記錄
GitHub Raw → storage/backtest_result.json   → 回測績效圖
```

## 環境變數（GitHub Secrets）
```
FINMIND_TOKEN    ← FinMind API Token
GEMINI_API_KEY   ← Gemini AI 分析
GITHUB_TOKEN     ← 自動 push storage（Actions 內建）
GITHUB_OWNER     ← 你的 GitHub 帳號
GITHUB_REPO      ← 倉庫名稱
```

## 回測使用方式
```python
from backtest.backtest_engine import BacktestEngine

bt = BacktestEngine(
    storage_dir="storage",
    start_date="2025-01-01",
    end_date="2026-04-13",
    initial_capital=1_000_000,
)
result = bt.run()
# → 輸出 storage/backtest_result.json
```
