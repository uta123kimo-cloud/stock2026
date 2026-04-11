# 資源法 AI 戰情室 v3.0 — 架構說明

## 🏗️ 系統架構

```
┌──────────────────────────────┐
│        USER / WEB APP        │
│   Streamlit app.py (展示層)  │
│   - V4 TOP20 市場強度        │
│   - V12.1 路徑決策           │
│   - Regime Dashboard        │
│   - Gemini AI 分析          │
└─────────────┬────────────────┘
              │  READ ONLY JSON（透過 GitHub Raw URL）
              ▼
┌──────────────────────────────────────────────┐
│              DATA LAYER (GitHub)             │
│                                              │
│  storage/v4/v4_latest.json      ← 覆蓋檔    │
│  storage/v4/v4_20260411_1330.json ← 歷史檔  │
│  storage/v12/v12_latest.json                │
│  storage/v12/v12_20260411_1330.json         │
│  storage/regime/regime_state.json           │
│  storage/market/market_snapshot.json        │
│  storage/logs/trade_history.json            │
└──────────────────────┬───────────────────────┘
                       │ WRITE (GitHub Actions)
                       ▼
┌──────────────────────────────────────────────┐
│          COMPUTE LAYER (daily_run.py)        │
│                                              │
│  Step 1: market  → 大盤指數快照              │
│  Step 2: regime  → 牛熊震盪機率             │
│  Step 3: v4      → TOP20 市場強度            │
│  Step 4: v12     → 路徑決策 + 部位監控      │
│  Step 5: history → 交易記錄追加             │
└──────────────────────────────────────────────┘
```

---

## 📅 自動排程時間（台灣時間，週一～週五）

| 時間  | 任務                  |
|-------|-----------------------|
| 09:30 | 開盤快照（V4 掃描）  |
| 12:00 | 盤中更新              |
| 13:30 | 午後計算              |
| 14:30 | V12.1 決策更新        |
| 15:30 | 收盤後計算            |
| 18:00 | 日結存檔              |

> ⚠️ **手動觸發已停用** — workflow 只允許 cron 自動觸發

---

## 📂 儲存檔案結構

### `storage/v4/v4_latest.json`（V4 最新快照）
```json
{
  "generated_at": "2026-04-11 15:30",
  "date": "2026-04-11",
  "market": "TW",
  "pool_mu": 62.3,
  "pool_sigma": 11.5,
  "win_rate": 57.1,
  "top20": [
    {
      "rank": 1,
      "symbol": "2330",
      "score": 84.5,
      "pvo": 12.3,
      "vri": 68.5,
      "slope_z": 1.85,
      "slope": 0.42,
      "action": "強力買進",
      "signal": "三合一(ABC)",
      "close": 845.0,
      "regime": "trend"
    }
  ]
}
```

### `storage/v12/v12_latest.json`（V12.1 最新快照）
```json
{
  "generated_at": "2026-04-11 15:30",
  "date": "2026-04-11",
  "market": "TW",
  "regime": "range",
  "active_path": "423",
  "backup_path": "45",
  "stats": {
    "total_trades": 112,
    "win_rate": 57.1,
    "avg_ev": 5.29,
    "max_dd": -6.58,
    "sharpe": 5.36,
    "t_stat": 4.032,
    "simple_cagr": 96.9,
    "pl_ratio": 2.31
  },
  "positions": [
    {
      "symbol": "2330",
      "path": "423",
      "ev": 6.82,
      "ev_tier": "⭐核心",
      "action": "持有",
      "exit_signal": "—",
      "quality": "Pure",
      "days_held": 3,
      "curr_ret_pct": 4.2,
      "entry_price": 810.0,
      "tp1_price": 858.6,
      "stop_price": 786.0,
      "regime": "range",
      "close": 844.0
    }
  ]
}
```

### `storage/regime/regime_state.json`（Regime 狀態）
```json
{
  "generated_at": "2026-04-11 15:30",
  "bear": 0.22,
  "range": 0.41,
  "bull": 0.37,
  "label": "偏多震盪",
  "active_strategy": "range",
  "active_path": "423",
  "backup_path": "45",
  "slope_5d": 0.0312,
  "slope_20d": 0.0105,
  "mkt_rsi": 54.3,
  "adx": 22.1,
  "history": [
    {"month": "2026-03", "bear": 0.20, "range": 0.43, "bull": 0.37, "label": "偏多震盪"}
  ]
}
```

---

## 🚀 快速部署

### 1. Fork 此 repo 並設定 Secrets

在 GitHub Settings → Secrets and variables → Actions 新增：
- `FINMIND_TOKEN` — FinMind API Token（台股籌碼資料）
- `GEMINI_API_KEY` — Google Gemini API Key（AI 分析）

### 2. Streamlit Cloud 部署

```toml
# .streamlit/secrets.toml
GITHUB_OWNER = "your-username"
GITHUB_REPO  = "quant-storage"
GEMINI_API_KEY = "AIza..."
```

### 3. 手動初始化（首次）

```bash
pip install yfinance pandas pandas-ta numpy scipy pyarrow
python daily_run.py
```

---

## ⚙️ 引擎擴展

`daily_run.py` 採用引擎插件架構，可獨立替換：

| 引擎檔案 | 功能 |
|----------|------|
| `v4_engine.py` | V4 多因子共振評分 |
| `v12_engine.py` | V12.1 路徑 + EV 計算 |
| `regime_engine.py` | Soft Regime 分類 |
| `feature_engine.py` | 大盤特徵 + 技術指標 |
| `risk_engine.py` | 風控 + 停損計算 |

每個引擎只需實作 `run(**kwargs) → dict` 介面即可被主控制器自動載入。

---

## 📌 版本差異對照

| 版本 | 架構 | 計算位置 | 部署 |
|------|------|----------|------|
| v2.3 | 即時計算 | Streamlit Server | 緩慢 |
| v3.0 | 預計算 + 展示分離 | GitHub Actions | 快速 |

**v3.0 優勢**：
- ✅ App 載入速度提升 10x+（僅讀 JSON）
- ✅ 不依賴 Streamlit Server 算力
- ✅ 歷史快照自動存檔，支援回測驗證
- ✅ 6 個精確時間點計算，無手動觸發
