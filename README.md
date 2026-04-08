# ⚡ 資源法 AI 戰情室 v2.1

## 架構說明

```
stock_app/
├── app.py              # Streamlit 主程式（前端 + 控制層）
├── engine_21.py        # 資源法 2.1 核心引擎（技術指標 + 篩選邏輯）
├── alpha_seeds.json    # V12.1 盤後分析結果（可手動更新）
├── requirements.txt    # 依賴套件
├── start.bat           # Windows 一鍵啟動
└── README.md
```

## 快速啟動

### Windows
```
雙擊 start.bat
```

### Mac / Linux
```bash
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器開啟：http://localhost:8501

## 功能說明

### 雙軌市場支援
- **台股** 🇹🇼：自動判斷 `.TW` / `.TWO`（上市/上櫃）
- **美股** 🇺🇸：直接輸入代號（NVDA、TSLA...）

### 四層數據防火牆
1. **多源交叉比對**：自動偵測上市/上櫃後取得數據
2. **合理性過濾**：剔除異常價格（<1 / >100000）、成交量異常、OHLC 邏輯錯誤
3. **指標單元測試**：檢查 PVO NaN 比率、VRI 範圍、Score 標準差
4. **Streamlit 健康度面板**：Tab 4 即時顯示每檔數據狀態

### 兩階段篩選
- **Stage 1（資源法 2.1）**：Slope Z > 0.5 + VRI 40-75 + PVO 向上勾起
- **Stage 2（V12.1）**：讀取 `alpha_seeds.json`，驗證路徑有效性 + EV 門檻

### AI 整合
- 使用 **Gemini API**（在側欄輸入 Key）
- 點擊「呼叫 Gemini 深度分析」生成大盤摘要 + 標的建議

### 更新 V12.1 種子資料
1. 在您的電腦跑完 V12.1 盤後分析
2. 產出 `alpha_seeds.json`（格式如下）
3. 在 Streamlit 側欄上傳

```json
[
  {"Ticker": "3037", "Path": "Pure", "t_stat": 3.2, "EV_Threshold": 0.07},
  {"Ticker": "NVDA", "Path": "Pure", "t_stat": 4.1, "EV_Threshold": 0.09}
]
```

路徑類型：`Pure`（最強）> `Alive`（有效）> `Flicker`（閃爍，不執行）

## 技術指標說明

| 指標 | 說明 | 健康值 |
|------|------|--------|
| PVO | 快慢量能 EMA 差 | > 10 主力點火 |
| VRI | 上漲成交量/總量比 | 40-75 健康水溫 |
| Slope | 5日收盤線性斜率% | > 0 趨勢翻正 |
| Slope Z | 斜率對60日的標準化 | > 0.5 初步篩選 |
| Score | PVO×0.2 + VRI×0.2 + Slope×0.6 | 越高越佳 |

## 注意事項
- 本系統為分析輔助工具，不構成投資建議
- 所有指標均為統計模型，過去表現不代表未來結果
