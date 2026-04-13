"""
╔══════════════════════════════════════════════════════════════════╗
║  backtest_engine.py — 資源法 歷史回測引擎                        ║
║  版本：v1.0                                                      ║
║                                                                  ║
║  架構：                                                          ║
║  1. 從 storage/daily_snapshots/ 讀取每日 JSON（或重新計算）       ║
║  2. 逐日呼叫 PortfolioManager.process_day()                      ║
║  3. 計算回測績效指標：勝率/Sharpe/最大回撤/CAGR                   ║
║  4. 輸出 backtest_result.json + trades.csv 供 Streamlit 展示     ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, date, timedelta
from typing import Optional

log = logging.getLogger("backtest_engine")

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

from portfolio_manager import PortfolioManager, DEFAULT_CONFIG


# ──────────────────────────────────────────────────────────────
# 回測引擎
# ──────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    逐日回測引擎。
    讀取 storage/daily_snapshots/{date}_v4.json + v12.json + regime.json
    呼叫 PortfolioManager 模擬交易，產生績效報告。
    """

    def __init__(self,
                 storage_dir: str = "storage",
                 start_date: str  = None,
                 end_date: str    = None,
                 initial_capital: float = 1_000_000.0,
                 config: dict     = None):
        self.storage        = storage_dir
        self.start_date     = start_date
        self.end_date       = end_date
        self.initial_capital = initial_capital
        self.cfg            = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg["base_capital"] = initial_capital

        self.equity_curve:  list[dict] = []   # 每日總值
        self.trade_records: list[dict] = []   # 所有交易

    # ══════════════════════════════════════════════════════════
    # 主回測流程
    # ══════════════════════════════════════════════════════════

    def run(self) -> dict:
        """
        執行回測，回傳績效摘要。
        """
        log.info(f"回測啟動：{self.start_date} → {self.end_date}")

        snap_dir  = os.path.join(self.storage, "v4")
        v12_dir   = os.path.join(self.storage, "v12")
        reg_dir   = os.path.join(self.storage, "regime")

        # 取得所有可用日期
        all_dates = self._get_available_dates(snap_dir)
        if not all_dates:
            log.error("無可用的歷史快照資料")
            return {}

        if self.start_date:
            all_dates = [d for d in all_dates if d >= self.start_date]
        if self.end_date:
            all_dates = [d for d in all_dates if d <= self.end_date]

        log.info(f"  共 {len(all_dates)} 個交易日")

        # 初始化組合管理器（回測模式，使用臨時目錄）
        bt_storage = os.path.join(self.storage, "_backtest_tmp")
        os.makedirs(bt_storage, exist_ok=True)
        pm = PortfolioManager(config=self.cfg, storage_dir=bt_storage)

        for day in sorted(all_dates):
            v4_data  = self._load_json(os.path.join(snap_dir, "v4_latest.json"))   or {}
            v12_data = self._load_json(os.path.join(v12_dir, "v12_latest.json"))   or {}
            regime   = self._load_json(os.path.join(reg_dir, "regime_state.json")) or self._default_regime()

            # 嘗試載入當日快照（若有歸檔）
            day_v4 = self._load_json(os.path.join(snap_dir, f"v4_{day}.json"))
            if day_v4:
                v4_data = day_v4
            day_v12 = self._load_json(os.path.join(v12_dir, f"v12_{day}.json"))
            if day_v12:
                v12_data = day_v12
            day_reg = self._load_json(os.path.join(reg_dir, f"regime_{day}.json"))
            if day_reg:
                regime = day_reg

            if not v4_data:
                log.debug(f"  {day}: 無V4資料，跳過")
                continue

            snap = pm.process_day(day, v4_data, v12_data, regime)
            self.equity_curve.append({
                "date":      day,
                "total_val": snap["total_val"],
                "n_pos":     snap["n_positions"],
                "daily_pnl": snap["daily_pnl"],
            })
            self.trade_records.extend(snap.get("bought_today", []))
            self.trade_records.extend(snap.get("sold_today", []))

        # 計算績效
        result = self._calc_performance(pm)
        result["equity_curve"]  = self.equity_curve
        result["trade_records"] = self.trade_records

        # 儲存結果
        out_path = os.path.join(self.storage, "backtest_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"  ✅ 回測完成，結果存至 {out_path}")
        return result

    # ══════════════════════════════════════════════════════════
    # 績效計算
    # ══════════════════════════════════════════════════════════

    def _calc_performance(self, pm: PortfolioManager) -> dict:
        if not self.equity_curve or np is None:
            return pm.get_performance_summary()

        vals   = [e["total_val"] for e in self.equity_curve]
        rets_d = [(vals[i] - vals[i-1]) / (vals[i-1] + 1e-9)
                  for i in range(1, len(vals))]

        total_ret = (vals[-1] - self.initial_capital) / self.initial_capital * 100

        # 最大回撤
        peak   = vals[0]; max_dd = 0.0
        for v in vals:
            if v > peak: peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd: max_dd = dd

        # 年化報酬
        n_days = len(vals)
        cagr   = ((vals[-1] / self.initial_capital) ** (252 / max(n_days, 1)) - 1) * 100

        # Sharpe（年化）
        if rets_d:
            mu_d  = np.mean(rets_d)
            sd_d  = np.std(rets_d) + 1e-9
            sharpe = mu_d / sd_d * math.sqrt(252)
        else:
            sharpe = 0.0

        base = pm.get_performance_summary()
        base.update({
            "backtest_start":    self.equity_curve[0]["date"] if self.equity_curve else "—",
            "backtest_end":      self.equity_curve[-1]["date"] if self.equity_curve else "—",
            "n_trading_days":    n_days,
            "initial_capital":   self.initial_capital,
            "final_val":         round(vals[-1], 0) if vals else self.initial_capital,
            "total_ret_pct":     round(total_ret, 2),
            "cagr_pct":          round(cagr, 2),
            "max_drawdown_pct":  round(max_dd, 2),
            "sharpe":            round(sharpe, 3),
        })
        return base

    # ══════════════════════════════════════════════════════════
    # 工具函式
    # ══════════════════════════════════════════════════════════

    def _get_available_dates(self, snap_dir: str) -> list[str]:
        """取得 v4_YYYY-MM-DD.json 格式的歸檔日期清單。"""
        if not os.path.exists(snap_dir):
            return []
        dates = []
        for fn in os.listdir(snap_dir):
            if fn.startswith("v4_") and fn.endswith(".json"):
                d = fn[3:-5]
                try:
                    datetime.strptime(d, "%Y-%m-%d")
                    dates.append(d)
                except ValueError:
                    pass
        return sorted(dates)

    @staticmethod
    def _load_json(path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _default_regime() -> dict:
        return {"bear":0.33,"range":0.34,"bull":0.33,"label":"震盪",
                "active_path":"423","backup_path":"45"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    bt = BacktestEngine(
        storage_dir="storage",
        start_date="2025-01-01",
        end_date="2026-04-13",
        initial_capital=1_000_000.0,
    )
    result = bt.run()
    if result:
        print(f"\n回測結果摘要：")
        for k, v in result.items():
            if k not in ("equity_curve","trade_records","exit_breakdown"):
                print(f"  {k}: {v}")
