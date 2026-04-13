"""
╔══════════════════════════════════════════════════════════════════╗
║  portfolio_manager.py — 資源法 組合管理 & 賣出決策引擎           ║
║  版本：v1.0  相容 V4 + V12.1 訊號                               ║
║                                                                  ║
║  職責：                                                          ║
║  1. BUY  決策：整合 V4 強度 + V12 路徑 + Regime → 是否進場       ║
║  2. SELL 決策：多層出場規則（停利/停損/EV衰退/量能/時間衰減）      ║
║  3. 部位規模：ATR 波動錨定倉位 + Regime 調整係數                  ║
║  4. 換股邏輯：新進場EV > 現有最低EV × 1.20 才置換               ║
║  5. 產生 trade_log：記錄每筆買賣原因（供 Streamlit 顯示）         ║
║  6. 回測介面：可被 backtest_engine.py 逐日調用                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from typing import Optional

log = logging.getLogger("portfolio_manager")

# ──────────────────────────────────────────────────────────────
# 全域參數（可從 config.json 覆蓋）
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # ── 進場門檻 ──
    "min_v4_score":        58.0,    # V4 分數下限
    "min_ev_entry":        0.030,   # V12 EV 下限（3%）
    "min_slope_z_entry":   0.3,     # V4 slope_z 下限
    "min_vri_entry":       35.0,    # VRI 最低（過冷不進）
    "max_vri_entry":       88.0,    # VRI 最高（過熱不追）
    "require_v12_path":    True,    # 必須有V12路徑才進場

    # ── 部位規模 ──
    "max_positions":       5,       # 最大同時持倉數
    "base_capital":        1_000_000.0,   # 基礎資金（元）
    "atr_risk_pct":        0.015,   # 每筆最大風險：資金 × 1.5%
    "max_single_weight":   0.25,    # 單檔最高倉位比例
    "min_single_weight":   0.08,    # 單檔最低倉位比例

    # ── Regime 倉位調整 ──
    "regime_size_mult": {
        "bull":  1.10,
        "range": 1.00,
        "bear":  0.40,
    },

    # ── 停利（多層）──
    "tp1_ratio":           0.20,    # 第一目標：+20%
    "tp2_ratio":           0.35,    # 第二目標：+35%
    "tp1_exit_pct":        0.50,    # TP1 出清 50%
    "trailing_stop_pct":   0.08,    # TP1 達到後，回撤 8% 出場

    # ── 停損 ──
    "hard_stop_pct":       -0.10,   # 硬停損 -10%
    "atr_stop_mult":       1.5,     # 進場時 stop = close - ATR × 1.5

    # ── EV / 時間衰減出場 ──
    "ev_decay_thr":        0.005,   # EV 低於 0.5% → 出場
    "ev_decay_days":       7,       # 幾天後開始檢查衰減
    "ev_decay_pct_thr":    0.35,    # EV 衰減超過 35% 且斜率負
    "ev_accel_pct_thr":    0.20,    # EV 衰減超過 20% 且 slope 加速下滑
    "slope_exit_min":      -0.01,   # Slope 閾值

    # ── 量能出場 ──
    "vol_pvo_exit_thr":    -0.30,   # PVO < -0.30 且持倉 > 3 天
    "vol_pvo_days_min":    3,

    # ── 換股門檻 ──
    "slot_replace_ev_mult":  1.20,  # 新EV 必須 > 最弱現持EV × 1.20

    # ── 路徑TP覆蓋（優先 PATH_DEFS）──
    "path_tp_override": {
        "423": {"tp1": 0.20, "tp2": 0.35},
        "45":  {"tp1": 0.20, "tp2": 0.28},
    },
}

PATH_DEFS = {
    "423": {"tp1": 0.20, "tp2": 0.35, "ev_bear": 0.0096, "ev_range": 0.0834, "ev_bull": 0.0406},
    "45":  {"tp1": 0.20, "tp2": 0.28, "ev_bear": 0.0022, "ev_range": 0.0485, "ev_bull": 0.0432},
}

REGIME_PATH_PREF = {
    "bull":  {"active": "45",  "backup": "423"},
    "range": {"active": "423", "backup": "45"},
    "bear":  {"active": None,  "backup": None},
}


# ──────────────────────────────────────────────────────────────
# 資料結構
# ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    """單一持倉紀錄"""
    symbol:         str
    path:           str             # "423" or "45"
    entry_date:     str
    entry_price:    float
    shares:         int             # 股數（張數×1000）
    capital_used:   float           # 進場資金
    tp1_price:      float
    tp2_price:      float
    stop_price:     float
    ev_entry:       float           # 進場時 EV（小數）
    ev_now:         float           # 當前 EV
    regime_entry:   str
    v4_score_entry: float
    signal_entry:   str
    combo_key:      str
    # 動態更新欄位
    days_held:      int     = 0
    curr_price:     float   = 0.0
    curr_ret_pct:   float   = 0.0
    high_water:     float   = 0.0   # 最高達到價格（用於 trailing stop）
    tp1_hit:        bool    = False  # 已到 TP1
    tp1_sold_pct:   float   = 0.0   # TP1 出清比例
    partial_shares: int     = 0     # 剩餘股數（TP1後）
    exit_signal:    str     = "—"
    buy_reason:     str     = ""
    is_watchlist:   bool    = False


@dataclass
class TradeLog:
    """每筆交易記錄（買或賣）"""
    date:           str
    symbol:         str
    action:         str         # BUY / SELL_TP1 / SELL_TP2 / SELL_STOP / SELL_EV / SELL_TRAIL / SELL_REPLACE
    price:          float
    shares:         int
    capital:        float       # 資金金額
    ret_pct:        float       # 報酬率（賣出時）
    reason:         str         # 詳細原因文字（Streamlit 顯示用）
    path:           str
    ev_at_trade:    float
    regime:         str
    v4_score:       float
    signal:         str
    days_held:      int     = 0


# ──────────────────────────────────────────────────────────────
# 核心：PortfolioManager
# ──────────────────────────────────────────────────────────────

class PortfolioManager:
    """
    組合管理器。
    每個交易日呼叫 process_day(date, v4_data, v12_data, regime) 即可。
    """

    def __init__(self, config: dict = None, storage_dir: str = "storage"):
        self.cfg      = {**DEFAULT_CONFIG, **(config or {})}
        self.storage  = storage_dir
        self.positions: dict[str, Position] = {}   # symbol → Position
        self.trade_log: list[TradeLog]      = []
        self.capital   = self.cfg["base_capital"]  # 可用資金
        self.total_val = self.cfg["base_capital"]  # 組合總值（含持倉）

        # 每日快照列表（用於 Streamlit 和回測）
        self.daily_snapshots: list[dict] = []

        os.makedirs(storage_dir, exist_ok=True)
        self._load_state()

    # ══════════════════════════════════════════════════════════
    # 主入口：每日處理
    # ══════════════════════════════════════════════════════════

    def process_day(self,
                    today: str,
                    v4_data:  dict,
                    v12_data: dict,
                    regime:   dict,
                    price_map: dict = None) -> dict:
        """
        每日主流程：
        1. 更新現有持倉價格
        2. 執行賣出決策
        3. 執行買進決策
        4. 存檔 + 回傳今日摘要
        """
        log.info(f"=== PortfolioManager: {today} ===")

        regime_key = self._regime_key(regime)
        P_bear  = regime.get("bear",  0.33)
        P_range = regime.get("range", 0.34)
        P_bull  = regime.get("bull",  0.33)

        # price_map：{symbol: close_price}，若無則從 v4 TOP20 取
        if price_map is None:
            price_map = {r["symbol"]: r["close"]
                         for r in v4_data.get("top20", [])
                         if r.get("close", 0) > 0}

        # Step 1: 更新持倉市值
        self._update_positions(today, price_map, v4_data, v12_data, P_bear, P_range, P_bull)

        # Step 2: 賣出決策
        sold_today = self._execute_sells(today, regime_key)

        # Step 3: 買進決策
        bought_today = self._execute_buys(today, v4_data, v12_data, regime, regime_key,
                                           P_bear, P_range, P_bull)

        # Step 4: 快照
        snapshot = self._build_snapshot(today, regime_key, sold_today, bought_today)
        self.daily_snapshots.append(snapshot)

        # Step 5: 存檔
        self._save_state(today)

        return snapshot

    # ══════════════════════════════════════════════════════════
    # Step 1: 更新持倉
    # ══════════════════════════════════════════════════════════

    def _update_positions(self, today, price_map, v4_data, v12_data,
                          Pb, Pr, Pu):
        v4_map  = {r["symbol"]: r for r in v4_data.get("top20", [])}
        v12_map = {p["symbol"]: p for p in v12_data.get("positions", [])}

        for sym, pos in self.positions.items():
            curr = price_map.get(sym, pos.curr_price)
            pos.curr_price  = curr
            pos.curr_ret_pct = (curr - pos.entry_price) / (pos.entry_price + 1e-9) * 100
            pos.high_water   = max(pos.high_water, curr)
            pos.days_held   += 1

            # 更新 EV（從 V12 取最新值）
            v12_row = v12_map.get(sym, {})
            if v12_row:
                path = pos.path
                pdef = PATH_DEFS.get(path, {})
                pos.ev_now = (Pb * pdef.get("ev_bear",0) +
                              Pr * pdef.get("ev_range",0) +
                              Pu * pdef.get("ev_bull",0))
            else:
                pos.ev_now = pos.ev_now * 0.97  # 無V12資料時緩慢衰退

            # 更新出場訊號評估
            pos.exit_signal = self._check_exit_signal(pos, v4_map.get(sym, {}))

    # ══════════════════════════════════════════════════════════
    # Step 2: 賣出決策
    # ══════════════════════════════════════════════════════════

    def _execute_sells(self, today: str, regime_key: str) -> list[dict]:
        """
        遍歷持倉，判斷是否觸發出場。
        回傳今日賣出清單（含詳細原因）。
        """
        sold = []
        to_remove = []
        cfg = self.cfg

        for sym, pos in self.positions.items():
            action, reason = None, ""
            curr   = pos.curr_price
            ret    = pos.curr_ret_pct / 100.0
            days   = pos.days_held
            ev_now = pos.ev_now
            ev_ent = pos.ev_entry

            # ── 優先序 (由高到低) ──

            # [S1] 硬停損
            if curr <= pos.stop_price or ret <= cfg["hard_stop_pct"]:
                action = "SELL_STOP"
                reason = (f"硬停損觸發：現價{curr:.1f} ≤ 停損價{pos.stop_price:.1f}"
                          f"（損失 {ret*100:+.1f}%）")

            # [S2] 停利 TP2（全出）
            elif curr >= pos.tp2_price:
                action = "SELL_TP2"
                reason = (f"TP2 達標：現價{curr:.1f} ≥ 目標②{pos.tp2_price:.1f}"
                          f"（獲利 {ret*100:+.1f}%，路徑{pos.path}）")

            # [S3] Trailing Stop（TP1 後回撤）
            elif pos.tp1_hit and (pos.high_water - curr) / (pos.high_water + 1e-9) >= cfg["trailing_stop_pct"]:
                action = "SELL_TRAIL"
                reason = (f"移動停利：高點{pos.high_water:.1f} → 現{curr:.1f}"
                          f"，回撤{(pos.high_water-curr)/pos.high_water*100:.1f}%"
                          f" ≥ {cfg['trailing_stop_pct']*100:.0f}%")

            # [S4] EV 嚴重衰退
            elif ev_now < cfg["ev_decay_thr"]:
                action = "SELL_EV"
                reason = (f"EV 衰退至 {ev_now*100:.2f}%，低於門檻"
                          f" {cfg['ev_decay_thr']*100:.1f}%（已持 {days} 天）")

            # [S5] 時間衰減 + Slope 加速
            elif (days > cfg["ev_decay_days"] and ev_ent > 0 and
                  (ev_ent - ev_now) / ev_ent > cfg["ev_accel_pct_thr"] and
                  pos.exit_signal == "Slope加速出場"):
                action = "SELL_EV"
                reason = (f"EV 加速衰減 {(ev_ent-ev_now)/ev_ent*100:.1f}%"
                          f" + Slope 負值，時間衰減出場（持 {days} 天）")

            # [S6] 時間衰減（EV 衰減超過閾值）
            elif (days > cfg["ev_decay_days"] and ev_ent > 0 and
                  (ev_ent - ev_now) / ev_ent > cfg["ev_decay_pct_thr"]):
                action = "SELL_EV"
                reason = (f"EV 時間衰減 {(ev_ent-ev_now)/ev_ent*100:.1f}%"
                          f" > {cfg['ev_decay_pct_thr']*100:.0f}%（持 {days} 天）")

            # [S7] 量能枯竭
            elif (days > cfg["vol_pvo_days_min"] and
                  pos.exit_signal == "量能枯竭"):
                action = "SELL_EV"
                reason = f"量能枯竭：PVO 持續負值（持 {days} 天）"

            # [S8] 保本出場（TP1 達到後跌回）
            elif pos.tp1_hit and ret < 0.01:
                action = "SELL_TRAIL"
                reason = (f"保本出場：TP1 後現價 {curr:.1f} 回到成本"
                          f"（報酬 {ret*100:+.2f}%）")

            # ── TP1：部分出清（不完全賣出）──
            if not action and not pos.tp1_hit and curr >= pos.tp1_price:
                pos.tp1_hit = True
                pos.high_water = max(pos.high_water, curr)
                shares_sell = int(pos.shares * cfg["tp1_exit_pct"])
                if shares_sell > 0:
                    capital_back = shares_sell * curr
                    self.capital += capital_back
                    pos.shares -= shares_sell
                    pos.partial_shares = shares_sell
                    tlog = TradeLog(
                        date=today, symbol=sym, action="SELL_TP1",
                        price=curr, shares=shares_sell, capital=capital_back,
                        ret_pct=ret * 100,
                        reason=(f"TP1 達標（+{ret*100:.1f}%）：出清 {cfg['tp1_exit_pct']*100:.0f}%"
                                f"，剩餘 {pos.shares} 股繼續持有"),
                        path=pos.path, ev_at_trade=ev_now,
                        regime=regime_key, v4_score=pos.v4_score_entry,
                        signal=pos.signal_entry, days_held=days,
                    )
                    self.trade_log.append(tlog)
                    sold.append({"symbol": sym, "action": "SELL_TP1",
                                 "reason": tlog.reason, "ret_pct": ret*100})
                    log.info(f"  TP1 {sym}: 出清{shares_sell}股@{curr:.1f}，報酬{ret*100:+.1f}%")

            # 執行完全賣出
            if action:
                shares_sell = pos.shares
                capital_back = shares_sell * curr
                self.capital += capital_back
                ret_pct = (curr - pos.entry_price) / pos.entry_price * 100

                tlog = TradeLog(
                    date=today, symbol=sym, action=action,
                    price=curr, shares=shares_sell, capital=capital_back,
                    ret_pct=ret_pct, reason=reason,
                    path=pos.path, ev_at_trade=ev_now,
                    regime=regime_key, v4_score=pos.v4_score_entry,
                    signal=pos.signal_entry, days_held=days,
                )
                self.trade_log.append(tlog)
                sold.append({"symbol": sym, "action": action,
                             "reason": reason, "ret_pct": ret_pct,
                             "days": days, "ev_now": ev_now})
                to_remove.append(sym)
                log.info(f"  [{action}] {sym} @{curr:.1f}  ret={ret_pct:+.1f}%  {reason[:40]}")

        for sym in to_remove:
            del self.positions[sym]

        return sold

    # ══════════════════════════════════════════════════════════
    # Step 3: 買進決策
    # ══════════════════════════════════════════════════════════

    def _execute_buys(self, today, v4_data, v12_data, regime, regime_key,
                      Pb, Pr, Pu) -> list[dict]:
        """
        從 V4 TOP30 + V12 部位候選中，篩選進場標的。
        """
        cfg = self.cfg
        bought = []

        top30     = v4_data.get("top20", [])   # 即便 key 是 top20 也可能有30筆
        v12_pos   = {p["symbol"]: p for p in v12_data.get("positions", [])}
        v4_map    = {r["symbol"]: r for r in top30}

        # 依 V4 score 排序候選
        candidates = sorted(
            [r for r in top30
             if r.get("action") in ("強力買進", "買進")
             and r.get("score", 0) >= cfg["min_v4_score"]
             and r.get("slope_z", 0) >= cfg["min_slope_z_entry"]
             and cfg["min_vri_entry"] <= r.get("vri", 50) <= cfg["max_vri_entry"]
             and r["symbol"] not in self.positions],
            key=lambda x: x.get("score", 0), reverse=True,
        )

        regime_mult = cfg["regime_size_mult"].get(regime_key, 1.0)
        pref = REGIME_PATH_PREF.get(regime_key, {"active": "423", "backup": "45"})

        for cand in candidates:
            # 已滿倉：嘗試換股（新EV > 最弱 × 1.20）
            if len(self.positions) >= cfg["max_positions"]:
                if not self._try_replace(cand, v12_pos, Pb, Pr, Pu, today, regime_key, regime_mult):
                    continue
                else:
                    # 換股成功後重新走進場流程
                    pass

            sym   = cand["symbol"]
            v4r   = cand
            v12r  = v12_pos.get(sym, {})

            # ── V12 路徑確認 ──
            path = v12r.get("path", pref["active"] or "423")
            ev_raw = v12r.get("ev", 0.0)
            ev_float = ev_raw / 100.0 if ev_raw > 1.0 else ev_raw

            # 若 V12 無此股，自行計算 EV
            if not v12r:
                pdef = PATH_DEFS.get(path, {})
                ev_float = (Pb * pdef.get("ev_bear",0) +
                            Pr * pdef.get("ev_range",0) +
                            Pu * pdef.get("ev_bull",0))

            if cfg["require_v12_path"] and not v12r:
                log.debug(f"  跳過 {sym}：V12 無路徑確認")
                continue

            if ev_float < cfg["min_ev_entry"]:
                log.debug(f"  跳過 {sym}：EV={ev_float*100:.2f}% < {cfg['min_ev_entry']*100:.1f}%")
                continue

            # ── 計算進場參數 ──
            close   = float(v4r.get("close", 0))
            atr     = float(v4r.get("atr", close * 0.02))
            if close <= 0:
                continue

            pdef    = PATH_DEFS.get(path, {})
            tp1_px  = round(close * (1 + pdef.get("tp1", 0.20)), 1)
            tp2_px  = round(close * (1 + pdef.get("tp2", 0.28)), 1)
            stop_px = round(close - atr * cfg["atr_stop_mult"], 1)

            # ── 部位規模（ATR 錨定）──
            risk_cap  = self.capital * cfg["atr_risk_pct"]
            risk_per  = close - stop_px
            if risk_per <= 0:
                risk_per = atr
            shares_by_risk = int(risk_cap / (risk_per + 1e-9) / 1000) * 1000
            shares_by_weight = int(
                self.capital * min(cfg["max_single_weight"] * regime_mult,
                                   cfg["max_single_weight"])
                / close / 1000
            ) * 1000
            shares = min(shares_by_risk, shares_by_weight)
            shares = max(shares, 1000)  # 至少 1 張

            capital_need = shares * close
            if capital_need > self.capital * 0.95:
                shares = int(self.capital * 0.90 / close / 1000) * 1000
                capital_need = shares * close

            if shares <= 0 or capital_need <= 0:
                log.debug(f"  跳過 {sym}：資金不足")
                continue

            # ── 買進原因文字（Streamlit 顯示用）──
            buy_reason = self._build_buy_reason(v4r, v12r, path, ev_float, regime_key, regime)

            # ── 建立持倉 ──
            pos = Position(
                symbol=sym, path=path,
                entry_date=today, entry_price=close,
                shares=shares, capital_used=capital_need,
                tp1_price=tp1_px, tp2_price=tp2_px, stop_price=stop_px,
                ev_entry=ev_float, ev_now=ev_float,
                regime_entry=regime_key,
                v4_score_entry=v4r.get("score", 0),
                signal_entry=v4r.get("signal", "—"),
                combo_key=v4r.get("combo_key", "—"),
                curr_price=close, high_water=close,
                buy_reason=buy_reason,
                is_watchlist=v4r.get("is_watchlist", False),
            )
            self.positions[sym] = pos
            self.capital -= capital_need

            tlog = TradeLog(
                date=today, symbol=sym, action="BUY",
                price=close, shares=shares, capital=capital_need,
                ret_pct=0.0, reason=buy_reason,
                path=path, ev_at_trade=ev_float,
                regime=regime_key, v4_score=v4r.get("score", 0),
                signal=v4r.get("signal", "—"),
            )
            self.trade_log.append(tlog)
            bought.append({"symbol": sym, "reason": buy_reason,
                           "path": path, "ev": ev_float,
                           "price": close, "shares": shares})
            log.info(f"  [BUY] {sym} @{close:.1f}  {shares}股  EV={ev_float*100:.2f}%  路徑{path}")

            if len(self.positions) >= cfg["max_positions"]:
                break

        return bought

    # ══════════════════════════════════════════════════════════
    # 換股邏輯
    # ══════════════════════════════════════════════════════════

    def _try_replace(self, cand, v12_pos, Pb, Pr, Pu,
                     today, regime_key, regime_mult) -> bool:
        """若新候選EV優勢足夠，踢出最弱部位，回傳是否成功騰出空間。"""
        cfg = self.cfg
        if not self.positions:
            return False

        # 找最弱部位
        weakest_sym = min(self.positions, key=lambda s: self.positions[s].ev_now)
        weakest_ev  = self.positions[weakest_sym].ev_now

        sym  = cand["symbol"]
        v12r = v12_pos.get(sym, {})
        path = v12r.get("path", "423")
        pdef = PATH_DEFS.get(path, {})
        new_ev = (Pb * pdef.get("ev_bear",0) +
                  Pr * pdef.get("ev_range",0) +
                  Pu * pdef.get("ev_bull",0))

        if new_ev < weakest_ev * cfg["slot_replace_ev_mult"]:
            return False

        # 執行踢出
        weak_pos = self.positions[weakest_sym]
        curr  = weak_pos.curr_price
        ret   = (curr - weak_pos.entry_price) / weak_pos.entry_price * 100
        self.capital += weak_pos.shares * curr

        reason = (f"換股置換：新標的{sym} EV={new_ev*100:.2f}% "
                  f"> 現最弱{weakest_sym} EV={weakest_ev*100:.2f}% × {cfg['slot_replace_ev_mult']:.1f}")
        tlog = TradeLog(
            date=today, symbol=weakest_sym, action="SELL_REPLACE",
            price=curr, shares=weak_pos.shares, capital=weak_pos.shares * curr,
            ret_pct=ret, reason=reason,
            path=weak_pos.path, ev_at_trade=weak_pos.ev_now,
            regime=regime_key, v4_score=weak_pos.v4_score_entry,
            signal=weak_pos.signal_entry, days_held=weak_pos.days_held,
        )
        self.trade_log.append(tlog)
        del self.positions[weakest_sym]
        log.info(f"  [REPLACE] 踢出 {weakest_sym}（EV={weakest_ev*100:.2f}%），迎入 {sym}")
        return True

    # ══════════════════════════════════════════════════════════
    # 出場訊號評估（純判斷，不執行）
    # ══════════════════════════════════════════════════════════

    def _check_exit_signal(self, pos: Position, v4_row: dict) -> str:
        cfg = self.cfg
        curr = pos.curr_price
        ret  = pos.curr_ret_pct / 100.0

        if curr <= pos.stop_price or ret <= cfg["hard_stop_pct"]:
            return "硬停損"
        if pos.tp1_hit and (pos.high_water - curr) / (pos.high_water + 1e-9) >= cfg["trailing_stop_pct"]:
            return "移動停利"
        if pos.tp1_hit and ret < 0.01:
            return "保本出場"
        if pos.ev_now < cfg["ev_decay_thr"]:
            return "EV衰退"
        if pos.days_held > cfg["ev_decay_days"] and pos.ev_entry > 0:
            drop = (pos.ev_entry - pos.ev_now) / pos.ev_entry
            slope = v4_row.get("slope", 0.0)
            if drop > cfg["ev_accel_pct_thr"] and slope < cfg["slope_exit_min"]:
                return "Slope加速出場"
            if drop > cfg["ev_decay_pct_thr"]:
                return "時間衰減"
        pvo = v4_row.get("pvo", 0.0)
        if pos.days_held > cfg["vol_pvo_days_min"] and pvo < cfg["vol_pvo_exit_thr"]:
            return "量能枯竭"
        return "—"

    # ══════════════════════════════════════════════════════════
    # 買進原因文字生成（Streamlit 顯示用）
    # ══════════════════════════════════════════════════════════

    def _build_buy_reason(self, v4r, v12r, path, ev, regime_key, regime) -> str:
        """
        生成結構化的買進原因文字，供 Streamlit 展示與回測記錄。
        """
        score    = v4r.get("score", 0)
        pvo      = v4r.get("pvo", 0)
        vri      = v4r.get("vri", 0)
        slope_z  = v4r.get("slope_z", 0)
        signal   = v4r.get("signal", "—")
        combo    = v4r.get("combo_key", "—")
        label    = regime.get("label", "—")
        a_path   = regime.get("active_path", path)

        # 訊號說明
        signal_desc = {
            "三合一(ABC)": "量價強勢三訊號共振（A量縮冷卻+B火爆放量+C動能延續）",
            "二合一(AB)":  "雙訊號共振（量能爆發+動能突破）",
            "二合一(AC)":  "雙訊號共振（趨勢確認+量能延續）",
            "二合一(BC)":  "雙訊號共振（火爆放量+量能延續）",
            "單一(A)":     "量縮冷卻後反彈訊號",
            "單一(B)":     "火爆放量突破訊號",
            "單一(C)":     "量能動能延續訊號",
            "基準-強勢":   "Slope 斜率強勢基準進場",
        }.get(combo, signal)

        # Regime 路徑說明
        path_desc = {
            ("bull", "45"):  "牛市主路徑 45（Y4→Y5）適合趨勢追蹤",
            ("bull", "423"): "牛市備援路徑 423（Y4→Y2→Y3）",
            ("range","423"): "震盪主路徑 423（Y4→Y2→Y3）統計套利優先",
            ("range","45"):  "震盪備援路徑 45（Y4→Y5）",
        }.get((regime_key, path), f"路徑 {path}")

        reason = (
            f"【V4】Score={score:.1f}｜{signal}｜"
            f"PVO={pvo:+.2f}（量能）VRI={vri:.1f}（熱度）Slope_Z={slope_z:+.2f}（趨勢）"
            f"｜{signal_desc}"
            f"｜【V12】路徑{path} EV={ev*100:.2f}%｜{path_desc}"
            f"｜【Regime】{label}（{regime_key}）active_path={a_path}"
        )
        return reason

    # ══════════════════════════════════════════════════════════
    # 快照生成（供 Streamlit + 回測）
    # ══════════════════════════════════════════════════════════

    def _build_snapshot(self, today, regime_key, sold, bought) -> dict:
        """建立今日完整快照。"""
        pos_list = []
        total_market_val = self.capital
        for sym, pos in self.positions.items():
            val = pos.shares * pos.curr_price
            total_market_val += val
            pos_list.append({
                "symbol":        sym,
                "path":          pos.path,
                "entry_date":    pos.entry_date,
                "entry_price":   pos.entry_price,
                "curr_price":    pos.curr_price,
                "shares":        pos.shares,
                "market_val":    round(val, 0),
                "curr_ret_pct":  round(pos.curr_ret_pct, 2),
                "tp1_price":     pos.tp1_price,
                "tp2_price":     pos.tp2_price,
                "stop_price":    pos.stop_price,
                "ev_entry":      round(pos.ev_entry * 100, 2),
                "ev_now":        round(pos.ev_now * 100, 2),
                "days_held":     pos.days_held,
                "exit_signal":   pos.exit_signal,
                "buy_reason":    pos.buy_reason,
                "tp1_hit":       pos.tp1_hit,
                "high_water":    pos.high_water,
                "is_watchlist":  pos.is_watchlist,
            })

        daily_pnl = 0.0
        for t in self.trade_log:
            if t.date == today and t.action.startswith("SELL"):
                daily_pnl += t.ret_pct * t.capital / 100.0

        return {
            "date":           today,
            "regime":         regime_key,
            "positions":      pos_list,
            "n_positions":    len(pos_list),
            "available_cap":  round(self.capital, 0),
            "total_val":      round(total_market_val, 0),
            "sold_today":     sold,
            "bought_today":   bought,
            "daily_pnl":      round(daily_pnl, 0),
            "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    # ══════════════════════════════════════════════════════════
    # 工具函式
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _regime_key(regime: dict) -> str:
        label = str(regime.get("label", "震盪")).lower()
        bull  = regime.get("bull",  0.0)
        bear  = regime.get("bear",  0.0)
        if "牛" in label or "bull" in label or bull > 0.55: return "bull"
        if "熊" in label or "bear" in label or bear > 0.55: return "bear"
        return "range"

    # ══════════════════════════════════════════════════════════
    # 存檔 / 載入
    # ══════════════════════════════════════════════════════════

    def _save_state(self, today: str):
        """儲存 positions.csv + trades.csv + 今日 JSON 快照。"""
        s = self.storage

        # positions.csv（完整持倉）
        pos_path = os.path.join(s, "positions.csv")
        rows = [asdict(p) for p in self.positions.values()]
        if rows:
            keys = list(rows[0].keys())
            with open(pos_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(rows)

        # trades.csv（交易紀錄追加）
        trade_path = os.path.join(s, "trades.csv")
        today_trades = [asdict(t) for t in self.trade_log if t.date == today]
        if today_trades:
            keys   = list(today_trades[0].keys())
            exists = os.path.exists(trade_path)
            with open(trade_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                if not exists: w.writeheader()
                w.writerows(today_trades)

        # daily JSON 快照
        snap_dir = os.path.join(s, "daily_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap_path = os.path.join(snap_dir, f"snapshot_{today}.json")
        if self.daily_snapshots:
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(self.daily_snapshots[-1], f, ensure_ascii=False, indent=2, default=str)

        # 最新快照
        latest = os.path.join(s, "portfolio_latest.json")
        if self.daily_snapshots:
            with open(latest, "w", encoding="utf-8") as f:
                json.dump(self.daily_snapshots[-1], f, ensure_ascii=False, indent=2, default=str)

        log.info(f"  ✅ 存檔完成：{len(self.positions)} 持倉，{len(today_trades)} 筆今日交易")

    def _load_state(self):
        """從 positions.csv 恢復持倉（程式重啟時用）。"""
        pos_path = os.path.join(self.storage, "positions.csv")
        if not os.path.exists(pos_path):
            return
        try:
            with open(pos_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 型別轉換
                    for float_col in ["entry_price","shares","capital_used","tp1_price",
                                      "tp2_price","stop_price","ev_entry","ev_now",
                                      "v4_score_entry","curr_price","curr_ret_pct","high_water"]:
                        row[float_col] = float(row.get(float_col, 0) or 0)
                    for int_col in ["days_held","shares","partial_shares"]:
                        row[int_col] = int(float(row.get(int_col, 0) or 0))
                    for bool_col in ["tp1_hit","is_watchlist"]:
                        row[bool_col] = str(row.get(bool_col,"False")).lower() == "true"
                    pos = Position(**{k: v for k, v in row.items() if k in Position.__dataclass_fields__})
                    self.positions[pos.symbol] = pos
            log.info(f"  載入既有持倉：{len(self.positions)} 檔")
        except Exception as e:
            log.warning(f"  持倉載入失敗：{e}")

    # ══════════════════════════════════════════════════════════
    # 回測輔助介面
    # ══════════════════════════════════════════════════════════

    def get_performance_summary(self) -> dict:
        """
        生成績效摘要（供 backtest_engine.py 使用）。
        """
        sell_trades = [t for t in self.trade_log if t.action.startswith("SELL")]
        if not sell_trades:
            return {"message": "尚無已結束交易"}

        rets       = [t.ret_pct for t in sell_trades]
        wins       = [r for r in rets if r > 0]
        losses     = [r for r in rets if r <= 0]
        win_rate   = len(wins) / len(rets) * 100 if rets else 0
        avg_win    = sum(wins)  / len(wins)   if wins   else 0
        avg_loss   = sum(losses)/ len(losses) if losses else 0
        pl_ratio   = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        total_pnl  = sum(t.ret_pct * t.capital / 100 for t in sell_trades)
        total_ret  = total_pnl / self.cfg["base_capital"] * 100

        action_dist = {}
        for t in sell_trades:
            action_dist[t.action] = action_dist.get(t.action, 0) + 1

        return {
            "total_trades":  len(sell_trades),
            "win_rate_pct":  round(win_rate,  2),
            "avg_win_pct":   round(avg_win,   2),
            "avg_loss_pct":  round(avg_loss,  2),
            "pl_ratio":      round(pl_ratio,  2),
            "total_pnl":     round(total_pnl, 0),
            "total_ret_pct": round(total_ret, 2),
            "exit_breakdown": action_dist,
            "n_positions_now": len(self.positions),
            "available_cap": round(self.capital, 0),
        }


# ══════════════════════════════════════════════════════════════
# 快捷函式（供 daily_run.py 調用）
# ══════════════════════════════════════════════════════════════

def run(today: str, v4_data: dict, v12_data: dict, regime: dict,
        storage_dir: str = "storage", price_map: dict = None) -> dict:
    """
    daily_run.py 的調用入口。
    自動恢復持倉 → 執行今日決策 → 存檔 → 回傳快照。
    """
    pm = PortfolioManager(storage_dir=storage_dir)
    return pm.process_day(today, v4_data, v12_data, regime, price_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ── 快速 Demo 測試 ──
    mock_v4 = {
        "top20": [
            {"symbol":"2330","rank":1,"score":78.5,"pvo":12.3,"vri":62.0,
             "slope_z":1.8,"slope":0.005,"action":"強力買進",
             "signal":"三合一(ABC)","combo_key":"三合一(ABC)","close":950.0,
             "atr":18.5,"pos_weight":0.28,"regime":"range","is_watchlist":True,
             "tp1_price":1140.0,"stop_price":897.0},
            {"symbol":"2454","rank":2,"score":71.2,"pvo":8.1,"vri":55.0,
             "slope_z":1.2,"slope":0.003,"action":"買進",
             "signal":"單一(B)","combo_key":"單一(B)","close":420.0,
             "atr":8.2,"pos_weight":0.18,"regime":"range","is_watchlist":False,
             "tp1_price":504.0,"stop_price":407.7},
        ]
    }
    mock_v12 = {
        "positions": [
            {"symbol":"2330","path":"45","ev":5.83,"ev_tier":"⭐核心",
             "action":"進場","exit_signal":"—","quality":"Pure",
             "days_held":0,"curr_ret_pct":0.0,"entry_price":950.0,
             "tp1_price":1140.0,"tp2_price":1282.0,"stop_price":897.0,
             "close":950.0,"regime":"range"},
        ]
    }
    mock_regime = {
        "bear":0.00,"range":0.30,"bull":0.70,"label":"牛市",
        "active_strategy":"bull","active_path":"45","backup_path":"423",
    }

    pm = PortfolioManager(storage_dir="/tmp/pm_test")
    snap = pm.process_day("2026-04-13", mock_v4, mock_v12, mock_regime)

    print("\n=== 今日快照 ===")
    print(f"持倉數: {snap['n_positions']}")
    print(f"可用資金: {snap['available_cap']:,.0f}")
    for b in snap["bought_today"]:
        print(f"[BUY] {b['symbol']} - {b['reason'][:80]}...")
    for s in snap["sold_today"]:
        print(f"[SELL] {s['symbol']} - {s['reason'][:80]}...")
    print("\n績效:", pm.get_performance_summary())
