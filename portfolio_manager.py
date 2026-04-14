"""
╔══════════════════════════════════════════════════════════════╗
║  portfolio_manager.py  —  V412 持倉管理 & 賣出決策中樞       ║
║                                                              ║
║  設計哲學：                                                  ║
║  ★ 讀取 GitHub storage/ 的 JSON 快照                         ║
║  ★ V4 出場層（結構破壞）優先，任一觸發即出                    ║
║  ★ V12 出場層（EV/量能/時間）補充，任一觸發即出               ║
║  ★ 所有決策寫入 trade_history.json 以供 Streamlit 展示       ║
║                                                              ║
║  使用方式：                                                  ║
║    python portfolio_manager.py              # 執行賣出決策   ║
║    python portfolio_manager.py --backtest   # 讀歷史回測     ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("portfolio_manager")

# ─────────────────────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
STORAGE_DIR = ROOT_DIR / "storage"
V4_DIR      = STORAGE_DIR / "v4"
V12_DIR     = STORAGE_DIR / "v12"
REGIME_DIR  = STORAGE_DIR / "regime"
LOGS_DIR    = STORAGE_DIR / "logs"
POSITIONS_FILE     = STORAGE_DIR / "positions.json"
TRADE_HISTORY_FILE = LOGS_DIR / "trade_history.json"

for d in [STORAGE_DIR, V4_DIR, V12_DIR, REGIME_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TODAY_STR = date.today().isoformat()

# ─────────────────────────────────────────────────────────────
# ★ 閾值參數（對齊 5Y-3W-V12_1.py）
# ─────────────────────────────────────────────────────────────

# V4 出場閾值
V4_PVO_FLOW_THR       = 0.0
V4_VRI_DELTA_CUTOFF   = -4.0
V4_COND_STOP_PCT      = -0.05   # 條件式止損
V4_HARD_STOP_PCT      = -0.08   # 硬止損
V4_HOLD_BUFFER_DAYS   = 5       # 保護期（結構破壞不觸發）
V4_STRUCT_BREAK_PVO   = -2.0    # SB-PVO轉負閾值
V4_VRI_ACUTE_THR      = 60.0    # VRI急退需超過此值才算熱
V4_PARTIAL_EXIT_RET   = 0.06    # 分批止盈觸發
V4_PARTIAL_EXIT_PCT   = 0.50    # 分批出場比例

# V12 出場閾值
V12_EV_DECAY_PCT      = 0.35    # EV衰退幅度觸發出場
V12_EV_ACCEL_PCT      = 0.20    # EV快速衰退（配合slope）
V12_EV_DECAY_SLOPE    = -0.01   # 配合EV衰退的slope門檻
V12_EV_DEAD_THR       = 0.005   # EV趨近0直接出場
V12_PVO_DECAY_THR     = -0.30   # PVO衰減觸發量能枯竭
V12_VRI_COLD_THR      = 0.70    # VRI冷卻閾值（量能枯竭判斷）
V12_VRI_HOT_THR       = 2.50    # VRI過熱停利①
V12_VRI_VERY_HOT      = 3.00    # VRI極熱全倉出場
V12_PVO_EXIT_DAYS     = 3
V12_PVO_EXIT_CONSEC   = 2
V12_TIME_DECAY_DAYS   = 7
V12_TIME_EXIT_T1_DAYS = 10
V12_TIME_EXIT_T1_RET  = 0.03    # T1出場：持超過10天且報酬<3%
V12_TIME_EXIT_T2_DAYS = 20
V12_TIME_EXIT_T2_RET  = 0.10    # T2出場：持超過20天且報酬<10%
V12_MAX_HOLD_DAYS     = 40
V12_PROTECT_DAYS      = 10
V12_PROTECT_DAYS_EARLY= 5
V12_PROTECT_LOSS_EARLY= -0.06
V12_PROTECT_LOSS_LATE = -0.08
V12_HARD_STOP_PCT     = -0.10   # V12硬停損
V12_PROFIT_LOCK_THR   = 0.10    # 獲利保全觸發
V12_PROFIT_LOCK_FLOOR = 0.01    # 保本線

SELL_COST = 0.001425 + 0.003
SLIPPAGE  = 0.002


# ─────────────────────────────────────────────────────────────
# ★ 資料結構
# ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    """單一持倉狀態"""
    symbol:         str
    entry_price:    float
    entry_date:     str            # ISO date string
    pos_weight:     float = 0.20
    remain_weight:  float = 0.20

    # V4 資訊
    combo_key:      str   = "基準-持有"
    signal_label:   str   = "─"
    pvo_entry:      float = 0.0
    vri_entry:      float = 50.0
    mkt_state:      str   = "neutral"

    # V12 資訊
    v12_path:       str   = "─"
    ev_soft_entry:  float = 0.0
    ev_soft_cur:    float = 0.0
    tp1_px:         float = 0.0
    tp2_px:         float = 0.0

    # 狀態旗標
    tp1_hit:        bool  = False
    partial_sold:   bool  = False
    protect_over:   bool  = False
    profit_locked:  bool  = False
    profit_lock_px: float = 0.0
    addon_done:     bool  = False

    # 追蹤計數
    weak_window:    list  = field(default_factory=list)
    pvo_exit_count: int   = 0
    chandelier_stop:float = 0.0
    stop_price:     float = 0.0

    entry_source:   str   = "V4"
    is_addon:       bool  = False

    def days_held(self) -> int:
        try:
            d = date.fromisoformat(self.entry_date)
            return (date.today() - d).days
        except Exception:
            return 0

    def current_return(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class SellDecision:
    """賣出決策結果"""
    symbol:       str
    action:       str          # "SELL" | "HOLD" | "PARTIAL_SELL"
    reason:       str          # 出場原因文字
    layer:        str          # "V4" | "V12" | "NONE"
    exit_type:    str          # 對齊 trade_log exit_type
    curr_ret:     float        # 當前報酬（含成本前）
    net_ret:      float        # 扣除成本後報酬
    days_held:    int
    current_price:float
    exit_score:   int          # 0-5 出場緊急度分數
    combo_key:    str = "─"
    v12_path:     str = "─"
    ev_soft:      float = 0.0
    timestamp:    str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────
# ★ JSON 快照讀取
# ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"讀取 {path} 失敗: {e}")
        return {}


def load_v4_snapshot() -> dict:
    return _load_json(V4_DIR / "v4_latest.json")


def load_v12_snapshot() -> dict:
    return _load_json(V12_DIR / "v12_latest.json")


def load_regime_snapshot() -> dict:
    return _load_json(REGIME_DIR / "regime_latest.json")


def load_positions() -> list[Position]:
    """從 positions.json 讀取持倉（portfolio_manager 自行維護）"""
    raw = _load_json(POSITIONS_FILE)
    positions = []
    for item in raw.get("positions", []):
        try:
            pos = Position(**{k: item[k] for k in Position.__dataclass_fields__ if k in item})
            positions.append(pos)
        except Exception as e:
            log.warning(f"持倉格式錯誤 {item.get('symbol','?')}: {e}")
    return positions


def save_positions(positions: list[Position]):
    data = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(positions),
        "positions": [asdict(p) for p in positions],
    }
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"✅ 持倉已儲存：{len(positions)} 筆 → {POSITIONS_FILE}")


# ─────────────────────────────────────────────────────────────
# ★ V4 出場層判斷函數
# ─────────────────────────────────────────────────────────────

def check_v4_structural_break(
    pos: Position,
    pvo_now: float,
    vri_now: float,
    vri_delta_now: float,
    mkt_state: str,
) -> tuple[bool, str]:
    """
    V4 結構破壞（Structural Break）檢查。
    回傳 (是否觸發, 原因文字)。
    """
    if pos.days_held() < V4_HOLD_BUFFER_DAYS:
        return False, ""

    # SB-PVO轉負：資金持續流出
    if pvo_now < V4_STRUCT_BREAK_PVO and pos.pvo_entry >= V4_PVO_FLOW_THR:
        return True, "SB-PVO轉負"

    # SB-VRI急退：熱度驟降（VRI高位時才觸發）
    if vri_delta_now < V4_VRI_DELTA_CUTOFF and vri_now > V4_VRI_ACUTE_THR:
        return True, "SB-VRI急退"

    # SB-大盤轉空：環境惡化
    if mkt_state == "bear" and pos.mkt_state != "bear":
        return True, "SB-大盤轉空"

    return False, ""


def check_v4_stop_loss(
    pos: Position,
    curr_ret: float,
    pvo_now: float,
    vri_delta_now: float,
    mkt_state: str,
) -> tuple[bool, str]:
    """V4 止損邏輯"""
    # 硬止損 -8%
    if curr_ret <= V4_HARD_STOP_PCT:
        return True, f"S-硬止損({curr_ret:.1%})"

    # 條件式止損 -5%（需配合量能/大盤惡化）
    if curr_ret <= V4_COND_STOP_PCT:
        cond = (pvo_now < -1.0 or vri_delta_now < -3.0 or mkt_state == "bear")
        if cond:
            return True, f"S-條件止損({curr_ret:.1%})"

    return False, ""


# ─────────────────────────────────────────────────────────────
# ★ V12 出場層判斷函數
# ─────────────────────────────────────────────────────────────

def check_v12_ev_decay(
    pos: Position,
    ev_now: float,
    slope_now: float,
) -> tuple[bool, str]:
    """V12 EV 衰退出場"""
    if pos.days_held() < V12_PROTECT_DAYS:
        return False, ""

    ev_entry = pos.ev_soft_entry
    if ev_now < V12_EV_DEAD_THR:
        return True, f"EV衰退({ev_now*100:.1f}%→死亡)"

    if ev_entry > 0:
        drop_ratio = (ev_entry - ev_now) / ev_entry
        days_h = pos.days_held()

        # 快速衰退 + slope 惡化 → 加速出場
        if drop_ratio > V12_EV_ACCEL_PCT and slope_now < V12_EV_DECAY_SLOPE and days_h > V12_TIME_DECAY_DAYS:
            return True, f"Slope加速(ev{ev_entry*100:.1f}%→{ev_now*100:.1f}%)"

        # 累積衰退 > 35%
        if drop_ratio > V12_EV_DECAY_PCT and days_h > V12_TIME_DECAY_DAYS:
            return True, f"時間衰減(ev{ev_entry*100:.1f}%→{ev_now*100:.1f}%)"

    return False, ""


def check_v12_pvo_exhaustion(
    pos: Position,
    pvo_5d_chg: float,
    vri_now: float,
) -> tuple[bool, str]:
    """V12 量能枯竭出場"""
    if pos.days_held() < V12_PVO_EXIT_DAYS:
        return False, ""

    is_exhausted = (pvo_5d_chg < V12_PVO_DECAY_THR and vri_now < V12_VRI_COLD_THR)
    if is_exhausted:
        new_count = pos.pvo_exit_count + 1
        if new_count >= V12_PVO_EXIT_CONSEC:
            return True, f"量能枯竭(pvo{pvo_5d_chg:.2f})"
    return False, ""


def check_v12_vri_hot(
    curr_ret: float,
    vri_now: float,
    days_held: int,
) -> tuple[bool, str]:
    """V12 VRI過熱出場"""
    if days_held < 3:
        return False, ""
    if vri_now >= V12_VRI_VERY_HOT:
        return True, f"VRI極熱({vri_now:.2f})"
    if vri_now >= V12_VRI_HOT_THR and curr_ret >= 0.05:
        return True, f"VRI停利①({vri_now:.2f})"
    return False, ""


def check_v12_time_exit(
    pos: Position,
    curr_ret: float,
) -> tuple[bool, str]:
    """V12 時間出場"""
    days_h = pos.days_held()
    if pos.tp1_hit:
        return False, ""
    if days_h >= V12_TIME_EXIT_T1_DAYS and curr_ret < V12_TIME_EXIT_T1_RET and days_h < V12_TIME_EXIT_T2_DAYS:
        return True, f"時間T1({days_h}d)"
    if days_h >= V12_TIME_EXIT_T2_DAYS and curr_ret < V12_TIME_EXIT_T2_RET:
        return True, f"時間T2({days_h}d)"
    if days_h >= V12_MAX_HOLD_DAYS:
        return True, f"時間上限{V12_MAX_HOLD_DAYS}d"
    return False, ""


def check_v12_protection(
    pos: Position,
    curr_ret: float,
    current_price: float,
) -> tuple[bool, str]:
    """V12 保護期止損 & 保本出場"""
    days_h = pos.days_held()

    # 保本出場（帳面達10%後設保本線）
    if pos.profit_locked:
        lock_px = pos.profit_lock_px if pos.profit_lock_px > 0 else pos.entry_price * (1 + V12_PROFIT_LOCK_FLOOR)
        if current_price < lock_px:
            return True, f"保本出場({curr_ret:.1%})"

    # 硬停損
    if curr_ret <= V12_HARD_STOP_PCT:
        return True, f"硬停損({curr_ret:.1%})"

    # 保護期分層止損（protect_over = False 時）
    if not pos.protect_over:
        if days_h <= V12_PROTECT_DAYS_EARLY:
            if curr_ret < V12_PROTECT_LOSS_EARLY:
                return True, f"保護期早({curr_ret:.1%})"
        else:
            if curr_ret < V12_PROTECT_LOSS_LATE:
                return True, f"保護期({curr_ret:.1%})"

    return False, ""


# ─────────────────────────────────────────────────────────────
# ★ 核心決策函數
# ─────────────────────────────────────────────────────────────

def evaluate_sell_decision(
    pos: Position,
    current_price: float,
    v4_row: Optional[dict],
    v12_row: Optional[dict],
    mkt_state: str,
    regime: dict,
) -> SellDecision:
    """
    整合 V4 + V12 雙層出場決策。
    回傳 SellDecision，action 為 "SELL" / "PARTIAL_SELL" / "HOLD"。
    """
    curr_ret  = pos.current_return(current_price)
    days_h    = pos.days_held()
    net_ret   = curr_ret - SELL_COST - SLIPPAGE / 2

    # 從快照擷取即時指標
    pvo_now      = float((v4_row or {}).get("pvo", pos.pvo_entry))
    vri_now      = float((v4_row or {}).get("vri", pos.vri_entry))
    vri_delta    = float((v4_row or {}).get("vri_delta", 0.0))
    pvo_5d_chg   = float((v12_row or {}).get("pvo_5d_chg", 0.0))
    vri_v12      = float((v12_row or {}).get("vri", 1.0))
    ev_now       = float((v12_row or {}).get("ev", pos.ev_soft_entry))
    slope_now    = float((v12_row or {}).get("slope_5d", 0.0))

    # 保護期結束旗標
    if days_h > V12_PROTECT_DAYS:
        pos.protect_over = True

    # 獲利保全觸發
    if curr_ret >= V12_PROFIT_LOCK_THR and not pos.profit_locked:
        pos.profit_locked = True
        pos.profit_lock_px = pos.entry_price * (1 + V12_PROFIT_LOCK_FLOOR)

    action, reason, layer, exit_type, score = "HOLD", "", "NONE", "─", 0

    # ── 分批止盈（V4，不全出）─────────────────────────────────
    if not pos.partial_sold and curr_ret >= V4_PARTIAL_EXIT_RET:
        return SellDecision(
            symbol=pos.symbol, action="PARTIAL_SELL",
            reason=f"S-分批止盈(+{curr_ret:.1%})",
            layer="V4", exit_type="S-分批止盈",
            curr_ret=curr_ret, net_ret=net_ret,
            days_held=days_h, current_price=current_price,
            exit_score=3, combo_key=pos.combo_key,
            v12_path=pos.v12_path, ev_soft=ev_now,
        )

    # ══ V4 出場層（高優先）════════════════════════════════════

    # 結構破壞
    sb, sb_reason = check_v4_structural_break(pos, pvo_now, vri_now, vri_delta, mkt_state)
    if sb:
        action, reason, layer, exit_type, score = "SELL", sb_reason, "V4", sb_reason, 5

    # 止損（結構破壞優先，否則檢查止損）
    if action == "HOLD":
        sl, sl_reason = check_v4_stop_loss(pos, curr_ret, pvo_now, vri_delta, mkt_state)
        if sl:
            action, reason, layer, exit_type, score = "SELL", sl_reason, "V4", sl_reason, 5

    # ATR 尾隨止損
    if action == "HOLD" and current_price < pos.stop_price > 0:
        action, reason, layer, exit_type, score = "SELL", "S-ATR止損", "V4", "S-ATR止損", 5

    # ══ V12 出場層（V4 未觸發時繼續判斷）═════════════════════

    if action == "HOLD":
        # 保護 & 保本
        prot, prot_reason = check_v12_protection(pos, curr_ret, current_price)
        if prot:
            action, reason, layer, exit_type, score = "SELL", prot_reason, "V12", _map_exit_type(prot_reason), 5

    if action == "HOLD":
        # VRI 過熱
        hot, hot_reason = check_v12_vri_hot(curr_ret, vri_v12, days_h)
        if hot:
            action, reason, layer, exit_type, score = "SELL", hot_reason, "V12", _map_exit_type(hot_reason), 4

    if action == "HOLD":
        # 量能枯竭
        pos.pvo_exit_count = pos.pvo_exit_count + 1 if (pvo_5d_chg < V12_PVO_DECAY_THR and vri_v12 < V12_VRI_COLD_THR) else 0
        ex_p, ex_reason = check_v12_pvo_exhaustion(pos, pvo_5d_chg, vri_v12)
        if ex_p:
            action, reason, layer, exit_type, score = "SELL", ex_reason, "V12", "量能枯竭", 3

    if action == "HOLD":
        # EV 衰退
        ev_triggered, ev_reason = check_v12_ev_decay(pos, ev_now, slope_now)
        if ev_triggered:
            action, reason, layer, exit_type, score = "SELL", ev_reason, "V12", _map_exit_type(ev_reason), 3

    if action == "HOLD":
        # 時間出場
        te, te_reason = check_v12_time_exit(pos, curr_ret)
        if te:
            action, reason, layer, exit_type, score = "SELL", te_reason, "V12", _map_exit_type(te_reason), 2

    # 停利 ①②（V12 路徑 TP）
    if action == "HOLD" and pos.protect_over:
        if current_price >= pos.tp1_px > 0 and not pos.tp1_hit:
            action, reason, layer, exit_type, score = "SELL", f"停利①(+{curr_ret:.1%})", "V12", "停利①", 4
        elif pos.tp1_hit and current_price >= pos.tp2_px > 0:
            action, reason, layer, exit_type, score = "SELL", f"停利②(+{curr_ret:.1%})", "V12", "停利②", 4

    # 更新 EV
    pos.ev_soft_cur = ev_now

    return SellDecision(
        symbol=pos.symbol, action=action, reason=reason,
        layer=layer, exit_type=exit_type,
        curr_ret=curr_ret, net_ret=net_ret,
        days_held=days_h, current_price=current_price,
        exit_score=score, combo_key=pos.combo_key,
        v12_path=pos.v12_path, ev_soft=ev_now,
    )


def _map_exit_type(reason: str) -> str:
    mapping = {
        "停利②": "停利②", "停利①": "停利①",
        "EV衰退": "EV衰退", "Slope加速": "Slope加速",
        "時間衰減": "時間衰減", "量能枯竭": "量能枯竭",
        "VRI極熱": "VRI極熱", "VRI停利": "VRI停利①",
        "硬停損": "硬停損", "保本": "保本出場",
        "保護期": "保護期", "時間T1": "時間出場",
        "時間T2": "時間出場",
    }
    for k, v in mapping.items():
        if k in reason:
            return v
    return reason[:12]


# ─────────────────────────────────────────────────────────────
# ★ 交易歷史 Log
# ─────────────────────────────────────────────────────────────

def append_trade_log(decision: SellDecision, pos: Position, regime: dict):
    """將賣出決策追加到 trade_history.json（Streamlit 展示用）"""
    try:
        if TRADE_HISTORY_FILE.exists():
            with open(TRADE_HISTORY_FILE, encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
    except Exception:
        history = []

    record = {
        "date":         TODAY_STR,
        "sym":          decision.symbol,
        "action_type":  "賣出",
        "exit_type":    decision.exit_type,
        "layer":        decision.layer,
        "reason":       decision.reason,
        "ret":          round(decision.curr_ret, 5),
        "net_ret":      round(decision.net_ret, 5),
        "days_held":    decision.days_held,
        "current_price":decision.current_price,
        "entry_price":  pos.entry_price,
        "exit_score":   decision.exit_score,
        "combo_key":    pos.combo_key,
        "signal_label": pos.signal_label,
        "v12_path":     pos.v12_path,
        "ev_soft_entry":round(pos.ev_soft_entry, 5),
        "ev_soft_exit": round(decision.ev_soft, 5),
        "entry_source": pos.entry_source,
        "is_addon":     pos.is_addon,
        "mkt_state":    pos.mkt_state,
        "regime_label": regime.get("label", "─"),
        "bear":         regime.get("bear", 0),
        "bull":         regime.get("bull", 0),
        "year":         date.today().year,
        "timestamp":    decision.timestamp,
    }

    history.append(record)
    with open(TRADE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    log.info(f"📝 交易記錄已寫入: {decision.symbol} {decision.exit_type} {decision.curr_ret:+.2%}")


def save_sell_summary(decisions: list[SellDecision], positions_after: list[Position]):
    """將本次賣出摘要寫入 storage/logs/sell_summary_{today}.json，Streamlit 可讀取"""
    summary = {
        "date":       TODAY_STR,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sell_count": len([d for d in decisions if d.action != "HOLD"]),
        "hold_count": len([d for d in decisions if d.action == "HOLD"]),
        "sells": [asdict(d) for d in decisions if d.action != "HOLD"],
        "remaining_positions": len(positions_after),
    }
    path = LOGS_DIR / f"sell_summary_{TODAY_STR}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # 也覆蓋最新版
    latest = LOGS_DIR / "sell_summary_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"✅ 賣出摘要已儲存: {path}")


# ─────────────────────────────────────────────────────────────
# ★ 從快照取得現價（由 v4_latest.json 中的 top20 list）
# ─────────────────────────────────────────────────────────────

def get_current_price(symbol: str, v4_snapshot: dict, v12_snapshot: dict) -> Optional[float]:
    for row in v4_snapshot.get("top20", []):
        if row.get("symbol") == symbol:
            return float(row.get("price", 0)) or None
    for pos in v12_snapshot.get("positions", []):
        if pos.get("symbol") == symbol:
            return float(pos.get("current_price", 0)) or None
    return None


def get_v4_row(symbol: str, v4_snapshot: dict) -> Optional[dict]:
    for row in v4_snapshot.get("top20", []):
        if row.get("symbol") == symbol:
            return row
    return None


def get_v12_row(symbol: str, v12_snapshot: dict) -> Optional[dict]:
    for pos in v12_snapshot.get("positions", []):
        if pos.get("symbol") == symbol:
            return pos
    return None


# ─────────────────────────────────────────────────────────────
# ★ 主流程
# ─────────────────────────────────────────────────────────────

def run_sell_check():
    """
    每日執行：
    1. 讀取 V4 / V12 / Regime 快照
    2. 讀取持倉列表
    3. 對每筆持倉執行雙層賣出決策
    4. 輸出決策報告 & 寫入 trade_history.json
    """
    log.info("=" * 65)
    log.info("【portfolio_manager.py】V412 持倉管理 & 賣出決策")
    log.info(f"  執行日期: {TODAY_STR}")
    log.info("=" * 65)

    v4_snap  = load_v4_snapshot()
    v12_snap = load_v12_snapshot()
    regime   = load_regime_snapshot()
    positions = load_positions()

    if not positions:
        log.warning("⚠️ 無持倉資料（positions.json 空或不存在）。")
        log.info("  → 提示：可透過 V4 / V12 買進模組新增持倉至 positions.json")
        return

    mkt_state = regime.get("mkt_state", "neutral")
    log.info(f"  大盤狀態: {mkt_state} | 制度: {regime.get('label','─')}")
    log.info(f"  持倉數量: {len(positions)} 筆")
    log.info("")

    all_decisions: list[SellDecision] = []
    remaining: list[Position] = []
    sell_count = 0

    header = f"{'代號':10s} {'現價':>8} {'報酬':>8} {'持天':>5} {'觸發層':5} {'原因':30s} {'評分':>4}"
    sep    = "─" * 85
    log.info(header)
    log.info(sep)

    for pos in positions:
        sym = pos.symbol
        current_price = get_current_price(sym, v4_snap, v12_snap)
        if current_price is None or current_price <= 0:
            log.warning(f"  {sym}: 無法取得現價，維持持有")
            remaining.append(pos)
            continue

        v4_row  = get_v4_row(sym, v4_snap)
        v12_row = get_v12_row(sym, v12_snap)

        decision = evaluate_sell_decision(
            pos=pos,
            current_price=current_price,
            v4_row=v4_row,
            v12_row=v12_row,
            mkt_state=mkt_state,
            regime=regime,
        )
        all_decisions.append(decision)

        ret_str = f"{decision.curr_ret:+.2%}"
        log.info(
            f"  {sym:10s} {current_price:8.2f} {ret_str:8} {decision.days_held:5d}d "
            f"[{decision.layer:4s}] {decision.reason:30s} 分:{decision.exit_score}"
        )

        if decision.action in ("SELL", "PARTIAL_SELL"):
            sell_count += 1
            append_trade_log(decision, pos, regime)

            if decision.action == "PARTIAL_SELL":
                # 分批出場：更新持倉但不刪除
                pos.partial_sold = True
                pos.remain_weight *= (1 - V4_PARTIAL_EXIT_PCT)
                remaining.append(pos)
            elif decision.exit_type == "停利①":
                # 停利①：半倉出場，保留另一半
                pos.tp1_hit = True
                pos.remain_weight *= 0.5
                remaining.append(pos)
            # 其他全部出場：不加入 remaining
        else:
            remaining.append(pos)

    log.info(sep)
    log.info(f"  ✅ 本次賣出: {sell_count} 筆  剩餘持倉: {len(remaining)} 筆")

    save_positions(remaining)
    save_sell_summary(all_decisions, remaining)

    # ── 輸出摘要表 ──────────────────────────────────────────
    sells = [d for d in all_decisions if d.action != "HOLD"]
    if sells:
        log.info("\n【賣出摘要】")
        log.info(f"  {'代號':10s} {'動作':10s} {'報酬':>8} {'淨報酬':>8} {'原因'}")
        for d in sells:
            log.info(f"  {d.symbol:10s} {d.action:10s} {d.curr_ret:+8.2%} {d.net_ret:+8.2%} {d.reason}")
    else:
        log.info("\n  → 今日無賣出決策（全部持有）")

    log.info("\n  📁 結果已寫入:")
    log.info(f"    {POSITIONS_FILE}")
    log.info(f"    {TRADE_HISTORY_FILE}")
    log.info(f"    {LOGS_DIR / 'sell_summary_latest.json'}")
    log.info("=" * 65)


def demo_backtest_positions():
    """
    示範：從 V12 快照直接讀取其持倉並評估
    （用於驗證賣出邏輯是否與 5Y-3W-V12_1.py 回測一致）
    """
    log.info("\n【Backtest 模式】從 V12 快照重建持倉並評估")
    v12_snap = load_v12_snapshot()
    v4_snap  = load_v4_snapshot()
    regime   = load_regime_snapshot()
    mkt_state = regime.get("mkt_state", "neutral")

    raw_positions = v12_snap.get("positions", [])
    if not raw_positions:
        log.warning("V12 快照中無持倉資料")
        return

    for raw in raw_positions:
        pos = Position(
            symbol      = raw.get("symbol", ""),
            entry_price = float(raw.get("entry_price", 0)),
            entry_date  = raw.get("entry_date", TODAY_STR),
            combo_key   = raw.get("combo_key", "─"),
            signal_label= raw.get("signal_label", "─"),
            v12_path    = raw.get("path", "─"),
            ev_soft_entry=float(raw.get("ev", 0.0)),
            tp1_px      = float(raw.get("tp1", 0.0)),
            tp2_px      = float(raw.get("tp2", 0.0)),
            pvo_entry   = float(raw.get("pvo", 0.0)),
            vri_entry   = float(raw.get("vri", 50.0)),
            mkt_state   = raw.get("mkt_state", "neutral"),
        )
        current_price = float(raw.get("current_price", pos.entry_price))
        if current_price <= 0:
            continue

        v4_row  = get_v4_row(pos.symbol, v4_snap)
        v12_row = raw  # 使用 v12 快照本身的欄位

        decision = evaluate_sell_decision(pos, current_price, v4_row, v12_row, mkt_state, regime)
        print(f"  {pos.symbol:8s} | {decision.action:12s} | [{decision.layer}] {decision.reason}")


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V412 Portfolio Manager — 持倉管理 & 賣出決策")
    parser.add_argument("--backtest", action="store_true", help="從 V12 快照示範回測評估")
    args = parser.parse_args()

    if args.backtest:
        demo_backtest_positions()
    else:
        run_sell_check()


if __name__ == "__main__":
    main()
