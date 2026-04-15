"""
╔══════════════════════════════════════════════════════════════╗
║  portfolio_manager.py  V412 持倉管理 & 賣出決策             ║
║                                                              ║
║  [FIX-05] regime_latest.json → regime_state.json            ║
║  [FIX-07] positions.json 從 v12_latest.json 同步，           ║
║           不再要求獨立存在                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import os
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("portfolio_manager")

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(ROOT_DIR, "storage")

# [FIX-05] 修正：使用 regime_state.json，不是 regime_latest.json
REGIME_PATH    = os.path.join(STORAGE_DIR, "regime", "regime_state.json")
V12_PATH       = os.path.join(STORAGE_DIR, "v12", "v12_latest.json")
POSITIONS_PATH = os.path.join(STORAGE_DIR, "positions.json")


def load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"讀取 {path} 失敗: {e}")
        return None


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"  ✅ 儲存: {path}")
    except Exception as e:
        log.error(f"  ❌ 儲存失敗 {path}: {e}")


def main():
    log.info("=" * 60)
    log.info("【portfolio_manager.py】V412 持倉管理 & 賣出決策")
    log.info(f"  執行日期: {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # [FIX-05] 讀取 regime_state.json
    regime = load_json(REGIME_PATH)
    if not regime:
        log.warning(f"讀取 {REGIME_PATH} 失敗（找不到 regime_state.json）")
        log.info("  → 提示：請確認 daily_run.py 已正常執行並產生 storage/regime/regime_state.json")
    else:
        log.info(f"  Regime: {regime.get('label','?')} | 主路徑: {regime.get('active_path','?')}")

    # [FIX-07] 讀取 positions：優先讀 positions.json，不存在則從 v12_latest.json 同步
    positions_data = load_json(POSITIONS_PATH)

    if not positions_data:
        log.warning(f"positions.json 空或不存在，嘗試從 v12_latest.json 同步...")
        v12_data = load_json(V12_PATH)
        if v12_data and v12_data.get("positions"):
            positions_data = {
                "positions":    v12_data["positions"],
                "generated_at": v12_data.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
                "source":       "synced_from_v12_latest",
            }
            # 同步寫入 positions.json 供下次使用
            save_json(POSITIONS_PATH, positions_data)
            log.info(f"  ✅ 從 v12_latest.json 同步 {len(positions_data['positions'])} 筆持倉")
        else:
            log.warning("⚠️ 無持倉資料 (v12_latest.json 也為空)")
            log.info("  → 提示：可透過 V4 / V12 買進模組新增持倉至 positions.json")
            positions_data = {"positions": [], "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}

    positions = positions_data.get("positions", [])
    log.info(f"  持倉數量: {len(positions)} 檔")

    if not positions:
        log.info("  目前無持倉，跳過賣出決策")
        return

    # 賣出決策邏輯
    sell_list  = []
    hold_list  = []

    for pos in positions:
        sym     = pos.get("symbol", "?")
        exit_sig = pos.get("exit_signal", "—")
        ret_pct  = pos.get("curr_ret_pct", 0.0)
        close    = pos.get("close", 0)
        stop_px  = pos.get("stop_price", 0)
        tp1_px   = pos.get("tp1_price", 0)

        reasons = []

        # 硬停損
        if stop_px > 0 and close > 0 and close <= stop_px:
            reasons.append(f"觸及停損 ({close:.1f} ≤ {stop_px:.1f})")

        # 停利
        if tp1_px > 0 and close > 0 and close >= tp1_px:
            reasons.append(f"觸及停利① ({close:.1f} ≥ {tp1_px:.1f})")

        # exit_signal 觸發
        if exit_sig and exit_sig not in ("—", "無", ""):
            reasons.append(f"出場訊號: {exit_sig}")

        if reasons:
            log.info(f"  ▼ {sym} 建議賣出 | {' | '.join(reasons)}")
            sell_list.append({**pos, "sell_reasons": reasons})
        else:
            hold_list.append(pos)

    log.info(f"  建議賣出: {len(sell_list)} 檔 | 繼續持有: {len(hold_list)} 檔")
    if sell_list:
        for s in sell_list:
            log.info(f"    → {s['symbol']}: {s['sell_reasons']}")

    log.info("=" * 60)
    log.info("portfolio_manager 執行完成")


if __name__ == "__main__":
    main()
