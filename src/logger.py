"""
结构化回合日志系统 - 每轮训练/休息/商店/事件的完整记录。

输出: logs/run_YYYYMMDD_HHMMSS.json
每个 run 一个 JSON 文件，包含:
  - 运行元数据 (时间, 规则配置, 构建方向)
  - 每轮详细记录 (决策, 执行, 游戏状态, 度量指标)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


_SCREENSHOT_ACTION_TYPES = {
    "decision", "train", "rest", "event", "arcanum", "card_select", "shop",
}


class RunLogger:
    """单次跑马的完整日志记录器"""

    def __init__(self, profile_name: str = "", build_direction: str = "",
                 config_summary: dict | None = None):
        self._start_time = time.time()
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._turns: list[dict] = []
        self._current_turn: dict | None = None
        self._profile_name = profile_name
        self._build_direction = build_direction

        # 摘要统计 (运行中累积)
        self._train_count = 0
        self._rest_count = 0
        self._shop_visits = 0
        self._shop_items_bought = 0
        self._events_encountered = 0
        self._arcanum_clicks = 0
        self.stuck_screenshot: str | None = None
        self.stuck_context: dict | None = None

        # 回合计时
        self._turn_timing_start: float = 0.0
        self._turn_timings: list[dict] = []
        self._handler_timings: dict[str, list[float]] = {}

        # The trainer supplies the frame currently being evaluated.  Decision
        # actions keep a relative path to this frame so the inspector can open
        # the exact screen that produced a historical choice.
        self._current_screenshot = None
        self._current_screenshot_path: str | None = None
        self._screenshot_index = 0

    def set_screenshot(self, screenshot) -> None:
        """Associate the current game frame with subsequent action logs.

        A new frame clears the cached path.  Multiple actions produced from
        the same frame (for example a decision followed by its click) reuse
        one PNG instead of writing duplicate files.
        """
        self._current_screenshot = screenshot
        self._current_screenshot_path = None

    def _save_current_screenshot(self, action_type: str) -> str | None:
        if self._current_screenshot is None:
            return None
        if self._current_screenshot_path:
            return self._current_screenshot_path
        try:
            import cv2

            image = self._current_screenshot
            if getattr(image, "size", 0) == 0:
                return None
            root = Path(__file__).resolve().parent.parent
            folder = root / "logs" / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            self._screenshot_index += 1
            filename = f"{self._run_id}_{self._screenshot_index:04d}_{action_type}.png"
            path = folder / filename
            if not cv2.imwrite(str(path), image):
                return None
            self._current_screenshot_path = (Path("logs") / "screenshots" / filename).as_posix()
            return self._current_screenshot_path
        except Exception as exc:  # screenshot logging must never stop the run
            print(f"[日志] 决策截图保存失败: {exc}")
            return None

    # ===================== 回合生命周期 =====================

    def start_turn(self, round_num: int, turn_num: Optional[int],
                   stamina: float, mood: str, money: int):
        """开始新回合, 记录初始游戏状态"""
        self._current_turn = {
            "round": round_num,
            "turn": turn_num,
            "timestamp": datetime.now().isoformat(),
            "state_before": {
                "stamina": round(stamina, 1),
                "mood": mood,
                "money": money,
            },
            "actions": [],
        }

    def end_turn(self):
        """结束当前回合, 追加到列表"""
        if self._current_turn is not None:
            self._turns.append(self._current_turn)
            self._current_turn = None

    # ===================== 行动记录 =====================

    def log_arcanum(self, card_index: int, priority_attr: str,
                    current_turn: int, turn_limit: int):
        """记录支援卡点击"""
        self._arcanum_clicks += 1
        self._append_action("arcanum", {
            "card_index": card_index,
            "priority_attr": priority_attr,
            "turn_progress": f"{current_turn}/{turn_limit}",
        })

    def log_decision_rule_engine(self, result, context, scan_state):
        """记录规则引擎决策过程"""
        self._append_action("decision", {
            "mode": "rule_engine",
            "profile": context.profile_name,
            "matched_rule": result.matched_rule_id,
            "action": result.action.value,
            "target_row": result.target_row_index,
            "builtin_fallback": result.used_builtin_default,
            "metrics": {
                "icon_counts": list(scan_state.icon_counts),
                "fail_rates": list(scan_state.fail_rates),
                "gain_values": list(scan_state.gain_values),
                "advantage_training": list(getattr(scan_state, "is_advantage", [False] * 5)),
                "advantage_multiplier": round(float(getattr(context, "advantage_multiplier", 1.35)), 3),
                "strength_stat": scan_state.strength_stat,
                "stamina_stat": scan_state.stamina_stat,
            },
        })

    def log_decision_traditional(self, options, state, chosen):
        """记录传统权重模式决策过程"""
        opts = []
        for opt in options:
            opts.append({
                "attr": opt.attr_name,
                "gain": opt.gain,
                "fail_rate": round(opt.failure_rate, 1),
            })
        self._append_action("decision", {
            "mode": "traditional",
            "priority_attr_id": getattr(state, '_priority_id', None),
            "options": opts,
            "chosen": chosen.attr_name if chosen else None,
            "thresholds": {
                "failure": getattr(state, '_fail_threshold', None),
                "min_gain": getattr(state, '_min_gain', None),
            },
        })

    def log_train_exec(self, row_index: int, attr_name: str, metrics: dict | None = None):
        """记录实际训练执行"""
        self._train_count += 1
        entry = {"row_index": row_index, "attribute": attr_name}
        if metrics:
            entry["metrics"] = metrics
        self._append_action("train", entry)

    def log_rest(self, option_id: int, option_name: str, cost: int,
                 stamina_before: float, reason: str = ""):
        """记录休息决策与执行"""
        self._rest_count += 1
        self._append_action("rest", {
            "option_id": option_id,
            "option": option_name,
            "cost": cost,
            "stamina_before": round(stamina_before, 1),
            "reason": reason,
        })

    def log_shop(self, scanned: list, bought: list, stamina: float,
                 mood: str, money: int):
        """记录商店交易"""
        self._shop_visits += 1
        self._shop_items_bought += len(bought)

        scanned_info = []
        for slot, info, text in scanned:
            scanned_info.append({
                "slot": slot,
                "type": info.get("type", "?"),
                "name": info.get("attr_name", "") or text[:40],
                "value": info.get("value", 0),
                "from_profile": info.get("from_profile", False),
            })

        bought_info = []
        for slot, info, score in bought:
            bought_info.append({
                "slot": slot,
                "type": info.get("type", "?"),
                "name": info.get("attr_name", ""),
                "score": score,
            })

        self._append_action("shop", {
            "state": {"stamina": round(stamina, 1), "mood": mood, "money": money},
            "scanned": scanned_info,
            "bought": bought_info,
        })

    def log_event(self, event_id: str, event_name: str, option_selected: int,
                  num_options: int, auto_learned: bool, matched: bool):
        """记录事件选择"""
        self._events_encountered += 1
        self._append_action("event", {
            "event_id": event_id,
            "event_name": event_name,
            "option_selected": option_selected,
            "num_options": num_options,
            "auto_learned": auto_learned,
            "matched": matched,
        })

    def log_handler_dispatch(self, handler_name: str, elapsed_ms: float = 0):
        """记录处理器调度 (轻量)"""
        self._append_action("handler", {
            "name": handler_name,
            "elapsed_ms": round(elapsed_ms, 1),
        })
        # 累计每个handler的耗时用于后续分析
        if handler_name not in self._handler_timings:
            self._handler_timings[handler_name] = []
        self._handler_timings[handler_name].append(elapsed_ms)

    def start_turn_timing(self):
        """开始为本轮回合计时"""
        self._turn_timing_start = time.perf_counter()

    def end_turn_timing(self) -> float:
        """结束回合计时, 记录耗时"""
        elapsed = time.perf_counter() - self._turn_timing_start
        if self._current_turn is not None:
            self._current_turn["duration_seconds"] = round(elapsed, 2)
            self._current_turn["handler_chain"] = [
                a for a in self._current_turn.get("actions", [])
                if a.get("type") == "handler"
            ]
        self._turn_timings.append({
            "round": self._current_turn.get("round", 0) if self._current_turn else 0,
            "duration_seconds": round(elapsed, 2),
        })
        return elapsed

    def log_commission(self, tier: str, my_rank: int, recommended: int):
        """记录讨伐委托执行"""
        self._append_action("commission", {
            "tier": tier,
            "my_rank": my_rank,
            "recommended": recommended,
        })

    def log_card_select(self, card_index: int, reason: str, texts: list[str]):
        """记录卡片选择"""
        self._append_action("card_select", {
            "card_index": card_index,
            "reason": reason,
            "card_texts": texts,
        })

    # ===================== 持久化 =====================

    def save(self, exit_reason: str = "normal") -> str:
        """保存完整日志到 JSON 文件, 返回文件路径"""
        if self._current_turn is not None:
            self.end_turn()

        elapsed = time.time() - self._start_time
        total_actions = sum(len(t["actions"]) for t in self._turns)

        # 耗时分析
        timing = self._build_timing_summary()

        log_data = {
            "run_id": self._run_id,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(),
            "duration_seconds": round(elapsed, 1),
            "profile": self._profile_name,
            "build_direction": self._build_direction,
            "exit_reason": exit_reason,
            "summary": {
                "total_turns": len(self._turns),
                "total_actions": total_actions,
                "train_count": self._train_count,
                "rest_count": self._rest_count,
                "shop_visits": self._shop_visits,
                "shop_items_bought": self._shop_items_bought,
                "events_encountered": self._events_encountered,
                "arcanum_clicks": self._arcanum_clicks,
            },
            "timing": timing,
            "turns": self._turns,
        }

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"run_{self._run_id}.json"

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"[日志] 已保存: {log_path} ({len(self._turns)} 回合, {total_actions} 动作)")
        return str(log_path)

    # ===================== 耗时分析 =====================

    def _build_timing_summary(self) -> dict:
        """构建耗时分析摘要: 各handler平均耗时, 最慢回合, 等待耗时等"""
        summary: dict = {
            "total_seconds": round(time.time() - self._start_time, 1),
            "turn_count": len(self._turns),
        }

        # 各handler耗时统计
        handler_stats = {}
        for name, times in self._handler_timings.items():
            if times:
                handler_stats[name] = {
                    "total_seconds": round(sum(times) / 1000, 2),
                    "count": len(times),
                    "avg_ms": round(sum(times) / len(times), 1),
                    "max_ms": round(max(times), 1),
                }
        summary["handlers"] = handler_stats

        # 回合计时
        if self._turn_timings:
            durations = [t["duration_seconds"] for t in self._turn_timings]
            summary["per_turn"] = {
                "avg_seconds": round(sum(durations) / len(durations), 2),
                "max_seconds": round(max(durations), 2),
                "min_seconds": round(min(durations), 2),
            }
            # 最慢的5个回合
            sorted_turns = sorted(self._turn_timings,
                                  key=lambda x: x["duration_seconds"], reverse=True)
            summary["slowest_turns"] = sorted_turns[:5]

        # 总耗时构成
        handler_total = sum(s["total_seconds"] for s in handler_stats.values())
        total = summary["total_seconds"]
        summary["breakdown"] = {
            "handler_time_seconds": round(handler_total, 2),
            "other_time_seconds": round(total - handler_total, 2),
            "handler_pct": round(handler_total / total * 100, 1) if total > 0 else 0,
        }

        return summary

    # ===================== 内部 =====================

    def _append_action(self, action_type: str, data: dict):
        """向当前回合追加一个动作记录"""
        if self._current_turn is None:
            # 未调用 start_turn 时的兜底
            self._current_turn = {
                "round": 0,
                "turn": None,
                "timestamp": datetime.now().isoformat(),
                "state_before": {},
                "actions": [],
            }
        data["type"] = action_type
        data["ts"] = datetime.now().isoformat()
        if action_type in _SCREENSHOT_ACTION_TYPES:
            screenshot_path = self._save_current_screenshot(action_type)
            if screenshot_path:
                data["screenshot"] = screenshot_path
        self._current_turn["actions"].append(data)
