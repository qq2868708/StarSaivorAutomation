"""Guarded home-to-home daily for 圣铠 / 策略战 / NPC / 老手."""
import re
from dataclasses import dataclass

import numpy as np

from .daily import identify_daily_page


OPPONENTS = {"老手骑士团", "老手先锋队", "老手袭击组", "老手支持组", "老手射击团"}
BACK = (.043, .064)


def _blue(frame, x, y):
    h, w = frame.shape[:2]
    pixel = frame[round(y * h), round(x * w)].astype(int)
    b, g, r = pixel
    return bool(b > 150 and g > 70 and b > r + 80)


def read_arena(frame, items):
    """Read client-relative controls; absence of OCR is never zero attempts."""
    h, w = frame.shape[:2]
    words = []
    for text, score, box in items:
        if float(score) < .8 or box is None:
            continue
        points = np.asarray(box)
        x, y = points.mean(axis=0) / [w, h]
        words.append((re.sub(r"\s+", "", str(text)), float(x), float(y)))
    texts = {t for t, _, _ in words}

    def button(label, region=(0, 0, 1, 1)):
        left, top, right, bottom = region
        found = [(x, y) for t, x, y in words
                 if t == label and left <= x <= right and top <= y <= bottom]
        return found[0] if len(found) == 1 else None

    counts = {int(m[1]) for t, x, y in words if x > .59 and y < .10
              for m in re.finditer(r"(?<!\d)([0-6])/6(?!\d)", t)}
    state = {"page": "unknown", "remaining": next(iter(counts)) if len(counts) == 1 else None,
             "buttons": {}, "opponents": [], "challenges": [], "cooldowns": [],
             "npc_selected": False, "senior_selected": False}
    for label in ("圣铠", "策略战", "NPC", "老手", "跳过战斗", "确认", "取消", "点击以继续"):
        point = button(label)
        if point:
            state["buttons"][label] = point

    # Modals take precedence over the darkened page behind them.
    compact = "".join(texts)
    if "确认" in texts and "取消" in texts:
        if ("跳过战斗" in texts and "确定要消耗1个圣铠钥匙" in texts
                and "跳过该场战斗吗" in compact
                and not any(t in compact for t in ("钻石", "购买", "补充"))):
            state["page"] = "arena_confirm"
        else:
            state["page"] = "unsupported_dialog"
    elif "点击以继续" in texts and "战术现状" in texts and "获得奖励" in texts:
        if "VICTORY" in texts or "DEFEAT" in texts:
            state["page"] = "arena_result"
            state["outcome"] = "victory" if "VICTORY" in texts else "defeat"
    elif "跳过战斗" in texts and "开始战斗" in texts and "策略战" in texts:
        state["page"] = "arena_prepare"
        state["prepared_opponent"] = next(
            (name for name in OPPONENTS if any(name in t for t in texts)), None)
    elif {"策略战", "NPC", "对战列表", "防御编制"} <= texts:
        state["page"] = "arena_list"
        state["npc_selected"] = _blue(frame, .50, .164)
        state["senior_selected"] = _blue(frame, .525, .211)
        state["opponents"] = sorted(
            [(t, round(y, 3)) for t, x, y in words
             if t in OPPONENTS and .38 < x < .50 and .23 < y < .88], key=lambda v: v[1])
        state["cooldowns"] = [(x, y) for t, x, y in words
                              if re.fullmatch(r"\d{2}:\d{2}:\d{2}", t)
                              and .81 < x < .94 and .23 < y < .88]
        for t, x, y in words:
            if t != "挑战" or not (.81 < x < .94 and .23 < y < .88):
                continue
            if not _blue(frame, x - .035, y):
                continue
            nearby = [(abs(name_y - y), name) for name, name_y in state["opponents"]]
            if nearby and min(nearby)[0] < .035:
                state["challenges"].append({"name": min(nearby)[1], "point": (x, y)})
        state["challenges"].sort(key=lambda row: row["point"][1])
    elif {"圣铠", "策略战"} <= texts and any("排位战" in t for t in texts):
        state["page"] = "arena_hub"
    elif {"主线故事", "阿尔克那", "圣铠", "任务"} <= texts:
        state["page"] = "menu"
    elif identify_daily_page(items)["page"] == "home":
        state["page"] = "home"
    return state


@dataclass
class ArenaDaily:
    """One observed action at a time; confirmation is never retried blindly."""
    started: bool = False
    returning: bool = False
    stop_reason: str | None = None
    pending: str | None = None
    selected: str | None = None
    before_attempt: int | None = None
    expected_remaining: int | None = None
    confirmed: int = 0
    victories: int = 0
    initial_remaining: int | None = None
    last_remaining: int | None = None
    last_signature: tuple | None = None
    stable_scrolls: int = 0
    scrolls: int = 0
    returning_steps: int = 0

    def _click(self, name, point, expect):
        if point is None:
            return {"kind": "pause", "reason": f"未识别到按钮：{name}"}
        self.pending = expect
        return {"kind": "click", "name": name, "point": point}

    def step(self, observation):
        page = observation["page"]
        buttons = observation["buttons"]
        remaining = observation["remaining"]
        if remaining is not None and page in {"arena_hub", "arena_list", "arena_prepare"}:
            self.last_remaining = remaining
        if not self.started:
            if page != "home":
                return {"kind": "pause", "reason": "每项日常必须从主界面开始"}
            self.started = True
            return self._click("功能菜单", (.956, .052), "menu")

        # The server can briefly show the list before presenting the result.
        # Wait for the actual result instead of issuing the next challenge.
        if self.pending and page != self.pending:
            if page == "unsupported_dialog":
                return {"kind": "pause", "reason": "遇到未校准的确认框"}
            return {"kind": "wait", "reason": f"等待 {self.pending}"}
        self.pending = None

        if self.returning:
            if page == "home":
                return {"kind": "complete", "reason": self.stop_reason,
                        "returned_home": True, "confirmed": self.confirmed,
                        "victories": self.victories, "remaining": self.last_remaining}
            if page not in {"arena_list", "arena_hub"} or self.returning_steps >= 2:
                return {"kind": "pause", "reason": "返回主界面途中出现未知页面"}
            self.returning_steps += 1
            return self._click("返回", BACK, "arena_hub" if page == "arena_list" else "home")

        if page == "menu":
            return self._click("圣铠", buttons.get("圣铠"), "arena_hub")
        if page == "arena_hub":
            if remaining is None:
                return {"kind": "wait", "reason": "等待圣铠次数识别"}
            if self.initial_remaining is None:
                self.initial_remaining = remaining
            if remaining == 0:
                return self._finish("attempts_exhausted", page)
            # Click the card body. Its footer did not consistently respond in calibration.
            return self._click("策略战", (.50, .53), "arena_list")
        if page == "arena_list":
            if remaining is None:
                return {"kind": "wait", "reason": "等待圣铠次数识别"}
            if self.expected_remaining is not None:
                if remaining != self.expected_remaining:
                    return {"kind": "pause", "reason": "扣次与结算不一致，停止追加挑战"}
                self.expected_remaining = None
            if remaining == 0:
                return self._finish("attempts_exhausted", page)
            if not observation["npc_selected"]:
                return self._click("NPC", buttons.get("NPC"), "arena_list")
            if not observation["senior_selected"]:
                return self._click("老手", buttons.get("老手"), "arena_list")
            if observation["challenges"]:
                target = observation["challenges"][0]
                self.selected = target["name"]
                self.before_attempt = remaining
                self.last_signature = None
                self.stable_scrolls = self.scrolls = 0
                return self._click("挑战", target["point"], "arena_prepare")
            opponents = observation["opponents"]
            cooled = {name for name, y in opponents
                      if any(abs(cy - y) < .035 for _, cy in observation["cooldowns"])}
            signature = tuple((name, round(y, 2)) for name, y in opponents)
            if signature and signature == self.last_signature:
                self.stable_scrolls += 1
            else:
                self.stable_scrolls = 0
            self.last_signature = signature
            # Five known NPCs, each with a visible cooldown, plus a verified list end.
            if self.stable_scrolls >= 2 and cooled == OPPONENTS:
                return self._finish("no_challengeable_opponents", page)
            self.scrolls += 1
            if self.scrolls > 10:
                return {"kind": "pause", "reason": "未能确认整个老手页签已无可挑战对手"}
            return {"kind": "scroll", "point": (.70, .70), "clicks": -12}
        if page == "arena_prepare":
            if remaining is None:
                return {"kind": "wait", "reason": "等待战前次数识别"}
            if remaining == 0:
                return {"kind": "pause", "reason": "战前次数意外归零"}
            if remaining != self.before_attempt or observation.get("prepared_opponent") != self.selected:
                return {"kind": "pause", "reason": "战前对手或次数不匹配"}
            return self._click("跳过战斗", buttons.get("跳过战斗"), "arena_confirm")
        if page == "arena_confirm":
            if not self.selected or not self.before_attempt:
                return {"kind": "pause", "reason": "缺少已核对的挑战上下文"}
            return self._click("确认消耗1个圣铠钥匙", buttons.get("确认"), "arena_result")
        if page == "arena_result":
            self.confirmed += 1
            self.victories += observation["outcome"] == "victory"
            self.expected_remaining = self.before_attempt - 1
            return self._click("关闭结算", buttons.get("点击以继续"), "arena_list")
        return {"kind": "wait", "reason": "等待页面加载或识别"}

    def _finish(self, reason, page):
        self.returning = True
        self.stop_reason = reason
        self.returning_steps = 1
        return self._click("返回", BACK, "arena_hub" if page == "arena_list" else "home")
