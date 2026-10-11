"""Goal-local action plans. Counts are backed by the recorded starlight route."""
import copy
import re

GOALS = [
    {"id": "basic", "name": "基础评鉴战", "turns": 6, "months": [3, 4]},
    {"id": "subjugation_exam", "name": "讨伐评鉴战", "turns": 6, "months": [5, 6]},
    {"id": "expedition", "name": "远征评鉴战", "turns": 6, "months": [7, 8]},
    {"id": "competition", "name": "竞争评鉴战", "turns": 6, "months": [9, 10]},
    {"id": "camp", "name": "完成集训", "turns": 6, "months": [11, 12]},
    {"id": "void", "name": "虚空评鉴战", "turns": 3, "months": [1]},
    {"id": "final", "name": "最终评鉴战", "turns": 3, "months": [2]},
]
GOAL_MAP = {goal["id"]: goal for goal in GOALS}
ACTION_POLICIES = {"prefer": "优先执行", "allow": "按回合安排", "skip": "不执行"}


def default_goals():
    return {goal["id"]: {"turn_count": goal["turns"], "default_action": "inherit",
                         "subjugation_policy": "inherit", "commission_policy": "inherit", "turns": {}}
            for goal in GOALS}


def goal_instruction(plan, progress):
    goal_id, local_turn = progress.get("goal_id"), progress.get("goal_turn")
    goal = plan.get("goals", {}).get(goal_id, {})
    action = goal.get("turns", {}).get(str(local_turn), "inherit")
    source = "goal_turn"
    if action == "inherit":
        action, source = plan.get("turns", {}).get(str(progress.get("turn")), "inherit"), "legacy_turn"
    if action == "inherit":
        action, source = goal.get("default_action", "inherit"), "goal"
    if action == "inherit":
        key = "after_void_action" if progress.get("after_void") else "default_action"
        action, source = plan.get(key, "rules"), key
    result = {"action": action if plan.get("enabled") else "rules",
              "source": source if plan.get("enabled") else "disabled",
              "goal_id": goal_id, "goal_turn": local_turn, "turn": progress.get("turn")}
    for key, default in (("subjugation_policy", "prefer"), ("commission_policy", "skip")):
        value = goal.get(key, "inherit") if plan.get("enabled") else "inherit"
        result[key] = plan.get(key, default) if value == "inherit" else value
    return result


def menu_choice(instruction, available):
    """Hard bans are never bypassed by a fallback or an explicit turn choice."""
    allowed = set(available)
    for kind in ("subjugation", "commission"):
        if instruction.get(kind + "_policy") == "skip":
            allowed.discard(kind)
    for kind in ("subjugation", "commission"):
        if kind in allowed and instruction.get(kind + "_policy") == "prefer":
            return kind, "preferred_activity"
    action = instruction.get("action", "rules")
    wanted = "train" if action.startswith("train") else action
    if wanted in allowed:
        return wanted, "planned_activity"
    for kind in ("train", "rest", "subjugation", "commission"):
        if kind in allowed:
            return kind, "rules" if wanted in {"rules", "inherit"} else "unavailable_or_forbidden_fallback"
    return None, "no_allowed_activity"


def observe_goal(data, items, page, goals=None, size=(1600, 900)):
    definitions = goals or default_goals()
    rows = []
    for text, confidence, *rest in items:
        if confidence < .85:
            continue
        box = rest[0] if rest else None
        center = None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            center = (sum(p[0] for p in box) / 4 * 1600 / size[0],
                      sum(p[1] for p in box) / 4 * 900 / size[1])
        rows.append((re.sub(r"\s+", "", text), center))
    # Preparation banners identify a target; completed checklist text does not.
    titles = {goal["id"] for goal in GOALS for text, _ in rows
              if text in {goal["name"] + "准备", goal["name"] + "即将开始"}}
    hud_dates = []
    remaining = []
    for text, center in rows:
        if center and center[0] < 470 and center[1] < 125:
            match = re.search(r"(?<!\d)(1[0-2]|[1-9])月(上旬|中旬|下旬)", text)
            if match:
                hud_dates.append((int(match[1]), ("上旬", "中旬", "下旬").index(match[2])))
            if center[0] < 180 and center[1] < 115 and re.fullmatch(r"\d{1,2}", text):
                remaining.append(int(text))
    if len(set(hud_dates)) == 1:
        month, part = hud_dates[0]
        candidate = next(goal for goal in GOALS if month in goal["months"])
        goal_id = candidate["id"]
        index = candidate["months"].index(month) * 3 + part + 1
        # Countdown determines the local slot when a user changes this route's budget.
        count = definitions.get(goal_id, {}).get("turn_count", candidate["turns"])
        expected = count - index + 1
        if len(set(remaining)) == 1 and 1 <= remaining[0] <= count:
            expected = remaining[0]
            index = count - expected + 1
            data.update(goal_id=goal_id, goal_turn=index, goal_source="hud_countdown",
                        goal_remaining=expected)
        elif not remaining and count == candidate["turns"]:
            data.update(goal_id=goal_id, goal_turn=index, goal_source="hud_calendar",
                        goal_remaining=expected)
    elif len(titles) == 1:
        goal_id = next(iter(titles))
        if data.get("goal_id") != goal_id:
            data.update(goal_id=goal_id, goal_turn=None, goal_source="goal_banner")
    return copy.deepcopy(data)
