"""Validated user overrides and evidence-based journey decision progress."""
import copy
import re
from .journey_plan import default_goals, GOAL_MAP, ACTION_POLICIES, goal_instruction, observe_goal


SHOP_TYPES = {
    "mood_recovery": "心情恢复", "stamina_recovery": "耐力恢复",
    "potential_points": "潜质点数", "vitality_potion": "活力药水",
    "training_exp": "训练经验", "attribute_boost": "属性增加",
    "status_buff": "状态增益", "random_effect": "随机效果",
    "consumable": "其他消耗品", "money_gain": "古币获得",
}
TURN_ACTIONS = {
    "inherit": "继承默认", "rules": "按规则", "train": "训练（自动选属性）",
    "train_strength": "力量训练", "train_stamina": "体力训练",
    "train_agility": "韧性训练", "train_focus": "集中训练",
    "train_guard": "保护训练", "rest": "休息",
    "subjugation": "讨伐", "commission": "普通委托",
}
DEFAULTS = {
    "notifications": {"enabled": True, "volume": 50,
                      "events": {"journey_end": True, "paused_safe": False, "recovered": False}},
    "purchase_rules": {"default": "auto", "categories": {}, "items": {}},
    "turn_plan": {"enabled": False, "start_turn": None, "default_action": "rules",
                  "after_void_action": "rules", "turns": {}, "goals": default_goals(),
                  "subjugation_policy": "prefer", "commission_policy": "skip"},
}


def preferences(config):
    result = copy.deepcopy(DEFAULTS)
    sources = {"notifications": config.get("notifications", {}),
               "purchase_rules": config.get("shop", {}).get("purchase_rules", {}),
               "turn_plan": config.get("turn_plan", {})}
    for section, source in sources.items():
        if isinstance(source, dict):
            result[section].update(copy.deepcopy(source))
    result["notifications"]["events"] = {
        **DEFAULTS["notifications"]["events"], **result["notifications"].get("events", {})}
    goals = default_goals()
    for goal_id, setting in result["turn_plan"].get("goals", {}).items():
        goals[goal_id] = {**goals.get(goal_id, {}), **setting} if isinstance(setting, dict) else setting
    result["turn_plan"]["goals"] = goals
    return validate_preferences(result)


def validate_preferences(body):
    if not isinstance(body, dict) or set(body) != set(DEFAULTS):
        raise ValueError("配置必须包含音效、购买策略和回合计划")
    result = copy.deepcopy(body)
    for section in DEFAULTS:
        if not isinstance(result[section], dict) or set(result[section]) != set(DEFAULTS[section]):
            raise ValueError(f"配置字段无效：{section}")
    sound, shop, plan = (result[key] for key in DEFAULTS)
    if type(sound["enabled"]) is not bool or type(sound["volume"]) is not int or not 0 <= sound["volume"] <= 100:
        raise ValueError("音效开关或音量无效")
    if (not isinstance(sound["events"], dict) or set(sound["events"]) != set(DEFAULTS["notifications"]["events"])
            or any(type(v) is not bool for v in sound["events"].values())):
        raise ValueError("提示音事件无效")
    modes = {"auto", "buy", "skip"}
    if shop["default"] not in modes:
        raise ValueError("默认购买策略无效")
    for name in ("categories", "items"):
        rules = shop[name]
        if not isinstance(rules, dict) or len(rules) > 500:
            raise ValueError("购买规则过多或格式无效")
        for key, mode in rules.items():
            if (not isinstance(key, str) or not key.strip() or len(key) > 120 or mode not in modes
                    or name == "categories" and key not in SHOP_TYPES):
                raise ValueError("商品或类别购买策略无效")
        normalized = {re.sub(r"\s+", "", key): value for key, value in rules.items()}
        if len(normalized) != len(rules):
            raise ValueError("商品名去除空白后不能重复")
        shop[name] = normalized
    if type(plan["enabled"]) is not bool:
        raise ValueError("回合计划开关无效")
    start = plan["start_turn"]
    if start is not None and (type(start) is not int or not 1 <= start <= 45):
        raise ValueError("起始回合须为1至45，或自动识别")
    for key in ("default_action", "after_void_action"):
        if plan[key] not in TURN_ACTIONS or plan[key] == "inherit":
            raise ValueError("默认或阶段行为无效")
    if not isinstance(plan["turns"], dict):
        raise ValueError("逐回合行为须为对象")
    for turn, action in plan["turns"].items():
        if not isinstance(turn, str) or turn not in {str(i) for i in range(1, 46)} or action not in TURN_ACTIONS:
            raise ValueError("逐回合行为无效")
    for key in ("subjugation_policy", "commission_policy"):
        if plan[key] not in ACTION_POLICIES:
            raise ValueError("讨伐或普通委托策略无效")
    if not isinstance(plan["goals"], dict) or set(plan["goals"]) != set(GOAL_MAP):
        raise ValueError("旅程目标列表无效")
    for goal_id, goal in plan["goals"].items():
        if not isinstance(goal, dict) or set(goal) != set(default_goals()[goal_id]):
            raise ValueError("目标内回合配置无效")
        count = goal["turn_count"]
        if type(count) is not int or not 1 <= count <= 60:
            raise ValueError("目标回合数量须为1至60")
        if goal["default_action"] not in TURN_ACTIONS:
            raise ValueError("目标默认行为无效")
        for key in ("subjugation_policy", "commission_policy"):
            if goal[key] not in {"inherit", *ACTION_POLICIES}:
                raise ValueError("目标行动策略无效")
        if not isinstance(goal["turns"], dict) or any(
                key not in {str(i) for i in range(1, count + 1)} or value not in TURN_ACTIONS
                for key, value in goal["turns"].items()):
            raise ValueError("目标内回合行为超出范围")
    return result


def purchase_mode(rules, item):
    names = [item.get(key, "") for key in ("profile_name", "display_name", "name")]
    matches = [(name, rules.get("items", {}).get(re.sub(r"\s+", "", str(name)))) for name in names]
    # A deny on either the observed or canonical name always wins.
    for mode in ("skip", "buy", "auto"):
        for name, value in matches:
            if value == mode:
                return mode, f"item:{name}"
    category = item.get("type", "")
    if category in rules.get("categories", {}):
        return rules["categories"][category], f"category:{category}"
    return rules.get("default", "auto"), "default"


def turn_instruction(plan, turn, after_void=False, progress=None):
    if progress is not None and progress.get("goal_id") and plan.get("goals"):
        return goal_instruction(plan, progress)
    if not plan.get("enabled"):
        return {"action": "rules", "source": "disabled", "turn": turn}
    explicit = plan.get("turns", {}).get(str(turn), "inherit")
    if explicit != "inherit":
        return {"action": explicit, "source": "turn", "turn": turn}
    key = "after_void_action" if after_void else "default_action"
    return {"action": plan.get(key, "rules"), "source": key, "turn": turn}


class JourneyProgress:
    """Count confirmed choices, never dispatcher iterations or attempted clicks."""
    def __init__(self, start_turn=None, saved=None):
        self.data = {"turn": start_turn, "source": "configured" if start_turn else "unknown",
                     "after_void": False, "battle": None, "pending": None, "date": None,
                     "last_evidence": None, "goal_id": None, "goal_turn": None, "goal_source": "unknown"}
        if saved:
            self.data.update(copy.deepcopy(saved))
            if self.data["turn"] is None and start_turn is not None:
                self.data.update(turn=start_turn, source="configured")

    def prepare(self, kind):
        if not self.data["pending"] or self.data["pending"]["kind"] != kind:
            self.data["pending"] = {"kind": kind, "turn": self.data["turn"], "date": self.data["date"]}

    def observe(self, items, page, observation_id=None, size=(1600, 900), goals=None):
        observe_goal(self.data, items, page, goals, size)
        texts = [str(item[0]) for item in items if len(item) > 1 and item[1] >= .85]
        text = re.sub(r"\s+", "", " ".join(texts))
        fractions = {int(m[1]) for t in texts for m in re.finditer(r"(?<!\d)(\d{1,2})\s*/\s*45(?!\d)", t)
                     if 1 <= int(m[1]) <= 45}
        dates = {m[0] for t in texts if (m := re.search(r"(?:[1-9]|1[0-2])月(?:上旬|中旬|下旬)", t))}
        date = next(iter(dates)) if len(dates) == 1 else None
        pending = self.data["pending"]
        if len(fractions) == 1:
            turn = next(iter(fractions))
            if self.data["turn"] is None or turn >= self.data["turn"]:
                self.data.update(turn=turn, source="ocr")
            if pending and pending["turn"] is not None and turn > pending["turn"]:
                self.data["pending"] = None
                pending = None
        if pending:
            confirmed = (
                pending["kind"] == "train" and page in {"训练结果", "属性结算"}
                or pending["kind"] == "rest" and "点击以继续" in text
                and any(word in text for word in ("露宿", "住处", "冥想室"))
                and any(word in text.upper() for word in ("SUCCESS", "FAIL"))
                or date is not None and pending["date"] is not None and date != pending["date"])
            if confirmed:
                if pending["turn"] is not None:
                    self.data.update(turn=min(46, pending["turn"] + 1), source="confirmed_action")
                self.data["pending"] = None
                self.data["last_evidence"] = observation_id
        if date:
            self.data["date"] = date
        if page == "评鉴战":
            if "虚空评鉴战" in text:
                self.data["battle"] = "void"
            elif any(word in text for word in ("竞争评鉴战", "最终评鉴战", "远征评鉴战")):
                self.data["battle"] = "other"
        if self.data["battle"] == "void" and page in {"战后离开", "战斗失败重试", "战斗结果剧情"}:
            if any(word in text.upper() for word in ("VICTORY", "DEFEAT", "评鉴战胜利", "评鉴战失败")):
                self.data.update(after_void=True, last_evidence=observation_id)
        return copy.deepcopy(self.data)
