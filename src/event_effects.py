"""量化旅程事件效果，并按当前构筑方向比较选项。

表格里直接给出的属性/资源数字会参与评分；潜能、职业限制、状态名称等
没有统一数值的内容会保留为定性字段，只在数字效果相同的时候作为决胜依据。
"""
from __future__ import annotations

import re
from typing import Any


# 游戏中的“体力”是永久生命属性；“耐力”是训练体力条，不能混为一谈。
STAT_ALIASES = {
    "力量": "力量",
    "体力": "体力",
    "生命": "体力",
    "韧性": "韧性",
    "专注": "专注",
    "集中": "专注",
    "保护": "保护",
}
STAMINA_ALIASES = ("耐力", "忍耐")

# 权重写入代码是为了让日志和测试可以复现。用户仍可通过 config 的
# events.decision_weights 覆盖它们；这些只是默认的攻击向/生存向排序。
DEFAULT_DIRECTION_WEIGHTS = {
    "attack": {"力量": 5.0, "体力": 4.0, "韧性": 4.0, "专注": 3.0, "保护": 2.0},
    "survival": {"韧性": 5.0, "体力": 4.0, "保护": 4.0, "力量": 3.0, "专注": 2.0},
}


def _number_after(text: str, label: str) -> int:
    """Read a signed number in either ``力量+10`` or ``+10 力量`` form."""
    escaped = re.escape(label)
    patterns = (
        rf"{escaped}\s*(?:[:：])?\s*([+-])\s*(\d+)",
        rf"([+-])\s*(\d+)\s*{escaped}",
        rf"(?<!\d)(\d+)\s*{escaped}",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if len(match.groups()) == 2:
            sign, number = match.groups()
            value = int(number)
            return value if sign == "+" else -value
        return int(match.group(1))
    return 0


def _first_number_after(text: str, labels: tuple[str, ...]) -> int:
    for label in labels:
        value = _number_after(text, label)
        if value:
            return value
    return 0


def quantify_effect_text(effect_text: str) -> dict[str, Any]:
    """从表格原文提取可计算的数字，同时保留不可量化效果。"""
    text = str(effect_text or "").replace("\ufeff", "")
    stats: dict[str, int] = {}
    for alias, canonical in STAT_ALIASES.items():
        delta = _number_after(text, alias)
        if delta:
            stats[canonical] = stats.get(canonical, 0) + delta

    stamina_delta = 0
    for alias in STAMINA_ALIASES:
        stamina_delta += _number_after(text, alias)

    potential_points = _first_number_after(text, ("潜质点数", "潜质点", "潜力", "技能点"))
    mood_delta = _number_after(text, "心情")
    status_points = _number_after(text, "状态")
    currency_delta = _first_number_after(text, ("硬币", "银币", "金币"))

    relic_names: list[str] = []
    for name in re.findall(r"(?:获得遗物|选择遗物)\s*[:：]?\s*([^\n【】]+)", text):
        name = name.strip(" ：:，,；; ")
        if name and name not in relic_names:
            relic_names.append(name)

    potential_names: list[str] = []
    for name in re.findall(r"『([^』]+)』", text):
        if name not in potential_names:
            potential_names.append(name)
    # 有些表格单元格使用“潜能/潜力”但没有书名号。
    for name in re.findall(r"(?:潜能|潜力)\s*[:：]?\s*([\u4e00-\u9fffA-Za-z0-9（）()！!？?]+)", text):
        name = name.strip("（）()【】[] ，,；; ")
        if name and name not in potential_names:
            potential_names.append(name)

    # Some rows omit the words “潜能/潜力” and leave the named effect after
    # the numeric block. Keep common names so they still break score ties.
    common_potentials = (
        "攻击感知", "保护感知", "体力感知", "集中感知", "速度感知", "防御感知",
        "攻击技巧", "保护技巧", "集中技巧", "速度技巧", "生命技巧", "攻击天赋",
        "生命天赋", "专注天赋", "洞察天赋", "洞察技巧", "破坏技巧", "锐利攻势",
        "第二颗心脏", "急救措施", "从容应对", "强韧意志", "防御意识", "感官性守备",
        "洞察圣痕", "攻击圣痕", "速度星痕", "洞察星痕", "坚固的破坏", "敏捷姿态",
        "触发层叠", "生命圣痕", "速度天赋", "星光恩宠", "速度的圣痕", "目标校准魔法阵",
        "灾难应对协议", "紧急咒语咏唱法", "深红誓约", "命运之路", "笑声触发机制",
        "战斗热意", "催化剂", "随机应变", "双连击", "弹力加速", "侵占的暗影",
        "清酿之香", "爱恋行进", "作战指令", "责任感", "糖分亢奋", "特长是访查",
    )
    for name in common_potentials:
        if name in text and name not in potential_names:
            potential_names.append(name)

    discounts: dict[str, float] = {}
    for match in re.finditer(r"『([^』]+)』[^\n（）()]*[（(]?\s*(\d+)\s*折", text):
        discounts[match.group(1)] = int(match.group(2)) / 10.0
    for name in potential_names:
        match = re.search(rf"{re.escape(name)}[^\n（）()]*[（(]?\s*(\d+)\s*折", text)
        if match:
            discounts[name] = int(match.group(1)) / 10.0

    duration = None
    duration_match = re.search(r"(\d+)\s*回合", text)
    if duration_match:
        duration = int(duration_match.group(1))

    buffs: list[dict[str, Any]] = []
    buff_match = re.search(r"(?:旅行|旅程)?\s*(?:buff|BUFF|增益)\s*[:：]?\s*([^\n]+)", text)
    if buff_match:
        buff_name = re.split(r"[；;]", buff_match.group(1), maxsplit=1)[0]
        buff_name = re.sub(r"\s*持续\s*\d+\s*回合", "", buff_name)
        buff_name = buff_name.strip().replace("﻿", "")
        if buff_name:
            buffs.append({"name": buff_name, "duration_rounds": duration})
    if not buffs:
        gain_match = re.search(r"(?:旅行状态|旅程增益)\s*[:：]?\s*([^（\n]+)", text)
        if gain_match:
            buff_name = gain_match.group(1).strip(" ：:，,；; ")
            if buff_name:
                buffs.append({"name": buff_name, "duration_rounds": duration})

    role_requirement = None
    for role in ("游侠", "坦克", "术师", "法师", "刺客", "大剑", "辅助"):
        if f"{role}专用" in text or f"{role}专属" in text:
            role_requirement = role
            break

    unknown_effects: list[str] = []
    status_match = re.search(r"(?:旅行状态|旅程状态)\s*[:：]?\s*([^（\n]+)", text)
    if status_match:
        unknown_effects.append(status_match.group(1).strip())

    result: dict[str, Any] = {
        "stats": stats,
        "stamina_delta": stamina_delta,
        "potential_points": potential_points,
        "mood_delta": mood_delta,
        "status_points": status_points,
        "currency_delta": currency_delta,
        "relic_names": relic_names,
        "potential_names": potential_names,
        "potential_discounts": discounts,
        "buffs": buffs,
        "role_requirement": role_requirement,
        "unknown_effects": unknown_effects,
    }
    return result


def option_score(option: dict[str, Any], direction: str = "attack",
                 weights: dict[str, float] | None = None) -> float:
    """只用表格给出的数字计算基础分；潜能数量仅作为平局决胜。"""
    quantified = option.get("quantified") or quantify_effect_text(option.get("effect_text", ""))
    direction = direction if direction in DEFAULT_DIRECTION_WEIGHTS else "attack"
    attr_weights = dict(DEFAULT_DIRECTION_WEIGHTS[direction])
    if isinstance(weights, dict):
        attr_weights.update({str(k): float(v) for k, v in weights.items()})

    score = sum(float(value) * attr_weights.get(name, 0.0)
                for name, value in quantified.get("stats", {}).items())
    # 耐力/心情/状态/潜质点数的数字来源相同，但它们不是永久属性，
    # 因此使用明确较低的资源权重。
    score += float(quantified.get("stamina_delta", 0)) * 0.25
    score += float(quantified.get("potential_points", 0)) * 0.5
    score += float(quantified.get("mood_delta", 0)) * 5.0
    score += float(quantified.get("status_points", 0)) * 2.0
    score += float(quantified.get("currency_delta", 0)) * 0.5
    score += len(quantified.get("relic_names", [])) * 4.0
    return round(score, 2)


def score_options(event: dict[str, Any], direction: str = "attack",
                  weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """返回每个选项的可审计评分，按原选项顺序保留。"""
    scored = []
    for position, option in enumerate(event.get("options", []), start=1):
        quantified = option.get("quantified") or quantify_effect_text(option.get("effect_text", ""))
        score = option_score({**option, "quantified": quantified}, direction, weights)
        scored.append({
            "index": int(option.get("index", position)),
            "score": score,
            "numeric_effect": quantified,
            "potential_count": len(quantified.get("potential_names", [])),
            "buff_count": len(quantified.get("buffs", [])),
            "unknown_count": len(quantified.get("unknown_effects", [])),
        })
    return scored


def choose_option(event: dict[str, Any], direction: str = "attack",
                  weights: dict[str, float] | None = None) -> tuple[int | None, list[dict[str, Any]]]:
    """按数字分数选择；相同分数时依次看潜能、buff、未知副作用，再看原顺序。"""
    scored = score_options(event, direction, weights)
    if not scored:
        return None, []
    best = max(scored, key=lambda item: (
        item["score"], item["potential_count"], item["buff_count"],
        -item["unknown_count"], -item["index"],
    ))
    return int(best["index"]), scored

