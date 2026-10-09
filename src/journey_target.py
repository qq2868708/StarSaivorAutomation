"""Local, explicit journey goals. Normal decisions never require an AI service."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from .runtime_control import RuntimeControl
from .journey_record import normalize_setup


ROOT = Path(__file__).resolve().parent.parent
DIRECTION_LABELS = {"attack": "攻击", "survival": "生存",
                    "comprehensive": "综合", "tactical": "战术"}
PLAYSTYLE_LABELS = {"pve": "PVE（主线）", "pvp": "PVP（圣凯）", "cosmos": "宇宙门"}


class JourneyTargetPending(RuntimeError):
    pass


def load_catalog():
    return json.loads((ROOT / "profiles" / "journey_targets.json").read_text(encoding="utf-8"))


def precheck_target(body, catalog=None):
    catalog = catalog or load_catalog()
    errors, warnings = [], []
    character = str(body.get("character") or "").strip()
    if not character or len(character) > 80:
        errors.append("请输入目标角色（最多 80 字）")
    playstyle = str(body.get("playstyle") or "")
    if playstyle not in PLAYSTYLE_LABELS:
        errors.append("请选择玩法")
    code = str(body.get("combo_code") or "").strip().upper()
    combo = next((item for item in catalog["combos"] if item["code"] == code), None)
    if combo is None:
        errors.append("请选择表格中已核对的技能组合")
    training_direction = str(body.get("training_direction") or "")
    if training_direction not in {"attack", "survival"}:
        errors.append("请选择训练与普通事件的属性方向")
    event_direction = combo.get("event_direction") if combo else None
    if combo and event_direction not in DIRECTION_LABELS:
        errors.append("目标组合缺少已知声援分组，无法自动决策")
    if not isinstance(body.get("has_partner", False), bool):
        errors.append("现有材料状态无效")
    try:
        actual_setup = normalize_setup(body.get("actual_setup") if body.get("actual_setup") is not None else {})
    except ValueError as error:
        actual_setup = {}
        errors.append(str(error))
    matched = next((item for item in catalog["characters"]
                    if character in [item["name"]] + item.get("aliases", [])), None)
    recommendations = matched.get("recommendations", {}).get(playstyle, []) if matched else []
    if not matched:
        warnings.append("该角色推荐尚未收录，按你手动指定的技能和方向执行")
    elif combo and code not in recommendations:
        warnings.append("目标组合不在该角色此玩法的表格推荐中，按你的自定义目标执行")
    if combo and combo.get("note"):
        warnings.append(combo["note"])
    target = {"character": character, "playstyle": playstyle, "combo_code": code,
              "training_direction": training_direction, "event_direction": event_direction,
              "direction_source": "combo_code",
              "required_initial_material": combo.get("initial_material") if combo else None,
              "actual_setup": actual_setup,
              "has_partner": body.get("has_partner", False)}
    return {"ready": not errors, "errors": errors, "warnings": warnings,
            "target": target, "combo": combo, "recommendations": recommendations}


class JourneyTargetStore:
    def __init__(self, root=ROOT):
        self.path = Path(root) / "runtime" / "journey_target.json"

    def read(self):
        if not self.path.exists():
            return None
        try:
            target = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(target, dict):
                raise ValueError("目标文件必须是对象")
            return target
        except (OSError, ValueError) as error:
            raise JourneyTargetPending(f"目标文件无法读取：{error}") from error

    def save(self, body, scope_run_id=None):
        checked = precheck_target(body)
        if not checked["ready"]:
            raise JourneyTargetPending("；".join(checked["errors"]))
        target = {**checked["target"], "revision": uuid.uuid4().hex,
                  "saved_at": datetime.now().isoformat(), "scope_run_id": scope_run_id}
        RuntimeControl._atomic_write(self.path, target)
        return target

    def bind(self, target, run_id):
        # A concurrent editor must not have its new revision overwritten.
        current = self.read()
        if not current or current.get("revision") != target.get("revision"):
            raise JourneyTargetPending("目标在读取期间变化，请重新检查")
        target = {**target, "scope_run_id": run_id}
        RuntimeControl._atomic_write(self.path, target)
        return target


def direction_choice(target, rows):
    if not target:
        raise JourneyTargetPending("训练的方向性需要先填写目标角色和技能组合")
    combo = next((item for item in load_catalog()["combos"] if item["code"] == target.get("combo_code")), None)
    direction = combo.get("event_direction") if combo else None
    if direction not in DIRECTION_LABELS:
        raise JourneyTargetPending("目标技能组合无法匹配声援分组，保持安全暂停")
    terms = {"attack": ("攻击",), "survival": ("生存",),
             "comprehensive": ("综合", "各种状况"), "tactical": ("战术",)}
    hits = {}
    for row_direction, words in terms.items():
        indices = [i for i, row in enumerate(rows)
                   if any(term in row.get("text", "") for term in words)]
        if len(indices) != 1:
            raise JourneyTargetPending("声援选项文字无法逐项核对，保持安全暂停")
        hits[row_direction] = indices[0]
    if len(set(hits.values())) != 4 or len(rows) != 4:
        raise JourneyTargetPending("声援界面与四个已知方向不符")
    index = hits[direction]
    if not rows[index].get("enabled"):
        raise JourneyTargetPending("目标声援方向当前不可选，不改选其他方向")
    return index


MATERIAL_WORDS = {
    "攻击": ("攻击力", "攻击"), "生命": ("生命值", "生命力", "生命"),
    "防御": ("防御力", "防御"), "速度": ("速度",),
    "暴击": ("暴击率", "暴击"), "爆伤": ("暴击伤害", "爆伤"),
    "命中": ("效果命中", "命中"), "抗性": ("效果抗性", "效果抵抗", "抗性"),
    "反击": ("反击几率", "反击概率", "反击"), "吸血": ("吸血",),
    "韧性": ("最大韧性", "韧性"),
}


def equipment_choice(target, texts, equipment_context=False):
    """Return a unique needed material; descriptions of ordinary buffs aren't gear."""
    if not target:
        raise JourneyTargetPending("装备选择需要预输入目标")
    combo = next(item for item in load_catalog()["combos"] if item["code"] == target["combo_code"])
    explicit_code = re.compile(r"(?<![A-Z0-9])(?:[A-H]X[1-8]|XX[1-3])(?![A-Z0-9])", re.I)
    equipment_context = equipment_context or any("装备" in text or explicit_code.search(text) for text in texts)
    if not equipment_context:
        return None
    needed = combo["partner_material"]
    candidates = []
    for index, text in enumerate(texts):
        if any(word in text for word in ('失去', '丢弃', '卸下', '移除', '分解')):
            continue
        codes = [code.upper() for code in explicit_code.findall(text)]
        if codes:
            if codes == [combo["code"]]:
                candidates.append(index)
            continue
        # Read the labelled type or an unambiguous stat line, not the prose effect.
        header = re.split(r"每回合|受到|使用|普通攻击|发生暴击|攻击后", text)[0]
        if "暴击伤害" in header:
            header = header.replace("暴击伤害", "爆伤")
        found = {material for material, words in MATERIAL_WORDS.items()
                 if any(re.search(re.escape(word) + r"(?:[型类]?装备|\s*[:：+＋]\s*\d)", header)
                        or header.strip() == word for word in words)}
        if found == {needed}:
            candidates.append(index)
    if len(candidates) != 1:
        raise JourneyTargetPending(f"需要「{needed}」装备，但当前选项无法唯一核对，未选择")
    return {"index": candidates[0], "material": needed, "combo_code": combo["code"]}
