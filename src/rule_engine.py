"""
规则引擎模块 - 从 SleepRunner 移植的训练决策规则系统

核心概念:
  - TrainingRuleField: 规则可引用的数据字段 (图标数/失败率/属性值)
  - TrainingRuleCard: 单条规则 (field + operator + value → action)
  - TrainingRuleProfile: 规则集合 (有序列表 + legacy策略参数)
  - TrainingDecisionContext: 规则评估所需的运行时数据

引擎流程 (lazy scan):
  1. Probe(context, profile):
     - 依次检查每条规则的 field 是否已知
     - 若 field 未知 → 返回 MissingField，由调用方按需读取
     - 若 field 已知但不满足条件 → 继续下一条
     - 若 field 已知且满足 + action 的 cap 检查 → 返回 Decision
     - 最后走 fallback 规则 (BuiltinDefault)
  2. Evaluate(context, profile):
     - 与 Probe 逻辑相同，但遇到缺失字段直接跳过 (不返回 Missing)
     - 用于全量扫描后的完整评估
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


# ============================================================
# 枚举定义
# ============================================================

class TrainingRuleField(Enum):
    """规则可引用的数据字段"""
    STRENGTH_ICONS = auto()
    STAMINA_ICONS = auto()
    AGILITY_ICONS = auto()
    FOCUS_ICONS = auto()
    GUARD_ICONS = auto()
    STRENGTH_FAIL_RATE = auto()
    STAMINA_FAIL_RATE = auto()
    AGILITY_FAIL_RATE = auto()
    FOCUS_FAIL_RATE = auto()
    GUARD_FAIL_RATE = auto()
    ANY_FAIL_RATE = auto()
    STRENGTH_GAIN = auto()
    STAMINA_GAIN = auto()
    AGILITY_GAIN = auto()
    FOCUS_GAIN = auto()
    GUARD_GAIN = auto()
    ANY_GAIN = auto()
    STRENGTH_STAT = auto()
    STAMINA_STAT = auto()


class TrainingRuleOperator(Enum):
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="


class TrainingDecisionAction(Enum):
    TRAIN_STRENGTH = "train_strength"
    TRAIN_STAMINA = "train_stamina"
    TRAIN_AGILITY = "train_agility"
    TRAIN_FOCUS = "train_focus"
    TRAIN_GUARD = "train_guard"
    TRAIN_BEST_GAIN = "train_best_gain"
    TRAIN_MOST_ICONS = "train_most_icons"
    REST = "rest"
    BUILTIN_DEFAULT = "builtin_default"


class BuildDirection(Enum):
    ATTACK = "attack"
    SURVIVAL = "survival"


# ============================================================
# 数据类
# ============================================================

@dataclass
class TrainingRuleCard:
    """单条规则"""
    id: str = ""
    field: Optional[TrainingRuleField] = None
    operator: Optional[TrainingRuleOperator] = None
    value: Optional[int] = None
    action: TrainingDecisionAction = TrainingDecisionAction.BUILTIN_DEFAULT
    enabled: bool = True
    is_fallback: bool = False


@dataclass
class TrainingLegacyStrategy:
    """内置默认决策的参数"""
    build_direction: BuildDirection = BuildDirection.ATTACK
    fail_rate_threshold: int = 30
    rush_threshold: int = 450


@dataclass
class TrainingRuleProfile:
    """训练规则配置 (可序列化为 JSON)"""
    source_path: str = ""
    legacy_strategy: TrainingLegacyStrategy = field(default_factory=TrainingLegacyStrategy)
    rules: list[TrainingRuleCard] = field(default_factory=list)


@dataclass
class TrainingDecisionContext:
    """规则评估的运行时数据 (由扫描结果填充)"""
    icon_counts: list[int] = field(default_factory=list)
    fail_rates: list[int] = field(default_factory=list)
    gain_values: list[int] = field(default_factory=list)
    is_flash: list[bool] = field(default_factory=list)
    # 游戏训练行右侧的蓝色大拇指标记（优势训练）。
    is_advantage: list[bool] = field(default_factory=list)
    known_icon_mask: Optional[int] = None
    known_fail_rate_mask: Optional[int] = None
    known_gain_mask: Optional[int] = None
    known_flash_mask: Optional[int] = None
    known_advantage_mask: Optional[int] = None
    strength_stat: Optional[int] = None
    stamina_stat: Optional[int] = None
    stamina_pct: float = 100.0
    build_direction: BuildDirection = BuildDirection.ATTACK
    legacy_fail_rate_threshold: int = 30
    legacy_rush_threshold: int = 450
    adventure_gain_threshold: int = 100
    advantage_multiplier: float = 1.35
    profile_name: str = ""
    target_stats: dict = field(default_factory=dict)
    current_stats: dict = field(default_factory=dict)

    FULL_ROW_MASK = 0b11111

    @property
    def any_fail_rate(self) -> Optional[int]:
        """优先属性行的失败率 (用于休息决策: 攻击→力量, 生存→体力)"""
        effective_mask = self._effective_mask(self.fail_rates, self.known_fail_rate_mask)
        if (effective_mask & self.FULL_ROW_MASK) != self.FULL_ROW_MASK:
            return None
        if len(self.fail_rates) < 5:
            return None
        priority_idx = 0 if self.build_direction == BuildDirection.ATTACK else 1
        if not (effective_mask & (1 << priority_idx)):
            return None
        return self.fail_rates[priority_idx]

    @property
    def any_gain(self) -> Optional[int]:
        """优先属性行的增益值 (用于冒险决策: 攻击→力量, 生存→体力)"""
        effective_mask = self._effective_mask(self.gain_values, self.known_gain_mask)
        if effective_mask == 0 or len(self.gain_values) < 5:
            return None
        priority_idx = 0 if self.build_direction == BuildDirection.ATTACK else 1
        if not (effective_mask & (1 << priority_idx)):
            return None
        return self.gain_values[priority_idx]

    def try_get_metric(self, field: TrainingRuleField) -> tuple[bool, int]:
        """安全获取字段值，返回 (known, value)"""
        if field == TrainingRuleField.STRENGTH_ICONS:
            return self._try_get_indexed(self.icon_counts, self.known_icon_mask, 0)
        elif field == TrainingRuleField.STAMINA_ICONS:
            return self._try_get_indexed(self.icon_counts, self.known_icon_mask, 1)
        elif field == TrainingRuleField.AGILITY_ICONS:
            return self._try_get_indexed(self.icon_counts, self.known_icon_mask, 2)
        elif field == TrainingRuleField.FOCUS_ICONS:
            return self._try_get_indexed(self.icon_counts, self.known_icon_mask, 3)
        elif field == TrainingRuleField.GUARD_ICONS:
            return self._try_get_indexed(self.icon_counts, self.known_icon_mask, 4)
        elif field == TrainingRuleField.STRENGTH_FAIL_RATE:
            return self._try_get_indexed(self.fail_rates, self.known_fail_rate_mask, 0)
        elif field == TrainingRuleField.STAMINA_FAIL_RATE:
            return self._try_get_indexed(self.fail_rates, self.known_fail_rate_mask, 1)
        elif field == TrainingRuleField.AGILITY_FAIL_RATE:
            return self._try_get_indexed(self.fail_rates, self.known_fail_rate_mask, 2)
        elif field == TrainingRuleField.FOCUS_FAIL_RATE:
            return self._try_get_indexed(self.fail_rates, self.known_fail_rate_mask, 3)
        elif field == TrainingRuleField.GUARD_FAIL_RATE:
            return self._try_get_indexed(self.fail_rates, self.known_fail_rate_mask, 4)
        elif field == TrainingRuleField.STRENGTH_GAIN:
            return self._try_get_indexed(self.gain_values, self.known_gain_mask, 0)
        elif field == TrainingRuleField.STAMINA_GAIN:
            return self._try_get_indexed(self.gain_values, self.known_gain_mask, 1)
        elif field == TrainingRuleField.AGILITY_GAIN:
            return self._try_get_indexed(self.gain_values, self.known_gain_mask, 2)
        elif field == TrainingRuleField.FOCUS_GAIN:
            return self._try_get_indexed(self.gain_values, self.known_gain_mask, 3)
        elif field == TrainingRuleField.GUARD_GAIN:
            return self._try_get_indexed(self.gain_values, self.known_gain_mask, 4)
        elif field == TrainingRuleField.ANY_GAIN:
            val = self.any_gain
            return (True, val) if val is not None else (False, 0)
        elif field == TrainingRuleField.ANY_FAIL_RATE:
            val = self.any_fail_rate
            return (True, val) if val is not None else (False, 0)
        elif field == TrainingRuleField.STRENGTH_STAT:
            return (True, self.strength_stat) if self.strength_stat is not None else (False, 0)
        elif field == TrainingRuleField.STAMINA_STAT:
            return (True, self.stamina_stat) if self.stamina_stat is not None else (False, 0)
        return (False, 0)

    def with_updated_stats(self, strength_stat: Optional[int], stamina_stat: Optional[int]) -> TrainingDecisionContext:
        """创建更新了 stat 值的新 context (不可变)"""
        return TrainingDecisionContext(
            icon_counts=list(self.icon_counts),
            fail_rates=list(self.fail_rates),
            gain_values=list(self.gain_values),
            is_flash=list(self.is_flash),
            is_advantage=list(self.is_advantage),
            known_icon_mask=self.known_icon_mask,
            known_fail_rate_mask=self.known_fail_rate_mask,
            known_gain_mask=self.known_gain_mask,
            known_flash_mask=self.known_flash_mask,
            known_advantage_mask=self.known_advantage_mask,
            strength_stat=self._merge_non_decreasing(self.strength_stat, strength_stat),
            stamina_stat=self._merge_non_decreasing(self.stamina_stat, stamina_stat),
            stamina_pct=self.stamina_pct,
            build_direction=self.build_direction,
            legacy_fail_rate_threshold=self.legacy_fail_rate_threshold,
            legacy_rush_threshold=self.legacy_rush_threshold,
            adventure_gain_threshold=self.adventure_gain_threshold,
            profile_name=self.profile_name,
            target_stats=dict(self.target_stats),
            current_stats=dict(self.current_stats),
            advantage_multiplier=self.advantage_multiplier,
        )

    @staticmethod
    def _try_get_indexed(values: list[int], known_mask: Optional[int], index: int) -> tuple[bool, int]:
        effective_mask = TrainingDecisionContext._effective_mask(values, known_mask)
        if index < len(values) and (effective_mask & (1 << index)) != 0:
            return (True, values[index])
        return (False, 0)

    @staticmethod
    def _effective_mask(values: list[int], known_mask: Optional[int]) -> int:
        if known_mask is not None:
            return known_mask
        length = min(len(values), 5)
        return 0 if length <= 0 else (1 << length) - 1

    @staticmethod
    def _merge_non_decreasing(previous: Optional[int], current: Optional[int]) -> Optional[int]:
        if current is None:
            return previous
        if previous is None:
            return current
        return max(previous, current)


@dataclass
class TrainingDecisionResult:
    """规则评估的输出"""
    matched_rule_id: str = ""
    action: TrainingDecisionAction = TrainingDecisionAction.REST
    target_row_index: Optional[int] = None
    used_builtin_default: bool = False
    summary: str = ""


@dataclass
class TrainingRuleProbeResult:
    """Probe 的结果: 要么有决策，要么需要更多数据"""
    decision: Optional[TrainingDecisionResult] = None
    missing_field: Optional[TrainingRuleField] = None


# ============================================================
# 属性上限策略 (移植自 RaceStatCapPolicy)
# ============================================================

ATTRIBUTE_CAP = 1250

# Action → cap field 映射
_ACTION_CAP_FIELDS = {
    TrainingDecisionAction.TRAIN_STRENGTH: TrainingRuleField.STRENGTH_STAT,
    TrainingDecisionAction.TRAIN_STAMINA: TrainingRuleField.STAMINA_STAT,
}


def _is_action_blocked_by_cap(action: TrainingDecisionAction, stat: int) -> bool:
    if action == TrainingDecisionAction.TRAIN_STRENGTH:
        return stat >= ATTRIBUTE_CAP
    elif action == TrainingDecisionAction.TRAIN_STAMINA:
        return stat >= ATTRIBUTE_CAP
    return False


# ============================================================
# Action → 行索引映射
# ============================================================

_ACTION_TO_ROW = {
    TrainingDecisionAction.TRAIN_STRENGTH: 0,
    TrainingDecisionAction.TRAIN_STAMINA: 1,
    TrainingDecisionAction.TRAIN_AGILITY: 2,
    TrainingDecisionAction.TRAIN_FOCUS: 3,
    TrainingDecisionAction.TRAIN_GUARD: 4,
}

_ROW_TO_ACTION = {v: k for k, v in _ACTION_TO_ROW.items()}

# Field → 行索引
_FIELD_TO_ROW_INDEX = {
    TrainingRuleField.STRENGTH_ICONS: 0,
    TrainingRuleField.STAMINA_ICONS: 1,
    TrainingRuleField.AGILITY_ICONS: 2,
    TrainingRuleField.FOCUS_ICONS: 3,
    TrainingRuleField.GUARD_ICONS: 4,
    TrainingRuleField.STRENGTH_FAIL_RATE: 0,
    TrainingRuleField.STAMINA_FAIL_RATE: 1,
    TrainingRuleField.AGILITY_FAIL_RATE: 2,
    TrainingRuleField.FOCUS_FAIL_RATE: 3,
    TrainingRuleField.GUARD_FAIL_RATE: 4,
    TrainingRuleField.STRENGTH_GAIN: 0,
    TrainingRuleField.STAMINA_GAIN: 1,
    TrainingRuleField.AGILITY_GAIN: 2,
    TrainingRuleField.FOCUS_GAIN: 3,
    TrainingRuleField.GUARD_GAIN: 4,
}

ROW_MAP = [
    (TrainingDecisionAction.TRAIN_STRENGTH, 0),
    (TrainingDecisionAction.TRAIN_STAMINA, 1),
    (TrainingDecisionAction.TRAIN_AGILITY, 2),
    (TrainingDecisionAction.TRAIN_FOCUS, 3),
    (TrainingDecisionAction.TRAIN_GUARD, 4),
]


# ============================================================
# 规则引擎 (移植自 TrainingRuleEngine)
# ============================================================

class TrainingRuleEngine:
    """规则评估引擎"""

    @staticmethod
    def probe(context: TrainingDecisionContext, profile: TrainingRuleProfile) -> TrainingRuleProbeResult:
        """
        懒加载扫描的规则评估。
        返回: TrainingRuleProbeResult (decision=None 且 missing_field!=None 表示需要读取字段)
        """
        for rule in profile.rules:
            if not rule.enabled:
                continue

            if rule.is_fallback:
                # fallback: 检查 cap
                cap_field = _ACTION_CAP_FIELDS.get(rule.action)
                if cap_field is not None:
                    known, stat = context.try_get_metric(cap_field)
                    if not known:
                        return TrainingRuleProbeResult(missing_field=cap_field)
                    if _is_action_blocked_by_cap(rule.action, stat):
                        continue

                if rule.action == TrainingDecisionAction.BUILTIN_DEFAULT:
                    return TrainingRuleEngine._builtin_probe(context, rule.id)
                if rule.action == TrainingDecisionAction.TRAIN_MOST_ICONS:
                    return TrainingRuleEngine._most_icons_probe(context, rule.id)
                return TrainingRuleProbeResult(
                    decision=TrainingRuleEngine._resolve_result(context, rule, rule.action)
                )

            # 条件规则
            if rule.field is None or rule.operator is None or rule.value is None:
                continue

            known, metric = context.try_get_metric(rule.field)
            if not known:
                return TrainingRuleProbeResult(missing_field=rule.field)

            if not TrainingRuleEngine._matches(metric, rule.operator, rule.value):
                continue

            # 值匹配 → 检查 action 的 cap
            cap_field = _ACTION_CAP_FIELDS.get(rule.action)
            if cap_field is not None:
                known, stat = context.try_get_metric(cap_field)
                if not known:
                    return TrainingRuleProbeResult(missing_field=cap_field)
                if _is_action_blocked_by_cap(rule.action, stat):
                    continue

            if rule.action == TrainingDecisionAction.BUILTIN_DEFAULT:
                return TrainingRuleEngine._builtin_probe(context, rule.id)
            return TrainingRuleProbeResult(
                decision=TrainingRuleEngine._resolve_result(context, rule, rule.action)
            )

        # 无规则命中 → 走内置默认
        return TrainingRuleEngine._builtin_probe(context, "builtin_default")

    @staticmethod
    def evaluate(context: TrainingDecisionContext, profile: TrainingRuleProfile) -> TrainingDecisionResult:
        """
        完整评估 (全量扫描后使用)。
        遇到未知字段时跳过，不会要求重新扫描。
        """
        for rule in profile.rules:
            if not rule.enabled:
                continue

            if rule.is_fallback:
                cap_field = _ACTION_CAP_FIELDS.get(rule.action)
                if cap_field is not None:
                    known, stat = context.try_get_metric(cap_field)
                    if known and _is_action_blocked_by_cap(rule.action, stat):
                        continue

                if rule.action == TrainingDecisionAction.BUILTIN_DEFAULT:
                    return TrainingRuleEngine._builtin_evaluate(context, "builtin_default")
                return TrainingRuleEngine._resolve_result(context, rule, rule.action)

            if rule.field is None or rule.operator is None or rule.value is None:
                continue

            known, metric = context.try_get_metric(rule.field)
            if not known:
                continue

            if not TrainingRuleEngine._matches(metric, rule.operator, rule.value):
                continue

            cap_field = _ACTION_CAP_FIELDS.get(rule.action)
            if cap_field is not None:
                known, stat = context.try_get_metric(cap_field)
                if known and _is_action_blocked_by_cap(rule.action, stat):
                    continue

            if rule.action == TrainingDecisionAction.BUILTIN_DEFAULT:
                return TrainingRuleEngine._builtin_evaluate(context, "builtin_default")
            return TrainingRuleEngine._resolve_result(context, rule, rule.action)

        return TrainingRuleEngine._builtin_evaluate(context, "builtin_default")

    @staticmethod
    def _matches(metric: int, op: TrainingRuleOperator, value: int) -> bool:
        if op == TrainingRuleOperator.GREATER_THAN:
            return metric > value
        elif op == TrainingRuleOperator.GREATER_THAN_OR_EQUAL:
            return metric >= value
        elif op == TrainingRuleOperator.LESS_THAN:
            return metric < value
        elif op == TrainingRuleOperator.LESS_THAN_OR_EQUAL:
            return metric <= value
        return False

    @staticmethod
    def _resolve_result(context: TrainingDecisionContext, rule: TrainingRuleCard,
                        action: TrainingDecisionAction) -> TrainingDecisionResult:
        if action == TrainingDecisionAction.TRAIN_BEST_GAIN:
            target_row = TrainingRuleEngine._find_best_gain_row(context)
        elif action == TrainingDecisionAction.TRAIN_MOST_ICONS:
            target_row = TrainingRuleEngine._find_most_icons_row(context)
        else:
            target_row = _ACTION_TO_ROW.get(action)

        # 配置文件中的图标阈值规则来自旧版策略。若当前画面明确显示了
        # 「优势训练」大拇指标记，则让同一套评分模型参与最终行选择；
        # 只有优势行的有效得分不低于规则命中的行时才改道，失败率/冒险/休息
        # 等更高优先级规则仍在这里之前生效。
        if (action in _ACTION_TO_ROW or
                action == TrainingDecisionAction.TRAIN_MOST_ICONS) and target_row is not None:
            advantage_row = TrainingRuleEngine._best_advantage_row(context)
            advantage_action = (_ROW_TO_ACTION.get(advantage_row)
                                if advantage_row is not None else None)
            advantage_blocked = False
            if advantage_action is not None:
                cap_field = _ACTION_CAP_FIELDS.get(advantage_action)
                if cap_field is not None:
                    known_cap, cap_value = context.try_get_metric(cap_field)
                    advantage_blocked = known_cap and _is_action_blocked_by_cap(
                        advantage_action, cap_value)
            if (advantage_row is not None and advantage_row != target_row and
                    not advantage_blocked and
                    TrainingRuleEngine._score_training_row(context, advantage_row) >=
                    TrainingRuleEngine._score_training_row(context, target_row)):
                target_row = advantage_row
                action = _ROW_TO_ACTION[advantage_row]
                matched_id = f"{rule.id}+advantage"
            else:
                matched_id = rule.id
        else:
            matched_id = rule.id
        return TrainingDecisionResult(
            matched_rule_id=matched_id,
            action=action,
            target_row_index=target_row,
            used_builtin_default=False,
            summary=f"profile={context.profile_name}, matched={matched_id}, action={action.value}, target={target_row}, builtin_default=False",
        )

    # ===================== 最多图标决策 =====================

    @staticmethod
    def _most_icons_probe(context: TrainingDecisionContext, matched_rule_id: str) -> TrainingRuleProbeResult:
        """最多图标决策: 选择支援卡图标最多的行训练，并列时按构建方向决定"""
        for i in range(5):
            field = TrainingRuleEngine._icon_field_for_row(i)
            known, _ = context.try_get_metric(field)
            if not known:
                return TrainingRuleProbeResult(missing_field=field)

        row = TrainingRuleEngine._find_most_icons_row(context)
        return TrainingRuleProbeResult(decision=TrainingRuleEngine._build_result(
            context, matched_rule_id, ROW_MAP[row][0], row, True))

    # ===================== 内置默认逻辑 (移植自 TrainingBuiltInDecision) =====================

    @staticmethod
    def _builtin_probe(context: TrainingDecisionContext, matched_rule_id: str) -> TrainingRuleProbeResult:
        """内置默认决策: 冒险(增益>=阈值) → 失败率检查 → Umaplay 评分选最优"""
        known, priority_fail_rate = context.try_get_metric(TrainingRuleField.ANY_FAIL_RATE)
        if not known:
            return TrainingRuleProbeResult(missing_field=TrainingRuleField.ANY_FAIL_RATE)

        known, priority_gain = context.try_get_metric(TrainingRuleField.ANY_GAIN)
        if not known:
            return TrainingRuleProbeResult(missing_field=TrainingRuleField.ANY_GAIN)

        priority_idx = 1 if context.build_direction == BuildDirection.SURVIVAL else 0
        adventure_threshold = context.adventure_gain_threshold

        # 1. 冒险: 优先行增益 >= 阈值 且 失败率 < 100% → 强制训练 (忽略失败率阈值)
        if priority_gain >= adventure_threshold and priority_fail_rate < 100:
            return TrainingRuleProbeResult(decision=TrainingRuleEngine._build_result(
                context, "adventure_gain", ROW_MAP[priority_idx][0], priority_idx, True))

        # 2. 优先行失败率 > 阈值 → 休息
        if priority_fail_rate > context.legacy_fail_rate_threshold:
            return TrainingRuleProbeResult(decision=TrainingRuleEngine._build_result(
                context, matched_rule_id, TrainingDecisionAction.REST, None, True))

        # 3. 需要全部 5 行数据 → Umaplay 评分
        for i in range(5):
            field = TrainingRuleEngine._icon_field_for_row(i)
            known, _ = context.try_get_metric(field)
            if not known:
                return TrainingRuleProbeResult(missing_field=field)

        best_row = TrainingRuleEngine._score_training_rows(context)
        return TrainingRuleProbeResult(decision=TrainingRuleEngine._build_result(
            context, matched_rule_id, ROW_MAP[best_row][0], best_row, True))

    @staticmethod
    def _builtin_evaluate(context: TrainingDecisionContext, matched_rule_id: str) -> TrainingDecisionResult:
        """内置默认决策的 Evaluate 版本: 缺失字段用 0 兜底, Umaplay 评分"""
        _, priority_fail_rate = context.try_get_metric(TrainingRuleField.ANY_FAIL_RATE)
        _, priority_gain = context.try_get_metric(TrainingRuleField.ANY_GAIN)

        priority_idx = 1 if context.build_direction == BuildDirection.SURVIVAL else 0
        adventure_threshold = context.adventure_gain_threshold

        if priority_gain >= adventure_threshold and priority_fail_rate < 100:
            return TrainingRuleEngine._build_result(
                context, "adventure_gain", ROW_MAP[priority_idx][0], priority_idx, True)

        if priority_fail_rate > context.legacy_fail_rate_threshold:
            return TrainingRuleEngine._build_result(
                context, matched_rule_id, TrainingDecisionAction.REST, None, True)

        best_row = TrainingRuleEngine._score_training_rows(context)
        return TrainingRuleEngine._build_result(
            context, matched_rule_id, ROW_MAP[best_row][0], best_row, True)

    _STAT_ROW_NAMES = ["力量", "体力", "韧性", "专注", "保护"]

    @staticmethod
    def _score_training_rows(context: TrainingDecisionContext) -> int:
        """Umaplay 训练评分并选最高分。

        评分因子为：基础增益 × 支援卡图标加成 × 闪光加成 × 优势训练加成 ×
        体力/失败率惩罚 × 目标降权。优势训练倍率由配置传入 context，默认为
        1.35；未识别到大拇指标记时该因子为 1，不会改变旧行为。
        """
        gains = TrainingRuleEngine._ensure_five_counts(context.gain_values)
        counts = TrainingRuleEngine._ensure_five_counts(context.icon_counts)
        fails = TrainingRuleEngine._ensure_five_counts(context.fail_rates)
        flash = context.is_flash if len(context.is_flash) == 5 else [False] * 5
        advantage = context.is_advantage if len(context.is_advantage) == 5 else [False] * 5
        stamina = context.stamina_pct
        targets = context.target_stats or {}
        current = context.current_stats or {}

        best_row = 0
        best_score = -999.0

        for i in range(5):
            score = TrainingRuleEngine._score_training_row(context, i)

            print(f"  [评分] row={i}: gain={gains[i]}, icons={counts[i]}, "
                  f"flash={flash[i] if i < len(flash) else False}, "
                  f"advantage={advantage[i] if i < len(advantage) else False}, "
                  f"fail={fails[i]}%, stamina={stamina:.0f}% → score={score:.1f}")

            if score > best_score:
                best_score = score
                best_row = i

        return best_row

    @staticmethod
    def _score_training_row(context: TrainingDecisionContext, row_index: int) -> float:
        """计算单行训练分数，供总评分和旧规则的优势行改道共用。"""
        gains = TrainingRuleEngine._ensure_five_counts(context.gain_values)
        counts = TrainingRuleEngine._ensure_five_counts(context.icon_counts)
        fails = TrainingRuleEngine._ensure_five_counts(context.fail_rates)
        flash = context.is_flash if len(context.is_flash) == 5 else [False] * 5
        advantage = context.is_advantage if len(context.is_advantage) == 5 else [False] * 5
        targets = context.target_stats or {}
        current = context.current_stats or {}

        i = max(0, min(4, int(row_index)))
        score = float(gains[i]) if gains[i] > 0 else 1.0

        # 图标人数加成: ≥3人 → ×1.5 (高概率触发闪光训练)
        if counts[i] >= 3:
            score *= 1.5

        # 闪光训练加成: 按钮亮黄 → ×1.3
        if i < len(flash) and flash[i]:
            score *= 1.3

        # 蓝色大拇指标记: 游戏明确标识的优势训练。
        if i < len(advantage) and advantage[i]:
            multiplier = float(getattr(context, "advantage_multiplier", 1.35) or 1.0)
            score *= max(1.0, multiplier)

        # 体力风险惩罚 (训练消耗约 20% 体力)
        expected = context.stamina_pct - 20
        if expected < 30:
            score *= 0.3
        elif expected < 50:
            score *= 0.7

        # 失败率惩罚
        if fails[i] >= 50:
            score *= 0.5
        elif fails[i] >= 30:
            score *= 0.8

        # 属性目标接近降权: 当前值接近目标 → 降低该训练优先级
        stat_name = TrainingRuleEngine._STAT_ROW_NAMES[i]
        target = targets.get(stat_name, 0)
        cur = current.get(stat_name, 0)
        if target > 0 and cur > 0:
            gap = target - cur
            if gap <= 0:
                score *= 0.3
            elif gap <= 10:
                score *= 0.5
            elif gap <= 30:
                score *= 0.8
        return score

    @staticmethod
    def _best_advantage_row(context: TrainingDecisionContext) -> Optional[int]:
        """在已识别的大拇指行中选有效分最高的一行；没有标记则返回 None。"""
        advantage = context.is_advantage if len(context.is_advantage) == 5 else [False] * 5
        rows = [i for i, marked in enumerate(advantage) if marked]
        if not rows:
            return None
        return max(rows, key=lambda i: TrainingRuleEngine._score_training_row(context, i))

    @staticmethod
    def _find_best_gain_row(context: TrainingDecisionContext) -> int:
        """找到增益最高的行索引 (冒险训练用)"""
        counts = TrainingRuleEngine._ensure_five_counts(context.gain_values)
        effective_mask = TrainingDecisionContext._effective_mask(
            context.gain_values, context.known_gain_mask)
        best_row = 0
        best_gain = -1
        for i in range(5):
            if (effective_mask & (1 << i)) and counts[i] > best_gain:
                best_gain = counts[i]
                best_row = i
        return best_row

    @staticmethod
    def _find_most_icons_row(context: TrainingDecisionContext) -> int:
        """找到支援卡图标最多的行索引，并列时按构建方向决定"""
        counts = TrainingRuleEngine._ensure_five_counts(context.icon_counts)
        return TrainingRuleEngine._apply_priority_rule(counts, context.build_direction)

    @staticmethod
    def _icon_field_for_row(row_index: int) -> TrainingRuleField:
        mapping = {
            0: TrainingRuleField.STRENGTH_ICONS,
            1: TrainingRuleField.STAMINA_ICONS,
            2: TrainingRuleField.AGILITY_ICONS,
            3: TrainingRuleField.FOCUS_ICONS,
            4: TrainingRuleField.GUARD_ICONS,
        }
        return mapping[row_index]

    @staticmethod
    def _ensure_five_counts(counts: list[int]) -> list[int]:
        result = [0] * 5
        for i in range(min(len(counts), 5)):
            result[i] = counts[i]
        return result

    @staticmethod
    def _select_best_front_three_except_rush(counts: list[int], rush_row_index: int,
                                             fallback: int = None) -> int:
        if fallback is None:
            fallback = 1 if rush_row_index == 0 else 0
        safe = TrainingRuleEngine._ensure_five_counts(counts)
        selected = -1
        best_count = -1
        for i in range(3):
            if i == rush_row_index:
                continue
            if safe[i] > best_count:
                best_count = safe[i]
                selected = i
        return selected if selected >= 0 else fallback

    @staticmethod
    def _apply_priority_rule(counts: list[int], build_direction: BuildDirection) -> int:
        """
        图标优先级规则 (移植自 TrainingIconCounter.ApplyPriorityRule):
        集中/保护 ≥4 时择优；否则前三项中取最多者；
        并列时按基调: attack → 力量/体力/韧性, survival → 韧性/体力/力量
        """
        if counts[3] >= 4 or counts[4] >= 4:
            return 3 if counts[3] >= counts[4] else 4

        best_count = max(counts[0], counts[1], counts[2])
        tied = [i for i in range(3) if counts[i] == best_count]

        if len(tied) == 1:
            return tied[0]

        preference = [0, 1, 2] if build_direction == BuildDirection.ATTACK else [2, 1, 0]
        for idx in preference:
            if idx in tied:
                return idx
        return tied[0]

    @staticmethod
    def _build_result(context: TrainingDecisionContext, matched_rule_id: str,
                      action: TrainingDecisionAction, target_row_index: Optional[int],
                      used_builtin_default: bool) -> TrainingDecisionResult:
        return TrainingDecisionResult(
            matched_rule_id=matched_rule_id,
            action=action,
            target_row_index=target_row_index,
            used_builtin_default=used_builtin_default,
            summary=(f"profile={context.profile_name}, matched={matched_rule_id}, "
                     f"action={action.value}, target={target_row_index}, "
                     f"builtin_default={used_builtin_default}"),
        )


# ============================================================
# Rule Profile Loader (移植自 TrainingRuleLoader)
# ============================================================

class TrainingRuleLoader:
    """JSON 规则配置的加载/保存"""

    # Field token → enum
    _FIELD_TOKENS = {
        "strength_icons": TrainingRuleField.STRENGTH_ICONS,
        "stamina_icons": TrainingRuleField.STAMINA_ICONS,
        "agility_icons": TrainingRuleField.AGILITY_ICONS,
        "focus_icons": TrainingRuleField.FOCUS_ICONS,
        "guard_icons": TrainingRuleField.GUARD_ICONS,
        "strength_fail_rate": TrainingRuleField.STRENGTH_FAIL_RATE,
        "stamina_fail_rate": TrainingRuleField.STAMINA_FAIL_RATE,
        "agility_fail_rate": TrainingRuleField.AGILITY_FAIL_RATE,
        "focus_fail_rate": TrainingRuleField.FOCUS_FAIL_RATE,
        "guard_fail_rate": TrainingRuleField.GUARD_FAIL_RATE,
        "strength_gain": TrainingRuleField.STRENGTH_GAIN,
        "stamina_gain": TrainingRuleField.STAMINA_GAIN,
        "agility_gain": TrainingRuleField.AGILITY_GAIN,
        "focus_gain": TrainingRuleField.FOCUS_GAIN,
        "guard_gain": TrainingRuleField.GUARD_GAIN,
        "any_gain": TrainingRuleField.ANY_GAIN,
        "any_fail_rate": TrainingRuleField.ANY_FAIL_RATE,
        "strength_stat": TrainingRuleField.STRENGTH_STAT,
        "stamina_stat": TrainingRuleField.STAMINA_STAT,
    }

    _OPERATOR_TOKENS = {
        ">": TrainingRuleOperator.GREATER_THAN,
        ">=": TrainingRuleOperator.GREATER_THAN_OR_EQUAL,
        "<": TrainingRuleOperator.LESS_THAN,
        "<=": TrainingRuleOperator.LESS_THAN_OR_EQUAL,
    }

    _ACTION_TOKENS = {
        "train_strength": TrainingDecisionAction.TRAIN_STRENGTH,
        "train_stamina": TrainingDecisionAction.TRAIN_STAMINA,
        "train_agility": TrainingDecisionAction.TRAIN_AGILITY,
        "train_focus": TrainingDecisionAction.TRAIN_FOCUS,
        "train_guard": TrainingDecisionAction.TRAIN_GUARD,
        "train_best_gain": TrainingDecisionAction.TRAIN_BEST_GAIN,
        "train_most_icons": TrainingDecisionAction.TRAIN_MOST_ICONS,
        "rest": TrainingDecisionAction.REST,
        "builtin_default": TrainingDecisionAction.BUILTIN_DEFAULT,
    }

    _DIRECTION_TOKENS = {
        "attack": BuildDirection.ATTACK,
        "survival": BuildDirection.SURVIVAL,
    }

    @classmethod
    def load_from_json(cls, json_str: str, source_path: str = "") -> TrainingRuleProfile:
        """从 JSON 字符串加载规则配置"""
        data = json.loads(json_str)

        profile = TrainingRuleProfile(source_path=source_path)

        # 加载 legacy_strategy
        if "legacy_strategy" in data and isinstance(data["legacy_strategy"], dict):
            ls = data["legacy_strategy"]
            if "build_direction" in ls:
                direction = ls["build_direction"].lower()
                if direction in cls._DIRECTION_TOKENS:
                    profile.legacy_strategy.build_direction = cls._DIRECTION_TOKENS[direction]
            if "fail_rate_threshold" in ls:
                profile.legacy_strategy.fail_rate_threshold = max(0, min(100, int(ls["fail_rate_threshold"])))
            if "rush_threshold" in ls:
                profile.legacy_strategy.rush_threshold = max(100, min(1200, int(ls["rush_threshold"])))

        # 加载 rules
        used_ids = set()
        has_fallback = False

        if "rules" in data and isinstance(data["rules"], list):
            for idx, rule_data in enumerate(data["rules"]):
                if has_fallback:
                    raise ValueError("Fallback rules must be the last rule in a training profile.")

                has_field = "field" in rule_data
                has_operator = "operator" in rule_data
                has_value = "value" in rule_data
                condition_count = (1 if has_field else 0) + (1 if has_operator else 0) + (1 if has_value else 0)
                is_fallback = condition_count == 0

                if not is_fallback and condition_count != 3:
                    raise ValueError(f"Rule #{idx + 1}: must be fully conditional or fallback")

                # 解析 ID
                rule_id = rule_data.get("id", "")
                if not rule_id:
                    if is_fallback:
                        rule_id = cls._ensure_unique_id("fallback", used_ids)
                    else:
                        field_token = rule_data.get("field", "")
                        op_token = rule_data.get("operator", "")
                        val = rule_data.get("value", 0)
                        rule_id = cls._ensure_unique_id(
                            f"{field_token}_{op_token}_{val}", used_ids)
                else:
                    if rule_id in used_ids:
                        raise ValueError(f"Duplicate rule id '{rule_id}'")
                    used_ids.add(rule_id)

                rule = TrainingRuleCard(
                    id=rule_id,
                    enabled=rule_data.get("enabled", True),
                    is_fallback=is_fallback,
                )

                if is_fallback:
                    has_fallback = True

                if has_field:
                    field_token = rule_data["field"].lower()
                    if field_token not in cls._FIELD_TOKENS:
                        raise ValueError(f"Unknown field '{rule_data['field']}'")
                    rule.field = cls._FIELD_TOKENS[field_token]

                if has_operator:
                    op_token = rule_data["operator"]
                    if op_token not in cls._OPERATOR_TOKENS:
                        raise ValueError(f"Unknown operator '{rule_data['operator']}'")
                    rule.operator = cls._OPERATOR_TOKENS[op_token]

                if has_value:
                    rule.value = int(rule_data["value"])

                if "action" in rule_data:
                    action_token = rule_data["action"].lower()
                    if action_token not in cls._ACTION_TOKENS:
                        raise ValueError(f"Unknown action '{rule_data['action']}'")
                    rule.action = cls._ACTION_TOKENS[action_token]
                elif is_fallback:
                    rule.action = TrainingDecisionAction.BUILTIN_DEFAULT
                else:
                    raise ValueError(f"Rule #{idx + 1}: missing action")

                profile.rules.append(rule)

        # 确保有 fallback
        if not has_fallback:
            rule_id = cls._ensure_unique_id("fallback", used_ids)
            profile.rules.append(TrainingRuleCard(
                id=rule_id,
                action=TrainingDecisionAction.BUILTIN_DEFAULT,
                enabled=True,
                is_fallback=True,
            ))

        return profile

    @classmethod
    def load_from_file(cls, filepath: str) -> TrainingRuleProfile:
        """从文件加载规则配置"""
        with open(filepath, "r", encoding="utf-8") as f:
            json_str = f.read()
        return cls.load_from_json(json_str, source_path=filepath)

    @classmethod
    def create_builtin_default(cls) -> TrainingRuleProfile:
        """创建纯内置默认的 profile (无自定义规则)"""
        profile = TrainingRuleProfile(source_path="builtin_default")
        profile.rules.append(TrainingRuleCard(
            id="builtin_default",
            action=TrainingDecisionAction.BUILTIN_DEFAULT,
            enabled=True,
            is_fallback=True,
        ))
        return profile

    @classmethod
    def _ensure_unique_id(cls, base: str, used: set) -> str:
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    @classmethod
    def field_to_row_index(cls, field: TrainingRuleField) -> Optional[int]:
        """获取字段对应的训练行索引 (仅图标和失败率字段)"""
        return _FIELD_TO_ROW_INDEX.get(field)

    @classmethod
    def action_to_row_index(cls, action: TrainingDecisionAction) -> Optional[int]:
        """获取 action 对应的训练行索引"""
        return _ACTION_TO_ROW.get(action)


# ============================================================
# Rule Profile Manager (移植自 TrainingRuleProfileManager)
# ============================================================

class TrainingRuleProfileManager:
    """管理训练规则配置的加载/切换"""

    DEFAULT_PROFILE = "speed"
    PROFILES_DIR = Path("profiles/training")

    def __init__(self):
        self._current = self.DEFAULT_PROFILE
        self._profile_cache: dict[str, TrainingRuleProfile] = {}
        self._builtin_default: Optional[TrainingRuleProfile] = None

    @property
    def current_profile_name(self) -> str:
        return self._current

    @property
    def current_profile(self) -> TrainingRuleProfile:
        """获取当前激活的 profile (优先加载文件，回退到内置默认)"""
        if self._current in self._profile_cache:
            return self._profile_cache[self._current]

        filepath = self._resolve_path(self._current)
        if os.path.isfile(filepath):
            try:
                profile = TrainingRuleLoader.load_from_file(filepath)
                self._profile_cache[self._current] = profile
                return profile
            except Exception as e:
                print(f"[RuleProfile] 加载失败 '{filepath}': {e}, 使用内置默认")

        return self.builtin_default

    @property
    def builtin_default(self) -> TrainingRuleProfile:
        if self._builtin_default is None:
            self._builtin_default = TrainingRuleLoader.create_builtin_default()
        return self._builtin_default

    def set_profile(self, name: str):
        cleaned = name.strip()
        if cleaned.endswith(".json"):
            cleaned = cleaned[:-5]
        if not cleaned:
            cleaned = self.DEFAULT_PROFILE

        if cleaned != self._current:
            self._current = cleaned
            print(f"[RuleProfile] 训练策略 -> '{cleaned}'")

    def list_profiles(self) -> list[str]:
        """列出所有可用的 profile 文件名 (不含扩展名)"""
        if not self.PROFILES_DIR.is_dir():
            return []
        profiles = []
        for f in self.PROFILES_DIR.glob("*.json"):
            profiles.append(f.stem)
        return sorted(profiles)

    def _resolve_path(self, name: str) -> str:
        return str(self.PROFILES_DIR / f"{name}.json")

    def invalidate_cache(self):
        self._profile_cache.clear()


# ============================================================
# 全局单例
# ============================================================

_rule_profile_manager: Optional[TrainingRuleProfileManager] = None


def get_profile_manager() -> TrainingRuleProfileManager:
    global _rule_profile_manager
    if _rule_profile_manager is None:
        _rule_profile_manager = TrainingRuleProfileManager()
    return _rule_profile_manager
