"""
AI 决策引擎 - 训练选择 + 休息决策

支持两种模式:
  1. 规则引擎模式 (推荐): 使用 JSON 规则配置 + 内置默认逻辑
     - 参考 SleepRunner 的 TrainingRuleEngine + TrainingBuiltInDecision
     - 支持图标计数、每行独立失败率、属性上限检测等高级特性
  2. 传统权重模式: 加权评分 + 优先级检查

整体决策流程:
  1. 读取游戏状态 (体力/心情/金钱/年份)
  2. 读取训练选项
  3. 优先属性达标 → 训练
  4. 优先属性不达标:
     a. 失败率超标 + 没好属性 → 休息
     b. 第一年 + 没好属性 + 心情非Best → 休息(方式2)
     c. 否则 → 评分选最优训练

休息选择逻辑:
  1. 心情Best 且 体力>40% → 免费休息
  2. 金钱 >= 60 → 高级休息($60)
  3. 其他 → 付费休息($30)
"""
from __future__ import annotations

from typing import Optional
from .rule_engine import (
    TrainingDecisionContext,
    TrainingDecisionResult,
    TrainingRuleProfile,
    TrainingRuleEngine,
    TrainingRuleLoader,
    TrainingDecisionAction,
    BuildDirection,
    get_profile_manager,
)


class GameState:
    """当前游戏状态"""

    def __init__(self, stamina=100, mood="Normal", money=0, round_num=1):
        self.stamina = stamina           # 体力值 (0-100)
        self.mood = mood                 # 心情 (Worst/Bad/Normal/Good/Best)
        self.money = money               # 金钱
        self.round_num = round_num       # 当前训练轮次

    @property
    def failure_rate(self):
        """从体力值计算失败率: 失败率 = 100% - 体力%"""
        return max(0, 100 - self.stamina)

    @property
    def mood_level(self):
        """心情等级 (0-4)"""
        mapping = {"Worst": 0, "Bad": 1, "Normal": 2, "Good": 3, "Best": 4}
        return mapping.get(self.mood, 2)

    def __repr__(self):
        return (f"GameState(体力={self.stamina}%, 失败率={self.failure_rate}%, "
                f"心情={self.mood}, 金钱=${self.money}, 轮次={self.round_num})")


class TrainingOption:
    """单个训练选项"""

    def __init__(self, attr_id, attr_name, gain, failure_rate):
        self.attr_id = attr_id
        self.attr_name = attr_name
        self.gain = gain
        self.failure_rate = failure_rate

    def __repr__(self):
        return (f"TrainingOption({self.attr_name}, "
                f"增益={self.gain}, 失败率={self.failure_rate}%)")


class RestOption:
    """单个休息选项"""

    def __init__(self, rest_id, name, cost, stamina_recover, description=""):
        self.rest_id = rest_id              # 1/2/3
        self.name = name
        self.cost = cost                    # 金钱消耗
        self.stamina_recover = stamina_recover  # 恢复体力%
        self.description = description

    def __repr__(self):
        return f"RestOption({self.name}, 花费${self.cost}, 恢复{self.stamina_recover}%)"


class DecisionEngine:
    """AI 训练决策引擎"""

    def __init__(self, config):
        training_cfg = config.get("training", {})
        rest_cfg = config.get("rest", {})
        mood_cfg = config.get("mood", {})
        attr_cfg = config.get("attributes", {})

        # 规则引擎模式
        self.rule_profile_name = training_cfg.get("rule_profile", "")
        self.use_per_row_fail_rate = training_cfg.get("per_row_fail_rate", False)
        self.use_icon_counting = training_cfg.get("icon_counting", False)
        self.fail_rate_regions = training_cfg.get("fail_rate_regions", [])
        self.adventure_gain_threshold = training_cfg.get("adventure_gain_threshold", 100)
        self._rule_profile: Optional[TrainingRuleProfile] = None

        # 传统模式参数
        self.priority = training_cfg.get("priority", 1)
        self.failure_threshold = training_cfg.get("failure_threshold", 30)
        self.min_gain_threshold = training_cfg.get("min_gain_threshold", 3)

        weights = training_cfg.get("weights", {})
        self.w_priority = weights.get("priority_bonus", 40)
        self.w_gain = weights.get("gain_score", 35)
        self.w_failure = weights.get("failure_penalty", 25)

        # 休息参数
        self.rest_trigger_failure = rest_cfg.get("trigger_failure_threshold", 30)
        self.first_year_rounds = rest_cfg.get("first_year_rounds", 12)

        # 休息选择参数
        selection = rest_cfg.get("selection", {})
        self.best_mood_free_stamina = selection.get("best_mood_free_stamina", 40)
        self.premium_money_threshold = selection.get("premium_money_threshold", 60)

        # 构建休息选项
        rest_options_cfg = rest_cfg.get("options", {})
        self.rest_options = {}
        for k, v in rest_options_cfg.items():
            rid = int(k)
            self.rest_options[rid] = RestOption(
                rest_id=rid,
                name=v.get("name", f"Option {rid}"),
                cost=v.get("cost", 0),
                stamina_recover=v.get("stamina_recover", 30),
                description=v.get("description", ""),
            )

        # 心情配置
        self.mood_levels = mood_cfg.get("levels", {
            0: "Worst", 1: "Bad", 2: "Normal", 3: "Good", 4: "Best"
        })
        # 反转映射: name -> level
        self._mood_name_to_level = {v: int(k) for k, v in self.mood_levels.items()}

        self.attributes = {int(k): v for k, v in attr_cfg.items()}

    @property
    def rule_profile(self) -> TrainingRuleProfile:
        """获取当前规则配置"""
        if self._rule_profile is None:
            if self.rule_profile_name:
                mgr = get_profile_manager()
                mgr.set_profile(self.rule_profile_name)
                self._rule_profile = mgr.current_profile
            else:
                self._rule_profile = TrainingRuleLoader.create_builtin_default()
        return self._rule_profile

    @property
    def is_rule_engine_mode(self) -> bool:
        """是否使用规则引擎模式"""
        return bool(self.rule_profile_name)

    # ===================== 规则引擎模式决策 =====================

    def make_decision_from_context(
        self,
        context: TrainingDecisionContext,
        game_state: GameState,
        debug: bool = False,
    ) -> tuple[str, object]:
        """
        基于规则引擎的决策 (规则引擎模式)。

        Args:
            context: 扫描得到的 TrainingDecisionContext
            game_state: 游戏状态 (用于休息决策)
            debug: 是否打印调试信息

        Returns:
            (action, target) - action: "train"/"rest", target: TrainingDecisionResult/RestOption
        """
        result = TrainingRuleEngine.evaluate(context, self.rule_profile)

        if debug:
            print(f"\n{'='*60}")
            print(f"  [规则引擎决策] profile={self.rule_profile_name}")
            print(f"  状态: 体力={game_state.stamina}%, 失败率={game_state.failure_rate}%, "
                  f"心情={game_state.mood}, 金钱=${game_state.money}")
            print(f"  规则结果: matched={result.matched_rule_id}, action={result.action.value}, "
                  f"target={result.target_row_index}, builtin={result.used_builtin_default}")
            print(f"{'='*60}\n")

        if result.action == TrainingDecisionAction.REST:
            rest_choice = self.decide_rest(game_state, debug=debug)
            return ("rest", rest_choice)

        return ("train", result)

    def get_rule_action_target_name(self, result: TrainingDecisionResult) -> str:
        """获取规则决策对应的属性名称"""
        if result.target_row_index is None:
            return "休息"
        attr_keys = sorted(self.attributes.keys())
        if result.target_row_index < len(attr_keys):
            return self.attributes.get(attr_keys[result.target_row_index], "未知")
        return "未知"

    # ===================== 训练决策 =====================

    def _find_priority_option(self, options):
        for opt in options:
            if opt.attr_id == self.priority:
                return opt
        return None

    def _is_priority_good(self, priority_opt):
        if priority_opt is None:
            return False
        gain_ok = priority_opt.gain >= self.min_gain_threshold
        failure_ok = priority_opt.failure_rate <= self.failure_threshold
        return gain_ok and failure_ok

    def decide_train(self, options, state, debug=False):
        """
        训练决策。
        返回: TrainingOption (选中的训练项)
        """
        if not options:
            return None

        priority_opt = self._find_priority_option(options)
        priority_name = self.attributes.get(self.priority, "未知")

        if debug:
            print(f"\n{'='*60}")
            print(f"  [训练决策] 优先属性: {priority_name}")
            print(f"  状态: 体力={state.stamina}%, 失败率={state.failure_rate}%, "
                  f"心情={state.mood}, 金钱=${state.money}")
            print(f"  阈值: 失败率<={self.failure_threshold}%, "
                  f"增益>={self.min_gain_threshold}")
            print(f"{'─'*60}")

        if self._is_priority_good(priority_opt):
            if debug:
                print(f"  {priority_opt.attr_name}: 增益={priority_opt.gain}, "
                      f"失败率={priority_opt.failure_rate:.1f}% → 达标，直接训练")
                for opt in options:
                    if opt.attr_id != self.priority:
                        print(f"  {opt.attr_name}: 增益={opt.gain}, "
                              f"失败率={opt.failure_rate:.1f}% (跳过)")
                print(f"{'='*60}\n")
            return priority_opt

        if debug:
            if priority_opt:
                reasons = []
                if priority_opt.gain < self.min_gain_threshold:
                    reasons.append(f"增益过低({priority_opt.gain}<{self.min_gain_threshold})")
                if priority_opt.failure_rate > self.failure_threshold:
                    reasons.append(f"失败率超标({priority_opt.failure_rate:.1f}%>{self.failure_threshold}%)")
                print(f"  {priority_opt.attr_name} 不达标 ({', '.join(reasons)})")
            else:
                print(f"  未找到优先属性 {priority_name}")

        scored = self._score_all(options)
        scored.sort(key=lambda x: x[1], reverse=True)

        if debug:
            print(f"  启用AI评分比价:")
            for opt, score in scored:
                marker = " ← 选择" if opt == scored[0][0] else ""
                print(f"    {opt.attr_name}: 增益={opt.gain}, "
                      f"失败率={opt.failure_rate:.1f}% → 总分={score:.1f}{marker}")
            print(f"{'='*60}\n")

        return scored[0][0] if scored else None

    # ===================== 休息决策 =====================

    def should_rest(self, options, state, debug=False):
        """
        判断是否应该休息（而非训练）。

        触发条件:
        1. 失败率超标 + 没有好的训练属性
        """
        # 检查是否有好的优先属性
        priority_opt = self._find_priority_option(options)
        has_good_training = self._is_priority_good(priority_opt)

        # 条件: 失败率超标 且 没有好训练
        failure_high = state.failure_rate > self.rest_trigger_failure
        if failure_high and not has_good_training:
            if debug:
                print(f"  [休息判断] 失败率={state.failure_rate:.1f}% > "
                      f"{self.rest_trigger_failure}% 且无好属性 → 休息")
            return True

        if debug:
            print(f"  [休息判断] 无需休息 (失败率={state.failure_rate:.1f}%, "
                  f"有好训练={has_good_training})")

        return False

    def decide_rest(self, state, debug=False):
        """
        选择哪种休息方式 (与 SleepRunner RestDecisionPolicy 完全一致)。

        优先级:
        1. 心情Best 且 体力 > best_mood_free_stamina → 免费休息
        2. 心情Best 且 体力 <= best_mood_free_stamina → 高级休息($60)
        3. 心情非Best → 付费休息($30)
        4. 其他 → 免费休息
        """
        if debug:
            print(f"\n  [休息选择] 体力={state.stamina}%, 心情={state.mood}, "
                  f"金钱=${state.money}")

        # 规则 1: 心情Best 且 体力充沛 → 免费休息
        if state.mood == "Best" and state.stamina > self.best_mood_free_stamina:
            choice = self.rest_options[1]
            if debug:
                print(f"  心情Best + 体力>{self.best_mood_free_stamina}% → {choice.name}")
            return choice

        # 规则 2: 心情Best 但 体力不足 → 高级休息($60)
        if state.mood == "Best" and state.stamina <= self.best_mood_free_stamina:
            choice = self.rest_options[3]
            if debug:
                print(f"  心情Best + 体力<={self.best_mood_free_stamina}% → {choice.name}")
            return choice

        # 规则 3: 心情非Best → 付费休息($30)
        if state.mood != "Best":
            choice = self.rest_options[2]
            if debug:
                print(f"  心情非Best → {choice.name}")
            return choice

        # 规则 4: 默认 → 免费休息
        choice = self.rest_options[1]
        if debug:
            print(f"  默认 → {choice.name}")
        return choice

    def make_decision(self, options, state, debug=False):
        """
        综合决策: 返回 (action, target)
        action: "train" 或 "rest"
        target: TrainingOption 或 RestOption
        """
        # Step 1: 检查是否该休息
        if self.should_rest(options, state, debug=debug):
            rest_choice = self.decide_rest(state, debug=debug)
            return ("rest", rest_choice)

        # Step 2: 训练决策
        train_choice = self.decide_train(options, state, debug=debug)
        return ("train", train_choice)

    # ===================== 评分系统 =====================

    def _score_all(self, options):
        if not options:
            return []

        max_gain = max(opt.gain for opt in options)
        if max_gain == 0:
            max_gain = 1

        scored = []
        for opt in options:
            score = self._score_option(opt, max_gain)
            scored.append((opt, score))
        return scored

    def _score_option(self, option, max_gain):
        total = 0.0

        # 1. 优先级加分
        if option.attr_id == self.priority:
            total += self.w_priority
        else:
            distance = abs(option.attr_id - self.priority)
            priority_score = self.w_priority * max(0, 1.0 - distance * 0.3)
            total += priority_score

        # 2. 增益评分（归一化）
        gain_normalized = option.gain / max_gain
        total += self.w_gain * gain_normalized

        # 3. 失败率评分
        if option.failure_rate <= self.failure_threshold:
            score = (1.0 - option.failure_rate / 100.0) * self.w_failure
        else:
            penalty_ratio = (option.failure_rate - self.failure_threshold) / (100 - self.failure_threshold)
            score = -self.w_failure * penalty_ratio * 3.0
        total += score

        return total
