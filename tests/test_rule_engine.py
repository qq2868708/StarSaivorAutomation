"""
规则引擎集成测试

验证从 SleepRunner 移植的模块:
  - 规则配置加载 (TrainingRuleLoader)
  - 规则评估 (TrainingRuleEngine.probe / evaluate)
  - 内置默认决策 (TrainingBuiltInDecision 逻辑)
  - 图标优先级规则 (apply_priority_rule)
  - 失败率解析 (parse_fail_rate_from_ocr_texts)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rule_engine import (
    TrainingRuleLoader,
    TrainingRuleEngine,
    TrainingDecisionContext,
    TrainingDecisionAction,
    TrainingRuleField,
    BuildDirection,
    TrainingRuleProfile,
    TrainingLegacyStrategy,
)
from fail_rate_ocr import parse_fail_rate_from_ocr_texts
from icon_counter import apply_priority_rule


def test_load_profiles():
    """测试加载规则配置文件"""
    profiles_dir = os.path.join(os.path.dirname(__file__), '..', 'profiles', 'training')

    for name in ['attack', 'survival', 'default', 'speed']:
        path = os.path.join(profiles_dir, f'{name}.json')
        assert os.path.exists(path), f"Profile {name} not found at {path}"

        profile = TrainingRuleLoader.load_from_file(path)
        assert len(profile.rules) > 0, f"Profile {name} has no rules"
        assert profile.rules[-1].is_fallback, f"Profile {name} last rule should be fallback"

        # 验证 legacy strategy
        assert profile.legacy_strategy.fail_rate_threshold in range(0, 101)
        assert profile.legacy_strategy.rush_threshold in range(100, 1201)
        assert profile.legacy_strategy.build_direction in (BuildDirection.ATTACK, BuildDirection.SURVIVAL)

        print(f"  [PASS] {name}: {len(profile.rules)} rules, "
              f"build={profile.legacy_strategy.build_direction.value}, "
              f"fail_threshold={profile.legacy_strategy.fail_rate_threshold}%, "
              f"rush_threshold={profile.legacy_strategy.rush_threshold}")


def test_builtin_default():
    """测试内置默认 profile"""
    profile = TrainingRuleLoader.create_builtin_default()
    assert len(profile.rules) == 1
    assert profile.rules[0].action == TrainingDecisionAction.BUILTIN_DEFAULT
    assert profile.rules[0].is_fallback
    print("  [PASS] builtin_default")


def test_rule_engine_probe_missing_field():
    """测试 Probe 在字段缺失时返回 MissingField"""
    profile = TrainingRuleLoader.load_from_file(
        os.path.join(os.path.dirname(__file__), '..', 'profiles', 'training', 'attack.json'))

    # 空 context - 所有字段未知
    context = TrainingDecisionContext(
        profile_name="test",
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
    )

    result = TrainingRuleEngine.probe(context, profile)
    assert result.decision is None, "Should not have decision with empty context"
    assert result.missing_field is not None, "Should report missing field"
    print(f"  [PASS] probe missing field: {result.missing_field.name}")


def test_rule_engine_probe_decision():
    """测试 Probe 在数据足够时返回 Decision"""
    profile = TrainingRuleLoader.load_from_file(
        os.path.join(os.path.dirname(__file__), '..', 'profiles', 'training', 'attack.json'))

    # 完整 context
    context = TrainingDecisionContext(
        icon_counts=[0, 0, 0, 0, 0],
        fail_rates=[50, 50, 50, 50, 50],  # 全部 > 25% → 第一条规则 rest
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        strength_stat=None,
        stamina_stat=None,
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
        profile_name="test",
    )

    result = TrainingRuleEngine.probe(context, profile)
    assert result.decision is not None, "Should have decision"
    # 力量失败率 50% > 25% → rest
    assert result.decision.action == TrainingDecisionAction.REST, \
        f"Expected REST, got {result.decision.action}"
    print(f"  [PASS] probe decision: {result.decision.action.value} "
          f"(matched={result.decision.matched_rule_id})")


def test_rule_engine_evaluate_full():
    """测试 Evaluate 全量评估: 图标好但失败率低的场景"""
    profile = TrainingRuleLoader.load_from_file(
        os.path.join(os.path.dirname(__file__), '..', 'profiles', 'training', 'attack.json'))

    # 失败率低 + 力量图标 ≥3 → 应该选力量训练
    context = TrainingDecisionContext(
        icon_counts=[3, 0, 0, 0, 0],   # 力量 3 图标
        fail_rates=[15, 50, 50, 50, 50],  # 力量失败率 15% ≤ 25% 阈值
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        strength_stat=400,
        stamina_stat=300,
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
        profile_name="test",
    )

    result = TrainingRuleEngine.evaluate(context, profile)
    assert result.action != TrainingDecisionAction.REST, "Should not rest with low fail rate"

    # 力量图标=3 >= 3 → 应匹配 strength_icons_3 规则
    assert result.action == TrainingDecisionAction.TRAIN_STRENGTH, \
        f"Expected TRAIN_STRENGTH, got {result.action}"
    print(f"  [PASS] evaluate: {result.action.value} (matched={result.matched_rule_id})")


def test_builtin_decision_flow():
    """测试内置默认决策逻辑的各个分支"""
    profile = TrainingRuleLoader.create_builtin_default()

    # Case 1: 高失败率 → rest
    context = TrainingDecisionContext(
        icon_counts=[3, 3, 3, 3, 3],
        fail_rates=[35, 35, 35, 35, 35],
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        strength_stat=400,
        stamina_stat=300,
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
        profile_name="test",
    )
    result = TrainingRuleEngine.evaluate(context, profile)
    assert result.action == TrainingDecisionAction.REST, \
        f"Case 1 (high fail): expected REST, got {result.action}"
    print(f"  [PASS] builtin case 1 (high fail → rest)")

    # Case 2: rush stat ≥ cap → 选前3中非rush最好的
    context2 = TrainingDecisionContext(
        icon_counts=[5, 8, 0, 0, 0],
        fail_rates=[10, 10, 10, 10, 10],
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        strength_stat=1250,  # 达到 cap
        stamina_stat=300,
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
        profile_name="test",
    )
    result2 = TrainingRuleEngine.evaluate(context2, profile2) if False else None
    # 需要重新构建
    result2 = TrainingRuleEngine.evaluate(context2, profile)
    # strength capped → 选 front-3 中非 strength(0) 的最好: stamina(1) 图标=8
    assert result2.action == TrainingDecisionAction.TRAIN_STAMINA, \
        f"Case 2 (cap): expected TRAIN_STAMINA, got {result2.action}"
    print(f"  [PASS] builtin case 2 (stat capped → alternative)")

    # Case 3: rush stat > threshold → train rush
    context3 = TrainingDecisionContext(
        icon_counts=[3, 3, 3, 3, 3],
        fail_rates=[10, 10, 10, 10, 10],
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        strength_stat=500,  # > 450 rush threshold
        stamina_stat=300,
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
        profile_name="test",
    )
    result3 = TrainingRuleEngine.evaluate(context3, profile)
    assert result3.action == TrainingDecisionAction.TRAIN_STRENGTH, \
        f"Case 3 (rush): expected TRAIN_STRENGTH, got {result3.action}"
    print(f"  [PASS] builtin case 3 (rush stat → train rush)")

    # Case 4: normal → priority rule
    context4 = TrainingDecisionContext(
        icon_counts=[2, 5, 1, 0, 0],
        fail_rates=[10, 10, 10, 10, 10],
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        strength_stat=400,
        stamina_stat=300,
        build_direction=BuildDirection.ATTACK,
        legacy_fail_rate_threshold=30,
        legacy_rush_threshold=450,
        profile_name="test",
    )
    result4 = TrainingRuleEngine.evaluate(context4, profile)
    assert result4.action == TrainingDecisionAction.TRAIN_STAMINA, \
        f"Case 4 (normal): expected TRAIN_STAMINA (highest icons), got {result4.action}"
    print(f"  [PASS] builtin case 4 (normal → priority rule)")


def test_fail_rate_parsing():
    """测试失败率 OCR 文本解析"""
    # 精确匹配
    texts = ["力量训练", "失败率25%", "Lv.3"]
    rate = parse_fail_rate_from_ocr_texts(texts)
    assert rate == 25, f"Expected 25, got {rate}"
    print(f"  [PASS] fail_rate parse '25%' → {rate}")

    # 多个候选, 选置信度最高的
    texts2 = ["败率", "30%", "训练"]
    rate2 = parse_fail_rate_from_ocr_texts(texts2)
    assert rate2 == 30, f"Expected 30, got {rate2}"
    print(f"  [PASS] fail_rate parse '30%' with fail keyword → {rate2}")

    # OCR 错误修正: O→0 字符替换
    # 注: 单个来源的 0% 被调度为低置信度 (SleepRunner 的安全设计)
    # 需要多次读到 0% 才可信
    texts3 = ["失败率O%", "0%", "失败率 0"]  # O→0 修正 + 多源确认
    rate3 = parse_fail_rate_from_ocr_texts(texts3)
    assert rate3 == 0, f"Expected 0 (O→0, multi-source), got {rate3}"
    print(f"  [PASS] fail_rate parse 'O%' + multi-source → {rate3}")

    # 无有效候选 - 应回退到 100
    texts4 = ["训练", "Lv"]  # 完全没有数字
    rate4 = parse_fail_rate_from_ocr_texts(texts4)
    assert rate4 == 100, f"Expected fallback 100, got {rate4}"
    print(f"  [PASS] fail_rate parse (no candidates) → {rate4} (fallback)")


def test_icon_priority_rule():
    """测试图标优先级规则"""
    # Attack: preference order 力量(0) > 体力(1) > 韧性(2)
    counts = [3, 3, 1, 0, 0]
    selected = apply_priority_rule(counts, "attack")
    assert selected == 0, f"Attack tie 3/3/1: expected 0 (力量), got {selected}"
    print(f"  [PASS] icon priority attack tie → {selected}")

    # Survival: preference 韧性(2) > 体力(1) > 力量(0)
    selected2 = apply_priority_rule(counts, "survival")
    assert selected2 == 1, f"Survival tie 3/3/1: expected 1 (体力), got {selected2}"
    print(f"  [PASS] icon priority survival tie → {selected2}")

    # 集中≥4 → 选中集中
    counts3 = [0, 0, 0, 4, 1]
    selected3 = apply_priority_rule(counts3, "attack")
    assert selected3 == 3, f"Expected 3 (集中≥4), got {selected3}"
    print(f"  [PASS] icon priority focus≥4 → {selected3}")

    # 保护≥4 → 选保护
    counts4 = [0, 0, 0, 2, 4]
    selected4 = apply_priority_rule(counts4, "attack")
    assert selected4 == 4, f"Expected 4 (保护≥4), got {selected4}"
    print(f"  [PASS] icon priority guard≥4 → {selected4}")

    # No contest
    counts5 = [5, 2, 1, 0, 0]
    selected5 = apply_priority_rule(counts5, "attack")
    assert selected5 == 0, f"Expected 0 (clear winner), got {selected5}"
    print(f"  [PASS] icon priority clear winner → {selected5}")


def test_decision_context():
    """测试 TrainingDecisionContext 的 any_fail_rate 和 try_get_metric"""
    context = TrainingDecisionContext(
        icon_counts=[3, 2, 1, 4, 5],
        fail_rates=[10, 20, 30, 40, 50],
        known_icon_mask=0b11111,
        known_fail_rate_mask=0b11111,
        profile_name="test",
    )

    # any_fail_rate
    assert context.any_fail_rate == 50, f"Expected max fail rate 50, got {context.any_fail_rate}"

    # try_get_metric
    known, val = context.try_get_metric(TrainingRuleField.STRENGTH_ICONS)
    assert known and val == 3

    known, val = context.try_get_metric(TrainingRuleField.GUARD_FAIL_RATE)
    assert known and val == 50

    known, val = context.try_get_metric(TrainingRuleField.STRENGTH_STAT)
    assert not known  # None

    print("  [PASS] decision context metrics")


if __name__ == "__main__":
    print("=" * 60)
    print("  规则引擎集成测试")
    print("=" * 60)

    test_load_profiles()
    test_builtin_default()
    test_rule_engine_probe_missing_field()
    test_rule_engine_probe_decision()
    test_rule_engine_evaluate_full()
    test_builtin_decision_flow()
    test_fail_rate_parsing()
    test_icon_priority_rule()
    test_decision_context()

    print("\n" + "=" * 60)
    print("  全部测试通过!")
    print("=" * 60)
