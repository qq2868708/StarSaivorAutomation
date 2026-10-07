"""优势训练（蓝色大拇指）识别与评分回归测试。"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.advantage_detector import detect_advantage_training
from src.rule_engine import (
    TrainingDecisionAction,
    TrainingDecisionContext,
    TrainingDecisionResult,
    TrainingRuleCard,
    TrainingRuleEngine,
    TrainingRuleField,
    TrainingRuleOperator,
)


class AdvantageTrainingTests(unittest.TestCase):
    def test_real_training_frame_detects_thumb_rows(self):
        frame_path = (Path(__file__).resolve().parents[1] /
                      ".." / "verification" / "training-fix-20261006" /
                      "frames" / "strength.png")
        if not frame_path.is_file():
            self.skipTest("optional captured training frame is not included in the source checkout")
        frame = cv2.imread(str(frame_path))
        self.assertIsNotNone(frame)
        self.assertEqual(detect_advantage_training(frame),
                         [False, True, True, False, False])

    def test_blank_frame_has_no_advantage_rows(self):
        self.assertEqual(detect_advantage_training(np.zeros((900, 1600, 3), np.uint8)),
                         [False, False, False, False, False])

    def test_advantage_multiplier_changes_builtin_score(self):
        base = TrainingDecisionContext(
            gain_values=[30, 40, 10, 0, 0],
            icon_counts=[3, 0, 0, 0, 0],
            fail_rates=[0, 0, 0, 0, 0],
            is_advantage=[False, True, False, False, False],
            known_gain_mask=0b11111,
            known_icon_mask=0b11111,
            known_fail_rate_mask=0b11111,
            stamina_pct=100,
            advantage_multiplier=1.35,
        )
        # 力量：30×1.5=45；体力：40×1.35=54，应选择带大拇指的体力行。
        self.assertEqual(TrainingRuleEngine._score_training_rows(base), 1)

    def test_legacy_icon_rule_can_follow_advantage_score(self):
        context = TrainingDecisionContext(
            gain_values=[30, 40, 10, 0, 0],
            icon_counts=[3, 0, 0, 0, 0],
            fail_rates=[0, 0, 0, 0, 0],
            is_advantage=[False, True, False, False, False],
            known_gain_mask=0b11111,
            known_icon_mask=0b11111,
            known_fail_rate_mask=0b11111,
            stamina_pct=100,
            advantage_multiplier=1.35,
        )
        rule = TrainingRuleCard(
            id="strength_icons_3",
            field=TrainingRuleField.STRENGTH_ICONS,
            operator=TrainingRuleOperator.GREATER_THAN_OR_EQUAL,
            value=3,
            action=TrainingDecisionAction.TRAIN_STRENGTH,
        )
        result = TrainingRuleEngine._resolve_result(
            context, rule, TrainingDecisionAction.TRAIN_STRENGTH)
        self.assertEqual(result.target_row_index, 1)
        self.assertEqual(result.action, TrainingDecisionAction.TRAIN_STAMINA)
        self.assertTrue(result.matched_rule_id.endswith("+advantage"))


if __name__ == "__main__":
    unittest.main()

