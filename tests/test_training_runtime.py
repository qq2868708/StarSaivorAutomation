"""Regression tests for training stalls; no desktop input or game required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.recognition import GameStateReader
from src.trainer import Trainer, TrainingMetricScanState, _TRAINING_ROWS


class TrainingRuntimeTests(unittest.TestCase):
    def test_background_card_match_cannot_preempt_training(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = {'arcanum': {'enabled': True}}
        trainer.round_count = 1
        trainer._handler_ctx = SimpleNamespace(round_count=0, detected_turn=None,
                                              try_detect_turn=Mock())
        trainer._try_click_arcanum = Mock(side_effect=AssertionError('legacy card click'))
        trainer._read_game_state = Mock(return_value=SimpleNamespace(
            stamina=80, mood='Good', money=30))
        trainer.logger = Mock()
        trainer.engine = SimpleNamespace(is_rule_engine_mode=True)
        target = object()
        trainer._decide_rule_engine = Mock(return_value=('train', target))
        trainer._execute_train = Mock(return_value=True)
        self.assertTrue(trainer._handle_training_screen(np.zeros((900, 1600, 3), np.uint8)))
        trainer._execute_train.assert_called_once_with(target)
        trainer._try_click_arcanum.assert_not_called()

    def test_strength_is_selected_when_resuming_another_row(self):
        trainer = Trainer.__new__(Trainer)
        trainer.controller = Mock()
        trainer.capture = Mock()
        trainer.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        trainer.engine = SimpleNamespace(use_icon_counting=False, use_per_row_fail_rate=True,
                                       fail_rate_regions=[(0.85, 0.265, 0.1, 0.06)] * 5)
        trainer.ui = SimpleNamespace(ocr=Mock())
        trainer.ui.ocr.recognize.return_value = [('失败率2%', 0.99)]
        trainer.ui.ocr.recognize_region.return_value = '+23'
        trainer._last_stamina = 0
        scan = TrainingMetricScanState()
        trainer._read_row_metrics(scan, 0)
        trainer.controller.click_at_percent.assert_called_once_with(*_TRAINING_ROWS[0][1:])
        self.assertEqual(scan.fail_rates[0], 2)
        self.assertEqual(scan.gain_values[0], 23)

    def test_zero_stamina_is_not_replaced_with_full_stamina(self):
        trainer = Trainer.__new__(Trainer)
        trainer._last_stamina = 0
        self.assertEqual(trainer._stamina_fail_rate(0), 100)
        self.assertEqual(trainer._stamina_fail_rate(4), 100)

    def test_hud_ignores_bare_numbers_and_reads_empty_bar(self):
        ocr = Mock()
        ocr.recognize_region.return_value = '距离目标 5 3月中旬'
        reader = GameStateReader(ocr)
        frame = np.zeros((900, 1600, 3), np.uint8)
        # The game can have 0% stamina; absence of green is a valid empty bar.
        self.assertEqual(reader._read_stamina(frame), 0)
        for call in ocr.recognize_region.call_args_list:
            self.assertGreater(call.args[1], 0.3)
        frame[46:56, 579:764] = (0, 255, 0)
        self.assertTrue(75 <= reader._read_stamina(frame) <= 85)
        ocr.recognize_region.return_value = '0%'
        self.assertEqual(reader._read_stamina(frame), 0)

    def test_turn_detection_is_read_only_and_requires_valid_fraction(self):
        trainer = Trainer.__new__(Trainer)
        trainer.capture = Mock()
        trainer.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        trainer.controller = Mock()
        trainer.ui = SimpleNamespace(ocr=Mock())
        for text, expected in [('距离目标 5', None), ('99/45', None),
                               ('21/1250', None), ('23 / 45', 23)]:
            trainer.ui.ocr.recognize.return_value = [(text, 1.0)]
            self.assertEqual(trainer._detect_game_turn(), expected)
        trainer.controller.click_at_percent.assert_not_called()


if __name__ == '__main__':
    unittest.main()
