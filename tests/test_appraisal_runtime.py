"""Appraisal navigation regressions; no game input is sent."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.controller import InputTargetError
from src.handlers import AppraisalHandler, FrameContext, ShopHandler
from src.recognition import ScreenDetector


class AppraisalRuntimeTests(unittest.TestCase):
    def test_due_requires_day_and_real_menu(self):
        detector = ScreenDetector(Mock())
        frame = np.zeros((900, 1600, 3), np.uint8)
        for goal, menu, expected in [
            ('目标D-DAY参加评鉴战', '评鉴战交易', True),
            ('距离目标5参加评鉴战', '评鉴战交易', False),
            ('D-DAY', '训练委托休息', False),
        ]:
            detector.read_region_text = Mock(side_effect=[goal, menu])
            self.assertEqual(detector.is_appraisal_due_screen(frame), expected)

    def test_button_locator_rejects_ambiguous_or_nonexact_labels(self):
        ocr = Mock()
        detector = ScreenDetector(ocr)
        frame = np.zeros((900, 1600, 3), np.uint8)
        box = [[100, 10], [160, 10], [160, 30], [100, 30]]
        region = (0.78, 0.84, 0.18, 0.1)
        ocr.recognize.return_value = [('接受', 0.99, box)]
        self.assertEqual(detector.find_action_position(frame, '接受', region),
                         ((1248 + 130) / 1600, (756 + 20) / 900))
        ocr.recognize.return_value = [('接受', 0.99, box)] * 2
        self.assertIsNone(detector.find_action_position(frame, '接受', region))
        ocr.recognize.return_value = [('不接受', 0.99, box)]
        self.assertIsNone(detector.find_action_position(frame, '接受', region))

    def test_appraisal_precedes_shop_even_when_shop_is_disabled(self):
        handler = AppraisalHandler()
        self.assertLess(handler.priority, ShopHandler({'shop': {'enabled': False}}).priority)
        with patch('src.recognition.ScreenDetector') as cls:
            cls.return_value.is_appraisal_screen.return_value = False
            cls.return_value.is_appraisal_due_screen.return_value = True
            self.assertTrue(handler.can_handle(FrameContext(np.zeros((900, 1600, 3), np.uint8)), Mock()))

    def test_failed_entry_transition_stops_after_one_click(self):
        ctx = SimpleNamespace(ocr=Mock(), controller=Mock(), capture=Mock(), pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        with patch('src.recognition.ScreenDetector') as cls, patch('src.handlers._wait_for_screen', return_value=False):
            detector = cls.return_value
            detector.is_appraisal_screen.return_value = False
            detector.is_appraisal_due_screen.return_value = True
            detector.find_action_position.return_value = (0.88, 0.47)
            with self.assertRaises(InputTargetError):
                AppraisalHandler().handle(ctx)
        ctx.controller.click_at_percent.assert_called_once_with(0.941, 0.466)


if __name__ == '__main__':
    unittest.main()
