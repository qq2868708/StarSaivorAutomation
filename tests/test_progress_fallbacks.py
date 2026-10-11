"""Policy fallbacks require visible enabled options and recorded deviations."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.handlers import EventHandler, EventDecisionPending
from src.journey_target import JourneyTargetPending


class ProgressFallbackTests(unittest.TestCase):
    def context(self):
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), logger=Mock(), ocr=Mock(),
                              journey_target={"training_direction": "attack"}, pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        return ctx

    def test_event_rule_or_target_gap_uses_enabled_no_payment_option(self):
        for error in (EventDecisionPending("规则缺失"), JourneyTargetPending("材料不匹配")):
            with self.subTest(error=type(error).__name__):
                handler = EventHandler({"safety": {"progress_fallbacks": True}})
                handler._handle_document_event = Mock(side_effect=error)
                ctx = self.context()
                rows = [{"text": "支付钻石", "enabled": True, "y": .4},
                        {"text": "继续前进", "enabled": True, "y": .6},
                        {"text": "已锁定", "enabled": False, "y": .8}]
                with patch("src.recognition.ScreenDetector") as detector:
                    detector.return_value.read_region_text.return_value = "旅程事件"
                    detector.return_value.read_event_option_rows.side_effect = [rows, []]
                    self.assertTrue(handler.handle(ctx))
                ctx.controller.click_at_percent.assert_called_once_with(.80, .6)
                ctx.logger.log_fallback.assert_called_once()
                self.assertEqual(ctx.logger.log_fallback.call_args.args[3]["index"], 2)
                ctx.logger.log_event.assert_called_once()

    def test_no_payment_free_option_never_invents_or_buys_one(self):
        handler = EventHandler({"safety": {"progress_fallbacks": True}})
        handler._handle_document_event = Mock(side_effect=EventDecisionPending("规则缺失"))
        ctx = self.context()
        with patch("src.recognition.ScreenDetector") as detector:
            detector.return_value.read_region_text.return_value = "旅程事件"
            detector.return_value.read_event_option_rows.return_value = [
                {"text": "购买", "enabled": True, "y": .4},
                {"text": "锁定", "enabled": False, "y": .6}]
            with self.assertRaises(EventDecisionPending):
                handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
