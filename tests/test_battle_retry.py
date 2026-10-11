"""Read actual retry counts and acknowledge each battle transition separately."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.controller import InputTargetError
from src.handlers import BattleDefeatHandler, FrameContext, RewardAdvanceHandler


class BattleRetryTests(unittest.TestCase):
    def test_fail_requires_battle_controls_not_just_failure_text(self):
        handler = BattleDefeatHandler()
        for controls, expected in [("", False), ("点击以继续", False),
                                   ("重新挑战", True), ("离开", True)]:
            with self.subTest(controls=controls):
                frame = Mock()
                frame.get_ocr.side_effect = lambda _, x, y, w, h: (
                    "FAIL" if y < .5 else controls)
                self.assertEqual(handler.can_handle(frame, Mock()), expected)

    def test_defeat_banner_and_retry_dialog_still_match(self):
        for text in ("DEFEAT", "EFEAT", "重新挑战通知"):
            frame = Mock()
            frame.get_ocr.return_value = text
            self.assertTrue(BattleDefeatHandler().can_handle(frame, Mock()))

    def context(self, *, page=0, remaining=3, respond=True):
        state = {"page": page}
        handler = BattleDefeatHandler()
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=Mock(),
                              logger=Mock(), pending_screenshot=None)

        def capture():
            image = np.zeros((900, 1600, 3), np.uint8)
            image[0, 0] = state["page"]
            return image
        ctx.capture.capture_game.side_effect = capture
        handler._is_retry_dialog = lambda frame, _: int(frame.screenshot[0, 0, 0]) == 1
        handler._is_defeat_screen = lambda frame, _: int(frame.screenshot[0, 0, 0]) == 0
        handler._read_retries = Mock(return_value=remaining)
        ctx.ocr.recognize_region.return_value = "是否要重新挑战，再次尝试战斗？"

        def click(*args, **kwargs):
            if respond:
                state["page"] = 1 if state["page"] == 0 and remaining else 2
        ctx.controller.click_at_percent.side_effect = click
        return handler, ctx, state

    def test_retry_uses_detected_button_and_does_not_blindly_click_confirmation(self):
        handler, ctx, state = self.context()
        with patch("src.recognition.ScreenDetector.find_action_position", return_value=(.77, .93)) as find:
            self.assertTrue(handler.handle(ctx))
        self.assertEqual(find.call_args.args[1], "重新挑战")
        ctx.controller.click_at_percent.assert_called_once_with(.77, .93)
        self.assertEqual(state["page"], 1)
        self.assertEqual(handler._retries_remaining, 3)
        ctx.controller.click_center_multi.assert_not_called()

    def test_confirmation_is_a_separate_verified_action(self):
        handler, ctx, state = self.context(page=1)
        with patch("src.recognition.ScreenDetector.find_action_position", return_value=(.57, .69)) as find:
            self.assertTrue(handler.handle(ctx))
        self.assertEqual(find.call_args.args[1], "确认")
        ctx.controller.click_at_percent.assert_called_once_with(.57, .69)
        self.assertEqual(state["page"], 2)

    def test_zero_displayed_retries_leaves_without_resetting_allowance(self):
        handler, ctx, _ = self.context(remaining=0)
        with patch("src.recognition.ScreenDetector.find_action_position", return_value=(.90, .93)) as find:
            handler.handle(ctx)
        self.assertEqual(find.call_args.args[1], "离开")
        self.assertEqual(handler._retries_remaining, 0)

    def test_unresponsive_button_does_not_consume_local_count_or_repeat_clicks(self):
        handler, ctx, _ = self.context(respond=False)
        with patch("src.recognition.ScreenDetector.find_action_position", return_value=(.77, .93)), \
             patch("src.handlers.time.perf_counter", side_effect=[0, 0, 7]), \
             self.assertRaises(InputTargetError):
            handler.handle(ctx)
        ctx.controller.click_at_percent.assert_called_once()
        self.assertEqual(handler._retries_remaining, 3)

    def test_extra_resource_confirmation_never_clicks(self):
        handler, ctx, _ = self.context(page=1)
        ctx.ocr.recognize_region.return_value = "重新挑战需要支付钻石"
        with self.assertRaises(InputTargetError):
            handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_unknown_retry_count_never_clicks(self):
        handler, ctx, _ = self.context(remaining=None)
        with self.assertRaises(InputTargetError):
            handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_stale_defeat_detection_yields_to_new_scene(self):
        handler, ctx, _ = self.context(page=2)
        self.assertFalse(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_not_called()

    def test_count_parser_requires_unique_confident_bounded_value(self):
        image = np.zeros((900, 1600, 3), np.uint8)
        ocr = Mock()
        for text, score, dialog, expected in [
            ("3", .99, False, 3), ("0", .99, False, 0),
            ("×3", .61, False, None), ("30", .99, False, None),
            ("重新挑战可再3次", .99, True, 3), ("重新挑战", .99, True, None)]:
            ocr.recognize_detailed.return_value = [(text, score, None)]
            self.assertEqual(BattleDefeatHandler._read_retries(image, ocr, dialog=dialog), expected)
        ocr.recognize_detailed.return_value = [("2", .99, None), ("3", .99, None)]
        self.assertIsNone(BattleDefeatHandler._read_retries(image, ocr))

class RestFailureTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((900, 1600, 3), np.uint8)
        self.rows = [
            ("露宿", .99, [[776, 80], [824, 80], [824, 104], [776, 104]]),
            ("FAIL", .99, [[704, 128], [894, 128], [894, 205], [704, 205]]),
            ("点击以继续", .99, [[745, 814], [850, 814], [850, 845], [745, 845]]),
        ]
        self.ocr = Mock()
        self.ocr.recognize_detailed.return_value = self.rows

    def test_rest_fail_advances_at_verified_continue_position(self):
        handler = RewardAdvanceHandler()
        self.assertTrue(handler.can_handle(FrameContext(self.image), self.ocr))
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=self.ocr)
        ctx.capture.capture_game.return_value = self.image
        self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(
            797.5/1600, 829.5/900, source="rest_result_continue")

    def test_changed_scene_does_not_replay_continue(self):
        handler = RewardAdvanceHandler()
        self.assertTrue(handler.can_handle(FrameContext(self.image), self.ocr))
        self.ocr.recognize_detailed.return_value = [self.rows[2]]
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=self.ocr)
        ctx.capture.capture_game.return_value = self.image
        self.assertFalse(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_not_called()
        self.assertIs(ctx.pending_screenshot, self.image)

    def test_rest_fail_requires_unique_confident_positioned_markers(self):
        for i in range(3):
            for change in ("missing", "duplicate", "low_confidence", "misplaced"):
                with self.subTest(marker=i, change=change):
                    rows = list(self.rows)
                    text, _, box = rows[i]
                    if change == "missing":
                        rows.pop(i)
                    elif change == "duplicate":
                        rows.append(rows[i])
                    elif change == "low_confidence":
                        rows[i] = (text, .5, box)
                    else:
                        rows[i] = (text, .99, [[0, 0], [10, 0], [10, 10], [0, 10]])
                    self.ocr.recognize_detailed.return_value = rows
                    self.assertIsNone(RewardAdvanceHandler._rest_failure_continue_position(
                        FrameContext(self.image), self.ocr))


if __name__ == "__main__":
    unittest.main()
