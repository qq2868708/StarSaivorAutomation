"""Unavailable rewards and acknowledged card selection, without desktop input."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.controller import InputTargetError
from src.handlers import CardSelectHandler, RewardAdvanceHandler, FrameContext
from src.journey_target import JourneyTargetPending


def frame(selected=None, blocked=(), complete=False):
    image = np.zeros((900, 1600, 3), np.uint8)
    image[0, 0] = int(complete)
    for index in blocked:
        x, _, w, _ = CardSelectHandler.CARD_TEXT_REGIONS[index]
        image[275:320, int(1600*(x+.015)):int(1600*(x+w-.015))] = (30, 40, 190)
    if selected is not None:
        center = CardSelectHandler.CARD_CLICK_POINTS[selected][0]
        for edge in (center-.10, center+.10):
            x = int(1600*edge)
            image[200:685, x-4:x+5] = (0, 200, 255)
        image[723:776, 675:925] = (210, 115, 30)
    return image


class CardAvailabilityTests(unittest.TestCase):
    def setup_game(self, *, ignored=(), selected=None, texts=None):
        handler = CardSelectHandler()
        state = {"selected": selected, "complete": False, "clicks": []}
        texts = texts or ["持有数量达到上限 发生暴击，攻击力提高", "攻击力减少", "特殊技冷却缩短"]
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=Mock(), logger=Mock(),
                              engine=SimpleNamespace(is_rule_engine_mode=False), journey_target=None,
                              pending_screenshot=None, target_selection_phase=Mock())
        ctx.capture.capture_game.side_effect = lambda: frame(state["selected"], complete=state["complete"])

        def region(image, *area):
            if bool(image[0, 0, 0]):
                return "获得奖励 暴击装备"
            if area == handler.TITLE_REGION:
                return "选择奖励"
            if area in handler.CARD_TEXT_REGIONS:
                return texts[handler.CARD_TEXT_REGIONS.index(area)]
            return ""

        ctx.ocr.recognize_region.side_effect = region
        ctx.ocr.recognize.return_value = [
            ("选择完成", .99, [[115, 24], [205, 24], [205, 48], [115, 48]])]

        def click(x, y, **kwargs):
            state["clicks"].append((x, y))
            if (x, y) in handler.CARD_CLICK_POINTS:
                index = handler.CARD_CLICK_POINTS.index((x, y))
                self.assertNotIn("上限", texts[index], "Must not click a capped card")
                if index not in ignored:
                    state["selected"] = None if state["selected"] == index else index
            else:
                self.assertIsNotNone(state["selected"], "Must select before confirming")
                state["complete"] = True
        ctx.controller.click_at_percent.side_effect = click
        return handler, ctx, state

    def test_holding_limit_is_filtered_and_next_ranked_card_finishes(self):
        handler, ctx, state = self.setup_game()
        with patch("src.recognition.ScreenDetector.is_main_menu_screen", return_value=False):
            self.assertTrue(handler.handle(ctx))
        self.assertTrue(state["complete"])
        self.assertEqual(state["clicks"][0], handler.CARD_CLICK_POINTS[1])
        self.assertEqual(len(state["clicks"]), 2)
        final = ctx.logger.log_card_select.call_args
        self.assertEqual(final.args[0], 1)
        self.assertEqual(final.kwargs["phase"], "confirmed")
        self.assertFalse(final.kwargs["candidates"][0]["enabled"])
        self.assertEqual(ctx.logger.record_decision_frame.call_count, 5)

    def test_zero_secondary_scores_cannot_fall_back_to_disabled_slot_zero(self):
        handler, ctx, state = self.setup_game(texts=["达到上限", "恢复效果", "冷却缩短"])
        with patch("src.recognition.ScreenDetector.is_main_menu_screen", return_value=False):
            handler.handle(ctx)
        self.assertEqual(state["clicks"][0], handler.CARD_CLICK_POINTS[1])

    def test_already_selected_card_is_not_toggled_off(self):
        handler, ctx, state = self.setup_game(selected=1)
        with patch("src.recognition.ScreenDetector.is_main_menu_screen", return_value=False):
            handler.handle(ctx)
        self.assertTrue(state["complete"])
        self.assertEqual(len(state["clicks"]), 1)

    def test_nonresponsive_selection_is_bounded_then_uses_next_available(self):
        handler, ctx, state = self.setup_game(ignored=(1,))
        with patch("src.recognition.ScreenDetector.is_main_menu_screen", return_value=False):
            handler.handle(ctx)
        self.assertEqual(state["clicks"].count(handler.CARD_CLICK_POINTS[1]), 2)
        self.assertEqual(state["selected"], 2)
        self.assertTrue(state["complete"])

    def test_no_responsive_card_never_clicks_confirmation(self):
        handler, ctx, state = self.setup_game(ignored=(1, 2))
        with self.assertRaises(InputTargetError):
            handler.handle(ctx)
        self.assertEqual(len(state["clicks"]), 4)
        self.assertFalse(state["complete"])

    def test_all_capped_cards_send_no_input(self):
        handler, ctx, state = self.setup_game(texts=["达到上限"]*3)
        with self.assertRaises(InputTargetError):
            handler.handle(ctx)
        self.assertFalse(state["clicks"])

    def test_red_disabled_banner_is_rejected_even_when_ocr_misses_it(self):
        rows = CardSelectHandler._candidates(frame(blocked=(0,)), ["暴击率", "攻击力", "冷却"])
        self.assertFalse(rows[0]["enabled"])
        self.assertTrue(rows[1]["enabled"])

    def test_selected_highlight_and_blue_button_are_both_required(self):
        handler = CardSelectHandler()
        image = frame()
        ocr = Mock()
        self.assertIsNone(handler._selected_slot(image))
        self.assertIsNone(handler._done_position(image, ocr))
        image = frame(1)
        self.assertEqual(handler._selected_slot(image), 1)
        image[723:776, 675:925] = 60
        self.assertIsNone(handler._done_position(image, ocr))
        ocr.recognize.assert_not_called()

    def test_target_material_remains_unconfirmed_on_failed_selection(self):
        handler, ctx, state = self.setup_game(ignored=(1,), texts=["攻击装备", "暴击装备", "速度装备"])
        choice = {"index": 1, "material": "暴击", "combo_code": "AX5"}
        with self.assertRaises(InputTargetError):
            handler._execute_target_card(ctx, ["攻击装备", "暴击装备", "速度装备"], choice)
        self.assertEqual([c.args[0] for c in ctx.target_selection_phase.call_args_list], ["prepared"])
        self.assertFalse(state["complete"])

    def test_capped_target_material_does_not_mark_material_obtained(self):
        handler, ctx, state = self.setup_game()
        with self.assertRaises(JourneyTargetPending):
            handler._execute_target_card(ctx, ["暴击装备 达到上限", "其他", "其他"],
                                         {"index": 0, "material": "暴击", "combo_code": "AX5"})
        ctx.target_selection_phase.assert_not_called()
        self.assertFalse(state["clicks"])

    def test_receipt_screen_cannot_be_treated_as_choice_screen(self):
        handler = CardSelectHandler()
        observation = Mock()
        observation.get_ocr.side_effect = ["", "获得奖励"]
        self.assertFalse(handler.can_handle(observation, Mock()))

    def test_unavailable_target_falls_back_without_claiming_target_obtained(self):
        handler, ctx, state = self.setup_game()
        handler.config["safety"] = {"progress_fallbacks": True}
        ctx.journey_target = {"training_direction": "attack", "combo_code": "AX5"}
        with patch("src.handlers.equipment_choice", side_effect=JourneyTargetPending("目标材料不可用")), \
             patch("src.recognition.ScreenDetector.is_main_menu_screen", return_value=False):
            self.assertTrue(handler.handle(ctx))
        ctx.logger.log_fallback.assert_called_once()
        self.assertEqual(ctx.logger.log_fallback.call_args.args[0], "target_card")
        ctx.target_selection_phase.assert_not_called()
        self.assertTrue(state["complete"])

    def test_status_removed_advances_only_after_rechecking_title_and_button(self):
        handler = RewardAdvanceHandler()
        image = frame()
        title = ("状态解除!", .99, [[680, 200], [900, 200], [900, 245], [680, 245]])
        button = ("点击以继续", .99, [[745, 814], [850, 814], [850, 845], [745, 845]])
        ocr = Mock()
        ocr.recognize_detailed.return_value = [title, button]
        self.assertTrue(handler.can_handle(FrameContext(image), ocr))
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=ocr, pending_screenshot=None)
        ctx.capture.capture_game.return_value = image
        self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(
            797.5/1600, 829.5/900, source="status_continue")
        ctx.controller.reset_mock()
        ocr.recognize_detailed.return_value = [button]
        self.assertFalse(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_not_called()

    def test_status_removed_rejects_misplaced_or_low_confidence_continue(self):
        image = frame()
        ocr = Mock()
        title = ("状态解除!", .99, [[680, 200], [900, 200], [900, 245], [680, 245]])
        for button in [
                ("点击以继续", .4, [[745, 814], [850, 814], [850, 845], [745, 845]]),
                ("点击以继续", .99, [[745, 514], [850, 514], [850, 545], [745, 545]])]:
            ocr.recognize_detailed.return_value = [title, button]
            self.assertIsNone(RewardAdvanceHandler._status_continue_position(FrameContext(image), ocr))


if __name__ == "__main__":
    unittest.main()
