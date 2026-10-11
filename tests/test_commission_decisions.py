"""Commission selection, battle choices and page separation without game input."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.commission import TIERS, battle_mode, read_panel, read_battle_prompt, select_commission
from src.controller import InputTargetError
from src.handlers import CommissionHandler, SkipHandler, CombatHandler, ShopHandler, FrameContext
from src.recognition import ScreenDetector


def text(value, x, y, confidence=.99):
    return value, confidence, [[x-40, y-10], [x+40, y-10], [x+40, y+10], [x-40, y+10]]


def panel_rows(tier="低阶", rank=19, recommended=17):
    return [text("参加评鉴战", 220, 55), text("RANK"+str(rank), 130, 282),
            text("史莱姆讨伐委托工", 980, 270), text(tier+"委托", 980, 299),
            text("建议综合等级", 907, 512), text("RANK"+str(recommended), 885, 547),
            text("高阶委托", 1360, 563), text("中阶委托", 1360, 452),
            text("低阶委托", 1360, 343), text("接受", 1370, 807)]


def prompt_rows(recommended=23, recommendation=None):
    rows = [text("史莱姆讨伐委托！", 808, 278),
            text("建议综合等级"+str(recommended), 735, 503),
            text("是否要进行战斗委托？", 800, 540),
            text("跳过战斗", 685, 652), text("开始委托", 925, 653)]
    if recommendation:
        rows.append(text("推荐", 925 if recommendation == "battle" else 685, 612))
    return rows


class CommissionDecisionTests(unittest.TestCase):
    def test_rank_gap_boundaries_and_old_five_level_setting(self):
        for gap, mode in [(-3, "skip"), (0, "skip"), (1, "skip"), (2, "skip"),
                          (3, "battle"), (4, "battle"), (5, None)]:
            self.assertEqual(battle_mode(20, 20+gap, max_gap=5), mode)
        self.assertIsNone(battle_mode(None, 20))
        self.assertIsNone(battle_mode(20, None))

    def test_highest_eligible_rank_wins_instead_of_first_safe_low_tier(self):
        candidates = [{"tier": tier, "my_rank": 20, "recommended": rank, "verified": True}
                      for tier, rank in zip(TIERS, [25, 23, 18])]
        self.assertEqual(select_commission(candidates)["tier"], "中阶")
        candidates[0]["recommended"] = 24
        self.assertEqual(select_commission(candidates)["tier"], "高阶")
        candidates[0]["verified"] = False
        self.assertEqual(select_commission(candidates)["tier"], "中阶")

    def test_panel_separates_own_and_recommended_rank(self):
        result = read_panel(panel_rows(), 1600, 900)
        self.assertEqual((result["my_rank"], result["recommended"], result["tier"]), (19, 17, "低阶"))
        self.assertEqual(set(result["tier_positions"]), set(TIERS))
        wrong = panel_rows() + [text("RANK25", 885, 551)]
        self.assertIsNone(read_panel(wrong, 1600, 900)["recommended"])

    def test_prompt_accepts_joined_rank_and_locates_recommendation(self):
        for recommendation in ("battle", "skip", None):
            result = read_battle_prompt(prompt_rows(recommendation=recommendation), 1600, 900)
            self.assertEqual(result["recommended"], 23)
            self.assertEqual(result["recommendation"], recommendation)
            self.assertTrue(result["commission"])

    def test_ambiguous_low_confidence_or_missing_buttons_are_not_clickable(self):
        rows = prompt_rows()
        self.assertIsNone(read_battle_prompt(rows[:-1], 1600, 900))
        self.assertIsNone(read_battle_prompt(rows + [text("开始委托", 930, 660)], 1600, 900))
        self.assertIsNone(read_battle_prompt(rows[:-1] + [text("开始委托", 925, 653, .4)], 1600, 900))
        rows.extend([text("推荐", 685, 610), text("推荐", 925, 610)])
        self.assertTrue(read_battle_prompt(rows, 1600, 900)["recommendation_ambiguous"])

    def test_goal_text_cannot_classify_a_commission_as_appraisal(self):
        detector = ScreenDetector(Mock())
        detector.read_region_text = Mock(side_effect=["参加评鉴战", "史莱姆讨伐委托I低阶委托"])
        self.assertFalse(detector.is_appraisal_screen(np.zeros((900, 1600, 3), np.uint8)))

    def test_native_title_crop_repairs_full_frame_character_error(self):
        rows = panel_rows()
        rows[2] = text("史莱姆讨代委托工", 980, 270)
        ocr = Mock()
        crop_title = text("史莱姆讨伐委托I", 180, 36, .87)
        ocr.recognize_detailed.side_effect = [rows, [crop_title]]
        image = np.zeros((900, 1600, 3), np.uint8)
        result = read_panel(CommissionHandler._panel_items(image, ocr), 1600, 900)
        self.assertEqual(result["title"], "史莱姆讨伐委托I")
        self.assertEqual(result["recommended"], 17)
        self.assertEqual(ocr.recognize_detailed.call_args.args[0].shape[:2], (58, 400))

    def test_native_title_crop_does_not_relax_confidence(self):
        rows = panel_rows()
        rows[2] = text("史莱姆讨代委托工", 980, 270)
        ocr = Mock()
        ocr.recognize_detailed.side_effect = [rows, [text("史莱姆讨伐委托I", 180, 36, .4)]]
        image = np.zeros((900, 1600, 3), np.uint8)
        self.assertIsNone(read_panel(CommissionHandler._panel_items(image, ocr), 1600, 900)["title"])

    def context(self):
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=Mock(),
                              logger=Mock(), pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        return ctx

    def test_all_tiers_are_logged_before_accept_and_four_level_gap_battles(self):
        handler, ctx = CommissionHandler(), self.context()
        frame = ctx.capture.capture_game.return_value
        panels = [read_panel(panel_rows(tier, 19, rank), 1600, 900)
                  for tier, rank in zip(TIERS, [23, 21, 17])]
        handler._initial_panel = Mock(return_value=(frame, panels[2]))
        handler._wait_panel = Mock(side_effect=[*[(frame, p) for p in panels], (frame, panels[0])])
        ctx.ocr.recognize_detailed.return_value = prompt_rows(23)
        self.assertTrue(handler.handle(ctx))
        candidate_calls = [call for call in ctx.logger.record_decision_frame.call_args_list
                           if call.args[0] in TIERS]
        self.assertEqual([call.args[0] for call in candidate_calls], list(TIERS))
        self.assertEqual(ctx.logger.log_commission.call_args.args, ("高阶", 19, 23))
        self.assertEqual(ctx.logger.log_commission.call_args.kwargs["mode"], "battle")
        self.assertEqual(ctx.controller.click_at_percent.call_args.args, (925/1600, 653/900))
        ctx.controller.click_center_multi.assert_not_called()

    def test_selected_low_tier_is_not_clicked_again_and_collapsed(self):
        handler, ctx = CommissionHandler(), self.context()
        image = ctx.capture.capture_game.return_value
        panels = [read_panel(panel_rows(tier, 38, rank), 1600, 900)
                  for tier, rank in zip(TIERS, [52, 45, 42])]
        handler._initial_panel = Mock(return_value=(image, panels[2]))
        handler._wait_panel = Mock(side_effect=[*[(image, p) for p in panels], (image, panels[2])])
        ctx.ocr.recognize_detailed.return_value = prompt_rows(42)
        handler.handle(ctx)
        clicks = [call.args for call in ctx.controller.click_at_percent.call_args_list]
        self.assertEqual(clicks.count(handler.TIER_CLICK_POINTS["低阶"]), 1)
        self.assertEqual(ctx.logger.log_commission.call_args.args, ("低阶", 38, 42))
        self.assertEqual(ctx.logger.log_battle_choice.call_args.args[0], "battle")

    def test_collapsed_verified_list_opens_high_tier_before_scanning(self):
        handler, ctx = CommissionHandler(), self.context()
        rows = [row for row in panel_rows() if not (
            row[0].startswith("史莱姆") or row[0] in {"建议综合等级", "RANK17", "接受"}
            or row[0] == "低阶委托" and row[2][0][0] < 1100)]
        handler._panel_items = Mock(return_value=rows)
        handler._wait_panel = Mock(return_value=(ctx.capture.capture_game.return_value, {}))
        handler._initial_panel(ctx)
        ctx.controller.click_at_percent.assert_called_once_with(*handler.TIER_CLICK_POINTS["高阶"])
        handler._wait_panel.assert_called_once_with(ctx, "高阶")

    def test_stale_commission_frame_yields_to_current_battle_without_input(self):
        handler, ctx = CommissionHandler(), self.context()
        handler._panel_items = Mock(return_value=[
            text("BURST", 1140, 810), text("1/1", 125, 56)])
        handler._wait_panel = Mock()
        self.assertFalse(handler.handle(ctx))
        handler._wait_panel.assert_not_called()
        ctx.controller.click_at_percent.assert_not_called()
        self.assertIs(ctx.pending_screenshot, ctx.capture.capture_game.return_value)

    def emergency_panel(self, rank=51, recommended=60):
        rows = [row for row in panel_rows("高阶", rank, recommended)
                if not (row[0] in {"中阶委托", "低阶委托"}
                        or row[0] == "高阶委托" and row[2][0][0] > 1100)]
        rows[2] = text("弗洛拉紧急讨伐委托", 980, 270)
        rows += [text("弗洛拉紧急讨伐委托", 1380, 416),
                 text("高阶委托", 1340, 452)]
        return rows

    def test_single_emergency_requires_matching_list_and_detail_titles(self):
        rows = self.emergency_panel()
        self.assertTrue(read_panel(rows, 1600, 900)["single_tier_emergency"])
        for changed in (rows[:-2] + rows[-1:],
                        rows + [text("中阶委托", 1340, 563)],
                        rows[:-2] + [text("其他紧急讨伐委托", 1380, 416), rows[-1]]):
            self.assertFalse(read_panel(changed, 1600, 900)["single_tier_emergency"])

    def test_single_emergency_over_gap_records_actual_candidate_and_exits(self):
        handler, ctx = CommissionHandler(), self.context()
        image = ctx.capture.capture_game.return_value
        panel = read_panel(self.emergency_panel(), 1600, 900)
        handler._initial_panel = Mock(return_value=(image, panel))
        handler._wait_panel = Mock(return_value=(image, panel))
        handler._exit_commission = Mock()
        with self.assertRaisesRegex(InputTargetError, "无可降档项"):
            handler.handle(ctx)
        ctx.logger.begin_decision.assert_called_once_with("commission", ("高阶",))
        self.assertEqual(ctx.logger.log_commission.call_args.kwargs["reason"], "no_eligible_tier")
        self.assertEqual(len(ctx.logger.log_commission.call_args.kwargs["candidates"]), 1)
        handler._exit_commission.assert_called_once_with(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_emergency_exit_accepts_verified_commission_trade_menu(self):
        handler, ctx = CommissionHandler(), self.context()
        rows = [text("RANK51", 130, 282), text("委托", 1428, 416), text("交易", 1428, 506)]
        ctx.ocr.recognize_detailed.return_value = rows
        with patch("src.handlers._wait_for_screen", return_value=False):
            handler._exit_commission(ctx)
        ctx.controller.send_escape.assert_called_once()
        self.assertIs(ctx.pending_screenshot, ctx.capture.capture_game.return_value)
        for bad in (rows[:-1], rows + [rows[1]], [text("RANK51", 900, 500)] + rows[1:]):
            ctx.ocr.recognize_detailed.return_value = bad
            self.assertFalse(handler._is_commission_trade_menu(ctx.pending_screenshot, ctx.ocr))

    def test_authorized_single_emergency_battles_without_lowering_normal_limit(self):
        handler = CommissionHandler({"commission": {"single_tier_emergency_battle": True}})
        ctx = self.context()
        image = ctx.capture.capture_game.return_value
        panel = read_panel(self.emergency_panel(), 1600, 900)
        handler._initial_panel = Mock(return_value=(image, panel))
        handler._wait_panel = Mock(return_value=(image, panel))
        rows = prompt_rows(60, "skip")
        rows[0] = text("弗洛拉紧急讨伐委托", 808, 278)
        ctx.ocr.recognize_detailed.return_value = rows
        self.assertTrue(handler.handle(ctx))
        self.assertEqual(ctx.logger.log_commission.call_args.kwargs["mode"], "battle")
        self.assertEqual(ctx.logger.log_commission.call_args.kwargs["reason"],
                         "single_tier_emergency_progress")
        self.assertFalse(ctx.logger.log_commission.call_args.kwargs["candidates"][0]["eligible"])
        self.assertEqual(ctx.controller.click_at_percent.call_args.args, (925/1600, 653/900))
        self.assertIsNone(battle_mode(51, 60))
        ctx.controller.send_escape.assert_not_called()

    def test_all_normal_tiers_over_limit_choose_lowest_rank_and_log_deviation(self):
        handler = CommissionHandler({"safety": {"progress_fallbacks": True}})
        ctx = self.context()
        image = ctx.capture.capture_game.return_value
        panels = [read_panel(panel_rows(tier, 38, rank), 1600, 900)
                  for tier, rank in zip(TIERS, [60, 52, 45])]
        handler._initial_panel = Mock(return_value=(image, panels[2]))
        handler._wait_panel = Mock(side_effect=[*[(image, p) for p in panels], (image, panels[2])])
        ctx.ocr.recognize_detailed.return_value = prompt_rows(45)
        self.assertTrue(handler.handle(ctx))
        self.assertEqual(ctx.logger.log_commission.call_args.args, ("低阶", 38, 45))
        self.assertEqual(ctx.logger.log_commission.call_args.kwargs["reason"],
                         "no_eligible_tier_lowest_rank_battle")
        self.assertEqual(ctx.logger.log_battle_choice.call_args.args[0], "battle")
        ctx.logger.log_fallback.assert_called_once()
        self.assertEqual(len(ctx.logger.log_fallback.call_args.kwargs["candidates"]), 3)
        ctx.controller.send_escape.assert_not_called()

    def test_emergency_menu_uses_verified_icon_and_shop_yields(self):
        config = {"commission": {"single_tier_emergency_battle": True}}
        handler, ctx = CommissionHandler(config), self.context()
        image = ctx.capture.capture_game.return_value
        ctx.ocr.recognize_detailed.return_value = [
            text("RANK51", 130, 282), text("委托", 1428, 416), text("交易", 1428, 506)]
        handler._wait_panel = Mock(return_value=(image, {}))
        self.assertTrue(handler.can_handle(FrameContext(image), ctx.ocr))
        self.assertFalse(ShopHandler(config).can_handle(FrameContext(image), ctx.ocr))
        self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(.941, 416/900)
        handler._wait_panel.assert_called_once_with(ctx)

    def test_emergency_option_does_not_accept_incomplete_normal_list(self):
        handler = CommissionHandler({"commission": {"single_tier_emergency_battle": True}})
        ctx = self.context()
        ctx.ocr.recognize_detailed.return_value = []
        panel = read_panel(panel_rows("高阶", 51, 60), 1600, 900)
        del panel["tier_positions"]["中阶"]
        handler._initial_panel = Mock(return_value=(ctx.capture.capture_game.return_value, panel))
        with self.assertRaises(InputTargetError):
            handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_missing_normal_tiers_blocks_before_any_input(self):
        handler, ctx = CommissionHandler(), self.context()
        panel = read_panel(panel_rows(), 1600, 900)
        del panel["tier_positions"]["中阶"]
        handler._initial_panel = Mock(return_value=(ctx.capture.capture_game.return_value, panel))
        with self.assertRaises(InputTargetError):
            handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_ui_battle_recommendation_overrides_skip_but_not_the_reverse(self):
        for gap, recommendation, expected in [(2, "battle", "battle"), (4, "skip", "battle"),
                                                (2, None, "skip")]:
            with self.subTest(gap=gap, recommendation=recommendation):
                ctx = self.context()
                ctx.ocr.recognize_detailed.return_value = prompt_rows(19+gap, recommendation)
                selected = {"title": "史莱姆讨伐委托工", "recommended": 19+gap,
                            "mode": battle_mode(19, 19+gap)}
                CommissionHandler()._choose_battle(ctx, selected)
                self.assertEqual(ctx.logger.log_battle_choice.call_args.args[0], expected)

    def test_changed_confirmation_rank_blocks_input(self):
        ctx = self.context()
        ctx.ocr.recognize_detailed.return_value = prompt_rows(24)
        with self.assertRaises(InputTargetError):
            CommissionHandler()._choose_battle(ctx, {"title": "史莱姆讨伐委托工", "recommended": 23, "mode": "battle"})
        ctx.controller.click_at_percent.assert_not_called()

    def test_generic_skip_cannot_skip_an_unrecorded_commission(self):
        ctx, handler = self.context(), SkipHandler()
        handler._skip_type = "battle"
        ctx.ocr.recognize_detailed.return_value = prompt_rows()
        with patch.object(CommissionHandler, "_wait_panel"):
            self.assertTrue(handler.handle(ctx))
        ctx.controller.send_escape.assert_called_once()
        ctx.controller.click_at_percent.assert_not_called()
        ctx.controller.click_center_multi.assert_not_called()

    def test_combat_auto_is_requested_only_once_on_confirmed_battle(self):
        ctx, handler = self.context(), CombatHandler()
        handler.can_handle = Mock(return_value=True)
        handler._auto_is_off = Mock(return_value=True)
        handler.handle(ctx)
        handler.handle(ctx)
        ctx.controller.click_at_percent.assert_called_once_with(*handler.AUTO_POSITION)

    def test_combat_does_not_toggle_unknown_auto_icon(self):
        ctx, handler = self.context(), CombatHandler()
        handler.can_handle = Mock(return_value=True)
        handler._auto_is_off = Mock(return_value=False)
        handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_combat_timeout_stops_without_extra_input(self):
        ctx, handler = self.context(), CombatHandler()
        handler.can_handle = Mock(return_value=True)
        ctx.battle_started_at = 1
        with patch("src.handlers.time.monotonic", return_value=200), self.assertRaises(InputTargetError):
            handler.handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_combat_animation_without_burst_stays_in_battle_dispatch(self):
        frame = Mock()
        frame.get_ocr.side_effect = ["", "1/1", "总伤害量"]
        self.assertTrue(CombatHandler().can_handle(frame, Mock()))
        frame.get_ocr.side_effect = ["", "1/1", "获得奖励"]
        self.assertFalse(CombatHandler().can_handle(frame, Mock()))

    def test_moving_combat_is_not_timed_out_by_total_duration(self):
        ctx, handler = self.context(), CombatHandler()
        handler.can_handle = Mock(return_value=True)
        handler._auto_is_off = Mock(return_value=False)
        ctx.battle_started_at = 1
        ctx.battle_progress_frame = np.full((36, 64, 3), 100, np.uint8)
        with patch("src.handlers.time.monotonic", return_value=200):
            self.assertTrue(handler.handle(ctx))
        self.assertEqual(ctx.battle_last_progress_at, 200)
        ctx.controller.click_at_percent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
