"""Goal-local planning and activity routing; no desktop input."""
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import web_ui
from src.automation_preferences import DEFAULTS, JourneyProgress, preferences, validate_preferences
from src.journey_plan import GOALS, default_goals, goal_instruction, menu_choice
from src.handlers import FrameContext, GoalMenuHandler, CommissionHandler
from src.notifications import SoundNotifier


def row(text, x, y):
    return text, .99, [[x-20, y-10], [x+20, y-10], [x+20, y+10], [x-20, y+10]]


class GoalPlanTests(unittest.TestCase):
    def test_verified_late_goals_are_three_action_slots_not_three_training_actions(self):
        config = default_goals()
        self.assertEqual(config["void"]["turn_count"], 3)
        self.assertEqual(config["final"]["turn_count"], 3)
        progress = JourneyProgress()
        for month, goal in [(1, "void"), (2, "final")]:
            for index, part in enumerate(("上旬", "中旬", "下旬"), 1):
                state = progress.observe([row(f"{month}月{part}", 260, 89), row(str(4-index), 100, 83)], "训练决策")
                self.assertEqual((state["goal_id"], state["goal_turn"]), (goal, index))
                progress.prepare("rest" if index == 3 else "train")

    def test_checklist_does_not_select_future_goal(self):
        progress = JourneyProgress()
        progress.observe([row("在最终评鉴战中获胜。", 800, 550)], "目标提示")
        self.assertIsNone(progress.data["goal_id"])
        progress.observe([row("虚空评鉴战准备", 800, 550)], "目标提示")
        self.assertEqual(progress.data["goal_id"], "void")
        self.assertIsNone(progress.data["goal_turn"])

    def test_rest_resumes_same_stage_from_checkpoint_without_becoming_dispatch_count(self):
        progress = JourneyProgress(saved={"goal_id": "void", "goal_turn": 2, "after_void": False})
        for _ in range(20):
            progress.observe([], "未知兜底")
        self.assertEqual(progress.data["goal_turn"], 2)

    def test_goal_specific_action_and_policy_override_global(self):
        plan = copy.deepcopy(DEFAULTS["turn_plan"])
        plan.update(enabled=True, default_action="rest", subjugation_policy="prefer")
        plan["goals"]["void"].update(default_action="train", subjugation_policy="allow",
                                    commission_policy="allow", turns={"2": "commission"})
        result = goal_instruction(plan, {"goal_id": "void", "goal_turn": 2})
        self.assertEqual(result["action"], "commission")
        self.assertEqual(result["source"], "goal_turn")
        self.assertEqual(menu_choice(result, {"train", "rest", "commission", "subjugation"})[0], "commission")

    def test_default_prefers_announced_subjugation_and_never_optional_commission(self):
        instruction = goal_instruction(DEFAULTS["turn_plan"], {})
        self.assertEqual(menu_choice(instruction, {"train", "rest", "subjugation"})[0], "subjugation")
        self.assertEqual(menu_choice(instruction, {"train", "rest", "commission"})[0], "train")
        instruction["action"] = "commission"
        self.assertEqual(menu_choice(instruction, {"train", "rest", "commission"}),
                         ("train", "unavailable_or_forbidden_fallback"))
        instruction["commission_policy"] = "allow"
        self.assertEqual(menu_choice(instruction, {"train", "rest", "commission"})[0], "commission")

    def test_forbidden_subjugation_is_not_a_fallback(self):
        instruction = {"action": "subjugation", "subjugation_policy": "skip", "commission_policy": "skip"}
        self.assertEqual(menu_choice(instruction, {"subjugation", "rest"})[0], "rest")
        self.assertIsNone(menu_choice(instruction, {"subjugation", "commission"})[0])

    def test_goal_turn_outside_verified_budget_is_rejected(self):
        data = preferences({})
        data["turn_plan"]["goals"]["void"]["turns"]["4"] = "train"
        with self.assertRaises(ValueError):
            validate_preferences(data)

    def test_menu_routes_revalidated_actual_entry_not_training_only(self):
        image = np.zeros((900, 1600, 3), np.uint8)
        ocr = Mock()
        ocr.recognize_detailed.return_value = [
            row("训练", 1425, 355), row("委托", 1425, 466), row("休息", 1425, 575)]
        ctx = SimpleNamespace(capture=Mock(), ocr=ocr, controller=Mock(), logger=Mock(),
                              activity_instruction=lambda: {"action": "commission",
                                  "commission_policy": "allow", "subjugation_policy": "allow"})
        ctx.capture.capture_game.return_value = image
        handler = GoalMenuHandler()
        self.assertTrue(handler.can_handle(FrameContext(image), ocr))
        self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(.941, 466/900)
        self.assertEqual(ctx.commission_origin, "commission")
        ocr.recognize_detailed.return_value.append(row("受理讨伐委托!", 800, 600))
        self.assertIn("subjugation", handler._entries(FrameContext(image), ocr))
        self.assertNotIn("commission", handler._entries(FrameContext(image), ocr))

    def test_changed_menu_never_replays_entry_click(self):
        ctx = SimpleNamespace(capture=Mock(), ocr=Mock(), controller=Mock(), logger=Mock())
        ctx.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        ctx.ocr.recognize_detailed.return_value = [row("继承旅程", 800, 200)]
        self.assertFalse(GoalMenuHandler().handle(ctx))
        ctx.controller.click_at_percent.assert_not_called()

    def test_native_preview_executes_same_player_and_reports_failure(self):
        with patch.object(web_ui.SoundNotifier, "_play") as play:
            self.assertTrue(web_ui.play_notification_preview({"volume": 50})["played"])
            self.assertTrue(play.call_args.args[0].startswith(b"RIFF"))
            self.assertTrue(web_ui.play_notification_preview({"volume": 0})["muted"])
            play.assert_called_once()
        with patch.object(web_ui.SoundNotifier, "_play", side_effect=RuntimeError("no device")):
            with self.assertRaises(web_ui.HTTPException) as result:
                web_ui.play_notification_preview({"volume": 50})
            self.assertEqual(result.exception.status_code, 503)

    def test_native_player_supports_python_312_without_snd_sync(self):
        winsound = SimpleNamespace(SND_MEMORY=4, PlaySound=Mock())
        with patch.dict(sys.modules, {"winsound": winsound}):
            SoundNotifier._play(b"wave")
        winsound.PlaySound.assert_called_once_with(b"wave", 4)


if __name__ == "__main__":
    unittest.main()
