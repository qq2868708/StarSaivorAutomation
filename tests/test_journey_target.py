"""Goals must change real decisions and fail before input on ambiguous evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

import web_ui
from src.handlers import EventHandler, CardSelectHandler
from src.journey_target import (JourneyTargetStore, JourneyTargetPending, direction_choice,
                                equipment_choice, load_catalog, precheck_target)
from src.rule_engine import BuildDirection, TrainingRuleLoader
from src.runtime_control import RuntimeControl
from test_runtime_session import make_trainer


def goal(**changes):
    return {"character": "格温", "playstyle": "pve", "combo_code": "AX5",
            "training_direction": "attack",
            "has_partner": False, **changes}


class JourneyTargetTests(unittest.TestCase):
    def test_catalog_contains_real_equipment_not_legacy_skill_names(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog["combos"]), 67)
        self.assertEqual(len({item["code"] for item in catalog["combos"]}), 67)
        ax5 = precheck_target(goal())["combo"]
        self.assertEqual((ax5["initial_material"], ax5["partner_material"]), ("攻击", "暴击"))
        self.assertIn("每回合", ax5["effect_text"])

    def test_precheck_blocks_missing_or_invalid_inputs(self):
        for changes in ({"character": ""}, {"combo_code": "IX1"},
                        {"playstyle": "unknown"}, {"has_partner": "yes"},
                        {"training_direction": ["attack"]}, {"training_direction": "auto"}):
            with self.subTest(changes=changes):
                self.assertFalse(precheck_target(goal(**changes))["ready"])

    def test_precheck_derives_direction_without_claiming_owned_equipment(self):
        result = precheck_target(goal())
        self.assertTrue(result["ready"])
        target = result["target"]
        self.assertEqual(target["event_direction"], "attack")
        self.assertEqual(target["direction_source"], "combo_code")
        self.assertEqual(target["required_initial_material"], "攻击")
        self.assertFalse(target["has_partner"])
        self.assertNotIn("initial_material", target)
        self.assertNotIn("direction_confirmed", target)

    def test_missing_catalog_direction_blocks_precheck_and_selection(self):
        catalog = load_catalog()
        next(item for item in catalog["combos"] if item["code"] == "AX5").pop("event_direction")
        self.assertFalse(precheck_target(goal(), catalog)["ready"])
        with patch("src.journey_target.load_catalog", return_value=catalog):
            with self.assertRaises(JourneyTargetPending):
                direction_choice(goal(), [])

    def test_manual_character_supported_without_inventing_recommendations(self):
        result = precheck_target(goal(character="尚未收录的角色"))
        self.assertTrue(result["ready"])
        self.assertEqual(result["recommendations"], [])
        self.assertIn("手动", result["warnings"][0])

    def test_direction_uses_text_even_when_rows_are_reordered(self):
        rows = [{"text": text, "enabled": True} for text in
                ["战术向声援", "生存向声援", "攻击向声援", "综合向声援"]]
        self.assertEqual(direction_choice(goal(), rows), 2)
        self.assertEqual(direction_choice(goal(combo_code="BX5"), rows), 1)
        rows[2]["enabled"] = False
        with self.assertRaises(JourneyTargetPending):
            direction_choice(goal(), rows)
        with self.assertRaises(JourneyTargetPending):
            direction_choice(None, rows)

    def test_direction_follows_combo_starting_type_across_all_groups(self):
        rows = [{"text": text, "enabled": True} for text in
                ["有助于应对各种状况的训练声援", "有助于战术运用的声援",
                 "对生存有帮助的训练声援", "对攻击有帮助的训练声援"]]
        for code, expected in [("AX2", 3), ("BX1", 2), ("CX1", 2), ("DX4", 2),
                               ("EX5", 3), ("FX6", 3), ("GX7", 0), ("HX8", 0),
                               ("XX1", 1), ("XX2", 1), ("XX3", 1)]:
            with self.subTest(code=code):
                self.assertEqual(direction_choice(goal(combo_code=code), rows), expected)

    def test_legacy_manual_inputs_cannot_override_combo_direction(self):
        old = goal(initial_material="生命", event_direction="survival", direction_confirmed=False)
        result = precheck_target(old)
        self.assertTrue(result["ready"])
        self.assertEqual(result["target"]["event_direction"], "attack")
        rows = [{"text": text, "enabled": True} for text in
                ["攻击向声援", "生存向声援", "综合向声援", "战术向声援"]]
        self.assertEqual(direction_choice(old, rows), 0)
        with self.assertRaises(JourneyTargetPending):
            direction_choice(goal(combo_code="IX1"), rows)

    def test_equipment_matches_partner_and_does_not_confuse_crit_damage(self):
        texts = ["爆伤装备 暴击伤害+10%", "暴击装备 暴击率+5%", "速度装备 速度+4"]
        self.assertEqual(equipment_choice(goal(), texts)["index"], 1)
        fx6 = goal(combo_code="FX6")
        self.assertEqual(equipment_choice(fx6, texts)["index"], 0)
        self.assertIsNone(equipment_choice(goal(), ["每回合提高攻击力", "速度提高", "普通攻击暴击"]))
        for texts in (["攻击装备 攻击力+4%"], ["暴击装备", "暴击装备"], ["装备图片"]):
            with self.subTest(texts=texts), self.assertRaises(JourneyTargetPending):
                equipment_choice(goal(), texts)

    def test_direction_event_needs_goal_and_sends_no_input_without_one(self):
        detector = Mock()
        detector.read_region_text.return_value = "旅程事件训练的方向性"
        detector.read_event_option_rows.return_value = [{"text": term, "enabled": True, "y": .5}
            for term in ("攻击向声援", "生存向声援", "综合向声援", "战术向声援")]
        handler = EventHandler({"events": {"require_document_effects": True}})
        ctx = SimpleNamespace(ocr=Mock(), controller=Mock(), journey_target=None)
        with patch("src.recognition.ScreenDetector", return_value=detector):
            with self.assertRaises(JourneyTargetPending):
                handler._handle_document_event(ctx, Mock())
        ctx.controller.click_at_percent.assert_not_called()

    def test_direction_handler_uses_goal_then_confirms_screen_transition(self):
        detector = Mock()
        detector.read_region_text.side_effect = ["旅程事件训练的方向性", "主菜单"]
        rows = [{"text": term, "enabled": True, "y": y} for term, y in
                [("对攻击有帮助的训练声援", .52), ("对生存有帮助的训练声援", .59),
                 ("有助于应对各种状况的训练声援", .66), ("有助于战术运用的声援", .73)]]
        detector.read_event_option_rows.side_effect = [rows, []]
        handler = EventHandler({"events": {"require_document_effects": True}})
        target = {**precheck_target(goal(combo_code="BX5"))["target"], "revision": "target"}
        ctx = SimpleNamespace(ocr=Mock(), controller=Mock(), capture=Mock(), logger=Mock(),
                              journey_target=target)
        with patch("src.recognition.ScreenDetector", return_value=detector):
            self.assertTrue(handler._handle_document_event(ctx, Mock()))
        ctx.controller.click_at_percent.assert_called_once_with(.8, .59)
        self.assertEqual(ctx.logger.log_event.call_args.kwargs["target_decision"]["character"], "格温")
        self.assertEqual(ctx.logger.log_event.call_args.kwargs["target_decision"]["event_direction"], "survival")
        self.assertEqual(ctx.logger.log_event.call_args.kwargs["target_decision"]["direction_source"], "combo_code")

    def test_ambiguous_equipment_handler_never_clicks(self):
        ctx = SimpleNamespace(capture=Mock(), ocr=Mock(), controller=Mock(),
                              engine=SimpleNamespace(is_rule_engine_mode=False), journey_target=goal())
        with patch("src.handlers._ocr_region", return_value="装备图片，属性不明"):
            with self.assertRaises(JourneyTargetPending):
                CardSelectHandler().handle(ctx)
        ctx.controller.click_at_percent.assert_not_called()

    def test_equipment_receipt_is_checkpointed_after_confirmation(self):
        detector = Mock()
        detector.is_main_menu_screen.return_value = 'main_menu'
        detector.read_region_text.return_value = '主菜单'
        ctx = SimpleNamespace(capture=Mock(), ocr=Mock(), controller=Mock(), logger=Mock(),
            target_selection_phase=Mock(), journey_target=goal(revision='goal'))
        choice = {'index': 1, 'material': '暴击', 'combo_code': 'AX5'}
        with patch('src.recognition.ScreenDetector', return_value=detector), \
             patch('src.handlers._ocr_region', return_value='主菜单'):
            self.assertTrue(CardSelectHandler()._execute_target_card(ctx, ['攻击装备', '暴击装备', '速度装备'], choice))
        self.assertEqual(ctx.target_selection_phase.call_args_list[0].args[0], 'prepared')
        self.assertEqual(ctx.target_selection_phase.call_args_list[1].args[0], 'confirmed')
        self.assertEqual(ctx.controller.click_at_percent.call_count, 2)

    def test_runtime_applies_direction_and_keeps_uncertain_progress_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            target_store = JourneyTargetStore(root)
            target_store.save(goal(training_direction="survival"))
            trainer = make_trainer(root)
            trainer.engine.rule_profile = TrainingRuleLoader.create_builtin_default()
            trainer._refresh_journey_target()
            self.assertEqual(trainer.engine.rule_profile.legacy_strategy.build_direction, BuildDirection.SURVIVAL)
            self.assertEqual(trainer.engine.priority, 2)
            self.assertEqual(EventHandler._event_build_direction(trainer._handler_ctx), "survival")
            self.assertEqual(trainer._handler_ctx.journey_target["event_direction"], "attack")
            trainer._target_selection_phase("prepared", {"index": 1, "material": "暴击"})
            resumed = make_trainer(root)
            resumed.engine.rule_profile = TrainingRuleLoader.create_builtin_default()
            resumed._run_id = "new_run"
            resumed._load_resume_checkpoint()
            with self.assertRaises(JourneyTargetPending):
                resumed._refresh_journey_target()
            self.assertEqual(resumed.controller.sent, [])
            target_store.save(goal(has_partner=True), "new_run")
            resumed._refresh_journey_target()
            self.assertTrue(resumed._handler_ctx.journey_target["has_partner"])
            self.assertIsNone(resumed._journey_target_state["pending"])

    def test_runtime_normalizes_legacy_manual_direction_and_keeps_inventory_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            store = JourneyTargetStore(root)
            target = store.save(goal())
            target.update(initial_material="生命", event_direction="tactical", direction_confirmed=True)
            store.path.write_text(json.dumps(target), encoding="utf-8")
            trainer = make_trainer(root)
            trainer.engine.rule_profile = TrainingRuleLoader.create_builtin_default()
            trainer._refresh_journey_target()
            active = trainer._handler_ctx.journey_target
            self.assertEqual(active["event_direction"], "attack")
            self.assertEqual(active["direction_source"], "combo_code")
            self.assertEqual(active["required_initial_material"], "攻击")
            self.assertFalse(active["has_partner"])
            self.assertNotIn("initial_material", active)
            self.assertNotIn("direction_confirmed", active)

    def test_different_journey_goal_cannot_silently_be_reused(self):
        with tempfile.TemporaryDirectory() as root:
            JourneyTargetStore(root).save(goal(), "another_journey")
            trainer = make_trainer(root)
            with self.assertRaises(JourneyTargetPending):
                trainer._do_round()
            self.assertEqual(trainer.controller.sent, [])

    def test_api_saves_only_at_pause_and_cannot_start_without_valid_goal(self):
        with tempfile.TemporaryDirectory() as root:
            control = RuntimeControl(root)
            with patch.object(web_ui, "_runtime_control", control), patch.object(web_ui, "_automation_process", None):
                with patch.object(web_ui.subprocess, "Popen") as start:
                    with self.assertRaises(HTTPException):
                        web_ui.start_runtime()
                    start.assert_not_called()
                control.write_status("running", run_id="run")
                with self.assertRaises(HTTPException):
                    web_ui.save_journey_target(goal())
                control.write_status("paused_safe", run_id="run")
                result = web_ui.save_journey_target(goal())
                self.assertEqual(result["target"]["scope_run_id"], "run")
                self.assertFalse(result["applied"])
                with self.assertRaises(HTTPException):
                    web_ui.resume_runtime()  # Old script must load the new code explicitly.
                self.assertIsNone(control.consume())


if __name__ == "__main__":
    unittest.main()
