"""Evidence quality, interrupted inputs and resumed journey grouping; no game input."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

import web_ui
from types import SimpleNamespace
from unittest.mock import Mock
from src.handlers import JourneyEndHandler, JourneyEndException
from src.journey_record import JourneyRecordStore, normalize_setup, screen_observation
from src.journey_target import precheck_target
from src.logger import RunLogger
from src.runtime_control import RuntimeControl, RuntimeInterrupt
from src.rule_engine import TrainingRuleLoader
from test_runtime_session import make_trainer


def target(**changes):
    checked = precheck_target({"character": "格温", "playstyle": "pve", "combo_code": "AX5",
        "training_direction": "attack", "actual_setup": {"character": "格温",
        "journey_records": ["记录甲"], "support_cards": [{"name": "支援甲", "level": 60, "borrowed": True}]}, **changes})
    assert checked["ready"], checked
    return {**checked["target"], "revision": "goal", "saved_at": "2026-10-08T10:00:00+08:00"}


class JourneyRecordTests(unittest.TestCase):
    def test_unknown_setup_is_different_from_confirmed_empty_loadout(self):
        self.assertIsNone(normalize_setup({})["support_cards"])
        self.assertEqual(normalize_setup({"support_cards": []})["support_cards"], [])
        self.assertEqual(normalize_setup({"support_cards": [{"name": " 卡甲 ", "level": 60, "limit_break": 2, "borrowed": True, "slot": 1}]})["support_cards"][0]["name"], "卡甲")
        for setup in ({"support_cards": "卡甲"}, {"support_cards": [{"name": "卡", "level": True}]},
                      {"journey_records": [{"name": "卡", "borrowed": "yes"}]}):
            with self.subTest(setup=setup), self.assertRaises(ValueError):
                normalize_setup(setup)

    def test_only_labelled_current_values_are_recorded_not_rewards_or_suggestions(self):
        rows = [(text, .99, None) for text in ("力量 339/1250", "体力: 0", "韧性+15", "建议综合等级 30",
             "专注训练 +20", "保护减少10", "装备技能: AX5、XX1", "潜质点数: 80", "SUCCESS")]
        values = screen_observation(rows, "旅程结束", 5)["values"]
        self.assertEqual(values["力量"], 339)
        self.assertEqual(values["体力"], 0)
        self.assertEqual(values["equipped_skills"], ["AX5", "XX1"])
        self.assertEqual(values["potential_points"], 80)
        for key in ("韧性", "专注", "保护", "rank", "appraisal_result"):
            self.assertNotIn(key, values)

    def test_separate_label_and_number_require_unambiguous_geometry_and_confidence(self):
        def box(x, y, width=40):
            return [[x, y], [x+width, y], [x+width, y+20], [x, y+20]]
        rows = [("力量", .99, box(20, 20)), ("339/1250", .99, box(20, 44)),
                ("体力", .99, box(20, 100)), ("200/1250", .5, box(20, 124))]
        observation = screen_observation(rows, "主菜单", 1)
        self.assertEqual(observation["values"], {"力量": 339})
        self.assertEqual(observation["value_evidence"]["力量"], [0, 1])
        rows.append(("340/1250", .99, box(80, 20)))
        self.assertNotIn("力量", screen_observation(rows, "主菜单", 1)["values"])

    def test_duplicate_or_low_confidence_labels_remain_unknown(self):
        rows = [("力量 20", .99), ("力量 300", .99), ("体力 50", .4), ("专注 20", float('nan'))]
        self.assertEqual(screen_observation(rows, "事件选择", 1)["values"], {})

    def test_game_date_and_turn_are_observed_without_using_dispatch_count(self):
        observed = screen_observation([('@3月中旬', .99), ('12/45', .99), ('保护 191/1250', .99)], '主菜单', 300)
        self.assertEqual(observed['values']['game_date'], '3月中旬')
        self.assertEqual(observed['values']['game_turn'], 12)
        self.assertEqual(observed['round'], 300)
        self.assertNotIn('game_turn', screen_observation([('距离目标 12', .99)], '主菜单', 300)['values'])

    def test_snapshot_before_input_preserves_open_turn_and_records_sources(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.configure_run("config", {}, TrainingRuleLoader.create_builtin_default())
            logger.set_target(target())
            logger.set_screenshot(np.zeros((64, 96, 3), np.uint8))
            before = logger.observe_frame([("力量 100", .99)], "训练决策", 1)
            logger.start_turn(1, 1, 80, "Good", 50)
            logger.log_train_exec(0, "力量")
            logger.log_input({"action_id": "input", "phase": "prepared", "source": "script"})
            log_path = logger.save("running", announce=False, close_turn=False)
            self.assertIsNotNone(logger._current_turn)
            data = json.loads(Path(log_path).read_text(encoding='utf-8'))
            self.assertEqual(data["turns"][0]["actions"][0]["observation_id"], before["id"])
            self.assertEqual(data["inputs"][0]["phase"], "prepared")
            self.assertTrue((Path(root) / before["screenshot"]).is_file())
            logger.log_input({"action_id": "input", "phase": "sent", "source": "script"})
            after = logger.observe_frame([("力量 120", .99)], "主菜单", 2)
            self.assertEqual(after["state_changes"], {"力量": 20})
            logger.save("safe_paused", announce=False)
            record = JourneyRecordStore(root).read(logger._journey["journey_id"])
            self.assertFalse(record["outcome"]["journey_ended"])
            self.assertIsNone(record["outcome"]["final_stats"]["力量"]["value"])
            self.assertEqual(record["actual_setup"]["support_cards"]["value"][0]["level"], 60)
            self.assertEqual(record["actual_setup"]["character"]["source"], "user_declared")
            self.assertEqual(len(record["setup_history"]), 1)

    def test_restart_segments_share_journey_without_fabricating_final_stats(self):
        with tempfile.TemporaryDirectory() as root:
            first = RunLogger(root=root)
            first.set_target(target())
            first.observe_frame([("力量 500", .99)], "主菜单", 10)
            first.save("safe_paused", announce=False)
            resumed = RunLogger(root=root)
            resumed.bind_journey(first._journey["journey_id"], first._journey["started_at"], first._run_id)
            resumed.set_target(target())
            resumed.observe_frame([("旅程结束", .99)], "旅程结束", 1, final=True)
            resumed.save("journey_end", announce=False)
            resumed.save("journey_end", announce=False)
            record = JourneyRecordStore(root).read(first._journey["journey_id"])
            self.assertEqual(len(record["segments"]), 2)
            self.assertEqual(record["segments"][-1]["parent_run_id"], first._run_id)
            self.assertEqual(len(record["target_history"]), 1)
            self.assertTrue(record["outcome"]["journey_ended"])
            self.assertIsNone(record["outcome"]["final_stats"]["力量"]["value"])
            self.assertIsNone(record["outcome"]["target_skill_achieved"]["value"])
            self.assertFalse(record["data_quality"]["ready_for_comparison"])

    def test_final_result_uses_equipped_skills_not_material_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.set_target(target(has_partner=True))
            logger.log_target_progress("confirmed", {"material": "暴击"})
            logger.observe_frame([("装备技能: BX1", .99), ("评鉴结果: 胜利", .99), ("力量 600", .99)], "旅程结束", 1, final=True)
            logger.save("journey_end", announce=False)
            record = JourneyRecordStore(root).read(logger._journey["journey_id"])
            self.assertFalse(record["outcome"]["target_skill_achieved"]["value"])
            self.assertEqual(record["outcome"]["appraisal_result"]["value"], "victory")
            self.assertEqual(record["outcome"]["final_stats"]["力量"]["value"], 600)

    def test_journey_summary_marks_exploration_after_completion(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.set_target(target())
            logger.log_exploration({"key": "e", "index": 1, "phase": "observed",
                                    "effects": {"numeric": {}, "qualitative": ["获得增益"]}})
            logger.observe_frame([("旅程结束", .99)], "旅程结束", 1, final=True)
            logger.save("journey_end", announce=False)
            record = JourneyRecordStore(root).read(logger._journey["journey_id"])
            self.assertEqual(record["outcome"]["execution_mode"], "exploration")
            self.assertTrue(record["exploration"]["included"])

    def test_manual_supplements_keep_provenance_and_survive_log_replay(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.set_target(target())
            logger.observe_frame([("力量 200", .99)], "旅程结束", 1, final=True)
            logger.save("journey_end", announce=False)
            store = JourneyRecordStore(root)
            store.supplement(logger._journey["journey_id"], {"final_stats": {"力量": 250, "保护": 0}, "equipped_skills": []})
            logger.save("journey_end", announce=False)
            record = store.read(logger._journey["journey_id"])
            self.assertEqual(record["outcome"]["final_stats"]["力量"]["value"], 250)
            self.assertEqual(record["outcome"]["final_stats"]["力量"]["source"], "user_declared")
            self.assertEqual(record["outcome"]["final_stats"]["保护"]["value"], 0)
            self.assertFalse(record["outcome"]["target_skill_achieved"]["value"])
            original = store.read(logger._journey["journey_id"])
            with self.assertRaises(ValueError):
                store.supplement(logger._journey["journey_id"], {"final_stats": {"力量": 300}, "equipped_skills": ["IX1"]})
            self.assertEqual(store.read(logger._journey["journey_id"]), original)

    def test_rule_snapshot_changes_when_profile_changes_even_with_same_config(self):
        logger = RunLogger()
        profile = TrainingRuleLoader.create_builtin_default()
        logger.configure_run("config", {}, profile)
        first = logger._strategy["strategy_hash"]
        profile.legacy_strategy.fail_rate_threshold += 1
        logger.configure_run("config", {}, profile)
        self.assertNotEqual(logger._strategy["strategy_hash"], first)
        self.assertEqual(len(logger._strategy_history), 2)

    def test_real_runtime_records_input_before_send_and_restores_journey_identity(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root, lambda ctx: ctx.controller.click_at_percent(.8, .5) or True)
            trainer.logger = RunLogger(root=root)
            trainer._runtime_init(root)
            trainer.ui.ocr.recognize_detailed.return_value = [("力量 39", .99, None)]
            trainer._runtime_active = True
            trainer.controller.before_action, trainer.controller.after_action = trainer._before_input, trainer._after_input
            self.assertTrue(trainer._do_round())
            data = json.loads(next((Path(root) / 'logs').glob('run_*.json')).read_text(encoding='utf-8'))
            self.assertEqual([item["phase"] for item in data["inputs"]], ["prepared", "sent"])
            self.assertEqual(trainer.runtime.read_checkpoint()["journey_id"], trainer._journey_id)
            resumed = make_trainer(root)
            resumed._load_resume_checkpoint()
            self.assertEqual(resumed._journey_id, trainer._journey_id)

    def test_record_write_failure_prevents_input(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._runtime_active = True
            trainer.controller.before_action = trainer._before_input
            trainer.logger.save.side_effect = OSError("disk write failed")
            with self.assertRaises(RuntimeInterrupt):
                trainer.controller.click_at_percent(.8, .5)
            self.assertEqual(trainer.controller.sent, [])

    def test_logging_failure_still_exposes_safe_pause(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer.logger.save.side_effect = OSError("disk write failed")
            trainer._enter_safe_pause("checkpoint_error")
            status = trainer.runtime.read_status()
            self.assertEqual(status["state"], "paused_safe")
            self.assertEqual(status["diagnostics"]["recording_error"], "disk write failed")
            self.assertEqual(trainer.controller.sent, [])

    def test_final_handler_saves_frame_before_manual_handoff_without_input(self):
        handler = JourneyEndHandler({'journey_end': {'pause_for_manual': True}})
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=Mock(), logger=Mock(), round_count=45)
        screenshot = np.zeros((64, 96, 3), np.uint8)
        ctx.capture.capture_game.return_value = screenshot
        ctx.ocr.recognize_region.return_value = "旅程结束"
        ctx.ocr.recognize_detailed.return_value = [("力量 600", .99, None)]
        with self.assertRaises(JourneyEndException):
            handler.handle(ctx)
        ctx.logger.observe_frame.assert_called_once_with([("力量 600", .99, None)], '旅程结束', 45, error=None, final=True)
        ctx.controller.click_at_percent.assert_not_called()

    def test_supplement_can_complete_setup_but_keeps_mid_journey_warning(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.set_target(target(actual_setup={}))
            logger.save("journey_end", announce=False)
            store = JourneyRecordStore(root)
            body = {"actual_setup": {"character": "格温", "journey_records": [], "support_cards": ["支援甲"]},
                    "final_stats": {name: 0 for name in ("力量", "体力", "韧性", "专注", "保护")},
                    "equipped_skills": ["AX5"], "appraisal_result": "victory"}
            result = store.supplement(logger._journey["journey_id"], body)
            self.assertEqual(result["data_quality"]["missing_fields"], [])
            self.assertTrue(result["outcome"]["target_skill_achieved"]["value"])
            self.assertFalse(result["data_quality"]["ready_for_comparison"])
            self.assertEqual(result["actual_setup"]["support_cards"]["source"], "user_declared")

    def test_damaged_summary_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            store = JourneyRecordStore(root)
            path = store.path(logger._journey["journey_id"])
            path.parent.mkdir(parents=True)
            path.write_text('damaged record', encoding='utf-8')
            with self.assertRaises(ValueError):
                logger.save("running", announce=False)
            self.assertEqual(path.read_text(encoding='utf-8'), 'damaged record')

    def test_api_blocks_results_on_live_or_unfinished_journey_and_validates_path(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.save("safe_paused", announce=False)
            journey_id = logger._journey["journey_id"]
            control = RuntimeControl(root)
            with patch.object(web_ui, '_runtime_control', control), patch.object(web_ui, '_automation_process', None):
                with self.assertRaises(HTTPException):
                    web_ui.supplement_journey_record(journey_id, {"final_stats": {"力量": 100}})
                logger.save("journey_end", announce=False)
                control.write_status("paused_safe", run_id='active', journey_id=journey_id)
                with patch.object(web_ui, 'process_alive', return_value=True):
                    with self.assertRaises(HTTPException) as raised:
                        web_ui.supplement_journey_record(journey_id, {"final_stats": {"力量": 100}})
                    self.assertEqual(raised.exception.status_code, 409)
                control.write_status("journey_end", journey_id=journey_id)
                result = web_ui.supplement_journey_record(journey_id, {"equipped_skills": ["AX5"]})
                self.assertIsNone(result["outcome"]["target_skill_achieved"]["value"])  # No saved target.
                self.assertEqual(len(web_ui.journey_records()["records"]), 1)
                with self.assertRaises(HTTPException):
                    web_ui.supplement_journey_record('../status', {})


if __name__ == '__main__':
    unittest.main()
