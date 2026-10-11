"""Multi-frame decision provenance and replay; no desktop input."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import web_ui
from src.logger import RunLogger
from src.rule_engine import BuildDirection, TrainingDecisionContext, TrainingRuleEngine, TrainingRuleLoader
from src.trainer import Trainer, TrainingMetricScanState


class DecisionEvidenceTests(unittest.TestCase):
    def make_logger(self, root):
        logger = RunLogger(root=root)
        logger.configure_run("cfg", {"commission": {"rank_threshold": 4}},
                             TrainingRuleLoader.create_builtin_default())
        logger.start_turn(2, None, 80, "Good", 30)
        return logger

    def test_distinct_frames_values_and_hashes_persist_before_selection(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            logger.begin_decision("commission", ["高阶", "中阶", "低阶"])
            values = {"my_rank": 20, "recommended": 24}
            for index, tier in enumerate(("高阶", "中阶", "低阶")):
                logger.record_decision_frame(tier, tier, np.full((40, 60, 3), 40+index, np.uint8), values)
            values["my_rank"] = 99
            saved = json.loads(next((Path(root)/"logs").glob("run_*.json")).read_text(encoding="utf-8"))
            frames = saved["decision_evidence"][0]["frames"]
            self.assertEqual(len({frame["screenshot"] for frame in frames}), 3)
            self.assertEqual(len({frame["sha256"] for frame in frames}), 3)
            for index, frame in enumerate(frames):
                self.assertEqual(frame["values"]["my_rank"], 20)
                self.assertEqual(int(cv2.imread(str(Path(root)/frame["screenshot"]))[0, 0, 0]), 40+index)
            self.assertFalse(saved["turns"][0]["actions"])

    def test_partial_evidence_blocks_selection_but_remains_saved(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            logger.begin_decision("commission", ["高阶", "中阶", "低阶"])
            logger.record_decision_frame("高阶", "高阶", np.zeros((40, 60, 3), np.uint8), {})
            with self.assertRaisesRegex(RuntimeError, "候选证据不完整"):
                logger.log_commission("高阶", 20, 23)
            self.assertFalse(logger._current_turn["actions"])
            self.assertEqual(logger.decision_evidence()["missing_candidates"], ["中阶", "低阶"])

    def test_starting_training_keeps_previous_commission_history(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            logger.log_commission("高阶", 20, 24)
            commission_id = logger._current_turn["actions"][0]["action_id"]
            logger.start_turn(3, None, 70, "Good", 50)
            data = json.loads(Path(logger.save("running", announce=False, close_turn=False)).read_text(encoding="utf-8"))
            self.assertEqual(data["turns"][0]["actions"][0]["action_id"], commission_id)
            self.assertEqual(len(data["turns"]), 2)

    def test_failed_image_write_cannot_produce_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            logger.begin_decision("training", ["0"])
            with patch("cv2.imwrite", return_value=False), self.assertRaisesRegex(RuntimeError, "保存失败"):
                logger.record_decision_frame("0", "力量", np.zeros((40, 60, 3), np.uint8), {})
            self.assertFalse(logger.decision_evidence()["frames"])

    def test_recorded_training_context_replays_same_rule_and_target(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            profile = TrainingRuleLoader.create_builtin_default()
            logger.begin_decision("training", [str(i) for i in range(5)])
            scan = TrainingMetricScanState()
            for i in range(5):
                scan.set_row_metrics(i, i, 10, 10+i*5, is_flash=i == 1)
                logger.record_decision_frame(str(i), str(i), np.full((40, 60, 3), i, np.uint8),
                                             {"gain": scan.gain_values[i], "fail_rate": 10, "icons": i})
            context = scan.to_decision_context(profile, "builtin")
            result = TrainingRuleEngine.evaluate(context, profile)
            logger.log_decision_rule_engine(result, context, scan)
            action = logger._current_turn["actions"][-1]
            replay_values = dict(action["decision_context"])
            replay_values["build_direction"] = BuildDirection(replay_values["build_direction"])
            replay = TrainingRuleEngine.evaluate(TrainingDecisionContext(**replay_values), profile)
            self.assertEqual((replay.action.value, replay.target_row_index, replay.matched_rule_id),
                             (action["action"], action["target_row"], action["matched_rule"]))
            self.assertTrue(action["evidence"]["complete"])
            logger.log_train_exec(result.target_row_index, "test")
            logger.log_input({"phase": "prepared"})
            self.assertEqual(logger._current_turn["actions"][-1]["decision_id"], action["action_id"])
            self.assertEqual(logger._inputs[-1]["decision_id"], action["action_id"])
            self.assertEqual(len(logger._current_turn["actions"][-1]["evidence"]["frames"]), 5)

    def test_old_decision_snapshot_does_not_change_when_new_frames_are_added(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            logger.begin_decision("commission", ["高阶"])
            frame = np.zeros((40, 60, 3), np.uint8)
            logger.record_decision_frame("高阶", "高阶", frame, {"recommended": 23})
            logger.log_commission("高阶", 20, 23, mode="battle")
            action = logger._current_turn["actions"][-1]
            logger.record_decision_frame(None, "确认", frame, {}, role="confirmation")
            self.assertEqual(len(action["evidence"]["frames"]), 1)
            logger.clear_decision()
            self.assertIsNone(logger.decision_evidence())

    def test_ui_exposes_commissions_frames_and_missing_images(self):
        with tempfile.TemporaryDirectory() as root:
            logger = self.make_logger(root)
            logger.begin_decision("commission", ["高阶"])
            logger.record_decision_frame("高阶", "高阶", np.zeros((40, 60, 3), np.uint8), {})
            logger.log_commission("高阶", 20, 23, mode="battle")
            path = Path(logger.save("running", announce=False, close_turn=False))
            payload = json.loads(path.read_text(encoding="utf-8"))
            store = web_ui.DataStore()
            with patch.object(store, "logs", return_value=[(path, payload)]), patch.object(web_ui, "ROOT", Path(root)):
                row = store.action_rows()[0]
                self.assertEqual(row["type"], "commission")
                self.assertTrue(row["evidence"]["complete"])
                self.assertIn("rule_profile", row["strategy_snapshot"])
                self.assertTrue(row["evidence"]["frames"][0]["screenshot_url"])
                (Path(root)/row["evidence"]["frames"][0]["screenshot"]).unlink()
                self.assertFalse(store.action_rows()[0]["evidence"]["complete"])

    def test_history_orders_actual_timestamps_across_restart_segments(self):
        store = web_ui.DataStore()
        entries = [(Path("new.json"), {"run_id": "new", "actions": [
            {"type": "commission", "ts": "2026-10-11T01:30:00.123456", "tier": "高阶"}]}),
                   (Path("old.json"), {"run_id": "old", "actions": [
            {"type": "decision", "ts": "2026-10-11T01:20:00.123456"}]})]
        with patch.object(store, "logs", return_value=entries):
            rows = store.action_rows(1)
        self.assertEqual(rows[0]["run_id"], "new")
        self.assertEqual(rows[0]["time"], "01:30:00")

    def test_reading_row_keeps_flash_and_records_real_selected_frame(self):
        trainer = Trainer.__new__(Trainer)
        trainer.controller, trainer.capture, trainer.logger = Mock(), Mock(), Mock()
        frame = np.zeros((900, 1600, 3), np.uint8)
        trainer.capture.capture_game.return_value = frame
        trainer.engine = SimpleNamespace(use_icon_counting=False, use_per_row_fail_rate=False)
        trainer.ui = SimpleNamespace(ocr=Mock())
        trainer.ui.ocr.recognize_region.return_value = "+23"
        trainer._last_stamina = 70
        scan = TrainingMetricScanState()
        scan.is_flash[0] = True
        trainer._read_row_metrics(scan, 0)
        self.assertTrue(scan.is_flash[0])
        entry = trainer.logger.record_decision_frame.call_args
        self.assertIs(entry.args[2], frame)
        self.assertEqual(entry.args[3]["sources"]["fail_rate"], "stamina_fallback")
        self.assertEqual(entry.args[3]["gain"], 23)

    def test_final_camp_override_is_logged_instead_of_provisional_choice(self):
        from src.decision import GameState
        from src.rule_engine import TrainingDecisionAction, TrainingDecisionResult
        trainer = Trainer.__new__(Trainer)
        trainer.config = {"camp": {"pre_camp_min_gain": 35}}
        trainer.logger = Mock()
        trainer.engine = SimpleNamespace(rule_profile=TrainingRuleLoader.create_builtin_default(),
                                         rule_profile_name="builtin", adventure_gain_threshold=100)
        before = TrainingDecisionResult(action=TrainingDecisionAction.TRAIN_STRENGTH, target_row_index=0)
        after = TrainingDecisionResult(action=TrainingDecisionAction.REST, matched_rule_id="pre_camp_rest")
        trainer._apply_camp_strategy = Mock(return_value=("rest", after))
        trainer._finish_training_decision(("train", before), TrainingMetricScanState(), 26, GameState())
        call = trainer.logger.log_decision_rule_engine.call_args
        self.assertIs(call.args[0], after)
        self.assertEqual(call.kwargs["base_result"], before)
        self.assertEqual(call.kwargs["overrides"]["final_action"], "rest")


if __name__ == "__main__":
    unittest.main()
