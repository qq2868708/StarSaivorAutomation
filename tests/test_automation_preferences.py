"""User-configured sounds, purchases and confirmed decision turns; no game input."""
import copy
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml
from fastapi import HTTPException

import web_ui
from src.automation_preferences import DEFAULTS, JourneyProgress, preferences, purchase_mode, turn_instruction, validate_preferences
from src.handlers import ShopHandler
from src.notifications import SoundNotifier, tone_wav
from src.trainer import Trainer, TrainingMetricScanState
from src.rule_engine import TrainingDecisionAction, TrainingDecisionResult
from test_runtime_session import make_trainer


class PreferencesTests(unittest.TestCase):
    def test_defaults_only_notify_completion_and_do_not_override_choices(self):
        settings = preferences({})
        self.assertEqual(settings["notifications"]["events"],
                         {"journey_end": True, "paused_safe": False, "recovered": False})
        self.assertFalse(settings["turn_plan"]["enabled"])
        self.assertEqual(settings["purchase_rules"]["default"], "auto")

    def test_validation_rejects_invalid_and_duplicate_values(self):
        for section, key, value in [
            ("notifications", "volume", 101), ("notifications", "enabled", 1),
            ("turn_plan", "start_turn", 0), ("turn_plan", "turns", {"46": "train"}),
            ("turn_plan", "turns", {"1": "invalid"}),
            ("purchase_rules", "categories", {"unknown": "buy"}),
            ("purchase_rules", "items", {"a b": "buy", "ab": "skip"}),
        ]:
            with self.subTest(key=key):
                settings = copy.deepcopy(DEFAULTS)
                settings[section][key] = value
                with self.assertRaises(ValueError):
                    validate_preferences(settings)

    def test_shop_specific_rules_override_category_and_deny_wins_alias_conflict(self):
        rules = {"default": "auto", "categories": {"stamina_recovery": "skip"},
                 "items": {"茶": "buy", "坏茶": "skip"}}
        self.assertEqual(purchase_mode(rules, {"type": "stamina_recovery", "display_name": "茶"})[0], "buy")
        self.assertEqual(purchase_mode(rules, {"type": "stamina_recovery"})[0], "skip")
        self.assertEqual(purchase_mode(rules, {"profile_name": "茶", "display_name": "坏茶"})[0], "skip")

    def test_shop_rank_never_buys_denied_items_and_can_buy_explicit_override(self):
        handler = ShopHandler({"shop": {"purchase_rules": {
            "default": "skip", "items": {"茶": "buy"}, "categories": {}}}})
        items = [(0, {"type": "stamina_recovery", "display_name": "茶", "value": 1}, ""),
                 (1, {"type": "potential_points", "display_name": "书", "value": 100}, "")]
        result = handler._rank_items(items, 100, "Best", True)
        self.assertEqual([item[0] for item in result], [0])
        self.assertEqual(items[0][1]["purchase_policy"]["mode"], "buy")
        self.assertEqual(items[1][1]["purchase_policy"]["mode"], "skip")

    def test_explicit_turn_including_rules_overrides_stage(self):
        plan = {"enabled": True, "after_void_action": "train", "default_action": "rules",
                "turns": {"40": "rules", "41": "rest"}}
        self.assertEqual(turn_instruction(plan, 39, True)["action"], "train")
        self.assertEqual(turn_instruction(plan, 40, True)["action"], "rules")
        self.assertEqual(turn_instruction(plan, 41, True)["action"], "rest")
        self.assertEqual(turn_instruction(plan, None, True)["action"], "train")
        self.assertEqual(turn_instruction(plan, None, False)["action"], "rules")

    def test_turn_override_is_applied_after_camp_and_logged(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = {"turn_plan": {"enabled": True, "turns": {"40": "train_strength"}}}
        trainer._journey_progress = JourneyProgress(40)
        trainer._apply_camp_strategy = Mock(return_value=("rest", TrainingDecisionResult()))
        trainer.logger = Mock()
        trainer.engine = SimpleNamespace(rule_profile=Mock(), rule_profile_name="test", adventure_gain_threshold=0)
        scan = Mock()
        state = SimpleNamespace(mood="Best")
        result = trainer._finish_training_decision(("rest", TrainingDecisionResult()), scan, 40, state)
        self.assertEqual(result[0], "train")
        self.assertEqual(result[1].target_row_index, 0)
        self.assertEqual(result[1].action, TrainingDecisionAction.TRAIN_STRENGTH)
        self.assertEqual(trainer.logger.log_turn_policy.call_args.args[0]["source"], "turn")


class ProgressTests(unittest.TestCase):
    def test_dispatch_and_rest_entry_do_not_advance_turn(self):
        progress = JourneyProgress(20)
        for _ in range(30):
            progress.observe([], "主菜单")
        progress.prepare("rest")
        progress.observe([], "休息决策")
        self.assertEqual(progress.data["turn"], 20)
        progress.observe([("露宿 SUCCESS 点击以继续", .99)], "奖励继续", "evidence")
        self.assertEqual(progress.data["turn"], 21)
        progress.observe([("露宿 SUCCESS 点击以继续", .99)], "奖励继续")
        self.assertEqual(progress.data["turn"], 21)

    def test_restarts_keep_pending_turn_and_completion_counts_once(self):
        original = JourneyProgress(39)
        original.prepare("train")
        resumed = JourneyProgress(saved=original.data)
        resumed.observe([("训练成功", .99)], "训练结果", "proof")
        self.assertEqual(resumed.data["turn"], 40)
        self.assertEqual(resumed.data["last_evidence"], "proof")
        resumed.observe([("39/45", .99)], "未知兜底")
        self.assertEqual(resumed.data["turn"], 40)

    def test_ocr_calibrates_and_unknown_start_is_never_guessed(self):
        progress = JourneyProgress()
        progress.prepare("train")
        progress.observe([("训练成功", .99)], "训练结果")
        self.assertIsNone(progress.data["turn"])
        progress.observe([("23 / 45", .99)], "训练决策")
        self.assertEqual(progress.data["turn"], 23)
        progress.prepare("train")
        progress.observe([("24/45", .99)], "训练决策")
        self.assertEqual(progress.data["turn"], 24)
        self.assertIsNone(progress.data["pending"])

    def test_void_goal_list_alone_is_not_completed_battle(self):
        progress = JourneyProgress()
        progress.observe([("在虚空评鉴战中获胜", .99)], "目标提示")
        self.assertFalse(progress.data["after_void"])
        progress.observe([("虚空评鉴战", .99)], "评鉴战")
        progress.observe([("BURST 1/1", .99)], "实际战斗")
        self.assertFalse(progress.data["after_void"])
        progress.observe([("VICTORY", .99)], "战后离开", "result")
        self.assertTrue(progress.data["after_void"])
        self.assertTrue(JourneyProgress(saved=progress.data).data["after_void"])

    def test_runtime_checkpoint_restores_decision_progress(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._journey_progress = JourneyProgress(40)
            trainer._journey_progress.data["after_void"] = True
            trainer._prepare_turn_action("train")
            other = make_trainer(root)
            other._load_resume_checkpoint()
            self.assertEqual(other._journey_progress.data, trainer._journey_progress.data)

    def test_configured_anchor_fills_unknown_but_does_not_reset_verified_turn(self):
        unknown = JourneyProgress()
        self.assertEqual(JourneyProgress(12, unknown.data).data["turn"], 12)
        known = JourneyProgress(32)
        self.assertEqual(JourneyProgress(12, known.data).data["turn"], 32)

    def test_commission_only_counts_when_calendar_actually_advances(self):
        progress = JourneyProgress(20)
        progress.observe([("5月中旬", .99)], "主菜单")
        progress.prepare("commission")
        progress.observe([("VICTORY", .99)], "战后离开")
        self.assertEqual(progress.data["turn"], 20)
        progress.observe([("5月下旬", .99)], "主菜单")
        self.assertEqual(progress.data["turn"], 21)
        progress.observe([("5月下旬", .99)], "主菜单")
        self.assertEqual(progress.data["turn"], 21)

    def test_aborted_rest_is_replaced_by_training_without_double_count(self):
        progress = JourneyProgress(20)
        progress.prepare("rest")
        progress.prepare("train")
        progress.observe([("训练成功", .99)], "训练结果")
        self.assertEqual(progress.data["turn"], 21)


class NotificationTests(unittest.TestCase):
    def test_wave_volume_and_format(self):
        for volume in (0, 50, 100):
            with wave.open(io.BytesIO(tone_wav("journey_end", volume)), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertGreater(wav.getnframes(), 0)
                if volume == 0:
                    self.assertEqual(set(wav.readframes(wav.getnframes())), {0})

    def test_only_once_on_completion_not_on_poll_or_start(self):
        play = Mock()
        notifier = SoundNotifier(DEFAULTS["notifications"], player=play)
        for state in ("starting", "running", "paused_safe", "running", "journey_end", "journey_end"):
            notifier.state_changed(state)
        notifier.close()
        play.assert_called_once()

    def test_mute_and_playback_failure_do_not_break_runtime(self):
        notifier = SoundNotifier({**DEFAULTS["notifications"], "enabled": False}, player=Mock())
        self.assertFalse(notifier.state_changed("journey_end"))
        notifier.player.assert_not_called()
        notifier = SoundNotifier(DEFAULTS["notifications"], player=Mock(side_effect=RuntimeError("no audio")))
        notifier.state_changed("journey_end")
        notifier.close()
        self.assertEqual(notifier.last_error, "no audio")

    def test_completion_is_not_dropped_while_another_tone_is_playing(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        calls = []
        def play(data):
            calls.append(data)
            entered.set()
            release.wait(1)
        settings = copy.deepcopy(DEFAULTS["notifications"])
        settings["events"]["paused_safe"] = True
        notifier = SoundNotifier(settings, player=play)
        notifier.state_changed("paused_safe")
        self.assertTrue(entered.wait(1))
        notifier.state_changed("journey_end")
        release.set()
        notifier.close()
        self.assertEqual(len(calls), 2)


class PreferencesApiTests(unittest.TestCase):
    def test_save_roundtrip_is_atomic_and_preserves_other_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("game:\n  window_title: StarSavior\nshop:\n  enabled: false\n", encoding="utf-8")
            with patch.object(web_ui, "CONFIG_PATH", path):
                with patch.object(web_ui, "store", web_ui.DataStore()):
                    prefs = preferences({})
                    prefs["turn_plan"].update(enabled=True, after_void_action="train", turns={"40": "rest"})
                    response = web_ui.update_config({"preferences": prefs})
                    self.assertEqual(response["preferences"], prefs)
                    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
                    self.assertFalse(saved["shop"]["enabled"])
                    self.assertEqual(saved["game"]["window_title"], "StarSavior")
                    before = path.read_bytes()
                    prefs["notifications"]["volume"] = -1
                    with self.assertRaises(HTTPException):
                        web_ui.update_config({"preferences": prefs})
                    self.assertEqual(path.read_bytes(), before)

    def test_preview_is_audio_and_rejects_invalid_input(self):
        result = web_ui.notification_preview("journey_end", 50)
        self.assertEqual(result.media_type, "audio/wav")
        with self.assertRaises(HTTPException):
            web_ui.notification_preview("unknown", 50)


if __name__ == "__main__":
    unittest.main()
