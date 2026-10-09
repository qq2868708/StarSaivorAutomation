"""Fault and restart scenarios with fake game frames; no desktop input is sent."""
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.handlers import EventDecisionPending, HandlerChain, JourneyEndException
from src.runtime_control import RuntimeInterrupt
from src.trainer import Trainer


class FakeCapture:
    def __init__(self, value=40):
        self.frame = np.full((64, 96, 3), value, np.uint8)
        self.before_capture = self.after_capture = None

    def find_window(self):
        return (0, 0, 96, 64)

    def get_hwnd(self):
        return 1

    def capture_game(self):
        if self.before_capture:
            self.before_capture()
        if self.after_capture:
            self.after_capture(self.frame)
        return self.frame


class FakeController:
    def __init__(self):
        self.before_action = self.after_action = self.poll_control = None
        self.sent = []
        self.on_sent = None

    def set_game_region(self, region):
        pass

    def set_window_handle(self, hwnd):
        pass

    def focus_game_window(self):
        pass

    def click_at_percent(self, x, y, source="script"):
        action = {"kind": "click", "x": x, "y": y, "source": source}
        if self.before_action:
            self.before_action(action)
        self.sent.append((x, y))
        if self.after_action:
            self.after_action(action)
        if self.on_sent:
            self.on_sent()

    def wait(self, seconds):
        if self.poll_control:
            self.poll_control()


class FakeHandler:
    priority = 1
    name = "fake-page"

    def __init__(self, action):
        self.action = action

    def can_handle(self, frame, ocr):
        return True

    def handle(self, ctx):
        return self.action(ctx)


def make_trainer(root, action=None):
    trainer = Trainer.__new__(Trainer)
    trainer.config = {}
    trainer.logger = Mock()
    trainer.logger._run_id = "test_run"
    trainer.logger._current_turn = None
    trainer.capture = FakeCapture()
    trainer.controller = FakeController()
    trainer.ui = SimpleNamespace(ocr=Mock())
    trainer.ui.ocr.recognize.return_value = [("测试界面", 1.0)]
    trainer.engine = SimpleNamespace(is_rule_engine_mode=False)
    trainer._handler_ctx = SimpleNamespace(detected_turn=21, turn_detected=True,
        consecutive_unknown=0, pending_screenshot=None, pending_action="",
        controller=trainer.controller, round_count=0)
    trainer._handlers = HandlerChain([FakeHandler(action or (lambda ctx: True))])
    trainer.round_count = 0
    trainer.max_rounds = 1
    trainer._max_runtime_seconds = 60
    trainer._round_timeout_seconds = 5
    trainer._stuck_threshold = 5
    trainer._development_mode = False
    trainer._consecutive_unknown = 0
    trainer._exit_reason = "unknown"
    trainer._print_config = Mock()
    trainer._runtime_init(root)
    return trainer


class RuntimeSessionTests(unittest.TestCase):
    def test_click_is_checkpointed_before_input_and_after_input(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._runtime_active = True
            trainer._last_runtime_frame = trainer.capture.frame
            action = {"kind": "click", "x": .8, "y": .5}
            trainer._before_input(action)
            self.assertEqual(trainer.runtime.read_checkpoint()["last_input"]["phase"], "prepared")
            trainer._after_input(action)
            self.assertEqual(trainer.runtime.read_checkpoint()["last_input"]["phase"], "sent")

    def test_pause_unwinds_before_a_second_input(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._runtime_active = True
            trainer.runtime.issue("pause", run_id=trainer._run_id)
            with self.assertRaises(RuntimeInterrupt):
                trainer._before_input({"kind": "click"})
            self.assertTrue(trainer._paused)
            self.assertEqual(trainer.controller.sent, [])

    def test_screenshot_persistence_failure_cannot_fall_through_to_input(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._runtime_active = True
            trainer._last_runtime_frame = trainer.capture.frame
            trainer.controller.before_action = trainer._before_input
            with patch.object(trainer, "_write_runtime_diagnostic_screenshot",
                              side_effect=OSError("disk full")):
                with self.assertRaises(RuntimeInterrupt) as caught:
                    trainer.controller.click_at_percent(.8, .5)
            self.assertEqual(caught.exception.reason, "checkpoint_error")
            self.assertEqual(trainer.controller.sent, [])

    def test_resume_discards_pending_frame_and_rebinds_current_page(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._handler_ctx.pending_screenshot = np.zeros((4, 4, 3), np.uint8)
            trainer._handler_ctx.pending_action = "training"
            trainer._paused = True
            trainer._set_running_status()
            self.assertIsNone(trainer._handler_ctx.pending_screenshot)
            trainer._recover_current_screen()
            self.assertIs(trainer._handler_ctx.pending_screenshot, trainer.capture.frame)
            self.assertIsNone(trainer._handler_ctx.detected_turn)
            self.assertEqual(trainer.controller.sent, [])

    def test_restart_after_uncertain_click_never_replays_last_click(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer.runtime.write_checkpoint(phase="before_input", resumable=True,
                round_count=17, detected_turn=33, last_input={"kind": "click", "x": .9, "y": .9})
            trainer._load_resume_checkpoint()
            trainer._recover_current_screen()
            self.assertEqual(trainer.controller.sent, [])
            self.assertIsNone(trainer._handler_ctx.detected_turn)

    def test_new_instance_can_start_on_each_current_page_without_checkpoint(self):
        for page in ("training", "rest", "event", "appraisal", "shop", "result", "story"):
            with self.subTest(page=page), tempfile.TemporaryDirectory() as root:
                trainer = make_trainer(root)
                trainer._handlers.handlers[0].name = page
                trainer._recover_current_screen()
                self.assertTrue(trainer._do_round())
                self.assertEqual(trainer._active_handler_name, page)

    def test_stale_run_command_cannot_pause_new_instance(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._runtime_active = True
            command = trainer.runtime.issue("pause", run_id="old-run")
            trainer._runtime_poll()
            self.assertFalse(trainer._paused)
            ack = trainer.runtime._read(trainer.runtime.directory / "acks" / f"{command['id']}.json")
            self.assertFalse(ack["accepted"])

    def test_handler_exception_is_not_swallowed_by_dispatch(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root, lambda ctx: (_ for _ in ()).throw(ValueError("test")))
            with self.assertRaises(ValueError):
                trainer._do_round()

    def test_running_and_manual_pause_reject_ai_even_with_a_request(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            for paused in (False, True):
                trainer._paused = paused
                command = trainer.runtime.issue("ai_action", run_id=trainer._run_id,
                                                 action_id="one", kind="click", x=.8, y=.5)
                trainer._handle_ai_action(command)
                ack = trainer.runtime._read(trainer.runtime.directory / "acks" / f"{command['id']}.json")
                self.assertFalse(ack["accepted"])
            self.assertEqual(trainer.controller.sent, [])

    def test_safe_pause_requires_ai_registration_and_stale_screen_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._enter_safe_pause("test")
            command = trainer.runtime.issue("ai_action", run_id=trainer._run_id,
                pause_id=trainer._pause_id, kind="click", action_id="one", x=.8, y=.5)
            trainer._handle_ai_action(command)
            self.assertEqual(trainer.controller.sent, [])
            lease = trainer.runtime.claim_ai("test-client", trainer._run_id, trainer._pause_id)
            command = trainer.runtime.issue("ai_action", **lease, kind="click", action_id="two",
                expected_frame_id="outdated", x=.8, y=.5)
            trainer._handle_ai_action(command)
            self.assertEqual(trainer.controller.sent, [])

    def test_ai_click_changes_keep_script_paused_and_duplicates_do_not_click(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._enter_safe_pause("test")
            lease = trainer.runtime.claim_ai("test-client", trainer._run_id, trainer._pause_id)
            payload = {**lease, "kind": "click", "action_id": "once", "x": .8, "y": .5}
            trainer._handle_ai_action(trainer.runtime.issue("ai_action", **payload,
                                                          expected_frame_id=trainer._frame_id))
            trainer._handle_ai_action(trainer.runtime.issue("ai_action", **payload,
                                                          expected_frame_id=trainer._frame_id))
            self.assertEqual(trainer.controller.sent, [(.8, .5)])
            self.assertTrue(trainer._safe_paused)

    def test_old_ai_action_remains_rejected_after_many_actions_and_restart(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._ai_action_ids = [f"action-{index}" for index in range(150)]
            trainer._write_checkpoint("safe_paused")
            recovered = make_trainer(root)
            recovered._load_resume_checkpoint()
            recovered._enter_safe_pause("test")
            lease = recovered.runtime.claim_ai("test-client", recovered._run_id, recovered._pause_id)
            command = recovered.runtime.issue("ai_action", **lease, kind="click", action_id="action-0",
                expected_frame_id=recovered._frame_id, x=.8, y=.5)
            recovered._handle_ai_action(command)
            self.assertEqual(recovered.controller.sent, [])
            ack = recovered.runtime._read(recovered.runtime.directory / "acks" / f"{command['id']}.json")
            self.assertFalse(ack["accepted"])

    def test_ai_changed_game_screen_does_not_click(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._enter_safe_pause("test")
            lease = trainer.runtime.claim_ai("test-client", trainer._run_id, trainer._pause_id)
            trainer.capture.frame = np.full((64, 96, 3), 200, np.uint8)
            command = trainer.runtime.issue("ai_action", **lease, kind="click", action_id="one",
                expected_frame_id=trainer._frame_id, x=.8, y=.5)
            trainer._handle_ai_action(command)
            self.assertEqual(trainer.controller.sent, [])

    def test_script_without_ai_keeps_safe_pause_until_manual_stop(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._enter_safe_pause("test")
            worker = threading.Thread(target=trainer._wait_for_runtime, daemon=True)
            worker.start()
            try:
                time.sleep(.15)
                self.assertTrue(worker.is_alive())
                self.assertEqual(trainer.controller.sent, [])
            finally:
                trainer.runtime.issue("stop", run_id=trainer._run_id)
                worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertTrue(trainer._stop_requested)

    def test_watchdog_works_inside_handler_before_input(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer._runtime_active = True
            trainer._round_deadline = time.monotonic() - 1
            with self.assertRaises(RuntimeInterrupt):
                trainer._before_input({"kind": "click"})
            self.assertEqual(trainer.controller.sent, [])

    def test_full_loop_resume_reenters_dispatch_instead_of_old_click_sequence(self):
        with tempfile.TemporaryDirectory() as root:
            def old_flow(ctx):
                ctx.controller.click_at_percent(.1, .1)
                ctx.controller.click_at_percent(.9, .9)  # must never execute after pause
                return True

            trainer = make_trainer(root, old_flow)
            trainer.max_rounds = 0
            old = trainer._handlers.handlers[0]
            old.can_handle = lambda frame, ocr: int(frame.screenshot[0, 0, 0]) == 40

            def completed(ctx):
                raise JourneyEndException("旅程结束")

            end = FakeHandler(completed)
            end.name = "final-result"
            trainer._handlers = HandlerChain([old, end])

            def pause_then_continue():
                trainer.capture.frame = np.full((64, 96, 3), 80, np.uint8)
                trainer.runtime.issue("pause", run_id=trainer._run_id)
                trainer.runtime.issue("resume", run_id=trainer._run_id)

            trainer.controller.on_sent = pause_then_continue
            with patch("src.trainer.preload_profiles"):
                trainer.run(auto_start=True)
            self.assertEqual(trainer.controller.sent, [(.1, .1)])
            self.assertEqual(trainer.runtime.read_status()["state"], "journey_end")
            self.assertFalse(trainer.runtime.read_checkpoint()["resumable"])

    def test_missing_game_window_safely_pauses_instead_of_exiting_startup(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = make_trainer(root)
            trainer.capture.find_window = Mock(return_value=None)
            pause = trainer._enter_safe_pause
            captured = []

            def observe_pause(*args, **kwargs):
                pause(*args, **kwargs)
                captured.append(trainer.runtime.read_status()["state"])
                trainer.runtime.issue("stop", run_id=trainer._run_id)

            trainer._enter_safe_pause = observe_pause
            with patch("src.trainer.preload_profiles"):
                trainer.run(auto_start=True)
            self.assertEqual(captured, ["paused_safe"])
            self.assertEqual(trainer.controller.sent, [])


if __name__ == "__main__":
    unittest.main()
