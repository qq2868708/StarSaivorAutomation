"""Local control API contracts; all mutations target a temporary runtime directory."""
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import web_ui
from src.runtime_control import RuntimeControl


class RuntimeAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = RuntimeControl(self.temp.name)
        self.patches = [patch.object(web_ui, "_runtime_control", self.control),
                        patch.object(web_ui, "_automation_process", None)]
        for value in self.patches:
            value.start()

    def tearDown(self):
        for value in self.patches:
            value.stop()
        self.temp.cleanup()

    def pause(self):
        self.control.write_status("paused_safe", run_id="run", pause_id="pause", frame_id="frame")
        return web_ui.runtime_ai_claim({"client_id": "test-ai", "run_id": "run", "pause_id": "pause"})

    def test_normal_run_cannot_accept_ai_commands(self):
        self.control.write_status("running", run_id="run")
        with self.assertRaises(HTTPException) as error:
            web_ui.runtime_ai_action({"kind": "click", "x": .5, "y": .5})
        self.assertEqual(error.exception.status_code, 409)
        self.assertIsNone(self.control.consume())

    def test_click_requires_lease_frame_and_unique_action(self):
        lease = self.pause()
        payload = {**lease, "kind": "click", "action_id": "once", "x": .8, "y": .5}
        with self.assertRaises(HTTPException):
            web_ui.runtime_ai_action(payload)
        result = web_ui.runtime_ai_action({**payload, "expected_frame_id": "frame"})
        self.assertTrue(result["queued"])
        command = self.control.consume()
        self.assertEqual(command["command"], "ai_action")
        self.assertEqual(command["action_id"], "once")

    def test_nan_and_out_of_bounds_coordinates_are_rejected(self):
        lease = self.pause()
        for x in (float("nan"), float("inf"), -1, 2):
            with self.subTest(x=x), self.assertRaises(HTTPException):
                web_ui.runtime_ai_action({**lease, "kind": "click", "action_id": "test",
                    "expected_frame_id": "frame", "x": x, "y": .5})

    def test_resume_commands_are_scoped_to_live_run(self):
        self.pause()
        result = web_ui.resume_runtime()
        self.assertEqual(result["command"]["run_id"], "run")
        self.assertEqual(self.control.consume()["command"], "resume")

    def test_ui_restart_can_observe_cli_owned_process(self):
        self.pause()
        status = web_ui.runtime_status()
        self.assertTrue(status["running"])
        self.assertFalse(status["automatic_active"])
        self.assertEqual(status["state"], "paused_safe")

    def test_dead_process_does_not_claim_safe_for_ai(self):
        self.control.write_status("paused_safe", run_id="run", pause_id="pause")
        with patch.object(web_ui, "process_alive", return_value=False):
            status = web_ui.runtime_diagnostics()
        self.assertFalse(status["safe_for_ai"])
        self.assertEqual(status["state"], "process_exit")

    def test_api_payload_cannot_change_command_type(self):
        lease = self.pause()
        web_ui.runtime_ai_action({**lease, "kind": "inspect", "action_id": "test",
                                 "command": "resume", "id": "injected"})
        command = self.control.consume()
        self.assertEqual(command["command"], "ai_action")
        self.assertNotEqual(command["id"], "injected")


if __name__ == "__main__":
    unittest.main()
