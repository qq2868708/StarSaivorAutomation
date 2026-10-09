"""运行时控制面测试，不启动游戏或发送 Win32 输入。"""
import tempfile
import unittest
from pathlib import Path

from src.runtime_control import RuntimeControl


class RuntimeControlTests(unittest.TestCase):
    def test_commands_are_atomic_and_consumed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RuntimeControl(Path(directory))
            issued = control.issue("pause", source="test")
            self.assertEqual(control.consume()["id"], issued["id"])
            self.assertIsNone(control.consume())

    def test_checkpoint_and_status_are_json_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RuntimeControl(Path(directory))
            control.write_status("paused_safe", "unknown_screen", round_count=7)
            control.write_checkpoint(phase="safe_paused", round_count=7,
                                    screen_fingerprint="abc")
            self.assertEqual(control.read_status()["state"], "paused_safe")
            self.assertEqual(control.read_checkpoint()["screen_fingerprint"], "abc")

    def test_ai_action_is_a_distinct_command(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RuntimeControl(Path(directory))
            control.issue("ai_action", kind="inspect")
            command = control.consume()
            self.assertEqual(command["command"], "ai_action")
            self.assertEqual(command["kind"], "inspect")

    def test_multiple_commands_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RuntimeControl(directory)
            ids = [control.issue(command)["id"] for command in ("pause", "resume", "stop")]
            self.assertEqual([control.consume()["id"] for _ in ids], ids)

    def test_payload_cannot_replace_protocol_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RuntimeControl(directory)
            value = control.issue("pause", id="injected", created_at="injected")
            self.assertNotEqual(value["id"], "injected")
            self.assertNotEqual(value["created_at"], "injected")

    def test_os_owner_lock_prevents_second_script(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = RuntimeControl(directory), RuntimeControl(directory)
            a.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    b.acquire()
            finally:
                a.release()
            b.acquire()
            b.release()

    def test_ai_registration_is_bound_to_one_safe_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            control = RuntimeControl(directory)
            control.write_status("running", run_id="run", pause_id=None)
            with self.assertRaises(ValueError):
                control.claim_ai("test", "run", "pause")
            control.write_status("paused_safe", run_id="run", pause_id="pause")
            lease = control.claim_ai("test", "run", "pause")
            self.assertTrue(control.valid_ai(lease, "run", "pause"))
            self.assertFalse(control.valid_ai(lease, "run", "next-pause"))


if __name__ == "__main__":
    unittest.main()
