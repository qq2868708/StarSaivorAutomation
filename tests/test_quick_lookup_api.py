"""Scan state and diagnostics contracts without sending game input."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
import web_ui
from src.quick_lookup_log import append_event, log_path, recent_events
from src.runtime_control import RuntimeControl


class QuickLookupAPITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for item in (patch.object(web_ui, "_runtime_control", RuntimeControl(self.root)),
                     patch.object(web_ui, "_quick_lookup_tasks", {}),
                     patch.object(web_ui, "_automation_status", return_value={"running": False})):
            item.start()
            self.addCleanup(item.stop)

    def task(self, kind, state="waiting_foreground"):
        task = {"task_id": kind + "-id", "kind": kind, "state": state,
                "started_at": "2026-10-10T12:00:00+08:00"}
        web_ui._quick_lookup_tasks[task["task_id"]] = task
        return task

    def test_other_family_and_stopping_workers_cannot_overlap(self):
        task = self.task("rescuer")
        with patch.object(web_ui.threading, "Thread") as thread:
            with self.assertRaises(HTTPException) as error:
                web_ui.scan_quick_arcanum()
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(web_ui.scan_quick_rescuer()["task"]["task_id"], task["task_id"])
            task["state"] = "stopping"
            with self.assertRaises(HTTPException):
                web_ui.scan_quick_rescuer()
            thread.assert_not_called()

    def test_stop_targets_only_requested_family(self):
        rescuer = self.task("rescuer")
        with self.assertRaises(HTTPException):
            web_ui.stop_quick_lookup("arcanum")
        self.assertEqual(rescuer["state"], "waiting_foreground")
        with self.assertRaises(HTTPException):
            web_ui.stop_quick_lookup("../../secret")
        web_ui.stop_quick_lookup("rescuer")
        self.assertEqual(rescuer["state"], "stopping")

    def test_initialization_failure_is_logged_separately_with_redacted_frames(self):
        task = self.task("arcanum")
        with patch.object(web_ui, "QuickLookupScanner", side_effect=KeyError("model_path")):
            web_ui._run_quick_lookup_task(task["task_id"], "arcanum", "StarSavior")
        data = web_ui.quick_lookup_state()
        self.assertEqual(data["tasks"]["arcanum"]["state"], "failed")
        self.assertEqual(data["logs"]["rescuer"], [])
        failure = data["logs"]["arcanum"][-1]
        self.assertEqual(failure["event"], "failed")
        self.assertEqual(failure["error_type"], "KeyError")
        self.assertTrue(failure["traceback"])
        for frame in failure["traceback"]:
            self.assertNotIn("\\", frame["file"])
            self.assertNotIn("/", frame["file"])

    def test_stop_before_worker_start_is_not_reset(self):
        task = self.task("rescuer", "stopping")
        scanner = Mock()
        with patch.object(web_ui, "QuickLookupScanner", return_value=scanner):
            web_ui._run_quick_lookup_task(task["task_id"], "rescuer", "StarSavior")
        self.assertEqual(task["state"], "stopped")
        scanner.scan.assert_not_called()

    def test_progress_and_both_family_states_are_returned(self):
        old = self.task("arcanum", "failed")
        task = self.task("rescuer_single")
        scanner = Mock()
        scanner.scan_selected_rescuer.side_effect = RuntimeError("test failure")
        def construct(*args, progress, **kwargs):
            progress({"stage": "rescuer_detail", "retries": 2, "current": "role",
                      "last_retry_error": "transition"})
            return scanner
        with patch.object(web_ui, "QuickLookupScanner", side_effect=construct):
            web_ui._run_quick_lookup_task(task["task_id"], task["kind"], "StarSavior")
        data = web_ui.quick_lookup_state()
        self.assertEqual(data["tasks"]["arcanum"]["task_id"], old["task_id"])
        self.assertEqual(data["tasks"]["rescuer"]["stage"], "rescuer_detail")
        self.assertEqual(data["tasks"]["rescuer"]["last_retry_error"], "transition")

    def test_logs_are_redacted_bounded_and_survive_task_reset(self):
        append_event(self.root, "rescuer_single", "failed",
                     error=f"{self.root}/private.py {Path.home()}/private token=abc123")
        with patch("src.quick_lookup_log._MAX_BYTES", 1):
            append_event(self.root, "arcanum", "first")
            append_event(self.root, "arcanum", "second")
        data = web_ui.quick_lookup_state()
        self.assertEqual(data["tasks"], {})
        text = json.dumps(data["logs"])
        self.assertNotIn("abc123", text)
        self.assertIn("<project>", text)
        self.assertIn("<home>", text)
        self.assertEqual(recent_events(self.root, "arcanum")[-1]["event"], "second")
        self.assertTrue(log_path(self.root, "arcanum").with_suffix(".jsonl.1").exists())
        with self.assertRaises(ValueError):
            recent_events(self.root, "../secret")

    def test_finished_traversal_below_ninety_percent_is_partial(self):
        task = self.task("rescuer")
        scanner = Mock()
        scanner.scan.return_value = {
            "roster_scan": {"complete": True, "accepted": False, "inventory_complete": True,
                            "attempted": 10, "succeeded": 8, "failed": 2,
                            "total": 10, "success_rate": .8,
                            "failed_characters": ["甲", "乙"],
                            "stop_reason": "本轮通过率 80.0%，未达到 90%"}}
        with patch.object(web_ui, "QuickLookupScanner", return_value=scanner):
            web_ui._run_quick_lookup_task(task["task_id"], "rescuer", "StarSavior")
        public = web_ui.quick_lookup_state()["tasks"]["rescuer"]
        self.assertEqual(public["state"], "partial")
        self.assertFalse(public["accepted"])
        self.assertEqual(public["success_rate"], .8)
        self.assertEqual(public["total"], 10)
        self.assertEqual(public["failed"], 2)


if __name__ == "__main__":
    unittest.main()
