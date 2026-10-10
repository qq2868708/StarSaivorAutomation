"""Offline checks for dependency setup; no packages are installed by these tests."""
import importlib.metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import bootstrap_ui


class BootstrapTests(unittest.TestCase):
    def test_only_supported_64_bit_python_is_accepted(self):
        self.assertTrue(bootstrap_ui.supported_python((3, 12), 64))
        self.assertTrue(bootstrap_ui.supported_python((3, 11), 64))
        self.assertFalse(bootstrap_ui.supported_python((3, 13), 64))
        self.assertFalse(bootstrap_ui.supported_python((3, 14), 64))
        self.assertFalse(bootstrap_ui.supported_python((3, 12), 32))

    def test_missing_and_wrong_versions_require_installation(self):
        def version(name):
            if name == "missing":
                raise importlib.metadata.PackageNotFoundError(name)
            return "1.0"
        self.assertEqual(bootstrap_ui.mismatched_packages(
            {"correct": "1.0", "wrong": "2.0", "missing": "1.0"}, version),
            ["wrong", "missing"])

    def test_installed_environment_never_invokes_pip_install(self):
        with patch.object(bootstrap_ui, "mismatched_packages", return_value=[]), \
                patch.object(bootstrap_ui, "run_check") as run:
            bootstrap_ui.prepare_environment()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], ["-m", "pip", "check"])
        self.assertTrue(run.call_args_list[1].args[0][0].endswith("check_environment.py"))

    def test_changed_environment_installs_then_checks_actual_ocr(self):
        with patch.object(bootstrap_ui, "mismatched_packages", side_effect=[["numpy"], []]), \
                patch.object(bootstrap_ui, "panel_ready", return_value=False), \
                patch.object(bootstrap_ui, "run_check") as run:
            bootstrap_ui.prepare_environment()
        labels = [call.args[1] for call in run.call_args_list]
        self.assertEqual(labels, ["pip bootstrap", "dependency installation",
                                  "dependency consistency", "OCR runtime check"])
        self.assertIn("requirements-lock.txt", run.call_args_list[1].args[0])

    def test_running_server_is_not_modified(self):
        with patch.object(bootstrap_ui, "mismatched_packages", return_value=["numpy"]), \
                patch.object(bootstrap_ui, "panel_ready", return_value=True), \
                patch.object(bootstrap_ui, "run_check") as run:
            with self.assertRaisesRegex(RuntimeError, "Stop the running panel"):
                bootstrap_ui.prepare_environment()
            run.assert_not_called()

    def test_explicit_repair_reinstalls_even_when_metadata_matches(self):
        with patch.object(bootstrap_ui, "mismatched_packages", return_value=[]), \
                patch.object(bootstrap_ui, "panel_ready", return_value=False), \
                patch.object(bootstrap_ui, "run_check") as run:
            bootstrap_ui.prepare_environment(force=True)
        self.assertIn("--force-reinstall", run.call_args_list[1].args[0])

    def test_lock_pins_original_ocr_stack(self):
        pins = bootstrap_ui.locked_versions(bootstrap_ui.ROOT / "requirements-lock.txt")
        self.assertEqual(pins["rapidocr-onnxruntime"], "1.4.4")
        self.assertEqual(pins["numpy"], "1.26.4")

    def test_setup_log_redacts_paths_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime").mkdir()
            with patch.object(bootstrap_ui, "ROOT", root):
                bootstrap_ui.log_output(f"{root}/private token=secretvalue\nhttps://user:pass@example.test")
            text = (root / "runtime" / "bootstrap.log").read_text(encoding="utf-8")
            self.assertNotIn(str(root), text)
            self.assertNotIn("secretvalue", text)
            self.assertNotIn("user:pass", text)


if __name__ == "__main__":
    unittest.main()
