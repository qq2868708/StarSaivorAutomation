"""Cross-page Arcana tests: real image matching, mocked window input only."""
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.quick_lookup import QuickLookupScanner, QuickLookupInterrupted, _arcanum_owned_count
from test_arcanum_scan import GRID, detail, text


class ArcanumLibraryTests(unittest.TestCase):
    def setup_scanner(self, expected=5, frozen=False, auto_scroll=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        controller = Mock()
        scanner = QuickLookupScanner(temp.name, controller=controller, sleep=lambda _: None)
        scanner.scan_kind = "arcanum"
        scanner.on_arcanum_entry = Mock()
        state = {"page": 2, "open": None, "opened": []}
        names = ["卡片甲", "卡片乙", "卡片丙", "卡片丁", "卡片戊"]
        art = [np.random.default_rng(i).integers(0, 255, (50, 55, 3), dtype=np.uint8)
               for i in range(5)]
        frames = []
        for i in range(4):
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)
            frame[360:410, 337:392] = art[i]
            frame[360:410, 531:586] = art[i + 1]
            frames.append(frame)
        def view(*args, **kwargs):
            if state["open"] is not None:
                return frames[state["page"]], detail(names[state["open"]])
            return frames[state["page"]], GRID + [text(f"{expected}/80", 1194, 114)]
        scanner._wait_arcanum_page = view
        scanner._read_frame = view
        def scroll(*args, clicks, **kwargs):
            self.assertIsNone(state["open"])
            if not frozen:
                state["page"] = min(3, max(0, state["page"] + (1 if clicks < 0 else -1)))
        controller.scroll_at_percent.side_effect = scroll
        def click(x, y, *, source, **kwargs):
            if source == "quick_lookup_open_arcanum":
                state["open"] = state["page"] + (0 if x < .3 else 1)
                state["opened"].append(state["open"])
            elif source == "quick_lookup_close_arcanum":
                if auto_scroll:
                    state["page"] = min(3, state["open"])
                state["open"] = None
            else:
                raise AssertionError(source)
        controller.click_at_percent.side_effect = click
        return scanner, controller, state

    def test_cross_page_scan_reaches_bottom_and_visits_each_card_once(self):
        scanner, controller, state = self.setup_scanner()
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            result = scanner._scan_arcanum_library("abc", (0, 0, 1600, 900), 1)
        summary = result["scan_summary"]
        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["attempted"], 5)
        self.assertEqual(summary["succeeded"], 5)
        self.assertTrue(summary["complete"])
        self.assertEqual(state["opened"], [0, 1, 2, 3, 4])
        self.assertIsNone(state["open"])
        self.assertEqual(state["page"], 3)
        self.assertEqual(scanner.on_arcanum_entry.call_count, 5)
        self.assertGreater(controller.scroll_at_percent.call_count, 4)

    def test_auto_scroll_after_close_does_not_skip_newly_visible_cards(self):
        scanner, _, state = self.setup_scanner(auto_scroll=True)
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            result = scanner._scan_arcanum_library("abc", (0, 0, 1600, 900), 1)
        self.assertTrue(result["scan_summary"]["complete"])
        self.assertEqual(state["opened"], [0, 1, 2, 3, 4])

    def test_ignored_scroll_cannot_claim_full_library(self):
        scanner, _, state = self.setup_scanner(frozen=True)
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "全库清点未完成"):
                scanner._scan_arcanum_library("abc", (0, 0, 1600, 900), 1)
        self.assertEqual(state["opened"], [])
        scanner.on_arcanum_entry.assert_not_called()

    def test_missing_card_count_prevents_complete(self):
        scanner, _, state = self.setup_scanner(expected=6)
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "识别 5 / 总数 6"):
                scanner._scan_arcanum_library("abc", (0, 0, 1600, 900), 1)
        self.assertEqual(state["opened"], [])

    def test_stop_preserves_saved_card_and_sends_no_further_input(self):
        scanner, _, state = self.setup_scanner()
        scanner.should_stop = lambda: scanner.completed_count >= 1
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            with self.assertRaises(QuickLookupInterrupted):
                scanner._scan_arcanum_library("abc", (0, 0, 1600, 900), 1)
        self.assertEqual(state["opened"], [0])
        self.assertIsNone(state["open"])
        scanner.on_arcanum_entry.assert_called_once()

    def test_count_requires_correct_region_and_valid_capacity(self):
        self.assertEqual(_arcanum_owned_count([text("52/80", 1194, 114)]), 52)
        self.assertIsNone(_arcanum_owned_count([text("52/80", 200, 400)]))
        self.assertIsNone(_arcanum_owned_count([text("90/80", 1194, 114)]))
        self.assertIsNone(_arcanum_owned_count([text("52/80", 1194, 114, .4)]))


if __name__ == "__main__":
    unittest.main()
