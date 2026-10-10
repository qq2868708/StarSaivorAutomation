"""Arcana page navigation tests without game input."""
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.quick_lookup import (
    QuickLookupScanner, QuickLookupInterrupted,
    _arcanum_detail_confirmed, _arcanum_grid_tiles, parse_arcanum,
    _arcanum_art_patch, _locate_arcanum_art,
    _same_arcanum_view, _index_arcanum_tiles,
)


def text(value, x, y, confidence=.99):
    return value, confidence, [[x, y], [x + 80, y], [x + 80, y + 24], [x, y + 24]]


GRID = [text("阿尔克那", 206, 43), text("管理", 143, 142),
        text("SR", 387, 445), text("SR", 581, 445)]


def detail(name):
    return [text("旅程效果", 574, 259), text("专属效果", 575, 317),
            text("·旅程初始效果", 752, 383),
            text("莉迪亚", 740, 249), text(name, 744, 281)]


class ArcanumScanTests(unittest.TestCase):
    def test_bullet_heading_is_accepted_only_in_expected_region(self):
        rows = detail("令人满意的开花")
        self.assertTrue(_arcanum_detail_confirmed(rows))
        wrong = [r for r in rows if r[0] != "·旅程初始效果"]
        self.assertFalse(_arcanum_detail_confirmed(wrong + [text("·旅程初始效果", 100, 100)]))
        self.assertFalse(_arcanum_detail_confirmed(wrong + [text("·旅程初始效果", 752, 383, .5)]))
        self.assertEqual(parse_arcanum(detail("归途"))["cards"][0]["name"]["value"], "归途")
        support = [(t.replace("旅程初始效果", "支持委托效果"), s, b) for t, s, b in rows]
        self.assertTrue(_arcanum_detail_confirmed(support))
        training_only = [(t.replace("旅程初始效果", "训练效果"), s, b) for t, s, b in rows]
        self.assertTrue(_arcanum_detail_confirmed(training_only))
        self.assertEqual(parse_arcanum(detail("清扫，不速之客"))["cards"][0]["name"]["value"], "清扫，不速之客")

    def test_grid_needs_page_and_management_markers(self):
        self.assertEqual(len(_arcanum_grid_tiles(GRID)), 2)
        self.assertFalse(_arcanum_grid_tiles(GRID[1:]))
        self.assertFalse(_arcanum_grid_tiles([r for r in GRID if r[0] != "管理"]))
        self.assertFalse(_arcanum_grid_tiles(detail("卡片甲")))

    def scanner(self, root, stop_after_first=False):
        state = {"open": False, "index": 0}
        controller, capture, ocr, persist = Mock(), Mock(), Mock(), Mock()
        frame = np.random.default_rng(42).integers(0, 255, (900, 1600, 3), dtype=np.uint8)
        capture.capture_game.return_value = frame
        names = ["令人满意的开花", "归途"]
        ocr.recognize_detailed.side_effect = lambda _: detail(names[state["index"]]) if state["open"] else GRID
        def click(x, y, *, source, **kwargs):
            if source == "quick_lookup_open_arcanum":
                state.update(open=True, index=0 if x < .3 else 1)
            elif source == "quick_lookup_close_arcanum":
                state["open"] = False
            else:
                raise AssertionError(source)
        controller.click_at_percent.side_effect = click
        scanner = QuickLookupScanner(root, capture=capture, controller=controller, ocr=ocr,
                                     on_arcanum_entry=persist, sleep=lambda _: None)
        scanner.scan_kind = "arcanum"
        if stop_after_first:
            scanner.should_stop = lambda: scanner.completed_count >= 1
        return scanner, state, frame, controller, persist

    def test_visible_cards_are_read_closed_and_incrementally_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            scanner, state, frame, controller, persist = self.scanner(root)
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                result = scanner._scan_arcanum_page("abc", (0, 0, 1600, 900), 1, frame, GRID)
            self.assertFalse(state["open"])
            self.assertEqual([card["name"] for card in result["cards"]], ["令人满意的开花", "归途"])
            self.assertEqual(result["scan_summary"]["succeeded"], 2)
            self.assertTrue(result["scan_summary"]["complete"])
            self.assertEqual(persist.call_count, 2)
            self.assertEqual([call.kwargs["source"] for call in controller.click_at_percent.call_args_list],
                             ["quick_lookup_open_arcanum", "quick_lookup_close_arcanum"] * 2)

    def test_stop_keeps_first_record_and_does_not_click_next_card(self):
        with tempfile.TemporaryDirectory() as root:
            scanner, state, frame, controller, persist = self.scanner(root, True)
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                with self.assertRaises(QuickLookupInterrupted):
                    scanner._scan_arcanum_page("abc", (0, 0, 1600, 900), 1, frame, GRID)
            persist.assert_called_once()
            self.assertEqual(controller.click_at_percent.call_count, 2)
            self.assertFalse(state["open"])

    def test_unknown_page_does_not_send_a_close_click(self):
        with tempfile.TemporaryDirectory() as root:
            scanner = QuickLookupScanner(root, controller=Mock())
            scanner._read_frame = Mock(return_value=(None, []))
            with self.assertRaisesRegex(RuntimeError, "未确认"):
                scanner._close_arcanum("abc", 1, (0, 0, 1600, 900))
            scanner.controller.click_at_percent.assert_not_called()

    def test_art_locator_tracks_scrolled_card_and_rejects_duplicates(self):
        frame = np.random.default_rng(4).integers(0, 255, (900, 1600, 3), dtype=np.uint8)
        tile = _arcanum_grid_tiles(GRID)[1]
        patch = _arcanum_art_patch(frame, tile)
        moved = np.zeros_like(frame)
        moved[220:270, 510:565] = patch
        self.assertEqual(_locate_arcanum_art(moved, patch), (537.5/1600, 245/900))
        moved[320:370, 710:765] = patch
        with self.assertRaisesRegex(ValueError, "多个"):
            _locate_arcanum_art(moved, patch)
        self.assertFalse(_arcanum_grid_tiles(
            [text("阿尔克那", 206, 43), text("管理", 143, 142), text("SR", 387, 273)]))

    def test_mixed_rarity_box_jitter_is_stable_and_does_not_duplicate_cards(self):
        frame = np.random.default_rng(42).integers(0, 255, (900, 1600, 3), dtype=np.uint8)
        jittered = GRID[:2] + [text("SSR", 378, 449), text("R", 580, 446)]
        self.assertTrue(_same_arcanum_view((frame, GRID), (frame, jittered)))
        patches = []
        self.assertEqual(_index_arcanum_tiles(frame, GRID, patches, discover=True), [0, 1])
        self.assertEqual(_index_arcanum_tiles(frame, jittered, patches, discover=True), [0, 1])
        self.assertEqual(len(patches), 2)

    def test_grid_wait_uses_art_not_exact_ocr_box_rounding(self):
        with tempfile.TemporaryDirectory() as root:
            scanner, _, frame, _, _ = self.scanner(root)
            jittered = GRID[:2] + [text("SSR", 378, 449), text("R", 580, 446)]
            scanner._read_frame = Mock(side_effect=[(frame, GRID), (frame, jittered)])
            result = scanner._wait_arcanum_page(1, "grid")
            self.assertIs(result[0], frame)
            self.assertEqual(scanner._read_frame.call_count, 2)

    def test_unknown_art_reports_geometry_and_score_without_accepting_or_adding_it(self):
        frame = np.random.default_rng(42).integers(0, 255, (900, 1600, 3), dtype=np.uint8)
        patch = np.random.default_rng(43).integers(0, 255, (50, 55, 3), dtype=np.uint8)
        patches = [patch]
        with self.assertRaisesRegex(
                ValueError, r"标准化位置=\(410,305\)，最高卡面匹配分数=0\.\d{4}"):
            _index_arcanum_tiles(frame, GRID, patches)
        self.assertEqual(len(patches), 1)
        self.assertIs(patches[0], patch)


if __name__ == "__main__":
    unittest.main()
