import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.quick_lookup import (QuickLookupScanner, _name_compatible, _page_kind,
                              _roster_click_percent, _IDENTITY_BOXES,
                              build_match, parse_arcanum, parse_journey_initial,
                              parse_rescuer, QuickLookupInterrupted,
                              QuickLookupStore, sanitize_quick_lookup_scan)


def box(x, y, w=60, h=24):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class QuickLookupTests(unittest.TestCase):
    def test_rescuer_reads_selected_character_and_visible_roster(self):
        rows = [
            ("救援者", .99, box(204, 42, 80, 32)),
            ("SSR", .99, box(159, 110, 50, 32)),
            ("星光华尔兹", .99, box(158, 144, 110, 26)),
            ("亚瑟菈", .99, box(157, 172, 70, 25)),
            ("Lv.180/180", .99, box(64, 213, 124, 24)),
            ("45,381", .99, box(61, 242, 184, 48)),
            ("Lv.180", .99, box(1290, 170, 110, 23)),
            ("亚瑟菈", .99, box(1336, 194, 75, 28)),
            ("Lv.179", .99, box(1335, 260, 110, 24)),
            ("克拉丽莎", .99, box(1336, 284, 95, 28)),
        ]
        result = parse_rescuer(rows)
        self.assertEqual(result["character"]["value"], "亚瑟菈")
        self.assertEqual(result["power"]["value"], 45381)
        self.assertEqual(result["rarity"], "SSR")
        self.assertGreaterEqual(len(result["roster"]), 2)

    def test_arcanum_effects_and_match_keep_final_stats_separate(self):
        rows = [
            ("阿尔克那", .99, box(100, 40)),
            ("莉迪亚", .99, box(150, 100)),
            ("令人满意的开花", .99, box(300, 100, 160, 24)),
            ("SR", .99, box(300, 140)),
            ("Lv.45", .99, box(300, 170)),
            ("旅程初始效果", .99, box(100, 250)),
            ("羁绊点数增加 +9", .99, box(100, 280, 180, 24)),
            ("训练效果", .99, box(100, 330)),
            ("训练效果上升 +11.4%", .99, box(100, 360, 210, 24)),
            ("训练失败率", .99, box(100, 410)),
            ("训练失败率减少 +3.1%", .99, box(100, 440, 220, 24)),
        ]
        arcanum = parse_arcanum(rows)
        self.assertEqual(arcanum["cards"][0]["name"]["value"], "令人满意的开花")
        self.assertEqual(arcanum["cards"][0]["level"], 45)
        match = build_match({"character": {"value": "亚瑟菈"}}, arcanum)
        self.assertEqual(match["character"], "亚瑟菈")
        self.assertIsNone(match["journey_end_mapping"]["outcome.final_stats"]["value"])

    def test_journey_initial_info_maps_each_roster_entry(self):
        rows = [
            ("潜质", .99, box(646, 172, 47, 30)),
            ("救援者介绍", .99, box(780, 173, 100, 28)),
            ("生命感知", .99, box(677, 268, 82, 27)),
            ("未习得", .99, box(916, 268, 63, 28)),
            ("亚瑟菈", .99, box(323, 407, 85, 32)),
            ("力量", .99, box(208, 481, 55, 22)),
            ("体力", .99, box(375, 481, 54, 22)),
            ("韧性", .99, box(207, 539, 57, 26)),
            ("专注", .99, box(373, 539, 57, 26)),
            ("保护", .99, box(208, 601, 55, 22)),
        ]
        initial = parse_journey_initial(rows)
        self.assertEqual(initial["journey_initial_character"], "亚瑟菈")
        match = build_match(
            {"character": {"value": "亚瑟菈"}, "roster_details": [{
                "character": "亚瑟菈", "journey_initial": initial,
            }]},
            {"cards": []},
        )
        self.assertEqual(match["character_mappings"][0]["journey_end_character"], "亚瑟菈")

    def test_initial_modal_is_detected_without_a_title_and_matches_ocr_variant(self):
        rows = [
            ("潜质", .99, box(646, 172)),
            ("救援者介绍", .99, box(780, 173)),
            ("生命感知", .99, box(677, 268)),
            ("未习得", .99, box(916, 268)),
            ("亚瑟", .99, box(323, 407, 85, 32)),
            ("力量", .99, box(208, 481)),
            ("体力", .99, box(375, 481)),
            ("韧性", .99, box(207, 539)),
            ("专注", .99, box(373, 539)),
        ]
        self.assertEqual(_page_kind(rows), "journey_initial")
        parsed = parse_journey_initial(rows, "亚瑟菈")
        self.assertEqual(parsed["journey_initial_character"], "亚瑟")
        self.assertTrue(_name_compatible("亚瑟菈", "亚瑟"))

    def test_page_markers_are_mutually_exclusive(self):
        self.assertEqual(_page_kind([("救援者", .99, box(200, 40)),
                                     ("查看详情", .99, box(1370, 825))]), "rescuer_list")
        self.assertEqual(_page_kind([("救援者", .99, box(200, 40)),
                                     ("基本", .99, box(1180, 120)),
                                     ("旅程初始信息", .99, box(1320, 795))]), "rescuer_detail")

    def test_weak_basic_tab_is_rejected_even_with_stat_labels(self):
        rows = [
            ("救援者", .749, box(205, 42, 78, 32)),
            ("基本", .604, box(1203, 123, 46, 26)),
            ("旅程初始信息", .832, box(1321, 796, 110, 23)),
            ("攻击力", .699, box(1211, 309)),
            ("生命力", .706, box(1211, 341)),
            ("防御力", .745, box(1191, 367)),
        ]
        self.assertIsNone(_page_kind(rows))
        strong = [(text, .99, bounds) for text, _, bounds in rows]
        self.assertEqual(_page_kind(strong), "rescuer_detail")
        self.assertIsNone(_page_kind(rows[:3]))
        self.assertIsNone(_page_kind(rows[1:]))
        self.assertIsNone(_page_kind([row for row in rows if row[0] != "旅程初始信息"]))
        misplaced = rows[:3] + [(t, c, box(200, 320)) for t, c, _ in rows[3:]]
        self.assertIsNone(_page_kind(misplaced))

    def test_recovery_unknown_page_is_bounded_without_clicking(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as root:
            scanner = QuickLookupScanner(root, sleep=lambda _: None, controller=Mock())
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)
            scanner._read_frame = lambda hwnd: (frame, [])
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "连续三次"):
                    scanner._retry_rescuer_entry("scan", (0, 0, 1600, 900), 1, "甲", 1)
            self.assertEqual(scanner.retry_count, 3)
            scanner.controller.click_at_percent.assert_not_called()

    def test_detail_confirmation_still_requires_two_frames_and_identity(self):
        from unittest.mock import Mock
        rows = [
            ("救援者", .75, box(205, 42)),
            ("基本", .99, box(1203, 123)),
            ("旅程初始信息", .832, box(1321, 796)),
            ("攻击力", .70, box(1211, 309)),
            ("生命力", .71, box(1211, 341)),
            ("防御力", .75, box(1191, 367)),
            ("克拉丽莎", .79, box(159, 171)),
        ]
        with tempfile.TemporaryDirectory() as root:
            scanner = QuickLookupScanner(root, sleep=lambda _: None, controller=Mock())
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)
            scanner._read_frame = Mock(return_value=(frame, rows))
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                scanner._wait_page(1, "rescuer_detail", expected_name="克拉丽莎")
            self.assertEqual(scanner._read_frame.call_count, 2)
            scanner.controller.click_at_percent.assert_not_called()

    def test_failed_traversal_row_never_enters_character_mapping(self):
        match = build_match({"character": {"value": "亚瑟菈"}, "roster_details": [
            {"character": "错误角色", "verified": False,
             "journey_initial": None, "uncertain": ["transition"]},
        ]}, {"cards": []})
        self.assertEqual(match["character_mappings"], [])
        self.assertIn("roster_details", match["uncertain"])

    def test_detail_transition_uses_modal_close_then_detail_back_arrow(self):
        list_rows = [
            ("救援者", .99, box(204, 42, 80, 32)),
            ("亚瑟菈", .99, box(157, 172, 70, 25)),
            ("查看详情", .99, box(1370, 825, 86, 28)),
            ("Lv.180", .99, box(1290, 170, 110, 23)),
            ("亚瑟", .99, box(1336, 194, 75, 28)),
        ]
        detail_rows = list_rows[:2] + [
            ("基本", .99, box(1180, 120, 72, 32)),
            ("旅程初始信息", .99, box(1321, 795, 110, 24)),
        ]
        initial_rows = [
            ("潜质", .99, box(646, 172, 47, 30)),
            ("救援者介绍", .99, box(780, 173, 100, 28)),
            ("生命感知", .99, box(677, 268, 82, 27)),
            ("未习得", .99, box(916, 268, 63, 28)),
            ("亚瑟", .99, box(323, 407, 85, 32)),
            ("力量", .99, box(208, 481)), ("体力", .99, box(375, 481)),
            ("韧性", .99, box(207, 539)), ("专注", .99, box(373, 539)),
        ]
        scanner = QuickLookupScanner(tempfile.mkdtemp(), sleep=lambda _: None)
        scanner._save_crop = lambda *args, **kwargs: "logs/screenshots/portrait.png"
        scanner._click_row = lambda *args, **kwargs: None
        clicked = []
        scanner._click_fixed = lambda hwnd, region, x, y, source: clicked.append((x, y, source))
        frame = type("Frame", (), {"shape": (900, 1600, 3)})()
        pages = iter([(frame, detail_rows), (frame, initial_rows),
                      (frame, detail_rows), (frame, list_rows)])
        scanner._wait_page = lambda hwnd, kind, **kwargs: next(pages)
        page = parse_rescuer(list_rows)
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            result = scanner._read_rescuer_detail("scan", (0, 0, 1600, 900), 1,
                                                frame, page, 1, expected_name="亚瑟")
        self.assertTrue(result["can_continue"])
        self.assertEqual(clicked[0][2], "quick_lookup_close_initial")
        self.assertEqual(clicked[1][2], "quick_lookup_back_detail")
        self.assertEqual(clicked[0][:2], (.889, .203))
        self.assertEqual(clicked[1][:2], (.043, .064))

    def test_initial_modal_requires_potential_label_as_well_as_identity_and_stats(self):
        rows = [("救援者介绍", .833, box(780, 173)),
                ("克拉丽莎", .85, box(323, 407, 85, 32)),
                ("力量", .90, box(208, 481)),
                ("体力", .90, box(375, 481)),
                ("韧性", .90, box(207, 539))]
        self.assertIsNone(_page_kind(rows))
        confirmed = rows + [("潜质", .99, box(646, 172))]
        self.assertEqual(_page_kind(confirmed), "journey_initial")
        self.assertEqual(parse_journey_initial(confirmed, "克拉丽莎")["character"]["value"], "克拉丽莎")
        self.assertIsNone(parse_journey_initial(confirmed, "其他角色")["character"])
        self.assertIsNone(_page_kind(rows[:1] + rows[2:]))
        self.assertIsNone(_page_kind(rows[:3]))
        self.assertIsNone(_page_kind(rows[1:]))

    def test_collect_close_and_return_flow_with_real_page_waits(self):
        from unittest.mock import Mock
        from src.quick_lookup_log import recent_events
        list_rows = [("救援者", .95, box(205, 42)), ("克拉丽莎", .95, box(159, 171)),
                     ("查看详情", .95, box(1370, 825))]
        detail_rows = list_rows[:2] + [("基本", .95, box(1203, 123)),
                                      ("旅程初始信息", .95, box(1321, 796))]
        modal_rows = [("潜质", .99, box(646, 172)),
                      ("救援者介绍", .99, box(780, 173)),
                      ("克拉丽莎", .95, box(323, 407, 85, 32)),
                      ("力量", .95, box(208, 481)), ("体力", .95, box(375, 481)),
                      ("韧性", .95, box(207, 539))]
        with tempfile.TemporaryDirectory() as root:
            pages = {"list": list_rows, "detail": detail_rows, "modal": modal_rows}
            current = {"page": "list"}
            sources = []
            def click(*args, source, **kwargs):
                sources.append(source)
                current["page"] = {
                    "quick_lookup_view_details": "detail",
                    "quick_lookup_journey_initial": "modal",
                    "quick_lookup_close_initial": "detail",
                    "quick_lookup_back_detail": "list",
                }[source]
            controller, capture, ocr = Mock(), Mock(), Mock()
            controller.click_at_percent.side_effect = click
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)
            capture.capture_game.return_value = frame
            ocr.recognize_detailed.side_effect = lambda _: pages[current["page"]]
            scanner = QuickLookupScanner(root, controller=controller, capture=capture,
                                         ocr=ocr, sleep=lambda _: None)
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                result = scanner._read_rescuer_detail(
                    "abc", (0, 0, 1600, 900), 1, frame,
                    parse_rescuer(list_rows), 1, expected_name="克拉丽莎")
            self.assertTrue(result["can_continue"])
            self.assertEqual(current["page"], "list")
            self.assertEqual(sources, ["quick_lookup_view_details", "quick_lookup_journey_initial",
                                      "quick_lookup_close_initial", "quick_lookup_back_detail"])
            self.assertEqual(ocr.recognize_detailed.call_count, 8)
            self.assertEqual([event["event"] for event in recent_events(root, "rescuer")],
                             ["initial_info_collected", "closing_initial",
                              "returning_to_list", "returned_to_list"])

    def test_failed_row_is_retried_until_it_succeeds(self):
        first = [
            ("救援者", .99, box(204, 42, 80, 32)),
            ("亚瑟菈", .99, box(157, 172, 70, 25)),
            ("查看详情", .99, box(1370, 825, 86, 28)),
            ("Lv.180", .99, box(1290, 170, 110, 23)),
            ("亚瑟", .99, box(1336, 194, 75, 28)),
            ("Lv.179", .99, box(1335, 260, 110, 24)),
            ("克拉丽莎", .99, box(1336, 284, 95, 28)),
        ]
        frame = type("Frame", (), {"shape": (900, 1600, 3)})()
        scanner = QuickLookupScanner(tempfile.mkdtemp(), sleep=lambda _: None)
        scanner._save_crop = lambda *args, **kwargs: "logs/screenshots/portrait.png"
        scanner._read_frame = lambda hwnd: (frame, first)
        scanner.controller.click_at_percent = lambda *args, **kwargs: None
        calls = {"count": 0}
        def read_detail(*args, expected_name=None, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("旅程初始信息角色未确认")
            return {"can_continue": True,
                    "journey_initial": {"journey_initial_character": expected_name,
                                        "character": {"value": expected_name}},
                    "uncertain": []}
        scanner._read_rescuer_detail = read_detail
        scanner._recover_to_rescuer_list = lambda *args, **kwargs: (frame, parse_rescuer(first))
        scanner._wait_page = lambda hwnd, kind, **kwargs: (frame, first)
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            selected, page, detail, screenshot, attempts, errors = scanner._retry_rescuer_entry(
                "scan", (0, 0, 1600, 900), 1, "亚瑟", 1)
        self.assertTrue(_name_compatible("亚瑟", selected))
        self.assertEqual(attempts, 2)
        self.assertEqual(len(errors), 1)

    def test_list_click_uses_right_avatar_hotspot(self):
        # Regression guard for the live failure: OCR text is around x=0.86,
        # while the clickable portrait is around x=0.94.
        self.assertEqual(_roster_click_percent({"center": [1370, 208]}, 1600, 900),
                         (0.94, 208 / 900))

    def test_single_rescuer_merge_preserves_other_records(self):
        root = tempfile.mkdtemp()
        store = QuickLookupStore(root)
        store.save_scan("rescuer", {
            "page": "rescuer", "character": {"value": "甲"},
            "roster": [{"name": "甲"}, {"name": "乙"}],
            "roster_details": [{"character": "甲", "verified": True,
                                "journey_initial": {"journey_initial_character": "甲"}},
                               {"character": "乙", "verified": True,
                                "journey_initial": {"journey_initial_character": "乙"}}],
        })
        store.merge_rescuer_detail(
            {"character": "乙", "verified": True,
             "journey_initial": {"journey_initial_character": "乙"},
             "screenshot": "logs/screenshots/乙.png"})
        state = store.read()
        self.assertEqual([item["character"] for item in state["rescuer"]["roster_details"]], ["甲", "乙"])
        self.assertEqual(state["rescuer"]["roster_details"][0]["character"], "甲")

    def test_full_rescuer_scan_is_incremental_and_keeps_prior_rows(self):
        root = tempfile.mkdtemp()
        store = QuickLookupStore(root)
        store.save_scan("rescuer", {
            "page": "rescuer", "character": {"value": "甲"},
            "roster": [{"name": "甲"}, {"name": "乙"}],
            "roster_details": [{"character": "甲", "verified": True,
                                "journey_initial": {"journey_initial_character": "甲"}},
                               {"character": "乙", "verified": True,
                                "journey_initial": {"journey_initial_character": "乙"}}],
            "roster_scan": {"complete": True, "attempted": 2},
        })
        store.save_scan("rescuer", {
            "page": "rescuer", "character": {"value": "乙"},
            "roster": [{"name": "乙"}, {"name": "丙"}],
            "roster_details": [{"character": "乙", "verified": True,
                                "journey_initial": {"journey_initial_character": "乙"},
                                "read_attempts": 3}],
            "roster_scan": {"complete": False, "attempted": 1,
                            "stop_reason": "游戏失去前台"},
        })
        state = store.read()
        rows = state["rescuer"]["roster_details"]
        self.assertEqual([item["character"] for item in rows], ["甲", "乙"])
        self.assertNotIn("read_attempts", rows[1])
        self.assertEqual([item["name"] for item in state["rescuer"]["roster"]], ["甲", "乙", "丙"])
        self.assertFalse(state["rescuer"]["roster_scan"]["complete"])
        self.assertEqual(state["rescuer"]["roster_scan"]["retained"], 1)

    def test_full_rescuer_scan_scrolls_and_discovers_new_rows(self):
        frame = type("Frame", (), {"shape": (900, 1600, 3)})()
        page_one = [
            ("救援者", .99, box(204, 42, 80, 32)),
            ("查看详情", .99, box(1370, 825, 86, 28)),
            ("角色甲", .99, box(1336, 194, 75, 28)),
            ("Lv.180", .99, box(1335, 170, 110, 23)),
            ("角色乙", .99, box(1336, 284, 75, 28)),
            ("Lv.179", .99, box(1335, 260, 110, 23)),
        ]
        page_two = [
            ("救援者", .99, box(204, 42, 80, 32)),
            ("查看详情", .99, box(1370, 825, 86, 28)),
            ("角色乙", .99, box(1336, 194, 75, 28)),
            ("Lv.179", .99, box(1335, 170, 110, 23)),
            ("角色丙", .99, box(1336, 284, 75, 28)),
            ("Lv.178", .99, box(1335, 260, 110, 23)),
        ]
        scanner = QuickLookupScanner(tempfile.mkdtemp(), sleep=lambda _: None)
        pages = iter([page_one, page_two, page_two, page_two, page_two, page_two])
        scanner._wait_page = lambda hwnd, kind, **kwargs: (frame, next(pages))
        clicked_scrolls = []
        scanner.controller.scroll_at_percent = lambda *args, **kwargs: clicked_scrolls.append(args)

        def fake_entry(scan_id, region, hwnd, name, slot, **kwargs):
            return name, {"character": {"value": name}, "roster": []}, {
                "verified": True, "character": name,
                "journey_initial": {"journey_initial_character": name},
                "can_continue": True,
            }, "logs/screenshots/row.png", 1, []

        scanner._retry_rescuer_entry = fake_entry
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            details, traversal = scanner._traverse_rescuers(
                "scan", (0, 0, 1600, 900), 1, frame, {}, max_rescuers=10)
        self.assertEqual([item["character"] for item in details], ["角色甲", "角色乙", "角色丙"])
        self.assertTrue(traversal["complete"])
        self.assertGreaterEqual(len(clicked_scrolls), 2)
        self.assertEqual([item["name"] for item in scanner.discovered_roster],
                         ["角色甲", "角色乙", "角色丙"])

    def test_arcanum_scan_is_incremental_and_keeps_prior_cards(self):
        root = tempfile.mkdtemp()
        store = QuickLookupStore(root)
        store.save_scan("arcanum", {
            "page": "arcanum",
            "cards": [{"name": {"value": "旧卡"}, "level": 40}],
            "screenshot": "old.png",
        })
        store.save_scan("arcanum", {
            "page": "arcanum",
            "cards": [{"name": {"value": "新卡"}, "level": 50}],
            "screenshot": "new.png",
        })
        state = store.read()
        self.assertEqual([card["name"] for card in state["arcanum"]["cards"]],
                         ["旧卡", "新卡"])
        self.assertEqual(state["arcanum"]["incremental"]["retained"], 1)

    def test_stop_callback_interrupts_unbounded_retry(self):
        frame = type("Frame", (), {"shape": (900, 1600, 3)})()
        scanner = QuickLookupScanner(tempfile.mkdtemp(), sleep=lambda _: None,
                                     should_stop=lambda: True)
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            with self.assertRaises(QuickLookupInterrupted):
                scanner._retry_rescuer_entry("scan", (0, 0, 1600, 900), 1, "甲", 1)

    def assert_identity_only(self, value):
        forbidden = {"level", "level_max", "stars", "star", "rarity", "power", "stats",
                     "potential", "effects", "collection", "ocr", "evidence", "candidates",
                     "screenshot", "detail_screenshot", "journey_initial_screenshot",
                     "retry_errors"}
        if isinstance(value, dict):
            self.assertFalse(set(value) & forbidden)
            for child in value.values():
                self.assert_identity_only(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_identity_only(child)

    def test_persistence_redacts_nested_fields_and_preserves_full_names(self):
        with tempfile.TemporaryDirectory() as root:
            store = QuickLookupStore(root)
            scan = {"character": {"value": "亚瑟菈", "evidence": [1]}, "title": "星光华尔兹",
                    "level": 180, "ocr": [{"text": "Lv.180"}],
                    "scan_id": "abcdef", "observed_at": "2026-10-10T12:00:00+08:00",
                    "roster_details": [{"character": "亚瑟菈", "title": "星光华尔兹",
                        "verified": True, "rarity": "SSR", "stars": 2,
                        "journey_initial": {"journey_initial_character": "亚瑟菈",
                            "stats": {"health": 180}, "ocr": [{"text": "SSR"}]}}]}
            store.save_scan("rescuer", scan)
            store.save_scan("arcanum", {"cards": [{"name": {"value": "令人满意的开花", "evidence": [1]},
                "level": 45, "stars": 3, "effects": {"training": 11.4}, "collection": {"owned": 52}}]})
            store.save_match(build_match(scan, store.read()["arcanum"], screenshots={"rescuer": "full.png"}))
            data = json.loads(store.path.read_text(encoding="utf-8"))
            self.assert_identity_only(data)
            self.assertEqual(data["rescuer"]["character"]["value"], "星光华尔兹：亚瑟菈")
            self.assertEqual(data["rescuer"]["roster_details"][0]["character"], "星光华尔兹：亚瑟菈")
            self.assertEqual(data["rescuer"]["scan_id"], "abcdef")
            self.assertEqual(data["arcanum"]["cards"][0]["name"], "令人满意的开花")
            before = store.path.read_bytes()
            self.assertEqual(store.read(), store.read())
            self.assertEqual(before, store.path.read_bytes())

    def test_sanitizer_is_idempotent_including_scan_metadata(self):
        scan = {"character": {"value": "亚瑟菈"}, "title": "星光华尔兹",
                "scan_id": "abc", "observed_at": "time",
                "roster_details": [{"character": "亚瑟菈", "title": "星光华尔兹",
                                     "journey_initial": {"journey_initial_character": "亚瑟菈"}}]}
        clean = sanitize_quick_lookup_scan("rescuer", scan)
        self.assertEqual(clean, sanitize_quick_lookup_scan("rescuer", clean))

    def test_unread_and_unverified_updates_do_not_erase_identity(self):
        with tempfile.TemporaryDirectory() as root:
            store = QuickLookupStore(root)
            picture = "logs/screenshots/quick_lookup_abc_journey_identity_1.png"
            store.merge_rescuer_detail({"character": "角色甲", "title": "称号甲", "verified": True,
                                       "journey_initial_portrait": picture})
            store.save_scan("rescuer", {"roster_details": [
                {"character": "角色甲", "verified": True, "journey_initial_portrait": None},
                {"character": "错误角色", "verified": False}]})
            details = store.read()["rescuer"]["roster_details"]
            self.assertEqual(len(details), 1)
            self.assertEqual(details[0]["character"], "称号甲：角色甲")
            self.assertEqual(details[0]["journey_initial_portrait"], picture)
            store.save_scan("arcanum", {"cards": [{"name": "卡甲", "image": picture}]})
            store.save_scan("arcanum", {"cards": [{"name": "卡甲", "image": None}]})
            self.assertEqual(store.read()["arcanum"]["cards"][0]["image"], picture)
            with self.assertRaises(ValueError):
                store.merge_rescuer_detail({"verified": True})

    def test_different_rescuer_titles_have_distinct_records(self):
        with tempfile.TemporaryDirectory() as root:
            store = QuickLookupStore(root)
            for title in ("称号甲", "称号乙", "称号甲"):
                store.merge_rescuer_detail({"character": "同名角色", "title": title,
                                           "verified": True})
            self.assertEqual([row["character"] for row in store.read()["rescuer"]["roster_details"]],
                             ["称号甲：同名角色", "称号乙：同名角色"])

    def test_corrupt_file_cannot_be_overwritten_by_a_scan(self):
        with tempfile.TemporaryDirectory() as root:
            store = QuickLookupStore(root)
            store.path.parent.mkdir(parents=True)
            store.path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                store.save_scan("rescuer", {"character": {"value": "角色甲"}})
            self.assertEqual(store.path.read_text(encoding="utf-8"), "{broken")

    def test_legacy_migration_recrops_images_and_preserves_other_logs(self):
        with tempfile.TemporaryDirectory() as root:
            store = QuickLookupStore(root)
            folder = Path(root) / "logs" / "screenshots"
            folder.mkdir(parents=True)
            frame = np.full((900, 1600, 3), 19, dtype=np.uint8)
            frame[330:395, 330:410] = 255  # old portrait's rarity/star overlay
            full = folder / "quick_lookup_abc_journey_initial_1.png"
            old_portrait = folder / "quick_lookup_abc_journey_initial_portrait_1.png"
            cv2.imwrite(str(full), frame)
            cv2.imwrite(str(old_portrait), frame[145:408, 170:560])
            unrelated = folder / "journey_log.png"
            cv2.imwrite(str(unrelated), frame)
            store.path.parent.mkdir(parents=True)
            store.path.write_text(json.dumps({"rescuer": {"roster_details": [{
                "character": "角色甲", "ocr": [("救援者", .99, box(200, 42)),
                    ("称号甲", .99, box(158, 144)), ("角色甲", .99, box(157, 172))],
                "journey_initial_portrait": "logs/screenshots/" + old_portrait.name,
                "journey_initial_screenshot": "logs/screenshots/" + full.name,
                "level": 180}]}}, ensure_ascii=False), encoding="utf-8")
            report = store.redact_existing()
            row = store.read()["rescuer"]["roster_details"][0]
            self.assertEqual(row["character"], "称号甲：角色甲")
            self.assertEqual(report["removed_captures"], 2)
            self.assertTrue(unrelated.exists())
            crop = cv2.imread(str(Path(root) / row["journey_initial_portrait"]))
            self.assertEqual(crop.shape[:2], (175, 295))
            self.assertTrue(np.all(crop == 19))
            self.assert_identity_only(json.loads(store.path.read_text(encoding="utf-8")))

    def test_arcanum_scanner_saves_only_art_in_confirmed_detail(self):
        rows = [("旅程效果", .99, box(570, 259)),
                ("旅程初始效果", .99, box(775, 385)),
                ("令人满意的开花", .99, box(745, 285, 240)),
                ("莉迪亚", .99, box(745, 253)), ("Lv.45", .99, box(315, 594))]
        with tempfile.TemporaryDirectory() as root:
            scanner = QuickLookupScanner(root)
            scanner.wait_for_foreground = lambda: ((0, 0, 1600, 900), 1)
            frame = np.full((900, 1600, 3), 17, dtype=np.uint8)
            frame[594:740, 200:500] = 255
            scanner._capture = lambda: ((0, 0, 1600, 900), 1, frame)
            scanner.ocr.recognize_detailed = lambda _: rows
            scan = scanner.scan("arcanum")
            self.assert_identity_only(scan)
            crop = cv2.imread(str(Path(root) / scan["cards"][0]["image"]))
            self.assertEqual(crop.shape[:2], (335, 255))
            self.assertTrue(np.all(crop == 17))
            self.assertEqual(len(list((Path(root) / "logs" / "screenshots").glob("*.png"))), 1)
            scanner.ocr.recognize_detailed = lambda _: [("阿尔克那", .99, box(100, 40))]
            with self.assertRaises(ValueError):
                scanner.scan("arcanum")
            self.assertEqual(len(list((Path(root) / "logs" / "screenshots").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
