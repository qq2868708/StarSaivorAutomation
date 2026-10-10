"""Full-pass denominator, bounded retries and safe navigation contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.quick_lookup import (QuickLookupScanner, QuickLookupEntryFailed,
                              QuickLookupStore, parse_rescuer, _join_roster_pages)


def box(x, y):
    return [[x, y], [x + 70, y], [x + 70, y + 22], [x, y + 22]]


def roster_rows(count=10):
    rows = [("救援者", .99, box(200, 42)), ("查看详情", .99, box(1370, 825))]
    for i in range(count):
        rows.extend([(f"角色{chr(0x7532+i)}", .99, box(1330, 192 + i * 48)),
                     ("Lv.180", .99, box(1300, 167 + i * 48))])
    return rows


class AcceptanceTests(unittest.TestCase):
    def scan(self, failed=(), limit=100, inventory=True):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        frame = np.zeros((900, 1600, 3), np.uint8)
        rows = roster_rows()
        page = parse_rescuer(rows)
        scanner = QuickLookupScanner(temp.name, controller=Mock(), sleep=lambda _: None)
        scanner.discovered_roster = list(page["roster"])
        scanner.inventory_complete = inventory
        scanner._wait_page = Mock(return_value=(frame, rows))
        names = [row["name"] for row in page["roster"]]
        attempted = []
        def entry(scan_id, region, hwnd, name, slot):
            attempted.append(name)
            if names.index(name) in failed:
                raise QuickLookupEntryFailed("Identity mismatch, recovered to list.")
            return name, {"title": None}, {"verified": True}, None, 1, []
        scanner._retry_rescuer_entry = entry
        with patch("src.quick_lookup.is_window_foreground", return_value=True):
            details, report = scanner._traverse_rescuers("scan", (0, 0, 1600, 900), 1,
                                                       frame, page, max_rescuers=limit)
        return details, report, attempted

    def test_nine_of_ten_meets_threshold_and_failed_role_does_not_block(self):
        details, report, attempted = self.scan(failed=(1,))
        self.assertEqual(len(attempted), 10)
        self.assertEqual(len(details), 9)
        self.assertEqual(report["total"], 10)
        self.assertEqual(report["attempted"], 10)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["success_rate"], .9)
        self.assertTrue(report["accepted"])

    def test_eight_of_ten_is_not_accepted(self):
        _, report, _ = self.scan(failed=(1, 2))
        self.assertTrue(report["complete"])
        self.assertFalse(report["accepted"])
        self.assertEqual(report["success_rate"], .8)

    def test_incomplete_inventory_or_capped_pass_cannot_pass(self):
        _, unknown, _ = self.scan(inventory=False)
        self.assertIsNone(unknown["success_rate"])
        self.assertFalse(unknown["accepted"])
        _, capped, _ = self.scan(limit=5)
        self.assertEqual(capped["total"], 10)
        self.assertFalse(capped["complete"])
        self.assertFalse(capped["accepted"])

    def test_previous_saved_roles_do_not_inflate_current_attempt_count(self):
        with tempfile.TemporaryDirectory() as root:
            store = QuickLookupStore(root)
            store.save_scan("rescuer", {"roster_details": [
                {"character": "甲", "verified": True},
                {"character": "乙", "verified": True}]})
            store.save_scan("rescuer", {"roster_details": [{"character": "丙", "verified": True}],
                                       "roster_scan": {"attempted": 1, "succeeded": 1, "total": 10,
                                                       "success_rate": .1, "accepted": False}})
            result = store.read()["rescuer"]
            self.assertEqual(len(result["roster_details"]), 3)
            self.assertEqual(result["roster_scan"]["attempted"], 1)
            self.assertEqual(result["roster_scan"]["success_rate"], .1)

    def test_unread_level_row_marks_census_incomplete(self):
        rows = roster_rows(2)
        page = parse_rescuer(rows)
        self.assertTrue(QuickLookupScanner._roster_page_complete(page))
        rows.append(("Lv.90", .99, box(1300, 400)))
        self.assertFalse(QuickLookupScanner._roster_page_complete(parse_rescuer(rows)))

    def test_partial_top_row_does_not_create_false_missing_anchor(self):
        rows = roster_rows(2) + [("额外角色", .99, box(1330, 155)),
                                ("Lv.99", .99, box(1300, 130))]
        self.assertTrue(QuickLookupScanner._roster_page_complete(parse_rescuer(rows)))

    def test_last_visible_bottom_row_is_included_in_denominator(self):
        from src.quick_lookup import _physical_roster
        rows = roster_rows(2) + [("贝拉", .99, box(1330, 765)),
                                ("Lv.1", .99, box(1330, 740))]
        physical = _physical_roster(parse_rescuer(rows))
        self.assertEqual(len(physical), 3)
        self.assertEqual(physical[-1]["name"], "贝拉")

    def test_highlighted_level_loss_does_not_remove_name_row_or_unknown_slot(self):
        from src.quick_lookup import _physical_roster
        rows = [("救援者", .99, box(200, 42)),
                ("甲角色", .99, box(1330, 194)), ("Lv.19", .99, box(1330, 167)),
                ("乙角色", .99, box(1330, 284)),
                ("丙角色", .99, box(1330, 374)), ("Lv.1", .99, box(1330, 347))]
        physical = _physical_roster(parse_rescuer(rows))
        self.assertEqual([r["name"] for r in physical], ["甲角色", "乙角色", "丙角色"])
        self.assertIsNone(physical[1]["level"])
        unknown = _physical_roster(parse_rescuer([r for r in rows if r[0] != "乙角色"]))
        self.assertEqual(len(unknown), 3)
        self.assertIsNone(unknown[1]["name"])

    def test_census_checks_both_directions_and_rejects_ignored_scroll(self):
        with tempfile.TemporaryDirectory() as root:
            first = roster_rows(3)
            renamed = {f"角色{chr(0x7532+i)}": f"角色{chr(0x7533+i)}" for i in range(3)}
            second = [(renamed.get(text, text), confidence, bounds)
                      for text, confidence, bounds in first]
            current = {"page": 0}
            controller = Mock()
            def scroll(*args, clicks, **kwargs):
                current["page"] = min(1, max(0, current["page"] + (1 if clicks < 0 else -1)))
            controller.scroll_at_percent.side_effect = scroll
            scanner = QuickLookupScanner(root, controller=controller, sleep=lambda _: None)
            scanner._wait_page = lambda *args, **kwargs: (None, [first, second][current["page"]])
            scanner._inventory_rescuers("scan", (0, 0, 1600, 900), 1)
            self.assertTrue(scanner.inventory_complete)
            self.assertEqual(len(scanner.discovered_roster), 4)
            self.assertEqual(current["page"], 0)
            scanner.inventory_complete = False
            controller.scroll_at_percent.side_effect = None
            with self.assertRaisesRegex(RuntimeError, "未观察到列表实际滚动"):
                scanner._inventory_rescuers("scan", (0, 0, 1600, 900), 1)
            self.assertFalse(scanner.inventory_complete)

    def test_ordered_inventory_preserves_same_names_and_unreadable_rows(self):
        def row(name, level=100):
            return {"name": name, "level": level}
        first = [row("甲角色"), row("乙角色"), row(None), row("丙角色")]
        second = [row("乙角色"), row(None), row("丙角色"),
                  row("甲角色", 90), row("丁角色")]
        joined = _join_roster_pages(first, second)
        self.assertEqual(len(joined), 6)
        self.assertEqual(sum(r["name"] == "甲角色" for r in joined), 2)
        self.assertEqual(sum(r["name"] is None for r in joined), 1)
        with self.assertRaises(RuntimeError):
            _join_roster_pages(first, [row("其他角色"), row("另一角色")])

    def test_single_character_names_and_level_comma_are_not_dropped(self):
        rows = [("救援者", .99, box(200, 42)), ("凌", .99, box(159, 172)),
                ("凌", .99, box(1335, 194)), ("Lv,19", .99, box(1335, 170))]
        page = parse_rescuer(rows)
        self.assertEqual(page["character"]["value"], "凌")
        self.assertEqual(page["roster"][0]["name"], "凌")
        self.assertTrue(QuickLookupScanner._roster_page_complete(page))

    def test_one_weak_glyph_can_align_a_page_but_does_not_verify_identity(self):
        from src.quick_lookup import _same_roster_rows, _name_compatible
        def row(name, confidence=.99):
            return {"name": name, "level": 160, "confidence": confidence}
        left = [row("克莱儿"), row("罪", .8), row("缪莉尔")]
        right = [row("克莱儿"), row("霏", .86), row("缪莉尔")]
        self.assertTrue(_same_roster_rows(left, right))
        self.assertFalse(_name_compatible("罪", "霏"))
        self.assertFalse(_same_roster_rows(left[1:], right[1:]))
        self.assertFalse(_same_roster_rows([row("甲"), row("乙"), row("丙")],
                                          [row("甲"), row("丁"), row("丙")]))

    def test_list_position_distinguishes_repeated_names_without_deleting_rows(self):
        from src.quick_lookup import _physical_roster
        with tempfile.TemporaryDirectory() as root:
            rows = roster_rows(3)
            rows += [("角色甲", .99, box(1330, 432)), ("Lv.90", .99, box(1300, 407))]
            page = parse_rescuer(rows)
            scanner = QuickLookupScanner(root, controller=Mock(), sleep=lambda _: None)
            scanner.discovered_roster = [{**entry, "row_id": index} for index, entry in
                                         enumerate(_physical_roster(page))]
            located = scanner._locate_roster_view(page)
            self.assertEqual([entry["row_id"] for entry in located], [0, 1, 2, 3])
            self.assertEqual([entry["name"] for entry in located].count("角色甲"), 2)

    def test_identity_failure_stops_after_three_attempts_on_confirmed_list(self):
        with tempfile.TemporaryDirectory() as root:
            scanner = QuickLookupScanner(root, controller=Mock(), sleep=lambda _: None)
            frame = np.zeros((900, 1600, 3), np.uint8)
            rows = roster_rows(2)
            scanner._read_frame = Mock(return_value=(frame, rows))
            scanner._wait_page = Mock(side_effect=RuntimeError("mismatch"))
            scanner._recover_to_rescuer_list = Mock(return_value=(frame, parse_rescuer(rows)))
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                with self.assertRaises(QuickLookupEntryFailed):
                    scanner._retry_rescuer_entry("scan", (0, 0, 1600, 900), 1, "角色甲", 1)
            self.assertEqual(scanner.retry_count, 3)
            self.assertEqual(scanner._recover_to_rescuer_list.call_count, 3)

    def test_navigation_retries_only_when_origin_is_still_confirmed(self):
        with tempfile.TemporaryDirectory() as root:
            scanner = QuickLookupScanner(root, controller=Mock(), sleep=lambda _: None)
            scanner._click_fixed = Mock()
            scanner._read_frame = Mock(return_value=(None, roster_rows(2)))
            scanner._wait_page = Mock(side_effect=[RuntimeError("ignored click"), (None, [])])
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                scanner._click_transition(1, (0, 0, 1600, 900), .889, .203,
                                          "test", "rescuer_list", "rescuer_detail")
            self.assertEqual(scanner._click_fixed.call_count, 2)
            scanner._click_fixed.reset_mock()
            scanner._read_frame.return_value = (None, [])
            scanner._wait_page.side_effect = RuntimeError("unknown")
            with patch("src.quick_lookup.is_window_foreground", return_value=True):
                with self.assertRaises(RuntimeError):
                    scanner._click_transition(1, (0, 0, 1600, 900), .889, .203,
                                              "test", "rescuer_list", "rescuer_detail")
            self.assertEqual(scanner._click_fixed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
