import tempfile
import unittest
from pathlib import Path

from src.quick_lookup import build_match, parse_arcanum, parse_rescuer


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


if __name__ == "__main__":
    unittest.main()
