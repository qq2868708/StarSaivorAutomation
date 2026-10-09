import tempfile
import unittest
from pathlib import Path

from src.journey_info_scan import parse_journey_info
from src.journey_record import JourneyRecordStore
from src.logger import RunLogger


def box(x, y, w=40, h=22):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class JourneyInfoScanTests(unittest.TestCase):
    def test_parser_reads_labelled_values_and_leaves_cards_unknown(self):
        rows = [
            ("旅程信息", .99, box(500, 80)), ("继承星辰档案", .99, box(470, 500)),
            ("亚瑟菈", .95, box(195, 317)), ("RANK69", .99, box(200, 356)),
            ("力量", .99, box(80, 390)), ("222/1250", .99, box(80, 413)),
            ("体力", .99, box(245, 390)), ("1054/1250", .99, box(245, 413)),
            ("韧性", .99, box(80, 448)), ("598/1250", .99, box(80, 472)),
            ("专注", .99, box(245, 448)), ("149/1250", .99, box(245, 472)),
            ("保护", .99, box(80, 508)), ("178/1250", .99, box(80, 530)),
            ("潜质点数", .99, box(245, 509)), ("2573", .99, box(245, 531)),
            ("卡蜜", .99, box(497, 578)), ("芙蕾", .99, box(898, 578)),
            ("Lv.40", .99, box(542, 397)), ("SSR", .99, box(534, 414)),
        ]
        result = parse_journey_info(rows)
        self.assertEqual(result["actual_setup"]["character"], "亚瑟菈")
        self.assertEqual([item["name"] for item in result["actual_setup"]["journey_records"]], ["卡蜜", "芙蕾"])
        self.assertEqual(result["outcome"]["rank"], 69)
        self.assertEqual(result["outcome"]["final_stats"], {"力量": 222, "体力": 1054,
                         "韧性": 598, "专注": 149, "保护": 178})
        self.assertEqual(result["outcome"]["potential_points"], 2573)
        self.assertNotIn("support_cards", result["actual_setup"])
        self.assertEqual(result["support_card_slots"][0]["level"], 40)

    def test_screen_scan_merges_evidence_and_does_not_replace_manual_value(self):
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root=root)
            logger.save("journey_end", announce=False)
            store = JourneyRecordStore(root)
            journey_id = logger._journey["journey_id"]
            result = store.apply_screen_scan(journey_id, {
                "scan_id": "a" * 32, "observed_at": "2026-10-09T01:00:00+08:00",
                "screenshot": "logs/screenshots/journey_info_a_dialog.png",
                "actual_setup": {"character": "亚瑟菈", "journey_records": [{"name": "卡蜜", "slot": 1}]},
                "outcome": {"rank": 69, "final_stats": {"力量": 222}, "potential_points": 2573},
                "evidence": {"character": [1]},
                "ocr": [{"text": "亚瑟菈", "confidence": .95, "box": box(1, 1)}],
            })
            self.assertEqual(result["actual_setup"]["character"]["source"], "screen_confirmed")
            self.assertEqual(result["actual_setup"]["journey_records"]["value"][0]["name"], "卡蜜")
            self.assertEqual(result["outcome"]["rank"]["value"], 69)
            self.assertEqual(result["outcome"]["final_stats"]["力量"]["value"], 222)
            store.supplement(journey_id, {"actual_setup": {"character": "人工核对角色"}})
            result = store.apply_screen_scan(journey_id, {
                "scan_id": "b" * 32, "observed_at": "2026-10-09T02:00:00+08:00",
                "actual_setup": {"character": "错误识别"}, "outcome": {},
            })
            self.assertEqual(result["actual_setup"]["character"]["value"], "人工核对角色")


if __name__ == "__main__":
    unittest.main()
