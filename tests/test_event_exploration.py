import tempfile
import unittest

from src.event_exploration import EventExplorationStore, aggregate_effects, receipt_effects


class EventExplorationTests(unittest.TestCase):
    def test_receipt_parser_rejects_training_preview_and_keeps_effects_unknown(self):
        self.assertEqual(receipt_effects({"page": "训练决策", "ocr": [
            {"text": "力量训练 +20", "confidence": .99},
        ]})["numeric"], {})
        effect = receipt_effects({"page": "属性结算", "ocr": [
            {"text": "力量+20", "confidence": .99},
            {"text": "获得潜能：攻击技巧", "confidence": .99},
        ]})
        self.assertEqual(effect["numeric"], {"力量": 20})
        self.assertEqual(effect["qualitative"], ["获得潜能：攻击技巧"])

    def test_unknown_event_rotates_unseen_enabled_branches(self):
        with tempfile.TemporaryDirectory() as root:
            store = EventExplorationStore(root)
            rows = [{"text": "选项甲", "enabled": True, "y": .58},
                    {"text": "选项乙", "enabled": True, "y": .66},
                    {"text": "锁定", "enabled": False, "y": .73}]
            first = store.select("旅程事件测试", rows, {}, "attack")
            store.put_attempt({"key": first["key"], "index": first["index"], "id": "a",
                               "phase": "observed", "effects": {"numeric": {"力量": 10}}})
            second = store.select("旅程事件测试", rows, {}, "attack")
            self.assertEqual((first["index"], second["index"]), (1, 2))
            self.assertFalse(second["branch_coverage_complete"])

    def test_after_two_samples_per_branch_uses_empirical_direction_score(self):
        with tempfile.TemporaryDirectory() as root:
            store = EventExplorationStore(root)
            rows = [{"text": "甲", "enabled": True, "y": .58},
                    {"text": "乙", "enabled": True, "y": .66}]
            first = store.select("事件", rows, {}, "survival")
            for index in (1, 2):
                for sample in (1, 2):
                    store.put_attempt({"key": first["key"], "index": index,
                                       "id": f"{index}-{sample}", "phase": "observed",
                                       "effects": {"numeric": {"体力": 20 if index == 1 else 1}}})
            result = store.select("事件", rows, {}, "survival")
            self.assertEqual(result["mode"], "empirical")
            self.assertEqual(result["index"], 1)
            self.assertTrue(result["branch_coverage_complete"])
            self.assertTrue(result["effects_ready"])

    def test_qualitative_only_branches_converge_without_numeric_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            store = EventExplorationStore(root)
            rows = [{"text": "获得增益", "enabled": True, "y": .58},
                    {"text": "获得道具", "enabled": True, "y": .66}]
            first = store.select("未知文字效果", rows, {"target_revision": "r1"}, "survival")
            for index in (1, 2):
                for sample in (1, 2):
                    store.put_attempt({"key": first["key"], "index": index,
                                       "id": f"q-{index}-{sample}", "phase": "observed",
                                       "effects": {"numeric": {}, "qualitative": ["效果"],
                                                    "observed": True}})
            result = store.select("未知文字效果", rows, {"target_revision": "r1"}, "survival")
            self.assertEqual(result["mode"], "empirical_qualitative")
            self.assertEqual(result["index"], 1)
            self.assertTrue(result["effects_ready"])


if __name__ == "__main__":
    unittest.main()
