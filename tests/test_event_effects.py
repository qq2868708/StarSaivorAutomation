import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.event_effects import quantify_effect_text, choose_option
from src.handlers import EventHandler


class EventEffectTests(unittest.TestCase):
    def test_local_evidence_is_development_only(self):
        event = {'document_verified': False, 'source_type': 'screen_observed',
                 'source_evidence': 'frame.png', 'development_decision': 'only_enabled'}
        self.assertFalse(EventHandler._event_source_allowed(event, 'document', False))
        self.assertTrue(EventHandler._event_source_allowed(event, 'document', True))
        event.pop('source_evidence')
        self.assertFalse(EventHandler._event_source_allowed(event, 'document', True))

    def test_sheet_prefix_numbers_are_quantified(self):
        result = quantify_effect_text("+10 力量，+8 体力，+12 专注，+5 耐力")
        self.assertEqual(result["stats"], {"力量": 10, "体力": 8, "专注": 12})
        self.assertEqual(result["stamina_delta"], 5)

    def test_potential_points_and_travel_status_are_preserved(self):
        result = quantify_effect_text("潜力+10；旅行状态：疲惫不堪（5回合）")
        self.assertEqual(result["potential_points"], 10)
        self.assertEqual(result["unknown_effects"], ["疲惫不堪"])

    def test_base_event_currency_and_relic_are_quantified(self):
        result = quantify_effect_text("心情+1；硬币+30；获得遗物：传说中的仿制剑")
        self.assertEqual(result["currency_delta"], 30)
        self.assertEqual(result["relic_names"], ["传说中的仿制剑"])

    def test_option_choice_uses_numeric_effects(self):
        event = {"options": [
            {"index": 1, "effect_text": "力量+5、体力+5"},
            {"index": 2, "effect_text": "力量+15、体力+15"},
        ]}
        selected, scored = choose_option(event, "attack")
        self.assertEqual(selected, 2)
        self.assertGreater(scored[1]["score"], scored[0]["score"])

    def test_duplicate_title_resolves_to_card_visible_in_marker(self):
        candidates = [
            {"event_name": "传说之剑", "card": "文档堆上的责任感-萝贝塔",
             "options": [{"index": 1, "effect_text": "体力+20"}]},
            {"event_name": "传说之剑", "card": "慵懒午后的空闲（限定）-婚纱卡密",
             "options": [{"index": 1, "effect_text": "体力+5"},
                         {"index": 2, "effect_text": "潜质点数+15"}]},
        ]
        result = EventHandler._resolve_document_match(
            candidates, "支援卡事件 传说之剑 婚纱卡密", "attack")
        self.assertIs(result, candidates[1])

    def test_support_card_effects_have_verified_source_and_scores(self):
        index = Path(__file__).resolve().parents[1] / "profiles" / "events" / "document_verified.json"
        import json
        data = json.loads(index.read_text(encoding="utf-8"))
        event = next(item for item in data["events"] if item.get("source_cell") == "O7:P8")
        self.assertTrue(event["document_verified"])
        self.assertEqual(len(event["options"]), 2)
        self.assertEqual(event["options"][1]["quantified"]["stats"]["力量"], 15)
        self.assertGreater(event["options"][1]["score_attack"], event["options"][0]["score_attack"])


if __name__ == "__main__":
    unittest.main()
