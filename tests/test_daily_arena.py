"""Daily boundaries, spending acknowledgements, and both arena stop conditions."""
import unittest

import numpy as np

from src.daily_arena import ArenaDaily, OPPONENTS, read_arena


def observation(page, remaining=6, **extra):
    result = {"page": page, "remaining": remaining,
              "buttons": {name: (.5, .5) for name in
                          ("圣铠", "NPC", "老手", "跳过战斗", "确认", "点击以继续")},
              "npc_selected": True, "senior_selected": True,
              "opponents": [(name, .28 + index * .13)
                            for index, name in enumerate(sorted(OPPONENTS))],
              "challenges": [], "cooldowns": []}
    result.update(extra)
    return result


def ready():
    planner = ArenaDaily()
    for page in ("home", "menu", "arena_hub"):
        assert planner.step(observation(page))["kind"] == "click"
    return planner


def start_battle(planner, remaining=6):
    target = sorted(OPPONENTS)[0]
    planner.step(observation("arena_list", remaining,
                             challenges=[{"name": target, "point": (.87, .3)}]))
    planner.step(observation("arena_prepare", remaining, prepared_opponent=target))
    return planner.step(observation("arena_confirm"))


class ArenaDailyTests(unittest.TestCase):
    def test_skip_confirmation_rejects_other_costs(self):
        image = np.zeros((900, 1600, 3), np.uint8)
        for cost, expected in [("确定要消耗1个圣铠钥匙", "arena_confirm"),
                               ("确定要消耗2个圣铠钥匙", "unsupported_dialog"),
                               ("确定要消耗1个圣铠钥匙和100金币", "unsupported_dialog")]:
            items = [(text, .99, [[400, 300], [500, 300], [500, 330], [400, 330]])
                     for text in ("跳过战斗", cost, "跳过该场战斗吗？", "确认", "取消")]
            self.assertEqual(read_arena(image, items)["page"], expected)

    def test_player_list_can_route_to_npc_without_senior_tab_visible(self):
        image = np.zeros((900, 1600, 3), np.uint8)
        items = [(text, .99, [[700, 100], [800, 100], [800, 130], [700, 130]])
                 for text in ("策略战", "NPC", "对战列表", "防御编制")]
        result = read_arena(image, items)
        self.assertEqual(result["page"], "arena_list")
        self.assertFalse(result["npc_selected"])

    def test_requires_home_start(self):
        self.assertEqual(ArenaDaily().step(observation("arena_list"))["kind"], "pause")

    def test_zero_attempts_never_opens_strategy_battle(self):
        planner = ArenaDaily()
        planner.step(observation("home"))
        planner.step(observation("menu"))
        action = planner.step(observation("arena_hub", 0))
        self.assertEqual(action["name"], "返回")
        self.assertEqual(planner.stop_reason, "attempts_exhausted")
        self.assertEqual(planner.step(observation("home"))["kind"], "complete")

    def test_zero_attempts_precedes_available_challenge(self):
        planner = ready()
        action = planner.step(observation("arena_list", 0,
                               challenges=[{"name": "老手骑士团", "point": (.87, .3)}]))
        self.assertEqual(action["name"], "返回")

    def test_missing_count_is_not_zero_and_never_spends(self):
        planner = ready()
        action = planner.step(observation("arena_list", None,
                               challenges=[{"name": "老手骑士团", "point": (.87, .3)}]))
        self.assertEqual(action["kind"], "wait")
        self.assertIsNone(planner.stop_reason)

    def test_wrong_tab_is_selected_before_challenge(self):
        planner = ready()
        self.assertEqual(planner.step(observation("arena_list", npc_selected=False))["name"], "NPC")
        self.assertEqual(planner.step(observation("arena_list", senior_selected=False))["name"], "老手")

    def test_no_opponents_requires_scanning_and_cooldown_evidence(self):
        planner = ready()
        state = observation("arena_list", 1)
        state["cooldowns"] = [(.87, y) for _, y in state["opponents"]]
        self.assertEqual(planner.step(state)["kind"], "scroll")
        self.assertEqual(planner.step(state)["kind"], "scroll")
        self.assertEqual(planner.step(state)["name"], "返回")
        self.assertEqual(planner.stop_reason, "no_challengeable_opponents")
        self.assertEqual(planner.step(observation("arena_hub", 1))["name"], "返回")
        self.assertEqual(planner.step(observation("home"))["kind"], "complete")

    def test_empty_or_partial_ocr_never_means_all_opponents_exhausted(self):
        for opponents in ([], [("老手骑士团", .3)]):
            planner = ready()
            for _ in range(11):
                action = planner.step(observation("arena_list", 1, opponents=opponents))
            self.assertEqual(action["kind"], "pause")
            self.assertIsNone(planner.stop_reason)

    def test_confirmation_sent_once_while_server_shows_stale_list(self):
        planner = ready()
        self.assertEqual(start_battle(planner)["kind"], "click")
        for page in ("arena_confirm", "arena_list", "arena_prepare", "unknown"):
            self.assertEqual(planner.step(observation(page))["kind"], "wait")
        self.assertEqual(planner.confirmed, 0)

    def test_actual_result_and_matching_deduction_are_required(self):
        planner = ready()
        start_battle(planner)
        self.assertEqual(planner.step(observation("arena_result", outcome="victory"))["name"], "关闭结算")
        self.assertEqual(planner.confirmed, 1)
        self.assertEqual(planner.victories, 1)
        self.assertEqual(planner.step(observation("arena_list", 6))["kind"], "pause")

    def test_last_attempt_returns_home_only_after_result_is_closed(self):
        planner = ready()
        start_battle(planner, 1)
        planner.step(observation("arena_result", outcome="victory"))
        self.assertEqual(planner.step(observation("arena_list", 0))["name"], "返回")
        self.assertEqual(planner.step(observation("arena_hub", 0))["name"], "返回")
        result = planner.step(observation("home"))
        self.assertEqual(result["reason"], "attempts_exhausted")
        self.assertTrue(result["returned_home"])

    def test_prepared_opponent_must_match_selected_npc(self):
        planner = ready()
        planner.step(observation("arena_list", challenges=[{"name": "老手骑士团", "point": (.87, .3)}]))
        action = planner.step(observation("arena_prepare", prepared_opponent="老手先锋队"))
        self.assertEqual(action["kind"], "pause")

    def test_unrecognized_dialog_is_never_confirmed(self):
        planner = ready()
        self.assertEqual(planner.step(observation("unsupported_dialog"))["kind"], "pause")

    def test_occluded_count_and_wrong_fraction_are_not_parsed_as_zero(self):
        image = np.zeros((900, 1600, 3), np.uint8)
        def word(text, x, y):
            return (text, .99, [[x, y], [x + 60, y], [x + 60, y + 20], [x, y + 20]])
        for items in ([], [word("0/45", 990, 50)], [word("0/6", 100, 500)],
                      [word("0/6", 990, 50), word("1/6", 1100, 50)]):
            self.assertIsNone(read_arena(image, items)["remaining"])
        self.assertEqual(read_arena(image, [word("0/6", 990, 50)])["remaining"], 0)


if __name__ == "__main__":
    unittest.main()
