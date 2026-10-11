"""Position-checked commission observations and deterministic battle policy."""
import math
import re

import numpy as np


TIERS = ("高阶", "中阶", "低阶")


def normalized_rows(items, width, height):
    rows = []
    for item in items:
        if len(item) < 3 or item[2] is None:
            continue
        text, confidence, box = item[:3]
        points = np.asarray(box, dtype=float)
        if (points.shape != (4, 2) or not np.isfinite(points).all()
                or not math.isfinite(float(confidence)) or confidence < .80):
            continue
        x, y = points.mean(axis=0) / [width, height]
        if not (0 <= x <= 1 and 0 <= y <= 1):
            continue
        rows.append({"text": re.sub(r"\s+", "", text), "confidence": float(confidence),
                     "x": float(x), "y": float(y), "box": points.tolist()})
    return rows


def inside(row, region):
    x, y, width, height = region
    return x <= row["x"] <= x + width and y <= row["y"] <= y + height


def unique_rank(rows, region):
    values = []
    for row in rows:
        match = re.fullmatch(r"RANK(\d{1,2})", row["text"], re.IGNORECASE)
        if inside(row, region) and match and 1 <= int(match[1]) <= 99:
            values.append(int(match[1]))
    return values[0] if len(values) == 1 else None


def read_panel(items, width, height):
    rows = normalized_rows(items, width, height)
    card = [row for row in rows if inside(row, (.49, .24, .26, .59))]
    titles = [row["text"] for row in card if "讨伐委托" in row["text"] and row["y"] < .33]
    tiers = [tier for tier in TIERS if any(row["text"] == tier + "委托"
                                         and row["y"] < .37 for row in card)]
    tier_positions = {tier: (row["x"], row["y"]) for tier in TIERS for row in rows
                      if row["text"] == tier + "委托" and inside(row, (.78, .30, .20, .38))}
    emergency_titles = [row for row in rows if "紧急讨伐委托" in row["text"]
                        and inside(row, (.78, .30, .20, .38))]
    single_tier_emergency = (
        len(titles) == 1 and len(tiers) == 1 and len(emergency_titles) == 1
        and "紧急讨伐委托" in titles[0]
        and emergency_titles[0]["text"] == titles[0]
        and set(tier_positions) == set(tiers))
    return {
        "title": titles[0] if len(titles) == 1 else None,
        "tier": tiers[0] if len(tiers) == 1 else None,
        "my_rank": unique_rank(rows, (.035, .27, .15, .10)),
        "recommended": unique_rank(rows, (.50, .57, .22, .09)),
        "card_rows": card,
        "tier_positions": tier_positions,
        "single_tier_emergency": single_tier_emergency,
        "accept_positions": [(row["x"], row["y"]) for row in rows
                             if row["text"] == "接受" and inside(row, (.77, .82, .20, .12))],
    }


def read_battle_prompt(items, width, height):
    rows = [row for row in normalized_rows(items, width, height)
            if inside(row, (.20, .20, .60, .60))]
    skip = [row for row in rows if row["text"] == "跳过战斗"
            and inside(row, (.30, .65, .22, .14))]
    battle = [row for row in rows if row["text"] in {"开始委托", "开始战斗"}
              and inside(row, (.50, .65, .20, .14))]
    questions = [row for row in rows if "是否" in row["text"] and
                 any(word in row["text"] for word in ("委托", "战斗", "评鉴"))]
    if len(skip) != 1 or len(battle) != 1 or not questions:
        return None
    recommendations = set()
    for row in rows:
        if row["text"] in {"建议战斗", "推荐战斗"}:
            recommendations.add("battle")
        elif row["text"] in {"建议跳过", "推荐跳过", "推荐跳过战斗"}:
            recommendations.add("skip")
        elif row["text"] == "推荐" and .55 <= row["y"] <= .77:
            distances = [(abs(row["x"] - candidate["x"]), mode)
                         for candidate, mode in ((skip[0], "skip"), (battle[0], "battle"))]
            distances.sort()
            if distances[0][0] < .08 and distances[1][0] - distances[0][0] > .05:
                recommendations.add(distances[0][1])
    titles = [row["text"] for row in rows if row["y"] < .38 and "讨伐委托" in row["text"]]
    levels = [int(match[1]) for row in rows
              if inside(row, (.38, .53, .22, .06))
              and (match := re.fullmatch(r"(?:RANK|建议综合等级)?(\d{1,2})", row["text"], re.IGNORECASE))]
    return {"commission": bool(titles), "title": titles[0] if len(titles) == 1 else None,
            "recommended": levels[0] if len(levels) == 1 else None,
            "skip_position": (skip[0]["x"], skip[0]["y"]),
            "battle_position": (battle[0]["x"], battle[0]["y"]),
            "recommendation": next(iter(recommendations)) if len(recommendations) == 1 else None,
            "recommendation_ambiguous": len(recommendations) > 1,
            "rows": rows}


def battle_mode(my_rank, recommended, max_gap=4):
    if my_rank is None or recommended is None:
        return None
    gap = recommended - my_rank
    if gap > min(4, max_gap):
        return None
    return "skip" if gap <= 2 else "battle"


def same_commission(left, right):
    # Roman tier suffixes are often read as punctuation. The name and exact
    # recommended rank are independent, visible identifiers on both pages.
    return bool(left and right and "讨伐委托" in left and "讨伐委托" in right
                and left.partition("讨伐委托")[0] == right.partition("讨伐委托")[0])


def select_commission(candidates, max_gap=4):
    eligible = [row for row in candidates if row.get("verified") and
                battle_mode(row.get("my_rank"), row.get("recommended"), max_gap)]
    return max(eligible, key=lambda row: (row["recommended"], -TIERS.index(row["tier"]))) if eligible else None
