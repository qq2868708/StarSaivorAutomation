#!/usr/bin/env python3
"""Classify Starsavior items by game function type based on name patterns."""

import json
import re

# Map of item name patterns to (type, attr_name, value)
# Order matters - first match wins

RULES = [
    # === Vitality Potions ===
    (r"活力药水", "vitality_potion", "", 0),
    (r"能量饮料", "vitality_potion", "", 0),

    # === Training EXP - 力量 ===
    (r"力量训练秘籍", "training_exp", "力量", 0),
    (r"禁忌力量训练之书", "training_exp", "力量", 0),
    (r"力量特训证书", "training_exp", "力量", 0),
    (r"力量训练结业奖牌", "training_exp", "力量", 0),

    # === Training EXP - 体力/体能 ===
    (r"体力训练秘籍", "training_exp", "体力", 0),
    (r"禁忌体力训练之书", "training_exp", "体力", 0),
    (r"体能特训证书", "training_exp", "体力", 0),
    (r"体能训练结业奖牌", "training_exp", "体力", 0),

    # === Training EXP - 保护 ===
    (r"保护训练秘籍", "training_exp", "保护", 0),
    (r"禁忌保护训练之书", "training_exp", "保护", 0),

    # === Training EXP - 忍耐 ===
    (r"忍耐训练秘籍", "training_exp", "忍耐", 0),
    (r"禁忌忍耐训练之书", "training_exp", "忍耐", 0),

    # === Training EXP - 集中 ===
    (r"集中训练秘籍", "training_exp", "集中", 0),
    (r"禁忌集中训练之书", "training_exp", "集中", 0),

    # === Attribute Boost - 奖章/勋章 ===
    (r"力量奖章", "attribute_boost", "力量", 0),
    (r"洞察力奖章", "attribute_boost", "洞察力", 0),
    (r"技能奖章", "attribute_boost", "技能", 0),
    (r"英勇勋章", "attribute_boost", "英勇", 0),
    (r"智慧勋章", "attribute_boost", "智慧", 0),
    (r"宁静勋章", "attribute_boost", "宁静", 0),
    (r"星光勋章", "attribute_boost", "星光", 0),
    (r"星光功勋服务徽章", "attribute_boost", "星光", 0),
    (r"银色功勋徽章", "attribute_boost", "功勋", 0),
    (r"金质功绩奖章", "attribute_boost", "功绩", 0),

    # === Mood Recovery ===
    (r"月光润唇膏", "mood_recovery", "", 0),
    (r"月光精华", "mood_recovery", "", 0),
    (r"月光香水", "mood_recovery", "", 0),
    (r"温泉纪念毛巾", "mood_recovery", "", 0),
    (r"温泉度假村", "mood_recovery", "", 0),
    (r"暖手宝", "mood_recovery", "", 0),

    # === Status Buff ===
    (r"苍蓝之息", "status_buff", "", 0),
    (r"专注的凝视", "status_buff", "", 0),
    (r"智者之手", "status_buff", "", 0),
    (r"学习助手", "status_buff", "", 0),
    (r"宇宙视野", "status_buff", "", 0),
    (r"能量控制器", "status_buff", "", 0),

    # === Potential Points ===
    (r"洛洛卡的包裹", "potential_points", "", 0),
    (r"蕾塞特的抽奖券", "potential_points", "", 0),
    (r"蕾塞特的存钱罐", "potential_points", "", 0),
    (r"珍贵的日记本", "potential_points", "", 0),

    # === Stamina Recovery - 皇家套餐 ===
    (r"皇家肉炸套餐", "stamina_recovery", "", 0),
    (r"皇家水果沙拉套餐", "stamina_recovery", "", 0),
    (r"皇家牛肉意面套餐", "stamina_recovery", "", 0),
    (r"皇家蒸菜天妇套餐", "stamina_recovery", "", 0),
    (r"皇家奶油意面套餐", "stamina_recovery", "", 0),

    # === Stamina Recovery - 优质/美味/高级 ===
    (r"优质炸肉", "stamina_recovery", "", 0),
    (r"美味水果沙拉", "stamina_recovery", "", 0),
    (r"优质牛肉意面", "stamina_recovery", "", 0),
    (r"美味蔬菜沙拉", "stamina_recovery", "", 0),
    (r"高级蔬菜天妇罗", "stamina_recovery", "", 0),
    (r"高级奶油意面", "stamina_recovery", "", 0),

    # === Stamina Recovery - 普通食物 ===
    (r"炸肉", "stamina_recovery", "", 0),
    (r"牛排", "stamina_recovery", "", 0),
    (r"鸡排", "stamina_recovery", "", 0),
    (r"水果沙拉", "stamina_recovery", "", 0),
    (r"蔬菜沙拉", "stamina_recovery", "", 0),
    (r"蔬菜天妇罗", "stamina_recovery", "", 0),
    (r"牛肉意面", "stamina_recovery", "", 0),
    (r"奶油意面", "stamina_recovery", "", 0),
    (r"蓬松蛋糕", "stamina_recovery", "", 0),
    (r"普通蛋糕", "stamina_recovery", "", 0),
    (r"甜甜圈", "stamina_recovery", "", 0),
    (r"普通甜甜圈", "stamina_recovery", "", 0),
    (r"探险家肉炖菜", "stamina_recovery", "", 0),
    (r"探险家蔬菜炖菜", "stamina_recovery", "", 0),
    (r"莫里安特色饼干", "stamina_recovery", "", 0),
    (r"特制橙汁", "stamina_recovery", "", 0),
    (r"特制章鱼烧", "stamina_recovery", "", 0),
    (r"特制芭菲", "stamina_recovery", "", 0),
    (r"特制蜂蜜糖", "stamina_recovery", "", 0),
    (r"深渊炸鱼", "stamina_recovery", "", 0),
    (r"女巫汤", "stamina_recovery", "", 0),
    (r"南瓜之夜", "stamina_recovery", "", 0),
    (r"街头烘焙食品", "stamina_recovery", "", 0),
    (r"街头冰淇淋", "stamina_recovery", "", 0),
    (r"早晨的咖啡", "stamina_recovery", "", 0),
    (r"街头小贩邦吉奥邦", "stamina_recovery", "", 0),
    (r"鲜牛奶", "stamina_recovery", "", 0),
    (r"纯牛奶", "stamina_recovery", "", 0),
    (r"草莓", "stamina_recovery", "", 0),
    (r"苹果", "stamina_recovery", "", 0),
    (r"弗洛拉西瓜", "stamina_recovery", "", 0),
    (r"弗洛拉橙", "stamina_recovery", "", 0),
    (r"奇迹柑橘", "stamina_recovery", "", 0),
    (r"综合营养补充剂", "stamina_recovery", "", 0),
    (r"街头面包店", "stamina_recovery", "", 0),
    (r"Cafe Strega优惠券", "stamina_recovery", "", 0),
    (r"特辑海报", "stamina_recovery", "", 0),

    # === Random Effect (gacha/random) ===
    (r"可疑盒子", "random_effect", "", 0),
    (r"可疑的药瓶", "random_effect", "", 0),
    (r"被诅咒的香炉", "random_effect", "", 0),
    (r"诅咒荆棘棒", "random_effect", "", 0),
    (r"古老沙漏", "random_effect", "", 0),
    (r"有缺陷的扑克牌", "random_effect", "", 0),
    (r"魔法瓶", "random_effect", "", 0),
    (r"来历不明的遗物", "random_effect", "", 0),

    # === Status Buff - more items ===
    (r"狼之契约", "status_buff", "", 0),
    (r"逃亡者斗篷", "status_buff", "", 0),
    (r"冰封吊坠", "status_buff", "", 0),
    (r"火种玻璃瓶", "status_buff", "", 0),
    (r"有毒鳞片", "status_buff", "", 0),
    (r"活字印剧", "status_buff", "", 0),
    (r"满溢的爱", "status_buff", "", 0),
    (r"便携式风扇", "status_buff", "", 0),
    (r"德鲁伊竖琴", "status_buff", "", 0),
    (r"磁带录音机", "status_buff", "", 0),
    (r"温柔的呜鸣祖拉", "status_buff", "", 0),
    (r"可爱的呜呜祖拉", "status_buff", "", 0),
    (r"温柔的呜呜祖拉", "status_buff", "", 0),
    (r"回响的祝福", "status_buff", "", 0),
    (r"不容错过的机会", "status_buff", "", 0),
    (r"债务人的策略", "status_buff", "", 0),
    (r"满意度调查", "status_buff", "", 0),
    (r"购物狂的推荐", "status_buff", "", 0),
    (r"投资推荐传单", "status_buff", "", 0),
    (r"害羞", "status_buff", "", 0),
    (r"号角", "status_buff", "", 0),
    (r"送货员的号角", "status_buff", "", 0),
    (r"最后一弹", "status_buff", "", 0),
    (r"玩偶服装制作套件", "status_buff", "", 0),
    (r"闪亮手部喷雾器9号", "status_buff", "", 0),
    (r"手工兔子头箍", "status_buff", "", 0),
    (r"兔子绅士", "status_buff", "", 0),
    (r"马形石", "status_buff", "", 0),
    (r"毛绒绒的朋友", "status_buff", "", 0),
    (r"蓝色蝴蝶", "status_buff", "", 0),
    (r"幸运木制骰子套装", "status_buff", "", 0),
    (r"K.I.杰出员工奖杯", "status_buff", "", 0),
    (r"佩特拉工具 Ver.00", "status_buff", "", 0),
    (r"霓虹特工钥匙扣", "status_buff", "", 0),
    (r"压铸机车模型", "status_buff", "", 0),
    (r"坚固的卡钳", "status_buff", "", 0),
    (r"蘸墨笔系列", "status_buff", "", 0),
    (r"烦恼的布谷鸟钟", "status_buff", "", 0),
    (r"烦人的布谷鸟钟", "status_buff", "", 0),
    (r"眼睛闪闪发光", "status_buff", "", 0),
    (r"观赏性吸血植物", "status_buff", "", 0),
    (r"饥饿小丑面具", "status_buff", "", 0),
    (r"石像之花", "status_buff", "", 0),
    (r"钢铁之花", "status_buff", "", 0),
    (r"空笼子", "status_buff", "", 0),
    (r"空药瓶", "status_buff", "", 0),
    (r"星光游侠面具", "status_buff", "", 0),
    (r"邪恶巨龙刻耳柏洛斯模型", "status_buff", "", 0),
    (r"孩童的英雄", "status_buff", "", 0),
    (r"星际守护者的旅程", "status_buff", "", 0),
    (r"寒尔凡德 噩梦 EX", "status_buff", "", 0),
    (r"Maison Delièr", "status_buff", "", 0),
    (r"新手女佣指南", "status_buff", "", 0),
    (r"城镇卫队标准手铐", "status_buff", "", 0),
    (r"特尔米尔萨安全腰带", "status_buff", "", 0),
    (r"早鸟", "status_buff", "", 0),
    (r"永不从战斗中撤退", "status_buff", "", 0),

    # === Equipment (consumable fallback - low priority) ===
    (r"装饰玻璃剑", "consumable", "", 0),
    (r"舞动的锯片", "consumable", "", 0),
    (r"石剑", "consumable", "", 0),
    (r"老式猎人步枪", "consumable", "", 0),
    (r"剧毒守望者", "consumable", "", 0),
    (r"狙击手外套", "consumable", "", 0),
    (r"伤痕收割镰", "consumable", "", 0),
    (r"古董钥匙", "consumable", "", 0),
    (r"红舞鞋", "consumable", "", 0),
    (r"坚毅决斗手套", "consumable", "", 0),
    (r"永久旋转装置", "consumable", "", 0),
    (r"用于安全目的的自动装置", "consumable", "", 0),
    (r"灰烬王座", "consumable", "", 0),
    (r"平衡的天平", "consumable", "", 0),
    (r"荣誉纪念碑", "consumable", "", 0),
    (r"传说之剑复制品", "consumable", "", 0),
    (r"骑士的荣耀", "consumable", "", 0),
    (r"骑士的荣誉", "consumable", "", 0),
    (r"骑士救援急救包", "consumable", "", 0),
    (r"沙滩排球二等奖", "consumable", "", 0),
    (r"沙滩排球锦标奖杯", "consumable", "", 0),
    (r"危机逃脱玩具", "consumable", "", 0),
    (r"轨道搜索队徽章", "consumable", "", 0),
    # Currency / coins
    (r"之币", "consumable", "", 0),
    # Equipment sets
    (r"人偶遗留的", "consumable", "", 0),
    (r"史莱姆附着的", "consumable", "", 0),
    (r"启示录新型", "consumable", "", 0),
    (r"远古救世主的", "consumable", "", 0),
    # Exchange tickets
    (r"领取预约单表", "consumable", "", 0),
    # Badges/medals (non-stat)
    (r"强盗男爵讨伐徽章", "consumable", "", 0),
    (r"史莱姆讨伐徽章", "consumable", "", 0),
    (r"人偶讨伐徽章", "consumable", "", 0),
]


def classify(name):
    """Return (type, attr_name, value) for an item name."""
    for pattern, item_type, attr_name, value in RULES:
        if re.search(pattern, name):
            return item_type, attr_name, value
    # Fallback
    return "consumable", "", 0


def main():
    path = "e:/Starsavior/profiles/shop/speed.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {}
    for item in data["items"]:
        name = item["name"]
        new_type, attr_name, value = classify(name)

        old_type = item.get("type", "consumable")
        item["type"] = new_type
        if attr_name:
            item["attr_name"] = attr_name
        elif old_type != new_type and not attr_name:
            # Clear attr_name when type changes and no specific attr
            if item.get("attr_name") == name:
                item["attr_name"] = ""
        item["value"] = value

        # Build keywords from name
        keywords = [name]
        # Add variant without spaces
        compact = name.replace(" ", "").replace(".", "").replace("·", "")
        if compact != name:
            keywords.append(compact)

        # Trim OCR-unfriendly characters for fuzzy matching
        base = name.rstrip("．\"'ⅠⅢ")
        if base != name:
            keywords.append(base)

        item["keywords"] = keywords

        # Track stats
        stats[new_type] = stats.get(new_type, 0) + 1

        if old_type != new_type:
            print(f"  {old_type} → {new_type}: {name}")

    # Update comment
    total = len(data["items"])
    data["_comment"] = (
        f"Starsavior 全物品库 (已分类, 共{total}个) - "
        + ", ".join(f"{k}:{v}" for k, v in sorted(stats.items()))
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n=== 分类完成 ===")
    print(f"总计: {total} 个物品")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
