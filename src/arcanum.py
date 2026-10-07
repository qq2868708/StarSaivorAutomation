"""
阿尔克那 (Arcanum) 支援卡检测与选择模块

在训练第一年，优先点击与优先训练属性匹配的支援卡角色，
将羁绊(bond)提升至黄色以触发闪光训练。

检测策略:
  1. OCR 读取支援卡附近的属性名称文字来识别卡片类型
  2. HSV 颜色检测羁绊进度条是否达到黄色
  3. 根据优先属性决定点击哪张卡
"""
from __future__ import annotations

import time
import re
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np


# ============================================================
# 属性映射
# ============================================================

# 属性名称关键词 → 属性ID
ATTRIBUTE_KEYWORD_TO_ID: Dict[str, int] = {
    "力量": 1,
    "生命": 2,
    "体力": 2,
    "韧性": 3,
    "命中": 4,
    "保护": 5,
}

# 属性ID → 图标HSV颜色特征 (用于颜色检测回退)
# 各属性图标的主色调范围 (H: 0-179, S: 0-255, V: 0-255)
ATTRIBUTE_ICON_COLORS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    1: (np.array([0, 40, 60]), np.array([10, 255, 255])),      # 力量: 红色 (降低S/V阈值)
    2: (np.array([0, 20, 100]), np.array([25, 150, 255])),     # 生命: 心形 (橙/粉/白)
    3: (np.array([30, 50, 40]), np.array([80, 255, 200])),     # 韧性: 绿/黄绿
    4: (np.array([100, 50, 40]), np.array([130, 255, 200])),   # 命中: 蓝色 (V阈值降低)
    5: (np.array([140, 40, 30]), np.array([175, 255, 200])),   # 保护: 紫/品红
}

# ============================================================
# 羁绊进度条 HSV 检测
# ============================================================

# 黄色阈值 (羁绊达到黄色 = 可触发闪光训练)
YELLOW_HSV_LOWER = np.array([18, 80, 100])
YELLOW_HSV_UPPER = np.array([38, 255, 255])

# 进度条区域颜色判定: 黄色像素占比超过此值认为已达标
BOND_YELLOW_RATIO = 0.35


# ============================================================
# Arcanum 卡片检测
# ============================================================

class ArcanumDetector:
    """
    阿尔克那支援卡检测器。

    在训练选择界面检测支援卡角色头像、左上角属性图标和下方羁绊进度条。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 完整配置字典，读取 arcanum 和 training 部分
        """
        ac = config.get("arcanum", {})
        tc = config.get("training", {})

        # 支援卡区域 (相对于游戏窗口的百分比坐标)
        # 默认: 左侧 0-80% 宽度, 下方 70%-92% 高度 (底部一排卡片)
        self.card_row_y = ac.get("card_row_y", 0.70)
        self.card_row_h = ac.get("card_row_h", 0.22)

        # 单张卡片的宽度占比 (5张卡平分左侧区域)
        self.card_count = ac.get("card_count", 5)
        self.card_area_x = ac.get("card_area_x", 0.02)
        self.card_area_w = ac.get("card_area_w", 0.76)

        # 每张卡片内: 左上角图标的相对位置 (相对于单张卡片)
        self.icon_x_offset = ac.get("icon_x_offset", 0.05)
        self.icon_y_offset = ac.get("icon_y_offset", 0.02)
        self.icon_w = ac.get("icon_w", 0.30)
        self.icon_h = ac.get("icon_h", 0.18)

        # 每张卡片内: 属性名称文字区域 (图标旁边)
        self.label_x_offset = ac.get("label_x_offset", 0.05)
        self.label_y_offset = ac.get("label_y_offset", 0.02)
        self.label_w = ac.get("label_w", 0.70)
        self.label_h = ac.get("label_h", 0.18)

        # 每张卡片内: 羁绊进度条区域 (头像下方)
        self.bond_bar_y_offset = ac.get("bond_bar_y_offset", 0.82)
        self.bond_bar_h = ac.get("bond_bar_h", 0.08)
        self.bond_bar_x_offset = ac.get("bond_bar_x_offset", 0.10)
        self.bond_bar_w = ac.get("bond_bar_w", 0.80)

        # 点击位置: 卡片中心偏上 (角色头像位置)
        self.click_y_offset = ac.get("click_y_offset", 0.45)

        # 第一年轮数 (与 rest 配置共用)
        self.first_year_rounds = config.get("rest", {}).get("first_year_rounds", 12)

        # 优先属性ID (从 training 配置读取)
        self.priority_attr_id = tc.get("priority", 1)

    # ===================== 主接口 =====================

    def is_first_year(self, round_count: int) -> bool:
        """判断是否处于第一年"""
        if self.first_year_rounds <= 0:
            return False
        return round_count <= self.first_year_rounds

    def get_target_card(self, screenshot: np.ndarray, ocr_engine,
                        priority_attr_id: int) -> Optional[int]:
        """
        扫描所有支援卡，返回应优先点击的卡片索引 (0-based)。

        优先选择左上角图标与 priority_attr_id 匹配的卡片。
        如果该卡片的羁绊进度条已变黄则返回 None (无需再点)。
        如果没有匹配的卡片也返回 None。

        Args:
            screenshot: 训练选择界面的完整截图 (BGR)
            ocr_engine: OCR 引擎
            priority_attr_id: 优先训练属性ID (1-5)

        Returns:
            卡片索引 (0-4)，或 None 表示无需点击支援卡
        """
        h, w = screenshot.shape[:2]

        cards = self._detect_cards(screenshot, ocr_engine, h, w)
        if not cards:
            return None

        # 筛选属性匹配的卡片
        matching = [c for c in cards if c["attr_id"] == priority_attr_id]

        if not matching:
            return None

        # 检查羁绊是否已黄
        for card in matching:
            if not card["bond_yellow"]:
                return card["index"]

        # 所有匹配卡的羁绊都已黄
        return None

    def get_card_click_pos(self, card_index: int) -> Tuple[float, float]:
        """
        获取卡片的点击百分比坐标。

        Args:
            card_index: 卡片索引 (0-4)

        Returns:
            (x_pct, y_pct) 百分比坐标
        """
        card_w = self.card_area_w / self.card_count
        card_center_x = self.card_area_x + card_w * (card_index + 0.5)
        card_center_y = self.card_row_y + self.card_row_h * self.click_y_offset
        return (card_center_x, card_center_y)

    def has_bond_reached_yellow(self, screenshot: np.ndarray,
                                card_index: int) -> bool:
        """检查指定卡片的羁绊进度条是否已达到黄色"""
        h, w = screenshot.shape[:2]
        bar_roi = self._get_bond_bar_roi(screenshot, card_index, h, w)
        if bar_roi is None or bar_roi.size == 0:
            return False
        return self._detect_yellow_bar(bar_roi)

    # ===================== 卡片检测 =====================

    def _detect_cards(self, screenshot: np.ndarray, ocr_engine,
                      img_h: int, img_w: int) -> List[dict]:
        """检测所有支援卡，返回每张卡的信息列表"""
        cards = []
        card_w = self.card_area_w / self.card_count

        for i in range(self.card_count):
            card_left = self.card_area_x + card_w * i
            card_top = self.card_row_y

            # OCR 读取卡片上的属性名称文字
            attr_id = self._read_card_attribute(
                screenshot, ocr_engine, img_h, img_w,
                card_left, card_top, card_w
            )

            # 读取羁绊进度条
            bond_yellow = False
            bar_roi = self._get_bond_bar_roi_idx(screenshot, card_left, card_top,
                                                  card_w, img_h, img_w)
            if bar_roi is not None and bar_roi.size > 0:
                bond_yellow = self._detect_yellow_bar(bar_roi)

            cards.append({
                "index": i,
                "attr_id": attr_id,
                "bond_yellow": bond_yellow,
            })

        return cards

    def _read_card_attribute(self, screenshot, ocr_engine,
                              img_h, img_w, card_left, card_top,
                              card_w) -> Optional[int]:
        """OCR 读取卡片上的属性名称，返回属性ID"""
        # 策略1: 读取左上角文字区域 (图标旁边的属性名)
        label_x = int(img_w * (card_left + card_w * self.label_x_offset))
        label_y = int(img_h * (card_top + self.card_row_h * self.label_y_offset))
        label_w_px = max(1, int(img_w * card_w * self.label_w))
        label_h_px = max(1, int(img_h * self.card_row_h * self.label_h))

        if label_x + label_w_px <= img_w and label_y + label_h_px <= img_h:
            roi = screenshot[label_y:label_y + label_h_px, label_x:label_x + label_w_px]
            items = ocr_engine.recognize(roi)
            for text, _, _ in items:
                attr_id = self._match_attribute_text(text)
                if attr_id is not None:
                    return attr_id

        # 策略2: HSV 颜色检测左上角图标
        icon_x = int(img_w * (card_left + card_w * self.icon_x_offset))
        icon_y = int(img_h * (card_top + self.card_row_h * self.icon_y_offset))
        icon_w_px = max(1, int(img_w * card_w * self.icon_w))
        icon_h_px = max(1, int(img_h * self.card_row_h * self.icon_h))

        if icon_x + icon_w_px <= img_w and icon_y + icon_h_px <= img_h:
            icon_roi = screenshot[icon_y:icon_y + icon_h_px, icon_x:icon_x + icon_w_px]
            return self._detect_icon_by_color(icon_roi)

        return None

    def _match_attribute_text(self, text: str) -> Optional[int]:
        """从 OCR 文字匹配属性名称"""
        if not text:
            return None
        clean = text.replace(" ", "").replace("\r", "").replace("\n", "").strip()
        for keyword, attr_id in ATTRIBUTE_KEYWORD_TO_ID.items():
            if keyword in clean:
                return attr_id
        return None

    def _detect_icon_by_color(self, roi: np.ndarray) -> Optional[int]:
        """通过 HSV 颜色检测图标类型"""
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        best_id = None
        best_score = 0.0

        for attr_id, (lower, upper) in ATTRIBUTE_ICON_COLORS.items():
            if attr_id == 1:
                # 力量 (红色): H 需要 wrap-around (0-10 和 160-180)
                mask1 = cv2.inRange(hsv, lower, upper)
                mask2 = cv2.inRange(hsv,
                                    np.array([160, lower[1], lower[2]]),
                                    np.array([180, upper[1], upper[2]]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, lower, upper)

            ratio = np.count_nonzero(mask) / max(mask.size, 1)
            # 权重: 饱和度越高越可信
            if ratio > 0.03:
                mean_s = np.mean(hsv[:, :, 1][mask > 0]) if np.count_nonzero(mask) > 0 else 0
                score = ratio * (1.0 + min(mean_s / 255.0, 0.5))
                if score > best_score:
                    best_score = score
                    best_id = attr_id

        return best_id

    # ===================== 进度条检测 =====================

    def _get_bond_bar_roi(self, screenshot, card_index, img_h, img_w):
        """获取指定卡片索引的羁绊进度条区域"""
        card_w = self.card_area_w / self.card_count
        card_left = self.card_area_x + card_w * card_index
        card_top = self.card_row_y
        return self._get_bond_bar_roi_idx(screenshot, card_left, card_top,
                                           card_w, img_h, img_w)

    def _get_bond_bar_roi_idx(self, screenshot, card_left, card_top,
                                card_w, img_h, img_w):
        """获取羁绊进度条 ROI"""
        bar_x = int(img_w * (card_left + card_w * self.bond_bar_x_offset))
        bar_y = int(img_h * (card_top + self.card_row_h * self.bond_bar_y_offset))
        bar_w = max(1, int(img_w * card_w * self.bond_bar_w))
        bar_h = max(1, int(img_h * self.card_row_h * self.bond_bar_h))

        if (bar_x >= 0 and bar_y >= 0 and
            bar_x + bar_w <= img_w and bar_y + bar_h <= img_h):
            return screenshot[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w]
        return None

    @staticmethod
    def _detect_yellow_bar(bar_roi: np.ndarray) -> bool:
        """检测进度条中黄色像素占比"""
        if bar_roi.size == 0:
            return False

        hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

        # 膨胀操作连接黄色区域
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)

        yellow_ratio = np.count_nonzero(dilated) / max(mask.size, 1)
        return yellow_ratio >= BOND_YELLOW_RATIO


# ============================================================
# 校准 / 调试工具
# ============================================================

def calibrate_arcanum(screenshot: np.ndarray, config: dict,
                      ocr_engine, save_dir: str = "templates") -> dict:
    """
    校准模式: 对当前截图进行全区域扫描，输出诊断信息。

    用于调整 arcanum 配置中的坐标参数。

    Returns:
        诊断结果字典
    """
    detector = ArcanumDetector(config)
    h, w = screenshot.shape[:2]

    result = {
        "image_size": (w, h),
        "cards": [],
    }

    card_w = detector.card_area_w / detector.card_count

    for i in range(detector.card_count):
        card_left = detector.card_area_x + card_w * i
        card_top = detector.card_row_y

        # 保存卡片区域截图
        cx = int(w * card_left)
        cy = int(h * card_top)
        cw = int(w * card_w)
        ch = int(h * detector.card_row_h)

        card_info = {
            "index": i,
            "region": (cx, cy, cw, ch),
            "attribute": None,
            "bond_yellow": False,
        }

        if cx + cw <= w and cy + ch <= h:
            card_roi = screenshot[cy:cy + ch, cx:cx + cw]

            # OCR 全卡片文字
            items = ocr_engine.recognize(card_roi)
            texts = [t[0] for t in items]
            card_info["ocr_texts"] = texts[:20]

            # 属性匹配
            for text in texts:
                clean = text.replace(" ", "").replace("\r", "").replace("\n", "").strip()
                for kw, aid in ATTRIBUTE_KEYWORD_TO_ID.items():
                    if kw in clean:
                        card_info["attribute"] = kw
                        card_info["attr_id"] = aid
                        break
                if card_info["attribute"]:
                    break

            # 进度条 HSV
            bar_roi = detector._get_bond_bar_roi_idx(
                screenshot, card_left, card_top, card_w, h, w)
            if bar_roi is not None and bar_roi.size > 0:
                card_info["bond_yellow"] = detector._detect_yellow_bar(bar_roi)

            # 保存卡片截图用于调试
            cv2.imwrite(f"{save_dir}/_arcanum_card{i}.png", card_roi)

        result["cards"].append(card_info)

    return result
