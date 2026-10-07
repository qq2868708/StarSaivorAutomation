"""
失败率 OCR 模块 - 移植自 SleepRunner 的 TrainingFailRateOcr

读取训练页每行的红色"失败率"百分比数值。

策略:
  1. 通过红色 HSV 范围定位"失败率"标签的 Y 坐标
  2. 在标签附近的多个区域做 OCR
  3. 对 OCR 结果打分投票选出最可信的百分比值

失败回退值: 100 (保守策略 - 宁可休息也不误点训练)
"""
from __future__ import annotations

import re
import cv2
import numpy as np

# 选项 Y 边界 (用于从红色标记 Y 推断当前展开的行)
# 从上到下 5 行的分界线 (百分比)
_OPTION_Y_BOUNDARIES = [0.33, 0.43, 0.53, 0.63]

# "失败率"红色标记搜索区域 (百分比)
_FAIL_RATE_SEARCH_X = 0.74
_FAIL_RATE_SEARCH_Y = 0.20
_FAIL_RATE_SEARCH_W = 0.24
_FAIL_RATE_SEARCH_H = 0.56

# 失败率识别失败时的保守值
UNKNOWN_FAIL_RATE_FALLBACK = 100


def detect_selected_option(screenshot: np.ndarray) -> int:
    """
    通过红色"失败率"标记的 Y 坐标判断当前展开了哪个训练行。

    Args:
        screenshot: 训练页截屏 (BGR numpy array)

    Returns:
        行索引 (0-4)，未识别返回 -1
    """
    result = _find_fail_rate_marker_by_color(screenshot)
    if result is None:
        return -1

    marker_center, red_ratio = result
    y_pct = marker_center[1] / screenshot.shape[0]

    for i, boundary in enumerate(_OPTION_Y_BOUNDARIES):
        if y_pct < boundary:
            return i
    return 4


def read_fail_rate_percent(screenshot: np.ndarray, selected_index: int = 0) -> int:
    """
    OCR 识别当前展开行的失败率百分比值。

    Args:
        screenshot: 训练行截屏 (BGR numpy array)
        selected_index: 当前展开的行索引 (用于回退定位)

    Returns:
        失败率百分比 (0-100)
    """
    if screenshot is None or screenshot.size == 0:
        return UNKNOWN_FAIL_RATE_FALLBACK

    h, w = screenshot.shape[:2]

    # Step 1: 用红色标记定位区域
    regions = []
    result = _find_fail_rate_marker_by_color(screenshot)
    if result is not None:
        marker_center, red_ratio = result
        match_x_pct = marker_center[0] / w
        match_y_pct = marker_center[1] / h
        # 红色标记附近的三个扩展区域
        regions.append((max(0, match_x_pct - 0.03), max(0, match_y_pct - 0.025), 0.10, 0.05))
        regions.append((max(0, match_x_pct - 0.06), max(0, match_y_pct - 0.03), 0.12, 0.06))
        regions.append((max(0, match_x_pct - 0.08), max(0, match_y_pct - 0.035), 0.14, 0.07))

    # Step 2: 固定位置回退区域 (基于选中行)
    row_y = 0.28 + selected_index * 0.10  # 近似估算每行 Y 位置
    regions.append((0.80, max(0, row_y - 0.03), 0.16, 0.07))
    regions.append((0.76, max(0, row_y - 0.04), 0.20, 0.09))

    # Step 3: OCR 并投票
    candidates = []
    candidate_freq = {}
    candidate_score = {}
    saw_fail_word = False

    for region_idx, (rx, ry, rw, rh) in enumerate(regions):
        # 裁剪区域
        px = max(0, int(w * rx))
        py = max(0, int(h * ry))
        pw = max(1, min(int(w * rw), w - px))
        ph = max(1, min(int(h * rh), h - py))

        if pw <= 1 or ph <= 1:
            continue

        roi = screenshot[py:py + ph, px:px + pw]
        texts = _recognize_fail_rate_variants(roi)

        for text in texts:
            has_fail_word = "败率" in text or "失败率" in text
            has_percent = "%" in text or "％" in text
            has_lv = "lv" in text.lower()
            saw_fail_word = saw_fail_word or has_fail_word

            parsed = _parse_fail_rate_candidates(text)
            for p in parsed:
                candidates.append(p)
                candidate_freq[p] = candidate_freq.get(p, 0) + 1

                score = 1
                if has_fail_word:
                    score += 4
                if has_percent:
                    score += 3
                if re.search(rf"(失败率|败率|率)\D*{p}\s*[％%]?", text, re.IGNORECASE):
                    score += 3
                if has_lv and not has_fail_word:
                    score -= 2
                if p == 0 and re.search(r"0{3,}", text):
                    score -= 4
                if p == 0 and has_lv and not has_percent and not has_fail_word:
                    score -= 2

                candidate_score[p] = candidate_score.get(p, 0) + score

        # 早退: 前 2 个区域已扫完 + 高置信候选
        if region_idx >= 1 and saw_fail_word and candidate_score:
            top_val = max(candidate_score, key=candidate_score.get)
            top_score = candidate_score[top_val]
            top_freq = candidate_freq.get(top_val, 0)
            if top_score >= 10 and top_freq >= 2:
                break

    if not candidates:
        return UNKNOWN_FAIL_RATE_FALLBACK

    # 选最优候选
    best = max(candidate_score, key=lambda k: (
        candidate_score[k],
        candidate_freq.get(k, 0),
        k,
    ))

    # 0% 低置信度检查 (高分单票也信任, 避免因没有跑OCR变体而误拦截)
    zero_freq = candidate_freq.get(0, 0)
    zero_score = candidate_score.get(0, 0)
    if best == 0 and zero_freq < 2 and zero_score < 10 and saw_fail_word:
        return UNKNOWN_FAIL_RATE_FALLBACK

    return best


def _find_fail_rate_marker_by_color(screenshot: np.ndarray) -> tuple[tuple[int, int], float] | None:
    """通过 HSV 红色范围定位"失败率"标签"""
    if screenshot is None or screenshot.size == 0:
        return None

    h, w = screenshot.shape[:2]
    sx = max(0, int(w * _FAIL_RATE_SEARCH_X))
    sy = max(0, int(h * _FAIL_RATE_SEARCH_Y))
    sw = max(1, min(int(w * _FAIL_RATE_SEARCH_W), w - sx))
    sh = max(1, min(int(h * _FAIL_RATE_SEARCH_H), h - sy))

    if sw <= 1 or sh <= 1:
        return None

    roi = screenshot[sy:sy + sh, sx:sx + sw]
    if roi.size == 0:
        return None

    # 转 HSV
    if roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 红色范围 (HSV: 0-12 + 168-180)
    low_red = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(low_red, high_red)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    red_pixels = cv2.countNonZero(mask)
    red_ratio = red_pixels / max(1, sw * sh)
    if red_pixels < 24:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_rect = None
    best_area = 0
    for contour in contours:
        rect = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 16 or rect[2] < 4 or rect[3] < 4:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    if best_area <= 0 or best_rect is None:
        return None

    marker_center = (
        sx + best_rect[0] + best_rect[2] // 2,
        sy + best_rect[1] + best_rect[3] // 2,
    )
    return (marker_center, red_ratio)


def _recognize_fail_rate_variants(roi: np.ndarray) -> list[str]:
    """对 ROI 做多种预处理的 OCR (不使用 PaddleOCR，只用 OpenCV 做区域特征提取)"""
    # 注意: 实际的 OCR 文本识别仍由 PaddleOCR 完成
    # 这里只返回空列表，让调用方通过 PaddleOCR 来做
    return []


def _parse_fail_rate_candidates(text: str) -> list[int]:
    """从 OCR 文本中提取失败率候选数字"""
    results = []
    if not text:
        return results

    normalized = text
    normalized = normalized.replace('O', '0').replace('o', '0').replace('〇', '0')
    normalized = normalized.replace('８', '8').replace('Ｂ', '8').replace('B', '8')
    normalized = normalized.replace('帙', '8').replace('捌', '8')
    normalized = normalized.replace('Ｓ', '5').replace('S', '5')

    for m in re.finditer(r"(\d{1,3})", normalized):
        val = int(m.group(1))
        if 0 <= val <= 100:
            results.append(val)
    return results


def parse_fail_rate_from_ocr_texts(ocr_texts: list[str]) -> int:
    """
    从 PaddleOCR 输出的文本列表中解析失败率。
    这是 Python 版本的集成点 - 传入 OCR 识别的所有文本。

    Args:
        ocr_texts: OCR 识别的文本列表 (如 ["力量训练", "失败率", "25%", "Lv.3"])

    Returns:
        失败率百分比 (0-100), 无法识别返回 UNKNOWN_FAIL_RATE_FALLBACK
    """
    candidates = []
    candidate_freq = {}
    candidate_score = {}
    saw_fail_word = False

    for text in ocr_texts:
        if not text:
            continue

        text = text.replace(" ", "").replace("\r", "").replace("\n", "").replace("　", "").strip()
        has_fail_word = "败率" in text or "失败率" in text
        has_percent = "%" in text or "％" in text
        has_lv = "lv" in text.lower()
        saw_fail_word = saw_fail_word or has_fail_word

        parsed = _parse_fail_rate_candidates(text)
        for p in parsed:
            candidates.append(p)
            candidate_freq[p] = candidate_freq.get(p, 0) + 1

            score = 1
            if has_fail_word:
                score += 4
            if has_percent:
                score += 3
            if re.search(rf"(失败率|败率|率)\D*{p}\s*[％%]?", text, re.IGNORECASE):
                score += 3
            if has_lv and not has_fail_word:
                score -= 2
            if p == 0 and re.search(r"0{3,}", text):
                score -= 4
            if p == 0 and has_lv and not has_percent and not has_fail_word:
                score -= 2

            candidate_score[p] = candidate_score.get(p, 0) + score

    if not candidates:
        return UNKNOWN_FAIL_RATE_FALLBACK

    best = max(candidate_score, key=lambda k: (
        candidate_score[k],
        candidate_freq.get(k, 0),
        k,
    ))

    zero_freq = candidate_freq.get(0, 0)
    zero_score = candidate_score.get(0, 0)
    # 0% 低置信度检查 (高分单票也信任, 避免因没有跑OCR变体而误拦截)
    if best == 0 and zero_freq < 2 and zero_score < 10 and saw_fail_word:
        return UNKNOWN_FAIL_RATE_FALLBACK

    return best
