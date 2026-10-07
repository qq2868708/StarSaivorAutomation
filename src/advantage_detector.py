"""训练行「优势训练」标记检测。

游戏会在部分训练行右侧显示蓝色大拇指标记。它不是支援卡圆形图标，
也不是闪光训练按钮，因此单独在右侧窄区域检测，避免和人物立绘混淆。
"""
from __future__ import annotations

import cv2
import numpy as np


# 训练行中心位置与 trainer._TRAINING_ROWS 保持一致。
_ROW_Y = (0.28, 0.38, 0.48, 0.58, 0.68)

# 1600x900 实测：蓝色大拇指标记中心在对应训练行中心上方约 20px，
# x 约为 0.92。使用相对坐标可适应同一 16:9 客户区的缩放。
_ROI_X = (0.90, 0.96)
_ROI_Y_OFFSET = (-0.055, -0.005)

# 蓝色图标的 HSV 范围。文字/背景的蓝色不会落入这个窄区域，
# 实测图标占 ROI 约 17%~22%，无图标为 0%。
_BLUE_LOWER = np.array([90, 80, 80], dtype=np.uint8)
_BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)
_MIN_COMPONENT_AREA = 120
_MIN_BLUE_RATIO = 0.035


def detect_advantage_training(screenshot: np.ndarray) -> list[bool]:
    """返回五行是否显示蓝色大拇指标记。

    识别失败时返回五个 ``False``，调用方应把识别结果作为观测值记录，
    不应因为没有识别到标记而停止训练。
    """
    result = [False] * 5
    if screenshot is None or not isinstance(screenshot, np.ndarray) or screenshot.ndim != 3:
        return result

    height, width = screenshot.shape[:2]
    if height <= 0 or width <= 0:
        return result

    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    x1 = max(0, int(width * _ROI_X[0]))
    x2 = min(width, int(width * _ROI_X[1]))
    if x2 <= x1:
        return result

    for row_index, row_y in enumerate(_ROW_Y):
        y1 = max(0, int(height * (row_y + _ROI_Y_OFFSET[0])))
        y2 = min(height, int(height * (row_y + _ROI_Y_OFFSET[1])))
        if y2 <= y1:
            continue

        roi = hsv[y1:y2, x1:x2]
        mask = cv2.inRange(roi, _BLUE_LOWER, _BLUE_UPPER)
        # 去掉孤立抗锯齿像素，同时保留 30px 左右的图标主体。
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)

        component_ok = False
        if ratio >= _MIN_BLUE_RATIO:
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            component_ok = any(int(stat[cv2.CC_STAT_AREA]) >= _MIN_COMPONENT_AREA
                               for stat in stats[1:])
        result[row_index] = component_ok

    return result

