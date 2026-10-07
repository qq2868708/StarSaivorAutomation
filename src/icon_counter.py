"""
图标计数模块 - 移植自 SleepRunner 的 TrainingIconCounter

通过分析训练选项右侧固定位置的像素 HSV 值来计数圆形图标。
图标表示训练的质量/收益 (参与训练的支援卡角色数), 每行最多 8 个图标。

原理:
  - 彩色图标: satMean > 40, valMean > 70, satStd > 25 (带额外细分阈值)
  - 灰色/高亮立绘: valMean > 135, valStd > 30
  - 从第一个空 slot 开始停止计数

注: SleepRunner 没有闪光训练检测, 图标数量本身就是训练质量的指标。
"""
from __future__ import annotations

import cv2
import numpy as np

# 图标检测区域 (训练行右侧的圆形图标区域, 相对于截图百分比)
ICON_CENTER_X = 0.73
ICON_START_Y = 0.13
ICON_SPACING = 0.08
MAX_ICON_SLOTS = 8
ICON_CHECK_RADIUS = 0.015

# 调试开关
DEBUG_DUMP_ENABLED = False


def count_circular_icons(screenshot: np.ndarray, row_label: str = "") -> int:
    """
    统计单行训练选项右侧的圆形图标数

    Args:
        screenshot: 单行截屏 (BGR numpy array)
        row_label: 可选的训练行名 (仅用于日志)

    Returns:
        图标数量 (0-8)
    """
    count = 0
    h, w = screenshot.shape[:2]

    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    check_size = max(1, int(w * ICON_CHECK_RADIUS))

    for slot in range(MAX_ICON_SLOTS):
        slot_y = ICON_START_Y + slot * ICON_SPACING
        cx = int(w * ICON_CENTER_X)
        cy = int(h * slot_y)

        x1 = max(0, cx - check_size)
        y1 = max(0, cy - check_size)
        x2 = min(w, cx + check_size)
        y2 = min(h, cy + check_size)

        if x2 <= x1 or y2 <= y1:
            break

        region = hsv[y1:y2, x1:x2]
        mean, stddev = cv2.meanStdDev(region)
        sat_mean = mean[1][0]
        val_mean = mean[2][0]
        sat_std = stddev[1][0]
        val_std = stddev[2][0]

        # 彩色图标检测
        color_icon = sat_mean > 40 and val_mean > 70 and sat_std > 25
        if color_icon:
            if sat_mean < 75:
                color_icon = sat_std > 35
            else:
                color_icon = sat_std > 40

        # 灰色/高亮立绘图标检测
        gray_icon = val_mean > 135 and val_std > 30

        has_icon = color_icon or gray_icon

        if has_icon:
            count += 1
        else:
            break

    return count


def apply_priority_rule(counts: list[int], build_direction: str = "attack") -> int:
    """
    图标优先级决策 (移植自 SleepRunner TrainingIconCounter.ApplyPriorityRule)

    集中/保护 ≥4 时择优；否则前三项取最多者；
    并列时按基调: attack → 力量/体力/韧性, survival → 韧性/体力/力量

    Args:
        counts: 5 行的图标数 [力量, 体力, 韧性, 集中, 保护]
        build_direction: "attack" 或 "survival"

    Returns:
        选中的行索引 (0-4)
    """
    # 集中(3)/保护(4) 任一 ≥4 → 择优
    if counts[3] >= 4 or counts[4] >= 4:
        return 3 if counts[3] >= counts[4] else 4

    # 前三项中取最多
    best_count = max(counts[0], counts[1], counts[2])
    tied = [i for i in range(3) if counts[i] == best_count]

    if len(tied) == 1:
        return tied[0]

    # 并列时按基调偏好
    preference = [0, 1, 2] if build_direction == "attack" else [2, 1, 0]
    for idx in preference:
        if idx in tied:
            return idx
    return tied[0]
