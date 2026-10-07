"""
图像识别模块 - 移植自 SleepRunner 的屏幕检测 + 状态读取

使用相对坐标 (百分比) 进行所有定位，无需像素坐标校准。
屏幕检测通过 OCR 右侧分支区域文字来判断当前界面。
"""
import re
import time
import cv2
import numpy as np
from pathlib import Path

from .capture import ScreenCapture
from .light_ocr import LightOCR, DirectMLOCR


# ============================================================
# OCR 引擎 (PaddleOCR)
# ============================================================

class OCREngine:
    """Compatibility OCR interface backed by bundled RapidOCR CPU models."""

    def __init__(self):
        self._ocr = None
        self._gpu_available = False
        self._checked_gpu = False

    def _init_ocr(self):
        self._ocr = DirectMLOCR()

    @property
    def ocr(self):
        if self._ocr is None:
            self._init_ocr()
        return self._ocr

    def recognize(self, image):
        """
        对图片进行 OCR 识别。
        image: numpy array (BGR)
        Returns: [(text, confidence, bbox), ...]
        """
        if image is None or image.size == 0:
            return []
        results = self.ocr.ocr(image, cls=False)
        if results is None or results[0] is None:
            return []
        items = []
        for line in results[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            items.append((text, confidence, bbox))
        return items

    def recognize_region(self, screenshot, x_pct, y_pct, w_pct, h_pct):
        """识别截图指定百分比区域内的文字，返回拼接文本"""
        h, w = screenshot.shape[:2]
        px = max(0, int(w * x_pct))
        py = max(0, int(h * y_pct))
        pw = max(1, min(int(w * w_pct), w - px))
        ph = max(1, min(int(h * h_pct), h - py))
        roi = screenshot[py:py + ph, px:px + pw]
        items = self.recognize(roi)
        texts = [t[0] for t in items]
        return " ".join(texts)


# ============================================================
# HybridOCR — 混合 OCR: WinOCR 快速检测 + DirectMLOCR 精准识别
# ============================================================

class HybridOCR:
    """
    本地补齐的 OCR 引擎 (RapidOCR / ONNX Runtime CPU):
      - recognize()           → LightOCR (返回文本与坐标)
        用于绝大多数场景: handler 检测, 状态读取, 训练指标, 支援卡检测
      - recognize_region()    → LightOCR, 百分比坐标便捷方法
      - recognize_detailed()  → DirectMLOCR 兼容接口 (CPU)
        用于需要 bbox/置信度的场景: 全屏校准, 详细识别
      - paddle                → 底层 DirectMLOCR 实例 (向后兼容)
    """

    def __init__(self):
        self._light = LightOCR()
        self._paddle = None

    @property
    def paddle(self):
        """Lazy CPU OCR adapter, retaining the upstream property name."""
        if self._paddle is None:
            print("[OCR] 已启用 RapidOCR 中文模型 / ONNX Runtime CPU")
            self._paddle = DirectMLOCR()
        return self._paddle

    def recognize(self, image):
        """
        全图/区域 OCR — 通过 WinOCR (快速, 即时可用).

        返回与 PaddleOCR 兼容的 [(text, confidence, bbox), ...] 格式,
        其中 bbox 为 None (WinOCR 不返回坐标).

        用于绝大多数场景: handler 检测, 状态读取, 训练指标, 支援卡检测等.
        """
        return self._light.recognize(image)

    def recognize_region(self, screenshot, x_pct, y_pct, w_pct, h_pct) -> str:
        """
        区域 OCR — 通过 WinOCR (快速, 百分比坐标).

        用于: handler can_handle 屏幕检测, FrameContext 缓存 OCR.
        这是热路径, 每帧每个 handler 调用 1-5 次.
        """
        return self._light.recognize_region(screenshot, x_pct, y_pct, w_pct, h_pct)

    def recognize_detailed(self, image):
        """
        全图/区域 OCR — 通过 DirectMLOCR (PP-OCRv4 ONNX, 含 bbox+置信度).

        用于需要 bbox 坐标的场景: 全屏校准, 详细识别.
        首次调用 ~1.7s (DirectML 初始化 + 推理), 后续 ~0.5s.
        """
        return self.paddle.recognize(image)


# 高灵敏度 OCR 实例 (模块级缓存, 仅用于增益值小文字识别)
_sensitive_ocr = None


def get_sensitive_ocr():
    """Legacy diagnostic adapter using the installed CPU OCR models."""
    global _sensitive_ocr
    if _sensitive_ocr is not None:
        return _sensitive_ocr
    _sensitive_ocr = DirectMLOCR()
    return _sensitive_ocr


# ============================================================
# 屏幕检测器 (移植自 SleepRunner TrainingScreenChecks + RestDecisionHandler.CanHandle)
# ============================================================

class ScreenDetector:
    """
    通过 OCR 识别右侧分支区域 + 底部按钮区域，判断当前游戏界面。

    SleepRunner 的核心设计: CanHandle() 先用 OCR 读一段文字，
    再用关键字匹配判断是否应该接管当前界面。
    """

    # 右侧分支区域 (SleepRunner: ReadBranchText 0.68, 0.40, 0.30, 0.40)
    BRANCH_REGION = (0.68, 0.40, 0.30, 0.40)

    # 训练屏检测: 右下角"训练"按钮区域
    TRAIN_CONFIRM_REGION = (0.78, 0.82, 0.18, 0.14)

    # 休息屏检测: 右下角"休息"按钮区域 + 选项区域
    REST_CONFIRM_REGIONS = [
        (0.78, 0.82, 0.18, 0.14),
        (0.80, 0.84, 0.16, 0.12),
    ]
    REST_OPTION_REGION = (0.56, 0.40, 0.40, 0.36)

    # 主菜单检测: 菱形按钮行
    MAIN_MENU_ROWS = [
        (0.74, 0.34, 0.22, 0.10),   # 训练
        (0.74, 0.50, 0.22, 0.10),   # 委托/讨伐
        (0.74, 0.58, 0.22, 0.10),   # 休息
    ]

    def __init__(self, ocr: OCREngine, frame=None):
        self.ocr = ocr
        self._frame = frame  # FrameContext, 用于 OCR 缓存

    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        return text.replace(" ", "").replace("\r", "").replace("\n", "").replace("　", "").strip()

    def _read_region(self, screenshot, x, y, w, h) -> str:
        """读取区域 OCR — 优先使用 FrameContext 缓存"""
        if self._frame is not None:
            return self._normalize(self._frame.get_ocr(self.ocr, x, y, w, h))
        return self._normalize(self.ocr.recognize_region(screenshot, x, y, w, h))

    def read_branch_text(self, screenshot) -> str:
        """读取右侧分支区域文字 (与 SleepRunner 的 ReadBranchText 一致)"""
        return self._read_region(screenshot, *self.BRANCH_REGION)

    def read_region_text(self, screenshot, x, y, w, h) -> str:
        return self._read_region(screenshot, x, y, w, h)

    # ===================== 训练界面检测 =====================

    def is_training_screen(self, screenshot) -> bool:
        """
        判断是否在训练详细页。
        移植自 SleepRunner TrainingScreenChecks.IsTrainingDetailText
        """
        branch_text = self.read_branch_text(screenshot)

        # 排除事件界面: 选项含"继续训练"会被训练检测误匹配 (天气事件等)
        if self._is_event_context(branch_text):
            return False

        # 先检查训练指纹 (long training text with digits can falsely match _is_trade_text)
        if self._is_training_detail_text(branch_text):
            return True

        # 排除交易页
        if self._is_trade_text(branch_text):
            return False
        # 排除委托/讨伐
        if "讨伐委托" in branch_text or "受理委托" in branch_text:
            return False

        return False

    def _is_event_context(self, text: str) -> bool:
        """事件界面排除: 排除含有事件选项特征但被训练检测/交易检测误匹配的界面."""
        if not text:
            return False
        # 天气事件选项特征
        if "回旅馆" in text and "继续训练" in text:
            return True
        # 事件指示词 (选项风格)
        event_indicators = ["该怎么办", "怎么办", "请选择", "要不要", "是否要",
                            "等待下次机会", "下次机会", "还是",
                            "再说", "尽力而为", "放松",
                            "填饱肚子", "深呼吸"]
        if any(kw in text for kw in event_indicators):
            return True
        # 检测多个选项编号 (如 "1.", "①", "2.") 且非交易界面
        option_markers = len(re.findall(r'[①②③④⑤⑥]|\d\.', text))
        if option_markers >= 2:
            return True
        return False

    def _is_training_detail_text(self, text: str) -> bool:
        """训练详细页指纹 (与 SleepRunner 的 IsTrainingDetailText 一致)"""
        if not text:
            return False

        training_hits = 0
        for keyword in ["力量训练", "体力训练", "韧性训练", "集中训练", "专注训练", "保护训练"]:
            if keyword in text:
                training_hits += 1
        if training_hits >= 2:
            return True

        if re.search(r'(力量|体力|韧性|集中|专注|保护)训练L?V?\.?\d', text, re.IGNORECASE):
            return True

        if ("继续训练" in text or "返回住处" in text) and "训练" in text:
            return True

        if len(re.findall(r'训练L?V?\.?\d', text)) >= 2:
            return True

        return False

    def _is_trade_text(self, text: str) -> bool:
        """交易页指纹 (与 SleepRunner 的 IsTradeItemText 一致)"""
        if not text:
            return False

        # SOLDOUT 是强交易指纹
        if "SOLDOUT" in text.upper() or "SOLD" in text.upper():
            return True
        trade_hits = 0
        if "购买" in text:
            trade_hits += 2
        for kw in ["禁书", "秘笈", "商品", "评鉴战", "交易"]:
            if kw in text:
                trade_hits += 1
        if trade_hits >= 2:
            return True
        if trade_hits > 0 and bool(re.search(r'\d{1,4}', text)):
            return True
        # 回退: 物品名+数字组合 (如 "活力药水 30", "牛排 80") + 排除事件文本
        # \d{2,4}: 至少2位数字 = 价格, 排除事件选项编号(单个数字"1""2""3")
        if len(text) > 15 and bool(re.search(r'\d{2,4}', text)) and not self._is_event_text(text):
            return True
        # 较长文本 + 较大数字 → 很可能是交易界面(价格)
        if len(text) > 30 and bool(re.search(r'\d{3,5}', text)):
            if not self._is_event_text(text):
                return True
        return False

    @staticmethod
    def _is_event_text(text: str) -> bool:
        """检测文本是否包含事件选项特征, 避免事件界面被误判为交易界面."""
        if not text:
            return False
        event_keywords = [
            "探索", "迷宫", "探勘", "冒险", "挑战",
            "赞成", "反对", "选择", "选项",
            "获得", "失去", "体力", "干劲",
            "训练", "休息", "外出", "出游",
            "事件", "剧情", "返回住处",
            "该怎么办", "怎么办", "要不要", "是否要",
            "还是", "再说", "尽力而为", "放松",
            "填饱肚子", "深呼吸",
        ]
        hits = sum(1 for kw in event_keywords if kw in text)
        if hits >= 2:
            return True
        # 选项编号模式: 2个以上 "数字." 或 "①②③" 标记 → 事件选项
        option_number_markers = len(re.findall(r'[①②③④⑤⑥]|\d\.', text))
        return option_number_markers >= 2

    # 事件界面硬标记: 左上角必定有 "旅程事件" 四个字
    EVENT_MARKER_REGION = (0.04, 0.12, 0.40, 0.18)

    def read_event_option_rows(self, screenshot):
        """读取右侧选项文字和实际中心，保留灰色/锁定行以维持序号。"""
        if screenshot is None:
            return []
        cache = self._frame._ocr_cache if self._frame is not None else {}
        if '_event_rows' in cache:
            return cache['_event_rows']
        height, width = screenshot.shape[:2]
        left, top = int(width * 0.68), int(height * 0.45)
        roi = screenshot[top:int(height * 0.79), left:int(width * 0.945)]
        rows = []
        for text, confidence, box in self.ocr.recognize(roi):
            if confidence < 0.75 or box is None or len(re.findall(r'[\u4e00-\u9fff]', text)) < 2:
                continue
            points = np.asarray(box, dtype=float)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            cx, cy = points.mean(axis=0)
            y = (top + cy) / height
            if not 0.46 <= y <= 0.78:
                continue
            x1, y1 = np.maximum(points.min(axis=0).astype(int), 0)
            x2, y2 = np.minimum(points.max(axis=0).astype(int), [roi.shape[1], roi.shape[0]])
            pixels = roi[y1:y2, x1:x2]
            # 禁用选项文字呈灰色；可用选项文字为白色。
            bright = np.max(pixels, axis=2) if pixels.size else np.array([0])
            enabled = bool(np.count_nonzero(bright >= 210) / bright.size > 0.06)
            rows.append({'text': text.strip(), 'y': y, 'x': (left + cx) / width,
                         'enabled': enabled, 'confidence': float(confidence)})
        rows.sort(key=lambda row: row['y'])
        merged = []
        for row in rows:
            if merged and abs(row['y'] - merged[-1]['y']) < 0.025:
                merged[-1]['text'] += row['text']
                merged[-1]['enabled'] &= row['enabled']
            else:
                merged.append(row)
        cache['_event_rows'] = merged
        return merged

    def find_story_skip(self, screenshot):
        """匹配剧情页的跳过按钮。

        普通旅程页使用体力条右侧的白色图标；新旅程开场/加载剧情会把
        同一功能显示成右上角的文字 ``SKIP``。两种位置都返回可点击坐标，
        但不会把双三角快进按钮当成跳过。
        """
        if screenshot is None:
            return None
        template = cv2.imread(str(Path(__file__).resolve().parents[1] / 'templates/story_skip.png'))
        if template is None:
            return None
        height, width = screenshot.shape[:2]
        scaled = cv2.resize(screenshot, (1600, 900))
        roi = scaled[28:76, 1144:1220]
        def white_mask(image):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 50, 255]))
        mask = white_mask(template)
        if np.count_nonzero(mask) < 30:
            return None
        result = cv2.matchTemplate(white_mask(roi), mask, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        # 游戏实际窗口中同一白色图标通常约为 0.67~0.85；过高阈值会漏掉
        # 叙事过场的蓝白变体，随后把剧情误交给未知兜底并等待 30 秒。
        # 灰色禁用图标仍由 white_mask 的亮度/饱和度条件排除；ROI 只覆盖
        # 体力条右侧第一个图标，不会把旁边的快进按钮当成跳过。
        if score < 0.62:
            # 开场剧情的文字版 SKIP 较小，原图 OCR 不稳定；放大后识别。
            top_roi = scaled[0:100, 1400:1600]
            if top_roi.size:
                enlarged = cv2.resize(top_roi, None, fx=3.0, fy=3.0,
                                      interpolation=cv2.INTER_CUBIC)
                try:
                    texts = self.ocr.recognize(enlarged)
                except Exception:
                    texts = []
                if not isinstance(texts, (list, tuple)):
                    texts = []
                if any("SKIP" in str(item[0]).upper() or "跳过" in str(item[0])
                       for item in texts):
                    # 1600x900 实测文字按钮中心约 (1525, 50)。
                    return (0.953, 0.055)
            return None
        return ((1144 + location[0] + template.shape[1] / 2) / 1600,
                (28 + location[1] + template.shape[0] / 2) / 900)

    def is_story_dialogue(self, screenshot):
        # 选项已出现时跳过图标仍然为白色，必须优先保留选择界面。
        if self.read_event_option_rows(screenshot):
            return False
        dialogue = self.read_region_text(screenshot, 0.10, 0.83, 0.80, 0.13)
        return len(re.findall(r'[\u4e00-\u9fff]', dialogue)) >= 8

    def is_event_marker_screen(self, screenshot) -> bool:
        """检测左上角事件标记 (旅程事件 / 合宿事件 / 阿尔克那事件 / 支援卡事件 / 救援者事件)."""
        text = self.read_region_text(screenshot, *self.EVENT_MARKER_REGION)
        if not text:
            return False
        if any(kw in text for kw in ["旅程事件", "阿尔克那事件", "阿克那事件", "支援卡事件",
                                       "救援者事件", "合宿", "固定事件", "连续事件"]):
            return True
        # 回退: 检查分支区域是否有事件特征
        branch_text = self.read_branch_text(screenshot)
        return self._is_event_context(branch_text)

    # ===================== 商店事件检测 =====================

    # 商店事件检测区域: 训练主屏右侧按钮区
    SHOP_EVENT_REGIONS = [
        (0.68, 0.30, 0.30, 0.50),   # 右侧全高区域
        (0.74, 0.34, 0.22, 0.42),   # 按钮行区域
    ]

    # 交易物品界面检测区域
    TRADE_SCREEN_REGIONS = [
        (0.60, 0.30, 0.35, 0.50),   # 右侧物品列表区域 (扩展到底部按钮)
        (0.68, 0.40, 0.30, 0.50),   # 分支区域 (含物品+按钮)
    ]
    TRADE_BUY_BUTTON_REGION = (0.75, 0.80, 0.22, 0.18)  # 购买按钮区域 (扩大以提升OCR可靠性)

    def is_shop_event_screen(self, screenshot) -> bool:
        """
        检测训练主屏是否触发了商店事件。
        特征: 右侧出现 "交易" + "评鉴战" 取代正常的 "训练/委托/休息"。
        两遍扫描: 不要求两个关键词在同一 OCR 区域, 容忍 OCR 漏字。
        """
        has_trade = False
        has_appraisal = False
        for rx, ry, rw, rh in self.SHOP_EVENT_REGIONS:
            text = self.read_region_text(screenshot, rx, ry, rw, rh)
            if "交易" in text:
                has_trade = True
            if "评鉴战" in text or "评鉴" in text:
                has_appraisal = True
            if self._is_shop_event_text(text):
                return True
            if has_trade and has_appraisal:
                return True

        # 回退: 右侧有 "交易" → 检查顶部横幅是否有 "评鉴战"
        if has_trade:
            top = self._read_top_banner(screenshot)
            if "评鉴战" in top or "评鉴" in top:
                return True
            # 检查是否有交易通知 "全新商品到货" 等
            for rx, ry, rw, rh in self.SHOP_EVENT_REGIONS:
                text = self.read_region_text(screenshot, rx, ry, rw, rh)
                if self._is_trade_notification_text(text):
                    return True
        return False

    # 顶部通知横幅区域 (滚动通知 "委托交易全新商品到货!" / "评鉴战准备" 等)
    TOP_BANNER_REGIONS = [
        (0.05, 0.00, 0.90, 0.10),
        (0.08, 0.01, 0.85, 0.09),
    ]

    def _read_top_banner(self, screenshot) -> str:
        """读取顶部横幅通知文字 (合并所有区域)"""
        texts = []
        for rx, ry, rw, rh in self.TOP_BANNER_REGIONS:
            text = self.read_region_text(screenshot, rx, ry, rw, rh)
            if text:
                texts.append(text)
        return " ".join(texts)

    def is_trade_screen(self, screenshot) -> bool:
        """
        检测是否进入了交易/购买物品界面。
        特征: 右侧显示物品 + "购买"/"SOLDOUT" 关键词。
        """
        # 排除事件界面: 事件选项可能含编号被误判为交易
        branch_text = self.read_branch_text(screenshot)
        if self._is_event_context(branch_text):
            return False

        # 购买按钮检测 (放宽: 购/買 单字也可, 且用更大区域)
        buy_text = self.read_region_text(screenshot, *self.TRADE_BUY_BUTTON_REGION)
        has_buy = "购买" in buy_text or "購入" in buy_text or "购" in buy_text or "買" in buy_text

        for rx, ry, rw, rh in self.TRADE_SCREEN_REGIONS:
            text = self.read_region_text(screenshot, rx, ry, rw, rh)
            if self._is_trade_text(text):
                # 强信号: 购买按钮 或 SOLDOUT → 直接确认
                if has_buy or "SOLDOUT" in text or "SOLD" in text.upper():
                    return True
                # 中信号: 文本含有"购买"或"購入" → 确认
                if "购买" in text or "購入" in text:
                    return True
                # 弱信号: 有交易关键词+数字, 排除事件文本
                if len(text) > 20 and bool(re.search(r'\d{3,5}', text)) and not self._is_event_text(text):
                    return True
        # 回退: branch 区域含交易关键词 (复用上面已读取的 branch_text)
        if self._is_trade_text(branch_text):
            if has_buy or "SOLDOUT" in branch_text or "SOLD" in branch_text.upper():
                return True
            if "购买" in branch_text or "購入" in branch_text:
                return True
        return False

    def is_trade_notification_screen(self, screenshot) -> bool:
        """检测交易通知弹窗 (委托交易全新商品到货等), 不含购买/SOLDOUT"""
        # 先检查顶部横幅
        if self._is_trade_notification_text(self._read_top_banner(screenshot)):
            return True
        # 回退检查右侧分支区域 (全屏弹窗情况)
        branch_text = self.read_branch_text(screenshot)
        return self._is_trade_notification_text(branch_text)

    @staticmethod
    def _is_trade_notification_text(text: str) -> bool:
        """检测交易通知关键词: '委托交易全新商品到货！' 等"""
        if not text:
            return False
        return ("委托交易" in text) or ("交易" in text and ("到货" in text or "全新" in text))

    @staticmethod
    def _is_shop_event_text(text: str) -> bool:
        """检测商店事件关键词: "交易"+"评鉴战" 或 "委托"+"交易" """
        if not text:
            return False
        if "交易" in text and "评鉴战" in text:
            return True
        if "委托" in text and "交易" in text:
            return True
        return False

    # ===================== 评鉴战界面检测 =====================

    APPRAISAL_REGIONS = [
        (0.48, 0.50, 0.28, 0.15),   # 建议综合等级 + RANK 区域
        (0.78, 0.85, 0.18, 0.12),   # 接受按钮区域
    ]

    APPRAISAL_TITLE_REGION = (0.05, 0.02, 0.20, 0.10)  # 评鉴战标题

    def is_appraisal_due_screen(self, screenshot) -> bool:
        """D-DAY 主屏的评鉴战入口，排除提前显示的目标文字和事件选项。"""
        goal = self.read_region_text(screenshot, 0.035, 0.04, 0.20, 0.09).upper()
        if "D-DAY" not in goal:
            return False
        menu = self.read_region_text(screenshot, 0.78, 0.38, 0.19, 0.23)
        return "评鉴战" in menu and "交易" in menu

    def find_action_position(self, screenshot, label, region):
        """在已确认界面的指定区域定位按钮文字，只接受精确标签和有效坐标。"""
        if screenshot is None:
            return None
        height, width = screenshot.shape[:2]
        rx, ry, rw, rh = region
        left, top = int(width * rx), int(height * ry)
        right, bottom = int(width * (rx + rw)), int(height * (ry + rh))
        roi = screenshot[top:bottom, left:right]
        if roi.size == 0:
            return None
        positions = []
        for text, confidence, box in self.ocr.recognize(roi):
            if self._normalize(text) != label or confidence < 0.6 or box is None:
                continue
            points = np.asarray(box, dtype=float)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            cx, cy = points.mean(axis=0)
            if 0 <= cx < roi.shape[1] and 0 <= cy < roi.shape[0]:
                positions.append(((left + cx) / width, (top + cy) / height))
        return positions[0] if len(positions) == 1 else None

    def is_appraisal_screen(self, screenshot) -> bool:
        """
        检测是否进入评鉴战界面。
        特征: 标题 "评鉴战" + "建议综合等级" 或 "接受" 按钮。
        """
        title_text = self.read_region_text(screenshot, *self.APPRAISAL_TITLE_REGION)
        has_title = "评鉴战" in title_text or "评鉴" in title_text
        has_commission = "讨伐" in title_text or "委托" in title_text
        if has_commission:
            return False

        # 点击“接受”后会出现确认层。它仍保留准备页背景和“建议综合等级”，
        # 但已经应由 SkipHandler 处理“跳过战斗”，不能继续被当作准备页等待。
        confirmation = self.read_region_text(screenshot, 0.48, 0.38, 0.42, 0.34)
        if any(keyword in confirmation for keyword in ("是否要进行", "建议战斗", "跳过战斗", "开始委托")):
            return False

        for rx, ry, rw, rh in self.APPRAISAL_REGIONS:
            text = self.read_region_text(screenshot, rx, ry, rw, rh)
            if "建议综合等级" in text and has_title:
                return True
            if "接受" in text and has_title:
                return True
        return False

    # ---- 古币不足对话框检测 ----

    @staticmethod
    def _is_insufficient_money_text(text: str) -> bool:
        """检测 '古币不足' 对话框"""
        if not text:
            return False
        return "古币不足" in text or "古币" in text or "不足" in text

    # ===================== 休息界面检测 =====================

    def is_rest_screen(self, screenshot) -> bool:
        """
        判断是否在休息决策界面。
        移植自 SleepRunner RestDecisionHandler.CanHandle + IsRestDecisionContext
        """
        option_text = self._read_region(screenshot, *self.REST_OPTION_REGION)

        # 快速路径: 一个区域足够判定
        confirm_text = self._read_region(screenshot, *self.REST_CONFIRM_REGIONS[0])

        if self._is_rest_context(option_text, confirm_text):
            return True

        # 扩展路径
        confirm_text2 = self._read_region(screenshot, *self.REST_CONFIRM_REGIONS[1])
        return self._is_rest_context(option_text, confirm_text2)

    def _is_rest_context(self, option_text: str, confirm_text: str) -> bool:
        """与 SleepRunner 的 IsRestDecisionContext 一致"""
        if not option_text:
            return False

        score = 0
        if "免费住处" in option_text:
            score += 4
        if "冥想室" in option_text:
            score += 4
        if "免费" in option_text:
            score += 1
        if "住处" in option_text:
            score += 1
        if "30" in option_text:
            score += 1
        if "60" in option_text:
            score += 1

        has_confirm_rest = confirm_text and "休息" in confirm_text
        if has_confirm_rest:
            return score >= 5

        if self._looks_like_rest_detail(option_text):
            return True

        return score >= 7

    def _looks_like_rest_detail(self, option_text: str) -> bool:
        if not option_text:
            return False
        has_place = any(kw in option_text for kw in ["住处", "露宿", "冥想室", "冥想"])
        has_recovery = any(kw in option_text for kw in ["效果", "恢复", "耐力"])
        has_cost = "30" in option_text or "60" in option_text
        has_menu = any(kw in option_text for kw in ["冥想", "免费", "住处"])
        return has_place and has_recovery and has_cost and has_menu

    # ===================== 弹窗检测 =====================

    INVENTORY_TITLE_REGION = (0.12, 0.12, 0.32, 0.24)

    def is_inventory_screen(self, screenshot) -> bool:
        """检测游戏的背包弹窗。

        背包会覆盖主菜单/旅程画面，必须优先由脚本关闭，否则下层界面
        即使已识别也无法继续。标题 OCR 足够稳定，且不会把背包内容区
        的物品名称误认为弹窗。
        """
        title = self.read_region_text(screenshot, *self.INVENTORY_TITLE_REGION)
        return "背包" in title

    # ===================== 主菜单检测 =====================

    def is_main_menu_screen(self, screenshot) -> str:
        """
        判断是否在主菜单，返回检测到的菜单类型。
        Returns: "training" / "commission" / "rest" / "shop" / ""(非主菜单)

        移植自 SleepRunner MainMenuScreenChecks.IsMainMenuScreen
        """
        has_train = False
        has_commission = False
        has_rest = False
        has_trade = False
        has_appraisal = False

        for i, (rx, ry, rw, rh) in enumerate(self.MAIN_MENU_ROWS):
            text = self.read_region_text(screenshot, rx, ry, rw, rh)
            label = re.sub(r'[^\u4e00-\u9fff]', '', text)
            if label == "训练":
                has_train = True
            if label in ("委托", "讨伐", "讨伐委托"):
                has_commission = True
            if label == "休息":
                has_rest = True
            if label == "交易":
                has_trade = True
            if label in ("评鉴战", "评鉴"):
                has_appraisal = True

        # 命中 ≥2: 正常菜单(训练+委托+休息) 或 商店事件菜单(交易+评鉴战+休息)
        normal_hits = sum([has_train, has_commission, has_rest])
        shop_hits = sum([has_trade, has_appraisal, has_rest])
        if normal_hits < 2 and shop_hits < 2:
            return ""

        if has_train and has_rest:
            return "training"
        if has_trade and has_appraisal:
            return "shop"
        return "training"

    def detect_screen(self, screenshot) -> str:
        """
        综合检测: 返回当前界面类型。
        Returns: "training" / "rest" / "main_menu" / "unknown"
        """
        if self.is_training_screen(screenshot):
            return "training"
        if self.is_rest_screen(screenshot):
            return "rest"
        if self.is_main_menu_screen(screenshot):
            return "main_menu"
        return "unknown"


# ============================================================
# 绿色体力条 HSV 检测
# ============================================================

def detect_green_stamina_bar(screenshot, region=(0.30, 0.00, 0.20, 0.06),
                              green_threshold=0.25) -> bool:
    """
    通过 HSV 色彩检测判断体力条是否满 (绿色像素占比)。

    Args:
        screenshot: 游戏窗口截屏 (BGR)
        region: 检测区域百分比 (x, y, w, h), 默认顶部偏中
        green_threshold: 绿色像素占比阈值

    Returns:
        True 如果绿色像素占比 >= 阈值 (体力条满)
    """
    import cv2
    import numpy as np

    if screenshot is None:
        return False
    h, w = screenshot.shape[:2]
    rx, ry, rw, rh = region
    px = max(0, int(w * rx))
    py = max(0, int(h * ry))
    pw = max(1, min(int(w * rw), w - px))
    ph = max(1, min(int(h * rh), h - py))
    roi = screenshot[py:py + ph, px:px + pw]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / mask.size
    return green_ratio >= green_threshold


def estimate_stamina_bar_pct(screenshot,
                              bar_region=(0.362, 0.052, 0.144, 0.012)) -> int:
    """
    通过绿色耐力条的填充宽度估算耐力百分比。

    检测耐力条中绿色像素的水平延伸比例，映射为耐力百分比。
    坐标通过实际截图校准: 左端(0.3629), 右端(0.5059)。

    Args:
        screenshot: 游戏窗口截屏 (BGR)
        bar_region: 耐力条区域百分比 (x, y, w, h)

    Returns:
        估算的耐力百分比 (0-100), 无法判断返回 50
    """
    import cv2
    import numpy as np

    if screenshot is None:
        return 50

    h, w = screenshot.shape[:2]
    rx, ry, rw, rh = bar_region
    px = max(0, int(w * rx))
    py = max(0, int(h * ry))
    pw = max(1, min(int(w * rw), w - px))
    ph = max(1, min(int(h * rh), h - py))

    if pw < 5 or ph < 3:
        return 50

    roi = screenshot[py:py + ph, px:px + pw]
    if roi.size == 0:
        return 50

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 按列统计绿色像素
    col_green = np.sum(mask, axis=0)
    green_threshold = max(1, ph * 0.25)

    # 从右往左找连续>=3列的绿色段, 避免噪音
    consecutive = 0
    rightmost = 0
    for col in range(mask.shape[1] - 1, -1, -1):
        if col_green[col] >= green_threshold:
            consecutive += 1
            if consecutive >= 3 and rightmost == 0:
                rightmost = col
        else:
            consecutive = 0

    if rightmost == 0:
        return 0

    bar_width = mask.shape[1]
    pct = int((rightmost / max(1, bar_width - 1)) * 100)
    return max(0, min(100, pct))


# ============================================================
# 游戏状态读取器 (相对坐标版)
# ============================================================

class GameStateReader:
    """
    从屏幕 OCR 读取体力/心情/金钱。

    所有坐标均为相对于游戏窗口的百分比 (0.0-1.0)。
    移植自 SleepRunner 的多个 OCR 模块:
      - 体力: 从体力值显示区域 OCR (类似 TrainingPowerStat 但读百分比)
      - 心情: 从 HUD 心情文字 OCR
      - 金钱: 从 HudMoneyOcr 多区域投票
    """

    # 心情关键字 (游戏使用英文)
    MOOD_KEYWORDS = ["Worst", "Bad", "Normal", "Good", "Best"]
    MOOD_FUZZY = {
        "worst": "Worst", "bad": "Bad", "normal": "Normal",
        "good": "Good", "best": "Best",
        "wost": "Worst", "badd": "Bad", "nommal": "Normal",
        "god": "Good", "bets": "Best", "bost": "Best",
        "g00d": "Good", "go0d": "Good", "b3st": "Best",
        "norma": "Normal", "nomal": "Normal",
    }

    # 中央顶部体力 HUD；左上角是目标距离和日期，不是体力。
    STAMINA_REGIONS = [
        (0.355, 0.035, 0.16, 0.04),
    ]

    # 心情区域 (右上角, 体力/金钱之间, 相对坐标 - 校准: 2560x1440)
    MOOD_REGIONS = [
        (0.52, 0.03, 0.08, 0.06),
        (0.50, 0.03, 0.10, 0.06),
        (0.53, 0.04, 0.06, 0.04),
    ]

    # 金钱区域 (右上角, 心情右侧, 相对坐标 - 校准: 2560x1440)
    MONEY_REGIONS = [
        (0.57, 0.03, 0.07, 0.06),
        (0.58, 0.03, 0.06, 0.06),
        (0.56, 0.03, 0.10, 0.07),
        (0.60, 0.03, 0.10, 0.06),
    ]

    def __init__(self, ocr: OCREngine):
        self.ocr = ocr

    def read_state(self, screenshot, old_stamina=None, old_mood=None) -> dict:
        """
        从截图读取完整游戏状态 (使用相对坐标区域)。

        Args:
            screenshot: 游戏窗口截屏 (BGR numpy array)
            old_stamina: 上一次读到的体力值 (用于 OCR 不稳定时的平滑处理)
            old_mood: 上一次读到的心情

        Returns: dict with stamina, mood, money
        """
        result = {"stamina": 100, "mood": "Normal", "money": 0}

        stamina = self._read_stamina(screenshot)
        if stamina is not None:
            result["stamina"] = stamina
        elif old_stamina is not None:
            result["stamina"] = old_stamina

        mood = self._read_mood(screenshot)
        if mood is not None:
            result["mood"] = mood
        elif old_mood is not None:
            result["mood"] = old_mood

        money = self._read_money(screenshot)
        if money is not None:
            result["money"] = money

        return result

    def _read_stamina(self, screenshot) -> float | None:
        """读取体力百分比: OCR优先, 绿色条估算回退"""
        # 1. 尝试 OCR 读取百分比数字
        all_texts = []
        for rx, ry, rw, rh in self.STAMINA_REGIONS:
            text = self._ocr_region_text(screenshot, rx, ry, rw, rh)
            if text:
                all_texts.append(text)

        combined = " ".join(all_texts)
        numbers = re.findall(r'(\d{1,3})\s*[％%]', combined)
        for n in sorted([int(x) for x in numbers], reverse=True):
            if 0 <= n <= 100:
                return float(n)

        # 2. 绿色体力条估算 (无精确数字时)
        green_pct = estimate_stamina_bar_pct(screenshot)
        if screenshot is not None:
            return float(green_pct)
        return None

    def _read_mood(self, screenshot) -> str | None:
        """读取心情 (英文单词)"""
        all_text = ""
        for rx, ry, rw, rh in self.MOOD_REGIONS:
            text = self._ocr_region_text(screenshot, rx, ry, rw, rh)
            if text:
                all_text += " " + text

        all_lower = all_text.lower()

        # 精确匹配
        for keyword in self.MOOD_KEYWORDS:
            if keyword.lower() in all_lower:
                return keyword

        # 模糊匹配
        for word in all_lower.split():
            clean = word.strip().strip(".,;:!?")
            if clean in self.MOOD_FUZZY:
                return self.MOOD_FUZZY[clean]

        return None

    def _read_money(self, screenshot) -> int | None:
        """
        读取金钱 (移植自 SleepRunner HudMoneyOcr 的多区域投票策略)

        返回: 金钱数值, None 表示无法识别
        """
        candidates = {}  # value -> score

        for region_idx, (rx, ry, rw, rh) in enumerate(self.MONEY_REGIONS):
            text = self._ocr_region_text(screenshot, rx, ry, rw, rh)
            if not text:
                continue

            # 排除噪声: 日期/排名/百分比
            if self._has_non_money_noise(text):
                continue

            # 提取数字候选
            numbers = re.findall(r'\d+', text)
            for n_str in numbers:
                val = int(n_str)
                if not (1 <= val <= 999):
                    continue

                # 评分
                score = 0
                if region_idx == 0:
                    score += 20
                elif region_idx == 1:
                    score += 10

                digits = len(n_str)
                if digits == 2:
                    score += 18
                elif digits == 3:
                    score += 24

                if 20 <= val <= 399:
                    score += 16
                elif val <= 999:
                    score += 6

                # 整10倍数加分 (游戏金钱是10的倍数)
                if val >= 30 and val % 10 == 0:
                    score += 8

                candidates[val] = max(candidates.get(val, 0), score)

        if not candidates:
            return None

        # 选最高分
        best_val = max(candidates, key=lambda k: (candidates[k], k))
        if candidates[best_val] < 20:
            return None

        return best_val

    def _has_non_money_noise(self, text: str) -> bool:
        """检测是否是噪声 (日期/排名/百分比等非金钱内容)"""
        noise_keywords = ["月", "上旬", "中旬", "下旬", "日", "距离",
                          "评鉴", "RANK", "D-DAY", "DAY", ":", "：", "/", "％", "%"]
        for kw in noise_keywords:
            if kw.upper() in text.upper():
                return True
        return False

    def _ocr_region_text(self, screenshot, x_pct, y_pct, w_pct, h_pct) -> str:
        """读取指定百分比区域内的 OCR 文本 (去空格+归一化).

        通过 recognize_region() → WinOCR, 即时可用, 无需等待 PaddleOCR 加载.
        """
        return self.ocr.recognize_region(screenshot, x_pct, y_pct, w_pct, h_pct)

    def read_money_fast(self, screenshot) -> int | None:
        """快速读金钱 (单区域, 用于休息决策)"""
        text = self._ocr_region_text(screenshot, 0.62, 0.00, 0.12, 0.07)
        if not text or self._has_non_money_noise(text):
            return None
        numbers = re.findall(r'\d+', text)
        for n_str in sorted(numbers, key=lambda x: len(x), reverse=True):
            val = int(n_str)
            if 20 <= val <= 999:
                return val
        return None


# ============================================================
# 综合 UI 接口
# ============================================================

class TemplateMatcher:
    """基于模板匹配的图像识别 (OpenCV)"""

    def __init__(self, confidence=0.8):
        self.confidence = confidence
        self.templates_dir = Path("templates")
        self._cache = {}

    def load_template(self, name):
        if name in self._cache:
            return self._cache[name]
        filepath = self.templates_dir / f"{name}.png"
        if not filepath.exists():
            raise FileNotFoundError(f"模板图片不存在: {filepath}")
        img = cv2.imread(str(filepath))
        if img is None:
            raise ValueError(f"无法读取模板图片: {filepath}")
        self._cache[name] = img
        return img

    def find(self, screenshot, template_name):
        template = self.load_template(template_name)
        h, w = template.shape[:2]
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.confidence:
            return None
        return (max_loc[0] + w // 2, max_loc[1] + h // 2, max_val)


class GameUI:
    """游戏 UI 识别的综合接口"""

    def __init__(self, capture: ScreenCapture, config):
        self.capture = capture
        self.config = config
        self.matcher = TemplateMatcher(
            confidence=config.get("recognition", {}).get("confidence", 0.8)
        )
        self._ocr_engine = None
        self._state_reader = None
        self._screen_detector = None

    @property
    def ocr(self) -> HybridOCR:
        if self._ocr_engine is None:
            self._ocr_engine = HybridOCR()
        return self._ocr_engine

    @property
    def state_reader(self) -> GameStateReader:
        if self._state_reader is None:
            self._state_reader = GameStateReader(self.ocr)
        return self._state_reader

    @property
    def detector(self) -> ScreenDetector:
        if self._screen_detector is None:
            self._screen_detector = ScreenDetector(self.ocr)
        return self._screen_detector

    def read_game_state(self, screenshot, old_stamina=None, old_mood=None) -> dict:
        """读取游戏状态 (使用相对坐标)"""
        return self.state_reader.read_state(screenshot, old_stamina, old_mood)

    def detect_screen(self, screenshot) -> str:
        """检测当前游戏界面类型"""
        return self.detector.detect_screen(screenshot)

    def find_button(self, button_name):
        screenshot = self.capture.capture_game()
        result = self.matcher.find(screenshot, button_name)
        if result:
            return (result[0], result[1])
        return None
