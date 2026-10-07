"""
Handler 处理器链 - 移植自 SleepRunner 的 IRaceHandler 优先级管道

处理器按优先级排序:
  0: SkipHandler        - 检测 SKIP/跳过 按钮, 快进对话/加载画面
  3: CommissionHandler  - 检测讨伐委托并自动执行
  4: ShopHandler        - 检测商店交易事件, 自动购买最优物品
  5: EventHandler       - 检测事件选项屏幕, 自动选择
  6: TrainingHandler    - 训练决策 + 执行训练
 10: CardSelectHandler  - 检测奖励卡片选择界面
 18: RestHandler        - 休息决策 + 执行休息
 20: MainMenuHandler    - 从主菜单点击训练入口
 99: UnknownHandler     - 未知界面: 点中心推进 + 按 ESC 兜底
"""
from __future__ import annotations

import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from .event_effects import choose_option, score_options


# ============================================================
# FrameContext - 每一帧的上下文
# ============================================================

class FrameContext:
    """单帧截图 + OCR 缓存 (按区域缓存 + 热门区域预计算)"""

    # 热门区域: 被多个 handler 频繁读取, 首次 FrameContext 创建时预 OCR
    # BRANCH_REGION 用于训练/交易/休息/事件等屏幕检测
    HOT_REGIONS = [
        (0.68, 0.40, 0.30, 0.40),   # BRANCH_REGION
        (0.02, 0.02, 0.28, 0.10),   # 卡牌标题
        (0.05, 0.05, 0.15, 0.10),   # 左上角标记
        (0.05, 0.00, 0.90, 0.10),   # 顶部通知横幅
        (0.12, 0.12, 0.32, 0.24),   # 弹窗标题（背包等）
    ]

    def __init__(self, screenshot):
        self.screenshot = screenshot
        self._h = screenshot.shape[0] if screenshot is not None else 0
        self._w = screenshot.shape[1] if screenshot is not None else 0
        self._ocr_cache = {}
        self._precomputed = False

    @property
    def size(self):
        return (self._w, self._h)

    def _precompute_hot_regions(self, engine):
        """预 OCR 被多个 handler 共享的热门区域, 避免重复 OCR"""
        if self._precomputed:
            return
        self._precomputed = True
        for rx, ry, rw, rh in self.HOT_REGIONS:
            key = f"{rx:.3f},{ry:.3f},{rw:.3f},{rh:.3f}"
            if key not in self._ocr_cache:
                self._ocr_cache[key] = _ocr_region(engine, self.screenshot, rx, ry, rw, rh)

    def get_ocr(self, engine, x, y, w_pct, h_pct) -> str:
        """区域 OCR — 带缓存, 首次访问时预计算热门区域"""
        if not self._precomputed:
            self._precompute_hot_regions(engine)
        key = f"{x:.3f},{y:.3f},{w_pct:.3f},{h_pct:.3f}"
        if key not in self._ocr_cache:
            self._ocr_cache[key] = _ocr_region(engine, self.screenshot, x, y, w_pct, h_pct)
        return self._ocr_cache[key]

    def recognize_full(self, engine):
        """全屏 OCR (含 bbox, 通过 PaddleOCR — 仅用于需要坐标的场景)"""
        if '_full_ocr' not in self._ocr_cache:
            items = engine.recognize_detailed(self.screenshot)
            self._ocr_cache['_full_ocr'] = [(t[0], t[1], t[2]) for t in items]
        return self._ocr_cache['_full_ocr']


def _ocr_region(engine, screenshot, x_pct, y_pct, w_pct, h_pct) -> str:
    """读取区域内 OCR 文本 (去空格归一化).

    通过 engine.recognize_region() → HybridOCR 路由到 WinOCR (快速),
    用于 handler can_handle 屏幕检测热路径.
    """
    return engine.recognize_region(screenshot, x_pct, y_pct, w_pct, h_pct)


def _ocr_region_enhanced(engine, screenshot, x_pct, y_pct, w_pct, h_pct) -> str:
    """增强版 OCR: 轻度对比度增强, 不改变分辨率和锐度.

    仅做 CLAHE 对比度增强 (不影响字形的条件下提升可读性),
    避免放大/锐化等操作导致 WinOCR 误判.
    """
    h, w = screenshot.shape[:2]
    rx = int(w * x_pct)
    ry = int(h * y_pct)
    rw = int(w * w_pct)
    rh = int(h * h_pct)
    rx = max(0, rx)
    ry = max(0, ry)
    rw = min(rw, w - rx)
    rh = min(rh, h - ry)
    if rw <= 0 or rh <= 0:
        return ""

    roi = screenshot[ry:ry + rh, rx:rx + rw]

    # 轻度 CLAHE 对比度增强 (只调对比度, 不改大小/锐度)
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return engine.recognize_region(enhanced_bgr, 0, 0, 1.0, 1.0)


# ============================================================
# 模块级配置缓存 — 启动时预加载, 避免 Handler 构造时重复读盘
# ============================================================

_profile_cache: dict[str, dict] = {}


def preload_profiles():
    """预加载所有 events/ 和 shop/ 目录下的 JSON 配置文件到内存缓存."""
    if _profile_cache:
        return  # 已加载

    profiles_root = Path("profiles")
    for subdir in ("events", "shop"):
        dir_path = profiles_root / subdir
        if dir_path.is_dir():
            for json_file in dir_path.glob("*.json"):
                cache_key = str(json_file).replace("\\", "/")
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        _profile_cache[cache_key] = json.load(f)
                except Exception as e:
                    print(f"[缓存] 预加载失败 '{json_file}': {e}")

    if _profile_cache:
        print(f"[缓存] 已预加载 {len(_profile_cache)} 个配置文件到内存")


def _cached_load_json(path: Path, list_key: str) -> list:
    """从缓存读取 JSON, 未命中时读盘并写入缓存. 返回 list_key 对应的 list."""
    cache_key = str(path).replace("\\", "/")
    if cache_key in _profile_cache:
        return _profile_cache[cache_key].get(list_key, [])
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _profile_cache[cache_key] = data
            return data.get(list_key, [])
        except Exception:
            return []
    return []


def _cached_load_json_value(path: Path, key: str, default=None):
    """从同一份 JSON 快查文件读取任意顶层字段。

    事件库仍通过 :func:`_cached_load_json` 读取列表；规则字段使用这个
    辅助函数，避免每次事件都重新打开磁盘文件，同时保留旧配置格式兼容性。
    """
    cache_key = str(path).replace("\\", "/")
    data = _profile_cache.get(cache_key)
    if data is None and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _profile_cache[cache_key] = data
        except Exception:
            return default
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def _cached_save_json(path: Path, data: dict):
    """写入 JSON 文件并同步更新内存缓存."""
    cache_key = str(path).replace("\\", "/")
    _profile_cache[cache_key] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _register_unknown_item(profile_name: str, item_name: str, context: str = "") -> bool:
    """
    模块级函数: 当任何 handler 遇到疑似物品名称的文字时调用。
    如果该物品不在已知库中, 以 type="consumable" 注册, 等待商店中看到效果文字后升级类型。
    返回 True 表示是新物品, False 表示已存在或名称无效。
    """
    if not item_name or len(item_name.strip()) <= 1:
        return False
    if item_name.strip().isdigit():
        return False

    path = Path(f"profiles/shop/{profile_name}.json")
    known = _cached_load_json(path, "items")

    name_clean = item_name.strip()
    for item in known:
        existing_name = item.get("name", "")
        if existing_name == name_clean or name_clean in existing_name or existing_name in name_clean:
            return False  # 已存在

    # 注册新物品 (泛型, 待商店中升级)
    item_id = f"shop_{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_item = {
        "id": item_id,
        "name": name_clean,
        "type": "consumable",
        "keywords": [name_clean],
        "attr_name": name_clean,
        "value": 0,
        "priority": None,
        "note": f"Registered from OCR {now}. Context: {context[:80]}"
    }
    known.append(new_item)
    _cached_save_json(path, {"_comment": "商店物品配置 - 遇到未知物品时自动添加", "items": known})
    return True


# ============================================================
# 模块级转场等待
# ============================================================

def _wait_for_screen(ctx, target: str, timeout: float = 3.0) -> bool:
    """轮询等待目标界面出现 (避免动画期间的 miss-loop).
    支持: training, main_menu, commission, trade, rest.
    返回 True=已检测到, False=超时."""
    from .recognition import ScreenDetector
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        time.sleep(0.1)
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            continue
        detector = ScreenDetector(ctx.ocr)
        if target == "training" and detector.is_training_screen(screenshot):
            print(f"  训练界面已加载 ({time.perf_counter() - t0:.1f}s)")
            return True
        elif target == "main_menu" and detector.is_main_menu_screen(screenshot):
            print(f"  主菜单已加载 ({time.perf_counter() - t0:.1f}s)")
            return True
        elif target == "commission":
            h, w = screenshot.shape[:2]
            for rx, ry, rw_, rh_ in CommissionHandler.COMMISSION_TITLE_REGIONS:
                roi = screenshot[int(h*ry):int(h*(ry+rh_)), int(w*rx):int(w*(rx+rw_))]
                text = " ".join([t[0] for t in ctx.ocr.recognize(roi)])
                if "讨伐委托" in text or "受理讨伐" in text or "建议综合等级" in text:
                    print(f"  委托界面已加载 ({time.perf_counter() - t0:.1f}s)")
                    return True
        elif target == "trade" and detector.is_trade_screen(screenshot):
            print(f"  交易界面已加载 ({time.perf_counter() - t0:.1f}s)")
            return True
        elif target == "rest" and detector.is_rest_screen(screenshot):
            print(f"  休息界面已加载 ({time.perf_counter() - t0:.1f}s)")
            return True
        elif target == "appraisal" and detector.is_appraisal_screen(screenshot):
            print(f"  评鉴战准备页已加载 ({time.perf_counter() - t0:.1f}s)")
            return True
    print(f"  _wait_for_screen('{target}') 超时 ({timeout}s)")
    return False


# ============================================================
# 处理器基类
# ============================================================

class Handler(ABC):
    """处理器基类"""

    @property
    @abstractmethod
    def priority(self) -> int:
        """优先级 (越小越优先)"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """处理器名称"""
        ...

    @abstractmethod
    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        """是否能处理当前画面"""
        ...

    @abstractmethod
    def handle(self, ctx: HandlerContext) -> bool:
        """执行处理逻辑, 返回 True 表示成功"""
        ...


# ============================================================
# HandlerContext - 处理器执行上下文 (只注入必要依赖)
# ============================================================

class HandlerContext:
    """处理器执行上下文"""

    def __init__(self, capture, controller, ui, engine, logger=None):
        self.capture = capture
        self.controller = controller
        self.ui = ui
        self.engine = engine
        self.ocr = ui.ocr
        self.logger = logger
        self.round_count = 0
        self._last_stamina = None
        self._last_mood = "Normal"
        self._last_money = 0
        self._turn_detector = None
        self.turn_detected = False
        self.detected_turn = None
        self.last_branch_text = ""
        self.consecutive_unknown = 0
        # 快速跳转: handler 确认目标界面后设置, 让 _do_round 跳过 dispatch
        self.pending_action = ""  # "training" 等
        self.pending_screenshot = None  # 已确认的截图, 避免重复 capture
        self.development_mode = False

    def set_turn_detector(self, callback):
        self._turn_detector = callback

    def try_detect_turn(self):
        if self.turn_detected or self._turn_detector is None:
            return
        self.turn_detected = True
        turn = self._turn_detector()
        if turn is not None:
            self.detected_turn = turn


# ============================================================
# SkipHandler (Priority 0) - 跳过按钮
# ============================================================

class SkipHandler(Handler):
    """
    统一跳过按钮处理 — 检测任意界面的 SKIP/跳过 按钮并点击。

    覆盖场景:
      - 对话/加载画面右上角 SKIP 按钮
      - 讨伐/评鉴战第1次 "跳过战斗" 按钮 (右下区域)
      - 评鉴战第2次 "跳过战斗" 确认窗口 (中央) ← 新窗口, 实测坐标 (0.5855, 0.6542)
    """

    # 右上角 SKIP 区域 (对话/加载画面)
    SKIP_REGIONS_TOP = [
        (0.90, 0.00, 0.10, 0.08),
        (0.86, 0.00, 0.14, 0.10),
    ]
    SKIP_CLICK_TOP_X = 0.96
    SKIP_CLICK_TOP_Y = 0.04

    # 第1次战斗跳过区域 (讨伐/评鉴战初始对话框)
    SKIP_REGIONS_BATTLE = [
        (0.28, 0.66, 0.25, 0.14),
        (0.30, 0.68, 0.26, 0.12),
        (0.35, 0.65, 0.30, 0.15),
    ]
    SKIP_CLICK_BATTLE_X = 0.43
    SKIP_CLICK_BATTLE_Y = 0.73

    # 第2次战斗跳过确认 (评鉴战后弹出的新窗口, 实测坐标 0.5855, 0.6542)
    SKIP_REGIONS_BATTLE2 = [
        (0.44, 0.58, 0.28, 0.15),   # 以 (0.5855, 0.6542) 为中心
        (0.42, 0.60, 0.32, 0.12),   # 变体
        (0.40, 0.55, 0.35, 0.18),   # 宽区域
    ]
    SKIP_CLICK_BATTLE2_X = 0.5855
    SKIP_CLICK_BATTLE2_Y = 0.6542

    @property
    def priority(self): return 0

    @property
    def name(self): return "SKIP"

    def can_handle(self, frame, ocr_engine) -> bool:
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        if detector.find_story_skip(frame.screenshot) and detector.is_story_dialogue(frame.screenshot):
            self._skip_type = "story"
            return True
        # 检查第1次战斗跳过
        for rx, ry, rw, rh in self.SKIP_REGIONS_BATTLE:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if self._is_skip_text(text):
                self._skip_type = "battle"
                return True
        # 检查第2次战斗跳过确认 (评鉴战新窗口)
        for rx, ry, rw, rh in self.SKIP_REGIONS_BATTLE2:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if self._is_skip_text(text):
                self._skip_type = "battle2"
                return True
        return False

    def handle(self, ctx: HandlerContext) -> bool:
        skip_type = getattr(self, '_skip_type', 'top')
        if skip_type == "story":
            from .recognition import ScreenDetector
            screenshot = ctx.capture.capture_game()
            detector = ScreenDetector(ctx.ocr)
            position = detector.find_story_skip(screenshot)
            if position is None or not detector.is_story_dialogue(screenshot):
                return False
            print("[SKIP] 使用体力条右侧第一个剧情跳过按钮")
            ctx.controller.click_at_percent(*position)
            ctx.controller.wait(0.5)
            ctx.pending_screenshot = ctx.capture.capture_game()
        elif skip_type == "battle2":
            print("[SKIP] 评鉴战确认")
            ctx.controller.click_at_percent(self.SKIP_CLICK_BATTLE2_X, self.SKIP_CLICK_BATTLE2_Y)
            time.sleep(0.3)
            ctx.controller.send_space()
            time.sleep(0.1)
            ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
            time.sleep(0.3)
        elif skip_type == "battle":
            print("[SKIP] 跳过战斗")
            ctx.controller.click_at_percent(self.SKIP_CLICK_BATTLE_X, self.SKIP_CLICK_BATTLE_Y)
            time.sleep(0.3)
            ctx.controller.click_at_percent(self.SKIP_CLICK_BATTLE2_X, self.SKIP_CLICK_BATTLE2_Y)
            time.sleep(0.3)
            ctx.controller.send_space()
            time.sleep(0.1)
            ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
            time.sleep(0.3)
        else:
            print("[SKIP] 快进")
            ctx.controller.click_at_percent(self.SKIP_CLICK_TOP_X, self.SKIP_CLICK_TOP_Y)
            time.sleep(0.1)
            ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        return True

    @staticmethod
    def _is_skip_text(text: str) -> bool:
        if not text:
            return False
        return "SKIP" in text.upper() or "跳过" in text or "跳過" in text


# ============================================================
# InventoryHandler (Priority 1) - 背包弹窗
# ============================================================

class InventoryHandler(Handler):
    """由脚本自动关闭背包弹窗，把控制权交还给下层旅程界面。"""

    CLOSE_X = 0.785
    CLOSE_Y = 0.270

    @property
    def priority(self): return 1

    @property
    def name(self): return "背包弹窗"

    def can_handle(self, frame, ocr_engine) -> bool:
        from .recognition import ScreenDetector
        return ScreenDetector(ocr_engine, frame).is_inventory_screen(frame.screenshot)

    def handle(self, ctx: HandlerContext) -> bool:
        print("[背包] 检测到背包弹窗，点击右上角关闭")
        ctx.controller.click_at_percent(self.CLOSE_X, self.CLOSE_Y)
        ctx.controller.wait(0.35)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


# ============================================================
# StoryResultHandler (Priority 1.5) - 评鉴战胜利剧情结果
# ============================================================

class StoryResultHandler(Handler):
    """处理评鉴战结束后覆盖在角色画面上的胜利结果卡片。"""

    @property
    def priority(self): return 1.5

    @property
    def name(self): return "战斗结果剧情"

    def can_handle(self, frame, ocr_engine) -> bool:
        if frame.screenshot is None:
            return False
        try:
            text = " ".join(item[0] for item in frame.recognize_full(ocr_engine))
        except Exception:
            text = ""
        normalized = re.sub(r"\s+", "", text)
        # 主菜单左上角会长期保留“评鉴战胜利”标题；只有同时出现
        # 结果卡片正文“在远征评鉴战中获胜”时才属于待推进的剧情页。
        return "评鉴战胜利" in normalized and "远征评鉴战中获胜" in normalized

    def handle(self, ctx: HandlerContext) -> bool:
        print("[战斗结果剧情] 评鉴战胜利 → 点击结果卡片继续")
        # 结果卡片下缘是推进区域；这是单次、已识别的精确点击，
        # 不属于 UnknownHandler 的中央连点兜底。
        ctx.controller.click_at_percent(0.50, 0.84)
        time.sleep(0.6)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


# ============================================================
# BattleLeaveHandler (Priority 2) - 战后离开
# ============================================================

class BattleLeaveHandler(Handler):
    """
    检测战斗胜利结算画面, 点击 "离开" 按钮精准退出。

    移植自 SleepRunner BattleLeaveHandler:
    - 检测 VICTORY / 回合以内获胜 / 达成度 文字
    - 检测 "离开" 按钮
    - 点击 (0.91, 0.90) 离开结算画面
    - 离开后快速推进对话
    """

    # 胜利/结算文字检测区域 (右上/中央)
    VICTORY_REGIONS = [
        (0.72, 0.10, 0.26, 0.16),   # 右上胜利文字
        (0.68, 0.08, 0.30, 0.20),   # 扩展
        (0.50, 0.10, 0.45, 0.15),   # 中上
    ]

    # "离开" 按钮检测区域 (右下)
    LEAVE_REGIONS = [
        (0.82, 0.82, 0.16, 0.14),   # 右下按钮区
        (0.80, 0.80, 0.18, 0.16),   # 扩展
        (0.78, 0.78, 0.20, 0.18),   # 更宽
    ]

    # 蓝色 "离开" 按钮 HSV 颜色检测区域 (兜底)
    LEAVE_BLUE_REGION = (0.82, 0.82, 0.16, 0.14)

    LEAVE_CLICK_X = 0.91
    LEAVE_CLICK_Y = 0.90

    @property
    def priority(self): return 2

    @property
    def name(self): return "战后离开"

    def can_handle(self, frame, ocr_engine) -> bool:
        screenshot = frame.screenshot
        if screenshot is None:
            return False

        # 检测1: 胜利/结算文字
        has_victory = self._detect_victory_text(frame, ocr_engine)

        # 检测2: "离开" 按钮文字
        has_leave = self._detect_leave_button(frame, ocr_engine)

        if has_victory and has_leave:
            return True

        # 检测3: 仅 "离开" 按钮 + 蓝色按钮 HSV 兜底
        if has_leave and self._detect_blue_leave_button(screenshot):
            return True

        # 检测4: 排除训练/休息/商店界面后, 检测到 "离开"
        if has_leave:
            from .recognition import ScreenDetector
            detector = ScreenDetector(ocr_engine, frame)
            if (not detector.is_training_screen(screenshot) and
                not detector.is_rest_screen(screenshot) and
                not detector.is_shop_event_screen(screenshot)):
                return True

        return False

    def handle(self, ctx: HandlerContext) -> bool:
        print("[战后] 胜利, 离开")
        ctx.controller.click_at_percent(self.LEAVE_CLICK_X, self.LEAVE_CLICK_Y)
        time.sleep(0.5)
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        time.sleep(0.3)
        return True

    def _detect_victory_text(self, frame, ocr_engine) -> bool:
        """检测胜利/结算相关文字"""
        for rx, ry, rw, rh in self.VICTORY_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if not text:
                continue
            if any(kw in text.upper() for kw in ["VICTORY", "VICTOR"]):
                return True
            if "回合以内获胜" in text or "回合内获胜" in text:
                return True
            if "达成度" in text:
                return True
        return False

    def _detect_leave_button(self, frame, ocr_engine) -> bool:
        """检测 '离开' 按钮文字"""
        for rx, ry, rw, rh in self.LEAVE_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if text and "离开" in text:
                return True
        return False

    def _detect_blue_leave_button(self, screenshot) -> bool:
        """HSV 检测蓝色 '离开' 按钮 (兜底, OCR 可能漏读)"""
        import cv2
        import numpy as np

        if screenshot is None:
            return False
        h, w = screenshot.shape[:2]
        rx, ry, rw, rh = self.LEAVE_BLUE_REGION
        x, y = int(w * rx), int(h * ry)
        rw_px, rh_px = int(w * rw), int(h * rh)
        if rw_px <= 0 or rh_px <= 0:
            return False
        roi = screenshot[y:y + rh_px, x:x + rw_px]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 蓝色 HSV 范围
        lower = np.array([100, 50, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        blue_ratio = np.count_nonzero(mask) / mask.size
        return blue_ratio >= 0.08  # 至少 8% 蓝色像素


# ============================================================
# BattleDefeatHandler (Priority 1) - 战斗失败重试
# ============================================================

class BattleDefeatHandler(Handler):
    """
    战斗失败处理 — 移植自 SleepRunner BattleDefeatHandler.

    每把跑马有 3 次重新挑战机会, 失败时:
      - 剩余次数 > 0: 点击"重新挑战" → 直接点击第二个确认 (双确认同时显示)
      - 剩余次数 = 0: 点击第一个确认退出, 重置次数

    优先级 1, 比 BattleLeaveHandler(2) 高一级。
    """

    MAX_RETRIES = 3

    # 失败文字检测区 (FAIL/DEFEAT)
    DEFEAT_REGIONS = [
        (0.66, 0.08, 0.34, 0.40),
        (0.60, 0.05, 0.38, 0.45),
        (0.20, 0.10, 0.60, 0.40),
    ]

    # ---- 按钮坐标 (用户实测校准) ----
    # 重新挑战按钮
    RETRY_CLICK_X = 0.4137
    RETRY_CLICK_Y = 0.9069

    # 第一个确认 (重试次数=0时, 点击退出)
    CONFIRM1_X = 0.5711
    CONFIRM1_Y = 0.9097

    # 第二个确认 (点击重新挑战后的弹窗确认, 与第一个确认同时显示)
    CONFIRM2_X = 0.5715
    CONFIRM2_Y = 0.6910

    def __init__(self, config=None):
        self._retries_remaining = self.MAX_RETRIES

    def reset_retries(self):
        """重置重新挑战次数 (新跑马开始时调用)."""
        self._retries_remaining = self.MAX_RETRIES
    @property
    def priority(self): return 1

    @property
    def name(self): return "战斗失败重试"

    def can_handle(self, frame, ocr_engine) -> bool:
        screenshot = frame.screenshot
        if screenshot is None:
            return False
        if self._is_retry_dialog(frame, ocr_engine):
            return True
        if not self._is_defeat_screen(frame, ocr_engine):
            return False
        return True

    def handle(self, ctx: HandlerContext) -> bool:
        screenshot = ctx.capture.capture_game()
        if screenshot is not None:
            if self._is_retry_dialog(FrameContext(screenshot), ctx.ocr):
                print("[战斗] 重试确认弹窗, 点击确认")
                ctx.controller.click_at_percent(self.CONFIRM2_X, self.CONFIRM2_Y)
                time.sleep(0.5)
                return True

        if self._retries_remaining > 0:
            self._retries_remaining -= 1
            print(f"[战斗] FAIL, 重试 ({self._retries_remaining}/{self.MAX_RETRIES})")
            ctx.controller.click_at_percent(self.RETRY_CLICK_X, self.RETRY_CLICK_Y)
            time.sleep(0.2)
            ctx.controller.click_at_percent(self.CONFIRM2_X, self.CONFIRM2_Y)
            time.sleep(0.5)
        else:
            print(f"[战斗] 重试用尽, 退出")
            ctx.controller.click_at_percent(self.CONFIRM1_X, self.CONFIRM1_Y)
            time.sleep(0.5)
            ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
            time.sleep(0.3)
            self._retries_remaining = self.MAX_RETRIES

        return True

    def _is_defeat_screen(self, frame, ocr_engine) -> bool:
        """检测失败文字: DEFEAT (SleepRunner) 或 FAIL (中文游戏)"""
        for rx, ry, rw, rh in self.DEFEAT_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if not text:
                continue
            normalized = text.replace(" ", "").replace("\r", "").replace("\n", "").replace("　", "").strip().upper()
            if "DEFEAT" in normalized:
                return True
            if "EFEAT" in normalized:
                return True
            if "FAIL" in normalized:
                return True
        return False

    def _has_retry_button(self, frame, ocr_engine) -> bool:
        """检测右下角 '重新挑战' 按钮文字 (保留用于将来可能的 OCR 辅助判断)"""
        text = frame.get_ocr(ocr_engine, *(0.60, 0.88, 0.22, 0.10))
        if not text:
            return False
        normalized = text.replace(" ", "").replace("\r", "").replace("\n", "").replace("　", "").strip()
        if "重新挑战" in normalized:
            return True
        if "重新" in normalized and "挑战" in normalized:
            return True
        return False

    @classmethod
    def _is_retry_dialog(cls, frame, ocr_engine) -> bool:
        """
        检测 '重新挑战通知' 弹窗 (供 EventHandler 交叉检测, 防止死循环).

        移植自 SleepRunner EventScreenChecks.IsRetryDialogContext:
        弹窗正文"是否要重新挑战"会被 EventHandler._is_event_option_hint("是否")
        误判为事件选项 → 点取消 → BattleDefeat 又点重新挑战 → 死循环.
        """
        regions = [
            (0.20, 0.15, 0.60, 0.30),
            (0.25, 0.20, 0.50, 0.35),
        ]
        for rx, ry, rw, rh in regions:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if not text:
                continue
            normalized = text.replace(" ", "").replace("\r", "").replace("\n", "").replace("　", "").strip()
            if "重新挑战通知" in normalized:
                return True
            if "是否要重新挑战" in normalized:
                return True
            if "再次尝试战斗" in normalized:
                return True
            # 弱兜底: 同时含"重新挑战"+"再"+"次"
            if "重新挑战" in normalized and "再" in normalized and "次" in normalized:
                return True
        return False


# ============================================================
# ============================================================
# CampSelectHandler (Priority 2) - 合宿地区选择
# ============================================================

class CampSelectHandler(Handler):
    """
    检测合宿地区选择界面并自动选择。

    合宿是"地区移动"的变体（左上角同样显示"地区移动"标题）：
    - 弗洛拉 (力量加成 + 专注加成) → 攻击/速度方向
    - 卡莱德 (体力加成 + 保护加成) → 生存方向 (体力/韧性/保护优先)
    """

    # --- 检测区域 (百分比坐标) ---
    TITLE_REGION = (0.00, 0.04, 0.30, 0.16)

    CAMP_NAME_REGIONS = [
        (0.74, 0.10, 0.24, 0.14),
        (0.72, 0.08, 0.26, 0.16),
        (0.76, 0.12, 0.22, 0.12),
        (0.74, 0.22, 0.24, 0.16),
        (0.72, 0.20, 0.26, 0.20),
    ]

    # --- 选项探测 (已根据实测坐标校准) ---
    OPTION_PROBE_1 = (0.74, 0.25, 0.20, 0.07)
    OPTION_PROBE_2 = (0.74, 0.36, 0.20, 0.07)

    # --- 点击位置 (已根据实测坐标校准) ---
    FLORA_CLICK = (0.8203, 0.2826)
    KALEID_CLICK = (0.8352, 0.3903)
    GO_BUTTON_X = 0.8578
    GO_BUTTON_Y = 0.8958
    SKIP_CLICK_X = 0.9520
    SKIP_CLICK_Y = 0.0549

    # 传统模式下选择卡莱德的属性ID: 生命(2), 韧性(3), 保护(5)
    _KALEID_TRAD_ATTRS = {2, 3, 5}

    @property
    def priority(self): return 2

    @property
    def name(self): return "合宿选择"

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        # 1. 左上角 "地区移动" 标题
        title = frame.get_ocr(ocr_engine, *self.TITLE_REGION)
        if "地区移动" not in title and "目标地区移动" not in title:
            return False

        # 2. 右侧地区名是否包含弗洛拉或卡莱德
        for rx, ry, rw, rh in self.CAMP_NAME_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if "弗洛拉" in text or "卡莱德" in text:
                return True
        return False

    def handle(self, ctx: HandlerContext) -> bool:
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False
        frame = FrameContext(screenshot)

        # 1. 探测各选项对应的地区名
        option_map: dict[str, tuple[float, float]] = {}
        for text, (cx, cy) in [
            (frame.get_ocr(ctx.ocr, *self.OPTION_PROBE_1), self.FLORA_CLICK),
            (frame.get_ocr(ctx.ocr, *self.OPTION_PROBE_2), self.KALEID_CLICK),
        ]:
            for camp_name in ("弗洛拉", "卡莱德"):
                if camp_name in text:
                    option_map[camp_name] = (cx, cy)
                    break

        # 2. 决定目标地区
        target = self._resolve_target(ctx)
        click_x, click_y = option_map.get(target, self.FLORA_CLICK)

        print(f"[合宿] '{target}'")

        # 3. 点击选项
        ctx.controller.click_at_percent(click_x, click_y)
        time.sleep(0.5)

        # 4. 点击 "前往" 按钮
        ctx.controller.click_at_percent(self.GO_BUTTON_X, self.GO_BUTTON_Y)
        time.sleep(1.0)

        # 5. 点击 SKIP 跳过后续对话
        ctx.controller.click_at_percent(self.SKIP_CLICK_X, self.SKIP_CLICK_Y)
        time.sleep(0.3)

        # 6. 推进后续画面
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        time.sleep(0.5)
        return True

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _resolve_target(self, ctx: HandlerContext) -> str:
        """根据训练方向决定目标合宿地区"""
        try:
            if ctx.engine.is_rule_engine_mode:
                direction = ctx.engine.rule_profile.legacy_strategy.build_direction.value
                # survival → 卡莱德; attack/speed → 弗洛拉
                return "卡莱德" if direction == "survival" else "弗洛拉"
            else:
                # 传统模式: 生命(2)/韧性(3)/保护(5) → 卡莱德; 其余→弗洛拉
                return "卡莱德" if ctx.engine.priority in self._KALEID_TRAD_ATTRS else "弗洛拉"
        except Exception:
            return "弗洛拉"  # fallback


# ============================================================
# MovePlatformHandler (Priority 3) - 地区移动/列车月台
# ============================================================

class MovePlatformHandler(Handler):
    """
    检测 "地区移动" / "列车月台" 界面并自动选择选项。

    移植自 SleepRunner MovePlatformHandler:
    - 左上角检测 "地区移动" 或 "目标地区移动" 标题
    - 右上角检测 "列车月台" / "车月台" / "车站台" 平台文字
    - 根据配置选择选项 1 或 2 (默认选 1)
    - 点击选项后点击 "前往" 按钮
    """

    # 左上标题检测区域 (移植自 SleepRunner)
    TITLE_REGION = (0.00, 0.04, 0.30, 0.16)

    # 右上平台标题检测区域
    PLATFORM_HEADER_REGIONS = [
        (0.74, 0.10, 0.24, 0.14),
        (0.72, 0.08, 0.26, 0.16),
        (0.76, 0.12, 0.22, 0.12),
    ]

    # 右侧选项检测区域
    RIGHT_OPTION_REGIONS = [
        (0.74, 0.22, 0.24, 0.16),
        (0.72, 0.20, 0.26, 0.20),
        (0.76, 0.24, 0.22, 0.14),
    ]

    # 选项行探测区域 (检测有几个选项)
    OPTION_PROBE_1 = (0.74, 0.30, 0.24, 0.10)
    OPTION_PROBE_2 = (0.74, 0.42, 0.24, 0.10)

    # 点击位置
    OPTION_1_CLICK_X = 0.86
    OPTION_1_CLICK_Y = 0.30
    OPTION_2_CLICK_Y = 0.42
    GO_BUTTON_X = 0.88
    GO_BUTTON_Y = 0.91
    SKIP_CLICK_X = 0.9520
    SKIP_CLICK_Y = 0.0549

    @property
    def priority(self): return 3

    @property
    def name(self): return "地区移动"

    def __init__(self, config=None):
        self.config = config or {}
        mp_cfg = self.config.get("move_platform", {})
        self.option_index = mp_cfg.get("option_index", 1)

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        # 1. 检测左上角 "地区移动" 标题
        title_text = frame.get_ocr(ocr_engine, *self.TITLE_REGION)
        if not self._is_move_stage_title(title_text):
            return False

        # 2. 检测右上角平台文字 (需要至少一个区域匹配)
        has_platform = False
        for rx, ry, rw, rh in self.PLATFORM_HEADER_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if self._is_platform_header_text(text):
                has_platform = True
                break
        if not has_platform:
            for rx, ry, rw, rh in self.RIGHT_OPTION_REGIONS:
                text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
                if self._is_train_platform_text(text):
                    has_platform = True
                    break

        return has_platform

    def handle(self, ctx: HandlerContext) -> bool:
        print("[月台] 地区移动")

        # 截图
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False
        frame = FrameContext(screenshot)

        # 1. 探测有几个选项
        text1 = frame.get_ocr(ctx.ocr, *self.OPTION_PROBE_1)
        text2 = frame.get_ocr(ctx.ocr, *self.OPTION_PROBE_2)
        has_option1 = self._is_option_line_text(text1)
        has_option2 = self._is_option_line_text(text2)
        num_options = 2 if (has_option1 and has_option2) else 1

        # 2. 选择选项
        opt_idx = min(self.option_index, num_options)
        click_y = self.OPTION_1_CLICK_Y if opt_idx == 1 else self.OPTION_2_CLICK_Y
        print(f"[月台] 选项{opt_idx}/{num_options}")
        ctx.controller.click_at_percent(self.OPTION_1_CLICK_X, click_y)
        time.sleep(0.5)

        # 3. 点击 "前往" 按钮
        ctx.controller.click_at_percent(self.GO_BUTTON_X, self.GO_BUTTON_Y)
        time.sleep(1.0)

        # 4. 点击 SKIP 跳过后续对话
        ctx.controller.click_at_percent(self.SKIP_CLICK_X, self.SKIP_CLICK_Y)
        time.sleep(0.3)

        # 5. 推进后续画面
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        time.sleep(0.5)
        return True

    # ===================== 文本判定 =====================

    @staticmethod
    def _is_move_stage_title(text: str) -> bool:
        if not text:
            return False
        return "地区移动" in text or "目标地区移动" in text

    @staticmethod
    def _is_platform_header_text(text: str) -> bool:
        return MovePlatformHandler._is_train_platform_text(text)

    @staticmethod
    def _is_train_platform_text(text: str) -> bool:
        if not text:
            return False
        import re
        if "列车月台" in text or "车月台" in text or "车站台" in text:
            return True
        if "列车" in text and ("月台" in text or "站台" in text):
            return True
        if re.search(r'(列|刂)?.?车.*(月.?台|站.?台)', text):
            return True
        return False

    @staticmethod
    def _is_option_line_text(text: str) -> bool:
        """选项行需包含至少2个中文字符, 且不含导航按钮文字"""
        if not text:
            return False
        if "前往" in text or "返回" in text:
            return False
        # 统计中文字符数
        zh_count = sum(1 for c in text if '一' <= c <= '鿿')
        return zh_count >= 2


# ============================================================
# AttributeGainHandler (Priority 4.0) - 属性结算动画
# ============================================================

class AttributeGainHandler(Handler):
    """推进训练后的属性结算动画。

    训练完成后游戏会短暂显示“某角色的体力提升了 +40”之类的覆盖层。
    该画面没有训练按钮、事件选项或剧情 SKIP，若交给未知兜底会一直等待。
    """

    @property
    def priority(self): return 4.0

    @property
    def name(self): return "属性结算"

    def can_handle(self, frame, ocr_engine) -> bool:
        if frame.screenshot is None:
            return False
        try:
            text = " ".join(item[0] for item in frame.recognize_full(ocr_engine))
        except Exception:
            text = ""
        normalized = re.sub(r"\s+", "", text)
        if not re.search(r"\+\s*\d{1,3}", normalized):
            return False
        if not any(marker in normalized for marker in ("提升了", "增加了", "上升了")):
            return False
        # 已有明确处理器的界面不交给结算动画，避免 +N 物品/训练文字误判。
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        if detector.is_training_screen(frame.screenshot):
            return False
        if detector.is_rest_screen(frame.screenshot) or detector.is_shop_event_screen(frame.screenshot):
            return False
        if detector.read_event_option_rows(frame.screenshot):
            return False
        return True

    def handle(self, ctx: HandlerContext) -> bool:
        print("[属性结算] 检测到属性提升动画 → 点击结算层继续")
        # 结算图标位于画面中央偏下；单次点击，不使用未知兜底连点。
        ctx.controller.click_at_percent(0.50, 0.70)
        time.sleep(0.6)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


# ============================================================
# TrainingResultHandler (Priority 4.5) - 训练结果过渡页
# ============================================================

class TrainingResultHandler(Handler):
    """推进“训练失败/训练成功”结果层，交给后续事件或菜单处理。"""

    @property
    def priority(self): return 4.5

    @property
    def name(self): return "训练结果"

    def can_handle(self, frame, ocr_engine) -> bool:
        if frame.screenshot is None:
            return False
        try:
            text = " ".join(item[0] for item in frame.recognize_full(ocr_engine))
        except Exception:
            text = ""
        normalized = re.sub(r"\s+", "", text)
        if any(choice in normalized for choice in (
                "训练到此为止", "休息吧", "再试一次", "活力药水")):
            # 结果页推进后会留下“训练失败”标题；出现选项文字时交给
            # EventHandler 的体力/道具策略处理。
            return False
        return any(word in normalized for word in ("训练失败", "训练成功", "训练超感应"))

    def handle(self, ctx: HandlerContext) -> bool:
        print("[训练结果] 识别到训练结果 → 点击结果层继续")
        # 结果字样位于画面中下部，单次精确推进；不使用未知兜底连点。
        ctx.controller.click_at_percent(0.50, 0.76)
        time.sleep(0.6)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


# ============================================================
# EventHandler (Priority 6) - 事件选择 (原P5)
# ============================================================

class EventDecisionPending(Exception):
    """缺少经过文档核对的事件效果时，在选择前暂停。"""

class EventHandler(Handler):
    """
    检测事件选择界面并自动选择。

    移植自 SleepRunner EventHandler + EventScreenChecks + EventOptionGeometry:
    - 多区域检测旅程事件标识 + 选项提示文字
    - 根据选项数自动计算点击 Y 坐标
    - 默认选第一个选项
    """

    # 事件选项区域
    OPTION_REGIONS = [
        (0.55, 0.45, 0.40, 0.35),
        (0.52, 0.42, 0.44, 0.40),
    ]

    # 旅程标识区域 (移植自 SleepRunner JourneyMarkerRegions — 需要覆盖到 Y=0.46)
    JOURNEY_REGIONS = [
        (0.00, 0.00, 0.22, 0.08),
        (0.00, 0.00, 0.26, 0.12),
        (0.02, 0.00, 0.30, 0.14),
        (0.02, 0.20, 0.38, 0.16),
        (0.02, 0.22, 0.42, 0.18),
        (0.00, 0.12, 0.42, 0.20),
        (0.00, 0.18, 0.55, 0.28),
        (0.04, 0.22, 0.52, 0.24),
        (0.02, 0.24, 0.62, 0.30),
        (0.00, 0.28, 0.68, 0.34),
        (0.00, 0.30, 0.72, 0.36),
    ]

    # 选项点击 X (移植自 SleepRunner EventOptionGeometry.OptionClickX)
    OPTION_CLICK_X = 0.75

    # 按选项数固定的 Y 中心 (移植自 SleepRunner EventOptionGeometry)
    TWO_OPTION_CENTERS = [0.65, 0.74]
    THREE_OPTION_CENTERS = [0.57, 0.65, 0.73]
    FOUR_OPTION_CENTERS = [0.52, 0.59, 0.66, 0.73]

    @property
    def priority(self): return 6

    @property
    def name(self): return "事件选择"

    def __init__(self, config=None):
        self.config = config or {}
        training_cfg = self.config.get("training", {})
        self._profile_name = training_cfg.get("rule_profile", "default")
        self._profile_path = Path(f"profiles/events/{self._profile_name}.json")
        event_cfg = self.config.get("events", {})
        self._quick_lookup_path = Path(event_cfg.get(
            "quick_lookup", "profiles/events/document_verified.json"))
        self._events: list = []
        self._last_failed_text = ""   # 防止同一事件死循环
        self._load_profile()

    def _load_profile(self):
        """加载事件配置 JSON (优先从内存缓存读取)."""
        cached = _cached_load_json(self._profile_path, "events")
        if cached:
            self._events = cached
            print(f"[事件] 已加载 {len(self._events)} 个已知事件 ({self._profile_name})")
        else:
            self._events = []

    def _save_profile(self):
        """持久化事件列表到 JSON 文件 (同步更新缓存)."""
        data = {
            "_comment": "事件决策配置 - 匹配逻辑：OCR识别选项文字 -> 模糊匹配 options 中的 keyword/alias",
            "events": self._events,
        }
        _cached_save_json(self._profile_path, data)

    def _match_known_event(self, option_text: str) -> dict | None:
        """将 OCR 选项文字与已知事件进行模糊匹配, 返回最佳匹配或 None."""
        if not option_text or not self._events:
            return None
        normalized = self._normalize(option_text)
        if not normalized:
            return None

        candidates = []
        for event in self._events:
            options = event.get("options", [])
            if not options:
                continue

            # 同 auto-learn 条目用字符相似度匹配 (OCR 变体多, 关键字匹配不可靠)
            if event.get("id", "").startswith("auto_"):
                keyword = self._normalize(options[0].get("keyword", ""))
                if len(keyword) > 4 and len(normalized) > 4:
                    sa, sb = set(keyword), set(normalized)
                    sim = len(sa & sb) / len(sa | sb) if (sa | sb) else 0
                    if sim > 0.55:
                        candidates.append((event, 1, sim))
                continue

            # 标准匹配: keyword/alias 子串命中
            matched = 0
            for opt in options:
                terms = [opt.get("keyword", "")] + opt.get("alias", [])
                for term in terms:
                    if term and self._normalize(term) in normalized:
                        matched += 1
                        break
            total = len(options)
            ratio = matched / total if total > 0 else 0
            if matched > 0:
                candidates.append((event, matched, ratio))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[2], -x[1]))
        best, best_matched, best_ratio = candidates[0]

        if best_ratio < 0.33:
            return None

        # 平局时优先 confirmed 状态
        if len(candidates) >= 2:
            second = candidates[1]
            if abs(best_ratio - second[2]) < 0.05 and best_matched == second[1]:
                if best.get("status") != "confirmed" and second[0].get("status") == "confirmed":
                    best = second[0]
                elif best.get("status") == "confirmed":
                    pass
                else:
                    return None

        return best

    # 事件类型标签 (用于从 OCR 文本中提取事件名)
    _EVENT_TYPE_LABELS = ["旅程事件", "阿尔克那事件", "阿克那事件", "支援卡事件", "救援者事件"]

    def _auto_learn_event(self, option_text: str, marker_text: str, num_options: int) -> dict:
        """为无法匹配的事件创建新条目并持久化 (自动去重)."""
        event_id = f"auto_{uuid.uuid4().hex[:8]}"
        # 从 marker_text 提取真实事件名 (去除事件类型标签后的剩余文本)
        raw_marker = self._normalize(marker_text)
        event_name_raw = raw_marker
        for label in self._EVENT_TYPE_LABELS:
            event_name_raw = event_name_raw.replace(label, "")
        event_name_raw = event_name_raw.strip()
        # 取前20字作为事件名, 没有则用选项文字兜底
        display_name = event_name_raw[:20] if event_name_raw else self._normalize(option_text)[:30]
        if not display_name:
            display_name = f"Unknown-{len(self._events) + 1}"

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        raw_snapshot = f"{marker_text} | {option_text}"[:200]
        raw_norm = self._normalize(raw_snapshot)

        # 去重: 优先用事件名比对, 回退到 raw note 相似度
        for existing in self._events:
            if not existing.get("id", "").startswith("auto_"):
                continue
            existing_name = self._normalize(existing.get("event_name", ""))
            if existing_name and event_name_raw and len(existing_name) > 2 and len(event_name_raw) > 2:
                sa = set(existing_name)
                sb = set(event_name_raw)
                sim = len(sa & sb) / len(sa | sb) if (sa | sb) else 0
                if sim > 0.6:
                    print(f"  [事件] 去重跳过 (事件名相似{sim:.0%})")
                    return existing
            existing_raw = self._normalize(existing.get("note", ""))
            if existing_raw and len(existing_raw) > 5 and len(raw_norm) > 5:
                sa = set(existing_raw)
                sb = set(raw_norm)
                sim = len(sa & sb) / len(sa | sb) if (sa | sb) else 0
                if sim > 0.6:
                    print(f"  [事件] 去重跳过 (内容相似{sim:.0%})")
                    return existing

        # 将事件名关键词加入 alias 提升后续匹配准确度
        alias = []
        if event_name_raw:
            alias.append(event_name_raw[:15])

        new_event = {
            "id": event_id,
            "event_name": display_name,
            "status": "pending",
            "recommended_option": None,
            "num_options": num_options,
            "options": [
                {"index": 1, "keyword": self._normalize(option_text)[:60], "alias": alias}
            ],
            "note": f"Auto-learned {now}. Options: {num_options}. Raw: {raw_snapshot}"
        }
        self._events.append(new_event)
        self._save_profile()
        print(f"[事件] 新事件: {display_name} ({num_options}选项)")
        return new_event

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        screenshot = frame.screenshot
        if screenshot is None:
            return False

        if self.config.get('events', {}).get('require_document_effects', False):
            from .recognition import ScreenDetector
            detector = ScreenDetector(ocr_engine, frame)
            marker = detector.read_region_text(screenshot, 0.035, 0.14, 0.40, 0.16)
            return ('事件' in marker or '训练失败' in marker) and bool(detector.read_event_option_rows(screenshot))

        # 获取各个区域的文字
        marker = self._read_marker(frame, ocr_engine)
        option_text = self._read_options(frame, ocr_engine)

        # 重复事件防护: 如果与上一次失败的 OCR 文本高度相似, 跳过
        pre_ocr = self._normalize(option_text)
        if pre_ocr and self._last_failed_text:
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, pre_ocr[:60], self._last_failed_text[:60]).ratio()
            if ratio > 0.7:
                return False

        # 排除训练页
        if self._is_training_context(marker, option_text):
            return False
        # 排除主菜单
        if self._is_main_menu_like(option_text):
            return False
        # 排除休息页
        if self._is_rest_context(option_text):
            return False
        # 排除评鉴目标列表
        if self._is_appraise_goal(option_text):
            return False
        # 排除交易界面 (购买按钮区域有"购买"文字)
        if self._is_trade_screen(ocr_engine, frame):
            return False
        # 排除主屏商店事件 (右侧有评鉴战+交易按钮, 不是事件选项)
        if "评鉴战" in option_text and "交易" in option_text:
            return False
        # 商店事件屏兜底: 两处文字各自出现 评鉴战/交易 也排除
        if ("评鉴战" in option_text or "评鉴战" in marker) and \
           ("交易" in option_text or "交易" in marker):
            return False
        # 排除重试确认弹窗 ("是否要重新挑战" 会误匹配 _is_event_option_hint("是否"))
        # 否则 EventHandler 点取消 → BattleDefeat 又点重新挑战 → 死循环
        if BattleDefeatHandler._is_retry_dialog(frame, ocr_engine):
            return False
        # 排除卡片奖励选择界面 (标题 "选择奖励" → CardSelectHandler 处理)
        if self._is_card_select_screen(ocr_engine, frame):
            return False

        # 旅程事件标识
        if self._is_journey_marker(marker):
            return True

        # 事件选项提示
        if self._is_event_option_hint(option_text):
            # 排除商店/交易界面: 物品属性加成带+号会误触发事件检测
            if self._is_shop_context(ocr_engine, frame, marker, option_text):
                return False
            # 需要 marker hint 也命中
            if self._contains_journey_hint(marker) and not self._is_journey_noise(marker):
                return True
            # 或者 ≥2 个 +号且 marker 含事件关键词 (排除商店物品的+号误判)
            if option_text.count('+') >= 2 and self._is_journey_marker(marker):
                return True
            # 或者长中文文本(≥30字)且 marker 含事件关键词
            zh_count = sum(1 for c in option_text if '一' <= c <= '鿿')
            if zh_count >= 30 and self._is_journey_marker(marker):
                return True

        # 兜底: 分支文字区包含事件特征词 (捕获选项区 OCR 失败的事件屏)
        branch_text = frame.get_ocr(ocr_engine, 0.08, 0.55, 0.84, 0.30)
        if self._contains_journey_hint(branch_text) and not self._is_journey_noise(branch_text):
            return True

        return False

    def handle(self, ctx: HandlerContext) -> bool:
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False

        if self.config.get('events', {}).get('require_document_effects', False):
            return self._handle_document_event(ctx, screenshot)

        # OCR 选项文字和标记文字
        option_text = _ocr_region(ctx.ocr, screenshot, *self.OPTION_REGIONS[0])
        frame_ctx = FrameContext(screenshot)
        marker_text = self._read_marker(frame_ctx, ctx.ocr)
        num_options = self._count_options(option_text)

        # 保存点击前归一化文本, 用于后续相似度验证
        pre_ocr = self._normalize(option_text)

        # 重复事件防护: 如果与上一次失败的 OCR 文本高度相似, 跳过让 UnknownHandler 恢复
        if pre_ocr and self._last_failed_text:
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, pre_ocr[:60], self._last_failed_text[:60]).ratio()
            if ratio > 0.7:
                print(f"  [事件] 重复事件 (相似度{ratio:.0%}), 跳过...")
                return False

        # 1. 匹配已知事件
        matched = self._match_known_event(option_text) if self._events else None

        if matched:
            event_id = matched.get("id", "unknown")
            recommended = matched.get("recommended_option")
            event_name = matched.get("event_name", event_id)
            # 已知事件的选项数以配置为准: num_options 字段 > options 数组长度 > OCR
            stored_num = matched.get("num_options")
            if stored_num and stored_num > 0:
                num_options = stored_num
            else:
                profile_opts = len(matched.get("options", []))
                if profile_opts > 1 and profile_opts > num_options:
                    num_options = profile_opts
            if recommended is not None:
                selected_option = recommended
                print(f"[事件] {event_name} → 选项{selected_option} (已学习)")
            else:
                selected_option = 1
                status = matched.get("status", "pending")
                print(f"[事件] {event_name} (待定) → 选项{selected_option}")
            if ctx.logger:
                ctx.logger.log_event(
                    event_id=event_id, event_name=event_name,
                    option_selected=selected_option, num_options=num_options,
                    auto_learned=False, matched=True)
        else:
            if option_text.strip():
                matched = self._auto_learn_event(option_text, marker_text, num_options)
            selected_option = 1
            display = matched.get("event_name", "未知") if matched else "未知"
            print(f"[事件] {display} → 选项{selected_option} (自动学习)")
            if ctx.logger and matched:
                ctx.logger.log_event(
                    event_id=matched.get("id", "auto_unknown"),
                    event_name=matched.get("event_name", "Unknown"),
                    option_selected=selected_option, num_options=num_options,
                    auto_learned=True, matched=False)

        # 热键选择: 特定事件使用键盘数字键而非鼠标点击 (如弗洛拉沙滩排球)
        if matched and matched.get("selection_method") == "hotkey":
            return self._handle_hotkey_event(ctx, matched, pre_ocr)

        # 2. 构建尝试顺序: 文档推荐 → 1 → 2 → 3 → 4
        max_opts = max(num_options, 4)
        try_order = []
        if matched and matched.get("recommended_option"):
            rec = matched["recommended_option"]
            if 1 <= rec <= max_opts:
                try_order.append(rec)
        for i in range(1, max_opts + 1):
            if i not in try_order:
                try_order.append(i)
        try_order = [o for o in try_order if o <= max_opts]

        # 3. 验证辅助: 用文本相似度判断事件是否真的变了
        def _char_sim(a: str, b: str) -> float:
            """字符集 Jaccard 相似度"""
            if not a or not b:
                return 0.0
            sa, sb = set(a), set(b)
            if not sa:
                return 0.0
            return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

        def _event_gone(verify_text: str) -> bool:
            vt = self._normalize(verify_text)
            if not vt:
                return True  # 空文本 = 不在事件界面
            sim = _char_sim(pre_ocr, vt)
            return sim < 0.35

        def _try_click(opt_idx: int, num_opts: int) -> bool:
            y = self._get_option_y(num_opts, opt_idx, matched)
            ctx.controller.click_at_percent(self.OPTION_CLICK_X, y)
            time.sleep(0.15)
            ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
            time.sleep(0.15)
            verify_ss = ctx.capture.capture_game()
            if verify_ss is not None:
                verify_text = _ocr_region(ctx.ocr, verify_ss, *self.OPTION_REGIONS[0])
                if _event_gone(verify_text):
                    print(f"  → 选项{opt_idx}/{num_opts} 有效")
                    self._last_failed_text = ""
                    return True
            print(f"  → 选项{opt_idx}/{num_opts} 无效")
            return False

        # 先用 OCR 检测到的布局
        for opt_idx in try_order:
            if _try_click(opt_idx, num_options):
                return True

        # 再尝试其他布局 (OCR 可能漏算/错算选项数)
        for candidate_n in (2, 3, 4):
            if candidate_n == num_options:
                continue
            for opt_idx in try_order:
                if opt_idx > candidate_n:
                    continue
                if _try_click(opt_idx, candidate_n):
                    return True

        print(f"  [事件] 所有选项无效, 兜底推进...")
        self._last_failed_text = pre_ocr
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        time.sleep(0.15)
        ctx.controller.send_escape()
        time.sleep(0.15)
        return True

    def _handle_document_event(self, ctx, screenshot):
        """用标题和所有选项共同核对文档记录；只点一次实测文字中心。"""
        from difflib import SequenceMatcher
        from .recognition import ScreenDetector
        from .controller import InputTargetError
        detector = ScreenDetector(ctx.ocr)
        marker = detector.read_region_text(screenshot, 0.035, 0.14, 0.40, 0.16)
        rows = detector.read_event_option_rows(screenshot)
        normalize = lambda value: re.sub(r'[^\w\u4e00-\u9fff]', '', value)
        reference = self.config.get('events', {}).get('reference_document', '')
        quick_lookup_path = getattr(self, '_quick_lookup_path', Path('profiles/events/document_verified.json'))
        verified = _cached_load_json(quick_lookup_path, 'events')
        matches = []
        matched_rows = {}
        for event in verified:
            if not self._event_source_allowed(event, reference, getattr(ctx, 'development_mode', False)):
                continue
            names = [event.get('event_name', '')] + event.get('title_aliases', [])
            if not any(normalize(name) and normalize(name) in normalize(marker) for name in names):
                continue
            options = event.get('options', [])
            for candidate_rows in self._event_row_variants(rows, len(options)):
                # 支援卡表格通常只记录“事件标题→效果”，不记录游戏内选项原文。
                # 此时标题和选项数量已经核对，允许用数量匹配；有 keyword 的旧条目
                # 仍然要求逐项 OCR 相似，避免把不同事件误合并。
                has_keywords = any(option.get('keyword') or option.get('alias') for option in options)
                rows_match = all(option.get('effect_text') and max(
                    [SequenceMatcher(None, normalize(row['text']), normalize(term)).ratio()
                     for term in [option.get('keyword', '')] + option.get('alias', []) if term]
                ) >= 0.84 for row, option in zip(candidate_rows, options)) if has_keywords else True
                if rows_match:
                    matches.append(event)
                    matched_rows[event.get('id', id(event))] = candidate_rows
                    break
        if len(matches) > 1:
            # The sheet legitimately reuses titles such as “猎头” and
            # “传说之剑” for different support cards.  Prefer a card name
            # visible in the marker; when the client omits it, resolve the
            # verified variants by the same quantitative build score instead
            # of stopping on a known event.
            direction = self._event_build_direction(ctx)
            resolved = self._resolve_document_match(matches, marker, direction)
            if resolved is not None:
                matches = [resolved]

        if len(matches) != 1:
            # 少数训练失败特殊事件不在公开事件表中；沿用原始配置中已确认的安全选项，
            # 明确记录来源，避免把它伪装成文档匹配或盲试其他选项。
            local_match = self._match_known_event(_ocr_region(ctx.ocr, screenshot, *self.OPTION_REGIONS[0]))
            if local_match and local_match.get('status') == 'confirmed' and local_match.get('id') in {'training_failure'}:
                rows = detector.read_event_option_rows(screenshot)
                stamina = self._read_event_stamina(ctx, screenshot)
                index = self._select_training_failure_option(rows, stamina)
                if isinstance(index, int) and 1 <= index <= len(rows) and rows[index - 1].get('enabled'):
                    row = rows[index - 1]
                    rules = self._training_failure_rules()
                    threshold = float(rules.get('high_stamina_threshold', 60))
                    potion_available = bool(
                        len(rows) >= 3 and rows[2].get('enabled') and
                        any(term in rows[2].get('text', '') for term in ('活力', '药水'))
                    )
                    if potion_available and index == int(rules.get('potion_option', 3)):
                        reason = '活力药水可用→选项3'
                    elif stamina >= threshold and index == int(rules.get('high_stamina_option', 2)):
                        reason = f'体力{stamina:.0f}%≥{threshold:.0f}%→选项2'
                    else:
                        reason = f'体力{stamina:.0f}%<{threshold:.0f}%或目标项锁定→选项1'
                    print(f"[事件] {local_match.get('event_name', '训练失败')} → {row['text']}；{reason}；文档未收录该特例，使用本地规则")
                    ctx.controller.click_at_percent(0.80, row['y'])
                    ctx.pending_screenshot = ctx.capture.capture_game()
                    if ctx.logger:
                        ctx.logger.log_event(event_id=local_match['id'], event_name=local_match.get('event_name', '训练失败'),
                            option_selected=index, num_options=len(rows), auto_learned=False, matched=False)
                    return True
            folder = Path('../verification/pending-event')
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / 'current.png'), screenshot)
            (folder / 'current.json').write_text(json.dumps({'marker': marker, 'options': rows,
                'reference_document': reference, 'input_sent': False}, ensure_ascii=False, indent=2), encoding='utf-8')
            raise EventDecisionPending(f'事件「{marker}」尚未核对文档效果，已暂停，未选择选项。')
        event = matches[0]
        selected_rows = matched_rows.get(event.get('id', id(event)), rows)
        direction = self._event_build_direction(ctx)
        weights = self.config.get('events', {}).get('decision_weights', {}).get(direction)
        local_rule = not event.get('document_verified')
        if local_rule:
            # 本地/画面规则只在开发模式按明确决策使用，未知效果不冒充零收益评分。
            enabled = [i for i, row in enumerate(selected_rows, 1) if row.get('enabled')]
            if event.get('development_decision') == 'only_enabled':
                if len(enabled) != 1:
                    raise EventDecisionPending('画面规则要求唯一可用选项，当前条件不满足。')
                index = enabled[0]
            else:
                index = event.get('recommended_by_direction', {}).get(direction, event.get('recommended_option'))
            scored = []
        else:
            index, scored = choose_option(event, direction, weights)
        # 先按量化效果评分，再过滤当前画面中不可用/锁定的选项。
        # 例如“探索迷宫”选项2在没有对应条件时会显示灰色，不能因
        # 理论最高分而暂停整个开发流程。
        enabled_indices = {
            row_index for row_index, row in enumerate(selected_rows, start=1)
            if row.get('enabled')
        }
        if index not in enabled_indices:
            enabled_scored = [item for item in scored if item['index'] in enabled_indices]
            if enabled_scored:
                best_enabled = max(enabled_scored, key=lambda item: (
                    item['score'], item['potential_count'], item['buff_count'],
                    -item['unknown_count'], -item['index'],
                ))
                index = best_enabled['index']
            else:
                # Local development rules may include a documented fallback
                # for a locked preferred branch (for example, a conditional
                # event whose first two choices require an item/stat).
                fallback = event.get('fallback_option') if local_rule else None
                index = fallback if fallback in enabled_indices else event.get('recommended_option')
        if not isinstance(index, int) or not 1 <= index <= len(selected_rows) or not selected_rows[index - 1]['enabled']:
            raise EventDecisionPending('文档推荐选项无法使用，暂停等待处理。')
        row = selected_rows[index - 1]
        effect = event['options'][index - 1].get('effect_text', '')
        score_text = ''
        if scored:
            score_text = '；量化评分=' + ','.join(
                f"{item['index']}:{item['score']}" for item in scored)
        source_text = '；开发本地规则，效果尚未文档核验' if local_rule else ''
        print(f"[事件] {event['event_name']} → {row['text']}；效果：{effect}{score_text}{source_text}")
        ctx.controller.click_at_percent(0.80, row['y'])
        deadline = time.perf_counter() + 6
        while time.perf_counter() < deadline:
            ctx.controller.wait(0.4)
            current = ctx.capture.capture_game()
            if current is None:
                continue
            current_rows = detector.read_event_option_rows(current)
            if [normalize(r['text']) for r in current_rows] != [normalize(r['text']) for r in rows]:
                ctx.pending_screenshot = current
                if ctx.logger:
                    ctx.logger.log_event(event_id=event['id'], event_name=event['event_name'],
                        option_selected=index, num_options=len(selected_rows), auto_learned=False, matched=not local_rule)
                return True
        raise InputTargetError('事件选择后界面未变化，停止重复或改点其他选项。')

    @staticmethod
    def _resolve_document_match(candidates: list[dict], marker: str,
                                direction: str) -> dict | None:
        """Resolve duplicate sheet titles without pretending they are unknown.

        Support-card events can share an event title while their effects come
        from different cards.  The card name is sometimes present in OCR; if
        it is absent, use the verified effect score as a deterministic
        fallback and leave the choice auditable in the log.
        """
        if not candidates:
            return None
        normalize = lambda value: re.sub(r'[^\w\u4e00-\u9fff]', '', str(value or ''))
        marker_norm = normalize(marker)
        card_hits = []
        for event in candidates:
            card = normalize(event.get('card', ''))
            if card and card in marker_norm:
                card_hits.append(event)
                continue
            # The marker often keeps only the character after the dash.
            tail = re.split(r'[-－]', str(event.get('card', '')))[-1]
            tail = normalize(tail)
            if len(tail) >= 2 and tail in marker_norm:
                card_hits.append(event)
        if len(card_hits) == 1:
            return card_hits[0]

        from .event_effects import score_options
        ranked = []
        for event in candidates:
            scored = score_options(event, direction)
            scores = [item['score'] for item in scored]
            ranked.append((max(scores, default=float('-inf')),
                           sum(scores), -len(event.get('options', [])), event))
        ranked.sort(key=lambda item: item[:3], reverse=True)
        chosen = ranked[0][3]
        if len(candidates) > 1:
            print(f"[事件] 标题重复，按{direction}向量化效果选择："
                  f"{chosen.get('event_name', '')} / {chosen.get('card', '未知卡')}")
        return chosen

    @staticmethod
    def _event_source_allowed(event: dict, reference: str, development_mode: bool) -> bool:
        if event.get('document_verified'):
            return EventHandler._document_source_matches(event, reference)
        return bool(development_mode and event.get('source_type') in {'local_rule', 'screen_observed'}
                    and event.get('development_decision') in {'explicit_recommendation', 'only_enabled'}
                    and event.get('source_evidence'))

    @staticmethod
    def _document_source_matches(event: dict, reference: str) -> bool:
        """允许同一腾讯文档不同 tab 锚点链接互相核对。"""
        references = [event.get('source_document', '')] + event.get('source_document_aliases', [])
        if reference in references:
            return True
        from urllib.parse import urlsplit
        def base(value):
            parsed = urlsplit(str(value or ''))
            return (parsed.scheme, parsed.netloc, parsed.path) if parsed.scheme else str(value)
        ref_base = base(reference)
        return bool(ref_base) and any(base(value) == ref_base for value in references if value)

    @staticmethod
    def _event_build_direction(ctx) -> str:
        """读取当前训练构筑方向，缺少引擎上下文时使用攻击向默认。"""
        try:
            value = ctx.engine.rule_profile.legacy_strategy.build_direction.value
            return value if value in {'attack', 'survival'} else 'attack'
        except Exception:
            return 'attack'

    @staticmethod
    def _event_row_variants(rows, target_count):
        """生成事件选项行候选，兼容一条选项被 OCR 拆成两行的情况。"""
        from itertools import combinations
        if target_count <= 0 or len(rows) < target_count:
            return []
        if len(rows) == target_count:
            return [rows]

        variants = []
        for cuts in combinations(range(1, len(rows)), target_count - 1):
            bounds = (0,) + cuts + (len(rows),)
            merged = []
            for start, end in zip(bounds, bounds[1:]):
                group = rows[start:end]
                merged.append({
                    'text': ''.join(row.get('text', '') for row in group),
                    'y': sum(float(row.get('y', 0)) for row in group) / len(group),
                    'x': sum(float(row.get('x', 0.8)) for row in group) / len(group),
                    'enabled': all(row.get('enabled', True) for row in group),
                    'confidence': min(float(row.get('confidence', 0)) for row in group),
                })
            variants.append(merged)
        return variants

    def _training_failure_rules(self) -> dict:
        """读取训练失败的动态规则；阈值保存在静态快查 JSON 中。"""
        quick_lookup_path = getattr(self, '_quick_lookup_path', Path('profiles/events/document_verified.json'))
        rules = _cached_load_json_value(quick_lookup_path, 'rules', {})
        if not isinstance(rules, dict):
            rules = {}
        event_rules = rules.get('training_failure', rules)
        return event_rules if isinstance(event_rules, dict) else {}

    def _read_event_stamina(self, ctx, screenshot) -> float:
        """读取事件出现瞬间的体力，优先 HUD 状态，最后才用缓存/绿色条估算。"""
        fallback = getattr(ctx, '_last_stamina', None)
        ui = getattr(ctx, 'ui', None)
        if ui is not None and hasattr(ui, 'read_game_state'):
            try:
                state = ui.read_game_state(
                    screenshot, fallback, getattr(ctx, '_last_mood', None))
                value = state.get('stamina') if isinstance(state, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return max(0.0, min(100.0, float(value)))
            except Exception:
                pass
        if isinstance(fallback, (int, float)) and not isinstance(fallback, bool):
            return max(0.0, min(100.0, float(fallback)))
        try:
            from .recognition import estimate_stamina_bar_pct
            value = estimate_stamina_bar_pct(screenshot)
            if isinstance(value, (int, float)):
                return max(0.0, min(100.0, float(value)))
        except Exception:
            pass
        return 50.0

    def _select_training_failure_option(self, rows, stamina: float) -> int | None:
        """按道具和体力选择训练失败事件选项，并跳过锁定行。

        选项3只有在当前画面明确可用时才会进入候选；没有药水时，
        体力达到阈值选2重试，否则选1休息。候选顺序允许在 OCR 锁定
        误判时安全回退到另一个可用选项。
        """
        if not rows:
            return None
        rules = self._training_failure_rules()
        threshold = float(rules.get('high_stamina_threshold', 60))
        potion_idx = int(rules.get('potion_option', 3))
        high_idx = int(rules.get('high_stamina_option', 2))
        low_idx = int(rules.get('low_stamina_option', 1))

        def enabled(index: int) -> bool:
            return 1 <= index <= len(rows) and rows[index - 1].get('enabled', True) is not False

        potion_available = enabled(potion_idx) and any(
            term in rows[potion_idx - 1].get('text', '') for term in ('活力', '药水'))
        if potion_available:
            preferred = [potion_idx, low_idx, high_idx]
        elif float(stamina) >= threshold:
            preferred = [high_idx, low_idx]
        else:
            preferred = [low_idx, high_idx]

        # 最后按安全顺序兜底，永不点击 enabled=False 的行。
        order = preferred + [1, 2, 3]
        seen = set()
        for index in order:
            if index in seen:
                continue
            seen.add(index)
            if enabled(index):
                return index
        return None

    def _get_option_y(self, num_options: int, option_index: int, matched_event: dict = None) -> float:
        """根据选项数和选项序号获取点击 Y 坐标, 支持事件级 Y 覆盖.

        统一使用 4-选项布局通过偏移量映射:
        选项在游戏中底部对齐, N 选项事件的选项位于 4 选项布局的最后 N 个位置.
        例如: 2 选项事件 → 选项 1,2 映射到 4 选项的 3,4 号位置.
        """
        # 事件级 Y 覆盖优先
        if matched_event and "option_y_overrides" in matched_event:
            overrides = matched_event["option_y_overrides"]
            idx = option_index - 1
            if 0 <= idx < len(overrides):
                return overrides[idx]
        # 统一映射: option_index + (4 - num_options) → 4-选项位置
        four_idx = option_index + (4 - num_options) - 1
        if 0 <= four_idx < 4:
            return self.FOUR_OPTION_CENTERS[four_idx]
        return 0.52 + (option_index - 1) * 0.07

    @staticmethod
    def _normalize(text: str) -> str:
        if not text: return ""
        return text.replace(" ", "").replace("\r", "").replace("\n", "").replace("　", "").strip()

    def _handle_hotkey_event(self, ctx: HandlerContext, matched: dict, pre_ocr: str) -> bool:
        """热键事件处理: 使用键盘数字键选择选项, 支持耐力条件判断.
        用于沙滩排球等特定事件, 不依赖鼠标点击坐标.
        """
        event_name = matched.get("event_name", "unknown")
        stamina_rule = matched.get("stamina_rule", {})
        threshold = stamina_rule.get("high_stamina_threshold", 80)
        opt_high = stamina_rule.get("option_when_high_stamina", 2)
        opt_default = stamina_rule.get("default_option", 1)
        opt_fallback = stamina_rule.get("fallback_option", 3)

        # 读取当前体力
        stamina = getattr(ctx, '_last_stamina', None) or 50

        # 根据体力决定目标选项
        if stamina > threshold:
            target_option = opt_high
            rule_reason = f"耐力{stamina:.0f}%>{threshold}%→选项{target_option}"
        else:
            target_option = opt_default
            rule_reason = f"耐力{stamina:.0f}%≤{threshold}%→选项{target_option}"

        print(f"[事件] {event_name} 热键 | {rule_reason}")

        # 发送热键
        ctx.controller.send_number_key(target_option)
        time.sleep(0.3)

        # 验证事件是否消失
        verify_ss = ctx.capture.capture_game()
        if verify_ss is not None:
            verify_text = _ocr_region(ctx.ocr, verify_ss, *self.OPTION_REGIONS[0])
            vt = self._normalize(verify_text)
            if vt:
                pre_norm = self._normalize(pre_ocr)
                sa, sb = set(pre_norm), set(vt)
                sim = len(sa & sb) / len(sa | sb) if (sa | sb) else 0
                if sim < 0.35:
                    print(f"  → 热键{target_option} 有效")
                    self._last_failed_text = ""
                    if ctx.logger:
                        ctx.logger.log_event(
                            event_id=matched.get("id", ""), event_name=event_name,
                            option_selected=target_option,
                            num_options=matched.get("num_options", 3),
                            auto_learned=False, matched=True)
                    return True

        # 选项2无效 → 回退选项3
        if target_option == opt_high and opt_fallback:
            print(f"  → 热键{target_option} 无效, 回退选项{opt_fallback}")
            ctx.controller.send_number_key(opt_fallback)
            time.sleep(0.3)
            verify_ss = ctx.capture.capture_game()
            if verify_ss is not None:
                verify_text = _ocr_region(ctx.ocr, verify_ss, *self.OPTION_REGIONS[0])
                vt = self._normalize(verify_text)
                if not vt:
                    print(f"  → 热键{opt_fallback} 有效")
                    self._last_failed_text = ""
                    if ctx.logger:
                        ctx.logger.log_event(
                            event_id=matched.get("id", ""), event_name=event_name,
                            option_selected=opt_fallback,
                            num_options=matched.get("num_options", 3),
                            auto_learned=False, matched=True)
                    return True

        # 兜底: 仍无效则推进
        print(f"  → 热键无效, 兜底推进...")
        self._last_failed_text = pre_ocr
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        time.sleep(0.15)
        return True

    def _read_marker(self, frame, engine) -> str:
        """合并全部旅程标识区域的OCR文字 (单区域可能遗漏"旅程事件"关键词)"""
        texts = []
        for rx, ry, rw, rh in self.JOURNEY_REGIONS:
            text = frame.get_ocr(engine, rx, ry, rw, rh)
            if text:
                texts.append(text)
        return " ".join(texts)

    def _read_options(self, frame, engine) -> str:
        for rx, ry, rw, rh in self.OPTION_REGIONS:
            text = frame.get_ocr(engine, rx, ry, rw, rh)
            if text: return text
        return ""

    @staticmethod
    def _is_journey_marker(text: str) -> bool:
        if not text: return False
        if "旅程事件" in text: return True
        if "阿尔克那事件" in text or "阿克那事件" in text: return True
        if "支援卡事件" in text: return True
        if "救援者事件" in text: return True
        if "训练失败" in text: return True
        return "旅程" in text and "事件" in text

    @staticmethod
    def _contains_journey_hint(text: str) -> bool:
        if not text: return False
        return any(kw in text for kw in [
            "旅程", "事件", "距离目标", "选择项", "选项",
            "育成", "剧情", "固定事件", "连续事件", "花样追加",
            "阿尔克那", "支援卡", "救援者",
        ])

    @staticmethod
    def _is_journey_noise(text: str) -> bool:
        if not text: return False
        return bool(re.search(r'\d{1,2}月(上旬|中旬|下旬)', text))

    @staticmethod
    def _is_card_select_screen(ocr_engine, frame) -> bool:
        """OCR 标题区域检测是否为卡片奖励选择界面, 防止 EventHandler 抢卡片."""
        try:
            title = frame.get_ocr(ocr_engine, 0.02, 0.02, 0.28, 0.10)
            return bool(title) and ("选择奖励" in title or "奖励选择" in title or
                                    ("选择" in title and "奖励" in title))
        except Exception:
            return False

    @staticmethod
    def _is_event_option_hint(text: str) -> bool:
        if not text: return False
        # ≥2 个 + 号或 ≥2 个 。 选项分隔符
        if text.count('+') >= 2: return True
        if text.count('。') >= 2: return True
        if text.count('选') >= 2: return True
        event_hints = ["该怎么办", "怎么办", "请选择", "选项",
                       "是否", "要不要", "等待下次机会", "下次机会",
                       "训练教材", "有帮助的", "有助于"]  # 方向选择类事件
        for hint in event_hints:
            if hint in text: return True
        # 中文 8 字 + 问号/呢/吧
        zh_count = sum(1 for c in text if '一' <= c <= '鿿')
        if zh_count >= 8 and any(c in text for c in "？?呢吧"):
            return True
        # 长中文文本(≥40字) → 大概率是事件选项 (排除训练/休息/菜单)
        if zh_count >= 40:
            return True
        return False

    @staticmethod
    def _is_training_context(marker: str, option: str) -> bool:
        # marker 包含事件关键词 → 不是训练界面 (天气事件选项常含"继续训练", 不能仅靠选项排除)
        event_marker_hints = ["今日天气", "旅程事件", "阿尔克那事件", "阿克那事件",
                               "支援卡事件", "救援者事件", "固定事件", "连续事件"]
        if any(h in marker for h in event_marker_hints):
            return False
        # 选项包含事件提示词 → 非训练界面
        event_option_hints = ["教材", "方向性", "有帮助的", "有助于", "该怎么", "怎么办",
                               "回旅馆", "要不要", "是否"]
        if any(h in option for h in event_option_hints):
            return False
        # 训练关键词: 需要 ≥2 个匹配才判定为训练界面 (避免单一"继续训练"误伤天气事件)
        train_kw = ["力量训练", "体力训练", "韧性训练", "集中训练", "专注训练", "保护训练",
                     "继续训练", "返回住处"]
        train_hits = sum(1 for kw in train_kw if kw in option)
        if train_hits >= 2:
            return True
        return "训练" in marker and any(
            kw in option for kw in ["训练", "力量", "体力", "韧性", "集中", "专注", "保护"])

    @staticmethod
    def _is_main_menu_like(text: str) -> bool:
        if not text: return False
        menu_kw = ["训练", "委托", "讨伐", "休息", "评鉴战", "交易", "出击"]
        count = sum(1 for kw in menu_kw if kw in text)
        return count >= 2

    @staticmethod
    def _is_rest_context(text: str) -> bool:
        if not text: return False
        rest_kw = ["免费住处", "冥想室", "冥想", "露宿"]
        has_rest = any(kw in text for kw in rest_kw)
        has_price = "30" in text or "60" in text
        return has_rest and has_price

    @staticmethod
    def _is_appraise_goal(text: str) -> bool:
        if not text: return False
        if "回合以内获胜" in text or "回合内获胜" in text:
            return True
        return len(re.findall(r'\d{1,3}/\d{1,3}', text)) >= 2

    @staticmethod
    def _is_trade_screen(ocr_engine, frame) -> bool:
        """通过 OCR 购买按钮区域检测是否为交易界面, 防止误匹配事件选择.

        使用与 recognition.ScreenDetector.is_trade_screen 一致的宽区域,
        同时检查 "购买"/"購入"/"SOLDOUT" 等多种信号, 避免 OCR 漏字导致漏判.
        """
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        # 用与 recognition.py 一致的扩展区域 (0.75, 0.80, 0.22, 0.18)
        buy_text = detector.read_region_text(frame.screenshot, 0.75, 0.80, 0.22, 0.18)
        if "购买" in buy_text or "購入" in buy_text or "购" in buy_text or "買" in buy_text:
            return True
        if "SOLDOUT" in buy_text.upper() or "SOLD" in buy_text.upper():
            return True
        # 回退: 用 ScreenDetector 的完整逻辑 (含弱信号)
        return detector.is_trade_screen(frame.screenshot)

    @classmethod
    def _is_shop_context(cls, ocr_engine, frame, marker: str, option_text: str) -> bool:
        """检测是否为商店/交易场景, 防止物品属性加成(+号)误触发事件检测."""
        # 直接检查 marker + option_text 组合
        combined = f"{marker} {option_text}"
        shop_kw = ["交易", "购买", "購入", "评鉴战", "评鉴", "古币", "商店",
                    "SOLDOUT", "SOLD", "资金", "结余"]
        hits = sum(1 for kw in shop_kw if kw in combined)
        if hits >= 2:
            return True
        # 检查购买按钮区域
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        buy_text = detector.read_region_text(frame.screenshot, 0.75, 0.80, 0.22, 0.18)
        if "购买" in buy_text or "購入" in buy_text or "购" in buy_text or "買" in buy_text:
            return True
        return False

    @staticmethod
    def _count_options(text: str) -> int:
        """从 OCR 文本数选项数量 (通过分隔符)"""
        if not text:
            return 1
        # + 号分隔 (典型事件格式)
        if text.count('+') >= 1:
            return text.count('+') + 1
        # 。 句号分隔 (选项文字 OCR 常丢失 + 号读成句号)
        period_count = text.count('。')
        if period_count >= 1:
            return period_count
        # 选 字开头 (OCR 误读 + 为选项内容)
        select_count = text.count('选')
        if select_count >= 2:
            return select_count
        return 1


# ============================================================
# RewardAdvanceHandler (Priority 9.5) - 奖励详情继续
# ============================================================

class RewardAdvanceHandler(Handler):
    """推进获得遗物/旅程奖励详情页，避免把已知奖励页当未知。"""

    @property
    def priority(self): return 9.5

    @property
    def name(self): return "奖励继续"

    def can_handle(self, frame, ocr_engine) -> bool:
        if frame.screenshot is None:
            return False
        try:
            text = " ".join(item[0] for item in frame.recognize_full(ocr_engine))
        except Exception:
            text = ""
        normalized = re.sub(r"\s+", "", text)
        if "获得奖励" in normalized and "点击以继续" in normalized:
            self._result_kind = "reward"
            return True
        # 休息/住处结算页使用英文 GREAT SUCCESS，但同样只有一个底部继续按钮。
        # 若不单独识别，该页面会落入 UnknownHandler 并连续等待。
        if "SUCCESS" in normalized.upper() and "点击以继续" in normalized:
            self._result_kind = "rest" if "GREAT" in normalized.upper() else "battle"
            return True
        if "新伙伴加入" in normalized and "点击以继续" in normalized:
            self._result_kind = "companion"
            return True
        return False

    def handle(self, ctx: HandlerContext) -> bool:
        if getattr(self, "_result_kind", "reward") == "rest":
            print("[休息结算] GREAT SUCCESS → 点击底部继续")
        elif getattr(self, "_result_kind", "reward") == "battle":
            print("[战斗结算] SUCCESS → 点击底部继续")
        elif getattr(self, "_result_kind", "reward") == "companion":
            print("[伙伴加入] 新伙伴加入 → 点击底部继续")
        else:
            print("[奖励继续] 获得奖励详情 → 点击底部继续")
        ctx.controller.click_at_percent(0.50, 0.89)
        time.sleep(0.6)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


# ============================================================
# CardSelectHandler (Priority 10) - 卡片选择
# ============================================================

class CardSelectHandler(Handler):
    """
    奖励卡片选择界面。

    移植自 SleepRunner CardSelectHandler:
    - 左上角标题 OCR 检测 "选择奖励"
    - 3 张卡片, 每张 OCR 文字后按优先级排序
    - 速度最优先 (可配置), 攻击向优先暴击→爆伤→攻击, 生存向优先生命→防御
    - 都不匹配时选金额最大的卡片
    - 点击卡片 → 等 SelectDone 按钮亮起 → 点击确认
    """

    # 标题检测区域 (SleepRunner: TitleRegion)
    TITLE_REGION = (0.02, 0.02, 0.28, 0.10)

    # 3 张卡片点击位置 (SleepRunner: CardClickPercents)
    CARD_CLICK_POINTS = [
        (0.20, 0.48),   # Card 1
        (0.50, 0.48),   # Card 2
        (0.80, 0.48),   # Card 3
    ]

    # 3 张卡片 OCR 区域 (扩边减少边缘裁切)
    CARD_TEXT_REGIONS = [
        (0.04, 0.24, 0.30, 0.42),
        (0.35, 0.24, 0.30, 0.42),
        (0.66, 0.24, 0.30, 0.42),
    ]

    # 确认按钮 (SleepRunner: SelectDone)
    SELECT_DONE_X = 0.47
    SELECT_DONE_Y = 0.83

    # 不选按钮 (SleepRunner: Unselected)
    UNSELECTED_X = 0.884
    UNSELECTED_Y = 0.057

    @property
    def priority(self): return 10

    @property
    def name(self): return "卡片选择"

    def __init__(self, config=None):
        self.config = config or {}
        card_cfg = self.config.get("card_select", {})
        self.speed_first = card_cfg.get("speed_first", True)
        self.attack_priority = card_cfg.get("attack_priority", ["暴击率", "暴击伤害", "攻击力"])
        self.survival_priority = card_cfg.get("survival_priority", ["生命力", "防御力"])

    def can_handle(self, frame, ocr_engine) -> bool:
        title = frame.get_ocr(ocr_engine, *self.TITLE_REGION)

        # 排除事件屏幕: 如果存在旅程事件标记, 即使标题匹配也是事件不是卡片
        if self._has_event_marker(frame, ocr_engine):
            return False

        # 主检测: 标题区有"选择奖励"
        if self._is_card_select_title(title):
            # 排除主菜单: 三张"卡片"中任一张包含菜单关键词 → 不是奖励界面
            for rx, ry, rw, rh in self.CARD_TEXT_REGIONS:
                text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
                if text and self._is_main_menu_card(text):
                    return False
            return True

        # 兜底检测: 标题 OCR 可能失败, 检查卡片内容区是否有卡牌效果文字
        # 如果 ≥2 个卡片区域包含卡牌特征词且不含菜单词 → 大概率是卡片选择界面
        card_like_count = 0
        for rx, ry, rw, rh in self.CARD_TEXT_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            # 先排除主菜单 (RANK/属性面板也会命中卡牌特征词, 误判率很高)
            if text and self._is_main_menu_card(text):
                return False
            if text and self._looks_like_card_effect(text):
                card_like_count += 1
        if card_like_count >= 2:
            return True

        return False

    def handle(self, ctx: HandlerContext) -> bool:
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False

        # 读取 3 张卡片文字
        card_texts = []
        for i, (rx, ry, rw, rh) in enumerate(self.CARD_TEXT_REGIONS):
            text = _ocr_region(ctx.ocr, screenshot, rx, ry, rw, rh)
            card_texts.append(text)

        # 获取构建方向
        build_direction = "attack"
        try:
            if ctx.engine.is_rule_engine_mode:
                build_direction = ctx.engine.rule_profile.legacy_strategy.build_direction.value
        except Exception:
            pass

        # 确定优先级列表
        priority_keywords = self._get_priority_keywords(build_direction)

        # 评分排序
        ranked = self._rank_cards(card_texts, priority_keywords)

        if ranked:
            best_slot = ranked[0][0]
            reason = f"优先级匹配 (rank={ranked[0][1]})"
            print(f"[卡片] {build_direction} → #{best_slot + 1} (评分最高)")
        else:
            best_slot = self._pick_richest(card_texts)
            reason = "无匹配, 选金额最大"
            print(f"[卡片] {build_direction} → #{best_slot + 1} (金额最大)")

        if ctx.logger:
            ctx.logger.log_card_select(best_slot, reason, card_texts)

        # 点击卡片
        cx, cy = self.CARD_CLICK_POINTS[best_slot]
        ctx.controller.click_at_percent(cx, cy)
        time.sleep(0.3)

        # 点击确认按钮
        ctx.controller.click_at_percent(self.SELECT_DONE_X, self.SELECT_DONE_Y)
        time.sleep(0.3)

        return True

    def _get_priority_keywords(self, build_direction: str) -> list:
        """根据构建方向获取优先级关键词列表"""
        keywords = []
        if self.speed_first:
            keywords.append("速度")
        if build_direction == "attack":
            keywords.extend(self.attack_priority)
        else:
            keywords.extend(self.survival_priority)
        return keywords

    def _rank_cards(self, card_texts: list, priority_keywords: list) -> list:
        """按优先级排序卡片, 返回 [(slot_index, rank, score), ...]"""
        ranked = []
        for i, text in enumerate(card_texts):
            if not text:
                continue
            rank = self._find_keyword_rank(text, priority_keywords)
            if rank >= 0:
                # 同时提取金额作为二级排序
                money = self._extract_money(text)
                ranked.append((i, rank, money))
        # 按 rank 升序, rank 相同按 money 降序
        ranked.sort(key=lambda x: (x[1], -x[2]))
        return ranked

    # 关键词别名: OCR可能出现的缩写/变体 → 标准关键词
    KEYWORD_ALIASES = {
        "暴伤": "暴击伤害",
        "爆伤": "暴击伤害",
        "暴击概率": "暴击率",
        "会心率": "暴击率",
        "会心": "暴击率",
    }

    # 单字匹配表: 关键词包含这些核心字时视为匹配 (用于OCR漏字场景)
    KEYWORD_CORE_CHARS = {
        "速度": ["速"],
        "暴击率": ["暴", "击率", "会心"],
        "暴击伤害": ["暴", "暴伤", "爆伤", "击伤害"],
        "攻击力": ["攻击", "击力"],
        "生命力": ["生命", "命力"],
        "防御力": ["防御", "御力"],
    }

    @staticmethod
    def _fuzzy_match_keyword(text: str, kw: str) -> bool:
        """模糊匹配关键词,容忍OCR漏字/错字"""
        if not text:
            return False
        # 1. 精确匹配
        if kw in text:
            return True
        # 2. 去空格匹配
        compact = text.replace(" ", "").replace("　", "")
        if kw in compact:
            return True
        # 3. 核心字匹配 (OCR可能丢失个别字符)
        core_chars = CardSelectHandler.KEYWORD_CORE_CHARS.get(kw, [])
        for core in core_chars:
            if core in compact:
                return True
        # 4. 别名匹配
        for alias, canonical in CardSelectHandler.KEYWORD_ALIASES.items():
            if canonical == kw and alias in compact:
                return True
        return False

    @staticmethod
    def _find_keyword_rank(text: str, keywords: list) -> int:
        """在文字中查找关键词, 返回优先级序号 (0=最高), -1=未找到"""
        if not text:
            return -1
        for rank, kw in enumerate(keywords):
            if CardSelectHandler._fuzzy_match_keyword(text, kw):
                return rank
        return -1

    @staticmethod
    def _extract_money(text: str) -> int:
        """从卡片文字提取金额"""
        if not text:
            return 0
        # 匹配 G/金/钱 前面的数字
        import re
        # 如 "100G", "50金", "$30"
        match = re.search(r'(\d{1,4})\s*[G金]', text)
        if match:
            return int(match.group(1))
        # 纯大数字
        nums = re.findall(r'\d{2,4}', text)
        if nums:
            vals = [int(n) for n in nums if 10 <= int(n) <= 500]
            if vals:
                return max(vals)
        return 0

    @staticmethod
    def _pick_richest(card_texts: list) -> int:
        """选钱最多的卡片"""
        best_slot = 0
        best_money = 0
        for i, text in enumerate(card_texts):
            money = CardSelectHandler._extract_money(text)
            if money > best_money:
                best_money = money
                best_slot = i
        return best_slot

    @staticmethod
    def _is_card_select_title(title: str) -> bool:
        if not title:
            return False
        return ("选择奖励" in title or "奖励选择" in title or
                ("选择" in title and "奖励" in title))

    @staticmethod
    def _is_main_menu_card(text: str) -> bool:
        """单张"卡片"是否实际是主菜单/训练屏元素 (排除用)."""
        if not text:
            return False
        # "/1250" 是训练/菜单属性面板的强指纹, 出现即排除
        if "/1250" in text:
            return True
        # 数值/上限 模式 (如 552/1250, 98/1200) — 属性面板强指纹, OCR 可能漏掉具体数字
        import re
        if re.search(r'\d{2,4}\s*/\s*\d{2,4}', text):
            return True
        menu_hints = ["训练", "委托", "休息", "讨伐", "评鉴战", "交易", "出击",
                      "RANK"]
        count = sum(1 for kw in menu_hints if kw in text)
        return count >= 2

    @staticmethod
    def _looks_like_card_effect(text: str) -> bool:
        """检测文字是否像卡牌效果描述 (兜底卡片界面识别)."""
        if not text:
            return False
        card_hints = [
            "队员全体", "自身", "开始时", "发动时", "训练",
            "速度", "力量", "体力", "韧性", "专注", "保护",
            "防御力", "攻击力", "生命力", "暴击率", "暴击伤害",
            "％", "技能", "能力",
        ]
        count = sum(1 for kw in card_hints if kw in text)
        return count >= 2

    # 事件标记检测区域 (与 EventHandler.JOURNEY_REGIONS 一致)
    _EVENT_MARKER_REGIONS = [
        (0.00, 0.00, 0.22, 0.08),
        (0.02, 0.20, 0.38, 0.16),
        (0.00, 0.28, 0.68, 0.34),
    ]

    # 事件特征词
    _EVENT_HINTS = ["旅程事件", "花样追加", "固定事件", "支援卡", "故事",
                    "连续事件", "育成", "剧情", "竞赛", "大奖赛",
                    "休息", "外出", "保健室", "新年", "夏合宿"]

    @classmethod
    def _has_event_marker(cls, frame, ocr_engine) -> bool:
        """检查屏幕是否包含旅程事件标记 (用于卡片界面排除事件屏)."""
        for rx, ry, rw, rh in cls._EVENT_MARKER_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if text:
                for hint in cls._EVENT_HINTS:
                    if hint in text:
                        return True
        return False


# ============================================================
# TrainingHandler (Priority 6) - 训练决策
# ============================================================

class TrainingHandler(Handler):
    """
    训练界面处理器。
    can_handle 检测训练屏；handle 委托给 trainer 的复杂逻辑。
    """

    @property
    def priority(self): return 6

    @property
    def name(self): return "训练决策"

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        return detector.is_training_screen(frame.screenshot)

    def handle(self, ctx: HandlerContext) -> bool:
        # 训练逻辑复杂, 标记返回 False 让 trainer 自己接管
        return False


# ============================================================
# RestHandler (Priority 18) - 休息决策
# ============================================================

class RestHandler(Handler):
    """
    休息界面处理器。

    移植自 SleepRunner RestDecisionHandler:
    - 读取金钱
    - 预算制选择: 有钱→冥想室(60G), 一般→住处(30G), 没钱→露宿(免费)
    - 点击选项 → 点击确认
    - 如果付费休息失败 (古币不足), 根据体力条估算值回退:
      ≥70% → 退出休息继续训练, <70% → 露宿(免费)
    - 连续调用检测: 防止付费失败导致的死循环
    """

    # 休息选项点击位置 (露宿 / 住处 / 冥想室) — 已通过 OCR 校准
    REST_CLICK_POINTS = [
        (0.83, 0.37),   # 方案1: 露宿 (免费)
        (0.84, 0.47),   # 方案2: 住处 ($30)
        (0.85, 0.57),   # 方案3: 冥想室 ($60)
    ]
    CONFIRM_X = 0.89
    CONFIRM_Y = 0.925

    # 回退阈值: 体力≥此值且没钱付费休息 → 继续训练而非免费休息
    STAMINA_FALLBACK_THRESHOLD = 80

    def __init__(self, config=None):
        self._consecutive_calls = 0
        self._stamina_continue = 80
        self._first_year_rounds = 0
        self._best_mood_free_stamina = 40
        if config:
            rest_cfg = config.get("rest", {})
            self._stamina_continue = rest_cfg.get("stamina_continue_threshold", 80)
            self._first_year_rounds = rest_cfg.get("first_year_rounds", 0)
            self._best_mood_free_stamina = rest_cfg.get("selection", {}).get(
                "best_mood_free_stamina", 40)

    @property
    def priority(self): return 18

    @property
    def name(self): return "休息决策"

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        if detector.is_rest_screen(frame.screenshot):
            self._consecutive_calls += 1
            return True
        self._consecutive_calls = 0
        return False

    def handle(self, ctx: HandlerContext) -> bool:
        screenshot = ctx.capture.capture_game()

        from .recognition import estimate_stamina_bar_pct
        stamina_pct = estimate_stamina_bar_pct(screenshot) if screenshot is not None else 50

        money = 0
        if screenshot is not None:
            raw = ctx.ui.read_game_state(screenshot)
            money = raw.get("money", 0)
            if money > 0:
                ctx._last_money = money
        if money <= 0:
            money = ctx._last_money

        # 读取心情和回合
        mood = getattr(ctx, '_last_mood', 'Normal') or 'Normal'
        turn = getattr(ctx, 'detected_turn', None)

        names = ["露宿(免费)", "住处($30)", "冥想室($60)"]

        # 连续调用 ≥2 → 付费失败循环, 强制回退
        if self._consecutive_calls >= 2:
            self._consecutive_calls = 0
            return self._handle_fallback(ctx, stamina_pct)

        # ── 心情规则 (config.yaml rest.selection) ──
        is_first_year = self._first_year_rounds > 0 and turn is not None and turn <= self._first_year_rounds

        if mood == "Best":
            if stamina_pct > self._best_mood_free_stamina:
                option_idx = 0  # 规则1: Best + 耐力>40% → 免费
                reason = f"Best+耐力{stamina_pct}%>{self._best_mood_free_stamina}%→免费"
            else:
                option_idx = 2 if money >= 60 else (1 if money >= 30 else 0)
                reason = f"Best+耐力{stamina_pct}%≤{self._best_mood_free_stamina}%→{'冥想室' if option_idx==2 else '住处' if option_idx==1 else '露宿'}"
        elif is_first_year:
            # 第一年特殊策略: 心情非Best → 强制付费休息提高心情
            option_idx = 1 if money >= 30 else 0
            reason = f"第一年(turn={turn})+心情{mood}→付费休息"
        else:
            # 规则3: 心情非Best → 付费休息($30) 提高心情
            option_idx = 1 if money >= 30 else 0
            reason = f"心情{mood}→{'住处($30)' if money>=30 else '露宿(免费,钱不够)'}"

        # 如果选了付费休息但钱不够 → 耐力高则继续训练, 低则免费
        if option_idx > 0:
            needed = 60 if option_idx == 2 else 30
            if money < needed:
                if stamina_pct >= self._stamina_continue:
                    print(f"[休息] ${money}<${needed} 耐力={stamina_pct}%≥{self._stamina_continue}% → 继续训练")
                    self._consecutive_calls = 0
                    return self._exit_to_training(ctx)
                print(f"[休息] ${money}<${needed} 耐力={stamina_pct}%<{self._stamina_continue}% → 露宿(免费)")
                option_idx = 0
                reason = f"钱不够(${money}<${needed})+耐力低→免费"

        print(f"[休息] ${money} | {stamina_pct}% | {mood} → {names[option_idx]}")

        if ctx.logger:
            ctx.logger.log_rest(
                option_id=option_idx + 1, option_name=names[option_idx],
                cost=[0, 30, 60][option_idx], stamina_before=stamina_pct,
                reason=reason)
        if option_idx > 0:
            rx, ry = self.REST_CLICK_POINTS[option_idx]
            ctx.controller.click_at_percent(rx, ry)
            time.sleep(0.2)

        ctx.controller.click_at_percent(self.CONFIRM_X, self.CONFIRM_Y)
        # 轮询等待休息动画结束 (画面离开休息界面) 或超时 (付费失败)
        from .recognition import ScreenDetector
        rest_gone = False
        for _ in range(8):  # 8 × 0.15s = 1.2s 总超时
            time.sleep(0.15)
            new_screenshot = ctx.capture.capture_game()
            if new_screenshot is not None:
                detector = ScreenDetector(ctx.ocr)
                if not detector.is_rest_screen(new_screenshot):
                    # 休息界面消失 → 休息成功, 缓存截图推进
                    rest_gone = True
                    ctx.pending_screenshot = new_screenshot
                    break

        # 付费休息后检测是否卡住 (古币不足)
        if option_idx > 0 and not rest_gone:
            new_screenshot = ctx.capture.capture_game()
            if new_screenshot is not None:
                from .recognition import ScreenDetector
                detector = ScreenDetector(ctx.ocr)
                if detector.is_rest_screen(new_screenshot):
                    return self._handle_fallback(ctx, stamina_pct)

        self._consecutive_calls = 0
        return True

    def _handle_fallback(self, ctx, stamina_pct):
        """付费休息失败后的回退处理"""
        print(f"[休息] 钱不足, 耐力={stamina_pct}%", end="")
        if stamina_pct >= self._stamina_continue:
            print(" → 继续训练")
            return self._exit_to_training(ctx)
        print(" → 露宿(免费)")
        ctx.controller.click_at_percent(self.CONFIRM_X, self.CONFIRM_Y)
        time.sleep(0.5)
        return True

    def _exit_to_training(self, ctx):
        """退出休息界面, 返回训练"""
        ctx.controller.send_escape()
        time.sleep(0.5)
        ctx.controller.click_center_multi(count=10, interval=0.01, click_delay=0.01)
        _wait_for_screen(ctx, "training", timeout=5.0)
        ctx.pending_screenshot = ctx.capture.capture_game()
        ctx.pending_action = "training"
        return True


# ============================================================
# MainMenuHandler (Priority 20) - 主菜单
# ============================================================

class MainMenuHandler(Handler):
    """
    主菜单: 优先检测回合, 然后按优先级处理委托/交易/训练。

    优先级顺序:
      1. 检测当前回合 (第一件事)
      2. 检测 "受理讨伐委托!" 通知 → 进入委托
      3. 检测 "交易"+"评鉴战" (商店事件) → 进入交易
      4. 默认: 点击训练入口
    """

    TRAIN_X = 0.91
    TRAIN_Y = 0.40
    COMMISSION_X = 0.91
    COMMISSION_Y = 0.51
    TRADE_X = 0.89
    TRADE_Y = 0.56

    # 委托通知检测区域 — 通知出现在屏幕中央而非右侧边栏
    COMMISSION_NOTIFY_REGIONS = [
        (0.10, 0.45, 0.80, 0.30),   # 中央下 (与 capture_screen.py 一致)
        (0.35, 0.55, 0.50, 0.18),   # 中央偏下聚焦
        (0.55, 0.40, 0.35, 0.30),   # 中央偏右
    ]

    # 商店事件检测区域 (训练主屏右侧)
    SHOP_EVENT_REGIONS = [
        (0.68, 0.30, 0.30, 0.50),
        (0.74, 0.34, 0.22, 0.42),
    ]

    @property
    def priority(self): return 5

    @property
    def name(self): return "主菜单"

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        return bool(detector.is_main_menu_screen(frame.screenshot))

    def handle(self, ctx: HandlerContext) -> bool:
        # 1. 第一件事: 检测当前回合
        ctx.try_detect_turn()

        # 截图用于后续检测
        screenshot = ctx.capture.capture_game()
        frame = FrameContext(screenshot) if screenshot is not None else None
        from .recognition import ScreenDetector
        if frame is None or not ScreenDetector(ctx.ocr, frame).is_main_menu_screen(screenshot):
            return False
        ctx.pending_action = ''

        # 2. 检测委托通知 → 优先进入委托
        if frame is not None and self._has_commission_notify(frame, ctx.ocr):
            print("[菜单] 委托通知 → 进入委托")
            ctx.controller.click_at_percent(self.COMMISSION_X, self.COMMISSION_Y)
            _wait_for_screen(ctx, "commission", timeout=3.0)
            ctx.pending_screenshot = ctx.capture.capture_game()
            return True

        # 3. 检测商店事件 (交易+评鉴战) → 优先进入交易
        if frame is not None and self._has_shop_event(frame, ctx.ocr):
            print("[菜单] 商店事件 → 进入交易")
            ctx.controller.click_at_percent(self.TRADE_X, self.TRADE_Y)
            _wait_for_screen(ctx, "trade", timeout=3.0)
            ctx.pending_screenshot = ctx.capture.capture_game()
            return True

        # 4. 默认: 点击训练入口
        print("[菜单] 进入训练...")
        ctx.controller.click_at_percent(self.TRAIN_X, self.TRAIN_Y)
        if not _wait_for_screen(ctx, "training", timeout=5.0):
            print('[菜单] 未到达训练界面，重新识别当前画面')
            ctx.pending_screenshot = ctx.capture.capture_game()
            return True
        # 已确认训练界面, 缓存截图让 trainer 跳过 dispatch 直接决策
        ctx.pending_screenshot = ctx.capture.capture_game()
        if ctx.pending_screenshot is not None and ScreenDetector(ctx.ocr).is_training_screen(ctx.pending_screenshot):
            ctx.pending_action = "training"
        return True

    def _has_commission_notify(self, frame: FrameContext, ocr_engine) -> bool:
        """检测主菜单上是否有委托通知"""
        for rx, ry, rw, rh in self.COMMISSION_NOTIFY_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if not text:
                continue
            # "讨伐委托" 是关键指纹 (OCR 可能漏字如 "理讨伐委托")
            if "讨伐委托" in text or "受理讨伐" in text:
                print(f"  [委托通知] 检测到: '{text[:60]}'")
                return True
            # 模糊: 同时出现 "讨伐"+"委托" (被 OCR 拆散的情况)
            if "讨伐" in text and "委托" in text:
                print(f"  [委托通知] 模糊匹配: '{text[:60]}'")
                return True
        return False

    def _has_shop_event(self, frame: FrameContext, ocr_engine) -> bool:
        """检测主菜单/训练屏上是否有商店事件 (交易+评鉴战).
        两遍扫描: 不要求两个关键词在同一 OCR 区域, 容忍 OCR 漏字."""
        has_trade = False
        has_appraisal = False
        for rx, ry, rw, rh in self.SHOP_EVENT_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if not text:
                continue
            if "交易" in text:
                has_trade = True
            if "评鉴战" in text or "评鉴" in text:
                has_appraisal = True
            if has_trade and has_appraisal:
                return True
        # 回退: 交易 + 顶部横幅含 "评鉴战"
        if has_trade and not has_appraisal:
            for rx, ry, rw, rh in [(0.05, 0.00, 0.90, 0.10)]:
                banner = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
                if "评鉴战" in banner or "评鉴" in banner:
                    return True
        return False


# ============================================================
# CommissionHandler (Priority 3) - 讨伐委托
# ============================================================

class AppraisalHandler(Handler):
    """到期评鉴战独立于商店：定位入口 → 准备页接受 → 战斗处理器接管。"""

    ENTRY_REGION = (0.78, 0.38, 0.19, 0.23)
    ACCEPT_REGION = (0.78, 0.84, 0.18, 0.10)
    # 入口文字左侧只是标签，实际可点击热区是右侧菱形图标。
    # 使用相对客户区坐标，适配当前 1600x900 游戏客户区。
    ENTRY_ICON_CLICK = (0.941, 0.466)

    @property
    def priority(self): return 1

    @property
    def name(self): return "评鉴战"

    def can_handle(self, frame, ocr_engine):
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)
        return (detector.is_appraisal_screen(frame.screenshot) or
                detector.is_appraisal_due_screen(frame.screenshot))

    def handle(self, ctx):
        from .controller import InputTargetError
        from .recognition import ScreenDetector
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            raise InputTargetError("评鉴战截图失败，停止点击")
        detector = ScreenDetector(ctx.ocr)
        if detector.is_appraisal_screen(screenshot):
            label, region = "接受", self.ACCEPT_REGION
        elif detector.is_appraisal_due_screen(screenshot):
            label, region = "评鉴战", self.ENTRY_REGION
        else:
            return False
        position = detector.find_action_position(screenshot, label, region)
        if position is None:
            raise InputTargetError(f"无法唯一定位评鉴战的「{label}」按钮，停止点击")
        click_position = self.ENTRY_ICON_CLICK if label == "评鉴战" else position
        print(f"[评鉴战] {label}，文字=({position[0]:.3f}, {position[1]:.3f})，点击=({click_position[0]:.3f}, {click_position[1]:.3f})")
        ctx.controller.click_at_percent(*click_position)
        if label == "评鉴战":
            # 首次打开虚空评鉴战会加载角色/奖励卡片，转场可能超过 6 秒。
            if not _wait_for_screen(ctx, "appraisal", timeout=12.0):
                raise InputTargetError("评鉴战入口点击后未到达准备页，停止重复点击")
        else:
            deadline = time.perf_counter() + 6.0
            while time.perf_counter() < deadline:
                ctx.controller.wait(0.4)
                current = ctx.capture.capture_game()
                if current is not None and not ScreenDetector(ctx.ocr).is_appraisal_screen(current):
                    print("[评鉴战] 已接受，交由战斗处理器接管")
                    ctx.pending_screenshot = current
                    return True
            raise InputTargetError("评鉴战接受后未离开准备页，停止重复点击")
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


class CommissionHandler(Handler):
    """
    讨伐委托检测与自动执行。

    主菜单检测 "受理讨伐委托!" 通知 + 左上角 "距离目标" 倒计时。
    委托界面读取 RANK，按 高阶→中阶→低阶 尝试，
    满足 RANK >= 建议等级 - 阈值 则执行。

    移植自 SleepRunner 的讨伐委托处理逻辑。
    """

    # 委托界面标题检测
    COMMISSION_TITLE_REGIONS = [
        (0.50, 0.26, 0.40, 0.14),
        (0.55, 0.28, 0.35, 0.12),
        (0.48, 0.25, 0.42, 0.16),
    ]

    # 当前 RANK 检测 (委托界面左侧)
    RANK_REGIONS = [
        (0.04, 0.28, 0.15, 0.08),
        (0.05, 0.30, 0.14, 0.06),
        (0.03, 0.27, 0.16, 0.10),
    ]

    # 三个委托阶数点击位置 (右侧按钮, 从上到下: 低→中→高)
    TIER_CLICK_POINTS = {
        "低阶": (0.88, 0.36),
        "中阶": (0.88, 0.48),
        "高阶": (0.88, 0.60),
    }

    # "建议综合等级" 标签检测区域 (中央)
    RECOMMENDED_LABEL_REGIONS = [
        (0.50, 0.55, 0.25, 0.10),
        (0.48, 0.56, 0.28, 0.08),
        (0.52, 0.54, 0.22, 0.12),
    ]

    # 建议综合等级 RANK 数值检测区域 (选择阶数后中央显示)
    RECOMMENDED_LEVEL_REGIONS = [
        (0.52, 0.58, 0.18, 0.08),
        (0.50, 0.59, 0.20, 0.06),
        (0.52, 0.57, 0.16, 0.10),
    ]

    # 挑战/出击按钮
    CHALLENGE_BTN_X = 0.88
    CHALLENGE_BTN_Y = 0.88

    # "跳过战斗" 按钮检测区域和点击位置
    SKIP_BATTLE_REGIONS = [
        (0.35, 0.68, 0.30, 0.12),
        (0.38, 0.70, 0.25, 0.10),
        (0.40, 0.66, 0.25, 0.14),
    ]
    SKIP_BATTLE_CLICK_X = 0.43
    SKIP_BATTLE_CLICK_Y = 0.73

    # 委托完成后确认按钮
    COMMISSION_DONE_X = 0.88
    COMMISSION_DONE_Y = 0.90

    # 退出委托按钮 (返回主菜单)
    EXIT_COMMISSION_X = 0.91
    EXIT_COMMISSION_Y = 0.92

    # 默认 RANK 差阈值 (可配置)
    DEFAULT_RANK_THRESHOLD = 5

    def __init__(self, config=None):
        self.config = config or {}
        cc = self.config.get("commission", {})
        self.rank_threshold = cc.get("rank_threshold", self.DEFAULT_RANK_THRESHOLD)
        self.enabled = cc.get("enabled", True)

    @property
    def priority(self): return 4

    @property
    def name(self): return "讨伐委托"

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        if not self.enabled:
            return False
        return self._is_commission_screen(frame, ocr_engine)

    def handle(self, ctx: HandlerContext) -> bool:
        return self._handle_commission_screen(ctx)

    # ===================== 委托界面处理 =====================

    def _is_commission_screen(self, frame, ocr_engine) -> bool:
        result = self._is_commission_screen_raw(frame.screenshot, ocr_engine)
        if not result:
            return False
        # 检测 "建议综合等级" 标签 (委托详情页独有)
        for rx, ry, rw, rh in self.RECOMMENDED_LABEL_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if "建议综合等级" in text or "综合等级" in text:
                return True
        # 或者有多个委托阶数文字 (低阶+中阶+高阶)
        tiers_found = set()
        for rx, ry, rw, rh in self.COMMISSION_TITLE_REGIONS:
            text = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            for tier in ["低阶", "中阶", "高阶"]:
                if tier in text:
                    tiers_found.add(tier)
        return len(tiers_found) >= 2

    def _is_commission_screen_raw(self, screenshot, ocr_engine) -> bool:
        """宽松检测: 是否有委托相关文字"""
        for rx, ry, rw, rh in self.COMMISSION_TITLE_REGIONS:
            text = _ocr_region(ocr_engine, screenshot, rx, ry, rw, rh)
            if self._is_commission_title(text):
                return True
        return False

    def _handle_commission_screen(self, ctx: HandlerContext) -> bool:
        """处理委托界面: 读RANK → 逐级尝试 → 执行"""
        print("[委托] 进入")

        # 读取当前 RANK
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False

        my_rank = self._read_rank(screenshot, ctx.ocr)
        if my_rank is None:
            print("[委托] 无法读RANK, 退出")
            self._exit_commission(ctx)
            return True

        print(f"[委托] RANK={my_rank}")

        # 按 高阶 → 中阶 → 低阶 尝试
        tier_order = ["高阶", "中阶", "低阶"]
        for tier_name in tier_order:
            click_x, click_y = self.TIER_CLICK_POINTS[tier_name]

            ctx.controller.click_at_percent(click_x, click_y)
            time.sleep(0.3)

            screenshot = ctx.capture.capture_game()
            if screenshot is None:
                continue

            recommended = self._read_recommended_level(screenshot, ctx.ocr)
            if recommended is None:
                print(f"[委托] {tier_name}: 无建议等级, 跳过")
                continue

            diff = recommended - my_rank
            can_do = my_rank >= recommended - self.rank_threshold
            print(f"[委托] {tier_name} 建议RANK{recommended} diff={diff} {'→ 执行' if can_do else '→ 跳过'}")

            if can_do:
                ctx.controller.click_at_percent(
                    self.CHALLENGE_BTN_X, self.CHALLENGE_BTN_Y)
                time.sleep(0.5)

                self._try_skip_battle(ctx)

                ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
                time.sleep(0.5)

                ctx.controller.click_at_percent(
                    self.COMMISSION_DONE_X, self.COMMISSION_DONE_Y)
                time.sleep(0.5)

                return True

        print("[委托] 全部不满足, 退出")
        self._exit_commission(ctx)
        return True

    def _exit_commission(self, ctx: HandlerContext):
        """退出委托界面"""
        ctx.controller.send_escape()
        time.sleep(0.1)
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        _wait_for_screen(ctx, "main_menu", timeout=3.0)
        ctx.pending_screenshot = ctx.capture.capture_game()

    # ===================== 跳过战斗 =====================

    def _try_skip_battle(self, ctx: HandlerContext):
        """检测并点击 '跳过战斗' 按钮"""
        for attempt in range(3):
            time.sleep(0.3)
            screenshot = ctx.capture.capture_game()
            if screenshot is None:
                continue

            for rx, ry, rw, rh in self.SKIP_BATTLE_REGIONS:
                text = _ocr_region(ctx.ocr, screenshot, rx, ry, rw, rh)
                if self._is_skip_battle_text(text):
                    print(f"[委托] 跳过战斗")
                    ctx.controller.click_at_percent(
                        self.SKIP_BATTLE_CLICK_X, self.SKIP_BATTLE_CLICK_Y)
                    time.sleep(0.5)
                    return

    @staticmethod
    def _is_skip_battle_text(text: str) -> bool:
        if not text:
            return False
        return ("跳过战斗" in text or "跳过" in text or
                "SKIP" in text.upper())

    # ===================== OCR 读取 =====================

    def _read_rank(self, screenshot, ocr_engine) -> Optional[int]:
        """读取当前 RANK 数值"""
        import re
        for rx, ry, rw, rh in self.RANK_REGIONS:
            text = _ocr_region(ocr_engine, screenshot, rx, ry, rw, rh)
            if not text:
                continue
            match = re.search(r'RANK\s*(\d{1,2})', text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            # 纯数字回退
            nums = re.findall(r'\b(\d{1,2})\b', text)
            for n in nums:
                val = int(n)
                if 1 <= val <= 30:
                    return val
        return None

    def _read_recommended_level(self, screenshot, ocr_engine) -> Optional[int]:
        """读取建议综合等级 RANK 数值"""
        import re
        for rx, ry, rw, rh in self.RECOMMENDED_LEVEL_REGIONS:
            text = _ocr_region(ocr_engine, screenshot, rx, ry, rw, rh)
            if not text:
                continue
            # 匹配 RANK 后跟数字
            match = re.search(r'RANK\s*(\d{1,2})', text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            # 纯数字回退
            nums = re.findall(r'\b(\d{1,2})\b', text)
            for n in nums:
                val = int(n)
                if 10 <= val <= 30:
                    return val
        return None

    # ===================== 文本判定 =====================

    @staticmethod
    def _is_commission_title(text: str) -> bool:
        if not text:
            return False
        return ("讨伐委托" in text or "受理讨伐" in text or
                "建议综合等级" in text)



# ============================================================
# ShopHandler (Priority 4) - 商店/交易
# ============================================================

class ShopHandler(Handler):
    """
    商店/交易事件处理器。

    当训练主屏右侧按钮从 "训练/委托/休息" 变为 "交易"+"评鉴战" 时,
    说明触发了商店事件。检测到后:
      1. 点击 "交易" 进入交易界面
      2. OCR 3 个物品文字
      3. 按物品类型分类
      4. 按优先级评分排序
      5. 点击最优物品购买
      6. 处理购买后确认/推进

    物品类型 (按正则匹配):
      - mood_recovery:     exact "心情恢复"
      - stamina_recovery:  耐力(\\d+)恢复
      - potential_points:  潜质点数(\\d+)增加
      - vitality_potion:  活力药水 (训练失败后可继续训练)
      - attribute_boost:   (.+?)(\\d+)增加
      - training_exp:      (.+)训练经验增加 (只买匹配优先属性的, 专注/保护永不购买)
      - status_buff:       (.+)状态获得
      - random_effect:     随机效果 (解析详情中的加成属性, 匹配优先属性则购买)
      - consumable:        通用消耗品 (兜底匹配)
    """

    # ----- 物品类型正则 (顺序重要: 具体→通用, 避免误匹配) -----
    # ----- 物品名称模式 (根据物品名称直接分类, 比效果文字更可靠) -----
    _NAME_PATTERNS: list[tuple] = [
        # (.+)训练秘[籍笈]: "力量训练秘籍", "体力训练秘籍" 等
        (re.compile(r'(.+)训练秘[籍笈]'), "training_exp"),
        # 禁忌(.+)训练之书: "禁忌力量训练之书", "禁忌保护训练之书" 等
        (re.compile(r'禁忌(.+)训练之书'), "training_exp"),
        # (.+)训练结业奖牌: "力量训练结业奖牌", "体能训练结业奖牌" 等
        (re.compile(r'(.+)训练结业奖牌'), "training_exp"),
        # (.+)特训证书: "力量特训证书", "体能特训证书" 等
        (re.compile(r'(.+)特训证书'), "training_exp"),
    ]

    ITEM_PATTERNS = [
        # --- 恢复类 (耐力=训练体力条, 不要和"体力"属性混淆) ---
        ("mood_recovery",      re.compile(r'心情\s*恢复')),
        ("stamina_recovery",   re.compile(r'耐力\s*(\d+)\s*恢复')),
        ("stamina_recovery",   re.compile(r'耐力\s*(?:恢复|回復)\s*(\d+)')),
        # --- 潜力/活力 ---
        ("potential_points",   re.compile(r'潜质点数\s*(\d+)\s*增加')),
        ("vitality_potion",    re.compile(r'活力药水')),
        ("vitality_potion",    re.compile(r'活力\s*(\d+)\s*恢复')),
        ("vitality_potion",    re.compile(r'特定事件中[亻吏使]用')),
        # --- 训练经验 ---
        ("training_exp",       re.compile(r'(.+)训练经验\s*增加')),
        # --- 属性增加 (数值+单位) ---
        ("attribute_boost",    re.compile(r'(.+?)\s*(\d+)\s*增加')),
        ("attribute_boost",    re.compile(r'(.+?)\s*\+\s*(\d+%?)')),          # "暴击率+5%"
        ("attribute_boost",    re.compile(r'(.+?)\s*(?:提升|上升)\s*(\d+)')),  # "力量提升10"
        # --- 状态 ---
        ("status_buff",        re.compile(r'(.+)状态\s*获得')),
        ("status_buff",        re.compile(r'(.+)状态\s*付与')),
        ("status_buff",        re.compile(r'(.+)效果\s*获得')),
        # --- 随机 ---
        ("random_effect",      re.compile(r'随机\s*效果')),
        # --- 古币获得 ---
        ("money_gain",         re.compile(r'(?:古币|金币|金钱)\s*(\d+)\s*(?:获得|入手|增加|取得)')),
        # --- 兜底: 任何属性名+数字 (放在最后, 最通用) ---
        ("consumable",         re.compile(r'(.+?)\s+(\d{2,6})\b')),
    ]

    # ----- 类型显示名 -----
    TYPE_LABELS = {
        "mood_recovery":      "心情恢复",
        "stamina_recovery":   "耐力恢复",
        "potential_points":   "潜质点数增加",
        "vitality_potion":    "活力药水",
        "training_exp":       "训练经验增加",
        "attribute_boost":    "属性增加",
        "status_buff":        "状态获得",
        "random_effect":      "随机效果",
        "money_gain":         "古币获得",
        "consumable":         "消耗品",
    }

    # ----- 坐标常量 (可被 config 覆盖, 已通过实际截图校准) -----
    TRADE_BUTTON_X = 0.89
    TRADE_BUTTON_Y = 0.56

    # 评鉴战按钮位置 (训练主屏右侧)
    APPRAISAL_BUTTON_X = 0.88
    APPRAISAL_BUTTON_Y = 0.47

    # 评鉴战界面: 接受按钮
    APPRAISAL_ACCEPT_X = 0.86
    APPRAISAL_ACCEPT_Y = 0.90

    # ----- 物品槽位点击位置 (由 mouse_coords.py 校准) -----
    ITEM_CLICK_POINTS = [
        (0.8490, 0.4102),   # 物品1
        (0.8521, 0.5287),   # 物品2
        (0.8547, 0.6333),   # 物品3
    ]

    ITEM_TEXT_REGIONS = [
        (0.80, 0.37, 0.12, 0.11),   # 物品1: 右侧面板 (旧, 仅作参考)
        (0.80, 0.48, 0.12, 0.09),   # 物品2
        (0.80, 0.59, 0.12, 0.09),   # 物品3
    ]

    # ----- 物品详情面板 OCR 区域 (由 mouse_coords.py 校准) -----

    # 物品名称区域 (右侧物品列表, 3个槽位从上到下, 始终可见)
    # 高度收紧以排除价格数字, 避免 OCR 混入数字导致识别错误
    ITEM_NAME_REGIONS = [
        [0.8151, 0.3806, 0.1078, 0.055],    # 物品1 (不含价格 60)
        [0.8141, 0.4898, 0.1104, 0.048],    # 物品2 (不含价格 50)
        [0.8130, 0.5843, 0.1115, 0.048],    # 物品3 (不含价格 15)
    ]

    # 物品效果/作用文字区域 (中间详情面板, 点击物品后显示对应效果)
    ITEM_EFFECT_REGION = (0.50, 0.58, 0.23, 0.19)

    # SOLD OUT 告罄标记区域 (斜着写的 SOLD OUT)
    ITEM_SOLDOUT_REGION = (0.5542, 0.3556, 0.1192, 0.1296)

    # 兼容旧代码: 保留全版面 OCR 但缩小为不包含右侧列表的区域
    TRADE_DETAIL_REGION = (0.4943, 0.2306, 0.2349, 0.5963)

    BUY_CONFIRM_X = 0.86
    BUY_CONFIRM_Y = 0.90

    POST_PURCHASE_CLICKS = 3
    POST_PURCHASE_INTERVAL = 0.01
    POST_PURCHASE_WAIT = 0.5

    @property
    def priority(self): return 2

    @property
    def name(self): return "商店交易"

    def __init__(self, config=None):
        self.config = config or {}
        self._trade_handled = False  # 防止重复进入交易
        sc = self.config.get("shop", {})
        self.enabled = sc.get("enabled", True)

        # 物品类型优先级 (用户可配置顺序)
        self.item_priority_order = sc.get("item_priority", [
            "mood_recovery", "stamina_recovery", "potential_points",
            "vitality_potion", "training_exp", "attribute_boost",
            "status_buff", "random_effect", "money_gain", "consumable",
        ])
        self._priority_rank = {
            type_id: idx
            for idx, type_id in enumerate(self.item_priority_order)
        }

        # 属性匹配关键词 (由 trainer 动态同步)
        self.attribute_match = "力量"

        # 覆盖坐标常量
        self.trade_button_x = sc.get("trade_button_x", self.TRADE_BUTTON_X)
        self.trade_button_y = sc.get("trade_button_y", self.TRADE_BUTTON_Y)

        item_regions = sc.get("item_regions", None)
        self.item_text_regions = [tuple(r) for r in item_regions] if item_regions else list(self.ITEM_TEXT_REGIONS)

        item_clicks = sc.get("item_click_points", None)
        self.item_click_points = [tuple(p) for p in item_clicks] if item_clicks else list(self.ITEM_CLICK_POINTS)

        self.buy_confirm_x = sc.get("buy_confirm_x", self.BUY_CONFIRM_X)
        self.buy_confirm_y = sc.get("buy_confirm_y", self.BUY_CONFIRM_Y)
        self.post_clicks = sc.get("post_purchase_clicks", self.POST_PURCHASE_CLICKS)
        self.post_interval = sc.get("post_purchase_interval", self.POST_PURCHASE_INTERVAL)
        self.post_wait = sc.get("post_purchase_wait", self.POST_PURCHASE_WAIT)

        # === 商店物品配置加载 ===
        training_cfg = self.config.get("training", {})
        self._shop_profile_name = training_cfg.get("rule_profile", "default")
        self._shop_profile_path = Path(f"profiles/shop/{self._shop_profile_name}.json")
        self._known_items: list = []
        self._load_shop_profile()

    def _load_shop_profile(self):
        """加载商店物品配置 JSON (优先从内存缓存读取)."""
        cached = _cached_load_json(self._shop_profile_path, "items")
        if cached:
            self._known_items = cached
            print(f"[商店] 已加载 {len(self._known_items)} 个已知物品 ({self._shop_profile_name})")
        else:
            self._known_items = []

    def _save_shop_profile(self):
        """持久化物品列表到 JSON 文件 (同步更新缓存)."""
        data = {
            "_comment": "商店物品配置 - 遇到未知物品时自动添加。手动编辑以调整优先级。",
            "items": self._known_items,
        }
        _cached_save_json(self._shop_profile_path, data)

    def _auto_learn_item(self, raw_text: str, classified_info: dict, name_text: str = ""):
        """将新遇到的物品添加到配置并持久化.
        若物品已存在但类型为泛型(consumable), 用新的具体类型更新它。
        跳过明显的 OCR 噪声."""
        # 优先用 OCR 到的名称区域文字, 其次用分类的 attr_name, 最后用原始文字
        name = (name_text or "").strip()
        if len(name) <= 1:
            name = classified_info.get("attr_name", "") or raw_text[:30].strip()
        else:
            name = name[:30].strip()

        # 跳过 OCR 噪声: 纯数字/纯符号/过短的"物品名"
        if len(name.strip()) <= 1:
            print(f"  [商店] 跳过: 名称太短 '{name}'")
            return
        if name.strip().isdigit():
            print(f"  [商店] 跳过: 纯数字 '{name}'")
            return

        new_type = classified_info["type"]
        new_attr = classified_info.get("attr_name", "")
        new_value = classified_info.get("value", 0)

        # 检查是否已存在同名/高度相似物品 → 升级类型或跳过
        import difflib
        for existing in self._known_items:
            existing_name = existing.get("name", "")
            # 精确/子串匹配
            if existing_name == name or existing_name in name or name in existing_name:
                old_type = existing.get("type", "")
                if old_type == "consumable" and new_type != "consumable":
                    print(f"  [商店] ↑ {name}: {old_type}→{new_type}")
                    existing["type"] = new_type
                    if new_attr:
                        existing["attr_name"] = new_attr
                    if new_value:
                        existing["value"] = new_value
                    existing["note"] = f"Upgraded {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Raw: {raw_text[:100]}"
                    if raw_text.strip()[:80] not in existing.get("keywords", []):
                        existing.setdefault("keywords", []).append(raw_text.strip()[:80])
                    self._save_shop_profile()
                    return
                elif old_type == new_type:
                    if raw_text.strip()[:80] not in existing.get("keywords", []):
                        existing.setdefault("keywords", []).append(raw_text.strip()[:80])
                        self._save_shop_profile()
                    return
            # 模糊匹配: 截取短名对比 (防止OCR错字+后缀垃圾产生重复条目)
            # 取较短名的长度作为对比窗口, 只比较前段 (OCR后缀垃圾通常在后)
            short_len = min(len(existing_name), len(name), 6)
            if short_len >= 2:
                a = existing_name[:short_len]
                b = name[:short_len]
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                if ratio > 0.65:
                    print(f"  [商店] 跳过: '{name[:20]}' 与已知 '{existing_name}' 相似({ratio:.0%}), OCR噪声")
                    return

        # 新物品
        item_id = f"shop_{uuid.uuid4().hex[:8]}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 新类型的优先级: 插入到 consumable 之前
        if new_type not in self._priority_rank:
            self._priority_rank[new_type] = max(0, len(self.item_priority_order) - 1)
            self.item_priority_order.insert(self._priority_rank[new_type], new_type)
            self.TYPE_LABELS[new_type] = new_type

        default_priority = self._priority_rank.get(new_type, len(self.item_priority_order) - 1)

        new_item = {
            "id": item_id,
            "name": name,
            "type": new_type,
            "keywords": [raw_text.strip()[:80]],
            "name_keys": [name.strip()[:40]] if name.strip() else [],
            "effect_keys": [raw_text.strip()[:80]],
            "attr_name": new_attr,
            "value": new_value,
            "priority": default_priority,
            "note": f"Auto-learned {now}. Raw: {raw_text[:100]}"
        }
        self._known_items.append(new_item)
        self._save_shop_profile()
        print(f"  [商店] 新物品: {name} [{new_type}]")

    # ===================== 状态检测 =====================

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        if not self.enabled:
            return False

        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine, frame)

        # 硬排除: 左上角有 "旅程事件" → 绝对是事件界面, 不是商店
        if detector.is_event_marker_screen(frame.screenshot):
            return False

        if detector.is_shop_event_screen(frame.screenshot):
            return True
        if detector.is_trade_screen(frame.screenshot):
            return True
        if detector.is_trade_notification_screen(frame.screenshot):
            return True
        if detector.is_appraisal_screen(frame.screenshot):
            return True
        return False

    # ===================== 主处理流程 =====================

    def handle(self, ctx: HandlerContext) -> bool:
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False

        from .recognition import ScreenDetector
        detector = ScreenDetector(ctx.ocr)

        # 硬排除: 左上角有 "旅程事件" → 绝对是事件界面, 不是商店
        if detector.is_event_marker_screen(screenshot):
            return False

        if detector.is_appraisal_screen(screenshot):
            self._trade_handled = False
            print("[商店] 检测到评鉴战界面")
            return self._handle_appraisal_screen(ctx)
        elif detector.is_trade_notification_screen(screenshot):
            print("[商店] 检测到交易通知")
            return self._handle_trade_notification(ctx)
        elif detector.is_trade_screen(screenshot):
            print("[商店] 检测到交易界面")
            return self._handle_trade_screen(ctx)
        elif detector.is_shop_event_screen(screenshot):
            if self._trade_handled:
                self._trade_handled = False
                print("[商店] 已交易过, 点评鉴战")
                return self._click_appraisal(ctx)
            print("[商店] 检测到商店事件, 进入交易")
            return self._enter_trade(ctx)
        return False

    # ===================== 进入交易 =====================

    def _enter_trade(self, ctx: HandlerContext) -> bool:
        """从训练主屏点击 '交易' 按钮"""
        ctx.controller.click_at_percent(self.trade_button_x, self.trade_button_y)
        _wait_for_screen(ctx, "trade", timeout=3.0)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True

    # ===================== 交易通知处理 =====================

    def _handle_trade_notification(self, ctx: HandlerContext) -> bool:
        """处理 '委托交易全新商品到货!' 等交易通知: 点击交易按钮 → 短轮询等待交易界面."""
        ctx.controller.click_at_percent(self.trade_button_x, self.trade_button_y)
        from .recognition import ScreenDetector
        for _ in range(10):  # 最多 1.0s (10 × 0.10s)
            time.sleep(0.10)
            screenshot = ctx.capture.capture_game()
            if screenshot is not None:
                detector = ScreenDetector(ctx.ocr)
                if detector.is_trade_screen(screenshot):
                    print("[商店] 检测到交易界面")
                    return self._handle_trade_screen(ctx)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True

    # ===================== 交易界面处理 =====================

    def _handle_trade_screen(self, ctx: HandlerContext) -> bool:
        """交易界面: 一次扫描全部物品 → 按优先级依次购买合适的 → 退出"""
        from .recognition import detect_green_stamina_bar

        # 1. 一次扫描全部 3 个物品, 记下详情
        items, last_clicked = self._scan_trade_items(ctx)
        self._trade_handled = True  # 已进入交易界面, 防止回主菜单后重入

        # 2. 读状态 (先于打印, 确保心情显示准确)
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False
        game_state = ctx.ui.read_game_state(screenshot)
        stamina = game_state.get("stamina", 100)
        mood = game_state.get("mood", "Normal")
        self._sync_attribute_match(ctx)

        if not items:
            self._print_shop_header(ctx, mood)
            print("  (无可用物品)")
            self._exit_trade(ctx)
            return self._click_appraisal(ctx)

        # 3. 评分排序 + 格式化打印
        stamina_bar_green = detect_green_stamina_bar(screenshot)

        self._print_shop_header(ctx, mood)
        ranked = self._rank_items(items, stamina, mood, stamina_bar_green)

        for i in range(3):
            match = next(((info, score) for slot, info, score in ranked if slot == i), None)
            raw = next((t for slot, _, t in items if slot == i), None)

            if raw is None:
                print(f"  [{i + 1}] (空)")
            elif match:
                info, score = match
                self._print_item_block(i, info, score, "buy")
            else:
                info = next((info for slot, info, _ in items if slot == i), None)
                skip_reason = self._skip_reason(info, stamina, mood, stamina_bar_green) if info else ""
                self._print_item_block(i, info, 0, skip_reason)

        if not ranked:
            if ctx.logger:
                ctx.logger.log_shop(
                    scanned=items, bought=[], stamina=stamina,
                    mood=mood, money=game_state.get("money", 0))
            self._exit_trade(ctx)
            return self._click_appraisal(ctx)

        # 4. 按优先级依次购买
        buy_count = 0
        bought_items = []
        for best_slot, best_info, best_score in ranked:
            print(f"  → 购买[{best_slot + 1}] ... ", end="", flush=True)
            if not self._purchase_item(ctx, best_slot, last_clicked):
                print("失败")
                break
            buy_count += 1
            bought_items.append((best_slot, best_info, best_score))
            print("OK")

        if ctx.logger:
            ctx.logger.log_shop(
                scanned=items, bought=bought_items, stamina=stamina,
                mood=mood, money=game_state.get("money", 0))

        if buy_count == 0:
            self._exit_trade(ctx)
            return self._click_appraisal(ctx)

        print(f"  购买 {buy_count} 件, 推进回合...")
        ctx.controller.click_center_multi(
            count=self.post_clicks, interval=0.01, click_delay=0.01)
        time.sleep(self.post_wait)
        self._exit_trade(ctx)
        return self._click_appraisal(ctx)

    # ===================== 格式化打印 =====================

    def _print_shop_header(self, ctx: HandlerContext, mood: str):
        mood_display = mood if isinstance(mood, str) else "?"
        print(f"\n--- 商店 {self.attribute_match}/{mood_display} ---")

    def _print_item_block(self, slot: int, info: dict | None, score: int, action: str):
        """打印单个物品信息块 (压缩为一行)"""
        if info is None:
            print(f"  [{slot + 1}] (空)")
            return

        type_id = info.get("type", "?")
        type_label = self.TYPE_LABELS.get(type_id, type_id)
        name = info.get("display_name", "") or info.get("attr_name", "") or "?"
        from_prof = info.get("from_profile", False)
        source = "已知" if from_prof else "OCR"

        if action == "buy":
            print(f"  [{slot + 1}] {name} [{type_label}] {source} 评分{score} → 购买")
        else:
            print(f"  [{slot + 1}] {name} [{type_label}] {source} → {action}")

    def _skip_reason(self, info: dict, stamina: float, mood: str,
                     stamina_bar_green: bool) -> str:
        """返回跳过原因"""
        type_id = info.get("type", "")
        if type_id == "mood_recovery" and mood == "Best":
            return "心情已是Best, 跳过"
        if type_id == "stamina_recovery" and (stamina >= 95 or stamina_bar_green):
            return "体力已满, 跳过"
        if type_id == "training_exp":
            name = info.get("attr_name", "")
            if any(blocked in name for blocked in self._TRAINING_EXP_BLOCKED):
                return f"训练经验(专注/保护)永不购买, 跳过"
            if not self._attr_matches_priority(name):
                return f"属性不匹配({self.attribute_match}), 跳过"
        if type_id in ("attribute_boost", "random_effect"):
            name = info.get("attr_name", "")
            if not self._attr_matches_priority(name):
                return f"属性不匹配({self.attribute_match}), 跳过"
        return "不满足条件, 跳过"

    def _scan_trade_items(self, ctx: HandlerContext) -> tuple[list, int]:
        """逐个点击物品槽位, OCR 名称+效果+价格分区.
        效果文字用于分类, 名称用于 profile 匹配.
        物品1 进入时已默认展开, 不点击直接读; 物品2/3 点击后读取.
        Returns: ([(slot_idx, item_info, raw_text), ...], last_clicked_slot)"""
        items = []
        last_clicked = -1  # -1 表示没有槽位被主动点击过
        for i in range(3):
            if i == 0:
                # 物品1 进入交易界面时详情已默认展开, 不点击直接读
                pass
            else:
                cx, cy = self.item_click_points[i]
                ctx.controller.click_at_percent(cx, cy)
                time.sleep(0.08)
            last_clicked = i

            screenshot = ctx.capture.capture_game()
            if screenshot is None:
                continue

            # 1. 先检查告罄
            soldout_text = _ocr_region(ctx.ocr, screenshot, *self.ITEM_SOLDOUT_REGION)
            if soldout_text and ("SOLD" in soldout_text.upper() or "SOLDOUT" in soldout_text.upper()
                                 or "售罄" in soldout_text or "売切" in soldout_text):
                print(f"  [扫描] 物品{i + 1}: SOLDOUT")
                continue

            # 2. OCR 效果文字 (增强预处理, 提高中文识别率)
            effect_text = _ocr_region_enhanced(ctx.ocr, screenshot, *self.ITEM_EFFECT_REGION)
            # 3. OCR 名称 (增强预处理)
            name_text = _ocr_region_enhanced(ctx.ocr, screenshot, *self.ITEM_NAME_REGIONS[i])

            # 合并效果+名称作为分类输入 (效果优先)
            combined_text = f"{name_text} {effect_text}".strip()

            if not combined_text or len(combined_text.strip()) <= 1:
                continue

            # 用效果文字分类 (效果是主要分类依据)
            classify_input = effect_text if effect_text.strip() else combined_text
            item_info = self._classify_item(classify_input, name_text)
            if item_info:
                # 优先用 profile 已知名称, 其次用右侧列表 OCR 名称, 最后用分类属性名
                profile_name = item_info.get("profile_name", "")
                ocr_name = (name_text or "").strip()
                if profile_name and len(profile_name) >= 2:
                    display_name = profile_name
                elif ocr_name and len(ocr_name) >= 2:
                    display_name = ocr_name
                else:
                    display_name = item_info.get("attr_name", "") or combined_text[:20].strip()
                item_info["display_name"] = display_name
                items.append((i, item_info, combined_text))
                if not item_info.get("from_profile") and combined_text.strip():
                    self._auto_learn_item(combined_text, item_info, name_text)
            else:
                print(f"  [扫描] 物品{i + 1}: 分类失败, 跳过")

        return items, last_clicked

    def _purchase_item(self, ctx: HandlerContext, slot_idx: int,
                       last_clicked: int = -1) -> bool:
        """点击物品 + 购买确认 + 关闭购买后对话框. 返回是否成功.
        若 slot_idx 与扫描时最后点击的槽位相同, 跳过点击 (详情已展开)."""
        if slot_idx == last_clicked:
            # 扫描时最后点击的就是此槽位, 详情已展开, 直接购买
            pass
        else:
            cx, cy = self.item_click_points[slot_idx]
            ctx.controller.click_at_percent(cx, cy)
            time.sleep(0.05)

        ctx.controller.click_at_percent(self.buy_confirm_x, self.buy_confirm_y)
        time.sleep(0.15)

        post_buy = ctx.capture.capture_game()
        if self._check_insufficient_money(post_buy, ctx.ocr):
            print("(古币不足)", end="")
            self._dismiss_dialog(ctx)
            return False

        ctx.controller.click_center_multi(count=3, interval=0.01, click_delay=0.01)
        time.sleep(0.15)
        return True

    # ===================== 属性同步 =====================

    def _sync_attribute_match(self, ctx: HandlerContext):
        """同步当前优先训练属性名称到 attribute_match"""
        try:
            if ctx.engine.is_rule_engine_mode:
                direction = ctx.engine.rule_profile.legacy_strategy.build_direction.value
                # 构建方向 → 主属性映射
                direction_map = {
                    "attack":   "力量",
                    "survival": "生命",
                    "speed":    "速度",
                    "defense":  "防御力",
                }
                self.attribute_match = direction_map.get(direction, "力量")
            else:
                attr_id = ctx.engine.priority
                attr_name = ctx.engine.attributes.get(attr_id, "力量")
                self.attribute_match = attr_name
        except Exception:
            pass

    # ===================== 效果语义自动分类 =====================

    # 效果关键词 → 类型 映射 (按优先级排序, 第一个命中即返回)
    _EFFECT_SEMANTIC_MAP: list[tuple[str, list[str]]] = [
        ("mood_recovery",      ["心情恢复", "情绪恢复", "心情回復", "気分回復"]),
        ("stamina_recovery",   ["耐力恢复", "スタミナ恢复", "スタミナ回復"]),
        ("potential_points",   ["潜质点数", "潜在能力", "潜力"]),
        ("vitality_potion",    ["活力", "生命力恢复", "特定事件中使用", "特定事件中"]),
        ("training_exp",       ["训练经验", "トレーニング経験", "经验值增加"]),
        ("attribute_boost",    ["增加", "提升", "上升", "UP", "アップ"]),
        ("status_buff",        ["状态获得", "状态付与", "状態獲得", "効果獲得"]),
        ("random_effect",      ["随机效果", "随机", "ランダム"]),
        ("money_gain",         ["古币获得", "古币入手", "金币获得", "金钱"]),
    ]

    # 通用效果值提取: "XXX 24 恢复" / "XXX +5%" / "XXX 10 增加"
    _GENERIC_VALUE_PATTERN = re.compile(r'(\d+%?)\s*(?:恢复|回復|增加|提升|上升|获得|入手)')

    def _classify_by_effect_semantics(self, text: str) -> dict | None:
        """
        从 OCR 详情文本中提取效果语义, 自动映射到已知类型。
        无法识别时, 提取效果关键词作为新类型名, 自动创建分类。

        这个方法只在正则匹配全部失败后作为最后兜底。
        """
        normalized = text.replace(" ", "").replace("　", "").replace("\n", "")

        # 1. 关键字映射
        for type_id, keywords in self._EFFECT_SEMANTIC_MAP:
            for kw in keywords:
                if kw in normalized:
                    info = {"type": type_id, "from_profile": False, "priority_override": None}
                    # 提取数值
                    val_match = self._GENERIC_VALUE_PATTERN.search(normalized)
                    info["value"] = int(val_match.group(1).rstrip('%')) if val_match else 0

                    if type_id == "attribute_boost":
                        info["attr_name"] = self._extract_attr_from_text(normalized)
                    elif type_id == "training_exp":
                        info["attr_name"] = self._extract_attr_from_text(normalized)
                    else:
                        info["attr_name"] = kw
                    return info

        # 2. 尝试提取"属性名 + 数值" 模糊模式
        fuzzy_match = re.search(r'([一-鿿]{1,4})\s*(\d+%?)\s*(?:恢复|回復|增加|提升|上升|获得)', normalized)
        if fuzzy_match:
            attr = fuzzy_match.group(1)
            val_str = fuzzy_match.group(2)
            val = int(val_str.rstrip('%'))
            # 根据关键词判断类型 (耐力=训练体力条, 体力=HP属性)
            if "恢复" in normalized or "回復" in normalized:
                if "耐力" in attr:
                    type_id = "stamina_recovery"
                elif "心情" in attr:
                    type_id = "mood_recovery"
                elif "活力" in attr:
                    type_id = "vitality_potion"
                else:
                    type_id = "mood_recovery"
            elif "训练" in normalized or "经验" in normalized:
                type_id = "training_exp"
            else:
                type_id = "attribute_boost"
            return {"type": type_id, "attr_name": attr, "value": val,
                    "from_profile": False, "priority_override": None}

        # 3. 完全无法识别 → 创建新 type (用效果描述的前几个字作为 type 名)
        #    提取看起来像效果的部分 (去掉纯价格/数字)
        cleaned = re.sub(r'\d{3,6}\s*[G古金币]', '', normalized)  # 去掉价格
        cleaned = re.sub(r'SOLDOUT|SOLD', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if len(cleaned) >= 4:
            # 用效果描述的关键词部分作为新 type
            new_type = "effect_" + cleaned[:12].strip()
            # 确保 type 名合法 (只保留字母/中文/数字/下划线)
            new_type = re.sub(r'[^\w一-鿿]', '_', new_type).strip('_')
            if not new_type.startswith("effect_"):
                new_type = "effect_unknown"
            print(f"  [语义] 新类型: {new_type} ({text[:50]})")
            # 注册新类型
            if new_type not in self._priority_rank:
                self._priority_rank[new_type] = len(self.item_priority_order)
                self.item_priority_order.append(new_type)
                self.TYPE_LABELS[new_type] = cleaned[:20]
            return {"type": new_type, "attr_name": cleaned[:30], "value": 0,
                    "from_profile": False, "priority_override": None}

        return None

    def _extract_attr_from_text(self, text: str) -> str:
        """从 OCR 文本中提取属性名 (按长度降序匹配, 优先长词)"""
        all_attrs = sorted(self._ALL_ATTR_NAMES, key=len, reverse=True)
        for attr in all_attrs:
            if attr in text:
                return attr
        return "未知属性"

    # ===================== 物品分类 =====================

    def _classify_item(self, text: str, name_text: str = "") -> dict | None:
        """
        双向匹配 + 正则兜底:
          1. 已知物品双向验证 (名称↔效果互相验证) — 杜绝关键词误匹配
          2. 物品名称模式匹配 (NAME_PATTERNS) — 训练秘笈/禁忌之书等
          3. 正则模式匹配 (ITEM_PATTERNS)
          4. 效果语义自动分类 (兜底, 自动发现新类型)

        Returns:
            dict with keys: type, attr_name, value, from_profile (bool), priority_override (int|None)
        """
        profile_match = None

        # 1. 双向验证: 名称→效果 + 效果→名称 必须指向同一物品
        name_norm = (name_text or "").replace(" ", "").replace("　", "").strip()
        effect_norm = text.replace(" ", "").replace("　", "").strip()
        # 旧式组合文本 (仅用于无 effect_keys 的物品向后兼容)
        combined = f"{effect_norm} {name_norm}".strip()

        import difflib as _difflib

        def _kw_hit(kw: str, target: str) -> bool:
            """关键词命中检测: 先精确子串匹配, 失败则滑动模糊匹配 (容忍OCR错字)."""
            kw_norm = kw.replace(" ", "").replace("　", "")
            if not kw_norm or len(kw_norm) <= 1:
                return False
            if kw_norm in target:
                return True
            # 短关键字不模糊 (2字以下精确就够了)
            if len(kw_norm) <= 2:
                return False
            # 模糊匹配: 滑动窗口, 相似度≥0.7即视为匹配
            kw_len = len(kw_norm)
            for i in range(max(0, len(target) - kw_len + 1)):
                window = target[i:i + kw_len]
                # 快速路径: 首或尾字符相同才进入相似度计算
                if kw_norm[0] != window[0] and kw_norm[-1] != window[-1]:
                    continue
                if _difflib.SequenceMatcher(None, kw_norm, window).ratio() >= 0.7:
                    return True
            return False

        for item in self._known_items:
            name_keys = item.get("name_keys")
            effect_keys = item.get("effect_keys")

            if effect_keys is not None:
                # 双向验证模式: 名称必须命中 name_keys, 效果必须命中 effect_keys
                name_hit = any(_kw_hit(k, name_norm) for k in (name_keys or item.get("keywords", [])))
                if not name_hit:
                    continue
                effect_hit = any(_kw_hit(k, effect_norm) for k in effect_keys)
                if not effect_hit:
                    continue
            else:
                # 无 effect_keys → 向后兼容旧式组合文本匹配
                keywords = item.get("keywords", [])
                if not any(_kw_hit(k, combined) for k in keywords):
                    continue

            profile_match = {
                "type": item["type"],
                "attr_name": item.get("attr_name", ""),
                "value": item.get("value", 0),
                "from_profile": True,
                "priority_override": item.get("priority"),
                "profile_name": item.get("name", ""),
            }
            break

        # 如果 profile 匹配到的是泛型 consumable, 不直接返回, 继续尝试正则/语义升级
        if profile_match and profile_match["type"] != "consumable":
            return profile_match

        # 2. 物品名称模式匹配 (根据名称判断类型, 比效果OCR更稳定)
        name_match = None
        search_text = (name_text or "") + " " + text
        for pattern, type_id in self._NAME_PATTERNS:
            m = pattern.search(search_text)
            if m:
                name_match = {"type": type_id, "from_profile": False, "priority_override": None}
                if type_id == "training_exp":
                    name_match["attr_name"] = m.group(1)
                    name_match["value"] = 0
                break
        if name_match:
            if profile_match:
                print(f"  [分类] ↑ {profile_match['type']}→{name_match['type']} (名称: {search_text[:40]})")
            return name_match

        # 3. 回退到正则匹配 (或从 consumable 升级)
        regex_match = None
        for type_id, pattern in self.ITEM_PATTERNS:
            match = pattern.search(text)
            if match:
                regex_match = {"type": type_id, "from_profile": False, "priority_override": None}

                if type_id == "stamina_recovery":
                    regex_match["value"] = int(match.group(1))
                    regex_match["attr_name"] = "耐力"
                elif type_id == "potential_points":
                    regex_match["value"] = int(match.group(1))
                    regex_match["attr_name"] = "潜质"
                elif type_id == "training_exp":
                    regex_match["attr_name"] = match.group(1)
                    regex_match["value"] = 0
                elif type_id == "attribute_boost":
                    regex_match["attr_name"] = match.group(1)
                    val_str = match.group(2).rstrip('%')
                    regex_match["value"] = int(val_str) if val_str.isdigit() else 0
                elif type_id == "status_buff":
                    regex_match["attr_name"] = match.group(1)
                    regex_match["value"] = 0
                elif type_id == "mood_recovery":
                    regex_match["attr_name"] = "心情"
                    regex_match["value"] = 0
                elif type_id == "vitality_potion":
                    regex_match["attr_name"] = "活力药水"
                    regex_match["value"] = 0
                elif type_id == "random_effect":
                    regex_match["attr_name"] = self._parse_random_attrs(text)
                    regex_match["value"] = 0
                elif type_id == "money_gain":
                    regex_match["value"] = int(match.group(1))
                    regex_match["attr_name"] = "古币"
                elif type_id == "consumable":
                    regex_match["attr_name"] = match.group(1)
                    regex_match["value"] = 0

                break

        # 正则命中且比 profile 的 consumable 更具体 → 用正则结果
        if regex_match and regex_match["type"] != "consumable":
            if profile_match:
                print(f"  [分类] ↑ {profile_match['type']}→{regex_match['type']} ({text[:40]})")
            return regex_match

        # 4. 语义兜底
        semantic_info = self._classify_by_effect_semantics(text)
        if semantic_info:
            if profile_match:
                print(f"  [分类] ↑ 语义 {profile_match['type']}→{semantic_info['type']} ({text[:40]})")
            else:
                print(f"  [分类] 语义 → {semantic_info['type']} ({text[:40]})")
            return semantic_info

        # 语义也没命中 → 有 profile 就用 profile, 没有则返回正则的 consumable
        if profile_match:
            return profile_match
        if regex_match:
            return regex_match
        return None

    # ===================== 物品评分 =====================

    def _rank_items(self, items: list, stamina: float, mood: str,
                    stamina_bar_green: bool) -> list:
        """
        过滤 + 评分排序。
        Returns: [(slot_index, item_info, score), ...] 按 score 降序
        """
        ranked = []
        for slot_idx, item_info, text in items:
            type_id = item_info["type"]

            if not self._can_buy(type_id, item_info, stamina, mood, stamina_bar_green):
                continue

            score = self._score_item(type_id, item_info)
            if score < 0:
                continue
            ranked.append((slot_idx, item_info, score))

        ranked.sort(key=lambda x: (-x[2], self._priority_rank.get(x[1]["type"], 99)))
        return ranked

    # 训练经验永不购买的属性 (无论当前优先级)
    _TRAINING_EXP_BLOCKED = {"专注", "保护"}

    def _can_buy(self, type_id: str, item_info: dict, stamina: float,
                 mood: str, stamina_bar_green: bool) -> bool:
        """判断物品是否满足购买条件"""
        stamina_full = stamina >= 95 or stamina_bar_green
        mood_best = mood == "Best"

        if type_id == "mood_recovery":
            return not mood_best

        if type_id == "stamina_recovery":
            return not stamina_full

        if type_id == "potential_points":
            return True

        if type_id == "vitality_potion":
            return True

        if type_id == "training_exp":
            attr_name = item_info.get("attr_name", "")
            if any(blocked in attr_name for blocked in self._TRAINING_EXP_BLOCKED):
                return False
            return self._attr_matches_priority(attr_name)

        if type_id == "attribute_boost":
            attr_name = item_info.get("attr_name", "")
            return self._attr_matches_priority(attr_name)

        if type_id == "random_effect":
            attr_name = item_info.get("attr_name", "")
            return self._attr_matches_priority(attr_name)

        if type_id == "status_buff":
            return True

        if type_id == "money_gain":
            return True

        if type_id == "consumable":
            return True

        # 自动发现的新类型 (type 名就是效果关键词): 默认允许购买
        return True

    # 属性名称同义词 (OCR 可能用不同的字)
    # 注意: 体力≠耐力! 体力=生命属性值, 耐力=训练体力条(stamina)
    ATTR_SYNONYMS = {
        "体力":  ["体力", "生命", "HP"],              # 体力 = HP
        "力量":  ["力量", "攻击力", "攻击"],            # 力量 = Attack
        "韧性":  ["韧性", "防御力", "防御"],            # 韧性 = Defense
        "速度":  ["速度", "迅捷"],                     # Speed
        "保护":  ["保护", "效果抵抗", "抗性", "抵抗"],   # 保护 = Effect RES
        "专注":  ["专注", "效果命中", "命中率"],         # 专注 = Effect Hit
        "暴击率": ["暴击率", "暴击", "会心率", "会心"],   # Crit Rate
        "暴伤":  ["暴伤", "暴击伤害", "会心伤害"],       # Crit DMG
    }

    # 所有可识别的属性名, 用于解析效果描述中的属性词
    _ALL_ATTR_NAMES = [
        "体力", "力量", "韧性", "专注", "保护",
        "速度", "暴击率", "暴击伤害", "暴伤",
        "攻击力", "防御力", "生命",
        "效果抗性", "效果命中",
    ]

    def _attr_matches_priority(self, attr_name: str) -> bool:
        """检查 OCR 提取的属性名是否包含优先属性关键词 (含同义词)"""
        if not attr_name or not self.attribute_match:
            return False
        synonyms = self.ATTR_SYNONYMS.get(self.attribute_match, [self.attribute_match])
        return any(syn in attr_name for syn in synonyms)

    def _parse_random_attrs(self, text: str) -> str:
        """从随机效果物品的详情 OCR 文本中解析被加成的属性名, 以逗号连接"""
        found = []
        for attr in self._ALL_ATTR_NAMES:
            if attr in text:
                found.append(attr)
        return ",".join(found) if found else "随机"

    def _score_item(self, type_id: str, item_info: dict) -> int:
        """评分: 基础分 = 100 - rank * 15, 加 value 附加值. 支持物品级 priority 覆盖."""
        priority = item_info.get("priority_override")
        if priority is not None:
            base = 100 - priority * 15
        else:
            rank = self._priority_rank.get(type_id, 99)
            base = 100 - rank * 15
        return base + item_info.get("value", 0)

    # ===================== 古币不足检测 =====================

    INSUFFICIENT_MONEY_REGIONS = [
        (0.30, 0.40, 0.40, 0.20),   # 对话框中央
        (0.35, 0.35, 0.30, 0.30),   # 宽区域
    ]

    @classmethod
    def _check_insufficient_money(cls, screenshot, ocr_engine) -> bool:
        """检测是否弹出 '古币不足' 对话框"""
        if screenshot is None:
            return False
        for rx, ry, rw, rh in cls.INSUFFICIENT_MONEY_REGIONS:
            text = _ocr_region(ocr_engine, screenshot, rx, ry, rw, rh)
            if text and ("古币不足" in text or "古币" in text):
                return True
        return False

    @classmethod
    def _check_all_items_unaffordable(cls, screenshot, ocr_engine) -> bool:
        """检测交易界面是否没有可购买的物品 (资金不足导致购买按钮灰掉)"""
        if screenshot is None:
            return False
        from .recognition import ScreenDetector
        detector = ScreenDetector(ocr_engine)
        # 如果还在交易界面但没有购买按钮, 可能是买不了
        if detector.is_trade_screen(screenshot):
            # 检查是否有 "购买" 按钮
            return "购买" not in detector.read_region_text(
                screenshot, 0.78, 0.85, 0.18, 0.12)
        return False

    @staticmethod
    def _dismiss_dialog(ctx: HandlerContext):
        """关闭对话框 (点击中央)"""
        ctx.controller.click_center_multi(count=10, interval=0.01, click_delay=0.01)
        time.sleep(0.2)

    # ===================== 评鉴战流程 =====================

    def _click_appraisal(self, ctx: HandlerContext) -> bool:
        """在训练主屏点击评鉴战按钮. 等待主菜单稳定后再点击."""
        from .recognition import ScreenDetector
        # 等待主菜单出现 (退出交易界面有动画过渡)
        for _ in range(8):
            screenshot = ctx.capture.capture_game()
            if screenshot is not None:
                detector = ScreenDetector(ctx.ocr)
                if detector.is_main_menu_screen(screenshot):
                    break
            time.sleep(0.1)
        ctx.controller.click_at_percent(self.APPRAISAL_BUTTON_X, self.APPRAISAL_BUTTON_Y)
        time.sleep(0.15)
        return True

    def _handle_appraisal_screen(self, ctx: HandlerContext) -> bool:
        """评鉴战界面: 点击接受 → SkipHandler 会自动处理后续跳过战斗"""
        ctx.controller.click_at_percent(self.APPRAISAL_ACCEPT_X, self.APPRAISAL_ACCEPT_Y)
        time.sleep(0.15)
        return True

    # ===================== 退出交易 =====================

    def _exit_trade(self, ctx: HandlerContext):
        """退出交易界面, 返回主屏"""
        ctx.controller.send_escape()
        time.sleep(0.1)
        ctx.controller.click_center_multi(count=5, interval=0.01, click_delay=0.01)
        time.sleep(0.15)


# ============================================================
# JourneyEndHandler (Priority 19) - 旅程结束/技能选择
# ============================================================

class JourneyEndException(Exception):
    """旅程结束, 脚本正常终止"""
    pass


class JourneyTransitionHandler(Handler):
    """单次推进阶段目标达成/新目标提示层；它们并不表示整局旅程结束。"""

    @property
    def priority(self): return 18

    @property
    def name(self): return "目标提示"

    def can_handle(self, frame, ocr_engine):
        text = re.sub(r"\s+", "", " ".join(
            item[0] for item in frame.recognize_full(ocr_engine)))
        if "达成目标" in text and "点击以继续" in text:
            return True
        # 新目标卡片位于中央下方。右上角有 HUD 的普通主菜单不能命中。
        center = frame.get_ocr(ocr_engine, 0.30, 0.73, 0.40, 0.19)
        if ("地区移动" in center and
                ("D-DAY" in center.upper() or "参与" in center or "远征" in center)):
            return True
        return (any(word in center for word in ("距目标", "距离目标")) and
                "评鉴战" in center and "/1250" not in text and
                "RANK" not in text.upper())

    def handle(self, ctx):
        print("[目标提示] 推进阶段目标提示，继续当前旅程")
        ctx.controller.click_at_percent(0.50, 0.87)
        time.sleep(0.7)
        ctx.pending_screenshot = ctx.capture.capture_game()
        return True


class JourneyEndHandler(Handler):
    """
    旅程结束界面处理 + 技能自动选择。

    流程:
    1. 检测旅程结束界面 (多区域OCR)
    2. 如开启 auto_learn_skills:
       a. 点击"潜质"标签页
       b. OCR 可习得技能列表
       c. 从 profiles/skills/{profile}.json 加载技能关键词 → 优先级映射
       d. 按构建方向匹配技能, 选优先级最高的
       e. 点击技能 → 可习得 → 确认
       f. 循环直到潜质点不足或无匹配
    3. 推进旅程结束流程 (连续点击中央)
    """

    # 旅程结束检测区域 (多区域覆盖不同位置的文字)
    JOURNEY_END_REGIONS = [
        (0.34, 0.12, 0.32, 0.12),
        (0.30, 0.18, 0.40, 0.14),
        (0.38, 0.27, 0.28, 0.08),
        (0.42, 0.88, 0.20, 0.08),
        (0.74, 0.50, 0.22, 0.10),
        (0.72, 0.48, 0.24, 0.14),
    ]

    # 潜质标签
    TAB_CLICK_X = 0.71
    TAB_CLICK_Y = 0.96

    # 技能列表区域: 左半屏 5 个可见槽位
    SKILL_SLOT_REGIONS = [
        (0.04, 0.22, 0.52, 0.08),
        (0.04, 0.30, 0.52, 0.08),
        (0.04, 0.38, 0.52, 0.08),
        (0.04, 0.46, 0.52, 0.08),
        (0.04, 0.54, 0.52, 0.08),
    ]
    SKILL_SLOT_CLICK_X = 0.28  # 技能名点击位置 (左半屏中央)

    # 技能槽位 Y 点击坐标
    SKILL_SLOT_CLICK_Y = [0.26, 0.34, 0.42, 0.50, 0.58]

    # 可习得按钮
    LEARNABLE_X = 0.69
    LEARNABLE_Y = 0.91

    # 确认按钮
    CONFIRM_X = 0.50
    CONFIRM_Y = 0.85

    # 潜质点数 OCR 区域
    POTENTIAL_POINTS_REGION = (0.04, 0.68, 0.15, 0.10)

    @property
    def priority(self): return 19

    @property
    def name(self): return "旅程结束"

    def __init__(self, config=None):
        self.config = config or {}
        je_cfg = self.config.get("journey_end", {})
        self._auto_learn = je_cfg.get("auto_learn_skills", False)
        training_cfg = self.config.get("training", {})
        self._profile_name = training_cfg.get("rule_profile", "default")
        self._skills: list[dict] = []
        self._load_skill_profile()

    def _load_skill_profile(self):
        """加载技能选择配置"""
        path = Path(f"profiles/skills/{self._profile_name}.json")
        if not path.exists():
            path = Path("profiles/skills/default.json")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._skills = data.get("skills", [])
            print(f"[技能] 已加载 {len(self._skills)} 个技能定义 ({self._profile_name})")
        except Exception as e:
            print(f"[技能] 加载失败: {e}")
            self._skills = []

    def can_handle(self, frame: FrameContext, ocr_engine) -> bool:
        screenshot = frame.screenshot
        if screenshot is None:
            return False

        texts = []
        for rx, ry, rw, rh in self.JOURNEY_END_REGIONS:
            t = frame.get_ocr(ocr_engine, rx, ry, rw, rh)
            if t:
                texts.append(t)
        combined = " ".join(texts)
        if self._is_journey_end_text(combined):
            return True
        # PrintWindow/HybridOCR 在部分结算页的局部裁剪上会返回空文本，
        # 而全屏 OCR 仍能稳定读出“达成目标/点击以继续”。
        try:
            full_text = " ".join(item[0] for item in frame.recognize_full(ocr_engine))
        except Exception:
            full_text = ""
        return self._is_journey_end_text(full_text)

    def _is_journey_end_text(self, text: str) -> bool:
        if not text:
            return False
        # 旅程结算的目标达成过渡页只有一个底部“点击以继续”按钮，
        # 先由本处理器精准推进到最终结算页，再执行技能选择/退出。
        # 阶段目标达成、奖励中的“获得潜质点数”都可能出现在旅程中途。
        # 只接受明确整局结束文字，目标卡片由 JourneyTransitionHandler 推进。
        if "旅程结束" in text or "继承旅程" in text:
            return True
        if "旅程" in text and ("画下句号" in text or "告别" in text):
            if "古币" in text or "护符" in text or "奖励" in text:
                return True
        return False

    def handle(self, ctx: HandlerContext) -> bool:
        print("[旅程结束] 检测到")

        # 目标达成过渡页不是技能列表，不能直接点潜质标签。
        screenshot = ctx.capture.capture_game()
        if screenshot is not None:
            goal_text = " ".join(
                _ocr_region(ctx.ocr, screenshot, rx, ry, rw, rh)
                for rx, ry, rw, rh in self.JOURNEY_END_REGIONS
            )
            if not goal_text.strip():
                try:
                    goal_text = " ".join(
                        item[0] for item in ctx.ocr.recognize_detailed(screenshot)
                    )
                except Exception:
                    goal_text = ""
            if "达成目标" in goal_text and "点击以继续" in goal_text:
                print("[旅程结束] 目标达成过渡页 → 点击底部继续")
                ctx.controller.click_at_percent(0.50, 0.89)
                time.sleep(0.8)
                ctx.pending_screenshot = ctx.capture.capture_game()
                return True

        # 已到最终“旅程结束”结算页时交还给人工接管。结算页后续的
        # 潜质/奖励操作不属于旅程推进，不能反复点击或进入商店处理。
        if self.config.get("journey_end", {}).get("pause_for_manual", True):
            print("[旅程结束] 已到最终结算页，暂停等待人工接管")
            raise JourneyEndException("旅程结束，等待人工接管")

        if not self._auto_learn:
            print("[旅程结束] auto_learn_skills=false, 退出")
            raise JourneyEndException("旅程结束")

        # --- 自动学习技能 ---
        direction = self._get_direction(ctx)
        print(f"[旅程结束] 构建方向={direction}, 开始技能选择...")

        # 读取潜质点数
        screenshot = ctx.capture.capture_game()
        if screenshot is None:
            return False
        potential = self._read_potential(screenshot, ctx)
        print(f"[旅程结束] 潜质点数={potential}")

        # 点击潜质标签
        ctx.controller.click_at_percent(self.TAB_CLICK_X, self.TAB_CLICK_Y)
        time.sleep(0.6)

        # 技能选择循环
        selected_count = 0
        for attempt in range(5):  # 最多选5个技能
            screenshot = ctx.capture.capture_game()
            if screenshot is None:
                break

            # OCR 各技能槽位
            slots = []
            for i, (rx, ry, rw, rh) in enumerate(self.SKILL_SLOT_REGIONS):
                text = _ocr_region(ctx.ocr, screenshot, rx, ry, rw, rh)
                if text and len(text.strip()) > 2:
                    slots.append({"index": i, "text": text.strip()})

            if not slots:
                print("[旅程结束] 无可见技能槽位")
                break

            # 匹配技能配置
            best = self._find_best_skill(slots)
            if best is None:
                print("[旅程结束] 无匹配技能, 停止选择")
                break

            skill_name = best["skill"]["name"]
            slot_idx = best["slot_index"]
            priority = best["skill"]["priority"]
            print(f"[旅程结束] 选中技能: {skill_name} (优先级={priority}, 槽位={slot_idx})")

            # 点击技能槽位
            click_y = self.SKILL_SLOT_CLICK_Y[min(slot_idx, len(self.SKILL_SLOT_CLICK_Y) - 1)]
            ctx.controller.click_at_percent(self.SKILL_SLOT_CLICK_X, click_y)
            time.sleep(0.3)

            # 点击可习得按钮
            ctx.controller.click_at_percent(self.LEARNABLE_X, self.LEARNABLE_Y)
            time.sleep(0.4)

            # 点击确认
            ctx.controller.click_at_percent(self.CONFIRM_X, self.CONFIRM_Y)
            time.sleep(0.5)

            selected_count += 1
            # 短暂等待UI更新
            time.sleep(0.3)

        print(f"[旅程结束] 共选择 {selected_count} 个技能")

        # 推进结算
        print("[旅程结束] 推进结算...")
        ctx.controller.click_center_multi(count=12, interval=0.02, click_delay=0.02)
        time.sleep(1.5)
        return True

    def _read_potential(self, screenshot, ctx) -> int:
        """读取潜质点数"""
        pts_text = _ocr_region(ctx.ocr, screenshot, *self.POTENTIAL_POINTS_REGION)
        if pts_text:
            nums = re.findall(r'\d+', pts_text.replace(",", ""))
            if nums:
                return int(nums[0])
        return 999  # 未知时假设充足

    def _find_best_skill(self, slots: list[dict]) -> dict | None:
        """在可读槽位中查找优先级最高的匹配技能"""
        best = None
        best_priority = -1
        for slot in slots:
            text = slot["text"]
            for skill in self._skills:
                for kw in skill.get("keywords", []):
                    if kw in text:
                        prio = skill.get("priority", 0)
                        if prio > best_priority:
                            best_priority = prio
                            best = {"skill": skill, "slot_index": slot["index"], "match_keyword": kw}
                        break  # 一个技能只统计一次
        return best

    def _get_direction(self, ctx: HandlerContext) -> str:
        """获取当前构建方向"""
        try:
            if ctx.engine.is_rule_engine_mode:
                return ctx.engine.rule_profile.legacy_strategy.build_direction.value
        except Exception:
            pass
        try:
            attr_id = ctx.engine.priority
            if attr_id == 1:
                return "speed"
            elif attr_id == 2:
                return "attack"
            elif attr_id == 3:
                return "survival"
        except Exception:
            pass
        return "default"


# ============================================================
# UnknownHandler (Priority 99) - 未知界面兜底
# ============================================================

class UnknownHandler(Handler):
    """
    未知界面观察处理器:
    1. 只轮询截图和已知界面检测
    2. 发现体力条右侧的剧情跳过按钮时只点击一次
    3. 不点击屏幕中央，不抢占用户鼠标
    """

    ADVANCE_TIMEOUT = 30.0       # 单轮推进超时 (秒)

    @property
    def priority(self): return 99

    @property
    def name(self): return "未知兜底"

    def can_handle(self, frame, ocr_engine) -> bool:
        return True

    def handle(self, ctx: HandlerContext) -> bool:
        ctx.consecutive_unknown += 1
        count = ctx.consecutive_unknown

        print(f"[未知] 观察并寻找跳过按钮({count})")
        return self._smart_advance(ctx)

    # ===================== 智能推进 =====================

    def _smart_advance(self, ctx: HandlerContext) -> bool:
        """只观察未知画面；找到剧情跳过按钮时点击一次并等待转场。"""
        from .recognition import ScreenDetector

        deadline = time.perf_counter() + self.ADVANCE_TIMEOUT
        skip_clicked = False
        started = time.perf_counter()

        while time.perf_counter() < deadline:
            screenshot = ctx.capture.capture_game()
            if screenshot is None:
                time.sleep(0.15)
                continue

            detector = ScreenDetector(ctx.ocr)
            if self._is_known_screen(screenshot, detector, ctx.ocr):
                elapsed = time.perf_counter() - started
                print(f"  → 检测到已知界面 ({elapsed:.1f}s)")
                ctx.pending_screenshot = screenshot
                return True

            # 事件选项页也可能显示顶部图标；有选项时绝不能把它当剧情跳过。
            skip_position = detector.find_story_skip(screenshot)
            has_options = bool(detector.read_event_option_rows(screenshot))
            if skip_position and not has_options and not skip_clicked:
                print("  → 使用体力条右侧第一个剧情跳过按钮")
                ctx.controller.click_at_percent(*skip_position)
                skip_clicked = True
                time.sleep(0.5)
                continue

            time.sleep(0.15)

        print(f"  → 未识别画面仍在等待 ({self.ADVANCE_TIMEOUT}s)，未发送中央点击")
        return True

    @staticmethod
    def _is_known_screen(screenshot, detector, ocr_engine) -> bool:
        """检测是否为已知界面. 按出现概率排序, 早命中早退出."""
        frame = FrameContext(screenshot)
        # 轮询时也检查已实现的中途结果层；只交还调度，不在未知处理器发送输入。
        if any(handler.can_handle(frame, ocr_engine) for handler in (
                InventoryHandler(), TrainingResultHandler(), AttributeGainHandler(), RewardAdvanceHandler(),
                JourneyTransitionHandler(), StoryResultHandler())):
            return True
        # 加载转为剧情时退出中心连点，让剧情跳过处理器使用顶部图标。
        if detector.find_story_skip(screenshot) and detector.is_story_dialogue(screenshot):
            return True
        marker = detector.read_region_text(screenshot, 0.035, 0.14, 0.40, 0.16)
        if '事件' in marker and detector.read_event_option_rows(screenshot):
            return True
        # 1. 事件界面 (训练后最常出现) — 需同时满足标记+选项, 否则是过场
        if detector.is_event_marker_screen(screenshot):
            if UnknownHandler._has_event_options(screenshot, ocr_engine):
                return True
            # 只有标记没有选项 → 过场事件, 不停止

        # 2. 卡片奖励选择 (训练后可能弹出)
        title = _ocr_region(ocr_engine, screenshot, 0.02, 0.02, 0.28, 0.10)
        if title and ("选择奖励" in title or "奖励选择" in title):
            return True

        # 3. 主菜单
        if detector.is_main_menu_screen(screenshot):
            return True

        # 4. 训练界面
        if detector.is_training_screen(screenshot):
            return True

        # 5. 休息界面
        if detector.is_rest_screen(screenshot):
            return True

        # 6. 商店事件 / 交易
        if detector.is_shop_event_screen(screenshot):
            return True

        return False

    @staticmethod
    def _has_event_options(screenshot, ocr_engine) -> bool:
        """检测事件屏幕是否有可点击选项 (区分过场事件和真事件)."""
        text = _ocr_region(ocr_engine, screenshot, 0.55, 0.45, 0.40, 0.35)
        if not text:
            return False
        # 复用 EventHandler 的选项检测逻辑
        if text.count('+') >= 2 or text.count('。') >= 2:
            return True
        if any(kw in text for kw in ["该怎么办", "怎么办", "请选择", "是否", "要不要"]):
            return True
        zh_count = sum(1 for c in text if '一' <= c <= '鿿')
        return zh_count >= 8 and any(c in text for c in "？?呢吧")


# ============================================================
# 处理器链调度器
# ============================================================

class HandlerChain:
    """处理器优先级链 (移植自 SleepRunner RaceRunner 的 Handler 调度)"""

    def __init__(self, handlers: List[Handler]):
        self.handlers = sorted(handlers, key=lambda h: h.priority)
        self._consecutive_misses = 0

    @property
    def adaptive_wait(self) -> float:
        """自适应等待时间 (根据连续未命中次数渐进延长)"""
        if self._consecutive_misses <= 1:
            return 0.05
        elif self._consecutive_misses == 2:
            return 0.15
        elif self._consecutive_misses == 3:
            return 0.25
        else:
            return 0.4

    def dispatch(self, frame: FrameContext, ctx: HandlerContext, ocr_engine) -> str:
        """
        调度处理器: 找到第一个能处理的 Handler 并执行。

        Returns: 处理的 Handler 名 (用于日志)
        """
        for handler in self.handlers:
            try:
                if handler.can_handle(frame, ocr_engine):
                    self._consecutive_misses = 0
                    handler.handle(ctx)
                    return handler.name
            except Exception as e:
                print(f"[{handler.name}] 异常: {e}")

        self._consecutive_misses += 1
        return "none"
