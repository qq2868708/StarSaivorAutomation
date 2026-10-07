"""
训练主循环 - 完全移植自 SleepRunner 的 Handler 流水线架构

核心改进 (v5.0):
  - 后台窗口截图 (PrintWindow/BitBlt), 窗口被遮挡也能工作
  - 后台点击 (PostMessage), 不移动实际鼠标光标
  - 处理器优先级链: Skip → Event → Training → Rest → MainMenu → Unknown
  - 自适应轮询: 连续未命中时渐进延长等待时间
  - 训练后自动点击推进加载/结算画面
  - 事件选择自动处理

SleepRunner Handler 映射:
  SkipHandler        → 检测 SKIP 按钮快进对话
  EventHandler       → 检测事件选项并自动选择
  TrainingHandler    → 训练决策 (规则引擎/传统模式)
  RestHandler        → 休息预算制选择
  MainMenuHandler    → 从主菜单进训练
  UnknownHandler     → 点击推进 + 兜底
"""
from __future__ import annotations

import json
import sys
import time
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .capture import ScreenCapture
from .controller import Controller, InputTargetError
from .recognition import GameUI, OCREngine, GameStateReader, ScreenDetector
from .decision import DecisionEngine, TrainingOption, GameState
from .rule_engine import (
    TrainingDecisionContext,
    TrainingDecisionResult,
    TrainingRuleEngine,
    TrainingRuleField,
    TrainingDecisionAction,
    TrainingRuleLoader,
    BuildDirection,
)
from .fail_rate_ocr import (
    parse_fail_rate_from_ocr_texts,
    UNKNOWN_FAIL_RATE_FALLBACK,
)
from .icon_counter import count_circular_icons
from .advantage_detector import detect_advantage_training
from .handlers import (
    FrameContext,
    HandlerContext,
    HandlerChain,
    SkipHandler,
    InventoryHandler,
    BattleDefeatHandler,
    StoryResultHandler,
    BattleLeaveHandler,
    CampSelectHandler,
    MovePlatformHandler,
    EventHandler,
    EventDecisionPending,
    CardSelectHandler,
    CommissionHandler,
    ShopHandler,
    TrainingResultHandler,
    AttributeGainHandler,
    RewardAdvanceHandler,
    TrainingHandler,
    RestHandler,
    JourneyEndHandler,
    JourneyEndException,
    JourneyTransitionHandler,
    MainMenuHandler,
    AppraisalHandler,
    UnknownHandler,
    preload_profiles,
)
from .arcanum import ArcanumDetector, calibrate_arcanum
from .logger import RunLogger


# ============================================================
# 相对坐标常量
# ============================================================

_TRAINING_ROWS = [
    ("力量训练", 0.86, 0.28),
    ("体力训练", 0.86, 0.38),
    ("韧性训练", 0.86, 0.48),
    ("集中训练", 0.86, 0.58),
    ("保护训练", 0.86, 0.68),
]

_TRAIN_BTN_X = 0.89
_TRAIN_BTN_Y = 0.89

_CLICK_ROW_WAIT = 0.8
_CLICK_TRAIN_WAIT = 0.12

# 训练后点击推进 (新增)
_POST_TRAIN_CLICKS = 8
_POST_TRAIN_CLICK_INTERVAL = 0.01
_POST_TRAIN_WAIT = 0.05

_CONFIRM_REST_BTN_X = 0.89
_CONFIRM_REST_BTN_Y = 0.925

_MAIN_MENU_DIAMOND_X = 0.91
_MAIN_MENU_TRAIN_Y = 0.40
_MAIN_MENU_REST_Y = 0.626

# 属性面板区域 (校准: 0.035~0.175, 0.28~0.78)
_STAT_PANEL = (0.035, 0.28, 0.140, 0.50)
_STAT_NAMES = ["力量", "体力", "韧性", "专注", "保护"]

_REST_CLICK_POINTS = [
    (0.83, 0.37),   # 露宿 (免费)
    (0.84, 0.47),   # 住处 ($30)
    (0.85, 0.57),   # 冥想室 ($60)
]


# ============================================================
# 训练指标扫描状态
# ============================================================

class TrainingMetricScanState:
    """跟踪哪些指标已被扫描，支持懒加载"""

    def __init__(self):
        self.icon_counts = [0] * 5
        self.fail_rates = [0] * 5
        self.gain_values = [0] * 5
        self.is_flash = [False] * 5
        self.is_advantage = [False] * 5
        self.known_icon_mask = 0
        self.known_fail_rate_mask = 0
        self.known_gain_mask = 0
        self.known_flash_mask = 0
        self.known_advantage_mask = 0
        self.strength_stat: Optional[int] = None
        self.stamina_stat: Optional[int] = None
        self.stamina_bar_pct: float = 100.0  # OCR体力条百分比

    def has_icon_count(self, row_index: int) -> bool:
        return (self.known_icon_mask & (1 << row_index)) != 0

    def has_fail_rate(self, row_index: int) -> bool:
        return (self.known_fail_rate_mask & (1 << row_index)) != 0

    def has_gain(self, row_index: int) -> bool:
        return (self.known_gain_mask & (1 << row_index)) != 0

    def has_flash(self, row_index: int) -> bool:
        return (self.known_flash_mask & (1 << row_index)) != 0

    def update_stats(self, strength: Optional[int], stamina: Optional[int]) -> bool:
        prev_strength = self.strength_stat
        prev_stamina = self.stamina_stat
        if strength is not None:
            self.strength_stat = (max(self.strength_stat, strength)
                                  if self.strength_stat is not None else strength)
        if stamina is not None:
            self.stamina_stat = (max(self.stamina_stat, stamina)
                                 if self.stamina_stat is not None else stamina)
        return self.strength_stat != prev_strength or self.stamina_stat != prev_stamina

    def set_row_metrics(self, row_index: int, icon_count: int, fail_rate: int, gain: int = 0,
                        is_flash: bool = False, is_advantage: Optional[bool] = None):
        self.icon_counts[row_index] = icon_count
        self.fail_rates[row_index] = fail_rate
        self.gain_values[row_index] = gain
        self.is_flash[row_index] = is_flash
        # 优势标记由当前训练页一次性检测；逐行读取失败率时不能把它覆盖掉。
        if is_advantage is not None:
            self.is_advantage[row_index] = bool(is_advantage)
            self.known_advantage_mask |= 1 << row_index
        self.known_icon_mask |= 1 << row_index
        self.known_fail_rate_mask |= 1 << row_index
        self.known_gain_mask |= 1 << row_index
        self.known_flash_mask |= 1 << row_index

    def to_decision_context(self, profile, profile_name: str,
                            adventure_gain_threshold: int = 100,
                            target_stats: dict = None,
                            current_stats: dict = None) -> TrainingDecisionContext:
        strategy = profile.legacy_strategy
        return TrainingDecisionContext(
            icon_counts=list(self.icon_counts),
            fail_rates=list(self.fail_rates),
            gain_values=list(self.gain_values),
            is_flash=list(self.is_flash),
            is_advantage=list(self.is_advantage),
            known_icon_mask=self.known_icon_mask,
            known_fail_rate_mask=self.known_fail_rate_mask,
            known_gain_mask=self.known_gain_mask,
            known_flash_mask=self.known_flash_mask,
            known_advantage_mask=self.known_advantage_mask,
            strength_stat=self.strength_stat,
            stamina_stat=self.stamina_stat,
            stamina_pct=self.stamina_bar_pct,
            build_direction=strategy.build_direction,
            legacy_fail_rate_threshold=strategy.fail_rate_threshold,
            legacy_rush_threshold=strategy.rush_threshold,
            adventure_gain_threshold=adventure_gain_threshold,
            profile_name=profile_name,
            target_stats=target_stats or {},
            current_stats=current_stats or {},
            advantage_multiplier=getattr(self, "_advantage_multiplier", 1.35),
        )


# ============================================================
# Trainer 主类
# ============================================================

class Trainer:
    def __init__(self, config, development_mode=False):
        self.config = config
        # 开发阶段允许 UnknownHandler 持续观察并收集现场；正式运行默认
        # 仍由连续未知/超时保护安全暂停。通过 main.py --develop 显式开启。
        self._development_mode = bool(development_mode)
        game_cfg = config.get("game", {})
        op_cfg = config.get("operation", {})

        self.capture = ScreenCapture(
            window_title=game_cfg.get("window_title", "StarSavior")
        )
        self.controller = Controller(
            click_delay=op_cfg.get("click_delay", 0.5),
            action_interval=op_cfg.get("action_interval", 1.0),
            auto_refocus=development_mode,
        )
        self.ui = GameUI(self.capture, config)
        self.engine = DecisionEngine(config)

        self.max_rounds = op_cfg.get("max_rounds", 100)
        self.round_count = 0
        self._exit_reason = "max_rounds"

        # 构建日志器
        # 扫描模式: "fast"=只读主属性 | "normal"=读力量体力韧性 | "focus"=全5行扫描
        training_cfg = config.get("training", {})
        self._scan_mode = training_cfg.get("scan_mode", "normal")
        self._advantage_multiplier = float(training_cfg.get("advantage_multiplier", 1.35) or 1.35)
        if self._advantage_multiplier < 1.0:
            self._advantage_multiplier = 1.0

        profile_name = self.engine.rule_profile_name if self.engine.is_rule_engine_mode else "traditional"
        build_dir = ""
        if self.engine.is_rule_engine_mode:
            build_dir = self.engine.rule_profile.legacy_strategy.build_direction.value
        self.logger = RunLogger(
            profile_name=profile_name,
            build_direction=build_dir,
        )

        # 状态缓存
        self._last_stamina: Optional[float] = None
        self._last_mood: str = "Normal"
        self._last_money: int = 0

        # 构建处理器链
        self._handlers = HandlerChain([
            SkipHandler(),
            InventoryHandler(),
            BattleDefeatHandler(),
            StoryResultHandler(),
            AppraisalHandler(),
            BattleLeaveHandler(),
            CampSelectHandler(),
            MovePlatformHandler(config),
            CommissionHandler(config),
            ShopHandler(config),
            AttributeGainHandler(),
            TrainingResultHandler(),
            EventHandler(config),
            RewardAdvanceHandler(),
            CardSelectHandler(config),
            TrainingHandler(),
            RestHandler(config),
            JourneyTransitionHandler(),
            JourneyEndHandler(config),
            MainMenuHandler(),
            UnknownHandler(),
        ])

        # Arcanum 支援卡检测器
        self._arcanum = ArcanumDetector(config)

        # 合宿策略配置
        camp_cfg = config.get("camp", {})
        self._camp_start_turn = camp_cfg.get("camp_start_turn", 30)
        self._camp_end_turn = camp_cfg.get("camp_end_turn", 38)
        self._pre_camp_start_turn = camp_cfg.get("pre_camp_start_turn", 25)
        self._pre_camp_min_gain = camp_cfg.get("pre_camp_min_gain", 35)
        self._camp_min_coins = camp_cfg.get("camp_min_coins", 80)
        self._camp_rest_fail_threshold = camp_cfg.get("camp_rest_fail_threshold", 50)
        self._camp_type = camp_cfg.get("camp_type", "strength")  # strength=力量加成, stamina=体力加成
        self._camp_high_gain_threshold = camp_cfg.get("camp_high_gain_threshold", 45)
        # 合宿加成属性对应的行索引
        self._camp_bonus_row = 0 if self._camp_type == "strength" else 1

        # 回合追踪: 首次进入主菜单时从游戏读取, 之后每次训练/休息递增
        arcanum_cfg = config.get("arcanum", {})
        self._turn_limit = arcanum_cfg.get("turn_limit", 10)
        self._icon_priority_limit = arcanum_cfg.get("icon_priority_limit", 20)
        self._turn_menu_pos = (
            arcanum_cfg.get("turn_menu_x", 0.0656),
            arcanum_cfg.get("turn_menu_y", 0.0694),
        )
        turn_ocr = arcanum_cfg.get("turn_ocr_region", [0.25, 0.68, 0.25, 0.12])
        self._turn_ocr_region = tuple(turn_ocr)

        # Handler 上下文
        self._handler_ctx = HandlerContext(
            capture=self.capture,
            controller=self.controller,
            ui=self.ui,
            engine=self.engine,
            logger=self.logger,
        )
        self._handler_ctx.set_turn_detector(self._detect_game_turn)
        self._handler_ctx.development_mode = self._development_mode

        # 保护机制
        safety_cfg = config.get("safety", {})
        self._max_runtime_seconds = safety_cfg.get("max_runtime_minutes", 45) * 60
        self._stuck_threshold = safety_cfg.get("stuck_threshold", 5)
        self._round_timeout_seconds = safety_cfg.get("round_timeout_seconds", 60)
        self._exit_reason = "unknown"
        self._consecutive_unknown = 0
        self._last_handler_name = ""
        self._same_handler_count = 0
        self._last_branch_text = ""
        self._round_start_time = 0.0
        self._last_stuck_check_round = -1

        # 属性目标值 (用于训练评分降权)
        target_cfg = config.get("training", {}).get("target_stats", {})
        self._target_stats = {
            "力量": target_cfg.get("strength", 0) or 0,
            "体力": target_cfg.get("stamina", 0) or 0,
            "韧性": target_cfg.get("agility", 0) or 0,
            "专注": target_cfg.get("focus", 0) or 0,
            "保护": target_cfg.get("guard", 0) or 0,
        }
        self._target_stats_enabled = any(v > 0 for v in self._target_stats.values())

    # ===================== 主循环 =====================

    def run(self, auto_start=False):
        if not auto_start and not sys.stdin.isatty():
            raise RuntimeError("请通过启动训练.bat 在交互窗口运行；环境检查使用 --check")
        mode = "规则引擎" if self.engine.is_rule_engine_mode else "传统权重"
        print("\n" + "=" * 60)
        print(f"  StarSavior 训练自动化机器人 v5.0")
        print(f"  决策模式: {mode}")
        print(f"  截图方式: PrintWindow；黑屏时使用前台客户区截图")
        print(f"  点击方式: SendInput (硬件级, 保存恢复光标)")
        if self.engine.is_rule_engine_mode:
            print(f"  规则配置: {self.engine.rule_profile_name}")
            s = self.engine.rule_profile.legacy_strategy
            print(f"  构建方向: {s.build_direction.value}, "
                  f"失败率阈值={s.fail_rate_threshold}%, rush={s.rush_threshold}")
        scan_labels = {"fast": "快速(1行)", "normal": "普通(3行)", "focus": "专注(5行全扫)"}
        label = scan_labels.get(self._scan_mode, self._scan_mode)
        print(f"  扫描模式: {label}")
        print(f"  优势训练: 蓝色大拇指标记 ×{self._advantage_multiplier:.2f}")
        print(f"  未知界面模式: {'开发持续观察' if self._development_mode else '正式安全暂停'}")
        if self.engine.use_icon_counting and self._icon_priority_limit > 0:
            print(f"  图标优先: 前{self._icon_priority_limit}回合 (羁绊优先)")
        print("=" * 60)

        region = self.capture.find_window()
        if region is None:
            raise RuntimeError("无法找到 StarSavior 游戏窗口，请先打开游戏并进入旅程")

        self.controller.set_game_region(region)
        self.controller.set_window_handle(self.capture.get_hwnd())
        print(f"[信息] 游戏窗口: {region[2]}x{region[3]}")

        self._print_config()

        # 属性目标确认 (始终询问用户输入期望值)
        print("\n--- 属性目标 (训练评分降权用) ---")
        if self._target_stats_enabled:
            for name, val in self._target_stats.items():
                status = "(目标)" if val > 0 else "(无限制)"
                print(f"  {name}: {val} {status}")
        else:
            print("  尚未设置目标值, 训练评分将不启用目标接近降权")
        if not auto_start and sys.stdin.isatty():
            try:
                resp = input("\n输入各属性期望值 (如 力量=500 体力=400) 或按 Enter 跳过: ").strip()
                while resp:
                    for part in resp.split():
                        if "=" in part:
                            k, v = part.split("=", 1)
                            for name in _STAT_NAMES:
                                if k in name:
                                    try:
                                        self._target_stats[name] = int(v)
                                        print(f"  {name} → {v}")
                                    except ValueError:
                                        pass
                                    break
                    resp = input("继续修改? (Enter 完成): ").strip()
                self._target_stats_enabled = any(v > 0 for v in self._target_stats.values())
                if self._target_stats_enabled:
                    print("  目标已保存:")
                    for name, val in self._target_stats.items():
                        if val > 0:
                            print(f"    {name}: {val}")
                else:
                    print("  未设置目标, 跳过目标接近降权")
            except EOFError:
                print("  [跳过] 非交互环境, 使用配置文件中的目标值")

        if not auto_start and sys.stdin.isatty():
            try:
                input("\n按 Enter 开始训练...")
            except EOFError:
                pass
        else:
            print("\n[自动] 直接开始训练...")

        preload_profiles()
        self.controller.focus_game_window()

        try:
            while self.max_rounds == 0 or self.round_count < self.max_rounds:
                self.round_count += 1
                self._round_start_time = time.time()
                self._last_handler_name = ""
                self._same_handler_count = 0
                # 商店的已交易状态随评鉴战流程结束重置，不能每轮清空。
                turn_info = f" T{self._handler_ctx.detected_turn}/45" if self._handler_ctx.detected_turn is not None else ""
                print(f"\n--- 第{self.round_count}轮{turn_info} ---")
                self.logger.start_turn_timing()

                # --- 超时检查 ---
                elapsed = time.time() - self.logger._start_time
                if elapsed > self._max_runtime_seconds:
                    print(f"\n[保护] 超时 ({elapsed / 60:.1f}min > {self._max_runtime_seconds / 60:.0f}min), 触发分析...")
                    self._exit_reason = "timeout"
                    break

                # --- 卡死检查1: 连续 UnknownHandler 轮数 ---
                if self._consecutive_unknown >= self._stuck_threshold and not self._development_mode:
                    print(f"\n[保护] 连续{self._consecutive_unknown}轮Unknown (卡死), 触发分析...")
                    self._exit_reason = "stuck"
                    path = self._save_stuck_screenshot()
                    self._trigger_stuck_analysis("连续UnknownHandler卡死", path)
                    break

                # --- 卡死检查2: 全局每轮超时 (任何handler都可能卡住) ---
                round_elapsed = time.time() - self._round_start_time
                if (self._round_start_time > 0 and round_elapsed > self._round_timeout_seconds
                        and not (self._development_mode and self._consecutive_unknown > 0)):
                    print(f"\n[保护] 轮超时 ({round_elapsed:.0f}s > {self._round_timeout_seconds}s), 触发分析...")
                    self._exit_reason = "stuck"
                    path = self._save_stuck_screenshot()
                    self._trigger_stuck_analysis("回合超时卡死", path)
                    break

                success = self._do_round()
                turn_elapsed = self.logger.end_turn_timing()
                print(f"[计时] 本轮耗时 {turn_elapsed:.1f}s")

                # 检查 _do_round 里设置的卡死标记
                if self._exit_reason == "stuck":
                    break

                if not success:
                    wait_time = self._handlers.adaptive_wait
                    print(f"[等待] 未命中 ({wait_time:.1f}s)...")
                    time.sleep(wait_time)

                if self.max_rounds == 0 or self.round_count < self.max_rounds:
                    time.sleep(0.1)

        except JourneyEndException:
            print("\n[信息] 旅程结束, 脚本正常退出")
            self._exit_reason = "journey_end"
        except EventDecisionPending as error:
            print(f'\n[暂停] {error}')
            self._exit_reason = 'event_reference_pending'
        except KeyboardInterrupt:
            print("\n[信息] 用户中断")
            self._exit_reason = "keyboard_interrupt"
        finally:
            self._save_log()
            print(f"\n[完成] 共执行 {self.round_count} 轮")

    def _do_round(self) -> bool:
        """单轮: 截屏 → 处理器链调度"""
        # 快速路径: handler 已缓存转场后的截图, 直接复用省去 capture_game + dispatch 周期
        if self._handler_ctx.pending_screenshot is not None:
            screenshot = self._handler_ctx.pending_screenshot
            self._handler_ctx.pending_screenshot = None
        else:
            screenshot = self.capture.capture_game()
        if screenshot is None:
            print("[错误] 截屏失败")
            return False

        # Keep the frame that caused this handler decision.  RunLogger stores
        # it next to the action record so the UI can reopen the exact screen.
        if self.logger is not None:
            self.logger.set_screenshot(screenshot)

        frame = FrameContext(screenshot)

        for handler in self._handlers.handlers:
            try:
                if handler.can_handle(frame, self.ui.ocr):
                    self._handlers._consecutive_misses = 0

                    if isinstance(handler, TrainingHandler):
                        self._consecutive_unknown = 0
                        self._last_handler_name = ""
                        self._same_handler_count = 0
                        return self._handle_training_screen(screenshot)
                    else:
                        t0 = time.perf_counter()
                        handler.handle(self._handler_ctx)
                        if isinstance(handler, AppraisalHandler):
                            # 新的评鉴战阶段开始后，下一阶段商店可重新处理。
                            for shop_handler in self._handlers.handlers:
                                if hasattr(shop_handler, '_trade_handled'):
                                    shop_handler._trade_handled = False
                        elapsed = (time.perf_counter() - t0) * 1000
                        self.logger.log_handler_dispatch(handler.name, elapsed)

                        # 快速跳转: handler 已确认目标界面, 直接处理省去 dispatch 周期
                        if self._handler_ctx.pending_action == "training":
                            self._handler_ctx.pending_action = ""
                            self._consecutive_unknown = 0
                            self._last_handler_name = ""
                            self._same_handler_count = 0
                            ss = self._handler_ctx.pending_screenshot
                            self._handler_ctx.pending_screenshot = None
                            if ss is not None:
                                return self._handle_training_screen(ss)

                        # 卡死检测1: 连续 UnknownHandler 匹配 → 卡死
                        if isinstance(handler, UnknownHandler):
                            self._consecutive_unknown += 1
                            if (self._consecutive_unknown >= self._stuck_threshold
                                    and not self._development_mode):
                                print(f"\n[保护] 连续{self._consecutive_unknown}轮Unknown → 卡死!")
                                self._exit_reason = "stuck"
                                path = self._save_stuck_screenshot()
                                self._trigger_stuck_analysis("连续UnknownHandler卡死", path)
                                return False
                        else:
                            self._consecutive_unknown = 0
                            self._handler_ctx.consecutive_unknown = 0
                            # 卡死检测2: 同handler连续触发 (非Unknown/Skip/战后)
                            if handler.name == self._last_handler_name:
                                self._same_handler_count += 1
                                if self._same_handler_count >= self._stuck_threshold:
                                    print(f"\n[保护] 连续{self._same_handler_count}次 {handler.name} → 卡死!")
                                    self._exit_reason = "stuck"
                                    path = self._save_stuck_screenshot()
                                    self._trigger_stuck_analysis(f"连续{handler.name}卡死", path)
                                    return False
                            else:
                                self._last_handler_name = handler.name
                                self._same_handler_count = 1
                        return True
            except (JourneyEndException, InputTargetError, EventDecisionPending):
                raise
            except Exception as e:
                print(f"[{handler.name}] 异常: {e}")

        self._handlers._consecutive_misses += 1
        return False

    # ===================== 训练界面处理 =====================

    def _detect_game_turn(self) -> Optional[int]:
        """只读取当前画面明确显示的 X/45，不点击旧布局的左上角坐标。"""
        img = self.capture.capture_game()
        if img is None:
            return None
        items = self.ui.ocr.recognize(img)
        all_text = " ".join([t[0] for t in items])
        match = re.search(r'(?<!\d)(\d{1,2})\s*/\s*45(?!\d)', all_text)
        if match and 1 <= int(match.group(1)) <= 45:
            turn = int(match.group(1))
            print(f"[回合] {turn}/45")
            return turn
        print("[回合] 当前画面未显示 X/45，使用普通训练规则")
        return None

    def _save_stuck_screenshot(self):
        """卡死时自动保存现场截图供 AI 分析"""
        try:
            import cv2
            from datetime import datetime
            screenshot = self.capture.capture_game()
            if screenshot is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"templates/_stuck_{ts}.png"
                cv2.imwrite(path, screenshot)
                print(f"[保护] 卡死截图已保存: {path}")
                self.logger.stuck_screenshot = path
                return path
        except Exception as e:
            print(f"[保护] 截图保存失败: {e}")
        return None

    def _trigger_stuck_analysis(self, reason: str, screenshot_path: str = None):
        """卡死时写入分析上下文 + 创建 trigger.txt 触发 AI 分析流水线"""
        try:
            import json
            from datetime import datetime
            ai_dir = Path("ai_suggestions")
            ai_dir.mkdir(exist_ok=True)

            # 保存日志
            log_path = self.logger.save(reason)

            # 收集最近回合摘要
            turns_summary = []
            for turn in self.logger._turns[-5:]:
                actions_summary = []
                for a in turn.get("actions", []):
                    t = a.get("type", "?")
                    if t == "decision":
                        actions_summary.append(f"decision: {a.get('action', '?')} (rule={a.get('matched_rule', 'N/A')})")
                    elif t == "rest":
                        actions_summary.append(f"rest: {a.get('option', '?')} cost={a.get('cost', 0)}")
                    elif t == "train":
                        actions_summary.append(f"train: {a.get('attribute', '?')}")
                    elif t == "event":
                        actions_summary.append(f"event: {a.get('event_name', '?')} opt={a.get('option_selected', 0)}")
                    elif t == "shop":
                        actions_summary.append(f"shop: bought={a.get('bought', [])}")
                    elif t == "handler":
                        actions_summary.append(f"handler: {a.get('name', '?')}")
                    else:
                        actions_summary.append(t)
                turns_summary.append({
                    "round": turn["round"],
                    "turn": turn.get("turn"),
                    "stamina": turn.get("state_before", {}).get("stamina"),
                    "mood": turn.get("state_before", {}).get("mood"),
                    "actions": actions_summary,
                })

            profile = self.engine.rule_profile_name if self.engine.is_rule_engine_mode else "traditional"
            bd = "N/A"
            if self.engine.is_rule_engine_mode:
                try:
                    bd = self.engine.rule_profile.legacy_strategy.build_direction.value
                except Exception:
                    pass

            context = {
                "stuck_reason": reason,
                "screenshot": screenshot_path,
                "log_file": str(log_path),
                "round_count": self.round_count,
                "consecutive_unknown": self._consecutive_unknown,
                "profile": profile,
                "build_direction": bd,
                "recent_turns": turns_summary,
            }

            # 写入 latest.md
            latest = ai_dir / "latest.md"
            ts = datetime.now().strftime("%m-%d %H:%M")
            md = f"# 卡死分析 ({ts})\n\n"
            md += f"**原因**: {reason}\n\n"
            md += f"**截图**: {screenshot_path or 'N/A'}\n\n"
            md += f"**日志**: {log_path}\n\n"
            md += f"**轮次**: 第{self.round_count}轮, **连续Unknown**: {self._consecutive_unknown}轮\n\n"
            md += f"**配置**: {profile} / {bd}\n\n"
            branch_text = self._handler_ctx.last_branch_text
            if branch_text:
                md += f"**最后分支文字**: `{branch_text}`\n\n"
            md += f"## 最近回合\n\n"
            md += "```json\n" + json.dumps(turns_summary, ensure_ascii=False, indent=2) + "\n```\n"
            latest.write_text(md, encoding="utf-8")

            # 创建 trigger.txt
            trigger = ai_dir / "trigger.txt"
            trigger.write_text(f"stuck|{reason}|{screenshot_path or 'N/A'}|{log_path}", encoding="utf-8")

            print(f"[流水线] 分析上下文已写入: {latest}")
            print(f"[流水线] 触发器已创建: {trigger}")
        except Exception as e:
            print(f"[流水线] 写入失败: {e}")

    def _increment_turn(self):
        """训练或休息完成后递增回合计数"""
        if self._handler_ctx.detected_turn is not None:
            self._handler_ctx.detected_turn += 1

    def _handle_training_screen(self, screenshot) -> bool:
        """训练决策 + 执行 (移植自 SleepRunner TrainingSelectHandler)"""
        t0 = time.perf_counter()
        self._handler_ctx.round_count = self.round_count
        if self.logger is not None:
            self.logger.set_screenshot(screenshot)

        # 当前客户端没有底部横排支援卡。旧颜色检测会把立绘误判为卡片，
        # 因而不能在训练决策前执行该点击分支。
        self._handler_ctx.try_detect_turn()

        # 读取游戏状态
        state = self._read_game_state(screenshot)

        # 开始回合日志
        self.logger.start_turn(
            round_num=self.round_count,
            turn_num=self._handler_ctx.detected_turn,
            stamina=state.stamina,
            mood=state.mood,
            money=state.money,
        )

        if self.engine.is_rule_engine_mode:
            action, target = self._decide_rule_engine(state)
        else:
            failure_rate = max(0, 100 - state.stamina)
            options = self._read_training_options_legacy(screenshot, failure_rate)
            if not options:
                print("[警告] 未识别到训练选项")
                self.logger.end_turn()
                return False
            action, target = self.engine.make_decision(options, state, debug=True)

        if action == "train":
            result = self._execute_train(target)
        else:
            result = self._execute_rest_flow(target)

        self.logger.end_turn()
        elapsed = time.perf_counter() - t0
        if elapsed > 5.0:
            print(f"[计时] 训练处理: {elapsed:.2f}s")
        return result

    def _read_game_state(self, screenshot) -> GameState:
        raw = self.ui.read_game_state(screenshot, self._last_stamina, self._last_mood)
        stamina = raw["stamina"]
        mood = raw["mood"]
        money = raw["money"]

        self._last_stamina = stamina
        self._last_mood = mood
        self._handler_ctx._last_stamina = stamina
        self._handler_ctx._last_mood = mood
        if money and money > 0:
            self._last_money = money
            self._handler_ctx._last_money = money

        print(f"[状态] 耐力={stamina:.0f}% 失败率={100 - stamina:.0f}% 心情={mood} ${self._last_money}")

        return GameState(
            stamina=stamina,
            mood=mood,
            money=self._last_money,
            round_num=self.round_count,
        )

    # ===================== 规则引擎懒加载扫描 =====================

    def _decide_rule_engine(self, state: GameState) -> tuple:
        profile = self.engine.rule_profile
        profile_name = self.engine.rule_profile_name
        scan_state = TrainingMetricScanState()
        scan_state.stamina_bar_pct = state.stamina

        # 构建当前属性快照 (用于评分降权)
        current_stats = {
            "力量": scan_state.strength_stat,
            "体力": scan_state.stamina_stat,
            "韧性": None, "专注": None, "保护": None,
        }
        target_stats = getattr(self, '_target_stats', {})
        ctx_kwargs = dict(target_stats=target_stats, current_stats=current_stats)

        self._try_read_stats(scan_state)

        # 优势训练标记会影响所有训练行的权重。若画面上存在大拇指，
        # 先把五行指标补齐，避免 speed.json 的旧图标阈值规则在尚未读取
        # 优势行时提前返回。
        advantage_rows = [i for i, marked in enumerate(scan_state.is_advantage) if marked]
        if advantage_rows:
            print(f"[优势] 检测到训练行: {[i + 1 for i in advantage_rows]}，补齐指标后参与评分")
            for row_index in range(5):
                if (not scan_state.has_icon_count(row_index) or
                        not scan_state.has_fail_rate(row_index) or
                        not scan_state.has_gain(row_index)):
                    self._read_row_metrics(scan_state, row_index)

        # 前20回合: 全扫描找图标最多的行, 优先训练以增加支援卡羁绊
        turn = self._handler_ctx.detected_turn
        icon_priority_limit = getattr(self, '_icon_priority_limit', 20)
        if turn is not None and turn <= icon_priority_limit:
            icon_result = self._early_icon_priority(scan_state, turn, state.mood)
            if icon_result is not None:
                return self._apply_camp_strategy(icon_result, scan_state, turn, state)

        for iteration in range(10):
            context = scan_state.to_decision_context(
                profile, profile_name, self.engine.adventure_gain_threshold, **ctx_kwargs)
            probe = TrainingRuleEngine.probe(context, profile)

            if probe.decision is not None:
                result = probe.decision
                print(f"[规则] {result.action.value} row={result.target_row_index} (rule={result.matched_rule_id})")
                # 更新 current_stats (scan_state 可能已更新)
                ctx_kwargs["current_stats"] = {
                    "力量": scan_state.strength_stat,
                    "体力": scan_state.stamina_stat,
                    "韧性": None, "专注": None, "保护": None,
                }
                context = scan_state.to_decision_context(
                    profile, profile_name, self.engine.adventure_gain_threshold, **ctx_kwargs)
                self.logger.log_decision_rule_engine(result, context, scan_state)
                if result.action == TrainingDecisionAction.REST:
                    return self._apply_camp_strategy(("rest", result), scan_state, turn, state)
                return self._apply_camp_strategy(("train", result), scan_state, turn, state)

            if probe.missing_field is None:
                break

            if iteration == 0:
                print(f"[规则] 扫描... (需{probe.missing_field.name})")

            made_progress = self._ensure_metric(scan_state, probe.missing_field)
            if not made_progress:
                break

        print("[规则] 已有数据评估")
        ctx_kwargs["current_stats"] = {
            "力量": scan_state.strength_stat,
            "体力": scan_state.stamina_stat,
            "韧性": None, "专注": None, "保护": None,
        }
        context = scan_state.to_decision_context(
            profile, profile_name, self.engine.adventure_gain_threshold, **ctx_kwargs)
        result = TrainingRuleEngine.evaluate(context, profile)
        print(f"[规则] {result.action.value} row={result.target_row_index}")

        self.logger.log_decision_rule_engine(result, context, scan_state)

        if result.action == TrainingDecisionAction.REST:
            return self._apply_camp_strategy(("rest", result), scan_state, turn, state)
        return self._apply_camp_strategy(("train", result), scan_state, turn, state)

    def _apply_camp_strategy(self, decision, scan_state, turn, state):
        """合宿策略: 25-29回合预合宿蓄力, 30+合宿激进训练(条件性).

        预合宿: 训练增益低时改为休息蓄力, 确保进入合宿时体力接近100%.
        合宿中: 仅当加成属性增益 > camp_high_gain_threshold 时才激进训练
                (失败率阈值从20→50), 否则20%失败率就休息.
        """
        action_type, result = decision

        if turn is None or self._pre_camp_min_gain <= 0:
            return decision

        stamina = scan_state.stamina_bar_pct
        bonus_row = self._camp_bonus_row  # 0=力量(弗洛拉), 1=体力

        # ---- 预合宿阶段 (25-29回合) ----
        if self._pre_camp_start_turn <= turn < self._camp_start_turn:
            if action_type == "train" and result.target_row_index is not None:
                gain = 0
                if scan_state.has_gain(result.target_row_index):
                    gain = scan_state.gain_values[result.target_row_index]
                if gain < self._pre_camp_min_gain and stamina < 95:
                    row_name = _TRAINING_ROWS[result.target_row_index][0]
                    print(f"[合宿] T{turn} {row_name}增益{gain}<{self._pre_camp_min_gain} + "
                          f"体力{stamina:.0f}% → 改为休息蓄力")
                    return ("rest", TrainingDecisionResult(
                        action=TrainingDecisionAction.REST,
                        target_row_index=None,
                        matched_rule_id="pre_camp_rest",
                    ))

        # ---- 合宿阶段 (30-37回合) ----
        if self._camp_start_turn <= turn < self._camp_end_turn:
            if action_type == "rest":
                # 确保加成属性行的增益已读取 (规则引擎可能只读了失败率就决定休息)
                if not scan_state.has_gain(bonus_row) or not scan_state.has_fail_rate(bonus_row):
                    self._read_row_metrics(scan_state, bonus_row)
                fail_rate = scan_state.fail_rates[bonus_row] if scan_state.has_fail_rate(bonus_row) else 0
                bonus_gain = scan_state.gain_values[bonus_row] if scan_state.has_gain(bonus_row) else 0
                # 仅当加成属性增益 > 阈值时才激进训练, 否则20%就休息
                if (bonus_gain > self._camp_high_gain_threshold and
                        fail_rate < self._camp_rest_fail_threshold and
                        stamina > 25):
                    bonus_name = _TRAINING_ROWS[bonus_row][0]
                    print(f"[合宿] T{turn} {bonus_name}增益{bonus_gain}>{self._camp_high_gain_threshold} + "
                          f"失败率{fail_rate}%<{self._camp_rest_fail_threshold}% → 坚持训练")
                    return ("train", TrainingDecisionResult(
                        action=TrainingDecisionAction.TRAIN_BEST_GAIN,
                        target_row_index=None,
                        matched_rule_id="camp_force_train",
                    ))
                elif bonus_gain <= self._camp_high_gain_threshold:
                    bonus_name = _TRAINING_ROWS[bonus_row][0]
                    print(f"[合宿] T{turn} {bonus_name}增益{bonus_gain}≤{self._camp_high_gain_threshold} → "
                          f"按正常阈值休息")

        return decision

    def _try_read_stats(self, scan_state: TrainingMetricScanState):
        try:
            screenshot = self.capture.capture_game()
            if screenshot is None:
                return
            all_stats = self._ocr_all_stats(screenshot)
            strength = all_stats.get("力量")
            stamina = all_stats.get("体力")
            scan_state.update_stats(strength, stamina)

            # 闪光训练检测 (HSV 黄色检测, 5行按钮)
            flash = self._detect_flash_training(screenshot)
            for i, f in enumerate(flash):
                scan_state.is_flash[i] = f
                scan_state.known_flash_mask |= 1 << i
            flash_rows = [i + 1 for i, f in enumerate(flash) if f]
            if flash_rows:
                print(f"[闪光] {flash_rows}")

            # 蓝色大拇指标记表示游戏当前推荐的优势训练。
            advantage = detect_advantage_training(screenshot)
            for i, marked in enumerate(advantage):
                scan_state.is_advantage[i] = marked
                scan_state.known_advantage_mask |= 1 << i
            advantage_rows = [i + 1 for i, marked in enumerate(advantage) if marked]
            if advantage_rows:
                print(f"[优势] 大拇指标记: {advantage_rows}")

            parts = []
            for name in _STAT_NAMES:
                val = all_stats.get(name)
                parts.append(f"{name}={val}" if val is not None else f"{name}=N/A")
            print(f"[属性] {' '.join(parts)}")
        except Exception:
            pass

    def _early_icon_priority(self, scan_state: TrainingMetricScanState, turn: int, mood: str):
        """前N回合: 全扫描5行找图标最多的训练, 优先训练以增加支援卡羁绊.
        图标最多的行 → 支援卡集中, 羁绊涨得快 → 更早触发闪光训练.
        图标都不多且心情非Best → 优先休息提高心情再训练."""
        if not self.engine.use_icon_counting:
            return None

        # 强制全扫描 (无论 scan_mode)
        self._focus_read_metrics(scan_state)

        # 有大拇指标记时，前期羁绊策略也必须把优势因素纳入评分；
        # 没有标记时保留原来的“图标最多”策略。
        if any(scan_state.is_advantage):
            score_context = scan_state.to_decision_context(
                self.engine.rule_profile, self.engine.rule_profile_name,
                self.engine.adventure_gain_threshold,
                target_stats=getattr(self, '_target_stats', {}),
                current_stats={
                    "力量": scan_state.strength_stat,
                    "体力": scan_state.stamina_stat,
                })
            best_row = TrainingRuleEngine._score_training_rows(score_context)
            print(f"[羁绊] 优势标记参与前期评分 → {_TRAINING_ROWS[best_row][0]}")
        else:
            best_row = max(range(5), key=lambda i: scan_state.icon_counts[i])
        best_icons = scan_state.icon_counts[best_row]

        if best_icons < 2:
            if mood != "Best":
                print(f"[羁绊] T{turn} 图标不足(最大{best_icons}) + 心情{mood} → 优先休息($30)")
                return ("rest", TrainingDecisionResult(
                    action=TrainingDecisionAction.REST,
                    target_row_index=None,
                    matched_rule_id="early_icon_priority_rest",
                ))
            print(f"[羁绊] T{turn} 图标不足(最大{best_icons}), 回退规则引擎")
            return None

        row_name = _TRAINING_ROWS[best_row][0]
        print(f"[羁绊] T{turn} {row_name}: {best_icons}图标 → 优先训练")
        self.logger.log_decision_rule_engine(
            TrainingDecisionResult(
                action=TrainingDecisionAction.TRAIN,
                target_row_index=best_row,
                matched_rule_id="early_icon_priority",
            ),
            scan_state.to_decision_context(
                self.engine.rule_profile, self.engine.rule_profile_name,
                self.engine.adventure_gain_threshold,
                target_stats=getattr(self, '_target_stats', {}),
                current_stats={
                    "力量": scan_state.strength_stat,
                    "体力": scan_state.stamina_stat,
                }),
            scan_state,
        )
        return ("train", TrainingDecisionResult(
            action=TrainingDecisionAction.TRAIN,
            target_row_index=best_row,
            matched_rule_id="early_icon_priority",
        ))

    def _get_primary_row(self) -> int:
        """根据构建方向/优先属性返回主训练行索引 (0-4).
        快速模式下只点击此行读取真实失败率, 其他行用耐力推导."""
        try:
            if self.engine.is_rule_engine_mode:
                direction = self.engine.rule_profile.legacy_strategy.build_direction.value
                mapping = {"attack": 0, "survival": 1, "speed": 0}
                return mapping.get(direction, 0)
        except Exception:
            pass
        try:
            return self.engine.priority - 1
        except Exception:
            pass
        return 0

    def _stamina_fail_rate(self, row_index: int) -> int:
        """耐力推导失败率. 专注(3)/保护(4)额外+10%安全余量."""
        stamina_pct = self._last_stamina if self._last_stamina is not None else 100
        base = max(0, 100 - int(stamina_pct))
        if row_index >= 3:  # 专注/保护: 失败率略高于其他训练
            return min(100, base + 10)
        return base

    def _fast_read_metrics(self, scan_state: TrainingMetricScanState):
        """快速模式: 只点击主属性行, 其他4行耐力推导."""
        primary_row = self._get_primary_row()
        self._read_row_metrics(scan_state, primary_row)

        for i in range(5):
            if i == primary_row:
                continue
            if not scan_state.has_fail_rate(i) or not scan_state.has_icon_count(i):
                scan_state.set_row_metrics(i, 3, self._stamina_fail_rate(i))

        name = _TRAINING_ROWS[primary_row][0]
        fr = scan_state.fail_rates[primary_row]
        ic = scan_state.icon_counts[primary_row]
        others = ",".join(f"{_TRAINING_ROWS[i][0]}={scan_state.fail_rates[i]}%"
                         for i in range(5) if i != primary_row)
        print(f"[快速] 主={name} {fr}%/{ic}图标, 推导: {others}")

    def _read_all_rows_metrics(self, scan_state: TrainingMetricScanState):
        """普通模式: 点力量/体力/韧性(行0-2), 专注/保护(行3-4)耐力推导(+10%)."""
        for i in range(3):
            if not scan_state.has_fail_rate(i) or not scan_state.has_icon_count(i):
                self._read_row_metrics(scan_state, i)

        for i in range(3, 5):
            if not scan_state.has_fail_rate(i) or not scan_state.has_icon_count(i):
                scan_state.set_row_metrics(i, 3, self._stamina_fail_rate(i))

    def _focus_read_metrics(self, scan_state: TrainingMetricScanState):
        """专注模式: 逐行点击全部5行, 读取真实失败率."""
        for i in range(5):
            if not scan_state.has_fail_rate(i) or not scan_state.has_icon_count(i):
                self._read_row_metrics(scan_state, i)

    def _ensure_metric(self, scan_state: TrainingMetricScanState,
                       field: TrainingRuleField) -> bool:
        row_index = TrainingRuleLoader.field_to_row_index(field)

        if field in (TrainingRuleField.STRENGTH_STAT, TrainingRuleField.STAMINA_STAT):
            if (field == TrainingRuleField.STRENGTH_STAT and scan_state.strength_stat is None) or \
               (field == TrainingRuleField.STAMINA_STAT and scan_state.stamina_stat is None):
                return self._try_read_stats(scan_state)
            return True

        if field == TrainingRuleField.ANY_FAIL_RATE:
            if self._scan_mode == "fast":
                self._fast_read_metrics(scan_state)
            elif self._scan_mode == "focus":
                self._focus_read_metrics(scan_state)
            else:
                self._read_all_rows_metrics(scan_state)
            return True

        if field == TrainingRuleField.ANY_GAIN:
            if self._scan_mode == "fast":
                self._fast_read_metrics(scan_state)
            elif self._scan_mode == "focus":
                self._focus_read_metrics(scan_state)
            else:
                self._read_all_rows_metrics(scan_state)
            for i in range(5):
                if not scan_state.has_gain(i):
                    self._read_row_metrics(scan_state, i)
            return True

        if row_index is not None:
            self._read_row_metrics(scan_state, row_index)
            return True

        return False

    def _detect_flash_training(self, screenshot) -> list[bool]:
        """检测5行训练按钮是否闪光.

        闪光训练按钮呈暖白色/浅琥珀色, 比普通按钮饱和度高。
        双重检测: (1) "浅黄" HSV 像素占比 + (2) 相对饱和度差异。
        两方法独立运行, 结果取并集。
        """
        h, w = screenshot.shape[:2]
        rois_hsv = []
        s_means = []

        for i, (name, rx, ry) in enumerate(_TRAINING_ROWS):
            bx = max(0, int(w * (rx - 0.05)))
            by = max(0, int(h * (ry - 0.025)))
            bw = max(1, int(w * 0.10))
            bh = max(1, int(h * 0.05))
            roi = screenshot[by:by + bh, bx:bx + bw]
            if roi.size == 0:
                rois_hsv.append(None)
                s_means.append(0.0)
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            rois_hsv.append(hsv)
            s_means.append(float(np.mean(hsv[:, :, 1])))

        result = [False] * 5

        # 方法1: "浅黄" 像素占比 (H:12-45, S>=35, V>=130)
        lower = np.array([12, 35, 130])
        upper = np.array([45, 255, 255])
        for i, hsv in enumerate(rois_hsv):
            if hsv is None:
                continue
            mask = cv2.inRange(hsv, lower, upper)
            ratio = np.count_nonzero(mask) / mask.size
            if ratio > 0.03:
                result[i] = True

        # 方法2: 饱和度相对差异 (各行独立判断, 闪光行 S 显著高于均值)
        # 始终运行, 补充方法1可能遗漏的行
        if s_means:
            mean_s = sum(s_means) / len(s_means)
            for i in range(5):
                if result[i]:
                    continue
                if s_means[i] > mean_s + 8 and s_means[i] > 50:
                    result[i] = True

        return result

    def _read_row_metrics(self, scan_state: TrainingMetricScanState, row_index: int):
        if scan_state.has_icon_count(row_index) and scan_state.has_fail_rate(row_index):
            return

        name, rx, ry = _TRAINING_ROWS[row_index]

        # 中途接管时任意行都可能被选中，力量行也必须显式选择。
        self.controller.click_at_percent(rx, ry)
        self.controller.wait(_CLICK_ROW_WAIT)

        screenshot = self.capture.capture_game()
        if screenshot is None:
            scan_state.set_row_metrics(row_index, 0, UNKNOWN_FAIL_RATE_FALLBACK)
            return

        h, w = screenshot.shape[:2]

        icon_count = 3
        if self.engine.use_icon_counting:
            try:
                icon_count = count_circular_icons(screenshot, name)
            except Exception:
                pass

        fail_rate = UNKNOWN_FAIL_RATE_FALLBACK
        stamina_pct = self._last_stamina if self._last_stamina is not None else 100
        stamina_fallback = max(0, 100 - int(stamina_pct))

        if self.engine.use_per_row_fail_rate:
            regions = self.engine.fail_rate_regions
            if regions and len(regions) == 5:
                rrx, rry, rrw, rrh = regions[row_index]
            else:
                rrx, rry, rrw, rrh = 0.74, 0.20, 0.21, 0.40

            ocr_texts = []
            try:
                # 方案1: DirectML OCR 直接读取 ROI
                px = max(0, int(w * rrx))
                py = max(0, int(h * rry))
                pw = max(1, min(int(w * rrw), w - px))
                ph = max(1, min(int(h * rrh), h - py))
                if pw > 1 and ph > 1:
                    fail_roi = screenshot[py:py + ph, px:px + pw]
                    texts = self.ui.ocr.recognize(fail_roi)
                    ocr_texts = [t[0] for t in texts]
                    fail_rate = parse_fail_rate_from_ocr_texts(ocr_texts)
            except Exception:
                pass

            # 方案2: DirectML 失败 → 红色增强预处理后重试
            if fail_rate >= UNKNOWN_FAIL_RATE_FALLBACK:
                try:
                    px = max(0, int(w * rrx))
                    py = max(0, int(h * rry))
                    pw = max(1, min(int(w * rrw), w - px))
                    ph = max(1, min(int(h * rrh), h - py))
                    if pw > 1 and ph > 1:
                        fail_roi = screenshot[py:py + ph, px:px + pw]
                        # 提取红色通道增强 (失败率文字为红色)
                        if fail_roi.shape[2] >= 3:
                            b, g, r = cv2.split(fail_roi[:, :, :3])
                            enhanced = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
                            _, enhanced = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
                            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                            texts = self.ui.ocr.recognize(enhanced_bgr)
                            ocr_texts2 = [t[0] for t in texts]
                            fail_rate2 = parse_fail_rate_from_ocr_texts(ocr_texts2)
                            if fail_rate2 < UNKNOWN_FAIL_RATE_FALLBACK:
                                fail_rate = fail_rate2
                except Exception:
                    pass

            # 方案3: 红色增强也失败 → WinOCR 回退 (对红色文字识别更好)
            if fail_rate >= UNKNOWN_FAIL_RATE_FALLBACK:
                try:
                    win_text = self.ui.ocr.recognize_region(screenshot, rrx, rry, rrw, rrh)
                    if win_text:
                        fail_rate = parse_fail_rate_from_ocr_texts([win_text])
                except Exception:
                    pass

            # 所有方案失败 → 耐力推导回退
            if fail_rate >= UNKNOWN_FAIL_RATE_FALLBACK:
                fail_rate = stamina_fallback
        else:
            fail_rate = stamina_fallback

        # +N 徽标位于属性面板右侧，属性名、当前值和徽标是独立 OCR 框。
        # 按属性所在行裁剪，避免把副属性/潜质点数的增益当成主属性增益。
        gain = 0
        try:
            gain_text = self.ui.ocr.recognize_region(
                screenshot, 0.164, 0.365 + row_index * 0.0694, 0.05, 0.043)
            m = re.search(r'[+＋]\s*(\d{1,3})(?!\d)', gain_text)
            if m:
                gain = int(m.group(1))
        except Exception:
            pass

        scan_state.set_row_metrics(row_index, icon_count, fail_rate, gain)
        gain_str = f" +{gain}" if gain > 0 else ""
        print(f"  [{row_index + 1}] {name}: {icon_count}图标 {fail_rate}%失败{gain_str}")

    def _ocr_stat_value(self, screenshot, stat_name: str) -> Optional[int]:
        """从校准的属性面板区域读取单个属性值 (格式: 属性名 数字/1250)"""
        h, w = screenshot.shape[:2]
        rx, ry, rrw, rrh = _STAT_PANEL
        px = int(w * rx)
        py = int(h * ry)
        pw = int(w * rrw)
        ph = int(h * rrh)
        if pw < 10 or ph < 10:
            return None
        roi = screenshot[py:py + ph, px:px + pw]
        items = self.ui.ocr.recognize(roi)
        all_text = " ".join([t[0] for t in items])

        # 精确匹配: 属性名 数字/1250
        m = re.search(rf'{stat_name}\s*(\d{{1,4}})\s*/\s*1250', all_text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 1300:
                return val

        # 回退1: 属性名后跟数字/被截断 (如 OCR 把 511/1250 读成 51/1)
        m = re.search(rf'{stat_name}[^\d]*?(\d{{2,3}})\s*/\s*\d{{1,2}}(?:\s|$|[^\d])', all_text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 1300:
                return val

        # 回退2: 属性名后任意至少3位数字
        m = re.search(rf'{stat_name}[^\d]*?(\d{{3,4}})', all_text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 1300:
                return val

        # 回退3: 2位数字兜底 (OCR 严重截断时)
        m = re.search(rf'{stat_name}[^\d]*?(\d{{2}})', all_text)
        if m:
            val = int(m.group(1))
            if 10 <= val <= 99:
                return val
        return None

    def _ocr_all_stats(self, screenshot) -> dict[str, Optional[int]]:
        """从属性面板一次性读取全部5个属性值"""
        h, w = screenshot.shape[:2]
        rx, ry, rrw, rrh = _STAT_PANEL
        px = int(w * rx)
        py = int(h * ry)
        pw = int(w * rrw)
        ph = int(h * rrh)
        if pw < 10 or ph < 10:
            return {name: None for name in _STAT_NAMES}

        roi = screenshot[py:py + ph, px:px + pw]
        # 优先用 recognize_region (WinOCR) 获取干净文本, 失败则尝试 recognize
        raw_text = self.ui.ocr.recognize_region(screenshot, rx, ry, rrw, rrh)
        items = self.ui.ocr.recognize(roi)
        all_text = " ".join([t[0] for t in items])
        if len(raw_text) > len(all_text):
            all_text = raw_text

        results = {}
        for stat_name in _STAT_NAMES:
            # 精确匹配: 属性名 数字/1250
            m = re.search(rf'{stat_name}\s*(\d{{1,4}})\s*/\s*1250', all_text)
            if m:
                val = int(m.group(1))
                results[stat_name] = val if 0 <= val <= 1300 else None
            else:
                # 回退1: 属性名后跟数字/被截断 (如 OCR 把 511/1250 读成 51/1)
                m2 = re.search(rf'{stat_name}[^\d]*?(\d{{2,3}})\s*/\s*\d{{1,2}}(?:\s|$|[^\d])', all_text)
                if m2:
                    val = int(m2.group(1))
                    results[stat_name] = val if 0 <= val <= 1300 else None
                else:
                    # 回退2: 属性名后任意至少3位数字
                    m3 = re.search(rf'{stat_name}[^\d]*?(\d{{3,4}})', all_text)
                    if m3:
                        val = int(m3.group(1))
                        results[stat_name] = val if 0 <= val <= 1300 else None
                    else:
                        # 回退3: 2位数字兜底 (OCR 严重截断时)
                        m4 = re.search(rf'{stat_name}[^\d]*?(\d{{2}})', all_text)
                        if m4:
                            val = int(m4.group(1))
                            results[stat_name] = val if 10 <= val <= 99 else None
                        else:
                            results[stat_name] = None

        missing = [n for n, v in results.items() if v is None]
        if missing:
            print(f"[属性OCR] 未识别: {missing}, 原始: '{all_text[:150]}'")

        return results

    # ===================== 训练执行 =====================

    def _execute_train(self, target) -> bool:
        """执行训练 + 训练后点击推进"""
        if target is None:
            print("[警告] 无训练目标")
            return False

        if isinstance(target, TrainingDecisionResult):
            row_index = target.target_row_index
            if row_index is None or not (0 <= row_index < 5):
                print(f"[警告] 无效行索引: {row_index}")
                return False
        else:
            row_index = target.attr_id - 1
            if not (0 <= row_index < 5):
                print(f"[警告] 无效属性ID: {target.attr_id}")
                return False

        name, rx, ry = _TRAINING_ROWS[row_index]
        print(f"[训练] {name} (行{row_index + 1})")

        self.controller.click_at_percent(rx, ry)
        self.controller.wait(_CLICK_ROW_WAIT)

        self.controller.click_at_percent(_TRAIN_BTN_X, _TRAIN_BTN_Y)
        self.controller.wait(_CLICK_TRAIN_WAIT)

        if isinstance(target, TrainingDecisionResult):
            self.logger.log_train_exec(
                row_index, _TRAINING_ROWS[row_index][0],
                metrics={
                    "rule_id": target.matched_rule_id,
                    "builtin_default": target.used_builtin_default,
                })
        else:
            self.logger.log_train_exec(
                row_index, target.attr_name,
                metrics={
                    "gain": target.gain,
                    "failure_rate": target.failure_rate,
                })

        # 训练后点击推进 (穿透加载动画/结算画面)
        print(f"  推进加载 ({_POST_TRAIN_CLICKS}次)...")
        self.controller.click_center_multi(
            count=_POST_TRAIN_CLICKS,
            interval=_POST_TRAIN_CLICK_INTERVAL,
            click_delay=0.01,
        )
        time.sleep(_POST_TRAIN_WAIT)
        # 缓存截图跳过下一轮 dispatch capture
        ss = self.capture.capture_game()
        if ss is not None:
            self._handler_ctx.pending_screenshot = ss

        self._increment_turn()
        return True

    # ===================== 休息流程 =====================

    def _execute_rest_flow(self, target) -> bool:
        """从训练界面切换到休息界面. 合宿期间内联处理休息选择.
        超时时触发重检测而非死循环."""
        turn = self._handler_ctx.detected_turn
        in_camp = (turn is not None and self._camp_start_turn <= turn < self._camp_end_turn)

        self.controller.send_escape()
        self.controller.wait(0.15)

        screenshot = self.capture.capture_game()
        if screenshot is not None:
            menu_text = self.ui.ocr.recognize_region(screenshot, 0.74, 0.34, 0.22, 0.34)
            print(f"[休息流程] ESC → 菜单 (rest={'休息' in menu_text})")
        # 点击休息按钮
        self.controller.click_at_percent(_MAIN_MENU_DIAMOND_X, _MAIN_MENU_REST_Y)
        # 等待休息界面加载
        from .handlers import _wait_for_screen
        if _wait_for_screen(self._handler_ctx, "rest", timeout=3.0):
            if in_camp:
                # 合宿休息策略: 默认免费, 心情非Best且钱足够预留80后才付费$30
                self._execute_camp_rest()
                return True
            else:
                self._handler_ctx.pending_screenshot = self.capture.capture_game()
                self.logger.log_rest(
                    option_id=0, option_name="休息流程",
                    cost=0, stamina_before=self._last_stamina or 0,
                    reason="规则引擎判定休息")
                self._increment_turn()
                return True
        else:
            # 超时: 可能被事件/弹窗挡住, ESC 回退让 dispatch 重新检测
            print("[休息流程] 超时, ESC回退重新检测...")
            self.controller.send_escape()
            self.controller.wait(0.15)
            self._handler_ctx.pending_screenshot = None
            return True

    def _execute_camp_rest(self):
        """合宿期间的休息选择: 默认免费, 心情非Best且(金钱-80)>30才付费$30."""
        screenshot = self.capture.capture_game()
        money = self._last_money or 0
        if screenshot is not None:
            raw = self.ui.read_game_state(screenshot)
            m = raw.get("money", 0)
            if m > 0:
                money = m
                self._last_money = money
        mood = self._last_mood or "Normal"
        stamina = self._last_stamina or 50

        # 合宿休息规则 (优先级从高到低):
        # 1. Best + (金钱-80) >= 80: 冥想室($60)
        # 2. 非Best + (金钱-80) > 30: 住处($30)
        # 3. 其他: 免费休息
        reserve = self._camp_min_coins
        if mood == "Best" and (money - reserve) >= 80:
            option_idx = 2  # 冥想室 $60
            reason = f"合宿+Best+${money}(预留{reserve}后${money-reserve})→冥想室$60"
        elif mood != "Best" and (money - reserve) > 30:
            option_idx = 1  # 住处 $30
            reason = f"合宿+心情{mood}+${money}(预留{reserve})→住处$30"
        else:
            option_idx = 0  # 露宿 免费
            reason = f"合宿+心情{mood}+${money}(预留{reserve})→免费"

        names = ["露宿(免费)", "住处($30)", "冥想室($60)"]
        print(f"[休息] 合宿 | ${money} | {stamina:.0f}% | {mood} → {names[option_idx]}")

        # 点击选项
        REST_CLICK = [(0.83, 0.37), (0.84, 0.47), (0.85, 0.57)]
        if option_idx > 0:
            rx, ry = REST_CLICK[option_idx]
            self.controller.click_at_percent(rx, ry)
            time.sleep(0.15)
        # 点击确认
        self.controller.click_at_percent(0.89, 0.925)
        time.sleep(0.3)

        # 等待休息完成 (离开休息界面)
        from .recognition import ScreenDetector
        for _ in range(10):
            time.sleep(0.15)
            new_ss = self.capture.capture_game()
            if new_ss is not None:
                detector = ScreenDetector(self.ui.ocr)
                if not detector.is_rest_screen(new_ss):
                    self._handler_ctx.pending_screenshot = new_ss
                    break

        self.logger.log_rest(
            option_id=option_idx + 1, option_name=names[option_idx],
            cost=[0, 30, 60][option_idx], stamina_before=stamina,
            reason=reason)
        self._increment_turn()

    # ===================== 阿尔克那支援卡 (第一年) =====================

    def _try_click_arcanum(self, screenshot) -> bool:
        """
        第一年优先点击与训练属性匹配的支援卡以提升羁绊。

        羁绊达到黄色后触发闪光训练，属性增益大幅提升。

        Returns:
            True 如果点击了支援卡 (本轮跳过训练)
        """
        arcanum_cfg = self.config.get("arcanum", {})
        if not arcanum_cfg.get("enabled", False):
            return False

        current_turn = self._handler_ctx.detected_turn if self._handler_ctx.detected_turn is not None else self.round_count
        if current_turn > self._turn_limit:
            return False

        # 获取当前优先属性
        priority_attr = self._get_priority_attr_id()

        # 扫描支援卡
        card_index = self._arcanum.get_target_card(
            screenshot, self.ui.ocr, priority_attr)

        if card_index is None:
            if current_turn == self._turn_limit:
                print(f"[支援卡] T{self._turn_limit}, 无匹配/羁绊已满, 正常训练")
            return False

        click_x, click_y = self._arcanum.get_card_click_pos(card_index)

        attr_name = self.config.get("attributes", {}).get(priority_attr, str(priority_attr))
        print(f"[支援卡] T{current_turn}/{self._turn_limit} {attr_name} (卡{card_index})")

        self.controller.click_at_percent(click_x, click_y)
        post_wait = arcanum_cfg.get("post_click_wait", 1.5)
        self.controller.wait(post_wait)

        # 点完后需要推进画面
        self.controller.click_center_multi(count=10, interval=0.01, click_delay=0.01)
        time.sleep(0.5)

        self.logger.log_arcanum(
            card_index=card_index,
            priority_attr=attr_name,
            current_turn=current_turn,
            turn_limit=self._turn_limit,
        )

        return True

    def _get_priority_attr_id(self) -> int:
        """获取当前优先训练属性ID"""
        if self.engine.is_rule_engine_mode:
            strategy = self.engine.rule_profile.legacy_strategy
            direction = strategy.build_direction.value
            if direction == "attack":
                return 1  # 力量
            else:
                return 2  # 生命/体力
        else:
            return self.engine.priority

    def _get_priority_attr_name(self) -> str:
        """获取当前优先训练属性的名称字符串"""
        attr_id = self._get_priority_attr_id()
        return self.engine.attributes.get(attr_id, "力量")

    # ===================== 传统模式 (向后兼容) =====================

    def _read_training_options_legacy(self, screenshot, failure_rate) -> list:
        options = []
        h, w = screenshot.shape[:2]
        for i, (name, _, _) in enumerate(_TRAINING_ROWS):
            rx, ry, rw, rh = 0.50, 0.26 + i * 0.10, 0.45, 0.08
            px = max(0, int(w * rx))
            py = max(0, int(h * ry))
            pw = min(int(w * rw), w - px)
            ph = min(int(h * rh), h - py)
            roi = screenshot[py:py + ph, px:px + pw]
            texts = self.ui.ocr.recognize(roi)
            all_text = " ".join([t[0] for t in texts])
            gain = 3
            nums = re.findall(r'\+?(\d+)', all_text)
            if nums:
                gain = max(int(n) for n in nums if 0 < int(n) <= 20)
            attr_name = self._match_attribute(all_text) or name
            options.append(TrainingOption(i + 1, attr_name, gain, failure_rate))
        return options

    def _match_attribute(self, text: str) -> str | None:
        for attr_id, attr_name in self.engine.attributes.items():
            if attr_name in text:
                return attr_name
        return None

    # ===================== 日志 =====================

    def _print_config(self):
        if self.engine.is_rule_engine_mode:
            s = self.engine.rule_profile.legacy_strategy
            print(f"\n当前配置:")
            print(f"  决策模式: 规则引擎")
            print(f"  规则配置: {self.engine.rule_profile_name}")
            print(f"  构建方向: {s.build_direction.value}")
            print(f"  失败率阈值: {s.fail_rate_threshold}%")
            print(f"  Rush阈值: {s.rush_threshold}")
            print(f"  规则数: {len(self.engine.rule_profile.rules)}")
            print(f"  图标计数: {'开' if self.engine.use_icon_counting else '关'}")
            print(f"  独立失败率: {'开' if self.engine.use_per_row_fail_rate else '关'}")
            print(f"  最大轮次: {self.max_rounds if self.max_rounds > 0 else '无限'}")

    def _save_log(self):
        self.logger.save(exit_reason=self._exit_reason)

    def calibrate(self):
        """校准模式: 截图 + 全屏 OCR 输出"""
        print("\n" + "=" * 60)
        print("  校准模式 - 验证 OCR + 屏幕检测")
        print("=" * 60)
        input("请先打开游戏并进入训练界面，然后按 Enter...")

        screenshot = self.capture.capture_game()
        self.capture.save_capture(screenshot, "_calibration_full")
        h, w = screenshot.shape[:2]
        print(f"\n截图尺寸: {w}x{h}")

        screen_type = self.ui.detect_screen(screenshot)
        print(f"屏幕检测: {screen_type}")

        branch = self.ui.detector.read_branch_text(screenshot)
        print(f"分支区域文字: '{branch[:120] if branch else '(空)'}'")

        state = self.ui.read_game_state(screenshot)
        print(f"识别状态: 耐力={state['stamina']:.0f}%, "
              f"心情={state['mood']}, 金钱=${state['money']}")

        print(f"\n全屏 OCR 结果 (前30条):")
        texts = self.ui.ocr.recognize_detailed(screenshot)
        for i, (text, conf, bbox) in enumerate(texts[:30]):
            if bbox is not None:
                pts = np.array(bbox)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                xp = cx / w
                yp = cy / h
                print(f"  [{conf:.2f}] ({xp:.3f},{yp:.3f}) {text}")
            else:
                print(f"  [{conf:.2f}] (--,--) {text}")

        print(f"\n截图已保存: templates/_calibration_full.png")

        # Arcanum 支援卡校准
        print(f"\n{'─'*40}")
        print(f"  支援卡 (阿尔克那) 检测")
        print(f"{'─'*40}")
        arcanum_result = calibrate_arcanum(
            screenshot, self.config, self.ui.ocr, save_dir="templates"
        )
        for card in arcanum_result.get("cards", []):
            i = card["index"]
            attr = card.get("attribute", "N/A")
            bond = "黄色✓" if card.get("bond_yellow") else "未达标"
            texts = card.get("ocr_texts", [])[:5]
            print(f"  卡片{i}: 属性={attr}, 羁绊={bond}")
            if texts:
                print(f"    OCR: {' | '.join(texts)}")
        print(f"  卡片截图已保存: templates/_arcanum_card*.png")
