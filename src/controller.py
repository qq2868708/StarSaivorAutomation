"""
输入控制器 - SendInput 硬件级鼠标模拟

使用 Win32 SendInput API 注入鼠标事件到系统输入队列。
所有游戏 (包括 Unity/DirectX) 都能正常响应。
点击前保存光标位置，点击后恢复，尽量减少对用户操作的影响。

移植自 SleepRunner MouseSimulator.cs + GameContext.cs
"""
import time
import ctypes
import ctypes.wintypes
import random

from .capture import user32, gdi32

kernel32 = ctypes.windll.kernel32

# SendInput 常量
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
SM_CXSCREEN = 0
SM_CYSCREEN = 1
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_1 = 0x31
VK_2 = 0x32
VK_3 = 0x33
VK_4 = 0x34


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("union", INPUT_UNION),
    ]


class InputTargetError(RuntimeError):
    """Stop the run when input can no longer reach the selected game window."""


class Controller:
    """输入控制器 - SendInput 硬件级点击 + 光标位置恢复"""

    def __init__(self, click_delay=0.5, action_interval=1.0, auto_refocus=False):
        self.click_delay = click_delay
        self.action_interval = action_interval
        self.auto_refocus = bool(auto_refocus)
        self._hwnd = None
        self._game_region = None
        self._dpi_scale = None
        self._rng = random.Random()

    def set_game_region(self, region):
        self._game_region = region

    def set_window_handle(self, hwnd):
        self._hwnd = hwnd

    @property
    def dpi_scale(self):
        if self._dpi_scale is None:
            logical_w = user32.GetSystemMetrics(SM_CXSCREEN)
            dc = user32.GetDC(0)
            physical_w = gdi32.GetDeviceCaps(dc, 118)
            user32.ReleaseDC(0, dc)
            self._dpi_scale = max(1.0, physical_w / max(logical_w, 1))
        return self._dpi_scale

    def focus_game_window(self):
        if not self._hwnd or not user32.IsWindow(self._hwnd):
            raise InputTargetError("游戏窗口不可用，请先打开 StarSavior")
        if user32.IsIconic(self._hwnd):
            raise InputTargetError("请恢复游戏窗口，不能最小化运行")
        user32.ShowWindow(self._hwnd, 5)  # SW_SHOW
        user32.BringWindowToTop(self._hwnd)
        user32.SetForegroundWindow(self._hwnd)
        # Windows 前台锁定会让终端/调试器保留前台资格；临时连接当前
        # 线程、目标窗口线程与前台线程后再重试，保证脚本重启时能回到游戏窗口。
        if user32.GetForegroundWindow() != self._hwnd:
            foreground = user32.GetForegroundWindow()
            foreground_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
            target_thread = user32.GetWindowThreadProcessId(self._hwnd, None)
            current_thread = kernel32.GetCurrentThreadId()
            attached_pairs = []
            for source, target in ((current_thread, foreground_thread),
                                   (target_thread, foreground_thread),
                                   (current_thread, target_thread)):
                if source and target and source != target and user32.AttachThreadInput(source, target, True):
                    attached_pairs.append((source, target))
            try:
                user32.BringWindowToTop(self._hwnd)
                user32.SetActiveWindow(self._hwnd)
                user32.SetForegroundWindow(self._hwnd)
            finally:
                for source, target in reversed(attached_pairs):
                    user32.AttachThreadInput(source, target, False)
        time.sleep(0.15)
        if user32.GetForegroundWindow() != self._hwnd:
            raise InputTargetError("请把 StarSavior 切到前台后再启动")

    def _check_input_target(self):
        if not self._hwnd or not user32.IsWindow(self._hwnd):
            raise InputTargetError("游戏窗口已关闭，停止发送输入")
        if user32.GetForegroundWindow() != self._hwnd:
            if self.auto_refocus:
                self.focus_game_window()
            if user32.GetForegroundWindow() == self._hwnd:
                return
            raise InputTargetError("游戏不在前台，停止发送输入；请切回游戏")

    # ===================== 核心点击 (SendInput) =====================

    def click_at_percent(self, x_pct, y_pct, fast=False):
        """
        在游戏窗口的百分比坐标处点击。

        使用 SendInput (硬件级输入), 所有游戏都响应。
        点击前保存光标位置, 点击后恢复。
        fast=True 时跳过随机延迟, 用于快速推进。
        """
        if self._hwnd is None:
            self._fallback_click(x_pct, y_pct)
            return
        self._check_input_target()

        # 1. 保存当前光标位置
        saved_pos = POINT()
        user32.GetCursorPos(ctypes.byref(saved_pos))

        try:
            # 2. 客户区百分比 → 屏幕绝对坐标
            screen_x, screen_y = self._percent_to_screen(x_pct, y_pct)

            # 3. 移动光标到目标
            self._send_move(screen_x, screen_y)
            if not fast:
                time.sleep(self._rng.uniform(0.03, 0.08))

            # 4. 点击
            self._send_mouse_event(MOUSEEVENTF_LEFTDOWN)
            if fast:
                time.sleep(0.01)
            else:
                time.sleep(self._rng.uniform(0.06, 0.14))
            self._send_mouse_event(MOUSEEVENTF_LEFTUP)
            time.sleep(0.01 if fast else self.click_delay)
        finally:
            # 每次点击都立即恢复，避免连续未知状态把真实鼠标留在游戏上。
            self._send_move(saved_pos.x, saved_pos.y)
            self._saved_cursor = (saved_pos.x, saved_pos.y)

    def restore_cursor(self):
        """恢复到操作前的光标位置"""
        if hasattr(self, '_saved_cursor') and self._saved_cursor:
            self._send_move(self._saved_cursor[0], self._saved_cursor[1])

    def click_center(self):
        self.click_at_percent(0.5, 0.75)

    def click_center_multi(self, count=3, interval=0.3, click_delay=None):
        """多次快速点击中央 (推进对话/加载), 支持临时覆盖 click_delay"""
        saved_delay = self.click_delay
        fast = click_delay is not None and click_delay <= 0.01
        if click_delay is not None:
            self.click_delay = click_delay
        try:
            for i in range(count):
                self.click_at_percent(0.5, 0.75, fast=fast)
                if i < count - 1:
                    time.sleep(interval)
        finally:
            self.click_delay = saved_delay

    def send_escape(self):
        """发送 ESC 键 (SendInput 键盘事件)"""
        self._send_key(VK_ESCAPE, key_up=False)
        time.sleep(0.05)
        self._send_key(VK_ESCAPE, key_up=True)
        time.sleep(0.1)

    def send_space(self):
        """发送空格键 (用于跳过战斗兜底)"""
        self._send_key(VK_SPACE, key_up=False)
        time.sleep(0.05)
        self._send_key(VK_SPACE, key_up=True)
        time.sleep(0.1)

    def send_number_key(self, n: int):
        """发送数字键 1-4 (用于事件选项热键选择)"""
        vk_map = {1: VK_1, 2: VK_2, 3: VK_3, 4: VK_4}
        vk = vk_map.get(n)
        if vk is None:
            return
        self._send_key(vk, key_up=False)
        time.sleep(0.05)
        self._send_key(vk, key_up=True)
        time.sleep(0.1)

    # ===================== 内部实现 =====================

    def _percent_to_screen(self, x_pct, y_pct):
        """客户区百分比 → 屏幕绝对坐标"""
        rect = ctypes.wintypes.RECT()
        user32.GetClientRect(self._hwnd, ctypes.byref(rect))
        client_w = rect.right - rect.left
        client_h = rect.bottom - rect.top

        if client_w <= 0:
            user32.GetWindowRect(self._hwnd, ctypes.byref(rect))
            client_w = rect.right - rect.left
            client_h = rect.bottom - rect.top
            # WindowRect: 直接屏幕坐标
            screen_x = rect.left + int(client_w * x_pct)
            screen_y = rect.top + int(client_h * y_pct)
        else:
            # ClientRect: 需要转换到屏幕坐标
            pt = POINT()
            pt.x = int(client_w * x_pct)
            pt.y = int(client_h * y_pct)
            user32.ClientToScreen(self._hwnd, ctypes.byref(pt))
            screen_x = pt.x
            screen_y = pt.y

        return screen_x, screen_y

    def _send_move(self, screen_x, screen_y):
        """移动光标到屏幕绝对坐标 (SendInput)"""
        self._send_mouse_input(screen_x, screen_y,
                               MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)

    def _send_mouse_event(self, flags):
        """发送鼠标事件 (在当前光标位置)"""
        self._send_mouse_input(0, 0, flags)

    def _send_mouse_input(self, dx, dy, flags):
        """SendInput 底层调用 (鼠标)"""
        screen_w = user32.GetSystemMetrics(SM_CXSCREEN)
        screen_h = user32.GetSystemMetrics(SM_CYSCREEN)

        # 绝对坐标: dx/dy 范围是 0-65535 (映射到整个屏幕)
        normalized_x = int((dx * 65535) / screen_w) if screen_w > 0 else 0
        normalized_y = int((dy * 65535) / screen_h) if screen_h > 0 else 0

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi = MOUSEINPUT()
        inp.union.mi.dx = normalized_x
        inp.union.mi.dy = normalized_y
        inp.union.mi.mouseData = 0
        inp.union.mi.dwFlags = flags
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = 0

        if not flags & MOUSEEVENTF_LEFTUP:
            self._check_input_target()
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise InputTargetError("鼠标输入失败，请核对游戏与脚本的运行权限")

    def _send_key(self, vk_code, key_up=False):
        """SendInput 键盘事件"""
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki = KEYBDINPUT()
        inp.union.ki.wVk = vk_code
        inp.union.ki.wScan = 0
        inp.union.ki.dwFlags = KEYEVENTF_KEYUP if key_up else 0
        inp.union.ki.time = 0
        inp.union.ki.dwExtraInfo = 0
        if not key_up:
            self._check_input_target()
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise InputTargetError("键盘输入失败，请核对游戏与脚本的运行权限")

    def _fallback_click(self, x_pct, y_pct):
        """回退: pyautogui 点击"""
        import pyautogui
        if self._game_region:
            left, top, w, h = self._game_region
            abs_x = left + int(w * x_pct)
            abs_y = top + int(h * y_pct)
            pyautogui.click(abs_x, abs_y)
        time.sleep(self.click_delay)

    # ===================== 等待 =====================

    def wait(self, seconds=None):
        if seconds is None:
            seconds = self.action_interval
        time.sleep(seconds)
