"""
屏幕截图模块 - 支持后台窗口捕获

使用 win32gui (PrintWindow/BitBlt) 直接从窗口 DC 截取画面。
即使游戏窗口被遮挡或在后台也能正常截图。
移植自 SleepRunner BitBltCapture.cs
"""
import threading
import time
import ctypes
import ctypes.wintypes
from pathlib import Path

import cv2
import numpy as np

# Win32 API
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# 常量
SRCCOPY = 0x00CC0020
PW_CLIENTONLY = 0x01
PW_RENDERFULLCONTENT = 0x02
SM_CXSCREEN = 0
SM_CYSCREEN = 1

# 截图超时 (秒) — PrintWindow 在目标窗口无响应时会永久阻塞
CAPTURE_TIMEOUT = 5.0

# 结构体
class RECT(ctypes.Structure):
    _fields_ = [
        ("Left", ctypes.c_long),
        ("Top", ctypes.c_long),
        ("Right", ctypes.c_long),
        ("Bottom", ctypes.c_long),
    ]


# Windows handles are pointer-sized. ctypes defaults to 32-bit integers, which
# truncates HDC/HBITMAP values on this 64-bit Python build.
_WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL,
                                 ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
_user_signatures = {
    "FindWindowW": ([ctypes.c_wchar_p, ctypes.c_wchar_p], ctypes.wintypes.HWND),
    "GetWindowRect": ([ctypes.wintypes.HWND, ctypes.c_void_p], ctypes.wintypes.BOOL),
    "GetClientRect": ([ctypes.wintypes.HWND, ctypes.c_void_p], ctypes.wintypes.BOOL),
    "GetWindowTextLengthW": ([ctypes.wintypes.HWND], ctypes.c_int),
    "GetWindowTextW": ([ctypes.wintypes.HWND, ctypes.c_wchar_p, ctypes.c_int], ctypes.c_int),
    "EnumWindows": ([_WNDENUMPROC, ctypes.wintypes.LPARAM], ctypes.wintypes.BOOL),
    "GetDC": ([ctypes.wintypes.HWND], ctypes.wintypes.HDC),
    "ReleaseDC": ([ctypes.wintypes.HWND, ctypes.wintypes.HDC], ctypes.c_int),
    "PrintWindow": ([ctypes.wintypes.HWND, ctypes.wintypes.HDC, ctypes.c_uint], ctypes.wintypes.BOOL),
    "GetSystemMetrics": ([ctypes.c_int], ctypes.c_int),
    "IsWindow": ([ctypes.wintypes.HWND], ctypes.wintypes.BOOL),
    "IsIconic": ([ctypes.wintypes.HWND], ctypes.wintypes.BOOL),
    "GetCursorPos": ([ctypes.c_void_p], ctypes.wintypes.BOOL),
    "ClientToScreen": ([ctypes.wintypes.HWND, ctypes.c_void_p], ctypes.wintypes.BOOL),
    "GetForegroundWindow": ([], ctypes.wintypes.HWND),
    "SetForegroundWindow": ([ctypes.wintypes.HWND], ctypes.wintypes.BOOL),
    "SendInput": ([ctypes.c_uint, ctypes.c_void_p, ctypes.c_int], ctypes.c_uint),
}
_gdi_signatures = {
    "CreateCompatibleDC": ([ctypes.wintypes.HDC], ctypes.wintypes.HDC),
    "CreateCompatibleBitmap": ([ctypes.wintypes.HDC, ctypes.c_int, ctypes.c_int], ctypes.wintypes.HANDLE),
    "SelectObject": ([ctypes.wintypes.HDC, ctypes.wintypes.HANDLE], ctypes.wintypes.HANDLE),
    "DeleteObject": ([ctypes.wintypes.HANDLE], ctypes.wintypes.BOOL),
    "DeleteDC": ([ctypes.wintypes.HDC], ctypes.wintypes.BOOL),
    "GetDeviceCaps": ([ctypes.wintypes.HDC, ctypes.c_int], ctypes.c_int),
    "BitBlt": ([ctypes.wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_uint32], ctypes.wintypes.BOOL),
    "GetDIBits": ([ctypes.wintypes.HDC, ctypes.wintypes.HANDLE, ctypes.c_uint, ctypes.c_uint,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint], ctypes.c_int),
}
for _library, _signatures in ((user32, _user_signatures), (gdi32, _gdi_signatures)):
    for _name, (_args, _result) in _signatures.items():
        getattr(_library, _name).argtypes = _args
        getattr(_library, _name).restype = _result

# Keep client rectangles and screen capture in the same physical pixel space.
user32.SetProcessDPIAware()


class ScreenCapture:
    """后台窗口截图 (移植自 SleepRunner BitBltCapture)"""

    def __init__(self, window_title="StarSavior"):
        self.window_title = window_title
        self._hwnd = None
        self._dpi_scale = None
        self.last_capture_backend = None

    # ===================== 窗口查找 =====================

    def find_window(self):
        """查找游戏窗口，返回 (left, top, width, height) 和窗口句柄"""
        hwnd = user32.FindWindowW(None, self.window_title)
        if hwnd:
            self._hwnd = hwnd
            rect = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.Right - rect.Left
            h = rect.Bottom - rect.Top
            return (rect.Left, rect.Top, w, h)

        # 回退: 枚举所有窗口
        matches = self._enum_windows(self.window_title)
        if matches:
            hwnd, rect = matches[0]
            self._hwnd = hwnd
            return (rect.Left, rect.Top, rect.Right - rect.Left, rect.Bottom - rect.Top)

        return None

    def get_hwnd(self):
        """获取绑定的窗口句柄"""
        if self._hwnd is None:
            self.find_window()
        return self._hwnd

    @staticmethod
    def _enum_windows(title_match):
        """枚举所有顶层窗口，找到标题包含指定字符串的"""
        results = []

        def callback(hwnd, _):
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if title_match.lower() in buf.value.lower():
                rect = RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                results.append((hwnd, rect, buf.value == title_match))
            return True

        user32.EnumWindows(_WNDENUMPROC(callback), 0)
        # 精确匹配优先
        return ([(h, r) for h, r, exact in results if exact]
                + [(h, r) for h, r, exact in results if not exact])

    # ===================== DPI =====================

    @property
    def dpi_scale(self):
        """获取 DPI 缩放因子 (物理像素 / 逻辑像素)"""
        if self._dpi_scale is None:
            logical_w = user32.GetSystemMetrics(SM_CXSCREEN)
            dc = user32.GetDC(0)
            physical_w = gdi32.GetDeviceCaps(dc, 118)  # DESKTOPHORZRES
            user32.ReleaseDC(0, dc)
            if logical_w > 0 and physical_w > logical_w:
                self._dpi_scale = physical_w / logical_w
            else:
                self._dpi_scale = 1.0
        return self._dpi_scale

    # ===================== 截图 =====================

    def capture_game(self, region=None):
        """
        捕获游戏窗口画面 (BGR numpy array)。

        使用 PrintWindow API 直接从窗口 DC 获取画面。
        即使窗口被遮挡也能正常工作。

        关键: PrintWindow 在目标窗口无响应时会永久阻塞主线程,
        所以放在子线程中执行并设置超时 (CAPTURE_TIMEOUT=5s)。
        超时后抛出 RuntimeError 让调用方处理。
        """
        hwnd = self.get_hwnd()
        if hwnd is None:
            raise RuntimeError("无法找到游戏窗口")
        if not user32.IsWindow(hwnd):
            self._hwnd = None
            raise RuntimeError("游戏窗口已关闭，请重新打开游戏")
        if user32.IsIconic(hwnd):
            raise RuntimeError("游戏窗口已最小化，请恢复窗口后重试")

        # 获取窗口尺寸
        try:
            client_rect = RECT()
            if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
                user32.GetWindowRect(hwnd, ctypes.byref(client_rect))
                log_w = client_rect.Right - client_rect.Left
                log_h = client_rect.Bottom - client_rect.Top
            else:
                log_w = client_rect.Right - client_rect.Left
                log_h = client_rect.Bottom - client_rect.Top
            if log_w <= 0 or log_h <= 0:
                user32.GetWindowRect(hwnd, ctypes.byref(client_rect))
                log_w = client_rect.Right - client_rect.Left
                log_h = client_rect.Bottom - client_rect.Top
        except Exception:
            user32.GetWindowRect(hwnd, ctypes.byref(client_rect))
            log_w = client_rect.Right - client_rect.Left
            log_h = client_rect.Bottom - client_rect.Top

        if log_w <= 0 or log_h <= 0:
            raise RuntimeError(f"窗口尺寸无效: {log_w}x{log_h}")

        dpi = self.dpi_scale
        width = int(log_w * dpi)
        height = int(log_h * dpi)

        # 在子线程中执行 GDI 截图 (防止 PrintWindow 阻塞主线程)
        result = [None]
        error = [None]

        def _capture_thread():
            hdc_window = None
            hdc_mem = None
            hbitmap = None
            h_old = 0
            try:
                hdc_window = user32.GetDC(hwnd)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
                hbitmap = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
                if not hdc_window or not hdc_mem or not hbitmap:
                    raise RuntimeError("无法创建窗口截图资源")
                h_old = gdi32.SelectObject(hdc_mem, hbitmap)

                ok = user32.PrintWindow(hwnd, hdc_mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
                if not ok:
                    gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_window, 0, 0, SRCCOPY)

                # GetDIBits
                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ("biSize", ctypes.c_uint32),
                        ("biWidth", ctypes.c_int32),
                        ("biHeight", ctypes.c_int32),
                        ("biPlanes", ctypes.c_uint16),
                        ("biBitCount", ctypes.c_uint16),
                        ("biCompression", ctypes.c_uint32),
                        ("biSizeImage", ctypes.c_uint32),
                        ("biXPelsPerMeter", ctypes.c_int32),
                        ("biYPelsPerMeter", ctypes.c_int32),
                        ("biClrUsed", ctypes.c_uint32),
                        ("biClrImportant", ctypes.c_uint32),
                    ]

                bmi = BITMAPINFOHEADER()
                bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.biWidth = width
                bmi.biHeight = -height
                bmi.biPlanes = 1
                bmi.biBitCount = 32
                bmi.biCompression = 0

                buf_size = width * height * 4
                pixels = (ctypes.c_ubyte * buf_size)()
                # GetDIBits requires the bitmap to be deselected from its DC.
                gdi32.SelectObject(hdc_mem, h_old)
                copied = gdi32.GetDIBits(hdc_mem, hbitmap, 0, height, pixels,
                                       ctypes.byref(bmi), 0)
                if copied != height:
                    raise RuntimeError(f"截图读取失败: {copied}/{height} 行")

                # BGRA → BGR numpy array
                arr = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
                result[0] = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            except Exception as e:
                error[0] = e
            finally:
                # 清理 GDI 资源
                if hdc_mem and hbitmap and h_old:
                    try:
                        gdi32.SelectObject(hdc_mem, h_old)
                    except Exception:
                        pass
                if hbitmap:
                    try:
                        gdi32.DeleteObject(hbitmap)
                    except Exception:
                        pass
                if hdc_mem:
                    try:
                        gdi32.DeleteDC(hdc_mem)
                    except Exception:
                        pass
                if hdc_window:
                    try:
                        user32.ReleaseDC(hwnd, hdc_window)
                    except Exception:
                        pass

        t = threading.Thread(target=_capture_thread, daemon=True)
        t.start()
        t.join(timeout=CAPTURE_TIMEOUT)

        if t.is_alive():
            print(f"[截图] PrintWindow 超时 ({CAPTURE_TIMEOUT}s), 窗口可能无响应")
            # 无法安全终止 GDI 线程, 但主线程可继续运行
            raise RuntimeError(f"截图超时 ({CAPTURE_TIMEOUT}s)")

        if error[0]:
            raise RuntimeError(f"截图失败: {error[0]}")

        if result[0] is None:
            raise RuntimeError("截图返回空")

        self.last_capture_backend = "PrintWindow/GDI"
        if float(result[0].std()) < 2:
            # Unity/DirectX can report PrintWindow success while returning black.
            # Capture only this game's client area when it is the foreground app.
            if user32.GetForegroundWindow() != hwnd:
                raise RuntimeError("后台截图为黑屏，请将 StarSavior 切到前台后重试")
            import mss
            rect = ctypes.wintypes.RECT()
            point = ctypes.wintypes.POINT(0, 0)
            if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
                raise RuntimeError("无法读取游戏客户区尺寸")
            if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
                raise RuntimeError("无法定位游戏客户区")
            with mss.mss() as screen:
                pixels = np.asarray(screen.grab({
                    "left": point.x, "top": point.y,
                    "width": rect.right - rect.left,
                    "height": rect.bottom - rect.top,
                }))
            if user32.GetForegroundWindow() != hwnd:
                raise RuntimeError("截图时游戏焦点改变，请切回游戏后重试")
            result[0] = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
            self.last_capture_backend = "MSS foreground client area"

        return result[0]

    def capture_region(self, x, y, w, h):
        """截取全屏的指定区域 (回退用)"""
        import mss
        with mss.mss() as sct:
            monitor = {"left": x, "top": y, "width": w, "height": h}
            img = np.array(sct.grab(monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def save_capture(self, img, name):
        """保存截图到 templates 目录"""
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        filepath = templates_dir / f"{name}.png"
        cv2.imwrite(str(filepath), img)
        print(f"[截图] 已保存: {filepath}")
        return filepath

    @property
    def window_handle(self):
        return self._hwnd
