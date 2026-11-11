"""Run the calibrated arena daily, or observe without input by default."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.capture import ScreenCapture, is_window_foreground, user32
from src.controller import Controller
from src.daily_arena import ArenaDaily, read_arena
from src.light_ocr import LightOCR
from src.runtime_control import RuntimeControl


def main():
    parser = argparse.ArgumentParser(description="圣铠 NPC 老手日常；默认只观察")
    parser.add_argument("--run", action="store_true", help="从主界面开始实际执行")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    directory = ROOT / "runtime" / "daily-arena" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    control = RuntimeControl(ROOT)
    planner = ArenaDaily()
    report = {"task": "arena_npc_senior", "state": "observing", "returned_home": False,
              "mode": "run" if args.run else "observe", "inputs_sent": 0}
    capture = ScreenCapture("StarSavior")
    ocr = LightOCR()
    previous_action = None

    def save():
        report["planner"] = asdict(planner)
        RuntimeControl._atomic_write(directory / "status.json", report)
        RuntimeControl._atomic_write(directory.parent / "latest.json", report)

    def check_stop():
        if user32.GetAsyncKeyState(0x1B) & 0x8000:
            raise KeyboardInterrupt("用户按下 Escape")

    try:
        if args.run:
            control.acquire()
        if not capture.find_window():
            raise RuntimeError("未找到 StarSavior")
        hwnd = capture.get_hwnd()
        controller = Controller(click_delay=.25, auto_refocus=False)
        controller.set_window_handle(hwnd)
        controller.poll_control = check_stop
        deadline = time.monotonic() + 600
        progress_time = time.monotonic()
        report["state"] = "running" if args.run else "observing"
        for index in range(500):
            if args.run:
                check_stop()
                if not is_window_foreground(hwnd):
                    raise RuntimeError("游戏失去前台，已停止输入；重新运行须先回到主界面")
            frame = capture.capture_game()
            image_path = directory / f"{index:04}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError("保存证据截图失败")
            items = ocr.recognize(frame)
            observation = read_arena(frame, items)
            report.update(observation=observation, screenshot=str(image_path),
                          updated_at=datetime.now().astimezone().isoformat())
            if not args.run:
                save()
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
            action = planner.step(observation)
            evidence = {"observation": observation, "action": action,
                        "ocr": items, "screenshot": str(image_path)}
            RuntimeControl._atomic_write(directory / f"{index:04}.json", evidence)
            report["action"] = action
            if action["kind"] == "pause":
                raise RuntimeError(action["reason"])
            if action["kind"] == "complete":
                report.update(state="complete", returned_home=True, result=action)
                save()
                print(json.dumps(action, ensure_ascii=False, indent=2))
                return 0
            now = time.monotonic()
            if now > deadline or now - progress_time > 25:
                raise RuntimeError("页面或动作超时；未自动重放消耗次数的确认操作")
            # A button may remain on screen while its first click is being processed.
            # Do not repeat an unchanged navigation/tab action. Spending confirmation
            # is also protected by ArenaDaily.pending until the actual result arrives.
            if action["kind"] in {"click", "scroll"}:
                fingerprint = (observation["page"], action["kind"], action.get("name"))
                if fingerprint == previous_action:
                    raise RuntimeError("上次操作未产生预期变化，停止重复输入")
                previous_action = fingerprint if action["kind"] == "click" else None
                progress_time = now
                save()  # Persist intent before any input, including the spending confirmation.
                check_stop()
                if action["kind"] == "click":
                    controller.click_at_percent(*action["point"], source="daily_arena")
                else:
                    controller.scroll_at_percent(*action["point"], clicks=action["clicks"],
                                                 source="daily_arena")
                report["inputs_sent"] += 1
            save()
            controller.wait(.8)
        raise RuntimeError("识别次数超出限制")
    except KeyboardInterrupt:
        report.update(state="stopped", reason="用户中止，保留现场且不继续输入")
        save()
        return 130
    except Exception as exc:
        report.update(state="paused_safe", reason=str(exc))
        save()
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        control.release()


if __name__ == "__main__":
    sys.exit(main())
