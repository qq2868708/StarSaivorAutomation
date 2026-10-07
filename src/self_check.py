"""Read-only startup verification; never dispatches a handler or sends input."""
import ctypes
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .controller import INPUT
from .light_ocr import LightOCR
from .rule_engine import TrainingRuleLoader


def run_startup_check(trainer, require_game=False):
    output = Path(__file__).resolve().parents[2] / "verification"
    output.mkdir(exist_ok=True)
    report = {
        "python": sys.version.split()[0],
        "architecture": platform.architecture()[0],
        "python_executable": sys.executable,
        "backend": LightOCR.backend_name,
        "game_input_sent": False,
        "checks": {},
    }
    try:
        for package in ("numpy", "opencv-python", "rapidocr-onnxruntime",
                        "onnxruntime", "PyYAML", "Pillow"):
            report.setdefault("packages", {})[package] = importlib.metadata.version(package)
        expected_input_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        assert ctypes.sizeof(INPUT) == expected_input_size, "Windows input structure size mismatch"
        report["checks"]["win32_input_layout"] = True

        # Load each selectable training profile, including paths used after switching strategy.
        profile_files = sorted(Path("profiles/training").glob("*.json"))
        for profile_file in profile_files:
            TrainingRuleLoader.load_from_file(str(profile_file))
        assert profile_files, "No training profiles found"
        report["profiles"] = [path.stem for path in profile_files]
        report["checks"]["trainer_initialization_and_profiles"] = True

        canvas = Image.new("RGB", (1100, 220), "white")
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 48)
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 25), "力量训练 失败率 10%", font=font, fill="black")
        draw.text((30, 115), "StarSavior OCR Ready", font=font, fill="black")
        canvas.save(output / "synthetic_ocr.png")
        sample = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        detailed = trainer.ui.ocr.recognize_detailed(sample)
        text = " ".join(item[0] for item in detailed)
        assert "力量训练" in text and "失败率" in text and "10" in text and "StarSavior" in text, f"OCR mismatch: {text}"
        assert detailed and all(item[2] is not None for item in detailed), "OCR boxes missing"
        region_text = trainer.ui.ocr.recognize_region(sample, 0, 0, 1, 0.5)
        assert "力量" in region_text and "StarSavior" not in region_text, f"ROI mismatch: {region_text}"
        report["synthetic_ocr_text"] = text
        report["checks"]["chinese_english_ocr_and_region_cropping"] = True

        region = trainer.capture.find_window()
        report["game_window_found"] = region is not None
        if region is not None:
            screenshot = trainer.capture.capture_game()
            assert screenshot.size and float(screenshot.std()) > 2, "Game capture is empty or black"
            cv2.imwrite(str(output / "game_capture.png"), screenshot)
            recognized = trainer.ui.ocr.recognize_detailed(screenshot)
            assert recognized, "Game capture contained no recognized text"
            report["game_capture_size"] = [screenshot.shape[1], screenshot.shape[0]]
            report["capture_backend"] = trainer.capture.last_capture_backend
            report["game_ocr"] = [{"text": text, "score": score, "box": box}
                                  for text, score, box in recognized]
            report["checks"]["game_capture_and_ocr"] = True
        elif require_game:
            raise RuntimeError("未找到 StarSavior 游戏窗口，请先打开游戏")
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    (output / "startup_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "game_ocr"}
    if "game_ocr" in report:
        summary["game_ocr_text_regions"] = len(report["game_ocr"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n检查记录：{output / 'startup_check.json'}")
    return 0 if report["passed"] else 1
