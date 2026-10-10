"""Check imports and bundled OCR inference without capturing or clicking."""
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import web_ui
    from src.light_ocr import LightOCR

    canvas = Image.new("RGB", (600, 180), "white")
    font = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "C:/Windows"))
                                 / "Fonts" / "msyh.ttc"), 48)
    ImageDraw.Draw(canvas).text((30, 45), "\u529b\u91cf", font=font, fill="black")
    rows = LightOCR().recognize(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    if not any(text.strip() == "\u529b\u91cf" and score >= .85 for text, score, _ in rows):
        raise RuntimeError("OCR short-label confidence check failed; verify the locked OCR version.")
    assert web_ui.app is not None
    print("Runtime imports and two-character OCR inference passed (score >= 0.85).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
