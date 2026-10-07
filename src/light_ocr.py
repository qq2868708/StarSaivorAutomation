"""OCR interfaces omitted by upstream, implemented with bundled CPU models.

The existing LightOCR/DirectMLOCR names remain compatible with the trainer.
Both use RapidOCR on CPU; no Windows language pack or external model path is
required, and detection results retain their bounding boxes.
"""
from functools import lru_cache
from threading import Lock

import numpy as np


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False,
                    intra_op_num_threads=2, inter_op_num_threads=1,
                    det_limit_type="max", det_limit_side_len=960)


_ocr_lock = Lock()


class LightOCR:
    backend_name = "RapidOCR / ONNX Runtime CPU"

    def recognize(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []
        with _ocr_lock:
            result, _ = _engine()(np.ascontiguousarray(image))
        return [(str(text), float(score), np.asarray(box).tolist())
                for box, text, score in (result or [])]

    def recognize_region(self, screenshot, x_pct, y_pct, w_pct, h_pct):
        if screenshot is None or screenshot.size == 0:
            return ""
        height, width = screenshot.shape[:2]
        left = max(0, min(width, int(width * x_pct)))
        top = max(0, min(height, int(height * y_pct)))
        right = max(0, min(width, int(width * (x_pct + w_pct))))
        bottom = max(0, min(height, int(height * (y_pct + h_pct))))
        if right <= left or bottom <= top:
            return ""
        return " ".join(item[0] for item in
                        self.recognize(screenshot[top:bottom, left:right]))

    def recognize_detailed(self, image):
        return self.recognize(image)

    def ocr(self, image, cls=False, **kwargs):
        """Retain the PaddleOCR-style output used by legacy diagnostic callers."""
        return [[[box, (text, score)]
                 for text, score, box in self.recognize(image)]]


class DirectMLOCR(LightOCR):
    """Compatibility name; this local implementation uses CPU inference."""
