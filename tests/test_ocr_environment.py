"""Integration guard for the OCR version and short-label confidence regression."""
import importlib.metadata
import unittest

from tools.check_environment import main


class OCREnvironmentTests(unittest.TestCase):
    def test_two_character_inference_uses_the_pinned_engine(self):
        self.assertEqual(importlib.metadata.version("rapidocr-onnxruntime"), "1.4.4")
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
