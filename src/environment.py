"""Write a redacted runtime environment snapshot for local diagnostics."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from datetime import datetime
from pathlib import Path
import sys
import uuid


_RUNTIME_PACKAGES = (
    "numpy",
    "opencv-python",
    "rapidocr-onnxruntime",
    "onnxruntime",
    "PyYAML",
    "Pillow",
    "fastapi",
    "uvicorn",
    "mss",
    "pyautogui",
    "pynput",
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _python_executable(root: Path) -> str:
    executable = Path(sys.executable)
    try:
        relative = executable.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return executable.name
    return relative.as_posix()


def record_environment(root: Path | str, *, source: str, port: int | None = None) -> Path:
    """Persist a local, redacted snapshot and return its path."""
    root = Path(root).resolve()
    output = root / "runtime" / "environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema_version": 1,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source": source,
        "process_id": os.getpid(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": _python_executable(root),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.architecture()[0],
        },
        "runtime": {
            "port": port,
            "python_utf8": os.environ.get("PYTHONUTF8") == "1",
        },
        "packages": {package: _version(package) for package in _RUNTIME_PACKAGES},
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
