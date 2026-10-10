"""Open the existing StarSavior panel or start its local WebUI once."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import msvcrt
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import uuid
import webbrowser


@contextmanager
def launch_lock(path: Path):
    """Serialize double clicks while the HTTP server is still starting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + 35
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("另一次面板启动仍在进行，请稍后重试")
                time.sleep(.2)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def panel_ready(url: str) -> bool:
    try:
        with urlopen(url + "api/runtime/diagnostics", timeout=2) as response:
            status = json.load(response)
        return isinstance(status, dict) and {"automatic_active", "safe_for_ai"} <= status.keys()
    except (OSError, URLError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="只检查或启动服务，不打开浏览器")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        port = int(os.environ.get("STARSAVIOR_UI_PORT", "8765"))
        if not 1 <= port <= 65535:
            raise ValueError("STARSAVIOR_UI_PORT 必须在 1 至 65535 之间")
        url = f"http://127.0.0.1:{port}/"
        with launch_lock(root / "runtime" / f"ui-launcher-{port}.lock"):
            if not panel_ready(url):
                with socket.socket() as probe:
                    probe.settimeout(1)
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        raise RuntimeError(f"端口 {port} 已占用，但 StarSavior 面板未就绪")
                log_id = uuid.uuid4().hex
                stdout_path = root / "runtime" / f"ui-launcher-{log_id}.stdout.log"
                stderr_path = root / "runtime" / f"ui-launcher-{log_id}.stderr.log"
                environment = {**os.environ, "PYTHONUTF8": "1", "STARSAVIOR_UI_NO_BROWSER": "1"}
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(root / "ui_app.py")], cwd=root,
                        env=environment, stdout=stdout, stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                deadline = time.monotonic() + 30
                while not panel_ready(url):
                    if process.poll() is not None:
                        raise RuntimeError(f"面板启动失败，查看日志：{stderr_path}")
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"面板启动超时，查看日志：{stderr_path}")
                    time.sleep(.3)
        if not args.no_browser:
            webbrowser.open(url)
        print(f"StarSavior 面板已就绪：{url}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"启动面板失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
