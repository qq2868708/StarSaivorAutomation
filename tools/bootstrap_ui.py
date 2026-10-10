"""Prepare the pinned local environment before starting the panel."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launch_ui import launch_lock, panel_ready
from src.quick_lookup_log import redact


def supported_python(version=None, bits=None):
    version = sys.version_info[:2] if version is None else version
    bits = struct.calcsize("P") * 8 if bits is None else bits
    return version in {(3, 11), (3, 12)} and bits == 64


def locked_versions(path):
    """The committed lock format intentionally permits exact pins only."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise ValueError("requirements-lock.txt must contain exact version pins.")
        pins[match[1]] = match[2]
    if not pins:
        raise ValueError("Dependency lock is empty.")
    return pins


def mismatched_packages(pins, version=importlib.metadata.version):
    result = []
    for name, expected in pins.items():
        try:
            installed = version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            result.append(name)
    return result


def log_output(text):
    with (ROOT / "runtime" / "bootstrap.log").open("a", encoding="utf-8") as stream:
        for line in str(text).splitlines():
            stream.write(redact(ROOT, line) + "\n")


def run_check(arguments, label, timeout=600):
    log_output(f"[{label}]")
    result = subprocess.run(
        [sys.executable, "-X", "utf8", *arguments], cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log_output(result.stdout)
    log_output(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{label} failed. See runtime/bootstrap.log.")


def prepare_environment(force=False):
    pins = locked_versions(ROOT / "requirements-lock.txt")
    missing = mismatched_packages(pins)
    if missing or force:
        port = int(os.environ.get("STARSAVIOR_UI_PORT", "8765"))
        if panel_ready(f"http://127.0.0.1:{port}/"):
            raise RuntimeError("Stop the running panel before changing its dependencies.")
        print("[2/3] Installing locked dependencies...", flush=True)
        run_check(["-m", "ensurepip", "--upgrade"], "pip bootstrap")
        install = ["-m", "pip", "install", "--disable-pip-version-check",
                   "-r", "requirements.txt", "-r", "requirements-lock.txt"]
        if force:
            install.append("--force-reinstall")
        run_check(install, "dependency installation")
        if mismatched_packages(pins):
            raise RuntimeError("Installed packages do not match the lock.")
    else:
        print("[2/3] Locked dependencies already installed.", flush=True)
    run_check(["-m", "pip", "check"], "dependency consistency", timeout=60)
    print("[3/3] Checking imports and real OCR inference...", flush=True)
    run_check([str(ROOT / "tools" / "check_environment.py")], "OCR runtime check", timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--repair", action="store_true", help="Run dependency installation again.")
    args = parser.parse_args()
    try:
        if not supported_python():
            raise RuntimeError("64-bit Python 3.11 or 3.12 is required; 3.13/3.14 are unsupported. "
                               "Back up an incompatible .venv before recreating it.")
        target = ROOT / ".venv" / "Scripts" / "python.exe"
        if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            with launch_lock(ROOT / "runtime" / "environment-setup.lock", timeout=600):
                if not target.exists():
                    print("[1/3] Creating .venv...", flush=True)
                    venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
            return subprocess.call([str(target), "-X", "utf8", __file__, *sys.argv[1:]],
                                   cwd=ROOT)
        with launch_lock(ROOT / "runtime" / "environment-setup.lock", timeout=600):
            log = ROOT / "runtime" / "bootstrap.log"
            if log.exists():
                log.replace(log.with_name("bootstrap.previous.log"))
            print(f"[1/3] Python {sys.version.split()[0]} / .venv", flush=True)
            prepare_environment(force=args.repair)
        if args.check_only:
            print("Environment verified. No game input was sent.")
            return 0
        command = [sys.executable, "-X", "utf8", str(ROOT / "launch_ui.py")]
        if args.no_browser:
            command.append("--no-browser")
        return subprocess.call(command, cwd=ROOT)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        message = redact(ROOT, str(error))
        try:
            (ROOT / "runtime").mkdir(exist_ok=True)
            log_output(message)
        except OSError:
            pass
        print(f"Startup failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
