"""Local, AI-independent control queue, checkpoints and exclusive process ownership."""
from __future__ import annotations

import ctypes
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class RuntimeInterrupt(BaseException):
    """Unwind a handler before its next input; ordinary handler catches cannot swallow it."""

    def __init__(self, reason="control"):
        self.reason = reason
        super().__init__(reason)


def process_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        if os.name == "nt":
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            api.OpenProcess.restype = ctypes.c_void_p
            api.CloseHandle.argtypes = [ctypes.c_void_p]
            api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            handle = api.OpenProcess(0x1000, False, pid)
            if not handle:
                return ctypes.get_last_error() == 5
            try:
                code = ctypes.c_ulong()
                return bool(api.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
            finally:
                api.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except (ValueError, TypeError, OSError):
        return False


class RuntimeControl:
    VALID_COMMANDS = {"pause", "resume", "step", "stop", "ai_action"}

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or Path(__file__).resolve().parent.parent)
        self.directory = self.root / "runtime"
        self.status_path = self.directory / "status.json"
        self.checkpoint_path = self.directory / "checkpoint.json"
        self.commands = self.directory / "commands"
        self._owner = None

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _read(path):
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def acquire(self):
        """OS lock released automatically on process death, including CLI vs WebUI."""
        self.directory.mkdir(parents=True, exist_ok=True)
        stream = (self.directory / "owner.lock").open("a+b")
        try:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("已有自动化脚本运行，请先暂停或停止该实例") from error
        self._owner = stream

    def release(self):
        if self._owner:
            self._owner.close()
            self._owner = None

    def read_status(self):
        return self._read(self.status_path) or {"state": "offline"}

    def read_checkpoint(self):
        return self._read(self.checkpoint_path)

    def write_status(self, state, reason="", **details):
        value = {"schema_version": 1, "updated_at": datetime.now().isoformat(),
                 **details, "state": state, "reason": reason, "pid": os.getpid()}
        self._atomic_write(self.status_path, value)
        return value

    def write_checkpoint(self, **details):
        value = {"schema_version": 1, "updated_at": datetime.now().isoformat(), **details}
        self._atomic_write(self.checkpoint_path, value)
        return value

    def issue(self, command, **payload):
        if command not in self.VALID_COMMANDS:
            raise ValueError(f"未知命令: {command}")
        value = {**payload, "id": uuid.uuid4().hex, "command": command,
                 "created_at": datetime.now().isoformat()}
        self._atomic_write(self.commands / f"{time.time_ns():020d}_{value['id']}.json", value)
        return value

    def clear_pending_command(self):
        # Only the owner may clear commands while starting a new run.
        if not self._owner:
            raise RuntimeError("只有运行实例可以清理旧命令")
        if self.commands.exists():
            for path in self.commands.glob("*.json"):
                path.unlink(missing_ok=True)

    def consume(self):
        """Atomically claim a queue entry; separate writers cannot overwrite commands."""
        for path in sorted(self.commands.glob("*.json")):
            claim = path.with_suffix(".claimed")
            try:
                os.replace(path, claim)
            except FileNotFoundError:
                continue
            value = self._read(claim)
            claim.unlink(missing_ok=True)
            if value and value.get("id") and value.get("command") in self.VALID_COMMANDS:
                return value
        return None

    def acknowledge(self, command, accepted, reason=""):
        self._atomic_write(self.directory / "acks" / f"{command['id']}.json", {
            "id": command["id"], "accepted": accepted, "reason": reason,
            "updated_at": datetime.now().isoformat(),
        })

    def wait_for_command(self, timeout=0.25):
        deadline = time.monotonic() + max(0, timeout)
        while True:
            command = self.consume()
            if command or time.monotonic() >= deadline:
                return command
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def claim_ai(self, client_id, run_id, pause_id):
        """Explicitly register a present AI client for this one safe pause."""
        if not str(client_id).strip():
            raise ValueError("client_id 不能为空")
        status = self.read_status()
        if (status.get("state") != "paused_safe" or not process_alive(status.get("pid"))
                or status.get("run_id") != run_id or status.get("pause_id") != pause_id):
            raise ValueError("脚本已离开本次安全暂停")
        previous = self._read(self.directory / "ai_client.json") or {}
        same_pause = previous.get("run_id") == run_id and previous.get("pause_id") == pause_id
        if same_pause and previous.get("expires_at", 0) > time.time():
            if previous.get("client_id") != str(client_id)[:100]:
                raise ValueError("已有另一个 AI 客户端接管本次暂停")
            previous["expires_at"] = time.time() + 120
            self._atomic_write(self.directory / "ai_client.json", previous)
            return previous
        lease = {"client_id": str(client_id)[:100], "lease_id": uuid.uuid4().hex,
                 "run_id": run_id, "pause_id": pause_id, "expires_at": time.time() + 120}
        self._atomic_write(self.directory / "ai_client.json", lease)
        return lease

    def valid_ai(self, command, run_id, pause_id):
        lease = self._read(self.directory / "ai_client.json") or {}
        return bool(lease.get("lease_id") and command.get("lease_id") == lease["lease_id"]
                    and lease.get("run_id") == run_id == command.get("run_id")
                    and lease.get("pause_id") == pause_id == command.get("pause_id")
                    and lease.get("expires_at", 0) > time.time())

    def emit(self, run_id, event, **details):
        """Append-only event hooks, readable without a running AI."""
        self.directory.mkdir(parents=True, exist_ok=True)
        row = {"run_id": run_id, "event": event, "ts": datetime.now().isoformat(), **details}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
