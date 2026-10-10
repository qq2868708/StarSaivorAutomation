"""Bounded, redacted diagnostic streams for the two lookup families."""
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import re
from threading import RLock


_LOCK = RLock()
_MAX_BYTES = 2 * 1024 * 1024


def family(kind):
    if kind in {"rescuer", "rescuer_single"}:
        return "rescuer"
    if kind == "arcanum":
        return kind
    raise ValueError("Invalid scan kind")


def log_path(root, kind):
    return Path(root) / "runtime" / f"quick_lookup_{family(kind)}.jsonl"


def redact(root, value):
    if isinstance(value, dict):
        return {key: redact(root, child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(root, child) for child in value]
    if not isinstance(value, str):
        return value
    for path, replacement in ((Path(root).resolve(), "<project>"),
                              (Path.home(), "<home>")):
        for spelling in (str(path).replace("\\", "\\\\"), str(path), path.as_posix()):
            value = re.sub(re.escape(spelling), lambda _: replacement, value,
                           flags=re.IGNORECASE)
    value = re.sub(r"https?://[^\s\"']+", "<url>", value)
    value = re.sub(r"(?i)\b[a-z]:[\\/][^\r\n\"']+", "<path>", value)
    value = re.sub(r"\\\\[^\\\s]+\\[^\r\n\"']+", "<path>", value)
    value = re.sub(r"(?i)\b(token|api_key|password|secret)\s*[:=]\s*[^\s,;]+",
                   r"\1=<redacted>", value)
    return value[:4000]


def append_event(root, kind, event, **details):
    record = redact(root, {
        **details, "kind": kind, "event": event,
        "ts": datetime.now().astimezone().isoformat(),
    })
    path = log_path(root, kind)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            path.replace(path.with_suffix(".jsonl.1"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def recent_events(root, kind, limit=60):
    path = log_path(root, kind)
    with _LOCK:
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 131072))
                if size > 131072:
                    stream.readline()
                lines = deque(stream, maxlen=limit)
        except FileNotFoundError:
            return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
            if isinstance(entry, dict):
                entries.append(redact(root, entry))
        except (ValueError, UnicodeError):
            continue
    return entries
