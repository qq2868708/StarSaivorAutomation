"""Local WebUI for inspecting StarSavior automation runs.

The server only reads the game artifacts written by the automation and writes
the event/config files after an explicit save request from the browser.  It
never attaches to the game window or sends input.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.event_effects import quantify_effect_text


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
EVENTS_PATH = ROOT / "profiles" / "events" / "document_verified.json"
CONFIG_PATH = ROOT / "config.yaml"
LOG_DIR = ROOT / "logs"

TRAINING_TARGETS = {
    "力量训练": (0.86, 0.28),
    "体力训练": (0.86, 0.38),
    "韧性训练": (0.86, 0.48),
    "集中训练": (0.86, 0.58),
    "保护训练": (0.86, 0.68),
}
ACTION_TYPES = {"decision", "train", "rest", "event", "arcanum", "card_select", "shop"}

_automation_lock = threading.RLock()
_automation_process: subprocess.Popen[str] | None = None
_automation_log_handle = None


def _compact(value: Any, limit: int = 64) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _verified_label(event: dict[str, Any]) -> str:
    if event.get("document_verified"):
        return "文档核验"
    return "本地规则" if event.get("source_type") else "未核验"


def _option_summary(option: dict[str, Any]) -> str:
    quantified = option.get("quantified") or quantify_effect_text(option.get("effect_text", ""))
    parts: list[str] = []
    for name, value in (quantified.get("stats") or {}).items():
        parts.append(f"{name}{value:+g}")
    for key, label in (("stamina_delta", "耐力"), ("potential_points", "潜质"),
                       ("mood_delta", "心情"), ("currency_delta", "硬币")):
        value = quantified.get(key, 0)
        if value:
            parts.append(f"{label}{value:+g}")
    for name in quantified.get("potential_names") or []:
        parts.append(f"潜能:{name}")
    for name in quantified.get("relic_names") or []:
        parts.append(f"遗物:{name}")
    if not parts:
        parts.extend(quantified.get("unknown_effects") or [])
    return _compact("；".join(parts) or "未量化")


def _target_text(action: dict[str, Any]) -> str:
    if "target_x" in action and "target_y" in action:
        return f"客户区 ({float(action['target_x']):.3f}, {float(action['target_y']):.3f})"
    row = action.get("row_index", action.get("target_row"))
    if row is not None:
        names = list(TRAINING_TARGETS.items())
        try:
            name, (x, y) = names[int(row)]
            return f"{name} ({x:.3f}, {y:.3f})"
        except (IndexError, ValueError, TypeError):
            pass
    if action.get("option_selected") is not None:
        option = int(action["option_selected"])
        count = int(action.get("num_options") or 3)
        centers = {2: [0.65, 0.74], 3: [0.57, 0.65, 0.73], 4: [0.52, 0.59, 0.66, 0.73]}
        positions = centers.get(count) or centers[3]
        y = positions[max(0, min(option - 1, len(positions) - 1))]
        return f"事件选项 (0.800, {y:.3f})"
    return "—"


class DataStore:
    def __init__(self) -> None:
        self.events_data: dict[str, Any] = {"events": []}
        self.config: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if EVENTS_PATH.is_file():
            try:
                self.events_data = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.events_data = {"events": []}
        else:
            self.events_data = {"events": []}
        if CONFIG_PATH.is_file():
            try:
                self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                self.config = {}
        else:
            self.config = {}

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.events_data.setdefault("events", [])

    def save_events(self) -> None:
        self.events_data["event_count"] = len(self.events)
        EVENTS_PATH.write_text(json.dumps(self.events_data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    def save_config(self) -> None:
        CONFIG_PATH.write_text(yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False),
                               encoding="utf-8")

    def logs(self) -> list[tuple[Path, dict[str, Any]]]:
        result: list[tuple[Path, dict[str, Any]]] = []
        if not LOG_DIR.exists():
            return result
        for path in sorted(LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                result.append((path, value))
        return result

    @staticmethod
    def resolve_screenshot(reference: Any) -> Path | None:
        if not reference:
            return None
        raw = Path(str(reference))
        candidate = raw if raw.is_absolute() else ROOT / raw
        try:
            candidate = candidate.resolve()
            allowed = (ROOT.resolve(), (ROOT.parent / "verification").resolve())
            if not any(candidate.is_relative_to(root) for root in allowed):
                return None
        except (OSError, ValueError):
            return None
        return candidate if candidate.is_file() else None

    def action_rows(self, limit: int = 180) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path, data in self.logs()[:5]:
            run_id = str(data.get("run_id") or path.stem.replace("run_", ""))
            actions = list(data.get("actions", []))[-30:]
            for turn in data.get("turns", []):
                actions.extend(turn.get("actions", []))
            for action in actions:
                if action.get("type") not in ACTION_TYPES:
                    continue
                screenshot = self.resolve_screenshot(action.get("screenshot"))
                rows.append({
                    "run_id": run_id,
                    "type": action.get("type", "action"),
                    "time": str(action.get("ts") or action.get("timestamp") or "")[-8:],
                    "decision": action.get("attribute") or action.get("action") or
                                action.get("event_name") or action.get("name") or "—",
                    "target": _target_text(action),
                    "screenshot": action.get("screenshot"),
                    "screenshot_url": f"/api/screenshot?path={quote(str(action.get('screenshot')), safe='')}"
                                      if screenshot else None,
                    "has_screenshot": screenshot is not None,
                    "details": action,
                })
        return rows[-limit:][::-1]

    def event_rows(self, query: str = "") -> list[dict[str, Any]]:
        query = query.strip().lower()
        result = []
        for index, event in enumerate(self.events):
            searchable = " ".join(str(event.get(key, "")) for key in
                                   ("event_name", "id", "category", "source_sheet", "source_type"))
            if query and query not in searchable.lower():
                continue
            result.append({
                "index": index,
                "id": event.get("id", ""),
                "name": event.get("event_name") or event.get("id") or "未命名事件",
                "category": event.get("category", "—"),
                "source": event.get("source_sheet") or event.get("source_type") or "—",
                "verified": _verified_label(event),
                "recommended": event.get("recommended_option", "—"),
                "options": len(event.get("options") or []),
            })
        return result

    @staticmethod
    def get_path(root: dict[str, Any], path: str, default: Any = None) -> Any:
        node: Any = root
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @staticmethod
    def set_path(root: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def weight_specs(self) -> list[dict[str, Any]]:
        training = self.config.setdefault("training", {})
        weights = training.setdefault("weights", {})
        events = self.config.setdefault("events", {})
        directions = events.setdefault("decision_weights", {})
        defaults = {
            "attack": {"力量": 5.0, "体力": 4.0, "韧性": 4.0, "专注": 3.0, "保护": 2.0},
            "survival": {"韧性": 5.0, "体力": 4.0, "保护": 4.0, "力量": 3.0, "专注": 2.0},
        }
        specs = []
        for key, label in (("priority_bonus", "训练：优先属性加分"),
                           ("gain_score", "训练：增益权重"),
                           ("failure_penalty", "训练：失败率惩罚")):
            specs.append({"path": f"training.weights.{key}", "label": label,
                          "group": "训练评分", "value": weights.get(key, 0)})
        for key, label in (("failure_threshold", "训练：失败率阈值"),
                           ("adventure_gain_threshold", "训练：冒险增益阈值"),
                           ("advantage_multiplier", "训练：优势训练倍率")):
            specs.append({"path": f"training.{key}", "label": label,
                          "group": "训练安全", "value": training.get(key, 0)})
        for direction, values in defaults.items():
            current = directions.setdefault(direction, {})
            for stat, default in values.items():
                specs.append({"path": f"events.decision_weights.{direction}.{stat}",
                              "label": f"事件：{direction}向 {stat}", "group": f"事件权重 · {direction}",
                              "value": current.get(stat, default)})
        return specs


store = DataStore()
app = FastAPI(title="StarSavior 自动化检查台", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


def _automation_status() -> dict[str, Any]:
    with _automation_lock:
        process = _automation_process
        if process is None:
            return {"running": False, "pid": None, "log_path": None, "returncode": None}
        return {
            "running": process.poll() is None,
            "pid": process.pid,
            "log_path": getattr(process, "_starsavior_log_path", None),
            "returncode": process.poll(),
        }


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = _automation_status()
    return {"ok": True, "events": len(store.events), "logs": len(store.logs()), "automation": status}


@app.get("/api/runtime")
def runtime_status() -> dict[str, Any]:
    return _automation_status()


@app.post("/api/runtime/start")
def start_runtime() -> dict[str, Any]:
    global _automation_process, _automation_log_handle
    with _automation_lock:
        if _automation_process is not None and _automation_process.poll() is None:
            return _automation_status()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"ui_automation_{stamp}.log"
        _automation_log_handle = log_path.open("a", encoding="utf-8", errors="replace")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        _automation_process = subprocess.Popen(
            [sys.executable, "main.py", "--auto"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=_automation_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        _automation_process._starsavior_log_path = str(log_path.relative_to(ROOT))
        return _automation_status()


@app.post("/api/runtime/stop")
def stop_runtime() -> dict[str, Any]:
    global _automation_process, _automation_log_handle
    with _automation_lock:
        process = _automation_process
        if process is None or process.poll() is not None:
            return _automation_status()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        if _automation_log_handle is not None:
            _automation_log_handle.close()
            _automation_log_handle = None
        return _automation_status()


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    store.reload()
    logs = store.logs()
    latest = logs[0][1] if logs else {}
    latest_path = logs[0][0].name if logs else None
    return {
        "summary": latest.get("summary", {}),
        "run": {
            "id": latest.get("run_id") or (latest_path or "").replace("run_", "").replace(".json", ""),
            "started_at": latest.get("started_at"),
            "duration_seconds": latest.get("duration_seconds"),
            "exit_reason": latest.get("exit_reason"),
        },
        "logs": [{"name": path.name, "run_id": data.get("run_id"),
                  "started_at": data.get("started_at"), "exit_reason": data.get("exit_reason")}
                 for path, data in logs[:12]],
        "actions": store.action_rows(),
    }


@app.get("/api/events")
def events(query: str = Query(default="")) -> dict[str, Any]:
    store.reload()
    return {"total": len(store.events), "items": store.event_rows(query)}


@app.get("/api/events/{index}")
def event_detail(index: int) -> dict[str, Any]:
    store.reload()
    if index < 0 or index >= len(store.events):
        raise HTTPException(404, "事件不存在")
    event = store.events[index]
    options = []
    for option in event.get("options") or []:
        options.append({**option, "quantified_summary": _option_summary(option),
                        "quantified": option.get("quantified") or quantify_effect_text(option.get("effect_text", ""))})
    return {"index": index, "event": event, "options": options}


@app.put("/api/events/{index}")
def update_event(index: int, body: dict[str, Any]) -> dict[str, Any]:
    store.reload()
    if index < 0 or index >= len(store.events):
        raise HTTPException(404, "事件不存在")
    options = body.get("options")
    if not isinstance(options, list):
        raise HTTPException(400, "options 必须是数组")
    try:
        body["recommended_option"] = int(body.get("recommended_option", 1))
    except (TypeError, ValueError):
        raise HTTPException(400, "推荐分支必须是整数")
    original = store.events[index]
    # Keep the event identity and source metadata intact while allowing the
    # browser to edit the visible name, recommendation, and branch records.
    for key in ("event_name", "recommended_option", "options", "title_aliases", "category"):
        if key in body:
            original[key] = body[key]
    store.save_events()
    return {"ok": True, "event": original}


@app.get("/api/config")
def config_values() -> dict[str, Any]:
    store.reload()
    return {"specs": store.weight_specs()}


@app.put("/api/config")
def update_config(body: dict[str, Any]) -> dict[str, Any]:
    store.reload()
    updates = body.get("updates")
    if not isinstance(updates, dict):
        raise HTTPException(400, "updates 必须是对象")
    for path, raw in updates.items():
        try:
            value = float(raw) if isinstance(raw, str) and "." in raw else int(raw)
        except (TypeError, ValueError):
            raise HTTPException(400, f"参数无效：{path}")
        store.set_path(store.config, str(path), value)
    store.save_config()
    return {"ok": True, "specs": store.weight_specs()}


@app.get("/api/screenshot")
def screenshot(path: str = Query(...)) -> FileResponse:
    resolved = store.resolve_screenshot(path)
    if resolved is None:
        raise HTTPException(404, "截图不存在或路径不允许")
    return FileResponse(resolved, media_type="image/png", filename=resolved.name)


def main() -> None:
    port = int(os.environ.get("STARSAVIOR_UI_PORT", "8765"))
    url = f"http://127.0.0.1:{port}/"
    print(f"[WebUI] StarSavior 自动化检查台: {url}")
    if os.environ.get("STARSAVIOR_UI_NO_BROWSER") != "1":
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
