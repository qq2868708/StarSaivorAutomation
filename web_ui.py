"""Local WebUI for inspecting StarSavior automation runs.

The server reads the game artifacts written by the automation and writes the
event/config files after an explicit save request from the browser. Runtime
controls are sent to the automation process through a local JSON command
queue. The post-run journey-information scan is the one explicit exception:
it verifies the foreground game and may click the magnifier once.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
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
from src.runtime_control import RuntimeControl, process_alive
from src.journey_target import JourneyTargetStore, JourneyTargetPending, load_catalog, precheck_target
from src.journey_record import JourneyRecordStore
from src.journey_info_scan import JourneyInfoScanner, JourneyInfoScanError
from src.quick_lookup import QuickLookupScanner, QuickLookupStore, build_match


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
_runtime_control = RuntimeControl(ROOT)


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
    runtime = _runtime_control.read_status()
    checkpoint = _runtime_control.read_checkpoint()
    process = _automation_process
    managed_alive = process is not None and process.poll() is None
    owner_alive = process_alive(runtime.get("pid")) and runtime.get("state") not in {
        "stopped", "journey_end", "finished", "offline"}
    state = runtime.get("state", "offline")
    if managed_alive and not owner_alive:
        state = "starting"
    elif not owner_alive and state not in {"stopped", "journey_end", "finished", "offline"}:
        state = "process_exit"
    return {
        "running": bool(managed_alive or owner_alive),
        "automatic_active": owner_alive and state == "running",
        "pid": runtime.get("pid") if owner_alive else (process.pid if managed_alive else None),
        "log_path": getattr(process, "_starsavior_log_path", None) if process else None,
        "returncode": process.poll() if process else None,
        "state": state, "reason": runtime.get("reason", ""), "status": runtime,
        "checkpoint": checkpoint,
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


def _target_store():
    return JourneyTargetStore(_runtime_control.root)


@app.get("/api/journey-target")
def journey_target():
    runtime = _automation_status()
    try:
        target = _target_store().read()
        checked = precheck_target(target or {})
    except JourneyTargetPending as error:
        target = None
        checked = {"ready": False, "errors": [str(error)], "warnings": []}
    applied = runtime["status"].get("journey_target") or {}
    return {"catalog": load_catalog(), "target": target, "precheck": checked,
            "applied": bool(target and applied.get("revision") == target.get("revision")),
            "progress": runtime["status"].get("journey_target_state") if target and applied.get("revision") == target.get("revision") else None,
            "script_supports_targets": "journey_target" in runtime["status"],
            "runtime_state": runtime["state"], "running": runtime["running"]}


@app.post("/api/journey-target/precheck")
def check_journey_target(body: dict[str, Any]):
    return precheck_target(body)


@app.get("/api/journey-records")
def journey_records():
    records = JourneyRecordStore(_runtime_control.root)
    result = []
    if records.directory.exists():
        for path in sorted(records.directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
            try:
                result.append(records.read(path.stem))
            except (OSError, ValueError):
                continue
    return {"records": result, "active_journey_id": _automation_status()["status"].get("journey_id")}


@app.patch("/api/journey-records/{journey_id}")
def supplement_journey_record(journey_id: str, body: dict[str, Any]):
    runtime = _automation_status()
    if runtime["running"] and runtime["status"].get("journey_id") == journey_id:
        raise HTTPException(409, "本轮脚本尚未退出，退出后再补录结算结果")
    try:
        return JourneyRecordStore(_runtime_control.root).supplement(journey_id, body)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/journey-records/{journey_id}/scan-current")
def scan_current_journey_record(journey_id: str):
    """Read the currently visible journey-information panel.

    The automation process must be stopped first.  When the terminal page is
    visible the scanner may perform the single, verified magnifier click; it
    never resumes a journey or chooses rewards/skills.
    """
    runtime = _automation_status()
    if runtime["running"]:
        raise HTTPException(409, "脚本仍在运行，请等旅程结束后再读取旅程信息")
    records = JourneyRecordStore(_runtime_control.root)
    try:
        record = records.read(journey_id)
        if record is None:
            raise FileNotFoundError("旅程记录不存在")
        if not record["outcome"]["journey_ended"]:
            raise ValueError("旅程尚未结束，请结束后再读取旅程信息")
        game = store.config.get("game", {}) if isinstance(store.config, dict) else {}
        scan = JourneyInfoScanner(_runtime_control.root,
                                  window_title=game.get("window_title", "StarSavior")).scan()
        updated = records.apply_screen_scan(journey_id, scan)
        missing = list(updated.get("data_quality", {}).get("missing_fields", []))
        return {"record": updated, "scan": scan, "missing_fields": missing}
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except JourneyInfoScanError as error:
        raise HTTPException(409, str(error)) from error
    except (ValueError, OSError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


def _quick_lookup_store():
    return QuickLookupStore(_runtime_control.root)


@app.get("/api/quick-lookup")
def quick_lookup_state():
    """Return the latest explicit rescuer/Arcana scans and their mapping."""
    return _quick_lookup_store().read()


@app.post("/api/quick-lookup/scan-rescuer")
def scan_quick_rescuer():
    """Read the visible rescuer page and OCR-confirmed right-side rows.

    The user opens the page and keeps the game foreground.  The scanner may
    click only the OCR-confirmed visible rows; it never scrolls, refocuses, or
    changes a running journey.
    """
    runtime = _automation_status()
    if runtime["running"]:
        raise HTTPException(409, "脚本仍在运行，请停止后再做数据快查")
    game = store.config.get("game", {}) if isinstance(store.config, dict) else {}
    try:
        scan = QuickLookupScanner(_runtime_control.root,
                                  window_title=game.get("window_title", "StarSavior")).scan("rescuer")
        return _quick_lookup_store().save_scan("rescuer", scan, scan.get("screenshot"))
    except (ValueError, OSError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/quick-lookup/scan-arcanum")
def scan_quick_arcanum():
    """Read the currently visible Arcana/support-card page without input."""
    runtime = _automation_status()
    if runtime["running"]:
        raise HTTPException(409, "脚本仍在运行，请停止后再做数据快查")
    game = store.config.get("game", {}) if isinstance(store.config, dict) else {}
    try:
        scan = QuickLookupScanner(_runtime_control.root,
                                  window_title=game.get("window_title", "StarSavior")).scan("arcanum",
                                                                                          traverse_rescuer=False)
        return _quick_lookup_store().save_scan("arcanum", scan, scan.get("screenshot"))
    except (ValueError, OSError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/quick-lookup/match")
def match_quick_lookup():
    state = _quick_lookup_store().read()
    if not state.get("rescuer") and not state.get("arcanum"):
        raise HTTPException(409, "请先读取救援者或阿尔克那界面")
    match = build_match(state.get("rescuer"), state.get("arcanum"), screenshots={
        key: (state.get(key) or {}).get("screenshot") for key in ("rescuer", "arcanum")
    })
    _quick_lookup_store().save_match(match)
    return {**state, "match": match}


@app.put("/api/journey-target")
def save_journey_target(body: dict[str, Any]):
    with _automation_lock:
        runtime = _automation_status()
        if runtime["running"] and runtime["state"] not in {"paused_safe", "paused_manual"}:
            raise HTTPException(409, "请先暂停脚本，再更改本轮目标")
        try:
            _target_store().save(body, runtime["status"].get("run_id") if runtime["running"] else None)
        except JourneyTargetPending as error:
            raise HTTPException(400, str(error))
        return journey_target()


def _require_start_target(runtime):
    try:
        target = _target_store().read()
        checked = precheck_target(target or {})
    except JourneyTargetPending as error:
        raise HTTPException(409, str(error))
    if not target or not checked["ready"]:
        raise HTTPException(409, "请先完成旅程目标预检查并保存")
    scope = target.get("scope_run_id")
    previous = runtime.get("checkpoint") or {}
    if scope and (scope != previous.get("run_id") or previous.get("resumable") is False):
        raise HTTPException(409, "目标属于已结束或其他旅程，请重新保存本轮目标")


@app.post("/api/runtime/start")
def start_runtime() -> dict[str, Any]:
    global _automation_process, _automation_log_handle
    with _automation_lock:
        if _automation_status()["running"]:
            return _automation_status()
        _require_start_target(_automation_status())
        if _automation_log_handle is not None:
            _automation_log_handle.close()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        log_path = LOG_DIR / f"ui_automation_{stamp}.log"
        _automation_log_handle = log_path.open("a", encoding="utf-8", errors="replace")
        try:
            _automation_process = subprocess.Popen(
                [sys.executable, "-u", "main.py", "--auto", "--resume"], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=_automation_log_handle,
                stderr=subprocess.STDOUT, text=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except Exception:
            _automation_log_handle.close()
            _automation_log_handle = None
            raise
        _automation_process._starsavior_log_path = str(log_path.relative_to(ROOT))
        return _automation_status()


def _issue_runtime_command(command: str, payload=None):
    runtime = _automation_status()
    if not runtime["running"] or runtime["state"] in {"starting", "process_exit"}:
        raise HTTPException(409, "当前没有可控制的运行实例")
    status = runtime["status"]
    if command in {"resume", "step"} and status.get("state") not in {"paused_manual", "paused_safe"}:
        raise HTTPException(409, "继续和单步只能在暂停状态使用")
    if command in {"resume", "step"} and _target_store().read() and "journey_target" not in status:
        raise HTTPException(409, "当前脚本尚未加载目标功能，请使用重启恢复加载新版本")
    return _runtime_control.issue(command, **(payload or {}), run_id=status["run_id"])


@app.post("/api/runtime/stop")
def stop_runtime() -> dict[str, Any]:
    if not _automation_status()["running"]:
        return _automation_status()
    command = _issue_runtime_command("stop")
    return {**_automation_status(), "command": command}


@app.post("/api/runtime/restart")
def restart_runtime() -> dict[str, Any]:
    # Let the script unwind and save its checkpoint. Never start a second owner.
    with _automation_lock:
        status = _automation_status()
        _require_start_target(status)
        if status["running"]:
            if status["state"] == "starting":
                raise HTTPException(409, "脚本仍在启动，请稍后重试")
            _issue_runtime_command("stop")
            import time
            deadline = time.monotonic() + 8
            while _automation_status()["running"] and time.monotonic() < deadline:
                time.sleep(0.1)
            if _automation_status()["running"]:
                raise HTTPException(409, "脚本仍在收束当前操作，尚未重启；等待停止后再启动")
        return start_runtime()


@app.post("/api/runtime/pause")
def pause_runtime() -> dict[str, Any]:
    command = _issue_runtime_command("pause")
    return {**_automation_status(), "command": command}


@app.post("/api/runtime/resume")
def resume_runtime() -> dict[str, Any]:
    command = _issue_runtime_command("resume")
    return {**_automation_status(), "command": command}


@app.post("/api/runtime/step")
def step_runtime() -> dict[str, Any]:
    command = _issue_runtime_command("step")
    return {**_automation_status(), "command": command}


@app.get("/api/runtime/diagnostics")
def runtime_diagnostics() -> dict[str, Any]:
    runtime = _automation_status()
    status = runtime["status"]
    snapshot = status.get("screenshot")
    current = store.resolve_screenshot(snapshot)
    return {**runtime, "safe_for_ai": runtime["running"] and runtime["state"] == "paused_safe",
            "screenshot_url": f"/api/screenshot?path={quote(str(snapshot), safe='')}" if current else None}


@app.post("/api/runtime/ai-claim")
def runtime_ai_claim(body: dict[str, Any]):
    try:
        return _runtime_control.claim_ai(body.get("client_id", ""), body.get("run_id"), body.get("pause_id"))
    except ValueError as error:
        raise HTTPException(409, str(error))


@app.post("/api/runtime/ai-action")
def runtime_ai_action(body: dict[str, Any]):
    runtime = _automation_status()
    status = runtime["status"]
    if not runtime["running"] or runtime["state"] != "paused_safe":
        raise HTTPException(409, "AI 接管只允许在脚本安全暂停后使用")
    if not _runtime_control.valid_ai(body, status["run_id"], status.get("pause_id")):
        raise HTTPException(409, "AI 客户端未接入、会话过期或本次暂停已变化")
    kind = body.get("kind")
    if kind not in {"inspect", "click", "resume", "stop"}:
        raise HTTPException(400, "支持的 AI 操作为 inspect/click/resume/stop")
    if kind == "resume" and _target_store().read() and "journey_target" not in status:
        raise HTTPException(409, "当前脚本尚未加载目标功能，需要人工重启恢复加载新版本")
    if not isinstance(body.get("action_id"), str) or not body["action_id"].strip():
        raise HTTPException(400, "需要唯一 action_id")
    if kind == "click":
        import math
        try:
            x, y = float(body["x"]), float(body["y"])
        except (ValueError, TypeError, KeyError):
            raise HTTPException(400, "click 需要数值 x/y")
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            raise HTTPException(400, "x/y 必须在 0..1")
        if body.get("expected_frame_id") != status.get("frame_id"):
            raise HTTPException(409, "画面已更新，需要读取最新诊断")
    # Copy only protocol fields. The request cannot replace command ids or kinds.
    fields = ("kind", "run_id", "pause_id", "lease_id", "action_id", "expected_frame_id", "x", "y")
    payload = {key: body[key] for key in fields if key in body}
    command = _runtime_control.issue("ai_action", **payload)
    return {"queued": True, "command_id": command["id"]}


@app.get("/api/runtime/commands/{command_id}")
def command_result(command_id: str):
    if len(command_id) != 32 or any(c not in "0123456789abcdef" for c in command_id):
        raise HTTPException(400, "无效命令 ID")
    return _runtime_control._read(_runtime_control.directory / "acks" / f"{command_id}.json") or {
        "id": command_id, "pending": True}


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


@app.post("/api/events")
def create_event(body: dict[str, Any]) -> dict[str, Any]:
    """Create a local rule from the event editor.

    Manually entered events are deliberately marked as local rules.  They can
    be used in development mode after their option text and effects are
    filled in, but are never presented as document-verified data.
    """
    store.reload()
    event_name = str(body.get("event_name") or "").strip()
    if not event_name:
        raise HTTPException(400, "事件名不能为空")
    raw_options = body.get("options", [])
    if not isinstance(raw_options, list):
        raise HTTPException(400, "options 必须是数组")
    options: list[dict[str, Any]] = []
    for position, raw_option in enumerate(raw_options, start=1):
        if not isinstance(raw_option, dict):
            raise HTTPException(400, f"分支{position}格式无效")
        try:
            option_index = int(raw_option.get("index", position))
        except (TypeError, ValueError):
            raise HTTPException(400, f"分支{position}序号必须是整数")
        if option_index < 1:
            raise HTTPException(400, f"分支{position}序号必须是正整数")
        aliases = raw_option.get("alias", [])
        if isinstance(aliases, str):
            aliases = [item.strip() for item in aliases.split(",") if item.strip()]
        if not isinstance(aliases, list):
            raise HTTPException(400, f"分支{position}别名格式无效")
        options.append({
            "index": option_index,
            "keyword": str(raw_option.get("keyword") or "").strip(),
            "alias": [str(item).strip() for item in aliases if str(item).strip()],
            "effect_text": str(raw_option.get("effect_text") or "").strip(),
        })
    try:
        recommended = int(body.get("recommended_option", 1))
    except (TypeError, ValueError):
        raise HTTPException(400, "默认分支必须是整数")
    if recommended < 1 or (options and recommended > len(options)):
        raise HTTPException(400, "默认分支超出当前分支数量")
    event = {
        "id": f"manual_{uuid.uuid4().hex[:12]}",
        "event_name": event_name,
        "title_aliases": [event_name],
        "category": str(body.get("category") or "手动事件").strip(),
        "source_sheet": "手动维护",
        "source_type": "local_rule",
        "document_verified": False,
        "status": "manual",
        "recommended_option": recommended,
        "note": "手动维护事件；效果需要自行填写并在开发模式验证。",
        "options": options,
    }
    store.events.append(event)
    store.save_events()
    return {"ok": True, "index": len(store.events) - 1, "event": event}


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
        raise HTTPException(400, "默认分支必须是整数")
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
