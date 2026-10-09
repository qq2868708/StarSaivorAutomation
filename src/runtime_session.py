"""Cooperative runtime state machine. Recovery and normal decisions belong to the script."""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .runtime_control import RuntimeControl, RuntimeInterrupt
from .journey_target import JourneyTargetStore, JourneyTargetPending, precheck_target
from .event_exploration import EventExplorationStore, aggregate_effects, new_attempt
from .journey_record import now_iso


class SafePauseError(RuntimeError):
    pass


class RuntimeSession:
    """Trainer mixin; no AI calls, polling services, or model dependency."""

    def _runtime_init(self, root=None):
        self.runtime = RuntimeControl(root)
        self._run_id = self.logger._run_id
        self._runtime_active = False
        self._paused = False
        self._safe_paused = False
        self._stop_requested = False
        self._step_once = False
        self._safe_pause_reason = ""
        self._pause_id = None
        self._frame_id = None
        self._last_runtime_frame = None
        self._last_frame_screenshot = None
        self._last_frame_fingerprint = ""
        self._last_input = None
        self._last_logged_action = None
        self._ai_action_ids = []
        self._resume_checkpoint = None
        self._recovering = True
        self._active_handler_name = ""
        self._last_handler_name = ""
        self._same_handler_count = 0
        self._stagnant_frame = None
        self._stagnant_since = time.monotonic()
        self._runtime_started = time.monotonic()
        self._round_deadline = 0
        self._paused_at = None
        self._diagnostics = {}
        self._journey_target_store = JourneyTargetStore(self.runtime.root)
        self._journey_target = None
        self._journey_target_state = {}
        self._event_exploration = EventExplorationStore(self.runtime.root)
        self._exploration_attempt = None
        self._journey_id = uuid.uuid4().hex
        self._journey_started_at = datetime.now().astimezone().isoformat()
        self._journey_entry_point = "mid_journey"
        self.logger.bind_journey(self._journey_id, self._journey_started_at)
        self._handler_ctx.journey_target = None
        self._handler_ctx.target_selection_phase = self._target_selection_phase
        self._handler_ctx.exploration_prepare = self._exploration_prepare
        self._handler_ctx.exploration_finish = self._exploration_finish
        self._config_hash = self._configuration_hash()
        self._record_strategy()
        self.logger.set_checkpoint_callback(self._on_logged_action)

    def _exploration_prepare(self, marker, rows, direction):
        before = self.logger._observations[-1] if self.logger._observations else {}
        selection = self._event_exploration.select(
            # Keep empirical knowledge reusable across runs of the same target.
            # The attempt itself still carries run_id/journey_id for provenance;
            # putting journey_id in the key would reset exploration every run.
            marker, rows, {'target_revision': self._target_revision()}, direction)
        self._exploration_attempt = new_attempt(
            selection, marker, rows, before, self._run_id, self._journey_id,
            self._target_revision())
        return selection

    def _target_revision(self):
        return (self._journey_target or {}).get('revision')

    def _exploration_finish(self, screenshot):
        attempt = self._exploration_attempt
        if not attempt:
            return
        from .handlers import FrameContext
        try:
            items = FrameContext(screenshot).recognize_full(self.ui.ocr)
            after = self.logger.observe_frame(items, '事件后续', self.round_count)
            before_values = (attempt.get('before') or {}).get('values', {})
            changes = {}
            for field, value in after.get('values', {}).items():
                old = before_values.get(field)
                if type(value) is int and type(old) is int:
                    changes[field] = value - old
            attempt['after'] = after
            attempt['effects'] = aggregate_effects([after])
            # A branch that only grants a buff/item or advances to a page has
            # still been observed even when OCR contains no numeric receipt.
            # Keep that fact so the bounded sampler can converge and continue.
            attempt['effects']['observed'] = True
            attempt['observation_quality'] = (
                'numeric' if attempt['effects'].get('numeric') else
                'qualitative' if attempt['effects'].get('qualitative') else
                'transition_only')
            if changes:
                attempt['effects']['numeric'].update(changes)
            attempt['phase'] = 'observed'
            attempt['finished_at'] = now_iso()
            self._event_exploration.put_attempt(attempt)
            self.logger.log_exploration(attempt)
        finally:
            self._exploration_attempt = None

    def _record_strategy(self):
        self.logger.configure_run(self._config_hash, self.config,
                                  self.engine.rule_profile if self.engine.is_rule_engine_mode else None)

    def _configuration_hash(self):
        return hashlib.sha256(json.dumps(self.config, ensure_ascii=False,
                                        sort_keys=True).encode()).hexdigest()[:20]

    @staticmethod
    def _fingerprint_frame(frame):
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        return hashlib.sha256(small.tobytes()).hexdigest()[:20]

    @staticmethod
    def _frame_difference(a, b):
        if a is None or b is None or a.shape != b.shape:
            return 255.0
        a = cv2.resize(a, (64, 36), interpolation=cv2.INTER_AREA).astype(float)
        b = cv2.resize(b, (64, 36), interpolation=cv2.INTER_AREA).astype(float)
        return float(np.abs(a - b).mean())

    def _runtime_status(self, state, reason="", **extra):
        return self.runtime.write_status(
            state, reason, run_id=self._run_id, round_count=self.round_count,
            pause_id=self._pause_id, frame_id=self._frame_id,
            last_handler=self._active_handler_name,
            journey_id=self._journey_id, journey_started_at=self._journey_started_at,
            screenshot=self._last_frame_screenshot, diagnostics=self._diagnostics,
            journey_target=self._journey_target, journey_target_state=self._journey_target_state, **extra)

    def _event(self, event, **data):
        self.runtime.emit(self._run_id, event, round_count=self.round_count, journey_id=self._journey_id, **data)

    def _write_checkpoint(self, phase, screenshot=None, reason="", **extra):
        if isinstance(screenshot, str):
            self._last_frame_screenshot = screenshot
        elif screenshot is not None:
            self._last_frame_fingerprint = self._fingerprint_frame(screenshot)
        handler_state = {}
        for handler in self._handlers.handlers:
            if hasattr(handler, "_trade_handled"):
                handler_state[type(handler).__name__] = {"trade_handled": handler._trade_handled}
        return self.runtime.write_checkpoint(
            run_id=self._run_id, round_count=self.round_count,
            journey_id=self._journey_id, journey_started_at=self._journey_started_at,
            journey_entry_point=self._journey_entry_point,
            detected_turn=self._handler_ctx.detected_turn, phase=phase,
            resumable=phase != "journey_end", config_hash=self._config_hash,
            screen_fingerprint=self._last_frame_fingerprint,
            screenshot=self._last_frame_screenshot, last_handler=self._active_handler_name,
            last_action=self._last_logged_action, last_input=self._last_input,
            handler_state=handler_state, ai_action_ids=self._ai_action_ids,
            journey_target=self._journey_target, journey_target_state=self._journey_target_state,
            safe_pause_reason=reason or self._safe_pause_reason, **extra)

    def _refresh_journey_target(self):
        target = self._journey_target_store.read()
        if target is None:
            if self._journey_target:
                raise JourneyTargetPending("已生效的目标文件被移除，请在暂停时重新保存")
            return
        if target.get("revision") == (self._journey_target or {}).get("revision"):
            if self._journey_target_state.get("pending"):
                raise JourneyTargetPending("上次装备输入结果不确定；请核对现有材料后重新保存目标")
            return
        checked = precheck_target(target)
        if not checked["ready"] or not target.get("revision"):
            raise JourneyTargetPending("目标预检查未通过：" + "；".join(checked["errors"]))
        # Older goals may contain manual direction overrides. The combo is authoritative.
        target = {**target, **checked["target"]}
        target.pop("initial_material", None)
        target.pop("direction_confirmed", None)
        previous = self._resume_checkpoint or {}
        scope = target.get("scope_run_id")
        if scope not in {None, self._run_id, previous.get("run_id")}:
            raise JourneyTargetPending("目标属于另一轮旅程，请重新保存本轮目标")
        if scope != self._run_id:
            target = self._journey_target_store.bind(target, self._run_id)
        self._journey_target = target
        state = previous.get("journey_target_state", {})
        if (previous.get("journey_target") or {}).get("revision") != target["revision"]:
            state = {"revision": target["revision"], "has_partner": target["has_partner"], "pending": None}
        self._journey_target_state = dict(state)
        self._handler_ctx.journey_target = {**target, "has_partner": bool(state.get("has_partner"))}
        direction = target["training_direction"]
        from .rule_engine import BuildDirection
        self.engine.rule_profile.legacy_strategy.build_direction = BuildDirection(direction)
        self.engine.priority = 1 if direction == "attack" else 2
        self.logger.set_target(target)
        self._record_strategy()
        self._event("journey_target_applied", target=target, progress=state)
        if state.get("pending"):
            raise JourneyTargetPending("重启前的装备输入结果不确定；请核对现有材料后重新保存目标")

    def _target_selection_phase(self, phase, choice):
        if phase == "prepared":
            self._journey_target_state["pending"] = choice
        elif phase == "confirmed":
            self._journey_target_state.update(has_partner=True, pending=None)
            self._handler_ctx.journey_target["has_partner"] = True
        self.logger.log_target_progress(phase, choice)
        self._write_checkpoint("target_selection_" + phase)
        self._event("target_selection_" + phase, choice=choice,
                    target_revision=(self._journey_target or {}).get("revision"))

    def _on_logged_action(self, action):
        self._last_logged_action = dict(action)
        if self._runtime_active:
            self._event("decision" if action.get("type") == "decision" else "action_logged",
                        action=action)
            self._write_checkpoint("action_logged")

    def _load_resume_checkpoint(self):
        checkpoint = self.runtime.read_checkpoint()
        if not checkpoint or checkpoint.get("resumable") is False or checkpoint.get("phase") == "journey_end":
            return
        self._resume_checkpoint = checkpoint
        self._journey_id = checkpoint.get("journey_id") or self._journey_id
        self._journey_started_at = checkpoint.get("journey_started_at") or self._journey_started_at
        self._journey_entry_point = checkpoint.get("journey_entry_point") or "mid_journey"
        self.logger.bind_journey(self._journey_id, self._journey_started_at,
                                 checkpoint.get("run_id"), self._journey_entry_point)
        self._last_input = checkpoint.get("last_input")
        self._ai_action_ids = list(checkpoint.get("ai_action_ids") or [])
        # Never restore a pending click, cached frame, or guessed game turn.
        # The game screen is the authority, even when the user manually progressed.
        self._event("recovery_requested", previous_run=checkpoint.get("run_id"),
                    previous_phase=checkpoint.get("phase"))

    def _runtime_poll(self):
        """Called before capture/input; never blocks inside an old handler stack."""
        if not self._runtime_active or self._paused:
            return
        command = self.runtime.consume()
        if command:
            self._process_command(command)
        if self._paused or self._stop_requested:
            raise RuntimeInterrupt("stop" if self._stop_requested else "pause")
        now = time.monotonic()
        if now - self._runtime_started >= self._max_runtime_seconds:
            raise RuntimeInterrupt("runtime_timeout")
        if self._round_deadline and now >= self._round_deadline:
            raise RuntimeInterrupt("round_timeout")

    def _before_input(self, action):
        self._runtime_poll()
        if self._paused and not action.get("source") == "ai":
            raise RuntimeInterrupt("pause")
        self._last_input = {**action, "action_id": uuid.uuid4().hex, "phase": "prepared",
                            "handler": self._active_handler_name, "ts": datetime.now().isoformat()}
        # Persistence failure aborts before sending input.
        try:
            self.logger.log_input(self._last_input)
            self.logger.save("running", announce=False, close_turn=False)
            if self._last_runtime_frame is not None:
                self._last_frame_screenshot = self._write_runtime_diagnostic_screenshot(self._last_runtime_frame)
            self._write_checkpoint("before_input")
            self._event("before_input", action=self._last_input)
        except Exception as error:
            raise RuntimeInterrupt("checkpoint_error") from error

    def _after_input(self, action, error=None):
        self._last_input = {**(self._last_input or action),
                            "phase": "uncertain" if error else "sent", "error": str(error) if error else None}
        try:
            self.logger.log_input(self._last_input)
            self.logger.save("running", announce=False, close_turn=False)
            self._write_checkpoint("input_uncertain" if error else "after_input")
            self._event("input_error" if error else "after_input", action=self._last_input)
        except Exception as exc:
            raise RuntimeInterrupt("checkpoint_error") from exc

    def _process_command(self, command):
        if command.get("run_id") != self._run_id:
            self.runtime.acknowledge(command, False, "命令属于旧运行实例")
            return
        kind = command["command"]
        if kind == "ai_action":
            self._handle_ai_action(command)
            return
        if kind == "pause":
            if not self._paused:
                self._paused = True
                self._safe_paused = False
                self._paused_at = time.monotonic()
                self._safe_pause_reason = "manual_pause"
                self._runtime_status("paused_manual", "manual_pause")
        elif kind in {"resume", "step"}:
            if not self._paused:
                self.runtime.acknowledge(command, False, "脚本当前未暂停")
                return
            self._step_once = kind == "step"
            self._set_running_status()
        elif kind == "stop":
            self._stop_requested = True
            self._exit_reason = "runtime_stop"
        self.runtime.acknowledge(command, True)

    def _set_running_status(self):
        paused_duration = time.monotonic() - self._paused_at if self._paused_at else 0
        self._runtime_started += paused_duration
        if self._safe_pause_reason == "runtime_timeout":
            self._runtime_started = time.monotonic()
        self._paused = self._safe_paused = False
        self._safe_pause_reason = ""
        self._pause_id = None
        self._paused_at = None
        self._consecutive_unknown = self._same_handler_count = 0
        self._handler_ctx.consecutive_unknown = 0
        self._handler_ctx.pending_action = ""
        self._handler_ctx.pending_screenshot = None
        self._handler_ctx.turn_detected = False
        self._round_deadline = 0
        self._recovering = True
        self._stagnant_frame = None
        self._diagnostics = {}
        for handler in self._handlers.handlers:
            if hasattr(handler, "_last_failed_text"):
                handler._last_failed_text = ""
        self._runtime_status("recovering")
        self._event("resume")

    def _wait_for_runtime(self):
        if not self._paused:
            command = self.runtime.consume()
            if command:
                self._process_command(command)
        while self._paused and not self._stop_requested:
            command = self.runtime.wait_for_command(0.25)
            if command:
                try:
                    self._process_command(command)
                except Exception as error:
                    self.runtime.acknowledge(command, False, str(error))
                    self._runtime_status("paused_safe" if self._safe_paused else "paused_manual",
                                         self._safe_pause_reason, error=str(error))
        return self._stop_requested

    def _enter_safe_pause(self, reason, screenshot_path=None, error=""):
        self._paused = self._safe_paused = True
        self._step_once = False
        self._paused_at = time.monotonic()
        self._pause_id = uuid.uuid4().hex
        self._safe_pause_reason = reason
        self._round_deadline = 0
        self._handler_ctx.pending_screenshot = None
        self._handler_ctx.pending_action = ""
        self._event("safe_pause", reason=reason, error=error)
        # Capture only after the handler has unwound and no input remains held.
        self._collect_diagnostics(error=error)
        self._write_checkpoint("safe_paused", reason=reason)
        try:
            self.logger.save("safe_paused")
        except (OSError, ValueError) as record_error:
            self._diagnostics["recording_error"] = str(record_error)
            self._event("recording_error", error=str(record_error))
        self._runtime_status("paused_safe", reason, error=error)
        print(f"[安全暂停] {reason}: {error}。等待人工或已接入的 AI；不继续发送输入。")

    def _write_runtime_diagnostic_screenshot(self, screenshot):
        path = self.runtime.directory / "screenshots" / f"{uuid.uuid4().hex}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), screenshot):
            raise OSError("无法保存运行现场截图")
        return path.relative_to(self.runtime.root).as_posix()

    def _collect_diagnostics(self, error=""):
        frame = None
        capture_error = ""
        try:
            frame = self.capture.capture_game()
        except Exception as exc:
            capture_error = str(exc)
        fresh = frame is not None
        if frame is None:
            frame = self._last_runtime_frame
        ocr = []
        ocr_error = ""
        if frame is not None:
            self._last_runtime_frame = frame
            self._last_frame_fingerprint = self._fingerprint_frame(frame)
            self._last_frame_screenshot = self._write_runtime_diagnostic_screenshot(frame)
            try:
                for item in self.ui.ocr.recognize(frame):
                    ocr.append({"text": str(item[0]), "confidence": float(item[1])})
            except Exception as exc:
                ocr_error = str(exc)
        self._frame_id = uuid.uuid4().hex
        self._diagnostics = {"ocr": ocr, "error": error, "capture_error": capture_error,
                             "ocr_error": ocr_error, "fresh_capture": fresh,
                             "screen_fingerprint": self._last_frame_fingerprint,
                             "last_input": self._last_input,
                             "handler": self._active_handler_name}
        self._event("diagnostics", frame_id=self._frame_id,
                    screenshot=self._last_frame_screenshot, **self._diagnostics)

    def _handle_ai_action(self, command):
        reject = lambda reason: self.runtime.acknowledge(command, False, reason)
        if not self._paused or not self._safe_paused:
            return reject("AI 操作只允许在安全暂停时执行")
        if not self.runtime.valid_ai(command, self._run_id, self._pause_id):
            return reject("AI 客户端未接入、会话过期或暂停已变化")
        action_id = str(command.get("action_id") or "")
        if not action_id or action_id in self._ai_action_ids:
            return reject("缺少 action_id 或动作已经处理")
        kind = command.get("kind")
        if kind == "click":
            try:
                x, y = float(command["x"]), float(command["y"])
            except (ValueError, TypeError, KeyError):
                return reject("无效点击坐标")
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
                return reject("坐标必须在 0..1")
            if command.get("expected_frame_id") != self._frame_id:
                return reject("诊断画面已更新，请重新读取")
            current = self.capture.capture_game()
            if self._frame_difference(current, self._last_runtime_frame) > 18:
                self._collect_diagnostics()
                self._runtime_status("paused_safe", self._safe_pause_reason)
                return reject("游戏画面已改变，未执行点击")
        elif kind not in {"inspect", "resume", "stop"}:
            return reject("不支持的 AI 动作")
        # Persist the id before any side effect so retries cannot double click.
        self._ai_action_ids.append(action_id)
        self._write_checkpoint("before_ai_action")
        try:
            if kind == "inspect":
                self._collect_diagnostics()
            elif kind == "click":
                self.controller.click_at_percent(x, y, source="ai")
                self._collect_diagnostics()
            elif kind == "resume":
                self._set_running_status()
            elif kind == "stop":
                self._stop_requested = True
                self._exit_reason = "ai_stop"
            if self._paused:
                self._write_checkpoint("safe_paused")
                self._runtime_status("paused_safe", self._safe_pause_reason)
            self.runtime.acknowledge(command, True)
            self._event("ai_action", kind=kind, action_id=action_id)
        except Exception as error:
            self._collect_diagnostics(error=str(error))
            self._runtime_status("paused_safe", self._safe_pause_reason, error=str(error))
            reject(str(error))

    def _recover_current_screen(self):
        """Rebind the window and rebuild script state; never replay the last action."""
        region = self.capture.find_window()
        if region is None:
            raise SafePauseError("找不到游戏窗口")
        self.controller.set_game_region(region)
        self.controller.set_window_handle(self.capture.get_hwnd())
        self.controller.focus_game_window()
        self._handler_ctx.pending_screenshot = None
        self._handler_ctx.pending_action = ""
        self._handler_ctx.turn_detected = False
        self._handler_ctx.detected_turn = None
        frame = self.capture.capture_game()
        if frame is None:
            raise SafePauseError("恢复时截图失败")
        self._last_runtime_frame = frame
        self._last_frame_fingerprint = self._fingerprint_frame(frame)
        checkpoint = self._resume_checkpoint or {}
        # Shopping phase is only reused for an unchanged screen and configuration.
        same = checkpoint.get("screen_fingerprint") == self._last_frame_fingerprint
        if same and checkpoint.get("config_hash") == self._config_hash:
            for handler in self._handlers.handlers:
                state = checkpoint.get("handler_state", {}).get(type(handler).__name__, {})
                if "trade_handled" in state and hasattr(handler, "_trade_handled"):
                    handler._trade_handled = bool(state["trade_handled"])
        self._resume_checkpoint = None
        self._handler_ctx.pending_screenshot = frame
        self._recovering = False
        self._write_checkpoint("recovered", screenshot=frame, previous_frame_matches=same)
        self._runtime_status("running")
        self._event("recovered", previous_frame_matches=same)
