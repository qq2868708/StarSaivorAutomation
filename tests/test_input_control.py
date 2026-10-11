"""Validate cancellation and input release using mocked Win32 calls."""
import unittest
from unittest.mock import Mock, patch

from src.controller import Controller, InputTargetError, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
from src.capture import is_window_foreground
from src.runtime_control import RuntimeInterrupt


class InputControlTests(unittest.TestCase):
    @patch("src.capture.user32.IsWindow", return_value=True)
    @patch("src.capture.user32.GetWindowThreadProcessId")
    @patch("src.capture.user32.GetAncestor", return_value=0)
    @patch("src.capture.user32.GetForegroundWindow", return_value=22)
    def test_foreground_accepts_same_game_process_child_window(
            self, _foreground, _ancestor, get_process_id, _is_window):
        def process_id(hwnd, pointer):
            pointer._obj.value = 700 if hwnd == 22 else 700
            return 1

        get_process_id.side_effect = process_id
        self.assertTrue(is_window_foreground(11))

    @patch("src.capture.user32.IsWindow", return_value=True)
    @patch("src.capture.user32.GetWindowThreadProcessId")
    @patch("src.capture.user32.GetAncestor", return_value=0)
    @patch("src.capture.user32.GetForegroundWindow", return_value=22)
    def test_foreground_rejects_unrelated_process(
            self, _foreground, _ancestor, get_process_id, _is_window):
        def process_id(hwnd, pointer):
            pointer._obj.value = 701 if hwnd == 22 else 700
            return 1

        get_process_id.side_effect = process_id
        self.assertFalse(is_window_foreground(11))

    def controller(self):
        controller = Controller()
        controller._hwnd = 1
        controller._check_input_target = Mock()
        controller._percent_to_screen = Mock(return_value=(20, 30))
        controller._send_move = Mock()
        controller._send_mouse_input = Mock()
        controller.after_action = Mock()
        return controller

    def test_each_input_refocuses_before_sending(self):
        for kind in ("click", "key", "scroll"):
            with self.subTest(kind=kind):
                events = []
                foreground = [False]
                controller = Controller(click_delay=0, auto_refocus=True)
                controller.set_window_handle(1)
                controller._percent_to_screen = Mock(return_value=(20, 30))
                controller.wait = Mock()

                def focus():
                    events.append("focus")
                    foreground[0] = True

                controller.focus_game_window = Mock(side_effect=focus)
                with patch("src.controller.user32.IsWindow", return_value=True), \
                     patch("src.controller.is_window_foreground", side_effect=lambda _: foreground[0]), \
                     patch("src.controller.user32.SendInput",
                           side_effect=lambda *args: (events.append("input"), 1)[1]), \
                     patch("src.controller.time.sleep"):
                    if kind == "click":
                        controller.click_at_percent(.5, .5)
                    elif kind == "key":
                        controller.send_escape()
                    else:
                        controller.scroll_at_percent(.5, .5)
                self.assertEqual(events[0], "focus")
                self.assertIn("input", events)
                controller.focus_game_window.assert_called_once()

    def test_failed_foreground_confirmation_blocks_all_inputs(self):
        for kind in ("click", "key", "scroll"):
            with self.subTest(kind=kind):
                controller = Controller(auto_refocus=True)
                controller.set_window_handle(1)
                controller.focus_game_window = Mock()
                with patch("src.controller.user32.IsWindow", return_value=True), \
                     patch("src.controller.is_window_foreground", return_value=False), \
                     patch("src.controller.user32.SendInput") as send:
                    with self.assertRaises(InputTargetError):
                        if kind == "click":
                            controller.click_at_percent(.5, .5)
                        elif kind == "key":
                            controller.send_escape()
                        else:
                            controller.scroll_at_percent(.5, .5)
                    send.assert_not_called()
                controller.focus_game_window.assert_called_once()

    def test_confirmed_game_foreground_needs_no_refocus(self):
        controller = Controller(auto_refocus=True)
        controller.set_window_handle(1)
        controller.focus_game_window = Mock()
        with patch("src.controller.user32.IsWindow", return_value=True), \
             patch("src.controller.is_window_foreground", return_value=True):
            controller._check_input_target()
        controller.focus_game_window.assert_not_called()

    def test_opt_out_keeps_scan_foreground_guard(self):
        controller = Controller(auto_refocus=False)
        controller.set_window_handle(1)
        controller.focus_game_window = Mock()
        with patch("src.controller.user32.IsWindow", return_value=True), \
             patch("src.controller.is_window_foreground", return_value=False):
            with self.assertRaises(InputTargetError):
                controller._check_input_target()
        controller.focus_game_window.assert_not_called()

    def test_closed_game_does_not_trigger_refocus(self):
        controller = Controller(auto_refocus=True)
        controller.set_window_handle(1)
        controller.focus_game_window = Mock()
        with patch("src.controller.user32.IsWindow", return_value=False):
            with self.assertRaises(InputTargetError):
                controller._check_input_target()
        controller.focus_game_window.assert_not_called()

    def test_pending_pause_prevents_mouse_down(self):
        controller = self.controller()
        controller.before_action = Mock(side_effect=RuntimeInterrupt("pause"))
        controller._send_mouse_event = Mock()
        with self.assertRaises(RuntimeInterrupt):
            controller.click_at_percent(.5, .5)
        controller._send_mouse_event.assert_not_called()
        controller.after_action.assert_not_called()

    def test_failed_click_attempt_releases_button_and_marks_uncertain(self):
        controller = self.controller()
        controller._send_mouse_event = Mock(side_effect=[None, InputTargetError("failed release"), None])
        with patch("src.controller.time.sleep"), self.assertRaises(InputTargetError):
            controller.click_at_percent(.5, .5)
        self.assertEqual([call.args[0] for call in controller._send_mouse_event.call_args_list],
                         [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_LEFTUP])
        self.assertIsInstance(controller.after_action.call_args.args[1], InputTargetError)
        self.assertFalse(controller._send_mouse_input.call_args.kwargs["check_target"])

    def test_keyboard_release_runs_even_if_interrupted_after_press(self):
        controller = self.controller()
        controller._send_key = Mock()
        with patch("src.controller.time.sleep", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            controller.send_escape()
        self.assertEqual([call.kwargs["key_up"] for call in controller._send_key.call_args_list], [False, True])

    def test_failed_release_retry_still_restores_cursor_and_checkpoints(self):
        controller = self.controller()
        first_error = InputTargetError("first release failed")
        controller._send_mouse_event = Mock(side_effect=[None, first_error, InputTargetError("retry failed")])
        with patch("src.controller.time.sleep"), self.assertRaises(InputTargetError) as caught:
            controller.click_at_percent(.5, .5)
        self.assertIs(caught.exception, first_error)
        controller._send_mouse_input.assert_called_once()
        self.assertIs(controller.after_action.call_args.args[1], first_error)

    def test_cursor_restore_failure_is_recorded_as_uncertain(self):
        controller = self.controller()
        controller._send_mouse_event = Mock()
        error = InputTargetError("restore failed")
        controller._send_mouse_input.side_effect = error
        with patch("src.controller.time.sleep"), self.assertRaises(InputTargetError):
            controller.click_at_percent(.5, .5)
        self.assertIs(controller.after_action.call_args.args[1], error)

    def test_keyboard_release_failure_is_checkpointed(self):
        controller = self.controller()
        error = InputTargetError("key release failed")
        controller._send_key = Mock(side_effect=[None, error])
        with patch("src.controller.time.sleep"), self.assertRaises(InputTargetError):
            controller.send_escape()
        self.assertIs(controller.after_action.call_args.args[1], error)

    def test_scroll_keeps_pointer_over_list_until_game_can_process_wheel(self):
        controller = self.controller()
        events = []
        controller._send_move.side_effect = lambda *args: events.append("move")
        controller.wait = lambda delay: events.append(("wait", delay))
        controller._send_mouse_input.side_effect = lambda *args, **kwargs: events.append("restore")
        with patch("src.controller.user32.SendInput",
                   side_effect=lambda *args: (events.append("wheel"), 1)[1]):
            controller.scroll_at_percent(.87, .54, clicks=-3)
        self.assertEqual(events[:5], ["move", ("wait", .08), "wheel", ("wait", .15), "restore"])


if __name__ == "__main__":
    unittest.main()
