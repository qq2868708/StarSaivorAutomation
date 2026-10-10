"""No game input: menu/choice separation, verified event choice and story skip."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from src.recognition import ScreenDetector
from src.handlers import (
    EventHandler, EventDecisionPending, MainMenuHandler, SkipHandler, FrameContext,
    UnknownHandler, AttributeGainHandler, JourneyEndHandler, RewardAdvanceHandler,
    InventoryHandler,
    AppraisalHandler,
    JourneyEndException,
)

class FlowRuntimeTests(unittest.TestCase):
    def test_inventory_title_is_detected(self):
        detector = ScreenDetector(Mock())
        detector.read_region_text = Mock(return_value='背包')
        self.assertTrue(detector.is_inventory_screen(np.zeros((900, 1600, 3), np.uint8)))

    def test_inventory_handler_closes_popup_without_center_click(self):
        handler = InventoryHandler()
        frame = SimpleNamespace(screenshot=np.zeros((900, 1600, 3), np.uint8))
        detector = Mock()
        detector.is_inventory_screen.return_value = True
        ctx = SimpleNamespace(capture=Mock(), controller=Mock())
        ctx.capture.capture_game.return_value = frame.screenshot
        with patch('src.recognition.ScreenDetector', return_value=detector):
            self.assertTrue(handler.can_handle(frame, Mock()))
            self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(0.785, 0.270)
        ctx.controller.click_center_multi.assert_not_called()

    def test_appraisal_entry_clicks_icon_hotspot(self):
        handler = AppraisalHandler()
        frame = SimpleNamespace(screenshot=np.zeros((900, 1600, 3), np.uint8))
        detector = Mock()
        detector.is_appraisal_screen.return_value = False
        detector.is_appraisal_due_screen.return_value = True
        detector.find_action_position.return_value = (0.885, 0.465)
        ctx = SimpleNamespace(capture=Mock(), controller=Mock(), ocr=Mock(), pending_screenshot=None)
        ctx.capture.capture_game.side_effect = [frame.screenshot, frame.screenshot]
        with patch('src.recognition.ScreenDetector', return_value=detector), \
             patch('src.handlers._wait_for_screen', return_value=True):
            self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(0.941, 0.466)

    def test_appraisal_confirmation_is_not_preparation_screen(self):
        detector = ScreenDetector(Mock())
        detector.read_region_text = Mock(side_effect=[
            '虚空评鉴战',
            '是否要进行评鉴战？建议战斗 跳过战斗 开始委托',
        ])
        self.assertFalse(detector.is_appraisal_screen(np.zeros((900, 1600, 3), np.uint8)))

    def test_stage_goal_and_potential_reward_are_not_journey_end(self):
        handler = JourneyEndHandler.__new__(JourneyEndHandler)
        for text in ('达成目标 点击以继续', '获得潜质点数 +40',
                     '总获得潜质点数 100', '距目标 6 参加评鉴战'):
            self.assertFalse(handler._is_journey_end_text(text))
        self.assertTrue(handler._is_journey_end_text('旅程结束'))

    def test_final_journey_end_pauses_for_manual_takeover(self):
        handler = JourneyEndHandler({'journey_end': {'auto_learn_skills': True,
                                                       'pause_for_manual': True}})
        ctx = SimpleNamespace(
            capture=Mock(), controller=Mock(), ocr=Mock(), pending_screenshot=None,
        )
        ctx.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        ctx.ocr.recognize_region.return_value = '旅程结束'
        with self.assertRaises(Exception) as raised:
            handler.handle(ctx)
        self.assertIsInstance(raised.exception, JourneyEndException)
        ctx.controller.click_at_percent.assert_not_called()
        ctx.controller.click_center_multi.assert_not_called()

    def test_attribute_gain_overlay_is_recognized(self):
        frame_path = (Path(__file__).resolve().parents[1] /
                      "verification" / "attribute-gain-20261007" /
                      "attribute_gain_overlay.png")
        if not frame_path.is_file():
            self.skipTest("optional captured attribute frame is not included in the source checkout")
        screenshot = cv2.imread(str(frame_path))
        self.assertIsNotNone(screenshot)
        from src.light_ocr import LightOCR
        self.assertTrue(AttributeGainHandler().can_handle(
            FrameContext(screenshot), LightOCR()))

    def test_rest_great_success_result_is_advanced(self):
        frame = SimpleNamespace(
            screenshot=np.zeros((900, 1600, 3), np.uint8),
            recognize_full=Mock(return_value=[('住处 GREAT SUCCESS 点击以继续', 0.99, None)]),
        )
        handler = RewardAdvanceHandler()
        self.assertTrue(handler.can_handle(frame, Mock()))
        ctx = SimpleNamespace(controller=Mock(), capture=Mock())
        ctx.capture.capture_game.return_value = frame.screenshot
        self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(0.50, 0.89)

    def test_skip_icon_distinguishes_enabled_disabled_and_fast_forward_location(self):
        glyph = cv2.imread(str(Path(__file__).resolve().parents[1] / 'templates/story_skip.png'))
        self.assertIsNotNone(glyph)
        frame = np.zeros((900,1600,3),np.uint8)
        frame[38:62,1158:1202] = glyph
        detector = ScreenDetector(Mock())
        self.assertEqual(detector.find_story_skip(frame), (0.7375,0.05555555555555555))
        frame[38:62,1158:1202] = np.minimum(glyph,120)
        self.assertIsNone(detector.find_story_skip(frame))
        frame[38:62,1158:1202] = 0
        frame[38:62,1230:1274] = glyph
        self.assertIsNone(detector.find_story_skip(frame))

    def test_event_phrases_are_not_menu_buttons(self):
        detector = ScreenDetector(Mock())
        for texts, result in [(['', '训练到此为止，休息吧。', '再试一次吧20'], ''),
                              (['训练', '委托?', '休息'], 'training')]:
            detector.read_region_text = Mock(side_effect=texts)
            self.assertEqual(detector.is_main_menu_screen(None), result)

    def test_failed_training_transition_cannot_set_pending_training(self):
        ctx = SimpleNamespace(try_detect_turn=Mock(), capture=Mock(), controller=Mock(),
                              ocr=Mock(), pending_action='training', pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900,1600,3), np.uint8)
        handler = MainMenuHandler()
        with patch('src.recognition.ScreenDetector') as cls, patch('src.handlers._wait_for_screen', return_value=False):
            cls.return_value.is_main_menu_screen.return_value = 'training'
            handler._has_commission_notify = Mock(return_value=False)
            handler._has_shop_event = Mock(return_value=False)
            self.assertTrue(handler.handle(ctx))
        self.assertEqual(ctx.pending_action, '')
        ctx.controller.click_at_percent.assert_called_once()

    def test_story_skip_must_not_consume_event_options(self):
        handler = SkipHandler()
        frame = Mock(screenshot=np.zeros((900,1600,3),np.uint8))
        frame.get_ocr.return_value = ''
        with patch('src.recognition.ScreenDetector') as cls:
            cls.return_value.find_story_skip.return_value = (0.7375,0.0556)
            cls.return_value.is_story_dialogue.return_value = False
            self.assertFalse(handler.can_handle(frame, Mock()))

    def test_unknown_handler_never_clicks_center_when_skip_is_absent(self):
        handler = UnknownHandler()
        frame = np.zeros((900, 1600, 3), np.uint8)
        ctx = SimpleNamespace(
            capture=Mock(), controller=Mock(), ocr=Mock(),
            consecutive_unknown=0, pending_screenshot=None,
        )
        ctx.capture.capture_game.return_value = frame
        detector = Mock()
        detector.find_story_skip.return_value = None
        detector.read_event_option_rows.return_value = []
        with patch('src.recognition.ScreenDetector', return_value=detector), \
             patch.object(UnknownHandler, '_is_known_screen', return_value=False), \
             patch.object(UnknownHandler, 'ADVANCE_TIMEOUT', 0.01), \
             patch('src.handlers.time.perf_counter', side_effect=[0.0, 0.0, 0.02]), \
             patch('src.handlers.time.sleep'):
            self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_not_called()
        ctx.controller.click_center_multi.assert_not_called()

    def test_unknown_handler_clicks_top_skip_once(self):
        handler = UnknownHandler()
        frame = np.zeros((900, 1600, 3), np.uint8)
        ctx = SimpleNamespace(
            capture=Mock(), controller=Mock(), ocr=Mock(),
            consecutive_unknown=0, pending_screenshot=None,
        )
        ctx.capture.capture_game.return_value = frame
        detector = Mock()
        detector.find_story_skip.return_value = (0.7375, 0.0555555)
        detector.read_event_option_rows.return_value = []
        with patch('src.recognition.ScreenDetector', return_value=detector), \
             patch.object(UnknownHandler, '_is_known_screen', return_value=False), \
             patch.object(UnknownHandler, 'ADVANCE_TIMEOUT', 0.01), \
             patch('src.handlers.time.perf_counter', side_effect=[0.0, 0.0, 0.001, 0.02]), \
             patch('src.handlers.time.sleep'):
            self.assertTrue(handler.handle(ctx))
        ctx.controller.click_at_percent.assert_called_once_with(0.7375, 0.0555555)
        ctx.controller.click_center_multi.assert_not_called()

    def test_unknown_effects_pause_before_input_and_do_not_rewrite_old_profile(self):
        handler = EventHandler.__new__(EventHandler)
        handler.config = {'events': {'reference_document': 'document'}}
        handler._events = []
        handler._events = []
        handler._auto_learn_event = Mock(side_effect=AssertionError('must preserve profile'))
        ctx = SimpleNamespace(ocr=Mock(),controller=Mock(),capture=Mock(),logger=Mock())
        rows = [{'text':'选项甲','enabled':True,'y':0.59}, {'text':'选项乙','enabled':True,'y':0.66}]
        with patch('src.recognition.ScreenDetector') as cls, patch('src.handlers._cached_load_json',return_value=[]), \
             patch('src.handlers.cv2.imwrite'), patch('src.handlers.Path.mkdir'), patch('src.handlers.Path.write_text'):
            cls.return_value.read_region_text.return_value = '旅程事件测试事件'
            cls.return_value.read_event_option_rows.return_value = rows
            with self.assertRaises(EventDecisionPending):
                handler._handle_document_event(ctx,np.zeros((900,1600,3),np.uint8))
        ctx.controller.click_at_percent.assert_not_called()
        handler._auto_learn_event.assert_not_called()

    def test_verified_effect_choice_uses_detected_row_and_only_one_click(self):
        handler = EventHandler.__new__(EventHandler)
        handler.config = {'events': {'reference_document':'document'}}
        rows = [{'text':'选项甲','enabled':True,'y':0.58}, {'text':'选项乙','enabled':True,'y':0.646}]
        event = {'id':'test','event_name':'测试事件','source_document':'document','document_verified':True,
                 'recommended_option':2,'options':[{'keyword':'选项甲','effect_text':'力量+5'},
                                                   {'keyword':'选项乙','effect_text':'体力+10'}]}
        ctx = SimpleNamespace(ocr=Mock(),controller=Mock(),capture=Mock(),logger=Mock(),pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900,1600,3),np.uint8)
        with patch('src.recognition.ScreenDetector') as cls, patch('src.handlers._cached_load_json',return_value=[event]):
            cls.return_value.read_region_text.return_value = '旅程事件测试事件'
            cls.return_value.read_event_option_rows.side_effect = [rows,[]]
            self.assertTrue(handler._handle_document_event(ctx,ctx.capture.capture_game.return_value))
        ctx.controller.click_at_percent.assert_called_once_with(0.80,0.646)
        ctx.logger.log_event.assert_called_once()

    def test_verified_event_collapses_wrapped_option_before_matching(self):
        handler = EventHandler.__new__(EventHandler)
        handler.config = {'events': {'reference_document':'document'}}
        rows = [
            {'text':'还是老实认错反省吧！','enabled':True,'y':0.626},
            {'text':'既然这样，也只能试着装可怜蒙混','enabled':True,'y':0.690},
            {'text':'过去了！','enabled':True,'y':0.722},
        ]
        event = {'id':'paperwork','event_name':'文书工作总是晴转阴',
                 'source_document':'document','document_verified':True,
                 'recommended_option':1,
                 'options':[{'keyword':'还是老实认错反省吧','alias':['认错反省'],
                             'effect_text':'韧性+10、专注+12'},
                            {'keyword':'既然这样也只能试着装可怜蒙混过去了',
                             'alias':['装可怜','蒙混过去'],
                             'effect_text':'获得潜能破坏感知'}]}
        ctx = SimpleNamespace(ocr=Mock(),controller=Mock(),capture=Mock(),logger=Mock(),pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900,1600,3),np.uint8)
        with patch('src.recognition.ScreenDetector') as cls, patch('src.handlers._cached_load_json',return_value=[event]):
            cls.return_value.read_region_text.return_value = '阿尔克那事件文书工作总是晴转阴'
            cls.return_value.read_event_option_rows.side_effect = [rows,[]]
            self.assertTrue(handler._handle_document_event(ctx,ctx.capture.capture_game.return_value))
        ctx.controller.click_at_percent.assert_called_once_with(0.80,0.626)

    def test_verified_event_skips_locked_high_score_option(self):
        handler = EventHandler.__new__(EventHandler)
        handler.config = {'events': {'reference_document': 'document'}}
        handler._events = []
        rows = [
            {'text': '用钱解决吧。', 'enabled': True, 'y': 0.585},
            {'text': '自己开起重机怎么样？', 'enabled': False, 'y': 0.652},
            {'text': '得找找看有没有替代路线了。', 'enabled': True, 'y': 0.717},
        ]
        event = {
            'id': 'maze', 'event_name': '探索迷宫',
            'title_aliases': ['迷宫探勘'],
            'source_document': 'document', 'document_verified': True,
            'options': [
                {'keyword': '用钱解决', 'alias': ['用钱解决吧'],
                 'effect_text': '耐力+15；获得遗物：未确认文物'},
                {'keyword': '自己开起重机', 'alias': ['自己开起重机怎么样'],
                 'effect_text': '韧性+15；潜质点+37'},
                {'keyword': '替代路线', 'alias': ['得找找看有没有替代路线了'], 'effect_text': '无'},
            ],
        }
        ctx = SimpleNamespace(ocr=Mock(), controller=Mock(), capture=Mock(),
                              logger=Mock(), pending_screenshot=None, engine=Mock())
        ctx.capture.capture_game.return_value = np.zeros((900, 1600, 3), np.uint8)
        with patch('src.recognition.ScreenDetector') as cls, \
             patch('src.handlers._cached_load_json', return_value=[event]):
            cls.return_value.read_region_text.return_value = '旅程事件迷宫探勘'
            cls.return_value.read_event_option_rows.side_effect = [rows, []]
            handler._event_build_direction = Mock(return_value='attack')
            self.assertTrue(handler._handle_document_event(ctx, ctx.capture.capture_game.return_value))
        ctx.controller.click_at_percent.assert_called_once_with(0.80, 0.585)

    def test_training_failure_special_case_uses_confirmed_local_safe_option(self):
        handler = EventHandler.__new__(EventHandler)
        handler.config = {'events': {'reference_document':'document'}}
        handler._match_known_event = Mock(return_value={'id':'training_failure','event_name':'训练失败',
            'status':'confirmed','recommended_option':3})
        rows = [{'text':'休息吧','enabled':True,'y':0.58}, {'text':'再试一次','enabled':True,'y':0.65},
                {'text':'活力药水','enabled':False,'y':0.72}]
        ctx = SimpleNamespace(ocr=Mock(),controller=Mock(),capture=Mock(),logger=Mock(),pending_screenshot=None)
        ctx.capture.capture_game.return_value = np.zeros((900,1600,3),np.uint8)
        with patch('src.recognition.ScreenDetector') as cls, patch('src.handlers._cached_load_json',return_value=[]):
            cls.return_value.read_region_text.return_value = '训练事件训练失败'
            cls.return_value.read_event_option_rows.return_value = rows
            self.assertTrue(handler._handle_document_event(ctx,ctx.capture.capture_game.return_value))
        ctx.controller.click_at_percent.assert_called_once_with(0.80,0.58)

    def _training_failure_handler(self):
        handler = EventHandler.__new__(EventHandler)
        handler._training_failure_rules = Mock(return_value={
            'high_stamina_threshold': 60,
            'potion_option': 3,
            'high_stamina_option': 2,
            'low_stamina_option': 1,
        })
        return handler

    def test_training_failure_prefers_available_potion(self):
        handler = self._training_failure_handler()
        rows = [
            {'text': '休息吧', 'enabled': True},
            {'text': '再试一次', 'enabled': True},
            {'text': '递给她活力药水', 'enabled': True},
        ]
        self.assertEqual(handler._select_training_failure_option(rows, 20), 3)

    def test_training_failure_uses_retry_when_stamina_is_high_without_potion(self):
        handler = self._training_failure_handler()
        rows = [
            {'text': '休息吧', 'enabled': True},
            {'text': '再试一次', 'enabled': True},
            {'text': '活力药水', 'enabled': False},
        ]
        self.assertEqual(handler._select_training_failure_option(rows, 75), 2)

    def test_training_failure_uses_rest_when_stamina_is_low_without_potion(self):
        handler = self._training_failure_handler()
        rows = [
            {'text': '休息吧', 'enabled': True},
            {'text': '再试一次', 'enabled': True},
            {'text': '活力药水', 'enabled': False},
        ]
        self.assertEqual(handler._select_training_failure_option(rows, 30), 1)

    def test_training_failure_skips_locked_preferred_option(self):
        handler = self._training_failure_handler()
        rows = [
            {'text': '休息吧', 'enabled': True},
            {'text': '再试一次', 'enabled': False},
            {'text': '活力药水', 'enabled': False},
        ]
        self.assertEqual(handler._select_training_failure_option(rows, 80), 1)

if __name__ == '__main__':
    unittest.main()
