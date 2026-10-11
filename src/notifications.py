"""Short local notification tones; playback failures never interrupt the game."""
import io
import math
import struct
import threading
import wave


PATTERNS = {"journey_end": (660, 880, 1100), "paused_safe": (440, 330), "recovered": (660, 880)}


def tone_wav(event, volume=50):
    if event not in PATTERNS or type(volume) is not int or not 0 <= volume <= 100:
        raise ValueError("无效提示音或音量")
    rate, duration = 22050, .12
    frames = bytearray()
    for frequency in PATTERNS[event]:
        count = int(rate * duration)
        for i in range(count):
            envelope = min(1, i / 180, (count - i) / 250)
            sample = int(10000 * volume / 100 * envelope * math.sin(2 * math.pi * frequency * i / rate))
            frames.extend(struct.pack("<h", sample))
        frames.extend(b"\0\0" * int(rate * .04))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(frames)
    return output.getvalue()


class SoundNotifier:
    def __init__(self, settings=None, player=None):
        self.settings = settings or {}
        self.player = player or self._play
        self.last_state = None
        self._lock = threading.Lock()
        self._threads = []
        self.last_error = None

    @staticmethod
    def _play(data):
        import winsound
        # Playback is synchronous by default; Python 3.12 has no SND_SYNC name.
        winsound.PlaySound(data, winsound.SND_MEMORY)

    def state_changed(self, state):
        previous, self.last_state = self.last_state, state
        if previous == state:
            return False
        event = state if state in {"journey_end", "paused_safe"} else (
            "recovered" if state == "running" and previous == "recovering" else None)
        try:
            return self.emit(event) if event else False
        except (TypeError, ValueError) as error:
            self.last_error = str(error)
            return False

    def emit(self, event):
        if (not self.settings.get("enabled", False) or not self.settings.get("events", {}).get(event, False)
                or not self.settings.get("volume", 50)):
            return False
        data = tone_wav(event, self.settings.get("volume", 50))
        self._threads = [t for t in self._threads if t.is_alive()]
        if len(self._threads) >= 3 and event != "journey_end":
            return False

        def play():
            try:
                with self._lock:
                    self.player(data)
            except Exception as error:
                self.last_error = str(error)
        thread = threading.Thread(target=play, daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()
        return True

    def close(self):
        for thread in self._threads:
            thread.join(timeout=1)
