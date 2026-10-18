"""Recognition on/off and voice-training-sample bookkeeping for the caption
page's control buttons.

Pulled out of `Pipeline` because that class also constructs the ASR/
translation/TTS models, which need a GPU and downloaded weights and so can't
be instantiated in a plain unit test; this class holds none of that and can
be exercised on its own.
"""

import threading

import numpy as np


class RecognitionControl:
    def __init__(self, sample_rate: int, target_seconds: float = 18.0):
        self.sample_rate = sample_rate
        self.target_seconds = target_seconds

        self.active = threading.Event()
        self.active.set()  # recognition/captions on by default
        self.translation_enabled = threading.Event()
        self.translation_enabled.set()  # translation on by default, independent of recognition
        self.voice_enabled = threading.Event()  # set by sts.cli at start (on unless --no-voice)
        self.training = threading.Event()

        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self.training_seconds = 0.0

    def start(self) -> None:
        self.active.set()

    def stop(self) -> None:
        self.active.clear()

    def enable_voice(self) -> None:
        self.voice_enabled.set()

    def disable_voice(self) -> None:
        self.voice_enabled.clear()

    def enable_translation(self) -> None:
        self.translation_enabled.set()

    def disable_translation(self) -> None:
        self.translation_enabled.clear()

    def start_training(self) -> None:
        with self._lock:
            self._buffer = []
            self.training_seconds = 0.0
        self.training.set()

    def stop_training(self, discard: bool = False) -> np.ndarray | None:
        self.training.clear()
        with self._lock:
            buffer = self._buffer
            self._buffer = []
            self.training_seconds = 0.0
        if discard or not buffer:
            return None
        return np.concatenate(buffer)

    def accumulate_training_segment(self, segment: np.ndarray) -> bool:
        """Feeds one more segment into the in-progress training sample.

        Returns True the moment the target duration is reached, so the
        caller knows to call `stop_training()` and use the result.
        """
        with self._lock:
            self._buffer.append(segment)
            self.training_seconds += len(segment) / self.sample_rate
            return self.training_seconds >= self.target_seconds
