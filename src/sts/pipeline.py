"""Wires capture -> VAD -> ASR -> translation -> TTS -> playback into one
streaming pipeline, each stage on its own thread connected by queues so a
slow stage (translation, TTS) doesn't stall audio capture.
"""

import queue
import threading

from sts.asr.whisper_asr import WhisperASR
from sts.audio.capture import AudioCapture
from sts.audio.playback import AudioPlayback
from sts.audio.vad import SpeechSegmenter
from sts.config import PipelineConfig
from sts.translate.translator import Translator
from sts.tts.xtts import SAMPLE_RATE, XTTSSynthesizer
from sts.web.server import CaptionBroadcaster

_SENTINEL = object()


class Pipeline:
    def __init__(self, cfg: PipelineConfig, broadcaster: CaptionBroadcaster | None = None):
        self.cfg = cfg
        self.capture = AudioCapture(cfg.audio)
        self.playback = AudioPlayback(cfg.audio)
        self.segmenter = SpeechSegmenter(cfg.audio, cfg.vad)
        self.asr = WhisperASR(cfg.asr)
        self.translator = Translator(cfg.translation)
        self.tts = XTTSSynthesizer(cfg.tts)
        self.broadcaster = broadcaster

        self._segments: queue.Queue = queue.Queue()
        self._transcripts: queue.Queue = queue.Queue()
        self._translations: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def clone_voice(self, reference_wav_path: str) -> None:
        self.tts.clone_voice(reference_wav_path)

    def _segment_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.capture.frames.get()
            segment = self.segmenter.process(frame)
            if segment is not None:
                self._segments.put(segment)

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            segment = self._segments.get()
            text = self.asr.transcribe(segment)
            if text:
                print(f"[ru] {text}")
                if self.broadcaster:
                    self.broadcaster.broadcast_original(text)
                self._transcripts.put(text)

    def _translate_loop(self) -> None:
        while not self._stop.is_set():
            text = self._transcripts.get()
            translated = self.translator.translate(text)
            if translated:
                print(f"[en] {translated}")
                if self.broadcaster:
                    self.broadcaster.broadcast_translated(translated)
                self._translations.put(translated)

    def _tts_loop(self) -> None:
        while not self._stop.is_set():
            text = self._translations.get()
            audio = self.tts.synthesize(text)
            if audio.size:
                self.playback.play(audio, sample_rate=SAMPLE_RATE, blocking=True)

    def start(self) -> None:
        self.capture.start()
        for target in (
            self._segment_loop,
            self._asr_loop,
            self._translate_loop,
            self._tts_loop,
        ):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
