"""Serves the live-caption page, streams original/translated/verse text to
it over a websocket, and relays operator control button presses back.

The pipeline's ASR/translation stages run on plain threads, not inside the
server's asyncio loop, so `CaptionBroadcaster` accepts text from any thread
and schedules the actual send onto the loop via `run_coroutine_threadsafe`.
Control messages flow the other way: the browser sends `{"action": ...}` on
the same websocket, which is handed to whatever callable the pipeline
registered as `on_control`.
"""

import asyncio
import json
import struct
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from sts.log import vlog

# Binary audio frames sent over /ws are a 4-byte little-endian sample rate
# header followed by raw float32 LE mono PCM - self-contained, no separate
# control message needed to keep the two in sync.
_AUDIO_HEADER = struct.Struct("<I")

_STATIC_DIR = Path(__file__).parent / "static"

# Recent caption lines (per kind) replayed to newly connected pages, so a
# refresh doesn't come back blank.
CAPTION_HISTORY_LINES = 4


class CaptionBroadcaster:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()
        self.on_control: Callable[[dict], None] | None = None
        self.on_mic_frame: Callable[[bytes], None] | None = None
        # The one connection currently allowed to feed browser-mic audio, so
        # frames from a stale/other tab can't get mixed into the stream.
        self._mic_source: WebSocket | None = None
        self._last_status: dict | None = None
        self._last_library: dict | None = None
        self._last_devices: dict | None = None
        self._last_engines: dict | None = None
        self._last_design: dict = {"type": "design", "state": "idle", "message": "", "has_voice": False}
        self._last_stage: dict = {"type": "stage", "text": "Starting…", "ready": False}
        # Latest real-time-factor reading per pipeline stage ("asr" |
        # "translate" | "voice"), so a page that connects mid-session sees
        # the current numbers immediately instead of blank ones until the
        # next segment.
        self._last_rtf: dict[str, dict] = {}
        self._history: dict[str, deque[dict]] = {
            "original": deque(maxlen=CAPTION_HISTORY_LINES),
            "translated": deque(maxlen=CAPTION_HISTORY_LINES),
        }
        self._history_lock = threading.Lock()
        self._last_recent: dict | None = None  # see broadcast_original_recent

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def register(self, websocket: WebSocket) -> None:
        self._clients.add(websocket)
        await websocket.send_text(json.dumps(self._last_stage))
        vlog("web", f"caption page connected ({len(self._clients)} client(s))")
        if self._last_status is not None:
            await websocket.send_text(json.dumps(self._last_status))
        if self._last_library is not None:
            await websocket.send_text(json.dumps(self._last_library))
        await websocket.send_text(json.dumps(self._last_design))
        if self._last_devices is not None:
            await websocket.send_text(json.dumps(self._last_devices))
        if self._last_engines is not None:
            await websocket.send_text(json.dumps(self._last_engines))
        for message in self._last_rtf.values():
            await websocket.send_text(json.dumps(message))
        with self._history_lock:
            history = {kind: list(lines) for kind, lines in self._history.items()}
        for messages in history.values():
            for message in messages:
                await websocket.send_text(json.dumps(message))
        if self._last_recent is not None:
            await websocket.send_text(json.dumps(self._last_recent))

    async def unregister(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)
        if websocket is self._mic_source:
            # This tab was streaming its mic in; falling back to ALSA beats
            # silently losing all audio input because a browser tab closed.
            self._mic_source = None
            if self.on_control:
                self.on_control({"action": "mic_stop"})
        vlog("web", f"caption page disconnected ({len(self._clients)} client(s))")

    async def _send_to_all(self, message: dict) -> None:
        dead = []
        payload = json.dumps(message)
        # A snapshot: a page connecting or disconnecting during one of the
        # awaits below changes the set, which would end this loop with
        # "Set changed size during iteration" - and the message would be
        # lost for every client after that point.
        for client in list(self._clients):
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    async def _send_bytes_to_all(self, payload: bytes) -> None:
        dead = []
        for client in list(self._clients):  # a snapshot - see _send_to_all
            try:
                await client.send_bytes(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    def _broadcast(self, kind: str, text: str, seq: int | None = None, batch: int | None = None) -> None:
        if not text:
            return
        message = {"type": kind, "text": text}
        if seq is not None:
            message["seq"] = seq
        if batch is not None:
            message["batch"] = batch
        with self._history_lock:
            self._history[kind].append(message)
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_original(self, text: str, seq: int | None = None, batch: int | None = None) -> None:
        """A caption sentence that's final (locked - see sts.transcript).
        `seq` identifies it so its translation - which can arrive much
        later, or never (translation off) - can be matched back to it exactly
        by the page's portrait layout, instead of by arrival order."""
        self._broadcast("original", text, seq, batch)

    def broadcast_original_recent(self, text: str, sentences: list[str] | None = None) -> None:
        """The caption text that isn't final yet - replaces whatever the page
        showed as such before ("" clears it) - and the same as its separate
        sentences, so the page can pick out the first one (the one the
        finalize clock is about). Remembered, so a page that connects
        mid-sentence shows it too."""
        if sentences is None:
            sentences = [text] if text else []
        message = {"type": "original_recent", "text": text, "sentences": sentences}
        self._last_recent = message
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_finalize_timer(self, timer: tuple[float, float] | None) -> None:
        """(seconds left, the whole wait) before the first still-open caption
        sentence is finalized - drawn as a circle emptying clockwise - or
        None while no finalize rule is running. Not replayed on connect: it
        changes with every re-transcription anyway."""
        if self.loop is None:
            return
        message = {"type": "finalize_timer", "left": None, "total": None}
        if timer is not None:
            message.update(left=round(timer[0], 2), total=round(timer[1], 2))
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_translated(self, text: str, seq: int | None = None, batch: int | None = None) -> None:
        """`seq` is the `seq` of the original sentence this translates, and
        `batch` the batch of the commit it belongs to - so a translation that
        lags behind (slow TTS or GPU contention) still groups with its own
        commit instead of looking like a new one."""
        self._broadcast("translated", text, seq, batch)

    def broadcast_verse(
        self,
        ru_label: str,
        chapter: int,
        verse: int,
        verse_end: int | None,
        ru_text: str | None,
        en_text: str | None,
    ) -> None:
        if self.loop is None or not (ru_text or en_text):
            return
        reference = f"{ru_label} {chapter}:{verse}" + (f"-{verse_end}" if verse_end else "")
        message = {
            "type": "verse",
            "reference": reference,
            "ru_text": ru_text,
            "en_text": en_text,
        }
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Streams a synthesized speech clip to every connected page (the
        speaker-icon toggle decides client-side whether to actually play it -
        cheaper than tracking per-client subscriptions here, and the browser
        never plays anything without TTS voicing already being on, since this
        is only called with clips the pipeline already produced for the local
        speaker output)."""
        if self.loop is None or not audio.size:
            return
        payload = _AUDIO_HEADER.pack(sample_rate) + np.ascontiguousarray(audio, dtype=np.float32).tobytes()
        asyncio.run_coroutine_threadsafe(self._send_bytes_to_all(payload), self.loop)

    def broadcast_stage(self, text: str, ready: bool = False) -> None:
        """Startup progress ("Loading speech recognition…" -> "Listening"). Remembered so
        pages that connect mid-startup see the current stage immediately."""
        self._last_stage = {"type": "stage", "text": text, "ready": ready}
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(self._last_stage), self.loop)

    def broadcast_devices(self, message: dict) -> None:
        """The audio device lists and current selection (plus an `error`/`applied` note after an Apply)."""
        self._last_devices = {**message, "error": None, "applied": False}
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_design(self, state: str, message: str = "", has_voice: bool = False) -> None:
        """Progress of the "Create Voice" dialog: idle | busy | ready | saved | error."""
        self._last_design = {"type": "design", "state": state, "message": message, "has_voice": has_voice}
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(self._last_design), self.loop)

    def broadcast_reset(self) -> None:
        with self._history_lock:
            for lines in self._history.values():
                lines.clear()
        self._last_recent = None
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all({"type": "reset"}), self.loop)

    def broadcast_rtf(self, stage: str, value: float) -> None:
        """Real-time factor (processing time / audio duration) for one
        pipeline stage: "asr", "translate", or "voice". Below 1.0 means that
        stage is keeping up with live speech; above 1.0 means it's falling
        behind."""
        message = {"type": "rtf", "stage": stage, "value": value}
        self._last_rtf[stage] = message
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_input_level(self, level: float) -> None:
        """RMS (0..~1) of the most recent input frame, throttled by the
        caller to about every 100ms - drives the volume-bar meter. Not
        cached/replayed on connect like the others: a fresh value arrives
        within 100ms anyway, and a stale level would be actively misleading."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all({"type": "input_level", "level": level}), self.loop)

    def broadcast_output_level(self, level: float) -> None:
        """RMS (0..~1) of the most recently played output chunk (synthesized
        voice audio) - drives the volume-bar meter next to the Voice button.
        Same reasoning as broadcast_input_level: not cached/replayed on
        connect, a fresh value follows quickly and a stale one would mislead."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all({"type": "output_level", "level": level}), self.loop)

    def broadcast_engines(self, message: dict) -> None:
        """Static ASR/translation/TTS model + device info for the header dropdowns."""
        self._last_engines = {"type": "engines", **message}
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(self._last_engines), self.loop)

    def broadcast_status(
        self,
        active: bool,
        translation_enabled: bool,
        voice_enabled: bool,
        training: bool,
        training_seconds: float,
        training_target_seconds: float,
        cloned: bool,
        active_speaker_id: str | None = None,
        can_save: bool = False,
        output_volume: int | None = None,
        audio_source: str = "alsa",
        can_design: bool = False,
        default_voice: str | None = None,
        voice_groups: dict[str, list[str]] | None = None,
    ) -> None:
        message = {
            "type": "status",
            "active": active,
            "translation_enabled": translation_enabled,
            "voice_enabled": voice_enabled,
            "training": training,
            "training_seconds": round(training_seconds, 1),
            "training_target_seconds": training_target_seconds,
            "cloned": cloned,
            "active_speaker_id": active_speaker_id,
            "can_save": can_save,
            "output_volume": output_volume,
            "audio_source": audio_source,
            "can_design": can_design,
            "default_voice": default_voice,
            "voice_groups": voice_groups,
        }
        self._last_status = message
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def broadcast_library(self, speakers: list[dict]) -> None:
        message = {"type": "library", "speakers": speakers}
        self._last_library = message
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)

    def _handle_incoming(self, websocket: WebSocket, raw: str) -> None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict) or not data.get("action"):
            return
        action = data["action"]
        if action == "mic_start":
            self._mic_source = websocket  # displaces any other tab that was streaming
        elif action == "mic_stop" and self._mic_source is websocket:
            self._mic_source = None
        if self.on_control:
            self.on_control(data)

    def _handle_incoming_bytes(self, websocket: WebSocket, data: bytes) -> None:
        """Binary frames are only ever browser-mic audio (the one binary
        payload a client sends us; TTS audio flows the other way). Frames
        from any connection other than the current mic source are dropped -
        a stale/leftover stream from a tab that isn't in control right now."""
        if websocket is self._mic_source and self.on_mic_frame:
            self.on_mic_frame(data)


def create_app(broadcaster: CaptionBroadcaster) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def _bind_loop() -> None:
        broadcaster.bind_loop(asyncio.get_running_loop())

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await broadcaster.register(websocket)
        try:
            while True:
                # The generic receive() (rather than receive_text()) is needed
                # because this socket now carries both directions: JSON control
                # messages, and - only when this tab is the mic source -
                # binary microphone frames going the opposite way from the
                # binary TTS audio the server sends out.
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (text := message.get("text")) is not None:
                    broadcaster._handle_incoming(websocket, text)
                elif (data := message.get("bytes")) is not None:
                    broadcaster._handle_incoming_bytes(websocket, data)
        except WebSocketDisconnect:
            pass
        finally:
            await broadcaster.unregister(websocket)

    return app


def run_server(
    broadcaster: CaptionBroadcaster,
    host: str = "0.0.0.0",
    port: int = 8765,
    *,
    https_port: int | None = None,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
) -> None:
    """Serves the caption page over HTTP, and - when an HTTPS port and cert
    are given - also over HTTPS on the same app, concurrently. Both listeners
    run in the *same* asyncio event loop (via `asyncio.gather`, not separate
    threads), so `CaptionBroadcaster` still only ever has one loop to
    schedule broadcasts on regardless of which listener a client connected
    through. HTTPS matters because browsers only allow microphone capture
    (`getUserMedia`) from a secure context - plain LAN HTTP doesn't qualify."""
    app = create_app(broadcaster)
    configs = [uvicorn.Config(app, host=host, port=port, log_level="warning")]
    if https_port is not None:
        configs.append(
            uvicorn.Config(
                app, host=host, port=https_port, log_level="warning",
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
            )
        )

    async def _serve_all() -> None:
        await asyncio.gather(*(uvicorn.Server(cfg).serve() for cfg in configs))

    asyncio.run(_serve_all())
