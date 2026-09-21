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
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from sts.log import vlog

_STATIC_DIR = Path(__file__).parent / "static"

# Recent caption lines (per kind) replayed to newly connected pages, so a
# refresh doesn't come back blank.
CAPTION_HISTORY_LINES = 4


class CaptionBroadcaster:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()
        self.on_control: Callable[[dict], None] | None = None
        self._last_status: dict | None = None
        self._last_library: dict | None = None
        self._last_stage: dict = {"type": "stage", "text": "Starting…", "ready": False}
        self._history: dict[str, deque[str]] = {
            "original": deque(maxlen=CAPTION_HISTORY_LINES),
            "translated": deque(maxlen=CAPTION_HISTORY_LINES),
        }
        self._history_lock = threading.Lock()

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
        with self._history_lock:
            history = {kind: list(lines) for kind, lines in self._history.items()}
        for kind, lines in history.items():
            for text in lines:
                await websocket.send_text(json.dumps({"type": kind, "text": text}))

    async def unregister(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)
        vlog("web", f"caption page disconnected ({len(self._clients)} client(s))")

    async def _send_to_all(self, message: dict) -> None:
        dead = []
        payload = json.dumps(message)
        for client in self._clients:
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    def _broadcast(self, kind: str, text: str) -> None:
        if not text:
            return
        with self._history_lock:
            self._history[kind].append(text)
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all({"type": kind, "text": text}), self.loop)

    def broadcast_original(self, text: str) -> None:
        self._broadcast("original", text)

    def broadcast_translated(self, text: str) -> None:
        self._broadcast("translated", text)

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

    def broadcast_stage(self, text: str, ready: bool = False) -> None:
        """Startup progress ("Loading Whisper…" -> "Listening"). Remembered so
        pages that connect mid-startup see the current stage immediately."""
        self._last_stage = {"type": "stage", "text": text, "ready": ready}
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(self._last_stage), self.loop)

    def broadcast_reset(self) -> None:
        with self._history_lock:
            for lines in self._history.values():
                lines.clear()
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all({"type": "reset"}), self.loop)

    def broadcast_status(
        self,
        active: bool,
        voice_enabled: bool,
        training: bool,
        training_seconds: float,
        training_target_seconds: float,
        cloned: bool,
        active_speaker_id: str | None = None,
        default_voice: str | None = None,
        voice_groups: dict[str, list[str]] | None = None,
    ) -> None:
        message = {
            "type": "status",
            "active": active,
            "voice_enabled": voice_enabled,
            "training": training,
            "training_seconds": round(training_seconds, 1),
            "training_target_seconds": training_target_seconds,
            "cloned": cloned,
            "active_speaker_id": active_speaker_id,
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

    def _handle_incoming(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(data, dict) and data.get("action") and self.on_control:
            self.on_control(data)


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
                raw = await websocket.receive_text()
                broadcaster._handle_incoming(raw)
        except WebSocketDisconnect:
            pass
        finally:
            await broadcaster.unregister(websocket)

    return app


def run_server(broadcaster: CaptionBroadcaster, host: str = "0.0.0.0", port: int = 8765) -> None:
    app = create_app(broadcaster)
    uvicorn.run(app, host=host, port=port, log_level="warning")
