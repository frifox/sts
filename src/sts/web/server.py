"""Serves the live-caption page and streams original/translated text to it
over a websocket.

The pipeline's ASR/translation stages run on plain threads, not inside the
server's asyncio loop, so `CaptionBroadcaster` accepts text from any thread
and schedules the actual send onto the loop via `run_coroutine_threadsafe`.
"""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

_STATIC_DIR = Path(__file__).parent / "static"


class CaptionBroadcaster:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def register(self, websocket: WebSocket) -> None:
        self._clients.add(websocket)

    async def unregister(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

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
        if self.loop is None or not text:
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
                # Clients don't send anything; this just blocks until they
                # disconnect so we can clean them up.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await broadcaster.unregister(websocket)

    return app


def run_server(broadcaster: CaptionBroadcaster, host: str = "0.0.0.0", port: int = 8765) -> None:
    app = create_app(broadcaster)
    uvicorn.run(app, host=host, port=port, log_level="warning")
