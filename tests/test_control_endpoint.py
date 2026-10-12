"""/control: the websocket for remote controls (the Companion module in
src-companion/) - on/off status out, the six toggle actions in."""

import asyncio

from fastapi.testclient import TestClient

from sts.web.server import CaptionBroadcaster, create_app


def _client():
    broadcaster = CaptionBroadcaster()
    received = []
    broadcaster.on_control = received.append
    broadcaster._last_status = {"type": "status", "active": True, "translation_enabled": True,
                                "voice_enabled": False, "voice_stopping": False}
    return TestClient(create_app(broadcaster)), broadcaster, received


def test_a_remote_control_gets_the_stage_and_status_and_can_toggle():
    client, _, received = _client()
    with client.websocket_connect("/control") as ws:
        assert ws.receive_json()["type"] == "stage"
        status = ws.receive_json()
        assert status["type"] == "status" and status["voice_enabled"] is False
        ws.send_json({"action": "voice_on"})
        ws.send_json({"action": "get_status"})
        assert ws.receive_json()["type"] == "status"
    assert received == [{"action": "voice_on"}]


def test_only_the_toggle_actions_are_accepted():
    client, _, received = _client()
    with client.websocket_connect("/control") as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"action": "reset"})
        assert ws.receive_json()["type"] == "error"
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
    assert received == []


def test_remote_controls_get_status_and_stage_but_not_captions():
    broadcaster = CaptionBroadcaster()

    class _Socket:
        def __init__(self):
            self.sent = []

        async def send_text(self, payload):
            self.sent.append(payload)

    page, remote = _Socket(), _Socket()
    broadcaster._clients.add(page)
    broadcaster._control_clients.add(remote)
    for message in ({"type": "original", "text": "Слово."}, {"type": "status", "active": False},
                    {"type": "input_level", "level": 0.3}, {"type": "stage", "text": "Listening"}):
        asyncio.run(broadcaster._send_to_all(message))
    assert len(page.sent) == 4
    assert [__import__("json").loads(p)["type"] for p in remote.sent] == ["status", "stage"]


def test_a_remote_control_can_switch_voices():
    client, _, received = _client()
    with client.websocket_connect("/control") as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"action": "select_speaker", "speaker_id": "default-1-25"})
        ws.send_json({"action": "select_speaker"})  # no voice named
        assert ws.receive_json()["type"] == "error"
    assert received == [{"action": "select_speaker", "speaker_id": "default-1-25"}]
