import threading
import time

import pytest

from sts.worker import StageProcess, StageProcessDied


class FakeStage:
    """Stands in for a translator/synthesizer: plain calls, an attribute, streams."""

    def __init__(self, prefix: str):
        if prefix == "boom":
            raise ValueError("model failed to load")
        self.prefix = prefix
        self.cloned_speaker_wav = None
        self.produced = 0

    def translate(self, text: str) -> str:
        return f"{self.prefix}{text}"

    def clone_voice(self, path: str) -> None:
        self.cloned_speaker_wav = path

    def pieces(self, n: int, delay: float = 0.0):
        for i in range(n):
            time.sleep(delay)
            self.produced += 1
            yield i

    def failing_stream(self):
        yield 1
        raise KeyError("mid-stream")

    def fail(self):
        raise ValueError("bad voice")

    def die(self):
        import os

        os._exit(3)


@pytest.fixture
def stage():
    proc = StageProcess(FakeStage, "en:", name="test-stage")
    yield proc
    proc.close()


def test_calls_and_attributes_are_forwarded(stage):
    assert stage.translate("привет") == "en:привет"
    assert stage.cloned_speaker_wav is None
    stage.clone_voice("/tmp/a.wav")
    assert stage.cloned_speaker_wav == "/tmp/a.wav"


def test_missing_attributes_look_missing(stage):
    assert not hasattr(stage, "stream_chunks")
    assert getattr(stage, "stream_chunks", None) is None
    assert hasattr(stage, "translate")


def test_generators_stream_every_item(stage):
    assert list(stage.pieces(4)) == [0, 1, 2, 3]
    assert list(stage.pieces(0)) == []


def test_closing_a_stream_early_stops_the_child_and_frees_the_connection(stage):
    for item in stage.pieces(50, delay=0.02):
        if item == 1:
            break
    # The child stops after at most the item it was working on.
    assert stage.produced <= 4
    assert stage.translate("x") == "en:x"


def test_exceptions_are_raised_in_the_caller(stage):
    with pytest.raises(ValueError, match="bad voice"):
        stage.fail()
    with pytest.raises(KeyError):
        list(stage.failing_stream())
    assert stage.translate("still works") == "en:still works"


def test_load_failure_is_raised(tmp_path):
    proc = StageProcess(FakeStage, "boom", name="test-stage")
    with pytest.raises(ValueError, match="failed to load"):
        proc.wait_ready()
    proc.close()


def test_a_dead_child_is_reported(stage):
    with pytest.raises(StageProcessDied):
        stage.die()


def test_calls_from_several_threads_are_serialized(stage):
    results = {}

    def worker(i):
        results[i] = stage.translate(str(i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == {i: f"en:{i}" for i in range(8)}
