"""Runs a model stage (translation, TTS) in a process of its own.

On Apple Silicon every model in one process has to take turns on the GPU
(see sts.gpu: PyTorch's MPS backend aborts when two threads submit work at
once), and each takes its turn for a whole `generate()` - a Qwen3 sentence
or a MADLAD translation holds it for seconds. With ASR, translation and TTS
all busy, captions updated every ~4 s instead of every 0.5 s. Separate
processes each get their own Metal queue, so the GPU interleaves their work
instead, and they no longer share one GIL either. Measured on an M5 Max with
all three running (a sentence to translate every 6 s, TTS continuous): ASR
step 3.8 s -> 0.13 s (its solo speed), translation 3.0 s -> 1.15 s, TTS
unchanged.

`StageProcess` stands in for the stage object: method calls and attribute
reads are forwarded to the real object in the child process. A method that
returns a generator (TTS's `synthesize_stream`, `translate_sentences`) streams
each item back as the child produces it; closing it early (the pipeline
dropping a sentence on Reset) stops the child after its current item.
"""

import inspect
import multiprocessing
import pickle
import threading
import traceback


class StageProcessDied(RuntimeError):
    pass


def _portable(exc: BaseException) -> BaseException:
    """`exc` if it survives pickling, else a RuntimeError carrying its text."""
    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:
        return RuntimeError(f"{type(exc).__name__}: {exc}")


def _serve(conn, factory, args, verbose: bool, log_start: float) -> None:
    import signal

    from sts import log

    # Ctrl+C in the terminal reaches the whole process group; the parent's
    # Pipeline.stop() closes this process (and it exits on its own if the
    # parent dies: the pipe reads EOF).
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log.set_verbose(verbose)
    log._start = log_start  # same timestamps as the parent's log lines
    try:
        obj = factory(*args)
    except BaseException as exc:
        traceback.print_exc()
        conn.send(("error", _portable(exc)))
        return
    methods = {name for name in dir(obj) if not name.startswith("_") and callable(getattr(obj, name, None))}
    conn.send(("ready", methods))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return  # the parent is gone
        if message[0] == "cancel":
            continue  # arrived after that stream had already ended
        kind, name, a, kw = message
        try:
            result = getattr(obj, name) if kind == "getattr" else getattr(obj, name)(*a, **kw)
            if inspect.isgenerator(result):
                for item in result:
                    conn.send(("item", item))
                    if conn.poll() and conn.recv()[0] == "cancel":
                        result.close()
                        break
                conn.send(("end", None))
            else:
                conn.send(("value", result))
        except BaseException as exc:
            if kind != "getattr" or not isinstance(exc, AttributeError):
                traceback.print_exc()
            conn.send(("error", _portable(exc)))


class _RemoteStream:
    """Iterator over a generator running in the child. Holds the connection
    until it ends or is closed/garbage-collected."""

    def __init__(self, proc: "StageProcess", first):
        self._proc = proc
        self._pending = [first]
        self._open = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self._open:
            raise StopIteration
        kind, value = self._pending.pop() if self._pending else self._proc._recv()
        if kind == "item":
            return value
        self._finish()
        if kind == "error":
            raise value
        raise StopIteration

    def close(self) -> None:
        if not self._open:
            return
        try:
            self._proc._conn.send(("cancel",))
            if self._pending and self._pending[0][0] != "item":
                self._pending.clear()
            else:
                while self._proc._recv()[0] == "item":
                    pass
        except StageProcessDied:
            pass
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._open:
            self._open = False
            self._proc._lock.release()

    def __del__(self):
        self.close()


class StageProcess:
    """Proxy for `factory(*args)` built in a spawned child process.

    `factory` and `args` must be picklable (a class and its config dataclass).
    Construction returns at once; the model loads in the background until the
    first use, or `wait_ready()`.
    """

    def __init__(self, factory, *args, name: str):
        from sts import log

        ctx = multiprocessing.get_context("spawn")
        self._name = name
        self._conn, child = ctx.Pipe()
        self._process = ctx.Process(
            target=_serve, args=(child, factory, args, log.is_verbose(), log._start), name=name, daemon=True
        )
        self._process.start()
        child.close()
        self._lock = threading.Lock()  # one request (or open stream) at a time
        self._methods: set[str] | None = None

    def wait_ready(self) -> None:
        with self._lock:
            self._ensure_ready()

    def _ensure_ready(self) -> None:
        if self._methods is None:
            kind, value = self._recv()
            if kind == "error":
                raise value
            self._methods = value

    def _recv(self):
        try:
            return self._conn.recv()
        except (EOFError, OSError) as exc:
            raise StageProcessDied(f"the {self._name} process exited (code {self._process.exitcode})") from exc

    def _request(self, kind: str, name: str, a=(), kw=None):
        self._lock.acquire()
        try:
            self._ensure_ready()
            self._conn.send((kind, name, a, kw or {}))
            reply = self._recv()
        except BaseException:
            self._lock.release()
            raise
        if reply[0] in ("item", "end"):
            return _RemoteStream(self, reply)  # releases the lock when done
        self._lock.release()
        if reply[0] == "error":
            raise reply[1]
        return reply[1]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        with self._lock:
            self._ensure_ready()
        if name in self._methods:
            return lambda *a, **kw: self._request("call", name, a, kw)
        return self._request("getattr", name)

    def close(self) -> None:
        self._conn.close()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
