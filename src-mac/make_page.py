"""Runs `make` (build Live Translate.app and install it) and shows its output
live on a web page - what the Claude app's "make" and "make install" server
entries start (.claude/launch.json), so the tab they open shows the build
instead of nothing. Standard library only.

    python3 src-mac/make_page.py [port] [target]     (default 8765, or $PORT; no target: `make`)

The page follows the output as it comes, says whether it succeeded, and has
a "Run again" button; the server keeps running until it's stopped.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# The make target to run, if any: read here, as the first run starts on import.
TARGET = sys.argv[2:3]
COMMAND = " ".join(["make", *TARGET])


class Run:
    """One `make` run: its output lines, and how it ended."""

    def __init__(self):
        self.lines: list[str] = []
        self.status = "running"  # running, ok, failed
        self.code: int | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        process = subprocess.Popen(
            ["make", *TARGET], cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        )
        pending = b""
        while chunk := process.stdout.read(4096):
            pending += chunk
            *complete, pending = re.split(rb"\r?\n", pending)
            for raw in complete:
                self._add(raw)
        if pending:
            self._add(pending)
        code = process.wait()
        with self.lock:
            self.code, self.finished = code, time.time()
            self.status = "ok" if code == 0 else "failed"

    def _add(self, raw: bytes):
        # A progress bar redraws its line with \r: keep only what it ended as.
        line = ANSI.sub("", raw.decode(errors="replace").split("\r")[-1])
        with self.lock:
            self.lines.append(line)

    def since(self, start: int) -> dict:
        with self.lock:
            return {
                "lines": self.lines[start:], "next": len(self.lines), "status": self.status, "code": self.code,
                "seconds": round((self.finished or time.time()) - self.started),
            }


run = Run()
run_lock = threading.Lock()

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>COMMAND - Live Translate</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0b0b0f; color: #ddd; font: 14px -apple-system, system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  header { display: flex; align-items: center; gap: 1em; padding: 0.7em 1em; background: #111116;
           border-bottom: 1px solid #2a2a30; }
  h1 { font-size: 1em; margin: 0; font-weight: 600; }
  #status { padding: 0.2em 0.7em; border-radius: 999px; font-size: 0.85em; background: #333; }
  #status.running { background: #4d3a1f; color: #ffe9bf; }
  #status.ok { background: #1f4d2b; color: #d7ffdf; }
  #status.failed { background: #5a1f1f; color: #ffd7d7; }
  #elapsed { color: #888; font-size: 0.85em; }
  button { margin-left: auto; font: inherit; padding: 0.4em 1em; border-radius: 6px; border: 1px solid #444;
           background: #1a1a1e; color: #ccc; cursor: pointer; }
  button:disabled { opacity: 0.4; cursor: default; }
  pre { flex: 1; margin: 0; padding: 0.8em 1em; overflow: auto; font: 12px ui-monospace, Menlo, monospace;
        white-space: pre-wrap; word-break: break-word; }
</style></head>
<body>
<header><h1>COMMAND: build and install Live Translate.app</h1><span id="status">running</span>
<span id="elapsed"></span><button id="again" disabled>Run again</button></header>
<pre id="out"></pre>
<script>
  const out = document.getElementById('out'), statusEl = document.getElementById('status');
  const elapsed = document.getElementById('elapsed'), again = document.getElementById('again');
  let next = 0, follow = true;
  // Following the end until scrolled up; scrolled back to the bottom, it follows again.
  out.addEventListener('scroll', () => { follow = out.scrollTop + out.clientHeight >= out.scrollHeight - 20; });
  const LABEL = { running: 'running', ok: 'done', failed: 'failed' };
  async function poll() {
    try {
      const r = await (await fetch('/log?from=' + next)).json();
      if (r.next < next) { out.textContent = ''; next = 0; return poll(); }  // a new run started
      if (r.lines.length) out.append(r.lines.join('\\n') + '\\n');
      next = r.next;
      if (follow) out.scrollTop = out.scrollHeight;
      statusEl.className = r.status;
      statusEl.textContent = LABEL[r.status] + (r.status === 'failed' ? ` (exit ${r.code})` : '');
      elapsed.textContent = r.seconds + ' s';
      again.disabled = r.status === 'running';
      document.title = (r.status === 'running' ? '… ' : r.status === 'ok' ? '✓ ' : '✗ ') + 'COMMAND - Live Translate';
    } catch (e) { statusEl.textContent = 'not running'; statusEl.className = 'failed'; }
    setTimeout(poll, 500);
  }
  again.addEventListener('click', async () => {
    again.disabled = true;
    await fetch('/run', { method: 'POST' });
    out.textContent = ''; next = 0; follow = true;
  });
  poll();
</script>
</body></html>
""".replace("COMMAND", COMMAND)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/log"):
            try:
                start = int(self.path.split("from=", 1)[1]) if "from=" in self.path else 0
            except ValueError:
                start = 0
            self._send(200, "application/json", json.dumps(run.since(start)))
        else:
            self._send(200, "text/html; charset=utf-8", PAGE)

    def do_POST(self):
        global run
        if self.path == "/run":
            with run_lock:
                if run.status != "running":
                    run = Run()
            self._send(200, "application/json", "{}")
        else:
            self._send(404, "text/plain", "not found")

    def _send(self, code, kind, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # no per-request lines in the server log
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8765))
    print(f"{COMMAND} output: http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
