"""Remembers a handful of operator choices (ASR engine, audio devices, active
voice) across restarts of the live `sts` app, in a small SQLite key-value
store - so the app comes back up the way the operator last left it instead of
always resetting to the built-in defaults. Just a flat settings table:
there's no schema to migrate and nothing relational about a handful of scalar
preferences.
"""

import sqlite3
import threading


class PersistentConfig:
    def __init__(self, db_path: str):
        # check_same_thread=False: writes come from whichever thread handled
        # the operator action (a websocket-control thread, or one of the
        # background threads handle_control spawns for a slow device/engine
        # switch) - serialized by our own lock rather than sqlite3's default
        # same-thread restriction.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
