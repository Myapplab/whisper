"""İşlenmiş öğelerin kaydı (SQLite). Hash ile atlama + yarıda kalanı sürdürme."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class State:
    def __init__(self, state_dir: Path):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "durum.sqlite"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS items(
                key TEXT PRIMARY KEY, source TEXT, status TEXT, title TEXT,
                out_dir TEXT, error TEXT, info TEXT, updated REAL)"""
        )
        self._db.execute("CREATE TABLE IF NOT EXISTS alias(alias TEXT PRIMARY KEY, key TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY, size INT, mtime INT, key TEXT)")
        self._db.commit()

    def _resolve(self, key: str) -> str:
        row = self._db.execute("SELECT key FROM alias WHERE alias=?", (key,)).fetchone()
        return row[0] if row else key

    def get(self, key: str) -> dict | None:
        with self._lock:
            k = self._resolve(key)
            row = self._db.execute(
                "SELECT key,source,status,title,out_dir,error,info,updated FROM items WHERE key=?", (k,)
            ).fetchone()
        if not row:
            return None
        d = dict(zip(("key", "source", "status", "title", "out_dir", "error", "info", "updated"), row))
        d["info"] = json.loads(d["info"] or "{}")
        return d

    def is_done(self, key: str) -> bool:
        r = self.get(key)
        return bool(r and r["status"] == "done" and r["out_dir"] and Path(r["out_dir"]).exists())

    def set(self, key: str, *, source: str, status: str, title: str = "", out_dir: str = "",
            error: str = "", info: dict | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO items VALUES(?,?,?,?,?,?,?,?)",
                (key, source, status, title, out_dir, error, json.dumps(info or {}, ensure_ascii=False), time.time()),
            )
            self._db.commit()

    def add_alias(self, alias: str, key: str) -> None:
        if alias == key:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO alias VALUES(?,?)", (alias, key))
            self._db.commit()

    def file_key(self, path: str, compute) -> str:
        """Dosya hash'ini (yol, boyut, mtime) ile önbellekle; değişmeyen büyük dosya yeniden okunmaz."""
        st = Path(path).stat()
        with self._lock:
            row = self._db.execute("SELECT size, mtime, key FROM hashes WHERE path=?", (path,)).fetchone()
        if row and row[0] == st.st_size and row[1] == st.st_mtime_ns:
            return row[2]
        key = compute()
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO hashes VALUES(?,?,?,?)", (path, st.st_size, st.st_mtime_ns, key))
            self._db.commit()
        return key

    def all(self) -> list[dict]:
        with self._lock:
            keys = [r[0] for r in self._db.execute("SELECT key FROM items ORDER BY updated")]
        return [self.get(k) for k in keys]  # type: ignore[misc]
