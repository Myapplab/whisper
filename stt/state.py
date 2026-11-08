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
        self._db.execute("PRAGMA synchronous=NORMAL")  # WAL'da güvenli; yüzlerce iş eklerken her kayıt fsync etmesin
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS items(
                key TEXT PRIMARY KEY, source TEXT, status TEXT, title TEXT,
                out_dir TEXT, error TEXT, info TEXT, updated REAL)"""
        )
        self._db.execute("CREATE TABLE IF NOT EXISTS alias(alias TEXT PRIMARY KEY, key TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY, size INT, mtime INT, key TEXT)")
        # bitmemiş işler: uygulama kapanırsa / bilgisayar yeniden başlarsa bir sonraki açılışta sürdürülür.
        # 1.9.27: key (işin anahtarı), started (işe başlandı: temiz kapanışta silinir, çökmede kalır), tries (kaç kez
        # kirli kesintiyle yarıda kaldı; aynı video uygulamayı üst üste çökertirse sonsuz döngü olmasın)
        self._db.execute("CREATE TABLE IF NOT EXISTS queue(uid TEXT PRIMARY KEY, kind TEXT, source TEXT, "
                         "title TEXT, extra TEXT, added REAL, key TEXT, started REAL, tries INTEGER DEFAULT 0)")
        have = {r[1] for r in self._db.execute("PRAGMA table_info(queue)")}
        for col, typ in (("key", "TEXT"), ("started", "REAL"), ("tries", "INTEGER DEFAULT 0")):
            if col not in have:  # 1.9.26 ve öncesinin kuyruğu: işler kalır, sütun eklenir
                self._db.execute(f"ALTER TABLE queue ADD COLUMN {col} {typ}")
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
        try:
            d["info"] = json.loads(d["info"] or "{}")
        except ValueError:  # bozuk kayıt uygulamayı düşürmesin
            d["info"] = {}
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

    def done_keys(self, keys: list[str]) -> set[str]:
        """Verilen anahtarlardan bitmiş olanlar (takma adlar dahil) — tek sorgu grubu, klasörü duran."""
        keys = list(dict.fromkeys(keys))
        if not keys:
            return set()
        out: set[str] = set()
        with self._lock:
            for a in range(0, len(keys), 500):
                part = keys[a:a + 500]
                q = ",".join("?" * len(part))
                alias = dict(self._db.execute(f"SELECT alias, key FROM alias WHERE alias IN ({q})", part).fetchall())
                real = {k: alias.get(k, k) for k in part}
                rq = ",".join("?" * len(set(real.values())))
                rows = self._db.execute(f"SELECT key, out_dir FROM items WHERE status='done' AND key IN ({rq})",
                                        list(set(real.values()))).fetchall()
                ok = {k for k, d in rows if d and Path(d).exists()}
                out |= {k for k, r in real.items() if r in ok}
        return out

    # ---------------------------------------------------------- kalıcı kuyruk
    def queue_add(self, uid: str, kind: str, source: str, title: str, extra: dict) -> None:
        """İşi kalıcı kuyruğa yaz. Aynı iş yeniden yazılırsa (arayüz -> motor, sürdürme) sırası (added), deneme sayısı
        ve başlama işareti korunur."""
        ex = {k: v for k, v in extra.items() if not k.startswith("_")}
        with self._lock:
            self._db.execute(
                "INSERT INTO queue(uid,kind,source,title,extra,added) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(uid) DO UPDATE SET kind=excluded.kind, source=excluded.source, title=excluded.title, "
                "extra=excluded.extra",
                (uid, kind, source, title, json.dumps(ex, ensure_ascii=False, default=str), time.time()))
            self._db.commit()

    def queue_started(self, uid: str, key: str | None) -> None:
        """İşe başlandı (indirme / çeviri): bu sırada uygulama çökerse sürdürmede deneme sayılır."""
        with self._lock:
            self._db.execute("UPDATE queue SET started=?, key=COALESCE(?, key) WHERE uid=?", (time.time(), key, uid))
            self._db.commit()

    def queue_unstart(self, uid: str) -> None:
        with self._lock:
            self._db.execute("UPDATE queue SET started=NULL WHERE uid=?", (uid,))
            self._db.commit()

    def queue_unstart_all(self) -> None:
        """Temiz kapanış (uygulama kapatıldı, güncelleme, motor yeniden açılışı): yarım işler deneme sayılmaz."""
        with self._lock:
            self._db.execute("UPDATE queue SET started=NULL WHERE started IS NOT NULL")
            self._db.commit()

    def queue_bump(self, uid: str) -> int:
        """Kirli kesintiyle (çökme / motor durdu) yarıda kalan işin deneme sayısını artır; yeni sayı."""
        with self._lock:
            self._db.execute("UPDATE queue SET tries=COALESCE(tries,0)+1, started=NULL WHERE uid=?", (uid,))
            self._db.commit()
            row = self._db.execute("SELECT tries FROM queue WHERE uid=?", (uid,)).fetchone()
        return int(row[0] or 0) if row else 0

    def queue_count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM queue").fetchone()[0])

    def queue_remove(self, uid: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM queue WHERE uid=?", (uid,))
            self._db.commit()

    def queue_all(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT uid,kind,source,title,extra,added,key,started,tries FROM queue "
                                    "ORDER BY added").fetchall()
        out = []
        for uid, kind, source, title, extra, added, key, started, tries in rows:
            try:
                ex = json.loads(extra or "{}")
            except ValueError:
                ex = {}
            out.append({"uid": uid, "kind": kind, "source": source, "title": title, "extra": ex, "added": added,
                        "key": key, "started": started, "tries": int(tries or 0)})
        return out

    def queue_clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM queue")
            self._db.commit()

    def all(self) -> list[dict]:
        with self._lock:
            keys = [r[0] for r in self._db.execute("SELECT key FROM items ORDER BY updated")]
        return [self.get(k) for k in keys]  # type: ignore[misc]
