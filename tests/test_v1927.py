"""1.9.27: esnek güç 2 (bellek + kullanıcı yokken) ve ajanla (Hermes) otomatik çalışma; işler asla kaybolmasın."""
import time

from stt.config import Config
from stt.sources import Item


# ---------------------------------------------------------------- 1. kalıcı kuyruk
def test_queue_upsert_keeps_added_and_tries(tmp_path):
    from stt.state import State
    st = State(tmp_path / ".stt")
    st.queue_add("a", "url", "https://x/1", "t1", {"lang": "en", "_gen": 3})
    st.queue_add("b", "url", "https://x/2", "t2", {})
    first = {r["uid"]: r for r in st.queue_all()}
    assert first["a"]["extra"] == {"lang": "en"} and first["a"]["tries"] == 0 and first["a"]["started"] is None
    st.queue_started("a", "u_key_a")
    assert st.queue_bump("a") == 1                                      # kirli kesinti sayıldı, started temizlendi
    st.queue_started("a", "u_key_a")
    time.sleep(0.01)
    st.queue_add("a", "url", "https://x/1", "yeni başlık", {"lang": "tr"})   # yeniden ekleme (motora verilirken)
    rows = st.queue_all()
    assert [r["uid"] for r in rows] == ["a", "b"]                       # sıra korunur
    a = rows[0]
    assert a["added"] == first["a"]["added"] and a["tries"] == 1 and a["started"] and a["key"] == "u_key_a"
    assert a["title"] == "yeni başlık" and a["extra"] == {"lang": "tr"}
    st.queue_unstart_all()
    assert all(r["started"] is None for r in st.queue_all())
    assert st.queue_count() == 2


def test_queue_migrates_old_schema(tmp_path):
    import sqlite3

    from stt.state import State
    d = tmp_path / ".stt"
    d.mkdir()
    db = sqlite3.connect(d / "durum.sqlite")
    db.execute("CREATE TABLE queue(uid TEXT PRIMARY KEY, kind TEXT, source TEXT, title TEXT, extra TEXT, added REAL)")
    db.execute("INSERT INTO queue VALUES('old','url','https://x/9','eski','{}',1.0)")
    db.commit()
    db.close()
    st = State(d)                                                       # 1.9.26 kuyruğu: sütunlar eklenir, iş kalır
    r = st.queue_all()[0]
    assert r["uid"] == "old" and r["tries"] == 0 and r["started"] is None and r["key"] is None
    st.queue_started("old", "u_k")
    assert st.queue_bump("old") == 1
    State(d)                                                            # ikinci açılış: yeniden eklemeye çalışmaz


def test_closing_pipeline_keeps_waiting_items(tmp_path):
    from stt.pipeline import Pipeline
    p = Pipeline(Config(out_dir=tmp_path))
    ev = []
    p.events = ev.append
    it = Item("url", "https://www.youtube.com/watch?v=abcdefghijk", title="t")
    it.extra["uid"] = "w1"
    p.state.queue_add("w1", "url", it.source, "t", {})
    p._closing = True                                                   # motor kapanıyor (hata / yeniden açılış)
    p._prepare(it)
    assert [r["uid"] for r in p.state.queue_all()] == ["w1"]            # sıra bekleyen iş silinmez, sonra sürer
    assert not any(e["type"] == "failed" for e in ev)
    p.emit("failed", it, error="iç hata")                              # kapanırken düşen iş de silinmez
    assert p.state.queue_count() == 1
    p.emit("done", it)                                                  # biten iş her zaman çıkar
    assert p.state.queue_count() == 0
    p._closing = False
    p.state.queue_add("w1", "url", it.source, "t", {})
    it.extra["_cancel"] = True                                          # kullanıcının "Durdur"u: silinir
    p._prepare(it)
    assert p.state.queue_count() == 0 and any(e["type"] == "failed" and e.get("cancelled") for e in ev)
    p.close()


def test_prepare_marks_queue_started(tmp_path, monkeypatch):
    import stt.pipeline as pl
    p = pl.Pipeline(Config(out_dir=tmp_path))
    p.events = lambda e: None
    it = Item("url", "https://www.youtube.com/watch?v=abcdefghijk", title="t")
    it.extra["uid"] = "s1"
    p.state.queue_add("s1", "url", it.source, "t", {})
    seen = {}

    def fake_prepare(item, work, cfg):
        seen.update(p.state.queue_all()[0])
        raise pl.SkipItem("ses yok")
    monkeypatch.setattr(pl, "prepare_audio", fake_prepare)
    p._prepare(it)
    assert seen["started"] and seen["key"] == it.key                     # iş başladı: kirli kesintide deneme sayılır
    assert p.state.queue_count() == 0                                   # kalıcı hata: kuyruktan çıktı
    p.close()


def test_clean_close_does_not_count_as_try(tmp_path):
    from stt.pipeline import Pipeline
    p = Pipeline(Config(out_dir=tmp_path))
    p.state.queue_add("k1", "url", "https://x/1", "t", {})
    p.state.queue_started("k1", "u_x")
    p.close()                                                           # motor düzgün kapandı (ayar değişti)
    assert p.state.queue_all()[0]["started"] is None
    q = Pipeline(Config(out_dir=tmp_path))
    q.state.queue_started("k1", "u_x")
    q.kill()                                                            # uygulama kapandı / güncelleme
    assert q.state.queue_all()[0]["started"] is None
    q.close()


def test_enqueue_persists_before_engine(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    r = app.start("https://www.youtube.com/watch?v=abcdefghijk", [], {"lang": "en", "diarize": True})
    rows = app._state().queue_all()
    assert [x["uid"] for x in rows] == r["uids"] and rows[0]["extra"]["lang"] == "en"
    app2 = g.App()                                                      # motor hiç çalışmadan uygulama kapandı
    assert [x["uid"] for x in app2._pending] == r["uids"]


def test_cancel_all_clears_queue(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    r = app.start("https://www.youtube.com/watch?v=abcdefghijk\nhttps://www.youtube.com/watch?v=bbbbbbbbbbb", [], {})
    assert app._state().queue_count() == 2
    assert app.cancel(r["uids"][0])["ok"]
    assert [x["uid"] for x in app._state().queue_all()] == [r["uids"][1]]
    app._pending = [{"uid": "eski"}]
    app.cancel_all()                                                    # "Durdur": hiçbir şey sonra sürmesin
    assert app._state().queue_count() == 0 and app._pending == []


# ---------------------------------------------------------------- 2. motor durunca işler sürer
def test_all_workers_dead_interrupts_without_deleting(tmp_path):
    import pytest
    from test_stability import _job, _pipe
    p = _pipe(tmp_path, ("cuda:1",))
    ev = []
    p.events = ev.append
    job, tasks = _job(p, tmp_path)
    p.state.queue_add("u1", "url", "https://x/1", "t", {})
    p.state.queue_started("u1", "u_test")
    p.pool.dead.add("cuda:1")
    p.fatal.add("cuda:1")                                               # yeniden açılamıyor (bellek yok / sürücü)
    with pytest.raises(RuntimeError):
        p._health()
    assert job.status == "interrupted"                                  # eskiden: "failed" + kalıcı kuyruktan silinirdi
    assert p.state.queue_count() == 1 and p.state.get("u_test")["status"] == "interrupted"
    assert any(e["type"] == "interrupted" for e in ev) and not any(e["type"] == "failed" for e in ev)
    p.pool = None                                                       # sahte havuzun close'u yok
    p.close()
    assert p.state.queue_all()[0]["started"]                            # kirli kesinti: sürdürmede deneme sayılacak


def test_resume_skips_inbox_pending_and_fails_poison(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    pushed = []
    real_push = app._push
    app._push = lambda ev: (pushed.append(ev), real_push(ev))
    d = app.start("https://www.youtube.com/watch?v=dddddddddd1", [], {})["uids"][0]   # zaten sırada (inbox)
    st = app._state()
    for uid, n in (("a", "aaaaaaaaaa1"), ("b", "bbbbbbbbbb1"), ("c", "cccccccccc1"), ("e", "eeeeeeeeee1")):
        st.queue_add(uid, "url", f"https://www.youtube.com/watch?v={n}", uid, {"lang": "en"})
    st.queue_add("f", "file", str(tmp_path / "silinmis.mp4"), "f", {})
    st.queue_started("a", "u_a")                                        # çökme anında işleniyordu
    st.queue_started("c", "u_c")
    st.queue_bump("c")
    st.queue_bump("c")
    st.queue_started("c", "u_c")                                        # 3. kez yarıda: zehirli video
    app._pending = [r for r in st.queue_all() if r["uid"] == "e"]       # kullanıcının "Devam et" kararını bekliyor
    app._resume_after_failure = True
    app._after_engine_start(first=False)
    inbox = [it.extra["uid"] for it in list(app._inbox.queue)]
    assert inbox == [d, "a", "b"]                                       # inbox'taki ikinci kez girmez, e beklemede
    rows = {r["uid"]: r for r in st.queue_all()}
    assert set(rows) == {d, "a", "b", "e"}                              # c (zehirli) ve f (dosya yok) çıktı
    assert rows["a"]["tries"] == 1 and rows["a"]["started"] is None and rows["b"]["tries"] == 0
    assert st.get("u_c")["status"] == "failed" and "3 kez" in st.get("u_c")["error"]
    assert any(e.get("type") == "failed" and e.get("uid") == "c" for e in pushed)
    assert app._inbox.queue[-1].extra["lang"] == "en"                   # iş ayarları korunur


class _FakePipe:
    made: list = []

    def __init__(self, cfg):
        from pathlib import Path

        from stt.state import State
        self.cfg, self.events, self.failed_reason = cfg, None, None
        self.state = State(Path(cfg.out_dir).resolve() / ".stt")
        self.devices, self.helper_devices, self._h_note, self.got = ["cpu"], [], {}, []
        _FakePipe.made.append(self)

    def start(self):
        pass

    def submit(self, it):
        self.got.append(it.extra["uid"])
        self.state.queue_started(it.extra["uid"], it.compute_key())

    def pump(self, t):
        time.sleep(0.005)
        if len(_FakePipe.made) == 1 and self.got:
            raise RuntimeError("Tüm işçiler durdu: test")               # ilk motor: işçiler öldü

    def active(self):
        return bool(self.got)

    def close(self):
        pass

    def kill(self):
        pass


def test_runner_restarts_engine_and_resumes(tmp_path, monkeypatch):
    import threading

    import stt.engines as eng
    import stt.gui as g
    import stt.pipeline as pl
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pl, "Pipeline", _FakePipe)
    monkeypatch.setattr(eng, "gpu_info", lambda *a, **k: {})
    monkeypatch.setattr(g, "ENGINE_RETRY_S", (0.05, 0.05))
    monkeypatch.setattr(g, "RECOVER_POLL_S", 0.01)
    _FakePipe.made = []
    app = g.App()
    states = []
    real_engine = app._engine
    app._engine = lambda **kw: (states.append(kw.get("state")), real_engine(**kw))
    uid = app.start("https://www.youtube.com/watch?v=abcdefghijk", [], {})["uids"][0]
    th = threading.Thread(target=app._runner, daemon=True)
    th.start()
    for _ in range(400):
        if len(_FakePipe.made) >= 2 and _FakePipe.made[1].got:
            break
        time.sleep(0.01)
    app._stop.set()
    app._restart.set()
    th.join(5)
    assert len(_FakePipe.made) >= 2 and _FakePipe.made[1].got == [uid]  # yeni motor işi kaldığı yerden aldı
    assert "error" in states and states[-1] in ("ready", "loading", "error")
    assert app._state().queue_all()[0]["tries"] == 1                    # kirli kesinti bir deneme sayıldı


def test_close_stops_downloads_of_old_engine(tmp_path):
    from stt import fetch
    from stt.pipeline import Job, Pipeline
    p = Pipeline(Config(out_dir=tmp_path))
    it = Item("url", "https://x/dl", key="u_dl")
    job = Job(it, tmp_path / "w_dl", status="prep")
    p.jobs[it.key] = job
    p.close()                                                           # eski motorun indirmesi yenisiyle çakışmasın
    assert str(job.work) in fetch.CANCELLED
    fetch.CANCELLED.discard(str(job.work))
