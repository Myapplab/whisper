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


# ---------------------------------------------------------------- 3. esnek güç 2: bellek + kullanıcı yokken
def test_flex_memory_escalates_at_once():
    from stt.flex import Flex
    f = Flex()
    msg = f.update(10, 8.0, 0, commit_free=3.5)                         # sanal bellek az: tek ölçüm yeter
    assert "kademe 0 → 1" in msg and f.level == 1 and f.ram_low and not f.mem_crit
    assert "kademe 1 → 2" in f.update(10, 8.0, 10, commit_free=1.5) and f.mem_crit
    g = Flex()
    g.update(10, 0.8, 0, commit_free=20)                                # RAM kritik
    assert g.level == 2 and g.mem_crit
    h = Flex()
    assert h.update(50, 8.0, 0, commit_free=20) == "" and h.level == 0  # işlemci: tek sıçrama yetmez (eskisi gibi)


def test_flex_away_ignores_cpu_not_memory():
    from stt.flex import Flex
    f = Flex()
    f.update(80, 8, 0, commit_free=20, idle_s=10)
    f.update(80, 8, 10, commit_free=20, idle_s=20)
    assert f.level == 2                                                 # kullanıcı başında: başka programlar yavaşlatır
    msg = f.update(80, 8, 20, commit_free=20, idle_s=400)               # 5 dk'dır fare / klavye yok
    assert f.level == 0 and "kimse yok" in msg and f.away
    assert f.update(90, 8, 30, commit_free=20, idle_s=410) == "" and f.level == 0   # yokken işlemci yavaşlatmaz
    f.update(90, 8, 40, commit_free=3.0, idle_s=420)
    assert f.level == 1                                                 # bellek koruması yokken de geçerli
    f.update(90, 8, 50, commit_free=20, idle_s=1)                       # kullanıcı döndü: eski esnek davranış
    assert f.update(90, 8, 60, commit_free=20, idle_s=2) and f.level == 2


def test_flex_calm_needs_commit():
    from stt.flex import CALM_S, Flex
    f = Flex()
    f.update(10, 8, 0, commit_free=3.0)
    f.update(5, 8, 10, commit_free=5.0)                                 # 4-6 GB arası: sakin sayılmaz
    assert f.update(5, 8, 10 + CALM_S + 1, commit_free=5.0) == "" and f.level == 1
    f.update(5, 8, 200, commit_free=7.0)
    assert "kademe 1 → 0" in f.update(5, 8, 200 + CALM_S, commit_free=7.0)


def test_user_idle_unknown_off_windows():
    import os

    from stt.flex import idle_from_ticks, user_idle_s
    if os.name != "nt":
        assert user_idle_s() is None                                    # bilinmiyor = kullanıcı var sayılır
    assert idle_from_ticks(10_000, 4_000) == 6.0
    assert idle_from_ticks(5, 0xFFFFFFF0) == 0.021                      # GetTickCount 49,7 günde bir başa döner


MEM_LOW = {"ram_total": 24, "ram_free": 6.0, "commit_total": 32, "commit_free": 1.5}
MEM_OK = {"ram_total": 24, "ram_free": 9.0, "commit_total": 48, "commit_free": 12.0}


def test_mem_crit_pauses_prep_even_flex_off(tmp_path):
    from test_v1921 import _flex_pipe
    p, job, tasks = _flex_pipe(tmp_path, 0)
    p.cfg.gpu_flex = False                                              # esnek kapalı: bellek koruması yine çalışır
    p._flex_at = 0
    p._flex_sampler = type("S", (), {"sample": lambda s: (5.0, 10.0)})()
    p._memstat = lambda: dict(MEM_LOW)
    p._idle = lambda: None
    p._prep_ahead = lambda: 0
    p._flex_tick()
    assert p.flex.mem_crit and not p._pending_ahead_ok()                # yeni video indirilmez
    assert p._flex_level() == 0                                         # esnek kapalı: kart dağıtımı değişmez
    p._memstat = lambda: dict(MEM_OK)
    p._flex_at = 0
    p._flex_tick()
    assert not p.flex.mem_crit and p._pending_ahead_ok()


def test_memory_pressure_notice_once_per_hour(tmp_path):
    from test_v1921 import _flex_pipe
    p, job, tasks = _flex_pipe(tmp_path, 0)
    ev = []
    p.events = ev.append
    p._flex_sampler = type("S", (), {"sample": lambda s: (5.0, 10.0)})()
    p._idle = lambda: None
    for mem in (MEM_LOW, MEM_OK, MEM_LOW):
        p._memstat = lambda m=mem: dict(m)
        p._flex_at = 0
        p._flex_tick()
    notes = [e for e in ev if e["type"] == "notice"]
    assert len(notes) == 1 and "sanal bellek" in notes[0]["text"].lower()
    assert "16384" in notes[0]["text"]                                  # sınır 32 GB: sayfa dosyası ipucu


def test_helper_not_opened_when_memory_low(tmp_path):
    from test_v199 import _pipe, _ready

    from stt.pipeline import HELPER_OFF
    p, job, tasks = _pipe(tmp_path, n=6)
    p._h_parked["cuda:0"] = HELPER_OFF
    p.load = {"util": 0, "used_mb": 500.0, "total_mb": 10240.0}        # kartta yer var
    p._memstat = lambda: dict(MEM_LOW)                                  # ama bilgisayarın belleği az
    _ready(p, "cuda:1")
    p._dispatch()
    p._dispatch()
    assert p.hpool.respawned == [] and "bellek" in p._h_note["cuda:0"]
    p._memstat = lambda: dict(MEM_OK)
    p._h_ready["cuda:0"] = 0
    p._dispatch()
    p._dispatch()
    assert p.hpool.respawned == ["cuda:0"]                              # bellek açılınca ikinci kart açılır


def test_hdiar_not_opened_when_memory_low(tmp_path, monkeypatch):
    from test_v199 import FakeDiarPool, _diar_pipe
    p, ds = _diar_pipe(tmp_path, monkeypatch)
    p.load = {"util": 0, "used_mb": 1500.0, "total_mb": 10240.0}
    p._memstat = lambda: dict(MEM_LOW)
    p._h_probe.clear()
    p._dispatch()
    p._dispatch()
    assert "diar2" in p._h_parked and FakeDiarPool.made == [] and "bellek" in p._h_note["diar2"]


def test_pagefile_notice_two_cards(tmp_path, monkeypatch):
    import stt.engines as eng
    import stt.pipeline as pl
    import stt.power as pw
    from test_v1917 import _cards
    from test_v1921 import GI_FREE

    class P:
        def __init__(self, devices, ecfg, *a, **k):
            self.workers, self.load_lock = {d: d for d in devices}, None

        def start(self):
            pass
    monkeypatch.setattr(eng, "gpu_info", lambda *a, **k: dict(GI_FREE))
    monkeypatch.setattr(eng, "resolve_model_path", lambda m: "m")
    monkeypatch.setattr(pl, "WorkerPool", P)
    monkeypatch.setattr(pw, "query", lambda: _cards(224, 245))
    monkeypatch.setattr(pw, "reapply", lambda *a, **k: False)
    for total, want in ((32.0, 1), (56.0, 0)):
        p = pl.Pipeline(Config(out_dir=tmp_path / str(total)))
        ev = []
        p.events = ev.append
        p._memstat = lambda t=total: {"ram_total": 24, "ram_free": 9, "commit_total": t, "commit_free": 12}
        p.start()
        p.prep.shutdown(wait=False)
        p.fin.shutdown(wait=False)
        notes = [e for e in ev if e["type"] == "notice" and "sanal bellek" in e["text"].lower()]
        assert len(notes) == want


# ---------------------------------------------------------------- 4. uzun çalışma: yenileme + verim
def test_recycle_runs_outside_night_mode(tmp_path):
    from test_stability import _job, _pipe
    p = _pipe(tmp_path)                                                 # gece modu kapalı (esnek / normal)
    job, tasks = _job(p, tmp_path)
    p.backlog.clear()
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._born["cuda:1"] = time.time() - 181 * 60
    p._recycle_tick()
    assert p.pool.respawned == ["cuda:1"] and p.pool.respawn_idle == [True]   # 3 saattir açık, boşta: yenilendi
    q = _pipe(tmp_path / "b")
    q.cfg.recycle_min = 0                                               # kapalı
    _job(q, tmp_path / "b")
    q._on_msg({"type": "ready", "wid": "cuda:1"})
    q._born["cuda:1"] = 0
    q._recycle_tick()
    assert q.pool.respawned == []


def test_recycle_parks_old_idle_helper(tmp_path):
    from test_v199 import _pipe, _ready
    p, job, tasks = _pipe(tmp_path, n=1)
    p.backlog.clear()
    _ready(p, "cuda:1", "cuda:0")                                       # ikinci kart açık ve boşta
    p._born["cuda:0"] = time.time() - 200 * 60
    p._recycle_tick()
    assert "cuda:0" in p.hpool.stopped and "cuda:0" in p._h_parked     # kapatıldı; iş birikince yeniden açılır
    assert p.pool.respawned == []                                       # ana kart yeni açıldı


def test_stats_and_hourly_line(tmp_path, caplog):
    import logging

    from test_stability import _pipe
    p = _pipe(tmp_path)
    it = Item("url", "https://x/s1", key="u_s1")
    it.extra["uid"] = "s1"
    p.emit("done", it, duration=600.0)
    p.emit("done", it, duration=1200.0)
    p.emit("failed", it, error="indirilemedi")
    p.emit("failed", it, error="Durduruldu.", cancelled=True)          # iptal hata sayılmaz
    st = p.stats()
    assert st["done_1h"] == 2 and st["audio_min_1h"] == 30.0 and st["failed_1h"] == 1
    assert st["done_total"] == 2 and st["failed_total"] == 1
    p._stat_hour["t0"] -= 3601
    with caplog.at_level(logging.INFO, logger="stt"):
        p._stats_tick()
    assert any("[verim]" in r.message and "2 video" in r.message for r in caplog.records)
    assert time.time() - p._stat_hour["t0"] < 5                         # yeni saat başladı


# ---------------------------------------------------------------- 5. ajan modu, Windows açılışı, güncelleme
class _FakeReg:
    def __init__(self):
        self.d = {}

    def get(self, key, name):
        return self.d.get((key, name))

    def set(self, key, name, value):
        self.d[(key, name)] = value

    def delete(self, key, name):
        self.d.pop((key, name), None)


def test_autostart_toggle_with_fake_registry(monkeypatch):
    import os

    from stt import autostart as a
    reg = _FakeReg()
    monkeypatch.setattr(a, "REG", reg)
    assert a.status()["enabled"] is False
    assert a.set_enabled(True) == ""
    st = a.status()
    assert st["enabled"] and st["current"] and st["approved"] is None
    assert st["command"].endswith('" -m stt.gui --baslangic') and st["command"].startswith('"')
    reg.d[(a.APPROVED_KEY, a.VALUE)] = bytes([3] + [0] * 11)            # Görev Yöneticisi'nde kapatıldı
    assert a.status()["approved"] is False
    reg.d[(a.RUN_KEY, a.VALUE)] = '"C:\\eski\\pythonw.exe" -m stt.gui --baslangic'
    a.ensure()
    assert reg.d[(a.RUN_KEY, a.VALUE)].startswith('"C:\\eski')          # kullanıcının kapattığına dokunulmaz
    reg.d[(a.APPROVED_KEY, a.VALUE)] = bytes([2] + [0] * 11)
    a.ensure()
    assert reg.d[(a.RUN_KEY, a.VALUE)] == a.command()                   # kurulum yolu değişmişse düzeltilir
    assert a.set_enabled(False) == "" and a.status()["enabled"] is False
    monkeypatch.setattr(a, "REG", None)
    if os.name != "nt":
        assert a.status()["supported"] is False and a.set_enabled(True)


def _auto_app(tmp_path, monkeypatch, **cfg):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    store = {"out_dir": tmp_path / "cikti", **cfg}
    monkeypatch.setattr(g, "save_settings", lambda d: store.update(d))
    monkeypatch.setattr(g, "load_config", lambda: Config(**store))
    app = g.App()
    app._start_api = lambda: setattr(app, "_api_started", app._cfg.api_port)
    return g, app, store


def test_set_auto_mode_opens_api_and_autostart(tmp_path, monkeypatch):
    from stt import autostart as a
    reg = _FakeReg()
    monkeypatch.setattr(a, "REG", reg)
    g, app, store = _auto_app(tmp_path, monkeypatch)
    r = app.set_auto_mode(True)
    assert r["ok"] and store["auto_mode"] is True and store["api_port"] == 8765 and app._api_started == 8765
    assert reg.d[(a.RUN_KEY, a.VALUE)] == a.command()                   # Windows açılınca başlar
    s = app.get_settings()
    assert s["auto_mode"] is True and s["autostart"]["enabled"] is True
    app.set_auto_mode(False)
    assert store["auto_mode"] is False and (a.RUN_KEY, a.VALUE) not in reg.d
    assert store["api_port"] == 8765                                    # API açık kalır (başka program kullanıyor olabilir)


def test_note_unclean_end_counts_and_closes_window(tmp_path):
    import json

    from stt.gpuguard import mark_busy, note_unclean_end
    st = tmp_path / ".stt"
    assert note_unclean_end(st, now=1000.0) == 0                        # ilk kurulum: kayıt yok
    mark_busy(st, True)                                                 # iş sürerken uygulama çöktü (aralık açık)
    assert note_unclean_end(st, now=2000.0) == 1
    wins = json.loads((st / "isleme.json").read_text())["windows"]
    assert wins[-1][1] == 2000                                          # açık aralık kapatıldı: bir daha sayılmaz
    assert note_unclean_end(st, now=2100.0) == 1
    mark_busy(st, True)
    assert note_unclean_end(st, now=3000.0) == 2                        # 24 saatte ikinci kirli kapanış
    mark_busy(st, True)
    assert note_unclean_end(st, now=3000.0 + 2 * 86400) == 1            # eskiler 24 saat dışında


def test_auto_resume_on_first_start(tmp_path, monkeypatch):
    g, app, store = _auto_app(tmp_path, monkeypatch, auto_mode=True)
    st = app._state()
    st.queue_add("p1", "url", "https://www.youtube.com/watch?v=abcdefghijk", "eski iş", {"lang": "en"})
    app2 = g.App()                                                      # yeniden açılış (ör. Windows güncellemesi)
    pushed = []
    app2._push = pushed.append
    assert [r["uid"] for r in app2._pending] == ["p1"]
    app2._after_engine_start(first=True)
    assert [it.extra["uid"] for it in list(app2._inbox.queue)] == ["p1"] and app2._pending == []
    assert any(e.get("type") == "pending" and e.get("pending") == [] for e in pushed)
    store["auto_mode"] = False                                          # ajan modu kapalı: "Devam et" beklenir
    app3 = g.App()
    app3._after_engine_start(first=True)
    assert app3._inbox.empty() and [r["uid"] for r in app3._pending] == ["p1"]


def test_crash_loop_pauses_auto_resume(tmp_path, monkeypatch):
    from stt.gpuguard import mark_busy
    g, app, store = _auto_app(tmp_path, monkeypatch, auto_mode=True)
    sd = app._state_dir()
    app._state().queue_add("p1", "url", "https://www.youtube.com/watch?v=abcdefghijk", "iş", {})
    for _ in range(2):                                                  # 24 saatte iki kez iş yaparken çöktü
        mark_busy(sd, True)
        g.App()
    mark_busy(sd, True)
    app2 = g.App()
    assert app2._unclean_24h >= 2
    app2._after_engine_start(first=True)
    assert app2._inbox.empty() and app2._auto_paused                    # sürdürülmedi, ajandan iş alınmıyor
    app2.resume_pending()                                               # kullanıcı bakıp "Devam et"e bastı
    assert app2._auto_paused == "" and [it.extra["uid"] for it in list(app2._inbox.queue)] == ["p1"]


def test_update_while_busy_keeps_queue(tmp_path, monkeypatch):
    import stt.updater as up
    g, app, store = _auto_app(tmp_path, monkeypatch, auto_mode=True)
    uid = app.start("https://www.youtube.com/watch?v=abcdefghijk", [], {})["uids"][0]
    killed, relaunched, shut = [], [], []

    class P:
        def active(self):
            return True

        def kill(self):
            killed.append(1)
    app._pipe = P()
    monkeypatch.setattr(up, "apply", lambda progress: {"ok": True, "version": "9.9.9"})
    monkeypatch.setattr(up, "relaunch", lambda: relaunched.append(1))
    monkeypatch.setattr(g, "_shutdown", lambda a: shut.append(1))
    monkeypatch.setattr(g, "RELAUNCH_DELAY_S", 0.0)
    r = app.update_apply()
    for _ in range(200):
        if shut:
            break
        time.sleep(0.01)
    assert r["ok"] and killed == [1] and app._stop.is_set() and relaunched == [1] and shut == [1]
    from stt.state import State
    assert [x["uid"] for x in State(tmp_path / "cikti" / ".stt").queue_all()] == [uid]   # kuyruk korundu


# ---------------------------------------------------------------- 6. yerel API
def _api(app):
    import json as _j
    import socket
    import urllib.error
    import urllib.request

    from stt.api import start_api
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = start_api(app, port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def req(path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (_j.dumps(body).encode() if body is not None else None)
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                   headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with opener.open(r, timeout=10) as resp:
                return resp.status, _j.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, _j.loads(e.read() or b"{}")
    return srv, req


def test_api_rejects_browser_and_rebinding(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    srv, req = _api(app)
    try:
        assert req("/api/status")[0] == 200                             # yerel program (Hermes, curl)
        assert req("/api/status", headers={"Host": "evil.example:8765"})[0] == 403        # DNS yeniden bağlama
        assert req("/api/jobs", {"inputs": ["https://www.youtube.com/watch?v=abcdefghijk"]},
                   headers={"Origin": "https://evil.example"})[0] == 403                # web sayfası iş ekleyemez
        assert req("/api/status", headers={"Origin": "null"})[0] == 403
        assert req("/api/status", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
        assert req("/api/status", headers={"Origin": "http://localhost:3000"})[0] == 200
        assert app._inbox.empty()
    finally:
        srv.shutdown()


def test_api_jobs_busy_is_503(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    srv, req = _api(app)
    try:
        app._training = True                                            # eğitim sürüyor: eskiden 200 + ok:false (link kayboluyordu)
        code, r = req("/api/jobs", {"inputs": ["https://www.youtube.com/watch?v=abcdefghijk"]})
        assert code == 503 and r["retry"] is True and app._inbox.empty() and app._state().queue_count() == 0
        app._training = False
        app._auto_paused = "çökme döngüsü"
        code, r = req("/api/jobs", {"inputs": ["https://www.youtube.com/watch?v=abcdefghijk"]})
        assert code == 503 and "çökme" in r["error"] and app._inbox.empty()
        app._auto_paused = ""
        code, r = req("/api/jobs", {"inputs": ["https://www.youtube.com/watch?v=abcdefghijk"], "options": {"lang": "en"}})
        assert code == 200 and r["ok"] and r["count"] == 1 and len(r["uids"]) == 1
    finally:
        srv.shutdown()


def test_api_jobs_bad_input_4xx(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    srv, req = _api(app)
    try:
        code, r = req("/api/jobs", {"inputs": ["https://www.youtube.com/@Arsenal"]})
        assert code == 422 and r["ok"] is False                        # kanal linki: /api/channel
        assert req("/api/jobs", {"inputs": []})[0] == 400
        assert req("/api/jobs", raw=b"{bozuk")[0] == 400
        code, r = req("/api/jobs", {"url": "https://youtu.be/abcdefghijk"})   # tek link alanı da olur
        assert code == 200 and r["count"] == 1
    finally:
        srv.shutdown()


def test_api_status_queue_fields(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    srv, req = _api(app)
    try:
        req("/api/jobs", {"inputs": ["https://www.youtube.com/watch?v=abcdefghijk", "https://www.youtube.com/watch?v=bbbbbbbbbbb"]})
        app._push({"type": "notice", "text": "Bilgisayarın belleği azaldı"})
        code, st = req("/api/status")
        assert code == 200 and st["queue"] == {"total": 2, "working": 0, "waiting": 2}
        assert st["accepting"] is True and st["auto_mode"] is False and st["paused"] is None
        assert st["notices"][-1]["text"] == "Bilgisayarın belleği azaldı" and st["version"]
        assert "commit_free" in (st["memory"] or {"commit_free": 0})
        app._training = True
        assert req("/api/status")[1]["accepting"] is False
    finally:
        srv.shutdown()


def test_api_lookup_statuses(tmp_path, monkeypatch):
    import stt.gui as g
    from stt.sources import Item
    monkeypatch.chdir(tmp_path)
    app = g.App()
    st = app._state()
    U = {n: f"https://www.youtube.com/watch?v={n * 11}" for n in "abcde"}
    key = {n: Item("url", u).compute_key() for n, u in U.items()}
    out = tmp_path / "out" / "a"
    out.mkdir(parents=True)
    st.set(key["a"], source=U["a"], status="done", title="A", out_dir=str(out), info={"json": str(out / "a.json")})
    st.set(key["b"], source=U["b"], status="failed", title="B", error="Video bulunamadı ya da kaldırılmış.")
    st.set(key["c"], source=U["c"], status="failed", title="C", error="İnternet bağlantısı sorunu. (timed out)")
    st.queue_add("q", "url", U["d"], "D", {})
    srv, req = _api(app)
    try:
        code, r = req("/api/lookup", {"urls": [U[n] for n in "abcde"] + ["https://youtu.be/aaaaaaaaaaa"]})
        assert code == 200
        by = [(x["status"], x["retryable"]) for x in r["items"]]
        assert by == [("done", None), ("failed", False), ("failed", True), ("waiting", None), ("unknown", None),
                      ("done", None)]                                    # kısa link de aynı video
        a = r["items"][0]
        assert a["out_dir"] == str(out) and a["txt"].endswith("a.txt") and a["title"] == "A"
        code, r = req("/api/lookup?url=" + U["b"].replace("?", "%3F").replace("=", "%3D"))
        assert code == 200 and r["items"][0]["status"] == "failed"
    finally:
        srv.shutdown()
