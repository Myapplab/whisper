"""1.9.19: gece modu (yavaş ama dengeli, ısıya göre kendini ayarlar) + disk koruması."""
import time
from collections import deque
from pathlib import Path

from stt import governor as G
from stt.config import Config, normalize
from stt.governor import Governor
from stt.pipeline import HELPER_DIAR, HELPER_OFF, NIGHT_NOTE, Job, Pipeline
from stt.sources import Item

from test_stability import FakePool
from test_v199 import FakeHelperPool


# ---------------------------------------------------------------- Governor (saf mantık)
def test_governor_rest_grows_when_hot_and_shrinks_when_cool():
    g = Governor(72)
    assert g.rest == G.REST_MIN and not g.holding()
    g.update(75, 0)
    assert g.rest > G.REST_MIN
    r = g.rest
    g.update(77, 20)
    assert g.rest > r                       # hâlâ hedefin üstünde: mola uzar
    for i in range(40):
        g.update(60, 40 + i * 20)          # serin: en kısa molaya iner
    assert g.rest == G.REST_MIN
    for i in range(200):
        g.update(79, 1000 + i)             # hedefin üstünde ama soğuma eşiğinin altında: tavan
    assert g.rest == G.REST_MAX and not g.holding()


def test_governor_cooling_pause_and_resume():
    g = Governor(72)
    msg = g.update(g.hot, 0)
    assert g.holding() and "soğuma" in msg
    assert g.update(g.cool + 1, 30) == "" and g.holding()   # henüz soğumadı
    msg = g.update(g.cool, 60)
    assert not g.holding() and "bitti" in msg
    assert g.rest >= G.REST_MIN * 2


def test_governor_pause_never_blocks_forever():
    g = Governor(72)
    g.update(85, 0)
    assert g.holding()
    msg = g.update(84, G.PAUSE_MAX_S + 1)
    assert not g.holding() and g.rest == G.REST_MAX and "dakikadır" in msg   # en uzun molayla devam
    assert g.update(84, G.PAUSE_MAX_S + 30) == ""                            # tekrar tekrar yazılmaz
    g.update(g.cool, G.PAUSE_MAX_S + 60)
    assert not g.paused


def test_governor_without_temperature_uses_fixed_rest():
    g = Governor(72)
    g.update(90, 0)
    assert g.holding()
    g.update(None, 10)                      # tek okuma hatası: soğuma molası sürer
    g.update(None, 30)
    assert g.holding() and g.temp == 90
    g.update(None, 50)                      # art arda 3: nvidia-smi gitti, sonsuza dek beklemesin
    assert not g.holding() and g.rest == G.REST_BLIND
    assert "okunamıyor" in g.note()


def test_config_night_bounds():
    c = normalize(Config(night_temp=40, night_recycle_min=5, min_free_gb=-3))
    assert c.night_temp == 60 and c.night_recycle_min == 20 and c.min_free_gb == 0
    c = normalize(Config(night_temp=99, night_recycle_min=0))
    assert c.night_temp == 83 and c.night_recycle_min == 0
    assert Config().night_mode is False and Config().min_free_gb == 3.0


# ---------------------------------------------------------------- pipeline
def _pipe(tmp_path: Path, n=4, night=True, **cfg):
    p = Pipeline(Config(out_dir=tmp_path, night_mode=night, min_free_gb=0, **cfg))
    p.pool = FakePool(["cuda:1"])
    p.pool._spec = {"cuda:1": "cuda:1"}
    p.devices = ["cuda:1"]
    p._ecfg = {"gpu_mem": {1: 24.0}, "display_gpus": []}
    p._gi = {1: {"name": "RTX 3090", "gb": 24, "display": False}}
    p._spawn = lambda fn: fn()
    p.temps = []
    p._card_query = lambda: {1: {"temp": p.temps.pop(0) if p.temps else None}}
    item = Item("url", "https://x/1", key="u_test")
    item.extra["uid"] = "u1"
    job = Job(item, tmp_path / "w", status="run", info={"duration_s": 1200})
    (job.work / "parcalar").mkdir(parents=True)
    p.jobs[item.key] = job
    tasks = [{"id": f"u_test:t:{i:04d}", "kind": "transcribe", "job": item.key, "start": i * 300.0,
              "end": (i + 1) * 300.0, "idx": i, "n": n} for i in range(n)]
    for t in tasks:
        p.tasks[t["id"]] = t
    job.pending = {t["id"] for t in tasks}
    p.backlog = deque(tasks)
    p.events_seen = []
    p.events = p.events_seen.append
    return p, job, tasks


def _finish(p, wid, tid):
    p._on_msg({"type": "start", "wid": wid, "tid": tid})
    p._on_msg({"type": "done", "wid": wid, "tid": tid, "payload": {"segments": [], "language": "en"}, "elapsed": 5.0})


def test_night_rests_between_tasks(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._dispatch()
    assert len(p.pool.put_q) == 1
    _finish(p, "cuda:1", tasks[0]["id"])
    assert p._rest_until > time.time()
    p._dispatch()
    assert len(p.pool.put_q) == 1           # mola: yeni görev yok
    p._rest_until = time.time() - 0.1
    p._dispatch()
    assert len(p.pool.put_q) == 2           # mola bitti


def test_normal_mode_has_no_rest(tmp_path):
    p, job, tasks = _pipe(tmp_path, night=False)
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._dispatch()
    _finish(p, "cuda:1", tasks[0]["id"])
    assert p._rest_until == 0.0
    p._dispatch()
    assert len(p.pool.put_q) == 2


def test_night_cooling_pause_holds_text_and_diarization(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p.temps = [p.gov.hot + 1]
    p._pending_futs = 1                     # active()
    p._night_tick()                         # ölçüm (iş parçacığı yerine hemen)
    assert p.gov.holding()
    p.dpool = FakePool(["diar"])
    p.idle.add("diar")
    dt = {"id": "u_test:d", "kind": "diarize", "job": job.item.key}
    p.tasks[dt["id"]] = dt
    p.dbacklog.append(dt)
    p.backlog.clear()                       # metin bitti: sıra ayırmada
    job.pending = {dt["id"]}
    p._dispatch()
    assert p.pool.put_q == [] and p.dpool.put_q == []      # soğuma molası: hiçbir görev yok
    ev = [e for e in p.events_seen if e["type"] == "night"][-1]
    assert ev["on"] and ev["paused"] and "soğuma" in ev["note"] and ev["card"] == "RTX 3090"
    p.temps = [p.gov.cool - 1]
    p._gov_at = 0
    p._night_tick()
    assert not p.gov.holding()
    p._dispatch()
    assert len(p.dpool.put_q) == 1


def test_night_mode_ignored_during_live(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    job.live_active = True
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._dispatch()
    _finish(p, "cuda:1", tasks[0]["id"])
    p._dispatch()
    assert len(p.pool.put_q) == 2           # canlı metin gecikmesin
    assert p.night_state()["note"].startswith("canlı")


def test_night_parks_helper_and_restores_note(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=8)
    p.hpool = FakeHelperPool(["cuda:0"])
    p.helper_devices = ["cuda:0"]
    p._h_parked[HELPER_DIAR] = HELPER_OFF
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._on_msg({"type": "ready", "wid": "cuda:0"})   # yardımcı açık ve boşta
    p._dispatch()
    assert "cuda:0" in p.hpool.stopped and p.hpool.put_q == []
    assert p._h_note["cuda:0"] == NIGHT_NOTE and p._h_note[HELPER_DIAR] == NIGHT_NOTE
    p.cfg = Config(out_dir=tmp_path, night_mode=False, min_free_gb=0)
    p._night_was = True
    p._night_tick()                         # gece modu kapandı: notlar normale döner
    assert p._h_note["cuda:0"] == HELPER_OFF and p._h_note[HELPER_DIAR] == HELPER_OFF


def test_night_single_card_drops_other_main_card(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p.pool = FakePool(["cuda:0", "cuda:1"])
    p.pool._spec = {"cuda:0": "cuda:0", "cuda:1": "cuda:1"}
    p.pool.stop = lambda n, idle=False: None
    p.devices = ["cuda:0", "cuda:1"]
    p._ecfg = {"gpu_mem": {0: 10.0, 1: 24.0}, "display_gpus": [0]}
    p._dispatch()
    assert p._drop_wids == {"cuda:0"}       # 3080 ilk görevinde kapanır, 3090 sürer
    # kapanmadan gece modu kapatılırsa 3080 çalışmaya devam eder
    p.cfg = Config(out_dir=tmp_path, night_mode=False, min_free_gb=0)
    p._night_was = True
    p._night_tick()
    assert p._drop_wids == set()


def test_night_recycles_idle_worker_after_interval(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._pending_futs = 1
    now = time.time()
    p._night_recycle(now)
    assert p.pool.respawned == []           # yeni açıldı
    p._born["cuda:1"] = now - 91 * 60
    p.queued_n = 1                          # kuyruğunda görev var: yenilenmez
    p._night_recycle(now)
    assert p.pool.respawned == []
    p.queued_n = 0
    p.inflight["cuda:1"] = ("x", now)       # görevde: yenilenmez
    p._night_recycle(now)
    assert p.pool.respawned == []
    p.inflight.clear()
    p._night_recycle(now)
    assert p.pool.respawned == ["cuda:1"] and p.pool.respawn_idle == [True]
    assert "cuda:1" not in p.idle and p.restarts.get("cuda:1", 0) == 0   # hata sayılmaz
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._night_recycle(time.time())
    assert p.pool.respawned == ["cuda:1"]   # yeni süre baştan


def test_night_recycle_skips_multi_worker_pool_and_off_setting(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p.pool = FakePool(["cuda:1#0", "cuda:1#1"])
    for w in p.pool.workers:
        p._on_msg({"type": "ready", "wid": w})
        p._born[w] = 0
    p._night_recycle(time.time())
    assert p.pool.respawned == []           # ortak kuyruk: boştaki işçi güvenle kapatılamaz
    p, job, tasks = _pipe(tmp_path / "b", night_recycle_min=0)
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._born["cuda:1"] = 0
    p._night_recycle(time.time())
    assert p.pool.respawned == []


def test_night_prepares_fewer_videos_ahead(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    for i in range(3):
        it = Item("url", f"https://x/p{i}", key=f"u_p{i}")
        p.jobs[it.key] = Job(it, tmp_path / f"p{i}", status="prep")
    assert not p._pending_ahead_ok()        # gece: en çok 3 (+ çevrilen iş)
    p.cfg = Config(out_dir=tmp_path, night_mode=False, min_free_gb=0)
    assert p._pending_ahead_ok()            # normal: 6


def test_disk_guard_waits_and_warns_once(tmp_path, monkeypatch):
    import shutil as sh

    from stt import pipeline as P
    free = {"gb": 2.0}
    monkeypatch.setattr(sh, "disk_usage", lambda path: type("U", (), {"free": free["gb"] * 2 ** 30})())
    p = Pipeline(Config(out_dir=tmp_path, min_free_gb=8))
    seen = []
    p.events = seen.append
    assert not p._disk_free_ok()
    p._disk_at = 0
    assert not p._disk_free_ok()
    assert len([e for e in seen if e["type"] == "notice"]) == 1      # tek uyarı
    free["gb"] = 50.0
    assert not p._disk_free_ok()            # ölçüm aralığı dolmadan eski sonuç
    p._disk_at = time.time() - P.DISK_CHECK_S - 1
    assert p._disk_free_ok()
    p2 = Pipeline(Config(out_dir=tmp_path / "x", min_free_gb=0))
    free["gb"] = 0.1
    assert p2._disk_free_ok()               # 0 = kapalı


def test_disk_guard_blocks_prep_turn_until_space(tmp_path, monkeypatch):
    p, job, tasks = _pipe(tmp_path, night=False)
    state = {"ok": False, "calls": 0}

    def disk():
        state["calls"] += 1
        if state["calls"] >= 3:
            state["ok"] = True
        return state["ok"]
    p._disk_free_ok = disk
    monkeypatch.setattr("stt.pipeline.time.sleep", lambda s: None)
    it = Item("url", "https://x/new", key="u_new")
    assert p._wait_prep_turn(it) is True and state["calls"] == 3


def test_night_state_off_and_idle(tmp_path):
    p, job, tasks = _pipe(tmp_path, night=False)
    assert p.night_state()["on"] is False and p.night_state()["note"] == ""
    p, job, tasks = _pipe(tmp_path / "b")
    job.status = "done"
    assert p.night_state()["note"] == "açık · iş bekleniyor"


def test_night_tick_emits_on_toggle_and_probe_is_throttled(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p._pending_futs = 1
    p.temps = [60, 60, 60]
    p._night_tick()
    p._night_tick()                         # SAMPLE_S dolmadı: ikinci ölçüm yok
    assert len(p.temps) == 2
    assert any(e["type"] == "night" and e["on"] for e in p.events_seen)


def test_app_set_night_mode(tmp_path, monkeypatch):
    from stt import gui
    saved = {}
    monkeypatch.setattr(gui, "save_settings", lambda upd: saved.update(upd))
    monkeypatch.setattr(gui, "load_config", lambda: Config(out_dir=tmp_path, night_mode=bool(saved.get("night_mode"))))
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    r = app.set_night_mode(True)
    assert saved == {"night_mode": True} and r["night"]["on"] is True
    r = app.set_night_mode(False)
    assert r["night"]["on"] is False


def test_governor_single_miss_keeps_state_and_good_read_resets_counter():
    g = Governor(72)
    g.update(80, 0)
    g.update(None, 20)
    g.update(None, 40)
    g.update(75, 60)                        # okundu: sayaç sıfırlanır
    g.update(None, 80)
    g.update(None, 100)
    assert g.holding()


def test_display_guard_survives_night_off(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p.pool = FakePool(["cuda:0", "cuda:1"])
    p.pool._spec = {"cuda:0": "cuda:0", "cuda:1": "cuda:1"}
    p.pool.stop = lambda n, idle=False: None
    p.devices = ["cuda:0", "cuda:1"]
    p._ecfg = {"gpu_mem": {0: 10.0, 1: 24.0}, "display_gpus": [0]}
    p._dispatch()
    assert p._drop_wids == {"cuda:0"}
    p.disable_helper("ekran sürücüsü çöktü")
    p.cfg = Config(out_dir=tmp_path, night_mode=False, min_free_gb=0)
    p._night_was = True
    p._night_tick()
    assert p._drop_wids == {"cuda:0"}       # ekran koruması gece modu kapanınca kalkmaz
    p2, _, _ = _pipe(tmp_path / "b")
    p2.pool = p.pool
    p2.devices, p2._ecfg = p.devices, p._ecfg
    p2._dispatch()
    p2.limit_to_one("cuda:1", "güç sınırı kalktı")
    p2.cfg = Config(out_dir=tmp_path, night_mode=False, min_free_gb=0)
    p2._night_was = True
    p2._night_tick()
    assert p2._drop_wids == {"cuda:0"}


def test_night_chip_refreshes_when_queue_ends(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p._pending_futs = 1
    p.temps = [p.gov.hot + 2]
    p._night_tick()
    assert p.night_state()["paused"]
    p._pending_futs = 0
    job.status = "done"
    n = len(p.events_seen)
    p._night_tick()
    ev = p.events_seen[n:]
    assert ev and ev[-1]["type"] == "night" and ev[-1]["note"] == "açık · iş bekleniyor" and not ev[-1]["paused"]


def test_night_probe_spawn_failure_does_not_block(tmp_path):
    p, job, tasks = _pipe(tmp_path)
    p._pending_futs = 1

    def boom(fn):
        raise RuntimeError("can't start new thread")
    p._spawn = boom
    p._night_tick()
    assert p._gov_probing is False


def test_disk_notice_repeats_every_half_hour(tmp_path, monkeypatch):
    import shutil as sh
    monkeypatch.setattr(sh, "disk_usage", lambda path: type("U", (), {"free": 1 * 2 ** 30})())
    p = Pipeline(Config(out_dir=tmp_path, min_free_gb=3))
    seen = []
    p.events = seen.append
    p._disk_free_ok()
    p._disk_at = 0
    p._disk_free_ok()
    assert len(seen) == 1
    p._disk_at, p._disk_warned_at = 0, time.time() - 1801
    p._disk_free_ok()
    assert len(seen) == 2
