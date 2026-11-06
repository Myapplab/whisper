"""1.9.9: "yardim" kart modu — monitörlü kart (3080) yalnız ana kart (3090) yetişemeyince, düşük yükle çalışır."""
import time
from collections import deque
from pathlib import Path

from stt.config import Config, normalize
from stt.pipeline import Job, Pipeline, detect_devices, helper_devices
from stt.sources import Item

from test_stability import INFO_3090_3080, FakePool


class FakeHelperPool(FakePool):
    def __init__(self, names):
        super().__init__(names)
        self._spec = {n: n for n in names}
        self.stopped: list = []
        self.queue_left: list = []

    def stop(self, n, idle=False):
        self.stopped.append(n)
        self.stop_idle = getattr(self, "stop_idle", []) + [idle]
        self.dead.add(n)

    def drain(self):
        out, self.queue_left = self.queue_left, []
        return out


def _pipe(tmp_path: Path, n=6, **cfg):
    p = Pipeline(Config(out_dir=tmp_path, **cfg))
    p.pool = FakePool(["cuda:1"])
    p.pool._spec = {"cuda:1": "cuda:1"}
    p.hpool = FakeHelperPool(["cuda:0"])
    p.helper_devices = ["cuda:0"]
    p._spawn = lambda fn: fn()          # ölçüm hemen (iş parçacığı yok)
    p.loads = []
    p._gpu_load = lambda idx: (p.loads.append(idx), p.load)[1]
    p.load = {"util": 5, "used_mb": 4000.0, "total_mb": 10240.0}
    item = Item("url", "https://x/1", key="u_test")
    item.extra["uid"] = "u1"
    job = Job(item, tmp_path / "w", status="run", info={"duration_s": 600})
    (job.work / "parcalar").mkdir(parents=True)
    p.jobs[item.key] = job
    tasks = [{"id": f"u_test:t:{i:04d}", "kind": "transcribe", "job": item.key, "start": i * 300.0,
              "end": (i + 1) * 300.0, "idx": i, "n": n} for i in range(n)]
    for t in tasks:
        p.tasks[t["id"]] = t
    job.pending = {t["id"] for t in tasks}
    p.backlog = deque(tasks)
    return p, job, tasks


def _ready(p, *wids):
    for w in wids:
        p._on_msg({"type": "ready", "wid": w})


def test_helper_devices_only_in_helper_mode():
    assert helper_devices("yardim", INFO_3090_3080) == ["cuda:0"]
    assert detect_devices("yardim", INFO_3090_3080) == ["cuda:1"]       # ana kart = önerilen ayarla aynı
    assert helper_devices("auto", INFO_3090_3080) == []
    assert helper_devices("all", INFO_3090_3080) == []
    assert helper_devices("yardim", {0: {"name": "x", "gb": 10, "display": True}}) == []   # tek kart
    assert helper_devices("yardim", {}) == []                            # nvidia-smi yok: körlemesine gidilmez


def test_helper_waits_while_main_card_keeps_up(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=1)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    assert len(p.pool.put_q) == 1 and p.hpool.put_q == []      # 3090 aldı, 3080'e iş yok
    p._dispatch()
    assert p.hpool.put_q == [] and p.loads == []               # gereksiz ölçüm de yok


def test_helper_takes_surplus_after_probe(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()                                   # 3090: 1 görev; 3080: önce ölçüm
    assert len(p.pool.put_q) == 1 and p.loads == [0]
    p._dispatch()
    assert len(p.hpool.put_q) == 1                  # kart boş: tek görev
    p._dispatch()
    assert len(p.hpool.put_q) == 1                  # alınmadan ikinci görev verilmez
    p._on_msg({"type": "start", "wid": "cuda:0", "tid": p.hpool.put_q[0]["id"]})
    p._dispatch()
    assert len(p.hpool.put_q) == 1                  # çalışırken yeni görev yok


def test_helper_rests_after_each_task(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6, helper_rest_s=60)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    t = p.hpool.put_q[0]
    p._on_msg({"type": "start", "wid": "cuda:0", "tid": t["id"]})
    p._on_msg({"type": "done", "wid": "cuda:0", "tid": t["id"], "elapsed": 1.0,
               "payload": {"segments": [], "language": "en"}})
    n_loads = len(p.loads)
    for _ in range(3):
        p._dispatch()
    assert len(p.hpool.put_q) == 1 and len(p.loads) == n_loads    # dinleniyor: ölçüm de görev de yok
    p._h_ready["cuda:0"] = time.time() - 1                        # dinlenme bitti
    p._dispatch()
    assert len(p.loads) == n_loads + 1                            # yeni ölçüm (eskisi kullanılmaz)
    p._dispatch()
    assert len(p.hpool.put_q) == 2


def test_helper_waits_when_card_busy_with_other_program(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    events = []
    p.events = events.append
    p.load = {"util": 90, "used_mb": 4000.0, "total_mb": 10240.0}     # oyun
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    assert p.hpool.put_q == []
    paused = [e for e in events if e.get("state") == "paused"]
    assert paused and "%90" in paused[-1]["note"]
    p.load = {"util": 10, "used_mb": 4000.0, "total_mb": 10240.0}
    p._dispatch()
    assert p.hpool.put_q == []                       # 30 sn sonra yeniden bakılır
    p._h_ready["cuda:0"] = time.time() - 1
    p._dispatch()
    p._dispatch()
    assert len(p.hpool.put_q) == 1
    assert events[-1].get("state") == "idle"         # "bekliyor" yazısı kalktı


def test_helper_waits_when_card_memory_low(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    p.load = {"util": 5, "used_mb": 9500.0, "total_mb": 10240.0}     # 0.7 GB boş
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    assert p.hpool.put_q == []


def test_unreadable_load_does_not_block_forever(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    p.load = None                                                     # nvidia-smi "[N/A]"
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    assert len(p.hpool.put_q) == 1


def test_helper_parked_after_long_busy_and_reopened(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    p.load = {"util": 95, "used_mb": 9000.0, "total_mb": 10240.0}
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    p._h_busy_since["cuda:0"] = time.time() - 1000                    # uzun süredir meşgul
    p._h_ready["cuda:0"] = 0
    p._dispatch()
    p._dispatch()
    assert p.hpool.stopped == ["cuda:0"] and "cuda:0" in p._h_parked  # model belleği bırakıldı
    p._health()
    assert p.hpool.respawned == []                                    # çöktü sanılıp açılmaz
    # oyun kapandı ama bellek, modelin sığacağı kadar boş değil: kapalı kalır
    p.load = {"util": 5, "used_mb": 6000.0, "total_mb": 10240.0}
    p._h_ready["cuda:0"] = 0
    p._dispatch()
    p._dispatch()
    assert p.hpool.respawned == []
    p.load = {"util": 5, "used_mb": 1500.0, "total_mb": 10240.0}
    p._h_ready["cuda:0"] = 0
    p._dispatch()
    p._dispatch()
    assert p.hpool.respawned == ["cuda:0"] and not p._h_parked
    assert p.hpool.put_q == []                                        # önce model yüklensin (ready)


def test_helper_used_while_main_card_diarizes(tmp_path):
    """3090 konuşmacı ayırırken metin işçisi bekler: tek görev kalsa da 3080 alır (asıl kazanç bu)."""
    p, job, tasks = _pipe(tmp_path, n=1)
    p.dpool = FakePool(["diar"])
    p.dpool.devices = ["cuda:1"]
    d = {"id": "u_test:d", "kind": "diarize", "job": job.item.key}
    p.tasks[d["id"]] = d
    job.pending.add(d["id"])
    p.inflight["diar"] = (d["id"], time.time())
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    assert p.pool.put_q == [] and len(p.hpool.put_q) == 1


def test_helper_fatal_returns_queued_task(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    t = p.hpool.put_q[0]
    p.hpool.queue_left = [t]
    n = len(p.backlog)
    p._on_msg({"type": "fatal", "wid": "cuda:0", "error": "model yok"})
    assert p.backlog[0]["id"] == t["id"] and len(p.backlog) == n + 1 and p.hqueued_n == 0
    assert p.failed_reason is None                                    # ana kart çalışıyor: motor durmaz


def test_helper_crash_task_requeued(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    t = p.hpool.put_q[0]
    p._on_msg({"type": "start", "wid": "cuda:0", "tid": t["id"]})
    p.hpool.dead.add("cuda:0")
    p._health()
    assert p.hpool.respawned == ["cuda:0"]
    assert any(x["id"] == t["id"] for x in p.backlog)


def test_task_timeout_uses_helper_device(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    assert p._task_timeout(tasks[0], "cuda:0") == 900.0               # ekran kartı: 3 x 300 sn


def test_helper_config_bounds():
    c = normalize(Config(helper_rest_s=0, helper_busy_pct=500, helper_min_free_gb=float("nan")))
    assert c.helper_rest_s == 1.0 and c.helper_busy_pct == 100 and c.helper_min_free_gb == 1.5


def test_settings_accept_helper_mode(tmp_path, monkeypatch):
    import stt.gui as gui
    saved = {}
    monkeypatch.setattr(gui, "save_settings", lambda upd: saved.update(upd))
    monkeypatch.setattr(gui, "load_config", lambda: Config(out_dir=tmp_path, gpus=saved.get("gpus", "auto")))
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    app._gpus = {}
    app._llm = None
    import threading
    app._restart = threading.Event()
    app._start_api = lambda: None
    monkeypatch.setattr(gui.App, "llm", lambda self: type("L", (), {"key": lambda s, x: None,
                                                                     "status": lambda s: {}})())
    r = app.save_settings({"gpus": "yardim"})
    assert saved["gpus"] == "yardim" and r["gpus"] == "yardim"
    assert app._restart.is_set()                    # işler bitince motor yeni kart düzeniyle açılır


def test_worker_caps_batch_on_helper(monkeypatch):
    import stt.engines as eng
    import stt.workers as w
    seen = {}

    class FakeEngine:
        def __init__(self, path, device, ct, beam, cond, batch, vram, *a, **k):
            seen["batch"] = batch
    monkeypatch.setattr(eng, "WhisperEngine", FakeEngine)
    cfg = {"engine": "whisper", "batch_size": -1, "gpu_mem": {0: 10}, "display_gpus": [], "display_batch": 4,
           "helper": True, "model_path": "m", "compute_type": "float16", "beam_size": 5,
           "condition_on_previous_text": True}
    w._init("cuda:0", cfg, {})
    assert seen["batch"] == 4                       # 10 GB kart normalde 8: yardımcıda 4


# ---------------------------------------------------------------- inceleme bulguları (1.9.9)
def _wait(pool, typ, t=20.0):
    end = time.time() + t
    while time.time() < end:
        m = pool.get(0.2)
        if m and m["type"] == typ:
            return m
    return None


def test_real_queue_idle_worker_stop_keeps_queue_usable():
    """Boştaki işçi task_q.get() içinde kuyruk kilidini tutar; öldürülürse yeni süreç bir daha görev alamıyordu.
    Kapanma işaretiyle kapatılınca (stop idle=True) yeniden açılan süreç görev alır. Gerçek süreçlerle."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import _stub_worker
    from stt.workers import WorkerPool

    class P(WorkerPool):
        def _new_proc(self, name):
            return self.ctx.Process(target=_stub_worker.loop, args=(name, self._spec[name], self._ecfg, self.task_q,
                                                                    self.result_q, None, self.load_lock), daemon=True)
    hp = P(["cuda:0"], {"helper": True})
    try:
        hp.start()
        assert _wait(hp, "ready")
        hp.put({"id": "t1"})
        assert _wait(hp, "done")
        time.sleep(0.5)                      # işçi boşta: get() içinde
        hp.stop("cuda:0", idle=True)
        assert not hp.workers["cuda:0"].is_alive()
        hp.respawn("cuda:0")
        assert _wait(hp, "ready")
        hp.put({"id": "t2"})
        m = _wait(hp, "done")
        assert m and m["tid"] == "t2"
    finally:
        for w in hp.workers.values():
            if w.is_alive():
                w.kill()


def test_helper_starts_off_and_opens_only_on_surplus(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=1)
    p._h_parked["cuda:0"] = "kapalı"         # start(): model baştan yüklenmez
    _ready(p, "cuda:1")
    p._dispatch()
    p._dispatch()
    assert p.hpool.respawned == [] and p.loads == []          # ana kart yetişiyor: açılmaz, ölçülmez
    p.backlog.extend({**tasks[0], "id": f"x{i}"} for i in range(3))
    for t in list(p.backlog):
        p.tasks[t["id"]] = t
        job.pending.add(t["id"])
    p.load = {"util": 5, "used_mb": 2000.0, "total_mb": 10240.0}
    p._dispatch()
    p._dispatch()
    assert p.hpool.respawned == ["cuda:0"]                    # iş fazlası + kart boş: açılır
    assert p.hpool.put_q == []
    _ready(p, "cuda:0")
    p._dispatch()
    p._dispatch()
    assert len(p.hpool.put_q) == 1


def test_helper_closed_after_idle_minutes(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=1)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    assert p.hpool.stopped == []
    p._h_last["cuda:0"] = time.time() - 10_000
    p._dispatch()
    assert p.hpool.stopped == ["cuda:0"] and p.hpool.stop_idle == [True] and "cuda:0" in p._h_parked
    p._health()
    assert p.hpool.respawned == []


def test_helper_oom_during_task_requeues_without_losing_chunk(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    _ready(p, "cuda:1", "cuda:0")
    p._dispatch()
    p._dispatch()
    t = p.hpool.put_q[0]
    p._on_msg({"type": "start", "wid": "cuda:0", "tid": t["id"]})
    p._on_msg({"type": "error", "wid": "cuda:0", "tid": t["id"], "error": "RuntimeError: CUDA failed with error out of memory"})
    assert p.backlog[0]["id"] == t["id"] and t["id"] not in job.attempts       # deneme sayılmadı
    assert "cuda:0" in p._h_parked and p.hpool.stop_idle == [True]
    assert not job.info.get("skipped_chunks")


def test_helper_load_oom_parks_instead_of_restart_loop(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    _ready(p, "cuda:1")
    p._on_msg({"type": "fatal", "wid": "cuda:0", "error": "RuntimeError: CUDA failed with error out of memory",
               "retryable": True})
    assert "cuda:0" in p._h_parked and "cuda:0" not in p.fatal
    assert p.hpool.stop_idle == [False]          # süreç kendisi kapanıyor: kapanma işareti kuyruğa konmaz
    p.hpool.dead.add("cuda:0")
    p._health()
    assert p.hpool.respawned == [] and p.restarts.get("cuda:0", 0) == 0


def test_stale_probe_not_used_later(tmp_path):
    p, job, tasks = _pipe(tmp_path, n=6)
    _ready(p, "cuda:1", "cuda:0")
    p._h_probe["cuda:0"] = (time.time() - 3600, {"util": 0, "used_mb": 0.0, "total_mb": 10240.0})
    p.load = {"util": 95, "used_mb": 4000.0, "total_mb": 10240.0}   # şimdi oyun açık
    p._dispatch()
    p._dispatch()
    assert p.hpool.put_q == [] and p.loads == [0]


def test_only_one_helper_card():
    info = {0: {"name": "RTX 3080", "gb": 10, "display": True}, 1: {"name": "RTX 3090", "gb": 24, "display": False},
            2: {"name": "RTX 3060", "gb": 12, "display": True}}
    assert helper_devices("yardim", info) == ["cuda:2"]      # tek işçili havuz (kapanma işareti ona gider)


def test_settings_list_cards_before_engine_ready(tmp_path, monkeypatch):
    """Kullanıcıda 1.9.9'da "yardımcı" seçeneği görünmedi: Ayarlar uygulama açılırken (motor daha başlamadan)
    dolduruluyor, kart listesi boş geliyordu. Motor hazır değilse kartlar nvidia-smi'den okunur."""
    import stt.engines as eng
    import stt.gui as gui
    monkeypatch.setattr(eng, "gpu_info", lambda *a, **k: {0: {"name": "RTX 3080", "gb": 10.0, "display": True},
                                                            1: {"name": "RTX 3090", "gb": 24.0, "display": False}})
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    app._gpus = {}                                   # motor henüz başlamadı
    app._llm = None
    monkeypatch.setattr(gui.App, "llm", lambda self: type("L", (), {"key": lambda s, x: None,
                                                                     "status": lambda s: {}})())
    cards = app.get_settings()["gpu_cards"]
    assert [c["name"] for c in cards] == ["RTX 3080", "RTX 3090"] and cards[0]["display"] is True
