"""Donma / ekran kartı / iptal davranışları (GPU'suz çalışır)."""
import threading
import time
from collections import deque
from pathlib import Path

from stt.config import Config
from stt.engines import choose_gpus
from stt.fetch import explain_error
from stt.pipeline import Job, Pipeline, detect_devices
from stt.sources import Item

INFO_3090_3080 = {0: {"name": "RTX 3080", "gb": 10, "display": True},
                  1: {"name": "RTX 3090", "gb": 24, "display": False}}


def test_choose_gpus_leaves_display_card_free():
    assert choose_gpus(2, INFO_3090_3080, "auto") == [1]
    assert choose_gpus(2, INFO_3090_3080, "all") == [0, 1]


def test_choose_gpus_single_or_all_display():
    assert choose_gpus(1, {0: {"name": "x", "gb": 10, "display": True}}) == [0]
    both = {0: {"name": "a", "gb": 10, "display": True}, 1: {"name": "b", "gb": 24, "display": True}}
    assert choose_gpus(2, both) == [1]           # en büyük tek kart; öteki masaüstüne kalır
    assert choose_gpus(2, {}) == [0]             # bilgi yok: tek kart (eşitlikte ilki)


def test_detect_devices_explicit():
    assert detect_devices("0,1") == ["cuda:0", "cuda:1"]
    assert detect_devices("cpu,cpu") == ["cpu", "cpu"]
    assert detect_devices("cuda:1") == ["cuda:1"]


def test_explain_error_turkish_hints():
    assert "bot" in explain_error("ERROR: [youtube] abc: Sign in to confirm you're not a bot").lower()
    assert "kaldırılmış" in explain_error("ERROR: Video unavailable")
    assert explain_error("tamamen bilinmeyen") == "tamamen bilinmeyen"


class FakePool:
    threads = False

    def __init__(self, names):
        self.workers = {n: object() for n in names}
        self.put_q: list = []
        self.dead: set = set()
        self.respawned: list = []

    def put(self, t):
        self.put_q.append(t)

    def get(self, timeout):
        return None

    def alive(self):
        return [n for n in self.workers if n not in self.dead]

    def respawn(self, n):
        self.respawned.append(n)
        self.dead.discard(n)
        return True


def _pipe(tmp_path: Path, workers=("cuda:1",)):
    p = Pipeline(Config(out_dir=tmp_path))
    p.pool = FakePool(workers)
    return p


def _job(p: Pipeline, tmp_path: Path, n=4):
    item = Item("url", "https://x/1", key="u_test")
    item.extra["uid"] = "u1"
    job = Job(item, tmp_path / "w", status="run", info={"duration_s": 600})
    (job.work / "parcalar").mkdir(parents=True)
    p.jobs[item.key] = job
    tasks = [{"id": f"u_test:t:{i:04d}", "kind": "transcribe", "job": item.key, "start": i * 60.0,
              "end": (i + 1) * 60.0, "idx": i, "n": n} for i in range(n)]
    for t in tasks:
        p.tasks[t["id"]] = t
    job.pending = {t["id"] for t in tasks}
    p.backlog = deque(tasks)
    return job, tasks


def test_dispatch_only_to_idle_workers(tmp_path):
    p = _pipe(tmp_path, ("a", "b"))
    _job(p, tmp_path)
    p._dispatch()
    assert p.pool.put_q == []                    # işçi hazır değil
    p._on_msg({"type": "ready", "wid": "a"})
    p._dispatch()
    p._dispatch()
    assert len(p.pool.put_q) == 1                # tek boş işçi -> tek görev
    p._on_msg({"type": "ready", "wid": "b"})
    p._dispatch()
    assert len(p.pool.put_q) == 2


def test_dead_worker_is_respawned_and_task_requeued(tmp_path):
    p = _pipe(tmp_path, ("a",))
    job, tasks = _job(p, tmp_path)
    p._on_msg({"type": "ready", "wid": "a"})
    p._dispatch()
    p._on_msg({"type": "start", "wid": "a", "tid": tasks[0]["id"]})
    p.backlog.clear()
    p.pool.dead.add("a")
    p._health()
    assert p.pool.respawned == ["a"]
    assert p.backlog and p.backlog[0]["id"] == tasks[0]["id"]
    assert "a" not in p.idle and not p.inflight


def test_hung_task_restarts_worker(tmp_path):
    p = _pipe(tmp_path, ("a",))
    job, tasks = _job(p, tmp_path)
    p.backlog.clear()
    p.inflight["a"] = (tasks[0]["id"], time.time() - 10_000)
    p._health()
    assert p.pool.respawned == ["a"]
    assert p.backlog[0]["id"] == tasks[0]["id"]


def test_fatal_worker_not_respawned(tmp_path):
    p = _pipe(tmp_path, ("a", "b"))
    p._on_msg({"type": "fatal", "wid": "a", "error": "model yok"})
    p.pool.dead.add("a")
    p._health()
    assert p.pool.respawned == []


def test_cancel_all_stops_jobs_and_keeps_finished_parts(tmp_path):
    p = _pipe(tmp_path, ("a",))
    events = []
    p.events = events.append
    job, tasks = _job(p, tmp_path)
    p._on_msg({"type": "ready", "wid": "a"})
    p._dispatch()
    p._on_msg({"type": "start", "wid": "a", "tid": tasks[0]["id"]})
    assert p.cancel_all() == 1
    assert job.status == "cancelled" and not p.backlog and not p.active()
    assert any(e["type"] == "failed" and e.get("cancelled") for e in events)
    # çalışan parça bitince sonucu saklanır (tekrar eklenince kaldığı yerden devam)
    p._on_msg({"type": "done", "wid": "a", "tid": tasks[0]["id"], "payload": {"segments": []}, "elapsed": 1})
    assert (job.work / "parcalar" / "0000.json").exists()
    assert "a" in p.idle
    # durdurmadan önce gönderilmiş ama henüz hazırlanmamış öğe de atlanır
    it = Item("url", "https://x/2")
    it.extra["_gen"] = 0
    assert p._cancelled(it)


def test_gui_push_never_blocks(tmp_path, monkeypatch):
    """Pencere (WebView2) takılsa da olay gönderen iş parçacıkları beklememeli."""
    monkeypatch.setenv("STT_OUT", str(tmp_path))
    import stt.gui as g

    block = threading.Event()

    class StuckWindow:
        def evaluate_js(self, js):
            block.wait(30)

    app = g.App()
    app._window = StuckWindow()
    app._ready = True
    app._start_sender()
    t = time.time()
    for i in range(app.MAX_OUTBOX + 500):
        app._push({"type": "log", "level": "INFO", "text": f"satır {i}"})
    app._push({"type": "done", "uid": "x"})
    assert time.time() - t < 2.0
    assert len(app._outbox) <= app.MAX_OUTBOX
    assert any(e.get("type") == "done" for e in app._outbox)   # önemli olaylar atılmaz
    app._stop.set()
    block.set()


def test_pick_language_ignores_welsh():
    from stt.engines import DEFAULT_LANGS, pick_language
    allowed = set(DEFAULT_LANGS.split(","))
    probs = [{"cy": 0.93, "en": 0.06}, {"cy": 0.55, "en": 0.44}, {"en": 0.9, "cy": 0.1}]
    assert pick_language(probs, allowed)[0] == "en"
    assert pick_language([{"tr": 0.6, "az": 0.3}], allowed)[0] == "tr"


def test_collapse_repeats():
    from stt.merge import collapse_repeats, merge_chunks
    t = "Mae'n rhaid i ni ddweud hynny. " * 6 + "Thank you. Thank you."
    assert collapse_repeats(t) == "Mae'n rhaid i ni ddweud hynny. Thank you. Thank you."
    segs = [{"start": i, "end": i + 1, "text": "We have to keep going forward."} for i in range(5)]
    out, _ = merge_chunks([{"segments": segs, "language": "en"}])
    assert len(out) == 1 and out[0]["end"] == 5


def test_redo_reason(tmp_path):
    p = _pipe(tmp_path)
    it = Item("url", "https://x/1", key="u_1")
    assert p._redo_reason(it, {"info": {"language": "cy", "diarized": False}})
    assert not p._redo_reason(it, {"info": {"language": "en", "diarized": False}})
    it.extra["diarize"] = True
    assert "konuşmacı" in p._redo_reason(it, {"info": {"language": "en", "diarized": False}})
    it2 = Item("url", "https://x/2", key="u_2", extra={"lang": "cy"})   # kullanıcı dili kendisi seçtiyse dokunma
    assert not p._redo_reason(it2, {"info": {"language": "cy"}})


def test_diarize_runs_in_separate_pool(tmp_path):
    p = _pipe(tmp_path, ("cuda:1",))
    job, tasks = _job(p, tmp_path, n=1)
    d = {"id": "u_test:d", "kind": "diarize", "job": job.item.key}
    p.tasks[d["id"]] = d
    job.pending.add(d["id"])
    p.dbacklog.append(d)
    started = []

    def fake_start():
        p.dpool = FakePool(["diar"])
        started.append(1)
    p._start_diar_pool = fake_start
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._dispatch()
    assert started and [t["kind"] for t in p.pool.put_q] == ["transcribe"]
    assert p.dpool.put_q == []                    # ayırma süreci henüz hazır değil
    p._on_msg({"type": "ready", "wid": "diar"})
    p._dispatch()
    assert [t["kind"] for t in p.dpool.put_q] == ["diarize"]
    p._on_msg({"type": "start", "wid": "diar", "tid": d["id"]})
    p.dpool.dead.add("diar")                      # pyannote çöktü: Whisper işçisi etkilenmez
    p._health()
    assert p.dpool.respawned == ["diar"] and p.pool.respawned == []
    assert p.dbacklog and p.dbacklog[0]["id"] == d["id"]
