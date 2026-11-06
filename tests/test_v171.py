"""1.7.1: indirme takılması, kullanım verisi kapalı, konuşmacı ayırma sinyali / erken yeniden başlatma."""
import os
import time

from test_stability import _Pipe, _fake_diarizer, _job, _pipe, _wav


def test_net_env_disables_xet_and_telemetry(monkeypatch):
    for k in ("HF_HUB_DISABLE_XET", "HF_HUB_DOWNLOAD_TIMEOUT", "PYANNOTE_METRICS_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    from stt.util import net_env
    net_env()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]) > 0
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"


def test_beat_keeps_diarize_alive_and_hang_restarts_early(tmp_path):
    p = _pipe(tmp_path, ("a",))
    job, tasks = _job(p, tmp_path)
    p.backlog.clear()
    t = dict(tasks[0], kind="diarize")
    p.tasks[t["id"]] = t
    job.info["duration_s"] = 3000  # 50 dk video: eskiden 2,5 saat beklenirdi
    assert p._task_timeout(t) <= 900
    p.inflight["a"] = (t["id"], time.time() - 800)
    p._on_msg({"type": "beat", "wid": "a", "tid": t["id"]})
    p._health()
    assert p.pool.respawned == [] and time.time() - p.inflight["a"][1] < 5
    p.inflight["a"] = (t["id"], time.time() - 1000)
    p._health()
    assert p.pool.respawned == ["a"]


def test_beater_throttles():
    import queue

    from stt.workers import _beater
    q = queue.Queue()
    b = _beater(q, "diar", "t1", every=30)
    b(); b(); b()
    b(force=True)
    assert q.qsize() == 2 and q.get()["type"] == "beat"


def test_diarizer_sends_progress_via_hook(tmp_path):
    seen = []

    class HookPipe(_Pipe):
        def __call__(self, inp, **kw):
            kw["hook"]("segmentation", None, completed=1, total=2)
            return super().__call__(inp, **kw)
    d = _fake_diarizer(HookPipe([]), cpu_pipe=None)
    d.wait_memory = lambda *a, **k: None
    turns = d.run(_wav(tmp_path), beat=lambda force=False: seen.append(force))
    assert len(turns) == 2 and seen


def test_gpu_fault_detection():
    from stt.workers import gpu_fault
    assert gpu_fault("RuntimeError: parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")
    assert gpu_fault("RuntimeError: cuBLAS failed with status CUBLAS_STATUS_EXECUTION_FAILED")
    assert not gpu_fault("RuntimeError: CUDA failed with error out of memory")
    assert not gpu_fault("ValueError: bozuk ses")


def test_gpu_fault_requeues_and_worker_restarts(tmp_path):
    p = _pipe(tmp_path, ("a",))
    job, tasks = _job(p, tmp_path)
    p.backlog.clear()
    tid = tasks[0]["id"]
    err = "RuntimeError: parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal"
    for _ in range(2):  # eskiden 2. hatada iş başarısız oluyordu
        p._on_msg({"type": "start", "wid": "a", "tid": tid})
        p._on_msg({"type": "error", "wid": "a", "tid": tid, "error": err, "restart": True})
        assert job.status == "run" and p.backlog[0]["id"] == tid and "a" not in p.idle
        p.backlog.clear()
    p.pool.dead.add("a")
    p._health()
    assert p.pool.respawned == ["a"]


def test_diar_and_transcribe_take_turns_on_same_gpu(tmp_path):
    from test_stability import FakePool
    p = _pipe(tmp_path, ("cuda:1",))
    p.pool._spec = {"cuda:1": "cuda:1"}
    job, tasks = _job(p, tmp_path, n=2)
    d = {"id": "u_test:d", "kind": "diarize", "job": job.item.key}
    p.tasks[d["id"]] = d
    job.pending.add(d["id"])
    p.dbacklog.append(d)
    p.dpool = FakePool(["diar"])
    p.dpool.devices = ["cuda:1"]
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._on_msg({"type": "ready", "wid": "diar"})
    p._dispatch()
    assert [t["kind"] for t in p.pool.put_q] == ["transcribe"] and p.dpool.put_q == []  # aynı anda değil
    for t in tasks:
        p._on_msg({"type": "start", "wid": "cuda:1", "tid": t["id"]})
        p._dispatch()
        assert p.dpool.put_q == []
        p._on_msg({"type": "done", "wid": "cuda:1", "tid": t["id"], "elapsed": 1.0,
                   "payload": {"segments": [], "language": "en"}})
        p._dispatch()
    assert [t["kind"] for t in p.dpool.put_q] == ["diarize"]  # metin bitti: sıra ayırmada
