"""Senaryo denemeleri: takılma / kilitlenme olmasın (sahte işçilerle)."""
from test_stability import FakePool, _job, _pipe


def _diar_setup(tmp_path, n=2):
    p = _pipe(tmp_path, ("cuda:1",))
    p.pool._spec = {"cuda:1": "cuda:1"}
    job, tasks = _job(p, tmp_path, n=n)
    d = {"id": "u_test:d", "kind": "diarize", "job": job.item.key}
    p.tasks[d["id"]] = d
    job.pending.add(d["id"])
    p.dbacklog.append(d)
    p.dpool = FakePool(["diar"])
    p.dpool.devices = ["cuda:1"]
    fin = []
    p._maybe_finalize = lambda j: fin.append(j) if not j.pending else None
    return p, job, tasks, d, fin


def _finish_transcription(p, tasks):
    for t in tasks:
        p._on_msg({"type": "start", "wid": "cuda:1", "tid": t["id"]})
        p._on_msg({"type": "done", "wid": "cuda:1", "tid": t["id"], "elapsed": 1.0,
                   "payload": {"segments": [], "language": "en"}})
        p._dispatch()


def test_dead_diar_process_does_not_hang_job_or_queue(tmp_path):
    p, job, tasks, d, fin = _diar_setup(tmp_path)
    p.backlog.clear()
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    _finish_transcription(p, tasks)
    # ayırma süreci üst üste çöktü, yeniden açma hakkı bitti
    p.dpool.dead.add("diar")
    p.restarts["diar"] = p.cfg.worker_restarts
    p._health()
    assert not p.dbacklog and fin == [job] and job.diar_error
    # başka bir videonun metni artık engellenmemeli
    t2 = dict(tasks[0], id="u_test:t:9999", idx=9999)
    p.tasks[t2["id"]] = t2
    job.pending.add(t2["id"])
    p.backlog.append(t2)
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._dispatch()
    assert p.pool.put_q[-1]["id"] == t2["id"]


def test_two_jobs_alternate_without_starvation(tmp_path):
    p, job, tasks, d, fin = _diar_setup(tmp_path, n=2)
    p.backlog.clear()
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._on_msg({"type": "ready", "wid": "diar"})
    _finish_transcription(p, tasks)                     # A'nın metni bitti
    from test_stability import Item, Job
    b = Job(Item("url", "https://x/2", key="u_b"), tmp_path / "wb", status="run", info={"duration_s": 600})
    p.jobs["u_b"] = b
    tb = {"id": "u_b:t:0000", "kind": "transcribe", "job": "u_b", "start": 0.0, "end": 60.0, "idx": 0, "n": 1}
    p.tasks[tb["id"]] = tb
    b.pending = {tb["id"]}
    p.backlog.append(tb)
    n_main = len(p.pool.put_q)
    p._dispatch()
    assert [t["id"] for t in p.dpool.put_q] == [d["id"]] and len(p.pool.put_q) == n_main  # önce A'nın ayırması
    p._on_msg({"type": "start", "wid": "diar", "tid": d["id"]})
    p._dispatch()
    assert len(p.pool.put_q) == n_main                                            # ayırma sürerken metin yok
    p._on_msg({"type": "done", "wid": "diar", "tid": d["id"], "elapsed": 1.0,
               "payload": {"turns": [], "device": "cuda:1"}})
    p._dispatch()
    assert p.pool.put_q[-1]["id"] == tb["id"] and fin == [job]                    # sonra B'nin metni


def test_hung_gpu_chunk_detected_in_minutes(tmp_path):
    p = _pipe(tmp_path, ("cuda:1",))
    p.pool._spec = {"cuda:1": "cuda:1"}
    t = {"kind": "transcribe", "start": 0.0, "end": 270.0}
    assert p._task_timeout(t, "cuda:1") <= 900          # eskiden 36 dk
    p.pool._spec = {"cpu": "cpu"}
    assert p._task_timeout(t, "cpu") >= 2000            # işlemcide yavaş: kesilmesin


def test_bad_message_fails_one_job_not_engine(tmp_path):
    p = _pipe(tmp_path, ("cuda:1",))
    job, tasks = _job(p, tmp_path, n=2)
    p._on_msg({"type": "start", "wid": "cuda:1", "tid": tasks[0]["id"]})
    p.pool.get = lambda timeout, q=[{"type": "done", "wid": "cuda:1", "tid": tasks[0]["id"], "elapsed": 1.0,
                                     "payload": None}]: q.pop() if q else None  # bozuk sonuç
    p.pump(0.01)  # hata dışarı çıkmamalı
    assert job.status == "failed" and "İç hata" in job.error


def test_worker_stuck_loading_is_restarted(tmp_path):
    import time
    p = _pipe(tmp_path, ("cuda:1",))
    p._health()
    assert p.pool.respawned == []
    p._loading_since["cuda:1"] = time.time() - 4000
    p._health()
    assert p.pool.respawned == ["cuda:1"]


def test_cuda_error_in_music_separation_retries_without_it(tmp_path):
    p, job, tasks, d, fin = _diar_setup(tmp_path)
    d["vocals"] = True
    p._on_msg({"type": "start", "wid": "diar", "tid": d["id"]})
    p._on_msg({"type": "error", "wid": "diar", "tid": d["id"],
               "error": "RuntimeError: müzik ayıklama sırasında ekran kartı hatası: CUDA error: unknown error"})
    assert p.dpool.respawned == ["diar"] and p.dbacklog[0]["vocals"] is False and not job.diar_error


def test_retry_redoes_skipped_chunk_and_failed_diarization(tmp_path):
    import json

    from stt.pipeline import _finished
    d = tmp_path / "parcalar"
    d.mkdir()
    (d / "0000.json").write_text(json.dumps({"segments": [], "language": "en"}))
    (d / "0001.json").write_text(json.dumps({"segments": [], "language": None, "skipped": True}))
    (tmp_path / "konusmaci.json").write_text(json.dumps({"error": "CUDA error", "turns": []}))
    assert _finished(d / "0000.json") and not _finished(d / "0001.json")
    assert not _finished(tmp_path / "konusmaci.json") and not _finished(tmp_path / "yok.json")


def test_done_video_with_skipped_chunk_is_reprocessed(tmp_path):
    from test_stability import Item
    p = _pipe(tmp_path, ("cuda:1",))
    it = Item("url", "https://x/1", key="u_1")
    assert "çevrilemeyen" in p._redo_reason(it, {"info": {"language": "en", "skipped_chunks": [[0, 60]]}})
