"""1.6.1: tek iş durdurma, konuşmacı ayırmada bellek sınırı/bekleme, metindeki boşlukları doldurma."""
import time

import pytest

from stt import fetch
from stt.engines import find_gaps, keep_gap_segment
from stt.pipeline import is_ram_error

from test_stability import _fake_diarizer, _job, _pipe, _wav


def test_find_gaps():
    assert find_gaps([(0, 5), (30, 40)], 60, 12) == [(5, 30), (40, 60)]
    assert find_gaps([(0, 10), (15, 50)], 55, 12) == []
    assert find_gaps([], 20, 12) == [(0.0, 20)]


def test_keep_gap_segment_drops_music_hallucinations():
    assert keep_gap_segment("Nereye gidiyorsun?", -0.3, 0.1, 1.4)
    assert not keep_gap_segment("Altyazı M.K.", -0.2, 0.1, 1.2)
    assert not keep_gap_segment("♪ ♪", -0.2, 0.1, 1.2)
    assert not keep_gap_segment("evet evet evet", -1.4, 0.2, 1.2)
    assert not keep_gap_segment("hmm", -0.7, 0.9, 1.2)


def test_is_ram_error():
    assert is_ram_error("RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory")
    assert is_ram_error("MemoryError: ")
    assert not is_ram_error("cuDNN error")


def test_cancel_single_job(tmp_path):
    p = _pipe(tmp_path, ("a",))
    events = []
    p.events = events.append
    job, tasks = _job(p, tmp_path)
    p.items["u1"] = job.item
    assert p.cancel("u1")
    assert job.status == "cancelled" and not p.backlog
    assert str(job.work) in fetch.CANCELLED
    assert any(e["type"] == "failed" and e.get("cancelled") for e in events)
    assert p._cancelled(job.item)
    assert not p.cancel("yok")
    with pytest.raises(fetch.Cancelled):
        fetch.check_cancel(job.work)
    fetch.CANCELLED.discard(str(job.work))


def test_diar_ram_error_is_retried_later(tmp_path):
    p = _pipe(tmp_path, ("a",))
    job, tasks = _job(p, tmp_path)
    d = {"id": "u_test:d", "kind": "diarize", "job": job.item.key, "wav": "x"}
    p.tasks[d["id"]] = d
    job.pending.add(d["id"])
    p._on_msg({"type": "error", "wid": "diar", "tid": d["id"],
               "error": "RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 16 bytes."})
    assert not job.diar_error and d["id"] in job.pending
    assert p.ddelayed and p.ddelayed[0][1] is d
    p.ddelayed = [(time.time() - 1, d)]
    p._dispatch()                       # metin çevirisi sürüyor: ayırma bekler
    assert p.ddelayed
    p.backlog.clear()
    p._dispatch()
    assert not p.ddelayed and (list(p.dbacklog) == [d] or p.dpool is not None)


def test_diarizer_lowers_batch_on_ram_error(tmp_path, monkeypatch):
    from test_stability import _Pipe
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pipe = _Pipe(["DefaultCPUAllocator: not enough memory: you tried to allocate 16 bytes"])
    pipe.embedding_batch_size = pipe.segmentation_batch_size = 32
    dz = _fake_diarizer(pipe, cpu_pipe=None)
    dz.batch = 8
    turns = dz.run(_wav(tmp_path))
    assert turns and dz.batch == 2 and pipe.embedding_batch_size == 2
