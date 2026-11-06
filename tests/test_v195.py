"""1.9.5: ses kütüphanesi karışık küme imzasını öğrenmez; eğitimde imza düzeltilmiş satırların sesinden çıkar."""
import json
import threading
import time
import types

import numpy as np

from stt.voices import (PURE_MIN, VoiceLibrary, _FileLock, cluster_purity, ensure_orig_turns, learn_from_doc,
                        load_doc_voices, save_doc_voices)


def _vec(i, n=8):
    v = np.zeros(n)
    v[i] = 1.0
    return v.tolist()


def _seg(i, a, b, sid):
    return {"id": i, "start": a, "end": b, "speaker_id": sid, "text": "x"}


# ilk ayırma: S0 = 0-10 sn, S1 = 10-20 sn
ORIG = [[0.0, 10.0, "S0"], [10.0, 20.0, "S1"]]


def test_cluster_purity():
    # kullanıcı S0'ın 4-10 sn'sini S1'e taşıdı: S0'ın %60'ı başkasıymış
    segs = [_seg(0, 0, 4, "S0"), _seg(1, 4, 10, "S1"), _seg(2, 10, 20, "S1")]
    p = cluster_purity(ORIG, segs)
    assert abs(p["S0"] - 0.4) < 1e-9 and p["S1"] == 1.0
    assert cluster_purity(ORIG, [_seg(0, 0, 20, "S0")])["S1"] == 0.0
    assert cluster_purity([], segs) == {}


def _doc(segs, labels, key="k1", mode="herkes", td=None):
    return {"key": key, "speaker_mode": mode, "td_label": td,
            "speakers": {sid: {"label": lab} for sid, lab in labels.items()}, "segments": segs}


def test_mixed_cluster_is_not_learned_and_old_entry_is_removed(tmp_path):
    save_doc_voices(tmp_path, "k1", "m", {"S0": _vec(0), "S1": _vec(1)}, ORIG)
    lib = VoiceLibrary(tmp_path)
    lib.add("Ali", _vec(0), "m", "k1", "S0")      # önceki kaydetmede öğrenilmişti
    lib.add("Başka", _vec(5), "m", "k2", "S0")    # başka video: dokunulmaz
    segs = [_seg(0, 0, 4, "S0"), _seg(1, 4, 10, "S1"), _seg(2, 10, 20, "S1")]
    skipped = []
    learned = learn_from_doc(tmp_path, _doc(segs, {"S0": "Ali", "S1": "Veli"}), skipped=skipped)
    assert learned == ["Veli"] and skipped == ["Ali"]
    v = lib.load()["voices"]
    assert "Ali" not in v and [x["key"] for x in v["Veli"]["embs"]] == ["k1"]
    assert [x["key"] for x in v["Başka"]["embs"]] == ["k2"]


def test_small_corrections_keep_learning(tmp_path):
    save_doc_voices(tmp_path, "k1", "m", {"S0": _vec(0), "S1": _vec(1)}, ORIG)
    # S0'ın yalnız 0.5 sn'si taşındı (%95 temiz): öğrenilir
    segs = [_seg(0, 0, 9.5, "S0"), _seg(1, 9.5, 10, "S1"), _seg(2, 10, 20, "S1")]
    assert cluster_purity(ORIG, segs)["S0"] >= PURE_MIN
    assert learn_from_doc(tmp_path, _doc(segs, {"S0": "Ali", "S1": "Veli"})) == ["Ali", "Veli"]


def test_video_entries_are_rebuilt_on_each_save(tmp_path):
    """Adı değişen / genel ada dönen konuşmacının eski kaydı kalmaz; aynı ada iki küme ikisi de öğrenilir."""
    save_doc_voices(tmp_path, "k1", "m", {"S0": _vec(0), "S1": _vec(1)}, ORIG)
    segs = [_seg(0, 0, 10, "S0"), _seg(1, 10, 20, "S1")]
    learn_from_doc(tmp_path, _doc(segs, {"S0": "Ali", "S1": "Veli"}))
    learn_from_doc(tmp_path, _doc(segs, {"S0": "Ali", "S1": "Konuşmacı 2"}))
    assert [n["name"] for n in VoiceLibrary(tmp_path).names()] == ["Ali"]
    assert learn_from_doc(tmp_path, _doc(segs, {"S0": "Hoca", "S1": "Hoca"})) == ["Hoca"]
    v = VoiceLibrary(tmp_path).load()["voices"]
    assert list(v) == ["Hoca"] and sorted(x["spk"] for x in v["Hoca"]["embs"]) == ["S0", "S1"]


def test_new_person_without_signature_is_reported(tmp_path):
    save_doc_voices(tmp_path, "k1", "m", {"S0": _vec(0)}, [[0.0, 10.0, "S0"]])
    segs = [_seg(0, 0, 8, "S0"), _seg(1, 8, 10, "ELLE_1")]
    skipped = []
    learned = learn_from_doc(tmp_path, _doc(segs, {"S0": "Ali", "ELLE_1": "Yeni Muhabir"}), skipped=skipped)
    assert learned == [] and skipped == ["Ali", "Yeni Muhabir"]  # Ali %80 temiz: öğrenilmez


def test_old_output_gets_original_turns_before_first_correction(tmp_path):
    save_doc_voices(tmp_path, "k1", "m", {"S0": _vec(0), "S1": _vec(1)})  # 1.9.5 öncesi: tur yok
    doc = _doc([_seg(0, 0, 10, "S0"), _seg(1, 10, 20, "S1")], {"S0": "Ali", "S1": "Veli"})
    assert ensure_orig_turns(tmp_path, doc)
    assert load_doc_voices(tmp_path, "k1")["turns"] == ORIG
    assert not ensure_orig_turns(tmp_path, doc)  # bir kez
    save_doc_voices(tmp_path, "k2", "m", {"S0": _vec(0)})
    assert not ensure_orig_turns(tmp_path, {**doc, "key": "k2", "corrected": "2026-11-01 10:00"})  # önceden düzeltilmiş


def test_gui_save_reports_skipped_and_keeps_original_turns(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    st = tmp_path / ".stt"
    save_doc_voices(st, "k1", "m", {"S0": _vec(0), "S1": _vec(1)})  # eski çıktı (tur yok)
    doc = {"key": "k1", "speaker_mode": "herkes", "title": "t", "source": "https://x/1",
           "speakers": {"S0": {"label": "Konuşmacı 1"}, "S1": {"label": "Konuşmacı 2"}},
           "segments": [_seg(0, 0, 4, "S0"), _seg(1, 4, 10, "S0"), _seg(2, 10, 20, "S1")]}
    p = tmp_path / "metin.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    r = app.speakers_save(str(tmp_path / "metin.txt"), {"labels": {"S0": "Ali", "S1": "Veli"}, "moves": {"1": "S1"}})
    assert r["ok"], r
    assert r["learned"] == ["Veli"] and r["skipped"] == ["Ali"]
    assert load_doc_voices(st, "k1")["turns"] == [[0.0, 4.0, "S0"], [4.0, 10.0, "S0"], [10.0, 20.0, "S1"]]


def test_finalize_keeps_diarization_turns_with_signatures(tmp_path):
    from stt.config import Config
    from stt.pipeline import Item, Job, Pipeline
    out = tmp_path / "out"
    pipe = Pipeline(Config(out_dir=out))
    pipe.fpl = lambda: None
    try:
        item = Item("url", "https://www.youtube.com/watch?v=abc", title="Sohbet podcast #3", key="y_abc" + "0" * 20)
        item.extra.update({"diarize": True, "speaker_mode": "herkes"})
        work = tmp_path / "is"
        (work / "parcalar").mkdir(parents=True)
        segs = [{"start": 0.0, "end": 4.0, "text": "Merhaba.", "words": [{"start": 0.0, "end": 4.0, "word": "Merhaba."}]}]
        (work / "parcalar" / "0000.json").write_text(json.dumps({"segments": segs, "language": "tr"}), encoding="utf-8")
        (work / "konusmaci.json").write_text(json.dumps({"turns": [[0.0, 4.5, "SPEAKER_00"]], "model": "m",
                                                         "emb": {"SPEAKER_00": _vec(3)}}), encoding="utf-8")
        pipe._finalize(Job(item, work, chunks=[{}], info={"title": item.title, "duration_s": 5.0}))
    finally:
        pipe.close()
    assert load_doc_voices(out / ".stt", item.key)["turns"] == [[0.0, 4.5, "SPEAKER_00"]]


# ---------------------------------------------------------------- imzayı düzeltilmiş satırlardan çıkarma
class _EmbModel:
    """Sahte imza modeli: parçanın ortalama genliği -> imza (konuşmacıyı sesin seviyesinden 'tanır')."""
    min_num_samples = 8000

    def __init__(self):
        self.calls = []

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 3 and x.shape[:2] == (1, 1)
        self.calls.append(x.shape[2] / 16000)
        return np.array([[float(np.abs(x).mean()), 1.0, 0.0]])


_TORCH = types.SimpleNamespace(from_numpy=lambda a: np.asarray(a))


def _audio(*parts):
    """[(sn, genlik), ...] -> 16 kHz ses."""
    return np.concatenate([np.full(int(s * 16000), a, dtype=np.float32) for s, a in parts])


def test_voice_embeddings_use_long_clean_segments():
    from stt.tune import VOICE_TRIM, voice_embeddings
    audio = _audio((10, 0.1), (25, 0.5), (1, 0.9))
    segs = [[0, 10, "Ali"], [10, 35, "Veli"], [35, 36, "Ali"]]  # Ali'nin 1 sn'lik satırı kullanılmaz
    m = _EmbModel()
    got = voice_embeddings(m, audio, segs, ["Ali", "Veli", "Yok"], torch=_TORCH)
    assert set(got) == {"Ali", "Veli"}
    assert abs(got["Ali"][0][0] - 0.1) < 1e-6 and abs(got["Veli"][0][0] - 0.5) < 1e-6
    assert abs(got["Ali"][1] - (10 - 2 * VOICE_TRIM)) < 0.05
    assert max(m.calls) <= 10.0 + 1e-6  # uzun satır 10 sn'lik parçalara bölünür
    assert voice_embeddings(None, audio, segs, ["Ali"]) == {}


def test_voice_embeddings_cap_per_person():
    from stt.tune import VOICE_MAX, voice_embeddings
    audio = _audio((400, 0.2))
    got = voice_embeddings(_EmbModel(), audio, [[0, 400, "Ali"]], ["Ali"], torch=_TORCH)
    assert got["Ali"][1] <= VOICE_MAX + 10


def test_run_tune_relearns_voices_from_corrected_lines(tmp_path, monkeypatch):
    from stt import engines, tune
    from stt.config import Config
    from stt.voices import save_reference
    st = tmp_path / "out" / ".stt"
    doc = {"key": "u_1", "title": "P", "source": "https://x/1", "speaker_mode": "basin", "td_label": "Hoca Ali",
           "segments": [{"start": 0, "end": 10, "speaker": "Hoca Ali"}, {"start": 10, "end": 20, "speaker": "Muhabir"},
                        {"start": 20, "end": 30, "speaker": "Yeni Kişi"}, {"start": 30, "end": 31, "speaker": "Kısa"}]}
    save_reference(st, doc)
    lib = VoiceLibrary(st)
    lib.add("Hoca Ali", _vec(0, 3), "m", "u_1", "S0")   # "Kaydet"te küme imzası (karışık olabilir)
    lib.add("Kısa", _vec(1, 3), "m", "u_1", "S9")       # sesi az: eğitimde yenisi çıkmaz, eskisi kalır
    lib.add("Başka", _vec(2, 3), "m", "u_9", "S0")      # başka video: dokunulmaz

    class Ann:
        def itertracks(self, yield_label=True):
            for a, b, s in [[0, 10, "A"], [10, 20, "B"], [20, 30, "C"]]:
                yield types.SimpleNamespace(start=a, end=b), None, s

    class Pipe:
        training = False
        _embedding = _EmbModel()

        def parameters(self, instantiated=True):
            return {"clustering": {"threshold": 0.6}}

        def instantiate(self, p):
            pass

        def __call__(self, f):
            return types.SimpleNamespace(exclusive_speaker_diarization=Ann())

    class FakeD:
        def __init__(self, *a, **k):
            self.pipe, self.model_used, self.agc = Pipe(), "m", False
            self.torch = _TORCH

    monkeypatch.setattr(engines, "Diarizer", FakeD)
    monkeypatch.setattr(tune, "_audio_for", lambda ref, sd, cfg: "x.wav")
    import stt.audio
    monkeypatch.setattr(stt.audio, "load_wav", lambda p: _audio((10, 0.1), (10, 0.3), (10, 0.6), (1, 0.9)))
    res = tune.run_tune(Config(out_dir=tmp_path / "out"), "cuda:0")
    assert res["voices"] == ["Hoca Ali", "Yeni Kişi"]  # "Muhabir" genel etiket: öğrenilmez
    v = VoiceLibrary(st).load()["voices"]
    assert [x["src"] for x in v["Hoca Ali"]["embs"]] == ["duzeltme"] and v["Hoca Ali"]["td"]
    assert abs(v["Hoca Ali"]["embs"][0]["emb"][0] - 0.1) < 1e-4  # karışık küme imzası yerine temiz imza
    assert [x["src"] for x in v["Yeni Kişi"]["embs"]] == ["duzeltme"] and not v["Yeni Kişi"]["td"]
    assert [x["spk"] for x in v["Kısa"]["embs"]] == ["S9"] and [x["key"] for x in v["Başka"]["embs"]] == ["u_9"]
    assert "Muhabir" not in v


# ---------------------------------------------------------------- iki süreç aynı anda yazınca kayıp olmasın
def test_library_lock_prevents_lost_updates(tmp_path):
    lib = VoiceLibrary(tmp_path)

    def work(prefix):
        for i in range(15):
            VoiceLibrary(tmp_path).add(f"{prefix}{i}", _vec(i % 8), "m", f"{prefix}-{i}", "S0")
    ts = [threading.Thread(target=work, args=(p,)) for p in ("a", "b", "c")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(lib.load()["voices"]) == 45
    assert not list(tmp_path.glob("*.lock"))


def test_stale_lock_is_removed_and_unreadable_library_is_not_wiped(tmp_path, monkeypatch):
    lib = VoiceLibrary(tmp_path)
    lib.add("Ali", _vec(0), "m", "k", "S0")
    lock = tmp_path / "sesler.json.lock"
    lock.write_text("")
    old = time.time() - 120
    import os
    os.utime(lock, (old, old))
    with _FileLock(lib.path, timeout=2) as lk:  # çöken süreçten kalmış kilit
        assert lk.held
    # bozuk dosya: yedeklenip boş başlanır (eskiden sessizce "boş" sanılıp üzerine yazılıyordu)
    lib.path.write_text("{bozuk", encoding="utf-8")
    lib.add("Veli", _vec(1), "m", "k2", "S0")
    assert [n["name"] for n in lib.names()] == ["Veli"] and list(tmp_path.glob("sesler.json.bozuk-*"))
    # okunamayan (kilitli) dosya: üzerine yazılmaz, hata verir
    from pathlib import Path
    real = Path.read_text

    def locked(self, *a, **k):
        if self.name == "sesler.json":
            raise PermissionError("kilitli")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", locked)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(PermissionError):
        lib.add("Can", _vec(2), "m", "k3", "S0")
    monkeypatch.setattr(Path, "read_text", real)
    assert [n["name"] for n in lib.names()] == ["Veli"]


def test_gui_save_succeeds_even_if_library_is_locked(tmp_path, monkeypatch):
    """Ses kütüphanesi o an yazılamasa da düzeltmeler kaydedilmiş sayılır (eskiden hepsi 'Kaydedilemedi' olurdu)."""
    import stt.gui as g
    import stt.voices as v
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    save_doc_voices(tmp_path / ".stt", "k1", "m", {"S0": _vec(0)}, [[0.0, 10.0, "S0"]])
    doc = {"key": "k1", "speaker_mode": "herkes", "title": "t", "source": "https://x/1",
           "speakers": {"S0": {"label": "Konuşmacı 1"}}, "segments": [_seg(0, 0, 10, "S0")]}
    (tmp_path / "metin.json").write_text(json.dumps(doc), encoding="utf-8")

    def locked(*a, **k):
        raise PermissionError("kilitli")
    monkeypatch.setattr(v.VoiceLibrary, "replace_video", locked)
    r = app.speakers_save(str(tmp_path / "metin.txt"), {"labels": {"S0": "Ali"}})
    assert r["ok"] and "kaydedildi" in r["warning"]
    assert json.loads((tmp_path / "metin.json").read_text(encoding="utf-8"))["speakers"]["S0"]["label"] == "Ali"
