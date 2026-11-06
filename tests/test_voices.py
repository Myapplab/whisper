import json

import numpy as np

from stt.engines import chunk_spans, speaker_embeddings
from stt.voices import (VoiceLibrary, apply_corrections, apply_voice_names, is_generic, learn_from_doc,
                        load_doc_voices, reference_turns, save_doc_voices)


def _vec(seed, dim=16):
    return list(np.random.default_rng(seed).normal(size=dim))


def _near(v, seed, eps=0.05):
    return list(np.asarray(v) + eps * np.random.default_rng(seed).normal(size=len(v)))


def test_chunk_spans_cover_everything_once():
    n = 1000
    seen = np.zeros(n, int)
    for a, b, c, d in chunk_spans(n, 300, 20):
        assert a <= c < d <= b and a >= 0 and b <= n
        seen[c:d] += 1
    assert (seen == 1).all()
    assert list(chunk_spans(0, 300, 20)) == []


def test_speaker_embeddings_skip_empty_rows():
    cents = np.array([[1.0, 0.0], [0.0, 0.0], [np.nan, 1.0]])
    assert speaker_embeddings(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], cents) == {"SPEAKER_00": [1.0, 0.0]}
    assert speaker_embeddings(["A"], None) is None


def test_library_identifies_known_voice(tmp_path):
    lib = VoiceLibrary(tmp_path)
    arteta, slot = _vec(1), _vec(2)
    assert lib.add("Arteta", arteta, "m", "k1", "SPEAKER_00", td=True)
    lib.add("Slot", slot, "m", "k1", "SPEAKER_01")
    got = lib.identify({"SPEAKER_03": _near(arteta, 5), "SPEAKER_04": _vec(9)}, "m", 0.6)
    assert set(got) == {"SPEAKER_03"} and got["SPEAKER_03"][0] == "Arteta"
    assert lib.identify({"SPEAKER_03": _near(arteta, 5)}, "baska-model", 0.6) == {}  # modeller karışmaz
    assert lib.td_names() == {"Arteta"}


def test_library_one_name_per_speaker(tmp_path):
    lib = VoiceLibrary(tmp_path)
    v = _vec(1)
    lib.add("Arteta", v, "m", "k1", "S0")
    got = lib.identify({"A": _near(v, 2, 0.01), "B": _near(v, 3, 0.2)}, "m", 0.5)
    assert list(got) == ["A"]


def test_library_rename_does_not_duplicate(tmp_path):
    lib = VoiceLibrary(tmp_path)
    lib.add("Konuk", _vec(1), "m", "k1", "S0")
    lib.add("Ali Koç", _vec(1), "m", "k1", "S0")  # aynı video + konuşmacı yeniden adlandırıldı
    assert [x["name"] for x in lib.names()] == ["Ali Koç"]
    assert lib.remove("Ali Koç") and lib.names() == []


def test_apply_voice_names_presser_moves_td():
    sp = {"S0": {"label": "Teknik direktör", "seconds": 300}, "S1": {"label": "Muhabir", "seconds": 200}}
    td = apply_voice_names(sp, {"S1": ("Arteta", 0.8)}, "basin", "Teknik direktör", "Muhabir", {"Arteta"})
    assert td == "Arteta" and sp["S1"]["label"] == "Arteta" and sp["S0"]["label"] == "Muhabir"


def test_apply_voice_names_everyone():
    sp = {"S0": {"label": "Konuşmacı 1"}, "S1": {"label": "Konuşmacı 2"}}
    apply_voice_names(sp, {"S1": ("Ali", 0.7)}, "herkes", "x", "Muhabir")
    assert sp["S1"]["label"] == "Ali" and sp["S0"]["label"] == "Konuşmacı 1" and sp["S1"]["voice"] == 0.7


def _doc():
    return {"key": "k1", "title": "t", "speaker_mode": "herkes", "td_label": None,
            "speakers": {"S0": {"label": "Konuşmacı 1", "seconds": 4}, "S1": {"label": "Konuşmacı 2", "seconds": 2}},
            "segments": [{"id": 0, "start": 0, "end": 2, "speaker_id": "S0", "speaker": "Konuşmacı 1", "text": "a"},
                         {"id": 1, "start": 2, "end": 4, "speaker_id": "S0", "speaker": "Konuşmacı 1", "text": "b"},
                         {"id": 2, "start": 4, "end": 6, "speaker_id": "S1", "speaker": "Konuşmacı 2", "text": "c"}]}


def test_apply_corrections_rename_move_and_new_speaker():
    doc = apply_corrections(_doc(), {"S0": "Ali", "ELLE_1": "Veli"}, {"1": "S1", 2: "ELLE_1"})
    assert [s["speaker"] for s in doc["segments"]] == ["Ali", "Konuşmacı 2", "Veli"]
    assert doc["speakers"]["S0"]["seconds"] == 2 and doc["speakers"]["ELLE_1"]["label"] == "Veli"
    assert reference_turns(doc) == [[0.0, 2.0, "Ali"], [2.0, 4.0, "Konuşmacı 2"], [4.0, 6.0, "Veli"]]


def test_apply_corrections_presser_td_follows_rename():
    d = _doc()
    d.update(speaker_mode="basin", td_label="Teknik direktör")
    d["speakers"]["S0"]["label"] = "Teknik direktör"
    apply_corrections(d, {"S0": "Arteta"})
    assert d["td_label"] == "Arteta"
    apply_corrections(d, {}, {}, td="S1")  # yanlış konuşmacı TD seçilmişti
    assert d["td_label"] == "Konuşmacı 2"


def test_learn_from_doc_only_named_speakers(tmp_path):
    save_doc_voices(tmp_path, "k1", "m", {"S0": _vec(1), "S1": _vec(2)})
    assert load_doc_voices(tmp_path, "k1")["model"] == "m"
    doc = apply_corrections(_doc(), {"S0": "Ali"})
    assert learn_from_doc(tmp_path, doc, ignore=("Muhabir",)) == ["Ali"]
    assert [x["name"] for x in VoiceLibrary(tmp_path).names()] == ["Ali"]
    assert is_generic("Konuşmacı 3") and is_generic("muhabir", ("Muhabir",)) and not is_generic("Ali")


def test_finalize_names_known_voice(tmp_path):
    from stt.config import Config
    from stt.pipeline import Item, Job, Pipeline
    out = tmp_path / "out"
    lib = VoiceLibrary(out / ".stt")
    ali = _vec(7)
    lib.add("Ali", ali, "m", "eski", "S9")
    pipe = Pipeline(Config(out_dir=out))
    pipe.fpl = lambda: None
    try:
        item = Item("url", "https://www.youtube.com/watch?v=abc", title="Sohbet podcast #3", key="y_abc" + "0" * 20)
        item.extra.update({"diarize": True, "speaker_mode": "herkes"})
        work = tmp_path / "is"
        (work / "parcalar").mkdir(parents=True)
        segs = [{"start": 0.0, "end": 4.0, "text": "Merhaba.", "words": [{"start": 0.0, "end": 4.0, "word": "Merhaba."}]},
                {"start": 5.0, "end": 9.0, "text": "Selam.", "words": [{"start": 5.0, "end": 9.0, "word": "Selam."}]}]
        (work / "parcalar" / "0000.json").write_text(json.dumps({"segments": segs, "language": "tr"}), encoding="utf-8")
        (work / "konusmaci.json").write_text(json.dumps({
            "turns": [[0.0, 4.5, "SPEAKER_00"], [4.8, 9.0, "SPEAKER_01"]], "model": "m",
            "emb": {"SPEAKER_00": _vec(3), "SPEAKER_01": _near(ali, 4)}}), encoding="utf-8")
        job = Job(item, work, chunks=[{}], info={"title": item.title, "duration_s": 10.0})
        pipe._finalize(job)
        assert job.status == "done", job.error
        assert pipe._want_vocals(item) is True
    finally:
        pipe.close()
    docs = [p for p in out.rglob("*.json") if ".stt" not in p.parts and p.name != "mac.json"]
    doc = next(json.loads(p.read_text(encoding="utf-8")) for p in docs if "segments" in p.read_text(encoding="utf-8"))
    assert [s["speaker"] for s in doc["segments"]] == ["Konuşmacı 1", "Ali"]
    assert doc["settings"]["diar_model"] == "m"
    assert load_doc_voices(out / ".stt", item.key)["emb"]


class _Seg:
    def __init__(self, s, e):
        self.start, self.end = s, e


class _Ann:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def labels(self):
        return sorted({r[2] for r in self.rows})

    def itertracks(self, yield_label=True):
        for s, e, k in self.rows:
            yield _Seg(s, e), None, k


def _fake(v4, out):
    import types

    from stt.engines import Diarizer
    d = Diarizer.__new__(Diarizer)
    d.torch = types.SimpleNamespace(from_numpy=lambda a: np.asarray(a))
    d.v4, d.kw = v4, None

    def pipe(file, **kw):
        d.kw = kw
        return out
    d.pipe = pipe
    return d


def test_diarizer_reads_v4_output_and_embeddings():
    full = _Ann([(0, 5, "SPEAKER_00"), (4, 9, "SPEAKER_01")])
    excl = _Ann([(0, 4.5, "SPEAKER_00"), (4.5, 9, "SPEAKER_01")])
    import types
    out = types.SimpleNamespace(speaker_diarization=full, exclusive_speaker_diarization=excl,
                                speaker_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]))
    d = _fake(True, out)
    turns = d._run(np.zeros(16000, np.float32), {"min_speakers": 2})
    assert turns == [[0, 4.5, "SPEAKER_00"], [4.5, 9, "SPEAKER_01"]] and "return_embeddings" not in d.kw
    assert d.last_emb == {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}


def test_diarizer_reads_v3_tuple_with_embeddings():
    ann = _Ann([(0, 5, "SPEAKER_00")])
    d = _fake(False, (ann, np.array([[0.5, 0.5]])))
    assert d._run(np.zeros(16000, np.float32), {}) == [[0, 5, "SPEAKER_00"]]
    assert d.kw.get("return_embeddings") is True and d.last_emb == {"SPEAKER_00": [0.5, 0.5]}
