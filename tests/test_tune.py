"""Konuşmacı ayırmayı düzeltilmiş videolarla eğitme (stt/tune.py)."""
import json

from stt import tune


def test_speaker_error_maps_names_and_counts_confusion():
    ref = [[0, 10, "Ahmet"], [10, 20, "Mehmet"]]
    assert tune.speaker_error(ref, [[0, 10, "SPEAKER_01"], [10, 20, "SPEAKER_00"]])[0] == 0  # adlar eşlenir
    bad, tot = tune.speaker_error(ref, [[0, 20, "SPEAKER_00"]])  # herkes tek kişi: yarısı yanlış
    assert 0.4 < bad / tot < 0.6
    bad, _ = tune.speaker_error(ref, [[0, 5, "A"], [5, 10, "B"], [10, 20, "C"]])  # Ahmet ikiye bölündü
    assert 4 < bad < 6


class _P:
    def __init__(self):
        self.p = {"segmentation": {"min_duration_off": 0.0}, "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}}
        self.inst = None

    def parameters(self, instantiated=False):
        return json.loads(json.dumps(self.p))

    def instantiate(self, params):
        self.p = params
        self.inst = params


def test_apply_params_and_candidates():
    p = _P()
    done = tune.apply_params(p, {"clustering.threshold": 0.7, "yok.boyle": 1})
    assert done == {"clustering.threshold": 0.7} and p.p["clustering"]["threshold"] == 0.7
    base = {k: v for k, v in tune.flat_params(p.parameters()).items() if k != "clustering.Fb"}
    cands = tune.candidates(base)
    assert len(cands) > 20 and all(set(c) == set(base) for c in cands)
    assert any(c["clustering.threshold"] == 0.4 for c in cands)


def test_references_and_saved_params(tmp_path):
    from stt.voices import save_reference
    doc = {"key": "u_1", "title": "Podcast", "source": "https://x/1", "speaker_mode": "herkes",
           "segments": [{"start": 0, "end": 5, "speaker": "Ali"}, {"start": 5, "end": 9, "speaker": "Veli"}]}
    save_reference(tmp_path, doc)
    save_reference(tmp_path, {"key": "u_2", "source": "x", "segments": [{"start": 0, "end": 5, "speaker": "Ali"}]})
    refs = tune.references(tmp_path)
    assert [r["key"] for r in refs] == ["u_1"] and refs[0]["speaker_mode"] == "herkes"
    tune.save_tuned(tmp_path, "m", {"clustering.threshold": 0.55}, {"before": 30, "after": 12})
    assert tune.load_tuned(tmp_path)["m"]["params"] == {"clustering.threshold": 0.55}


def test_diarizer_applies_tuned_params():
    from stt.engines import Diarizer
    d = Diarizer.__new__(Diarizer)
    d.tuned = {"m": {"params": {"clustering.threshold": 0.5}}}
    p = _P()
    import sys
    import types
    fake = types.ModuleType("pyannote.audio")
    fake.__version__ = "4.0.7"
    fake.Pipeline = types.SimpleNamespace(from_pretrained=lambda model, token=None: p)
    parent = sys.modules.setdefault("pyannote", types.ModuleType("pyannote"))
    old, old_attr = sys.modules.get("pyannote.audio"), getattr(parent, "audio", None)
    sys.modules["pyannote.audio"] = fake
    parent.audio = fake
    try:
        d.token, d.batch = "t", 8
        d._load_model("m")
    finally:
        if old is None:
            sys.modules.pop("pyannote.audio", None)
        else:
            sys.modules["pyannote.audio"] = old
        if old_attr is None:
            delattr(parent, "audio")
        else:
            parent.audio = old_attr
    assert p.p["clustering"]["threshold"] == 0.5


def test_run_tune_end_to_end_with_fake_model(tmp_path, monkeypatch):
    """Eşik 0.75'te doğru (2 kişi), 0.6'da herkes tek kişi çıkan sahte model: eğitim 0.75'i bulup kaydetmeli."""
    import types

    import numpy as np

    from stt import engines
    from stt.config import Config
    from stt.voices import save_reference
    st = tmp_path / "out" / ".stt"
    save_reference(st, {"key": "u_1", "title": "P", "source": "https://x/1", "speaker_mode": "basin",
                        "segments": [{"start": 0, "end": 30, "speaker": "Ali"}, {"start": 30, "end": 60, "speaker": "Veli"}]})

    class Ann:
        def __init__(self, turns):
            self.t = turns

        def itertracks(self, yield_label=True):
            for a, b, s in self.t:
                yield types.SimpleNamespace(start=a, end=b), None, s

    class Pipe(_P):
        training = False

        def __call__(self, f):
            th = self.p["clustering"]["threshold"]
            turns = [[0, 30, "A"], [30, 60, "B"]] if 0.7 <= th <= 0.8 else [[0, 60, "A"]]
            return types.SimpleNamespace(exclusive_speaker_diarization=Ann(turns))

    class FakeD:
        def __init__(self, *a, **k):
            self.pipe, self.model_used, self.agc = Pipe(), "m", False
            self.torch = types.SimpleNamespace(from_numpy=lambda a: np.asarray(a))

    monkeypatch.setattr(engines, "Diarizer", FakeD)
    monkeypatch.setattr(tune, "_audio_for", lambda ref, sd, cfg: "x.wav")
    import stt.audio
    monkeypatch.setattr(stt.audio, "load_wav", lambda p: np.zeros(16000, dtype=np.float32))
    res = tune.run_tune(Config(out_dir=tmp_path / "out"), "cuda:0")
    assert res["saved"] and res["after"] < res["before"]
    assert 0.7 <= tune.load_tuned(st)["m"]["params"]["clustering.threshold"] <= 0.8
