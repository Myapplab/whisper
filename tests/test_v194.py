"""1.9.4: konuşmacı sayısı sınırları, konuşmacı ekranında gerçek süre, anahtarlar, DiariZen yedeği, diar_agc,
eğitim referansında boşluk."""
from stt.config import Config
from stt.pipeline import Pipeline
from stt.sources import Item


def _bounds(tmp_path, title, mode=None, speakers=None, **cfg):
    pipe = Pipeline(Config(out_dir=tmp_path / "o", **cfg))
    try:
        it = Item("url", "u", title=title, key="k")
        if mode:
            it.extra["speaker_mode"] = mode
        if speakers is not None:
            it.extra["speakers"] = speakers
        t = pipe._diarize_task(it, tmp_path / "ses.wav")
        return t["min_speakers"], t["max_speakers"], t["vocals"]
    finally:
        pipe.close()


PRESS = "Mikel Arteta Press Conference | Arsenal vs Chelsea"


def test_press_conference_gets_at_least_two_speakers(tmp_path):
    assert _bounds(tmp_path, PRESS) == (2, None, False)


def test_podcast_or_single_person_video_is_not_forced_to_two(tmp_path):
    # eskiden her modda en az 2 -> pyannote tek kişiyi ikiye bölüyordu
    assert _bounds(tmp_path, "Amerikan Mutfak #11") == (None, None, True)
    assert _bounds(tmp_path, PRESS, mode="herkes") == (None, None, True)


def test_count_given_for_the_job_is_exact(tmp_path):
    assert _bounds(tmp_path, PRESS, speakers=3)[:2] == (3, 3)
    assert _bounds(tmp_path, "Amerikan Mutfak #11", speakers=1)[:2] == (1, 1)
    assert _bounds(tmp_path, PRESS, speakers="x")[:2] == (2, None)  # geçersiz sayı = verilmemiş


def test_config_min_speakers_is_a_bound_not_an_exact_count(tmp_path):
    # eskiden stt.toml'da yalnız min_speakers = 3 yazınca en çok da 3 oluyordu
    assert _bounds(tmp_path, "Amerikan Mutfak #11", min_speakers=3)[:2] == (3, None)
    assert _bounds(tmp_path, PRESS, min_speakers=3)[:2] == (3, None)
    assert _bounds(tmp_path, PRESS, max_speakers=6)[:2] == (2, 6)
    # CLI --speakers N ikisini birden N yapar: kesin sayı
    assert _bounds(tmp_path, "Amerikan Mutfak #11", min_speakers=4, max_speakers=4)[:2] == (4, 4)


def test_min_never_above_max(tmp_path):
    assert _bounds(tmp_path, PRESS, max_speakers=1)[:2] == (1, 1)
    assert _bounds(tmp_path, PRESS, min_speakers=5, max_speakers=3)[:2] == (3, 3)


def test_speakers_view_gets_real_segment_end(tmp_path, monkeypatch):
    import json

    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    doc = {"key": "k1", "speaker_mode": "basin", "td_label": "Hoca",
           "speakers": {"SPEAKER_00": {"label": "Hoca", "seconds": 4.0}, "SPEAKER_01": {"label": "Muhabir", "seconds": 1.0}},
           "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Soru?", "speaker_id": "SPEAKER_01"},
                        {"id": 1, "start": 26.0, "end": 30.0, "text": "Cevap.", "speaker_id": "SPEAKER_00"}]}
    p = tmp_path / "metin.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    r = app.speakers_load(str(tmp_path / "metin.txt"))
    assert r["ok"], r
    # arada 25 sn sessizlik var: muhabirin süresi 1 sn olmalı (eskiden `end` gitmiyordu, 26 sn sayılıyordu)
    assert [(s["start"], s["end"]) for s in r["segments"]] == [(0.0, 1.0), (26.0, 30.0)]


def test_gui_script_is_valid_javascript(tmp_path):
    """Arayüz burada açılamıyor: en azından betikte sözdizimi hatası olmasın (node varsa)."""
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node yok")
    html = (Path(__file__).resolve().parent.parent / "stt" / "gui.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert scripts
    js = tmp_path / "gui.js"
    js.write_text("\n".join(scripts), encoding="utf-8")
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]


def _settings_app(tmp_path, monkeypatch):
    """Gerçek stt.toml'a dokunmadan arayüz ayarları (geçici klasörde)."""
    import stt.config as c
    import stt.gui as g
    for env in c.ENV_MAP:
        monkeypatch.delenv(env, raising=False)
    toml = tmp_path / "stt.toml"
    monkeypatch.setattr(c, "config_files", lambda: [toml] if toml.exists() else [])
    orig = g.save_settings
    monkeypatch.setattr(g, "save_settings", lambda upd, path=toml: orig(upd, path))
    monkeypatch.chdir(tmp_path)
    return g.App(), toml


def test_openai_key_from_settings_and_keys_can_be_cleared(tmp_path, monkeypatch):
    app, toml = _settings_app(tmp_path, monkeypatch)
    s = app.save_settings({"openai_api_key": " sk-test ", "hf_token": "hf_x", "llm_provider": "openai"})
    assert s["openai_key_set"] and s["hf_token_set"] and s["secret_env"] == {}
    assert app._cfg.openai_api_key == "sk-test" and app.llm().key("openai") == "sk-test"
    s = app.save_settings({"openai_api_key": "", "hf_token": ""})  # boş kutu = mevcut kalsın
    assert s["openai_key_set"] and s["hf_token_set"]
    s = app.save_settings({"clear": ["openai_api_key", "out_dir", "bogus"]})  # yalnız anahtarlar silinebilir
    assert not s["openai_key_set"] and s["hf_token_set"]
    text = toml.read_text(encoding="utf-8")
    assert "openai_api_key" not in text and "hf_x" in text and "llm_provider" in text
    s = app.save_settings({"clear": ["hf_token"]})
    assert not s["hf_token_set"] and app._cfg.hf_token is None


def test_key_from_environment_is_reported_not_deletable(tmp_path, monkeypatch):
    app, toml = _settings_app(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    s = app.save_settings({"clear": ["openai_api_key"]})
    assert s["openai_key_set"] and s["secret_env"] == {"openai_api_key": "OPENAI_API_KEY"}


def test_openai_engine_takes_key_from_settings(monkeypatch):
    import pytest
    pytest.importorskip("openai")
    from stt.engines import OpenAIEngine
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="anahtarı yok"):
        OpenAIEngine("gpt-transcribe")
    assert OpenAIEngine("gpt-transcribe", api_key="sk-x").client.api_key == "sk-x"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert OpenAIEngine("gpt-transcribe").client.api_key == "sk-env"


def _fake_torch(monkeypatch):
    """torch kurulu olmayan test ortamı için Diarizer.__init__'in kullandığı kadarı."""
    import sys
    import types
    frac = []
    t = types.ModuleType("torch")
    t.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True, enabled=True))
    t.set_num_threads = lambda n: None
    t.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None,
                                   set_per_process_memory_fraction=lambda f, i: frac.append((f, i)))
    t.device = lambda s: ("device", s)
    monkeypatch.setitem(sys.modules, "torch", t)
    import stt.engines as e
    monkeypatch.setattr(e, "setup_cuda_dlls", lambda: None)
    return frac


class _LoadedPipe:
    def __init__(self):
        self.moved = []

    def to(self, dev):
        self.moved.append(dev)
        return self


def test_diarizen_does_not_load_pyannote_or_need_token(tmp_path, monkeypatch):
    import pytest

    from stt import diarizen
    from stt.engines import Diarizer
    frac = _fake_torch(monkeypatch)
    loads = []
    monkeypatch.setattr(Diarizer, "_load", lambda self: loads.append(1) or _LoadedPipe())
    d = Diarizer("m", None, "cuda:1", gpu_frac=0.9, engine="diarizen")  # token yok: hata vermez
    assert d.pipe is None and loads == [] and d.device == "cuda:1" and frac == [(0.9, 1)]
    # DiariZen hata verirse yedek pyannote ister: token yoksa açık mesaj
    monkeypatch.setattr(diarizen, "installed", lambda: True)
    monkeypatch.setattr(diarizen, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("çöktü")))
    from test_stability import _wav
    with pytest.raises(RuntimeError, match="yedek pyannote için HF token"):
        d.run(_wav(tmp_path))
    assert loads == []
    # pyannote seçiliyse eskisi gibi: token şart, model hemen yüklenip karta taşınır
    with pytest.raises(RuntimeError, match="HF token gerekli"):
        Diarizer("m", None, "cuda:1")
    d2 = Diarizer("m", "hf_x", "cuda:1")
    assert loads == [1] and d2.pipe.moved == [("device", "cuda:1")]
    # DiariZen + token: yedek gerekince bir kez yüklenir, karta taşınır
    d3 = Diarizer("m", "hf_x", "cuda:1", engine="diarizen")
    assert d3.pipe is None and loads == [1]
    p = d3._ensure_pipe()
    assert loads == [1, 1] and p.moved == [("device", "cuda:1")] and d3._ensure_pipe() is p


def test_no_token_warning_when_diarizen_is_ready(tmp_path, monkeypatch):
    from stt import diarizen
    app, _ = _settings_app(tmp_path, monkeypatch)
    r = app.start("https://www.youtube.com/watch?v=abcdefghijk", [], {"diarize": True})
    assert "HuggingFace token" in (r["warning"] or "")
    app._cfg.diar_engine = "diarizen"
    monkeypatch.setattr(diarizen, "installed", lambda: False)  # seçili ama kurulu değil: token yine gerekir
    assert not diarizen.token_free(app._cfg)
    monkeypatch.setattr(diarizen, "installed", lambda: True)
    assert diarizen.token_free(app._cfg)
    r = app.start("https://www.youtube.com/watch?v=bcdefghijkl", [], {"diarize": True})
    assert "HuggingFace token" not in (r["warning"] or "")


def test_diar_agc_follows_agc_unless_set():
    from stt.config import diar_agc_on
    assert diar_agc_on(Config()) is True  # eski davranış: ayırma da agc'li
    assert diar_agc_on(Config(agc=False)) is False
    assert diar_agc_on(Config(diar_agc=False)) is False
    assert diar_agc_on(Config(agc=False, diar_agc=True)) is True


def test_diar_agc_and_openai_key_reach_workers(tmp_path, monkeypatch):
    import stt.pipeline as pl
    seen = {}

    class FakePool:
        def __init__(self, devices, ecfg, threads=False, names=None):
            seen.update(ecfg)

        def start(self):
            pass

        def close(self):
            pass
    monkeypatch.setattr(pl, "WorkerPool", FakePool)
    pipe = Pipeline(Config(out_dir=tmp_path / "o", engine="openai", diar_agc=False, openai_api_key="sk-x"))
    try:
        pipe.start()
    finally:
        pipe.close()
    assert seen["diar_agc"] is False and seen["agc"] is True and seen["openai_api_key"] == "sk-x"


def test_diar_worker_uses_diar_agc_and_reports_it(tmp_path, monkeypatch):
    import stt.engines as e
    from stt.workers import _handle
    made = {}

    class FakeDiarizer:
        device, last_emb, model_used = "cuda:0", None, "m"

        def __init__(self, model, token, dev, agc=True, **kw):
            made["agc"], self.agc = agc, agc

        def run(self, wav, lo, hi, vocals=False, beat=None):
            return [[0.0, 1.0, "SPEAKER_00"]]
    monkeypatch.setattr(e, "Diarizer", FakeDiarizer)
    res = _handle({"kind": "diarize", "wav": "x.wav"}, "diar", "cuda:0",
                  {"diar_model": "m", "hf_token": "t", "agc": True, "diar_agc": False}, {})
    assert made["agc"] is False and res["agc"] is False


def test_gui_diar_agc_setting(tmp_path, monkeypatch):
    app, toml = _settings_app(tmp_path, monkeypatch)
    assert app.get_settings()["diar_agc"] == "auto"
    s = app.save_settings({"diar_agc": "off"})
    assert s["diar_agc"] == "off" and app._cfg.diar_agc is False and app._restart.is_set()
    s = app.save_settings({"diar_agc": "auto"})
    assert s["diar_agc"] == "auto" and app._cfg.diar_agc is None and "diar_agc" not in toml.read_text(encoding="utf-8")


def test_output_records_diar_agc_and_engine(tmp_path):
    """K5 ölçümünde hangi ayarla / hangi motorla ayrıldığı bilinsin: çıktıda diar_agc ve diar_model."""
    import json

    from stt.pipeline import Job
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
        (work / "konusmaci.json").write_text(json.dumps({
            "turns": [[0.0, 4.5, "SPEAKER_00"]], "model": "BUT-FIT/diarizen-wavlm-large-s80-md", "agc": False}),
            encoding="utf-8")
        job = Job(item, work, chunks=[{}], info={"title": item.title, "duration_s": 5.0})
        pipe._finalize(job)
        assert job.status == "done", job.error
    finally:
        pipe.close()
    doc = next(json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("*.json")
               if ".stt" not in p.parts and "segments" in p.read_text(encoding="utf-8"))
    assert doc["settings"]["diar_agc"] is False
    assert doc["settings"]["diar_model"] == "BUT-FIT/diarizen-wavlm-large-s80-md"


def test_reference_keeps_unmerged_segments_for_training(tmp_path):
    import json

    from stt.tune import speaker_error
    from stt.voices import reference_turns, save_reference
    doc = {"key": "u_1", "source": "https://x/1", "speaker_mode": "basin", "td_label": "Ali",
           "segments": [{"start": 0, "end": 3, "speaker": "Ali"}, {"start": 3.8, "end": 7, "speaker": "Ali"},
                        {"start": 8, "end": 12, "speaker": "Veli"}]}
    assert reference_turns(doc) == [[0, 7, "Ali"], [8, 12, "Veli"]]  # ekran / ölçüm: eskisi gibi birleşik
    assert reference_turns(doc, merge_gap=0) == [[0, 3, "Ali"], [3.8, 7, "Ali"], [8, 12, "Veli"]]
    p = save_reference(tmp_path, doc)
    saved = json.loads(p.read_text(encoding="utf-8"))
    assert saved["segs"] == reference_turns(doc, 0) and saved["td_label"] == "Ali"
    # pyannote aradaki 0.8 sn'de sessizlik diyor: bu konuşmacı hatası değil
    hyp = [[0, 3, "A"], [3.8, 7, "A"], [8, 12, "B"]]
    assert speaker_error(saved["turns"], hyp)[0] > 0.5     # birleşik turla: sessizlik "hata" sayılıyordu
    assert speaker_error(saved["segs"], hyp)[0] == 0.0


def test_run_tune_scores_with_unmerged_segments(tmp_path, monkeypatch):
    import types

    import numpy as np

    from stt import engines, tune
    from stt.voices import save_reference
    st = tmp_path / "out" / ".stt"
    save_reference(st, {"key": "u_1", "title": "P", "source": "https://x/1", "speaker_mode": "basin",
                        "segments": [{"start": 0, "end": 3, "speaker": "Ali"}, {"start": 3.8, "end": 7, "speaker": "Ali"},
                                     {"start": 8, "end": 12, "speaker": "Veli"}]})

    class Ann:
        def __init__(self, turns):
            self.t = turns

        def itertracks(self, yield_label=True):
            for a, b, s in self.t:
                yield types.SimpleNamespace(start=a, end=b), None, s

    class Pipe:
        training = False

        def parameters(self, instantiated=True):
            return {"clustering": {"threshold": 0.6}}

        def instantiate(self, p):
            pass

        def __call__(self, f):
            return types.SimpleNamespace(exclusive_speaker_diarization=Ann([[0, 3, "A"], [3.8, 7, "A"], [8, 12, "B"]]))

    class FakeD:
        def __init__(self, *a, **k):
            self.pipe, self.model_used, self.agc = Pipe(), "m", False
            self.torch = types.SimpleNamespace(from_numpy=lambda a: np.asarray(a))

    monkeypatch.setattr(engines, "Diarizer", FakeD)
    monkeypatch.setattr(tune, "_audio_for", lambda ref, sd, cfg: "x.wav")
    import stt.audio
    monkeypatch.setattr(stt.audio, "load_wav", lambda p: np.zeros(16000, dtype=np.float32))
    res = tune.run_tune(Config(out_dir=tmp_path / "out"), "cuda:0")
    assert res["before"] == 0.0  # eskiden aradaki sessizlik yüzünden ~%8 "hata" görünüyordu
