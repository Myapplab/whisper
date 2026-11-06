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
