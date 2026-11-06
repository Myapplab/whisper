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
