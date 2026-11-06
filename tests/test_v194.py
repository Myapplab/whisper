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
