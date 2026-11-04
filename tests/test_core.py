from pathlib import Path

import pytest

from stt.audio import group_chunks
from stt.fetch import find_media_urls, is_drm_error, manifest_has_drm, needs_cookies
from stt.fpl import fpl_target
from stt.merge import _smooth, apply_labels, assign_speakers, label_speakers, merge_chunks
from stt.outputs import to_srt, to_summary, to_txt, turns_from_segments
from stt.sources import expand_inputs, normalize_url


# ------------------------------------------------------------ linkler
@pytest.mark.parametrize("a,b", [
    ("https://youtu.be/abc123?si=xyz&t=40", "https://www.youtube.com/watch?v=abc123&feature=share"),
    ("https://www.youtube.com/shorts/abc123", "https://m.youtube.com/watch?v=abc123&pp=1"),
    ("https://twitter.com/Arsenal/status/1/", "https://x.com/Arsenal/status/1?s=20"),
    ("https://www.instagram.com/reel/XYZ/?igsh=aa&utm_source=ig", "https://instagram.com/reel/XYZ"),
])
def test_normalize_same(a, b):
    assert normalize_url(a) == normalize_url(b)


def test_normalize_different():
    assert normalize_url("https://youtube.com/watch?v=a") != normalize_url("https://youtube.com/watch?v=b")


def test_expand_inputs(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.mp4").write_bytes(b"0")
    (tmp_path / "a" / "y.mp3").write_bytes(b"0")
    (tmp_path / "a" / "z.mp4.part").write_bytes(b"0")
    (tmp_path / "a" / "notlar.docx").write_bytes(b"0")
    lst = tmp_path / "liste.txt"
    lst.write_text("# yorum\nhttps://youtu.be/abc\n\nhttps://youtube.com/watch?v=abc\na/x.mp4\n", encoding="utf-8")
    items = expand_inputs([str(tmp_path / "a"), str(lst)])
    kinds = sorted((i.kind, Path(i.source).name if i.kind == "file" else i.source) for i in items)
    assert kinds == [("file", "x.mp4"), ("file", "y.mp3"), ("url", "https://youtu.be/abc")]


# ------------------------------------------------------------ indirme yardımcıları
def test_drm_detection():
    assert manifest_has_drm('#EXT-X-KEY:METHOD=SAMPLE-AES,URI="skd://x",KEYFORMAT="com.apple.streamingkeydelivery"')
    assert manifest_has_drm('<ContentProtection schemeIdUri="urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"/>')
    assert not manifest_has_drm('#EXT-X-KEY:METHOD=AES-128,URI="https://k/key"')  # AES-128 DRM değil
    assert is_drm_error("ERROR: [Foo] 123: This video is DRM protected")
    assert needs_cookies("ERROR: [youtube] x: Sign in to confirm you're not a bot")
    assert needs_cookies("ERROR: [instagram] x: Requested content is not available, rate-limit reached or login required")


def test_find_media_urls():
    html = ('<video><source src="/v/clip.mp4"></video>'
            '<script>{"hls":"https:\\/\\/cdn.club.com\\/a\\/master.m3u8?tok=1\\u0026x=2"}</script>'
            '<meta property="og:video" content="https://cdn.club.com/og.mp4">')
    urls = find_media_urls(html, "https://club.com/news/presser")
    assert urls[0] == "https://cdn.club.com/a/master.m3u8?tok=1&x=2"
    assert "https://club.com/v/clip.mp4" in urls and "https://cdn.club.com/og.mp4" in urls


# ------------------------------------------------------------ VAD parçalama
def test_group_chunks_covers_timeline():
    speech = [(i * 10.0, i * 10.0 + 8.0) for i in range(60)]  # 10 dk, 2 sn sessizlikler
    ch = group_chunks(speech, 600.0, target_s=120)
    assert ch[0]["start"] == 0 and ch[-1]["end"] == 600.0
    for a, b in zip(ch, ch[1:]):
        assert a["end"] == b["start"]                       # boşluk/çakışma yok
        assert a["speech_end"] < a["end"] < b["speech_start"]  # kesim sessizlikte
    assert all(c["end"] - c["start"] <= 130 for c in ch)
    assert 4 <= len(ch) <= 6


def test_group_chunks_no_speech():
    assert group_chunks([], 30.0, 60) == [{"start": 0.0, "end": 30.0, "speech_start": 0.0, "speech_end": 30.0}]


# ------------------------------------------------------------ konuşmacılar
def _w(t0, t1, word):
    return {"start": t0, "end": t1, "word": word, "prob": 1.0}


def test_assign_and_label():
    # muhabir 0-4 sn soru, TD 4-20 sn cevap; whisper ikisini tek segmentte vermiş
    seg = {"start": 0.0, "end": 9.0, "text": "How is Saka? He is fine, he trained today.",
           "words": [_w(0.0, 0.5, " How"), _w(0.5, 0.8, " is"), _w(0.8, 1.5, " Saka?"),
                     _w(4.2, 4.5, " He"), _w(4.5, 4.7, " is"), _w(4.7, 5.2, " fine,"), _w(5.3, 5.5, " he"),
                     _w(5.5, 6.0, " trained"), _w(6.0, 6.6, " today.")]}
    seg2 = {"start": 10.0, "end": 19.0, "text": "We will see.", "words": [_w(10, 11, " We"), _w(11, 12, " will"),
                                                                          _w(12, 13, " see.")]}
    turns = [[0.0, 3.0, "SPEAKER_01"], [4.0, 20.0, "SPEAKER_00"], [21.0, 23.0, "SPEAKER_02"]]
    out = assign_speakers([seg, seg2], turns)
    assert [s["speaker_id"] for s in out] == ["SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
    assert out[0]["text"] == "How is Saka?" and out[1]["text"].startswith("He is fine")
    spk = label_speakers(turns, "Teknik direktör", "Muhabir")
    assert spk["SPEAKER_00"]["label"] == "Teknik direktör"
    assert spk["SPEAKER_01"]["label"] == spk["SPEAKER_02"]["label"] == "Muhabir"
    apply_labels(out, spk, "Muhabir")
    doc = {"segments": out, "speakers": spk, "td_label": "Teknik direktör", "title": "T",
           "match": {"home": "Arsenal", "away": "Slavia Prague", "manager": "Mikel Arteta"}}
    txt = to_txt(doc)
    assert txt.startswith("Arsenal vs Slavia Prague - Mikel Arteta | basın toplantısı\nVideo: T\n\n")
    assert "Soru 1 - Muhabir:\nHow is Saka?\n\nCevap - Teknik direktör:\nHe is fine" in txt
    assert len(turns_from_segments(out)) == 2


def test_smooth_single_word_flip():
    assert _smooth(["A", "A", "B", "A", "A"]) == ["A"] * 5
    assert _smooth(["A", "A", "B", "B", "B"]) == ["A", "A", "B", "B", "B"]


def test_merge_chunks_language_and_order():
    parts = [{"language": "tr", "segments": [{"start": 60, "end": 70, "text": "ikinci"}]},
             {"language": "tr", "segments": [{"start": 0, "end": 50, "text": "birinci"}]},
             {"language": "en", "segments": [{"start": 80, "end": 81, "text": "ok"}]}]
    segs, lang = merge_chunks(parts)
    assert [s["text"] for s in segs] == ["birinci", "ikinci", "ok"] and lang == "tr"


# ------------------------------------------------------------ çıktılar
def test_srt_format():
    words = [_w(i * 0.4, i * 0.4 + 0.35, f" kelime{i}" + ("." if i == 29 else "")) for i in range(30)]
    srt = to_srt([{"start": 0, "end": 12, "text": "x", "words": words, "speaker": "Muhabir"}], diarized=True)
    blocks = [b for b in srt.strip().split("\n\n") if b]
    assert blocks[0].startswith("1\n00:00:00,000 --> ")
    assert "[Muhabir]" in blocks[0] and "[Muhabir]" not in blocks[1]
    for b in blocks:
        lines = b.split("\n")[2:]
        assert len(lines) <= 2 and all(len(x) <= 50 for x in lines)


def test_srt_without_words():
    srt = to_srt([{"start": 0, "end": 30, "text": "a " * 120, "words": []}], diarized=False)
    assert srt.count("-->") >= 2


def test_summary_has_sections():
    doc = {"title": "T", "source": "s", "duration_s": 20, "language": "en", "engine": "whisper", "model": "m",
           "created": "now", "td_label": "Teknik direktör", "fpl": {"gw": 11, "team": "ARS"},
           "speakers": {"S0": {"label": "Teknik direktör", "seconds": 15}, "S1": {"label": "Muhabir", "seconds": 5}},
           "segments": [{"start": 0, "end": 4, "text": "How is the squad looking before the derby?", "speaker": "Muhabir"},
                        {"start": 4, "end": 20, "text": "The squad is looking good and everybody trained well today.",
                         "speaker": "Teknik direktör"}]}
    md = to_summary(doc, "x.txt")
    assert "GW11 ARS" in md and "## Konuşmacılar" in md and "Soru sayısı (muhabir turu): **1**" in md


def test_fpl_target(tmp_path):
    assert fpl_target(tmp_path, 7, "ars") == tmp_path / "inputs" / "pressers" / "GW07" / "ARS.txt"


# ------------------------------------------------------------ OpenAI motoru (sahte istemci)
class _Err(Exception):
    status_code = 400


class _FakeTranscriptions:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if kw["response_format"] == "verbose_json":
            raise _Err("response_format 'verbose_json' is not compatible with model")

        class R:
            text = "We are ready for Sunday."
        return R()


def test_openai_engine_fallback(tmp_path, monkeypatch):
    import wave

    import numpy as np

    from stt.engines import OpenAIEngine
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.zeros(16000 * 3, dtype=np.int16).tobytes())
    eng = OpenAIEngine.__new__(OpenAIEngine)
    fake = _FakeTranscriptions()
    eng.client = type("C", (), {"audio": type("A", (), {"transcriptions": fake})()})()
    eng.model, eng.verbose_ok, eng.device = "gpt-transcribe", None, "openai"
    r = eng.transcribe(str(wav), 0.0, 3.0, "en", None, speech_start=0.5, speech_end=2.5)
    assert r["segments"] == [{"start": 0.5, "end": 2.5, "text": "We are ready for Sunday.", "words": []}]
    assert eng.verbose_ok is False
    eng.transcribe(str(wav), 0.0, 3.0, "en", None)
    assert [c["response_format"] for c in fake.calls] == ["verbose_json", "json", "json"]


def test_link_list_with_fpl(tmp_path):
    from stt.sources import read_link_list
    p = tmp_path / "l.txt"
    p.write_text("https://youtu.be/a 11 ARS\nhttps://youtu.be/b GW7 liv\nhttps://youtu.be/c\nD:\\v\\x.mp4 12 MCI\n", "utf-8")
    assert read_link_list(p) == [("https://youtu.be/a", {"fpl": (11, "ARS")}), ("https://youtu.be/b", {"fpl": (7, "LIV")}),
                                 ("https://youtu.be/c", {}), ("D:\\v\\x.mp4", {"fpl": (12, "MCI")})]


def test_match_parsing():
    from stt.match import display_name, parse_match
    m = parse_match({"title": "PRESS CONFERENCE 🎙️ | Mikel Arteta previews Slavia Prague | Champions League",
                     "channel": "Arsenal", "upload_date": "20261103"})
    assert display_name(m, "x") == "Arsenal vs Slavia Prague - Mikel Arteta (2026-11-03)"
    m = parse_match({"title": "PEP GUARDIOLA PRESS CONFERENCE | Man City v Borussia Dortmund", "channel": "Man City"})
    assert (m["home"], m["away"], m["manager"]) == ("Manchester City", "Borussia Dortmund", "Pep Guardiola")
    assert display_name(parse_match({"title": "Post-match press conference"}), "yedek") == "yedek"
    m = parse_match({"title": "Teknik Direktörümüz Okan Buruk'un VfB Stuttgart Maçı Öncesi Açıklamaları",
                     "channel": "Galatasaray"})
    assert display_name(m, "x") == "Galatasaray vs Stuttgart - Okan Buruk"
    m = parse_match({"title": "Frank Lampard Pre-Match Press Conference | Everton vs Coventry City",
                     "channel": "Coventry City"})
    assert display_name(m, "x") == "Everton vs Coventry City - Frank Lampard"


def test_link_list_clipboard_style(tmp_path):
    from stt.sources import read_link_list
    p = tmp_path / "pano.txt"
    p.write_text("Arteta: https://youtu.be/a, Slot -> https://youtu.be/b\nhttps://x.com/A/status/1 https://x.com/B/status/2\n",
                 "utf-8")
    assert [u for u, _ in read_link_list(p)] == ["https://youtu.be/a", "https://youtu.be/b",
                                               "https://x.com/A/status/1", "https://x.com/B/status/2"]


FPL_SAMPLE = {
    "teams": {"1": {"id": 1, "name": "Arsenal", "short_name": "ARS"}, "7": {"id": 7, "name": "Chelsea", "short_name": "CHE"},
              "9": {"id": 9, "name": "Everton", "short_name": "EVE"}, "8": {"id": 8, "name": "Coventry", "short_name": "COV"},
              "14": {"id": 14, "name": "Man City", "short_name": "MCI"}},
    "fixtures": [
        {"id": 101, "event": 11, "kickoff_time": "2026-11-08T16:30:00Z", "team_h": 1, "team_a": 7, "finished": False},
        {"id": 95, "event": 10, "kickoff_time": "2026-11-01T15:00:00Z", "team_h": 14, "team_a": 1, "finished": True,
         "team_h_score": 1, "team_a_score": 2},
        {"id": 102, "event": 11, "kickoff_time": "2026-11-08T15:00:00Z", "team_h": 9, "team_a": 8, "finished": False},
    ]}


def test_dataset_classify_pl_prematch(tmp_path):
    from stt.dataset import classify, match_folder_name, presser_base_name, target_dir
    from stt.match import parse_match
    info = {"title": "Mikel Arteta's pre-Chelsea press conference", "channel": "Arsenal", "upload_date": "20261107",
            "timestamp": 1794052800}  # 2026-11-06 ~
    meta = classify(parse_match(info), info, FPL_SAMPLE)
    assert (meta["gw"], meta["home"], meta["away"], meta["phase"], meta["fpl_short"]) == (11, "Arsenal", "Chelsea",
                                                                                           "mac-oncesi", "ARS")
    assert match_folder_name(meta) == "2026-11-08 GW11 Arsenal vs Chelsea"
    assert presser_base_name(meta, "x").endswith("mac-oncesi - Mikel Arteta")
    assert target_dir(tmp_path, meta, "x") == tmp_path / "Premier League" / "Arsenal" / "2026-11-08 GW11 Arsenal vs Chelsea"


def test_dataset_classify_postmatch_and_away_team(tmp_path):
    from stt.dataset import classify, match_folder_name, target_dir
    from stt.match import parse_match
    info = {"title": "Frank Lampard Post-Match Press Conference | Everton vs Coventry City", "channel": "Coventry City",
            "upload_date": "20261108", "timestamp": 1794172800}  # 2026-11-08 ~ maçtan sonra
    meta = classify(parse_match(info), info, FPL_SAMPLE)
    assert meta["team"] == "Coventry City" and meta["phase"] == "mac-sonrasi" and meta["gw"] == 11
    assert target_dir(tmp_path, meta, "x").parts[-3:] == ("Premier League", "Coventry City",
                                                          "2026-11-08 GW11 Everton vs Coventry City")
    assert match_folder_name(meta) == "2026-11-08 GW11 Everton vs Coventry City"


def test_dataset_classify_cup_and_other(tmp_path):
    from stt.dataset import classify, target_dir
    from stt.match import parse_match
    info = {"title": "PRESS CONFERENCE | Mikel Arteta previews Slavia Prague | Champions League",
            "channel": "Arsenal", "upload_date": "20261103"}
    meta = classify(parse_match(info), info, FPL_SAMPLE)
    assert meta["gw"] is None and meta["competition"] == "Champions League"
    assert target_dir(tmp_path, meta, "x").name == "2026-11-03 Champions League Arsenal vs Slavia Prague"
    info = {"title": "Okan Buruk'un VfB Stuttgart maçı öncesi açıklamaları", "channel": "Galatasaray",
            "upload_date": "20261105"}
    meta = classify(parse_match(info), info, FPL_SAMPLE)
    assert target_dir(tmp_path, meta, "x").parts[-3:] == ("Diger Takimlar", "Galatasaray",
                                                          "2026-11-05 Galatasaray vs Stuttgart")
    assert meta["phase"] == "mac-oncesi"


def test_index_upsert(tmp_path):
    from stt.dataset import update_index
    update_index(tmp_path, {"key": "a", "team": "Arsenal", "league": "Premier League"})
    update_index(tmp_path, {"key": "b", "team": "Chelsea", "league": "Premier League"})
    update_index(tmp_path, {"key": "a", "team": "Arsenal", "league": "Premier League", "gw": 11})
    lines = (tmp_path / "index.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 2 and '"gw": 11' in lines[0]
    assert (tmp_path / "index.csv").read_text("utf-8-sig").splitlines()[0].startswith("key,presser_date")


def test_dataset_infers_team_from_opponent_fixture():
    from stt.dataset import classify, match_folder_name
    from stt.match import parse_match
    info = {"title": "Mikel Arteta's pre-Chelsea press conference", "timestamp": 1794052800}
    meta = classify(parse_match(info), info, FPL_SAMPLE)
    assert meta["team"] == "Arsenal" and meta["is_pl"] and match_folder_name(meta) == "2026-11-08 GW11 Arsenal vs Chelsea"
