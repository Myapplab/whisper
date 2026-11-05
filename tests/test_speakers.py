"""Podcast / film modu: her konuşmacı ayrı etiket, diyalog biçimi, ayrı klasör, yeniden işleme."""
import json

from stt.config import Config
from stt.match import looks_like_presser
from stt.merge import label_speakers_all
from stt.outputs import to_summary, to_txt
from stt.pipeline import Job, Pipeline
from stt.sources import Item


def _w(s, e, t):
    return {"start": s, "end": e, "word": t, "probability": 0.9}


SEGS = [
    {"start": 0.0, "end": 3.0, "text": "Welcome to the show.", "words": [_w(0, 1, " Welcome"), _w(1, 2, " to the"), _w(2, 3, " show.")]},
    {"start": 4.0, "end": 7.0, "text": "Thanks for having me.", "words": [_w(4, 5, " Thanks"), _w(5, 6, " for having"), _w(6, 7, " me.")]},
    {"start": 8.0, "end": 11.0, "text": "Hi everyone.", "words": [_w(8, 9, " Hi"), _w(9, 11, " everyone.")]},
    {"start": 12.0, "end": 20.0, "text": "So what about Curry?", "words": [_w(12, 14, " So what"), _w(14, 20, " about Curry?")]},
]
TURNS = [[0.0, 3.2, "SPEAKER_02"], [3.8, 7.2, "SPEAKER_00"], [7.8, 11.2, "SPEAKER_01"], [11.8, 20.0, "SPEAKER_02"]]


def test_label_all_by_first_appearance():
    spk = label_speakers_all(TURNS)
    assert [spk[k]["label"] for k in ("SPEAKER_02", "SPEAKER_00", "SPEAKER_01")] == \
        ["Konuşmacı 1", "Konuşmacı 2", "Konuşmacı 3"]
    assert list(spk) == ["SPEAKER_02", "SPEAKER_00", "SPEAKER_01"]


def test_looks_like_presser():
    assert looks_like_presser({"title": "Oliver Glasner Press Conference | Brentford vs Nottingham Forest"})
    assert looks_like_presser({"title": "Okan Buruk maç sonrası açıklamalar"})
    assert looks_like_presser({"title": "Qarabağ - məşqçi mətbuat konfransı"})
    assert looks_like_presser({"title": "Arteta on Saka"}, {"manager": "Mikel Arteta"})
    assert not looks_like_presser({"title": "Golden State vs Luka"}, {"manager": "Golden State"})
    assert not looks_like_presser({"title": "Golden State, Portland, Dallas, Washington, Curry vs. Luka | Amerikan Mutfak #11"})


def _run(tmp_path, title, mode=None, speakers=None):
    pipe = Pipeline(Config(out_dir=tmp_path / "out"))
    pipe.fpl = lambda: None
    try:
        item = Item("url", "https://www.youtube.com/watch?v=abc", title=title, key="y_abc" + "0" * 20)
        item.extra.update({"diarize": True, **({"speaker_mode": mode} if mode else {}),
                           **({"speakers": speakers} if speakers else {})})
        work = tmp_path / "is"
        (work / "parcalar").mkdir(parents=True)
        (work / "parcalar" / "0000.json").write_text(json.dumps({"segments": SEGS, "language": "en"}), encoding="utf-8")
        (work / "konusmaci.json").write_text(json.dumps({"turns": TURNS}), encoding="utf-8")
        job = Job(item, work, chunks=[{}], info={"title": title, "duration_s": 20.0})
        pipe._finalize(job)
        assert job.status == "done", job.error
        return pipe, job
    finally:
        pipe.close()


def test_podcast_auto_mode_everyone_separate(tmp_path):
    pipe, job = _run(tmp_path, "Golden State, Portland, Curry vs. Luka | Amerikan Mutfak #11")
    doc = job.result
    assert doc["speaker_mode"] == "herkes" and doc["td_label"] is None
    assert "Podcast ve Film" in str(job.out_dir)
    txt = job.paths["txt"].read_text(encoding="utf-8")
    assert "Soru" not in txt and "Muhabir" not in txt
    assert "[00:00] Konuşmacı 1:\nWelcome to the show." in txt
    assert "[00:04] Konuşmacı 2:\nThanks for having me." in txt
    assert "[00:12] Konuşmacı 1:\nSo what about Curry?" in txt
    srt = job.paths["srt"].read_text(encoding="utf-8")
    assert "[Konuşmacı 3]" in srt
    md = to_summary(doc, "x.txt")
    assert "Soru sayısı" not in md and "Konuşmacı 3" in md
    row = json.loads((tmp_path / "out" / "index.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["league"] == "Podcast ve Film" and row["questions"] == 0
    info = pipe.state.get(job.item.key)["info"]
    assert info["speaker_mode"] == "herkes"


def test_press_conference_keeps_qa(tmp_path):
    _, job = _run(tmp_path, "Oliver Glasner Press Conference | Brentford vs Nottingham Forest")
    txt = job.paths["txt"].read_text(encoding="utf-8")
    assert job.result["speaker_mode"] == "basin"
    assert "Soru 1 - Muhabir" in txt and "Konuşmacı 1" not in txt


def test_forced_mode_and_speaker_count(tmp_path):
    _, job = _run(tmp_path, "Oliver Glasner Press Conference", mode="herkes")
    assert job.result["speaker_mode"] == "herkes"
    pipe = Pipeline(Config(out_dir=tmp_path / "o2"))
    try:
        it = Item("url", "u", title="t", key="k")
        it.extra["speakers"] = 3
        assert pipe._speakers_wanted(it) == 3
        it.extra["speakers"] = "x"
        assert pipe._speakers_wanted(it) is None
    finally:
        pipe.close()


def test_old_qa_output_of_podcast_is_redone(tmp_path):
    pipe = Pipeline(Config(out_dir=tmp_path / "out"))
    try:
        it = Item("url", "u", title="Amerikan Mutfak #11", key="k")
        it.extra["diarize"] = True
        prev = {"status": "done", "title": "Golden State vs Luka | Amerikan Mutfak #11",
                "info": {"diarized": True, "language": "en"}}  # 1.5.0 öncesi: biçim kaydı yok = soru-cevap
        assert "herkes ayrı" in pipe._redo_reason(it, prev)
        prev["info"]["speaker_mode"] = "herkes"
        assert pipe._redo_reason(it, prev) == ""
        pp = {"status": "done", "title": "Arteta press conference", "info": {"diarized": True, "language": "en"}}
        assert pipe._redo_reason(it, pp) == ""  # basın toplantısı: eski çıktı doğru
        it.extra["speaker_mode"] = "herkes"
        assert pipe._redo_reason(it, pp)  # kullanıcı açıkça podcast seçti
    finally:
        pipe.close()


def test_general_txt_header():
    doc = {"title": "Pod", "speaker_mode": "herkes", "segments": [{**SEGS[0], "speaker": "Konuşmacı 1"}],
           "speakers": {"S0": {"label": "Konuşmacı 1", "seconds": 3.0}}, "match": {"team": "Arsenal"}}
    txt = to_txt(doc)
    assert txt.startswith("Pod\n1 konuşmacı: Konuşmacı 1 (00:03)\n") and "basın toplantısı" not in txt


# ------------------------------------------------ cümle düzeyinde konuşmacı atama (1.6.3)
def _ws(t0, words, step=0.4):
    return [_w(round(t0 + i * step, 2), round(t0 + (i + 1) * step, 2), " " + x) for i, x in enumerate(words)]


def test_boundary_drift_does_not_split_sentence():
    from stt.merge import assign_speakers
    # TD cümlesinin son 3 kelimesi kelime zamanı kaydığı için muhabirin turuna düşüyor
    ws = _ws(0.0, "I think the team played really well tonight honestly.".split())
    seg = {"start": 0.0, "end": ws[-1]["end"], "text": "x", "words": ws}
    turns = [[0.0, 2.3, "TD"], [2.3, 6.0, "MUH"]]
    out = assign_speakers([seg], turns)
    assert [s["speaker_id"] for s in out] == ["TD"]


def test_next_sentence_goes_to_other_speaker():
    from stt.merge import assign_speakers
    ws = _ws(0.0, "We will see. What about Saka?".split())
    seg = {"start": 0.0, "end": ws[-1]["end"], "text": "x", "words": ws}
    out = assign_speakers([seg], [[0.0, 1.3, "TD"], [1.3, 3.0, "MUH"]])
    assert [(s["speaker_id"], s["text"]) for s in out] == [("TD", "We will see."), ("MUH", "What about Saka?")]


def test_long_change_inside_sentence_is_split():
    from stt.merge import assign_speakers
    a = _ws(0.0, "and that is why we pressed so high today and".split())
    b = _ws(a[-1]["end"], "sorry can I ask about the injury news".split())
    seg = {"start": 0.0, "end": b[-1]["end"], "text": "x", "words": a + b}
    out = assign_speakers([seg], [[0.0, a[-1]["end"], "TD"], [b[0]["start"], 10.0, "MUH"]])
    assert [s["speaker_id"] for s in out] == ["TD", "MUH"]
    assert out[1]["text"].startswith("sorry")


def test_exclusive_turns_longer_wins_overlap():
    from stt.merge import exclusive_turns
    turns = [[0.0, 10.0, "TD"], [4.0, 4.6, "MUH"], [9.0, 12.0, "MUH"]]
    assert exclusive_turns(turns) == [[0.0, 10.0, "TD"], [10.0, 12.0, "MUH"]]


def test_smooth_by_duration():
    from stt.merge import _smooth
    assert _smooth(["A", "B", "B", "A"], durs=[1, 0.3, 0.3, 1]) == ["A"] * 4
    assert _smooth(["A", "B", "B", "A"], durs=[1, 0.8, 0.8, 1]) == ["A", "B", "B", "A"]
