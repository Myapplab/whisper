"""Senaryo turu 3: birleştirme, altyazı, ayarlar, linkler, uydurma satırlar, uyku engeli (eski turlar dahil)."""
import math

from stt import config, merge, outputs, util
from stt.match import safe_filename
from stt.sources import expand_inputs, is_url, normalize_url

from test_stability import _job, _pipe


def _w(t, word, d=0.3):
    return {"start": t, "end": t + d, "word": word}


# ---- tur 1
def test_far_apart_same_sentence_is_kept_twice():
    s = "Thank you very much everyone."
    out, _ = merge.merge_chunks([{"language": "en", "segments": [
        {"start": 0, "end": 2, "text": s}, {"start": 600, "end": 602, "text": s}]}])
    assert [(x["start"], x["end"]) for x in out] == [(0, 2), (600, 602)]


def test_back_to_back_loop_is_collapsed():
    s = "Thank you very much everyone."
    out, _ = merge.merge_chunks([{"segments": [{"start": 0, "end": 2, "text": s}, {"start": 2.1, "end": 4, "text": s}]}])
    assert len(out) == 1 and out[0]["end"] == 4


def test_srt_cues_do_not_overlap():
    srt = outputs.to_srt([{"start": 0, "end": 2, "text": "a", "speaker": "A"},
                          {"start": 1.9, "end": 2.0, "text": "b", "speaker": "B"}], True)
    assert "00:00:00,000 --> 00:00:01,900" in srt and "[B] b" in srt


def test_timestamps_survive_inf_and_nan():
    for v in (math.inf, math.nan, -5, None):
        assert util.fmt_ts(v) == "00:00:00,000" and util.fmt_clock(v) == "00:00"


def test_windows_reserved_names():
    assert safe_filename("CON") == "_CON" and safe_filename("nul.txt") == "_nul.txt"
    assert safe_filename("Conte basın toplantısı") == "Conte basın toplantısı"


# ---- tur 2
def test_bad_port_env_does_not_break_startup(monkeypatch):
    monkeypatch.setenv("STT_API_PORT", "abc")
    assert config.load_config().api_port == 0
    monkeypatch.setenv("STT_API_PORT", "70000")
    assert config.load_config().api_port == 0


def test_start_api_bad_port_returns_none():
    from stt.api import start_api
    assert start_api(None, 70000) is None


def test_follow_interval_clamped():
    c = config.Config()
    c.follow_interval_h = 0
    assert config.normalize(c).follow_interval_h == 1.0
    c.follow_interval_h = math.nan
    assert config.normalize(c).follow_interval_h == 6.0


def test_corrupt_follow_file(tmp_path):
    from stt.channel import Follows
    (tmp_path / "takip.json").write_text('{"a": 1}', encoding="utf-8")
    assert Follows(tmp_path).all() == [] and Follows(tmp_path).due(6) == []


# ---- tur 3
def test_youtube_link_forms_are_one_video():
    i = "dQw4w9WgXcQ"
    forms = [f"https://www.youtube.com/watch?v={i}", f"https://youtu.be/{i}?si=abc", f"https://m.youtube.com/watch?v={i}&t=3",
             f"https://www.youtube-nocookie.com/embed/{i}", f"https://www.youtube.com/v/{i}", f"youtube.com/watch?v={i}",
             f"https://www.youtube.com/shorts/{i}", f"https://music.youtube.com/watch?v={i}&list=PL1"]
    assert len({normalize_url(f) for f in forms}) == 1


def test_schemeless_link_is_url_and_gets_https():
    assert is_url("youtu.be/abc") and is_url("www.youtube.com/watch?v=1")
    items = expand_inputs(["youtube.com/watch?v=abc"])
    assert items[0].kind == "url" and items[0].source.startswith("https://")


def test_malformed_url_does_not_crash():
    assert normalize_url("http://[::1") == "http://[::1"


# ---- araştırma
def test_known_hallucinations_are_dropped():
    segs = [{"start": 0, "end": 2, "text": "Altyazı M.K."},
            {"start": 3, "end": 5, "text": "Thanks for watching!", "no_speech_prob": 0.8, "avg_logprob": -0.4},
            {"start": 6, "end": 8, "text": "Thanks for watching!", "no_speech_prob": 0.05, "avg_logprob": -0.2},
            {"start": 9, "end": 12, "text": "We made three subs by half time."}]
    out, _ = merge.merge_chunks([{"language": "en", "segments": segs}])
    assert [s["start"] for s in out] == [6, 9]


def test_keep_awake_follows_activity(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("stt.pipeline.keep_awake", calls.append)
    p = _pipe(tmp_path)
    _job(p, tmp_path)
    p.pump(0.01)
    p.jobs.clear()
    p.pump(0.01)
    p.pump(0.01)
    assert calls == [True, False]


# ---- tur 4
def test_repeat_loop_does_not_return_when_split_by_speaker():
    loop = "We will see on Sunday. " * 4
    ws, t = [], 0.0
    for w in loop.split():
        ws.append(_w(t, " " + w))
        t += 0.35
    ws.append(_w(t + 0.1, " Next"))
    segs, _ = merge.merge_chunks([{"segments": [{"start": 0, "end": t + 0.4, "text": loop.strip() + " Next",
                                                 "words": ws}]}])
    out = merge.assign_speakers(segs, [[0, 3, "A"], [3, t + 1, "B"]])
    assert " ".join(o["text"] for o in out).count("Sunday") == 1


def test_early_first_word_after_silence_goes_to_next_speaker():
    ws = [_w(7.0, " So")] + [_w(10 + i * 0.3, " w", 0.25) for i in range(8)] + [_w(12.7, " end.")]
    out = merge.assign_speakers([{"start": 7, "end": 13, "text": "So ...", "words": ws}], [[0, 8, "A"], [9.8, 14, "B"]])
    assert {o["speaker_id"] for o in out} == {"B"}


def test_real_short_answer_with_punctuation_keeps_speaker():
    ws = [_w(7.0, " Yes.")] + [_w(10 + i * 0.3, " w", 0.25) for i in range(8)] + [_w(12.7, " end.")]
    out = merge.assign_speakers([{"start": 7, "end": 13, "text": "Yes. ...", "words": ws}], [[0, 8, "A"], [9.8, 14, "B"]])
    assert out[0]["speaker_id"] == "A" and out[-1]["speaker_id"] == "B"


def test_bom_jsonl_first_row_kept(tmp_path):
    f = tmp_path / "i.jsonl"
    f.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8-sig")
    assert util.read_jsonl(f) == [{"a": 1}, {"b": 2}]


def test_atomic_write_retries_on_lock(tmp_path, monkeypatch):
    real, n = util.os.replace, []

    def flaky(a, b):
        n.append(1)
        if len(n) < 3:
            raise PermissionError("kilitli")
        real(a, b)
    monkeypatch.setattr(util.os, "replace", flaky)
    monkeypatch.setattr(util.time, "sleep", lambda s: None)
    util.atomic_write_text(tmp_path / "x.txt", "ok")
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "ok" and len(n) == 3


# ---- kullanıcı ekranı (1.8.6): "CUDA error: out of memory" ile ayırma hiç tekrar denenmeden düşüyordu
def test_diar_cuda_oom_is_retried_later_then_memory_hint(tmp_path):
    from test_sim import _diar_setup
    p, job, tasks, d, fin = _diar_setup(tmp_path)
    d["vocals"] = True
    err = ("RuntimeError: Konuşmacı ayırma ekran kartında başarısız oldu (CUDA error: out of memory CUDA kernel "
           "errors might be asynchronously reported)")
    for k in (1, 2):
        p._on_msg({"type": "start", "wid": "diar", "tid": d["id"]})
        p._on_msg({"type": "error", "wid": "diar", "tid": d["id"], "error": err})
        assert not job.diar_error and len(p.ddelayed) == 1 and "diar" in p.dpool.respawned
        _, retry = p.ddelayed.pop()
        assert retry["vocals"] is (k == 1)
    p._on_msg({"type": "start", "wid": "diar", "tid": d["id"]})
    p._on_msg({"type": "error", "wid": "diar", "tid": d["id"], "error": err})
    assert "bellek yetmedi" in job.diar_error and "NVIDIA" not in job.diar_error


# ---- K4: kısa soru cevabın içine karışmasın
def _conv(parts):
    ws, turns, t = [], [], 0.0
    for spk, text, gap in parts:
        t += gap
        a = t
        for w in text.split():
            ws.append(_w(t, " " + w))
            t += 0.35
        turns.append([a, t, spk])
    seg = {"start": ws[0]["start"], "end": ws[-1]["end"], "text": "".join(x["word"] for x in ws).strip(), "words": ws}
    return [(o["speaker_id"], o["text"]) for o in merge.assign_speakers([seg], turns)]


def test_short_question_at_sentence_end_is_kept():
    out = _conv([("TD", "I think we were good", 0), ("R", "and Saka?", 0.2), ("TD", "He is fine, he will play.", 0.2)])
    assert out == [("TD", "I think we were good"), ("R", "and Saka?"), ("TD", "He is fine, he will play.")]


def test_single_word_flicker_in_question_is_smoothed():
    out = _conv([("TD", "Why would we change", 0), ("R", "now?", 0.0)])
    assert [s for s, _ in out] == ["TD"]


# ---- K5: ölçüm (düzeltmeden önceki program çıktısı saklanır)
def test_measure_uses_saved_program_output(tmp_path):
    from stt.tune import measure
    from stt.voices import save_reference
    doc = {"key": "u_x", "title": "Basın", "segments": [
        {"start": 0, "end": 10, "speaker": "Hoca"}, {"start": 10, "end": 20, "speaker": "Muhabir"}]}
    hyp = [[0, 15, "S0"], [15, 20, "S1"]]  # program 10-15 arasını yanlış kişiye vermiş
    save_reference(tmp_path, doc, hyp)
    save_reference(tmp_path, doc, None)  # ikinci düzeltme ilk program çıktısını silmez
    m = measure(tmp_path)
    assert m["videos"] == 1 and 20 < m["error"] < 35


# ---- K6: DiariZen (ayrı ortam) kullanılır; hata verirse pyannote
def test_diarizen_used_and_falls_back(tmp_path, monkeypatch):
    from test_stability import _Pipe, _fake_diarizer, _wav

    from stt import diarizen
    monkeypatch.setattr(diarizen, "installed", lambda: True)
    monkeypatch.setattr(diarizen, "run", lambda wav, a, b, beat=None: [[0, 1, "dz0"], [1, 2, "dz1"]])
    pipe = _Pipe([])
    d = _fake_diarizer(pipe, cpu_pipe=None)
    d.engine = "diarizen"
    assert [t[2] for t in d.run(_wav(tmp_path))] == ["dz0", "dz1"] and pipe.calls == 0

    def boom(*a, **k):
        raise RuntimeError("DiariZen hata verdi")
    monkeypatch.setattr(diarizen, "run", boom)
    assert [t[2] for t in d.run(_wav(tmp_path))] == ["SPEAKER_00", "SPEAKER_01"] and pipe.calls == 1


# ---- K7: tek uzun açılış konuşması yapan basın sözcüsü TD sayılmasın
def test_td_is_many_long_answers_not_one_long_intro():
    turns = [[0, 70, "PR"]]  # sözcü 70 sn açılış
    t = 70.0
    for _ in range(4):  # 4 soru-cevap: soru 6 sn, cevap 15 sn
        turns += [[t, t + 6, "R"], [t + 6, t + 21, "TD"]]
        t += 21
    lab = merge.label_speakers(turns, "Hoca", "Muhabir")
    assert lab["TD"]["label"] == "Hoca" and lab["PR"]["label"] == "Muhabir"


def test_td_falls_back_to_total_without_long_turns():
    lab = merge.label_speakers([[0, 3, "A"], [3, 4, "B"], [4, 7, "A"]], "Hoca", "Muhabir")
    assert lab["A"]["label"] == "Hoca"
