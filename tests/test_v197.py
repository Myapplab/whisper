"""v1.9.7: 1.9.6'nın 'ilk cümle yutulmasın' kuralları ayırma sınırı geç kalınca yeni konuşmacının ilk kelimelerini
önceki konuşmacıya veriyordu (gerileme). Baştaki konuşmacı bir önceki konuşmacıysa eski davranış (çoğunluk oyu)."""
from __future__ import annotations

from stt.merge import assign_speakers


def _words(text: str, t0: float, step: float = 0.3) -> list[dict]:
    out, t = [], t0
    for tok in text.split():
        out.append({"start": round(t, 2), "end": round(t + step - 0.02, 2), "word": " " + tok})
        t += step
    return out


def _seg(ws: list[dict]) -> dict:
    return {"id": 0, "start": ws[0]["start"], "end": ws[-1]["end"],
            "text": "".join(w["word"] for w in ws).strip(), "words": ws}


def _labels(out):
    return [(s["speaker_id"], s["text"]) for s in out]


A_TEXT = "We were very good in the first half and I am happy."
B_TEXT = "Well I think the team showed a lot of character today and we deserved it."


def test_late_boundary_head_stays_with_new_speaker_same_segment():
    """A bitirdi, B başladı; ayırma B'ye 0.9 sn geç geçti ("Well I think" A'da). Hepsi B'nin cümlesi."""
    a = _words(A_TEXT, 0.0)
    b = _words(B_TEXT, a[-1]["end"] + 0.1)
    cut = b[0]["start"] + 0.9
    out = assign_speakers([_seg(a + b)], [[0, cut, "A"], [cut, 30, "B"]])
    assert _labels(out) == [("A", A_TEXT), ("B", B_TEXT)]


def test_late_boundary_head_stays_with_new_speaker_next_segment():
    a = _words(A_TEXT, 0.0)
    b = _words(B_TEXT, a[-1]["end"] + 0.1)
    cut = b[0]["start"] + 0.9
    out = assign_speakers([_seg(a), _seg(b)], [[0, cut, "A"], [cut, 30, "B"]])
    assert _labels(out) == [("A", A_TEXT), ("B", B_TEXT)]


def test_late_boundary_at_comma_not_split_off():
    """Geç sınır tam virgüle denk gelince de "Well, I think," A'ya geçmesin."""
    a = _words(A_TEXT, 0.0)
    text = "Well, I think, the team showed a lot of character today and we deserved it."
    b = _words(text, a[-1]["end"] + 0.1)
    cut = b[2]["end"] + 0.05
    out = assign_speakers([_seg(a + b)], [[0, cut, "A"], [cut, 30, "B"]])
    assert _labels(out) == [("A", A_TEXT), ("B", text)]


def test_new_speaker_short_head_after_other_speaker_still_split():
    """1.9.6'nın düzelttiği durum korunuyor: önceki konuşmacı yok/farklı, baştaki kısa söz ayrı kalır."""
    r = _words("Thank you Mikel", 0.0, 0.35)
    t = _words("so the game today was hard. We tried. We pushed. We lost.", r[-1]["end"] + 0.05)
    out = assign_speakers([_seg(r + t)], [[0, r[-1]["end"], "R"], [r[-1]["end"], 30, "T"]])
    assert _labels(out)[0] == ("R", "Thank you Mikel")
    assert all(s["speaker_id"] == "T" for s in out[1:])


# ------------------------------------------------------------------ ses kütüphanesi (1.9.6 inceleme bulguları)

def _doc(labels: dict, segs: list) -> dict:
    return {"key": "vid1", "speaker_mode": "herkes", "corrected": True,
            "speakers": {k: {"label": v, "seconds": 5.0} for k, v in labels.items()},
            "segments": [{"id": i, "start": a, "end": b, "speaker_id": sid, "speaker": labels[sid], "text": "x"}
                         for i, (a, b, sid) in enumerate(segs)]}


def test_trained_voice_not_kept_after_name_swap(tmp_path):
    """'Kaydet ve eğit': Ali = S1'in satırları. Sonra kullanıcı adları değiştiriyor (S1 = Veli, S2 = Ali) ve düz
    'Kaydet'. Eski temiz imza artık Veli'nin sesi: Ali'de kalmamalı, Ali S2'nin imzasını öğrenmeli."""
    from stt.voices import VoiceLibrary, learn_from_doc, save_doc_voices, spans_fp
    v1, v2, va, vv = [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]
    lib = VoiceLibrary(tmp_path)
    old = [[0.0, 5.0, "Ali"], [5.0, 10.0, "Veli"]]
    lib.replace_video("vid1", "m", [("Ali", va, False, "duzeltme:Ali", spans_fp(old, "Ali")),
                                    ("Veli", vv, False, "duzeltme:Veli", spans_fp(old, "Veli"))], src="duzeltme")
    save_doc_voices(tmp_path, "vid1", "m", {"S1": v1, "S2": v2}, turns=[[0.0, 5.0, "S1"], [5.0, 10.0, "S2"]])
    learn_from_doc(tmp_path, _doc({"S1": "Veli", "S2": "Ali"}, [(0.0, 5.0, "S1"), (5.0, 10.0, "S2")]))
    voices = VoiceLibrary(tmp_path).load()["voices"]
    assert [e["emb"] for e in voices["Ali"]["embs"]] == [v2]
    assert [e["emb"] for e in voices["Veli"]["embs"]] == [v1]


def test_trained_voice_kept_after_text_edit_and_same_speaker_split(tmp_path):
    """Yalnız metin düzeltme ya da aynı kişinin satırını ikiye bölme temiz imzayı silmez."""
    from stt.voices import VoiceLibrary, learn_from_doc, save_doc_voices, spans_fp
    va, v1 = [1.0, 0, 0, 0], [0, 1.0, 0, 0]
    lib = VoiceLibrary(tmp_path)
    old = [[0.0, 10.0, "Ali"], [10.0, 12.0, "Muhabir"]]
    lib.replace_video("vid1", "m", [("Ali", va, False, "duzeltme:Ali", spans_fp(old, "Ali"))], src="duzeltme")
    save_doc_voices(tmp_path, "vid1", "m", {"S1": v1}, turns=[[0.0, 12.0, "S1"]])  # karışık küme
    learned = learn_from_doc(tmp_path, _doc({"S1": "Ali", "S2": "Muhabir"},
                                            [(0.0, 4.8, "S1"), (5.0, 10.0, "S1"), (10.0, 12.0, "S2")]),
                             ignore=("Muhabir",))
    assert learned == ["Ali"]
    embs = VoiceLibrary(tmp_path).load()["voices"]["Ali"]["embs"]
    assert [(e["src"], e["emb"]) for e in embs] == [("duzeltme", va)]


def test_cluster_purity_counts_simultaneous_rows_once():
    """'👥 Aynı anda' satırı (aynı saniyede ikinci konuşmacı) kümeyi karışık göstermesin."""
    from stt.voices import cluster_purity
    segs = [{"start": 0.0, "end": 10.0, "speaker_id": "X"}, {"start": 4.0, "end": 6.0, "speaker_id": "Y"},
            {"start": 0.0, "end": 10.0, "speaker_id": "X2"}]
    assert cluster_purity([[0.0, 10.0, "X"]], segs[:2]) == {"X": 1.0}
    assert cluster_purity([[0.0, 10.0, "X"]], [segs[1], segs[2]]) == {"X": 0.0}
    # üst üste binmeyen satırlarda eski sonuç
    assert cluster_purity([[0.0, 10.0, "X"]], [{"start": 0.0, "end": 8.0, "speaker_id": "X"},
                                               {"start": 8.0, "end": 10.0, "speaker_id": "Y"}]) == {"X": 0.8}


def test_shortened_row_words_clamped_to_row_time():
    """Süresi elle kısaltılan satırın kelimeleri satır dışına taşmasın (.srt yeni bitişi göstersin)."""
    from stt.voices import apply_corrections
    w = lambda a, b, t: {"start": a, "end": b, "word": " " + t}  # noqa: E731
    doc = {"key": "k", "speaker_mode": "herkes",
           "speakers": {"S": {"label": "Konuşmacı 1", "seconds": 3.6}},
           "segments": [{"id": 0, "start": 9.0, "end": 12.6, "speaker_id": "S", "text": "Son söz burada.",
                         "words": [w(9, 10, "Son"), w(10, 11, "söz"), w(11, 12.6, "burada.")]}]}
    apply_corrections(doc, segments=[{"id": 0, "orig_id": 0, "start": 9.0, "end": 12.0, "sid": "S",
                                      "text": "Son söz burada.",
                                      "words": doc["segments"][0]["words"]}])
    s = doc["segments"][0]
    assert s["end"] == 12.0 and len(s["words"]) == 3 and s["words"][-1]["end"] == 12.0
