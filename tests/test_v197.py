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
