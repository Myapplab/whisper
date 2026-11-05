"""Parça sonuçlarını birleştirme ve konuşmacı etiketleme."""
from __future__ import annotations

from collections import Counter, defaultdict


def merge_chunks(parts: list[dict]) -> tuple[list[dict], str | None]:
    segs: list[dict] = []
    lang_dur: Counter = Counter()
    for p in parts:
        for s in p.get("segments", []):
            if not s.get("text"):
                continue
            s = {**s, "text": collapse_repeats(s["text"])}
            segs.append(s)
            if p.get("language"):
                lang_dur[p["language"]] += max(0.0, s["end"] - s["start"])
    segs.sort(key=lambda s: (s["start"], s["end"]))
    # olası tekrarları (parça sınırında) temizle
    out: list[dict] = []
    for s in segs:
        if out and s["text"] == out[-1]["text"] and s["start"] < out[-1]["end"]:
            continue
        # Whisper tekrar döngüsü: aynı uzun cümle art arda (ör. 8 kez) yazılmışsa bir kez tut
        if out and len(s["text"]) > 15 and _norm(s["text"]) == _norm(out[-1]["text"]):
            out[-1] = {**out[-1], "end": max(out[-1]["end"], s["end"])}
            continue
        out.append(s)
    lang = lang_dur.most_common(1)[0][0] if lang_dur else None
    return out, lang


_SENT = __import__("re").compile(r"(?<=[.!?…])\s+")


def collapse_repeats(text: str) -> str:
    """Bir segment içinde art arda tekrarlanan uzun cümleleri teke indir (Whisper tekrar döngüsü)."""
    parts = _SENT.split(text)
    out: list[str] = []
    for p in parts:
        if out and len(p) > 15 and _norm(p) == _norm(out[-1]):
            continue
        out.append(p)
    return " ".join(out)


def _norm(t: str) -> str:
    return "".join(ch for ch in t.lower() if ch.isalnum())


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_at(t0: float, t1: float, turns: list[list], idx_hint: int = 0) -> tuple[str | None, int]:
    best, best_ov = None, 0.0
    i = max(0, idx_hint - 3)
    while i < len(turns) and turns[i][0] <= t1 + 0.01:
        ov = _overlap(t0, t1, turns[i][0], turns[i][1])
        if ov > best_ov:
            best, best_ov = turns[i][2], ov
        i += 1
    if best is None:  # örtüşme yoksa 1 sn içindeki en yakın konuşma
        mid = (t0 + t1) / 2
        dist, near = 1.0, None
        for s, e, spk in turns:
            d = 0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
            if d < dist:
                dist, near = d, spk
        best = near
    j = idx_hint
    while j < len(turns) and turns[j][1] < t0:
        j += 1
    return best, j


def _smooth(labels: list[str | None], min_run: int = 2) -> list[str | None]:
    """Tek kelimelik konuşmacı sıçramalarını (min_run'dan kısa) komşu konuşmacıya kat."""
    out = list(labels)
    runs: list[list[int]] = []  # [başlangıç, bitiş)
    for i, lab in enumerate(out):
        if runs and out[runs[-1][0]] == lab:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    if len(runs) < 2:
        return out
    for k, (a, b) in enumerate(runs):
        if b - a >= min_run:
            continue
        left = out[runs[k - 1][0]] if k > 0 else None
        right = out[runs[k + 1][0]] if k + 1 < len(runs) else None
        repl = left if left is not None else right
        if repl is not None:
            for i in range(a, b):
                out[i] = repl
    return out


def assign_speakers(segments: list[dict], turns: list[list]) -> list[dict]:
    """Kelime düzeyinde konuşmacı ata; segment içinde konuşmacı değişirse segmenti böl."""
    turns = sorted(turns, key=lambda t: t[0])
    out: list[dict] = []
    hint = 0
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            spk, hint = _speaker_at(seg["start"], seg["end"], turns, hint)
            out.append({**seg, "speaker_id": spk})
            continue
        labels = []
        for w in words:
            spk, hint = _speaker_at(w["start"], max(w["end"], w["start"] + 0.02), turns, hint)
            labels.append(spk)
        labels = _smooth(labels)
        # ardışık aynı konuşmacılı kelime grupları -> alt segmentler
        groups: list[tuple[str | None, list[dict]]] = []
        for w, lab in zip(words, labels):
            if groups and groups[-1][0] == lab:
                groups[-1][1].append(w)
            else:
                groups.append((lab, [w]))
        if len(groups) == 1:
            out.append({**seg, "speaker_id": groups[0][0]})
            continue
        for lab, ws in groups:
            text = "".join(w["word"] for w in ws).strip()
            if not text:
                continue
            out.append({**seg, "start": ws[0]["start"], "end": ws[-1]["end"], "text": text,
                        "words": ws, "speaker_id": lab})
    return out


def label_speakers(turns: list[list], td_label: str, reporter_label: str) -> dict[str, dict]:
    """En uzun konuşan = teknik direktör, diğerleri = muhabir."""
    tot: dict[str, float] = defaultdict(float)
    for s, e, spk in turns:
        tot[spk] += max(0.0, e - s)
    if not tot:
        return {}
    td = max(tot, key=tot.get)
    return {spk: {"label": td_label if spk == td else reporter_label, "seconds": round(sec, 1)}
            for spk, sec in sorted(tot.items(), key=lambda kv: -kv[1])}


def apply_labels(segments: list[dict], speakers: dict[str, dict], reporter_label: str) -> None:
    for s in segments:
        sid = s.get("speaker_id")
        s["speaker"] = speakers[sid]["label"] if sid in speakers else (reporter_label if sid else None)
