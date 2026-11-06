"""Parça sonuçlarını birleştirme ve konuşmacı etiketleme."""
from __future__ import annotations

from collections import Counter, defaultdict


# Whisper'ın sessizlik/müzikte uydurduğu, eğitim verisindeki altyazı imzaları
_ALWAYS_JUNK = ("amara.org", "altyazı m.k", "altyazi m.k", "subtitles by", "transcribed by", "transcript by",
                "sottotitoli", "untertitel im auftrag", "untertitelung", "www.zeoranger")
_LOWCONF_JUNK = ("thanks for watching", "thank you for watching", "izlediğiniz için teşekkür", "abone ol",
                 "subscribe", "altyazı", "subtitle", "♪", "[müzik]", "[music]", "(music)")


def is_hallucination(seg: dict) -> bool:
    """Bilinen uydurma satır mı? İmza satırları her zaman; 'izlediğiniz için teşekkürler' gibi gerçek de olabilen
    satırlar yalnızca kısa ve modelin emin olmadığı (sessizlik olasılığı yüksek / güven düşük) durumda atılır."""
    t = (seg.get("text") or "").strip().lower()
    if any(j in t for j in _ALWAYS_JUNK):
        return True
    if len(t) > 60 or not any(j in t for j in _LOWCONF_JUNK):
        return False
    return (seg.get("no_speech_prob") or 0) > 0.5 or (seg.get("avg_logprob") or 0) < -0.8


REPEAT_GAP = 2.0  # sn: bundan uzak aynı cümle tekrar döngüsü sayılmaz


def merge_chunks(parts: list[dict]) -> tuple[list[dict], str | None]:
    segs: list[dict] = []
    lang_dur: Counter = Counter()
    for p in parts:
        for s in p.get("segments", []):
            if not s.get("text") or is_hallucination(s):
                continue
            text = collapse_repeats(s["text"])
            if text != s["text"] and s.get("words"):
                s = {**s, "words": collapse_word_repeats(s["words"])}
            s = {**s, "text": text}
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
        # (yalnızca bitişikse: dakikalar sonra aynı cümle gerçekten yeniden söylenmiş olabilir)
        if (out and len(s["text"]) > 15 and _norm(s["text"]) == _norm(out[-1]["text"])
                and s["start"] - out[-1]["end"] < REPEAT_GAP):
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


def _join(words: list[dict]) -> str:
    return "".join(w["word"] for w in words).strip()


def collapse_word_repeats(words: list[dict]) -> list[dict]:
    """collapse_repeats'in kelime karşılığı: tekrarlanan cümlenin kelimeleri de atılsın
    (yoksa konuşmacıya göre bölününce tekrar metne geri geliyordu)."""
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if w["word"].strip()[-1:] in ".!?…":
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    out: list[list[dict]] = []
    for snt in sents:
        if out and len(_join(snt)) > 15 and _norm(_join(snt)) == _norm(_join(out[-1])):
            continue
        out.append(snt)
    return [w for snt in out for w in snt]


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


def _dur(w: dict) -> float:
    return max(0.02, w["end"] - w["start"])


def _runs(labels: list) -> list[list[int]]:
    """Ardışık aynı etiketli bölgeler: [başlangıç, bitiş) listesi."""
    runs: list[list[int]] = []
    for i, lab in enumerate(labels):
        if runs and labels[runs[-1][0]] == lab:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return runs


def _smooth(labels: list[str | None], min_run: int = 2, durs: list[float] | None = None,
            min_sec: float = 1.0) -> list[str | None]:
    """Kısa konuşmacı sıçramalarını komşu konuşmacıya kat.

    `durs` verilirse kısa = toplam süresi `min_sec`'ten az; verilmezse kelime sayısı `min_run`'dan az."""
    out = list(labels)
    runs = _runs(out)
    if len(runs) < 2:
        return out
    for k, (a, b) in enumerate(runs):
        short = (sum(durs[a:b]) < min_sec) if durs is not None else (b - a < min_run)
        if not short:
            continue
        left = out[runs[k - 1][0]] if k > 0 else None
        right = out[runs[k + 1][0]] if k + 1 < len(runs) else None
        repl = left if left is not None else right
        if repl is not None:
            for i in range(a, b):
                out[i] = repl
    return out


SENT_END = ".?!…"
SENT_GAP = 0.6       # bu kadar sessizlik de cümle sonu sayılır (sn)
SPLIT_MIN_SEC = 1.5  # cümle ortasında konuşmacı değişimi için öbür konuşmacının en az süresi
SPLIT_MIN_WORDS = 4  # ... ve en az kelime sayısı
EARLY_GAP = 1.5      # segment başındaki kelimeden sonra bu kadar boşluk: kelime zamanı kaymış olabilir


def _sentences(words: list[dict]) -> list[tuple[int, int]]:
    """Kelimeleri cümlelere böl: noktalama (. ? ! …) ya da uzun sessizlik."""
    out, a = [], 0
    for i, w in enumerate(words):
        end = w["word"].strip().rstrip("\"'”’»)]").endswith(tuple(SENT_END))
        gap = i + 1 < len(words) and words[i + 1]["start"] - w["end"] > SENT_GAP
        if end or gap or i + 1 == len(words):
            out.append((a, i + 1))
            a = i + 1
    return out


def _sentence_labels(labels: list, durs: list[float]) -> list:
    """Cümleye süre ağırlıklı çoğunluk oyu; yalnızca iki konuşmacı da uzun (≥1.5 sn, ≥4 kelime) konuşmuşsa böl."""
    vote: dict = defaultdict(float)
    for lab, d in zip(labels, durs):
        if lab is not None:
            vote[lab] += d
    if not vote:
        return list(labels)
    major = max(vote, key=vote.get)
    runs = _runs(labels)
    strong = [labels[a] is not None and b - a >= SPLIT_MIN_WORDS and sum(durs[a:b]) >= SPLIT_MIN_SEC
              for a, b in runs]
    if len({labels[a] for (a, _), st in zip(runs, strong) if st}) < 2:
        return [major] * len(labels)
    out = list(labels)
    for k, (a, b) in enumerate(runs):
        if strong[k]:
            continue
        left = next((labels[runs[j][0]] for j in range(k - 1, -1, -1) if strong[j]), None)
        right = next((labels[runs[j][0]] for j in range(k + 1, len(runs)) if strong[j]), None)
        repl = right if (labels[a] == right or left is None) else left
        for i in range(a, b):
            out[i] = repl
    return out


def exclusive_turns(turns: list[list]) -> list[list]:
    """Örtüşen konuşmayı tek konuşmacıya indir (metinle hizalama için): örtüşen yerde uzun tur kazanır.

    pyannote 4.x'in `exclusive_speaker_diarization` çıktısının 3.x için yaklaşığı."""
    turns = [t for t in turns if t[1] > t[0]]
    if len(turns) < 2:
        return [list(t) for t in turns]
    turns = sorted(turns, key=lambda t: t[0])
    pts = sorted({p for t in turns for p in (t[0], t[1])})
    out: list[list] = []
    act: list[list] = []
    i = 0
    for t0, t1 in zip(pts, pts[1:]):
        while i < len(turns) and turns[i][0] <= t0:
            act.append(turns[i])
            i += 1
        act = [t for t in act if t[1] >= t1]
        if not act:
            continue
        spk = max(act, key=lambda t: t[1] - t[0])[2]
        if out and out[-1][2] == spk and abs(out[-1][1] - t0) < 1e-9:
            out[-1][1] = t1
        else:
            out.append([t0, t1, spk])
    return out


def assign_speakers(segments: list[dict], turns: list[list]) -> list[dict]:
    """Konuşmacı ata: kelime → tur, sonra cümle düzeyinde çoğunluk oyu; konuşmacı değişirse segmenti böl."""
    turns = exclusive_turns(sorted(turns, key=lambda t: t[0]))
    out: list[dict] = []
    hint = 0
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            spk, hint = _speaker_at(seg["start"], seg["end"], turns, hint)
            out.append({**seg, "speaker_id": spk})
            continue
        raw = []
        for w in words:
            spk, hint = _speaker_at(w["start"], max(w["end"], w["start"] + 0.02), turns, hint)
            raw.append(spk)
        durs = [_dur(w) for w in words]
        labels: list = []
        sents = _sentences(words)
        for a, b in sents:
            labels += _sentence_labels(_smooth(raw[a:b], durs=durs[a:b]), durs[a:b])
        # sessizlikten sonraki ilk kelimenin zamanı Whisper'da birkaç sn erken çıkabiliyor (faster-whisper #125):
        # segment başındaki 1-2 kelimelik, noktalamasız ve uzun boşlukla ayrılmış parça sonraki cümleye aittir
        if len(sents) > 1:
            a, b = sents[0]
            last = words[b - 1]["word"].strip()
            gap = words[b]["start"] - words[b - 1]["end"]
            if b - a <= 2 and gap >= EARLY_GAP and last[-1:] not in SENT_END and labels[b] is not None:
                labels[a:b] = [labels[b]] * (b - a)
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


def label_speakers_all(turns: list[list], prefix: str = "Konuşmacı") -> dict[str, dict]:
    """Podcast / film: her konuşmacı ayrı etiket alır (ilk konuşan = Konuşmacı 1)."""
    tot: dict[str, float] = defaultdict(float)
    first: dict[str, float] = {}
    for s, e, spk in turns:
        tot[spk] += max(0.0, e - s)
        first[spk] = min(first.get(spk, s), s)
    order = sorted(tot, key=lambda k: first[k])
    return {spk: {"label": f"{prefix} {i}", "seconds": round(tot[spk], 1)} for i, spk in enumerate(order, 1)}


def apply_labels(segments: list[dict], speakers: dict[str, dict], reporter_label: str) -> None:
    for s in segments:
        sid = s.get("speaker_id")
        s["speaker"] = speakers[sid]["label"] if sid in speakers else (reporter_label if sid else None)
