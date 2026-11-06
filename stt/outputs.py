"""Çıktılar: .txt, .srt, .json, ozet.md"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .util import atomic_write_text, fmt_clock, fmt_ts

MAX_LINE = 42
MAX_CUE_CHARS = 84
MAX_CUE_S = 6.0


# ------------------------------------------------------------------ TXT

def turns_from_segments(segments: list[dict]) -> list[dict]:
    """Ardışık aynı konuşmacılı segmentleri konuşma turlarına birleştir."""
    turns: list[dict] = []
    for s in segments:
        spk = s.get("speaker")
        gap = s["start"] - turns[-1]["end"] if turns else 0
        if turns and turns[-1]["speaker"] == spk and (spk is not None or gap < 2.0):
            turns[-1]["text"] += " " + s["text"]
            turns[-1]["end"] = s["end"]
        else:
            turns.append({"speaker": spk, "start": s["start"], "end": s["end"], "text": s["text"]})
    for t in turns:
        t["text"] = re.sub(r"\s+", " ", t["text"]).strip()
    return turns


def _paragraphs(segments: list[dict]) -> list[str]:
    paras, cur, last_end = [], [], None
    for s in segments:
        if cur and (s["start"] - (last_end or 0) > 2.0 or sum(len(x) for x in cur) > 600) and cur[-1][-1:] in ".?!…":
            paras.append(" ".join(cur))
            cur = []
        cur.append(s["text"])
        last_end = s["end"]
    if cur:
        paras.append(" ".join(cur))
    return [re.sub(r"\s+", " ", p).strip() for p in paras]


def qa_blocks(segments: list[dict], td_label: str | None) -> list[tuple[str, str, dict]]:
    """Konuşma turlarını soru-cevap bloklarına çevir: [(başlık, metin, tur)]."""
    out, q = [], 0
    for t in turns_from_segments(segments):
        spk = t["speaker"] or "?"
        if td_label and spk == td_label:
            head = f"Cevap - {spk}" if q else f"Açılış - {spk}"
        else:
            q += 1
            head = f"Soru {q} - {spk}"
        out.append((head, t["text"], t))
    return out


def dialog_blocks(segments: list[dict]) -> list[tuple[str, str, dict]]:
    """Podcast / film: her tur '[dakika] Konuşmacı N' başlığıyla."""
    return [(f"[{fmt_clock(t['start'])}] {t['speaker'] or '?'}", t["text"], t) for t in turns_from_segments(segments)]


def is_general(doc: dict) -> bool:
    return doc.get("speaker_mode") == "herkes" or bool((doc.get("dataset") or {}).get("general"))


def header_lines(doc: dict) -> list[str]:
    m = doc.get("match") or {}
    ds = doc.get("dataset") or {}
    if is_general(doc):
        lines = [doc.get("title") or "Kayıt"]
        if doc.get("speakers"):
            lines.append(f"{len(doc['speakers'])} konuşmacı: "
                         + ", ".join(f"{v['label']} ({fmt_clock(v['seconds'])})" for v in doc["speakers"].values()))
        if m.get("date"):
            lines.append(f"Yayın tarihi: {m['date']}")
        src = doc.get("webpage_url") or doc.get("source")
        if src:
            lines.append(f"Kaynak: {src}")
        if not doc.get("speakers") and doc.get("diarization_error"):
            lines.append(f"(Konuşmacı ayrımı yapılamadı: {doc['diarization_error']})")
        return lines
    home, away = ds.get("home") or m.get("home"), ds.get("away") or m.get("away")
    teams = " vs ".join(x for x in (home, away) if x)
    title = " - ".join(x for x in (teams, m.get("manager")) if x)
    lines = [f"{title} | basın toplantısı" if title else (doc.get("title") or "Kayıt")]
    meta = []
    tags = [x for x in (ds.get("competition"), f"GW{int(ds['gw']):02d}" if ds.get("gw") else None,
                        {"mac-oncesi": "maç öncesi", "mac-sonrasi": "maç sonrası"}.get(ds.get("phase") or "")) if x]
    if tags:
        meta.append(" | ".join(tags))
    if ds.get("match_date") and ds.get("match_date") != ds.get("presser_date"):
        meta.append(f"Maç tarihi: {ds['match_date']}")
    if ds.get("presser_date") or m.get("date"):
        meta.append(f"Basın toplantısı: {ds.get('presser_date') or m.get('date')}")
    if doc.get("title") and title:
        meta.append(f"Video: {doc['title']}")
    src = doc.get("webpage_url") or doc.get("source")
    if src:
        meta.append(f"Kaynak: {src}")
    lines += meta
    if not doc.get("speakers") and doc.get("diarization_error"):
        lines.append(f"(Konuşmacı ayrımı yapılamadı: {doc['diarization_error']})")
    return lines


def to_txt(doc: dict) -> str:
    segs = doc["segments"]
    body: list[str]
    if doc.get("speakers") and is_general(doc):
        body = [f"{head}:\n{text}" for head, text, _ in dialog_blocks(segs)]
    elif doc.get("speakers"):
        body = [f"{head}:\n{text}" for head, text, _ in qa_blocks(segs, doc.get("td_label"))]
    else:
        body = _paragraphs(segs)
    return "\n".join(header_lines(doc)) + "\n\n" + "\n\n".join(body) + "\n"


# ------------------------------------------------------------------ SRT

def _wrap(text: str) -> str:
    if len(text) <= MAX_LINE:
        return text
    words = text.split()
    best, best_diff = None, 10**9
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        diff = abs(len(a) - len(b))
        if diff < best_diff:
            best, best_diff = (a, b), diff
    return "\n".join(best) if best else text


def _cues_from_words(seg: dict) -> list[tuple[float, float, str]]:
    cues, cur = [], []
    for w in seg["words"]:
        cand = "".join(x["word"] for x in cur + [w]).strip()
        too_long = cur and (len(cand) > MAX_CUE_CHARS or w["end"] - cur[0]["start"] > MAX_CUE_S)
        gap = cur and w["start"] - cur[-1]["end"] > 0.8
        if too_long or gap:
            # mümkünse son virgülden böl (yarım cümle artığı kalmasın)
            cut = next((i for i in range(len(cur) - 1, len(cur) // 3, -1) if cur[i]["word"].strip()[-1:] in ",;:"),
                       None) if too_long else None
            if cut is not None:
                cues.append(cur[:cut + 1])
                cur = cur[cut + 1:]
            else:
                cues.append(cur)
                cur = []
        cur.append(w)
        if w["word"].strip()[-1:] in ".?!" and len("".join(x["word"] for x in cur)) > 20:
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    return [(c[0]["start"], c[-1]["end"], "".join(x["word"] for x in c).strip()) for c in cues if c]


def _cues_from_text(seg: dict) -> list[tuple[float, float, str]]:
    text, dur = seg["text"], max(0.01, seg["end"] - seg["start"])
    if len(text) <= MAX_CUE_CHARS:
        return [(seg["start"], seg["end"], text)]
    pieces, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > MAX_CUE_CHARS:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        pieces.append(cur)
    total = sum(len(p) for p in pieces)
    out, t = [], seg["start"]
    for p in pieces:
        d = dur * len(p) / total
        out.append((t, t + d, p))
        t += d
    return out


def to_srt(segments: list[dict], diarized: bool) -> str:
    cues, prev_spk = [], object()
    for seg in segments:
        parts = _cues_from_words(seg) if seg.get("words") else _cues_from_text(seg)
        first = True
        for a, b, text in parts:
            if not text:
                continue
            spk = seg.get("speaker")
            if diarized and spk and spk != prev_spk and first:
                text = f"[{spk}] {text}"
                prev_spk = spk
            first = False
            cues.append([a, max(b, a + 0.3), text])
    # altyazılar üst üste binmesin (oynatıcılar iki satırı birden gösterir): bir sonrakinin başında bitir
    for cur, nxt in zip(cues, cues[1:]):
        if cur[1] > nxt[0] > cur[0]:
            cur[1] = nxt[0]
    return "\n".join(f"{n}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{_wrap(text)}\n" for n, (a, b, text) in enumerate(cues, 1))


# ------------------------------------------------------------------ özet

_STOP = set("""
a an and are as at be been but by can do does for from had has have he her his how i if in into is it its
just me my no not of on or our so that the their them then there they this to too up us very was we were
what when which who will with would you your yeah yes well think know going get got really one also because
about all more out some like don't it's i'm we're they're that's there's
ve bir bu da de ile için çok ama ben biz siz onlar o şu ne mi mı mu mü gibi daha en kadar olarak olan
var yok şey değil ki her çünkü sonra önce evet hayır yani şimdi bunu buna bunun benim bizim tabii
və bu bir da də ilə üçün çox amma mən biz siz onlar o nə ki hər sonra əvvəl bəli xeyr indi
""".split())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if len(s.strip().split()) >= 6]


def _keywords(text: str, n: int = 12) -> list[str]:
    words = re.findall(r"[^\W\d_][\w'’-]{2,}", text.lower())
    c = Counter(w for w in words if w not in _STOP and len(w) > 3)
    return [w for w, _ in c.most_common(n)]


def _top_sentences(text: str, k: int = 6) -> list[str]:
    sents = _sentences(text)
    if len(sents) <= k:
        return sents
    freq = Counter(w for w in re.findall(r"[^\W\d_]{3,}", text.lower()) if w not in _STOP)
    scored = []
    for i, s in enumerate(sents):
        toks = [w for w in re.findall(r"[^\W\d_]{3,}", s.lower()) if w not in _STOP]
        if not toks:
            continue
        scored.append((sum(freq[t] for t in toks) / (len(toks) ** 0.7), i, s))
    best = sorted(sorted(scored, reverse=True)[:k], key=lambda x: x[1])
    return [s for _, _, s in best]


def to_summary(doc: dict, txt_name: str) -> str:
    segs = doc["segments"]
    dur = doc.get("duration_s") or (segs[-1]["end"] if segs else 0)
    words = sum(len(s["text"].split()) for s in segs)
    src = doc.get("webpage_url") or doc.get("source")
    L = [f"# {doc.get('title') or 'Kayıt'}", "",
         "| | |", "|---|---|",
         f"| Kaynak | {src} |",
         f"| Süre | {fmt_clock(dur)} |",
         f"| Dil | {doc.get('language') or '?'} |",
         f"| Motor | {doc.get('engine')} / {doc.get('model')} |",
         f"| Kelime | {words} |",
         f"| Oluşturma | {doc.get('created')} |"]
    general = is_general(doc)
    m = {} if general else (doc.get("match") or {})
    if m.get("home") or m.get("away"):
        L.append(f"| Maç | {' vs '.join(x for x in (m.get('home'), m.get('away')) if x)} |")
    if m.get("manager"):
        L.append(f"| Teknik direktör | {m['manager']} |")
    ds = {} if general else (doc.get("dataset") or {})
    if ds.get("gw"):
        L.append(f"| GW | {ds['gw']} |")
    if ds.get("competition"):
        L.append(f"| Turnuva | {ds['competition']} |")
    if ds.get("match_date"):
        L.append(f"| Maç tarihi | {ds['match_date']} |")
    if ds.get("phase"):
        L.append(f"| Zaman | {ds['phase']} |")
    if m.get("date"):
        L.append(f"| Basın toplantısı | {m['date']} |")
    if doc.get("fpl"):
        L.append(f"| FPL | GW{doc['fpl']['gw']:02d} {doc['fpl']['team']} |")
    if doc.get("diarization_error") and not doc.get("speakers"):
        L.append(f"| Konuşmacı ayrımı | YAPILAMADI: {doc['diarization_error']} |")
    L.append("")
    td_text = ""
    if doc.get("speakers"):
        L += ["## Konuşmacılar", "", "| Konuşmacı | Etiket | Süre | Pay |", "|---|---|---|---|"]
        tot = sum(v["seconds"] for v in doc["speakers"].values()) or 1
        for sid, v in doc["speakers"].items():
            L.append(f"| {sid} | {v['label']} | {fmt_clock(v['seconds'])} | %{100 * v['seconds'] / tot:.0f} |")
        if general:
            L.append("")
    if doc.get("speakers") and not general:
        td_label = doc.get("td_label") or next((v["label"] for v in doc["speakers"].values()), None)
        blocks = qa_blocks(segs, td_label)
        qs = [t for h, _, t in blocks if h.startswith("Soru")]
        td_text = " ".join(t["text"] for h, _, t in blocks if not h.startswith("Soru"))
        L += ["", f"Soru sayısı (muhabir turu): **{len(qs)}**", ""]
        if qs:
            L += ["## Sorular", ""]
            for t in qs[:40]:
                q = t["text"] if len(t["text"]) < 220 else t["text"][:217] + "…"
                L.append(f"- `{fmt_clock(t['start'])}` {q}")
            L.append("")
    text = td_text or " ".join(s["text"] for s in segs)
    kw = _keywords(text)
    if kw:
        L += ["## Anahtar kelimeler", "", ", ".join(kw), ""]
    top = _top_sentences(text)
    if top:
        L += ["## Öne çıkan cümleler" + (" (teknik direktör)" if td_text else ""), ""]
        L += [f"> {s}" for s in top]
        L.append("")
    L += ["## Dosyalar", "", f"- Metin: `{txt_name}`", f"- Altyazı: `{Path(txt_name).with_suffix('.srt')}`",
          f"- Zaman + konuşmacı: `{Path(txt_name).with_suffix('.json')}`", ""]
    return "\n".join(L)


def write_all(out_dir: Path, base: str, doc: dict, summary_name: str = "ozet.md") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    diarized = bool(doc.get("speakers"))
    doc.setdefault("created", datetime.now().strftime("%Y-%m-%d %H:%M"))
    paths = {"txt": out_dir / f"{base}.txt", "srt": out_dir / f"{base}.srt",
             "json": out_dir / f"{base}.json", "md": out_dir / summary_name}
    atomic_write_text(paths["txt"], to_txt(doc))
    atomic_write_text(paths["srt"], to_srt(doc["segments"], diarized))
    atomic_write_text(paths["json"], json.dumps(doc, ensure_ascii=False, indent=1))
    atomic_write_text(paths["md"], to_summary(doc, paths["txt"].name))
    return paths
