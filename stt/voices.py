"""Ses kütüphanesi: bilinen kişilerin ses imzaları (pyannote gömme vektörleri), elle düzeltme.

- Her konuşmacı ayırmada pyannote konuşmacı başına ortalama bir ses imzası verir; çıktıyla birlikte
  ``<out>/.stt/ses/<key>.json`` olarak saklanır.
- Kullanıcı "Metni göster > Konuşmacılar"da bir konuşmacıya ad verip kaydedince o imza
  ``<out>/.stt/sesler.json`` kütüphanesine eklenir (eğitim gerekmez).
- Yeni videoda her konuşmacının imzası kütüphanedekilerle kosinüs benzerliğine göre karşılaştırılır; eşiği
  geçen konuşmacı o adla yazılır.
- Düzeltilmiş konuşmacı turları ``<out>/.stt/duzeltme/<key>.json``'a yazılır (ölçüm / ileride ince ayar verisi).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

GENERIC = re.compile(r"^\s*(konuşmacı|konusmaci|speaker|kişi|kisi)\s*(\d+|\?)?\s*$", re.I)
MAX_PER_NAME = 30


MARGIN = 0.05  # en iyi iki aday arasındaki en az benzerlik farkı
SURE = 0.15    # eşiğin bu kadar üstündeyse fark aranmaz


def is_generic(label: str | None, extra=()) -> bool:
    """'Konuşmacı 2', 'Muhabir', 'Teknik direktör' gibi kişi adı olmayan etiketler kütüphaneye girmez."""
    lab = (label or "").strip()
    return not lab or bool(GENERIC.match(lab)) or lab.casefold() in {x.casefold() for x in extra if x}


def _safe(key: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(key))[:80] or "x"


def _unit(v):
    import numpy as np
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


# ------------------------------------------------------------------ videonun konuşmacı imzaları
def save_doc_voices(state_dir: Path, key: str, model: str | None, emb: dict | None) -> None:
    if emb:
        _write(Path(state_dir) / "ses" / f"{_safe(key)}.json", {"model": model, "emb": emb})


def load_doc_voices(state_dir: Path, key: str) -> dict:
    return _read(Path(state_dir) / "ses" / f"{_safe(key)}.json", {}) or {}


# ------------------------------------------------------------------ kütüphane
class VoiceLibrary:
    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "sesler.json"

    def load(self) -> dict:
        d = _read(self.path, {})
        return d if isinstance(d, dict) and isinstance(d.get("voices"), dict) else {"voices": {}}

    def save(self, d: dict) -> None:
        _write(self.path, d)

    def names(self) -> list[dict]:
        v = self.load()["voices"]
        return [{"name": n, "n": len(e.get("embs", [])), "td": bool(e.get("td"))}
                for n, e in sorted(v.items(), key=lambda kv: kv[0].casefold())]

    def td_names(self) -> set[str]:
        return {n for n, e in self.load()["voices"].items() if e.get("td")}

    def add(self, name: str, emb, model: str | None, key: str, spk: str, td: bool = False) -> bool:
        """Bir videodaki bir konuşmacının imzasını ``name`` altına ekle. Aynı video+konuşmacı önceden başka
        ada eklenmişse oradan alınır (yeniden adlandırma çift kayıt yapmasın)."""
        if _unit(emb) is None or not name.strip():
            return False
        name = name.strip()
        d = self.load()
        voices = d["voices"]
        for n in list(voices):
            voices[n]["embs"] = [x for x in voices[n].get("embs", []) if not (x.get("key") == key and x.get("spk") == spk)]
            if not voices[n]["embs"]:
                del voices[n]
        e = voices.setdefault(name, {"embs": []})
        e["td"] = bool(e.get("td")) or bool(td)
        e["embs"].append({"model": model, "emb": [round(float(x), 5) for x in emb], "key": key, "spk": spk,
                          "t": time.strftime("%Y-%m-%d")})
        same = [x for x in e["embs"] if x.get("model") == model]
        if len(same) > MAX_PER_NAME:  # en eskisi çıkar
            drop = same[0]
            e["embs"] = [x for x in e["embs"] if x is not drop]
        self.save(d)
        return True

    def remove(self, name: str) -> bool:
        d = self.load()
        if name in d["voices"]:
            del d["voices"][name]
            self.save(d)
            return True
        return False

    def centroids(self, model: str | None) -> dict:
        """Ad -> birim ortalama imza (yalnızca aynı modelden alınmış imzalar; modeller arası karşılaştırılmaz)."""
        import numpy as np
        out = {}
        for n, e in self.load()["voices"].items():
            vs = [u for x in e.get("embs", []) if x.get("model") == model for u in [_unit(x.get("emb"))] if u is not None]
            if vs:
                c = _unit(np.mean(vs, axis=0))
                if c is not None:
                    out[n] = c
        return out

    def identify(self, emb: dict | None, model: str | None, threshold: float = 0.6) -> dict:
        """{SPEAKER_xx: (ad, benzerlik)}; her ad en çok bir konuşmacıya, en benzerden başlayarak verilir."""
        if not emb:
            return {}
        cents = self.centroids(model)
        if not cents:
            return {}
        pairs = []
        for spk, v in emb.items():
            u = _unit(v)
            if u is None or len(u) != len(next(iter(cents.values()))):
                continue
            for n, c in cents.items():
                pairs.append((float(u @ c), spk, n))
        pairs.sort(reverse=True)
        # belirsiz eşleşme: iki kişiye neredeyse eşit benziyorsa (fark < MARGIN) ve eşiğin çok üstünde değilse ad
        # verme; yanlış kişinin adıyla yazmaktansa "Konuşmacı N" kalsın
        sims: dict = {}
        for sim, spk, n in pairs:
            sims.setdefault(spk, []).append(sim)
        got, used = {}, set()
        for sim, spk, n in pairs:
            if sim < threshold:
                break
            if spk in got or n in used:
                continue
            second = sims[spk][1] if len(sims[spk]) > 1 else -1.0
            if sim - second < MARGIN and sim < threshold + SURE:
                continue
            got[spk] = (n, sim)
            used.add(n)
        return got


def apply_voice_names(speakers: dict, names: dict, mode: str, td_label: str, reporter_label: str,
                      td_names: set | None = None) -> str:
    """Tanınan konuşmacılara adlarını ver. Basın toplantısında kütüphanede teknik direktör olarak işaretli
    (ya da adı maçtaki teknik direktörle aynı) bir ses bulunursa o konuşmacı TD olur. Döner: TD etiketi."""
    if not names:
        return td_label
    td_names = td_names or set()
    if mode == "basin":
        td_spk = next((k for k, v in speakers.items() if v.get("label") == td_label), None)
        hits = sorted(((s, k) for k, (n, s) in names.items() if k in speakers and (n in td_names or n == td_label)),
                      reverse=True)
        if hits:
            new_td = hits[0][1]
            if td_spk and td_spk != new_td:
                speakers[td_spk]["label"] = reporter_label
            td_spk, td_label = new_td, names[new_td][0]
        elif td_spk in names:
            td_label = names[td_spk][0]
        for k, (n, s) in names.items():
            if k in speakers:
                speakers[k]["voice"] = round(s, 3)
                if k != td_spk:
                    speakers[k]["label"] = n
        if td_spk:
            speakers[td_spk]["label"] = td_label
        return td_label
    for k, (n, s) in names.items():
        if k in speakers:
            speakers[k]["label"] = n
            speakers[k]["voice"] = round(s, 3)
    return td_label


# ------------------------------------------------------------------ elle düzeltme
def _sid_key(x) -> str:
    return str(x)


def apply_corrections(doc: dict, labels: dict | None = None, moves: dict | None = None,
                      td: str | None = None) -> dict:
    """Okuyucudaki düzeltmeleri çıktıya uygula.

    labels: {konuşmacı_id: yeni ad}; moves: {segment_id: konuşmacı_id} (yeni id'ler de olabilir);
    td: basın toplantısında teknik direktör olan konuşmacı_id. Süreler yeniden hesaplanır."""
    labels = {str(k): str(v).strip() for k, v in (labels or {}).items() if str(v).strip()}
    moves = {str(k): str(v) for k, v in (moves or {}).items() if v}
    segs = doc.get("segments") or []
    spk = {str(k): dict(v) for k, v in (doc.get("speakers") or {}).items()}
    old_td = doc.get("td_label")
    td_sid = td or next((k for k, v in spk.items() if old_td and v.get("label") == old_td), None)
    for s in segs:
        if _sid_key(s.get("id")) in moves:
            s["speaker_id"] = moves[_sid_key(s.get("id"))]
    n_new = 0
    for s in segs:
        sid = s.get("speaker_id")
        if sid and sid not in spk:
            n_new += 1
            spk[sid] = {"label": labels.get(sid) or f"Konuşmacı {len(spk) + 1}"}
    for sid, lab in labels.items():
        if sid in spk:
            spk[sid]["label"] = lab
            spk[sid].pop("voice", None)
    secs: dict[str, float] = {}
    for s in segs:
        sid = s.get("speaker_id")
        if sid:
            secs[sid] = secs.get(sid, 0.0) + max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
    spk = {k: {**v, "seconds": round(secs[k], 1)} for k, v in spk.items() if k in secs}
    for s in segs:
        sid = s.get("speaker_id")
        if sid in spk:
            s["speaker"] = spk[sid]["label"]
    doc["speakers"] = dict(sorted(spk.items(), key=lambda kv: -kv[1]["seconds"]))
    if doc.get("speaker_mode") == "basin" and td_sid in spk:
        doc["td_label"] = spk[td_sid]["label"]
    doc["corrected"] = time.strftime("%Y-%m-%d %H:%M")
    return doc


def reference_turns(doc: dict, merge_gap: float = 1.0) -> list[list]:
    """Düzeltilmiş çıktıdan [başla, bitir, etiket] turları. Art arda aynı konuşmacının satırları arada ``merge_gap``
    sn'den az boşluk varsa birleşir; ``merge_gap=0``: hiç birleştirilmez (satırlar olduğu gibi, aradaki sessizlik
    konuşma sayılmaz)."""
    out: list[list] = []
    for s in doc.get("segments") or []:
        lab = s.get("speaker")
        if not lab:
            continue
        a, b = round(float(s["start"]), 3), round(float(s["end"]), 3)
        if merge_gap > 0 and out and out[-1][2] == lab and a - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b, lab])
    return out


def save_reference(state_dir: Path, doc: dict, hyp: list | None = None) -> Path:
    """Doğru cevabı yaz. ``hyp``: programın düzeltilmeden önceki çıktısı (ölçüm için; ilk düzeltmede saklanır,
    sonraki düzeltmeler üzerine yazmaz)."""
    p = Path(state_dir) / "duzeltme" / f"{_safe(doc.get('key') or doc.get('title') or 'x')}.json"
    old = _read(p, {})
    data = {"key": doc.get("key"), "title": doc.get("title"), "source": doc.get("source"),
            "duration_s": doc.get("duration_s"), "speaker_mode": doc.get("speaker_mode"),
            "td_label": doc.get("td_label"),
            "turns": reference_turns(doc),
            # eğitim (stt egit) için birleştirilmemiş satırlar: birleşik turlarda <1 sn sessizlik "konuşma" sayılıp
            # pyannote'un orada sessizlik demesi hata çıkıyor, ayar (min_duration_off) yanlı seçiliyordu
            "segs": reference_turns(doc, merge_gap=0)}
    if old.get("hyp") or hyp:
        data["hyp"] = old.get("hyp") or hyp
        # programın çıktısı hangi ayarla üretildi (Kontrol'de ayara göre ayrı doğruluk: A/B ölçümü)
        setup = old.get("diar") if old.get("hyp") else _diar_setup(doc)
        if setup:
            data["diar"] = setup
    _write(p, data)
    return p


def _diar_setup(doc: dict) -> dict:
    st = doc.get("settings") or {}
    out = {}
    if st.get("diar_model"):
        out["model"] = st["diar_model"]
    if "diar_agc" in st:
        out["agc"] = bool(st["diar_agc"])
    return out


def learn_from_doc(state_dir: Path, doc: dict, ignore=()) -> list[str]:
    """Kaydedilen (düzeltilmiş) çıktıdaki adı konmuş konuşmacıların imzalarını kütüphaneye ekle."""
    side = load_doc_voices(state_dir, doc.get("key") or "")
    emb, model = side.get("emb") or {}, side.get("model")
    if not emb:
        return []
    lib = VoiceLibrary(state_dir)
    td = doc.get("td_label") if doc.get("speaker_mode") == "basin" else None
    learned = []
    for sid, v in (doc.get("speakers") or {}).items():
        lab = v.get("label")
        if sid in emb and not is_generic(lab, ignore):
            if lib.add(lab, emb[sid], model, doc.get("key") or "", sid, td=bool(td and lab == td)):
                learned.append(lab)
    return learned
