"""Ses kütüphanesi: bilinen kişilerin ses imzaları (pyannote gömme vektörleri), elle düzeltme.

- Her konuşmacı ayırmada pyannote konuşmacı başına ortalama bir ses imzası verir; çıktıyla birlikte
  ``<out>/.stt/ses/<key>.json`` olarak saklanır.
- Kullanıcı "Metni göster > Konuşmacılar"da bir konuşmacıya ad verip kaydedince o imza
  ``<out>/.stt/sesler.json`` kütüphanesine eklenir (eğitim gerekmez).
- Yeni videoda her konuşmacının imzası kütüphanedekilerle kosinüs benzerliğine göre karşılaştırılır; eşiği
  geçen konuşmacı o adla yazılır.
- Düzeltilmiş konuşmacı turları ``<out>/.stt/duzeltme/<key>.json``'a yazılır (ölçüm / ileride ince ayar verisi).
- 1.9.5: "Kaydet"te bir videonun kütüphanedeki kayıtları baştan kurulur. Konuşmasının %10'undan fazlası başka
  kişiye taşınmış (karışık) kümenin imzası öğrenilmez (``PURE_MIN``). "Kaydet ve eğit"te (``stt egit``, ekran kartında) imzalar
  düzeltilmiş satırların sesinden yeniden çıkarılır (``src="duzeltme"``); elle eklenen kişiler de böylece öğrenilir.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

GENERIC = re.compile(r"^\s*(konuşmacı|konusmaci|speaker|kişi|kisi)\s*(\d+|\?)?\s*$", re.I)
MAX_PER_NAME = 30


MARGIN = 0.05  # en iyi iki aday arasındaki en az benzerlik farkı
SURE = 0.15    # eşiğin bu kadar üstündeyse fark aranmaz
PURE_MIN = 0.9  # küme imzası, kümenin konuşmasının en az bu kadarı düzeltmeden sonra da aynı kişideyse öğrenilir


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
    from .util import replace_retry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")  # iki süreç aynı geçici dosyaya yazmasın
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    replace_retry(tmp, path)  # Windows'ta dosya bir anlığına kilitliyse tekrar dener


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


class _FileLock:
    """Kütüphaneye iki süreç yazabilir (arayüz "Kaydet" + eğitim süreci): oku-değiştir-yaz arasında kilit dosyası.
    Çöken süreçten kalan kilit ``stale`` sn sonra kaldırılır; ``timeout``da kilitsiz devam edilir (yazma yine atomik)."""

    def __init__(self, path: Path, timeout: float = 10.0, stale: float = 60.0):
        self.path = Path(str(path) + ".lock")
        self.timeout, self.stale, self.held = timeout, stale, False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                self.held = True
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() - t0 > self.timeout:
                    return self
                time.sleep(0.05)
            except OSError:  # salt okunur klasör vb.: kilitsiz devam
                return self

    def __exit__(self, *exc):
        if self.held:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self.held = False
        return False


# ------------------------------------------------------------------ videonun konuşmacı imzaları
def save_doc_voices(state_dir: Path, key: str, model: str | None, emb: dict | None,
                    turns: list | None = None) -> None:
    """Videonun konuşmacı imzaları. ``turns``: imzaların hesaplandığı ilk ayırma turları [başla, bitir, sid] —
    düzeltmeden sonra kümenin ne kadarının başka kişiye taşındığı (karışık küme) bunlardan bulunur."""
    if emb:
        data = {"model": model, "emb": emb}
        if turns:
            data["turns"] = [[round(float(a), 3), round(float(b), 3), str(s)] for a, b, s in turns]
        _write(Path(state_dir) / "ses" / f"{_safe(key)}.json", data)


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

    def _load_for_update(self) -> dict:
        """Değiştirmeden önce okuma: dosya o an okunamıyorsa (Windows'ta virüs tarayıcı / başka süreç) birkaç kez
        dener, yine olmazsa hata verir; kütüphane "boş" sanılıp üzerine yazılmasın (bütün sesler silinirdi).
        Bozuk dosya yedeklenip (``.bozuk-<zaman>``) boş başlanır."""
        for i in range(5):
            if not self.path.exists():
                return {"voices": {}}
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
            except OSError:
                if i == 4:
                    raise
                time.sleep(0.2)
                continue
            except ValueError:
                d = None
            if isinstance(d, dict) and isinstance(d.get("voices"), dict):
                return d
            bad = self.path.with_name(self.path.name + time.strftime(".bozuk-%Y%m%d-%H%M%S"))
            try:
                self.path.replace(bad)
            except OSError:
                pass
            return {"voices": {}}
        return {"voices": {}}

    def names(self) -> list[dict]:
        v = self.load()["voices"]
        return [{"name": n, "n": len(e.get("embs", [])), "td": bool(e.get("td"))}
                for n, e in sorted(v.items(), key=lambda kv: kv[0].casefold())]

    def td_names(self) -> set[str]:
        return {n for n, e in self.load()["voices"].items() if e.get("td")}

    @staticmethod
    def _drop(voices: dict, keep) -> int:
        """``keep(ad, x)`` False olan imzaları çıkar; boşalan adı sil. Döner: çıkan imza sayısı."""
        n = 0
        for name in list(voices):
            embs = voices[name].get("embs", [])
            kept = [x for x in embs if keep(name, x)]
            n += len(embs) - len(kept)
            voices[name]["embs"] = kept
            if not kept:
                del voices[name]
        return n

    @staticmethod
    def _append(voices: dict, name: str, emb, model, key: str, spk: str, td: bool, src: str) -> None:
        e = voices.setdefault(name, {"embs": []})
        e["td"] = bool(e.get("td")) or bool(td)
        e["embs"].append({"model": model, "emb": [round(float(x), 5) for x in emb], "key": key, "spk": spk,
                          "src": src, "t": time.strftime("%Y-%m-%d")})
        same = [x for x in e["embs"] if x.get("model") == model]
        if len(same) > MAX_PER_NAME:  # en eskisi çıkar
            drop = same[0]
            e["embs"] = [x for x in e["embs"] if x is not drop]

    def add(self, name: str, emb, model: str | None, key: str, spk: str, td: bool = False, src: str = "kume") -> bool:
        """Bir videodaki bir konuşmacının imzasını ``name`` altına ekle. Aynı video+konuşmacı önceden başka
        ada eklenmişse oradan alınır (yeniden adlandırma çift kayıt yapmasın)."""
        if _unit(emb) is None or not name.strip():
            return False
        with _FileLock(self.path):
            d = self._load_for_update()
            self._drop(d["voices"], lambda _n, x: not (x.get("key") == key and x.get("spk") == spk))
            self._append(d["voices"], name.strip(), emb, model, key, spk, td, src)
            self.save(d)
        return True

    def replace_video(self, key: str, model: str | None, entries, src: str = "kume",
                      names: set | None = None) -> list[str]:
        """Bir videonun bu modeldeki imzalarını ``entries`` ile değiştir (tek kilitte). ``names`` verilirse yalnız bu
        adların eski imzaları çıkar (eğitimde imzası çıkarılamayan kişinin önceki imzası kalsın), yoksa hepsi.
        ``entries``: [(ad, imza, td, spk), ...] (aynı ad birden çok kez olabilir) ya da {ad: (imza, td)}.
        Döner: eklenen adlar (sırasıyla, tekrarsız)."""
        if isinstance(entries, dict):
            entries = [(n, v[0], v[1] if len(v) > 1 else False, f"{src}:{n}") for n, v in entries.items()]
        good = [((n or "").strip(), e, bool(td), str(spk)) for n, e, td, spk in entries or []
                if (n or "").strip() and _unit(e) is not None]
        with _FileLock(self.path):
            d = self._load_for_update()
            self._drop(d["voices"], lambda n, x: not (x.get("key") == key and x.get("model") == model
                                                      and (names is None or n in names)))
            for name, emb, td, spk in good:
                self._append(d["voices"], name, emb, model, key, spk, td, src)
            self.save(d)
        out: list[str] = []
        for g in good:
            if g[0] not in out:
                out.append(g[0])
        return out

    def remove(self, name: str) -> bool:
        with _FileLock(self.path):
            d = self._load_for_update()
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


def cluster_purity(orig_turns: list, segments: list) -> dict:
    """İlk ayırmadaki her kümenin (sid) konuşmasının, düzeltilmiş satırlarda yine o sid'de kalan oranı (0..1).
    Kullanıcı kümenin satırlarını başka kişiye taşıdıkça düşer: o kümenin imzası başkasının sesini de içerir."""
    cur = sorted((float(s["start"]), float(s["end"]), str(s.get("speaker_id"))) for s in segments or []
                 if s.get("speaker_id") and s.get("start") is not None and s.get("end") is not None)
    same: dict = defaultdict(float)
    tot: dict = defaultdict(float)
    j0 = 0
    for a, b, sid in sorted((float(a), float(b), str(s)) for a, b, s in orig_turns or []):
        while j0 < len(cur) and cur[j0][1] <= a:
            j0 += 1
        j = j0
        while j < len(cur) and cur[j][0] < b:
            ov = min(b, cur[j][1]) - max(a, cur[j][0])
            if ov > 0:
                tot[sid] += ov
                if cur[j][2] == sid:
                    same[sid] += ov
            j += 1
    return {sid: same[sid] / t for sid, t in tot.items() if t > 0}


def ensure_orig_turns(state_dir: Path, doc: dict) -> bool:
    """Eski (1.9.5 öncesi) çıktıda ilk ayırma turları yok: ilk düzeltmeden ÖNCE (satırlar hâlâ programın
    ataması) satırlardan çıkarıp saklar. Döner: saklandı mı."""
    key = doc.get("key") or ""
    side = load_doc_voices(state_dir, key)
    if not side.get("emb") or side.get("turns") or doc.get("corrected"):
        return False
    turns = [[s["start"], s["end"], s["speaker_id"]] for s in doc.get("segments") or []
             if s.get("speaker_id") and s.get("start") is not None and s.get("end") is not None]
    if not turns:
        return False
    save_doc_voices(state_dir, key, side.get("model"), side["emb"], turns)
    return True


def learn_from_doc(state_dir: Path, doc: dict, ignore=(), skipped: list | None = None) -> list[str]:
    """Kaydedilen (düzeltilmiş) çıktıdaki adı konmuş konuşmacıların imzalarını kütüphaneye yaz.

    Bu videonun kütüphanedeki kayıtları baştan kurulur (adı değişen / genel ada dönen / silinen konuşmacının eski
    kaydı kalmasın). Satırlarının %10'undan fazlası başka kişiye taşınmış kümenin imzası öğrenilmez (karışık);
    elle eklenen kişinin imzası yoktur. İkisi de ``skipped``e yazılır: "Kaydet ve eğit" bunların temiz imzasını
    düzeltilmiş satırların sesinden çıkarır."""
    key = doc.get("key") or ""
    side = load_doc_voices(state_dir, key)
    emb, model = side.get("emb") or {}, side.get("model")
    td = doc.get("td_label") if doc.get("speaker_mode") == "basin" else None
    pur = cluster_purity(side.get("turns") or [], doc.get("segments") or []) if side.get("turns") else {}
    entries: list = []  # aynı ada iki küme olabilir (ör. hoca ikiye bölünmüştü): ikisi de öğrenilir
    missed: list = []
    for sid, v in (doc.get("speakers") or {}).items():
        lab = (v.get("label") or "").strip()
        if is_generic(lab, ignore):
            continue
        if sid not in emb or (sid in pur and pur[sid] < PURE_MIN):
            missed.append(lab)
            continue
        entries.append((lab, emb[sid], bool(td and lab == td), sid))
    learned = VoiceLibrary(state_dir).replace_video(key, model, entries) if emb else []
    if skipped is not None:
        skipped.extend(n for n in dict.fromkeys(missed) if n not in learned and n not in skipped)
    return learned
