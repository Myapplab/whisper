"""Yeniden sınıflandırma ("Veri seti" > "Yeniden sınıflandır", ``stt yeniden-siniflandir``).

İşlenmiş basın toplantılarının metnini yeniden çevirmeden, güncel eşleştirme kuralları ve güncel FPL fikstürüyle
lig / takım / maç klasörünü yeniden hesaplar. Yeri değişen kayıt yeni klasöre taşınır (metin, altyazı, json, özet,
haber taslağı); mac.json (GW, skor, erteleme), index.csv/jsonl, durum kaydı ve koleksiyon listeleri güncellenir.

Güvenlik kuralları:
- FPL fikstürü alınamazsa hiçbir şey yapılmaz (yoksa bütün GW'ler silinirdi).
- FPL verisinin sezonu dışındaki (eski sezon) kayıtlara dokunulmaz: o sezonun fikstürü elde yok.
- Podcast / film kayıtları taşınmaz (konuşmacı etiketleri farklı); artık basın toplantısına benzeyenler bildirilir.
- Önce yeni dosyalar yazılır, sonra silinecek eski dosyalar `.stt/siniflandirma_bekleyen.json`'a yazılır, sonra
  kayıtlar, en son eski dosyalar silinir. Silinemeyen (açık) ya da yarıda kalan silme bir sonraki çalıştırmada
  tamamlanır. Klasörde başka kayıt, tanınmayan dosya ya da mac.json'da elle eklenmiş "ek_bilgiler" varsa klasör
  silinmez.
- Elle düzeltilmiş .txt (gövdesi programın ürettiğinden farklı) yeniden üretilmez, olduğu gibi taşınır; .srt'de veri
  seti bilgisi yok, her zaman olduğu gibi taşınır.
- Yalnız harf büyüklüğü değişen ad ("Erik Ten Hag" -> "Erik ten Hag") taşınmaz: Windows'ta aynı dosyadır, yeni
  yazılan dosya "eski" diye silinirdi. Silmeden önce ayrıca aynı dosya mı diye bakılır.
- 1.9.8 öncesi json'larda video açıklaması yok: eski sınıflandırmanın yarışması ve başlıkta hiç geçmeyen rakibi
  (açıklamadan bulunmuştu) yerine konur ("dataset_hint" olarak saklanır, sonraki çalıştırmalarda aynısı kullanılır);
  yoksa açıklamadan doğru bulunmuş kupa maçı PL haftasına taşınırdı.
"""
from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from pathlib import Path

from .dataset import (_parse_dt, classify, fpl_data, index_row, out_paths, presser_time, rel_path, update_index,
                      write_match_json)
from .match import _ALIASES, _tnorm, is_team, looks_like_presser, parse_match
from .outputs import to_txt, write_all
from .util import atomic_write_text, log, read_jsonl, replace_retry, write_jsonl_csv

DOC_SUFFIXES = (".txt", ".srt", ".json", " - ozet.md")
NEWS_SUFFIXES = (" - haber.md", " - haber.json")
PENDING = "siniflandirma_bekleyen.json"


def _named_in_title(name: str, title: str) -> bool:
    """Takımın herhangi bir adı (belirsiz "derby", "Sporting" dahil) başlıkta geçiyor mu? Eski kurallar onu
    başlıktan (belki yanlışlıkla) bulmuş olabilir; açıklamadan geldiği varsayılmaz."""
    t = _tnorm(title)
    return any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", t) for a, c in _ALIASES if c == name)


def _old_hint(doc: dict, title: str) -> str | None:
    """1.9.8 öncesi kayıt: açıklamanın yerine eski sınıflandırmanın yarışması + başlıkta geçmeyen eski rakip."""
    old_ds = doc.get("dataset") if isinstance(doc.get("dataset"), dict) else {}
    old_m = doc.get("match") if isinstance(doc.get("match"), dict) else {}
    team = old_m.get("team") or old_m.get("home")
    opp = old_m.get("away") if team in (None, old_m.get("home")) else old_m.get("home")
    bits = [str(old_ds["competition"])] if old_ds.get("competition") else []
    if is_team(opp) and opp != team and not _named_in_title(opp, title):
        bits.append(opp)
    return " | ".join(bits) or None


def _info_of(doc: dict) -> dict:
    """Bitirmedeki ``job.info``'nun yerine: kaynak bilgisi + başlık + (1.9.8'den beri) açıklamanın başı.
    Açıklaması saklanmamış eski kayıtta, eski sınıflandırmanın açıklamadan gelmiş olabilecek bilgisi (yarışma,
    başlıkta geçmeyen rakip) "açıklama" olarak yerine konur."""
    si = doc.get("source_info")
    info = dict(si) if isinstance(si, dict) else {}
    info["title"] = str(doc.get("title") or info.get("title") or "")
    old_ds = doc.get("dataset") if isinstance(doc.get("dataset"), dict) else {}
    old_m = doc.get("match") if isinstance(doc.get("match"), dict) else {}
    if doc.get("description_head"):
        info["description"] = str(doc["description_head"])
    elif "dataset_hint" in doc:            # daha önce yeniden sınıflandırılmış eski kayıt: ilk çalıştırmanın ipucu
        if doc["dataset_hint"]:
            info["description"] = str(doc["dataset_hint"])
    elif "description_head" not in doc:   # 1.9.8 öncesi, ilk kez
        hint = _old_hint(doc, info["title"])
        if hint:
            info["description"] = hint
    if not any(info.get(k) for k in ("timestamp", "release_timestamp", "upload_date")):
        d = str(old_m.get("date") or old_ds.get("presser_date") or "")
        if len(d) == 10:
            info["upload_date"] = d.replace("-", "")
    return info


def _clean(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if v is not None and k != "fixture"}


def _season(data: dict) -> tuple | None:
    kos = [k for k in (_parse_dt(f.get("kickoff_time")) for f in data.get("fixtures") or []) if k]
    if not kos:
        return None
    return min(kos) - timedelta(days=30), max(kos) + timedelta(days=7)


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _same_name(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def _same_dir(a: Path, b: Path) -> bool:
    return _same(a, b) or str(a).casefold() == str(b).casefold()


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _has_extra(mac: Path) -> bool:
    """mac.json'a elle / başka araçla veri eklenmiş mi ("ek_bilgiler" dolu)?"""
    try:
        d = json.loads(mac.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True  # okunamıyorsa dokunma
    return isinstance(d, dict) and bool(d.get("ek_bilgiler"))


def _cleanup_folder(folder: Path, new_dir: Path, root: Path, key: str) -> None:
    """Taşınan kaydın eski klasörü boşaldıysa sil: yalnız mac.json, fpl_presser.log ve bu kaydın işareti kalmışsa.
    Başka basın toplantısı, başka kaydın işareti, tanınmayan dosya ya da dolu "ek_bilgiler" varsa dokunulmaz."""
    if _same_dir(folder, new_dir) or not folder.is_dir() or _same(folder, root):
        return
    try:
        entries = list(folder.iterdir())
        for e in entries:
            n = e.name
            if e.is_dir() or n.endswith(".txt"):
                return
            if n.startswith(".") and n.endswith(".key"):
                if e.read_text(encoding="utf-8", errors="replace").strip() != key:
                    return
            elif n == "mac.json":
                if _has_extra(e):
                    log.warning("Eski klasörde mac.json'a eklenmiş bilgi var, klasör silinmedi: %s", folder)
                    return
            elif n != "fpl_presser.log":
                return
        log_src = folder / "fpl_presser.log"
        if log_src.exists():
            dst, n = new_dir / "fpl_presser.log", 2
            while dst.exists():
                dst, n = new_dir / f"fpl_presser-{n}.log", n + 1
            replace_retry(log_src, dst)
        for e in entries:  # yalnız denetlenenler: arada yazılan dosya silinmez, klasör de silinemez (boş değil)
            if e.exists():
                e.unlink()
        folder.rmdir()
        parent = folder.parent  # boş kalan takım klasörü (lig klasörleri ve kök kalır)
        while not _same(parent, root) and not _same(parent.parent, root) and parent.is_dir() \
                and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    except OSError as e:
        log.warning("Eski klasör temizlenemedi (%s): %s", folder, e)


def _move_news(old_dir: Path, old_base: str, new_dir: Path, new_base: str, paths: dict) -> None:
    for suf in NEWS_SUFFIXES:
        src = old_dir / f"{old_base}{suf}"
        if not src.exists():
            continue
        dst = new_dir / f"{new_base}{suf}"
        replace_retry(src, dst)
        paths["news" if suf.endswith(".md") else "news_json"] = dst
        if suf.endswith(".json"):
            try:
                nj = json.loads(dst.read_text(encoding="utf-8"))
                if isinstance(nj, dict) and nj.get("metin"):
                    nj["metin"] = f"{new_base}.json"
                    atomic_write_text(dst, json.dumps(nj, ensure_ascii=False, indent=1))
            except (OSError, ValueError):
                pass


def _fix_collections(root: Path) -> None:
    """Koleksiyonlar/*/liste.jsonl: yolları index'teki güncel yerle eşitle + tum_metinler.txt yeniden. Her
    çalıştırmada bütün satırlara bakılır: önceki çalıştırma yarıda kaldıysa da tamamlanır."""
    cur = {}
    for r in read_jsonl(root / "index.jsonl"):
        if r.get("key") and r.get("txt") and r.get("json"):
            cur[r["key"]] = (str(root / r["txt"]), str(root / r["json"]))
    from .channel import _block
    for jl in (root / "Koleksiyonlar").glob("*/liste.jsonl"):
        rows = read_jsonl(jl)
        hit = False
        for r in rows:
            new = cur.get(r.get("key"))
            if new and (r.get("txt"), r.get("json")) != new and Path(new[0]).exists():
                r["txt"], r["json"] = new
                hit = True
        if not hit:
            continue
        try:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            write_jsonl_csv(jl, jl.with_name("liste.csv"), rows, fields)
            atomic_write_text(jl.with_name("tum_metinler.txt"), "\n".join(_block(r.get("txt") or "") for r in rows))
        except OSError as e:
            log.warning("Koleksiyon listesi güncellenemedi (%s): %s", jl.parent.name, e)


def _load_pending(root: Path) -> list[dict]:
    try:
        d = json.loads((root / ".stt" / PENDING).read_text(encoding="utf-8"))
        return [x for x in d if isinstance(x, dict) and x.get("key") and x.get("dir") and x.get("base")]
    except (OSError, ValueError, TypeError):
        return []


def _save_pending(root: Path, items: list[dict]) -> None:
    p = root / ".stt" / PENDING
    if items:
        atomic_write_text(p, json.dumps(items, ensure_ascii=False, indent=1))
    else:
        p.unlink(missing_ok=True)


def _finish_move(root: Path, item: dict) -> list[str]:
    """Taşınmış kaydın eski dosyalarını sil (yeni dosyayla aynı dosya olanlar hariç). Döner: silinemeyenler."""
    key = item["key"]
    old_dir, old_base = Path(item["dir"]), item["base"]
    new_dir, new_base = Path(item["new_dir"]), item["new_base"]
    written = [new_dir / f"{new_base}{suf}" for suf in DOC_SUFFIXES]
    failed = []
    for suf in DOC_SUFFIXES:
        old = old_dir / f"{old_base}{suf}"
        if not old.exists() or any(_same_file(old, w) for w in written):
            continue  # yok ya da aynı dosya (büyük/küçük harf, kısa ad...): silinmez
        try:
            old.unlink()
        except OSError as e:
            log.warning("Eski dosya silinemedi (açık olabilir), sonraki çalıştırmada denenecek: %s (%s)", old, e)
            failed.append(str(old))
    if failed:
        return failed
    marker = old_dir / f".{old_base}.key"
    try:
        if marker.exists() and not _same_file(marker, new_dir / f".{new_base}.key") \
                and marker.read_text(encoding="utf-8", errors="replace").strip() == key:
            marker.unlink()
    except OSError:
        pass
    _cleanup_folder(old_dir, new_dir, root, key)
    return []


def _retry_pending(root: Path, res: dict) -> None:
    """Önceki çalıştırmada silinemeyen / yarıda kalan eski dosyalar: index kayıt artık yeni yerdeyse silinir."""
    items = _load_pending(root)
    if not items:
        return
    where = {r["key"]: r.get("json") for r in read_jsonl(root / "index.jsonl") if r.get("key")}
    keep = []
    for it in items:
        cur = where.get(it["key"])
        old_json = Path(it["dir"]) / f"{it['base']}.json"
        if not cur or _same_file(root / cur, old_json) or str(root / cur).casefold() == str(old_json).casefold():
            continue  # index hâlâ eski yeri gösteriyor ya da kayıt yok: silinecek bir şey yok
        failed = _finish_move(root, it)
        if failed:
            keep.append(it)
            res["errors"].append({"key": it["key"], "title": Path(failed[0]).name,
                                  "error": "eski dosya silinemedi (açık olabilir)"})
    _save_pending(root, keep)


def reclassify(out_root: Path, apply: bool = True, data: dict | None = None, progress=None, state=None) -> dict:
    """``apply=False``: yalnız ne değişeceğini say (deneme turu, hiçbir dosya yazılmaz)."""
    root = Path(out_root).resolve()
    res = {"ok": True, "dry_run": not apply, "checked": 0, "moved": 0, "updated": 0, "unchanged": 0, "gw_found": 0,
           "other_season": 0, "podcast": 0, "missing": [], "errors": [], "presser_now": [], "changes": []}
    if data is None:
        data = fpl_data(root / ".stt", force=True)  # kullanıcı elle istedi: ağ bekleme süresine bakılmaz
    if not data or not data.get("fixtures"):
        return {**res, "ok": False,
                "error": "FPL fikstür verisi alınamadı (internet / fantasy.premierleague.com). Bağlantıyı kontrol "
                         "edip tekrar deneyin; veri olmadan yeniden sınıflandırma GW'leri silerdi."}
    season = _season(data)
    if apply:
        _retry_pending(root, res)
    rows = [r for r in read_jsonl(root / "index.jsonl") if r.get("key")]
    if state is None and apply:
        from .state import State
        state = State(root / ".stt")
    res["moved_files"] = {}
    for i, row in enumerate(rows):
        if progress:
            progress(i, len(rows))
        res["checked"] += 1
        try:
            _one(root, row, data, season, apply, state, res)
        except Exception as e:  # noqa: BLE001  tek bozuk kayıt bütün işi durdurmasın
            title = str(row.get("title") or row.get("json") or row.get("key"))
            log.exception("Yeniden sınıflandırılamadı: %s", title[:70])
            res["errors"].append({"key": row.get("key"), "title": title, "error": f"{type(e).__name__}: {e}"[:200]})
    if apply:
        try:
            _fix_collections(root)
        except Exception:  # noqa: BLE001
            log.exception("Koleksiyon listeleri güncellenemedi")
    res["text"] = summary_text(res)
    return res


def _one(root: Path, row: dict, data: dict, season, apply: bool, state, res: dict) -> None:
    key, rel = row["key"], str(row.get("json") or "")
    title = str(row.get("title") or rel)
    jp = root / rel if rel else None
    if jp is None or not jp.is_file():
        res["missing"].append({"key": key, "title": title, "json": rel})
        return
    try:
        doc = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not isinstance(doc.get("segments"), list):
            raise ValueError("geçersiz çıktı dosyası")
    except (OSError, ValueError) as e:
        res["errors"].append({"key": key, "title": title, "error": str(e)[:160]})
        return
    title = str(doc.get("title") or title)
    old_ds = doc.get("dataset") if isinstance(doc.get("dataset"), dict) else {}
    info = _info_of(doc)
    if old_ds.get("general") or row.get("league") == "Podcast ve Film":
        res["podcast"] += 1
        m = parse_match(info)
        if looks_like_presser(info, m) and classify(m, info, data).get("team"):
            res["presser_now"].append({"key": key, "title": title, "url": doc.get("webpage_url") or doc.get("source")})
        return
    when, _ = presser_time(info)
    if when and season and not (season[0] <= when <= season[1]):
        res["other_season"] += 1  # o sezonun fikstürü elde yok: eski sınıflandırma kalır
        return
    match = parse_match(info)
    meta = classify(match, info, data)
    new_ds = {k: v for k, v in meta.items() if k != "fixture"}
    old_dir, old_base = jp.parent, jp.stem
    new_dir, new_base = out_paths(root, meta, info["title"] or "kayit", key, create=False)
    # yalnız harf büyüklüğü farkı: aynı yer sayılır, eski ad kalır (Windows'ta aynı dosya)
    same_place = _same_dir(new_dir, old_dir) and _same_name(new_base, old_base)
    changed = _clean(new_ds) != _clean(old_ds) or _clean(match) != _clean(doc.get("match"))
    if same_place and not changed:
        res["unchanged"] += 1
        return
    res["moved" if not same_place else "updated"] += 1
    if meta.get("gw") and not old_ds.get("gw"):
        res["gw_found"] += 1
    if not same_place and len(res["changes"]) < 200:
        res["changes"].append({"key": key, "title": title, "from": rel_path(old_dir, root),
                               "to": rel_path(new_dir, root), "gw_old": old_ds.get("gw"), "gw": meta.get("gw")})
    if not apply:
        return
    if same_place:
        new_dir, new_base = old_dir, old_base
        marker = old_dir / f".{old_base}.key"
        if not marker.exists():
            marker.write_text(key)
    else:
        new_dir, new_base = out_paths(root, meta, info["title"] or "kayit", key, create=True)
        if _same_dir(new_dir, old_dir) and _same_name(new_base, old_base):   # işaret dosyası arada değişmiş
            same_place, new_dir, new_base = True, old_dir, old_base
    keep_txt = _edited_txt(old_dir / f"{old_base}.txt", doc)
    old_srt = old_dir / f"{old_base}.srt"
    keep_srt = old_srt.read_bytes() if not same_place and old_srt.exists() else None
    if "description_head" not in doc:   # 1.9.8 öncesi kayıt: ilk çalıştırmanın ipucu saklanır (sonrakiler aynısını kullanır)
        doc["description_head"] = None
        doc["dataset_hint"] = info.get("description")
    doc["match"], doc["dataset"] = match, new_ds
    paths = write_all(new_dir, new_base, doc, f"{new_base} - ozet.md")
    if keep_txt is not None:
        _write_bytes(paths["txt"], keep_txt)
        log.info("Elle düzeltilmiş metin korundu (başlık satırları eski kalabilir): %s", paths["txt"].name)
    if keep_srt is not None:
        _write_bytes(paths["srt"], keep_srt)
    if not same_place:
        _move_news(old_dir, old_base, new_dir, new_base, paths)
    if meta.get("team"):
        write_match_json(new_dir, meta)
    item = {"key": key, "dir": str(old_dir), "base": old_base, "new_dir": str(new_dir), "new_base": new_base}
    if not same_place:  # önce niyet yazılır: index güncellendikten sonra çökse de sonraki çalıştırma eski dosyaları siler
        pending = [x for x in _load_pending(root) if x.get("key") != key] + [item]
        _save_pending(root, pending)
    update_index(root, index_row(root, key, meta, doc, paths))
    prev = state.get(key) if state is not None else None
    if prev:
        pinfo = dict(prev.get("info") or {})
        pinfo["json"] = str(paths["json"])
        state.set(prev["key"], source=prev["source"] or doc.get("source") or "", status=prev["status"],
                  title=prev["title"] or title, out_dir=str(new_dir), error=prev.get("error") or "",
                  info=pinfo)
    if not same_place:
        res.setdefault("moved_files", {})[str(old_dir / f"{old_base}.txt")] = str(paths["txt"])
        failed = _finish_move(root, item)
        if failed:
            res["errors"].append({"key": key, "title": title,
                                  "error": "eski dosya silinemedi (açık olabilir); sonraki çalıştırmada silinecek"})
        else:
            _save_pending(root, [x for x in _load_pending(root) if x.get("key") != key])
    log.info("Yeniden sınıflandırıldı: %s -> %s", title[:70], rel_path(paths["txt"], root))


def _edited_txt(p: Path, doc: dict) -> bytes | None:
    """.txt elle düzeltilmiş mi? Gövde (başlık satırlarından sonrası) programın bu json'dan ürettiğinden farklıysa
    dosyanın kendisi (bayt olarak) döner, değilse None (yeniden üretilebilir)."""
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return raw  # başka kodlamayla kaydedilmiş: elle açılıp kaydedilmiş, dokunma
    gen = to_txt(doc)
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text
    gen_body = gen.split("\n\n", 1)[1] if "\n\n" in gen else gen
    return None if body.strip() == gen_body.strip() else raw


def _write_bytes(p: Path, data: bytes) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    replace_retry(tmp, p)


def summary_text(res: dict, limit: int = 12) -> str:
    if not res.get("ok"):
        return res.get("error") or "Yeniden sınıflandırma yapılamadı."
    verb = ("taşınacak", "güncellenecek") if res.get("dry_run") else ("taşındı", "güncellendi")
    lines = [f"{res['checked']} kayıt denetlendi: {res['moved']} {verb[0]}, {res['updated']} {verb[1]} "
             f"(GW / skor / maç bilgisi), {res['unchanged']} zaten doğru."]
    if res.get("gw_found"):
        lines.append(f"{res['gw_found']} kayda GW bulundu.")
    for c in res.get("changes", [])[:limit]:
        lines.append(f"  {c['from']}  ->  {c['to']}")
    if len(res.get("changes", [])) > limit:
        lines.append(f"  ... ve {len(res['changes']) - limit} kayıt daha")
    if res.get("other_season"):
        lines.append(f"{res['other_season']} kayıt eski sezondan (fikstürü yok), olduğu gibi bırakıldı.")
    if res.get("presser_now"):
        lines.append(f"'Podcast ve Film'deki {len(res['presser_now'])} kayıt basın toplantısına benziyor; konuşmacı "
                     "etiketleri için linkini tekrar ekleyip konuşmacı biçimini 'Basın toplantısı' seçin:")
        lines += [f"  {x['title']}" for x in res["presser_now"][:limit]]
    if res.get("missing"):
        lines.append(f"{len(res['missing'])} kaydın dosyası bulunamadı (silinmiş / taşınmış olabilir).")
    if res.get("errors"):
        lines.append(f"{len(res['errors'])} kayıt okunamadı ya da yazılamadı (ayrıntı günlükte).")
    return "\n".join(lines)
