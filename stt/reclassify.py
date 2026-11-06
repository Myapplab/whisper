"""Yeniden sınıflandırma ("Veri seti" > "Yeniden sınıflandır", ``stt yeniden-siniflandir``).

İşlenmiş basın toplantılarının metnini yeniden çevirmeden, güncel eşleştirme kuralları ve güncel FPL fikstürüyle
lig / takım / maç klasörünü yeniden hesaplar. Yeri değişen kayıt yeni klasöre taşınır (metin, altyazı, json, özet,
haber taslağı); mac.json (GW, skor, erteleme), index.csv/jsonl, durum kaydı ve koleksiyon listeleri güncellenir.

Güvenlik kuralları:
- FPL fikstürü alınamazsa hiçbir şey yapılmaz (yoksa bütün GW'ler silinirdi).
- FPL verisinin sezonu dışındaki (eski sezon) kayıtlara dokunulmaz: o sezonun fikstürü elde yok.
- Podcast / film kayıtları taşınmaz (konuşmacı etiketleri farklı); artık basın toplantısına benzeyenler bildirilir.
- Önce yeni dosyalar yazılır, sonra kayıtlar, en son eski dosyalar silinir: yarıda kalırsa bir sonraki çalıştırma
  tamamlar. Klasörde başka kayıt, tanınmayan dosya ya da mac.json'da elle eklenmiş "ek_bilgiler" varsa klasör
  silinmez.
- Yalnız harf büyüklüğü değişen ad ("Erik Ten Hag" -> "Erik ten Hag") taşınmaz: Windows'ta aynı dosyadır, yeni
  yazılan dosya "eski" diye silinirdi. Silmeden önce ayrıca aynı dosya mı diye bakılır.
- 1.9.8 öncesi json'larda video açıklaması yok: eski sınıflandırmanın yarışması ve başlıkta geçmeyen rakibi
  (açıklamadan bulunmuştu) yerine konur; yoksa açıklamadan doğru bulunmuş kupa maçı PL haftasına taşınırdı.
"""
from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

from .dataset import (_parse_dt, classify, fpl_data, index_row, out_paths, presser_time, rel_path, update_index,
                      write_match_json)
from .match import _team_hits, is_team, looks_like_presser, parse_match
from .outputs import write_all
from .util import atomic_write_text, log, read_jsonl, replace_retry, write_jsonl_csv

DOC_SUFFIXES = (".txt", ".srt", ".json", " - ozet.md")
NEWS_SUFFIXES = (" - haber.md", " - haber.json")


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
    elif "description_head" not in doc:
        in_title = {c for *_, c in _team_hits(info["title"])[1]}
        bits = [str(old_ds["competition"])] if old_ds.get("competition") else []
        bits += [x for x in (old_m.get("home"), old_m.get("away")) if is_team(x) and x not in in_title]
        if bits:
            info["description"] = " | ".join(bits)
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
        for e in folder.iterdir():
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
    rows = [r for r in read_jsonl(root / "index.jsonl") if r.get("key")]
    if state is None and apply:
        from .state import State
        state = State(root / ".stt")
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
    doc["match"], doc["dataset"] = match, new_ds
    paths = write_all(new_dir, new_base, doc, f"{new_base} - ozet.md")
    if not same_place:
        _move_news(old_dir, old_base, new_dir, new_base, paths)
    if meta.get("team"):
        write_match_json(new_dir, meta)
    update_index(root, index_row(root, key, meta, doc, paths))
    prev = state.get(key) if state is not None else None
    if prev:
        pinfo = dict(prev.get("info") or {})
        pinfo["json"] = str(paths["json"])
        state.set(prev["key"], source=prev["source"] or doc.get("source") or "", status=prev["status"],
                  title=prev["title"] or title, out_dir=str(new_dir), error=prev.get("error") or "",
                  info=pinfo)
    if not same_place:
        written = [paths[k] for k in ("txt", "srt", "json", "md")]
        for suf in DOC_SUFFIXES:
            old = old_dir / f"{old_base}{suf}"
            if any(_same_file(old, w) for w in written):
                continue  # aynı dosya (büyük/küçük harf, kısa ad...): silinmez
            try:
                old.unlink(missing_ok=True)
            except OSError as e:
                log.warning("Eski dosya silinemedi (açık olabilir): %s (%s)", old, e)
        marker = old_dir / f".{old_base}.key"
        new_marker = new_dir / f".{new_base}.key"
        if marker.exists() and not _same_file(marker, new_marker) \
                and marker.read_text(encoding="utf-8", errors="replace").strip() == key:
            marker.unlink()
        _cleanup_folder(old_dir, new_dir, root, key)
    log.info("Yeniden sınıflandırıldı: %s -> %s", title[:70], rel_path(paths["txt"], root))


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
