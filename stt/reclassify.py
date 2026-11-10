"""Yeniden sınıflandırma ("Veri seti" > "Yeniden sınıflandır", ``stt yeniden-siniflandir``).

İşlenmiş basın toplantılarının metnini yeniden çevirmeden, güncel eşleştirme kuralları ve güncel FPL fikstürüyle
lig / takım / maç klasörünü yeniden hesaplar. Yeri değişen kayıt yeni klasöre taşınır (metin, altyazı, json, özet,
haber taslağı); mac.json (GW, skor, erteleme), index.csv/jsonl, durum kaydı ve koleksiyon listeleri güncellenir.

Güvenlik kuralları:
- FPL fikstürü alınamazsa hiçbir şey yapılmaz (yoksa bütün GW'ler silinirdi).
- Eski sezonun kaydı o sezonun fikstürüyle yeniden sınıflandırılır (1.9.28: uygulamayla gelen 2023-24 / 2024-25 /
  2025-26 fikstürleri + indirilen sezonların kopyası; ``dataset.season_data``): maç + GW bulunur, o sezon PL'de olan
  takım "Diger Takimlar"dan "Premier League"e geçer. Fikstürü hiç olmayan sezonun kaydı yeniden sınıflandırılmaz,
  yalnız yeni düzene taşınır (takım / maç / ad aynı kalır).
- Düzen (1.9.28): "<lig>/<sezon>/<takım>/<hoca>/<maç>". Eski düzenler ("<lig>/<takım>/<maç>", 1.9.25
  "<lig>/<takım>/<sezon>/<maç>", 1.9.26 "<lig>/<takım>/<sezon> <hoca>/<maç>") tanınır ve taşınır.
- Podcast / film kayıtları taşınmaz (konuşmacı etiketleri farklı); artık basın toplantısına benzeyenler bildirilir.
- Önce yeni dosyalar yazılır, sonra silinecek eski dosyalar (boyut + değişiklik zamanıyla) `.stt/
  siniflandirma_bekleyen.json`'a yazılır, sonra kayıtlar, en son eski dosyalar silinir. Silinemeyen (açık) ya da
  yarıda kalan silme bir sonraki çalıştırmada tamamlanır; ama yalnız dosya o andan beri değişmediyse ve hâlâ bu
  kayda aitse (sonradan düzeltilmiş ya da aynı adı almış başka kaydın dosyası silinmez). Klasörde başka kayıt,
  tanınmayan dosya ya da mac.json'da elle eklenmiş "ek_bilgiler" varsa klasör silinmez.
- .txt yalnız programın ürettiği hâliyle aynıysa yeniden üretilir. Başlık satırları programınki, gövde elle
  düzeltilmişse gövde korunur, başlık yenilenir; başka her durumda (başlık silinmiş, en üste not eklenmiş, başka
  kodlama) dosya olduğu gibi kalır / taşınır. .srt'de veri seti bilgisi yok, olduğu gibi kalır. Yerinde güncellemede
  önce .json ve özet yazılır, .txt en son; eski .txt / .srt okunamıyorsa (açık) kayıt bu çalıştırmada taşınmaz.
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
from datetime import datetime
from pathlib import Path

from .dataset import (NO_SEASON, OTHER_DIR, PL_DIR, classify, fpl_data, index_row, match_folder_name, out_paths,
                      presser_time, rel_path, season_data, season_of, set_team_manager, update_index, write_match_json)
from .match import _ALIASES, _tnorm, is_team, looks_like_presser, parse_match
from .outputs import to_srt, to_summary, to_txt
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


def _channel(info: dict) -> str:
    return str(info.get("channel") or info.get("uploader") or "")


def _is_season_dir(name: str) -> bool:
    """Programın sezon klasörü mü: "2025-2026", "2025-2026 Pep Guardiola" (1.9.26; ad bilinen bir hoca),
    "Sezon bilinmiyor". Elle açılmış "2025-2026 yedek" değil (içindeki kayıtlara dokunulmaz)."""
    m = re.fullmatch(r"\d{4}-\d{4}(?: (.+))?", name)
    if not m:
        return name == NO_SEASON
    return not m.group(1) or m.group(1).casefold() in _manager_dir_names()


def _manager_dir_names() -> set[str]:
    from .match import MANAGER_HINTS, MANAGER_HISTORY, safe_filename
    names = set(MANAGER_HINTS.values()) | {n for rows in MANAGER_HISTORY.values() for _, n in rows if n}
    return {safe_filename(n).casefold() for n in names}


def _same_match_name(old: str, new: str, full: str | None) -> bool:
    """Aynı maç klasörü adı. ``full``: kısaltılmamış ad; Windows yol sınırı için yeni ad kısaltıldıysa (uzun hocalı
    sezon klasörü out_paths'te maç adını kısaltır) eski ad da yeni ad da onun başıysa aynı maç. Elle / eski sürümle
    uzatılmış ad ("... (eski ad)") aynı sayılmaz."""
    if _same_name(new, old):
        return True
    if not full or _same_name(new, full):
        return False
    f = full.casefold()
    return len(new) >= 20 and f.startswith(new.casefold()) and f.startswith(old.casefold())


_PLAIN_SEASON = re.compile(r"\d{4}-\d{4}")


def _is_plain_season(name: str) -> bool:
    """1.9.28 düzeninin ligin altındaki sezon klasörü: "2025-2026" ya da "Sezon bilinmiyor"."""
    return bool(_PLAIN_SEASON.fullmatch(name)) or name == NO_SEASON


def _team_of(d: Path, root: Path) -> str | None:
    """Maç klasörü programın hangi düzeninde, hangi takımın: "<lig>/<sezon>/<takım>[/<hoca>]/<maç>" (1.9.28),
    "<lig>/<takım>/<sezon[ hoca]>/<maç>" (1.9.25/26), "<lig>/<takım>/<maç>" (eski). Tanınmayan (elle açılmış
    "2025-2026 yedek", bilinmeyen alt klasör) yerde None."""
    try:
        parts = d.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 3 or parts[0] not in (PL_DIR, OTHER_DIR):
        return None
    mid = parts[1:-1]
    if len(mid) == 1:
        return None if _is_plain_season(mid[0]) else mid[0]
    if len(mid) == 2:
        if _is_plain_season(mid[0]):
            return mid[1]                                   # 1.9.28, hocasız
        return mid[0] if _is_season_dir(mid[1]) else None   # 1.9.25 / 1.9.26
    if len(mid) == 3 and _is_plain_season(mid[0]) and mid[2].casefold() in _manager_dir_names():
        return mid[1]                                       # 1.9.28, hocalı
    return None


def _same_match(old_dir: Path, new_dir: Path, full: str | None = None, root: Path | None = None) -> bool:
    """Aynı takımın aynı adlı maç klasörü; düzen (sezon / hoca katmanı, lig) farklı olabilir. ``root``: veri seti
    kökü (yoksa yeni klasörden çıkarılır)."""
    if not _same_match_name(old_dir.name, new_dir.name, full):
        return False
    if root is None:
        root = next((p.parent for p in new_dir.parents if p.name in (PL_DIR, OTHER_DIR)), None)
        if root is None:
            return False
    a, b = _team_of(old_dir, root), _team_of(new_dir, root)
    return bool(a and b and a.casefold() == b.casefold())


def _has_extra(mac: Path) -> bool:
    """mac.json'a elle / başka araçla veri eklenmiş mi ("ek_bilgiler" dolu)?"""
    try:
        d = json.loads(mac.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True  # okunamıyorsa dokunma
    return isinstance(d, dict) and bool(d.get("ek_bilgiler"))


def _extra_of(mac: Path):
    try:
        d = json.loads(mac.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d.get("ek_bilgiler") if isinstance(d, dict) else None


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
                # ek_bilgiler yeni klasörün mac.json'una aynen geçtiyse (sezon taşıması) eski klasör silinebilir
                if _has_extra(e) and _extra_of(e) != _extra_of(new_dir / "mac.json"):
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


class _Unreadable(Exception):
    """Eski metin / altyazı okunamadı (başka programda açık): elle düzeltilmiş mi bilinemez, kayıt taşınmaz."""


def _fingerprint(p: Path) -> list[int] | None:
    try:
        st = p.stat()
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def _load_pending(root: Path) -> list[dict]:
    try:
        d = json.loads((root / ".stt" / PENDING).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    keys = ("key", "dir", "base", "new_dir", "new_base")
    return [x for x in d if isinstance(x, dict) and all(isinstance(x.get(k), str) and x.get(k) for k in keys)] \
        if isinstance(d, list) else []


def _save_pending(root: Path, items: list[dict]) -> None:
    p = root / ".stt" / PENDING
    if items:
        atomic_write_text(p, json.dumps(items, ensure_ascii=False, indent=1))
    else:
        p.unlink(missing_ok=True)


def _owner(old_dir: Path, old_base: str) -> str | None:
    """Eski yerdeki kayıt kimin: json'daki anahtar, yoksa işaret dosyası."""
    try:
        d = json.loads((old_dir / f"{old_base}.json").read_text(encoding="utf-8"))
        if isinstance(d, dict) and d.get("key"):
            return str(d["key"])
    except (OSError, ValueError):
        pass
    try:
        return (old_dir / f".{old_base}.key").read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def _finish_move(root: Path, item: dict, retry: bool = False) -> tuple[list[str], list[str]]:
    """Taşınmış kaydın eski dosyalarını sil (yeni dosyayla aynı dosya olanlar hariç). ``retry``: önceki
    çalıştırmadan kalan silme; dosya kaydedildiği andan beri değişmişse ya da başka kayda aitse silinmez.
    Döner: (silinemeyenler, değiştiği için bırakılanlar)."""
    key = item["key"]
    old_dir, old_base = Path(item["dir"]), item["base"]
    new_dir, new_base = Path(item["new_dir"]), item["new_base"]
    written = [new_dir / f"{new_base}{suf}" for suf in DOC_SUFFIXES]
    fps = item.get("files") or {}
    owner = _owner(old_dir, old_base) if retry else key
    failed, changed = [], []
    for suf in DOC_SUFFIXES:
        old = old_dir / f"{old_base}{suf}"
        if not old.exists() or any(_same_file(old, w) for w in written):
            continue  # yok ya da aynı dosya (büyük/küçük harf, kısa ad...): silinmez
        if retry and (owner not in (None, key) or fps.get(suf) is None or _fingerprint(old) != fps.get(suf)):
            changed.append(str(old))  # sonradan düzeltilmiş ya da aynı adla başka kayıt: dokunma
            continue
        try:
            old.unlink()
        except OSError as e:
            log.warning("Eski dosya silinemedi (açık olabilir), sonraki çalıştırmada denenecek: %s (%s)", old, e)
            failed.append(str(old))
    if changed:
        log.warning("Eski yerdeki dosya taşımadan sonra değişmiş (ya da başka kaydın), silinmedi: %s", ", ".join(changed))
    if failed or changed:
        return failed, changed
    marker = old_dir / f".{old_base}.key"
    try:
        if marker.exists() and not _same_file(marker, new_dir / f".{new_base}.key") \
                and marker.read_text(encoding="utf-8", errors="replace").strip() == key:
            marker.unlink()
    except OSError:
        pass
    _cleanup_folder(old_dir, new_dir, root, key)
    return [], []


def _retry_pending(root: Path, res: dict) -> None:
    """Önceki çalıştırmada silinemeyen / yarıda kalan eski dosyalar: index kayıt artık yeni yerdeyse silinir."""
    items = _load_pending(root)
    if not items:
        return
    where = {r["key"]: r.get("json") for r in read_jsonl(root / "index.jsonl") if r.get("key")}
    keep = []
    for it in items:
        try:
            cur = where.get(it["key"])
            old_json = Path(it["dir"]) / f"{it['base']}.json"
            if not cur or not (root / cur).is_file() or _same_file(root / cur, old_json) \
                    or str(root / cur).casefold() == str(old_json).casefold():
                continue  # index hâlâ eski yeri gösteriyor, kayıt ya da yeni dosya yok: silinecek bir şey yok
            failed, changed = _finish_move(root, it, retry=True)
            if failed:
                keep.append(it)
                res["errors"].append({"key": it["key"], "title": Path(failed[0]).name,
                                      "error": "eski dosya silinemedi (açık olabilir)"})
            if changed:
                res["errors"].append({"key": it["key"], "title": Path(changed[0]).name,
                                      "error": "eski yerdeki dosya taşımadan sonra değişmiş; silinmedi, elle bakın"})
        except Exception as e:  # noqa: BLE001
            log.warning("Bekleyen silme yapılamadı (%s): %s", it.get("key"), e)
    _save_pending(root, keep)


def reclassify(out_root: Path, apply: bool = True, data: dict | None = None, progress=None, state=None) -> dict:
    """``apply=False``: yalnız ne değişeceğini say (deneme turu, hiçbir dosya yazılmaz)."""
    root = Path(out_root).resolve()
    res = {"ok": True, "dry_run": not apply, "checked": 0, "moved": 0, "updated": 0, "unchanged": 0, "gw_found": 0,
           "other_season": 0, "podcast": 0, "to_pl": 0, "by_season": {}, "missing": [], "errors": [],
           "presser_now": [], "changes": []}
    if data is None:
        data = fpl_data(root / ".stt", force=True)  # kullanıcı elle istedi: ağ bekleme süresine bakılmaz
    if not data or not data.get("fixtures"):
        return {**res, "ok": False,
                "error": "FPL fikstür verisi alınamadı (internet / fantasy.premierleague.com). Bağlantıyı kontrol "
                         "edip tekrar deneyin; veri olmadan yeniden sınıflandırma GW'leri silerdi."}
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
            _one(root, row, data, apply, state, res)
        except _Unreadable as e:  # beklenen durum (dosya açık): iz yazmadan bildir
            title = str(row.get("title") or row.get("json") or row.get("key"))
            log.warning("Yeniden sınıflandırılmadı: %s (%s)", title[:70], e)
            res["errors"].append({"key": row.get("key"), "title": title, "error": str(e)[:200]})
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


def _one(root: Path, row: dict, data: dict, apply: bool, state, res: dict) -> None:
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
    old_dir, old_base = jp.parent, jp.stem
    season_only = False
    if when and season_data(when, data, root / ".stt") is None:
        # o sezonun fikstürü elde yok: eski sınıflandırma kalır, yalnız yeni düzene (sezon / takım / hoca) taşınır
        meta = set_team_manager(root, dict(old_ds), key, _channel(info)) if old_ds.get("team") else dict(old_ds)
        new_dir = out_paths(root, meta, info["title"] or "kayit", key, create=False)[0] if old_ds.get("team") \
            else old_dir
        if _same_dir(new_dir, old_dir) or not _same_match(old_dir, new_dir, match_folder_name(meta), root):
            # zaten doğru sezon klasöründe, ya da başka bir yere / başka adlı maç klasörüne gidecekti (eski sürümün
            # ya da elle verilmiş ad): dokunma, yoksa farklı klasörler birleşirdi
            res["other_season"] += 1
            return
        match, season_only = doc.get("match"), True
    else:
        match = parse_match(info)
        meta = set_team_manager(root, classify(match, info, data, root / ".stt"), key, _channel(info))
    new_ds = {k: v for k, v in meta.items() if k != "fixture"}
    keep = old_base if season_only else None   # sezon taşıması: dosya adı aynen kalır
    new_dir, new_base = out_paths(root, meta, info["title"] or "kayit", key, create=False, base=keep)
    # yalnız harf büyüklüğü farkı: aynı yer sayılır, eski ad kalır (Windows'ta aynı dosya)
    same_place = _same_dir(new_dir, old_dir) and _same_name(new_base, old_base)
    changed = _clean(new_ds) != _clean(old_ds) or _clean(match) != _clean(doc.get("match"))
    if same_place and not changed:
        res["unchanged"] += 1
        return
    change = {"key": key, "title": title, "from": rel_path(old_dir, root), "to": rel_path(new_dir, root),
              "gw_old": old_ds.get("gw"), "gw": meta.get("gw")}

    def count() -> None:  # deneme turunda hemen, uygulamada yalnız başarınca sayılır
        res["moved" if not same_place else "updated"] += 1
        if meta.get("gw") and not old_ds.get("gw"):
            res["gw_found"] += 1
        if meta.get("is_pl") and old_ds.get("team") and not old_ds.get("is_pl"):
            res["to_pl"] += 1
        sk = season_of(meta) if meta.get("team") else None
        if sk:
            res["by_season"][sk] = res["by_season"].get(sk, 0) + 1
        if not same_place and len(res["changes"]) < 200:
            res["changes"].append(change)
    if not apply:
        count()
        return
    # önce eski metin / altyazı okunur (açıksa hiçbir şey yazılmadan bu kayıt atlanır)
    keep_txt = _edited_txt(old_dir / f"{old_base}.txt", doc)
    old_srt = old_dir / f"{old_base}.srt"
    try:
        keep_srt = old_srt.read_bytes() if old_srt.exists() else None
    except OSError as e:
        raise _Unreadable(f"altyazı dosyası okunamadı (açık olabilir), bu kayıt taşınmadı: {e}") from e
    if same_place:
        new_dir, new_base = old_dir, old_base
        marker = old_dir / f".{old_base}.key"
        if not marker.exists():
            marker.write_text(key)
    else:
        new_dir, new_base = out_paths(root, meta, info["title"] or "kayit", key, create=True, base=keep)
        if _same_dir(new_dir, old_dir) and _same_name(new_base, old_base):   # işaret dosyası arada değişmiş
            same_place, new_dir, new_base = True, old_dir, old_base
    if "description_head" not in doc:   # 1.9.8 öncesi kayıt: ilk çalıştırmanın ipucu saklanır (sonrakiler aynısını kullanır)
        doc["description_head"] = None
        doc["dataset_hint"] = info.get("description")
    doc["match"], doc["dataset"] = match, new_ds
    paths = _write_outputs(new_dir, new_base, doc, keep_txt, keep_srt, same_place)
    if not same_place:
        _move_news(old_dir, old_base, new_dir, new_base, paths)
    same_match = _same_match(old_dir, new_dir, match_folder_name(meta) if meta.get("team") else None, root)
    if (season_only or same_match) and (old_dir / "mac.json").is_file():
        # yalnız sezon klasörüne taşınan maç: eski mac.json (fikstür, skor, elle eklenen ek_bilgiler) yeni yere geçer
        _carry_mac(old_dir / "mac.json", new_dir / "mac.json")
    if season_only and (new_dir / "mac.json").is_file():
        _set_mac_season(new_dir / "mac.json", meta)   # eski bilgiler (skor, fikstür) boş alanlarla ezilmesin
    elif meta.get("team"):
        write_match_json(new_dir, meta)
    item = {"key": key, "dir": str(old_dir), "base": old_base, "new_dir": str(new_dir), "new_base": new_base,
            "files": {suf: _fingerprint(old_dir / f"{old_base}{suf}") for suf in DOC_SUFFIXES}}
    if not same_place:  # önce niyet yazılır: index güncellendikten sonra çökse de sonraki çalıştırma eski dosyaları siler
        _save_pending(root, [x for x in _load_pending(root) if not _same_item(x, key, old_dir, old_base)
                             and not _same_item(x, key, new_dir, new_base)] + [item])
    update_index(root, index_row(root, key, meta, doc, paths))
    prev = state.get(key) if state is not None else None
    if prev:
        pinfo = dict(prev.get("info") or {})
        pinfo["json"] = str(paths["json"])
        state.set(prev["key"], source=prev["source"] or doc.get("source") or "", status=prev["status"],
                  title=prev["title"] or title, out_dir=str(new_dir), error=prev.get("error") or "",
                  info=pinfo)
    change["to"] = rel_path(new_dir, root)
    count()
    if not same_place:
        moved = res.setdefault("moved_files", {})
        moved[str(old_dir / f"{old_base}.txt")] = str(paths["txt"])
        failed, _ = _finish_move(root, item)
        if not old_dir.exists():
            moved[str(old_dir)] = str(new_dir)  # klasör boşalıp silindiyse "Klasörü aç" yenisini açsın
        if failed:
            res["errors"].append({"key": key, "title": title,
                                  "error": "eski dosya silinemedi (açık olabilir); sonraki çalıştırmada silinecek"})
        else:
            _save_pending(root, [x for x in _load_pending(root) if not _same_item(x, key, old_dir, old_base)])
    log.info("Yeniden sınıflandırıldı: %s -> %s", title[:70], rel_path(paths["txt"], root))


def _carry_mac(old: Path, new: Path) -> None:
    """Eski mac.json yeni klasöre: yeni yoksa kopya; varsa (güncellemeden sonra o maçın yeni toplantısı işlenmiş)
    yenide boş olan alanlar (ek_bilgiler, skor, fikstür) eskisinden doldurulur."""
    try:
        if not new.exists():
            atomic_write_text(new, old.read_text(encoding="utf-8"))
            return
        o, n = json.loads(old.read_text(encoding="utf-8")), json.loads(new.read_text(encoding="utf-8"))
        if not (isinstance(o, dict) and isinstance(n, dict)):
            return
        empty = (None, "", {}, [])
        fill = {k: v for k, v in o.items() if v not in empty and n.get(k) in empty}
        if fill:
            n.update(fill)
            atomic_write_text(new, json.dumps(n, ensure_ascii=False, indent=1))
    except (OSError, ValueError) as e:
        log.warning("Eski mac.json yeni klasöre aktarılamadı (%s): %s", old, e)


def _set_mac_season(p: Path, meta: dict) -> None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        want = {"season": season_of(meta), "team_manager": meta.get("team_manager")}
        if isinstance(d, dict) and any(d.get(k) != v for k, v in want.items()):
            d.update(want)
            atomic_write_text(p, json.dumps(d, ensure_ascii=False, indent=1))
    except (OSError, ValueError) as e:
        log.warning("mac.json'a sezon yazılamadı (%s): %s", p, e)


def _same_item(x: dict, key: str, d: Path, base: str) -> bool:
    return x.get("key") == key and str(x.get("dir", "")).casefold() == str(d).casefold() \
        and str(x.get("base", "")).casefold() == base.casefold()


def _split_head(text: str) -> tuple[str, str]:
    head, _, body = text.partition("\n\n")
    return head.strip(), body.strip()


def _edited_txt(p: Path, doc: dict) -> str | bytes | None:
    """.txt elle düzeltilmiş mi (``doc``: dosyayı üreten, henüz güncellenmemiş kayıt)?
    - None: programın ürettiğiyle aynı, yeniden üretilebilir.
    - str: başlık satırları programınki, gövde düzeltilmiş: gövde (yeni başlıkla yazılır).
    - bytes: başka her durum (başlık silinmiş / değişmiş, en üste not, başka kodlama): dosya olduğu gibi kalır.
    Dosya var ama okunamıyorsa _Unreadable."""
    if not p.exists():
        return None
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise _Unreadable(f"metin dosyası okunamadı (açık olabilir), bu kayıt taşınmadı: {e}") from e
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return raw  # başka kodlamayla kaydedilmiş: elle açılıp kaydedilmiş, dokunma
    head, body = _split_head(text)
    gen_head, gen_body = _split_head(to_txt(doc))
    if head == gen_head:
        return None if body == gen_body else body
    return raw


def _write_outputs(d: Path, base: str, doc: dict, keep_txt, keep_srt: bytes | None, same_place: bool) -> dict:
    """outputs.write_all'ın yerine: önce .json ve özet, sonra .txt / .srt; elle düzeltilmiş metin / altyazı korunur.
    Yerinde güncellemede .srt hiç yazılmaz (veri seti bilgisi yok), olduğu gibi bırakılan .txt'ye de dokunulmaz:
    sonraki bir yazma başarısız olsa bile kullanıcının düzeltmesi kaybolmaz."""
    d.mkdir(parents=True, exist_ok=True)
    doc.setdefault("created", datetime.now().strftime("%Y-%m-%d %H:%M"))
    paths = {"txt": d / f"{base}.txt", "srt": d / f"{base}.srt", "json": d / f"{base}.json",
             "md": d / f"{base} - ozet.md"}
    atomic_write_text(paths["json"], json.dumps(doc, ensure_ascii=False, indent=1))
    atomic_write_text(paths["md"], to_summary(doc, paths["txt"].name))
    if keep_txt is None:
        atomic_write_text(paths["txt"], to_txt(doc))
    elif isinstance(keep_txt, str):  # düzeltilmiş gövde + yeni başlık satırları
        atomic_write_text(paths["txt"], to_txt(doc).partition("\n\n")[0] + "\n\n" + keep_txt + "\n")
        log.info("Elle düzeltilmiş metin korundu (başlık yenilendi): %s", paths["txt"].name)
    elif not same_place:
        _write_bytes(paths["txt"], keep_txt)
        log.info("Elle değiştirilmiş metin olduğu gibi taşındı: %s", paths["txt"].name)
    if keep_srt is not None:
        if not same_place:
            _write_bytes(paths["srt"], keep_srt)
    else:
        atomic_write_text(paths["srt"], to_srt(doc["segments"], bool(doc.get("speakers"))))
    return paths


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
    if res.get("to_pl"):
        lines.append(f"{res['to_pl']} kayıt 'Diğer Takımlar'dan Premier League'e geçiyor (takım o sezon PL'deydi).")
    if res.get("by_season"):
        lines.append("Sezonlara göre: " + ", ".join(f"{k}: {v}" for k, v in sorted(res["by_season"].items())))
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
