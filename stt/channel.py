"""Kanal / oynatma listesi: bütün videoları listele, konuya göre seç (ör. yalnızca basın toplantıları),
seçilenleri sıraya koy; 'takip' ile kanaldaki yeni videolar düzenli aralıklarla otomatik eklenir.

YouTube (kanal, oynatma listesi, /videos, /streams), Dailymotion, Vimeo, ok.ru, Twitch... yt-dlp'nin
listeleyebildiği her site çalışır. Listeleme 'düz' yapılır (video sayfaları tek tek açılmaz) — 1000 videoluk
kanal da ~1 dakikada listelenir.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .util import atomic_write_text, log, read_jsonl, write_jsonl_csv

# Konu hazırları: başlıkta (aksan/büyük harf farketmez) bunlardan biri geçerse seçilir
TOPICS = {
    "basin": ("press conference", "presser", "pre-match", "prematch", "post-match", "postmatch", "news conference",
              "media conference", "basın toplantısı", "basin toplantisi", "maç öncesi", "maç sonrası",
              "mətbuat konfransı", "rueda de prensa", "conferencia de prensa", "conférence de presse",
              "pressekonferenz", "conferenza stampa", "conferência de imprensa", "persconferentie",
              "пресс-конференция", "press talk", "pre-game", "post-game",
              # kulüp kanallarında basın toplantısı başka adlarla da yükleniyor (Man City: "MANAGER'S PREVIEW: Pep ...")
              "manager's preview", "managers preview", "manager preview", "head coach preview"),
    "roportaj": ("interview", "röportaj", "reaction", "speaks", "reacts", "müsahibə", "entrevista", "intervista"),
    "hepsi": (),
}
TOPIC_LABELS = {"basin": "basın toplantıları", "roportaj": "röportajlar", "hepsi": "tümü"}


def collection_name(channel: str, topic: str | None, custom: str | None = None) -> str:
    """Koleksiyon klasörü adı (GUI ve komut satırı aynı adı kullansın): '<kanal> - <konu>'."""
    label = TOPIC_LABELS.get((topic or "").lower()) or (custom or "").strip()
    return f"{channel} - {label}".strip(" -") if label else channel


def _words(s: str | None) -> list[str]:
    return [w.strip() for w in re.split(r"[,;\n]", s or "") if w.strip()]


def _norm(s: str) -> str:
    """Küçük harf, Türkçe/aksan sadeleştirme; Kiril/Arap harfleri KORUNUR; tire/çizgi = boşluk."""
    import unicodedata

    from .match import _TR
    s = unicodedata.normalize("NFKD", (s or "").translate(_TR))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[\W_]+", " ", s).split())


def is_collection_url(url: str) -> bool:
    """Kanal / oynatma listesi linki mi (tek video değil)?"""
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    path = p.path or "/"
    q = parse_qs(p.query)
    if host.endswith("youtube.com"):
        if path == "/playlist" and "list" in q:
            return True
        if path == "/watch":
            return False
        return bool(re.match(r"^/(@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)(/(videos|streams|shorts|playlists|featured)?)?/?$", path))
    if host == "youtu.be":
        return False
    if host.endswith("dailymotion.com"):
        return bool(re.match(r"^/(playlist/|[^/]+/?$)", path)) and not path.startswith("/video/")
    if host.endswith("vimeo.com"):
        return bool(re.match(r"^/(channels/|showcase/|album/|user\d+(/|$))", path))
    if host in ("ok.ru",):
        return "/video/c" in path or "/videos" in path   # listelenemez: list_videos açık mesaj verir
    if host.endswith("vkvideo.ru") or host.endswith("vk.com") or host == "vk.ru":
        return bool(re.match(r"^/(@[^/]+/?$|video/@[^/]+|playlist/)", path))
    if host.endswith("rutube.ru"):
        return bool(re.match(r"^/(channel/|u/|plst/|video/person/)", path))
    if host.endswith("twitch.tv"):
        return path.rstrip("/").endswith("/videos")
    return False


def is_playlist_url(url: str) -> bool:
    """Oynatma listesi linki mi (kanal değil)? Liste kullanıcının seçtiği videolardır: konu süzgeci uygulanmaz,
    işlenmemişlerin hepsi seçili gelir. YouTube'da ``list=`` (``/playlist`` ya da ``watch?v=..&list=``; otomatik
    "Mix" listeleri RD.. hariç), Dailymotion ``/playlist/``, Vimeo showcase/album."""
    p = urlsplit((url or "").strip())
    host = (p.hostname or "").lower().removeprefix("www.").removeprefix("m.").removeprefix("music.")
    q = parse_qs(p.query)
    if host.endswith("youtube.com") or host == "youtu.be":
        lid = (q.get("list") or [""])[0]
        return bool(lid) and not lid.startswith("RD")
    if host.endswith("dailymotion.com"):
        return p.path.startswith("/playlist/")
    if host.endswith("vimeo.com"):
        return bool(re.match(r"^/(showcase|album)/", p.path))
    return False


def _youtube_tabs(url: str) -> list[str]:
    """YouTube kanal kökü -> /videos + /streams (basın toplantıları çoğu zaman canlı yayın olarak yüklenir)."""
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    if "youtube.com" not in host:
        return [url]
    m = re.match(r"^(/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+))/?(featured)?/?$", p.path)
    if not m:
        return [url]
    base = f"https://www.youtube.com{m.group(1)}"
    return [base + "/videos", base + "/streams"]


def _entries(info: dict, depth: int = 0) -> list[dict]:
    out = []
    for e in info.get("entries") or []:
        if not e:
            continue
        if e.get("_type") == "playlist" or (e.get("entries") and depth < 2):
            out += _entries(e, depth + 1)
            continue
        url = e.get("url") or e.get("webpage_url")
        if not url:
            continue
        if not url.startswith("http") and e.get("ie_key") == "Youtube":
            url = f"https://www.youtube.com/watch?v={e.get('id') or url}"
        d = e.get("upload_date") or ""
        if not d and e.get("timestamp"):
            d = time.strftime("%Y%m%d", time.gmtime(e["timestamp"]))
        out.append({"url": url, "id": e.get("id"), "title": e.get("title") or url, "duration": e.get("duration"),
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else "",
                    "live": e.get("live_status") in ("is_live", "is_upcoming"),
                    "channel": e.get("channel") or e.get("uploader") or info.get("channel") or info.get("uploader")})
    return out


def list_videos(url: str, cfg, limit: int | None = None) -> dict:
    """{'title', 'channel', 'items': [...]} (yeniden eskiye). limit: sekme başına en fazla video."""
    from concurrent.futures import ThreadPoolExecutor

    from .fetch import ytdlp_info
    limit = int(limit or cfg.channel_max)
    host = (urlsplit(url).hostname or "").lower()
    if host.endswith("ok.ru") or host.endswith("mail.ru"):
        # yt-dlp'de ok.ru / mail.ru kanal-grup listeleyici yok (yalnız tek video): anlaşılır mesaj
        raise RuntimeError("ok.ru / mail.ru kanal ve grup listeleri desteklenmiyor; videoları tek tek "
                           "'Linkler' sekmesine ekleyin.")

    def fetch(tab):
        try:
            return ytdlp_info(tab, cfg.cookies_browser, flat=True, extra={"playlistend": limit}), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
    tabs = _youtube_tabs(url)
    with ThreadPoolExecutor(max_workers=len(tabs)) as ex:  # /videos ve /streams aynı anda
        results = list(ex.map(fetch, tabs))
    title, channel, items, seen = "", "", [], set()
    errors = [err for _, err in results if err]
    for info, _ in results:
        if info is None:
            continue
        title = title or info.get("title") or ""
        channel = channel or info.get("channel") or info.get("uploader") or ""
        for e in _entries(info):
            k = e.get("id") or e["url"]
            if k not in seen:
                seen.add(k)
                items.append(e)
    if not items and errors:
        raise RuntimeError("Liste alınamadı: " + " | ".join(errors))
    items.sort(key=lambda e: e.get("date") or "", reverse=True)  # tarihsizler sırasını korur (zaten yeniden eskiye)
    return {"title": re.sub(r"\s+-\s+(Videos|Live|Videolar)$", "", title), "channel": channel, "items": items}


# ------------------------------------------------------------------ kanalda basın toplantısı bulucu (deneme)
PRESSER_QUERIES = ("press conference", "pre-match", "post-match", "manager's preview", "presser")


def channel_root(url: str) -> str | None:
    """YouTube kanal linki -> kök (https://www.youtube.com/@Ad); kanal değilse None."""
    p = urlsplit((url or "").strip())
    host = (p.hostname or "").lower()
    if "youtube.com" not in host:
        return None
    m = re.match(r"^(/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+))(/.*)?$", p.path or "")
    return f"https://www.youtube.com{m.group(1)}" if m else None


def find_pressers(url: str, cfg, limit: int | None = None, queries=PRESSER_QUERIES) -> dict:
    """DENEME: kanalın tamamında basın toplantılarını bul. Normal listeleme kanalın son ``channel_max`` videosuna
    bakar; bu ise (1) kanal içi YouTube aramasını (her sorgu ayrı, ``/search?query=``) ve (2) adında basın toplantısı
    geçen oynatma listelerini tarar, sonuçları birleştirir. Konu süzgeci (``select``) sonra yine uygulanır.
    Döner: list_videos ile aynı biçim + 'sources' (nereden kaç video geldi)."""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote

    from .fetch import ytdlp_info
    base = channel_root(url)
    if not base:
        raise RuntimeError("Basın toplantısı bulucu yalnız YouTube kanal linkinde çalışır (ör. youtube.com/@Arsenal)")
    limit = int(limit or cfg.channel_max)

    def flat(u, n):
        try:
            return ytdlp_info(u, cfg.cookies_browser, flat=True, extra={"playlistend": n}), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
    targets = [(f"arama: {q}", f"{base}/search?query={quote(q)}") for q in queries]
    info, _ = flat(base + "/playlists", 300)
    kws = _prep(TOPICS["basin"])
    for e in (info or {}).get("entries") or []:
        title, purl = (e or {}).get("title") or "", (e or {}).get("url") or ""
        if purl and _hit(f" {_norm(title)} ", kws):
            targets.append((f"liste: {title}", purl))
    targets = targets[:16]  # aşırı istek olmasın (YouTube engelleyebilir)
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda t: flat(t[1], limit), targets))
    title, channel, items, seen, sources, errors = "", "", [], set(), {}, []
    for (label, _u), (inf, err) in zip(targets, results):
        if inf is None:
            errors.append(f"{label}: {err}")
            continue
        channel = channel or inf.get("channel") or inf.get("uploader") or ""
        n = 0
        for e in _entries(inf):
            k = e.get("id") or e["url"]
            if k not in seen:
                seen.add(k)
                items.append(e)
                n += 1
        sources[label] = n
    if not items and errors:
        raise RuntimeError("Basın toplantısı aranamadı: " + " | ".join(errors[:3]))
    items.sort(key=lambda e: e.get("date") or "", reverse=True)
    log.info("Basın toplantısı bulucu (%s): %d video; %s", base, len(items),
             ", ".join(f"{k}={v}" for k, v in sources.items()))
    return {"title": channel or base, "channel": channel, "items": items, "sources": sources}


def keywords_for(topic: str | None, custom: str | None = None) -> list[str]:
    return list(TOPICS.get((topic or "").lower(), ())) + _words(custom)


def _prep(words: list[str]) -> list[str]:
    return [f" {n}" for n in map(_norm, words) if n]


def _hit(title_norm: str, prepped: list[str]) -> bool:
    return any(k in title_norm for k in prepped)


def match_title(title: str, keywords: list[str], exclude: list[str] | None = None) -> bool:
    t = f" {_norm(title)} "
    if exclude and _hit(t, _prep(exclude)):
        return False
    return not keywords or _hit(t, _prep(keywords))


def select(items: list[dict], topic: str | None, custom: str | None = None, exclude: str | None = None,
           llm=None, llm_topic: str | None = None) -> list[dict]:
    """Her öğeye 'match' alanı ekler ve döner. llm verilirse anahtar kelime yerine yapay zekâ seçer."""
    kws = _prep(keywords_for(topic, custom))   # kelimeler bir kez sadeleştirilir (1000 başlıkta hızlı)
    exc = _prep(_words(exclude))
    for e in items:
        t = f" {_norm(e['title'])} "
        e["excluded"] = _hit(t, exc)
        e["match"] = not e["excluded"] and (not kws or _hit(t, kws))
    if llm is not None and llm_topic:
        try:
            from .llm import filter_titles
            flags = filter_titles(llm, [e["title"] for e in items], llm_topic)
            for e, f in zip(items, flags):
                e["match"] = bool(f) and not e["excluded"]
                e["ai"] = True
        except Exception as e:  # noqa: BLE001
            log.warning("Yapay zekâ ile seçilemedi, anahtar kelimeler kullanıldı: %s", e)
    return items


# ------------------------------------------------------------------ takip (kanal izleme)

class Follows:
    """Takip edilen kanallar: <out>/.stt/takip.json"""

    _lock = threading.Lock()

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "takip.json"

    def all(self) -> list[dict]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        return [r for r in rows if isinstance(r, dict) and r.get("id") and r.get("url")] if isinstance(rows, list) else []

    def _save(self, rows: list[dict]) -> None:
        atomic_write_text(self.path, json.dumps(rows, ensure_ascii=False, indent=1))

    def add(self, url: str, name: str, topic: str, custom: str = "", exclude: str = "", opts: dict | None = None,
            seen: list[str] | None = None) -> dict:
        with self._lock:
            rows = [r for r in self.all() if r["url"] != url]
            row = {"id": uuid.uuid4().hex[:8], "url": url, "name": name or url, "topic": topic, "custom": custom,
                   "exclude": exclude, "opts": opts or {}, "seen": list(seen or [])[-2000:], "last": 0, "added": time.time(),
                   "found": 0}
            rows.append(row)
            self._save(rows)
            return row

    def remove(self, fid: str) -> None:
        with self._lock:
            self._save([r for r in self.all() if r["id"] != fid])

    def update(self, fid: str, **kw) -> None:
        with self._lock:
            rows = self.all()
            for r in rows:
                if r["id"] == fid:
                    r.update(kw)
            self._save(rows)

    def due(self, interval_h: float) -> list[dict]:
        now = time.time()
        return [r for r in self.all() if now - (r.get("last") or 0) >= interval_h * 3600]


def check_follow(row: dict, cfg, llm=None, limit: int = 30) -> list[dict]:
    """Takip edilen kanaldaki yeni ve konuya uyan videolar (görülenler hariç)."""
    data = list_videos(row["url"], cfg, limit=limit)
    items = select(data["items"], row.get("topic"), row.get("custom"), row.get("exclude"),
                   llm=llm if (row.get("opts") or {}).get("ai") else None, llm_topic=row.get("custom") or row.get("topic"))
    seen = set(row.get("seen") or [])
    return [e for e in items if e["match"] and (e.get("id") or e["url"]) not in seen and not e.get("live")]


# ------------------------------------------------------------------ koleksiyon çıktısı

def add_to_collection(out_root: Path, name: str, doc: dict, paths: dict) -> Path:
    """<out>/Koleksiyonlar/<ad>/: liste.csv (tarih sırası) + tum_metinler.txt (bütün metinler tek dosyada).

    tum_metinler.txt'ye yeni metin sona eklenir (1000 videoluk kanalda her işte bütün metinleri baştan okumamak
    için); aynı video yeniden işlendiyse dosya baştan kurulur."""
    from .match import safe_filename
    folder = Path(out_root) / "Koleksiyonlar" / safe_filename(name, 80)
    folder.mkdir(parents=True, exist_ok=True)
    ds = doc.get("dataset") or {}
    row = {"date": ds.get("presser_date") or (doc.get("match") or {}).get("date") or "", "title": doc.get("title"),
           "url": doc.get("webpage_url") or doc.get("source"), "language": doc.get("language"),
           "duration_s": doc.get("duration_s"), "txt": str(paths["txt"]), "json": str(paths["json"]),
           "key": doc.get("key")}
    old = read_jsonl(folder / "liste.jsonl")
    again = any(r.get("key") == row["key"] for r in old)
    rows = sorted([r for r in old if r.get("key") != row["key"]] + [row],
                  key=lambda r: (r.get("date") or "", r.get("title") or ""))
    write_jsonl_csv(folder / "liste.jsonl", folder / "liste.csv", rows, list(row))
    allf = folder / "tum_metinler.txt"
    if again or not allf.exists():
        atomic_write_text(allf, "\n".join(_block(r["txt"]) for r in rows))
    else:
        with open(allf, "a", encoding="utf-8") as f:
            f.write("\n" + _block(row["txt"]))
    return folder


def _block(txt: str) -> str:
    try:
        body = Path(txt).read_text(encoding="utf-8")
    except OSError:
        body = ""
    return "=" * 80 + "\n" + body.strip() + "\n"
