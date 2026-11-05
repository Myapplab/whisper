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

from .util import log

# Konu hazırları: başlıkta (aksan/büyük harf farketmez) bunlardan biri geçerse seçilir
TOPICS = {
    "basin": ("press conference", "presser", "pre-match", "prematch", "post-match", "postmatch", "news conference",
              "media conference", "basın toplantısı", "basin toplantisi", "maç öncesi", "maç sonrası",
              "mətbuat konfransı", "rueda de prensa", "conferencia de prensa", "conférence de presse",
              "pressekonferenz", "conferenza stampa", "conferência de imprensa", "persconferentie",
              "пресс-конференция", "press talk", "pre-game", "post-game"),
    "roportaj": ("interview", "röportaj", "reaction", "speaks", "reacts", "müsahibə", "entrevista", "intervista"),
    "hepsi": (),
}


_TR = str.maketrans("ıİşŞğĞçÇöÖüÜəƏ", "iIsSgGcCoOuUeE")


def _norm(s: str) -> str:
    """Küçük harf, Türkçe/aksan sadeleştirme; Kiril/Arap harfleri KORUNUR; tire/çizgi = boşluk."""
    import unicodedata
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
        return bool(re.match(r"^/(channels|showcase|user\d+|album)/", path))
    if host in ("ok.ru",):
        return "/video/c" in path or "/videos" in path
    if host.endswith("twitch.tv"):
        return path.rstrip("/").endswith("/videos")
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
    from .fetch import ytdlp_info
    limit = int(limit or getattr(cfg, "channel_max", 200) or 200)
    title, channel, items, seen = "", "", [], set()
    errors = []
    for tab in _youtube_tabs(url):
        try:
            info = ytdlp_info(tab, cfg.cookies_browser, flat=True, extra={"playlistend": limit})
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:200])
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


def keywords_for(topic: str | None, custom: str | None = None) -> list[str]:
    words = list(TOPICS.get((topic or "").lower(), ()))
    if custom:
        words += [w.strip() for w in re.split(r"[,;\n]", custom) if w.strip()]
    return words


def match_title(title: str, keywords: list[str], exclude: list[str] | None = None) -> bool:
    t = f" {_norm(title)} "
    if exclude and any(_norm(x) and f" {_norm(x)}" in t for x in exclude):
        return False
    if not keywords:
        return True
    return any(_norm(k) and f" {_norm(k)}" in t for k in keywords)


def select(items: list[dict], topic: str | None, custom: str | None = None, exclude: str | None = None,
           llm=None, llm_topic: str | None = None) -> list[dict]:
    """Her öğeye 'match' alanı ekler ve döner. llm verilirse anahtar kelime yerine/yanında yapay zekâ seçer."""
    kws = keywords_for(topic, custom)
    exc = [w.strip() for w in re.split(r"[,;\n]", exclude or "") if w.strip()]
    for e in items:
        e["match"] = match_title(e["title"], kws, exc)
    if llm is not None and llm_topic:
        try:
            from .llm import filter_titles
            flags = filter_titles(llm, [e["title"] for e in items], llm_topic)
            for e, f in zip(items, flags):
                e["match"] = bool(f) and not (exc and not match_title(e["title"], [], exc))
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
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []

    def _save(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

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
    """<out>/Koleksiyonlar/<ad>/: liste.csv + tum_metinler.txt (tarih sırasıyla bütün metinler tek dosyada)."""
    import csv
    from .match import safe_filename
    from .outputs import to_txt
    folder = Path(out_root) / "Koleksiyonlar" / safe_filename(name, 80)
    folder.mkdir(parents=True, exist_ok=True)
    jl = folder / "liste.jsonl"
    rows = []
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    ds = doc.get("dataset") or {}
    date = ds.get("presser_date") or (doc.get("match") or {}).get("date") or ""
    row = {"date": date, "title": doc.get("title"), "url": doc.get("webpage_url") or doc.get("source"),
           "language": doc.get("language"), "duration_s": doc.get("duration_s"), "txt": str(paths["txt"]),
           "json": str(paths["json"]), "key": doc.get("key")}
    rows = [r for r in rows if r.get("key") != row["key"]] + [row]
    rows.sort(key=lambda r: (r.get("date") or "", r.get("title") or ""))
    jl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with open(folder / "liste.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)
    parts = []
    for r in rows:
        try:
            body = Path(r["txt"]).read_text(encoding="utf-8")
        except OSError:
            jp = Path(r["json"])
            body = to_txt(json.loads(jp.read_text(encoding="utf-8"))) if jp.exists() else ""
        parts.append("=" * 80 + "\n" + body.strip() + "\n")
    (folder / "tum_metinler.txt").write_text("\n".join(parts), encoding="utf-8")
    return folder
