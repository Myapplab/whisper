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


def collection_name(channel: str, topic: str | None, custom: str | None = None, search: str | None = None) -> str:
    """Koleksiyon klasörü adı (GUI ve komut satırı aynı adı kullansın): '<kanal> - <konu>'.
    Aranan varsa '<aranan> - <konu>' (ör. 'Arsenal - basın toplantıları'): farklı kanallardan bulunan aynı takımın
    videoları tek koleksiyonda toplanır."""
    label = TOPIC_LABELS.get((topic or "").lower()) or (custom or "").strip()
    terms = _words(search)
    if terms:
        head = ", ".join(_team_canon(t) or t for t in terms)
        return f"{head} - {label}" if label and (topic or "").lower() != "hepsi" else head
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
    if host.endswith("vkvideo.ru"):
        return bool(re.match(r"^/(@[^/]+/?$|playlist/)", path))
    if host.endswith("vk.com") or host == "vk.ru":  # vk.com/@ad bir makale sayfası; video kanalı /video/@ad
        return bool(re.match(r"^/(video/@[^/]+|video/playlist/|playlist/)", path))
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


# ------------------------------------------------------------------ kanalın tamamında arama (takım / kişi / konu)
PRESSER_QUERIES = ("press conference", "pre-match", "post-match", "manager's preview", "presser")
TOPIC_QUERIES = {"basin": PRESSER_QUERIES, "roportaj": ("interview", "reaction"), "hepsi": ()}
# aranan kelimeye eklenen sorgular ("Arsenal press conference", "Mikel Arteta pre-match")
TERM_QUERIES = {"basin": ("press conference", "pre-match", "post-match", "presser"), "roportaj": ("interview",),
                "hepsi": ("",)}
MAX_TARGETS = 24  # bir taramada en çok bu kadar arama / liste (aşırı istek olmasın: YouTube engelleyebilir)


def channel_root(url: str) -> str | None:
    """YouTube kanal linki -> kök (https://www.youtube.com/@Ad); kanal değilse None."""
    p = urlsplit((url or "").strip())
    host = (p.hostname or "").lower()
    if "youtube.com" not in host:
        return None
    m = re.match(r"^(/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+))(/.*)?$", p.path or "")
    return f"https://www.youtube.com{m.group(1)}" if m else None


def _term_names(term: str) -> list[str]:
    """Kanal içi aramada kullanılacak adlar: yazılan + (takımsa) kanonik ad ve bu sezonki hocası."""
    from .match import PL_MANAGERS
    names = [term.strip()]
    canon = _team_canon(term)
    if canon:
        names.append(canon)
        names += [m for m, t in PL_MANAGERS.items() if t == canon]
    out: list[str] = []
    for n in names:
        if n and _norm(n) not in {_norm(x) for x in out}:
            out.append(n)
    return out


def find_in_channel(url: str, cfg, limit: int | None = None, search: str | None = None, topic: str | None = "basin") -> dict:
    """Kanalın TAMAMINDA ara (normal listeleme yalnız son ``channel_max`` videoya bakar). Taranan: (1) kanal içi YouTube
    araması (``/search?query=``): aranan her kelime (takımsa hocası da) + konu sözcükleri ("Arsenal press conference"),
    aranan yoksa yalnız konu sözcükleri; (2) adı konuya ya da arananlara uyan oynatma listeleri; (3) son videolar ve canlı
    yayınlar (``/videos`` + ``/streams``). Sonuçlar birleştirilir (tekrar yok); hangilerinin uyduğuna ``select`` karar
    verir. Döner: list_videos ile aynı biçim + 'sources' (nereden kaç yeni video geldi)."""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote

    from .fetch import ytdlp_info
    base = channel_root(url)
    if not base:
        raise RuntimeError("Kanalın tamamında arama yalnız YouTube kanal linkinde çalışır (ör. youtube.com/@Arsenal)")
    limit = int(limit or cfg.channel_max)
    topic = (topic or "basin").lower()
    if topic not in TOPIC_QUERIES:
        topic = "hepsi"
    terms = _words(search)

    def flat(u, n):
        try:
            return ytdlp_info(u, cfg.cookies_browser, flat=True, extra={"playlistend": n}), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
    pinfo, _ = flat(base + "/playlists", 300)
    chan = (pinfo or {}).get("channel") or (pinfo or {}).get("uploader") or ""
    queries: list[str] = []
    for term in terms:
        for name in _term_names(term):
            queries += [f"{name} {q}".strip() for q in TERM_QUERIES[topic]]
    from .match import club_of_channel
    own = bool(chan) and any(_team_canon(t) and club_of_channel(chan) == _team_canon(t) for t in terms)
    if not terms or own:  # kulübün kendi kanalı: başlıkta takım adı yazmayabilir ("MANAGER'S PREVIEW: Pep on Wolves")
        queries += list(TOPIC_QUERIES[topic])
    seen_q: set[str] = set()
    targets = []
    for q in queries:
        if _norm(q) not in seen_q:
            seen_q.add(_norm(q))
            targets.append((f"arama: {q}", f"{base}/search?query={quote(q)}"))
    kws = _prep(list(TOPICS.get(topic, ())) + terms)
    lists = []
    for e in (pinfo or {}).get("entries") or []:
        title, purl = (e or {}).get("title") or "", (e or {}).get("url") or ""
        if purl and kws and _hit(f" {_norm(title)} ", kws):
            lists.append((f"liste: {title}", purl))
    tabs = [(f"son videolar ({tab.rsplit('/', 1)[-1]})", tab) for tab in _youtube_tabs(base)]
    targets = (tabs + targets[:MAX_TARGETS - len(tabs) - min(len(lists), 6)] + lists)[:MAX_TARGETS]
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda t: flat(t[1], limit), targets))
    channel, items, seen, sources, errors = chan, [], set(), {}, []
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
        raise RuntimeError("Kanalda aranamadı: " + " | ".join(errors[:3]))
    items.sort(key=lambda e: e.get("date") or "", reverse=True)
    log.info("Kanalda arama (%s; aranan %s; konu %s): %d video; %s", base, ", ".join(terms) or "-", topic, len(items),
             ", ".join(f"{k}={v}" for k, v in sources.items()))
    return {"title": channel or base, "channel": channel, "items": items, "sources": sources}


def find_pressers(url: str, cfg, limit: int | None = None) -> dict:
    """Kanalın tamamında basın toplantıları (aranan kelime yok)."""
    return find_in_channel(url, cfg, limit, None, "basin")


# ------------------------------------------------------------------ aranan: takım / kişi / konu
# Takım arandığında yalnız A takımı (erkek): kadın takımı, altyaş, akademi, efsaneler, e-spor, basketbol videoları alınmaz
MEN_ONLY_EXCLUDE = re.compile(
    r"(?<![a-z0-9])(?:wom[ae]n\w*|wsl|wfc|ladies|girls|lionesses|kadin\w*|bayan\w*|femenin\w*|feminin\w*|frauen\w*|"
    r"femminile|u ?(?:1[4-9]|2[0-3])s?|under ?(?:1[4-9]|2[0-3])|academy|akademi\w*|youth|pl2|reserves?|legends?|"
    r"futsal|e ?sports?|efootball|basketball|basketbol|nba)(?![a-z0-9])")


def _team_canon(term: str | None) -> str | None:
    """Yazılan bir takım adı mı (takma adlar dahil: "Man City", "Spurs", "Gunners")? Kanonik adı döner."""
    from .match import _ALIAS_MAP, _alias_norm
    return _ALIAS_MAP.get(_alias_norm(term or "")) if (term or "").strip() else None


def team_video(title: str, channel: str, team: str, date: str | None = None, own: bool = False) -> bool:
    """Video ``team`` ile ilgili mi? own=True (basın toplantısı): toplantı o takımın mı (konuşan o takımın hocası /
    oyuncusu; "Maresca previews Arsenal" Arsenal'in değil). own=False: başlıkta takım ya da bu sezonki hocası geçiyor.
    Takımın kendi kanalındaki (kanal adı yalnız kulüp adı: "Arsenal", "Liverpool FC") her video o takımındır."""
    from .match import _joined, _team_hits, club_of_channel, find_manager, find_teams, manager_team, parse_match
    club = club_of_channel(channel or "")
    if club == team:
        return True
    teams = find_teams(title)
    mgr = find_manager(title, teams)
    mt = manager_team(mgr, date) if mgr else None
    if not own:
        return team in teams or mt == team
    if club:
        return False  # başka kulübün kanalı: toplantı o kulübün
    d8 = (date or "").replace("-", "")
    if parse_match({"title": title, "channel": "", "upload_date": d8})["team"] == team:
        return True
    if mt is None and team in teams[:2]:  # "Arsenal vs Chelsea | Post-match press conference": hangisi belli değil
        t, hits = _team_hits(title)
        return _joined(t, hits)
    return False


def _presser_like(title: str, channel: str) -> bool:
    """Başlıkta basın toplantısı sözcüğü yoksa: bilinen hoca adı ya da (kulüp kanalında) tanınan rakip; özet, antrenman,
    canlı yayın vb. hariç (match.looks_like_presser)."""
    from .match import _NOT_PRESSER_RE, _norm as mnorm, looks_like_presser, parse_match
    info = {"title": title, "channel": channel}
    return looks_like_presser(info, parse_match(info)) and not _NOT_PRESSER_RE.search(mnorm(title))


def term_match(e: dict, term: str, topic: str | None, channel: str = "") -> bool:
    """Tek aranan kelime bu videoya uyuyor mu? Takımsa ``team_video`` (+ yalnız A takımı); değilse başlıkta geçiyor mu
    (kelime başı: "Arteta" -> "Arteta's" de)."""
    title = e.get("title") or ""
    canon = _team_canon(term)
    if canon:
        if MEN_ONLY_EXCLUDE.search(_norm(title)):
            return False
        return team_video(title, e.get("channel") or channel, canon, e.get("date") or None,
                          own=(topic or "").lower() == "basin")
    return _hit(f" {_norm(title)} ", _prep([term]))


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
           llm=None, llm_topic: str | None = None, search: str | None = None, channel: str = "") -> list[dict]:
    """Her öğeye 'match' alanı ekler ve döner. llm verilirse anahtar kelime yerine yapay zekâ seçer.
    search: aranan takım / kişi / konu (virgülle birden çok; biri uyması yeter). Verilirse video hem konuya hem
    arananlardan birine uymalı ("Arsenal" + Basın toplantısı = Arsenal'in basın toplantıları)."""
    kws = _prep(keywords_for(topic, custom))   # kelimeler bir kez sadeleştirilir (1000 başlıkta hızlı)
    exc = _prep(_words(exclude))
    terms = _words(search)
    team_terms = any(_team_canon(t) for t in terms)
    basin = (topic or "").lower() == "basin"
    for e in items:
        t = f" {_norm(e['title'])} "
        e["excluded"] = _hit(t, exc)
        ok = not e["excluded"] and (not kws or _hit(t, kws))
        if not ok and not e["excluded"] and basin and team_terms:
            # takımın kanalı / tarafsız kanal: "MANAGER'S PREVIEW" yok ama "Arteta on Saka's fitness", "X previews Chelsea"
            ok = _presser_like(e["title"], e.get("channel") or channel)
        if ok and terms:
            ok = any(term_match(e, term, topic, channel) for term in terms)
        e["match"] = ok
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
            seen: list[str] | None = None, search: str = "") -> dict:
        with self._lock:
            rows = [r for r in self.all() if r["url"] != url]
            row = {"id": uuid.uuid4().hex[:8], "url": url, "name": name or url, "topic": topic, "custom": custom,
                   "exclude": exclude, "search": search, "opts": opts or {}, "seen": list(seen or [])[-2000:], "last": 0, "added": time.time(),
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
                   llm=llm if (row.get("opts") or {}).get("ai") else None,
                   llm_topic=" ".join(x for x in (row.get("search"), row.get("custom") or row.get("topic")) if x),
                   search=row.get("search"), channel=data.get("channel") or "")
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
