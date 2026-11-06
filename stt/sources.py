"""Girdileri (link, dosya, klasör, .txt link listesi) işlenecek öğelere çevirir."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .util import MEDIA_EXTS, PARTIAL_EXTS, file_hash, log, text_hash

URL_RE = re.compile(r"^(https?://|www\.|(m\.|music\.)?(youtube\.com|youtu\.be|x\.com|twitter\.com)/)", re.I)

_DROP_PARAMS = {
    "si", "feature", "fbclid", "gclid", "igsh", "igshid", "mibextid", "ref", "ref_src", "ref_url",
    "s", "t", "pp", "ab_channel", "share", "_r", "is_from_webapp", "sender_device", "web_id",
}


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s.strip()))


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I) and URL_RE.match(url):  # "www..." / "youtube.com/..." (şemasız)
        url = "https://" + url
    try:
        p = urlsplit(url)
        p.port  # noqa: B018 - geçersiz port / IPv6 burada hata verir
    except ValueError:
        return url
    host = (p.hostname or "").lower()
    if host.startswith("www.") or host.startswith("m."):
        host = host.split(".", 1)[1]
    path = p.path or "/"
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if k.lower() not in _DROP_PARAMS and not k.lower().startswith("utm_")]
    if host == "youtu.be":
        host, q, path = "youtube.com", [("v", path.strip("/"))], "/watch"
    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        host = "youtube.com"
        m = re.match(r"^/(shorts|live|embed|v|e)/([\w-]+)", path)
        if m:
            path, q = "/watch", [("v", m.group(2))]
        if path == "/watch":
            q = [(k, v) for k, v in q if k == "v"]
    if host in ("twitter.com", "mobile.twitter.com", "fxtwitter.com", "vxtwitter.com"):
        host = "x.com"
    path = path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, urlencode(sorted(q)), ""))


@dataclass
class Item:
    kind: str                 # "file" | "url"
    source: str               # dosya yolu ya da link
    title: str = ""
    key: str = ""             # sha256 (dosya içeriği ya da normalize link)
    extra: dict = field(default_factory=dict)

    def compute_key(self) -> str:
        if not self.key:
            if self.kind == "file":
                self.key = "f_" + file_hash(Path(self.source))
            else:
                self.key = "u_" + text_hash(normalize_url(self.source))
        return self.key


def read_link_list(path: Path) -> list[tuple[str, dict]]:
    """Satır biçimi:  <link ya da dosya>  [GW TAKIM]   (# ile başlayan satırlar yorum)."""
    out = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        extra: dict = {}
        urls = re.findall(r"https?://[^\s<>\"']+", line)
        if len(urls) > 1 or (urls and not is_url(line)):
            # bir satırda birden çok link ya da link metin içinde (panodan yapıştırma)
            out.extend((u.rstrip(").,;"), {}) for u in urls)
            continue
        if is_url(line):
            parts = line.split()
            target, rest = parts[0], parts[1:]
        else:
            m = re.match(r"^(.*?)\s+(?:GW)?(\d{1,2})\s+([A-Za-z]{2,4})$", line, re.I)
            target, rest = (m.group(1), [m.group(2), m.group(3)]) if m else (line, [])
        if len(rest) >= 2 and rest[0].upper().lstrip("GW").isdigit() and rest[1].isalpha():
            extra["fpl"] = (int(rest[0].upper().lstrip("GW")), rest[1].upper())
        out.append((target.strip().strip('"'), extra))
    return out


def iter_media_files(folder: Path):
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() in PARTIAL_EXTS:
            continue
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        if p.suffix.lower() in MEDIA_EXTS:
            yield p


def expand_inputs(inputs: list[str], _depth: int = 0) -> list[Item]:
    items: list[Item] = []
    for raw in inputs:
        s = raw.strip().strip('"')
        if not s:
            continue
        if is_url(s):
            if not re.match(r"^https?://", s, re.I):
                s = "https://" + s
            items.append(Item("url", s, title=s))
            continue
        p = Path(s).expanduser()
        if p.is_dir():
            files = list(iter_media_files(p))
            log.info("Klasör %s: %d medya dosyası", p, len(files))
            items.extend(Item("file", str(f.resolve()), title=f.stem) for f in files)
        elif p.is_file() and p.suffix.lower() == ".txt":
            if _depth > 2:
                log.warning("İç içe liste çok derin, atlandı: %s", p)
                continue
            entries = read_link_list(p)
            log.info("Liste %s: %d satır", p, len(entries))
            for target, extra in entries:
                # listedeki göreli yollar listenin klasörüne göre
                t = target if is_url(target) or Path(target).is_absolute() else str(p.parent / target)
                for it in expand_inputs([t], _depth + 1):
                    it.extra.update(extra)
                    items.append(it)
        elif p.is_file():
            if p.suffix.lower() not in MEDIA_EXTS:
                log.warning("Bilinmeyen uzantı, yine de deneniyor: %s", p)
            items.append(Item("file", str(p.resolve()), title=p.stem))
        else:
            log.error("Bulunamadı (dosya/klasör/link değil): %s", s)
    # aynı girdiyi iki kez ekleme
    seen, uniq = set(), []
    for it in items:
        k = (it.kind, normalize_url(it.source) if it.kind == "url" else it.source)
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    return uniq
