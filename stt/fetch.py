"""Ses indirme (yt-dlp, çerez, m3u8/mp4 yedek yolu) ve ffmpeg ile 16 kHz mono WAV'a çevirme."""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import wave
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .util import log, which

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")


class SkipItem(Exception):
    """Bilerek atlanan öğe (DRM, ses yok...)."""


class DRMProtected(SkipItem):
    pass


class FetchError(Exception):
    pass


# ---------------------------------------------------------------- ffmpeg

def ffmpeg_bin() -> str:
    exe = which("ffmpeg")
    if not exe:
        raise FetchError("ffmpeg bulunamadı (PATH). 'winget install Gyan.FFmpeg' ile kurun.")
    return exe


def to_wav16k(src: str, dst: Path, headers: dict | None = None) -> Path:
    """Herhangi bir medya dosyası/URL -> 16 kHz mono PCM WAV (atomik)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    if headers and src.startswith("http"):
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
    cmd += ["-i", src, "-vn", "-sn", "-dn", "-map", "0:a:0?", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        err = (r.stderr or "").strip()[-800:]
        if "does not contain any stream" in err or "matches no streams" in err or not err:
            raise SkipItem(f"Ses izi yok: {src}")
        raise FetchError(f"ffmpeg dönüştürme hatası: {err}")
    os.replace(tmp, dst)
    return dst


def wav_ok(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == 16000 and w.getnchannels() == 1 and w.getnframes() > 1600
    except Exception:  # noqa: BLE001
        return False


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------------------------------------------------------- DRM

_DRM_MARKERS = ("SAMPLE-AES", "com.widevine", "com.apple.streamingkeydelivery", "com.microsoft.playready",
                'KEYFORMAT="urn:uuid', "urn:mpeg:dash:mp4protection", "cenc:default_KID", "<ContentProtection")


def manifest_has_drm(text: str) -> bool:
    return any(m.lower() in text.lower() for m in _DRM_MARKERS)


def is_drm_error(msg: str) -> bool:
    m = msg.lower()
    return "drm" in m and ("protect" in m or "drm-protected" in m or "widevine" in m)


_LOGIN_HINTS = ("sign in", "login", "log in", "cookies", "private", "age", "confirm you", "not a bot",
                "authentication", "rate-limit", "rate limit", "registered users", "members-only",
                "http error 401", "http error 403", "requested content is not available",
                "unable to extract", "empty media response", "this content isn't available")


LOGIN_FIRST = ("instagram.com", "facebook.com", "fb.watch", "threads.net", "patreon.com")


def needs_cookies(msg: str) -> bool:
    m = msg.lower()
    return any(h in m for h in _LOGIN_HINTS)


# ---------------------------------------------------------------- yt-dlp

class _QuietLogger:
    """yt-dlp mesajlarını kendi günlüğümüze (debug) yönlendir; hataları biz özetleriz."""

    def debug(self, msg):
        log.debug("yt-dlp: %s", msg)

    def info(self, msg):
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg):
        log.debug("yt-dlp uyarı: %s", msg)

    def error(self, msg):
        log.debug("yt-dlp hata: %s", msg)


def _ytdlp_opts(workdir: Path, extra: dict) -> dict:
    opts = {
        "format": "bestaudio/best[height<=480]/best",
        "outtmpl": str(workdir / "kaynak.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 30,
        "http_headers": {"User-Agent": UA},
        "allow_unplayable_formats": False,
    }
    rts = {n: {} for n in ("deno", "node", "bun", "quickjs") if which("qjs" if n == "quickjs" else n)}
    opts["js_runtimes"] = rts or {"deno": {}}
    opts.update(extra)
    return opts


def _find_download(workdir: Path) -> Path | None:
    cands = [p for p in workdir.glob("kaynak.*")
             if p.suffix not in (".part", ".ytdl", ".json") and not p.name.endswith(".part.wav")]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def ytdlp_download(url: str, workdir: Path, cookies_browser: str | None, cookies_file: str | None):
    """(dosya, info) döner. DRM ise DRMProtected; hiçbiri olmazsa FetchError."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    attempts: list[tuple[str, dict]] = [("çerezsiz", {})]
    if cookies_browser:
        attempts.append((f"--cookies-from-browser {cookies_browser}", {"cookiesfrombrowser": (cookies_browser,)}))
    if cookies_file and Path(cookies_file).is_file():
        attempts.append((f"--cookies {cookies_file}", {"cookiefile": cookies_file}))

    host = (urlsplit(url).hostname or "").lower()
    if any(host == d or host.endswith("." + d) for d in LOGIN_FIRST) and len(attempts) > 1:
        attempts = attempts[1:] + attempts[:1]  # giriş isteyen sitelerde önce çerezle dene
    last = ""
    for label, extra in attempts:
        try:
            with yt_dlp.YoutubeDL(_ytdlp_opts(workdir, extra)) as ydl:
                info = ydl.extract_info(url, download=True)
                if info and info.get("entries"):
                    entries = [e for e in info["entries"] if e]
                    if not entries:
                        raise FetchError("Oynatma listesi boş")
                    info = entries[0]
                    log.warning("Link bir oynatma listesi; yalnızca ilk video alındı. Hepsi için linkleri .txt listesine koyun.")
            f = _find_download(workdir)
            if not f:
                raise FetchError("yt-dlp dosya üretmedi")
            log.info("İndirildi (%s): %s", label, f.name)
            return f, info
        except DownloadError as e:
            last = str(e)
            if is_drm_error(last):
                raise DRMProtected(f"DRM korumalı: {url}") from None
            if "unsupported url" in last.lower():
                break  # çerez işe yaramaz; doğrudan yedek yola geç
            log.info("yt-dlp başarısız (%s): %s", label, last.splitlines()[-1][:200])
            if not needs_cookies(last) and label == "çerezsiz" and "http error 404" in last.lower():
                break
        except FetchError:
            raise
        except Exception as e:  # noqa: BLE001  (çerez okuma hataları vb.)
            last = f"{type(e).__name__}: {e}"
            hint = ""
            if "cookie" in last.lower() or "dpapi" in last.lower():
                hint = (" | İpucu: Chrome açıkken/uygulamaya bağlı şifreleme yüzünden çerezler okunamayabilir. "
                        "Chrome'u kapatın, ya da --cookies-browser firefox/edge, ya da --cookies cookies.txt kullanın.")
            log.info("yt-dlp hata (%s): %s%s", label, last[:300], hint)
    raise FetchError(last or "yt-dlp başarısız")


# ---------------------------------------------------------------- sayfadan m3u8/mp4 çekme

_MEDIA_URL_RE = re.compile(
    r"""https?:(?:\\?/){2}[^\s"'<>()\\]+?\.(?:m3u8|mp4|m4a|mp3|aac|webm|mpd)(?:\?[^\s"'<>\\]*)?""", re.I)
_ATTR_RE = re.compile(
    r"""<(?:video|source|audio)[^>]+?src=["']([^"']+)["']|<meta[^>]+?(?:property|name)=["'](?:og:video(?::secure_url|:url)?|twitter:player:stream)["'][^>]+?content=["']([^"']+)["']""",
    re.I)


def find_media_urls(page_html: str, base_url: str) -> list[str]:
    found: list[str] = []
    text = page_html.replace("\\u002F", "/").replace("\\u0026", "&").replace("\\/", "/")
    for m in _ATTR_RE.finditer(text):
        u = m.group(1) or m.group(2)
        if u and not u.startswith(("blob:", "data:")):
            found.append(urljoin(base_url, html.unescape(u)))
    for m in _MEDIA_URL_RE.finditer(text):
        found.append(html.unescape(m.group(0)))
    seen, out = set(), []
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)

    def rank(u: str) -> int:
        lu = u.lower()
        if ".m3u8" in lu:
            return 0 if "master" in lu or "index" in lu or "playlist" in lu else 1
        if ".mp4" in lu or ".m4a" in lu:
            return 2
        if ".mpd" in lu:
            return 4
        return 3
    return sorted(out, key=rank)


def _check_manifest_drm(session, url: str, headers: dict) -> None:
    lu = url.lower()
    if ".m3u8" not in lu and ".mpd" not in lu:
        return
    try:
        r = session.get(url, headers=headers, timeout=20)
        text = r.text
        if manifest_has_drm(text):
            raise DRMProtected(f"DRM korumalı akış: {url}")
        if ".m3u8" in lu and "#EXT-X-STREAM-INF" in text:  # ana liste: ilk alt listeye bak
            for line in text.splitlines():
                if line and not line.startswith("#"):
                    sub = session.get(urljoin(url, line.strip()), headers=headers, timeout=20).text
                    if manifest_has_drm(sub):
                        raise DRMProtected(f"DRM korumalı akış: {url}")
                    break
    except DRMProtected:
        raise
    except Exception as e:  # noqa: BLE001
        log.debug("manifest okunamadı %s: %s", url, e)


def scrape_and_convert(page_url: str, dst: Path) -> tuple[Path, dict]:
    import requests

    headers = {"User-Agent": UA, "Referer": page_url, "Accept-Language": "en-US,en;q=0.9,tr;q=0.8"}
    s = requests.Session()
    r = s.get(page_url, headers=headers, timeout=30)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "charset" not in ctype.lower():
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", r.content[:4000], re.I)
        r.encoding = m.group(1).decode() if m else "utf-8"
    if ctype.startswith(("video/", "audio/")) or "mpegurl" in ctype:
        cands = [r.url]
    else:
        cands = find_media_urls(r.text, r.url)
    if not cands:
        raise FetchError("Sayfada m3u8/mp4 linki bulunamadı")
    title_m = re.search(r"<title[^>]*>(.*?)</title>", r.text if "text" in ctype else "", re.I | re.S)
    title = html.unescape(title_m.group(1)).strip() if title_m else ""
    errors = []
    drm_seen = False
    for u in cands[:6]:
        try:
            _check_manifest_drm(s, u, headers)
            log.info("Yedek yol: ffmpeg ile çekiliyor: %s", u[:150])
            to_wav16k(u, dst, headers={"User-Agent": UA, "Referer": page_url})
            return dst, {"title": title, "webpage_url": page_url, "media_url": u, "extractor": "stt-scrape"}
        except DRMProtected:
            drm_seen = True
        except SkipItem as e:
            errors.append(str(e))
        except FetchError as e:
            errors.append(str(e)[:200])
    if drm_seen and not errors:
        raise DRMProtected(f"DRM korumalı: {page_url}")
    raise FetchError("Yedek yol başarısız: " + " | ".join(errors[-3:]))


# ---------------------------------------------------------------- ana giriş

def prepare_audio(item, workdir: Path, cfg) -> tuple[Path, dict]:
    """Öğeyi workdir/ses.wav olarak hazırlar. Var ve sağlamsa yeniden indirmez (sürdürme)."""
    workdir.mkdir(parents=True, exist_ok=True)
    wav = workdir / "ses.wav"
    info_path = workdir / "kaynak.json"
    if wav.exists() and wav_ok(wav):
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        log.info("Önceden hazırlanmış ses kullanılıyor: %s", item.title[:80])
        return wav, info

    if item.kind == "file":
        info = {"title": Path(item.source).stem, "source_path": item.source,
                "timestamp": int(Path(item.source).stat().st_mtime)}
        to_wav16k(item.source, wav)
    else:
        info = {}
        try:
            src, raw = ytdlp_download(item.source, workdir, cfg.cookies_browser, cfg.cookies_file)
            info = {k: raw.get(k) for k in ("id", "title", "extractor", "extractor_key", "webpage_url",
                                            "uploader", "channel", "upload_date", "timestamp", "release_timestamp", "duration",
                                            "description")
                    if raw.get(k) is not None}
            if info.get("description"):
                info["description"] = str(info["description"])[:1000]
            to_wav16k(str(src), wav)
            try:
                src.unlink()
            except OSError:
                pass
        except DRMProtected:
            raise
        except (FetchError, Exception) as e:  # noqa: BLE001
            if isinstance(e, SkipItem):
                raise
            log.info("yt-dlp olmadı (%s) -> sayfadan m3u8/mp4 aranıyor", str(e).splitlines()[-1][:160] if str(e) else e)
            try:
                _, info = scrape_and_convert(item.source, wav)
            except SkipItem:
                raise
            except Exception as e2:  # noqa: BLE001
                raise FetchError(f"İndirilemedi. yt-dlp: {str(e)[:300]} | yedek: {e2}") from None
    info["duration_s"] = wav_duration(wav)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return wav, info
