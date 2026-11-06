"""Ses indirme (yt-dlp, çerez, m3u8/mp4 yedek yolu) ve ffmpeg ile 16 kHz mono WAV'a çevirme."""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .util import console_python, log, popen_kw, which

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")


def user_agent() -> str:
    """ffmpeg / requests yolları için güncel tarayıcı kimliği (yt-dlp'nin kendi listesi, sürümle güncellenir).
    yt-dlp'ye ve curl_cffi'ye kimlik VERİLMEZ: tarayıcı taklidi kendi (uyumlu) kimliğini gönderir; eskiden sabit
    "Chrome/130" taklidin Chrome 15x parmak iziyle çelişiyordu (bot korumaları bu uyuşmazlığa bakar)."""
    try:
        from yt_dlp.utils.networking import random_user_agent
        return random_user_agent()
    except Exception:  # noqa: BLE001
        return UA


class SkipItem(Exception):
    """Bilerek atlanan öğe (DRM, ses yok...)."""


class DRMProtected(SkipItem):
    pass


class Cancelled(Exception):
    """Kullanıcı bu işi durdurdu."""


CANCELLED: set[str] = set()  # durdurulan işlerin çalışma klasörleri (indirme ilerledikçe kontrol edilir)


def check_cancel(workdir) -> None:
    if CANCELLED and str(workdir) in CANCELLED:
        raise Cancelled("Durduruldu.")


class FetchError(Exception):
    """İndirilemedi. transient=True: geçici (ağ / sunucu) hata, bir süre sonra tekrar denenir."""

    def __init__(self, msg: str = "", transient: bool = False):
        super().__init__(msg)
        self.transient = transient


_TRANSIENT = ("timed out", "timeout", "connection reset", "connection aborted", "connectionerror",
              "temporary failure", "remote end closed", "incompleteread", "incomplete read", "http error 500",
              "http error 502", "http error 503", "http error 504", "http error 429", "too many requests",
              "network is unreachable", "getaddrinfo", "ssl:", "eof occurred", "bad gateway", "service unavailable")


def is_transient(msg: str) -> bool:
    """Tek bir denemenin hata metni geçici (ağ) hatası mı?"""
    m = (msg or "").lower()
    if any(x in m for x in ("drm", "private video", "video unavailable", "has been removed", "http error 404",
                            "unsupported url")):
        return False
    return any(x in m for x in _TRANSIENT)


# ---------------------------------------------------------------- ffmpeg

def ffmpeg_bin() -> str:
    exe = which("ffmpeg")
    if not exe:
        raise FetchError("ffmpeg bulunamadı. STT-Kurulum.exe'yi tekrar çalıştırın (ffmpeg'i kendisi kurar).")
    return exe


def header_arg(headers: dict) -> str:
    return "".join(f"{k}: {v}\r\n" for k, v in headers.items())


def http_input_args(src: str, headers: dict | None, live: bool = False) -> list[str]:
    """ffmpeg'in http girdisi için başlık + yeniden bağlanma seçenekleri (dosya için boş)."""
    if not src.startswith("http"):
        return []
    out = ["-headers", header_arg(headers)] if headers else []
    # her http girdisinde: bağlantı 30 sn ses vermezse ffmpeg hata versin (eskiden başlıksız linkte sonsuza dek takılabilirdi)
    out += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10", "-rw_timeout", "30000000"]
    return out


def to_wav16k(src: str, dst: Path, headers: dict | None = None) -> Path:
    """Herhangi bir medya dosyası/URL -> 16 kHz mono PCM WAV (atomik)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *http_input_args(src, headers),
           "-i", src, "-vn", "-sn", "-dn", "-map", "0:a:0?", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
    # düşük öncelik: dönüştürme sırasında bilgisayar akıcı kalsın
    try:  # 3 saatlik film bile birkaç dakikada çevrilir; 2 saat = takıldı
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=7200, **popen_kw(low_priority=True))
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Ses dönüştürme 2 saatte bitmedi (ffmpeg takıldı); tekrar deneyin") from None
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


def explain_error(msg: str) -> str:
    """İndirme hatasını kullanıcının anlayacağı Türkçe bir cümleye çevir (orijinali kısaltılmış eklenir)."""
    full = (msg or "").lower()
    # asıl hata ilk denemedir; sonraki denemelerin (çerez vb.) hataları yanıltıcı ipucu vermesin
    primary = re.split(r"\s\s\[|\s\|\|\s", full.replace("i̇ndirilemedi. ", ""), maxsplit=1)[0]
    site = []  # siteye özgü: yt-dlp'nin genel mesajı yanıltıcı (sorun takipçisindeki örneklerle; 1.9.14)
    if "odnoklassniki" in full and "unable to extract player" in full:
        site.append("ok.ru bu videoyu vermedi: çoğu zaman bölgenize kapalıdır ya da giriş ister (ok.ru'da oturum "
                    "açtığınız tarayıcıyı Ayarlar'da seçin).")
    if "rutube" in full and "options json" in full and "404" in full:
        site.append("Rutube bu videoyu oynatmıyor: bölgenize kapalı, yaş sınırlı ya da henüz yayınlanmamış olabilir.")
    if "available to followers" in full:
        site.append("VK bu videoyu yalnız takipçilere açık gösteriyor (VK tarafında bilinen bir hata da olabilir; "
                    "VK'da oturum açtığınız tarayıcıyı Ayarlar'da seçip tekrar deneyin).")
    if site:
        short = (msg or "").strip().splitlines()[-1][:300]
        return f"{site[0]}  (ayrıntı: {short})"
    hints = [
        (("not available in your country", "geo restrict", "geo-restrict", "not available from your location",
          "blocked it in your country", "not available in your region", "use a vpn or a proxy", "geoblock",
          "geo restriction", "недоступно в вашей стране", "недоступно в вашем регионе"),
         "Bu video bulunduğunuz ülkeye kapalı (coğrafi kısıtlama)."),
        (("tarayıcı açılamadı",), "Gizli tarayıcı (Edge) açılamadı: uygulamada Güncelle'ye basın; olmazsa Ayarlar'dan "
                                   "'gizli tarayıcıyla yakala'yı kapatın."),
        (("playwright",), "Tarayıcıyla yakalama bileşeni kurulu değil: uygulamada Güncelle'ye basın."),
        (("ffmpeg bulunamadı",), "ffmpeg bulunamadı. Kurulumu (STT-Kurulum.exe) tekrar çalıştırın."),
        (("not a bot", "confirm you", "sign in to confirm"),
         "YouTube bot doğrulaması istedi. Ayarlar'da çerez tarayıcısını seçin (YouTube'a giriş yaptığınız "
         "tarayıcı) ve o tarayıcıyı kapatıp tekrar deneyin."),
        (("private video", "members-only", "join this channel"), "Video gizli ya da yalnızca üyelere açık."),
        # giriş: çerez okuma hatasından ÖNCE ("Use --cookies..." yazan giriş hatası eskiden "çerez okunamadı" sanılıyordu)
        (("registered users", "only available for registered", "login required", "account authentication",
          "logged-in", "requires authentication"),
         "Site giriş istiyor. Ayarlar'da çerez tarayıcısını seçin (siteye giriş yaptığınız tarayıcı)."),
        (("video unavailable", "this content isn't available", "has been removed", "http error 404"),
         "Video bulunamadı ya da kaldırılmış."),
        (("live event will begin", "premieres in", "is_upcoming"), "Yayın henüz başlamadı."),
        (("impersonate target", "impersonation dependenc", "dependencies required to support this target"),
         "Bu site (ör. Dailymotion) tarayıcı taklidi istiyor ama gerekli kütüphane (curl_cffi) kurulu değil ya da "
         "uyumsuz. Uygulamada Güncelle'ye basın; güncel sürüm bu kütüphaneyi kendisi kurar."),
        (("cloudflare", "anti-bot challenge"),
         "Site bot koruması (Cloudflare) kullanıyor; tarayıcı taklidi ve gizli tarayıcı denendi. Video tarayıcınızda "
         "açılıyorsa Ayarlar'da o tarayıcıyı çerez kaynağı seçip tekrar deneyin."),
        (("unable to extract", "unable to download webpage", "no video formats found",
          "expected string or bytes-like object"),
         "Site videoyu vermedi (site değişmiş ya da koruma var). STT yt-dlp'yi günde bir kez kendisi günceller; "
         "uygulamayı kapatıp açıp tekrar deneyin. Giriş isteyen bir siteyse Firefox'ta oturum açın."),
        (("dpapi", "could not copy chrome cookie", "failed to decrypt", "app-bound", "failed to load cookies",
          "cookies database", "çerez okunamadı"),
         "Tarayıcı çerezleri okunamadı. Chrome'u tamamen kapatıp tekrar deneyin ya da Ayarlar'dan Firefox/Edge seçin."),
        (("n challenge", "signature", "js runtime", "jsc", "deno"),
         "YouTube çözücüsü çalışmadı (deno). Kurulumu tekrar çalıştırın; olmazsa yt-dlp güncellenmeli."),
        (("timed out", "connection", "getaddrinfo", "network", "temporary failure", "ssl"),
         "İnternet bağlantısı sorunu. Bağlantıyı kontrol edip tekrar deneyin."),
        (("unsupported url", "m3u8/mp4 linki bulunamadı"), "Bu sayfada indirilebilir video bulunamadı."),
        (("http error 403", "http error 401", "login", "log in", "use --cookies"),
         "Site giriş istiyor. Ayarlar'da çerez tarayıcısını seçin (siteye giriş yaptığınız tarayıcı)."),
    ]
    for m in (primary, full):
        for keys, text in hints:
            if any(k in m for k in keys):
                short = (msg or "").strip().splitlines()[-1][:300] if msg else ""
                return f"{text}  (ayrıntı: {short})" if short else text
    return (msg or "Bilinmeyen hata")[:500]


# ---------------------------------------------------------------- yt-dlp

class _QuietLogger:
    """yt-dlp mesajlarını kendi günlüğümüze (debug) yönlendir; hataları biz özetleriz."""

    # yt-dlp bazı yerlerde ek argüman verir: çerez okuyucu warning(msg, only_once=True) çağırıyor. Eskiden bu
    # TypeError ile BÜTÜN çerez okumasını düşürüyordu (Chrome/Edge'in yeni şifreli çerezlerinde tam da bu uyarı çıkar).
    def debug(self, msg, *a, **k):
        log.debug("yt-dlp: %s", msg)

    def info(self, msg, *a, **k):
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg, *a, **k):
        log.debug("yt-dlp uyarı: %s", msg)

    def error(self, msg, *a, **k):
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
        "allow_unplayable_formats": False,
    }
    opts["js_runtimes"] = _js_runtimes(warn=True)
    opts["progress_hooks"] = [lambda d: check_cancel(workdir)]  # "Durdur": indirme hemen kesilsin
    opts.update(extra)
    return opts


def _js_runtimes(warn: bool = False) -> dict:
    rts = {}
    for n in ("deno", "node", "bun", "quickjs"):
        exe = which("qjs" if n == "quickjs" else n)
        if exe:
            rts[n] = {"path": exe}
    if not rts and warn:
        log.warning("deno bulunamadı: YouTube videoları indirilemeyebilir (kurulumu tekrar çalıştırın).")
    return rts or {"deno": {}}


def _find_download(workdir: Path) -> Path | None:
    cands = [p for p in workdir.glob("kaynak.*")
             if p.suffix not in (".part", ".ytdl", ".json") and not p.name.endswith(".part.wav")]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


class LiveStream(Exception):
    """Link şu an canlı yayında: indirilemez (bitmez), canlı çeviriye geçilir."""

    def __init__(self, info: dict):
        super().__init__("canlı yayın")
        self.info = info


# ---------------------------------------------------------------- çerezler

_BROWSER_DIRS = {  # Windows: tarayıcı kurulu mu (profil klasörü var mı)
    "firefox": ("APPDATA", r"Mozilla\Firefox\Profiles"),
    "edge": ("LOCALAPPDATA", r"Microsoft\Edge\User Data"),
    "chrome": ("LOCALAPPDATA", r"Google\Chrome\User Data"),
    "brave": ("LOCALAPPDATA", r"BraveSoftware\Brave-Browser\User Data"),
    "opera": ("APPDATA", r"Opera Software\Opera Stable"),
    "vivaldi": ("LOCALAPPDATA", r"Vivaldi\User Data"),
}
# Firefox önce: Chrome 127+ (ve Edge) Windows'ta çerezleri 'uygulamaya bağlı şifreleme' ile saklıyor,
# yt-dlp onları çoğu zaman okuyamıyor. Instagram/Facebook gibi giriş isteyen siteler bu yüzden inmiyordu.
_BROWSER_ORDER = ("firefox", "edge", "chrome", "brave", "opera", "vivaldi")


def installed_browsers() -> list[str]:
    if os.name != "nt":
        home = Path.home()
        cand = {"firefox": home / ".mozilla/firefox", "chrome": home / ".config/google-chrome",
                "brave": home / ".config/BraveSoftware", "edge": home / ".config/microsoft-edge",
                "opera": home / ".config/opera", "vivaldi": home / ".config/vivaldi"}
        return [b for b in _BROWSER_ORDER if cand[b].exists()]
    out = []
    for b in _BROWSER_ORDER:
        env, sub = _BROWSER_DIRS[b]
        base = os.environ.get(env)
        if base and (Path(base) / sub).exists():
            out.append(b)
    return out


def cookie_browsers(pref: str | None) -> list[str]:
    """Denenecek tarayıcılar: None = hiç (kullanıcı kapattı), 'auto' = kurulu olanlar, 'chrome' = önce o, sonra ötekiler."""
    if not pref:
        return []
    pref = pref.strip().lower()
    inst = installed_browsers()
    if pref in ("auto", "otomatik"):
        return inst
    return [pref] + [b for b in inst if b != pref]


_cookie_lock = threading.Lock()
_cookie_cache: dict[str, tuple[float, str | None, str]] = {}   # tarayıcı -> (zaman, dosya, hata)


def browser_cookie_file(browser: str) -> tuple[str | None, str]:
    """Tarayıcı çerezlerini bir kez okuyup Netscape cookies.txt'ye yazar (önbellek 30 dk; hata 10 dk).

    Her indirmede ayrı ayrı okumak hem yavaş hem de iki indirme aynı anda okuyunca 'database is locked'
    hatası veriyordu. Döner: (dosya, hata)."""
    import tempfile
    with _cookie_lock:
        hit = _cookie_cache.get(browser)
        if hit and time.time() - hit[0] < (1800 if hit[1] else 600) and (not hit[1] or Path(hit[1]).exists()):
            return hit[1], hit[2]
        path, err = None, ""
        try:
            from yt_dlp.cookies import extract_cookies_from_browser
            jar = extract_cookies_from_browser(browser, logger=_QuietLogger())
            d = Path(tempfile.gettempdir()) / f"stt-cerez-{os.getpid()}"
            d.mkdir(exist_ok=True)
            f = d / f"{browser}.txt"
            jar.save(str(f), ignore_discard=True, ignore_expires=True)
            path = str(f)
            log.info("Çerezler okundu: %s (%d)", browser, len(jar))
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            log.info("%s çerezleri okunamadı: %s", browser, err[:200])
        _cookie_cache[browser] = (time.time(), path, err)
        return path, err


def impersonate_target():
    """curl_cffi kuruluysa tarayıcı taklidi hedefi (Dailymotion, Cloudflare'li siteler...)."""
    try:
        import curl_cffi  # noqa: F401
        # yt-dlp yalnız desteklediği curl_cffi sürümlerinde taklidi açar; yoksa her denemede "Impersonate target
        # is not available" hatası verirdi (çerez denemeleri dahil). Bu içe aktarma uyumsuz sürümde hata verir.
        from yt_dlp.networking._curlcffi import CurlCFFIRH  # noqa: F401
        from yt_dlp.networking.impersonate import ImpersonateTarget
        return ImpersonateTarget.from_str("chrome")
    except Exception:  # noqa: BLE001
        return None


def _attempts(url: str, cookies_browser: str | None, cookies_file: str | None, workdir: Path):
    """yt-dlp denemeleri (etiket, ek seçenekler): düz -> tarayıcı taklidi -> tarayıcı çerezleri -> cookies.txt.

    Çerezler ancak sırası gelince okunur (düz deneme çoğu zaman yeter) ve her deneme kendi kopyasını kullanır
    (yt-dlp çerez dosyasına geri yazar; ortak önbellek dosyası bozulmasın). Giriş isteyen sitelerde çerez önce."""
    imp = impersonate_target()
    plain = [("çerezsiz", lambda: {})]
    if imp is not None:
        plain.append(("tarayıcı taklidi", lambda: {"impersonate": imp}))

    def browser(b):
        def make():
            cf, err = browser_cookie_file(b)
            if not cf:
                raise FetchError(err or "çerez okunamadı")
            return {"cookiefile": _private_copy(cf, workdir), **({"impersonate": imp} if imp is not None else {})}
        return make
    cookies = [(f"{b} çerezleri", browser(b)) for b in cookie_browsers(cookies_browser)]
    if cookies_file and Path(cookies_file).is_file():
        cookies.append(("cookies.txt", lambda: {"cookiefile": _private_copy(cookies_file, workdir)}))
    host = (urlsplit(url).hostname or "").lower()
    login_first = any(host == d or host.endswith("." + d) for d in LOGIN_FIRST)
    yield from (cookies + plain) if login_first else (plain + cookies)


def _private_copy(src: str, workdir: Path) -> str:
    import shutil
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / f"cerez-{threading.get_ident()}.txt"
    shutil.copyfile(src, dst)
    return str(dst)


def _import_ytdlp():
    """yt-dlp'yi içe aktar; günlük güncelleme sürüyorsa önce bitmesini bekle (yarım yazılmış paket yüklenmesin)."""
    import sys
    if "yt_dlp" not in sys.modules:
        wait_ytdlp_update()
    import yt_dlp
    return yt_dlp


def ytdlp_download(url: str, workdir: Path, cookies_browser: str | None, cookies_file: str | None,
                   headers: dict | None = None, detect_live: bool = True):
    """(dosya, info) döner. DRM ise DRMProtected; canlı yayınsa LiveStream; hiçbiri olmazsa FetchError."""
    yt_dlp = _import_ytdlp()
    from yt_dlp.utils import DownloadError

    live: dict = {}

    def live_filter(info, *, incomplete=False):
        if detect_live and (info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming")):
            live.update(info)
            return "canlı yayın: canlı çeviriye geçiliyor"
        return None

    errors: list[tuple[str, str]] = []   # (deneme, hata): ilk denemenin hatası çoğu zaman asıl sebeptir
    for label, make in _attempts(url, cookies_browser, cookies_file, workdir):
        check_cancel(workdir)
        try:
            extra = make()
            if headers:
                extra["http_headers"] = dict(headers)  # kimlik yt-dlp'nin (ya da taklidin) kendisi
            # yalnız ilk video: tanınmayan kanal / liste linkinde (VK, Rutube, Dailymotion kullanıcı sayfası
            # noplaylist'i dinlemiyor) bütün kanal indirilmesin
            with yt_dlp.YoutubeDL(_ytdlp_opts(workdir, {**extra, "match_filter": live_filter,
                                                        "playlist_items": "1"})) as ydl:
                info = ydl.extract_info(url, download=True)
            if live:
                raise LiveStream(dict(live))
            if info and info.get("entries"):
                info = next((e for e in info["entries"] if e), None)
                if not info:
                    raise FetchError("Oynatma listesi boş")
                log.warning("Link bir oynatma listesi; yalnızca ilk video alındı. Hepsi için 'Kanal / liste' "
                            "sekmesini kullanın.")
            f = _find_download(workdir)
            if not f:
                raise FetchError("yt-dlp dosya üretmedi")
            log.info("İndirildi (%s): %s", label, f.name)
            return f, info
        except DownloadError as e:
            msg = str(e)
            errors.append((label, msg))
            if is_drm_error(msg):
                raise DRMProtected(f"DRM korumalı: {url}") from None
            low = msg.lower()
            if "unsupported url" in low:
                break  # çerez işe yaramaz; doğrudan yedek yollara geç
            log.info("yt-dlp başarısız (%s): %s", label, _last_line(msg)[:200])
            if label == "çerezsiz" and "http error 404" in low and not needs_cookies(msg):
                break
        except (LiveStream, Cancelled):
            raise
        except Exception as e:  # noqa: BLE001  (çerez okuma hataları vb.)
            errors.append((label, f"{type(e).__name__}: {e}"))
            log.info("yt-dlp hata (%s): %s", label, str(e)[:300])
    first = errors[0][1] if errors else ""
    raise FetchError(ytdlp_error_summary(errors) or "yt-dlp başarısız", transient=is_transient(first))


def ytdlp_info(url: str, cookies_browser: str | None = None, flat: bool = False, extra: dict | None = None,
               workdir: Path | None = None) -> dict:
    """İndirmeden bilgi al (canlı yayın adresi, kanal/liste içeriği). Aynı deneme sırası, indirme yok."""
    import tempfile
    yt_dlp = _import_ytdlp()
    workdir = workdir or Path(tempfile.gettempdir()) / f"stt-bilgi-{os.getpid()}"
    opts = {**_ytdlp_opts(workdir, {}), "skip_download": True, **(extra or {})}
    if flat:
        opts["extract_flat"] = "in_playlist"
    last = ""
    for _label, make in _attempts(url, cookies_browser, None, workdir):
        try:
            with yt_dlp.YoutubeDL({**opts, **make()}) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except Exception as e:  # noqa: BLE001
            last = str(e)
            low = last.lower()
            if "unsupported url" in low or "does not have a" in low or "http error 404" in low:
                break  # çerezle de değişmez (ör. kanalın 'streams' sekmesi yok)
    raise FetchError(_last_line(last) if last else "bilgi alınamadı", transient=is_transient(last))


# ---------------------------------------------------------------- yt-dlp güncel tutma

_upd_done = threading.Event()
_upd_done.set()


def start_ytdlp_update(state_dir: Path, max_age_h: float = 20) -> bool:
    """yt-dlp'yi günde bir kez güncelle (arka planda). Siteler sık değiştiği için eski yt-dlp
    YouTube dışındaki sitelerde 'Unable to extract' gibi hatalar veriyor. yt_dlp içe aktarılmadan önce
    başlatılır; ilk içe aktarma bitmesini en fazla 3 dk bekler. Döner: güncelleme başlatıldı mı."""
    import sys
    stamp = Path(state_dir) / "ytdlp_guncelleme.txt"
    if stamp.exists() and time.time() - stamp.stat().st_mtime < max_age_h * 3600:
        return False
    if "yt_dlp" in sys.modules:
        return False  # zaten yüklü: güncelleme ancak yeniden açılışta etkili olur, sonraki açılışa bırak
    _upd_done.clear()

    def run():
        try:
            r = subprocess.run([console_python(), "-m", "pip", "install", "-q", "--upgrade",
                                "--disable-pip-version-check", "yt-dlp[default,curl-cffi]"],
                               capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace",
                               **popen_kw())
            if r.returncode == 0:
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
                log.info("yt-dlp güncel.")
            else:
                log.info("yt-dlp güncellenemedi: %s", (r.stderr or r.stdout)[-300:])
        except Exception as e:  # noqa: BLE001
            log.info("yt-dlp güncellenemedi: %s", e)
        finally:
            _upd_done.set()
    threading.Thread(target=run, name="ytdlp-guncelle", daemon=True).start()
    return True


def wait_ytdlp_update(timeout: float = 180) -> None:
    _upd_done.wait(timeout)


def _last_line(msg: str) -> str:
    lines = [x.strip() for x in str(msg).splitlines() if x.strip()]
    return (lines[-1] if lines else str(msg)).removeprefix("ERROR: ")[:300]


def ytdlp_hint(msg: str) -> str:
    m = msg.lower()
    if "impersonat" in m:
        return "Site tarayıcı taklidi istiyor: curl_cffi kurulu değil. Uygulamada Güncelle'ye basın (otomatik kurulur)."
    if "not a bot" in m or "sign in to confirm" in m:
        return "YouTube giriş/robot kontrolü istiyor: Ayarlar'da çerez tarayıcısı olarak Firefox ya da Edge seçin."
    if "js runtime" in m or "javascript" in m or "n challenge" in m or "deno" in m:
        return "YouTube için deno gerekli: STT-Kurulum.exe'yi tekrar çalıştırın."
    if "requested format is not available" in m or "only images are available" in m:
        return "Ses biçimi alınamadı (yt-dlp eski olabilir ya da deno eksik): STT-Kurulum.exe'yi tekrar çalıştırın."
    if "could not copy chrome cookie" in m or "dpapi" in m or "app-bound" in m or "failed to decrypt" in m:
        return "Chrome çerezleri okunamadı: Ayarlar'da Firefox/Edge seçin ya da 'Kullanma' deyin."
    if "private" in m or "login" in m or "log in" in m:
        return "Video giriş istiyor: Ayarlar'da o sitede oturum açık olan tarayıcıyı seçin."
    return ""


def ytdlp_error_summary(errors: list[tuple[str, str]]) -> str:
    """Asıl hatayı (ilk deneme) öne koy; çerez denemesi başka hata verdiyse onu da ekle."""
    if not errors:
        return ""
    first_label, first = errors[0]
    out = _last_line(first)
    for label, msg in errors[1:]:
        if _last_line(msg) != out:
            out += f"  [{label}: {_last_line(msg)[:160]}]"
    hint = ytdlp_hint(first) or next((ytdlp_hint(m) for _, m in errors[1:] if ytdlp_hint(m)), "")
    return out + (f"  -> {hint}" if hint else "")


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

    return sorted(out, key=rank_media)


def rank_media(url: str) -> int:
    """Medya adayının önceliği (küçük = önce): ana m3u8, m3u8, mp4/m4a, mpd, diğer."""
    u = url.lower().split("?")[0]
    if u.endswith(".m3u8"):
        return 0 if any(x in u for x in ("master", "index", "playlist", "manifest")) else 1
    if u.endswith((".mp4", ".m4v", ".m4a", ".webm", ".mp3", ".aac")):
        return 2
    if u.endswith(".mpd"):
        return 3
    return 5


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


def _http_get(url: str, headers: dict, timeout: float = 30):
    """Sayfa indir: curl_cffi varsa tarayıcı taklidiyle (Cloudflare / bot korumalı siteler), yoksa requests."""
    try:
        from curl_cffi import requests as creq
        h = {k: v for k, v in headers.items() if k.lower() != "user-agent"}  # taklidin kendi kimliği kalsın
        return creq.get(url, headers=h, timeout=timeout, impersonate="chrome", allow_redirects=True)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        log.debug("curl_cffi ile alınamadı (%s), requests deneniyor", e)
    import requests
    return requests.get(url, headers=headers, timeout=timeout)


_IFRAME_RE = re.compile(r"""<(?:iframe|embed)[^>]+?src=["']([^"']+)["']""", re.I)
_EMBED_HOSTS = ("youtube.com/embed", "youtube-nocookie.com/embed", "player.vimeo.com", "dailymotion.com/embed",
                "geo.dailymotion.com", "dai.ly", "ok.ru/videoembed", "odnoklassniki.ru/videoembed", "vk.com/video_ext",
                "vkvideo.ru/video_ext", "rutube.ru/play/embed", "players.brightcove.net", "cdn.jwplayer.com",
                "content.jwplatform.com", "fast.wistia", "streamable.com", "facebook.com/plugins/video",
                "platform.twitter.com", "kaltura.com", "uplynk", "vidyard", "twitch.tv", "kick.com", "rumble.com/embed",
                "mediadelivery.net", "bunny", "my.mail.ru/video/embed", "/+/video/meta/", "/embed/")
# mail.ru: sayfa yapılandırması yoksa yt-dlp TypeError veriyor (#15243); gömme / meta adresi sayfasız açılıyor
_MAILRU_RE = re.compile(r'(?:https?:)?//my\.mail\.ru/(?:video/embed/\d+|\+/video/meta/\d+)|"metaUrl"\s*:\s*"([^"]+)"')
_BC_TAG_RE = re.compile(r"<[^>]*data-account=[^>]*>", re.I | re.S)


def _attr(tag: str, name: str) -> str | None:
    m = re.search(name + r"""=["']([^"']+)["']""", tag, re.I)
    return m.group(1) if m else None


def find_embeds(page_html: str, base_url: str) -> list[str]:
    """Sayfaya gömülü oynatıcılar (YouTube, Vimeo, Dailymotion, ok.ru, Brightcove, JW Player...).
    Kulüp siteleri videoyu çoğu zaman böyle gömer; yt-dlp gömülü adresi doğrudan tanır."""
    text = page_html.replace("\\/", "/")
    out: list[str] = []
    for m in _IFRAME_RE.finditer(text):
        u = html.unescape(m.group(1)).strip()
        if u.startswith("//"):
            u = "https:" + u
        u = urljoin(base_url, u)
        if any(h in u.lower() for h in _EMBED_HOSTS):
            out.append(u)
    for m in _BC_TAG_RE.finditer(text):  # Brightcove <video-js data-account=.. data-video-id=.. data-player=..>
        tag = m.group(0)
        acc, vid = _attr(tag, "data-account"), _attr(tag, "data-video-id") or _attr(tag, "data-videoid")
        if acc and vid:
            player = _attr(tag, "data-player") or "default"
            out.append(f"https://players.brightcove.net/{acc}/{player}_default/index.html?videoId={vid}")
    for m in re.finditer(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}", text):
        out.append(m.group(0))
    if "mail.ru" in (urlsplit(base_url).hostname or ""):
        for m in _MAILRU_RE.finditer(text):
            u = m.group(1) or m.group(0)
            if "/video/" in u:
                out.append(urljoin("https://my.mail.ru/", u if not u.startswith("//") else "https:" + u))
    for m in _ATTR_RE.finditer(text):  # og:video bilinen bir oynatıcıysa ffmpeg yerine yt-dlp'ye (gömülü adım)
        u = m.group(2)
        if u and any(h in u.lower() for h in _EMBED_HOSTS if h != "/embed/"):
            out.append(urljoin(base_url, html.unescape(u)))
    seen, uniq = set(), []
    for u in out:
        if u not in seen and u.split("#")[0] != base_url:
            seen.add(u)
            uniq.append(u)
    return uniq[:6]


def probe_duration(url: str, headers: dict | None = None, timeout: float = 40) -> float:
    """ffprobe ile süre (sn); okunamazsa 0. Reklam (kısa) ile asıl videoyu ayırmak için."""
    exe = which("ffprobe")
    if not exe:
        return -1.0
    cmd = [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"]
    if headers and url.startswith("http"):
        cmd += ["-headers", header_arg(headers)]
    cmd += [url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **popen_kw())
        return float((r.stdout or "0").strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return 0.0


def _media_headers(req_headers: dict, page_url: str, cookies: str = "") -> dict:
    keep = {}
    for k, v in (req_headers or {}).items():
        lk = k.lower()
        if lk in ("referer", "origin", "user-agent", "cookie", "authorization") or lk.startswith("x-"):
            keep[k.title() if not lk.startswith("x-") else k] = v
    keep.setdefault("User-Agent", user_agent())
    keep.setdefault("Referer", page_url)
    if cookies and not any(k.lower() == "cookie" for k in keep):
        keep["Cookie"] = cookies
    return keep


def page_title(page_html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", page_html or "", re.I | re.S)
    return html.unescape(m.group(1)).strip() if m else ""


def scrape_and_convert(page_url: str, dst: Path, page_html: str) -> tuple[Path, dict]:
    """Sayfa HTML'inde doğrudan geçen m3u8/mp4 adreslerini ffmpeg ile çek."""
    headers = {"User-Agent": user_agent(), "Referer": page_url, "Accept-Language": "en-US,en;q=0.9,tr;q=0.8"}
    cands = find_media_urls(page_html, page_url)
    if not cands:
        raise FetchError("Sayfada m3u8/mp4 linki bulunamadı")
    title = page_title(page_html)
    errors = []
    drm_seen = False
    import requests
    s = requests.Session()
    for u in cands[:6]:
        try:
            _check_manifest_drm(s, u, headers)
            log.info("Yedek yol: ffmpeg ile çekiliyor: %s", u[:150])
            to_wav16k(u, dst, headers={"User-Agent": headers["User-Agent"], "Referer": page_url})
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


def browser_and_convert(page_url: str, dst: Path) -> tuple[Path, dict]:
    """Gizli tarayıcıyla yakala (stt/browser.py), en uzun medyayı (reklam değil) ffmpeg ile çevir."""
    try:
        from .browser import capture_media
    except ImportError as e:  # pragma: no cover
        raise FetchError(f"tarayıcı modülü yok: {e}") from None
    try:
        cap = capture_media(page_url, ua=UA)
    except ImportError:
        raise FetchError("Tarayıcıyla yakalama için 'playwright' kurulu değil (uygulamada Güncelle'ye basın).") from None
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"Tarayıcıyla yakalanamadı: {str(e).splitlines()[0][:200]}") from None
    cands = cap["candidates"][:8]
    if not cands:
        raise FetchError("Tarayıcıda da video adresi görünmedi (giriş/abonelik isteyen ya da DRM'li olabilir).")
    import requests
    from concurrent.futures import ThreadPoolExecutor

    def score(c):
        hdr = _media_headers(c["headers"], page_url, cap.get("cookies", ""))
        try:
            _check_manifest_drm(requests.Session(), c["url"], hdr)
        except DRMProtected:
            return None
        return (probe_duration(c["url"], hdr) if len(cands) > 1 else 0.0), c, hdr
    with ThreadPoolExecutor(max_workers=len(cands)) as ex:  # adaylar paralel denenir (reklam adresi yavaş olabilir)
        scored = [x for x in ex.map(score, cands) if x]
    if not scored:
        raise DRMProtected(f"DRM korumalı: {page_url}")
    scored.sort(key=lambda x: -x[0])  # en uzun = asıl video (reklamlar kısa)
    errors = []
    for d, c, hdr in scored[:4]:
        if 0 <= d < 20 and len(scored) > 1:
            continue
        try:
            log.info("Tarayıcı yolu: %s (%.0f sn) çekiliyor", c["url"][:150], d)
            to_wav16k(c["url"], dst, headers=hdr)
            return dst, {"title": cap.get("title") or "", "webpage_url": page_url, "media_url": c["url"],
                         "extractor": "stt-browser"}
        except (FetchError, SkipItem) as e:
            errors.append(str(e)[:160])
    raise FetchError("Tarayıcı yolu başarısız: " + " | ".join(errors[-2:]))


# ---------------------------------------------------------------- ana giriş

_INFO_KEYS = ("id", "title", "extractor", "extractor_key", "webpage_url", "uploader", "channel", "channel_id",
              "upload_date", "timestamp", "release_timestamp", "duration", "description", "playlist_title")


def _from_ytdlp(raw: dict) -> dict:
    info = {k: raw.get(k) for k in _INFO_KEYS if raw.get(k) is not None}
    if info.get("description"):
        info["description"] = str(info["description"])[:1000]
    return info


def site_url_variants(url: str) -> list[str]:
    """yt-dlp'nin tanımadığı / bozuk işlediği link biçimleri için denenecek sıra (önce düzeltilmiş, sonra asıl).
    - m.ok.ru, ok.ru/group/../video/<id> -> ok.ru/video/<id>; olmazsa ok.ru/videoembed/<id> (bölge kısıtında gerçek
      sebebi gösteriyor, bazı videolar oradan oynuyor; yt-dlp sorun #15845)
    - rutube.ru/shorts/<id> -> rutube.ru/video/<id>/ (shorts biçimi genel çıkarıcıya düşüyordu)
    - VK gömme biçimi video_ext.php?oid=..&id=.. yt-dlp'de bozuk (#16070, #17537); aynı video sayfa biçimiyle
      (vkvideo.ru/video<oid>_<id>) iniyor; özel gömmeler yalnız gömmeden açılabildiği için asıl link de denenir.
    (yt-dlp 2026.10.26 kaynağına göre; gerçek linklerle denenmedi.)"""
    from urllib.parse import parse_qs
    p = urlsplit((url or "").strip())
    host = (p.hostname or "").lower()
    out: list[str] = []
    if host == "ok.ru" or host.endswith(".ok.ru") or host.endswith("odnoklassniki.ru"):
        m = re.search(r"/video(?:embed)?/(\d+)", p.path or "")
        if m:
            out = [f"https://ok.ru/video/{m.group(1)}", f"https://ok.ru/videoembed/{m.group(1)}"]
    elif host.endswith("rutube.ru"):
        m = re.match(r"^/shorts/([0-9a-f]{32})", p.path or "")
        if m:
            out = [f"https://rutube.ru/video/{m.group(1)}/"]
    elif (host.endswith("vk.com") or host.endswith("vkvideo.ru") or host.endswith("vk.ru")) \
            and (p.path or "").endswith("video_ext.php"):
        q = parse_qs(p.query)
        oid, vid = (q.get("oid") or [""])[0], (q.get("id") or [""])[0]
        if re.fullmatch(r"-?\d+", oid) and re.fullmatch(r"\d+", vid):
            out = [f"https://vkvideo.ru/video{oid}_{vid}"]
    return list(dict.fromkeys(out + [url]))  # sıra korunur; asıl link zaten listedeyse yerinde kalır


def _ytdlp_to_wav(url: str, workdir: Path, wav: Path, cfg, **kw) -> dict:
    errors: list[FetchError] = []
    variants = site_url_variants(url)
    for i, u in enumerate(variants):
        try:
            src, raw = ytdlp_download(u, workdir, cfg.cookies_browser, cfg.cookies_file, **kw)
        except FetchError as e:
            errors.append(e)
            if e.transient or i == len(variants) - 1:
                break
            log.info("yt-dlp olmadı (%s); başka link biçimi deneniyor", u[:120])
            continue
        to_wav16k(str(src), wav)
        src.unlink(missing_ok=True)
        info = _from_ytdlp(raw)
        if u != url:
            info["webpage_url"] = url  # kayıt kullanıcının verdiği linkle kalsın
        return info
    # "Unable to extract" gibi genel hata yerine sitenin kendi sebebini (ör. ok.ru gömmesindeki bölge mesajı) göster
    best = next((e for e in errors if "unable to extract" not in str(e).lower()), errors[0])
    raise best


def download_url(url: str, workdir: Path, wav: Path, cfg) -> dict:
    """Link -> workdir/ses.wav. Sırayla: yt-dlp (düz, taklit, çerezler) -> sayfadaki gömülü oynatıcılar
    (yt-dlp ile) -> sayfadaki m3u8/mp4 -> gizli tarayıcıyla yakalama. Hepsinin hatası özetlenir."""
    steps: list[str] = []
    transient = False

    def fetch_page() -> list:
        """Sayfayı bir kez indir; gömülü oynatıcı ve m3u8/mp4 adımlarını üret."""
        r = _http_get(url, {"User-Agent": user_agent(), "Accept-Language": "en-US,en;q=0.9,tr;q=0.8"})
        ctype = (r.headers.get("content-type") or "").lower()
        if r.status_code >= 400:
            raise FetchError(f"HTTP {r.status_code}", transient=r.status_code in (429, 500, 502, 503, 504))
        if ctype.startswith(("video/", "audio/")) or "mpegurl" in ctype:  # link doğrudan medya dosyası
            def direct():
                to_wav16k(str(r.url), wav, headers={"User-Agent": user_agent()})
                return {"title": "", "webpage_url": url, "media_url": str(r.url), "extractor": "stt-direct"}
            return [("doğrudan", direct)]
        if "html" not in ctype:
            return []

        def embed(u):
            def run():
                log.info("Gömülü oynatıcı deneniyor: %s", u[:150])
                info = _ytdlp_to_wav(u, workdir, wav, cfg, headers={"Referer": url}, detect_live=False)
                return {**info, "webpage_url": url, "title": info.get("title") or page_title(r.text)}
            return run
        return ([("gömülü", embed(u)) for u in find_embeds(r.text, url)]
                + [("sayfa", lambda: scrape_and_convert(url, wav, r.text)[1])])

    queue: list = [("yt-dlp", lambda: _ytdlp_to_wav(url, workdir, wav, cfg)), ("sayfa", fetch_page)]
    if cfg.browser_capture:
        queue.append(("tarayıcı", lambda: browser_and_convert(url, wav)[1]))
    while queue:
        check_cancel(workdir)
        label, step = queue.pop(0)
        try:
            res = step()
        except (DRMProtected, LiveStream, SkipItem, Cancelled):
            raise
        except Exception as e:  # noqa: BLE001
            check_cancel(workdir)  # durdurulduysa yedek yolları deneme
            steps.append(f"{label}: " + _last_line(str(e))[:220])
            if len(steps) == 1:  # asıl hata ilk adımınki (yedek yolların hataları tekrar denemeyi belirlemez)
                transient = getattr(e, "transient", False)
            if label == "yt-dlp":
                log.info("yt-dlp olmadı (%s) -> sayfa ve gömülü oynatıcılar deneniyor", _last_line(str(e))[:160])
            continue
        if isinstance(res, list):  # sayfa indirildi: yeni adımlar tarayıcıdan önce denenir
            queue[0:0] = res
            continue
        return res
    raise FetchError("İndirilemedi. " + "  ||  ".join(steps), transient=transient)


def prepare_audio(item, workdir: Path, cfg) -> tuple[Path, dict]:
    """Öğeyi workdir/ses.wav olarak hazırlar. Var ve sağlamsa yeniden indirmez (sürdürme)."""
    workdir.mkdir(parents=True, exist_ok=True)
    wav = workdir / "ses.wav"
    info_path = workdir / "kaynak.json"
    if wav.exists() and wav_ok(wav):
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        log.info("Önceden hazırlanmış ses kullanılıyor: %s", item.title[:80])
        return wav, info

    ffmpeg_bin()  # ffmpeg yoksa açık mesajla hemen dur (yedek yollar yanıltıcı hata vermesin)
    if item.kind == "file":
        info = {"title": Path(item.source).stem, "source_path": item.source,
                "timestamp": int(Path(item.source).stat().st_mtime)}
        to_wav16k(item.source, wav)
    else:
        info = download_url(item.source, workdir, wav, cfg)
    info["duration_s"] = wav_duration(wav)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return wav, info
