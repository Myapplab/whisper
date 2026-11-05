"""Son çare indirme: sayfayı gizli bir tarayıcıda (Edge / Chrome) açıp videonun gerçek adresini yakalamak.

"X downloader" siteleri de aslında bunu yapar: sayfayı açar, oynatıcının istediği .m3u8 / .mp4 / .mpd
adresini bulur. Burada aynısı bilgisayarda, reklamsız ve üçüncü bir siteye link göndermeden yapılır.
Playwright, Windows'ta zaten kurulu olan Microsoft Edge'i kullanır (ayrıca tarayıcı indirmez).
"""
from __future__ import annotations

import os
import re
import time
from urllib.parse import urlsplit

from .util import log

MEDIA_RE = re.compile(r"\.(m3u8|mpd|mp4|m4a|m4v|webm|mp3|aac|ogg|opus|flv)(\?|$|#)", re.I)
SEGMENT_RE = re.compile(r"(\.ts|\.m4s|\.aac|/seg[-_]?\d+|/chunk[-_]?\d+|/frag[-_]?\d+|init\.mp4|range=)", re.I)
AD_HOSTS = ("doubleclick", "googlesyndication", "googleads", "imasdk", "adservice", "adnxs", "amazon-adsystem",
            "advertising", "adsrvr", "spotx", "teads", "moatads", "criteo", "taboola", "outbrain", "pubmatic",
            "rubiconproject", "smartadserver", "adform", "yandex.ru/ads", "an.yandex", "adfox", "vidazoo")
PLAY_SELECTORS = (".vjs-big-play-button", ".jw-icon-display", ".jw-display-icon-container", ".plyr__control--overlaid",
                  ".ytp-large-play-button", "button[aria-label*='Play' i]", "button[aria-label*='Oynat' i]",
                  "[class*='play-button' i]", "[class*='playButton' i]", "[class*='btn-play' i]", ".play", "video")


def is_ad(url: str) -> bool:
    u = url.lower()
    return any(a in u for a in AD_HOSTS)


def rank_media(url: str) -> int:
    u = url.lower().split("?")[0]
    if u.endswith(".m3u8"):
        return 0 if any(x in u for x in ("master", "index", "playlist", "manifest")) else 1
    if u.endswith(".mpd"):
        return 2
    if u.endswith((".mp4", ".m4v", ".webm", ".m4a", ".mp3")):
        return 3
    return 5


def _launch(p):
    """Kurulu tarayıcıyı bul: Edge (Windows'ta her zaman var) -> Chrome -> Playwright Chromium."""
    args = ["--autoplay-policy=no-user-gesture-required", "--mute-audio", "--disable-blink-features=AutomationControlled"]
    errors = []
    exe = os.environ.get("STT_BROWSER_EXE")
    tries = ([{"executable_path": exe}] if exe else []) + [{"channel": "msedge"}, {"channel": "chrome"}, {}]
    if os.path.isdir("/opt/pw-browsers"):
        import glob
        for c in sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome")):
            tries.append({"executable_path": c})
    for kw in tries:
        try:
            return p.chromium.launch(headless=True, args=args, **kw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{kw or 'chromium'}: {str(e).splitlines()[0][:120]}")
    raise RuntimeError("Tarayıcı açılamadı (Edge/Chrome bulunamadı): " + " | ".join(errors))


def capture_media(page_url: str, timeout: float = 45.0, ua: str | None = None) -> dict:
    """Sayfayı aç, oynatmayı başlat, medya isteklerini topla.

    Döner: {"title", "candidates": [{"url", "headers"}], "cookies": "a=b; c=d", "page_url"}.
    Playwright kurulu değilse ImportError.
    """
    from playwright.sync_api import sync_playwright  # isteğe bağlı bağımlılık

    found: dict[str, dict] = {}
    with sync_playwright() as p:
        browser = _launch(p)
        try:
            ctx = browser.new_context(user_agent=ua, locale="tr-TR", viewport={"width": 1280, "height": 720},
                                      ignore_https_errors=True)

            def on_request(req):
                u = req.url
                if u.startswith(("blob:", "data:")) or is_ad(u):
                    return
                rt = req.resource_type
                if (MEDIA_RE.search(u) or rt == "media") and not SEGMENT_RE.search(u.split("?")[0]):
                    if u not in found:
                        try:
                            hdr = dict(req.headers)
                        except Exception:  # noqa: BLE001
                            hdr = {}
                        found[u] = {"url": u, "headers": hdr, "t": time.time()}

            def on_response(resp):  # uzantısız adresler: içerik türünden anla
                try:
                    ct = (resp.headers.get("content-type") or "").lower()
                except Exception:  # noqa: BLE001
                    return
                u = resp.url
                if is_ad(u) or u in found:
                    return
                if "mpegurl" in ct or "dash+xml" in ct or ct.startswith(("video/", "audio/")):
                    if not SEGMENT_RE.search(u.split("?")[0]) and "mp2t" not in ct and "iso.segment" not in ct:
                        found[u] = {"url": u, "headers": dict(resp.request.headers), "t": time.time()}

            ctx.on("request", on_request)
            ctx.on("response", on_response)
            page = ctx.new_page()
            try:
                page.goto(page_url, wait_until="domcontentloaded", timeout=60_000)
            except Exception as e:  # noqa: BLE001  (yavaş sayfa: yüklenen kadarıyla devam)
                log.info("Tarayıcı: sayfa tam yüklenmedi (%s)", str(e).splitlines()[0][:120])
            end = time.time() + timeout
            tick = 0
            while time.time() < end:
                good = [f for f in found.values() if rank_media(f["url"]) <= 3]
                if good and time.time() - max(f["t"] for f in good) > 4:
                    break  # medya bulundu ve birkaç sn yeni istek gelmedi
                if tick % 3 == 0:
                    _try_play(page)
                tick += 1
                page.wait_for_timeout(1000)
            title = ""
            try:
                title = page.title()
            except Exception:  # noqa: BLE001
                pass
            host = urlsplit(page_url).hostname or ""
            cookies = []
            try:
                for c in ctx.cookies():
                    cookies.append(f"{c['name']}={c['value']}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            browser.close()
    cands = sorted(found.values(), key=lambda f: (rank_media(f["url"]), f["t"]))
    log.info("Tarayıcı %s: %d medya adresi bulundu", host, len(cands))
    return {"title": title, "candidates": cands, "cookies": "; ".join(cookies), "page_url": page_url}


def _try_play(page) -> None:
    """Her çerçevede (iframe) videoları sessiz başlat, oynat düğmelerine bas, çerez onayını kapat."""
    for frame in page.frames:
        try:
            frame.evaluate("""() => { document.querySelectorAll('video,audio').forEach(v => {
                try { v.muted = true; const p = v.play(); if (p) p.catch(() => {}); } catch (e) {} }); }""")
        except Exception:  # noqa: BLE001
            pass
    for sel in ("button:has-text('Accept')", "button:has-text('Kabul')", "button:has-text('Agree')",
                "button:has-text('Принять')", "#onetrust-accept-btn-handler"):
        try:
            page.locator(sel).first.click(timeout=500)
        except Exception:  # noqa: BLE001
            pass
    for frame in page.frames:
        for sel in PLAY_SELECTORS:
            try:
                loc = frame.locator(sel).first
                if loc.count() and loc.is_visible():
                    loc.click(timeout=700, force=True)
                    break
            except Exception:  # noqa: BLE001
                continue
