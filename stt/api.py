"""Yerel HTTP API: haber sitesi / günlük haber radarı gibi başka programlar STT'ye iş verip sonuç alabilsin.

Açmak: Ayarlar > 'Yerel API' (port, ör. 8765) ya da `stt serve --port 8765`. Yalnızca bu bilgisayardan
(127.0.0.1) erişilir. api_token ayarlıysa her istekte `X-STT-Token: <token>` başlığı (ya da ?token=) gerekir.

  GET  /api/status                      sürüm, motor, kuyruk {total, working, waiting}, accepting, busy, paused,
                                        auto_mode, flex, memory, stats (son 1 saat), notices (son bildirimler)
  POST /api/jobs   {"inputs": ["https://..."], "options": {"lang": "en", "diarize": true,
                    "speaker_mode": "auto|basin|herkes", "news": true, "live": false, "collection": "ad"}}
                                        -> 200 {"ok": true, "count", "uids": [...]}; 503 {"error", "retry": true}
                                        (eğitim / yeniden sınıflandırma / çökme döngüsü: sonra tekrar gönderin;
                                        link KUYRUĞA GİRMEDİ); 422 kanal / oynatma listesi linki; 400 geçersiz istek
  POST /api/lookup {"urls": [...]}  /  GET /api/lookup?url=...
                                        her link: status done | failed | waiting | working | interrupted | cancelled
                                        | missing | unknown, out_dir, txt, json, error, retryable (1.9.27)
  GET  /api/jobs                        son işler ve durumları (txt/json yolları dahil)
  GET  /api/results?limit=50            veri seti satırları (index.jsonl), yeniden eskiye
  GET  /api/text?path=...               çıktı klasöründeki bir metin dosyası
  POST /api/live/stop {"uid": "..."}    canlı yayın kaydını bitir
  POST /api/channel {"url": "...", "topic": "basin", "custom": "", "limit": 50, "start": false}
                                        kanal/liste videoları (+ konu süzgeci); start=true ise uyanları kuyruğa koyar
Her biten iş için ayrıca webhook_url'e JSON POST edilir (Ayarlar).

Güvenlik (1.9.27): yalnız bu bilgisayardaki programlar. Tarayıcıdaki bir web sayfasından gelen istek (Origin /
Sec-Fetch-Site) ve DNS yeniden bağlama (Host 127.0.0.1 / localhost değil) 403 alır; yoksa açık bir sayfa GPU işi
ekleyebilir, metinleri okuyabilirdi (TARAMA_2026-11-06 C2).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .util import log

ROUTES = {"durum": "status", "is": "jobs", "isler": "jobs", "sonuclar": "results", "metin": "text",
          "kanal": "channel", "canli/durdur": "live/stop", "sorgu": "lookup"}
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
# kalıcı hata (yeniden göndermek boşuna): video yok / gizli / DRM / bölgeye kapalı / ses yok
PERMANENT_ERRORS = ("kaldırılmış", "gizli ya da yalnızca üyelere", "drm", "ülkeye kapalı", "coğrafi kısıtlama",
                    "indirilebilir video bulunamadı", "ses yok", "private video", "video unavailable",
                    "has been removed", "members-only")


def retryable_error(err: str) -> bool:
    """Başarısız işi bir süre sonra yeniden göndermek işe yarar mı (ağ, bot doğrulaması, bellek, motor hatası: evet;
    video kaldırılmış / gizli / DRM: hayır)."""
    e = (err or "").lower()
    return not any(x in e for x in PERMANENT_ERRORS)


def foreign_request(headers) -> str:
    """İstek bu bilgisayardaki bir programdan değilse sebep ("" = yerel program). Tarayıcı her başka-site isteğinde
    Origin / Sec-Fetch-Site gönderir; yerel programlar (Hermes, curl) göndermez."""
    host = (headers.get("Host") or "").strip().lower()
    if host:
        name = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
        if name not in LOCAL_HOSTS:
            return "Host"
    origin = (headers.get("Origin") or "").strip().lower()
    if origin and (urlsplit(origin).hostname or "") not in LOCAL_HOSTS:
        return "Origin"  # "null" (sandbox / dosya) da
    if (headers.get("Sec-Fetch-Site") or "").strip().lower() == "cross-site":
        return "Sec-Fetch-Site"
    return ""


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        server_version = f"STT/{__version__}"

        def log_message(self, *a):  # sessiz
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self, q) -> bool:
            tok = app._cfg.api_token
            if not tok:
                return True
            got = self.headers.get("X-STT-Token") or (q.get("token") or [""])[0]
            return got == tok

        def _foreign(self) -> bool:
            why = foreign_request(self.headers)
            if why:
                log.warning("Yerel API: başka bir siteden gelen istek reddedildi (%s)", why)
                self._send(403, {"error": "Yerel API yalnız bu bilgisayardaki programlara açık (web sayfalarına değil)."})
            return bool(why)

        def _route(self) -> tuple[str, dict]:
            u = urlsplit(self.path)
            parts = [p for p in u.path.split("/") if p]
            if not parts or parts[0] != "api":
                return "", {}
            name = "/".join(parts[1:])
            name = ROUTES.get(name, name)
            return name, parse_qs(u.query)

        def do_GET(self):  # noqa: N802
            if self._foreign():
                return None
            name, q = self._route()
            if not self._auth(q):
                return self._send(401, {"error": "token gerekli"})
            try:
                if name == "status":
                    return self._send(200, app.api_status())
                if name == "lookup":
                    return self._send(200, {"items": app.lookup(q.get("url") or [])})
                if name == "jobs":
                    return self._send(200, {"jobs": app.jobs_list()})
                if name == "results":
                    lim = int((q.get("limit") or ["100"])[0])
                    return self._send(200, {"results": app.results(lim)})
                if name == "text":
                    p = (q.get("path") or [""])[0]
                    root = Path(app._cfg.out_dir).resolve()
                    rp = Path(p).resolve()
                    if root not in rp.parents:
                        return self._send(403, {"error": "yalnızca çıktı klasörü"})
                    return self._send(200, {"path": str(rp), "text": rp.read_text(encoding="utf-8")})
                return self._send(404, {"error": "bilinmeyen adres", "see": "/api/status"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):  # noqa: N802
            if self._foreign():
                return None
            name, q = self._route()
            if not self._auth(q):
                return self._send(401, {"error": "token gerekli"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                except ValueError:
                    return self._send(400, {"ok": False, "error": "Geçersiz JSON"})
                if not isinstance(body, dict):
                    return self._send(400, {"ok": False, "error": "JSON nesnesi bekleniyor"})
                if name == "jobs":
                    inputs = body.get("inputs") or ([body["url"]] if body.get("url") else [])
                    if isinstance(inputs, str):
                        inputs = [inputs]
                    opts = body.get("options") or {}
                    paused = getattr(app, "_auto_paused", "")
                    if paused:  # çökme döngüsü: kullanıcı bakana dek ajan yeni iş eklemesin
                        return self._send(503, {"ok": False, "error": paused, "retry": True})
                    r = app.start("\n".join(str(x) for x in inputs), [], opts if isinstance(opts, dict) else {})
                    if r.get("ok") is False:  # eğitim / yeniden sınıflandırma: iş eklenmedi, sonra tekrar gönderilmeli
                        return self._send(503, {**r, "retry": True})
                    if not r.get("count"):
                        return self._send(422 if r.get("collections") else 400,
                                          {**r, "ok": False, "error": r.get("warning") or "Link bulunamadı."})
                    return self._send(200, {"ok": True, **r})
                if name == "lookup":
                    urls = body.get("urls") or ([body["url"]] if body.get("url") else [])
                    return self._send(200, {"items": app.lookup(urls if isinstance(urls, list) else [urls])})
                if name == "live/stop":
                    return self._send(200, {"ok": app.stop_live(body.get("uid", ""))})
                if name == "channel":
                    res = app.list_channel(body.get("url", ""), body)
                    if body.get("start") and res.get("items"):
                        urls = [e["url"] for e in res["items"] if e.get("match")]
                        res["started"] = app.start_selected(urls, {**body, "collection": body.get("collection")
                                                                   or res.get("collection")})
                    return self._send(200, res)
                return self._send(404, {"error": "bilinmeyen adres"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def start_api(app, port: int) -> ThreadingHTTPServer | None:
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", int(port)), make_handler(app))
    except (OSError, OverflowError, ValueError, TypeError) as e:
        log.warning("Yerel API açılamadı (port %s): %s", port, e)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="api", daemon=True).start()
    log.info("Yerel API: http://127.0.0.1:%d/api/status", port)
    return srv
