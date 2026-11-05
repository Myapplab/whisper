"""Yerel HTTP API: haber sitesi / günlük haber radarı gibi başka programlar STT'ye iş verip sonuç alabilsin.

Açmak: Ayarlar > 'Yerel API' (port, ör. 8765) ya da `stt serve --port 8765`. Yalnızca bu bilgisayardan
(127.0.0.1) erişilir. api_token ayarlıysa her istekte `X-STT-Token: <token>` başlığı (ya da ?token=) gerekir.

  GET  /api/status                      sürüm, motor, kuyruk
  POST /api/jobs   {"inputs": ["https://..."], "options": {"lang": "en", "diarize": true,
                    "speaker_mode": "auto|basin|herkes", "news": true, "live": false, "collection": "ad"}}
                                        -> {"uids": [...]}
  GET  /api/jobs                        son işler ve durumları (txt/json yolları dahil)
  GET  /api/results?limit=50            veri seti satırları (index.jsonl), yeniden eskiye
  GET  /api/text?path=...               çıktı klasöründeki bir metin dosyası
  POST /api/live/stop {"uid": "..."}    canlı yayın kaydını bitir
  POST /api/channel {"url": "...", "topic": "basin", "custom": "", "limit": 50, "start": false}
                                        kanal/liste videoları (+ konu süzgeci); start=true ise uyanları kuyruğa koyar
Her biten iş için ayrıca webhook_url'e JSON POST edilir (Ayarlar).
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
          "kanal": "channel", "canli/durdur": "live/stop"}


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

        def _route(self) -> tuple[str, dict]:
            u = urlsplit(self.path)
            parts = [p for p in u.path.split("/") if p]
            if not parts or parts[0] != "api":
                return "", {}
            name = "/".join(parts[1:])
            name = ROUTES.get(name, name)
            return name, parse_qs(u.query)

        def do_GET(self):  # noqa: N802
            name, q = self._route()
            if not self._auth(q):
                return self._send(401, {"error": "token gerekli"})
            try:
                if name == "status":
                    return self._send(200, {"version": __version__, "engine": app._engine_state,
                                            "jobs": app.job_counts()})
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
            name, q = self._route()
            if not self._auth(q):
                return self._send(401, {"error": "token gerekli"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if name == "jobs":
                    inputs = body.get("inputs") or ([body["url"]] if body.get("url") else [])
                    opts = body.get("options") or {}
                    r = app.start("\n".join(inputs), [], opts)
                    return self._send(200, r)
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
    except OSError as e:
        log.warning("Yerel API açılamadı (port %s): %s", port, e)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="api", daemon=True).start()
    log.info("Yerel API: http://127.0.0.1:%d/api/status", port)
    return srv
