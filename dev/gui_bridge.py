# GUI testi: pywebview yerine HTTP köprüsü (aynı App sınıfı, aynı HTML)
import json, queue, sys, threading, re, logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import stt.gui as g
from stt.util import log

def main():
    events = queue.Queue()
    class FakeWindow:
        def evaluate_js(self, js):
            m = re.search(r"onEvents?\((.*)\)\s*$", js, re.S)
            if m: events.put(m.group(1))
    app = g.App(); app._window = FakeWindow()
    log.setLevel(logging.INFO); log.addHandler(g._UiLogHandler(app))
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    threading.Thread(target=app._runner, daemon=True).start()
    html = g.HTML_PATH.read_text(encoding="utf-8")
    SHIM = """<script>
    window.pywebview = {api: new Proxy({}, {get: (_, n) => (...a) => fetch('/api/' + n, {method: 'POST', body: JSON.stringify(a)}).then(r => r.json())})};
    const es = new EventSource('/events'); es.onmessage = e => { const d = JSON.parse(e.data); window.__stt && (Array.isArray(d) ? window.__stt.onEvents(d) : window.__stt.onEvent(d)); };
    </script>"""
    html = html.replace("<script>\n(() => {", SHIM + "\n<script>\n(() => {", 1)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path == "/events":
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
                while True:
                    try: data = events.get(timeout=15)
                    except queue.Empty: data = None
                    try:
                        self.wfile.write((f"data: {data}\n\n" if data else ": ping\n\n").encode()); self.wfile.flush()
                    except Exception: return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(html.encode())
        def do_POST(self):
            name = self.path.split("/")[-1]; args = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "[]")
            res = getattr(app, name)(*args)
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(res, default=str).encode())
    ThreadingHTTPServer(("127.0.0.1", 8799), H).serve_forever()

if __name__ == '__main__':
    main()
