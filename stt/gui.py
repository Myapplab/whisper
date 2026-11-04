"""STT masaüstü uygulaması (pywebview + Edge WebView2).

Çalıştırma:  pythonw -m stt.gui      (masaüstündeki STT kısayolu bunu açar)
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from . import __version__
from .config import TOOL_DIR, Config, load_config
from .sources import expand_inputs, is_url
from .util import log

HTML_PATH = Path(__file__).with_name("gui.html")
URL_RE = re.compile(r"https?://[^\s<>\"']+")
SETTING_KEYS = ("out_dir", "hf_token", "fpl_repo", "cookies_browser", "lang", "diarize", "fpl_auto", "gpus")


def _redirect_std() -> Path:
    """pythonw'da stdout/stderr yok: günlük dosyasına yönlendir (tqdm vb. çökmesin)."""
    logdir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "stt"
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / "stt_app.log"
    if sys.stdout is None or sys.stderr is None:
        f = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f
    return path


def open_path(p: str) -> None:
    if not p:
        return
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])  # noqa: S603,S607
    else:
        subprocess.Popen(["xdg-open", p])  # noqa: S603,S607


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "") + "'"  # literal string: ters bölü kaçışı gerekmez


def save_settings(updates: dict, path: Path = TOOL_DIR / "stt.toml") -> None:
    data = {}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            data = {}
    for k, v in updates.items():
        if v is None or v == "":
            data.pop(k, None)
        else:
            data[k] = v
    lines = ["# stt ayarları (uygulamadan kaydedildi)"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in data.items() if isinstance(v, (str, int, float, bool))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _UiLogHandler(logging.Handler):
    def __init__(self, app: "App"):
        super().__init__(logging.INFO)
        self.app = app
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.app._push({"type": "log", "level": record.levelname, "text": self.format(record)})
        except Exception:  # noqa: BLE001
            pass


class App:
    """JS'in çağırdığı API (window.pywebview.api.*) + arka planda çalışan motor."""

    def __init__(self):
        self._window = None
        self._cfg: Config = load_config()
        self._pipe = None
        self._inbox: queue.Queue = queue.Queue()
        self._restart = threading.Event()
        self._stop = threading.Event()
        self._engine_state = {"state": "starting", "devices": [], "error": None}
        self._events: list[dict] = []          # pencere hazır olmadan gelen olaylar
        self._ready = False
        self._lock = threading.Lock()

    # ---------------------------------------------------------- JS'e olay
    def _push(self, ev: dict) -> None:
        if ev.get("type") == "worker":  # son durum: sayfa yeniden açılırsa GPU kartları doğru görünsün
            for d in self._engine_state.get("devices", []):
                if d.get("wid") == ev.get("wid"):
                    d.update({k: ev.get(k) for k in ("state", "task", "title", "error")})
        with self._lock:
            if not self._ready or self._window is None:
                self._events.append(ev)
                return
        try:
            self._window.evaluate_js(f"window.__stt && window.__stt.onEvent({json.dumps(ev, default=str)})")
        except Exception:  # noqa: BLE001
            pass

    def _engine(self, **kw) -> None:
        self._engine_state.update(kw)
        self._push({"type": "engine", **self._engine_state})

    # ---------------------------------------------------------- motor döngüsü
    def _runner(self) -> None:
        from .pipeline import Pipeline
        from .engines import gpu_table
        while not self._stop.is_set():
            self._restart.clear()
            self._engine(state="loading", error=None, devices=[])
            try:
                self._pipe = Pipeline(self._cfg)
                self._pipe.events = self._push
                self._pipe.start()
                gt = gpu_table()
                devs = []
                ds = self._pipe.devices or ["api"]
                for i, d in enumerate(ds):
                    name, gb = ("CPU", 0) if d == "cpu" else ("OpenAI API", 0) if d == "api" else \
                        gt.get(int(d.split(":")[1]), (d, 0))
                    wid = d if ds.count(d) == 1 else f"{d}#{i}"
                    devs.append({"wid": wid, "name": name, "gb": round(gb)})
                self._engine(state="ready", devices=devs)
            except Exception as e:  # noqa: BLE001
                log.exception("Motor başlatılamadı")
                self._engine(state="error", error=f"{type(e).__name__}: {e}")
                self._restart.wait()
                continue
            try:
                while not self._stop.is_set():
                    while True:
                        try:
                            item = self._inbox.get_nowait()
                        except queue.Empty:
                            break
                        self._pipe.submit(item)
                    self._pipe.pump(0.3)
                    if self._restart.is_set() and not self._pipe.active():
                        break
            except Exception as e:  # noqa: BLE001
                log.error("Motor durdu: %s", e)
                self._engine(state="error", error=str(e))
            finally:
                try:
                    self._pipe.close()
                except Exception:  # noqa: BLE001
                    pass
            if not self._restart.is_set() and not self._stop.is_set():
                self._restart.wait()

    # ---------------------------------------------------------- API (JS'ten)
    def hello(self) -> dict:
        with self._lock:
            self._ready = True
            pending, self._events = self._events, []
        for ev in pending[-300:]:
            self._push(ev)
        return {"version": __version__, "settings": self.get_settings(), "engine": self._engine_state,
                "dataset": self.dataset()}

    def get_settings(self) -> dict:
        c = self._cfg
        return {"out_dir": str(Path(c.out_dir).resolve()), "hf_token_set": bool(c.hf_token),
                "fpl_repo": str(c.fpl_repo or ""), "fpl_ok": bool(c.fpl_repo and (Path(c.fpl_repo) / "presser.py").is_file()),
                "cookies_browser": c.cookies_browser or "none", "lang": c.lang or "auto", "diarize": bool(c.diarize),
                "fpl_auto": bool(c.fpl_auto), "gpus": c.gpus}

    def save_settings(self, s: dict) -> dict:
        upd = {}
        if s.get("out_dir"):
            upd["out_dir"] = s["out_dir"].replace("\\", "/")
        if s.get("hf_token"):
            upd["hf_token"] = s["hf_token"].strip()
        if "fpl_repo" in s:
            upd["fpl_repo"] = (s.get("fpl_repo") or "").replace("\\", "/")
        if s.get("cookies_browser"):
            upd["cookies_browser"] = s["cookies_browser"]
        for k in ("diarize", "fpl_auto"):
            if k in s:
                upd[k] = bool(s[k])
        if s.get("lang"):
            upd["lang"] = s["lang"]
        save_settings(upd)
        old = (str(self._cfg.out_dir), self._cfg.hf_token)
        self._cfg = load_config()
        if (str(self._cfg.out_dir), self._cfg.hf_token) != old:
            self._restart.set()  # işler bitince motor yeni ayarlarla açılır
        return self.get_settings()

    def pick_files(self) -> list[str]:
        import webview
        kind = getattr(getattr(webview, "FileDialog", None), "OPEN", None) or webview.OPEN_DIALOG
        res = self._window.create_file_dialog(kind, allow_multiple=True, file_types=(
            "Video / ses (*.mp4;*.mkv;*.mov;*.webm;*.avi;*.m4v;*.mp3;*.m4a;*.wav;*.flac;*.ogg;*.opus;*.aac)",
            "Link listesi (*.txt)", "Tüm dosyalar (*.*)"))
        return list(res or [])

    def pick_folder(self) -> str:
        import webview
        kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
        res = self._window.create_file_dialog(kind)
        return (res[0] if isinstance(res, (list, tuple)) else res) or ""

    def start(self, text: str, paths: list[str], opts: dict) -> dict:
        inputs = [u.rstrip(").,;") for u in URL_RE.findall(text or "")]
        for line in (text or "").splitlines():
            line = line.strip().strip('"')
            if line and not is_url(line) and not URL_RE.search(line) and Path(line).exists():
                inputs.append(line)
        inputs += [p for p in (paths or []) if p]
        items = expand_inputs(inputs)
        for it in items:
            it.extra.update({"lang": opts.get("lang") or "auto", "diarize": bool(opts.get("diarize")),
                             "fpl_auto": bool(opts.get("fpl_auto"))})
            self._inbox.put(it)
        warn = None
        if opts.get("diarize") and not self._cfg.hf_token:
            warn = "Konuşmacı ayırma için HuggingFace token gerekli (Ayarlar). Bu işler konuşmacısız yazılacak."
        return {"count": len(items), "warning": warn}

    def open(self, p: str) -> None:
        open_path(p)

    def open_out(self) -> None:
        Path(self._cfg.out_dir).mkdir(parents=True, exist_ok=True)
        open_path(str(Path(self._cfg.out_dir).resolve()))

    def read_text(self, p: str) -> str:
        try:
            return Path(p).read_text(encoding="utf-8")[:200_000]
        except Exception as e:  # noqa: BLE001
            return f"(okunamadı: {e})"

    def retry_engine(self) -> None:
        self._restart.set()

    def dataset(self) -> dict:
        root = Path(self._cfg.out_dir).resolve()
        rows = []
        jl = root / "index.jsonl"
        if jl.exists():
            for line in jl.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
        tree: dict = {}
        for r in rows:
            league = r.get("league") or "Diger"
            team = r.get("team") or "Bilinmeyen"
            txt = r.get("txt") or ""
            folder = str((root / txt).parent) if txt else str(root)
            m = tree.setdefault(league, {}).setdefault(team, {}).setdefault(folder, {
                "folder": folder, "name": Path(folder).name, "match_date": r.get("match_date"), "gw": r.get("gw"),
                "pressers": []})
            m["pressers"].append({"txt": str(root / txt) if txt else "", "manager": r.get("manager"),
                                  "phase": r.get("phase"), "date": r.get("presser_date"),
                                  "questions": r.get("questions"), "diarized": r.get("diarized")})
        out = []
        for league in sorted(tree, key=lambda x: (x != "Premier League", x)):
            teams = []
            for team in sorted(tree[league]):
                matches = sorted(tree[league][team].values(), key=lambda m: m.get("match_date") or "", reverse=True)
                teams.append({"team": team, "folder": str(Path(matches[0]["folder"]).parent), "matches": matches})
            out.append({"league": league, "folder": str(root / ("Premier League" if league == "Premier League"
                                                                else "Diger Takimlar")), "teams": teams})
        return {"root": str(root), "count": len(rows), "leagues": out,
                "index_csv": str(root / "index.csv") if (root / "index.csv").exists() else ""}


def _hide_child_consoles() -> None:
    """pythonw'dan açılan ffmpeg / nvidia-smi / presser.py vb. siyah pencere açmasın."""
    if sys.platform != "win32":
        return
    orig = subprocess.Popen.__init__

    def init(self, *a, **kw):
        kw["creationflags"] = kw.get("creationflags", 0) | 0x08000000  # CREATE_NO_WINDOW
        orig(self, *a, **kw)

    subprocess.Popen.__init__ = init


def main() -> int:
    os.chdir(TOOL_DIR)  # göreli 'cikti' klasörü hep C:\stt\cikti olsun
    logpath = _redirect_std()
    _hide_child_consoles()
    from .util import add_tool_paths
    add_tool_paths()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(logpath, encoding="utf-8")])
    from .util import quiet_warnings
    quiet_warnings()
    import webview

    app = App()
    log.addHandler(_UiLogHandler(app))
    log.info("STT %s başlıyor (günlük: %s)", __version__, logpath)
    app._window = webview.create_window(
        "STT — Basın Odası", url=str(HTML_PATH), js_api=app, width=1360, height=860, min_size=(1000, 640),
        background_color="#0f1318", text_select=True)

    def on_closing():
        app._stop.set()
        app._restart.set()

    app._window.events.closing += on_closing

    def on_loaded():
        try:  # dosyaları pencereye sürükleyip bırakma
            from webview.dom import DOMEventHandler

            def on_drop(e):
                files = (e.get("dataTransfer") or {}).get("files") or []
                paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
                if paths:
                    app._push({"type": "dropped", "paths": paths})

            doc = app._window.dom.document
            doc.events.dragover += DOMEventHandler(lambda e: None, True, True)
            doc.events.drop += DOMEventHandler(on_drop, True, True)
        except Exception:  # noqa: BLE001
            log.debug("sürükle-bırak etkin değil", exc_info=True)

    app._window.events.loaded += on_loaded
    threading.Thread(target=app._runner, name="motor", daemon=True).start()
    webview.start(debug=bool(os.environ.get("STT_DEBUG")))
    app._stop.set()
    time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
