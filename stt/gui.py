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
        self._engine_state = {"state": "starting", "devices": [], "spare": [], "error": None}
        self._outbox: list[dict] = []          # JS'e gidecek olaylar (ayrı iş parçacığı gönderir)
        self._ready = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._sender_started = False
        self._logpath: Path | None = None

    # ---------------------------------------------------------- JS'e olay
    MAX_OUTBOX = 3000

    def _push(self, ev: dict) -> None:
        """Olayı kuyruğa koy ve HEMEN dön.

        evaluate_js pencere iş parçacığını bekler; pencere (ya da ekran kartı) takılırsa doğrudan çağıran
        indirme/motor iş parçacıkları da kilitlenirdi ve uygulama tamamen donardı. Artık yalnızca
        _sender iş parçacığı bekler; iş arka planda sürer.
        """
        if ev.get("type") == "fatal":  # tüm işçiler başlatılamadı: motor hata durumunda görünsün
            self._engine_state.update(state="error", error=ev.get("error"))
        if ev.get("type") == "worker":  # son durum: sayfa yeniden açılırsa GPU kartları doğru görünsün
            for d in self._engine_state.get("devices", []):
                if d.get("wid") == ev.get("wid"):
                    d.update({k: ev.get(k) for k in ("state", "task", "title", "error")})
        with self._lock:
            self._outbox.append(ev)
            if len(self._outbox) > self.MAX_OUTBOX:  # pencere yanıt vermiyor: önce günlük satırlarını at
                keep = [e for e in self._outbox if e.get("type") != "log"]
                self._outbox = keep[-self.MAX_OUTBOX:]
        self._wake.set()

    def _take_batch(self) -> list[dict]:
        with self._lock:
            if not self._ready or self._window is None or not self._outbox:
                return []
            batch, self._outbox = self._outbox[:400], self._outbox[400:]
        return batch

    def _sender(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            batch = self._take_batch()
            while batch:
                try:
                    self._window.evaluate_js(
                        f"window.__stt && window.__stt.onEvents({json.dumps(batch, default=str)})")
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.05)  # olayları toplu gönder: arayüz saniyede en fazla ~20 kez çizilir
                batch = self._take_batch()

    def _start_sender(self) -> None:
        if not self._sender_started:
            self._sender_started = True
            threading.Thread(target=self._sender, name="arayuz", daemon=True).start()

    def _engine(self, **kw) -> None:
        self._engine_state.update(kw)
        self._push({"type": "engine", **self._engine_state})

    # ---------------------------------------------------------- motor döngüsü
    def _runner(self) -> None:
        from .pipeline import Pipeline
        from .engines import gpu_info
        while not self._stop.is_set():
            self._restart.clear()
            self._engine(state="loading", error=None, devices=[], spare=[])
            try:
                self._pipe = Pipeline(self._cfg)
                self._pipe.events = self._push
                self._pipe.start()
                gi = gpu_info()
                devs = []
                ds = self._pipe.devices or ["api"]
                used = set()
                for i, d in enumerate(ds):
                    g = {"name": "CPU", "gb": 0} if d == "cpu" else {"name": "OpenAI API", "gb": 0} if d == "api" \
                        else gi.get(int(d.split(":")[1]), {"name": d, "gb": 0})
                    if d.startswith("cuda"):
                        used.add(int(d.split(":")[1]))
                    wid = d if ds.count(d) == 1 else f"{d}#{i}"
                    devs.append({"wid": wid, "name": g["name"], "gb": round(g["gb"]), "display": bool(g.get("display"))})
                spare = [{"name": g["name"], "gb": round(g["gb"]), "display": bool(g.get("display"))}
                         for i, g in sorted(gi.items()) if i not in used and self._pipe.devices
                         and self._pipe.devices[0] != "api"]
                self._engine(state="ready", devices=devs, spare=spare)
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
                    if self._pipe.failed_reason:
                        raise RuntimeError(self._pipe.failed_reason)
                    if self._restart.is_set() and not self._pipe.active():
                        break
            except Exception as e:  # noqa: BLE001
                log.error("Motor durdu: %s", e)
                self._engine(state="error", error=str(e))
                self._restart.clear()  # 'Yeniden başlat' düğmesini bekle
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
            # sayfa (yeniden) açıldı: birikmiş günlük satırlarından yalnızca son 300'ü
            logs = [e for e in self._outbox if e.get("type") == "log"][-300:]
            self._outbox = [e for e in self._outbox if e.get("type") != "log" and e.get("type") != "engine"] + logs
        self._start_sender()
        self._wake.set()
        if not getattr(self, "_upd_checked", False):  # açılışta bir kez, arka planda
            self._upd_checked = True
            threading.Thread(target=self._auto_update_check, name="guncelleme", daemon=True).start()
        return {"version": __version__, "settings": self.get_settings(), "engine": self._engine_state,
                "dataset": self.dataset()}

    # ---------------------------------------------------------- güncelleme
    def _auto_update_check(self) -> None:
        time.sleep(3)
        info = self.update_check()
        if info.get("available"):
            log.info("Yeni sürüm var: %s (şu an %s)", info["latest"], info["current"])

    def update_check(self) -> dict:
        from . import updater
        info = updater.check()
        self._push({"type": "update", "stage": "available" if info.get("available") else "none", **info})
        return info

    def update_apply(self) -> dict:
        from . import updater
        if self._pipe is not None and self._pipe.active():
            return {"ok": False, "error": "Önce kuyruktaki işlerin bitmesini bekleyin (ya da Durdur'a basın), sonra güncelleyin."}
        if getattr(self, "_updating", False):
            return {"ok": False, "error": "Güncelleme zaten sürüyor."}
        self._updating = True
        try:
            res = updater.apply(lambda stage, text: self._push({"type": "update", "stage": stage, "text": text}))
        finally:
            self._updating = False
        if res.get("ok"):
            self._push({"type": "update", "stage": "restart", "text": f"STT {res['version']} açılıyor…"})

            def later():
                time.sleep(1.5)
                try:
                    updater.relaunch()
                finally:
                    _shutdown(self)
            threading.Thread(target=later, daemon=True).start()
        else:
            self._push({"type": "update", "stage": "error", "text": res.get("error")})
        return res

    def get_settings(self) -> dict:
        c = self._cfg
        return {"out_dir": str(Path(c.out_dir).resolve()), "hf_token_set": bool(c.hf_token),
                "fpl_repo": str(c.fpl_repo or ""), "fpl_ok": bool(c.fpl_repo and (Path(c.fpl_repo) / "presser.py").is_file()),
                "cookies_browser": c.cookies_browser or "none", "lang": c.lang or "auto", "diarize": bool(c.diarize),
                "fpl_auto": bool(c.fpl_auto), "gpus": "all" if (c.gpus or "").lower() == "all" else
                "auto" if (c.gpus or "auto").lower() == "auto" else c.gpus}

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
        if s.get("gpus") in ("auto", "all"):
            upd["gpus"] = s["gpus"]
        save_settings(upd)
        old = (str(self._cfg.out_dir), self._cfg.hf_token, self._cfg.gpus)
        self._cfg = load_config()
        if (str(self._cfg.out_dir), self._cfg.hf_token, self._cfg.gpus) != old:
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
        import uuid
        for it in items:
            it.extra.setdefault("uid", uuid.uuid4().hex[:10])
            it.extra.update({"lang": opts.get("lang") or "auto", "diarize": bool(opts.get("diarize")),
                             "fpl_auto": bool(opts.get("fpl_auto"))})
            self._push({"type": "queued", "uid": it.extra["uid"], "source": it.source, "title": it.title})
            self._inbox.put(it)
        warn = None
        if opts.get("diarize") and not self._cfg.hf_token:
            warn = "Konuşmacı ayırma için HuggingFace token gerekli (Ayarlar). Bu işler konuşmacısız yazılacak."
        if self._engine_state.get("state") == "error":
            warn = ("Motor çalışmıyor: " + str(self._engine_state.get("error") or "")[:200]
                    + " — İşler sırada bekliyor; üstteki 'Yeniden başlat'a basın ya da 'Kontrol'e bakın.")
        if not items:
            warn = "Link ya da dosya bulunamadı."
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
        self._cfg = load_config()
        self._restart.set()

    def cancel_all(self) -> dict:
        """Sıradaki ve çalışan bütün işleri durdur."""
        n = 0
        while True:  # henüz motora geçmemiş işler
            try:
                it = self._inbox.get_nowait()
            except queue.Empty:
                break
            n += 1
            self._push({"type": "failed", "uid": it.extra.get("uid"), "source": it.source, "title": it.title,
                        "error": "Durduruldu.", "cancelled": True})
        if self._pipe is not None:
            n += self._pipe.cancel_all()
        return {"count": n}

    def check(self) -> dict:
        """Kurulum kontrolü (stt kontrol) ayrı süreçte; çıktıyı pencerede göstermek için döndürür."""
        exe = Path(sys.executable)
        py = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
        if not py.exists():
            py = exe
        try:
            r = subprocess.run([str(py), "-m", "stt", "kontrol"], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=900, cwd=str(TOOL_DIR))
        except subprocess.TimeoutExpired:
            return {"ok": False, "text": "Kontrol 15 dakikada bitmedi."}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "text": f"Kontrol çalıştırılamadı: {type(e).__name__}: {e}"}
        text = r.stdout or ""
        if r.returncode not in (0, 1) and r.stderr:
            text += "\n" + r.stderr[-4000:]
        log.info("Kontrol bitti (%s)", "sorun yok" if r.returncode == 0 else "sorun var")
        return {"ok": r.returncode == 0, "text": text}

    def open_log(self) -> None:
        if self._logpath:
            open_path(str(self._logpath))

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


def _single_instance() -> bool:
    """Windows: uygulama zaten açıksa False. İki kopya iki kez model yükler (RAM/VRAM biter, bilgisayar donar)."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # güncelleme sonrası yeniden açılış: eski pencere kapanana kadar (en çok 30 sn) bekle
        tries = 60 if os.environ.pop("STT_RESTART", None) else 1
        for _ in range(tries):
            _single_instance.handle = k32.CreateMutexW(None, False, "Local\\STT-Basin-Odasi")  # type: ignore[attr-defined]
            if k32.GetLastError() != 183:
                break
            if tries > 1:
                k32.CloseHandle(_single_instance.handle)
                time.sleep(0.5)
        if k32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                None, "STT zaten açık. Görev çubuğundaki STT penceresini kullanın.\n\n"
                "Pencere görünmüyorsa birkaç saniye bekleyip tekrar deneyin.", "STT - Basın Odası", 0x40)
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _shutdown(app: "App") -> None:
    """Pencere kapandı: GPU işçilerini durdur ve süreci HEMEN bitir.

    Yarım kalan indirme iş parçacıkları normal çıkışı dakikalarca bekletirdi; o sırada uygulama yeniden
    açılınca iki kopya aynı anda ekran kartına model yüklüyordu.
    """
    app._stop.set()
    app._restart.set()
    pipe = app._pipe
    try:
        if pipe is not None and pipe.pool is not None and not pipe.pool.threads:
            for w in pipe.pool.workers.values():
                if w.is_alive():
                    w.terminate()
            for w in pipe.pool.workers.values():
                w.join(3)
    except Exception:  # noqa: BLE001
        pass
    logging.shutdown()
    os._exit(0)


def main() -> int:
    if not _single_instance():
        return 0
    os.chdir(TOOL_DIR)  # göreli 'cikti' klasörü hep C:\stt\cikti olsun
    logpath = _redirect_std()
    _hide_child_consoles()
    from .util import prepare_env
    prepare_env()  # ffmpeg / deno / GPU sırası / UTF-8: stt.bat'a gerek yok
    os.environ["PYTHONUTF8"] = "1"  # alt süreçler (GPU işçileri, ffmpeg, presser.py) Türkçe harfleri yazabilsin
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(logpath, encoding="utf-8")])
    from .util import quiet_warnings
    quiet_warnings()
    import webview

    app = App()
    app._logpath = logpath
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
    _shutdown(app)
    return 0


if __name__ == "__main__":
    sys.exit(main())
