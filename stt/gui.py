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
from .config import SECRETS, TOOL_DIR, Config, env_secrets, load_config
from .dataset import GENERAL_DIR
from .sources import expand_inputs, is_url
from .util import console_python, log, read_jsonl

HTML_PATH = Path(__file__).with_name("gui.html")
URL_RE = re.compile(r"https?://[^\s<>\"']+")
UI_STATE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "stt" / "pencere.json"


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
    from .util import atomic_write_text
    atomic_write_text(path, "\n".join(lines) + "\n")  # yarıda kalırsa token'lar silinmesin


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


def _diarizen_installed() -> bool:
    try:
        from .diarizen import installed
        return installed()
    except Exception:  # noqa: BLE001
        return False


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
        self._items: dict = {}                 # uid -> Item ("Tekrar dene" için)
        self._jobs: dict[str, dict] = {}       # uid -> son durum (yerel API / sayaçlar)
        self._gpus: dict = {}                  # nvidia-smi kart listesi (Ayarlar'daki kart seçimi)
        self._api = None
        self._llm = None
        self._st = None
        self._pending: list[dict] = []
        try:  # uygulama kapanınca yarım kalan işler: açılışta "Devam et" önerilir
            self._pending = self._state().queue_all()
        except Exception:  # noqa: BLE001
            log.debug("bekleyen işler okunamadı", exc_info=True)

    def _state_dir(self) -> Path:
        return Path(self._cfg.out_dir).resolve() / ".stt"

    def _state(self):
        """Durum veritabanı: motor açıksa onunki, değilse tek bir bağlantı (her çağrıda yenisi açılmasın)."""
        if self._pipe is not None:
            return self._pipe.state
        if self._st is None or self._st.path.parent != self._state_dir():
            from .state import State
            self._st = State(self._state_dir())
        return self._st

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
        self._track(ev)
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

    def _track(self, ev: dict) -> None:
        uid = ev.get("uid")
        typ = ev.get("type")
        if not uid or typ in ("log", "worker", "engine"):
            return
        j = self._jobs.setdefault(uid, {"uid": uid, "source": ev.get("source"), "status": "queued"})
        if ev.get("title") and not str(ev["title"]).startswith("http"):
            j["title"] = ev["title"]
        st = {"queued": "queued", "prep": "downloading", "planned": "running", "progress": "running",
              "retrying": "retrying", "done": "done", "failed": "cancelled" if ev.get("cancelled") else "failed",
              "skipped": "skipped"}.get(typ)
        if typ == "live":
            st = "live" if ev.get("state") in ("start", "info", "text") else "finalizing"
            if ev.get("txt"):
                j["live_txt"] = ev["txt"]
        if st:
            j["status"] = st
        for k in ("txt", "json", "out_dir", "error", "duration", "dataset", "done", "total"):
            if ev.get(k) is not None:
                j[k] = ev[k]
        if len(self._jobs) > 2000:
            for k in list(self._jobs)[:500]:
                self._jobs.pop(k, None)

    def job_counts(self) -> dict:
        out: dict = {}
        for j in list(self._jobs.values()):
            out[j["status"]] = out.get(j["status"], 0) + 1
        return out

    def jobs_list(self) -> list[dict]:
        return list(self._jobs.values())[-300:]

    def results(self, limit: int = 100) -> list[dict]:
        return read_jsonl(Path(self._cfg.out_dir).resolve() / "index.jsonl")[-limit:][::-1]

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
                self._gpus = gi
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
            threading.Thread(target=self._follow_loop, name="takip", daemon=True).start()
        return {"version": __version__, "settings": self.get_settings(), "engine": self._engine_state,
                "dataset": self.dataset(), "pending": [{"uid": x["uid"], "title": x["title"], "source": x["source"]}
                                                       for x in self._pending],
                "follows": self.follow_list()}

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
        g = (c.gpus or "auto").lower()
        cards = [{"id": str(i), "name": v["name"], "gb": round(v["gb"]), "display": bool(v.get("display"))}
                 for i, v in sorted(self._gpus.items())]
        llm = self.llm()
        return {"out_dir": str(Path(c.out_dir).resolve()), "hf_token_set": bool(c.hf_token),
                "fpl_repo": str(c.fpl_repo or ""), "fpl_ok": bool(c.fpl_repo and (Path(c.fpl_repo) / "presser.py").is_file()),
                "cookies_browser": c.cookies_browser or "none", "cookies_file": c.cookies_file or "",
                "lang": c.lang or "auto", "diarize": bool(c.diarize), "fpl_auto": bool(c.fpl_auto),
                "gpus": g if g in ("auto", "all") else str(c.gpus), "gpu_cards": cards,
                "browser_capture": bool(c.browser_capture), "ui_gpu": bool(c.ui_gpu),
                "llm_provider": c.llm_provider or "none", "llm_model": c.llm_model or "",
                "gemini_key_set": bool(llm.key("gemini")), "anthropic_key_set": bool(llm.key("claude")),
                "openai_key_set": bool(llm.key("openai")),
                "llm_budget_usd": c.llm_budget_usd, "llm_news": bool(c.llm_news), "llm_qa": bool(c.llm_qa),
                "llm": llm.status(), "api_port": c.api_port or 0, "api_token_set": bool(c.api_token),
                "webhook_url": c.webhook_url or "", "follow_interval_h": c.follow_interval_h,
                "diar_engine": c.diar_engine or "pyannote", "diarizen_installed": _diarizen_installed(),
                "diar_agc": "auto" if c.diar_agc is None else ("on" if c.diar_agc else "off"),
                "secret_env": env_secrets()}  # {ayar: ortam değişkeni}: bunlar arayüzden silinemez

    def save_settings(self, s: dict) -> dict:
        upd: dict = {}
        if s.get("out_dir"):
            upd["out_dir"] = s["out_dir"].replace("\\", "/")
        for k in SECRETS:
            if s.get(k):  # boş = mevcut anahtar kalsın
                upd[k] = str(s[k]).strip()
        for k in s.get("clear") or []:  # "Sil" düğmesi: stt.toml'dan çıkar (ortam değişkenindeki kalır)
            if k in SECRETS:
                upd[k] = ""
        if "fpl_repo" in s:
            upd["fpl_repo"] = (s.get("fpl_repo") or "").replace("\\", "/")
        if "cookies_file" in s:
            upd["cookies_file"] = (s.get("cookies_file") or "").replace("\\", "/")
        if s.get("cookies_browser"):
            upd["cookies_browser"] = s["cookies_browser"]
        for k in ("diarize", "fpl_auto", "browser_capture", "ui_gpu", "llm_news", "llm_qa"):
            if k in s:
                upd[k] = bool(s[k])
        if s.get("lang"):
            upd["lang"] = s["lang"]
        g = str(s.get("gpus") or "").strip()
        if g in ("auto", "all") or re.fullmatch(r"\d+(,\d+)*", g):
            upd["gpus"] = g
        if "llm_provider" in s:
            upd["llm_provider"] = s["llm_provider"] if s["llm_provider"] in ("gemini", "claude", "openai") else ""
        if "llm_model" in s:
            upd["llm_model"] = (s.get("llm_model") or "").strip()
        for k, typ in (("llm_budget_usd", float), ("api_port", int), ("follow_interval_h", float)):
            if s.get(k) not in (None, ""):
                try:
                    upd[k] = typ(s[k])
                except (TypeError, ValueError):
                    pass
        if s.get("diar_engine") in ("pyannote", "diarizen"):
            upd["diar_engine"] = s["diar_engine"]
        if s.get("diar_agc") in ("auto", "on", "off"):  # auto = stt.toml'dan çıkar (metin tanımadaki agc izlenir)
            upd["diar_agc"] = {"on": True, "off": False}.get(s["diar_agc"], "")
        if "webhook_url" in s:
            upd["webhook_url"] = (s.get("webhook_url") or "").strip()
        save_settings(upd)
        old = (str(self._cfg.out_dir), self._cfg.hf_token, self._cfg.gpus, self._cfg.diar_engine, self._cfg.diar_agc)
        old_port = self._cfg.api_port
        self._cfg = load_config()
        self._llm = None
        if self._pipe is not None:
            # motorun açılışta kopyaladığı değerler (kart, model...) etkilenmez; onlar için aşağıda yeniden başlatılır
            self._pipe.cfg = self._cfg
            self._pipe._llm = None
        if (str(self._cfg.out_dir), self._cfg.hf_token, self._cfg.gpus, self._cfg.diar_engine,
                self._cfg.diar_agc) != old:
            self._restart.set()  # işler bitince motor yeni ayarlarla açılır
        if self._cfg.api_port != old_port:
            self._start_api()
        return self.get_settings()

    # ---------------------------------------------------------- yapay zekâ
    def llm(self):
        if self._pipe is not None:
            return self._pipe.llm()  # tek örnek: model listesi / istemci iki kez kurulmasın
        if self._llm is None:
            from .llm import LLM
            self._llm = LLM(self._cfg, self._state_dir())
        return self._llm

    def llm_test(self) -> dict:
        try:
            out = self.llm().complete("Kısa yanıt ver.", "Sadece 'tamam' yaz.", max_tokens=50)
            return {"ok": True, "text": out.strip()[:100], "status": self.llm().status()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)[:300]}

    def make_news(self, path: str) -> dict:
        """Var olan bir metin için haber taslağı (txt ya da json yolu)."""
        p = Path(path)
        js = p if p.suffix == ".json" else p.with_suffix(".json")
        if not js.exists():
            return {"ok": False, "error": "Bu metnin .json dosyası bulunamadı."}
        try:
            from .llm import write_news
            res = write_news(self.llm(), js)
            return {"ok": True, "news": str(res["news"])}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)[:300]}

    # ---------------------------------------------------------- yerel API
    def _start_api(self) -> None:
        if self._api is not None:
            try:
                self._api.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._api = None
        if self._cfg.api_port:
            from .api import start_api
            self._api = start_api(self, self._cfg.api_port)


    def _dialog(self, kind: str, **kw) -> list[str]:
        import webview
        k = getattr(getattr(webview, "FileDialog", None), kind, None) or getattr(webview, f"{kind}_DIALOG")
        res = self._window.create_file_dialog(k, **kw)
        return list(res) if isinstance(res, (list, tuple)) else [res] if res else []

    def pick_files(self) -> list[str]:
        return self._dialog("OPEN", allow_multiple=True, file_types=(
            "Video / ses (*.mp4;*.mkv;*.mov;*.webm;*.avi;*.m4v;*.mp3;*.m4a;*.wav;*.flac;*.ogg;*.opus;*.aac)",
            "Link listesi (*.txt)", "Tüm dosyalar (*.*)"))

    def pick_cookie_file(self) -> str:
        return next(iter(self._dialog("OPEN", file_types=("cookies.txt (*.txt)", "Tüm dosyalar (*.*)"))), "")

    def pick_folder(self) -> str:
        return next(iter(self._dialog("FOLDER")), "")

    def _item_opts(self, it, opts: dict) -> None:
        it.extra.update({"lang": opts.get("lang") or "auto", "diarize": bool(opts.get("diarize")),
                         "fpl_auto": bool(opts.get("fpl_auto")),
                         "speaker_mode": opts.get("speaker_mode") if opts.get("speaker_mode") in
                         ("auto", "basin", "herkes") else "auto"})
        if opts.get("news"):
            it.extra["llm_news"] = True
        if opts.get("live"):
            it.extra["live"] = True
        if opts.get("collection"):
            it.extra["collection"] = str(opts["collection"])[:120]
        try:
            n = int(opts.get("speakers") or 0)
        except (TypeError, ValueError):
            n = 0
        if 0 < n <= 20:
            it.extra["speakers"] = n

    def _enqueue(self, it, retry: bool = False) -> None:
        import uuid
        it.extra.setdefault("uid", uuid.uuid4().hex[:10])
        it.extra.pop("_gen", None)
        self._items[it.extra["uid"]] = it
        self._push({"type": "queued", "uid": it.extra["uid"], "source": it.source, "title": it.title,
                    "live": bool(it.extra.get("live")), "retry": retry})
        self._inbox.put(it)

    def _busy_msg(self) -> str | None:
        """Başlat'ı engelleyen iş: eğitim (ekran kartı tek işe) ya da yeniden sınıflandırma (klasörler taşınıyor)."""
        if getattr(self, "_training", False):
            return "Konuşmacı eğitimi sürüyor; birkaç dakika sonra tekrar Başlat'a basın."
        if getattr(self, "_reclassifying", False):
            return "Veri seti yeniden sınıflandırılıyor; birkaç saniye sonra tekrar Başlat'a basın."
        return None

    def start(self, text: str, paths: list[str], opts: dict) -> dict:
        msg = self._busy_msg()
        if msg:
            return {"ok": False, "count": 0, "error": msg, "warning": msg}
        from .channel import is_collection_url
        urls = [u.rstrip(").,;") for u in URL_RE.findall(text or "")]
        for line in (text or "").splitlines():  # "www.youtube.com/..." / "youtu.be/..." (https:// olmadan yapıştırılmış)
            line = line.strip().strip('"')
            if line and is_url(line) and not URL_RE.search(line):
                urls.append("https://" + line.split()[0].rstrip(").,;"))
        collections = [u for u in urls if is_collection_url(u)] if not opts.get("live") else []
        inputs = [u for u in urls if u not in collections]
        for line in (text or "").splitlines():
            line = line.strip().strip('"')
            if line and not is_url(line) and not URL_RE.search(line) and Path(line).exists():
                inputs.append(line)
        inputs += [p for p in (paths or []) if p]
        items = expand_inputs(inputs)
        for it in items:
            self._item_opts(it, opts)
            self._enqueue(it)
        warn = None
        from .diarizen import token_free
        if opts.get("diarize") and not self._cfg.hf_token and not token_free(self._cfg):
            warn = "Konuşmacı ayırma için HuggingFace token gerekli (Ayarlar). Bu işler konuşmacısız yazılacak."
        if opts.get("news") and not self.llm().available():
            warn = "Haber taslağı için Ayarlar > Yapay zekâ bölümünde bir API anahtarı girin."
        if self._engine_state.get("state") == "error":
            warn = ("Motor çalışmıyor: " + str(self._engine_state.get("error") or "")[:200]
                    + " — İşler sırada bekliyor; üstteki 'Yeniden başlat'a basın ya da 'Kontrol'e bakın.")
        if collections:
            warn = (f"{len(collections)} kanal/oynatma listesi linki 'Kanal / liste' sekmesine taşındı: orada konuyu "
                    "seçip videoları listeleyin.")
        elif not items:
            warn = "Link ya da dosya bulunamadı."
        return {"count": len(items), "warning": warn, "collections": collections,
                "uids": [it.extra["uid"] for it in items]}

    # ---------------------------------------------------------- tekrar dene / yarım kalanlar
    def retry(self, uid: str) -> dict:
        if getattr(self, "_reclassifying", False):
            return {"ok": False, "error": self._busy_msg()}
        it = self._items.get(uid)
        if it is None:
            return {"ok": False, "error": "İş bulunamadı (uygulama yeniden açıldıysa linki tekrar ekleyin)."}
        self._enqueue(it, retry=True)
        return {"ok": True}

    def retry_failed(self) -> dict:
        n = 0
        for uid, j in list(self._jobs.items()):
            if j.get("status") in ("failed",) and uid in self._items:
                self.retry(uid)
                n += 1
        return {"count": n}

    def resume_pending(self) -> dict:
        from .sources import Item
        n = 0
        for row in self._pending:
            it = Item(row["kind"], row["source"], title=row["title"] or row["source"], extra=dict(row["extra"]))
            it.extra["uid"] = row["uid"]
            if it.kind == "file" and not Path(it.source).exists():
                continue
            self._enqueue(it)
            n += 1
        self._pending = []
        return {"count": n}

    def clear_pending(self) -> dict:
        try:
            self._state().queue_clear()
        except Exception:  # noqa: BLE001
            log.debug("kuyruk temizlenemedi", exc_info=True)
        self._pending = []
        return {"ok": True}

    def stop_live(self, uid: str) -> bool:
        return bool(self._pipe is not None and self._pipe.stop_live(uid))

    # ---------------------------------------------------------- kanal / oynatma listesi
    def list_channel(self, url: str, opts: dict) -> dict:
        from .channel import TOPIC_LABELS, collection_name, list_videos, select
        url = (url or "").strip()
        if not url:
            return {"ok": False, "error": "Kanal ya da oynatma listesi linki girin."}
        try:
            data = list_videos(url, self._cfg, limit=int(opts.get("limit") or self._cfg.channel_max))
        except Exception as e:  # noqa: BLE001
            from .fetch import explain_error
            return {"ok": False, "error": explain_error(str(e))}
        use_ai = bool(opts.get("ai")) and self.llm().available()
        topic = opts.get("topic") or "basin"
        items = select(data["items"], topic, opts.get("custom"), opts.get("exclude"),
                       llm=self.llm() if use_ai else None,
                       llm_topic=opts.get("custom") or ("" if topic == "hepsi" else TOPIC_LABELS.get(topic, "")))
        coll = collection_name(data.get("channel") or data.get("title") or url, topic, opts.get("custom"))
        try:
            from .sources import Item
            done = self._state().done_keys([Item("url", e["url"]).compute_key() for e in items])
            for e in items:
                e["done"] = Item("url", e["url"]).compute_key() in done
        except Exception:  # noqa: BLE001
            log.debug("işlenmiş videolar okunamadı", exc_info=True)
        return {"ok": True, "title": data.get("title"), "channel": data.get("channel"), "items": items,
                "matched": sum(1 for e in items if e.get("match")), "collection": coll, "ai": use_ai}

    def start_selected(self, urls: list[str], opts: dict) -> dict:
        msg = self._busy_msg()
        if msg:
            return {"ok": False, "count": 0, "error": msg, "warning": msg}
        from .sources import Item
        n = 0
        for u in urls or []:
            it = Item("url", u, title=u)
            self._item_opts(it, {**opts, "live": False})
            self._enqueue(it)
            n += 1
        return {"count": n}

    # ---------------------------------------------------------- kanal takibi
    def _follows(self):
        from .channel import Follows
        return Follows(self._state_dir())

    def follow_list(self) -> list[dict]:
        try:
            return [{k: r.get(k) for k in ("id", "url", "name", "topic", "custom", "last", "found")}
                    for r in self._follows().all()]
        except Exception:  # noqa: BLE001
            return []

    def follow_add(self, url: str, opts: dict) -> dict:
        """Kanalı takibe al. backfill=False ise ŞU ANKİ videolar 'görüldü' sayılır, yalnızca yenileri işlenir."""
        res = self.list_channel(url, opts) if not opts.get("items") else {"ok": True, "items": opts["items"]}
        if not res.get("ok"):
            return res
        seen = [] if opts.get("backfill") else [e.get("id") or e["url"] for e in res["items"]]
        keep = {k: opts.get(k) for k in ("lang", "diarize", "speaker_mode", "speakers", "news", "ai", "fpl_auto")}
        row = self._follows().add(url, res.get("collection") or opts.get("collection") or url, opts.get("topic") or "basin", opts.get("custom") or "",
                                  opts.get("exclude") or "", keep, seen)
        if opts.get("backfill"):
            self.start_selected([e["url"] for e in res["items"] if e.get("match") and not e.get("done")],
                                {**keep, "collection": row["name"]})
        self._follows().update(row["id"], last=time.time())
        return {"ok": True, "follow": row, "follows": self.follow_list()}

    def follow_remove(self, fid: str) -> list[dict]:
        self._follows().remove(fid)
        return self.follow_list()

    def follow_check_now(self) -> dict:
        return {"count": self._check_follows(force=True)}

    def _check_follows(self, force: bool = False) -> int:
        from .channel import check_follow
        f = self._follows()
        rows = f.all() if force else f.due(self._cfg.follow_interval_h)
        total = 0
        for row in rows:
            try:
                new = check_follow(row, self._cfg, llm=self.llm() if self.llm().available() else None)
            except Exception as e:  # noqa: BLE001
                log.warning("Takip (%s) kontrol edilemedi: %s", row.get("name"), str(e)[:200])
                f.update(row["id"], last=time.time())
                continue
            if new:
                log.info("Takip: %s -> %d yeni video", row.get("name"), len(new))
                self.start_selected([e["url"] for e in new], {**(row.get("opts") or {}), "collection": row["name"]})
            seen = (row.get("seen") or []) + [e.get("id") or e["url"] for e in new]
            f.update(row["id"], last=time.time(), seen=seen[-2000:], found=(row.get("found") or 0) + len(new))
            total += len(new)
        if rows:
            self._push({"type": "follows", "follows": self.follow_list()})
        return total

    def _follow_loop(self) -> None:
        time.sleep(60)
        while not self._stop.is_set():
            try:
                self._check_follows()
            except Exception:  # noqa: BLE001
                log.exception("kanal takibi")
            self._stop.wait(600)


    def open(self, p: str) -> None:
        open_path(p)

    def open_out(self) -> None:
        Path(self._cfg.out_dir).mkdir(parents=True, exist_ok=True)
        open_path(str(Path(self._cfg.out_dir).resolve()))

    def _moved_path(self, p: str) -> str:
        """Yeniden sınıflandırmayla taşınmış metin: açık okuyucu / Konuşmacılar ekranı eski yolu tutuyor olabilir."""
        moved = getattr(self, "_moved", {})
        seen = 0
        while p and not Path(p).exists() and p in moved and seen < 20:
            p, seen = moved[p], seen + 1
        return p

    def read_text(self, p: str) -> str:
        try:
            return Path(self._moved_path(p)).read_text(encoding="utf-8")[:200_000]
        except Exception as e:  # noqa: BLE001
            return f"(okunamadı: {e})"

    # ---------------------------------------------------------- konuşmacı düzeltme / ses kütüphanesi
    def _doc_path(self, p: str) -> Path:
        jp = Path(self._moved_path(p)).with_suffix(".json")
        if not jp.is_file():
            raise FileNotFoundError("Bu metnin .json dosyası yok")
        return jp

    def speakers_load(self, p: str) -> dict:
        """Okuyucunun 'Konuşmacılar' görünümü: konuşmacılar + satırlar + bilinen sesler."""
        try:
            from .voices import VoiceLibrary, load_doc_voices
            doc = json.loads(self._doc_path(p).read_text(encoding="utf-8"))
            if not doc.get("speakers"):
                return {"ok": False, "error": "Bu metinde konuşmacı ayrımı yok."}
            emb = load_doc_voices(self._state_dir(), doc.get("key") or "").get("emb") or {}
            return {"ok": True, "mode": doc.get("speaker_mode") or "basin", "td_label": doc.get("td_label"),
                    "speakers": [{"id": k, "label": v.get("label"), "seconds": v.get("seconds", 0),
                                  "voice": v.get("voice"), "sig": k in emb}
                                 for k, v in (doc.get("speakers") or {}).items()],
                    "segments": [{"id": sg.get("id"), "start": sg.get("start"), "end": sg.get("end"),
                                  "sid": sg.get("speaker_id"), "text": sg.get("text", ""),
                                  "words": [{"start": w["start"], "end": w["end"], "word": w["word"]}
                                            for w in (sg.get("words") or [])
                                            if isinstance(w, dict) and "start" in w and "end" in w and "word" in w]}
                                 for sg in doc.get("segments") or []],
                    "voices": VoiceLibrary(self._state_dir()).names()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def speakers_save(self, p: str, data: dict) -> dict:
        """Düzeltmeleri yaz (txt/srt/json/özet yeniden), adı konan seslerin imzasını kütüphaneye ekle."""
        try:
            from .outputs import write_all
            from .voices import apply_corrections, ensure_orig_turns, learn_from_doc, reference_turns, save_reference
            jp = self._doc_path(p)
            doc = json.loads(jp.read_text(encoding="utf-8"))
            st = self._state_dir()
            hyp = None if doc.get("corrected") else reference_turns(doc)  # programın kendi çıktısı (ölçüm için)
            ensure_orig_turns(st, doc)  # eski çıktı: ilk ayırma turları düzeltmeden önce saklansın
            apply_corrections(doc, (data or {}).get("labels"), (data or {}).get("moves"),
                              (data or {}).get("td"), (data or {}).get("segments"))
            base = jp.stem
            write_all(jp.parent, base, doc, f"{base} - ozet.md")
            save_reference(st, doc, hyp)
            skipped: list = []
            try:
                learned = learn_from_doc(st, doc, ignore=(self._cfg.reporter_label, self._cfg.td_label,
                                                          "Teknik direktör"), skipped=skipped)
            except OSError as e:  # sesler.json o an kilitli (virüs tarayıcı / eğitim süreci): düzeltmeler yazıldı
                log.warning("Ses kütüphanesi güncellenemedi: %s", e)
                return {"ok": True, "learned": [], "skipped": [],
                        "warning": "Düzeltmeler kaydedildi ama ses kütüphanesi şu an güncellenemedi; birazdan tekrar "
                                   "Kaydet'e basın."}
            log.info("Konuşmacılar düzeltildi: %s (öğrenilen sesler: %s; öğrenilmeyen: %s)", base[:60],
                     ", ".join(learned) or "-", ", ".join(skipped) or "-")
            return {"ok": True, "learned": learned, "skipped": skipped}
        except Exception as e:  # noqa: BLE001
            log.exception("Konuşmacı düzeltmesi kaydedilemedi")
            return {"ok": False, "error": str(e)}

    def speakers_train(self) -> dict:
        """Konuşmacı ayırmayı düzeltilmiş videolarla eğit (ayrı süreçte, monitörsüz ekran kartında).
        İlerleme ``train`` olaylarıyla gelir."""
        if getattr(self, "_training", False):
            return {"ok": False, "error": "Eğitim zaten sürüyor."}
        if self._pipe is not None and self._pipe.active():
            return {"ok": False, "error": "Kuyrukta iş varken eğitilmez (ekran kartı tek işe). Kuyruk bitince deneyin."}
        from .tune import references
        n = len(references(self._state_dir()))
        if not n:
            return {"ok": False, "error": "Düzeltilmiş video yok: önce bir videonun 'Konuşmacılar' ekranında "
                                          "tüm satırların konuşmacısını düzeltip kaydedin."}
        self._training = True

        def run():
            res, err = None, None
            try:
                pr = subprocess.Popen([console_python(), "-m", "stt", "egit", "-o", str(Path(self._cfg.out_dir).resolve())],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      errors="replace", cwd=str(TOOL_DIR))
                self._train_proc = pr
                for line in pr.stdout:  # type: ignore[union-attr]
                    line = line.strip()
                    if line.startswith("ILERLEME: "):
                        self._push({"type": "train", "text": line[10:]})
                    elif line.startswith("SONUC: "):
                        res = json.loads(line[7:])
                    elif line.startswith("HATA: "):
                        err = line[6:]
                    elif line:
                        log.info("[eğitim] %s", line[:300])
                pr.wait()
                if res is None and not err:
                    err = f"eğitim süreci bitti (kod {pr.returncode}); ayrıntı günlükte"
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            finally:
                self._training = False
            self._push({"type": "train", "done": True, "result": res, "error": err})
        threading.Thread(target=run, daemon=True, name="stt-egitim").start()
        return {"ok": True, "videos": n}

    def diarizen_install(self) -> dict:
        """Ayarlar > 'DiariZen'i kur': ayrı ortam (stt/diarizen.py), arka planda; ilerleme günlükte ve Ayarlar'da."""
        if getattr(self, "_dz_installing", False):
            return {"ok": False, "error": "Kurulum zaten sürüyor."}
        self._dz_installing = True

        def run():
            err = None
            try:
                from .diarizen import install
                install(lambda t: self._push({"type": "diarizen", "text": t}))
            except Exception as e:  # noqa: BLE001
                err = str(e)[:300]
                log.error("DiariZen kurulamadı: %s", e)
            finally:
                self._dz_installing = False
            self._push({"type": "diarizen", "done": True, "error": err})
        threading.Thread(target=run, daemon=True, name="stt-diarizen").start()
        return {"ok": True}

    def voices_list(self) -> list[dict]:
        from .voices import VoiceLibrary
        return VoiceLibrary(self._state_dir()).names()

    def voice_delete(self, name: str) -> list[dict]:
        from .voices import VoiceLibrary
        lib = VoiceLibrary(self._state_dir())
        lib.remove(name)
        return lib.names()

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

    def cancel(self, uid: str) -> dict:
        """Tek bir işi durdur (indiriliyor / çevriliyor / sırada)."""
        kept, hit = [], False
        while True:  # henüz motora geçmemişse kutudan çıkar
            try:
                it = self._inbox.get_nowait()
            except queue.Empty:
                break
            if it.extra.get("uid") == uid:
                hit = True
                self._push({"type": "failed", "uid": uid, "source": it.source, "title": it.title,
                            "error": "Durduruldu.", "cancelled": True})
            else:
                kept.append(it)
        for it in kept:
            self._inbox.put(it)
        if not hit and self._pipe is not None:
            hit = self._pipe.cancel(uid)
        return {"ok": hit}

    def check(self) -> dict:
        """Kurulum kontrolü (stt kontrol) ayrı süreçte; çıktıyı pencerede göstermek için döndürür."""
        try:
            r = subprocess.run([console_python(), "-m", "stt", "kontrol"], capture_output=True, text=True, encoding="utf-8",
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

    def dataset_reclassify(self, apply: bool = False) -> dict:
        """Veri seti > "Yeniden sınıflandır": önce deneme turu (ne değişecek), onaylanınca uygula.
        Kuyrukta iş varken yapılmaz (biten işin çıktısı aynı anda yazılabilir)."""
        if self._pipe is not None and self._pipe.active() or not self._inbox.empty():
            return {"ok": False, "error": "Kuyrukta iş varken yapılmaz: işler bitince (ya da Durdur'dan sonra) tekrar "
                                          "deneyin."}
        with self._lock:  # iki çağrı aynı anda başlamasın (çift tıklama)
            if getattr(self, "_reclassifying", False):
                return {"ok": False, "error": "Yeniden sınıflandırma zaten sürüyor."}
            self._reclassifying = True
        try:
            from .reclassify import reclassify
            res = reclassify(Path(self._cfg.out_dir).resolve(), apply=bool(apply), state=self._state())
            if not hasattr(self, "_moved"):
                self._moved = {}
            self._moved.update(res.get("moved_files") or {})
            if apply and res.get("ok"):
                log.info("Veri seti yeniden sınıflandırıldı: %s", res.get("text", "").splitlines()[0] if res.get("text") else "")
            return res
        except Exception as e:  # noqa: BLE001
            log.exception("Yeniden sınıflandırma yapılamadı")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self._reclassifying = False

    def dataset(self) -> dict:
        root = Path(self._cfg.out_dir).resolve()
        rows = read_jsonl(root / "index.jsonl")
        tree: dict = {}
        for r in rows:
            league = r.get("league") or "Diger"
            team = "Kayıtlar" if league == GENERAL_DIR else (r.get("team") or "Bilinmeyen")
            txt = r.get("txt") or ""
            folder = str((root / txt).parent) if txt else str(root)
            m = tree.setdefault(league, {}).setdefault(team, {}).setdefault(folder, {
                "folder": folder, "name": Path(folder).name, "match_date": r.get("match_date"), "gw": r.get("gw"),
                "pressers": []})
            m["pressers"].append({"txt": str(root / txt) if txt else "", "manager": r.get("manager"),
                                  "phase": r.get("phase"), "date": r.get("presser_date"),
                                  "questions": r.get("questions"), "diarized": r.get("diarized")})
        out = []
        for league in sorted(tree, key=lambda x: (x != "Premier League", x == GENERAL_DIR, x)):
            teams = []
            for team in sorted(tree[league]):
                matches = sorted(tree[league][team].values(), key=lambda m: m.get("match_date") or "", reverse=True)
                teams.append({"team": team, "folder": str(Path(matches[0]["folder"]).parent), "matches": matches})
            out.append({"league": league, "folder": str(root / ("Premier League" if league == "Premier League"
                                                                else league if league == GENERAL_DIR
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
    try:
        if app._pipe is not None:
            app._pipe.kill()
    except Exception:  # noqa: BLE001
        pass
    tp = getattr(app, "_train_proc", None)  # eğitim süreci ekran kartında açık kalmasın
    if tp is not None and tp.poll() is None:
        try:
            tp.kill()
        except Exception:  # noqa: BLE001
            pass
    logging.shutdown()
    os._exit(0)


def window_geometry(screen: tuple[int, int] | None, saved: dict | None) -> dict:
    """Açılış penceresi: ekranın ~%65'i (en çok 1180x760), son kullanılan boyut hatırlanır.
    Eskiden 1360x860 sabitti; dizüstü / %125-150 ölçekli ekranlarda pencere ekrandan taşıyordu."""
    sw, sh = screen or (1600, 900)
    w = int(min(1100, max(800, sw * 0.6)))
    h = int(min(720, max(540, sh * 0.68)))
    if saved:
        try:
            sw_, sh_ = int(saved.get("w", 0)), int(saved.get("h", 0))
            if 760 <= sw_ <= sw and 520 <= sh_ <= sh:
                w, h = sw_, sh_
        except (TypeError, ValueError):
            pass
    w, h = min(w, sw), min(h, sh)
    return {"width": w, "height": h, "min_size": (min(760, w), min(520, h))}


def _load_ui_state() -> dict:
    try:
        return json.loads(UI_STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_ui_state(d: dict) -> None:
    try:
        UI_STATE.parent.mkdir(parents=True, exist_ok=True)
        UI_STATE.write_text(json.dumps(d), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    if not _single_instance():
        return 0
    os.chdir(TOOL_DIR)  # göreli 'cikti' klasörü hep C:\stt\cikti olsun
    logpath = _redirect_std()
    _hide_child_consoles()
    from .util import prepare_env
    prepare_env()  # ffmpeg / deno / GPU sırası / UTF-8: stt.bat'a gerek yok
    os.environ["STT_LOG"] = str(logpath)  # işçi süreçleri (konuşmacı ayırma) de bu günlüğe yazsın
    os.environ["PYTHONUTF8"] = "1"  # alt süreçler (GPU işçileri, ffmpeg, presser.py) Türkçe harfleri yazabilsin
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(logpath, encoding="utf-8")])
    from .util import quiet_warnings
    quiet_warnings()
    app = App()
    if not app._cfg.ui_gpu:
        # arayüz ekran kartını kullanmasın: çoklu ekran kartında WebView2 açılırken ekran kararıp geri geliyordu
        os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "--disable-gpu --disable-gpu-compositing")
    import webview

    app._logpath = logpath
    log.addHandler(_UiLogHandler(app))
    log.info("STT %s başlıyor (günlük: %s)", __version__, logpath)
    if app._cfg.ytdlp_autoupdate and not (TOOL_DIR / ".git").exists():
        from .fetch import start_ytdlp_update
        start_ytdlp_update(Path(app._cfg.out_dir).resolve() / ".stt")
    screen = None
    try:
        sc = webview.screens[0]
        screen = (int(sc.width), int(sc.height))
    except Exception:  # noqa: BLE001
        pass
    ui = _load_ui_state()
    geo = window_geometry(screen, ui)
    wkw = dict(url=str(HTML_PATH), js_api=app, width=geo["width"], height=geo["height"], min_size=geo["min_size"],
               resizable=True, background_color="#0f1318", text_select=True)
    try:
        app._window = webview.create_window("STT — Basın Odası", maximized=bool(ui.get("max")), **wkw)
    except TypeError:  # eski pywebview
        app._window = webview.create_window("STT — Basın Odası", **wkw)
    size = {"w": geo["width"], "h": geo["height"], "max": bool(ui.get("max"))}

    def on_resized(width, height):
        if not size["max"]:
            size.update(w=int(width), h=int(height))

    def on_maximized():
        size["max"] = True

    def on_restored():
        size["max"] = False

    for ev, fn in (("resized", on_resized), ("maximized", on_maximized), ("restored", on_restored)):
        try:
            getattr(app._window.events, ev).__iadd__(fn)
        except Exception:  # noqa: BLE001
            pass

    def on_closing():
        _save_ui_state(size)
        app._stop.set()
        app._restart.set()

    app._window.events.closing += on_closing
    app._start_api()

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
