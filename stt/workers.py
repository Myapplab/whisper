"""GPU başına bir işçi süreci (ya da OpenAI için iş parçacıkları); ortak görev kuyruğu."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback

from .util import log


def shift_payload(payload: dict, offset: float) -> dict:
    """Ayrı dosyadan (canlı yayın parçası) gelen zamanları genel zaman çizgisine taşı."""
    if not offset:
        return payload
    for s in payload.get("segments", []):
        s["start"] = round(s["start"] + offset, 3)
        s["end"] = round(s["end"] + offset, 3)
        for w in s.get("words") or []:
            w["start"] = round(w["start"] + offset, 3)
            w["end"] = round(w["end"] + offset, 3)
    return payload


def _handle(task: dict, wid: str, device: str, engine_cfg: dict, cache: dict):
    kind = task["kind"]
    if kind == "transcribe":
        eng = cache.get("engine")
        if eng is None:
            raise RuntimeError("motor yok")
        if engine_cfg["engine"] == "openai":
            res = eng.transcribe(task["wav"], task["start"], task["end"], task["lang"], task["prompt"],
                                 task.get("speech_start"), task.get("speech_end"))
        else:
            res = eng.transcribe(task["wav"], task["start"], task["end"], task["lang"], task["prompt"])
        return shift_payload(res, task.get("offset") or 0.0)
    if kind == "diarize":
        lock = cache.setdefault("_diar_lock", threading.Lock())
        with lock:
            beat = cache.get("beat")
            if "diar" not in cache:
                # kart pipeline'da seçilir (monitörlü karta asla kendiliğinden gidilmez); "auto" = işlemci
                ddev = device if device.startswith("cuda") else "cpu"
                t0 = time.time()
                log.info("[diar] konuşmacı modeli yükleniyor (%s, ilk seferde indirilir)", engine_cfg["diar_model"])
                from .engines import Diarizer
                if beat:
                    beat(True)
                cache["diar"] = Diarizer(engine_cfg["diar_model"], engine_cfg["hf_token"], ddev,
                                         agc=engine_cfg.get("agc", True), batch=engine_cfg.get("diar_batch", 8),
                                         gpu_frac=engine_cfg.get("diar_gpu_frac"), ram_pct=engine_cfg.get("ram_limit_pct", 70),
                                         cpu_ok=engine_cfg.get("diar_cpu", False))
                log.info("[diar] konuşmacı modeli hazır (%.0f sn)", time.time() - t0)
            d = cache["diar"]
            turns = d.run(task["wav"], task.get("min_speakers"), task.get("max_speakers"),
                          vocals=bool(task.get("vocals")), beat=beat)
            return {"turns": turns, "device": d.device, "emb": d.last_emb, "model": d.model_used,
                    "vocals": bool(task.get("vocals"))}
    raise ValueError(f"bilinmeyen görev: {kind}")


def _beater(result_q, wid: str, tid: str, every: float = 30.0):
    """Uzun görevde "hâlâ çalışıyorum" sinyali (en çok ``every`` sn'de bir). Ana süreç takılma süresini
    son sinyalden sayar: ilerleyen iş kesilmez, gerçekten takılan iş erken yeniden başlatılır."""
    last = [0.0]

    def beat(force: bool = False) -> None:
        now = time.time()
        if force or now - last[0] >= every:
            last[0] = now
            try:
                result_q.put({"type": "beat", "wid": wid, "tid": tid})
            except Exception:  # noqa: BLE001
                pass
    return beat


def _worker_logging(diar: bool) -> None:
    """İşçi sürecinin kayıtları: uygulamada (pythonw) ekran yok, ``STT_LOG`` günlüğüne yazılır.
    Konuşmacı ayırma ayrıntılı (model yükleme, indirme, müzik ayıklama); metin işçileri yalnız uyarılar."""
    path = os.environ.get("STT_LOG")
    handlers: list[logging.Handler] = []
    if path:
        try:
            handlers.append(logging.FileHandler(path, encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(level=logging.INFO if (diar or not handlers) else logging.WARNING,
                        format="[%(asctime)s] (%(processName)s) %(message)s" if handlers else "[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers or None, force=True)
    for noisy in ("faster_whisper", "speechbrain", "lightning", "pytorch_lightning", "httpx", "httpcore",
                  "urllib3", "filelock", "fsspec", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def isolate_gpu(device: str) -> str:
    """Bu süreç YALNIZCA kendi kartını görsün (CUDA_VISIBLE_DEVICES). Döner: süreç içi aygıt adı (cuda:0).

    Böylece 3090'daki işçi, monitörün bağlı olduğu 3080'de hiçbir CUDA bağlamı açmaz (ekran kararması / TDR).
    CUDA kütüphaneleri yüklenmeden ÖNCE çağrılmalı (işçi süreci başında)."""
    if not device.startswith("cuda"):
        return device
    idx = device.split(":")[1] if ":" in device else "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = idx
    return "cuda:0"


def _init(device: str, engine_cfg: dict, cache: dict, load_lock=None, run_device: str | None = None) -> None:
    if engine_cfg.get("role") == "diar":  # yalnızca konuşmacı ayırma: model ilk görevde yüklenir
        cache["engine"] = None
        return
    if engine_cfg["engine"] == "openai":
        with cache.setdefault("_init_lock", threading.Lock()):
            if "engine" not in cache:
                from .engines import OpenAIEngine
                cache["engine"] = OpenAIEngine(engine_cfg["openai_model"])
    else:
        from .engines import WhisperEngine, parse_device
        _, idx = parse_device(device)
        vram = (engine_cfg.get("gpu_mem") or {}).get(idx)
        batch = engine_cfg.get("batch_size", -1)
        if device.startswith("cuda") and idx in (engine_cfg.get("display_gpus") or []):
            from .engines import auto_batch_size
            cap = max(1, int(engine_cfg.get("display_batch") or 4))
            batch = min(auto_batch_size("cuda", vram) if batch < 0 else batch, cap)
        # modeller SIRAYLA yüklenir: aynı anda iki 3 GB yükleme RAM'i (mkl_malloc) tüketebilir
        if load_lock is not None:
            load_lock.acquire()
        try:
            cache["engine"] = WhisperEngine(engine_cfg["model_path"], run_device or device, engine_cfg["compute_type"],
                                            engine_cfg["beam_size"], engine_cfg["condition_on_previous_text"],
                                            batch, vram, engine_cfg.get("languages"), label=device,
                                            agc=engine_cfg.get("agc", True),
                                            vad_threshold=engine_cfg.get("vad_threshold", 0.4),
                                            fill_gaps=engine_cfg.get("fill_gaps", True))
        finally:
            if load_lock is not None:
                load_lock.release()


def lower_priority() -> None:
    """Bu süreci düşük önceliğe al: masaüstü, tarayıcı ve oyunlar önce gelsin."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            from .util import BELOW_NORMAL
            k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001
        pass


def worker_main(wid: str, device: str, engine_cfg: dict, task_q, result_q, shared_cache: dict | None = None,
                load_lock=None):
    if shared_cache is None:  # ayrı süreç
        if engine_cfg.get("low_priority", True):
            lower_priority()
        os.environ.setdefault("PYTHONUTF8", "1")
        if sys.stdout is None or sys.stderr is None:  # pythonw (uygulama) altında
            # UTF-8 şart: Windows'un varsayılanı (cp1252) 'ş' gibi harfleri yazamıyor; pyannote/uyarılar
            # bir şey yazınca UnicodeEncodeError ile konuşmacı ayırma çöküyordu
            devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
            sys.stdout = sys.stdout or devnull
            sys.stderr = sys.stderr or devnull
        for st in (sys.stdout, sys.stderr):
            try:
                st.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        from .util import net_env, quiet_warnings
        quiet_warnings()
        net_env()
        _worker_logging(engine_cfg.get("role") == "diar")
        if engine_cfg.get("role") == "diar":
            import socket
            socket.setdefaulttimeout(120)  # müzik ayıklama modeli indirmesi (torch.hub) takılırsa hata versin
    cache: dict = shared_cache if shared_cache is not None else {}
    run_device = device
    if shared_cache is None and engine_cfg.get("isolate_gpu", True):
        run_device = isolate_gpu(device)
    try:
        _init(device, engine_cfg, cache, load_lock, run_device)
        result_q.put({"type": "ready", "wid": wid})
        while True:
            task = task_q.get()
            if task is None:
                break
            result_q.put({"type": "start", "wid": wid, "tid": task["id"]})
            cache["beat"] = _beater(result_q, wid, task["id"])
            t = time.time()
            try:
                payload = _handle(task, wid, run_device, engine_cfg, cache)
                result_q.put({"type": "done", "wid": wid, "tid": task["id"], "payload": payload,
                              "elapsed": time.time() - t})
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                # ekran kartı bağlamı bozuldu (invalid device ordinal, cuBLAS failed...): bu süreçte bir daha
                # düzelmez. Hatayı bildir ve çık; ana süreç işçiyi temiz açar, parça yeniden denenir.
                restart = (shared_cache is None and device.startswith("cuda") and engine_cfg.get("role") != "diar"
                           and gpu_fault(err))
                result_q.put({"type": "error", "wid": wid, "tid": task["id"], "error": err,
                              "trace": traceback.format_exc(), "restart": restart})
                if restart:
                    time.sleep(0.5)  # mesaj kuyruğa yazılsın
                    return
    except KeyboardInterrupt:  # Ctrl+C tüm konsol süreçlerine gider; sessizce çık
        return
    except Exception as e:  # noqa: BLE001  (model yükleme vb.)
        result_q.put({"type": "fatal", "wid": wid, "error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc(), "retryable": retryable(e)})


_RETRYABLE = ("mkl_malloc", "bad_alloc", "out of memory", "winerror 1455", "paging file", "cuda failed",
              "device-side", "unspecified launch failure", "illegal memory")


_GPU_FAULT = ("invalid device ordinal", "cudaerror", "cublas", "cuda failed", "cuda error", "illegal memory",
              "unspecified launch failure", "device-side", "cudnn_status_execution_failed")


def gpu_fault(err: str) -> bool:
    """Ekran kartı / sürücü takılması (bellek yetmemesi değil): süreci yeniden açmak gerekir."""
    e = err.lower()
    return "out of memory" not in e and any(x in e for x in _GPU_FAULT)


def retryable(e: BaseException) -> bool:
    """Model yüklenemedi ama geçici mi (RAM/GPU bir anlığına yetmedi, sürücü sıfırlandı)? Öyleyse işçi
    biraz sonra yeniden açılır; eksik model / kütüphane gibi kalıcı hatalarda açılmaz."""
    return isinstance(e, MemoryError) or any(x in str(e).lower() for x in _RETRYABLE)


class WorkerPool:
    """Süreç (GPU/CPU) ya da iş parçacığı (OpenAI) havuzu; aynı mesaj protokolü."""

    def __init__(self, devices: list[str], engine_cfg: dict, threads: bool = False, names: list[str] | None = None):
        self.devices = devices
        self.threads = threads
        if threads:
            self.task_q = queue.Queue()
            self.result_q = queue.Queue()
            shared: dict = {}
            self.workers = {
                f"api{i}": threading.Thread(target=worker_main, args=(f"api{i}", d, engine_cfg, self.task_q,
                                                                      self.result_q, shared), daemon=True)
                for i, d in enumerate(devices)}
        else:
            self.ctx = mp.get_context("spawn")
            self.task_q = self.ctx.Queue()
            self.result_q = self.ctx.Queue()
            self.load_lock = self.ctx.Lock()
            names = names or [d if devices.count(d) == 1 else f"{d}#{i}" for i, d in enumerate(devices)]
            self._spec = dict(zip(names, devices))
            self._ecfg = engine_cfg
            self.workers = {n: self._new_proc(n) for n in names}

    def _new_proc(self, name: str):
        return self.ctx.Process(target=worker_main, args=(name, self._spec[name], self._ecfg, self.task_q,
                                                          self.result_q, None, self.load_lock),
                                name=f"stt-{name}", daemon=True)

    def respawn(self, name: str) -> bool:
        """Çöken / takılan işçiyi yeniden başlat (yalnızca süreç havuzunda)."""
        if self.threads or name not in self.workers:
            return False
        old = self.workers[name]
        if old.is_alive():
            old.terminate()
            old.join(5)
            if old.is_alive():
                old.kill()
                old.join(2)
        self.workers[name] = self._new_proc(name)
        self.workers[name].start()
        return True

    def start(self):
        for w in self.workers.values():
            w.start()
        log.info("İşçiler başlatıldı: %s", ", ".join(self.workers))

    def alive(self) -> list[str]:
        return [k for k, w in self.workers.items() if w.is_alive()]

    def put(self, task: dict):
        self.task_q.put(task)

    def get(self, timeout: float):
        try:
            return self.result_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self, timeout: float = 10):
        for _ in self.workers:
            try:
                self.task_q.put(None)
            except Exception:  # noqa: BLE001
                pass
        end = time.time() + timeout
        for w in self.workers.values():
            w.join(max(0.1, end - time.time()))
        if not self.threads:
            for w in self.workers.values():
                if w.is_alive():
                    w.terminate()
