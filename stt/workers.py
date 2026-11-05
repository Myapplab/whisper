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


def _handle(task: dict, wid: str, device: str, engine_cfg: dict, cache: dict):
    kind = task["kind"]
    if kind == "transcribe":
        eng = cache.get("engine")
        if eng is None:
            raise RuntimeError("motor yok")
        if engine_cfg["engine"] == "openai":
            return eng.transcribe(task["wav"], task["start"], task["end"], task["lang"], task["prompt"],
                                  task.get("speech_start"), task.get("speech_end"))
        return eng.transcribe(task["wav"], task["start"], task["end"], task["lang"], task["prompt"])
    if kind == "diarize":
        lock = cache.setdefault("_diar_lock", threading.Lock())
        with lock:
            if "diar" not in cache:
                ddev = device
                if not device.startswith("cuda"):
                    try:
                        import torch
                        ddev = "cuda:0" if torch.cuda.is_available() else "cpu"
                    except Exception:  # noqa: BLE001
                        ddev = "cpu"
                from .engines import Diarizer
                cache["diar"] = Diarizer(engine_cfg["diar_model"], engine_cfg["hf_token"], ddev)
            return {"turns": cache["diar"].run(task["wav"], task.get("min_speakers"), task.get("max_speakers"))}
    raise ValueError(f"bilinmeyen görev: {kind}")


def _init(device: str, engine_cfg: dict, cache: dict, load_lock=None) -> None:
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
            cache["engine"] = WhisperEngine(engine_cfg["model_path"], device, engine_cfg["compute_type"],
                                            engine_cfg["beam_size"], engine_cfg["condition_on_previous_text"],
                                            batch, vram, engine_cfg.get("languages"))
        finally:
            if load_lock is not None:
                load_lock.release()


def lower_priority() -> None:
    """Bu süreci düşük önceliğe al: masaüstü, tarayıcı ve oyunlar önce gelsin."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
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
        from .util import quiet_warnings
        quiet_warnings()
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        for noisy in ("faster_whisper", "speechbrain", "lightning", "pytorch_lightning", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    cache: dict = shared_cache if shared_cache is not None else {}
    try:
        _init(device, engine_cfg, cache, load_lock)
        result_q.put({"type": "ready", "wid": wid})
        while True:
            task = task_q.get()
            if task is None:
                break
            result_q.put({"type": "start", "wid": wid, "tid": task["id"]})
            t = time.time()
            try:
                payload = _handle(task, wid, device, engine_cfg, cache)
                result_q.put({"type": "done", "wid": wid, "tid": task["id"], "payload": payload,
                              "elapsed": time.time() - t})
            except Exception as e:  # noqa: BLE001
                result_q.put({"type": "error", "wid": wid, "tid": task["id"], "error": f"{type(e).__name__}: {e}",
                              "trace": traceback.format_exc()})
    except KeyboardInterrupt:  # Ctrl+C tüm konsol süreçlerine gider; sessizce çık
        return
    except Exception as e:  # noqa: BLE001  (model yükleme vb.)
        result_q.put({"type": "fatal", "wid": wid, "error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc()})


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
