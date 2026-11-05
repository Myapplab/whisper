"""Orkestrasyon: hazırlık (indirme/dönüştürme/bölme) iş parçacıklarında, tanıma GPU işçilerinde."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .audio import plan_chunks
from .config import Config
from .fetch import DRMProtected, FetchError, SkipItem, prepare_audio
from .merge import apply_labels, assign_speakers, label_speakers, merge_chunks
from .outputs import to_txt, write_all
from .sources import Item
from .state import State
from .dataset import classify, fpl_data, presser_base_name, target_dir, update_index, write_match_json
from .match import parse_match
from .util import fmt_clock, log, text_hash
from .workers import WorkerPool

MAX_PATH = 250  # Windows yol sınırı (260) altında kalmak için


@dataclass
class Job:
    item: Item
    work: Path
    status: str = "queued"           # queued|prep|run|finalize|done|failed|skipped
    wav: Path | None = None
    info: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    pending: set = field(default_factory=set)
    attempts: dict = field(default_factory=dict)
    error: str = ""
    diar_error: str = ""
    match: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    done_chunks: int = 0
    out_dir: Path | None = None
    result: dict | None = None
    t0: float = field(default_factory=time.time)


def diar_hint(err: str) -> str:
    e = err.lower()
    if "memoryerror" in e or "bad_alloc" in e or "not enough memory" in e or "winerror 1455" in e:
        return ram_hint(err)
    if "cudnn" in e or "cuda error" in e:
        return ("Ekran kartında ve işlemcide konuşmacı ayırma başarısız oldu. NVIDIA sürücüsünü güncelleyin; "
                "'Kontrol'ün ekran görüntüsünü paylaşın. (" + err[:160] + ")")
    if "proxyerror" in e or "max retries" in e or "connectionerror" in e or "timed out" in e:
        return "huggingface.co'ya bağlanılamadı (internet/güvenlik duvarı). Model bir kez inince internet gerekmez."
    if "gated" in e or "403" in e or "accept" in e or "koşulları" in e or "nonetype" in e:
        return ("HuggingFace model koşulları kabul edilmemiş: huggingface.co/pyannote/speaker-diarization-3.1 ve "
                "huggingface.co/pyannote/segmentation-3.0 sayfalarında 'Agree' deyin.")
    if "401" in e or "token" in e or "unauthorized" in e:
        return "HF token geçersiz ya da okunamadı: huggingface.co'dan yeni token alıp uygulamada Ayarlar > HuggingFace token alanına yapıştırın."
    if "no module named" in e:
        return "pyannote/torch kurulu değil: STT-Kurulum.exe'yi tekrar çalıştırın."
    return err[:300]


def ram_hint(err: str = "") -> str:
    """RAM / sanal bellek yetmediğinde: anlık durum + kalıcı çözüm."""
    from .util import PAGEFILE_HINT, memory_status, memory_text
    m = memory_status()
    msg = "Bilgisayarın belleği (RAM + sanal bellek) yetmedi."
    if m:
        msg += " Şu an: " + memory_text() + "."
    msg += " Tarayıcı sekmelerini / oyunları kapatıp linki tekrar ekleyin (kaldığı yerden devam eder)."
    if not m or m["commit_total"] < 48:
        msg += " Kalıcı çözüm: " + PAGEFILE_HINT
    return msg + (f" ({err[:120]})" if err else "")


def worker_hint(err: str) -> str:
    """Model yükleme / GPU hatalarını anlaşılır Türkçeye çevir."""
    e = err.lower()
    if "mkl_malloc" in e or "memoryerror" in e or "bad_alloc" in e or "not enough memory" in e \
            or "paging file" in e or "winerror 1455" in e:
        return ram_hint(err)
    if "out of memory" in e:
        return ("Ekran kartı belleği yetmedi: GPU'yu kullanan başka bir program (oyun, başka bir STT penceresi) "
                "olabilir. Kapatıp 'Yeniden başlat'a basın. (" + err[:160] + ")")
    if any(x in e for x in ("cublas", "cudnn", ".dll", "library", "cannot load symbol")):
        return "CUDA kütüphaneleri yüklenemedi: STT-Kurulum.exe'yi tekrar çalıştırın. (" + err[:200] + ")"
    if "cuda driver" in e or "no cuda" in e or "cuda failed" in e or "insufficient for cuda" in e:
        return "Ekran kartı sürücüsü sorunu: NVIDIA sürücüsünü güncelleyip bilgisayarı yeniden başlatın. (" + err[:200] + ")"
    if "unable to open file 'model.bin'" in e or "no such file" in e or "localentrynotfound" in e:
        return "Whisper modeli bulunamadı/eksik: internet açıkken 'Yeniden başlat'a basın (model ~3 GB iner)."
    return err[:400]


def item_fpl(item: Item, cfg: Config) -> tuple[int, str] | None:
    if item.extra.get("fpl"):
        return item.extra["fpl"]
    if cfg.fpl_gw is not None and cfg.fpl_team:
        return cfg.fpl_gw, cfg.fpl_team.upper()
    return None


def detect_devices(spec: str, info: dict | None = None) -> list[str]:
    """spec: "auto" (monitörsüz kart(lar)) | "all" | "0,1" | "cuda:0" | "cpu" | "cpu,cpu" (test)."""
    spec = (spec or "auto").strip().lower()
    if spec not in ("auto", "all", "hepsi", "tum", "tümü"):
        out = []
        for x in spec.replace("cuda:", "").split(","):
            x = x.strip()
            if x:
                out.append("cpu" if x == "cpu" else f"cuda:{int(x)}")
        return out or ["cpu"]
    try:
        from .util import setup_cuda_dlls
        setup_cuda_dlls()
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        n = 0
    if n == 0:
        return ["cpu"]
    from .engines import choose_gpus
    return [f"cuda:{i}" for i in choose_gpus(n, info or {}, "auto" if spec == "auto" else "all")]


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_root = Path(cfg.out_dir).resolve()
        self.state = State(self.out_root / ".stt")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.prep = ThreadPoolExecutor(max_workers=max(1, cfg.prep_workers), thread_name_prefix="hazirlik")
        self.fin = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cikti")
        self.inflight: dict[str, tuple[str, float]] = {}   # wid -> (tid, başlama zamanı)
        self.tasks: dict[str, dict] = {}     # tid -> task
        self.pool: WorkerPool | None = None
        self.devices: list[str] = []
        self._fpl = None
        self._fpl_loaded = False
        self.events = None                   # isteğe bağlı: GUI için olay geri çağırma fonksiyonu
        self.results: list[Job] = []
        self._pending_futs = 0
        self.backlog: deque = deque()        # ortak kuyruk; boşta işçi oldukça dağıtılır
        self.dbacklog: deque = deque()       # konuşmacı ayırma görevleri (ayrı süreç: "diar")
        self.dpool: WorkerPool | None = None
        self.dqueued_n = 0
        self._ecfg: dict = {}
        self.idle: set[str] = set()          # boşta olduğunu bildiren işçiler
        self.queued_n = 0                    # işçi kuyruğuna konup henüz alınmamış görev sayısı
        self.fatal: set[str] = set()         # model yükleyemeyen işçiler (yeniden başlatılmaz)
        self.restarts: dict[str, int] = {}   # işçi -> yeniden başlatma sayısı
        self.cancel_gen = 0                  # "Durdur": bu sayıdan önce gönderilen işler iptal
        self.ready_workers: set = set()
        self.worker_errors: dict[str, str] = {}
        self.failed_reason: str | None = None  # tüm işçiler başlatılamadıysa sebep (GUI motoru hata durumuna alır)

    # ---------------------------------------------------------- başlatma
    def start(self):
        cfg = self.cfg
        ecfg = {"engine": cfg.engine, "compute_type": cfg.compute_type, "beam_size": cfg.beam_size,
                "batch_size": cfg.batch_size,
                "condition_on_previous_text": cfg.condition_on_previous_text, "openai_model": cfg.openai_model,
                "diar_model": cfg.diar_model, "hf_token": cfg.hf_token, "languages": cfg.languages}
        if cfg.engine == "openai":
            devices = ["api"] * max(1, cfg.openai_workers)
            self.pool = WorkerPool(devices, ecfg, threads=True)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # cuda:N == nvidia-smi N
            if os.environ.get("CUDA_VISIBLE_DEVICES") and (cfg.gpus or "auto").lower() in ("auto", "all"):
                log.info("CUDA_VISIBLE_DEVICES=%s kaldırıldı (kart seçimini uygulama yapar).",
                         os.environ.pop("CUDA_VISIBLE_DEVICES"))
            from .engines import gpu_info
            gi = gpu_info()
            devices = detect_devices(cfg.gpus, gi)
            self.devices = devices
            ecfg["gpu_mem"] = {i: g["gb"] for i, g in gi.items()}
            ecfg["display_gpus"] = [i for i, g in gi.items() if g.get("display")]
            ecfg["display_batch"] = cfg.display_batch
            ecfg["low_priority"] = cfg.low_priority
            used = {int(d.split(":")[1]) for d in devices if d.startswith("cuda")}
            for i, g in sorted(gi.items()):
                mon = " [monitör bağlı]" if g.get("display") else ""
                state = "kullanılıyor" if i in used else "BOŞ bırakıldı (masaüstü/tarayıcı için)"
                log.info("cuda:%d = %s (%.0f GB)%s -> %s", i, g["name"], g["gb"], mon, state)
            if any(i in ecfg["display_gpus"] for i in used):
                log.info("Monitörün bağlı olduğu kart kullanılıyor: toplu boyut en fazla %d (bilgisayar akıcı kalsın).",
                         cfg.display_batch)
            if all(d == "cpu" for d in devices):
                log.warning("CUDA GPU bulunamadı -> CPU (int8) ile çalışılacak (yavaş).")
            log.info("Model hazırlanıyor: %s", cfg.model)
            from .engines import resolve_model_path
            ecfg["model_path"] = resolve_model_path(cfg.model)
            self.pool = WorkerPool(devices, ecfg)
        self.pool.start()
        self._ecfg = ecfg

    def _start_diar_pool(self) -> None:
        """Konuşmacı ayırma (pyannote/torch) Whisper'dan AYRI bir süreçte çalışır.

        Aynı süreçte torch ile CTranslate2 ekran kartını paylaşınca 'invalid device ordinal' hataları
        görüldü; ayrıca pyannote çökerse Whisper işçisi etkilenmesin. İlk ayırma görevinde açılır.
        """
        cuda = [d for d in self.devices if d.startswith("cuda")]
        mem = self._ecfg.get("gpu_mem") or {}
        if not cuda and mem:  # Whisper CPU'da ama ekran kartı var
            cuda = [f"cuda:{i}" for i in mem]
        # belleği en büyük kart (3090): Whisper ile aynı kartı paylaşsa bile pyannote'a yer kalsın
        cuda.sort(key=lambda d: -(mem.get(int(d.split(":")[1])) or 0))
        dev = cuda[0] if cuda else "auto"
        self.dpool = WorkerPool([dev], {**self._ecfg, "role": "diar"}, names=["diar"])
        self.dpool.start()
        self.emit("worker", wid="diar", state="loading", name="Konuşmacı ayırma")

    def _pool_of(self, wid: str) -> WorkerPool | None:
        if self.dpool is not None and wid in self.dpool.workers:
            return self.dpool
        return self.pool

    def _pools(self) -> list:
        return [p for p in (self.pool, self.dpool) if p is not None]

    # ---------------------------------------------------------- olaylar (GUI)
    def emit(self, typ: str, item: Item | None = None, **data) -> None:
        if not self.events:
            return
        if item is not None:
            data.setdefault("uid", item.extra.get("uid"))
            data.setdefault("source", item.source)
            data.setdefault("title", item.title)
        try:
            self.events({"type": typ, **data})
        except Exception:  # noqa: BLE001
            log.debug("olay iletilemedi", exc_info=True)

    def _opt(self, item: Item, name: str):
        """İş başına ayar (GUI): item.extra[name] varsa o, yoksa genel ayar."""
        if name in item.extra:
            v = item.extra[name]
            return None if name == "lang" and v in ("auto", "", None) else v
        return getattr(self.cfg, name)

    # ---------------------------------------------------------- girdi ekleme
    def submit(self, item: Item) -> None:
        import uuid
        item.extra.setdefault("uid", uuid.uuid4().hex[:10])
        item.extra["_gen"] = self.cancel_gen
        self.emit("queued", item)
        self._pending_futs += 1
        self.prep.submit(self._prepare_safe, item)

    def _prepare_safe(self, item: Item):
        try:
            for attempt in range(3):
                try:
                    self._prepare(item)
                    break
                except MemoryError:
                    # RAM bir anlığına doldu (tarayıcı, model yükleme...): biraz bekleyip tekrar dene
                    with self.lock:
                        j = self.jobs.get(item.key)
                        if j is not None and j.status == "prep":
                            del self.jobs[item.key]
                    if attempt == 2:
                        raise
                    log.warning("Bellek (RAM) yetmedi; 30 sn sonra tekrar denenecek: %s", item.title[:60])
                    import gc
                    gc.collect()
                    time.sleep(30)
        except MemoryError:
            log.exception("Hazırlık hatası: %s", item.source)
            self._fail_item(item, ram_hint())
        except Exception as e:  # noqa: BLE001
            log.exception("Hazırlık hatası: %s", item.source)
            from .fetch import explain_error
            err = explain_error(str(e)) if isinstance(e, FetchError) else f"{type(e).__name__}: {e}"
            self._fail_item(item, err)
        finally:
            with self.lock:
                self._pending_futs -= 1

    def _fail_item(self, item: Item, err: str, status: str = "failed"):
        with self.lock:
            key = item.key or "?"
            job = self.jobs.get(key) or Job(item, self.out_root / ".stt" / "is" / key[:18])
            job.status, job.error = status, err
            self.jobs[key] = job
            self.results.append(job)
        if item.key:
            self.state.set(item.key, source=item.source, status=status, title=item.title, error=err)
        self.emit("failed" if status == "failed" else "skipped", item, status=status, error=err)

    def _redo_reason(self, item: Item, prev: dict | None) -> str:
        """Önceki çıktı varsa ama yetersizse sebebi (yeniden işlenir), yoksa boş."""
        if not prev:
            return ""
        info = prev.get("info") or {}
        if self._opt(item, "diarize") and not info.get("diarized"):
            return "Önceki çıktıda konuşmacı ayrımı yok"
        from .engines import DEFAULT_LANGS
        allowed = {x.strip() for x in (self.cfg.languages or DEFAULT_LANGS).split(",")}
        if not self._opt(item, "lang") and info.get("language") and info["language"] not in allowed:
            return f"Önceki çıktının dili yanlış algılanmış ({info['language']})"
        return ""

    def _cancelled(self, item: Item) -> bool:
        return item.extra.get("_gen", 0) < self.cancel_gen

    def _prepare(self, item: Item):
        cfg = self.cfg
        if self._cancelled(item):
            self.emit("failed", item, error="Durduruldu.", cancelled=True)
            return
        if item.kind == "file" and not item.key:
            item.key = self.state.file_key(item.source, item.compute_key)
        item.compute_key()
        prev = self.state.get(item.key) if not cfg.force else None
        why = self._redo_reason(item, prev)
        if prev and prev["status"] == "done" and why:
            log.info("%s; yeniden işleniyor: %s", why, item.title[:70])
        elif not cfg.force and self.state.is_done(item.key):
            log.info("Atlandı (daha önce işlendi): %s -> %s", item.title[:70], prev["out_dir"] if prev else "")
            if prev and item_fpl(item, cfg):
                self._fpl_from_previous(prev, item_fpl(item, cfg))
            job = Job(item, Path(), status="skipped", out_dir=Path(prev["out_dir"]) if prev else None)
            with self.lock:
                self.jobs[item.key] = job
                self.results.append(job)
            self.emit("skipped", item, status="done_before", out_dir=prev["out_dir"] if prev else None,
                      json=(prev.get("info") or {}).get("json") if prev else None)
            return
        with self.lock:
            if item.key in self.jobs and self.jobs[item.key].status in ("queued", "prep", "run", "finalize"):
                self.emit("skipped", item, status="duplicate", error="Bu video zaten sırada / işleniyor.")
                return
        job = Job(item, self.out_root / ".stt" / "is" / item.key[:18], status="prep")
        with self.lock:
            self.jobs[item.key] = job
        self.state.set(item.key, source=item.source, status="running", title=item.title)
        log.info("Hazırlanıyor: %s", item.source)
        self.emit("prep", item)
        try:
            job.wav, job.info = prepare_audio(item, job.work, cfg)
            if self._cancelled(item):
                with self.lock:
                    job.status = "cancelled"
                self.state.set(item.key, source=item.source, status="cancelled", title=item.title)
                self.emit("failed", item, error="Durduruldu.", cancelled=True)
                return
        except DRMProtected as e:
            log.warning("ATLANDI (DRM): %s", e)
            self._fail_item(item, str(e), status="drm")
            return
        except SkipItem as e:
            log.warning("ATLANDI: %s", e)
            self._fail_item(item, str(e), status="skipped_noaudio")
            return
        # link -> kanonik kimlik (ör. youtube:ID) ile de tekrar kontrolü
        if item.kind == "url" and job.info.get("id") and job.info.get("extractor"):
            canon = "u_" + text_hash(f"{job.info['extractor']}:{job.info['id']}")
            cprev = self.state.get(canon) or {}
            redo = bool(cprev) and bool(self._redo_reason(item, cprev))
            if not cfg.force and not redo and canon != item.key and self.state.is_done(canon):
                log.info("Atlandı (aynı video farklı link): %s", item.source)
                self.state.add_alias(item.key, canon)
                with self.lock:
                    job.status = "skipped"
                    self.results.append(job)
                self.emit("skipped", item, status="done_before")
                return
            self.state.add_alias(canon, item.key)
        if job.info.get("title"):
            item.title = job.info["title"]
        dur = job.info.get("duration_s", 0)
        plan_path = job.work / "parcalar.json"
        target = cfg.openai_chunk_s if (cfg.engine == "openai" and cfg.openai_model != "whisper-1") else cfg.chunk_min * 60
        split_s = min(cfg.split_min * 60, target) if cfg.engine == "openai" else cfg.split_min * 60
        if cfg.engine == "openai" and cfg.openai_model == "whisper-1":
            target = min(target, 600)  # 25 MB API sınırı
        plan_sig = f"{cfg.engine}:{split_s}:{target}"
        chunks = None
        if plan_path.exists():
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            if saved.get("sig") == plan_sig:
                chunks = saved["chunks"]
        if chunks is None:
            chunks = plan_chunks(job.wav, dur, split_s=split_s, target_s=target)
            plan_path.write_text(json.dumps({"sig": plan_sig, "chunks": chunks}), encoding="utf-8")
        job.chunks = chunks
        (job.work / "parcalar").mkdir(exist_ok=True)
        log.info("%s: %s, %d parça", item.title[:70], fmt_clock(dur), len(chunks))
        self.emit("planned", item, duration=dur, chunks=len(chunks))

        lang = self._opt(item, "lang")
        job.match = parse_match({**job.info, "title": job.info.get("title") or item.title})
        prompt = cfg.prompt
        if not prompt:
            m = job.match
            bits = [m.get("manager"), " vs ".join(x for x in (m.get("home"), m.get("away")) if x)]
            if any(bits):
                prompt = ", ".join(b for b in bits if b) + ", press conference."
            elif item_fpl(item, cfg):
                from .fpl import fpl_prompt
                prompt = fpl_prompt(item_fpl(item, cfg)[1])
        tasks = []
        for i, c in enumerate(chunks):
            tid = f"{item.key}:t:{i:04d}"
            if (job.work / "parcalar" / f"{i:04d}.json").exists():
                continue  # sürdürme: bu parça zaten bitmiş
            tasks.append({"id": tid, "kind": "transcribe", "wav": str(job.wav), "start": c["start"],
                          "end": c["end"], "speech_start": c["speech_start"], "speech_end": c["speech_end"],
                          "lang": lang, "prompt": prompt, "idx": i, "n": len(chunks)})
        if self._opt(item, "diarize") and not (job.work / "konusmaci.json").exists():
            tasks.insert(0, {"id": f"{item.key}:d", "kind": "diarize", "wav": str(job.wav),
                             "min_speakers": cfg.min_speakers or 2, "max_speakers": cfg.max_speakers})
        done_before = len(chunks) - sum(1 for t in tasks if t["kind"] == "transcribe")
        if done_before:
            log.info("Sürdürülüyor: %d/%d parça önceden bitmiş", done_before, len(chunks))
        job.done_chunks = done_before
        self.emit("progress", item, done=done_before, total=len(chunks), meta=job.match)
        with self.lock:
            if self._cancelled(item):
                job.status = "cancelled"
                self.emit("failed", item, error="Durduruldu.", cancelled=True)
                return
            job.pending = {t["id"] for t in tasks}
            job.status = "run"
            for t in tasks:
                t["job"] = item.key
                self.tasks[t["id"]] = t
            self.backlog.extend(t for t in tasks if t["kind"] != "diarize")
            self.dbacklog.extend(t for t in tasks if t["kind"] == "diarize")
        if not tasks:
            self._finalize_async(job)

    # ---------------------------------------------------------- sonuç döngüsü
    def pump(self, timeout: float = 0.5) -> None:
        for i, pool in enumerate(self._pools()):
            msg = pool.get(timeout if i == 0 else 0.01)
            while msg is not None:
                self._on_msg(msg)
                msg = pool.get(0.01)
        self._health()
        self._dispatch()

    def _dispatch(self):
        with self.lock:
            while len(self.idle) - self.queued_n > 0 and self.backlog:
                t = self.backlog.popleft()
                job = self.jobs.get(t["job"])
                if not job or job.status != "run":
                    continue  # iş iptal/başarısız: görevi at
                self.queued_n += 1
                self.pool.put(t)
            if self.dbacklog and self.dpool is None:
                self._start_diar_pool()
            didle = 1 if "diar" in self.idle else 0
            while didle - self.dqueued_n > 0 and self.dbacklog:
                t = self.dbacklog.popleft()
                job = self.jobs.get(t["job"])
                if not job or job.status != "run":
                    continue
                self.dqueued_n += 1
                self.dpool.put(t)

    def _on_msg(self, m: dict):
        typ, wid = m["type"], m.get("wid")
        if typ == "ready":
            self.idle.add(wid)
            self.ready_workers.add(wid)
            self.emit("worker", wid=wid, state="idle")
            return
        if typ == "fatal":
            hint = worker_hint(m["error"])
            self.worker_errors[wid] = hint
            self.fatal.add(wid)
            self.idle.discard(wid)
            self.emit("worker", wid=wid, state="error", error=hint)
            log.error("[%s] işçi başlatılamadı: %s", wid, m["error"])
            log.info("%s", m.get("trace", ""))
            main_ws = set(self.pool.workers) if self.pool else set()
            if main_ws and not (self.ready_workers & main_ws) and main_ws <= set(self.worker_errors):
                self.failed_reason = "Model yüklenemedi: " + hint
                self.emit("fatal", error=self.failed_reason)
            return
        if typ == "start":
            self.inflight[wid] = (m["tid"], time.time())
            self.idle.discard(wid)
            if wid == "diar":
                self.dqueued_n = max(0, self.dqueued_n - 1)
            else:
                self.queued_n = max(0, self.queued_n - 1)
            t = self.tasks.get(m["tid"]) or {}
            j = self.jobs.get(t.get("job", ""))
            self.emit("worker", wid=wid, state="busy", task=t.get("kind"),
                      title=j.item.title if j else "", uid=j.item.extra.get("uid") if j else None)
            return
        tid = m["tid"]
        self.inflight.pop(wid, None)
        self.idle.add(wid)
        self.emit("worker", wid=wid, state="idle")
        task = self.tasks.get(tid)
        if not task:
            return
        job = self.jobs.get(task["job"])
        if job and job.status == "cancelled" and typ == "done" and task["kind"] == "transcribe" \
                and (job.work / "parcalar").is_dir():
            # durdurulan iş: biten parçayı sakla, tekrar başlatınca kaldığı yerden devam eder
            (job.work / "parcalar" / f"{task['idx']:04d}.json").write_text(
                json.dumps(m["payload"], ensure_ascii=False), encoding="utf-8")
            return
        if not job or job.status != "run":
            return
        if typ == "error":
            n = job.attempts.get(tid, 0) + 1
            job.attempts[tid] = n
            log.error("[%s] görev hatası (%s, deneme %d): %s", wid, tid.split(":", 1)[1], n, m["error"])
            if n < 2 and task["kind"] == "transcribe":
                with self.lock:
                    self.backlog.appendleft(task)
                return
            if task["kind"] == "diarize":
                job.diar_error = diar_hint(m["error"])
                log.warning("KONUŞMACI AYIRMA YAPILAMADI: %s  -> metin konuşmacısız yazılacak.", job.diar_error)
                self.emit("diar", job.item, ok=False, error=job.diar_error)
                (job.work / "konusmaci.json").write_text(json.dumps({"error": m["error"], "turns": []}), "utf-8")
            else:
                job.status = "failed"
                job.error = worker_hint(m["error"])
                self.emit("failed", job.item, error=m["error"])
                self.state.set(job.item.key, source=job.item.source, status="failed", title=job.item.title,
                               error=m["error"])
                with self.lock:
                    self.results.append(job)
                return
        else:
            payload = m["payload"]
            if task["kind"] == "transcribe":
                p = job.work / "parcalar" / f"{task['idx']:04d}.json"
                p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                dur = task["end"] - task["start"]
                job.done_chunks += 1
                self.emit("progress", job.item, done=job.done_chunks, total=task["n"], speed=dur / max(0.01, m["elapsed"]))
                log.info("[%s] %s parça %d/%d bitti (%s ses, %.1fx hız)", wid, job.item.title[:50],
                         task["idx"] + 1, task["n"], fmt_clock(dur), dur / max(0.01, m["elapsed"]))
            else:
                (job.work / "konusmaci.json").write_text(json.dumps(payload), encoding="utf-8")
                self.emit("diar", job.item, ok=True, speakers=len({t[2] for t in payload.get("turns", [])}))
        job.pending.discard(tid)
        if not job.pending:
            self._finalize_async(job)

    def _task_timeout(self, task: dict) -> float:
        """Takılma sınırı (sn): GPU kilitlenirse işçi yeniden başlatılır."""
        if task.get("kind") == "diarize":
            job = self.jobs.get(task.get("job", ""))
            dur = (job.info.get("duration_s", 0) if job else 0) or 3600
            return max(1800.0, 3 * dur)
        return max(900.0, 8 * (task.get("end", 0) - task.get("start", 0)))

    def _requeue(self, wid: str, tid: str, why: str) -> None:
        task = self.tasks.get(tid)
        job = self.jobs.get(task["job"]) if task else None
        if not (task and job and job.status == "run"):
            return
        n = job.attempts.get(tid, 0) + 1
        job.attempts[tid] = n
        log.error("[%s] %s; görev yeniden kuyrukta (deneme %d)", wid, why, n)
        if n < 3:
            with self.lock:
                (self.dbacklog if task["kind"] == "diarize" else self.backlog).appendleft(task)
        else:
            self._on_msg({"type": "error", "wid": wid, "tid": tid, "error": f"{why} (3 deneme)"})
            self.idle.discard(wid)

    def _health(self):
        if not self.pool:
            return
        now = time.time()
        for wid, (tid, t0) in list(self.inflight.items()):  # takılan görev: işçiyi yeniden başlat
            task = self.tasks.get(tid) or {}
            pool = self._pool_of(wid)
            if task and now - t0 > self._task_timeout(task) and wid in pool.alive():
                log.error("[%s] görev %.0f dakikadır bitmedi (GPU takıldı?); işçi yeniden başlatılıyor", wid,
                          (now - t0) / 60)
                pool.respawn(wid)
                self.restarts[wid] = self.restarts.get(wid, 0) + 1
                self.inflight.pop(wid, None)
                self._requeue(wid, tid, "işçi takıldı")
                self.idle.discard(wid)
                self.emit("worker", wid=wid, state="loading")
        main_alive: set = set()
        for pool in self._pools():
            alive = set(pool.alive())
            for wid in list(pool.workers):
                if wid in alive:
                    continue
                self.idle.discard(wid)
                if wid in self.inflight:  # işçi çöktü: görevi yeniden kuyruğa koy
                    tid, _ = self.inflight.pop(wid)
                    self._requeue(wid, tid, "işçi öldü")
                    self.idle.discard(wid)
                if wid == "diar":
                    self.dqueued_n = 0  # kuyruktaki ayırma görevi yeni süreçte de alınır
                if wid not in self.fatal and self.restarts.get(wid, 0) < 3 and not pool.threads:
                    self.restarts[wid] = self.restarts.get(wid, 0) + 1
                    log.warning("[%s] işçi yeniden başlatılıyor (%d/3)", wid, self.restarts[wid])
                    pool.respawn(wid)
                    self.emit("worker", wid=wid, state="loading")
                    alive.add(wid)
            if pool is self.pool:
                main_alive = alive
        alive = main_alive
        if not alive and self.active():
            why = next(iter(self.worker_errors.values()), "") or "GPU/model hatası"
            for job in list(self.jobs.values()):
                if job.status in ("run", "prep", "queued"):
                    job.status, job.error = "failed", f"Çalışan işçi kalmadı: {why}"
                    self.emit("failed", job.item, error=job.error)
                    self.state.set(job.item.key, source=job.item.source, status="failed", title=job.item.title,
                                   error=job.error)
                    with self.lock:
                        self.results.append(job)
            self.failed_reason = self.failed_reason or f"Tüm işçiler durdu: {why}"
            self.emit("fatal", error=self.failed_reason)
            raise RuntimeError(self.failed_reason)

    # ---------------------------------------------------------- çıktı
    def _finalize_async(self, job: Job):
        job.status = "finalize"
        self.fin.submit(self._finalize_safe, job)

    def _finalize_safe(self, job: Job):
        try:
            self._finalize(job)
        except Exception as e:  # noqa: BLE001
            log.exception("Çıktı hatası: %s", job.item.source)
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            self.emit("failed", job.item, error=job.error)
            self.state.set(job.item.key, source=job.item.source, status="failed", title=job.item.title,
                           error=job.error)
        with self.lock:
            self.results.append(job)

    def _fpl_from_previous(self, prev: dict, fpl: tuple[int, str]):
        """Metin zaten var: yalnızca FPL adımını (yeniden) çalıştır."""
        cfg = self.cfg
        want = {"gw": fpl[0], "team": fpl[1]}
        info = prev.get("info") or {}
        if info.get("fpl") == want and info.get("fpl_returncode") == 0 and not cfg.force:
            return
        out = Path(prev["out_dir"])
        jpath = Path(info.get("json") or out / f"{out.name}.json")
        if not jpath.exists():
            return
        from .fpl import run_fpl
        doc = json.loads(jpath.read_text(encoding="utf-8"))
        if not cfg.fpl_repo:
            raise RuntimeError("--fpl için FPL reposu yolu gerekli (--fpl-repo, FPL_REPO ya da stt.toml)")
        res = run_fpl(cfg.fpl_repo, fpl[0], fpl[1], to_txt(doc), cfg.fpl_python)
        info.update({"fpl": want, "fpl_returncode": res["returncode"]})
        self.state.set(prev["key"], source=prev["source"], status="done", title=prev["title"],
                       out_dir=prev["out_dir"], info=info)

    def fpl(self) -> dict | None:
        if not self._fpl_loaded:
            self._fpl_loaded = True
            self._fpl = fpl_data(self.out_root / ".stt")
        return self._fpl

    def _out_paths(self, job: Job, meta: dict) -> tuple[Path, str]:
        info, item = job.info, job.item
        title = info.get("title") or item.title or "kayit"
        d = target_dir(self.out_root, meta, title)
        base = presser_base_name(meta, title)
        # Windows 260 karakter yol sınırı: en uzun dosya "<base> - ozet.md.tmp"
        budget = MAX_PATH - len(str(d)) - len(" - ozet.md.tmp") - 12  # 12: olası " [abcdef]" eki
        if budget < 20:  # klasör adı çok uzun (ör. video başlığı): kısalt
            d = d.parent / d.name[:max(20, len(d.name) - (20 - budget))].rstrip(" .-")
            budget = MAX_PATH - len(str(d)) - len(" - ozet.md.tmp") - 12
        if len(base) > budget:
            base = base[:max(20, budget)].rstrip(" .-")
        d.mkdir(parents=True, exist_ok=True)
        marker = d / f".{base}.key"
        if marker.exists() and marker.read_text().strip() != item.key:
            base = f"{base} [{item.key[2:8]}]"
            marker = d / f".{base}.key"
        marker.write_text(item.key)
        return d, base

    def _finalize(self, job: Job):
        cfg = self.cfg
        parts = []
        for i in range(len(job.chunks)):
            p = job.work / "parcalar" / f"{i:04d}.json"
            parts.append(json.loads(p.read_text(encoding="utf-8")))
        segments, lang = merge_chunks(parts)
        speakers: dict = {}
        diar_path = job.work / "konusmaci.json"
        if not job.match:
            job.match = parse_match({**job.info, "title": job.info.get("title") or job.item.title})
        td_label = cfg.td_label
        if td_label == "Teknik direktör" and job.match.get("manager"):
            td_label = job.match["manager"]
        if self._opt(job.item, "diarize") and diar_path.exists():
            dj = json.loads(diar_path.read_text(encoding="utf-8"))
            turns = dj.get("turns", [])
            if dj.get("error") and not job.diar_error:
                job.diar_error = diar_hint(dj["error"])
            if turns:
                segments = assign_speakers(segments, turns)
                speakers = label_speakers(turns, td_label, cfg.reporter_label)
                apply_labels(segments, speakers, cfg.reporter_label)
                if len(speakers) < 2:
                    job.diar_error = "Yalnızca 1 konuşmacı bulundu (muhabir sesi çok düşük olabilir)."
            elif not job.diar_error:
                job.diar_error = "Konuşma bulunamadı."
        for i, s in enumerate(segments):
            s["id"] = i
        info = job.info
        doc = {
            "title": info.get("title") or job.item.title, "source": job.item.source,
            "webpage_url": info.get("webpage_url"), "key": job.item.key,
            "duration_s": round(info.get("duration_s", 0), 2), "language": lang or self._opt(job.item, "lang"),
            "engine": cfg.engine, "model": cfg.openai_model if cfg.engine == "openai" else cfg.model,
            "settings": {"beam_size": cfg.beam_size, "vad_filter": True, "devices": self.devices,
                         "compute_type": cfg.compute_type if any(d.startswith("cuda") for d in self.devices)
                         else ("int8" if self.devices else None),
                         "word_timestamps": True, "chunks": len(job.chunks)},
            "match": job.match, "td_label": td_label if speakers else None,
            "diarization_error": job.diar_error or None,
            "speakers": speakers, "source_info": {k: v for k, v in info.items() if k != "description"},
            "segments": [{k: s.get(k) for k in ("id", "start", "end", "speaker", "speaker_id", "text", "words",
                                                "avg_logprob", "no_speech_prob") if k in s or k in ("speaker",)}
                         for s in segments],
        }
        if not speakers:
            for s in doc["segments"]:
                s.pop("speaker", None)
        meta = classify(job.match, info, self.fpl())
        doc["dataset"] = {k: v for k, v in meta.items() if k != "fixture"}
        fpl = item_fpl(job.item, cfg)
        if not fpl and self._opt(job.item, "fpl_auto") and meta.get("gw") and meta.get("fpl_short") and cfg.fpl_repo \
                and (Path(cfg.fpl_repo) / "presser.py").is_file():
            fpl = (int(meta["gw"]), meta["fpl_short"])
        if fpl:
            doc["fpl"] = {"gw": fpl[0], "team": fpl[1]}
        out, base = self._out_paths(job, meta)
        paths = write_all(out, base, doc, f"{base} - ozet.md")
        if meta.get("team"):
            write_match_json(out, meta)
        n_q = 0
        if speakers:
            from .outputs import qa_blocks
            n_q = sum(1 for h, _, _ in qa_blocks(doc["segments"], doc.get("td_label")) if h.startswith("Soru"))
        update_index(self.out_root, {
            **{k: meta.get(k) for k in ("presser_date", "phase", "team", "opponent", "competition", "gw",
                                        "match_date", "home", "away", "manager")},
            "key": job.item.key, "league": "Premier League" if meta.get("is_pl") else "Diger",
            "language": doc["language"], "duration_s": doc["duration_s"], "questions": n_q,
            "words": sum(len(s["text"].split()) for s in doc["segments"]), "diarized": bool(speakers),
            "url": doc.get("webpage_url") or doc["source"],
            "txt": str(paths["txt"].relative_to(self.out_root)), "json": str(paths["json"].relative_to(self.out_root))})
        job.out_dir, job.result = out, doc
        fpl_res = None
        if doc.get("fpl"):
            from .fpl import run_fpl
            if not cfg.fpl_repo:
                raise RuntimeError("--fpl için FPL reposu yolu gerekli (--fpl-repo, FPL_REPO ya da stt.toml)")
            fpl_res = run_fpl(cfg.fpl_repo, fpl[0], fpl[1], to_txt(doc), cfg.fpl_python)
            (out / "fpl_presser.log").write_text(
                f"$ {' '.join(fpl_res['cmd'])}\n[çıkış {fpl_res['returncode']}]\n{fpl_res['stdout']}\n{fpl_res['stderr']}",
                encoding="utf-8")
        job.paths = paths
        self.state.set(job.item.key, source=job.item.source, status="done", title=doc["title"], out_dir=str(out),
                       info={"language": doc["language"], "duration_s": doc["duration_s"], "diarized": bool(speakers),
                             "fpl": doc.get("fpl"), "fpl_returncode": fpl_res["returncode"] if fpl_res else None,
                             "json": str(paths["json"])})
        job.status = "done"
        el = time.time() - job.t0
        self.emit("done", job.item, title=doc["title"], out_dir=str(out), txt=str(paths["txt"]),
                  json=str(paths["json"]), dataset=doc["dataset"], speakers=len(speakers),
                  diar_error=None if speakers else job.diar_error, duration=doc["duration_s"], fpl=doc.get("fpl"),
                  fpl_rc=fpl_res["returncode"] if fpl_res else None)
        log.info("BİTTİ: %s  (%s ses, %.0fs, dil=%s%s)", paths["txt"], fmt_clock(doc["duration_s"]), el,
                 doc["language"], f", {len(speakers)} konuşmacı" if speakers else "")
        if not cfg.keep_audio:
            shutil.rmtree(job.work, ignore_errors=True)

    # ---------------------------------------------------------- durdurma
    def cancel_all(self) -> int:
        """Sıradaki ve çalışan bütün işleri durdur. Biten parçalar saklanır (tekrar eklenince devam eder)."""
        n = 0
        with self.lock:
            self.cancel_gen += 1
            self.backlog.clear()
            self.dbacklog.clear()
            for job in self.jobs.values():
                if job.status in ("queued", "prep", "run"):
                    job.status, job.error = "cancelled", "Durduruldu."
                    n += 1
                    if job.item.key:
                        self.state.set(job.item.key, source=job.item.source, status="cancelled",
                                       title=job.item.title)
                    self.emit("failed", job.item, error="Durduruldu.", cancelled=True)
        log.warning("Durduruldu: %d iş. Çalışan parça bitince işçiler boşa çıkar.", n)
        return n

    # ---------------------------------------------------------- bekleme
    def active(self) -> bool:
        with self.lock:
            if self._pending_futs > 0:
                return True
            return any(j.status in ("queued", "prep", "run", "finalize") for j in self.jobs.values())

    def wait_all(self):
        while self.active():
            self.pump(0.5)

    def close(self):
        self.prep.shutdown(wait=False, cancel_futures=True)
        self.fin.shutdown(wait=True)
        for pool in self._pools():
            pool.close()
