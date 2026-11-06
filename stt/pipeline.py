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
from .fetch import Cancelled, DRMProtected, FetchError, LiveStream, SkipItem, is_transient, prepare_audio  # noqa: F401
from .merge import apply_labels, assign_speakers, label_speakers, label_speakers_all, merge_chunks
from .outputs import to_txt, write_all
from .sources import Item
from .state import State
from .dataset import classify, fpl_data, presser_base_name, target_dir, update_index, write_match_json
from .match import looks_like_presser, parse_match
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
    # canlı yayın
    live: bool = False
    live_active: bool = False
    recorder: object | None = None
    live_lang: str | None = None
    live_txt: Path | None = None
    live_next: int = 0
    live_votes: dict = field(default_factory=dict)   # dil -> toplam (süre x olasılık)
    live_buf: dict = field(default_factory=dict)     # parça no -> segmentler (sırayla yazılmayı bekleyen)
    chunk_lang: dict = field(default_factory=dict)   # parça no -> algılanan dil
    lang_checked: bool = False


def is_ram_error(err: str) -> bool:
    e = err.lower()
    return any(x in e for x in ("memoryerror", "bad_alloc", "not enough memory", "winerror 1455", "mkl_malloc",
                                "paging file", "defaultcpuallocator"))


def diar_hint(err: str) -> str:
    e = err.lower()
    if is_ram_error(err):
        return ram_hint(err)
    if "cudnn" in e or "cuda error" in e:
        return ("Konuşmacı ayırma ekran kartında başarısız oldu (işlemciye geçilmedi; sonraki videoda kart "
                "baştan açılır). Sürerse NVIDIA sürücüsünü güncelleyin; "
                "'Kontrol'ün ekran görüntüsünü paylaşın. (" + err[:160] + ")")
    if "proxyerror" in e or "max retries" in e or "connectionerror" in e or "timed out" in e:
        return "huggingface.co'ya bağlanılamadı (internet/güvenlik duvarı). Model bir kez inince internet gerekmez."
    if "gated" in e or "403" in e or "accept" in e or "koşulları" in e or "nonetype" in e:
        return ("HuggingFace model koşulları kabul edilmemiş: huggingface.co/pyannote/speaker-diarization-community-1, "
                "huggingface.co/pyannote/speaker-diarization-3.1 ve huggingface.co/pyannote/segmentation-3.0 "
                "sayfalarında 'Agree' deyin.")
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
    if info:
        n = len(info)  # nvidia-smi: ana süreçte CUDA bağlamı açılmaz
    else:
        from .engines import cuda_device_count
        n = cuda_device_count()
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
        self.ddelayed: list = []             # (zaman, görev): bellek yetmeyen ayırma, sonra tekrar denenir
        self.items: dict = {}                # uid -> Item (tek tek "Durdur" için)
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
        self._respawn_due: dict[str, float] = {}  # işçi -> yeniden başlatma zamanı (geri çekilme)
        self._llm = None

    # ---------------------------------------------------------- başlatma
    def start(self):
        cfg = self.cfg
        ecfg = {"engine": cfg.engine, "compute_type": cfg.compute_type, "beam_size": cfg.beam_size,
                "batch_size": cfg.batch_size,
                "condition_on_previous_text": cfg.condition_on_previous_text, "openai_model": cfg.openai_model,
                "diar_model": cfg.diar_model, "hf_token": cfg.hf_token, "languages": cfg.languages,
                "agc": cfg.agc, "vad_threshold": cfg.vad_threshold, "isolate_gpu": cfg.isolate_gpu,
                "gpu_limit_pct": cfg.gpu_limit_pct, "display_gpu_limit_pct": cfg.display_gpu_limit_pct,
                "ram_limit_pct": cfg.ram_limit_pct, "diar_batch": cfg.diar_batch, "diar_cpu": cfg.diar_cpu, "fill_gaps": cfg.fill_gaps}
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
        dev = self._diar_device()
        log.info("Konuşmacı ayırma: %s", dev if dev != "cpu" else "işlemci (CPU)")
        frac = None
        if dev.startswith("cuda"):  # kart belleği sınırı: monitörlü kart %70, diğeri %90 (Ayarlar: stt.toml)
            on_display = int(dev.split(":")[1]) in (self._ecfg.get("display_gpus") or [])
            pct = self.cfg.display_gpu_limit_pct if on_display else self.cfg.gpu_limit_pct
            frac = pct / 100 if pct and 0 < pct < 100 else None
        from .tune import load_tuned
        self.dpool = WorkerPool([dev], {**self._ecfg, "role": "diar", "diar_gpu_frac": frac,
                                        "diar_params": load_tuned(self.out_root / ".stt")}, names=["diar"])
        self.dpool.start()
        self.emit("worker", wid="diar", state="loading", name="Konuşmacı ayırma")

    def _diar_device(self) -> str:
        """Konuşmacı ayırma kartı: Whisper'ın kullandığı kart(lar)dan belleği en büyüğü.

        Monitörün bağlı olduğu karta, Whisper de orada değilse ASLA gidilmez (ekran kararıyordu).
        Whisper işlemcideyse ve monitörsüz bir kart varsa o; yoksa işlemci (yavaş ama güvenli)."""
        mem = self._ecfg.get("gpu_mem") or {}
        display = set(self._ecfg.get("display_gpus") or [])
        cuda = [d for d in self.devices if d.startswith("cuda")]
        if not cuda and mem and (self.cfg.gpus or "auto").lower() in ("auto", "all"):
            cuda = [f"cuda:{i}" for i in mem if i not in display]
        cuda.sort(key=lambda d: (int(d.split(":")[1]) in display, -(mem.get(int(d.split(":")[1])) or 0)))
        return cuda[0] if cuda else "cpu"

    def _pool_of(self, wid: str) -> WorkerPool | None:
        if self.dpool is not None and wid in self.dpool.workers:
            return self.dpool
        return self.pool

    def _pools(self) -> list:
        return [p for p in (self.pool, self.dpool) if p is not None]

    # ---------------------------------------------------------- olaylar (GUI)
    def emit(self, typ: str, item: Item | None = None, **data) -> None:
        if item is not None and typ in ("done", "failed", "skipped") and item.extra.get("uid"):
            try:  # iş bitti (ya da kesin olarak başarısız): kalıcı kuyruktan çıkar
                self.state.queue_remove(item.extra["uid"])
            except Exception:  # noqa: BLE001
                log.debug("kuyruk kaydı silinemedi", exc_info=True)
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

    def _speaker_mode(self, item: Item, info: dict, match: dict | None) -> str:
        """'basin' (teknik direktör + muhabir) ya da 'herkes' (podcast / film: Konuşmacı 1..N)."""
        mode = (self._opt(item, "speaker_mode") or "auto").lower()
        if mode in ("basin", "herkes"):
            return mode
        return "basin" if looks_like_presser(info, match) else "herkes"

    def _speakers_wanted(self, item: Item) -> int | None:
        n = item.extra.get("speakers") or self.cfg.min_speakers
        try:
            return int(n) if n and int(n) > 0 else None
        except (TypeError, ValueError):
            return None

    # ---------------------------------------------------------- girdi ekleme
    def submit(self, item: Item) -> None:
        import uuid
        item.extra.setdefault("uid", uuid.uuid4().hex[:10])
        item.extra["_gen"] = self.cancel_gen
        item.extra.pop("_cancel", None)
        self.items[item.extra["uid"]] = item
        try:
            self.state.queue_add(item.extra["uid"], item.kind, item.source, item.title, item.extra)
        except Exception:  # noqa: BLE001
            log.debug("kuyruk kaydı yazılamadı", exc_info=True)
        self.emit("queued", item)
        self._pending_futs += 1
        self.prep.submit(self._prepare_safe, item)

    def _prepare_safe(self, item: Item):
        try:
            tries = max(3, self.cfg.download_retries + 1)
            for attempt in range(tries):
                try:
                    self._prepare(item)
                    break
                except (MemoryError, FetchError) as e:
                    transient = isinstance(e, MemoryError) or e.transient
                    with self.lock:
                        j = self.jobs.get(item.key)
                        if j is not None and j.status == "prep":
                            del self.jobs[item.key]
                    if not transient or attempt == tries - 1 or self._cancelled(item):
                        raise
                    wait = 30 * (attempt + 1)
                    if isinstance(e, MemoryError):
                        # RAM bir anlığına doldu (tarayıcı, model yükleme...): biraz bekleyip tekrar dene
                        log.warning("Bellek (RAM) yetmedi; %d sn sonra tekrar denenecek: %s", wait, item.title[:60])
                        import gc
                        gc.collect()
                    else:
                        log.warning("Geçici ağ hatası; %d sn sonra tekrar denenecek (%d/%d): %s — %s", wait,
                                    attempt + 1, tries - 1, item.title[:60], str(e).splitlines()[-1][:160])
                    self.emit("retrying", item, wait=wait, error=str(e)[:200])
                    time.sleep(wait)
        except Cancelled:
            with self.lock:
                j = self.jobs.get(item.key)
                if j is not None and j.status in ("prep", "cancelled"):
                    j.status = "cancelled"
            log.info("İndirme durduruldu: %s", item.source)
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
        if info.get("live") and not item.extra.get("live"):
            return "Önceki çıktı canlı yayın kaydıydı; videonun tamamı yeniden işleniyor"
        if self._opt(item, "diarize") and not info.get("diarized"):
            return "Önceki çıktıda konuşmacı ayrımı yok"
        if self._opt(item, "diarize") and info.get("diarized"):
            had = info.get("speaker_mode")  # 1.5.0 öncesi çıktılarda yok = soru-cevap
            asked = (self._opt(item, "speaker_mode") or "auto").lower()
            if asked in ("basin", "herkes"):
                want = asked
            elif had:
                want = had  # otomatik karar zaten verilmiş (açıklama da kullanıldı); tekrar işleme
            else:
                title = prev.get("title") or item.title or ""
                want = self._speaker_mode(item, {"title": title}, parse_match({"title": title}))
            if want != (had or "basin"):
                return "Konuşmacı etiketleme biçimi değişti (" + ("herkes ayrı" if want == "herkes" else "soru-cevap") + ")"
        from .engines import DEFAULT_LANGS
        allowed = {x.strip() for x in (self.cfg.languages or DEFAULT_LANGS).split(",")}
        if not self._opt(item, "lang") and info.get("language") and info["language"] not in allowed:
            return f"Önceki çıktının dili yanlış algılanmış ({info['language']})"
        return ""

    def _cancelled(self, item: Item) -> bool:
        return bool(item.extra.get("_cancel")) or item.extra.get("_gen", 0) < self.cancel_gen

    def _prepare(self, item: Item):
        cfg = self.cfg
        if self._cancelled(item):
            self.emit("failed", item, error="Durduruldu.", cancelled=True)
            return
        if item.kind == "file" and not item.key:
            item.key = self.state.file_key(item.source, item.compute_key)
        item.compute_key()
        prev = self.state.get(item.key) if not (cfg.force or item.extra.get("live")) else None
        why = self._redo_reason(item, prev)
        if item.extra.get("live"):
            pass  # canlı yayın: her seferinde yeniden kaydedilir
        elif prev and prev["status"] == "done" and why:
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
        from . import fetch as _fetch
        _fetch.CANCELLED.discard(str(job.work))  # daha önce durdurulduysa yeni deneme indirebilsin
        with self.lock:
            self.jobs[item.key] = job
        self.state.set(item.key, source=item.source, status="running", title=item.title)
        log.info("Hazırlanıyor: %s", item.source)
        self.emit("prep", item)
        if item.extra.get("live"):
            self._start_live(item, job, {})
            return
        try:
            job.wav, job.info = prepare_audio(item, job.work, cfg)
            if self._cancelled(item):
                with self.lock:
                    job.status = "cancelled"
                self.state.set(item.key, source=item.source, status="cancelled", title=item.title)
                self.emit("failed", item, error="Durduruldu.", cancelled=True)
                return
        except LiveStream as e:
            log.info("Canlı yayın algılandı: %s -> canlı çeviri başlıyor", item.source)
            self._start_live(item, job, e.info)
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
        prompt = self._prompt(item, job)
        tasks = [self._transcribe_task(item, i, job.wav, c["start"], c["end"], lang, prompt, n=len(chunks),
                                       speech=(c["speech_start"], c["speech_end"]))
                 for i, c in enumerate(chunks)
                 if not (job.work / "parcalar" / f"{i:04d}.json").exists()]  # sürdürme: biten parçalar atlanır
        if self._opt(item, "diarize") and not (job.work / "konusmaci.json").exists():
            tasks.insert(0, self._diarize_task(item, job.wav))
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
                self.tasks[t["id"]] = t
            self.backlog.extend(t for t in tasks if t["kind"] != "diarize")
            self.dbacklog.extend(t for t in tasks if t["kind"] == "diarize")
        if not tasks:
            self._maybe_finalize(job)

    @staticmethod
    def _transcribe_task(item: Item, idx: int, wav: Path, start: float, end: float, lang: str | None,
                         prompt: str | None, n: int | None = None, offset: float = 0.0,
                         speech: tuple[float, float] | None = None) -> dict:
        sp = speech or (start, end)
        return {"id": f"{item.key}:t:{idx:04d}", "kind": "transcribe", "wav": str(wav), "start": start, "end": end,
                "offset": offset, "speech_start": sp[0], "speech_end": sp[1], "lang": lang, "prompt": prompt,
                "idx": idx, "n": n, "job": item.key}

    def _diarize_task(self, item: Item, wav: Path) -> dict:
        n = self._speakers_wanted(item)
        return {"id": f"{item.key}:d", "kind": "diarize", "wav": str(wav), "job": item.key,
                "min_speakers": n or self.cfg.min_speakers or 2, "max_speakers": n or self.cfg.max_speakers,
                "vocals": self._want_vocals(item)}

    def _want_vocals(self, item: Item) -> bool:
        """Konuşmacı ayırmadan önce müzik ayıklansın mı: auto = podcast / film evet, basın toplantısı hayır."""
        v = str(self._opt(item, "diar_music") or "auto").strip().lower()
        if v in ("on", "acik", "açık", "1", "true", "evet"):
            return True
        if v in ("off", "kapali", "kapalı", "0", "false", "hayir", "hayır"):
            return False
        job = self.jobs.get(item.key)
        info = {**(job.info if job else {})}
        info["title"] = info.get("title") or item.title
        match = job.match if job is not None and job.match else parse_match(info)
        return self._speaker_mode(item, info, match) == "herkes"

    def _prompt(self, item: Item, job: Job) -> str | None:
        cfg = self.cfg
        if cfg.prompt:
            return cfg.prompt
        m = job.match or {}
        bits = [m.get("manager"), " vs ".join(x for x in (m.get("home"), m.get("away")) if x)]
        if any(bits):
            return ", ".join(b for b in bits if b) + ", press conference."
        if item_fpl(item, cfg):
            from .fpl import fpl_prompt
            return fpl_prompt(item_fpl(item, cfg)[1])
        return None

    # ---------------------------------------------------------- canlı yayın
    def _start_live(self, item: Item, job: Job, info: dict) -> None:
        from .fetch import _from_ytdlp
        from .live import LiveRecorder
        from .match import safe_filename
        job.live, job.live_active = True, True
        job.work = job.work / f"canli-{time.strftime('%Y%m%d-%H%M%S')}"
        (job.work / "parcalar").mkdir(parents=True, exist_ok=True)
        job.info = {**(_from_ytdlp(info) if info else {}), "live": True}
        if job.info.get("title"):
            item.title = job.info["title"]
        job.match = parse_match({**job.info, "title": job.info.get("title") or item.title})
        title = job.info.get("title") or item.title or "canli"
        job.live_txt = self.out_root / "Canli Yayinlar" / f"{time.strftime('%Y-%m-%d %H.%M')} {safe_filename(title, 80)}.txt"
        job.live_txt.parent.mkdir(parents=True, exist_ok=True)
        job.live_txt.write_text(f"{title}\nKaynak: {job.info.get('webpage_url') or item.source}\n"
                                f"Canlı kayıt başladı: {time.strftime('%Y-%m-%d %H:%M')}\n\n", encoding="utf-8")
        with self.lock:
            job.status = "run"
        rec = LiveRecorder(item.source, job.work, self.cfg,
                           on_chunk=lambda i, wav, off, dur: self._live_chunk(job, i, wav, off, dur),
                           on_end=lambda wav, why: self._live_end(job, wav, why),
                           readrate=item.extra.get("live_readrate"),
                           on_info=lambda inf: self._live_info(job, inf))
        job.recorder = rec
        self.emit("live", item, state="start", txt=str(job.live_txt), title=title)
        log.info("CANLI: %s  (metin: %s)", title[:80], job.live_txt)
        rec.start()

    def _live_info(self, job: Job, info: dict) -> None:
        if job.info.get("title") or not info:
            return
        from .fetch import _from_ytdlp
        job.info.update({k: v for k, v in _from_ytdlp(info).items() if k not in job.info})
        if job.info.get("title"):
            job.item.title = job.info["title"]
            self.emit("live", job.item, state="info", title=job.item.title)

    def _live_chunk(self, job: Job, idx: int, wav: Path, offset: float, dur: float) -> None:
        item = job.item
        t = self._transcribe_task(item, idx, wav, 0.0, dur, self._opt(item, "lang") or job.live_lang,
                                  self._prompt(item, job), offset=offset)
        with self.lock:
            if job.status != "run":
                return
            job.chunks.append({"start": offset, "end": offset + dur, "speech_start": offset, "speech_end": offset + dur})
            job.pending.add(t["id"])
            self.tasks[t["id"]] = t
            self.backlog.appendleft(t)  # canlı parçalar öne: gecikme az olsun
        self.emit("progress", item, done=job.done_chunks, total=len(job.chunks), live=True, recorded=offset + dur)

    def _live_vote(self, job: Job, payload: dict, dur: float) -> None:
        """Canlıda dil, ilk ~60 sn'nin oylamasıyla sabitlenir (20 sn'lik tek parça yanlış dil verebiliyor:
        İngilizce yayın Felemenkçe çevrilmişti)."""
        lang = payload.get("language")
        if not lang or job.live_lang or self._opt(job.item, "lang"):
            return
        w = dur * float(payload.get("language_probability") or 0.5)
        job.live_votes[lang] = job.live_votes.get(lang, 0.0) + w
        heard = sum(c["end"] - c["start"] for c in job.chunks[:job.done_chunks])
        if heard >= 55 and job.live_votes:
            job.live_lang = max(job.live_votes, key=job.live_votes.get)
            log.info("Canlı yayın dili: %s", job.live_lang)

    def _requeue_wrong_lang(self, job: Job) -> int:
        """Çoğunluk dilinden farklı çıkan parçaları o dille yeniden çevir (bitişte, bir kez).

        Yalnızca canlı yayında: 20 sn'lik parçada dil sık yanlış algılanıyor (İngilizce yayın Felemenkçe
        yazılmıştı). Normal işlerde 5 dk'lık parçalar farklı dildeyse gerçekten o dilde konuşulmuştur
        (ör. TD İspanyolca cevap veriyor); zorla çevirmek metni değiştirir."""
        if not job.live or self._opt(job.item, "lang") or not job.chunk_lang:
            return 0
        weight: dict[str, float] = {}
        for i, lg in job.chunk_lang.items():
            if lg:
                c = job.chunks[i]
                weight[lg] = weight.get(lg, 0.0) + c["end"] - c["start"]
        if not weight:
            return 0
        main = max(weight, key=weight.get)
        n = 0
        for i, lg in job.chunk_lang.items():
            base = self.tasks.get(f"{job.item.key}:t:{i:04d}")
            if lg and lg != main and base:
                t = {**base, "id": base["id"] + "~", "lang": main, "redo": True}
                self.tasks[t["id"]] = t
                job.pending.add(t["id"])
                self.backlog.append(t)
                n += 1
        if n:
            log.info("%s: %d parça '%s' diliyle yeniden çevriliyor", job.item.title[:60], n, main)
        return n

    def _live_end(self, job: Job, wav: Path, why: str) -> None:
        item = job.item
        log.info("CANLI bitti (%s): %s", why, item.title[:70])
        with self.lock:
            job.live_active = False
            job.wav = wav
            job.info["duration_s"] = sum(c["end"] - c["start"] for c in job.chunks)
            if job.status == "run" and self._opt(item, "diarize") and wav.exists() and job.info["duration_s"] > 5:
                d = self._diarize_task(item, wav)
                self.tasks[d["id"]] = d
                job.pending.add(d["id"])
                self.dbacklog.append(d)
        self.emit("live", item, state="end", reason=why)
        self._maybe_finalize(job)

    def _live_flush(self, job: Job) -> None:
        """Biten parçaları SIRAYLA canlı metin dosyasına ekle ve arayüze gönder (parçalar sırasız bitebilir)."""
        lines = []
        while job.live_next in job.live_buf:
            for sg in job.live_buf.pop(job.live_next):
                if sg.get("text"):
                    lines.append(f"[{fmt_clock(sg['start'])}] {sg['text'].strip()}")
            job.live_next += 1
        if lines and job.live_txt:
            with open(job.live_txt, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.emit("live", job.item, state="text", lines=lines)

    def stop_live(self, uid: str) -> bool:
        for job in list(self.jobs.values()):
            if job.item.extra.get("uid") == uid and job.recorder is not None and job.live_active:
                job.recorder.stop("kullanıcı bitirdi")
                return True
        return False

    def _maybe_finalize(self, job: Job) -> None:
        with self.lock:
            if job.status != "run" or job.pending or job.live_active:
                return
            if not job.lang_checked:
                job.lang_checked = True
                if self._requeue_wrong_lang(job):
                    return
            job.status = "finalize"
        self.fin.submit(self._finalize_safe, job)

    # ---------------------------------------------------------- sonuç döngüsü
    def pump(self, timeout: float = 0.5) -> None:
        for i, pool in enumerate(self._pools()):
            msg = pool.get(timeout if i == 0 else 0.01)
            while msg is not None:
                self._on_msg(msg)
                msg = pool.get(0.01)
        self._health()
        self._dispatch()

    def _shared_gpu(self) -> set:
        """Konuşmacı ayırmayla AYNI kartta çalışan metin işçileri. Bu kartta ikisi aynı anda çalışmaz
        (aynı anda çalışınca metin işçisi 'invalid device ordinal' ile hep düşüyordu): sırayla."""
        if not self.pool or getattr(self.pool, "threads", False):
            return set()
        ddev = (getattr(self.dpool, "devices", None) or [None])[0] if self.dpool is not None else None
        if ddev is None and self.dbacklog:
            ddev = self._diar_device()
        if not ddev or not ddev.startswith("cuda"):
            return set()
        spec = getattr(self.pool, "_spec", {}) or {}
        return {n for n, d in spec.items() if d == ddev}

    def _diar_turn(self) -> bool:
        """Sıradaki konuşmacı ayırmanın işinde metin çevirisi bitti mi? Öyleyse yeni metin görevi verilmez,
        ayırma araya girer (iş çabuk biter; metin işleri sonra devam eder)."""
        for t in self.dbacklog:
            job = self.jobs.get(t["job"])
            if job and job.status == "run" and not any(
                    (self.tasks.get(x) or {}).get("kind") == "transcribe" for x in job.pending):
                return True
        return False

    def _dispatch(self):
        with self.lock:
            main = set(self.pool.workers) if self.pool else set()
            shared = self._shared_gpu()
            if shared and ("diar" in self.inflight or self.dqueued_n > 0 or self._diar_turn()):
                main -= shared  # kart konuşmacı ayırmada (ya da sırası geldi): metin görevi verilmez
            while len(self.idle & main) - self.queued_n > 0 and self.backlog:
                t = self.backlog.popleft()
                job = self.jobs.get(t["job"])
                if not job or job.status != "run":
                    continue  # iş iptal/başarısız: görevi at
                self.queued_n += 1
                self.pool.put(t)
            if self.ddelayed:
                now = time.time()
                for due, t in list(self.ddelayed):
                    busy = any(b["job"] == t["job"] for b in self.backlog)  # önce metin çevirisi bitsin
                    if now >= due and (not busy or now >= due + 900):
                        self.ddelayed.remove((due, t))
                        self.dbacklog.append(t)
            if self.dbacklog and self.dpool is None:
                self._start_diar_pool()
            didle = 1 if "diar" in self.idle else 0
            if shared and (self.queued_n > 0 or any(w in self.inflight for w in shared)
                           or (self.backlog and not self._diar_turn())):
                didle = 0  # kartta metin çevirisi sürüyor: ayırma bekler
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
            if m.get("retryable") and self.restarts.get(wid, 0) < self.cfg.worker_restarts:
                # RAM/GPU bir anlığına yetmedi ya da sürücü sıfırlandı: süreç biter, _health biraz sonra açar
                log.warning("[%s] işçi açılamadı (%s); biraz sonra tekrar denenecek", wid, m["error"][:160])
                self.idle.discard(wid)
                self.emit("worker", wid=wid, state="loading", error=hint)
                return
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
        if typ == "beat":  # işçi hâlâ çalışıyor (konuşmacı ayırma): takılma süresi baştan sayılır
            cur = self.inflight.get(wid)
            if cur and cur[0] == m.get("tid"):
                self.inflight[wid] = (cur[0], time.time())
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
        if typ == "done":
            self.restarts[wid] = 0  # çalışıyor: yeniden başlatma sayacı sıfırlanır
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
            if m.get("trace"):
                log.info("%s", m["trace"][-2500:])  # nerede düştüğü günlükte görünsün
            if m.get("restart"):  # ekran kartı takıldı: işçi kendini kapattı, _health temiz açar
                self.idle.discard(wid)
                self.emit("worker", wid=wid, state="loading")
                if task["kind"] == "transcribe" and n < 3:
                    log.warning("[%s] ekran kartı hatası; işçi yeniden açılıyor, parça tekrar denenecek", wid)
                    with self.lock:
                        self.backlog.appendleft(task)
                    return
            if n < 2 and task["kind"] == "transcribe":
                with self.lock:
                    self.backlog.appendleft(task)
                return
            skipped = job.info.setdefault("skipped_chunks", []) if task["kind"] == "transcribe" else []
            if task["kind"] == "transcribe" and not (job.live or task.get("redo")) \
                    and len(skipped) < max(1, (task.get("n") or len(job.chunks) or 1) // 4):
                # tek parça yüzünden 1 saatlik iş çöpe gitmesin: o parça boş kalır, günlükte ve json'da yazar
                skipped.append([round(task["start"], 1), round(task["end"], 1)])
                log.warning("PARÇA ATLANDI (%s–%s) çevrilemedi: %s", fmt_clock(task["start"]), fmt_clock(task["end"]),
                            m["error"][:160])
                (job.work / "parcalar" / f"{task['idx']:04d}.json").write_text(
                    json.dumps({"segments": [], "language": None}), encoding="utf-8")
                job.done_chunks += 1
                job.pending.discard(tid)
                self._maybe_finalize(job)
                return
            if task["kind"] == "transcribe" and (job.live or task.get("redo")):
                # canlı yayında tek parça yüzünden kayıt durmasın: boş geç (dil düzeltmesi olmazsa eskisi kalır)
                log.warning("Parça %d yazılamadı, atlanıyor: %s", task["idx"], m["error"][:160])
                if not task.get("redo"):
                    (job.work / "parcalar" / f"{task['idx']:04d}.json").write_text(
                        json.dumps({"segments": [], "language": None}), encoding="utf-8")
                    job.live_buf[task["idx"]] = []
                    self._live_flush(job)
                job.pending.discard(tid)
                self._maybe_finalize(job)
                return
            if task["kind"] == "diarize" and n < 3 and is_ram_error(m["error"]):
                # RAM yetmedi: ayırma sürecini kapat (belleği tamamen boşalsın), metin çevirisi bitince tekrar dene
                wait = 60 * n
                log.warning("Konuşmacı ayırmada bellek yetmedi; metin çevirisi bitince (en erken %d sn sonra) "
                            "tekrar denenecek (%d/2): %s", wait, n, job.item.title[:60])
                with self.lock:
                    self.ddelayed.append((time.time() + wait, task))
                if self.dpool is not None and "diar" in self.dpool.workers:
                    self.dpool.respawn("diar")
                    self.idle.discard("diar")
                    self.dqueued_n = 0
                    self.emit("worker", wid="diar", state="loading")
                return
            if task["kind"] == "diarize" and self.dpool is not None and "diar" in self.dpool.workers \
                    and any(x in m["error"].lower() for x in ("cuda", "cudnn", "cublas", "ekran kart")):
                # ekran kartı hatası: süreci kapat, sonraki video temiz bir CUDA ile başlasın (CPU'ya düşmesin)
                self.dpool.respawn("diar")
                self.idle.discard("diar")
                self.dqueued_n = 0
                self.emit("worker", wid="diar", state="loading")
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
                job.chunk_lang[task["idx"]] = payload.get("language")
                if not task.get("redo"):  # dil düzeltmesi için yeniden çeviri ilerlemeye sayılmaz
                    job.done_chunks += 1
                total = task["n"] or len(job.chunks)
                self.emit("progress", job.item, done=job.done_chunks, total=total, speed=dur / max(0.01, m["elapsed"]),
                          live=job.live_active or None)
                if job.live and not task.get("redo"):
                    self._live_vote(job, payload, dur)
                    job.live_buf[task["idx"]] = payload.get("segments", [])
                    self._live_flush(job)
                else:
                    log.info("[%s] %s parça %d/%d bitti (%s ses, %.1fx hız)", wid, job.item.title[:50],
                             task["idx"] + 1, total, fmt_clock(dur), dur / max(0.01, m["elapsed"]))
            else:
                (job.work / "konusmaci.json").write_text(json.dumps(payload), encoding="utf-8")
                log.info("[diar] konuşmacı ayırma bitti (%s, %.0f sn): %s", payload.get("device") or "?",
                         m.get("elapsed", 0), job.item.title[:60])
                self.emit("diar", job.item, ok=True, speakers=len({t[2] for t in payload.get("turns", [])}))
        job.pending.discard(tid)
        self._maybe_finalize(job)

    def _task_timeout(self, task: dict) -> float:
        """Takılma sınırı (sn): GPU kilitlenir / indirme donarsa işçi yeniden başlatılır.
        Konuşmacı ayırma ilerledikçe sinyal gönderir; süre SON SİNYALDEN sayılır (eskiden 3 x video süresiydi,
        takılan iş saatlerce bekliyordu)."""
        if task.get("kind") == "diarize":
            job = self.jobs.get(task.get("job", ""))
            dur = (job.info.get("duration_s", 0) if job else 0) or 3600
            return max(900.0, 0.25 * dur)
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
                n = self.restarts.get(wid, 0)
                if wid not in self.fatal and n < self.cfg.worker_restarts and not pool.threads:
                    # geri çekilmeli yeniden başlatma: hemen, 5, 10, 20, 40 sn... (sürücü/RAM toparlansın)
                    due = self._respawn_due.setdefault(wid, now + (0.0 if n == 0 else min(120.0, 5.0 * 2 ** (n - 1))))
                    if now >= due:
                        self._respawn_due.pop(wid, None)
                        self.restarts[wid] = n + 1
                        log.warning("[%s] işçi yeniden başlatılıyor (%d/%d)", wid, n + 1, self.cfg.worker_restarts)
                        pool.respawn(wid)
                        self.emit("worker", wid=wid, state="loading")
                    alive.add(wid)  # yeniden başlatılmayı bekliyor: işler başarısız sayılmasın
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

    def _voice_names(self, job: Job, dj: dict, speakers: dict, mode: str, td_label: str) -> str:
        """Konuşmacı imzalarını sakla; ses kütüphanesinde tanınan kişilere adlarını ver. Döner: TD etiketi."""
        emb = dj.get("emb")
        if not emb or not speakers:
            return td_label
        from .voices import VoiceLibrary, apply_voice_names, save_doc_voices
        st = self.out_root / ".stt"
        try:
            save_doc_voices(st, job.item.key, dj.get("model"), emb)
            if not self.cfg.voice_match:
                return td_label
            lib = VoiceLibrary(st)
            names = lib.identify(emb, dj.get("model"), self.cfg.voice_threshold)
            if names:
                log.info("Ses kütüphanesinden tanındı: %s",
                         ", ".join(f"{n} ({sim:.2f})" for n, sim in names.values()))
            return apply_voice_names(speakers, names, mode, td_label, self.cfg.reporter_label, lib.td_names())
        except Exception as e:  # noqa: BLE001
            log.warning("Ses kütüphanesi kullanılamadı: %s", e)
            return td_label

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
        mode = self._speaker_mode(job.item, {**job.info, "title": job.info.get("title") or job.item.title}, job.match)
        td_label = cfg.td_label
        if td_label == "Teknik direktör" and job.match.get("manager"):
            td_label = job.match["manager"]
        diar_meta: dict = {}
        if self._opt(job.item, "diarize") and diar_path.exists():
            dj = json.loads(diar_path.read_text(encoding="utf-8"))
            turns = dj.get("turns", [])
            diar_meta = {k: dj[k] for k in ("model", "vocals") if dj.get(k) is not None}
            if dj.get("error") and not job.diar_error:
                job.diar_error = diar_hint(dj["error"])
            if turns:
                segments = assign_speakers(segments, turns)
                if mode == "herkes":
                    speakers = label_speakers_all(turns)
                else:
                    speakers = label_speakers(turns, td_label, cfg.reporter_label)
                td_label = self._voice_names(job, dj, speakers, mode, td_label)
                apply_labels(segments, speakers, "Konuşmacı ?" if mode == "herkes" else cfg.reporter_label)
                if len(speakers) < 2:
                    job.diar_error = ("Yalnızca 1 konuşmacı bulundu." if mode == "herkes" else
                                      "Yalnızca 1 konuşmacı bulundu (muhabir sesi çok düşük olabilir).")
            elif not job.diar_error:
                job.diar_error = "Konuşma bulunamadı."
        speaker_source = "ses" if speakers else None
        if (self._opt(job.item, "diarize") and mode == "basin" and len(speakers) < 2 and segments
                and self.cfg.llm_qa and self.llm().available()):
            # ses ile ayrılamadı (muhabir sesi çok düşük / pyannote hatası): soru-cevabı metinden ayır
            try:
                from .llm import label_qa
                flags = label_qa(self.llm(), [sg["text"] for sg in segments])
                rep = sum(flags)
                if 0 < rep < len(segments):
                    for sg, f in zip(segments, flags):
                        sg["speaker_id"] = "MUHABIR" if f else "TD"
                    speakers = {"TD": {"label": td_label, "seconds": round(sum(
                                    sg["end"] - sg["start"] for sg, f in zip(segments, flags) if not f), 1)},
                                "MUHABIR": {"label": cfg.reporter_label, "seconds": round(sum(
                                    sg["end"] - sg["start"] for sg, f in zip(segments, flags) if f), 1)}}
                    apply_labels(segments, speakers, cfg.reporter_label)
                    speaker_source = "yapay zekâ (metinden)"
                    log.info("Soru-cevap yapay zekâyla ayrıldı: %d muhabir cümlesi", rep)
            except Exception as e:  # noqa: BLE001
                log.warning("Soru-cevap yapay zekâyla ayrılamadı: %s", e)
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
                         "word_timestamps": True, "chunks": len(job.chunks),
                         **({"diar_model": diar_meta["model"]} if diar_meta.get("model") else {}),
                         **({"diar_music_removed": True} if diar_meta.get("vocals") else {})},
            "match": job.match, "td_label": td_label if speakers and mode == "basin" else None,
            "speaker_mode": mode if speakers else None,
            "diarization_error": job.diar_error or None, "speaker_source": speaker_source,
            "speakers": speakers, "source_info": {k: v for k, v in info.items() if k != "description"},
            "segments": [{k: s.get(k) for k in ("id", "start", "end", "speaker", "speaker_id", "text", "words",
                                                "avg_logprob", "no_speech_prob") if k in s or k in ("speaker",)}
                         for s in segments],
        }
        if not speakers:
            for s in doc["segments"]:
                s.pop("speaker", None)
        meta = classify(job.match, info, self.fpl())
        if mode == "herkes" and self._opt(job.item, "diarize"):
            meta["general"] = True  # podcast / film: veri setinin takım klasörlerine karışmasın
        doc["dataset"] = {k: v for k, v in meta.items() if k != "fixture"}
        fpl = item_fpl(job.item, cfg)
        if not fpl and not meta.get("general") and self._opt(job.item, "fpl_auto") and meta.get("gw") and meta.get("fpl_short") and cfg.fpl_repo \
                and (Path(cfg.fpl_repo) / "presser.py").is_file():
            fpl = (int(meta["gw"]), meta["fpl_short"])
        if fpl:
            doc["fpl"] = {"gw": fpl[0], "team": fpl[1]}
        out, base = self._out_paths(job, meta)
        paths = write_all(out, base, doc, f"{base} - ozet.md")
        if (self._opt(job.item, "llm_news") or self.cfg.llm_news) and self.llm().available():
            try:
                paths.update(self.make_news(paths["json"], doc))
            except Exception as e:  # noqa: BLE001
                log.warning("Haber taslağı yazılamadı: %s", e)
        if meta.get("team") and not meta.get("general"):
            write_match_json(out, meta)
        n_q = 0
        if speakers and mode == "basin":
            from .outputs import qa_blocks
            n_q = sum(1 for h, _, _ in qa_blocks(doc["segments"], doc.get("td_label")) if h.startswith("Soru"))
        update_index(self.out_root, {
            **{k: meta.get(k) for k in ("presser_date", "phase", "team", "opponent", "competition", "gw",
                                        "match_date", "home", "away", "manager")},
            "key": job.item.key,
            "league": "Podcast ve Film" if meta.get("general") else "Premier League" if meta.get("is_pl") else "Diger",
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
                             "json": str(paths["json"]), "speaker_mode": doc.get("speaker_mode"),
                             "live": True if job.live else None})
        if job.item.extra.get("collection"):
            try:
                from .channel import add_to_collection
                add_to_collection(self.out_root, job.item.extra["collection"], doc, paths)
            except Exception:  # noqa: BLE001
                log.exception("Koleksiyon dosyası güncellenemedi")
        job.status = "done"
        el = time.time() - job.t0
        self._webhook({"event": "done", "title": doc["title"], "url": doc.get("webpage_url") or doc["source"],
                       "language": doc["language"], "duration_s": doc["duration_s"], "txt": str(paths["txt"]),
                       "json": str(paths["json"]), "news": str(paths["news"]) if paths.get("news") else None,
                       "dataset": doc["dataset"], "speakers": len(speakers), "uid": job.item.extra.get("uid"),
                       "collection": job.item.extra.get("collection")})
        self.emit("done", job.item, title=doc["title"], out_dir=str(out), txt=str(paths["txt"]),
                  json=str(paths["json"]), dataset=doc["dataset"], speakers=len(speakers),
                  diar_error=None if speakers else job.diar_error, duration=doc["duration_s"], fpl=doc.get("fpl"),
                  fpl_rc=fpl_res["returncode"] if fpl_res else None,
                  live_txt=str(job.live_txt) if job.live_txt else None,
                  news=str(paths["news"]) if paths.get("news") else None)
        log.info("BİTTİ: %s  (%s ses, %.0fs, dil=%s%s)", paths["txt"], fmt_clock(doc["duration_s"]), el,
                 doc["language"], f", {len(speakers)} konuşmacı" if speakers else "")
        if not cfg.keep_audio:
            shutil.rmtree(job.work, ignore_errors=True)

    # ---------------------------------------------------------- yapay zekâ / dış programlar
    def llm(self):
        if self._llm is None:
            from .llm import LLM
            self._llm = LLM(self.cfg, self.out_root / ".stt")
        return self._llm

    def make_news(self, json_path: Path, doc: dict | None = None) -> dict:
        """Haber taslağı (+ özet, alıntılar, etiketler): '<ad> - haber.md' ve '- haber.json'."""
        from .llm import write_news
        return write_news(self.llm(), json_path, doc)

    def _webhook(self, payload: dict) -> None:
        url = self.cfg.webhook_url
        if not url:
            return

        def send():
            import requests
            for attempt in range(4):
                try:
                    r = requests.post(url, json=payload, timeout=20)
                    if r.status_code < 500:
                        return
                except Exception as e:  # noqa: BLE001
                    log.debug("webhook: %s", e)
                time.sleep(5 * 2 ** attempt)
            log.warning("Webhook'a gönderilemedi: %s", url)
        threading.Thread(target=send, name="webhook", daemon=True).start()

    # ---------------------------------------------------------- durdurma
    def cancel_all(self) -> int:
        """Sıradaki ve çalışan bütün işleri durdur. Biten parçalar saklanır (tekrar eklenince devam eder)."""
        n = 0
        with self.lock:
            self.cancel_gen += 1
            self.backlog.clear()
            self.dbacklog.clear()
            self.ddelayed.clear()
            for job in self.jobs.values():
                if job.recorder is not None and job.live_active:
                    job.recorder.stop("durduruldu")
                if job.status in ("queued", "prep", "run"):
                    job.status, job.error = "cancelled", "Durduruldu."
                    n += 1
                    if job.item.key:
                        self.state.set(job.item.key, source=job.item.source, status="cancelled",
                                       title=job.item.title)
                    self.emit("failed", job.item, error="Durduruldu.", cancelled=True)
        log.warning("Durduruldu: %d iş. Çalışan parça bitince işçiler boşa çıkar.", n)
        return n

    def cancel(self, uid: str) -> bool:
        """Tek bir işi durdur (indiriliyor / çevriliyor / sırada). Biten parçalar saklanır."""
        from . import fetch as _fetch
        item = self.items.get(uid)
        if item is None:
            return False
        item.extra["_cancel"] = True
        kill_diar = False
        with self.lock:
            job = self.jobs.get(item.key) if item.key else None
            if job is not None and job.item.extra.get("uid") not in (None, uid):
                job = None  # aynı videonun başka kopyası: ona dokunma
            if job is not None:
                _fetch.CANCELLED.add(str(job.work))  # indirme bir sonraki ilerleme adımında durur
                if job.recorder is not None and job.live_active:
                    job.recorder.stop("durduruldu")
                if job.status in ("queued", "prep", "run"):
                    job.status, job.error = "cancelled", "Durduruldu."
                    if job.item.key:
                        self.state.set(job.item.key, source=job.item.source, status="cancelled", title=job.item.title)
                self.backlog = deque(t for t in self.backlog if t["job"] != item.key)
                self.dbacklog = deque(t for t in self.dbacklog if t["job"] != item.key)
                self.ddelayed = [x for x in self.ddelayed if x[1]["job"] != item.key]
                tid = (self.inflight.get("diar") or (None,))[0]
                kill_diar = bool(tid and (self.tasks.get(tid) or {}).get("job") == item.key)
        if kill_diar and self.dpool is not None:  # uzun ayırma görevini beklemeden kes
            self.dpool.respawn("diar")
            self.inflight.pop("diar", None)
            self.idle.discard("diar")
            self.dqueued_n = 0
            self.emit("worker", wid="diar", state="loading")
        log.warning("Durduruldu: %s", item.title[:70] or item.source)
        self.emit("failed", item, error="Durduruldu.", cancelled=True)
        return True

    # ---------------------------------------------------------- bekleme
    def active(self) -> bool:
        with self.lock:
            if self._pending_futs > 0:
                return True
            return any(j.status in ("queued", "prep", "run", "finalize") for j in self.jobs.values())

    def wait_all(self):
        while self.active():
            self.pump(0.5)

    def kill(self) -> None:
        """Uygulama kapanıyor: canlı kayıtları ve işçi süreçlerini HEMEN durdur (beklemeden)."""
        for job in list(self.jobs.values()):
            if job.recorder is not None:
                job.recorder.stop("uygulama kapandı")
        for pool in self._pools():
            if pool.threads:
                continue
            for w in pool.workers.values():
                if w.is_alive():
                    w.terminate()
            for w in pool.workers.values():
                w.join(3)

    def close(self):
        self.prep.shutdown(wait=False, cancel_futures=True)
        self.fin.shutdown(wait=True)
        for pool in self._pools():
            pool.close()
