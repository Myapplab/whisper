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
from .config import Config, diar_agc_on
from .governor import NIGHT_PREP_AHEAD, SAMPLE_S, Governor
from .flex import NOTE_L1 as FLEX_NOTE, PREP_AHEAD_L1 as FLEX_PREP_AHEAD, REST_L2 as FLEX_REST_S
from .flex import AWAY_S, COMMIT_OK_GB, RAM_OK_GB, SAMPLE_S as FLEX_SAMPLE_S, Flex, media_active, user_idle_s
from .fetch import Cancelled, DRMProtected, FetchError, LiveStream, SkipItem, is_transient, prepare_audio  # noqa: F401
from .merge import apply_labels, assign_speakers, label_speakers, label_speakers_all, merge_chunks
from .outputs import to_txt, write_all
from .sources import Item
from .state import State
from .dataset import (MAX_PATH, classify, fpl_data, index_row, out_paths, set_team_manager, update_index,  # noqa: F401
                      write_match_json)
from .match import looks_like_presser, parse_match
from .util import PAGEFILE_DUAL_HINT, fmt_clock, keep_awake, log, memory_status, text_hash
from .workers import WorkerPool

DIAR_MEM_MAX_WAIT_S = 600.0  # RAM yetmezse ayırma en çok bu kadar sırada tutulur (sonra işçi kendisi bekler/dener)
PREP_AHEAD = 6       # en çok bu kadar video önden hazırlanır (indiriliyor + metni bitmemiş); kalanı "Sırada" bekler
FPL_MEM_S = 1800     # FPL fikstür verisi bellekte en çok bu kadar tutulur (erteleme / skor için yeniden okunur)
AUTO_SPECS = ("auto", "all", "hepsi", "tum", "tümü")
HELPER_SPECS = ("yardim", "yardım", "helper")   # monitörsüz kart(lar) tam + monitörlü kart yardımcı (düşük yük)
HELPER_RECHECK_S = 30.0  # yardımcı kartta yer yoksa bu kadar sonra yeniden ölçülür
HELPER_PROBE_S = 15.0    # açık yardımcıda kart ölçümü en çok bu kadar sn'de bir (görev arası)
HELPER_PARK_S = 60.0     # başka programlar ayrılan belleği bu kadar süre aşarsa yardımcı kapatılır (bellek onlara kalır)
# yardımcının toplu boyuta göre kartta kapladığı bellek (GB, tahmin + pay). faster-whisper README (large-v2 fp16,
# beam 5, RTX 3070 Ti): tekli 4.5 GB, toplu 8 6.1 GB. Gerçekte fazla gelirse kapanır, sonra bir küçük boyutla açılır.
HELPER_FOOT_GB = {8: 6.4, 4: 5.6, 2: 5.2}
HELPER_REGROW_S = 1800.0  # küçültülen toplu boyut / öğrenilen bellek bu kadar sonra varsayılana döner
HELPER_IDLE_PARK_S = 300.0  # bu kadar süre iş fazlası olmazsa yardımcı kapatılır (model belleği karta geri döner)
HELPER_OFF = "kapalı · gerekince açılır"
HELPER_DIAR = "diar2"     # yardımcı kartta ikinci konuşmacı ayırma işçisi (3090'daki "diar" yetişemeyince)
HELPER_DIAR_GB = 3.0      # konuşmacı ayırma işçisinin kartta kapladığı bellek (CUDA + pyannote, toplu 8; tahmin)
HELPER_DIAR_WAIT_S = 15.0  # ayırma işi bu kadar sn ana işçide beklemedikçe yardımcıda ayırma açılmaz (açılış anı)
HELPER_SWITCH_S = 20.0    # yardımcı kart bu kadar sn işsiz kalırsa öbür işe (metin <-> konuşmacı ayırma) geçer
NIGHT_NOTE = "kapalı · gece modu (tek kart)"
FLEX_RESERVE_GB = 1.0      # esnek ikinci kartta (monitörsüz) başka programlara bırakılan bellek: tarayıcı Intel'le çizer
DISK_CHECK_S = 30.0       # diskteki boş yer en çok bu kadar sn'de bir ölçülür
OLD_PREP_WAIT_S = 120.0   # yeni motor aynı işin klasöründe eski motorun indirmesi bitene dek en çok bu kadar bekler
PREP_POLL_S = 1.0
_PREP_LOCK = threading.Lock()
_PREP_ACTIVE: dict[str, int] = {}  # çalışma klasörü -> süren hazırlık (motorlar arası: kapanan motorun indirmesi)


def _prep_enter(path: str) -> None:
    with _PREP_LOCK:
        _PREP_ACTIVE[path] = _PREP_ACTIVE.get(path, 0) + 1


def _prep_exit(path: str) -> None:
    with _PREP_LOCK:
        n = _PREP_ACTIVE.get(path, 0) - 1
        if n > 0:
            _PREP_ACTIVE[path] = n
        else:
            _PREP_ACTIVE.pop(path, None)


def _prep_busy(path: str) -> bool:
    with _PREP_LOCK:
        return path in _PREP_ACTIVE
# ikinci kart (metin ya da konuşmacı ayırma) kendi belleğiyle bilgisayarı "bellek az" eşiğinin altına itmesin (1.9.27):
# açılmak için boş RAM >= RAM_OK_GB + kartın payı, boş sanal bellek >= COMMIT_OK_GB + kartın payı. Pay ölçülür
# (açılıştan sonraki ilk HELPER_FOOT_LEARN_S'de boş bellekteki en büyük düşüş), bilinmiyorsa HELPER_FOOT_DEFAULT.
# İnceleme: eşikler payı hesaba katmayınca kart aç -> bellek az -> bekle -> kapan -> aç döngüsünde saatte 40 model
# yüklüyordu. Bellek yüzünden kapatılan kart HELPER_MEM_COOL_S (tekrarında iki katı, en çok saat) açılmaz.
HELPER_FOOT_DEFAULT = {"ram": 2.0, "commit": 3.5}
HELPER_FOOT_LEARN_S = 600.0
HELPER_MEM_COOL_S = 900.0
HELPER_MEM_COOL_MAX_S = 3600.0
PRUNE_S = 60.0            # biten işlerin belleği en çok bu kadar sn'de bir temizlenir (arayüz modu)
KEEP_FINISHED = 200       # arayüz modunda bellekte tutulan son biten iş
_PAGEFILE_WARNED: set = set()  # sayfa dosyası uyarısı her veri klasörü için uygulama açıkken bir kez
PAGEFILE_DUAL_GB = 40.0   # iki kart çalışacakken sanal bellek sınırı bundan küçükse açılışta bir kez uyarılır
MEM_NOTICE_S = 3600.0     # bellek azaldı bildirimi en çok saatte bir
STATS_S = 3600.0          # günlüğe "[verim]" satırı aralığı


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


def _pos_int(v) -> int | None:
    """Pozitif tam sayı ya da None ("x", 0, -1, None -> None)."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _finished(p: Path) -> bool:
    """Sürdürmede bu parça / konuşmacı dosyası bitmiş sayılır mı? Atlanan parça ve başarısız ayırma sayılmaz:
    "Tekrar dene"de yeniden denenir (eskiden boş dosya "bitti" sayılıyor, hiç yeniden denenmiyordu)."""
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return not (isinstance(d, dict) and (d.get("skipped") or d.get("error")))


def is_ram_error(err: str) -> bool:
    e = err.lower()
    return any(x in e for x in ("memoryerror", "bad_alloc", "not enough memory", "winerror 1455", "mkl_malloc",
                                "paging file", "defaultcpuallocator"))


def is_oom(err: str) -> bool:
    """Ekran kartı (ya da sistem) belleği yetmedi: "CUDA out of memory" / "CUDA error: out of memory"."""
    e = err.lower()
    return "out of memory" in e or "outofmemory" in e


def diar_hint(err: str) -> str:
    e = err.lower()
    if is_ram_error(err):
        return ram_hint(err)
    if is_oom(err):  # sürücü sorunu değil: bellek doldu (ör. tarayıcı sekmeleri)
        return ("Konuşmacı ayırmada bellek yetmedi (ekran kartı / bilgisayar belleği). " + ram_hint(err))
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
    if "yedek pyannote" in e:
        return ("DiariZen bu videoda çalışmadı (ayrıntı günlükte) ve yedek pyannote için HF token yok: Ayarlar > "
                "HuggingFace token girin ya da günlükteki 'DiariZen' satırlarını paylaşın.")
    if "token gerekli" in e:
        return "HF token yok: huggingface.co'dan token alıp uygulamada Ayarlar > HuggingFace token alanına yapıştırın."
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


def flex_pair(info: dict | None) -> tuple[int, int] | None:
    """Esnek tam güç (1.9.21): tam iki NVIDIA kart ve ikisinde de monitör yok (monitör anakartta; biliniyor) ->
    (ana kart, esnek ikinci kart). Ana kart (belleği büyük, 3090) metin + konuşmacı ayırma; ikinci kart (3080) iş
    nerede birikmişse ona geçer (metin ya da konuşmacı ayırma; darboğaz çoğu zaman ayırma) ve başka programlar belleği
    / işlemciyi isteyince geri çekilir (yardımcı kart düzeni). Eskiden ikinci kart yalnız metin çeviriyordu."""
    if not info or len(info) != 2 or any(g.get("display") is not False for g in info.values()):
        return None
    a, b = sorted(info, key=lambda i: (-(info[i].get("gb") or 0), i))
    return a, b


def detect_devices(spec: str, info: dict | None = None, flex: bool = False) -> list[str]:
    """spec: "auto" (monitörsüz kart(lar)) | "yardim" (ana kartlar = auto; yardımcılar `helper_devices`) |
    "all" | "0,1" | "cuda:0" | "cpu" | "cpu,cpu" (test). flex: "auto"da iki boş kart varsa ana kart yalnız büyük kart
    (ikincisi esnek: ``helper_devices``)."""
    spec = (spec or "auto").strip().lower()
    if spec == "auto" and flex and flex_pair(info):  # "yardim" değil: orada ikinci kart monitörlü yardımcıdır
        return [f"cuda:{flex_pair(info)[0]}"]
    if spec in HELPER_SPECS:
        spec = "auto"
    if spec not in AUTO_SPECS:
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


def helper_devices(spec: str, info: dict | None, flex: bool = False) -> list[str]:
    """"yardim" modunda ana kartların (auto) dışında kalan kartlar: yalnız ana kart yetişemeyince metin çevirir.
    Kart bilgisi yoksa (nvidia-smi yok) ya da tek kart varsa yardımcı yok (monitörlü karta körlemesine gidilmez).
    flex: "auto"da iki boş kart varsa ikinci kart esnek (``flex_pair``)."""
    s = (spec or "").strip().lower()
    if s == "auto" and flex and flex_pair(info):
        return [f"cuda:{flex_pair(info)[1]}"]
    if s not in HELPER_SPECS or not info or len(info) < 2:
        return []
    from .engines import choose_gpus
    main = set(choose_gpus(len(info), info, "auto"))
    rest = [i for i in sorted(info) if i not in main]
    if not rest:
        return []
    # tek yardımcı (belleği en büyük): tek işçili havuz, kapanma işareti yalnız ona gider (WorkerPool.stop)
    return [f"cuda:{max(rest, key=lambda i: (info[i].get('gb', 0), -i))}"]


def auto_spec(spec: str | None) -> bool:
    """Kart seçimi uygulamada mı (elle kart numarası verilmemiş)?"""
    s = (spec or "auto").strip().lower()
    return s in AUTO_SPECS or s in HELPER_SPECS


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
        self._fpl_at = 0.0                   # FPL verisinin okunduğu an
        self.events = None                   # isteğe bağlı: GUI için olay geri çağırma fonksiyonu
        self.results: list[Job] = []
        self._pending_futs = 0
        self.backlog: deque = deque()        # ortak kuyruk; boşta işçi oldukça dağıtılır
        self.dbacklog: deque = deque()       # konuşmacı ayırma görevleri (ayrı süreç: "diar")
        self.ddelayed: list = []             # (zaman, görev): bellek yetmeyen ayırma, sonra tekrar denenir
        self.items: dict = {}                # uid -> Item (tek tek "Durdur" için)
        self.dpool: WorkerPool | None = None
        self.dqueued_n = 0
        # yardımcı kart (gpus="yardim"): ayrı havuz, ayrı kuyruk; görev yalnız ana kart(lar) yetişemeyince verilir
        self.hpool: WorkerPool | None = None
        self.helper_devices: list[str] = []
        self.hqueued_n = 0
        self._h_ready: dict[str, float] = {}     # yardımcı -> en erken yeni görev zamanı (dinlenme / meşgul)
        self._h_probe: dict[str, tuple] = {}     # yardımcı -> (ölçüm zamanı, gpu_load sonucu)
        self._h_probing: set[str] = set()
        self._h_busy_since: dict[str, float] = {}
        self._h_parked: dict[str, str] = {}      # kapatılmış yardımcı -> sebep (kart başka programda)
        self._h_note: dict[str, str] = {}        # arayüze son bildirilen bekleme sebebi
        self._h_gone: set[str] = set()           # bir daha açılmayacak yardımcılar (görevleri geri alındı)
        self._h_last: dict[str, float] = {}      # yardımcı -> son görev / hazır olma zamanı (boşta kapatma için)
        self._h_cap: dict[str, int] = {}         # yardımcı -> en büyük toplu boyut (bellek yetmeyince küçülür)
        self._h_cap_max: dict[str, int] = {}     # yardımcı -> kartın asıl en büyük toplu boyutu (geri büyüme)
        self._h_shrunk_at: dict[str, float] = {}  # yardımcı -> son küçültme zamanı
        self._h_foot: dict[str, dict] = {}       # yardımcı -> toplu boyut -> kapladığı bellek (GB; ölçümle büyür)
        self._h_open_free: dict[str, float] = {}  # yardımcı -> açılırken kartta boş bellek (kendi payını ölçmek için)
        self._h_own: dict[str, float] = {}       # yardımcı -> ölçülen kendi payı (GB; açılış başına bir kez)
        self._h_done: dict[str, int] = {}        # yardımcı -> açılıştan beri biten görev
        # yardımcı kartta konuşmacı ayırma (wid HELPER_DIAR): kart aynı anda TEK iş tutar (metin ya da ayırma;
        # ikisi belleğe sığmaz, aynı kartta torch + CTranslate2 da sırayla çalışmak zorunda)
        self.hdpool: WorkerPool | None = None
        self.hdqueued_n = 0
        self._d_need_since: float | None = None   # ana ayırma işçisinin alamadığı ayırma işi ne zamandır var
        self._w_need_since: float | None = None   # ana kartın alamadığı metin işi ne zamandır var
        self.helper_disabled = ""                  # dolu = yardımcı kart kapatıldı (ör. ekran sürücüsü çöktü), sebep
        self._h_batch: dict[str, int] = {}       # yardımcı -> şu an yüklü modelin toplu boyutu
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
        self._loading_since: dict[str, float] = {}  # işçi -> model yüklemeye başladığı an (takılma denetimi)
        self._llm = None
        self._closing = False
        self._dmem_at = 0.0          # ayırma için RAM denetimi (ana süreçte): son ölçüm zamanı
        self._dmem_ok = True
        self._dmem_since: float | None = None
        self._dmem_forced = False    # bekleme sınırı doldu: ayırma görevi verilene dek RAM'e bakılmaz
        self._drop_wids: set[str] = set()  # ekran koruması: ilk görevinde kapatılacak (monitörlü kart) ana işçiler
        # esnek tam güç (stt/flex.py): başka programlar işlemciyi / RAM'i kullanınca kademeli yavaşla, sonra hızlan
        self.flex = Flex()
        self._flex_sampler = None
        self._flex_at = 0.0
        self._flex_probing = False
        self._flex_was = False
        self._flex_rest_until = 0.0            # kademe 2: bu zamana dek ana karta yeni görev verilmez
        self._flex_card = False                # ikinci kart "esnek" düzende (monitörsüz; yardımcı düzeniyle)
        self._flex_card_possible = False       # iki kart monitörsüz ve "Önerilen": esnek düzen açılabilir
        self._memstat = memory_status          # RAM / sanal bellek ölçümü (testlerde değiştirilir)
        self._idle = user_idle_s               # klavye / fare en son kaç sn önce (Windows; bilinmiyorsa None)
        self._mem_noticed = 0.0
        self._media = media_active             # tam ekran / video oynatılıyor mu (kullanıcı dokunmadan izliyor olabilir)
        self._h_memfoot: dict[str, dict] = {}  # ikinci kart işçisi -> ölçülen bellek payı {"ram", "commit"} (GB)
        self._h_memopen: dict[str, tuple] = {}  # açılış: (zaman, boş RAM, boş sanal bellek, en büyük düşüş {ram, commit})
        self._h_cool_until = 0.0               # bellek yüzünden kapatılan ikinci kart bu zamana dek açılmaz
        self._h_cool_s = 0.0
        self._h_cool_last = 0.0
        self._q_marked: set[str] = set()       # ekran kartında görevi süren işler (kalıcı kuyrukta "started")
        self.keep_results = True               # komut satırı özet yazar; arayüz False: biten metinler bellekte birikmesin
        self._prune_at = 0.0
        # verim (1.9.27): biten / hatalı işler (zaman, ses sn) ve saatlik özet (günlükte "[verim]", yerel API'de stats)
        self._stat_done: deque = deque(maxlen=20000)
        self._stat_failed: deque = deque(maxlen=20000)
        self._stat_n = {"done": 0, "failed": 0}
        self._stat_hour = {"t0": time.time(), "ram_min": None, "commit_min": None, "level_s": [0.0, 0.0, 0.0],
                           "last": None}
        # gece modu (stt/governor.py): ısıya göre görev arası mola, soğuma molası, işçi yenileme, tek kart
        self.gov = Governor(cfg.night_temp)
        self._rest_until = 0.0                 # bu zamana dek ekran kartına yeni görev verilmez (gece modu molası)
        self._gov_at = 0.0
        self._gov_probing = False
        self._night_was: bool | None = None
        self._night_drops: set[str] = set()    # gece modunda kapatılacak öbür ana kart işçileri
        self._night_active = False             # son "night" olayında kuyrukta iş var mıydı
        self._born: dict[str, float] = {}      # işçi -> hazır olduğu an (gece modunda yenileme için)
        self._gi: dict = {}
        self._disk_lock = threading.Lock()
        self._disk_at = 0.0
        self._disk_ok = True
        self._disk_warned = False
        self._disk_warned_at = 0.0

    # ---------------------------------------------------------- başlatma
    def start(self):
        cfg = self.cfg
        ecfg = {"engine": cfg.engine, "compute_type": cfg.compute_type, "beam_size": cfg.beam_size,
                "batch_size": cfg.batch_size,
                "condition_on_previous_text": cfg.condition_on_previous_text, "openai_model": cfg.openai_model,
                "openai_api_key": cfg.openai_api_key,
                "diar_model": cfg.diar_model, "diar_engine": cfg.diar_engine, "hf_token": cfg.hf_token, "languages": cfg.languages,
                "agc": cfg.agc, "diar_agc": diar_agc_on(cfg), "vad_threshold": cfg.vad_threshold,
                "isolate_gpu": cfg.isolate_gpu,
                "gpu_limit_pct": cfg.gpu_limit_pct, "display_gpu_limit_pct": cfg.display_gpu_limit_pct,
                "ram_limit_pct": cfg.ram_limit_pct, "diar_batch": cfg.diar_batch, "diar_cpu": cfg.diar_cpu, "fill_gaps": cfg.fill_gaps}
        if cfg.engine == "openai":
            devices = ["api"] * max(1, cfg.openai_workers)
            self.pool = WorkerPool(devices, ecfg, threads=True)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # cuda:N == nvidia-smi N
            if os.environ.get("CUDA_VISIBLE_DEVICES") and auto_spec(cfg.gpus):
                log.info("CUDA_VISIBLE_DEVICES=%s kaldırıldı (kart seçimini uygulama yapar).",
                         os.environ.pop("CUDA_VISIBLE_DEVICES"))
            from .engines import gpu_info
            gi = gpu_info()
            self._gi = gi
            flex = bool(getattr(cfg, "gpu_flex", True))
            self._flex_card_possible = (cfg.gpus or "auto").strip().lower() == "auto" and flex_pair(gi) is not None
            devices = detect_devices(cfg.gpus, gi, flex=flex)
            self.devices = devices
            if all(d.startswith("cuda") for d in devices):
                self.helper_devices = helper_devices(cfg.gpus, gi, flex=flex)
                # iki kart birlikte yalnız güvenli güç ayarıyla (bilgisayar yeniden başlıyordu; stt/power.py)
                try:
                    from .power import card_line, restrict_or_reapply
                    devices, self.helper_devices, why, cards = restrict_or_reapply(devices, self.helper_devices, gi, cfg)
                    for i, c in sorted(cards.items()):
                        log.info("[güç] cuda:%d %s, sınır %s W (varsayılan %s W)", i, card_line(c),
                                 f"{c['limit']:.0f}" if c.get("limit") else "?", f"{c['default']:.0f}" if c.get("default") else "?")
                except Exception:  # noqa: BLE001
                    log.debug("güç sınırı okunamadı", exc_info=True)
                    why = ""
                if why:
                    self.devices = devices
                    log.warning("%s", why)
                    self.emit("notice", text=why)
                # ikinci kart esnek düzende mi (güç sınırı yüzünden düştüyse değil)
                self._flex_card = bool(self.helper_devices) and self._flex_card_possible and flex
                if len({d for d in list(devices) + list(self.helper_devices) if d.startswith("cuda")}) >= 2:
                    self._pagefile_check()  # iki kart: sanal bellek küçükse kalıcı çözüm bir kez söylenir (1.9.27)
            ecfg["gpu_mem"] = {i: g["gb"] for i, g in gi.items()}
            ecfg["display_gpus"] = [i for i, g in gi.items() if g.get("display")]
            ecfg["display_batch"] = cfg.display_batch
            ecfg["low_priority"] = cfg.low_priority
            used = {int(d.split(":")[1]) for d in devices if d.startswith("cuda")}
            helpers = {int(d.split(":")[1]) for d in self.helper_devices}
            for i, g in sorted(gi.items()):
                mon = " [monitör bağlı]" if g.get("display") else ""
                state = "kullanılıyor" if i in used else ("ESNEK İKİNCİ KART (metin ya da konuşmacı ayırma, iş nerede "
                    "birikmişse; başka programlar isteyince geri çekilir)" if self._flex_card else
                    "YARDIMCI (ana kart yetişemeyince, düşük yükle)") if i in helpers else "BOŞ bırakıldı (masaüstü/tarayıcı için)"
                log.info("cuda:%d = %s (%.0f GB)%s -> %s", i, g["name"], g["gb"], mon, state)
            if any(i in ecfg["display_gpus"] for i in used):
                log.info("Monitörün bağlı olduğu kart kullanılıyor: toplu boyut en fazla %d (bilgisayar akıcı kalsın).",
                         cfg.display_batch)
                self._mark_helper()  # ekran koruması: monitörlü kart iş yapıyor (aralık kapanışta kapanır)
            if all(d == "cpu" for d in devices):
                log.warning("CUDA GPU bulunamadı -> CPU (int8) ile çalışılacak (yavaş).")
            log.info("Model hazırlanıyor: %s", cfg.model)
            from .engines import resolve_model_path
            ecfg["model_path"] = resolve_model_path(cfg.model)
            self.pool = WorkerPool(devices, ecfg)
            if self.helper_devices:
                from .engines import auto_batch_size
                for d in self.helper_devices:
                    gb = (gi.get(int(d.split(":")[1])) or {}).get("gb")
                    self._h_cap[d] = cfg.batch_size if cfg.batch_size >= 1 else auto_batch_size("cuda", gb)
                    self._h_cap_max[d] = self._h_cap[d]
                log.info("%s: başka programlar için %.1f GB boş bırakılır, kalan bellek bu işe verilir "
                         "(toplu boyut ona göre, en çok %s); en düşük işlemci önceliği. Kart aynı anda tek iş yapar: metin ya da "
                         "(3090 yetişemeyince) konuşmacı ayırma.", "Esnek ikinci kart" if self._flex_card else "Yardımcı kart",
                         self._reserve(), "/".join(str(v) for v in self._h_cap.values()))
                # modeller iki havuzda da SIRAYLA yüklensin (ortak kilit: aynı anda iki 3 GB yükleme RAM'i tüketiyordu)
                self.hpool = WorkerPool(self.helper_devices, {**ecfg, "helper": True},
                                        load_lock=getattr(self.pool, "load_lock", None))
        self.pool.start()
        for wid in (self.hpool.workers if self.hpool is not None else ()):
            # yardımcı baştan açılmaz: model, iş fazlası çıkınca (ve kart boşsa) yüklenir; o zamana dek bellek karta kalır
            self._h_parked[wid] = HELPER_OFF
            self._h_note[wid] = HELPER_OFF
        if self.hpool is not None:
            self._h_parked[HELPER_DIAR] = HELPER_OFF
            self._h_note[HELPER_DIAR] = HELPER_OFF
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
        from .engines import diar_batch_size
        from .tune import load_tuned
        gb = (self._ecfg.get("gpu_mem") or {}).get(int(dev.split(":")[1])) if dev.startswith("cuda") else None
        batch = diar_batch_size(self.cfg.diar_batch, gb)
        log.info("Konuşmacı ayırma toplu boyutu: %d", batch)
        self.dpool = WorkerPool([dev], {**self._ecfg, "role": "diar", "diar_gpu_frac": frac, "diar_batch": batch,
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
        if not cuda and mem and auto_spec(self.cfg.gpus):
            cuda = [f"cuda:{i}" for i in mem if i not in display]
        cuda.sort(key=lambda d: (int(d.split(":")[1]) in display, -(mem.get(int(d.split(":")[1])) or 0)))
        return cuda[0] if cuda else "cpu"

    def _pool_of(self, wid: str) -> WorkerPool | None:
        if self.dpool is not None and wid in self.dpool.workers:
            return self.dpool
        if self.hpool is not None and wid in self.hpool.workers:
            return self.hpool
        if self.hdpool is not None and wid in self.hdpool.workers:
            return self.hdpool
        return self.pool

    def _pools(self) -> list:
        return [p for p in (self.pool, self.dpool, self.hpool, self.hdpool) if p is not None]

    def _is_helper(self, wid: str | None) -> bool:
        """Yardımcı kartta METİN işçisi mi."""
        return self.hpool is not None and wid in self.hpool.workers

    def _is_hdiar(self, wid: str | None) -> bool:
        """Yardımcı kartta KONUŞMACI AYIRMA işçisi mi."""
        return wid == HELPER_DIAR and self.hdpool is not None

    def _w_on(self) -> bool:
        """Yardımcı kartta metin işçisi açık (ya da açılıyor) mu. Kalıcı düşen sayılmaz (öbür işi kilitlemesin)."""
        return self.hpool is not None and any(w not in self._h_parked and w not in self.fatal and w not in self._h_gone
                                              for w in self.hpool.workers)

    def _d_on(self) -> bool:
        """Yardımcı kartta konuşmacı ayırma işçisi açık (ya da açılıyor) mu. Kalıcı düşen sayılmaz."""
        return self.hdpool is not None and HELPER_DIAR not in self._h_parked and HELPER_DIAR not in self.fatal \
            and HELPER_DIAR not in self._h_gone

    # ---------------------------------------------------------- olaylar (GUI)
    def emit(self, typ: str, item: Item | None = None, **data) -> None:
        if item is not None and (typ == "done" or (typ == "failed" and not data.get("cancelled"))):
            now = time.time()
            if typ == "done":
                self._stat_done.append((now, float(data.get("duration") or 0)))
            else:
                self._stat_failed.append(now)
            self._stat_n[typ] += 1
        # motor kapanırken (hata / yeniden açılış) düşen iş kuyruktan silinmez: yeni motor onu sürdürür (1.9.27;
        # eskiden kapanışta sıra bekleyen işler "Durduruldu" olayıyla kalıcı kuyruktan siliniyordu)
        if item is not None and item.extra.get("uid") and (typ == "done" or (
                typ in ("failed", "skipped") and not self._closing)):
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
        """Bu iş için verilen kesin konuşmacı sayısı (arayüzde 'kişi sayısı'); yoksa None."""
        return _pos_int(item.extra.get("speakers"))

    def _speaker_bounds(self, item: Item, mode: str) -> tuple[int | None, int | None]:
        """Konuşmacı ayırmaya verilecek (en az, en çok) kişi sayısı.

        - İşe verilen sayı kesindir (en az = en çok). CLI ``--speakers N`` de genel ayarda ikisini birden N yapar.
        - Genel ``min_speakers`` / ``max_speakers`` (stt.toml) yalnız sınırdır; eskiden ``min_speakers`` tek başına
          yazılınca kesin sayı gibi davranıyordu (en çok da o sayı oluyordu).
        - Basın toplantısında en az 2 (hoca + muhabir; yoksa muhabir hocanın içine karışıyordu). Podcast / film'de
          alt sınır yok: pyannote 4 (VBx) bulduğu kişi sayısı alt sınırdan azsa sesi zorla böler, tek kişilik video
          iki "konuşmacı" oluyordu."""
        n = self._speakers_wanted(item)
        if n:
            return n, n
        lo, hi = _pos_int(self.cfg.min_speakers), _pos_int(self.cfg.max_speakers)
        if lo is None and mode == "basin":
            lo = 2
        if lo and hi and lo > hi:
            lo = hi
        return lo, hi

    def _item_mode(self, item: Item) -> str:
        """İşin konuşmacı biçimi ('basin' / 'herkes'); bitirmedeki (`_finalize`) kararla aynı bilgiyle."""
        job = self.jobs.get(item.key)
        info = {**(job.info if job else {})}
        info["title"] = info.get("title") or item.title
        match = job.match if job is not None and job.match else parse_match(info)
        return self._speaker_mode(item, info, match)

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
        if item.extra.get("live"):  # canlı yayın sıra beklemesin (hazırlık iş parçacıkları sırada bekliyor olabilir)
            threading.Thread(target=self._prepare_safe, args=(item,), name="canli-hazirlik", daemon=True).start()
        else:
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

    def _prep_ahead(self) -> int:
        """Önden hazırlanmış (indiriliyor ya da metin çevirisi henüz bitmemiş) iş sayısı."""
        with self.lock:
            return sum(1 for j in self.jobs.values()
                       if not j.live and (j.status == "prep" or (j.status == "run" and any(
                           (self.tasks.get(x) or {}).get("kind") == "transcribe" for x in j.pending))))

    def _wait_prep_turn(self, item: Item) -> bool:
        """Kuyruğun tamamı önden indirilmesin: en çok ``PREP_AHEAD`` video hazırlanmış / çevriliyor olsun. Eskiden
        100 videonun hepsi saatlerce önden indiriliyor, VAD'den geçiriliyor (i3'te GPU işçileriyle yarış, fan; diskte
        100 x ses.wav; arayüzde 100 kart birden "Metne çevriliyor 0/N"). Sonuç değişmez. Döner: False = durduruldu."""
        while not (self._pending_ahead_ok() and self._disk_free_ok()):
            if self._cancelled(item) or self._closing:
                return False
            time.sleep(1.0)
        return not (self._cancelled(item) or self._closing)

    def _pending_ahead_ok(self) -> bool:
        if self._mem_crit():
            return False  # bellek kritik (1.9.27): yeni video indirilmez; çalışan işler biter, bellek açılınca sürer
        limit = NIGHT_PREP_AHEAD if getattr(self.cfg, "night_mode", False) else PREP_AHEAD
        if self._flex_level() >= 1:
            limit = min(limit, FLEX_PREP_AHEAD)  # bilgisayar kullanılıyor: indirme / ffmpeg işlemciyi doldurmasın
        return self._prep_ahead() < limit

    def _disk_free_ok(self) -> bool:
        """Çıktı klasörünün diskinde ``min_free_gb``'den az yer varsa yeni video indirilmez (çalışan işler biter, yer
        açılınca kendiliğinden sürer). Eskiden disk dolunca kuyruktaki her video tek tek "başarısız" oluyordu (200
        videoluk gece kuyruğu sabaha boşa giderdi)."""
        need = float(getattr(self.cfg, "min_free_gb", 0) or 0)
        if need <= 0:
            return True
        with self._disk_lock:
            now = time.time()
            if now - self._disk_at < DISK_CHECK_S:
                return self._disk_ok
            self._disk_at = now
            try:
                free = shutil.disk_usage(self.out_root).free / 2 ** 30
            except OSError:
                self._disk_ok = True
                return True
            ok = free >= need
            if not ok and (not self._disk_warned or now - self._disk_warned_at >= 1800):  # yarım saatte bir hatırlat
                self._disk_warned_at = now
                msg = (f"Diskte yer azaldı ({free:.1f} GB boş; en az {need:.0f} GB olmalı): yeni video indirilmiyor, "
                       "çalışan işler bitiyor. Çıktı klasörünün olduğu diskte yer açın (Geri Dönüşüm Kutusu, eski "
                       "indirmeler); yer açılınca kuyruk kendiliğinden devam eder.")
                log.warning("%s", msg)
                self.emit("notice", text=msg)
            elif ok and self._disk_warned:
                log.info("Diskte yer açıldı (%.1f GB boş): indirmeler sürüyor", free)
            self._disk_warned = not ok
            self._disk_ok = ok
            return ok

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
        if info.get("skipped_chunks"):
            return "Önceki çıktıda çevrilemeyen bölüm vardı; yeniden işleniyor"
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

    def _note_cancelled(self, item: Item) -> None:
        """Kullanıcının durdurduğu iş kalıcı kayıtta "cancelled" olsun (yerel API /api/lookup: ajan yeniden göndermesin;
        eskiden işe başlanmadan durdurulan link "unknown" görünüyordu)."""
        if item.kind != "url":
            return  # dosyanın anahtarı içerik özeti: yalnız bunun için okunmasın
        try:
            key = item.key or item.compute_key()
            prev = self.state.get(key)
            if not (prev and prev["status"] == "done"):
                self.state.set(key, source=item.source, status="cancelled", title=item.title)
        except Exception:  # noqa: BLE001
            log.debug("durdurma kaydı yazılamadı", exc_info=True)

    def _wait_old_prep(self, work: Path, item: Item) -> bool:
        """Kapanan (hata veren) motorun aynı işteki indirmesi / dönüştürmesi bitene dek bekle: ikisi aynı klasöre
        yazmasın. Dönüş: False = bu arada durduruldu / kapanıyor."""
        key, t0 = str(work), time.time()
        while _prep_busy(key):
            if self._cancelled(item) or self._closing:
                return False
            if time.time() - t0 >= OLD_PREP_WAIT_S:
                log.warning("Önceki motorun indirmesi %.0f sn'dir bitmedi; yine de başlanıyor: %s", OLD_PREP_WAIT_S,
                            item.title[:70])
                break
            time.sleep(PREP_POLL_S)
        return True

    def _prepare(self, item: Item):
        cfg = self.cfg
        if self._cancelled(item):
            self._note_cancelled(item)
            self.emit("failed", item, error="Durduruldu.", cancelled=True)
            return
        if item.kind == "file" and not item.key:
            item.key = self.state.file_key(item.source, item.compute_key)
        item.compute_key()
        if not item.extra.get("live"):
            # sıra beklemesi işlenmiş / kopya denetiminden ÖNCE (inceleme: beklerken öbür kopya bitince bu kopya
            # "duplicate" sayılmıyor, video baştan işleniyordu). Daha önce işlenmiş olan beklemeden atlanır.
            p0 = self.state.get(item.key) if not cfg.force else None
            done_before = not cfg.force and self.state.is_done(item.key) \
                and not (p0 and p0["status"] == "done" and self._redo_reason(item, p0))
            if not done_before and not self._wait_prep_turn(item):
                if self._closing and not self._cancelled(item):
                    return  # motor kapanıyor: iş kalıcı kuyrukta kalır, yeni motor sürdürür (iptal değil)
                self._note_cancelled(item)
                self.emit("failed", item, error="Durduruldu.", cancelled=True)
                return
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
                if self.jobs[item.key].item.extra.get("uid") == item.extra.get("uid"):
                    return  # aynı iş ikinci kez geldi (yarış): sessizce bırak; "duplicate" çalışanın kaydını silerdi
                self.emit("skipped", item, status="duplicate", error="Bu video zaten sırada / işleniyor.")
                return
        job = Job(item, self.out_root / ".stt" / "is" / item.key[:18], status="prep")
        if not self._wait_old_prep(job.work, item):
            if not self._closing:
                self._note_cancelled(item)
                self.emit("failed", item, error="Durduruldu.", cancelled=True)
            return
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
            _prep_enter(str(job.work))
            try:
                job.wav, job.info = prepare_audio(item, job.work, cfg)
            finally:
                _prep_exit(str(job.work))
            if self._closing:
                return  # motor kapandı (hata / yeniden açılış): ses hazır, yeni motor kuyruktan sürdürür
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
                 if not _finished(job.work / "parcalar" / f"{i:04d}.json")]  # sürdürme: biten parçalar atlanır
        if self._opt(item, "diarize") and not _finished(job.work / "konusmaci.json"):
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
        mode = self._item_mode(item)
        lo, hi = self._speaker_bounds(item, mode)
        return {"id": f"{item.key}:d", "kind": "diarize", "wav": str(wav), "job": item.key,
                "min_speakers": lo, "max_speakers": hi, "vocals": self._want_vocals(item, mode)}

    def _want_vocals(self, item: Item, mode: str | None = None) -> bool:
        """Konuşmacı ayırmadan önce müzik ayıklansın mı: auto = podcast / film evet, basın toplantısı hayır."""
        v = str(self._opt(item, "diar_music") or "auto").strip().lower()
        if v in ("on", "acik", "açık", "1", "true", "evet"):
            return True
        if v in ("off", "kapali", "kapalı", "0", "false", "hayir", "hayır"):
            return False
        return (mode or self._item_mode(item)) == "herkes"

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
                self._safe_msg(msg)
                msg = pool.get(0.01)
        for step in (self._health, self._night_tick, self._recycle_tick, self._stats_tick, self._dispatch):
            try:
                step()
            except Exception:  # noqa: BLE001  (tek bir hata bütün motoru durdurmasın)
                log.exception("Motor adımı hata verdi (%s); devam ediliyor", step.__name__)
        busy = self.active()
        if busy != getattr(self, "_awake", False):  # yalnız değişince (pump hep aynı iş parçacığında döner)
            self._awake = busy
            keep_awake(busy)
            try:  # iş sürerken bilgisayar yeniden başlarsa bir sonraki açılışta sebebiyle söylenir (gpuguard)
                from .gpuguard import mark_busy
                mark_busy(self.out_root / ".stt", busy)
            except Exception:  # noqa: BLE001
                pass

    def _safe_msg(self, msg: dict) -> None:
        """İşçi mesajını işle; beklenmedik hata (disk dolu, bozuk veri...) yalnız o videoyu düşürür.
        Eskiden hata pump'tan çıkıyor, motor duruyor, kuyruktaki her şey bekliyordu."""
        try:
            self._on_msg(msg)
        except Exception as e:  # noqa: BLE001
            log.exception("İşçi mesajı işlenemedi (%s)", msg.get("type"))
            task = self.tasks.get(msg.get("tid") or "") or {}
            job = self.jobs.get(task.get("job", ""))
            if job and job.status == "run":
                job.status, job.error = "failed", f"İç hata: {type(e).__name__}: {e}"
                self.emit("failed", job.item, error=job.error)
                self.state.set(job.item.key, source=job.item.source, status="failed", title=job.item.title,
                               error=job.error)
                with self.lock:
                    self.results.append(job)

    def _drop_dead_diar(self) -> None:
        """Konuşmacı ayırma süreci bir daha açılmayacaksa (üst üste çöktü / açılamadı) bekleyen ayırma görevleri
        hata sayılır: videolar konuşmacısız biter. Yoksa iş sonsuza dek bitmez ve (aynı kartta sırayla çalışma
        yüzünden) öbür videoların metin çevirisi de dururdu."""
        if self.dpool is None or "diar" not in self.dpool.workers or "diar" in self.dpool.alive():
            return
        if "diar" not in self.fatal and self.restarts.get("diar", 0) < self.cfg.worker_restarts:
            return
        with self.lock:
            waiting = list(self.dbacklog) + [t for _, t in self.ddelayed]
            self.dbacklog.clear()
            self.ddelayed.clear()
        self.dqueued_n = 0
        for t in waiting:
            job = self.jobs.get(t["job"])
            if not job or job.status != "run":
                continue
            job.attempts[t["id"]] = 9  # tekrar kuyruğa girmesin
            self._on_msg({"type": "error", "wid": "diar", "tid": t["id"],
                          "error": "Konuşmacı ayırma süreci açılamadı (üst üste durdu); ayrıntı günlükte"})
        self.idle.discard("diar")

    def _shared_gpu(self) -> set:
        """Konuşmacı ayırmayla AYNI kartta çalışan metin işçileri. Bu kartta ikisi aynı anda çalışmaz
        (aynı anda çalışınca metin işçisi 'invalid device ordinal' ile hep düşüyordu): sırayla."""
        if not self.pool or getattr(self.pool, "threads", False):
            return set()
        if any(j.live_active for j in self.jobs.values()):
            return set()  # canlı yayın sürüyor: metin gecikmesin (ayırma kısa, aynı anda çalışabilir)
        ddev = (getattr(self.dpool, "devices", None) or [None])[0] if self.dpool is not None else None
        if ddev is None and self.dbacklog:
            ddev = self._diar_device()
        if not ddev or not ddev.startswith("cuda"):
            return set()
        spec = getattr(self.pool, "_spec", {}) or {}
        return {n for n, d in spec.items() if d == ddev}

    def _diar_mem_ok(self) -> bool:
        """Sıradaki konuşmacı ayırmaya RAM yetiyor mu (ana süreçte, görev verilmeden ÖNCE; 5 sn'de bir ölçülür).
        Eskiden yalnız ayırma işçisi içinde bekleniyordu (en çok 10 dk); görev başlamış sayıldığı için aynı karttaki
        metin çevirisi de o süre boyunca duruyordu (3090 boşta). Artık RAM yetmezse ayırma sırada kalır, metin sürer.
        Ölçüt işçidekiyle aynı (``Diarizer.wait_memory``); ``DIAR_MEM_MAX_WAIT_S`` sonra yine de verilir."""
        now = time.time()
        if now - self._dmem_at < 5.0:
            return self._dmem_ok
        self._dmem_at = now
        t = next((t for t in self.dbacklog if (self.jobs.get(t["job"]) is not None
                                              and self.jobs[t["job"]].status == "run")), None)
        ok = True
        if t is not None:
            from .util import memory_status
            m = memory_status()
            if m:
                job = self.jobs[t["job"]]
                need = 2.5 + 1.5 * float(job.info.get("duration_s") or 3600) / 3600
                ram = m["ram_free"] * max(10, min(100, int(self.cfg.ram_limit_pct or 70))) / 100
                waited = now - (self._dmem_since or now)
                ok = min(ram, m["commit_free"]) >= need or (ram >= need and m["commit_free"] >= 1.0 and waited >= 60)
                if not ok and waited >= DIAR_MEM_MAX_WAIT_S:
                    if not self._dmem_forced:
                        log.warning("Konuşmacı ayırma için bellek %d dk'dır yetmiyor; yine de deneniyor", waited // 60)
                    self._dmem_forced = True  # görev verilene dek açık kalır (kart o an meşgulse süre sıfırlanmasın)
                    self._dmem_ok = True
                    return True
                elif not ok and self._dmem_since is None:
                    log.warning("Konuşmacı ayırma RAM bekliyor (~%.1f GB gerekli; boş RAM %.1f GB, sanal bellek %.1f GB): "
                                "bu sırada ekran kartı metin çevirmeye devam ediyor", need, m["ram_free"], m["commit_free"])
        if ok:
            self._dmem_since = None
            self._dmem_forced = False
        elif self._dmem_since is None:
            self._dmem_since = now
        self._dmem_ok = ok
        return ok

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
            night = bool(getattr(self.cfg, "night_mode", False))
            if night:
                self._night_single()
            hold = self._night_on() and self._night_hold()  # gece modu: görev arası mola / soğuma molası
            self._flex_tick()
            flex = self._flex_level()
            if flex >= 2 and time.time() < self._flex_rest_until:
                hold = True  # esnek kademe 2: ana kart görev arası dinleniyor
            main = set(self.pool.workers) if self.pool else set()
            shared = self._shared_gpu()
            dmem = self._diar_mem_ok() if self.dbacklog else True
            if shared and ("diar" in self.inflight or self.dqueued_n > 0 or (self._diar_turn() and dmem)):
                main -= shared  # kart konuşmacı ayırmada (ya da sırası geldi): metin görevi verilmez
            if hold:
                main = set()
            while len(self.idle & main) - self.queued_n > 0 and self.backlog:
                t = self.backlog.popleft()
                job = self.jobs.get(t["job"])
                if not job or job.status != "run":
                    continue  # iş iptal/başarısız: görevi at
                self.queued_n += 1
                self.pool.put(t)
            # bilgisayar kullanılıyor: ikinci karta yeni iş yok. Ana kartta çalışan metin işçisi yoksa (çöktü / açılamadı)
            # tutulmaz: yoksa kuyruk hiç ilerlemezdi
            flex_hold = flex >= 1 and self._main_takers() > 0
            if self.hpool is not None and not night and flex_hold:
                self._flex_hold_helpers()
            elif self.hpool is not None and not night:  # gece modunda yardımcı (monitörlü) karta iş yok
                self._helper_dispatch(len(main - self.fatal))
            if self.ddelayed:
                now = time.time()
                for due, t in list(self.ddelayed):
                    busy = any(b["job"] == t["job"] for b in self.backlog)  # önce metin çevirisi bitsin
                    if now >= due and (not busy or now >= due + 900):
                        self.ddelayed.remove((due, t))
                        self.dbacklog.append(t)
            if self.dbacklog and self.dpool is None:
                self._start_diar_pool()
            didle = 1 if "diar" in self.idle and dmem else 0  # RAM yetmezse ayırma sırada (metin sürer)
            if shared and (self.queued_n > 0 or any(w in self.inflight for w in shared)
                           or (self.backlog and not self._diar_turn())):
                didle = 0  # kartta metin çevirisi sürüyor: ayırma bekler
            if hold:
                didle = 0
            while didle - self.dqueued_n > 0 and self.dbacklog:
                t = self.dbacklog.popleft()
                job = self.jobs.get(t["job"])
                if not job or job.status != "run":
                    continue
                self.dqueued_n += 1
                self.dpool.put(t)
                self._dmem_since, self._dmem_forced, self._dmem_at = None, False, 0.0  # sonraki görev için yeniden
            if self.hpool is not None and not night and not (flex >= 1 and "diar" not in self.fatal):
                self._helper_diar_dispatch()

    # ---------------------------------------------------------- esnek tam güç (stt/flex.py)
    def _flex_on(self) -> bool:
        """Esnek yavaşlama geçerli mi: ayarda açık, gece modu kapalı (onun kendi molası var), ekran kartında
        çalışılıyor, iş var, canlı yayın yok (canlı metin gecikmesin)."""
        if not getattr(self.cfg, "gpu_flex", True) or getattr(self.cfg, "night_mode", False):
            return False
        if self.pool is None or getattr(self.pool, "threads", False):
            return False
        if not any(str(d).startswith("cuda") for d in self.devices):
            return False
        jobs = list(self.jobs.values())
        return any(j.status in ("prep", "run") for j in jobs) and not any(j.live_active for j in jobs)

    def _flex_level(self) -> int:
        return self.flex.level if self._flex_on() else 0

    def _mem_watch_on(self) -> bool:
        """Bellek koruması geçerli mi: ekran kartında çalışılıyor ve iş var. Esnek kapalı / gece modu / canlı yayında da
        (ayar ne olursa olsun bilgisayarın belleği dolmasın; yalnız indirme durdurulur, kart düzeni değişmez)."""
        if self.pool is None or getattr(self.pool, "threads", False):
            return False
        if not any(str(d).startswith("cuda") for d in self.devices):
            return False
        return any(j.status in ("prep", "run") for j in list(self.jobs.values()))

    def _mem_crit(self) -> bool:
        return bool(self.flex.mem_crit) and self._mem_watch_on()

    def _sysmem_short(self, wid: str | None = None) -> str:
        """İkinci karta (``wid``) model yüklemeye bilgisayarın belleği yetmiyorsa sebep, yetiyorsa "" (ölçülemezse de "").
        Kartın kendi payı hesaba katılır: açıldıktan sonra da boş bellek "az" eşiğinin üstünde kalmalı (yoksa aç-kapa)."""
        now = time.time()
        if now < self._h_cool_until:
            return (f"bilgisayarın belleği azaldığı için ikinci kart {max(1, round((self._h_cool_until - now) / 60))} "
                    "dk bekliyor")
        try:
            m = self._memstat()
        except Exception:  # noqa: BLE001
            m = None
        if not m:
            return ""
        foot = self._h_memfoot.get(wid or "") or HELPER_FOOT_DEFAULT
        need_ram, need_commit = RAM_OK_GB + foot["ram"], COMMIT_OK_GB + foot["commit"]
        if m["ram_free"] < need_ram or m["commit_free"] < need_commit:
            return (f"bilgisayarın belleği az (boş RAM {m['ram_free']:.1f} / {need_ram:.1f} GB, boş sanal bellek "
                    f"{m['commit_free']:.1f} / {need_commit:.1f} GB)")
        return ""

    def _foot_open(self, wid: str) -> None:
        """İkinci kart açılıyor: kendi bellek payını ölçmek için açılıştaki boş bellek."""
        try:
            m = self._memstat()
        except Exception:  # noqa: BLE001
            m = None
        if m:
            self._h_memopen[wid] = (time.time(), m["ram_free"], m["commit_free"], {"ram": 0.0, "commit": 0.0})

    def _foot_sample(self, m: dict | None, now: float) -> None:
        """Esnek ölçümünden: açık ikinci kartın payı = açılıştaki boş bellek - şimdiki (ilk HELPER_FOOT_LEARN_S içindeki
        en büyük düşüş). Kart kapanınca ya da süre dolunca öğrenilir (başka programların artışı da girer: temkinli)."""
        for wid, (t0, ram0, commit0, peak) in list(self._h_memopen.items()):
            closed = wid in self._h_parked or wid in self.fatal
            if m and not closed:
                peak["ram"] = max(peak["ram"], ram0 - m["ram_free"])
                peak["commit"] = max(peak["commit"], commit0 - m["commit_free"])
            if closed or now - t0 >= HELPER_FOOT_LEARN_S:
                del self._h_memopen[wid]
                if peak["commit"] >= 0.5 or peak["ram"] >= 0.5:
                    old = self._h_memfoot.get(wid) or HELPER_FOOT_DEFAULT
                    new = {k: round(min(10.0, max(0.5, peak[k] + 0.3)), 1) for k in ("ram", "commit")}
                    if closed:  # bellek yüzünden kapandıysa küçük ölçüm eskisini düşürmesin
                        new = {k: max(new[k], old[k]) for k in new}
                    self._h_memfoot[wid] = new
                    log.info("[esnek] %s ikinci kart işçisi bellekte ~%.1f GB RAM, ~%.1f GB sanal bellek kaplıyor: "
                             "sonraki açılış buna göre", wid, new["ram"], new["commit"])

    def _mem_cooldown(self, now: float) -> None:
        """İkinci kart bellek yüzünden kapatıldı: bir süre açılmasın (tekrarında süre iki katı, en çok saat)."""
        self._h_cool_s = min(HELPER_MEM_COOL_MAX_S, self._h_cool_s * 2) \
            if now - self._h_cool_last < HELPER_MEM_COOL_MAX_S and self._h_cool_s else HELPER_MEM_COOL_S
        self._h_cool_last = now
        self._h_cool_until = now + self._h_cool_s
        log.info("[esnek] bellek az: ikinci kart %.0f dk açılmayacak", self._h_cool_s / 60)

    def _mem_notice(self, m: dict | None) -> None:
        """Bellek azaldı: arayüze en çok saatte bir bildirim (sanal bellek sınırı küçükse kalıcı çözümle)."""
        now = time.time()
        if now - self._mem_noticed < MEM_NOTICE_S:
            return
        self._mem_noticed = now
        msg = ("Bilgisayarın belleği azaldı (" + self.flex._what() + "): STT ikinci kartı bekletiyor"
               + (" ve yeni video indirmiyor" if self.flex.mem_crit else "")
               + "; bellek açılınca kendiliğinden tam güce döner. Tarayıcıda çok sekme / oyun varsa kapatmak yardımcı olur.")
        if m and m.get("commit_total") is not None and m["commit_total"] < PAGEFILE_DUAL_GB:
            msg += f" Sanal bellek sınırı {m['commit_total']:.0f} GB (küçük); kalıcı çözüm: " + PAGEFILE_DUAL_HINT
        log.warning("[esnek] %s", msg)
        self.emit("notice", text=msg)

    def _pagefile_check(self) -> None:
        """İki kart birlikte çalışacak ve sanal bellek sınırı küçük: açılışta bir kez kalıcı çözümü söyle."""
        try:
            m = self._memstat()
        except Exception:  # noqa: BLE001
            m = None
        if not m or m.get("commit_total") is None or m["commit_total"] >= PAGEFILE_DUAL_GB \
                or str(self.out_root) in _PAGEFILE_WARNED:
            return
        _PAGEFILE_WARNED.add(str(self.out_root))  # motor yeniden açılınca tekrar söylenmez
        msg = (f"Sanal bellek sınırı {m['commit_total']:.0f} GB: iki ekran kartıyla uzun kuyrukta yetmeyebilir (dolunca "
               "Windows programları kapatır; STT bellek azalınca ikinci kartı bekletir, işler yavaşlar). "
               + PAGEFILE_DUAL_HINT)
        log.warning("%s", msg)
        self.emit("notice", text=msg)

    def _flex_tick(self) -> None:
        """``FLEX_SAMPLE_S``'de bir işlemci / RAM / sanal bellek ölçümü (arka planda: snapshot ve süreç süreleri motor
        döngüsünü bekletmesin). Bellek koruması esnek kapalıyken de ölçer (``_mem_watch_on``); işlemciye göre yavaşlama
        yalnız esnek açıkken. ``self.lock`` altında çağrılır."""
        now = time.time()
        on = self._mem_watch_on()
        if on != self._flex_was:
            self._flex_was = on
            if not on:  # iş bitti / gece modu / kapatıldı: sonraki kuyruk tam güçle başlasın, eski ölçüm kullanılmasın
                self.flex = Flex()
                self._flex_rest_until = 0.0
                if self._flex_sampler is not None:
                    self._flex_sampler.reset()
                for wid in self._helper_wids():
                    if self._h_note.get(wid) == FLEX_NOTE:
                        self._helper_note(wid, "")
            self._night_emit()
        if self._flex_probing or now - self._flex_at < FLEX_SAMPLE_S or not on:
            return
        self._flex_at = now
        self._flex_probing = True
        if self._flex_sampler is None:
            from .flex import CpuSampler
            self._flex_sampler = CpuSampler()

        def run():
            try:
                cpu = self._flex_sampler.sample()
                m = self._memstat()
                idle = self._idle()
                if idle is not None and idle >= AWAY_S and self._media():
                    idle = 0.0  # dokunmadan izliyor (tam ekran / video): bilgisayar başında biri var
                with self.lock:
                    old, was_low = self.flex.level, self.flex.ram_low
                    others = cpu[0] if cpu and self._flex_on() else None  # esnek kapalı: işlemci yavaşlatmaz
                    msg = self.flex.update(others, m["ram_free"] if m else None, time.time(),
                                           m["commit_free"] if m else None, idle)
                    if msg:
                        log.info("[esnek] %s", msg)
                    if self.flex.ram_low and not was_low:
                        self._mem_notice(m)
                    self._stat_sample(m, self.flex.level, time.time())
                    self._foot_sample(m, time.time())
                    if self.flex.level != old:
                        if self.flex.level < 2:
                            self._flex_rest_until = 0.0
                        if self.flex.level < 1:
                            for wid in self._helper_wids():
                                if self._h_note.get(wid) == FLEX_NOTE:
                                    self._helper_note(wid, "")  # yeniden iş alınca / kapalıysa kendi notu
                        self._night_emit()
            except Exception:  # noqa: BLE001
                log.debug("esnek ölçüm", exc_info=True)
            finally:
                self._flex_probing = False
        try:
            self._spawn(run)
        except Exception:  # noqa: BLE001  (iş parçacığı açılamadı: ölçüm takılı kalmasın)
            self._flex_probing = False
            log.debug("esnek ölçüm başlatılamadı", exc_info=True)

    def _helper_wids(self) -> list[str]:
        return (list(self.hpool.workers) if self.hpool is not None else []) + \
            ([HELPER_DIAR] if self.hpool is not None else [])

    def _main_takers(self) -> int:
        """Görev alabilecek ana metin işçisi sayısı (kalıcı düşen / kapatılan sayılmaz)."""
        return len(set(self.pool.workers) - self.fatal - self._drop_wids) if self.pool else 0

    def _flex_hold_helpers(self) -> None:
        """Kademe ≥ 1: ikinci karta yeni görev verilmez (sürmekte olan biter). Model bellekte kalır, bilgisayar
        sakinleşince hemen sürer (açıp kapamak işlemciyi daha çok yorardı); ama boşta uzun kalırsa kapatılır: RAM azken
        ``HELPER_SWITCH_S``, yoksa ``HELPER_IDLE_PARK_S`` sonra (bellek başka programlara kalsın; yoksa RAM yüzünden
        başlayan yavaşlama, ikinci kartın tuttuğu RAM yüzünden hiç bitmezdi). Kapatılmış (sürücü çökmesi / güç
        koruması) yardımcı önce kapatılır. ``self.lock`` altında."""
        if self.helper_disabled:
            self._helper_stop_all()
            return
        now = time.time()
        wait = HELPER_SWITCH_S if self.flex.ram_low else HELPER_IDLE_PARK_S
        for wid in self._helper_wids():
            if wid in self._h_parked or wid in self.inflight:
                continue
            queued = self.hqueued_n if wid != HELPER_DIAR else self.hdqueued_n
            if wid in self.idle and not queued and now - self._h_last.get(wid, now) >= wait:
                log.info("[%s] esnek: ikinci kart %d sn'dir bekliyor, kapatıldı (bellek boşalsın; gerekince açılır)",
                         wid, int(now - self._h_last.get(wid, now)))
                self._park_helper(wid, HELPER_OFF)
                if self.flex.ram_low:  # bellek yüzünden: hemen yeniden açılıp aynı döngüye girmesin
                    self._mem_cooldown(now)
                continue
            self._helper_note(wid, FLEX_NOTE)

    def flex_state(self) -> dict:
        on = bool(getattr(self.cfg, "gpu_flex", True))
        if not on:
            note = "kapalı"
        elif getattr(self.cfg, "night_mode", False):
            note = "gece modu açık (kendi molası var)"
        elif not self._flex_on():
            note = "açık · iş bekleniyor"
        else:
            note = self.flex.note()
        return {"on": on, "level": self._flex_level(), "note": note, "card2": self._flex_card}

    def _flex_spare(self) -> bool:
        """Esnek ikinci kart iş alabilir durumda mı (ana kart düşerse metni o sürdürür)."""
        hp = self.hpool
        return bool(self._flex_card and hp is not None and not self.helper_disabled and any(
            w not in self.fatal and w not in self._h_gone for w in hp.workers))

    def _reserve(self) -> float:
        """İkinci kartta başka programlara bırakılan bellek. Monitörsüz esnek kartta en çok ``FLEX_RESERVE_GB``
        (tarayıcı ekranı Intel'le çizer); monitörlü yardımcıda ayardaki değer (YouTube / sekmeler)."""
        r = float(self.cfg.helper_reserve_gb)
        return min(r, FLEX_RESERVE_GB) if self._flex_card else r

    # ---------------------------------------------------------- gece modu (stt/governor.py)
    def _night_on(self) -> bool:
        """Gece modu molaları şu an geçerli mi: ayarda açık, ekran kartında çalışılıyor, canlı yayın yok (canlı metin
        gecikmesin)."""
        if not getattr(self.cfg, "night_mode", False) or self.pool is None or getattr(self.pool, "threads", False):
            return False
        if not any(str(d).startswith("cuda") for d in self.devices):
            return False
        return not any(j.live_active for j in list(self.jobs.values()))

    def _night_hold(self) -> bool:
        return self.gov.holding() or time.time() < self._rest_until

    def _night_card(self) -> int | None:
        dev = self._diar_device()
        return int(dev.split(":")[1]) if dev.startswith("cuda:") else None

    @staticmethod
    def _card_query() -> dict:
        from .power import query
        return query()

    def _night_single(self) -> None:
        """Gece modu: yalnız bir kart (belleği en büyük monitörsüz kart) çalışır. Yardımcı (monitörlü) kart işçileri
        boştaysa kapatılır (görevdeki bitirince); "Hepsi" ile açılmış öbür ana kart işçileri ilk görevlerinde kapatılır
        (ortak kuyruktan boştaki işçi güvenle kapatılamaz; görev bu karta döner). ``self.lock`` altında."""
        if self.hpool is not None:
            for wid in list(self.hpool.workers) + [HELPER_DIAR]:
                pool = self.hdpool if wid == HELPER_DIAR else self.hpool
                if wid in self._h_parked or wid in self.fatal or wid in self._h_gone:
                    if wid in self._h_parked and self._h_note.get(wid) != NIGHT_NOTE:
                        self._helper_note(wid, NIGHT_NOTE)
                    continue
                if pool is None:
                    continue
                queued = self.hqueued_n if pool is self.hpool else self.hdqueued_n
                if wid in self.idle and wid not in self.inflight and not queued:
                    log.info("[%s] gece modu: yardımcı kart kapatıldı (yalnız bir kart çalışır)", wid)
                    self._park_helper(wid, "gece modu")
                    self._helper_note(wid, NIGHT_NOTE)
        spec = getattr(self.pool, "_spec", {}) or {}
        card = self._diar_device()
        if not card.startswith("cuda:") or not any(str(d) == card and w not in self.fatal for w, d in spec.items()):
            return
        others = {w for w, d in spec.items() if str(d).startswith("cuda:") and str(d) != card and w not in self.fatal
                  and w not in self._drop_wids}
        if others:
            log.info("Gece modu: yalnız %s çalışır; öbür kart(lar) ilk görevlerinde kapatılıyor (%s)", card,
                     ", ".join(sorted(others)))
            self._drop_wids |= others
            self._night_drops |= others
            self._drop_msg = "gece modu: tek kart çalışır, bu kart kapatıldı (uygulama yeniden açılınca döner)"
            for w in list(others):
                if w in self.inflight:
                    self._drop_worker(w)

    def _night_tick(self) -> None:
        """Gece modu adımı (pump): açılıp kapanmayı izler, ısıyı ``SAMPLE_S``'de bir ölçer (ayrı iş parçacığında),
        uzun süredir açık işçileri boştayken yeniler."""
        on = bool(getattr(self.cfg, "night_mode", False))
        self.gov.set_target(getattr(self.cfg, "night_temp", 72))
        if on != self._night_was:
            first = self._night_was is None
            self._night_was = on
            if on:
                log.info("Gece modu açık: tek kart, her görevden sonra ısıya göre mola (hedef %d°C, %d°C'de soğuma "
                         "molası), işçiler %s; önden en çok %d video hazırlanır", self.gov.target, self.gov.hot,
                         f"{self.cfg.night_recycle_min:.0f} dk'da bir yenilenir" if self.cfg.night_recycle_min
                         else "yenilenmez", NIGHT_PREP_AHEAD)
            else:
                with self.lock:
                    self._rest_until = 0.0
                    self.gov = Governor(getattr(self.cfg, "night_temp", 72))
                    self._drop_wids -= self._night_drops  # henüz kapatılmamış öbür kart işçileri çalışmaya devam eder
                    self._night_drops = set()
                    for wid, note in list(self._h_note.items()):
                        if note == NIGHT_NOTE:
                            self._helper_note(wid, HELPER_OFF)
                if not first:
                    log.info("Gece modu kapandı: normal hız")
            if on or not first:
                self._night_emit()
        if not on:
            return
        now = time.time()
        active = self.active()
        if active != self._night_active:  # kuyruk bitti / başladı: kutu eski ısıyı göstermesin
            self._night_active = active
            self._night_emit()
        if not active:
            return
        if self._night_on():
            self._night_recycle(now)
        if not self._gov_probing and now - self._gov_at >= SAMPLE_S:
            self._gov_probing = True
            self._gov_at = now
            try:
                self._spawn(self._night_probe)
            except Exception:  # noqa: BLE001  (iş parçacığı açılamadı: sonraki turda yeniden)
                self._gov_probing = False
                log.debug("ısı ölçümü başlatılamadı", exc_info=True)

    def _night_probe(self) -> None:
        """Çalışan kartın ısısını oku (nvidia-smi; ana süreçte CUDA açılmaz), mola süresini ayarla."""
        try:
            idx = self._night_card()
            temp = None
            if idx is not None:
                try:
                    temp = (self._card_query().get(idx) or {}).get("temp")
                except Exception:  # noqa: BLE001
                    log.debug("kart ısısı okunamadı", exc_info=True)
            with self.lock:
                msg = self.gov.update(temp, time.time())
            if msg:
                log.info("[gece] %s", msg)
            self._night_emit()
        finally:
            self._gov_probing = False

    def _night_recycle(self, now: float) -> None:
        self._recycle(now, float(getattr(self.cfg, "night_recycle_min", 0) or 0))

    def _recycle_tick(self) -> None:
        """Gece modu dışında uzun kuyruk (1.9.27): işçiler ``recycle_min`` dakikada bir boştayken yenilenir (gece
        modunda ``_night_tick`` kendi aralığıyla yeniler)."""
        if getattr(self.cfg, "night_mode", False) or not self.active():
            return
        self._recycle(time.time(), float(getattr(self.cfg, "recycle_min", 0) or 0))

    def _recycle(self, now: float, mins: float) -> None:
        """``mins`` dakikadır açık işçi, boşta ve kuyruğunda görev yokken yeniden açılır (saatlerce süren süreçte bellek
        parçalanması / sızıntı birikmesin; model ~30 sn'de yeniden yüklenir). Yalnız tek işçili havuzda: boştaki işçi
        ortak kuyruktan yalnız orada kapanma işaretiyle güvenle kapatılır. İkinci kart (yardımcı / esnek) işçisi boştaysa
        kapatılır; iş birikince yeniden açılır."""
        if mins <= 0:
            return
        if self.flex.ram_low and self._mem_watch_on():
            return  # bellek az: yenileme model yüklemesiyle bir an daha çok bellek ister; bellek açılınca
        for pool in (self.pool, self.dpool):
            if pool is None or getattr(pool, "threads", False) or len(pool.workers) != 1:
                continue
            wid = next(iter(pool.workers))
            born = self._born.get(wid)
            queued = self.dqueued_n if pool is self.dpool else self.queued_n
            if born is None or now - born < mins * 60 or wid in self.fatal or wid in self._drop_wids:
                continue
            if wid not in self.idle or wid in self.inflight or queued > 0 or wid not in pool.alive():
                continue
            log.info("[yenileme] %s işçisi %.0f dakikadır açık: bellek birikmesin diye yenileniyor (model yeniden "
                     "yüklenir)", wid, (now - born) / 60)
            self._born.pop(wid, None)
            pool.respawn(wid, idle=True)
            self.idle.discard(wid)
            if pool is self.dpool:
                self.dqueued_n = 0
            self.emit("worker", wid=wid, state="loading")
        with self.lock:
            for wid in self._helper_wids():
                born = self._born.get(wid)
                if born is None or now - born < mins * 60 or wid in self._h_parked or wid in self.fatal \
                        or wid in self._h_gone:
                    continue
                queued = self.hdqueued_n if wid == HELPER_DIAR else self.hqueued_n
                if wid not in self.idle or wid in self.inflight or queued:
                    continue
                log.info("[yenileme] %s ikinci kart işçisi %.0f dakikadır açık: bellek birikmesin diye kapatıldı "
                         "(iş birikince yeniden açılır)", wid, (now - born) / 60)
                self._born.pop(wid, None)
                self._park_helper(wid, HELPER_OFF)

    # ---------------------------------------------------------- verim (1.9.27)
    def stats(self, window: float = 3600.0) -> dict:
        """Son ``window`` sn'de biten / hatalı iş ve işlenen ses (dk); açılıştan beri toplam (yerel API, "[verim]")."""
        now = time.time()
        done = [(t, a) for t, a in list(self._stat_done) if now - t <= window]
        failed = [t for t in list(self._stat_failed) if now - t <= window]
        return {"done_1h": len(done), "failed_1h": len(failed), "audio_min_1h": round(sum(a for _, a in done) / 60, 1),
                "done_total": self._stat_n["done"], "failed_total": self._stat_n["failed"]}

    def _stat_sample(self, m: dict | None, level: int, now: float) -> None:
        """Esnek ölçümünden: saatin en düşük boş belleği ve kademelerde geçen süre."""
        h = self._stat_hour
        if m:
            h["ram_min"] = m["ram_free"] if h["ram_min"] is None else min(h["ram_min"], m["ram_free"])
            h["commit_min"] = m["commit_free"] if h["commit_min"] is None else min(h["commit_min"], m["commit_free"])
        if h["last"] is not None:
            h["level_s"][max(0, min(2, level))] += max(0.0, now - h["last"])
        h["last"] = now

    def _stats_tick(self) -> None:
        """Saatte bir günlüğe "[verim]" satırı (o saatte iş olduysa): bir sonraki oturumda hız / bellek / esnek kademe
        ölçümle ayarlansın. Arayüz modunda biten işler bellekten temizlenir."""
        now = time.time()
        if not self.keep_results and now - self._prune_at >= PRUNE_S:
            self._prune_at = now
            self._prune()
        h = self._stat_hour
        span = now - h["t0"]
        if span < STATS_S:
            return
        st = self.stats(span)
        if st["done_1h"] or st["failed_1h"] or self.active():
            with self.lock:
                waiting = self._pending_futs + sum(1 for j in self.jobs.values() if j.status in ("run", "finalize"))
            mem = ""
            if h["ram_min"] is not None:
                mem = f"; en düşük boş RAM {h['ram_min']:.1f} GB, sanal bellek {h['commit_min']:.1f} GB"
            lv = h["level_s"]
            log.info("[verim] son %.0f dk: %d video bitti (%.0f dk ses), %d hata; kuyrukta %d iş%s; esnek kademe süresi "
                     "0/1/2: %.0f/%.0f/%.0f dk", span / 60, st["done_1h"], st["audio_min_1h"], st["failed_1h"], waiting,
                     mem, lv[0] / 60, lv[1] / 60, lv[2] / 60)
        self._stat_hour = {"t0": now, "ram_min": None, "commit_min": None, "level_s": [0.0, 0.0, 0.0], "last": None}

    def night_state(self) -> dict:
        """Arayüzdeki gece modu kutusu: {on, note, temp, rest, paused, card}."""
        on = bool(getattr(self.cfg, "night_mode", False))
        idx = self._night_card() if self.devices else None
        card = (self._gi.get(idx) or {}).get("name") if idx is not None else None
        if not on:
            note = ""
        elif not any(str(d).startswith("cuda") for d in self.devices):
            note = "ekran kartı yok: mola uygulanmıyor"
        elif any(j.live_active for j in list(self.jobs.values())):
            note = "canlı yayın sürüyor: mola yok"
        elif not self.active():
            note = "açık · iş bekleniyor"
        else:
            note = self.gov.note()
        return {"on": on, "note": note, "temp": self.gov.temp, "rest": round(self.gov.rest),
                "paused": bool(on and self.gov.holding() and self._night_on() and self.active()), "card": card,
                "flex": self.flex_state()}

    def _night_emit(self) -> None:
        self.emit("night", **self.night_state())

    # ---------------------------------------------------------- yardımcı kart ("yardim")
    def _spawn(self, fn) -> None:
        threading.Thread(target=fn, name="kart-olcum", daemon=True).start()

    @staticmethod
    def _gpu_load(idx: int):
        from .engines import gpu_load
        return gpu_load(idx)

    def _helper_index(self, wid: str) -> int | None:
        if wid == HELPER_DIAR:
            wid = self.helper_devices[0] if self.helper_devices else wid
        dev = (getattr(self.hpool, "_spec", None) or {}).get(wid) or wid
        try:
            return int(str(dev).split(":")[1])
        except (IndexError, ValueError):
            return None

    def _probe_helper(self, wid: str) -> None:
        """Yardımcı kartın yükünü arka planda ölç (nvidia-smi yavaş olabilir; motor döngüsü beklemesin)."""
        if wid in self._h_probing:
            return
        self._h_probing.add(wid)
        idx = self._helper_index(wid)

        def run():
            load = None
            try:
                load = self._gpu_load(idx) if idx is not None else None
            except Exception:  # noqa: BLE001
                log.debug("kart yükü okunamadı", exc_info=True)
            finally:
                self._h_probe[wid] = (time.time(), load)
                self._h_probing.discard(wid)
        self._spawn(run)

    def _helper_check(self, wid: str, load: dict | None, parked: bool) -> tuple[str, int]:
        """(bekleme sebebi, açılışta toplu boyut). Kartta ``helper_reserve_gb`` YouTube / tarayıcı için boş kalır.
        Kapalı yardımcı: boş bellekten ayrılan çıkınca kalan yere sığan en büyük toplu boyutla açılır (yoksa bekler).
        Açık yardımcı: başka programlar ayrılan belleğin yarısından fazlasını kullandıysa bekler (uzarsa kapanır).
        Ölçülemezse (eski sürücü) çalışılır; açılışta küçük toplu boyut."""
        if parked and self._h_shrunk_at.get(wid) and time.time() - self._h_shrunk_at[wid] >= HELPER_REGROW_S:
            # küçültme kalıcı olmasın (sebep çoğu zaman geçici: oyun, ağır video): varsayılana dön
            self._h_shrunk_at.pop(wid, None)
            self._h_foot.pop(wid, None)
            if wid in self._h_cap_max:
                self._h_cap[wid] = self._h_cap_max[wid]
        cap = self._h_cap.get(wid) or 4
        foot = self._h_foot.get(wid) or HELPER_FOOT_GB
        sizes = [b for b in sorted(HELPER_FOOT_GB, reverse=True) if b <= cap] or [min(HELPER_FOOT_GB)]
        if parked:
            # kartta yer olsa da (ya da kart okunamasa da: nvidia-smi bellek baskısında zaman aşımına uğrayabilir)
            # bilgisayarın belleği yetmiyorsa açılmaz (1.9.27)
            short = self._sysmem_short(wid)
            if short:
                return short, 0
        used, total = (load or {}).get("used_mb"), (load or {}).get("total_mb")
        if not (isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0):
            return "", sizes[-1]
        free = (total - used) / 1024
        reserve = self._reserve()
        if parked:
            for b in sizes:
                if free - reserve >= foot[b]:
                    return "", b
            return f"kartta yer yok ({free:.1f} GB boş, {reserve:.1f} GB ayrılmış)", 0
        if free < reserve / 2:
            return f"başka programlar belleği kullanıyor ({free:.1f} GB boş)", 0
        self._helper_measure(wid, load)
        return "", self._h_batch.get(wid, 0)

    def _helper_shrink(self, wid: str) -> None:
        """Bellek yetmedi: bir sonraki açılış bir küçük toplu boyutla (aynı boyutla açılıp kapanma döngüsü olmasın).
        ``HELPER_REGROW_S`` sonra geri büyür."""
        cur = self._h_batch.get(wid) or self._h_cap.get(wid) or 4
        smaller = [b for b in HELPER_FOOT_GB if b < cur]
        self._h_cap[wid] = max(smaller) if smaller else min(HELPER_FOOT_GB)
        self._h_shrunk_at[wid] = time.time()

    def _helper_measure(self, wid: str, load: dict | None) -> None:
        """Yardımcının kendi bellek payını ölç: açılıştaki boş bellek − ilk parçadan sonraki boş bellek (açılışa yakın:
        başka programların değişimi az karışır). Tahmini 0.5 GB'tan fazla aşıyorsa o boyutun belleği öğrenilir; sonraki
        açılış buna göre yer ister (yoksa aç → bekle → kapan döngüsü olurdu). Kapanma anındaki ölçüm KULLANILMAZ: o an
        boşalan bellek çoğunlukla başka programındır (oyun) ve toplu boyut gereksiz küçülürdü."""
        if wid in self._h_own or not self._h_done.get(wid):
            return
        b = self._h_batch.get(wid)
        open_free = self._h_open_free.get(wid)
        used, total = (load or {}).get("used_mb"), (load or {}).get("total_mb")
        if not b or open_free is None or not (isinstance(used, (int, float)) and isinstance(total, (int, float))):
            return
        own = open_free - (total - used) / 1024
        self._h_own[wid] = own
        foot = self._h_foot.get(wid) or HELPER_FOOT_GB
        if own > foot.get(b, 0.0) + 0.5:
            self._h_foot[wid] = {**foot, b: round(own + 0.2, 2)}
            self._h_shrunk_at.setdefault(wid, time.time())  # öğrenilen değer de bir süre sonra varsayılana döner
            log.info("[%s] yardımcı toplu boyut %d ile ~%.1f GB kaplıyor (tahmin %.1f): sonraki açılış buna göre", wid,
                     b, own, HELPER_FOOT_GB.get(b, 0.0))

    def _helper_note(self, wid: str, note: str) -> None:
        if self._h_note.get(wid, "") == note:
            return
        self._h_note[wid] = note
        if note:
            log.info("[%s] yardımcı kart bekliyor: %s", wid, note)
            self.emit("worker", wid=wid, state="paused", note=note)
        else:
            log.info("[%s] yardımcı kart yeniden çalışıyor", wid)
            if wid in self.idle:
                self.emit("worker", wid=wid, state="idle")

    def _helper_dispatch(self, takers: int) -> None:
        """Yardımcıya görev: yalnız ana kart(lar)ın alabileceğinden fazla iş varsa (``takers`` = görev alabilen
        ana işçi sayısı; kart konuşmacı ayırmadaysa 0) ve kartta YouTube / tarayıcı için ayrılan bellek korunuyorsa
        (`_helper_check`). Başka programlar o belleği uzun süre aşarsa yardımcı kapatılır (bellek onlara kalır), yer
        açılınca sığan toplu boyutla yeniden açılır. ``self.lock`` altında çağrılır."""
        hp = self.hpool
        if hp is not None and self.helper_disabled:
            self._helper_stop_all()
            return
        if hp is None or self.hqueued_n > 0:
            return  # kuyruktaki görev henüz alınmadı: ikinci görev verilmez
        surplus = sum(1 for t in self.backlog
                      if (self.jobs.get(t["job"]) is not None and self.jobs[t["job"]].status == "run"))
        now = time.time()
        if surplus - takers > 0:
            if self._w_need_since is None:
                self._w_need_since = now
        else:
            self._w_need_since = None
        # konuşmacı ayırma işi bir süredir bekliyorsa (anlık değil: model açıp kapama boşa gitmesin)
        d_wait = self._d_need_since is not None and now - self._d_need_since >= HELPER_SWITCH_S
        if surplus - takers <= 0:  # ana kart yetişiyor: yardımcı boşta kalır (masaüstü, YouTube için)
            for wid in list(hp.workers):
                self._h_busy_since.pop(wid, None)
                self._h_probe.pop(wid, None)  # eski ölçümle sonradan görev verilmesin
                if wid in self._h_parked:
                    continue
                # konuşmacı ayırma bekliyorsa kart ona geçsin (metin işçisi kısa süre işsiz kalınca kapanır)
                if wid in self.idle and wid not in self.inflight and self.hqueued_n == 0 \
                        and now - self._h_last.get(wid, now) >= (HELPER_SWITCH_S if d_wait else HELPER_IDLE_PARK_S):
                    self._park_helper(wid, HELPER_OFF)
                    log.info("[%s] yardımcı kartta metin işi yok: metin işçisi kapatıldı (%s)", wid,
                             "kart konuşmacı ayırmaya geçiyor" if d_wait else "bellek boşaldı")
                else:
                    self._helper_note(wid, "")
            return
        for wid in list(hp.workers):
            parked = wid in self._h_parked
            if wid in self.fatal or now < self._h_ready.get(wid, 0.0):
                continue
            if not parked and wid not in self.idle:
                continue
            if parked and self._d_on():
                # kart konuşmacı ayırmada: o işsiz kalınca (HELPER_SWITCH_S) kapanır, sonra metin işçisi açılır
                self._helper_note(wid, "kapalı · kart konuşmacı ayırmada")
                continue
            pr = self._h_probe.get(wid)
            # kapalıyken her karar yeni ölçümle; açıkken ölçüm 15 sn geçerli (nvidia-smi her parçada çalışmasın: i3)
            fresh = max(self._h_ready.get(wid, 0.0), now - HELPER_RECHECK_S) if parked else now - HELPER_PROBE_S
            if pr is None or pr[0] < fresh:
                self._probe_helper(wid)  # kartı şimdi ölç (sonuç sonraki turda)
                continue
            reason, batch = self._helper_check(wid, pr[1], parked)
            if reason:
                since = self._h_busy_since.setdefault(wid, now)
                self._h_ready[wid] = now + HELPER_RECHECK_S
                self._h_probe.pop(wid, None)  # sonraki karar yeni ölçümle
                if not parked and now - since >= HELPER_PARK_S and wid not in self.inflight:
                    log.warning("[%s] yardımcı kart %.0f dakikadır bekliyor (%s): işçi kapatılıyor (bellek o programa "
                                "kalsın), kart boşalınca yeniden açılacak", wid, (now - since) / 60, reason)
                    self._park_helper(wid, reason)  # sebep başka programlar: toplu boyut küçülmez
                elif parked:
                    self._helper_note(wid, ("kapalı, bellek açılınca açılacak: " if reason.startswith("bilgisayarın belleği")
                                            else "kapalı, kart boşalınca açılacak: ") + reason)
                else:
                    self._helper_note(wid, reason)
                continue
            self._h_busy_since.pop(wid, None)
            if parked:
                self._h_probe.pop(wid, None)
                del self._h_parked[wid]
                log.info("[%s] yardımcı kart açılıyor (toplu boyut %d, %.1f GB YouTube / tarayıcıya ayrıldı)", wid,
                         batch, self._reserve())
                self._h_note[wid] = ""
                self._h_batch[wid] = batch
                self._h_own.pop(wid, None)
                self._h_done[wid] = 0
                ld = pr[1] or {}
                if isinstance(ld.get("used_mb"), (int, float)) and isinstance(ld.get("total_mb"), (int, float)):
                    self._h_open_free[wid] = (ld["total_mb"] - ld["used_mb"]) / 1024
                else:
                    self._h_open_free.pop(wid, None)
                if isinstance(getattr(hp, "_ecfg", None), dict):
                    hp._ecfg = {**hp._ecfg, "helper_batch": batch}  # yeni süreç bu toplu boyutla yükler
                self._mark_helper()
                self._foot_open(wid)
                hp.respawn(wid)
                self.emit("worker", wid=wid, state="loading")
                continue
            self._helper_note(wid, "")
            while self.backlog:
                t = self.backlog.popleft()
                job = self.jobs.get(t["job"])
                if not job or job.status != "run":
                    continue
                self.hqueued_n += 1
                hp.put(t)
                break
            return  # tek görev: alınınca (start) sıradaki yardımcı / görev

    def _park_helper(self, wid: str, reason: str, graceful: bool = True) -> None:
        """Yardımcı süreci kapat (model belleği karta geri döner); ``_health`` onu çöktü sanıp açmaz.
        graceful: süreç boşta görev bekliyor -> kapanma işaretiyle (kuyruk kilidi bırakılsın, bkz. WorkerPool.stop).
        Kendiliğinden kapanan süreçte (model yüklenemedi) işaret gönderilmez: kuyrukta kalıp yeni süreci kapatırdı."""
        self._h_parked[wid] = reason
        self._h_ready[wid] = time.time() + HELPER_RECHECK_S
        self._h_probe.pop(wid, None)
        self._h_busy_since.pop(wid, None)
        self.idle.discard(wid)
        pool = self.hdpool if wid == HELPER_DIAR else self.hpool
        if pool is not None:
            pool.stop(wid, idle=graceful)
        self._mark_helper_closed()
        self._helper_note(wid, reason if reason == HELPER_OFF else "kapalı, kart boşalınca açılacak: " + reason)

    def _helper_diar_dispatch(self) -> None:
        """Yardımcı kartta konuşmacı ayırma: ana ayırma işçisinin (3090) alamadığı ayırma işi varsa. Kart aynı anda
        tek iş tutar: metin işçisi açıksa önce o (işsiz kalınca) kapanır. Açılırken kartta ayrılan bellek
        (helper_reserve_gb) dışında ``HELPER_DIAR_GB`` yer olmalı; torch bellek sınırı da o boş yere göre konur.
        İşsiz kalınca kapanır (metin işi bekliyorsa ``HELPER_SWITCH_S``, yoksa ``HELPER_IDLE_PARK_S`` sonra).
        ``self.lock`` altında çağrılır."""
        if self.helper_disabled:
            self._helper_stop_all()
            return
        wid, now = HELPER_DIAR, time.time()
        # yardımcıda bellek yetmemiş görev (no_helper) yardımcıya bir daha verilmez: 3090'daki işçi alır
        need = sum(1 for t in self.dbacklog if not t.get("no_helper")
                   and self.jobs.get(t["job"]) is not None and self.jobs[t["job"]].status == "run")
        if not self._dmem_ok:
            need = 0  # RAM yetmediği için bekleyen ayırma yardımcıya (monitörlü karta) itilmesin
        if need:
            if self._d_need_since is None:
                self._d_need_since = now
        else:
            self._d_need_since = None
        if wid in self.fatal or wid in self._h_gone:
            return
        if wid in self._h_parked:
            if not need or self._w_on() or now < self._h_ready.get(wid, 0.0):
                return
            # yalnız iş gerçekten birikiyorsa: ana ayırma işçisi açılırken / yeniden başlarken bir an bekleyen işle
            # monitörlü kartta model açılmasın
            if now - self._d_need_since < HELPER_DIAR_WAIT_S or "diar" not in self.ready_workers:
                return
            pr = self._h_probe.get(wid)
            if pr is None or pr[0] < max(self._h_ready.get(wid, 0.0), now - HELPER_RECHECK_S):
                self._probe_helper(wid)
                return
            self._h_probe.pop(wid, None)
            ld = pr[1] or {}
            used, total = ld.get("used_mb"), ld.get("total_mb")
            frac = None
            if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
                free = (total - used) / 1024
                room = free - self._reserve()
                if room < HELPER_DIAR_GB:
                    self._h_ready[wid] = now + HELPER_RECHECK_S
                    self._helper_note(wid, f"kapalı, kart boşalınca açılacak: kartta yer yok ({free:.1f} GB boş, "
                                           f"{self._reserve():.1f} GB ayrılmış)")
                    return
                frac = min(0.9, max(0.2, room / (total / 1024)))  # torch bu kadarını aşmaz (ayrılan bellek kalsın)
            else:
                frac = self.cfg.display_gpu_limit_pct / 100 if 0 < self.cfg.display_gpu_limit_pct < 100 else None
            short = self._sysmem_short(wid)  # bilgisayarın belleği yetmiyorsa ikinci ayırma süreci açılmaz (1.9.27)
            if short:
                self._h_ready[wid] = now + HELPER_RECHECK_S
                self._helper_note(wid, "kapalı, bellek açılınca açılacak: " + short)
                return
            self._open_hdiar(frac)
            return
        # açık: boştaysa iş ver ya da kapan
        if wid not in self.idle or wid in self.inflight or self.hdqueued_n > 0:
            return
        if now < self._h_ready.get(wid, 0.0):
            return  # bellek ölçümü bekleniyor (nvidia-smi her turda çalışmasın: i3)
        if not need:
            idle_for = now - self._h_last.get(wid, now)
            w_wait = self._w_need_since is not None and now - self._w_need_since >= HELPER_SWITCH_S
            if idle_for >= (HELPER_SWITCH_S if w_wait else HELPER_IDLE_PARK_S):
                log.info("[%s] yardımcı kartta konuşmacı ayırma işi yok: kapatıldı (%s)", wid,
                         "kart metne geçiyor" if w_wait else "bellek boşaldı")
                self._park_helper(wid, HELPER_OFF)
            return
        pr = self._h_probe.get(wid)
        if pr is None or pr[0] < max(self._h_ready.get(wid, 0.0), now - HELPER_PROBE_S):
            self._probe_helper(wid)
            return
        ld = pr[1] or {}
        used, total = ld.get("used_mb"), ld.get("total_mb")
        if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
            free = (total - used) / 1024
            if free < self._reserve() / 2:
                since = self._h_busy_since.setdefault(wid, now)
                self._h_ready[wid] = now + HELPER_RECHECK_S
                self._h_probe.pop(wid, None)
                reason = f"başka programlar belleği kullanıyor ({free:.1f} GB boş)"
                if now - since >= HELPER_PARK_S:
                    self._park_helper(wid, reason)
                else:
                    self._helper_note(wid, reason)
                return
        self._h_busy_since.pop(wid, None)
        self._helper_note(wid, "")
        for t in list(self.dbacklog):
            job = self.jobs.get(t["job"])
            if not job or job.status != "run" or t.get("no_helper"):
                continue
            self.dbacklog.remove(t)
            self.hdqueued_n += 1
            self.hdpool.put(t)
            break

    def disable_helper(self, reason: str) -> None:
        """Monitörlü kartı bu oturumda bir daha kullanma: yardımcı işçiler (boştakiler hemen, çalışan görevini bitirince)
        kapanır; monitörlü kart ANA işçiyse ("Hepsi" / elle numara) bir sonraki görevini aldığı an kapatılır, görev
        öbür karta döner (boştaki işçi ortak kuyruktan kapatma işaretiyle güvenle kapatılamaz: işareti başkası alabilir)."""
        with self.lock:
            self.helper_disabled = reason or "kapatıldı"
            self._helper_stop_all()
            self._drop_msg = ""  # ekran koruması iletisi
            disp = set(self._ecfg.get("display_gpus") or [])
            spec = getattr(self.pool, "_spec", {}) or {}
            self._drop_wids = {w for w, d in spec.items() if str(d).startswith("cuda:") and int(str(d).split(":")[1]) in disp}
            others = [w for w in spec if w not in self._drop_wids and w not in self.fatal]
            if not others:
                self._drop_wids = set()  # tek kart monitörlü: kapatılırsa hiç işçi kalmaz (uyarı yeter)
            self._night_drops -= self._drop_wids  # ekran koruması: gece modu kapanınca da kapalı kalsın
            for w in list(self._drop_wids):
                if w in self.inflight:
                    self._drop_worker(w)

    def cards_in_use(self) -> tuple[list[str], list[str]]:
        """(ana kartlar, yardımcı kartlar) şu an iş alabilen: kapatılmış işçilerin kartı sayılmaz."""
        spec = getattr(self.pool, "_spec", {}) or {}
        main = sorted({str(d) for w, d in spec.items() if str(d).startswith("cuda:") and w not in self.fatal
                       and w not in self._drop_wids}) if spec else [d for d in self.devices if d.startswith("cuda:")]
        helpers = [] if self.helper_disabled else list(self.helper_devices)
        return main, helpers

    def limit_to_one(self, keep: str, reason: str) -> None:
        """Güç sınırı iş sürerken kalktı ve geri konamadı (uyku dönüşü, sürücü sıfırlanması, Afterburner): ``keep``
        dışındaki ana kart işçileri kapatılır (görevdeki hemen, görevi öbür karta döner; boştaki ilk görevinde),
        yardımcı kart kapatılır. Motor işler bitince yeniden açılınca kart seçimi baştan yapılır."""
        with self.lock:
            if self.helper_devices and not self.helper_disabled:
                self.helper_disabled = reason
                self._helper_stop_all()
            spec = getattr(self.pool, "_spec", {}) or {}
            drop = {w for w, d in spec.items() if str(d).startswith("cuda:") and str(d) != keep and w not in self.fatal}
            if not any(str(d) == keep and w not in self.fatal for w, d in spec.items()):
                drop = set()  # tutulacak kartta işçi yok: hepsini kapatmak işi durdurur
            self._drop_wids |= drop
            self._night_drops -= drop  # güç koruması: gece modu kapanınca da kapalı kalsın
            self._drop_msg = "güç koruması: güç sınırı kalktı, bu kart kapatıldı"
            for w in list(drop):
                if w in self.inflight:
                    self._drop_worker(w)

    def _drop_worker(self, wid: str) -> None:
        """Görevdeki (kuyruk kilidini tutmayan) monitörlü kart işçisini kapat; görevi geri sıraya koy, yeniden açma."""
        self._drop_wids.discard(wid)
        self.fatal.add(wid)
        msg = getattr(self, "_drop_msg", "") or "ekran koruması: monitörlü kart kapatıldı"
        self.worker_errors[wid] = msg
        cur = self.inflight.pop(wid, None)
        self.idle.discard(wid)
        if self.pool is not None:
            self.pool.stop(wid)
        if cur:
            task = self.tasks.get(cur[0])
            job = self.jobs.get(task["job"]) if task else None
            if task and job and job.status == "run":
                with self.lock:
                    (self.dbacklog if task["kind"] == "diarize" else self.backlog).appendleft(task)
        log.warning("[%s] %s; görevi öbür karta kaldı", wid, msg)
        self.emit("worker", wid=wid, state="error", error=msg[:1].upper() + msg[1:])

    def _helper_stop_all(self) -> None:
        note = "kapatıldı: " + self.helper_disabled
        for pool in (self.hpool, self.hdpool):
            for wid in (list(pool.workers) if pool is not None else []):
                if wid in self._h_parked:
                    self._helper_note(wid, note)
                    continue
                queued = self.hqueued_n if pool is self.hpool else self.hdqueued_n
                if wid in self.idle and wid not in self.inflight and not queued:
                    self._park_helper(wid, self.helper_disabled)
                    self._helper_note(wid, note)

    def _mark_helper(self) -> None:
        from .gpuguard import mark_helper_open
        mark_helper_open(self.out_root / ".stt")

    def _mark_helper_closed(self) -> None:
        if self.hpool is None or self._w_on() or self._d_on():
            return
        if any(int(d.split(":")[1]) in (self._ecfg.get("display_gpus") or []) for d in self.devices if d.startswith("cuda")):
            return  # monitörlü kart ana kart olarak hâlâ çalışıyor
        from .gpuguard import mark_helper_closed
        mark_helper_closed(self.out_root / ".stt")

    def _open_hdiar(self, frac: float | None) -> None:
        """Yardımcı kartta konuşmacı ayırma işçisini aç (model ilk görevde yüklenir)."""
        wid = HELPER_DIAR
        dev = self.helper_devices[0]
        from .tune import load_tuned
        # yardımcı kartta bellek sınırlı (ayrılan bellek dışı): küçük toplu boyut (ayarda sayı varsa o)
        ecfg = {**self._ecfg, "role": "diar", "diar_gpu_frac": frac, "diar_batch": self.cfg.diar_batch or 8,
                "diar_params": load_tuned(self.out_root / ".stt")}
        ecfg.pop("helper", None)
        del self._h_parked[wid]
        self._h_note[wid] = ""
        self._h_last[wid] = time.time()
        log.info("[%s] yardımcı kartta konuşmacı ayırma açılıyor (%s, bellek sınırı %s; %.1f GB YouTube / tarayıcıya "
                 "ayrıldı)", wid, dev, f"%{round(frac * 100)}" if frac else "yok", self._reserve())
        self._mark_helper()
        self._foot_open(wid)
        if self.hdpool is None:
            self.hdpool = WorkerPool([dev], ecfg, names=[wid])
            self.hdpool.start()
        else:
            self.hdpool._ecfg = ecfg
            self.hdpool.respawn(wid)
        self.emit("worker", wid=wid, state="loading", name="Konuşmacı ayırma (yardımcı)")

    def _helper_gone(self, wid: str) -> None:
        """Yardımcı bir daha açılmayacak: kuyruğunda bekleyen görev ana kartlara geri döner."""
        if self.hpool is None:
            return
        if any(w not in self.fatal and self.restarts.get(w, 0) < self.cfg.worker_restarts
               for w in self.hpool.workers if w != wid):
            return  # başka yardımcı var: kuyruktakini o alır
        back = self.hpool.drain()
        with self.lock:
            for t in reversed(back):
                self.backlog.appendleft(t)
            self.hqueued_n = 0

    def _on_msg(self, m: dict):
        typ, wid = m["type"], m.get("wid")
        if typ == "ready":
            self.idle.add(wid)
            self.ready_workers.add(wid)
            self._born[wid] = time.time()
            if self._is_helper(wid):
                self._h_last[wid] = time.time()
            self.emit("worker", wid=wid, state="idle")
            return
        if typ == "fatal":
            hint = worker_hint(m["error"])
            if (self._is_helper(wid) or self._is_hdiar(wid)) and m.get("retryable"):
                # yardımcı kartta model yüklenirken bellek yetmedi (oyun / başka program): yeniden açma döngüsüne
                # girmesin (6 denemeden sonra oturum boyu kaybolurdu); kart boşalınca açılır
                log.warning("[%s] yardımcı kartta model yüklenemedi (%s); kart boşalınca tekrar denenecek", wid,
                            m["error"][:160])
                if is_oom(m["error"]) and self._is_helper(wid):
                    self._helper_shrink(wid)
                self._park_helper(wid, "model yüklenemedi (kartta yer yok)", graceful=False)
                return
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
            if self._is_helper(wid):
                self._helper_gone(wid)
            elif self._is_hdiar(wid):  # kuyruğunda bekleyen ayırma görevi 3090'daki işçiye dönsün
                back = self.hdpool.drain()
                with self.lock:
                    for t in reversed(back):
                        self.dbacklog.appendleft(t)
                    self.hdqueued_n = 0
            log.error("[%s] işçi başlatılamadı: %s", wid, m["error"])
            log.info("%s", m.get("trace", ""))
            main_ws = set(self.pool.workers) if self.pool else set()
            if main_ws and not (self.ready_workers & main_ws) and main_ws <= set(self.worker_errors) \
                    and not self._flex_spare():
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
            if wid in self._drop_wids:  # ekran koruması: monitörlü kart işçisi görev aldı -> kapat (görev geri döner)
                if wid == "diar":
                    self.dqueued_n = max(0, self.dqueued_n - 1)
                else:
                    self.queued_n = max(0, self.queued_n - 1)
                self._drop_worker(wid)
                return
            if wid == "diar":
                self.dqueued_n = max(0, self.dqueued_n - 1)
            elif self._is_helper(wid):
                self.hqueued_n = max(0, self.hqueued_n - 1)
            elif self._is_hdiar(wid):
                self.hdqueued_n = max(0, self.hdqueued_n - 1)
            else:
                self.queued_n = max(0, self.queued_n - 1)
            t = self.tasks.get(m["tid"]) or {}
            j = self.jobs.get(t.get("job", ""))
            self._q_mark(j, True)
            self.emit("worker", wid=wid, state="busy", task=t.get("kind"),
                      title=j.item.title if j else "", uid=j.item.extra.get("uid") if j else None)
            return
        tid = m["tid"]
        self.inflight.pop(wid, None)
        self._q_mark(self.jobs.get((self.tasks.get(tid) or {}).get("job", "")), False)
        if typ == "done":
            self.restarts[wid] = 0  # çalışıyor: yeniden başlatma sayacı sıfırlanır
        self.idle.add(wid)
        if getattr(self.cfg, "night_mode", False) and not (self._is_helper(wid) or self._is_hdiar(wid)):
            # gece modu: kart her görevden sonra ısıya göre dinlenir (stt/governor.py)
            self._rest_until = max(self._rest_until, time.time() + self.gov.rest)
        elif not (self._is_helper(wid) or self._is_hdiar(wid)) and self._flex_level() >= 2:
            self._flex_rest_until = max(self._flex_rest_until, time.time() + FLEX_REST_S)  # esnek kademe 2
        if self._is_helper(wid):  # yardımcı kart dinlenir (kart serinler, ölçüm yalnız başka programları görür)
            self._h_ready[wid] = time.time() + self.cfg.helper_rest_s
            self._h_last[wid] = time.time()
            if typ == "done":
                self._h_done[wid] = self._h_done.get(wid, 0) + 1
                if self._h_done[wid] == 1:
                    self._h_probe.pop(wid, None)  # kendi payı ilk parçadan SONRAKİ ölçümle (_helper_measure)
        elif self._is_hdiar(wid):
            self._h_last[wid] = time.time()
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
        if typ == "error" and self._is_helper(wid) and task["kind"] == "transcribe" and is_oom(m["error"]):
            # başka program (oyun) görev sırasında kartın belleğini aldı: parça kaybolmasın (deneme sayılmaz),
            # ana karta / sonraya kalır; yardımcı kapanır, bellek o programa kalır
            log.warning("[%s] yardımcı kartta bellek yetmedi (başka program?): parça yeniden sırada", wid)
            with self.lock:
                self.backlog.appendleft(task)
            self._helper_shrink(wid)
            self._park_helper(wid, "görev sırasında kartta bellek yetmedi")
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
                    json.dumps({"segments": [], "language": None, "skipped": True}), encoding="utf-8")
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
            if task["kind"] == "diarize" and self._is_hdiar(wid) and is_oom(m["error"]):
                # yardımcı kartta bellek yetmedi (başka program aldı): görev kaybolmasın, deneme sayılmasın;
                # 3090'daki ayırma işçisi ya da sonra yardımcı alır. Yardımcı kapanır (bellek o programa kalır).
                job.attempts[tid] = n - 1
                log.warning("[%s] yardımcı kartta konuşmacı ayırmaya bellek yetmedi: görev 3090'daki ayırmaya kaldı",
                            wid)
                retry = {**task, "no_helper": True}  # aynı görev yardımcıda tekrar tekrar bellek aşmasın
                self.tasks[tid] = retry
                with self.lock:
                    self.dbacklog.appendleft(retry)
                self._park_helper(wid, "konuşmacı ayırmada kartta bellek yetmedi")
                return
            if task["kind"] == "diarize" and n < 3 and (is_ram_error(m["error"]) or is_oom(m["error"])):
                # RAM / ekran kartı belleği yetmedi (tarayıcı belleği doldurunca CUDA da "out of memory" veriyor):
                # ayırma sürecini kapat (bellek tamamen boşalsın), metin çevirisi bitince bekleyip tekrar dene
                wait = 60 * n
                retry = task if n < 2 else {**task, "vocals": False}  # 2. denemede müzik ayıklamasız (o da bellek ister)
                self.tasks[tid] = retry
                log.warning("Konuşmacı ayırmada bellek yetmedi; metin çevirisi bitince (en erken %d sn sonra) "
                            "tekrar denenecek (%d/2): %s", wait, n, job.item.title[:60])
                with self.lock:
                    self.ddelayed.append((time.time() + wait, retry))
                self._respawn_diar(wid)
                return
            dp = self._pool_of(wid) if wid in ("diar", HELPER_DIAR) else None
            if task["kind"] == "diarize" and dp is not None and wid in dp.workers \
                    and any(x in m["error"].lower() for x in ("cuda", "cudnn", "cublas", "ekran kart")):
                # ekran kartı hatası: süreci kapat, sonraki video temiz bir CUDA ile başlasın (CPU'ya düşmesin)
                self._respawn_diar(wid)
                if n < 3 and "out of memory" not in m["error"].lower():
                    # bozulan CUDA bağlamı yeni süreçte genelde düzelir: konuşmacısız bitirmeden önce tekrar dene.
                    # Müzik ayıklama sırasında bozulduysa (ya da ikinci deneme) müzik ayıklamadan dene.
                    retry = dict(task)
                    if n >= 2 or "müzik ayıklama" in m["error"].lower():
                        retry["vocals"] = False
                    self.tasks[tid] = retry
                    log.warning("Konuşmacı ayırmada ekran kartı hatası; süreç yeniden açıldı, tekrar deneniyor "
                                "(%d/2%s): %s", n, ", müzik ayıklamasız" if not retry.get("vocals") else "",
                                job.item.title[:60])
                    with self.lock:
                        self.dbacklog.appendleft(retry)
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
                job.chunk_lang[task["idx"]] = payload.get("language")
                if not task.get("redo"):  # dil düzeltmesi için yeniden çeviri ilerlemeye sayılmaz
                    job.done_chunks += 1
                total = task["n"] or len(job.chunks)
                # metin bitti ama konuşmacı ayırma sürüyor / bekliyor: arayüz "3/3"te takılı görünmesin
                diar_wait = any((self.tasks.get(x) or {}).get("kind") == "diarize" for x in job.pending)
                self.emit("progress", job.item, done=job.done_chunks, total=total, speed=dur / max(0.01, m["elapsed"]),
                          live=job.live_active or None, diar_wait=diar_wait or None, started=True)
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

    def _q_mark(self, job, on: bool) -> None:
        """Kalıcı kuyrukta "started": işin ekran kartında görevi sürüyor. Uygulama / motor bu sırada çökerse sürdürmede
        bu iş bir deneme sayılır (3'te kalıcı hata). İnceleme: indirme başında işaretlenince her kirli kesintide sırada
        hazırlanan 6 masum video da suçlanıyordu; artık yalnız görevi süren işler."""
        uid = job.item.extra.get("uid") if job is not None else None
        if not uid:
            return
        try:
            if on and uid not in self._q_marked:
                self._q_marked.add(uid)
                self.state.queue_started(uid, job.item.key)
            elif not on and uid in self._q_marked and not any(
                    (self.tasks.get(t) or {}).get("job") == job.item.key for t, _ in list(self.inflight.values())):
                self._q_marked.discard(uid)
                self.state.queue_unstart(uid)
        except Exception:  # noqa: BLE001
            log.debug("kuyruk işareti yazılamadı", exc_info=True)

    def _respawn_diar(self, wid: str) -> None:
        """Hatadan sonra boştaki ayırma sürecini yeniden aç (kapanma işaretiyle: kuyruk kilidi bırakılsın)."""
        pool = self._pool_of(wid) if wid in ("diar", HELPER_DIAR) else None
        if pool is None or wid not in pool.workers or wid in self._h_parked:
            return
        pool.respawn(wid, idle=True)
        self.idle.discard(wid)
        if wid == "diar":
            self.dqueued_n = 0
        else:
            self.hdqueued_n = 0
        self.emit("worker", wid=wid, state="loading")

    def _task_timeout(self, task: dict, wid: str | None = None) -> float:
        """Takılma sınırı (sn): GPU kilitlenir / indirme donarsa işçi yeniden başlatılır.
        Konuşmacı ayırma ilerledikçe sinyal gönderir; süre SON SİNYALDEN sayılır (eskiden 3 x video süresiydi,
        takılan iş saatlerce bekliyordu)."""
        if task.get("kind") == "diarize":
            job = self.jobs.get(task.get("job", ""))
            dur = (job.info.get("duration_s", 0) if job else 0) or 3600
            return max(900.0, 0.25 * dur)
        dur = task.get("end", 0) - task.get("start", 0)
        pool = self._pool_of(wid) if wid else self.pool
        dev = (getattr(pool, "_spec", None) or {}).get(wid, "") if pool else ""
        if str(dev).startswith("cuda"):  # ekran kartı 20-40x hızlı: 5 dk'lık parça ~10 sn; 40 dk beklemeye gerek yok
            return max(300.0, 3 * dur)
        return max(900.0, 8 * dur)

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
        if not self.pool or self._closing:
            return  # kapanıyor (uygulama kapandı / güncelleme): ölen işçiler yeniden açılmasın
        now = time.time()
        for wid, (tid, t0) in list(self.inflight.items()):  # takılan görev: işçiyi yeniden başlat
            task = self.tasks.get(tid) or {}
            pool = self._pool_of(wid)
            if task and now - t0 > self._task_timeout(task, wid) and wid in pool.alive():
                log.error("[%s] görev %.0f dakikadır bitmedi (GPU takıldı?); işçi yeniden başlatılıyor", wid,
                          (now - t0) / 60)
                pool.respawn(wid)
                self.restarts[wid] = self.restarts.get(wid, 0) + 1
                self.inflight.pop(wid, None)
                self._requeue(wid, tid, "işçi takıldı")
                self.idle.discard(wid)
                self.emit("worker", wid=wid, state="loading")
        for pool in self._pools():  # model yüklerken takılan işçi (görev yok, hazır da değil)
            if pool.threads:
                continue
            for wid in pool.alive():
                if wid in self.idle or wid in self.inflight or wid in self.fatal:
                    self._loading_since.pop(wid, None)
                    continue
                t0 = self._loading_since.setdefault(wid, now)
                if now - t0 > 1800:
                    log.error("[%s] işçi 30 dakikadır hazır olmadı (model yüklenirken takıldı?); yeniden başlatılıyor", wid)
                    self._loading_since.pop(wid, None)
                    self.restarts[wid] = self.restarts.get(wid, 0) + 1
                    pool.respawn(wid)
                    self.emit("worker", wid=wid, state="loading")
        main_alive: set = set()
        for pool in self._pools():
            alive = set(pool.alive())
            for wid in list(pool.workers):
                if wid in alive or wid in self._h_parked:  # kapatılmış yardımcı: kart boşalınca açılır
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
                elif pool is self.hpool and wid not in self._h_gone:
                    self._h_gone.add(wid)
                    self._helper_gone(wid)
                elif pool is self.hdpool and wid not in self._h_gone:
                    self._h_gone.add(wid)
                    back = self.hdpool.drain()
                    with self.lock:
                        for t in reversed(back):
                            self.dbacklog.appendleft(t)
                        self.hdqueued_n = 0
            if pool is self.pool:
                main_alive = main_alive | alive
            elif pool is self.hpool and self._flex_spare():
                # esnek düzen: ana kart düştüyse ikinci kart metni sürdürür (eskiden iki kart da ana işçiydi)
                main_alive = main_alive | {w for w in pool.workers if w not in self.fatal and w not in self._h_gone}
        self._drop_dead_diar()
        alive = main_alive
        if not alive and self.active():
            why = next(iter(self.worker_errors.values()), "") or "GPU/model hatası"
            # işler başarısız sayılıp silinmez (1.9.27; kullanıcıda bellek dolunca "sıradaki işlerin hepsi silindi"):
            # kalıcı kuyrukta kalır, uygulama motoru yeniden açınca kaldığı yerden sürer. Yarıda kalanlar deneme sayılır
            # (_dirty_stop): aynı video motoru hep düşürüyorsa sonsuz döngü olmasın.
            self._dirty_stop = True
            for job in list(self.jobs.values()):
                if job.status in ("run", "prep", "queued"):
                    job.status = "interrupted"
                    job.error = f"Motor durdu ({why[:160]}); yeniden açılınca kaldığı yerden sürecek"
                    self.emit("interrupted", job.item, error=job.error)
                    self.state.set(job.item.key, source=job.item.source, status="interrupted", title=job.item.title,
                                   error=job.error)
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
        if not self.keep_results:
            # arayüz modu (1.9.27 inceleme): metin + kelime zamanları iş başına ~2 MB; günlerce süren kuyrukta 1000 işte
            # ~2 GB bellekte kalıyordu. Metin diskte; komut satırı özeti için yalnız orada tutulur.
            job.result = None
            job.chunks, job.live_buf = [], {}
        with self.lock:
            self.results.append(job)

    def _prune(self) -> None:
        """Arayüz modu: biten işlerin son KEEP_FINISHED'i dışındakiler bellekten (iş, görevleri, sonuç listesi)."""
        if self.keep_results:
            return
        with self.lock:
            done = [k for k, j in self.jobs.items()
                    if j.status not in ("queued", "prep", "run", "finalize") and not j.live_active]
            drop = set(done[:-KEEP_FINISHED]) if len(done) > KEEP_FINISHED else set()
            for k in drop:
                job = self.jobs.pop(k)
                uid = job.item.extra.get("uid")
                if uid and self.items.get(uid) is job.item:
                    self.items.pop(uid, None)
            if drop:
                for tid in [t for t, task in self.tasks.items() if task.get("job") in drop]:
                    del self.tasks[tid]
            if len(self.results) > KEEP_FINISHED:
                del self.results[:-KEEP_FINISHED]

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
        """FPL fikstür verisi. Eskiden oturumda bir kez yükleniyordu: açılışta FPL'ye ulaşılamazsa günlerce açık
        kalan uygulamada her video GW'siz yazılıyordu; erteleme / skor da hiç yenilenmiyordu. Artık veri yoksa her
        bitirmede istenir (``fpl_data`` ulaşamayınca 10 dk ağı denemez, hemen döner), varsa FPL_MEM_S'de bir yeniden
        okunur (6 saatlik disk önbelleği)."""
        now = time.time()
        if self._fpl is not None and now - self._fpl_at < FPL_MEM_S:
            return self._fpl
        data = fpl_data(self.out_root / ".stt")
        if data is not None:
            self._fpl = data
        self._fpl_at = now  # yenilenemediyse eldeki veriyle devam, yarım saat sonra yine dene
        return self._fpl

    def _out_paths(self, job: Job, meta: dict) -> tuple[Path, str]:
        title = job.info.get("title") or job.item.title or "kayit"
        return out_paths(self.out_root, meta, title, job.item.key)

    def _voice_names(self, job: Job, dj: dict, speakers: dict, mode: str, td_label: str) -> str:
        """Konuşmacı imzalarını sakla; ses kütüphanesinde tanınan kişilere adlarını ver. Döner: TD etiketi."""
        emb = dj.get("emb")
        if not emb or not speakers:
            return td_label
        from .voices import VoiceLibrary, apply_voice_names, save_doc_voices
        st = self.out_root / ".stt"
        try:
            # turlar da saklanır: düzeltmede kümenin ne kadarı başka kişiye taşındı (karışık küme öğrenilmesin)
            save_doc_voices(st, job.item.key, dj.get("model"), emb, dj.get("turns"))
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
            diar_meta = {k: dj[k] for k in ("model", "vocals", "agc") if dj.get(k) is not None}
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
                         **({"diar_music_removed": True} if diar_meta.get("vocals") else {}),
                         **({"diar_agc": bool(diar_meta["agc"])} if "agc" in diar_meta else {})},
            "match": job.match, "td_label": td_label if speakers and mode == "basin" else None,
            "speaker_mode": mode if speakers else None,
            "diarization_error": job.diar_error or None, "speaker_source": speaker_source,
            "speakers": speakers, "source_info": {k: v for k, v in info.items() if k != "description"},
            # açıklamanın başı: yeniden sınıflandırma (stt yeniden-siniflandir) rakip / kupa adını buradan da okur
            "description_head": (str(info.get("description") or "")[:400]) or None,
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
        set_team_manager(self.out_root, meta, job.item.key,   # sezon klasörü "<sezon> <o günkü hoca>"
                         info.get("channel") or info.get("uploader") or "")
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
        update_index(self.out_root, index_row(self.out_root, job.item.key, meta, doc, paths))
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
                             "skipped_chunks": info.get("skipped_chunks") or None,
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
                  skipped=[f"{fmt_clock(a)}–{fmt_clock(b)}" for a, b in info.get("skipped_chunks") or []] or None,
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
        for it in list(self.items.values()):  # sırada bekleyip hazırlanmamışlar: ajan bunları yeniden göndermesin
            if not it.key or it.key not in self.jobs:
                self._note_cancelled(it)
        log.warning("Durduruldu: %d iş. Çalışan parça bitince işçiler boşa çıkar.", n)
        return n

    def cancel(self, uid: str) -> bool:
        """Tek bir işi durdur (indiriliyor / çevriliyor / sırada). Biten parçalar saklanır."""
        from . import fetch as _fetch
        item = self.items.get(uid)
        if item is None:
            return False
        item.extra["_cancel"] = True
        kill: list = []
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
                kill = [w for w in ("diar", HELPER_DIAR)
                        if (self.tasks.get((self.inflight.get(w) or (None,))[0] or "") or {}).get("job") == item.key]
        for w in kill:  # uzun ayırma görevini beklemeden kes (görevdeki süreç kuyruk kilidini tutmaz)
            pool = self._pool_of(w)
            if pool is None or w not in pool.workers:
                continue
            pool.respawn(w)
            self.inflight.pop(w, None)
            self.idle.discard(w)
            if w == "diar":
                self.dqueued_n = 0
            else:
                self.hdqueued_n = 0
            self.emit("worker", wid=w, state="loading")
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
        self._closing = True  # bu andan sonra düşen işler kalıcı kuyruktan silinmesin (sonraki açılışta sürer)
        try:  # düzgün kapanış: monitörlü kartın / iş yapmanın açık aralığı kapansın (ekran koruması)
            from .gpuguard import mark_busy, mark_helper_closed
            mark_helper_closed(self.out_root / ".stt")
            mark_busy(self.out_root / ".stt", False)
        except Exception:  # noqa: BLE001
            pass
        try:  # temiz kapanış: yarım işler deneme sayılmaz
            self.state.queue_unstart_all()
            self._q_marked.clear()
        except Exception:  # noqa: BLE001
            pass
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
        self._closing = True  # sıra bekleyen hazırlık iş parçacıkları çıksın (komut satırı kapanışta asılı kalmasın)
        try:  # düzgün kapanış: monitörlü kartın / iş yapmanın açık aralığı kapansın (ekran koruması)
            from .gpuguard import mark_busy, mark_helper_closed
            mark_helper_closed(self.out_root / ".stt")
            mark_busy(self.out_root / ".stt", False)
        except Exception:  # noqa: BLE001
            pass
        if not getattr(self, "_dirty_stop", False):  # işçiler hep birlikte öldüyse (çökme) yarım işler deneme sayılır
            try:
                self.state.queue_unstart_all()
                self._q_marked.clear()
            except Exception:  # noqa: BLE001
                pass
        from . import fetch as _fetch
        with self.lock:  # sürmekte olan indirmeler kesilsin: yeni motor aynı işi sürdürürken aynı klasöre yazmasın
            for job in self.jobs.values():
                if job.status == "prep":
                    _fetch.CANCELLED.add(str(job.work))
        self.prep.shutdown(wait=False, cancel_futures=True)
        self.fin.shutdown(wait=True)
        for pool in self._pools():
            pool.close()
