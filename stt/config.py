"""Ayarlar: varsayılanlar < stt.toml < ortam değişkenleri < komut satırı."""
from __future__ import annotations

import math
import os
import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    out_dir: Path = Path("cikti")
    lang: str | None = None              # None = otomatik
    languages: str | None = None         # otomatik algılamada izinli diller ("en,tr,az"); None = varsayılan liste
    engine: str = "whisper"              # whisper | openai
    model: str = "large-v3"
    compute_type: str = "float16"
    beam_size: int = 5
    batch_size: int = -1                 # -1 = karta göre otomatik (3090:16, 3080:8), 0/1 = kapalı
    gpus: str = "auto"                   # "auto" (monitörsüz kart), "yardim" (monitörlü kart yardımcı, düşük yük),
                                         # "all" (hepsi tam), "0,1", "0", "cpu"
    display_batch: int = 4               # monitörün bağlı olduğu kart kullanılırsa en fazla bu toplu boyut
    # "yardim": monitörlü kart (3080) ana kart (3090) yetişemeyince metin çevirir. Kartta YouTube / tarayıcı
    # sekmeleri için helper_reserve_gb boş bırakılır, kalan bellek ve güç bu işe verilir (toplu boyut ona göre).
    # Başka programlar belleği kullanırsa beklenir / yardımcı kapanır. 3090'daki konuşmacı ayırma yetişemeyince
    # kart konuşmacı ayırmaya geçer (aynı anda tek iş: metin ya da ayırma).
    helper_reserve_gb: float = 2.0       # yardımcı kartta başka programlara (YouTube, sekmeler) bırakılan boş bellek
    helper_rest_s: float = 0.0           # yardımcı kart her parçadan sonra bu kadar dinlenir (0 = tam güç)
    # iki kart birlikte (1.9.17): yalnız güç + saat sınırı uygulanmışken (stt/power.py). Kullanıcıda yalnız 3090 yükteyken
    # bile bilgisayar yeniden başlıyordu (güç sıçraması / hız aşırtma); iki kart tam yükte bu toplanır.
    safe_dual: bool = True               # sınır uygulanmamışsa tek kart (Ayarlar > Ekran kartı > Güvenli güç ayarı)
    gpu_power_pct: int = 70              # güç sınırı: kartın varsayılan gücünün yüzdesi (50-100)
    gpu_max_clock: int = 1600            # en yüksek çekirdek saati (MHz; boost sıçramalarını keser). 0 = sınır yok
    # esnek tam güç (1.9.21, stt/flex.py): "Önerilen"de iki kart da monitörsüzse ikinci kart metin ya da konuşmacı
    # ayırmaya (iş nerede birikmişse) geçer; başka programlar işlemciyi / RAM'i kullanınca iş kademeli yavaşlar, onlar
    # bırakınca tam güce döner. Kapalı = eski düzen (iki kart da yalnız tam metin, yavaşlama yok).
    gpu_flex: bool = True
    # açık hız aşırtma programı (MSI Afterburner) için kullanıcının "Reset + Save yaptım" onayı (program adları; uygulama
    # Afterburner'ın içindeki ayarı okuyamaz, yalnız açık olduğunu görür). Başka bir program açılırsa onay geçmez.
    oc_ack: str | None = None
    # gece modu (1.9.19, stt/governor.py): uzun kuyruk için yavaş ama dengeli. Tek kart, monitörlü karta / yardımcıya iş
    # yok, her görevden sonra ısıya göre mola (kart night_temp çevresinde), çok ısınırsa soğuma molası, işçiler
    # night_recycle_min dakikada bir boştayken yenilenir (bellek birikmesin), önden en çok 3 video hazırlanır.
    night_mode: bool = False
    night_temp: int = 72                 # gece modunda hedef kart ısısı (°C; 60-83)
    night_recycle_min: float = 90.0      # gece modunda işçi süreçleri bu kadar dakikada bir yenilenir (0 = kapalı)
    # 1.9.27: gece modu dışında da uzun kuyrukta işçiler bu kadar dakikada bir boştayken yenilenir (bellek birikmesin;
    # model ~30 sn'de yeniden yüklenir). 0 = kapalı
    recycle_min: float = 180.0
    min_free_gb: float = 3.0             # diskte bundan az boş yer varsa yeni video indirilmez (bekler; 0 = kapalı)
    low_priority: bool = True            # tanıma süreçleri düşük öncelikte (bilgisayar donmasın)
    isolate_gpu: bool = True             # her işçi yalnızca kendi kartını görür (monitörlü kartta CUDA bağlamı yok)
    ui_gpu: bool = False                 # arayüz (WebView2) ekran kartı hızlandırması; kapalı = ekran kararmasın
    prompt: str | None = None
    condition_on_previous_text: bool = True
    agc: bool = True                     # otomatik kazanç: uzaktaki muhabir sesini yükselt (tanıma + ayırma)
    diar_agc: bool | None = None         # konuşmacı ayırmada otomatik kazanç; boş = agc ile aynı (A/B ölçümü için)
    vad_threshold: float = 0.4           # Whisper VAD eşiği (düşük = sessiz konuşmayı da yakalar)
    # bellek sınırları (yüzde). Ekran kartı: konuşmacı ayırma (torch) bu oranı aşmaz.
    gpu_limit_pct: int = 90              # monitörsüz kart (ör. RTX 3090)
    display_gpu_limit_pct: int = 70      # monitörlü kart (ör. RTX 3080)
    ram_limit_pct: int = 70              # boş RAM'in en çok bu kadarı; yetmezse bekler, sonra tekrar dener
    diar_cpu: bool = False               # konuşmacı ayırma işlemcide (CPU) yapılabilsin mi (yavaş, fan çok döner)
    diar_batch: int = 0                  # konuşmacı ayırma toplu boyutu; 0 = karta göre (24 GB: 32, 10-12 GB: 16, diğer 8).
                                         # Sonucu değiştirmez (parçalar tek tek hesaplanır), yalnız hız / bellek
    fill_gaps: bool = True               # metinde uzun boşluk kalırsa (müzik altında konuşma) o kısım VAD'siz tekrar
    worker_restarts: int = 6             # art arda en fazla bu kadar yeniden başlatma (başarılı görev sayacı sıfırlar)
    # bölme
    split_min: float = 8.0               # bundan uzun dosyalar VAD ile parçalanır (dakika)
    chunk_min: float = 5.0               # hedef parça uzunluğu (dakika)
    # konuşmacı ayırma
    diarize: bool = False
    ui_diarize: bool | None = None       # arayüzdeki "Konuşmacıları ayır" seçimi (hatırlanır); boş = token varsa açık
    hf_token: str | None = None
    diar_model: str = "pyannote/speaker-diarization-community-1"  # olmazsa 3.1'e düşülür
    diar_engine: str = "pyannote"        # pyannote | diarizen (ayrı ortam: python -m stt diarizen-kur; CC BY-NC)
    diar_music: str = "auto"             # konuşmacı ayırmadan önce müzik ayıklama: auto (podcast/film) | on | off
    voice_match: bool = True             # ses kütüphanesindeki kişileri yeni videolarda adıyla tanı
    voice_threshold: float = 0.6         # ses imzası benzerliği (kosinüs) en az bu kadarsa aynı kişi sayılır
    min_speakers: int | None = None
    max_speakers: int | None = None
    speaker_mode: str = "auto"           # auto | basin (TD + muhabir) | herkes (podcast/film: Konuşmacı 1..N)
    td_label: str = "Teknik direktör"
    reporter_label: str = "Muhabir"
    # indirme
    cookies_browser: str | None = "auto"  # auto = kurulu tarayıcıları sırayla dene (firefox, edge, chrome, brave...)
    cookies_file: str | None = None
    prep_workers: int = 2
    download_retries: int = 3            # geçici ağ hatalarında indirme tekrarı
    browser_capture: bool = True         # son çare: gizli tarayıcıyla (Edge/Chrome) sayfadaki videoyu yakala
    ytdlp_autoupdate: bool = True        # yt-dlp günde bir kez güncellenir (siteler sık değişiyor)
    # kanal / oynatma listesi
    channel_max: int = 200               # bir kanaldan en fazla bu kadar video listelenir (yeniden eskiye)
    follow_interval_h: float = 6.0       # kanal takibi kaç saatte bir yeni video arasın
    # canlı yayın
    live_chunk_s: float = 20.0           # canlı yayında kaç saniyede bir metin çıksın
    live_max_h: float = 6.0              # canlı kayıt en fazla (saat)
    # yapay zekâ (isteğe bağlı): özet, haber taslağı, soru-cevap düzeltme, konu filtresi
    llm_provider: str | None = None      # None | gemini | claude | openai
    llm_model: str | None = None         # boş = sağlayıcının ucuz/hızlı modeli
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None    # yapay zekâ (openai) ve OpenAI yedek motoru; OPENAI_API_KEY de okunur
    llm_budget_usd: float = 5.0          # aylık harcama sınırı (tahmini); aşılınca yapay zekâ adımları atlanır
    llm_news: bool = False               # her metin için haber taslağı + alıntılar (haber.md)
    llm_qa: bool = True                  # konuşmacı ayrılamazsa soru-cevabı yapay zekâyla ayır
    # dış programlar (haber sitesi / haber radarı)
    api_port: int = 0                    # 0 = kapalı; ör. 8765 -> http://127.0.0.1:8765/api
    api_token: str | None = None
    webhook_url: str | None = None       # her biten iş için JSON POST
    # durum
    force: bool = False
    keep_audio: bool = False
    # FPL
    fpl_gw: int | None = None
    fpl_team: str | None = None
    fpl_repo: Path | None = None
    fpl_python: str | None = None        # None = python3 / py -3 / bu python
    fpl_auto: bool = True                # PL maçı bulunursa GW/takım otomatik, presser.py otomatik
    # OpenAI yedek motor
    openai_model: str = "gpt-transcribe"
    openai_workers: int = 4
    openai_chunk_s: float = 30.0
    # izleme
    watch_interval: float = 5.0

    @property
    def state_dir(self) -> Path:
        return Path(self.out_dir) / ".stt"


ENV_MAP = {
    "HF_TOKEN": "hf_token",
    "HUGGINGFACE_TOKEN": "hf_token",
    "FPL_REPO": "fpl_repo",
    "STT_OUT": "out_dir",
    "STT_COOKIES_BROWSER": "cookies_browser",
    "STT_COOKIES_FILE": "cookies_file",
    "STT_MODEL": "model",
    "STT_GPUS": "gpus",
    "GOOGLE_API_KEY": "gemini_api_key",
    "GEMINI_API_KEY": "gemini_api_key",   # sonra gelen önceliklidir
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENAI_API_KEY": "openai_api_key",
    "STT_API_PORT": "api_port",
    "STT_LLM": "llm_provider",
}


def diar_agc_on(cfg) -> bool:
    """Konuşmacı ayırmaya giden seste otomatik kazanç açık mı (``diar_agc`` boşsa metin tanımadaki ``agc``)."""
    v = getattr(cfg, "diar_agc", None)
    return bool(getattr(cfg, "agc", True)) if v is None else bool(v)


SECRETS = ("hf_token", "gemini_api_key", "anthropic_api_key", "openai_api_key", "api_token")  # arayüzden silinebilir


def env_secrets() -> dict[str, str]:
    """Ortam değişkeninden gelen anahtarlar: {ayar adı: değişken adı}. Bunlar stt.toml'dan silinse de geçerli kalır
    (ortam değişkeni önceliklidir), arayüz "buradan silinemez" der."""
    out: dict[str, str] = {}
    for env, attr in ENV_MAP.items():
        val = os.environ.get(env, "").strip()
        if attr in SECRETS and val and not (val.startswith("%") and val.endswith("%")):
            out[attr] = env
    return out


def config_files() -> list[Path]:
    cands = [TOOL_DIR / "stt.toml", Path.cwd() / "stt.toml"]
    if os.name == "nt" and os.environ.get("APPDATA"):
        cands.append(Path(os.environ["APPDATA"]) / "stt" / "stt.toml")
    else:
        cands.append(Path.home() / ".config" / "stt" / "stt.toml")
    seen, out = set(), []
    for c in cands:
        r = c.resolve()
        if r not in seen and c.is_file():
            seen.add(r)
            out.append(c)
    return out


def _coerce(name: str, value):
    types = {f.name: f.type for f in fields(Config)}
    t = str(types.get(name, ""))
    if value is None:
        return None
    if "Path" in t:
        return Path(str(value)).expanduser()
    if t.startswith("bool"):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "evet", "yes", "on")
        return bool(value)
    if t.startswith("int"):
        return int(value)
    if t.startswith("float"):
        return float(value)
    return value


def load_config() -> Config:
    cfg = Config()
    names = {f.name for f in fields(Config)}
    for path in reversed(config_files()):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8-sig"))  # PowerShell BOM ekleyebilir
        except Exception as e:  # noqa: BLE001
            print(f"[uyarı] {path} okunamadı: {e}", file=sys.stderr)
            continue
        for k, v in data.items():
            if k in names:
                _set(cfg, k, v, path)
    for env, attr in ENV_MAP.items():
        val = os.environ.get(env, "").strip()
        if val and not (val.startswith("%") and val.endswith("%")):  # boş/çözülmemiş %DEGISKEN% değil
            _set(cfg, attr, val, env)
    return normalize(cfg)


def _set(cfg: Config, name: str, value, where) -> None:
    """Hatalı tek değer (ör. port = "abc") uygulamayı açılmaz hâle getirmesin: varsayılan kalır."""
    try:
        setattr(cfg, name, _coerce(name, value))
    except (TypeError, ValueError) as e:
        print(f"[uyarı] {where}: {name} = {value!r} geçersiz, varsayılan kullanılıyor ({e})", file=sys.stderr)


def _num(v, default: float, lo: float, hi: float | None = None) -> float:
    """Sayısal ayarı sınırla: NaN/sonsuz/geçersiz -> varsayılan."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(max(v, lo), hi) if hi is not None else max(v, lo)


def normalize(cfg: Config) -> Config:
    """Uygulama/CLI ortak: 'none', 'auto' gibi değerleri None'a çevir."""
    if isinstance(cfg.cookies_browser, str) and cfg.cookies_browser.strip().lower() in ("", "none", "yok", "off", "kullanma"):
        cfg.cookies_browser = None
    if isinstance(cfg.lang, str) and cfg.lang.strip().lower() in ("", "auto", "otomatik", "oto"):
        cfg.lang = None
    if isinstance(cfg.llm_provider, str):
        v = cfg.llm_provider.strip().lower()
        v = {"google": "gemini", "anthropic": "claude", "gpt": "openai"}.get(v, v)
        cfg.llm_provider = v if v in ("gemini", "claude", "openai") else None
    if not (isinstance(cfg.api_port, int) and 0 <= cfg.api_port <= 65535):
        cfg.api_port = 0
    # 0 / NaN takip aralığı: kanallar 10 dakikada bir taranır (YouTube engeller) ya da hiç taranmaz
    cfg.follow_interval_h = _num(cfg.follow_interval_h, 6.0, 1.0)
    cfg.chunk_min = _num(cfg.chunk_min, 5.0, 0.5)
    cfg.split_min = _num(cfg.split_min, 8.0, 0.5)
    cfg.live_chunk_s = _num(cfg.live_chunk_s, 20.0, 5.0)
    cfg.llm_budget_usd = _num(cfg.llm_budget_usd, 5.0, 0.0)
    cfg.helper_rest_s = _num(cfg.helper_rest_s, 0.0, 0.0, 600.0)
    cfg.helper_reserve_gb = _num(cfg.helper_reserve_gb, 2.0, 0.5, 16.0)
    cfg.gpu_power_pct = int(_num(cfg.gpu_power_pct, 70, 50, 100))
    cfg.night_temp = int(_num(cfg.night_temp, 72, 60, 83))
    cfg.night_recycle_min = _num(cfg.night_recycle_min, 90.0, 0.0, 1440.0)
    if 0 < cfg.night_recycle_min < 20:
        cfg.night_recycle_min = 20.0  # model yüklemesi ~30 sn: daha sık yenileme işi yavaşlatır
    cfg.recycle_min = _num(cfg.recycle_min, 180.0, 0.0, 1440.0)
    if 0 < cfg.recycle_min < 20:
        cfg.recycle_min = 20.0
    cfg.min_free_gb = _num(cfg.min_free_gb, 3.0, 0.0, 500.0)
    clk = int(_num(cfg.gpu_max_clock, 1600, 0, 3000))
    cfg.gpu_max_clock = 0 if clk <= 0 else max(900, clk)
    for k in ("gemini_api_key", "anthropic_api_key", "openai_api_key", "hf_token", "api_token", "webhook_url",
              "llm_model"):
        v = getattr(cfg, k)
        if isinstance(v, str) and not v.strip():
            setattr(cfg, k, None)
    return cfg
