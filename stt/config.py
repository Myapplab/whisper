"""Ayarlar: varsayılanlar < stt.toml < ortam değişkenleri < komut satırı."""
from __future__ import annotations

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
    gpus: str = "auto"                   # "auto" (monitörsüz kart), "all" (hepsi), "0,1", "0", "cpu"
    display_batch: int = 4               # monitörün bağlı olduğu kart kullanılırsa en fazla bu toplu boyut
    low_priority: bool = True            # tanıma süreçleri düşük öncelikte (bilgisayar donmasın)
    isolate_gpu: bool = True             # her işçi yalnızca kendi kartını görür (monitörlü kartta CUDA bağlamı yok)
    ui_gpu: bool = False                 # arayüz (WebView2) ekran kartı hızlandırması; kapalı = ekran kararmasın
    prompt: str | None = None
    condition_on_previous_text: bool = True
    agc: bool = True                     # otomatik kazanç: uzaktaki muhabir sesini yükselt (tanıma + ayırma)
    vad_threshold: float = 0.4           # Whisper VAD eşiği (düşük = sessiz konuşmayı da yakalar)
    # bellek sınırları (yüzde). Ekran kartı: konuşmacı ayırma (torch) bu oranı aşmaz.
    gpu_limit_pct: int = 90              # monitörsüz kart (ör. RTX 3090)
    display_gpu_limit_pct: int = 70      # monitörlü kart (ör. RTX 3080)
    ram_limit_pct: int = 70              # boş RAM'in en çok bu kadarı; yetmezse bekler, sonra tekrar dener
    diar_batch: int = 8                  # konuşmacı ayırma toplu boyutu (küçük = daha az bellek, biraz yavaş)
    fill_gaps: bool = True               # metinde uzun boşluk kalırsa (müzik altında konuşma) o kısım VAD'siz tekrar
    worker_restarts: int = 6             # art arda en fazla bu kadar yeniden başlatma (başarılı görev sayacı sıfırlar)
    # bölme
    split_min: float = 8.0               # bundan uzun dosyalar VAD ile parçalanır (dakika)
    chunk_min: float = 5.0               # hedef parça uzunluğu (dakika)
    # konuşmacı ayırma
    diarize: bool = False
    hf_token: str | None = None
    diar_model: str = "pyannote/speaker-diarization-3.1"
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
    "STT_API_PORT": "api_port",
    "STT_LLM": "llm_provider",
}


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
                setattr(cfg, k, _coerce(k, v))
    for env, attr in ENV_MAP.items():
        val = os.environ.get(env, "").strip()
        if val and not (val.startswith("%") and val.endswith("%")):  # boş/çözülmemiş %DEGISKEN% değil
            setattr(cfg, attr, _coerce(attr, val))
    return normalize(cfg)


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
    for k in ("gemini_api_key", "anthropic_api_key", "hf_token", "api_token", "webhook_url", "llm_model"):
        v = getattr(cfg, k)
        if isinstance(v, str) and not v.strip():
            setattr(cfg, k, None)
    return cfg
