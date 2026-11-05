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
    prompt: str | None = None
    condition_on_previous_text: bool = True
    # bölme
    split_min: float = 8.0               # bundan uzun dosyalar VAD ile parçalanır (dakika)
    chunk_min: float = 5.0               # hedef parça uzunluğu (dakika)
    # konuşmacı ayırma
    diarize: bool = False
    hf_token: str | None = None
    diar_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    td_label: str = "Teknik direktör"
    reporter_label: str = "Muhabir"
    # indirme
    cookies_browser: str | None = "chrome"
    cookies_file: str | None = None
    prep_workers: int = 2
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
    return cfg
