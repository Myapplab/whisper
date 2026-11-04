"""Tanıma motorları (faster-whisper, OpenAI) ve konuşmacı ayırma (pyannote)."""
from __future__ import annotations

import io
import os
import time
import wave

import numpy as np

from .audio import SR, load_wav
from .util import log, setup_cuda_dlls


def parse_device(dev: str) -> tuple[str, int]:
    if dev.startswith("cuda"):
        return "cuda", int(dev.split(":")[1]) if ":" in dev else 0
    return "cpu", 0


def resolve_model_path(model: str) -> str:
    """Model yerel klasörse onu, değilse HF önbelleğine indirip yolunu döner (ana süreçte bir kez)."""
    if os.path.isdir(model):
        return model
    from faster_whisper.utils import download_model
    return download_model(model)


def auto_batch_size(kind: str, vram_gb: float | None) -> int:
    """Kart belleğine göre toplu boyut: 24 GB -> 16, 10-12 GB -> 8, bilinmiyor -> 8, CPU -> 1 (kapalı)."""
    if kind != "cuda":
        return 1
    if not vram_gb:
        return 8
    return 16 if vram_gb >= 20 else 8 if vram_gb >= 9 else 4


def _smi(query: str) -> list[list[str]]:
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
    try:
        r = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, **kw)
    except Exception:  # noqa: BLE001
        return []
    if r.returncode != 0:
        return []
    return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def gpu_info() -> dict[int, dict]:
    """nvidia-smi: {index: {name, gb, display}}. CUDA_DEVICE_ORDER=PCI_BUS_ID ile CUDA sırasıyla aynıdır.

    display: True = bu karta monitör bağlı (masaüstü bu kartla çiziliyor), False = boş kart, None = bilinmiyor.
    """
    out: dict[int, dict] = {}
    for row in _smi("index,name,memory.total"):
        try:
            i, name, mem = row[0], ",".join(row[1:-1]), row[-1]
            out[int(i)] = {"name": name.replace("NVIDIA GeForce ", ""), "gb": float(mem) / 1024, "display": None}
        except (ValueError, IndexError):
            continue
    for row in _smi("index,display_active"):  # ayrı sorgu: alan desteklenmezse ilk sorgu bozulmasın
        try:
            v = row[1].lower()
            if int(row[0]) in out and v in ("enabled", "disabled"):
                out[int(row[0])]["display"] = v == "enabled"
        except (ValueError, IndexError):
            continue
    return out


def gpu_table() -> dict[int, tuple[str, float]]:
    """Geriye uyumluluk: {index: (ad, GB)}."""
    return {i: (g["name"], g["gb"]) for i, g in gpu_info().items()}


def choose_gpus(n: int, info: dict[int, dict], mode: str = "auto") -> list[int]:
    """Kullanılacak CUDA kartları.

    auto: monitörün bağlı olduğu kart boş bırakılır (bilgisayar, YouTube, tarayıcı akıcı kalsın).
          Bütün kartlarda monitör varsa ya da tek kart varsa: belleği en büyük TEK kart.
    all : bütün kartlar (en hızlı; ekran kartı tam yükte kalır).
    """
    idx = list(range(n))
    if n <= 1 or mode == "all":
        return idx
    known = [i for i in idx if i in info and info[i].get("display") is not None]
    if len(known) < n:  # bilinmiyor: en büyük kartı al (ötekini masaüstüne bırak)
        return [max(idx, key=lambda i: (info.get(i, {}).get("gb", 0), -i))]
    free = [i for i in idx if not info[i]["display"]]
    if free:
        return free
    return [max(idx, key=lambda i: (info[i].get("gb", 0), -i))]


class WhisperEngine:
    def __init__(self, model_path: str, device: str, compute_type: str, beam_size: int,
                 condition_on_previous_text: bool = True, batch_size: int = -1, vram_gb: float | None = None):
        setup_cuda_dlls()
        kind, idx = parse_device(device)
        from faster_whisper import WhisperModel

        if kind == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"
        self.device = device
        self.beam_size = beam_size
        self.cond = condition_on_previous_text
        t = time.time()
        kw = dict(device=kind, device_index=idx, compute_type=compute_type,
                  cpu_threads=max(2, (os.cpu_count() or 4) - 2) if kind == "cpu" else 2)  # 2 çekirdek boşta kalsın
        try:
            self.model = WhisperModel(model_path, **kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if kind != "cuda" or not any(x in msg for x in ("dll", "cublas", "cudnn", "library", "symbol")):
                raise
            import torch  # noqa: F401  (Windows: torch'un cuBLAS/cuDNN DLL'lerini yükle, tekrar dene)
            self.model = WhisperModel(model_path, **kw)
        self.batch_size = auto_batch_size(kind, vram_gb) if batch_size < 0 else batch_size
        self.batched = None
        if self.batch_size > 1:
            from faster_whisper import BatchedInferencePipeline
            self.batched = BatchedInferencePipeline(model=self.model)
        log.info("[%s] model yüklendi (%s, %s, %s) %.1fs", device, os.path.basename(model_path.rstrip("/\\")),
                 compute_type, f"toplu={self.batch_size}" if self.batched else "tekli", time.time() - t)

    def transcribe(self, wav: str, start: float, end: float, lang: str | None, prompt: str | None) -> dict:
        audio = load_wav(wav, start, end)
        while True:
            try:
                return self._run(audio, start, lang, prompt)
            except RuntimeError as e:  # bellek yetmezse toplu boyutu yarıya indir
                if "out of memory" not in str(e).lower() or self.batched is None:
                    raise
                self.batch_size //= 2
                if self.batch_size <= 1:
                    self.batched = None
                log.warning("[%s] GPU belleği yetmedi; toplu boyut -> %s", self.device, self.batch_size)

    def _run(self, audio, start: float, lang: str | None, prompt: str | None) -> dict:
        if self.batched is not None:
            segs, info = self.batched.transcribe(
                audio, language=lang, beam_size=self.beam_size, vad_filter=True, word_timestamps=True,
                initial_prompt=prompt, batch_size=self.batch_size,
                vad_parameters={"min_silence_duration_ms": 500},
            )
        else:
            segs, info = self.model.transcribe(
                audio, language=lang, beam_size=self.beam_size, vad_filter=True, word_timestamps=True,
                initial_prompt=prompt, condition_on_previous_text=self.cond,
                vad_parameters={"min_silence_duration_ms": 500},
            )
        out = []
        for s in segs:
            words = [{"start": round(w.start + start, 3), "end": round(w.end + start, 3), "word": w.word,
                      "prob": round(w.probability, 3)} for w in (s.words or [])]
            out.append({"start": round(s.start + start, 3), "end": round(s.end + start, 3), "text": s.text.strip(),
                        "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                        "words": words})
        return {"segments": out, "language": info.language, "language_probability": round(info.language_probability, 3)}


class OpenAIEngine:
    """Yedek motor. OPENAI_API_KEY ortam değişkeninden okunur."""

    def __init__(self, model: str):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY ortam değişkeni tanımlı değil")
        self.client = OpenAI(max_retries=6, timeout=600)
        self.model = model
        self.verbose_ok: bool | None = None if model != "whisper-1" else True
        self.device = "openai"

    @staticmethod
    def _wav_bytes(audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue()

    def transcribe(self, wav: str, start: float, end: float, lang: str | None, prompt: str | None,
                   speech_start: float | None = None, speech_end: float | None = None) -> dict:
        audio = load_wav(wav, start, end)
        data = self._wav_bytes(audio)
        kw = {"model": self.model, "file": ("parca.wav", data, "audio/wav")}
        if lang:
            kw["language"] = lang
        if prompt:
            kw["prompt"] = prompt
        res = None
        if self.verbose_ok is not False:
            try:
                res = self.client.audio.transcriptions.create(
                    response_format="verbose_json", timestamp_granularities=["word", "segment"], **kw)
                self.verbose_ok = True
            except Exception as e:  # noqa: BLE001
                if getattr(e, "status_code", None) == 400 or "response_format" in str(e):
                    self.verbose_ok = False
                else:
                    raise
        if res is None:
            res = self.client.audio.transcriptions.create(response_format="json", **kw)
        segs = []
        rsegs = getattr(res, "segments", None)
        rwords = getattr(res, "words", None) or []
        if rsegs:
            words = [{"start": round(w.start + start, 3), "end": round(w.end + start, 3),
                      "word": " " + w.word.strip(), "prob": 1.0} for w in rwords]
            for s in rsegs:
                ss, se = s.start + start, s.end + start
                segs.append({"start": round(ss, 3), "end": round(se, 3), "text": s.text.strip(),
                             "words": [w for w in words if w["start"] >= ss - 0.01 and w["end"] <= se + 0.01]})
        else:
            text = (getattr(res, "text", "") or "").strip()
            if text:
                ss = speech_start if speech_start is not None else start
                se = speech_end if speech_end is not None else end
                segs.append({"start": round(ss, 3), "end": round(se, 3), "text": text, "words": []})
        return {"segments": segs, "language": getattr(res, "language", None) or lang, "language_probability": None}


class Diarizer:
    def __init__(self, model: str, token: str | None, device: str):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # pyannote 3.x + torch>=2.6
        setup_cuda_dlls()
        import torch
        from pyannote.audio import Pipeline

        if not token:
            raise RuntimeError("Konuşmacı ayırma için HF token gerekli (--hf-token ya da HF_TOKEN)")
        try:
            pipe = Pipeline.from_pretrained(model, use_auth_token=token)       # pyannote 3.x
        except TypeError as e:
            if "hf_hub_download" in str(e):
                raise RuntimeError("pyannote 3.x için huggingface_hub<1.0 gerekli: "
                                   'pip install "huggingface_hub<1.0"') from None
            pipe = Pipeline.from_pretrained(model, token=token)                # pyannote 4.x
        if pipe is None:
            raise RuntimeError(f"{model} yüklenemedi: HF'de model sayfasındaki koşulları kabul ettiniz mi? "
                               "(pyannote/speaker-diarization-3.1 ve pyannote/segmentation-3.0)")
        kind, idx = parse_device(device)
        if kind == "cuda" and torch.cuda.is_available():
            pipe.to(torch.device(f"cuda:{idx}"))
        self.pipe = pipe
        self.torch = torch
        self.device = device

    def run(self, wav: str, min_speakers=None, max_speakers=None) -> list[list]:
        audio = load_wav(wav)
        t = time.time()
        kw = {}
        if min_speakers:
            kw["min_speakers"] = min_speakers
        if max_speakers:
            kw["max_speakers"] = max_speakers
        out = self.pipe({"waveform": self.torch.from_numpy(audio)[None], "sample_rate": SR}, **kw)
        ann = getattr(out, "speaker_diarization", out)  # pyannote 4.x DiarizeOutput
        turns = [[round(seg.start, 3), round(seg.end, 3), str(spk)] for seg, _, spk in ann.itertracks(yield_label=True)]
        log.info("[%s] konuşmacı ayırma %.1fs, %d konuşmacı", self.device, time.time() - t, len({x[2] for x in turns}))
        return turns
