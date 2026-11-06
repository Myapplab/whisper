"""Tanıma motorları (faster-whisper, OpenAI) ve konuşmacı ayırma (pyannote)."""
from __future__ import annotations

import io
import os
import time
import wave

import numpy as np

from .audio import SR, load_wav
from .util import log, popen_kw, setup_cuda_dlls


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
    try:
        r = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, **popen_kw())
    except Exception:  # noqa: BLE001
        return []
    if r.returncode != 0:
        return []
    return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def gpu_load(idx: int) -> dict | None:
    """Kartın anlık yükü (nvidia-smi): {"util": %, "used_mb", "total_mb"}; okunamazsa None.
    utilization.gpu son ~1 sn'nin ortalamasıdır: yardımcı kart dinlenirken ölçülürse yalnız BAŞKA programları gösterir."""
    for row in _smi("index,utilization.gpu,memory.used,memory.total"):
        try:
            if int(row[0]) != idx:
                continue
            return {"util": int(float(row[1])), "used_mb": float(row[2]), "total_mb": float(row[3])}
        except (ValueError, IndexError):  # "[N/A]" (bazı sürücüler) ya da bozuk satır
            return None
    return None


def windows_display_adapters() -> list[str] | None:
    """Windows: masaüstünü çizen (monitör bağlı) ekran kartlarının adları. Windows değilse None.

    nvidia-smi'nin display_active alanı bazı sürücülerde monitörlü kartta da 'Disabled' diyor; o zaman iki kart
    da 'boş' sanılıp monitörlü kart (3080) tam yükte kullanılıyor, ekran kararıp geri geliyordu (TDR).
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DD(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                        ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                        ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]
        out, i = [], 0
        enum = ctypes.windll.user32.EnumDisplayDevicesW  # type: ignore[attr-defined]
        while i < 64:
            d = DD()
            d.cb = ctypes.sizeof(DD)
            if not enum(None, i, ctypes.byref(d), 0):
                break
            if d.StateFlags & 0x1:  # DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
                out.append(d.DeviceString)
            i += 1
        return out
    except Exception:  # noqa: BLE001
        return None


def _norm_gpu(name: str) -> str:
    n = name.lower()
    for p in ("nvidia ", "geforce ", "quadro "):
        n = n.replace(p, "")
    return " ".join(n.split())


def mark_display(out: dict[int, dict], adapters: list[str] | None) -> None:
    """Windows'un 'masaüstüne bağlı' listesiyle monitörlü kartları işaretle (adlar ayırt edilebiliyorsa)."""
    if adapters is None or not out:
        return
    names = [_norm_gpu(g["name"]) for g in out.values()]
    if len(set(names)) != len(names):  # iki aynı model kart: adla ayırt edilemez, nvidia-smi'ye güven
        return
    att = [_norm_gpu(a) for a in adapters]
    for g in out.values():
        n = _norm_gpu(g["name"])
        hit = n in att  # tam ad eşleşmesi ("rtx 3080" ile "rtx 3080 ti" karışmasın)
        if hit:
            g["display"] = True  # nvidia-smi 'Disabled' dese bile: Windows bu kartla çiziyor
        elif g.get("display") is None and att:
            g["display"] = False  # masaüstü listesinde yok (monitör başka kartta / iGPU'da)
        # nvidia-smi 'Enabled' dediyse koru (yanlışlıkla korumak yalnızca biraz hız kaybettirir)


_GPU_CACHE: tuple[float, dict] | None = None


def gpu_info(max_age: float = 60.0) -> dict[int, dict]:
    """Önbellekli (nvidia-smi her açılışta 4-5 kez çağrılıyordu)."""
    global _GPU_CACHE
    if _GPU_CACHE and time.time() - _GPU_CACHE[0] < max_age:
        return {i: dict(g) for i, g in _GPU_CACHE[1].items()}
    out = _gpu_info()
    _GPU_CACHE = (time.time(), out)
    return {i: dict(g) for i, g in out.items()}


def _gpu_info() -> dict[int, dict]:
    """nvidia-smi: {index: {name, gb, display}}. CUDA_DEVICE_ORDER=PCI_BUS_ID ile CUDA sırasıyla aynıdır.

    display: True = bu karta monitör bağlı (masaüstü bu kartla çiziliyor), False = boş kart, None = bilinmiyor.
    Kaynaklar: nvidia-smi display_attached / display_mode / display_active + Windows EnumDisplayDevices.
    """
    out: dict[int, dict] = {}
    for row in _smi("index,name,memory.total"):
        try:
            i, name, mem = row[0], ",".join(row[1:-1]), row[-1]
            out[int(i)] = {"name": name.replace("NVIDIA GeForce ", ""), "gb": float(mem) / 1024, "display": None}
        except (ValueError, IndexError):
            continue
    # önce tek sorgu; bir alan desteklenmiyorsa (sürücüye göre değişiyor) alan alan
    fields = ("display_mode", "display_active", "display_attached")
    rows = _smi("index," + ",".join(fields[:2])) or []
    if not rows:
        rows = [r for f in fields for r in _smi(f"index,{f}")]
    else:
        rows += _smi("index,display_attached")
    for row in rows:
        try:
            i = int(row[0])
        except (ValueError, IndexError):
            continue
        for v in (x.strip().lower() for x in row[1:]):
            if i not in out:
                continue
            if v in ("enabled", "yes", "true", "1"):
                out[i]["display"] = True
            elif v in ("disabled", "no", "false", "0") and out[i]["display"] is None:
                out[i]["display"] = False
    mark_display(out, windows_display_adapters())
    return out


def cuda_device_count() -> int:
    """CUDA kart sayısı. Önce nvidia-smi (ana süreçte CUDA başlatmamak için: monitörlü kartta bağlam açmak
    ekranın kararmasına yol açabiliyor), yoksa CTranslate2."""
    gi = gpu_info()
    if gi:
        return len(gi)
    try:
        setup_cuda_dlls()
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        return 0


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


# Otomatik dil algılamada izin verilen diller. Whisper İngiliz aksanını sık sık Galce (cy) sanıyor ve
# kendinden çok emin olduğu için metnin tamamı Galce "çeviri" olarak çıkıyordu.
DEFAULT_LANGS = ("en,tr,az,es,pt,fr,de,it,nl,ar,ru,uk,pl,hr,sr,bs,sl,da,sv,no,fi,el,cs,sk,ro,hu,bg,ja,ko,zh,"
                 "fa,he,ka")


def pick_language(probs: list[dict[str, float]], allowed: set[str]) -> tuple[str | None, float]:
    """Birkaç pencerenin dil olasılıklarını topla, izin verilen diller arasından en olasısını seç."""
    tot: dict[str, float] = {}
    for pr in probs:
        for lang, p in pr.items():
            tot[lang] = tot.get(lang, 0.0) + p
    cands = {k: v for k, v in tot.items() if k in allowed} or tot
    if not cands:
        return None, 0.0
    best = max(cands, key=cands.get)
    return best, cands[best] / max(1, len(probs))


GAP_S = 12.0  # metinde bundan uzun boşluk = VAD konuşmayı kaçırmış olabilir
_JUNK = ("altyazı", "subtitle", "abone ol", "izlediğiniz için", "thanks for watching", "♪", "[müzik]", "[music]",
         "müzik]", "music]")


def odd_cuda_error(err: str) -> bool:
    """CTranslate2'nin içeriğe bağlı CUDA hatası (bellek değil, kart çökmesi değil)."""
    e = err.lower()
    return "parallel_for failed" in e or "invalid device ordinal" in e or "cudaerrorinvaliddevice" in e


def seg_dict(s, off: float) -> dict:
    words = [{"start": round(w.start + off, 3), "end": round(w.end + off, 3), "word": w.word,
              "prob": round(w.probability, 3)} for w in (s.words or [])]
    return {"start": round(s.start + off, 3), "end": round(s.end + off, 3), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3), "words": words}


def find_gaps(spans: list[tuple[float, float]], dur: float, min_gap: float) -> list[tuple[float, float]]:
    """Metin olmayan, ``min_gap`` sn'den uzun aralıklar (parça içi saniye)."""
    gaps, prev = [], 0.0
    for a, b in sorted(spans):
        if a - prev >= min_gap:
            gaps.append((prev, a))
        prev = max(prev, b)
    if dur - prev >= min_gap:
        gaps.append((prev, dur))
    return gaps


def rms_db(x) -> float:
    if not len(x):
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9))


def keep_gap_segment(text: str, avg_logprob: float, no_speech_prob: float, compression_ratio: float) -> bool:
    """VAD'siz çeviride müzik/gürültüden uydurulan satırları ele."""
    t = (text or "").strip().lower()
    if len(t) < 2 or any(j in t for j in _JUNK):
        return False
    if avg_logprob < -1.0 or compression_ratio > 2.4:
        return False
    return not (no_speech_prob > 0.6 and avg_logprob < -0.5)


class WhisperEngine:
    def __init__(self, model_path: str, device: str, compute_type: str, beam_size: int,
                 condition_on_previous_text: bool = True, batch_size: int = -1, vram_gb: float | None = None,
                 languages: str | None = None, label: str | None = None, agc: bool = True,
                 vad_threshold: float = 0.4, fill_gaps: bool = True):
        setup_cuda_dlls()
        kind, idx = parse_device(device)
        from faster_whisper import WhisperModel

        if kind == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"
        self.device = label or device  # günlükte gerçek kart adı (işçi yalnızca kendi kartını görür: cuda:0)
        self.agc = agc
        self.vad = {"min_silence_duration_ms": 500, "threshold": vad_threshold}
        self.fill_gaps = fill_gaps
        self.beam_size = beam_size
        self.cond = condition_on_previous_text
        self.allowed = {x.strip() for x in (languages or DEFAULT_LANGS).split(",") if x.strip()}
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

    def detect_language(self, audio) -> str | None:
        """Konuşmanın başından, ortasından ve sonundan 30 sn'lik pencerelerle dil algıla (izinli diller)."""
        from .audio import speech_segments
        try:
            segs = speech_segments(audio)
            speech = np.concatenate([audio[int(a * SR):int(b * SR)] for a, b in segs]) if segs else audio
        except Exception:  # noqa: BLE001
            speech = audio
        win = 30 * SR
        if len(speech) < SR:
            return None
        starts = sorted({0, max(0, len(speech) // 2 - win // 2), max(0, len(speech) - win)})
        probs = []
        for st in starts[:3]:
            try:
                _, _, allp = self.model.detect_language(speech[st:st + win], language_detection_segments=1)
                probs.append(dict(allp))
            except Exception:  # noqa: BLE001  (eski faster-whisper: algılamayı modele bırak)
                return None
        lang, p = pick_language(probs, self.allowed)
        top = max(probs[0], key=probs[0].get) if probs and probs[0] else None
        if top and top != lang:
            log.info("[%s] dil: Whisper '%s' dedi, '%s' seçildi (%.2f)", self.device, top, lang, p)
        return lang

    def transcribe(self, wav: str, start: float, end: float, lang: str | None, prompt: str | None) -> dict:
        audio = load_wav(wav, start, end)
        if self.agc:
            from .audio import agc
            agc(audio)
        if not lang:
            lang = self.detect_language(audio)
        while True:
            try:
                res = self._run(audio, start, lang, prompt)
                break
            except RuntimeError as e:  # bellek yetmezse toplu boyutu yarıya indir
                if "out of memory" not in str(e).lower() and odd_cuda_error(str(e)):
                    res = self._fallback(audio, start, lang, prompt, e)
                    break
                if "out of memory" not in str(e).lower() or self.batched is None:
                    raise
                self.batch_size //= 2
                if self.batch_size <= 1:
                    self.batched = None
                log.warning("[%s] GPU belleği yetmedi; toplu boyut -> %s", self.device, self.batch_size)
        if self.fill_gaps:
            try:
                self._fill_gaps(audio, start, lang or res.get("language"), res)
            except Exception:  # noqa: BLE001  (boşluk doldurma ek iş; hata asıl metni bozmasın)
                log.warning("[%s] boşluk doldurma yapılamadı", self.device, exc_info=True)
        return res

    def _fill_gaps(self, audio, start: float, lang: str | None, res: dict, min_gap: float = GAP_S) -> None:
        """VAD'in müzik/gürültü altında kaçırdığı konuşmayı geri al (filmlerde metnin %30'u eksik kalıyordu).

        Metinde ``min_gap`` sn'den uzun boşluk varsa ve orada ses varsa, o kısım VAD'siz tekrar çevrilir;
        düşük güvenli / 'konuşma yok' / tekrarlı satırlar (müzikte uydurma) atılır."""
        segs = res["segments"]
        gaps = find_gaps([(x["start"] - start, x["end"] - start) for x in segs], len(audio) / SR, min_gap)
        added = []
        for a, b in gaps:
            piece = audio[int(a * SR):int(b * SR)]
            if rms_db(piece) < -50:  # gerçekten sessiz
                continue
            out, _ = self.model.transcribe(piece, language=lang, beam_size=self.beam_size, vad_filter=False,
                                           word_timestamps=True, condition_on_previous_text=False)
            for s in out:
                if keep_gap_segment(s.text, s.avg_logprob, s.no_speech_prob, s.compression_ratio):
                    added.append(seg_dict(s, start + a))
        if added:
            log.info("[%s] metindeki %d boşluktan %d satır geri alındı", self.device, len(gaps), len(added))
            res["segments"] = sorted(segs + added, key=lambda x: x["start"])

    def _fallback(self, audio, start: float, lang: str | None, prompt: str | None, err: Exception) -> dict:
        """CTranslate2 bazı parçalarda hep aynı yerde 'parallel_for failed: invalid device ordinal' veriyor
        (içeriğe bağlı; süreci yeniden açmak düzeltmiyor). Parçayı başka yollarla dene:
        1) toplu olmayan çeviri, 2) kelime zamanları olmadan (konuşmacı ataması satır düzeyinde kalır)."""
        log.warning("[%s] çeviri hatası (%s); parça toplu olmadan tekrar deneniyor", self.device, str(err)[:120])
        try:
            return self._run(audio, start, lang, prompt, batched=False)
        except RuntimeError as e2:
            if not odd_cuda_error(str(e2)):
                raise
        log.warning("[%s] yine olmadı; parça kelime zamanları olmadan çevriliyor", self.device)
        res = self._run(audio, start, lang, prompt, batched=False, words=False)
        res["no_words"] = True
        return res

    def _run(self, audio, start: float, lang: str | None, prompt: str | None, batched: bool = True,
             words: bool = True) -> dict:
        if self.batched is not None and batched:
            segs, info = self.batched.transcribe(
                audio, language=lang, beam_size=self.beam_size, vad_filter=True, word_timestamps=True,
                initial_prompt=prompt, batch_size=self.batch_size, vad_parameters=self.vad,
            )
        else:
            segs, info = self.model.transcribe(
                audio, language=lang, beam_size=self.beam_size, vad_filter=True, word_timestamps=words,
                initial_prompt=prompt, condition_on_previous_text=self.cond, vad_parameters=self.vad,
            )
        out = [seg_dict(s, start) for s in segs]
        return {"segments": out, "language": info.language, "language_probability": round(info.language_probability, 3)}


class OpenAIEngine:
    """Yedek motor. Anahtar: Ayarlar'daki OpenAI anahtarı (stt.toml ``openai_api_key``) ya da OPENAI_API_KEY."""

    def __init__(self, model: str, api_key: str | None = None):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI API anahtarı yok (Ayarlar > Yapay zekâ > OpenAI ya da OPENAI_API_KEY)")
        self.client = OpenAI(api_key=key, max_retries=6, timeout=600)
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


FALLBACK_DIAR_MODEL = "pyannote/speaker-diarization-3.1"


def chunk_spans(n: int, seg: int, ov: int):
    """Uzun sesi parça parça işlemek için: (a, b, c, d) = işlenen pencere [a, b), saklanan kısım [c, d).
    Her parçanın iki yanında ``ov`` örnek bağlam bırakılır; kenarlardaki bozulma atılır."""
    seg = max(1, int(seg))
    for c in range(0, max(0, n), seg):
        d = min(n, c + seg)
        yield max(0, c - ov), min(n, d + ov), c, d


def speaker_embeddings(labels, cents) -> dict | None:
    """pyannote'un konuşmacı başına ortalama ses imzası (labels() sırasıyla) -> {SPEAKER_xx: [..]}.
    Sıfır / NaN satırlar (kümeye düşmeyen konuşmacı) atlanır."""
    if cents is None:
        return None
    import numpy as np
    arr = np.asarray(cents, dtype=float)
    out = {}
    for lbl, row in zip(list(labels), arr):
        if row.ndim == 1 and np.all(np.isfinite(row)) and float(np.linalg.norm(row)) > 1e-6:
            out[str(lbl)] = [round(float(v), 5) for v in row]
    return out or None


class Diarizer:
    agc = True  # testler __new__ ile kurar
    batch = 8
    ram_pct = 70
    cpu_ok = False  # işlemciye (CPU) geçilmez: kullanıcıda fan çok dönüyor, i3'te çok yavaş
    v4 = False           # pyannote.audio 4.x (community-1 modeli, örtüşmesiz çıktı)
    model_used = None    # son ayırmayı yapan model (pyannote'ta yeni model olmazsa eskisi; DiariZen'de onun adı)
    pyannote_model = None  # yüklenen pyannote modeli (DiariZen seçiliyse yalnız yedek gerekince yüklenir)
    pipe = None
    token = None
    _cuda = None
    last_emb = None      # son ayırmadaki konuşmacı ses imzaları
    _sep = None          # müzik ayıklama modeli (HDemucs), ilk gerekince yüklenir
    _beat_fn = None      # "hâlâ çalışıyorum" sinyali (işçi -> ana süreç; takılma denetimi)
    def __init__(self, model: str, token: str | None, device: str, agc: bool = True, batch: int = 8,
                 gpu_frac: float | None = None, ram_pct: int = 70, cpu_ok: bool = False, params: dict | None = None,
                 engine: str | None = None):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # pyannote 3.x + torch>=2.6
        setup_cuda_dlls()
        import torch

        self.engine = (engine or "pyannote").lower()  # "diarizen": ayrı ortamda DiariZen (stt/diarizen.py)
        if not token and self.engine != "diarizen":  # DiariZen modelleri token istemez; pyannote yalnız yedek
            raise RuntimeError("Konuşmacı ayırma için HF token gerekli (--hf-token ya da HF_TOKEN)")
        self.torch = torch
        self.model, self.token = model, token
        self.agc = agc
        self.tuned = params or {}  # eğitilmiş ayarlar (stt/tune.py): {model: {"params": {...}}}
        self.batch, self.ram_pct, self.cpu_ok = max(1, int(batch or 8)), ram_pct, cpu_ok
        torch.backends.cudnn.benchmark = False
        try:  # ekran kartında da torch işlemcide iş parçacıkları açar; 2 ile sınırla (fan / donma)
            torch.set_num_threads(2)
        except Exception:  # noqa: BLE001
            pass
        kind, idx = parse_device(device)
        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Konuşmacı ayırma ekran kartını göremedi (torch CUDA yok); işlemcide yapılmıyor. "
                               "NVIDIA sürücüsünü güncelleyin ya da kurulumu tekrar çalıştırın.")
        if kind != "cuda" and not cpu_ok:
            raise RuntimeError("Konuşmacı ayırma için uygun ekran kartı yok; işlemcide (CPU) yapılmıyor.")
        self.pipe = None
        self.device = "cpu"
        self._cuda = f"cuda:{idx}" if kind == "cuda" else None
        if kind == "cuda":
            if gpu_frac:  # ekran kartı belleğinin en çok bu oranı (ör. 3090 %90, monitörlü 3080 %70)
                try:
                    torch.cuda.set_per_process_memory_fraction(min(0.95, max(0.2, gpu_frac)), idx)
                    log.info("[%s] konuşmacı ayırma ekran kartı belleği sınırı: %%%d", device, round(gpu_frac * 100))
                except Exception:  # noqa: BLE001
                    log.debug("bellek sınırı konamadı", exc_info=True)
            self.device = device
        if self.engine != "diarizen":
            # DiariZen'de pyannote yalnız DiariZen hata verirse yüklenir (eskiden hep yükleniyor, ekran kartı
            # belleğini boşuna tutuyor ve HF token istiyordu)
            self._ensure_pipe()

    def _ensure_pipe(self):
        """pyannote'u (ilk gerekince) yükle; kart seçiliyse karta taşı. Döner: pipeline."""
        if self.pipe is None:
            if not self.token:
                raise RuntimeError("DiariZen kullanılamadı; yedek pyannote için HF token gerekli "
                                   "(Ayarlar > HuggingFace token)")
            pipe = self._load()
            if self._cuda and str(self.device).startswith("cuda"):
                pipe.to(self.torch.device(self._cuda))
            self.pipe = pipe
        return self.pipe

    def _load(self):
        try:
            pipe = self._load_model(self.model)
            self.model_used = self.model
        except Exception as e:  # noqa: BLE001
            if self.model == FALLBACK_DIAR_MODEL:
                raise
            log.warning("Konuşmacı modeli %s yüklenemedi (%s); %s kullanılıyor. Yeni model için: "
                        "huggingface.co/%s sayfasında 'Agree' deyin.", self.model, str(e)[:200],
                        FALLBACK_DIAR_MODEL, self.model)
            pipe = self._load_model(FALLBACK_DIAR_MODEL)
            self.model_used = FALLBACK_DIAR_MODEL
        self.pyannote_model = self.model_used
        log.info("Konuşmacı modeli: %s", self.model_used)
        return pipe

    def _load_model(self, model: str):
        import pyannote.audio
        from pyannote.audio import Pipeline
        try:
            self.v4 = int(str(pyannote.audio.__version__).split(".")[0]) >= 4
        except ValueError:
            self.v4 = False
        if "community" in model and not self.v4:
            raise RuntimeError(f"{model} için pyannote.audio 4 gerekli (kurulu: {pyannote.audio.__version__})")
        try:
            if self.v4:
                pipe = Pipeline.from_pretrained(model, token=self.token)                 # pyannote 4.x
            else:
                pipe = Pipeline.from_pretrained(model, use_auth_token=self.token)        # pyannote 3.x
        except TypeError as e:
            if "hf_hub_download" in str(e):
                raise RuntimeError("pyannote 3.x için huggingface_hub<1.0 gerekli: "
                                   'pip install "huggingface_hub<1.0"') from None
            pipe = Pipeline.from_pretrained(model, token=self.token)                     # pyannote 4.x
        if pipe is None:
            raise RuntimeError(f"{model} yüklenemedi: HF'de model sayfasındaki koşulları kabul ettiniz mi? "
                               f"(huggingface.co/{model})")
        self._set_batch(pipe, self.batch)
        flat = ((getattr(self, "tuned", None) or {}).get(model) or {}).get("params")
        if flat:
            try:
                from .tune import apply_params
                log.info("Eğitilmiş konuşmacı ayarı kullanılıyor: %s", apply_params(pipe, flat))
            except Exception as e:  # noqa: BLE001
                log.warning("Eğitilmiş konuşmacı ayarı uygulanamadı: %s", str(e)[:160])
        return pipe

    @staticmethod
    def _set_batch(pipe, n: int) -> None:
        """Küçük toplu boyut: tepe bellek (RAM + ekran kartı) düşer; uzun filmde de sığar."""
        for name in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(pipe, name):
                try:
                    setattr(pipe, name, n)
                except Exception:  # noqa: BLE001
                    pass

    def wait_memory(self, dur_s: float, max_wait: float = 600.0) -> None:
        """Boş RAM'in ``ram_pct``'si işe yetene kadar bekle (en çok ``max_wait`` sn): Whisper o sırada
        model yüklüyor / tarayıcı RAM'i doldurmuşsa ayırma hata vermesin, biraz geç başlasın."""
        from .util import memory_status
        need = 2.5 + 1.5 * dur_s / 3600  # GB (pyannote + ses + gömmeler; küçük toplu boyutla)
        t0 = time.time()
        while time.time() - t0 < max_wait:
            m = memory_status()
            if not m:
                return
            usable = min(m["ram_free"] * self.ram_pct / 100, m["commit_free"])
            if usable >= need:
                return
            if m["ram_free"] * self.ram_pct / 100 >= need and m["commit_free"] >= 1.0 and time.time() - t0 >= 60:
                # RAM yetiyor, yalnız sanal bellek (sayfa dosyası) dolu görünüyor: Windows çoğu zaman sayfa
                # dosyasını büyütür. 20 dk beklemek yerine dene; gerçekten yetmezse bellek hatası yolu var.
                log.warning("Sanal bellek az (%.1f GB) ama boş RAM yeterli (%.1f GB); konuşmacı ayırma deneniyor",
                            m["commit_free"], m["ram_free"])
                return
            self._beat()
            if time.time() - t0 < 1:
                log.warning("Konuşmacı ayırma için bellek bekleniyor: ~%.1f GB gerekli, kullanılabilir %.1f GB "
                            "(boş RAM %.1f GB, sanal bellek %.1f GB)", need, usable, m["ram_free"], m["commit_free"])
            time.sleep(10)
        log.warning("Bellek beklemesi bitti; konuşmacı ayırma yine de deneniyor")

    def _run(self, audio, kw) -> list[list]:
        file = {"waveform": self.torch.from_numpy(audio)[None], "sample_rate": SR}
        cents = None
        if self._beat_fn is not None:  # pyannote her adımda / toplu işte çağırır: ilerleme sinyali
            kw = {**kw, "hook": self._hook}
        if self.v4:
            out = self.pipe(file, **kw)
            full = getattr(out, "speaker_diarization", out)
            # pyannote 4.x DiarizeOutput: metinle hizalama için örtüşmesiz çıktı önce
            ann = getattr(out, "exclusive_speaker_diarization", None) or full
            cents = getattr(out, "speaker_embeddings", None)
        else:
            try:  # 3.x: konuşmacı başına ses imzası da istenir (ses kütüphanesi için)
                out = self.pipe(file, return_embeddings=True, **kw)
            except TypeError:
                out = self.pipe(file, **kw)
            if isinstance(out, tuple):
                out, cents = out
            full = ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
        try:
            self.last_emb = speaker_embeddings(full.labels(), cents) if hasattr(full, "labels") else None
        except Exception:  # noqa: BLE001
            log.debug("ses imzaları alınamadı", exc_info=True)
            self.last_emb = None
        return [[round(seg.start, 3), round(seg.end, 3), str(spk)] for seg, _, spk in ann.itertracks(yield_label=True)]

    def _vocals(self, audio):
        """Müzik / efektleri ayıkla, yalnız konuşmayı bırak (HDemucs, torchaudio içinde; ek kurulum yok).
        Yalnızca ekran kartında; olmazsa özgün ses döner. Film / podcast müziği ses imzasını bozmasın diye."""
        if not str(self.device).startswith("cuda"):
            log.info("Müzik ayıklama atlandı: yalnızca ekran kartında yapılır")
            return audio
        import numpy as np
        torch = self.torch
        t0 = time.time()
        try:
            import torchaudio
            if self._sep is None:
                from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as bundle
                log.info("Müzik ayıklama modeli yükleniyor (ilk seferde ~300 MB iner)")
                self._beat(True)
                m = bundle.get_model().eval().to(torch.device(self.device))
                log.info("Müzik ayıklama modeli hazır (%.0f sn)", time.time() - t0)
                srcs = list(getattr(m, "sources", []) or [])
                self._sep = (m, bundle.sample_rate, srcs.index("vocals") if "vocals" in srcs else 3)
            model, msr, vi = self._sep
            x = np.asarray(audio, dtype=np.float32)
            mean, std = float(x.mean()), float(x.std()) or 1.0
            out = np.empty_like(x)
            dev = torch.device(self.device)
            with torch.inference_mode():
                for a, b, c, d in chunk_spans(len(x), 10 * SR, SR):  # 12 sn pencere: bellek az
                    self._beat()
                    ch = torch.from_numpy((x[a:b] - mean) / std).to(dev)
                    up = torchaudio.functional.resample(ch, SR, msr)
                    y = model(up[None, None].expand(1, 2, -1).contiguous())[0, vi].mean(0)
                    y = torchaudio.functional.resample(y, msr, SR)[: b - a]
                    if y.shape[0] < b - a:
                        y = torch.nn.functional.pad(y, (0, b - a - y.shape[0]))
                    out[c:d] = y[c - a:d - a].float().cpu().numpy() * std + mean
            log.info("[%s] müzik ayıklandı (%.0f sn ses, %.1f sn)", self.device, len(x) / SR, time.time() - t0)
            return out
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if ("cuda" in msg or "cublas" in msg or "cudnn" in msg) and "out of memory" not in msg:
                # ekran kartı bağlamı bozuldu: bu süreçte pyannote da düşer. Hemen bildir; ana süreç yeni
                # süreçte müzik ayıklamasız tekrar dener
                raise RuntimeError(f"müzik ayıklama sırasında ekran kartı hatası: {str(e)[:200]}") from e
            log.warning("Müzik ayıklanamadı (%s); konuşmacı ayırma özgün sesle yapılıyor", str(e)[:200])
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return audio

    def _run_lowmem(self, audio, kw) -> list[list]:
        """RAM / ekran kartı belleği yetmezse toplu boyutu küçültüp, belleği boşaltıp tekrar dene."""
        while True:
            try:
                return self._run(audio, kw)
            except (MemoryError, RuntimeError) as e:
                msg = str(e).lower()
                ram = isinstance(e, MemoryError) or "not enough memory" in msg or "bad_alloc" in msg \
                    or "defaultcpuallocator" in msg
                if not (ram or "out of memory" in msg) or self.batch <= 1:
                    raise
                self.batch = max(1, self.batch // 4)
                self._set_batch(self.pipe, self.batch)
                import gc
                gc.collect()
                if self.device != "cpu":
                    self.torch.cuda.empty_cache()
                log.warning("[%s] konuşmacı ayırmada bellek yetmedi; toplu boyut %d ile, 30 sn sonra tekrar",
                            self.device, self.batch)
                time.sleep(30)
                self.wait_memory(len(audio) / SR, max_wait=600)

    VOCALS_MIN_FREE_GB = 4.0  # müzik ayıklama için en az bu kadar boş RAM ve sanal bellek

    def _low_memory(self) -> bool:
        """Müzik ayıklama ek bellek ister (1.8.7: tarayıcı belleği doldurunca ayırma "out of memory" ile düştü)."""
        from .util import memory_status
        m = memory_status()
        return bool(m) and min(m["ram_free"], m["commit_free"]) < self.VOCALS_MIN_FREE_GB

    def _diarizen(self, audio, min_speakers, max_speakers) -> list[list] | None:
        """DiariZen ile dene (müzik ayıklanmış / agc'li ses); olmazsa None -> pyannote. Ses imzası çıkmaz."""
        from . import diarizen
        if not diarizen.installed():
            log.warning("diar_engine=diarizen ama DiariZen kurulu değil (python -m stt diarizen-kur); pyannote kullanılıyor")
            return None
        import tempfile
        import wave
        from pathlib import Path

        import numpy as np
        t = time.time()
        try:
            with tempfile.TemporaryDirectory(prefix="stt-dzw-") as td:
                wav = Path(td) / "ses.wav"
                with wave.open(str(wav), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
                turns = diarizen.run(str(wav), min_speakers, max_speakers, beat=lambda f=False: self._beat(f),
                                     token=self.token)
            log.info("[%s] konuşmacı ayırma (DiariZen) %.1fs, %d konuşmacı", self.device, time.time() - t,
                     len({x[2] for x in turns}))
            self.model_used = diarizen.MODEL  # çıktıya hangi motorun ayırdığı yazılsın (K5 ölçümü için)
            return turns
        except Exception as e:  # noqa: BLE001
            log.warning("DiariZen başarısız (%s); pyannote ile yapılıyor", str(e)[:300])
            return None

    def _beat(self, force: bool = False) -> None:
        if self._beat_fn is not None:
            self._beat_fn(force)

    def _hook(self, *_a, **_k) -> None:
        self._beat()

    def run(self, wav: str, min_speakers=None, max_speakers=None, vocals: bool = False, beat=None) -> list[list]:
        self._beat_fn = beat
        audio = load_wav(wav)
        log.info("[%s] konuşmacı ayırma başlıyor (%.0f dk ses%s)", self.device, len(audio) / SR / 60,
                 ", önce müzik ayıklanacak" if vocals else "")
        self.wait_memory(len(audio) / SR)
        self.last_emb = None
        if vocals and self._low_memory():
            log.warning("[%s] bellek az; müzik ayıklama atlandı (konuşmacı ayırma özgün sesle)", self.device)
            vocals = False
        if vocals:  # film / podcast: müzik ve efektler ses imzasını bozmasın
            audio = self._vocals(audio)
            self._beat(True)
        if self.agc:  # uzaktan konuşan muhabirler de ayrı konuşmacı olarak bulunsun
            from .audio import agc
            agc(audio)
        t = time.time()
        kw = {}
        if min_speakers:
            kw["min_speakers"] = min_speakers
        if max_speakers:
            kw["max_speakers"] = max_speakers
        if getattr(self, "engine", "pyannote") == "diarizen" and str(self.device).startswith("cuda"):
            turns = self._diarizen(audio, min_speakers, max_speakers)
            if turns is not None:
                return turns
        self._ensure_pipe()  # DiariZen'de yedek: pyannote ancak şimdi yüklenir
        self.model_used = self.pyannote_model or self.model_used
        try:
            turns = self._run_lowmem(audio, kw)
        except RuntimeError as e:
            msg = str(e).lower()
            if self.device == "cpu" or not any(x in msg for x in ("cudnn", "cuda", "out of memory", "cublas")):
                raise
            turns = None
            if "cudnn" in msg and self.torch.backends.cudnn.enabled:
                # cuDNN_STATUS_EXECUTION_FAILED vb.: cuDNN olmadan ekran kartında tekrar dene
                log.warning("[%s] cuDNN hatası (%s); cuDNN kapalı tekrar deneniyor", self.device, str(e)[:120])
                self.torch.backends.cudnn.enabled = False
                try:
                    self.torch.cuda.empty_cache()
                    turns = self._run(audio, kw)
                except RuntimeError as e2:
                    log.warning("[%s] yine olmadı (%s)", self.device, str(e2)[:120])
            if turns is None and not self.cpu_ok:
                raise RuntimeError(f"Konuşmacı ayırma ekran kartında başarısız oldu ({str(e)[:160]}); "
                                   "işlemciye geçilmedi. Bir sonraki videoda ekran kartı baştan açılır.")
            if turns is None:
                # ekran kartında olmuyor: işlemcide (CPU) baştan yükle; yavaş ama metin soru-cevap olarak çıkar
                log.warning("[%s] konuşmacı ayırma işlemciye (CPU) alındı; bu kayıt biraz daha uzun sürer", self.device)
                self.pipe = self._load()
                self.device = "cpu"
                turns = self._run(audio, kw)
        log.info("[%s] konuşmacı ayırma %.1fs, %d konuşmacı", self.device, time.time() - t, len({x[2] for x in turns}))
        return turns
