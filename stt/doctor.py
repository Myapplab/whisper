"""Kurulum kontrolü: Python, ffmpeg, yt-dlp, JS çalışma zamanı, CUDA/GPU, model, pyannote, OpenAI, FPL."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from .util import setup_cuda_dlls, which

OK, WARN, BAD = "[ OK ]", "[UYARI]", "[HATA]"


def _ver(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else None
    except Exception:  # noqa: BLE001
        return None


def run_doctor(cfg, deep: bool = False) -> int:
    bad = 0
    rows: list[tuple[str, str, str]] = []

    def add(status, name, msg):
        nonlocal bad
        rows.append((status, name, msg))
        print(f"{status} {name:<18} {msg}", flush=True)
        if status == BAD:
            bad += 1

    v = sys.version_info
    add(OK if (v.major, v.minor) == (3, 11) else WARN, "Python", f"{platform.python_version()} ({sys.executable})"
        + ("" if (v.major, v.minor) == (3, 11) else "  -> 3.11 önerilir"))
    add(OK, "Sistem", f"{platform.system()} {platform.release()} {platform.machine()}")

    for tool in ("ffmpeg", "ffprobe"):
        p = which(tool)
        if p:
            add(OK, tool, (_ver([p, "-version"]) or "?")[:70])
        else:
            add(BAD, tool, "bulunamadı -> STT-Kurulum.exe'yi tekrar çalıştırın (tools klasörüne indirir)")

    try:
        import yt_dlp
        add(OK, "yt-dlp", f"{yt_dlp.version.__version__}  (güncelleme: pip install -U \"yt-dlp[default]\")")
    except Exception as e:  # noqa: BLE001
        add(BAD, "yt-dlp", f"yok: {e}")
    js = next((n for n in ("deno", "node", "bun", "qjs") if which(n)), None)
    if js:
        add(OK, "JS çalışma zamanı", f"{js}: {(_ver([which(js), '--version']) or '?')[:50]} (YouTube için)")
    else:
        add(WARN, "JS çalışma zamanı", "yok -> YouTube indirmeleri çalışmaz; STT-Kurulum.exe'yi tekrar çalıştırın")

    # CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    setup_cuda_dlls()
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            add(OK, "torch", f"{torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
            for i in range(n):
                pr = torch.cuda.get_device_properties(i)
                add(OK, f"cuda:{i}", f"{pr.name}, {pr.total_memory / 2**30:.1f} GB")
        else:
            add(WARN, "torch", f"{torch.__version__} - CUDA görünmüyor (CPU sürümü kurulmuş olabilir; STT-Kurulum.exe'yi tekrar çalıştırın)")
    except Exception as e:  # noqa: BLE001
        add(WARN, "torch", f"yok/yüklenemedi ({e}) - konuşmacı ayırma çalışmaz")
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        if n:
            types = ctranslate2.get_supported_compute_types("cuda", 0)
            add(OK, "ctranslate2", f"{ctranslate2.__version__}, {n} GPU, float16={'float16' in types}")
        else:
            add(BAD if sys.platform == "win32" else WARN, "ctranslate2",
                f"{ctranslate2.__version__}, CUDA GPU yok (sürücü? 'nvidia-smi')")
    except Exception as e:  # noqa: BLE001
        add(BAD, "ctranslate2", f"yüklenemedi: {e}")
    smi = which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
            lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        except Exception:  # noqa: BLE001
            lines = []
        for ln in lines or ["?"]:
            add(OK, "nvidia-smi", ln)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        add(OK if not cvd else WARN, "CUDA_VISIBLE_DEVICES",
            "(boş - tüm kartlar görünür)" if not cvd else f"'{cvd}' -> yalnızca bu kart(lar) görünür; uygulama açılışta bunu kaldırır")
        try:
            import ctranslate2
            ncuda = ctranslate2.get_cuda_device_count()
        except Exception:  # noqa: BLE001
            ncuda = -1
        if lines and ncuda >= 0 and ncuda < len(lines):
            add(BAD, "GPU sayısı", f"nvidia-smi {len(lines)} kart görüyor ama CUDA {ncuda} kart görüyor. "
                "Sürücüyü yeniden kurun / Aygıt Yöneticisi'nde 3080'i kontrol edin; "
                "PowerShell: nvidia-smi -L ; echo $env:CUDA_VISIBLE_DEVICES")
        add(OK, "GPU sırası", "cuda:N = nvidia-smi N (PCI sırası)")
        try:
            from .engines import gpu_info
            from .pipeline import detect_devices
            gi = gpu_info()
            used = detect_devices(cfg.gpus, gi)
            for i, g in sorted(gi.items()):
                mon = "monitör bağlı" if g.get("display") else "monitör yok" if g.get("display") is False else "?"
                st = "KULLANILACAK" if f"cuda:{i}" in used else "boş bırakılacak"
                add(OK, f"Kart {i}", f"{g['name']} ({mon}) -> {st}  [gpus={cfg.gpus}]")
        except Exception as e:  # noqa: BLE001
            add(WARN, "Kart seçimi", str(e)[:120])

    try:
        import faster_whisper
        add(OK, "faster-whisper", faster_whisper.__version__)
        model = cfg.model
        if os.path.isdir(model):
            ok = (Path(model) / "model.bin").is_file()
            add(OK if ok else BAD, "Model", f"yerel: {model}" + ("" if ok else "  -> model.bin YOK (klasör eksik/bozuk)"))
        else:
            try:
                from faster_whisper.utils import download_model
                path = download_model(model, local_files_only=True)
                add(OK, "Model", f"{model} önbellekte: {path}")
            except Exception:  # noqa: BLE001
                add(WARN, "Model", f"{model} henüz indirilmedi (ilk çalıştırmada ~3 GB inecek)")
    except Exception as e:  # noqa: BLE001
        add(BAD, "faster-whisper", f"yok: {e}")

    if deep:
        try:
            import ctranslate2
            import numpy as np

            from .engines import WhisperEngine, resolve_model_path
            path = resolve_model_path(cfg.model)
            for d in [f"cuda:{i}" for i in range(ctranslate2.get_cuda_device_count())] or ["cpu"]:
                eng = WhisperEngine(path, d, cfg.compute_type, 1)
                list(eng.model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
                add(OK, f"Model test {d}", "yüklendi ve çalıştı (cuBLAS/cuDNN tamam)")
                del eng
        except Exception as e:  # noqa: BLE001
            add(BAD, "Model test", f"{type(e).__name__}: {e}")

    try:
        import pyannote.audio
        tok = cfg.hf_token
        add(OK if tok else WARN, "pyannote.audio", f"{pyannote.audio.__version__}; HF token: "
            + (f"var ({tok[:5]}…)" if tok else "YOK -> soru-cevap ayrımı çalışmaz (uygulamada Ayarlar > HuggingFace token)"))
        if tok:
            print("       konuşmacı ayırma modeli deneniyor (ilk seferde ~30 MB iner)...", flush=True)
            try:
                from .engines import Diarizer
                Diarizer(cfg.diar_model, tok, "cpu")
                add(OK, "Konuşmacı modeli", f"{cfg.diar_model} yüklendi")
            except Exception as e:  # noqa: BLE001
                from .pipeline import diar_hint
                add(BAD, "Konuşmacı modeli", diar_hint(f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001
        add(WARN, "pyannote.audio", f"yok/yüklenemedi ({str(e)[:80]}) - --diarize kullanılamaz")

    try:
        import jiwer  # noqa: F401
        add(OK, "jiwer", "var (stt olc)")
    except Exception:  # noqa: BLE001
        add(WARN, "jiwer", "yok -> pip install jiwer")

    try:
        import openai
        add(OK if os.environ.get("OPENAI_API_KEY") else WARN, "openai",
            f"{openai.__version__}; OPENAI_API_KEY " + ("var" if os.environ.get("OPENAI_API_KEY") else "YOK"))
    except Exception:  # noqa: BLE001
        add(WARN, "openai", "paket yok (yedek motor kullanılamaz)")

    if cfg.fpl_repo:
        rp = Path(cfg.fpl_repo)
        add(OK if (rp / "presser.py").is_file() else WARN, "FPL repo", f"{rp} " +
            ("(presser.py var)" if (rp / "presser.py").is_file() else
             "(presser.py YOK - yalnızca --fpl için gerekli; Ayarlar'dan düzeltin)"))
    else:
        add(WARN, "FPL repo", "ayarlı değil (--fpl için FPL_REPO ya da stt.toml fpl_repo)")

    if cfg.cookies_browser:
        add(OK, "Çerezler", f"tarayıcı: {cfg.cookies_browser}"
            + (" (Windows'ta Chrome çerezleri için Chrome kapalı olmalı; olmazsa firefox ya da cookies.txt)"
               if sys.platform == "win32" and cfg.cookies_browser == "chrome" else ""))
    print()
    print("Sonuç:", "her şey hazır." if bad == 0 else f"{bad} kritik sorun var.")
    return 1 if bad else 0
