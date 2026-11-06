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
        add(OK, "yt-dlp", f"{yt_dlp.version.__version__}  (güncelleme: pip install -U \"yt-dlp[default,curl-cffi]\")")
    except Exception as e:  # noqa: BLE001
        add(BAD, "yt-dlp", f"yok: {e}")
    try:
        import curl_cffi
        add(OK, "curl_cffi", f"{getattr(curl_cffi, '__version__', '?')} (Dailymotion vb. tarayıcı taklidi)")
    except Exception:  # noqa: BLE001
        add(WARN, "curl_cffi", "yok -> Dailymotion gibi siteler inmez; uygulamada Güncelle'ye basın")
    try:
        import playwright  # noqa: F401
        from .browser import _launch
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = _launch(pw)
            ver = br.version
            br.close()
        add(OK, "Gizli tarayıcı", f"Playwright + {ver} (doğrudan inmeyen siteler için)")
    except ImportError:
        add(WARN, "Gizli tarayıcı", "playwright yok -> bazı siteler inmeyebilir; uygulamada Güncelle'ye basın")
    except Exception as e:  # noqa: BLE001
        add(WARN, "Gizli tarayıcı", f"açılamadı: {str(e)[:120]}")
    from .fetch import cookie_browsers, installed_browsers
    inst = installed_browsers()
    add(OK if inst else WARN, "Tarayıcılar", (", ".join(inst) or "bulunamadı") + "  (çerez sırası: "
        + (", ".join(cookie_browsers(cfg.cookies_browser)) or "kapalı") + ")")
    js = next((n for n in ("deno", "node", "bun", "qjs") if which(n)), None)
    if js:
        add(OK, "JS çalışma zamanı", f"{js}: {(_ver([which(js), '--version']) or '?')[:50]} (YouTube için)")
    else:
        add(WARN, "JS çalışma zamanı", "yok -> YouTube indirmeleri çalışmaz; STT-Kurulum.exe'yi tekrar çalıştırın")

    # bellek: iki Whisper işçisi + konuşmacı ayırma birlikte ~12-16 GB sanal bellek ister
    from .util import PAGEFILE_HINT, memory_status, memory_text
    m = memory_status()
    if m:
        st = OK if m["commit_free"] >= 12 else WARN
        add(st, "Bellek", memory_text() + ("" if st == OK else "  -> AZ. " + PAGEFILE_HINT))

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
            from .pipeline import detect_devices, helper_devices
            gi = gpu_info()
            used = detect_devices(cfg.gpus, gi)
            helpers = helper_devices(cfg.gpus, gi) if all(d.startswith("cuda") for d in used) else []
            for i, g in sorted(gi.items()):
                mon = "monitör bağlı" if g.get("display") else "monitör yok" if g.get("display") is False else "?"
                st = "KULLANILACAK" if f"cuda:{i}" in used else \
                    "YARDIMCI (düşük yük, başka program kullanırken bekler)" if f"cuda:{i}" in helpers else "boş bırakılacak"
                add(OK, f"Kart {i}", f"{g['name']} ({mon}) -> {st}  [gpus={cfg.gpus}]")
        except Exception as e:  # noqa: BLE001
            add(WARN, "Kart seçimi", str(e)[:120])
        try:  # ekran kararması / "HDMI sinyali yok": Windows olay günlüğündeki sürücü çökmeleri
            from .gpuguard import describe, display_events
            ev = display_events(14)
            if os.name != "nt":
                pass
            elif ev:
                add(WARN, "Ekran sürücüsü", "Son 14 gün: " + describe(ev) + ". Olası sebepler: monitörlü karta yük "
                    "(Ayarlar › Ekran kartı: Önerilen), MSI Afterburner hız aşırtması (Reset), güç kaynağı.")
            else:
                add(OK, "Ekran sürücüsü", "son 14 günde çökme / beklenmedik kapanma yok")
        except Exception as e:  # noqa: BLE001
            add(WARN, "Ekran sürücüsü", f"olay günlüğü okunamadı: {str(e)[:100]}")

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
        from .diarizen import token_free
        no_tok = ("YOK -> DiariZen çalışır ama DiariZen hata verirse yedek pyannote çalışmaz (Ayarlar > HuggingFace token)"
                  if token_free(cfg) else
                  "YOK -> soru-cevap ayrımı çalışmaz (uygulamada Ayarlar > HuggingFace token)")
        add(OK if tok else WARN, "pyannote.audio", f"{pyannote.audio.__version__}; HF token: "
            + (f"var ({tok[:5]}…)" if tok else no_tok))
        if str(pyannote.audio.__version__).split(".")[0] in ("2", "3"):
            add(WARN, "Konuşmacı modeli sürümü", f"pyannote.audio {pyannote.audio.__version__}: yeni model (community-1) "
                "çalışmaz, eski 3.1 kullanılıyor. Uygulamada Güncelle'ye basın; olmazsa STT-Kurulum.exe'yi tekrar çalıştırın.")
        if tok:
            print("       konuşmacı ayırma modeli deneniyor (ilk seferde ~30 MB iner)...", flush=True)
            try:
                from .engines import Diarizer
                d = Diarizer(cfg.diar_model, tok, "cpu", cpu_ok=True)  # yalnızca yükleme denemesi; iş kartta yapılır
                if d.model_used and d.model_used != cfg.diar_model:
                    add(WARN, "Konuşmacı modeli", f"{cfg.diar_model} kullanılamadı; eski model {d.model_used} ile "
                        f"çalışılıyor. Yeni model için huggingface.co/{cfg.diar_model} sayfasında 'Agree' deyin"
                        + ("" if d.v4 else " ve uygulamayı güncelleyin (pyannote.audio 4 gerekli)") + ".")
                else:
                    add(OK, "Konuşmacı modeli", f"{d.model_used or cfg.diar_model} yüklendi")
                del d
            except Exception as e:  # noqa: BLE001
                from .pipeline import diar_hint
                add(BAD, "Konuşmacı modeli", diar_hint(f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001
        add(WARN, "pyannote.audio", f"yok/yüklenemedi ({str(e)[:80]}) - --diarize kullanılamaz")
    try:  # ölçüm: düzeltilmiş videolarda programın konuşmacı hatası (K5)
        from .tune import measure
        m = measure(Path(cfg.out_dir).resolve() / ".stt")
        if m["videos"]:
            worst = max(m["per"], key=lambda x: x["error"])
            add(OK if m["error"] < 10 else WARN, "Konuşmacı doğruluğu",
                f"düzeltilmiş {m['videos']} videoda ({m['minutes']} dk) hata %{m['error']}; en kötü: "
                f"%{worst['error']} {worst['title']}")
            if len(m.get("groups") or {}) > 1:  # farklı ayarla işlenmiş videolar: hangisi daha iyi (A/B)
                add(OK, "Konuşmacı doğruluğu (ayara göre)", " · ".join(
                    f"{k}: %{v['error']} ({v['videos']} video)" for k, v in sorted(m["groups"].items())))
        else:
            add(WARN, "Konuşmacı doğruluğu", "ölçülmedi: 'Metni göster' > 'Konuşmacılar' ekranında 3-5 basın "
                "toplantısını düzeltip kaydedin (yeni sürümle işlenmiş videolar)")
    except Exception as e:  # noqa: BLE001
        add(WARN, "Konuşmacı doğruluğu", f"ölçülemedi ({str(e)[:80]})")

    try:
        import jiwer  # noqa: F401
        add(OK, "jiwer", "var (stt olc)")
    except Exception:  # noqa: BLE001
        add(WARN, "jiwer", "yok -> pip install jiwer")

    try:
        import openai
        okey = bool(getattr(cfg, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY"))
        add(OK if okey else WARN, "openai", f"{openai.__version__}; OpenAI anahtarı " + ("var" if okey else "YOK"))
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
            + (" (Windows'ta Chrome çerezleri çoğu zaman okunamaz; diğer tarayıcılar da sırayla denenir)"
               if sys.platform == "win32" and cfg.cookies_browser == "chrome" else ""))
    if cfg.llm_provider:
        try:
            from .llm import LLM
            llm = LLM(cfg, Path(cfg.out_dir) / ".stt")
            st = llm.status()
            add(OK if st["ready"] else WARN, "Yapay zekâ", f"{cfg.llm_provider}: " + ("anahtar var" if st["ready"] else
                "ANAHTAR YOK (Ayarlar > Yapay zekâ)") + f"; bu ay ~${st['month_usd']:.3f} / sınır ${cfg.llm_budget_usd}")
        except Exception as e:  # noqa: BLE001
            add(WARN, "Yapay zekâ", str(e)[:120])
    else:
        add(OK, "Yapay zekâ", "kapalı (isteğe bağlı: haber taslağı, soru-cevap düzeltme, konu filtresi)")
    if cfg.api_port:
        add(OK, "Yerel API", f"http://127.0.0.1:{cfg.api_port}/api/status")
    print()
    print("Sonuç:", "her şey hazır." if bad == 0 else f"{bad} kritik sorun var.")
    return 1 if bad else 0
