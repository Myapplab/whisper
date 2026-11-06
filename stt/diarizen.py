"""DiariZen (BUT, Brno): isteğe bağlı konuşmacı ayırma motoru (K6).

Neden ayrı ortam: DiariZen Python 3.10 + torch 2.1.1 + kendi pyannote çatalını ister; ana ortamdaki
pyannote.audio 4 / torch 2.8 ile aynı yere kurulursa mevcut sistemi bozar. Bu yüzden:
- kurulum: ``<araç>/.venv-diarizen`` (Python 3.10) + ``<araç>/diarizen`` (GitHub zip; pyannote çatalı içinde)
- çalıştırma: ayırma işçisi sesi (müzik ayıklanmış / agc'li) geçici wav'a yazar, bu ortamda ayrı süreç çalışır,
  turlar JSON olarak döner. Hata olursa çağıran pyannote'a döner.
Ağırlıklar CC BY-NC 4.0 (yalnız ticari olmayan kullanım). Ayar: ``diar_engine = "diarizen"`` (stt.toml).
Kurulum: ``python -m stt diarizen-kur``.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from .config import TOOL_DIR
from .util import log, popen_kw

MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
ROOT = TOOL_DIR / "diarizen"
VENV = TOOL_DIR / ".venv-diarizen"
ZIP_URL = "https://codeload.github.com/BUTSpeechFIT/DiariZen/zip/refs/heads/main"
PY310_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
TORCH = ["torch==2.1.1", "torchvision==0.16.1", "torchaudio==2.1.1", "--index-url",
         "https://download.pytorch.org/whl/cu121"]

RUNNER = r'''
import json, sys, time
args = json.loads(sys.argv[1])
from diarizen.pipelines.inference import DiariZenPipeline
t0 = time.time()
pipe = DiariZenPipeline.from_pretrained(args["model"])
if args.get("min_speakers"):
    pipe.min_speakers = int(args["min_speakers"])
if args.get("max_speakers"):
    pipe.max_speakers = int(args["max_speakers"])
print("[diarizen] model hazir %.0f sn" % (time.time() - t0), flush=True)
ann = pipe(args["wav"], sess_name="stt")
turns = [[round(s.start, 3), round(s.end, 3), str(spk)] for s, _, spk in ann.itertracks(yield_label=True)]
with open(args["out"], "w", encoding="utf-8") as f:
    json.dump({"turns": turns, "elapsed": round(time.time() - t0, 1)}, f)
'''


def python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed() -> bool:
    return python().exists() and (VENV / "stt_ok").exists() and (ROOT / "diarizen").is_dir()


def token_free(cfg) -> bool:
    """DiariZen seçili ve kurulu: konuşmacı ayırma HF token'sız da yapılır (DiariZen modelleri herkese açık;
    token yalnız DiariZen hata verirse devreye giren pyannote yedeği için gerekir)."""
    return str(getattr(cfg, "diar_engine", "") or "").lower() == "diarizen" and installed()


def _run(cmd: list, progress, what: str, timeout: float = 3600, cwd=None) -> None:
    progress(what)
    log.info("[diarizen kurulum] %s", what)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, cwd=cwd, **popen_kw())
    if r.returncode != 0:
        raise RuntimeError(f"{what} başarısız: {(r.stderr or r.stdout)[-800:]}")


def _python310(progress) -> str:
    """Python 3.10: varsa py başlatıcısından, yoksa python.org kurucusuyla kullanıcı klasörüne (yönetici izni yok)."""
    if os.name != "nt":
        p = shutil.which("python3.10")
        if p:
            return p
        raise RuntimeError("Python 3.10 bulunamadı (python3.10)")
    try:
        r = subprocess.run(["py", "-3.10", "-c", "import sys;print(sys.executable)"], capture_output=True, text=True,
                           timeout=60, **popen_kw())
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[-1]
    except OSError:
        pass
    target = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Programs" / "Python" / "Python310"
    exe = target / "python.exe"
    if not exe.exists():
        from .updater import _get
        progress("Python 3.10 indiriliyor (python.org, ~28 MB)")
        inst = Path(tempfile.gettempdir()) / "python-3.10.11-amd64.exe"
        inst.write_bytes(_get(PY310_URL, timeout=600))
        _run([inst, "/quiet", "InstallAllUsers=0", "Include_launcher=0", "PrependPath=0", "Include_test=0",
              "Shortcuts=0", f"TargetDir={target}"], progress, "Python 3.10 kuruluyor", timeout=900)
    if not exe.exists():
        raise RuntimeError("Python 3.10 kurulamadı")
    return str(exe)


def install(progress=lambda text: None) -> dict:
    """Ayrı ortamı kur (~4 GB iner, 10-30 dk). Tekrar çalıştırmak güvenli."""
    t0 = time.time()
    py310 = _python310(progress)
    if not python().exists():
        _run([py310, "-m", "venv", VENV], progress, "DiariZen ortamı oluşturuluyor")
    py = python()
    _run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], progress, "pip güncelleniyor")
    _run([py, "-m", "pip", "install", "-q", *TORCH], progress, "torch 2.1.1 (CUDA 12.1) kuruluyor, ~2.5 GB")
    from .updater import _get
    progress("DiariZen kodu indiriliyor (GitHub)")
    data = _get(ZIP_URL, timeout=600)
    tmp = Path(tempfile.mkdtemp(prefix="stt-diarizen-"))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():  # zip içinden dışarı yazma girişimi olmasın
            t = (tmp / m.filename).resolve()
            if tmp.resolve() not in t.parents and t != tmp.resolve():
                raise RuntimeError(f"Geçersiz zip girdisi: {m.filename}")
        z.extractall(tmp)
    src = next(p for p in tmp.iterdir() if p.is_dir())
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.move(str(src), str(ROOT))
    shutil.rmtree(tmp, ignore_errors=True)
    cons = ["-c", "constraints.txt"] if (ROOT / "constraints.txt").exists() else []
    _run([py, "-m", "pip", "install", "-q", "-r", "requirements.txt", *cons], progress,
         "DiariZen kütüphaneleri kuruluyor", cwd=ROOT)
    _run([py, "-m", "pip", "install", "-q", "-e", "."], progress, "DiariZen kuruluyor", cwd=ROOT)
    _run([py, "-m", "pip", "install", "-q", "-e", ".", *(["-c", "../constraints.txt"] if cons else [])], progress,
         "DiariZen'in pyannote kopyası kuruluyor", cwd=ROOT / "pyannote-audio")
    _run([py, "-c", "from diarizen.pipelines.inference import DiariZenPipeline; import torch; "
                    "print(torch.__version__, torch.cuda.is_available())"], progress, "Deneniyor", timeout=600)
    (VENV / "stt_ok").write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    log.info("DiariZen kuruldu (%.0f dk)", (time.time() - t0) / 60)
    return {"ok": True, "minutes": round((time.time() - t0) / 60, 1)}


def run(wav: str, min_speakers=None, max_speakers=None, timeout: float = 4 * 3600, beat=None,
        token: str | None = None) -> list[list]:
    """Ayrı süreçte DiariZen. Ekran kartı seçimi ortamdan gelir (ayırma işçisi CUDA_VISIBLE_DEVICES ile tek kart görür;
    DiariZen hep cuda:0 kullanır = o kart). Döner: [[başla, bitir, konuşmacı], ...]."""
    if not installed():
        raise RuntimeError("DiariZen kurulu değil (python -m stt diarizen-kur)")
    with tempfile.TemporaryDirectory(prefix="stt-dz-") as td:
        out = Path(td) / "turns.json"
        script = Path(td) / "run.py"
        script.write_text(RUNNER, encoding="utf-8")
        args = {"wav": str(wav), "out": str(out), "model": MODEL, "min_speakers": min_speakers,
                "max_speakers": max_speakers}
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_XET": "1"}
        env.pop("PYTHONPATH", None)
        if token and not env.get("HF_TOKEN"):  # stt.toml'daki token (DiariZen modelleri herkese açık; olur da gerekirse)
            env["HF_TOKEN"] = token
        p = subprocess.Popen([str(python()), str(script), json.dumps(args)], cwd=str(ROOT), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", **popen_kw(low_priority=True))
        tail: list[str] = []

        def read():
            for line in p.stdout:  # type: ignore[union-attr]
                tail.append(line.rstrip())
                del tail[:-30]
        th = threading.Thread(target=read, daemon=True)
        th.start()
        t0 = time.time()
        while p.poll() is None:  # süreç yaşadıkça takılma denetimine sinyal; asıl sınır ``timeout``
            if beat:
                beat(False)
            if time.time() - t0 > timeout:
                p.kill()
                raise RuntimeError("DiariZen süre sınırını aştı")
            time.sleep(5)
        th.join(5)
        if p.returncode != 0 or not out.exists():
            raise RuntimeError("DiariZen hata verdi: " + " | ".join(tail[-6:])[-600:])
        res = json.loads(out.read_text(encoding="utf-8"))
        log.info("[diarizen] konuşmacı ayırma %.0f sn, %d konuşmacı", res.get("elapsed", 0),
                 len({t[2] for t in res["turns"]}))
        return res["turns"]


def main() -> int:
    """``python -m stt diarizen-kur``"""
    try:
        res = install(lambda t: print(t, flush=True))
    except Exception as e:  # noqa: BLE001
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    print(f"DiariZen kuruldu ({res['minutes']} dk). Kullanmak için stt.toml'a: diar_engine = \"diarizen\"")
    return 0
