from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
import shutil
import site
import sys
import unicodedata
from pathlib import Path

log = logging.getLogger("stt")

MEDIA_EXTS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts", ".wmv", ".mpg", ".mpeg", ".3gp",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr",
}
PARTIAL_EXTS = {".part", ".crdownload", ".tmp", ".download", ".ytdl", ".partial"}


def quiet_warnings() -> None:
    import warnings
    for pat in (".*list_audio_backends.*", ".*torchaudio._backend.*", ".*TorchCodec.*", ".*std\\(\\): degrees of freedom.*"):
        warnings.filterwarnings("ignore", message=pat)


def setup_logging(verbose: bool = False) -> None:
    quiet_warnings()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "faster_whisper", "speechbrain", "pytorch_lightning", "lightning"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_tool_paths() -> None:
    """Kurulumun indirdiği ffmpeg / deno (tools klasörü) PATH'e eklensin."""
    root = Path(__file__).resolve().parent.parent / "tools"
    if not root.is_dir():
        return
    for d in [*root.glob("ffmpeg*/bin"), root / "deno"]:
        if d.is_dir():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


_DLL_DONE = False


def setup_cuda_dlls() -> None:
    """Windows: torch\\lib ve nvidia\\*\\bin klasörlerini DLL arama yoluna ekle (cuBLAS/cuDNN)."""
    global _DLL_DONE
    if _DLL_DONE or os.name != "nt":
        return
    _DLL_DONE = True
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    dirs: list[str] = []
    for sp in roots:
        dirs.append(os.path.join(sp, "torch", "lib"))
        dirs.extend(glob.glob(os.path.join(sp, "nvidia", "*", "bin")))
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)  # type: ignore[attr-defined]
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def file_hash(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def slugify(s: str, maxlen: int = 80) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜəƏ", "cgiosuCGIOSUeE")
    s = unicodedata.normalize("NFKD", s.translate(tr)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")
    return (s[:maxlen].rstrip("._-")) or "kayit"


def fmt_ts(sec: float, sep: str = ",") -> str:
    sec = max(0.0, sec)
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_clock(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def which(name: str) -> str | None:
    return shutil.which(name)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
