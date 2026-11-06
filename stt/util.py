from __future__ import annotations

import glob
import hashlib
import logging
import math
import os
import re
import shutil
import site
import sys
import time
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


def _registry_path() -> list[str]:
    """Windows: kayıt defterindeki GÜNCEL PATH (kurulumdan hemen sonra açılan uygulamanın ortamı eskidir)."""
    if os.name != "nt":
        return []
    out: list[str] = []
    try:
        import winreg
        for hive, sub in ((winreg.HKEY_CURRENT_USER, "Environment"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, sub) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                out += [os.path.expandvars(x) for x in str(val).split(";") if x.strip()]
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


def tool_dirs(root: Path | None = None) -> list[Path]:
    """ffmpeg / deno'nun bulunabileceği klasörler (kurulumun 'tools' klasörü, winget, deno kurulumu)."""
    root = Path(root) if root else Path(__file__).resolve().parent.parent / "tools"
    dirs: list[Path] = []
    if root.is_dir():
        dirs += [*root.glob("ffmpeg*/bin"), root / "deno", root]
    la = os.environ.get("LOCALAPPDATA")
    if la:
        wg = Path(la) / "Microsoft" / "WinGet"
        dirs += [wg / "Links", *wg.glob("Packages/Gyan.FFmpeg*/ffmpeg-*/bin"), *wg.glob("Packages/DenoLand.Deno*")]
    dirs.append(Path.home() / ".deno" / "bin")
    return [d for d in dirs if d.is_dir()]


def add_tool_paths(root: Path | None = None) -> None:
    """ffmpeg / deno PATH'te değilse bilinen klasörleri ve kayıt defterindeki güncel PATH'i ekle."""
    cur = os.environ.get("PATH", "").split(os.pathsep)
    seen = {c.lower().rstrip("\\/") for c in cur if c}

    def fresh(dirs) -> list[str]:
        out = []
        for d in map(str, dirs):
            k = d.lower().rstrip("\\/")
            if d and k not in seen:
                seen.add(k)
                out.append(d)
        return out

    front = fresh(tool_dirs(root))          # kurulumun kendi araçları önce
    back = fresh(_registry_path())      # kayıt defterindeki yeni girdiler sonra
    if front or back:
        os.environ["PATH"] = os.pathsep.join(front + cur + back)


def prepare_env() -> None:
    """Uygulama/CLI açılışı: ortam hazırlığı (ffmpeg, deno, GPU sırası, UTF-8). Eskiden stt.bat yapıyordu."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # cuda:N == nvidia-smi N
    # başka programların koyduğu kart kısıtlaması (3080 görünmüyordu); kart seçimi stt.toml 'gpus' ile
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("PYTHONUTF8", "1")        # alt süreçler (presser.py vb.) UTF-8 yazsın
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    net_env()
    add_tool_paths()


def net_env() -> None:
    """Model indirmeleri takılmasın, dışarıya kullanım verisi gitmesin (alt süreçlere de geçer).

    - HF'nin yeni indirme yöntemi (Xet) Windows'ta zaman zaman donuyordu: kapalı, klasik HTTP indirme.
    - İndirmelere süre sınırı: bağlantı bu kadar saniye ses vermezse hata verir (sonsuza dek beklemez).
    - pyannote 4 ve huggingface_hub kullanım verisi (telemetri) göndermesin."""
    for k, v in (("HF_HUB_DISABLE_XET", "1"), ("HF_HUB_DOWNLOAD_TIMEOUT", "60"), ("HF_HUB_ETAG_TIMEOUT", "30"),
                 ("HF_HUB_DISABLE_TELEMETRY", "1"), ("HF_HUB_DISABLE_PROGRESS_BARS", "1")):
        os.environ.setdefault(k, v)
    os.environ["PYANNOTE_METRICS_ENABLED"] = "false"


def memory_status() -> dict | None:
    """RAM ve sanal bellek (commit) durumu, GB. Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo."""
    gb = 1024 ** 3
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):  # type: ignore[attr-defined]
            return None
        return {"ram_total": m.ullTotalPhys / gb, "ram_free": m.ullAvailPhys / gb,
                "commit_total": m.ullTotalPageFile / gb, "commit_free": m.ullAvailPageFile / gb}
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
        swap = info.get("SwapTotal", 0)
        return {"ram_total": info["MemTotal"] / gb, "ram_free": info["MemAvailable"] / gb,
                "commit_total": (info["MemTotal"] + swap) / gb,
                "commit_free": (info["MemAvailable"] + info.get("SwapFree", 0)) / gb}
    except Exception:  # noqa: BLE001
        return None


PAGEFILE_HINT = ("Sanal belleği büyütün: Başlat'a 'Gelişmiş sistem ayarlarını görüntüle' yazın → Performans: Ayarlar → "
                 "Gelişmiş → Sanal bellek: Değiştir → 'Tüm sürücüler için disk belleği dosyası boyutunu otomatik "
                 "yönet' işaretli olsun (ya da C: için Sistem tarafından yönetilen boyut) → Tamam → yeniden başlatın.")


def memory_text() -> str:
    m = memory_status()
    if not m:
        return ""
    return (f"RAM {m['ram_total']:.0f} GB (boş {m['ram_free']:.1f} GB), sanal bellek sınırı {m['commit_total']:.0f} GB "
            f"(boş {m['commit_free']:.1f} GB)")


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


def _finite(sec) -> float:
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return 0.0
    return sec if math.isfinite(sec) and sec > 0 else 0.0


def fmt_ts(sec: float, sep: str = ",") -> str:
    sec = _finite(sec)
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_clock(sec: float) -> str:
    sec = int(_finite(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def which(name: str) -> str | None:
    return shutil.which(name)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replace_retry(tmp, path)


def replace_retry(src: Path, dst: Path, tries: int = 5) -> None:
    """os.replace; Windows'ta virüs tarayıcı / arama dizini dosyayı bir anlığına kilitlerse birkaç kez dene."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i + 1 == tries:
                raise
            time.sleep(0.2 * (i + 1))


# Windows süreç bayrakları: konsol penceresi açılmasın / düşük öncelik (bilgisayar akıcı kalsın)
NO_WINDOW = 0x08000000
BELOW_NORMAL = 0x00004000


def popen_kw(low_priority: bool = False) -> dict:
    """subprocess için Windows bayrakları (diğer sistemlerde boş)."""
    if os.name != "nt":
        return {}
    return {"creationflags": NO_WINDOW | (BELOW_NORMAL if low_priority else 0)}


def keep_awake(on: bool) -> None:
    """İş sürerken Windows uyku moduna geçmesin (ekran kapanabilir). Gece kuyruğu uykuda durmasın;
    uykudan dönüşte ekran kartı sürücüsü de sık 'CUDA error: unknown error' veriyor."""
    if os.name != "nt":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))
    except Exception:  # noqa: BLE001
        pass


def console_python() -> str:
    """pythonw.exe altında çalışıyorsak yanındaki python.exe (pip / alt komutlar çıktı verebilsin)."""
    exe = Path(sys.executable)
    py = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
    return str(py if py.exists() else exe)


def read_jsonl(path: Path) -> list[dict]:
    """JSON satırları; bozuk satırlar atlanır, dosya yoksa boş liste."""
    import json
    out = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def write_jsonl_csv(jl: Path, csv_path: Path, rows: list[dict], fields: list[str]) -> None:
    """rows -> jsonl (atomik) + csv (Excel için utf-8-sig; Excel'de açıksa .tmp olarak kalır)."""
    import csv
    import json
    atomic_write_text(jl, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    try:
        replace_retry(tmp, csv_path, tries=3)
    except PermissionError:
        log.warning("%s açık (Excel?); güncel tablo %s olarak kaldı.", csv_path.name, tmp.name)
