"""Uygulama içi güncelleme: GitHub'daki (herkese açık) Myapplab/whisper deposunun main dalından.

Yeni .exe indirilmez: yalnızca uygulamanın Python dosyaları değişir. Bunları Windows'un imzalı Python'u
çalıştırdığı için Smart App Control engellemez; zip / PowerShell / "Engellemeyi kaldır" gerekmez.

Adımlar: sürümü oku -> zip'i indir -> geçici klasörde DENE (içe aktarma testi) -> eski dosyaları yedekle ->
yeni dosyaları yaz -> kütüphaneler değiştiyse pip -> hata olursa yedeği geri koy.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from . import __version__
from .config import TOOL_DIR
from .util import log

REPO = os.environ.get("STT_UPDATE_REPO", "Myapplab/whisper")
BRANCH = os.environ.get("STT_UPDATE_BRANCH", "main")
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
ZIP_URL = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"

ROOT_FILES = ["pyproject.toml", "constraints.txt", "requirements.txt", "README.md", "BENI_OKU.txt",
              "stt.toml.example"]
DEP_FILES = ["pyproject.toml", "constraints.txt"]  # bunlar değişirse pip çalışır
BACKUP = "guncelleme-yedek"


def parse_version(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3]) or (0,)


def _get(url: str, timeout: float = 20) -> bytes:
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": f"stt/{__version__}", "Cache-Control": "no-cache"})
    if r.status_code == 404:
        raise RuntimeError("Güncelleme deposuna ulaşılamadı (404). GitHub'da Myapplab/whisper herkese açık mı?")
    r.raise_for_status()
    return r.content


def latest_version() -> str:
    txt = _get(RAW + "stt/__init__.py", timeout=10).decode("utf-8", "replace")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', txt)
    if not m:
        raise RuntimeError("Sürüm bilgisi okunamadı")
    return m.group(1)


def check() -> dict:
    try:
        latest = latest_version()
    except Exception as e:  # noqa: BLE001
        return {"current": __version__, "latest": None, "available": False, "error": str(e)[:300]}
    return {"current": __version__, "latest": latest, "available": parse_version(latest) > parse_version(__version__),
            "error": None}


def _digest(root: Path, names: list[str]) -> str:
    h = hashlib.sha256()
    for n in names:
        p = root / n
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()


def _python_console() -> str:
    exe = Path(sys.executable)
    py = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
    return str(py if py.exists() else exe)


def _extract(data: bytes, dest: Path) -> Path:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():  # zip içinden dışarı yazma girişimi olmasın
            target = (dest / m.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise RuntimeError(f"Geçersiz zip girdisi: {m.filename}")
        z.extractall(dest)
    tops = [p for p in dest.iterdir() if p.is_dir()]
    root = tops[0] if len(tops) == 1 else dest
    if not (root / "stt" / "__init__.py").exists():
        raise RuntimeError("İndirilen pakette uygulama dosyaları yok")
    return root


def _validate(root: Path) -> str:
    """Yeni kodu ayrı bir süreçte içe aktar: sözdizimi / eksik modül hatası varsa güncelleme yapılmaz."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import stt, stt.gui, stt.pipeline, stt.cli, stt.updater; "
            "print(stt.__version__)")
    r = subprocess.run([_python_console(), "-c", code, str(root)], capture_output=True, text=True, timeout=180,
                       cwd=str(root), encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("Yeni sürüm denenirken hata (güncelleme yapılmadı): " + (r.stderr or r.stdout)[-600:])
    return r.stdout.strip().splitlines()[-1]


def _copy_tree(src: Path, dst: Path, backup: Path) -> None:
    for f in src.rglob("*"):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src)
        t = dst / rel
        if t.exists():
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(t, backup / rel)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)


def _restore(backup: Path, dst: Path) -> None:
    for f in backup.rglob("*"):
        if f.is_file():
            t = dst / f.relative_to(backup)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)


def apply(progress=lambda stage, text: None, tool_dir: Path = TOOL_DIR, data: bytes | None = None) -> dict:
    """Güncellemeyi uygula. Dönüş: {ok, version, deps_changed} ya da {ok: False, error}."""
    tool_dir = Path(tool_dir)
    if (tool_dir / ".git").exists() and not os.environ.get("STT_UPDATE_FORCE"):
        return {"ok": False, "error": "Bu bir geliştirici klasörü (git); güncelleme için git pull kullanın."}
    backup = tool_dir / BACKUP
    try:
        progress("download", "Yeni sürüm indiriliyor…")
        data = data if data is not None else _get(ZIP_URL, timeout=120)
        with tempfile.TemporaryDirectory(prefix="stt-guncelleme-") as td:
            root = _extract(data, Path(td))
            progress("validate", "Yeni sürüm deneniyor…")
            new_version = _validate(root)
            deps_changed = _digest(root, DEP_FILES) != _digest(tool_dir, DEP_FILES)
            progress("copy", "Dosyalar yazılıyor…")
            shutil.rmtree(backup, ignore_errors=True)
            backup.mkdir(parents=True)
            try:
                _copy_tree(root / "stt", tool_dir / "stt", backup / "stt")
                for n in ROOT_FILES:
                    if (root / n).exists():
                        _copy_tree_file(root / n, tool_dir / n, backup / n)
                if (root / "installer" / "kur.ps1").exists() and (tool_dir / "kur.ps1").exists():
                    _copy_tree_file(root / "installer" / "kur.ps1", tool_dir / "kur.ps1", backup / "kur.ps1")
                if deps_changed:
                    progress("deps", "Kütüphaneler güncelleniyor (birkaç dakika sürebilir)…")
                    r = subprocess.run([_python_console(), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                                        "-c", "constraints.txt", "-e", ".[diarize]"], cwd=str(tool_dir),
                                       capture_output=True, text=True, timeout=3600, encoding="utf-8", errors="replace")
                    if r.returncode != 0:
                        raise RuntimeError("Kütüphaneler kurulamadı: " + (r.stderr or r.stdout)[-600:])
            except Exception:
                log.exception("Güncelleme yazılırken hata; eski sürüm geri yükleniyor")
                _restore(backup, tool_dir)
                raise
        progress("done", f"Güncellendi: {new_version}")
        log.info("Güncellendi: %s -> %s", __version__, new_version)
        return {"ok": True, "version": new_version, "deps_changed": deps_changed}
    except Exception as e:  # noqa: BLE001
        log.error("Güncelleme başarısız: %s", e)
        return {"ok": False, "error": str(e)[:800]}


def _copy_tree_file(src: Path, dst: Path, backup: Path) -> None:
    if dst.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, backup)
    shutil.copy2(src, dst)


def relaunch() -> None:
    """Uygulamayı yeniden aç (yeni süreç eski kapanana kadar tek kopya kilidini bekler)."""
    env = {**os.environ, "STT_RESTART": "1"}
    flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, "-m", "stt.gui"], cwd=str(TOOL_DIR), env=env, creationflags=flags,
                     close_fds=True)
