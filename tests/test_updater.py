"""Uygulama içi güncelleme: sürüm karşılaştırma, deneme (bozuk sürüm reddedilir), dosya yazma, geri alma."""
import io
import zipfile
from pathlib import Path

import pytest

from stt import updater

REPO = Path(__file__).resolve().parent.parent


def _make_zip(version: str, broken: bool = False, extra_dep: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in (REPO / "stt").rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                data = f.read_bytes()
                rel = f.relative_to(REPO).as_posix()
                if rel == "stt/__init__.py":
                    data = f'"""stt"""\n__version__ = "{version}"\n'.encode()
                if broken and rel == "stt/pipeline.py":
                    data += b"\ndef bozuk(:\n"
                z.writestr("whisper-main/" + rel, data)
        for n in updater.ROOT_FILES:
            if (REPO / n).exists():
                data = (REPO / n).read_bytes()
                if extra_dep and n == "constraints.txt":
                    data += b"\n# degisti\n"
                z.writestr("whisper-main/" + n, data)
    return buf.getvalue()


def _install(tmp_path: Path) -> Path:
    d = tmp_path / "STT"
    (d / "stt").mkdir(parents=True)
    for f in (REPO / "stt").rglob("*.py"):
        t = d / f.relative_to(REPO)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(f.read_bytes())
    (d / "stt" / "__init__.py").write_text('__version__ = "1.2.0"\n', encoding="utf-8")
    (d / "stt" / "eski_modul.py").write_text("x = 1\n", encoding="utf-8")
    for n in updater.DEP_FILES:
        (d / n).write_bytes((REPO / n).read_bytes())
    (d / "stt.toml").write_text("hf_token = 'gizli'\n", encoding="utf-8")
    return d


def test_parse_version():
    assert updater.parse_version("1.10.0") > updater.parse_version("1.9.3") > updater.parse_version("1.9")


def test_check_compares(monkeypatch):
    monkeypatch.setattr(updater, "latest_version", lambda: "99.0.0")
    assert updater.check()["available"] is True
    monkeypatch.setattr(updater, "latest_version", lambda: "0.1.0")
    assert updater.check()["available"] is False


def test_apply_updates_files_and_keeps_settings(tmp_path):
    d = _install(tmp_path)
    stages = []
    r = updater.apply(lambda s, t: stages.append(s), tool_dir=d, data=_make_zip("9.9.9"))
    assert r["ok"] and r["version"] == "9.9.9" and r["deps_changed"] is False
    assert '9.9.9' in (d / "stt" / "__init__.py").read_text()
    assert (d / "stt" / "gui.html").exists() and (d / "stt" / "updater.py").exists()
    assert (d / "stt.toml").read_text() == "hf_token = 'gizli'\n"          # ayarlar korunur
    assert (d / updater.BACKUP / "stt" / "__init__.py").read_text().count("1.2.0")  # yedek alındı
    assert stages[:3] == ["download", "validate", "copy"] and stages[-1] == "done"


def test_broken_update_is_rejected(tmp_path):
    d = _install(tmp_path)
    before = (d / "stt" / "pipeline.py").read_bytes()
    r = updater.apply(tool_dir=d, data=_make_zip("9.9.9", broken=True))
    assert not r["ok"] and "denenirken" in r["error"]
    assert (d / "stt" / "pipeline.py").read_bytes() == before
    assert "1.2.0" in (d / "stt" / "__init__.py").read_text()


def test_failed_pip_restores_old_files(tmp_path, monkeypatch):
    d = _install(tmp_path)
    before = (d / "constraints.txt").read_bytes()

    class R:
        returncode, stdout, stderr = 1, "", "pip patladi"
    real_run = updater.subprocess.run
    monkeypatch.setattr(updater.subprocess, "run",
                        lambda cmd, **kw: R() if "pip" in cmd else real_run(cmd, **kw))
    r = updater.apply(tool_dir=d, data=_make_zip("9.9.9", extra_dep=True))
    assert not r["ok"] and "pip patladi" in r["error"]
    assert (d / "constraints.txt").read_bytes() == before
    assert "1.2.0" in (d / "stt" / "__init__.py").read_text()


def test_git_checkout_is_not_overwritten(tmp_path):
    d = _install(tmp_path)
    (d / ".git").mkdir()
    assert not updater.apply(tool_dir=d, data=_make_zip("9.9.9"))["ok"]


def test_zip_path_escape_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../kotu.txt", "x")
    with pytest.raises(RuntimeError):
        updater._extract(buf.getvalue(), tmp_path)
