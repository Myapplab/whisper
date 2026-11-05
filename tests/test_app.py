"""Uygulama (setup.exe + GUI) için: ortam hazırlığı, ayar normalizasyonu, hata mesajları, yol sınırı."""
import os

from stt.config import Config, load_config, normalize
from stt.fetch import ytdlp_error_summary
from stt.pipeline import Job, Pipeline, worker_hint
from stt.sources import Item
from stt.util import add_tool_paths, prepare_env


def test_tools_on_path_first(tmp_path, monkeypatch):
    (tmp_path / "ffmpeg-8.0-essentials_build" / "bin").mkdir(parents=True)
    (tmp_path / "deno").mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
    add_tool_paths(tmp_path)
    add_tool_paths(tmp_path)  # ikinci çağrı tekrar eklemesin
    parts = os.environ["PATH"].split(os.pathsep)
    assert set(parts[:2]) == {str(tmp_path / "deno"), str(tmp_path / "ffmpeg-8.0-essentials_build" / "bin")}
    assert len(parts) == 4


def test_prepare_env_clears_gpu_restriction(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    prepare_env()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_config_none_and_auto(tmp_path, monkeypatch):
    # GUI 'Kullanma' seçince stt.toml'a "none" yazar: yt-dlp'ye tarayıcı adı olarak gitmemeli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("STT_COOKIES_BROWSER", "STT_OUT"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "stt.toml").write_text("cookies_browser = 'none'\nlang = 'auto'\n", encoding="utf-8")
    cfg = load_config()
    assert cfg.cookies_browser is None and cfg.lang is None
    c = normalize(Config(cookies_browser="Kullanma", lang="en"))
    assert c.cookies_browser is None and c.lang == "en"
    assert normalize(Config(cookies_browser="firefox")).cookies_browser == "firefox"


def test_ytdlp_error_keeps_first_cause():
    errs = [("çerezsiz", "ERROR: [youtube] abc: Sign in to confirm you're not a bot."),
            ("--cookies-from-browser chrome", "PermissionError: Could not copy Chrome cookie database.")]
    msg = ytdlp_error_summary(errs)
    assert msg.startswith("[youtube] abc: Sign in to confirm")
    assert "Could not copy Chrome cookie" in msg and "Firefox" in msg
    assert ytdlp_error_summary([]) == ""


def test_worker_hint():
    assert "RAM" in worker_hint("RuntimeError: mkl_malloc: failed to allocate memory")
    assert "Ekran kartı belleği" in worker_hint("RuntimeError: CUDA failed with error out of memory")
    assert "STT-Kurulum" in worker_hint("RuntimeError: Library cublas64_12.dll is not found")
    assert worker_hint("x" * 1000) == "x" * 400


def _pipe(tmp_path, name="cikti"):
    return Pipeline(Config(out_dir=tmp_path / name))


def test_out_paths_fit_windows_limit(tmp_path):
    pipe = _pipe(tmp_path, "a" * 60)
    try:
        title = "PRESS CONFERENCE " + "çok uzun bir video başlığı " * 12
        job = Job(Item("file", "x.mp4", title=title, key="f_" + "0" * 64), tmp_path)
        job.info = {"title": title}
        d, base = pipe._out_paths(job, {})
        longest = d / f"{base} - ozet.md.tmp"
        assert len(str(longest)) + len(" [abcdef]") <= 260
        assert d.is_dir() and len(base) >= 20
    finally:
        pipe.close()


class _FakePool:
    workers = {"cuda:0": None, "cuda:1": None}

    def alive(self):
        return []


def test_all_workers_fatal_sets_reason(tmp_path):
    pipe = _pipe(tmp_path)
    events = []
    pipe.events = events.append
    pipe.pool = _FakePool()
    try:
        pipe._on_msg({"type": "fatal", "wid": "cuda:0", "error": "RuntimeError: CUDA failed with error out of memory"})
        assert pipe.failed_reason is None  # biri hâlâ yükleniyor olabilir
        pipe._on_msg({"type": "fatal", "wid": "cuda:1", "error": "RuntimeError: mkl_malloc: failed to allocate memory"})
        assert pipe.failed_reason and pipe.failed_reason.startswith("Model yüklenemedi")
        assert events[-1]["type"] == "fatal"
    finally:
        pipe.pool = None
        pipe.close()


def test_gui_engine_error_visible(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._push({"type": "fatal", "error": "Model yüklenemedi: RAM"})
    assert app._engine_state["state"] == "error"
    f = tmp_path / "a.mp4"
    f.write_bytes(b"0")
    r = app.start("", [str(f)], {})
    assert r["count"] == 1 and r["warning"].startswith("Motor çalışmıyor")
    assert app.start("", [], {})["warning"] == "Link ya da dosya bulunamadı."

