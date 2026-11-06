"""1.9.14: oynatma listesi seçimi, ekran sürücüsü çökmesi teşhisi, hızlandırma, kanal basın toplantısı bulucu."""
from pathlib import Path

from stt.config import Config


def test_is_playlist_url():
    from stt.channel import is_playlist_url
    assert is_playlist_url("https://www.youtube.com/playlist?list=PLuvwbYTkUzHcR-UjcwBisPBjH7W_eTA1k")
    assert is_playlist_url("https://www.youtube.com/watch?v=D_SlLqAQ__I&list=PLuvwbYTkUzHcR&index=19")
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc&list=RDabc")         # otomatik Mix
    assert not is_playlist_url("https://www.youtube.com/@ManCity")
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc")
    assert is_playlist_url("https://www.dailymotion.com/playlist/x6hynp")
    assert not is_playlist_url("https://www.dailymotion.com/arsenal")


def _app(tmp_path, monkeypatch, items):
    import stt.channel as ch
    import stt.gui as gui
    monkeypatch.setattr(ch, "list_videos", lambda url, cfg, limit=None: {"title": "Liste", "channel": "Arsenal",
                                                                          "items": [dict(e) for e in items]})
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    app._llm = None
    monkeypatch.setattr(gui.App, "llm", lambda self: type("L", (), {"available": lambda s: False})())

    class St:
        def done_keys(self, keys):
            from stt.sources import Item
            return {Item("url", "https://www.youtube.com/watch?v=a1").compute_key()}
    app._state = lambda: St()
    return app


ITEMS = [{"url": "https://www.youtube.com/watch?v=a1", "id": "a1", "title": "PRESS CONFERENCE | Arteta on Leeds"},
         {"url": "https://www.youtube.com/watch?v=a2", "id": "a2", "title": "MANAGER'S PREVIEW | Pep on Spurs"},
         {"url": "https://www.youtube.com/watch?v=a3", "id": "a3", "title": "MATCHDAY LIVE! | Brighton v Man City"},
         {"url": "https://www.youtube.com/watch?v=a4", "id": "a4", "title": "Women's highlights"}]


def test_playlist_selects_all_unprocessed(tmp_path, monkeypatch):
    """Kullanıcı: oynatma listesinde işlenmemişlerin hepsi her seferinde seçili gelsin (tek tek işaretlemeyeyim)."""
    app = _app(tmp_path, monkeypatch, ITEMS)
    r = app.list_channel("https://www.youtube.com/playlist?list=PLx", {"topic": "basin", "exclude": "women"})
    assert r["playlist"] is True
    m = {e["id"]: (e["match"], e.get("done")) for e in r["items"]}
    assert m == {"a1": (True, True), "a2": (True, False), "a3": (True, False), "a4": (False, False)}  # yalnız "geçerse alma"


def test_channel_keeps_topic_filter(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, ITEMS)
    r = app.list_channel("https://www.youtube.com/@Arsenal", {"topic": "basin"})
    assert r["playlist"] is False
    assert [e["id"] for e in r["items"] if e["match"]] == ["a1", "a2"]   # "MANAGER'S PREVIEW" de basın toplantısı


# ---------------------------------------------------------------- ekran kararması
def test_parse_display_events():
    from stt.gpuguard import describe, parse_events
    txt = ('[{"t":1794000000,"id":4101,"p":"Display","m":"Display driver nvlddmkm stopped responding"},'
           '{"t":1794000100,"id":153,"p":"nvlddmkm","m":""},'
           '{"t":1793990000,"id":41,"p":"Microsoft-Windows-Kernel-Power","m":"The system has rebooted"},'
           '{"t":1794000200,"id":1001,"p":"Windows Error Reporting","m":"Fault bucket LiveKernelEvent_141 ..."},'
           '{"t":1794000300,"id":1001,"p":"Windows Error Reporting","m":"Fault bucket APPCRASH"},'
           '{"t":"bozuk","id":1,"p":"x"}]')
    ev = parse_events(txt)
    assert [e["kind"] for e in ev] == ["reset", "driver", "driver", "driver"]
    assert "3 kez" in describe(ev) and "1 kez beklenmedik" in describe(ev)
    one = parse_events('{"t":1794000000,"id":4101,"p":"Display","m":"x"}')     # tek olay: liste değil nesne
    assert len(one) == 1
    assert parse_events("") == [] and parse_events("<hata>") == []


def test_helper_crash_only_after_helper_used(tmp_path):
    from stt.gpuguard import helper_crash, mark_helper_open
    ev = [{"t": 1000, "kind": "driver"}, {"t": 5000, "kind": "driver"}]
    assert helper_crash(tmp_path, ev) is None                       # yardımcı hiç açılmadı
    mark_helper_open(tmp_path)
    assert helper_crash(tmp_path, ev) is None                       # çökmeler yardımcıdan önce
    import time
    late = [{"t": int(time.time()) + 5, "kind": "driver"}]
    assert helper_crash(tmp_path, late)["t"] == late[0]["t"]


def test_display_guard_turns_helper_off(tmp_path, monkeypatch):
    import threading
    import time

    import stt.gui as gui
    from stt.gpuguard import mark_helper_open
    saved = {}
    monkeypatch.setattr(gui, "save_settings", lambda upd: saved.update(upd))
    monkeypatch.setattr(gui, "load_config", lambda: Config(out_dir=tmp_path, gpus=saved.get("gpus", "yardim")))
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path, gpus="yardim")
    app._restart = threading.Event()
    pushed = []
    app._push = pushed.append

    class P:
        cfg = None
        why = None

        def disable_helper(self, r):
            self.why = r
    app._pipe = P()
    app._state_dir = lambda: tmp_path
    assert app._display_guard([]) is None and not saved                  # çökme yok
    mark_helper_open(tmp_path)
    hit = app._display_guard([{"t": int(time.time()) + 1, "kind": "driver", "id": 4101, "p": "Display", "m": ""}])
    assert hit and saved["gpus"] == "auto" and app._pipe.why == "ekran sürücüsü çöktü"
    assert pushed and pushed[-1]["type"] == "notice" and "Afterburner" in pushed[-1]["text"]
    assert not (tmp_path / "yardimci_acik.json").exists()


def test_disable_helper_parks_and_never_reopens(tmp_path, monkeypatch):
    import time

    from test_v199 import FakeDiarPool, _diar_pipe, _ready
    p, ds = _diar_pipe(tmp_path, monkeypatch, n_diar=3, transcribe=6)
    p._h_probe.clear()
    p.load = {"util": 0, "used_mb": 1500.0, "total_mb": 10240.0}
    p._dispatch()
    p._dispatch()
    assert p._w_on() or p._d_on()
    assert (tmp_path / ".stt" / "yardimci_acik.json").exists()          # açılış zamanı yazıldı
    for w in ("cuda:0", "diar2"):
        if w not in p._h_parked:
            _ready(p, w)
    p.disable_helper("ekran sürücüsü çöktü")
    assert not p._w_on() and not p._d_on()
    for _ in range(5):
        p._h_ready = {}
        p._h_probe.clear()
        p._d_need_since = p._w_need_since = time.time() - 600
        p._dispatch()
    assert not p._w_on() and not p._d_on()
    assert "ekran sürücüsü çöktü" in p._h_note["cuda:0"]


# ---------------------------------------------------------------- hızlandırma
def test_diar_batch_by_card():
    from stt.engines import diar_batch_size
    assert diar_batch_size(0, 24.0) == 32 and diar_batch_size(0, 10.0) == 16 and diar_batch_size(0, 6.0) == 8
    assert diar_batch_size(0, None) == 8 and diar_batch_size(4, 24.0) == 4        # ayarda sayı varsa o
    assert diar_batch_size("x", 24.0) == 32


def test_main_diar_pool_uses_card_batch(tmp_path, monkeypatch):
    import stt.pipeline as pl
    from stt.pipeline import Pipeline
    made = {}

    class WP:
        def __init__(self, devices, ecfg, names=None, **k):
            made.update(ecfg)
            self.devices, self.workers = devices, {n: None for n in names}

        def start(self):
            pass
    monkeypatch.setattr(pl, "WorkerPool", WP)
    p = Pipeline(Config(out_dir=tmp_path))
    p.devices = ["cuda:1"]
    p._ecfg = {"gpu_mem": {0: 10.0, 1: 24.0}, "display_gpus": [0]}
    p._start_diar_pool()
    assert made["diar_batch"] == 32
    p2 = Pipeline(Config(out_dir=tmp_path, diar_batch=8))
    p2.devices = ["cuda:1"]
    p2._ecfg = {"gpu_mem": {1: 24.0}}
    p2._start_diar_pool()
    assert made["diar_batch"] == 8


# ---------------------------------------------------------------- kanalda basın toplantısı bulucu (deneme)
def test_channel_root():
    from stt.channel import channel_root
    assert channel_root("https://www.youtube.com/@ManCity/videos") == "https://www.youtube.com/@ManCity"
    assert channel_root("https://youtube.com/channel/UC123") == "https://www.youtube.com/channel/UC123"
    assert channel_root("https://www.youtube.com/watch?v=x") is None
    assert channel_root("https://www.dailymotion.com/arsenal") is None


def _fake_ytdlp(monkeypatch, calls):
    import stt.fetch as f

    def info(url, cookies=None, flat=False, extra=None, workdir=None):
        calls.append(url)
        if url.endswith("/playlists"):
            return {"entries": [{"title": "Press Conferences 2026/27", "url": "https://www.youtube.com/playlist?list=PLpc"},
                                {"title": "Goals & Highlights", "url": "https://www.youtube.com/playlist?list=PLhl"}]}
        if "PLpc" in url:
            return {"channel": "Man City", "entries": [
                {"id": "v1", "url": "https://www.youtube.com/watch?v=v1", "title": "Pep press conference", "upload_date": "20261101"},
                {"id": "v9", "url": "https://www.youtube.com/watch?v=v9", "title": "Old presser", "upload_date": "20240101"}]}
        if "search?query=press%20conference" in url:
            return {"channel": "Man City", "entries": [
                {"id": "v1", "url": "https://www.youtube.com/watch?v=v1", "title": "Pep press conference"},
                {"id": "v2", "url": "https://www.youtube.com/watch?v=v2", "title": "PRESS CONFERENCE | Pep on Spurs"}]}
        if "manager" in url:
            return {"channel": "Man City", "entries": [
                {"id": "v3", "url": "https://www.youtube.com/watch?v=v3", "title": "MANAGER'S PREVIEW: Pep on Wolves"}]}
        raise RuntimeError("ERROR: HTTP Error 429")
    monkeypatch.setattr(f, "ytdlp_info", info)


def test_find_pressers_searches_whole_channel(monkeypatch):
    from stt.channel import find_pressers, select
    calls = []
    _fake_ytdlp(monkeypatch, calls)
    r = find_pressers("https://www.youtube.com/@ManCity/videos", Config())
    ids = [e["id"] for e in r["items"]]
    assert sorted(ids) == ["v1", "v2", "v3", "v9"]                      # tekrar yok, eski liste videosu da
    assert not any("PLhl" in c for c in calls)                           # basın toplantısı olmayan liste taranmadı
    assert r["sources"]["liste: Press Conferences 2026/27"] == 1          # v1 aramada zaten bulundu (yeni olan sayılır)
    sel = select(r["items"], "basin")
    assert all(e["match"] for e in sel if e["id"] in ("v1", "v2", "v3"))  # "MANAGER'S PREVIEW" de basın toplantısı


def test_find_pressers_only_youtube_channel(monkeypatch):
    import pytest

    from stt.channel import find_pressers
    with pytest.raises(RuntimeError, match="YouTube kanal"):
        find_pressers("https://www.dailymotion.com/arsenal", Config())


def test_list_channel_finder_opt_in(tmp_path, monkeypatch):
    """Varsayılan değil: yalnız kutu işaretliyse kanal taranır."""
    import stt.channel as ch
    import stt.gui as gui
    used = []
    monkeypatch.setattr(ch, "find_pressers", lambda url, cfg, limit=None: used.append("find") or
                        {"title": "Man City", "channel": "Man City", "items": [], "sources": {"arama: x": 0}})
    monkeypatch.setattr(ch, "list_videos", lambda url, cfg, limit=None: used.append("list") or
                        {"title": "Man City", "channel": "Man City", "items": []})
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    monkeypatch.setattr(gui.App, "llm", lambda self: type("L", (), {"available": lambda s: False})())
    app._state = lambda: type("S", (), {"done_keys": lambda s, k: set()})()
    app.list_channel("https://www.youtube.com/@ManCity", {"topic": "basin"})
    r = app.list_channel("https://www.youtube.com/@ManCity", {"topic": "basin", "find": True})
    assert used == ["list", "find"] and r["finder"] is True
    app.list_channel("https://www.youtube.com/playlist?list=PL1", {"topic": "basin", "find": True})
    assert used[-1] == "list"                                            # oynatma listesinde bulucu yok
