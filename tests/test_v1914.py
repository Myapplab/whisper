"""1.9.14: oynatma listesi seçimi, ekran sürücüsü çökmesi teşhisi, hızlandırma, kanal basın toplantısı bulucu."""
import threading
import time
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
    info = {0: {"name": "RTX 3080", "gb": 10, "display": True}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}
    mark_helper_open(tmp_path)
    from stt.gpuguard import mark_helper_closed
    mark_helper_closed(tmp_path)
    assert app._display_guard([], info) is None and not saved            # çökme yok
    mark_helper_open(tmp_path)
    hit = app._display_guard([{"t": int(time.time()) + 1, "kind": "driver", "id": 4101, "p": "Display", "m": ""}], info)
    assert hit and saved["gpus"] == "auto" and app._pipe.why == "ekran sürücüsü çöktü"
    assert pushed and pushed[-1]["type"] == "notice" and "Afterburner" in pushed[-1]["text"]
    from stt.gpuguard import _read_windows
    assert _read_windows(tmp_path) == [] and (tmp_path / "yardimci_acik.json").exists()   # yeni başlangıç


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
    # 1.9.16: kararlılık doğrulanana dek otomatik değer yine 8 (kullanıcıda bilgisayar yeniden başlıyordu)
    assert diar_batch_size(0, 24.0) == 8 and diar_batch_size(0, None) == 8
    assert diar_batch_size(32, 24.0) == 32 and diar_batch_size("x", 24.0) == 8   # ayarda sayı varsa o


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
    assert made["diar_batch"] == 8
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
    monkeypatch.setattr(ch, "find_in_channel", lambda url, cfg, limit=None, search=None, topic=None: used.append("find") or
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


# ---------------------------------------------------------------- indirme siteleri
ERRS = {  # yt-dlp'nin gerçek hata metinleri (yt-dlp kaynağı / sorun takipçisi)
    "vk_login": "ERROR: [vk] -1_2: This video is only available for registered users. Use --cookies, "
                "--cookies-from-browser, --username and --password, --netrc-cmd, or --netrc (vk) to provide account "
                "credentials. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp",
    "dm_geo": "ERROR: [dailymotion] x8abc: Video geoblocked by the owner\nYou might want to use a VPN or a proxy "
              "server (with --proxy) to workaround.",
    "ok_player": "ERROR: [Odnoklassniki] 9987641379464: Unable to extract player; please report this issue on  "
                 "https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the appropriate issue template.",
    "rutube_404": "ERROR: [rutube] c5f0: Unable to download options JSON: HTTP Error 404: Not Found",
    "vk_followers": "ERROR: [vk] Video only available to followers",
    "mailru_type": "ERROR: expected string or bytes-like object, got 'NoneType'",
    "cloudflare": 'ERROR: [generic] Got HTTP Error 403 caused by Cloudflare anti-bot challenge; try again with  '
                  '--extractor-args "generic:impersonate"',
    "imp_missing": 'Impersonate target "chrome" is not available. Use --list-impersonate-targets to see available '
                   'targets. You may be missing dependencies required to support this target.',
    "chrome_cookie": "ERROR: Could not copy Chrome cookie database. See https://github.com/yt-dlp/yt-dlp/issues/7271",
}


def test_explain_error_site_messages():
    """İnceleme: giriş hatası "çerez okunamadı", bölge kısıtı İngilizce, Cloudflare "curl_cffi yok" görünüyordu."""
    from stt.fetch import explain_error
    e = {k: explain_error(v).split("  (ayrıntı")[0] for k, v in ERRS.items()}
    assert e["vk_login"].startswith("Site giriş istiyor")
    assert "ülkeye kapalı" in e["dm_geo"]
    assert e["ok_player"].startswith("ok.ru bu videoyu vermedi")
    assert e["rutube_404"].startswith("Rutube bu videoyu oynatmıyor")
    assert "takipçilere" in e["vk_followers"]
    assert e["mailru_type"].startswith("Site videoyu vermedi")
    assert "Cloudflare" in e["cloudflare"] and "curl_cffi" not in e["cloudflare"]
    assert "curl_cffi" in e["imp_missing"]
    assert e["chrome_cookie"].startswith("Tarayıcı çerezleri okunamadı")


def test_quiet_logger_accepts_yt_dlp_kwargs():
    """yt-dlp çerez okuyucu warning(msg, only_once=True) çağırıyor; eskiden TypeError ile bütün okuma düşüyordu."""
    from stt.fetch import _QuietLogger
    lg = _QuietLogger()
    lg.warning("cannot decrypt v10 cookies: no key found", only_once=True)
    lg.debug("x", once=True)
    lg.info("x")
    lg.error("x", tb=None)


def test_site_url_variants():
    from stt.fetch import site_url_variants
    assert site_url_variants("https://m.ok.ru/video/123456") == [
        "https://ok.ru/video/123456", "https://ok.ru/videoembed/123456", "https://m.ok.ru/video/123456"]
    assert site_url_variants("https://ok.ru/video/123456") == ["https://ok.ru/video/123456",
                                                               "https://ok.ru/videoembed/123456"]
    h = "a" * 32
    assert site_url_variants(f"https://rutube.ru/shorts/{h}/")[0] == f"https://rutube.ru/video/{h}/"
    vk = "https://vk.com/video_ext.php?oid=-22822305&id=456242110&hash=abc"
    assert site_url_variants(vk) == ["https://vkvideo.ru/video-22822305_456242110", vk]
    assert site_url_variants("https://www.dailymotion.com/video/x8abc") == ["https://www.dailymotion.com/video/x8abc"]


def test_variants_tried_in_order_and_best_error(tmp_path, monkeypatch):
    import pytest

    import stt.fetch as f
    tried = []

    def dl(u, *a, **k):
        tried.append(u)
        if "videoembed" in u:
            raise f.FetchError("ERROR: [Odnoklassniki] 1: This video is not available in your country")
        raise f.FetchError("ERROR: [Odnoklassniki] 1: Unable to extract player")
    monkeypatch.setattr(f, "ytdlp_download", dl)
    with pytest.raises(f.FetchError) as ei:
        f._ytdlp_to_wav("https://m.ok.ru/video/1", tmp_path, tmp_path / "s.wav", Config())
    assert tried == ["https://ok.ru/video/1", "https://ok.ru/videoembed/1", "https://m.ok.ru/video/1"]
    assert "not available in your country" in str(ei.value)          # genel "unable to extract" değil, asıl sebep


def test_variant_transient_error_stops(tmp_path, monkeypatch):
    import pytest

    import stt.fetch as f
    tried = []

    def dl(u, *a, **k):
        tried.append(u)
        raise f.FetchError("HTTP Error 503", transient=True)
    monkeypatch.setattr(f, "ytdlp_download", dl)
    with pytest.raises(f.FetchError):
        f._ytdlp_to_wav("https://m.ok.ru/video/1", tmp_path, tmp_path / "s.wav", Config())
    assert len(tried) == 1                                               # ağ hatası: biçim değiştirmek boşuna


def test_find_embeds_mailru_and_og_video():
    from stt.fetch import find_embeds
    page = ('<meta property="og:video" content="https://my.mail.ru/video/embed/123456789">'
            '<script>{"metaUrl":"//my.mail.ru/+/video/meta/987654321"}</script>')
    out = find_embeds(page, "https://my.mail.ru/mail/user/video/1/2.html")
    assert "https://my.mail.ru/video/embed/123456789" in out
    assert "https://my.mail.ru/+/video/meta/987654321" in out


def test_collection_urls_new_sites():
    from stt.channel import is_collection_url
    assert is_collection_url("https://vkvideo.ru/@arsenal")
    assert is_collection_url("https://rutube.ru/channel/12345/")
    assert is_collection_url("https://vimeo.com/user12345")
    assert not is_collection_url("https://vkvideo.ru/video-1_2")


def test_okru_channel_listing_clear_message():
    import pytest

    from stt.channel import list_videos
    with pytest.raises(RuntimeError, match="desteklenmiyor"):
        list_videos("https://ok.ru/video/c123", Config())


def test_no_fixed_user_agent_for_ytdlp(tmp_path):
    """Taklitte sabit "Chrome/130" kimliği taklidin parmak iziyle çelişiyordu: yt-dlp'ye kimlik verilmez."""
    from stt.fetch import _ytdlp_opts
    assert "http_headers" not in _ytdlp_opts(tmp_path, {})


# ---------------------------------------------------------------- önden hazırlık sınırı
def test_prep_waits_when_many_prepared(tmp_path, monkeypatch):
    """100 video önden indirilmesin: en çok PREP_AHEAD iş hazırlanmış / çevriliyor olsun."""
    import stt.pipeline as pl
    from stt.pipeline import Job, Pipeline
    from stt.sources import Item
    p = Pipeline(Config(out_dir=tmp_path))
    for i in range(pl.PREP_AHEAD):
        it = Item("url", f"https://x/{i}", key=f"u_{i}")
        j = Job(it, tmp_path / f"w{i}", status="run")
        t = {"id": f"u_{i}:t:0000", "kind": "transcribe", "job": it.key}
        p.tasks[t["id"]] = t
        j.pending = {t["id"]}
        p.jobs[it.key] = j
    assert p._prep_ahead() == pl.PREP_AHEAD and not p._pending_ahead_ok()
    # metni bitip yalnız konuşmacı ayırma bekleyen iş sayılmaz (onun sesi zaten hazır, GPU'yu bekletmesin)
    j0 = p.jobs["u_0"]
    d = {"id": "u_0:d", "kind": "diarize", "job": "u_0"}
    p.tasks[d["id"]] = d
    j0.pending = {d["id"]}
    assert p._pending_ahead_ok()
    # durdurulunca bekleme biter
    it = Item("url", "https://x/new", key="u_new")
    it.extra["_cancel"] = True
    p.jobs["u_0"].pending = {"u_0:t:0000"}
    assert p._wait_prep_turn(it) is False


def test_prepare_does_not_download_while_full(tmp_path, monkeypatch):
    import threading
    import time

    import stt.pipeline as pl
    from stt.pipeline import Job, Pipeline
    from stt.sources import Item
    p = Pipeline(Config(out_dir=tmp_path))
    monkeypatch.setattr(pl, "PREP_AHEAD", 1)
    it0 = Item("url", "https://x/0", key="u_0")
    j = Job(it0, tmp_path / "w0", status="prep")
    p.jobs["u_0"] = j
    called = []
    monkeypatch.setattr(pl, "prepare_audio", lambda item, work, cfg: called.append(item.key) or (_ for _ in ()).throw(
        pl.SkipItem("test")))
    it = Item("url", "https://x/1", key="u_1")
    it.extra["uid"] = "a"
    th = threading.Thread(target=p._prepare, args=(it,), daemon=True)
    th.start()
    time.sleep(1.5)
    assert called == []                       # sıra gelmedi: indirmeye başlamadı
    j.status = "done"
    th.join(5)
    assert called == ["u_1"]


# ---------------------------------------------------------------- bağımsız inceleme bulguları (1.9.14)
def _blocker(p, pl, tmp_path, key="u_block"):
    from stt.pipeline import Job
    from stt.sources import Item
    it = Item("url", "https://x/block", key=key)
    j = Job(it, tmp_path / "wb", status="run")
    t = {"id": f"{key}:t:0000", "kind": "transcribe", "job": key}
    p.tasks[t["id"]] = t
    j.pending = {t["id"]}
    p.jobs[key] = j
    return j


def test_duplicate_copy_not_reprocessed_after_wait(tmp_path, monkeypatch):
    """Same video queued twice: copy B passes the done-check BEFORE waiting; copy A finishes while B waits;
    B then re-downloads/re-transcribes A's video (old code: B was 'duplicate' immediately)."""
    import stt.pipeline as pl
    from stt.pipeline import Pipeline
    from stt.sources import Item
    monkeypatch.setattr(pl, "PREP_AHEAD", 1)
    p = Pipeline(Config(out_dir=tmp_path))
    a = _blocker(p, pl, tmp_path, key="u_dup")          # copy A: transcribing (counts as 'ahead')
    called = []
    monkeypatch.setattr(pl, "prepare_audio", lambda item, work, cfg: called.append(item.key) or
                        (_ for _ in ()).throw(pl.SkipItem("stop here")))
    b = Item("url", "https://x/dup", key="u_dup")
    b.extra["uid"] = "b"
    th = threading.Thread(target=p._prepare, args=(b,), daemon=True)
    th.start()
    time.sleep(1.5)
    assert called == []
    # A finishes normally
    out = tmp_path / "out_a"
    out.mkdir()
    p.state.set("u_dup", source="https://x/dup", status="done", out_dir=str(out))
    a.status = "done"
    th.join(5)
    assert called == [], "kopya B, A biterken yeniden indirildi"


def test_live_stream_not_stuck_behind_prep_wait(tmp_path, monkeypatch):
    """prep_workers=2: two normal items block both prep threads in _wait_prep_turn; a live stream added after
    them never reaches _prepare (its 'no wait' branch is inside the busy thread pool)."""
    import stt.pipeline as pl
    from stt.pipeline import Pipeline
    from stt.sources import Item
    monkeypatch.setattr(pl, "PREP_AHEAD", 1)
    p = Pipeline(Config(out_dir=tmp_path))
    _blocker(p, pl, tmp_path)
    started = []
    monkeypatch.setattr(p, "_start_live", lambda item, job, info: started.append(item.source))
    monkeypatch.setattr(pl, "prepare_audio", lambda item, work, cfg: (_ for _ in ()).throw(pl.SkipItem("x")))
    for i in range(2):
        p.submit(Item("url", f"https://x/v{i}"))
    live = Item("url", "https://x/live")
    live.extra["live"] = True
    p.submit(live)
    end = time.time() + 3
    while not started and time.time() < end:
        time.sleep(0.05)
    try:
        assert started == ["https://x/live"], "live stream did not start within 3 s"
    finally:
        p.cancel_all()
        p.prep.shutdown(wait=True)


def test_close_releases_waiting_prep_thread(tmp_path, monkeypatch):
    """CLI: Ctrl+C -> pipe.close() (pools closed, jobs still 'run'); a prep thread waiting in _wait_prep_turn never
    leaves, and the interpreter joins ThreadPoolExecutor threads at exit -> process hangs."""
    import stt.pipeline as pl
    from stt.pipeline import Pipeline
    from stt.sources import Item
    monkeypatch.setattr(pl, "PREP_AHEAD", 1)
    p = Pipeline(Config(out_dir=tmp_path))
    _blocker(p, pl, tmp_path)
    p.submit(Item("url", "https://x/v0"))
    time.sleep(1.2)
    p.close()
    end = time.time() + 3
    while time.time() < end and [t for t in threading.enumerate() if t.name.startswith("hazirlik")]:
        time.sleep(0.05)
    alive = [t for t in threading.enumerate() if t.name.startswith("hazirlik")]
    try:
        assert not alive, f"prep thread still waiting after close(): {alive}"
    finally:
        p.cancel_all()


def test_cloudflare_without_curl_cffi_says_update():
    """yt-dlp generic extractor, impersonation dependency missing (or incompatible curl_cffi -> impersonate_target()
    now returns None): old text said 'install curl_cffi (press Update)'; new text says impersonation was TRIED."""
    from stt.fetch import explain_error, ytdlp_error_summary
    raw = ('ERROR: [generic] Got HTTP Error 403 caused by Cloudflare anti-bot challenge; see  '
           'https://github.com/yt-dlp/yt-dlp#impersonation  for how to install the required impersonation '
           'dependency, and try again with  --extractor-args "generic:impersonate"')
    msg = explain_error("İndirilemedi. yt-dlp: " + ytdlp_error_summary([("çerezsiz", raw)]))
    assert "curl_cffi" in msg.split("(ayrıntı")[0], msg


def test_guard_ignores_old_marker_format(tmp_path):
    """Mark keeps the FIRST helper-open time forever; any later driver event (game TDR with STT closed, or an
    nvlddmkm Xid from a CUDA fault on the 3090) disables the helper and the notice says it happened 'while used'."""
    import json, time as _t
    from stt.gpuguard import helper_crash, parse_events
    (tmp_path / "yardimci_acik.json").write_text(json.dumps({"first": int(_t.time()) - 5 * 86400,
                                                            "last": int(_t.time()) - 5 * 86400}))
    ev = parse_events(json.dumps({"t": int(_t.time()) - 86400, "id": 13, "p": "nvlddmkm",
                                  "m": "\\Device\\Video5 Graphics Exception: ESR 0x404490=0x80000002"}))
    # 1.9.14'ün eski biçimi ("since" yok): "son 3 gün" kuralı bir kez işler (1.9.13/14'teki kararmalar sayılsın)
    assert helper_crash(tmp_path, ev) is not None
    from stt.gpuguard import clear_mark
    clear_mark(tmp_path)
    assert helper_crash(tmp_path, ev) is None


def test_guard_window_new_format(tmp_path):
    """Yalnız yardımcı kartın açık olduğu aralıktaki çökme sayılır; kapanışı yazılmamış aralık (bilgisayar o
    sırada kapandı) sayılır."""
    import time as _t

    from stt.gpuguard import clear_mark, helper_crash, mark_helper_closed, mark_helper_open
    now = int(_t.time())
    clear_mark(tmp_path)                                                               # mod seçildi (since = şimdi)
    mark_helper_open(tmp_path)
    mark_helper_closed(tmp_path)
    assert helper_crash(tmp_path, [{"t": now + 3600, "kind": "driver"}]) is None     # kapandıktan 1 saat sonra
    assert helper_crash(tmp_path, [{"t": now, "kind": "driver"}])                     # açıkken
    mark_helper_open(tmp_path)                                                         # yeniden açıldı, kapanmadı
    assert helper_crash(tmp_path, [{"t": now + 5, "kind": "reset"}])


def test_cloudflare_message_with_curl_cffi(tmp_path):
    from stt.fetch import explain_error
    m = explain_error('ERROR: [generic] Got HTTP Error 403 caused by Cloudflare anti-bot challenge; try again with  '
                      '--extractor-args "generic:impersonate"').split("  (ayrıntı")[0]
    assert "Cloudflare" in m


def test_vk_article_not_collection():
    from stt.channel import is_collection_url
    assert not is_collection_url("https://vk.com/@arsenal-news")
    assert is_collection_url("https://vk.com/video/@arsenal")



# ---------------------------------------------------------------- 1.9.15: 3080 kararması (5 kez üst üste)
INFO2 = {0: {"name": "RTX 3080", "gb": 10, "display": True}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}


def test_guard_counts_legacy_crashes_without_marker(tmp_path):
    """1.9.13 işaret yazmıyordu: dosya hiç yoksa son 3 gündeki çökmeler sayılır (kullanıcıda 5 kez kararma)."""
    import time as _t

    from stt.gpuguard import clear_mark, helper_crash
    now = int(_t.time())
    ev = [{"t": now - 3600, "kind": "reset"}, {"t": now - 10 * 86400, "kind": "driver"}]
    assert helper_crash(tmp_path, ev)["t"] == now - 3600
    assert helper_crash(tmp_path, [{"t": now - 10 * 86400, "kind": "driver"}]) is None
    from stt.gpuguard import mark_helper_open
    mark_helper_open(tmp_path)                    # inceleme: motor açılışta dosyayı yazsa da eski karar korunur
    assert helper_crash(tmp_path, ev)["t"] == now - 3600
    clear_mark(tmp_path)                                                   # kullanıcı modu yeniden seçti
    assert helper_crash(tmp_path, ev) is None


def test_display_card_used_modes():
    from stt.gpuguard import display_card_used
    assert display_card_used("yardim", INFO2) and display_card_used("all", INFO2) and display_card_used("0,1", INFO2)
    assert not display_card_used("auto", INFO2) and not display_card_used("1", INFO2)
    free = {0: {"name": "RTX 3080", "gb": 10, "display": False}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}
    assert not display_card_used("all", free)                              # monitör anakartta: iki kart da serbest


def test_guard_also_protects_all_mode(tmp_path, monkeypatch):
    import threading

    import stt.gui as gui
    saved = {}
    monkeypatch.setattr(gui, "save_settings", lambda upd: saved.update(upd))
    monkeypatch.setattr(gui, "load_config", lambda: Config(out_dir=tmp_path, gpus=saved.get("gpus", "all")))
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path, gpus="all")
    app._restart = threading.Event()
    app._push = lambda e: None
    app._pipe = None
    app._state_dir = lambda: tmp_path
    hit = app._display_guard([{"t": int(time.time()) - 600, "kind": "reset", "id": 41, "p": "x", "m": ""}], INFO2)
    assert hit and saved["gpus"] == "auto"


def test_igpu_advice():
    from stt.engines import igpu_advice
    assert igpu_advice(None, INFO2) is None
    ok, txt = igpu_advice([("Intel(R) UHD Graphics 630", True), ("NVIDIA GeForce RTX 3080", False),
                           ("NVIDIA GeForce RTX 3090", False)],
                          {0: {"name": "RTX 3080", "display": False}, 1: {"name": "RTX 3090", "display": False}})
    assert ok and "iki NVIDIA" in txt
    ok, txt = igpu_advice([("Intel(R) UHD Graphics 630", False), ("NVIDIA GeForce RTX 3080", True)], INFO2)
    assert not ok and "anakarttaki" in txt and "RTX 3080" in txt
    ok, txt = igpu_advice([("NVIDIA GeForce RTX 3080", True), ("NVIDIA GeForce RTX 3090", False)], INFO2)
    assert not ok and "BIOS" in txt


def test_low_ram_keeps_3090_transcribing(tmp_path, monkeypatch):
    """TARAMA B-1: ayırma RAM beklerken (eskiden işçinin içinde, 10 dk'ya kadar) 3090'da metin de duruyordu."""
    import stt.util as u
    from test_stability import FakePool, _job, _pipe
    p = _pipe(tmp_path, ("cuda:1",))
    p.pool._spec = {"cuda:1": "cuda:1"}
    job, tasks = _job(p, tmp_path, n=3)
    # metni bitmiş başka bir işin ayırması sırada (normalde "ayırmanın sırası" der, metin bekler)
    from stt.pipeline import Job
    from stt.sources import Item
    it = Item("url", "https://x/d", key="u_d")
    jd = Job(it, tmp_path / "wd", status="run", info={"duration_s": 3600})
    d = {"id": "u_d:d", "kind": "diarize", "job": "u_d"}
    p.tasks[d["id"]] = d
    jd.pending = {d["id"]}
    p.jobs["u_d"] = jd
    p.dbacklog.append(d)
    p.dpool = FakePool(["diar"])
    p.dpool.devices = ["cuda:1"]
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p._on_msg({"type": "ready", "wid": "diar"})
    mem = {"ram_total": 24, "ram_free": 2.0, "commit_total": 30, "commit_free": 3.0}   # RAM az
    monkeypatch.setattr(u, "memory_status", lambda: dict(mem))
    p._dispatch()
    assert [t["kind"] for t in p.pool.put_q] == ["transcribe"] and p.dpool.put_q == []   # metin sürdü
    mem.update(ram_free=12.0, commit_free=20.0)                                       # tarayıcı kapandı
    p._dmem_at = 0
    p._on_msg({"type": "start", "wid": "cuda:1", "tid": p.pool.put_q[0]["id"]})
    p._on_msg({"type": "done", "wid": "cuda:1", "tid": p.pool.put_q[0]["id"], "elapsed": 1.0,
               "payload": {"segments": [], "language": "en"}})
    p._dispatch()
    assert p.dpool.put_q and p.dpool.put_q[0]["kind"] == "diarize"                     # sıra ayırmada


def test_low_ram_diar_given_after_max_wait(tmp_path, monkeypatch):
    import stt.pipeline as pl
    import stt.util as u
    from stt.pipeline import Job, Pipeline
    from stt.sources import Item
    p = Pipeline(Config(out_dir=tmp_path))
    it = Item("url", "https://x/d", key="u_d")
    p.jobs["u_d"] = Job(it, tmp_path / "wd", status="run", info={"duration_s": 600})
    p.dbacklog.append({"id": "u_d:d", "kind": "diarize", "job": "u_d"})
    monkeypatch.setattr(u, "memory_status", lambda: {"ram_total": 24, "ram_free": 1.0, "commit_total": 30, "commit_free": 1.0})
    assert p._diar_mem_ok() is False
    p._dmem_at = 0
    p._dmem_since = time.time() - pl.DIAR_MEM_MAX_WAIT_S - 1
    assert p._diar_mem_ok() is True                                                    # sonsuza dek beklemez



def test_display_unknown_still_guarded():
    from stt.gpuguard import display_card_used
    unk = {0: {"name": "RTX 3080", "gb": 10, "display": None}, 1: {"name": "RTX 3090", "gb": 24, "display": None}}
    assert display_card_used("yardim", unk) and display_card_used("all", unk)


def test_mem_wait_forced_until_dispatched(tmp_path, monkeypatch):
    """İnceleme: 10 dk sınırı dolunca kart o an meşgulse süre sıfırlanıyor, ayırma kuyruk bitene dek bekliyordu."""
    import stt.pipeline as pl
    import stt.util as u
    from stt.pipeline import Job, Pipeline
    from stt.sources import Item
    p = Pipeline(Config(out_dir=tmp_path))
    it = Item("url", "https://x/d", key="u_d")
    p.jobs["u_d"] = Job(it, tmp_path / "wd", status="run", info={"duration_s": 600})
    p.dbacklog.append({"id": "u_d:d", "kind": "diarize", "job": "u_d"})
    monkeypatch.setattr(u, "memory_status", lambda: {"ram_total": 24, "ram_free": 1.0, "commit_total": 30, "commit_free": 1.0})
    p._dmem_since = time.time() - pl.DIAR_MEM_MAX_WAIT_S - 1
    assert p._diar_mem_ok() is True
    for _ in range(3):                                   # görev verilmeden yeniden ölçülse de açık kalır
        p._dmem_at = 0
        assert p._diar_mem_ok() is True


def test_low_ram_does_not_push_diar_to_monitor_card(tmp_path, monkeypatch):
    from test_v199 import FakeDiarPool, _diar_pipe
    p, ds = _diar_pipe(tmp_path, monkeypatch)
    p._h_probe.clear()
    p.load = {"util": 0, "used_mb": 1500.0, "total_mb": 10240.0}
    p._dmem_ok = False                                   # RAM bekleniyor
    p._dmem_at = time.time()
    for _ in range(3):
        p._dispatch()
    assert FakeDiarPool.made == []


def test_guard_drops_display_main_worker(tmp_path):
    """'Hepsi' modunda koruma devreye girince monitörlü karttaki ana işçi ilk görevinde kapatılır, görev geri döner."""
    from test_stability import FakePool, _job, _pipe
    p = _pipe(tmp_path, ("cuda:0", "cuda:1"))
    p.pool._spec = {"cuda:0": "cuda:0", "cuda:1": "cuda:1"}
    p.pool.stopped = []
    p.pool.stop = lambda w, idle=False: p.pool.stopped.append((w, idle))
    p._ecfg = {"display_gpus": [0]}
    job, tasks = _job(p, tmp_path, n=3)
    p._on_msg({"type": "ready", "wid": "cuda:0"})
    p._on_msg({"type": "ready", "wid": "cuda:1"})
    p.disable_helper("ekran sürücüsü çöktü")
    assert p._drop_wids == {"cuda:0"}
    p._dispatch()
    p._on_msg({"type": "start", "wid": "cuda:0", "tid": tasks[0]["id"]})
    assert p.pool.stopped == [("cuda:0", False)] and "cuda:0" in p.fatal
    assert p.backlog[0]["id"] == tasks[0]["id"]                          # görev öbür karta kaldı
    p.pool.dead.add("cuda:0")
    p._health()
    assert "cuda:0" not in p.pool.respawned                              # yeniden açılmaz



# ---------------------------------------------------------------- 1.9.16: STT çalışırken yeniden başlama
def test_restart_reason_and_busy_window(tmp_path):
    import json as _j

    from stt.gpuguard import crash_while_busy, mark_busy, parse_events, reset_busy_marks, restart_reason
    now = int(time.time())
    ev = parse_events(_j.dumps([
        {"t": now - 5, "id": 41, "p": "Microsoft-Windows-Kernel-Power", "m": "The system has rebooted", "bc": 278},
        {"t": now - 4, "id": 1001, "p": "Microsoft-Windows-WER-SystemErrorReporting",
         "m": "The computer has rebooted from a bugcheck. The bugcheck was: 0x00000116 (0x1, 0x2)."}]))
    assert [e["kind"] for e in ev] == ["reset", "bsod"]
    assert "0x116" in restart_reason(ev) and "VIDEO_TDR_FAILURE" in restart_reason(ev)
    power = parse_events(_j.dumps({"t": now, "id": 41, "p": "Microsoft-Windows-Kernel-Power", "m": "x", "bc": 0}))
    assert "güç kaynağı" in restart_reason(power)
    assert crash_while_busy(tmp_path, ev) == []                 # STT hiç iş yapmadı
    mark_busy(tmp_path, True)                                   # iş sürerken bilgisayar kapandı (aralık açık kaldı)
    assert len(crash_while_busy(tmp_path, ev)) == 2
    reset_busy_marks(tmp_path)
    mark_busy(tmp_path, True)
    mark_busy(tmp_path, False)
    assert crash_while_busy(tmp_path, [{"t": now + 7200, "kind": "reset"}]) == []   # iş bittikten çok sonra


def test_crash_check_notice_once(tmp_path):
    import stt.gui as gui
    from stt.gpuguard import mark_busy
    app = gui.App.__new__(gui.App)
    app._state_dir = lambda: tmp_path
    pushed = []
    app._push = pushed.append
    mark_busy(tmp_path, True)
    ev = [{"t": int(time.time()), "kind": "reset", "id": 41, "p": "Microsoft-Windows-Kernel-Power", "m": "", "bc": 0}]
    assert app._crash_check(ev) and "Afterburner" in pushed[-1]["text"]
    assert app._crash_check(ev) == []                           # bir kez söylenir
