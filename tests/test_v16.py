"""1.6.0: ekran kartı koruması, kalıcı kuyruk, indirme zinciri, kanal/liste, canlı yayın, yapay zekâ, yerel API."""
import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from stt.config import Config


# ------------------------------------------------------------ ekran kartı
def test_mark_display_from_windows_list():
    from stt.engines import choose_gpus, mark_display
    # nvidia-smi iki kart için de 'Disabled' dedi (bazı sürücüler) ama monitör 3080'de
    info = {0: {"name": "RTX 3080", "gb": 10, "display": False}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}
    mark_display(info, ["NVIDIA GeForce RTX 3080"])
    assert info[0]["display"] is True and info[1]["display"] is False
    assert choose_gpus(2, info, "auto") == [1]


def test_mark_display_unknown_and_igpu():
    from stt.engines import mark_display
    info = {0: {"name": "RTX 3080", "gb": 10, "display": None}, 1: {"name": "RTX 3090", "gb": 24, "display": None}}
    mark_display(info, ["Intel(R) UHD Graphics 770"])  # monitör anakartta: iki kart da boş
    assert info[0]["display"] is False and info[1]["display"] is False
    same = {0: {"name": "RTX 3090", "gb": 24, "display": None}, 1: {"name": "RTX 3090", "gb": 24, "display": None}}
    mark_display(same, ["NVIDIA GeForce RTX 3090"])  # aynı model: adla ayırt edilemez, dokunma
    assert same[0]["display"] is None
    keep = {0: {"name": "RTX 3080", "gb": 10, "display": True}}
    mark_display(keep, [])
    assert keep[0]["display"] is True


def test_isolate_gpu_sets_visible_devices(monkeypatch):
    from stt.workers import isolate_gpu
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert isolate_gpu("cpu") == "cpu"
    assert "CUDA_VISIBLE_DEVICES" not in __import__("os").environ
    assert isolate_gpu("cuda:1") == "cuda:0"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_diar_device_never_display(tmp_path):
    from stt.pipeline import Pipeline
    p = Pipeline(Config(out_dir=tmp_path))
    p.devices = ["cuda:1"]
    p._ecfg = {"gpu_mem": {0: 10, 1: 24}, "display_gpus": [0]}
    assert p._diar_device() == "cuda:1"
    p.devices = ["cpu"]  # Whisper işlemcide: monitörsüz kart varsa o
    assert p._diar_device() == "cuda:1"
    p._ecfg = {"gpu_mem": {0: 10}, "display_gpus": [0]}
    assert p._diar_device() == "cpu"  # tek kart monitörlü ve Whisper onu kullanmıyor: işlemci
    p.devices = ["cuda:0"]
    assert p._diar_device() == "cuda:0"  # tek kart: Whisper zaten orada
    p.close()


def test_shift_payload():
    from stt.workers import shift_payload
    pl = {"segments": [{"start": 1.0, "end": 2.0, "words": [{"start": 1.0, "end": 1.5}]}]}
    shift_payload(pl, 100.0)
    assert pl["segments"][0]["start"] == 101.0 and pl["segments"][0]["words"][0]["end"] == 101.5


# ------------------------------------------------------------ ses
def test_agc_boosts_quiet_not_noise():
    from stt.audio import SR, agc
    t = np.arange(SR * 4) / SR
    loud = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    quiet = (0.01 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    noise = np.full(SR * 4, 1e-4, dtype=np.float32)
    a = agc(np.concatenate([loud, quiet, noise]))
    db = lambda v: 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12)  # noqa: E731
    assert db(a[SR * 5:SR * 7]) > db(quiet) + 15         # muhabir sesi yükseldi
    assert abs(db(a[SR * 9:SR * 11]) - db(noise)) < 1      # sessizlik büyütülmedi
    assert np.max(np.abs(a)) <= 1.0


# ------------------------------------------------------------ kuyruk / sağlamlık
def test_state_queue_roundtrip(tmp_path):
    from stt.state import State
    st = State(tmp_path)
    st.queue_add("u1", "url", "https://x.com/a", "A", {"lang": "en", "_gen": 3, "fpl": (11, "ARS")})
    st.queue_add("u2", "file", "/a.mp4", "B", {})
    rows = st.queue_all()
    assert [r["uid"] for r in rows] == ["u1", "u2"] and "_gen" not in rows[0]["extra"]
    st.queue_remove("u1")
    assert [r["uid"] for r in st.queue_all()] == ["u2"]
    st.queue_clear()
    assert st.queue_all() == []


def test_finished_job_leaves_persistent_queue(tmp_path):
    from stt.pipeline import Pipeline
    from stt.sources import Item
    p = Pipeline(Config(out_dir=tmp_path))
    it = Item("url", "https://example.com/v", title="v")
    it.extra["uid"] = "abc"
    p.state.queue_add("abc", it.kind, it.source, it.title, it.extra)
    p.emit("progress", it)
    assert p.state.queue_all()
    p.emit("failed", it, error="x")
    assert p.state.queue_all() == []
    p.close()


def test_is_transient():
    from stt.pipeline import is_transient
    assert is_transient("ERROR: Read timed out")
    assert is_transient("HTTP Error 503: Service Unavailable")
    assert not is_transient("ERROR: Video unavailable")
    assert not is_transient("ERROR: Private video. Sign in")


def test_worker_restart_backoff(tmp_path):
    from stt.pipeline import Pipeline

    class Pool:
        threads = False

        def __init__(self):
            self.workers = {"a": None}
            self.respawned = []

        def alive(self):
            return []

        def respawn(self, w):
            self.respawned.append(w)
    p = Pipeline(Config(out_dir=tmp_path))
    p.pool = Pool()
    p._health()                               # ilk çöküş: hemen
    assert p.pool.respawned == ["a"]
    p._health()                               # ikinci: 5 sn beklemeli
    assert p.pool.respawned == ["a"]
    p._respawn_due["a"] = 0
    p._health()
    assert p.pool.respawned == ["a", "a"]
    p._on_msg({"type": "done", "wid": "a", "tid": "yok", "payload": {}, "elapsed": 1})
    assert p.restarts["a"] == 0               # başarılı görev sayacı sıfırlar
    p.pool = None
    p.close()


# ------------------------------------------------------------ indirme
def test_cookie_browsers_order(monkeypatch):
    import stt.fetch as f
    monkeypatch.setattr(f, "installed_browsers", lambda: ["firefox", "edge", "chrome"])
    assert f.cookie_browsers(None) == []
    assert f.cookie_browsers("auto") == ["firefox", "edge", "chrome"]
    assert f.cookie_browsers("chrome") == ["chrome", "firefox", "edge"]


def test_find_embeds():
    from stt.fetch import find_embeds
    html = ('<iframe src="//www.youtube.com/embed/abcdefghijk?rel=0"></iframe>'
            '<iframe src="https://ok.ru/videoembed/123"></iframe><iframe src="https://ads.example/x"></iframe>'
            '<video-js data-account="111" data-player="pl" data-video-id="999"></video-js>')
    out = find_embeds(html, "https://club.com/news")
    assert out[0] == "https://www.youtube.com/embed/abcdefghijk?rel=0"
    assert "https://ok.ru/videoembed/123" in out
    assert "https://players.brightcove.net/111/pl_default/index.html?videoId=999" in out
    assert not any("ads.example" in u for u in out)


def test_explain_error_primary_first():
    from stt.fetch import explain_error
    msg = ("İndirilemedi. yt-dlp: ERROR: [ok] 1: This video is not available in your country  "
           "[firefox çerezleri: Failed to decrypt with DPAPI]")
    assert explain_error(msg).startswith("Bu video bulunduğunuz ülkeye kapalı")


def test_browser_rank_and_ads():
    from stt.browser import is_ad, rank_media
    assert rank_media("https://cdn/x/master.m3u8?t=1") < rank_media("https://cdn/x/video.mp4")
    assert is_ad("https://pubads.g.doubleclick.net/x.mp4")


# ------------------------------------------------------------ kanal / liste
def test_collection_urls():
    from stt.channel import is_collection_url
    assert is_collection_url("https://www.youtube.com/@Arsenal")
    assert is_collection_url("https://www.youtube.com/@Arsenal/streams")
    assert is_collection_url("https://www.youtube.com/playlist?list=PLx")
    assert not is_collection_url("https://www.youtube.com/watch?v=abc&list=PLx")
    assert not is_collection_url("https://youtu.be/abc")
    assert not is_collection_url("https://www.youtube.com/@Arsenal/live")  # kanalın o anki canlı yayını
    assert is_collection_url("https://www.dailymotion.com/beinsports")
    assert not is_collection_url("https://www.dailymotion.com/video/x8abc")


def test_topic_select_multilang():
    from stt.channel import select
    items = [{"title": t, "url": f"u{i}"} for i, t in enumerate([
        "Mikel Arteta press conference | Chelsea v Arsenal", "Goals | Arsenal 3-0 Spurs",
        "Basın Toplantısı | Okan Buruk", "Пресс-конференция Слуцкого", "Arsenal Women press conference",
        "PRE-MATCH PRESSER: Slot"])]
    sel = select(items, "basin", exclude="women")
    assert [e["match"] for e in sel] == [True, False, True, True, False, True]
    sel = select(items, "hepsi", custom="Slot")
    assert [e["match"] for e in sel] == [False, False, False, False, False, True]


def test_channel_entries_flatten():
    from stt.channel import _entries, _youtube_tabs
    info = {"channel": "Arsenal", "entries": [
        {"_type": "url", "ie_key": "Youtube", "id": "aaaaaaaaaaa", "url": "aaaaaaaaaaa", "title": "A"},
        {"_type": "playlist", "entries": [{"url": "https://www.youtube.com/watch?v=bbbbbbbbbbb", "id": "b",
                                           "title": "B", "upload_date": "20261101"}]}]}
    out = _entries(info)
    assert out[0]["url"] == "https://www.youtube.com/watch?v=aaaaaaaaaaa" and out[1]["date"] == "2026-11-01"
    assert _youtube_tabs("https://www.youtube.com/@Arsenal") == ["https://www.youtube.com/@Arsenal/videos",
                                                                "https://www.youtube.com/@Arsenal/streams"]
    assert _youtube_tabs("https://www.youtube.com/@Arsenal/videos") == ["https://www.youtube.com/@Arsenal/videos"]


def test_follows_and_collection(tmp_path):
    from stt.channel import Follows, add_to_collection
    f = Follows(tmp_path)
    row = f.add("https://www.youtube.com/@Arsenal", "Arsenal - basın", "basin", seen=["x"])
    assert f.due(6) and f.all()[0]["seen"] == ["x"]
    f.update(row["id"], last=time.time())
    assert not f.due(6)
    f.remove(row["id"])
    assert f.all() == []
    out = tmp_path / "out"
    for i, d in enumerate(["2026-11-02", "2026-11-01"]):
        txt = out / f"{i}.txt"
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(f"metin {d}", encoding="utf-8")
        add_to_collection(out, "Arsenal - basın", {"key": f"k{i}", "title": f"t{i}", "dataset": {"presser_date": d}},
                          {"txt": txt, "json": txt.with_suffix(".json")})
    folder = out / "Koleksiyonlar" / "Arsenal - basın"
    allt = (folder / "tum_metinler.txt").read_text(encoding="utf-8")
    assert allt.index("2026-11-01") < allt.index("2026-11-02")  # tarih sırası
    assert (folder / "liste.csv").exists()


# ------------------------------------------------------------ canlı yayın
def test_live_cut_point_prefers_silence():
    from stt.audio import SR
    from stt.live import cut_point
    buf = (np.random.randn(SR * 30) * 3000).astype(np.int16)
    buf[SR * 22:SR * 22 + SR // 2] = 0
    k = cut_point(buf, SR * 18, SR * 28)
    assert SR * 22 <= k <= SR * 22 + SR // 2


def test_live_recorder_from_file(tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg yok")
    import subprocess
    src = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=12", "-ar", "16000", "-ac", "1", str(src)],
                   check=True)
    chunks, ended = [], threading.Event()
    res = {}
    cfg = Config(out_dir=tmp_path)
    cfg.live_chunk_s = 5

    def on_end(wav, why):
        res.update(wav=wav, why=why)
        ended.set()
    from stt.live import LiveRecorder
    rec = LiveRecorder(str(src), tmp_path / "w", cfg, on_chunk=lambda *a: chunks.append(a), on_end=on_end, readrate=20)
    rec.start()
    assert ended.wait(60)
    total = sum(c[3] for c in chunks)
    assert abs(total - 12) < 0.3 and len(chunks) >= 2
    assert chunks[1][2] == pytest.approx(chunks[0][3])  # ikinci parçanın başlangıcı = birincinin süresi
    from stt.fetch import wav_duration
    assert abs(wav_duration(res["wav"]) - 12) < 0.3 and res["why"] == "dosya bitti"


# ------------------------------------------------------------ yapay zekâ
def test_pick_gemini_and_parse_json():
    from stt.llm import parse_json, pick_gemini
    names = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.0-flash", "gemini-3.0-flash-preview-09",
             "gemini-3.0-pro", "gemini-3.0-flash-image", "text-embedding-004"]
    assert pick_gemini(names) == "gemini-3.0-flash"
    assert pick_gemini(["gemini-3.5-flash-preview"]) == "gemini-3.5-flash-preview"
    assert pick_gemini([]) is None
    assert parse_json('```json\n{"a": [1]}\n```') == {"a": [1]}
    assert parse_json('Tamam: {"uygun": [0, 2]} bitti') == {"uygun": [0, 2]}


def test_llm_budget_and_usage(tmp_path):
    from stt.llm import LLM, BudgetExceeded, Usage
    cfg = Config(out_dir=tmp_path)
    cfg.llm_provider, cfg.gemini_api_key, cfg.llm_budget_usd = "gemini", "k", 0.001
    llm = LLM(cfg, tmp_path)
    assert llm.available()
    Usage(tmp_path).add("claude-haiku-5-5", 1_000_000, 1_000_000)  # $0.60
    with pytest.raises(BudgetExceeded):
        llm.complete("s", "p")


def test_llm_qa_fallback_in_finalize(tmp_path, monkeypatch):
    """Konuşmacı ayırma 1 kişi bulunca soru-cevap yapay zekâyla ayrılır."""
    from stt import llm as llm_mod
    from stt.pipeline import Job, Pipeline
    from stt.sources import Item
    cfg = Config(out_dir=tmp_path)
    cfg.llm_provider, cfg.gemini_api_key = "gemini", "k"
    p = Pipeline(cfg)
    monkeypatch.setattr(llm_mod, "label_qa", lambda llm, sents: [s.endswith("?") for s in sents])
    it = Item("url", "https://youtube.com/watch?v=x", title="Arteta press conference", key="u_x")
    it.extra.update(diarize=True, speaker_mode="basin")
    job = Job(it, tmp_path / "w", status="finalize")
    (job.work / "parcalar").mkdir(parents=True)
    segs = [{"start": 0, "end": 5, "text": "Good morning."}, {"start": 5, "end": 8, "text": "How is Saka?"},
            {"start": 8, "end": 15, "text": "He is fine, he trained today."}]
    (job.work / "parcalar" / "0000.json").write_text(json.dumps({"segments": segs, "language": "en"}))
    (job.work / "konusmaci.json").write_text(json.dumps({"turns": [[0, 15, "S0"]]}))
    job.chunks = [{"start": 0, "end": 15}]
    job.info = {"title": "Arteta press conference", "duration_s": 15}
    p._finalize(job)
    txt = Path(job.paths["txt"]).read_text(encoding="utf-8")
    assert "Soru 1 - Muhabir" in txt and "How is Saka?" in txt
    assert job.result["speaker_source"].startswith("yapay")
    p.close()


# ------------------------------------------------------------ arayüz / API
def test_window_geometry():
    from stt.gui import window_geometry
    g = window_geometry((1920, 1080), None)
    assert g["width"] <= 1100 and g["height"] <= 720
    g = window_geometry((1366, 768), {"w": 5000, "h": 3000})  # bozuk kayıt yok sayılır
    assert g["width"] <= 1366 and g["height"] <= 768
    assert window_geometry((1920, 1080), {"w": 900, "h": 600})["width"] == 900


def test_local_api(tmp_path, monkeypatch):
    import socket
    import stt.gui as g
    from stt.api import start_api
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    app._cfg.api_token = "gizli"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = start_api(app, port)
    try:
        def req(path, body=None, tok="gizli"):
            r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body else None,
                                       headers={"X-STT-Token": tok, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=10) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        urllib.request.install_opener(opener)
        assert req("/api/status", tok="yanlis")[0] == 401
        code, st = req("/api/status")
        assert code == 200 and st["version"]
        code, r = req("/api/jobs", {"inputs": ["https://www.youtube.com/watch?v=abcdefghijk"], "options": {"lang": "en"}})
        assert code == 200 and r["count"] == 1 and app._inbox.qsize() == 1
        code, jobs = req("/api/jobs")
        assert jobs["jobs"][0]["status"] == "queued"
        assert req("/api/text?path=/etc/passwd")[0] == 403
    finally:
        srv.shutdown()


def test_gui_start_routes_collections(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    r = app.start("https://www.youtube.com/@Arsenal\nhttps://youtu.be/abcdefghijk", [], {})
    assert r["count"] == 1 and r["collections"] == ["https://www.youtube.com/@Arsenal"]
    r = app.start("https://www.youtube.com/watch?v=abcdefghijk", [], {"live": True, "news": True})
    it = app._inbox.queue[-1]
    assert it.extra["live"] and it.extra["llm_news"]
    assert app.retry(r["uids"][0])["ok"]


def test_gemini_request_and_news(tmp_path, monkeypatch):
    """Gemini REST yanıtı ayrıştırılır, kullanım kaydedilir, haber.md yazılır (ağsız)."""
    import requests
    from stt.llm import LLM, write_news
    calls = []

    class R:
        status_code = 200

        def __init__(self, data):
            self._d = data
            self.text = json.dumps(data)

        def json(self):
            return self._d

        def raise_for_status(self):
            pass
    pkg = {"baslik": "Arteta: Saka hazır", "spot": "Arteta açıkladı.", "haber": "Paragraf.", "ozet": ["a"],
           "alintilar": [{"kisi": "Arteta", "soz": "He is fine", "ceviri": "İyi durumda"}], "etiketler": ["Arsenal"]}

    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        calls.append((url, json))
        return R({"candidates": [{"content": {"parts": [{"text": __import__("json").dumps(pkg)}]}}],
                  "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200}})
    monkeypatch.setattr(requests, "post", post)
    cfg = Config(out_dir=tmp_path)
    cfg.llm_provider, cfg.gemini_api_key, cfg.llm_model = "gemini", "k", "gemini-2.5-flash"
    llm = LLM(cfg, tmp_path)
    jp = tmp_path / "a.json"
    jp.write_text(json.dumps({"title": "Arteta presser", "segments": [{"start": 0, "end": 1, "text": "He is fine"}]}))
    res = write_news(llm, jp)
    md = res["news"].read_text(encoding="utf-8")
    assert md.startswith("# Arteta: Saka hazır") and "He is fine" in md and "TASLAK" in md
    assert calls[0][0].endswith("gemini-2.5-flash:generateContent")
    assert calls[0][1]["generationConfig"]["responseMimeType"] == "application/json"
    assert llm.status()["calls"] == 1 and llm.status()["month_usd"] > 0
