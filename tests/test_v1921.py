"""1.9.21: kanalın tamamında arama (takım / kişi / konu) ve iki kartta esnek tam güç."""
from stt.config import Config


def _e(i, title, date="2026-11-01", channel=""):
    return {"url": f"https://www.youtube.com/watch?v={i}", "id": i, "title": title, "date": date, "channel": channel}


# ---------------------------------------------------------------- aranan: takım
NEUTRAL = [
    _e("a1", "Mikel Arteta press conference | Arsenal v Chelsea"),
    _e("a2", "Enzo Maresca previews Arsenal | Press Conference"),          # rakibin toplantısı
    _e("a3", "Arsenal Women press conference | Renée Slegers"),             # kadın takımı
    _e("a4", "Arsenal U21 post-match reaction"),                            # altyaş
    _e("a5", "Arteta on Saka's fitness ahead of Chelsea"),                  # anahtar sözcük yok, hoca adı var
    _e("a6", "Arsenal 3-1 Chelsea | Highlights"),                           # özet
    _e("a7", "Liverpool press conference | Iraola"),                        # başka takım
    _e("a8", "Arsenal vs Chelsea | Post-match press conference"),           # iki takım: hangisi belli değil
    _e("a9", "U-21 Arsenal pre-match"),
    _e("a10", "Gunners boss Arteta: pre-match press conference"),
]


def test_team_pressers_on_neutral_channel():
    from stt.channel import select
    r = select([dict(e) for e in NEUTRAL], "basin", search="Arsenal", channel="Sky Sports Premier League")
    assert [e["id"] for e in r if e["match"]] == ["a1", "a5", "a8", "a10"]


def test_team_alias_and_manager_out_of_season():
    from stt.channel import select
    items = [_e("b1", "Mikel Arteta press conference", date="2026-12-01"),
             _e("b2", "Xabi Alonso press conference", date="2025-10-01"),     # o zaman Real Madrid'deydi
             _e("b3", "Xabi Alonso press conference", date="2026-10-01")]
    assert [e["id"] for e in select(items, "basin", search="Gunners") if e["match"]] == ["b1"]
    assert [e["id"] for e in select(items, "basin", search="Chelsea") if e["match"]] == ["b3"]


def test_team_on_own_club_channel():
    from stt.channel import select
    items = [_e("c1", "PRESS CONFERENCE | Arteta on Leeds", channel="Arsenal"),
             _e("c2", "Women's press conference | Slegers", channel="Arsenal"),
             _e("c3", "Training session ahead of Leeds", channel="Arsenal"),
             _e("c4", "Arteta previews Leeds clash", channel="Arsenal"),        # kulüp kanalı + rakip
             _e("c5", "Behind the scenes v Leeds", channel="Arsenal")]
    assert [e["id"] for e in select(items, "basin", search="Arsenal") if e["match"]] == ["c1", "c4"]
    # başka kulübün kanalında Arsenal'in toplantısı yok (oradaki toplantı o kulübün)
    other = [_e("d1", "PRESS CONFERENCE | Maresca previews Arsenal", channel="Man City")]
    assert not select(other, "basin", search="Arsenal")[0]["match"]


def test_everything_about_a_team_or_person():
    """Kullanıcı: yalnız basın toplantısı değil, her şey: kanaldaki bütün ilgili videolar."""
    from stt.channel import select
    items = [_e("e1", "Arsenal transfer news: Saka contract"),
             _e("e2", "Maresca previews Arsenal"),            # konu 'hepsi': takım başlıkta geçiyorsa ilgili
             _e("e3", "Chelsea 2-0 Spurs | Highlights"),
             _e("e4", "Arsenal Women 2-0 Chelsea Women"),    # takım aranınca kadın takımı yine hariç
             _e("e5", "Arteta reacts to the draw")]          # hocası (bu sezon) geçiyor
    assert [e["id"] for e in select(items, "hepsi", search="Arsenal") if e["match"]] == ["e1", "e2", "e5"]
    assert [e["id"] for e in select(items, "hepsi", search="Saka") if e["match"]] == ["e1"]
    assert [e["id"] for e in select(items, "hepsi", search="transfer, highlights") if e["match"]] == ["e1", "e3"]
    assert [e["id"] for e in select(items, "hepsi") if e["match"]] == ["e1", "e2", "e3", "e4", "e5"]  # aranan yok


def test_search_keeps_exclude_and_topic():
    from stt.channel import select
    items = [_e("f1", "Saka interview after the win"), _e("f2", "Saka press conference"), _e("f3", "Saka interview | U21 days")]
    assert [e["id"] for e in select(items, "roportaj", search="Saka", exclude="u21") if e["match"]] == ["f1"]


def test_collection_name_with_search():
    from stt.channel import collection_name
    assert collection_name("Sky Sports", "basin", None, "Arsenal") == "Arsenal - basın toplantıları"
    assert collection_name("Sky Sports", "basin", None, "man city") == "Manchester City - basın toplantıları"
    assert collection_name("Sky Sports", "hepsi", None, "Saka, transfer") == "Saka, transfer"
    assert collection_name("Sky Sports", "basin") == "Sky Sports - basın toplantıları"   # eski davranış


# ---------------------------------------------------------------- kanal içi arama sorguları
def _fake(monkeypatch, calls, channel):
    import stt.fetch as f

    def info(url, cookies=None, flat=False, extra=None, workdir=None):
        calls.append(url)
        if url.endswith("/playlists"):
            return {"channel": channel, "entries": [
                {"title": "Arsenal press conferences", "url": "https://www.youtube.com/playlist?list=PLars"},
                {"title": "Goals of the month", "url": "https://www.youtube.com/playlist?list=PLgoal"}]}
        if "PLars" in url:
            return {"entries": [{"id": "p1", "url": "https://www.youtube.com/watch?v=p1", "title": "Arteta presser"}]}
        if "/search?query=" in url:
            return {"entries": [{"id": "s1", "url": "https://www.youtube.com/watch?v=s1", "title": "Arteta press conference"},
                                {"id": "p1", "url": "https://www.youtube.com/watch?v=p1", "title": "Arteta presser"}]}
        if url.endswith("/videos") or url.endswith("/streams"):
            return {"entries": [{"id": "v1", "url": "https://www.youtube.com/watch?v=v1", "title": "Latest"}]}
        raise RuntimeError("beklenmedik " + url)
    monkeypatch.setattr(f, "ytdlp_info", info)


def test_find_in_channel_team_queries_neutral(monkeypatch):
    from stt.channel import find_in_channel
    calls = []
    _fake(monkeypatch, calls, "Sky Sports Premier League")
    r = find_in_channel("https://www.youtube.com/@SkySportsPL/videos", Config(), search="Arsenal", topic="basin")
    q = [c.split("query=")[1] for c in calls if "query=" in c]
    assert "Arsenal%20press%20conference" in q and "Mikel%20Arteta%20press%20conference" in q
    assert "press%20conference" not in q                      # tarafsız kanalda konu sözcüğü tek başına aranmaz
    assert any("PLars" in c for c in calls) and not any("PLgoal" in c for c in calls)
    assert any(c.endswith("/videos") for c in calls) and any(c.endswith("/streams") for c in calls)
    assert sorted(e["id"] for e in r["items"]) == ["p1", "s1", "v1"]   # tekrar yok


def test_find_in_channel_own_club_adds_topic_queries(monkeypatch):
    from stt.channel import find_in_channel
    calls = []
    _fake(monkeypatch, calls, "Arsenal")
    find_in_channel("https://www.youtube.com/@Arsenal", Config(), search="Arsenal", topic="basin")
    q = [c.split("query=")[1] for c in calls if "query=" in c]
    assert "press%20conference" in q and "manager%27s%20preview" in q


def test_find_in_channel_everything_queries(monkeypatch):
    from stt.channel import MAX_TARGETS, find_in_channel
    calls = []
    _fake(monkeypatch, calls, "Sky Sports")
    find_in_channel("https://www.youtube.com/@Sky", Config(), search="transfer, Saka", topic="hepsi")
    q = [c.split("query=")[1] for c in calls if "query=" in c]
    assert q == ["transfer", "Saka"]
    calls.clear()
    find_in_channel("https://www.youtube.com/@Sky", Config(), search=",".join(f"x{i}" for i in range(40)), topic="basin")
    assert len([c for c in calls if not c.endswith("/playlists")]) == MAX_TARGETS


def test_list_channel_search_uses_finder_and_sorts(tmp_path, monkeypatch):
    import stt.channel as ch
    import stt.gui as gui
    used = []
    items = [_e("g1", "Liverpool press conference"), _e("g2", "Mikel Arteta press conference")]
    monkeypatch.setattr(ch, "find_in_channel", lambda url, cfg, limit=None, search=None, topic=None:
                        used.append(("find", search, topic)) or {"title": "Sky", "channel": "Sky", "items": [dict(e) for e in items]})
    monkeypatch.setattr(ch, "list_videos", lambda url, cfg, limit=None: used.append(("list",)) or
                        {"title": "DM", "channel": "DM", "items": [dict(e) for e in items]})
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    monkeypatch.setattr(gui.App, "llm", lambda self: type("L", (), {"available": lambda s: False})())
    app._state = lambda: type("S", (), {"done_keys": lambda s, k: set()})()
    r = app.list_channel("https://www.youtube.com/@Sky", {"topic": "basin", "search": " Arsenal ,"})
    assert used[-1] == ("find", "Arsenal", "basin") and r["finder"] and r["search"] == "Arsenal"
    assert [e["id"] for e in r["items"]] == ["g2", "g1"] and r["matched"] == 1
    assert r["collection"] == "Arsenal - basın toplantıları"
    r = app.list_channel("https://www.dailymotion.com/sky", {"topic": "basin", "search": "Arsenal"})
    assert used[-1] == ("list",) and r["matched"] == 1             # YouTube değil: son videolarda süzülür


def test_follow_keeps_search(tmp_path, monkeypatch):
    import stt.channel as ch
    f = ch.Follows(tmp_path)
    row = f.add("https://www.youtube.com/@Sky", "Arsenal - basın toplantıları", "basin", search="Arsenal")
    monkeypatch.setattr(ch, "list_videos", lambda url, cfg, limit=None: {
        "channel": "Sky", "items": [_e("h1", "Mikel Arteta press conference"), _e("h2", "Liverpool press conference")]})
    new = ch.check_follow(f.all()[0], Config())
    assert row["search"] == "Arsenal" and [e["id"] for e in new] == ["h1"]


# ---------------------------------------------------------------- esnek tam güç: iki boş kartın düzeni
GI_FREE = {0: {"name": "RTX 3080", "gb": 10.0, "display": False}, 1: {"name": "RTX 3090", "gb": 24.0, "display": False}}
GI_MON = {0: {"name": "RTX 3080", "gb": 10.0, "display": True}, 1: {"name": "RTX 3090", "gb": 24.0, "display": False}}


def test_flex_pair_layout():
    from stt.pipeline import detect_devices, flex_pair, helper_devices
    assert flex_pair(GI_FREE) == (1, 0)
    assert detect_devices("auto", GI_FREE, flex=True) == ["cuda:1"] and helper_devices("auto", GI_FREE, flex=True) == ["cuda:0"]
    assert detect_devices("auto", GI_FREE) == ["cuda:0", "cuda:1"] and helper_devices("auto", GI_FREE) == []  # kapalı: eski
    # monitör bir NVIDIA kartta / bilinmiyor / tek ya da üç kart: esnek düzen yok (monitörlü karta iş gitmez)
    assert flex_pair(GI_MON) is None and detect_devices("auto", GI_MON, flex=True) == ["cuda:1"]
    assert helper_devices("auto", GI_MON, flex=True) == []
    assert flex_pair({0: {"gb": 10, "display": None}, 1: {"gb": 24, "display": False}}) is None
    assert flex_pair({0: {"gb": 24, "display": False}}) is None
    assert flex_pair({**GI_FREE, 2: {"gb": 8, "display": False}}) is None
    for spec in ("all", "0,1", "yardim", "1"):  # elle seçim değişmez
        assert detect_devices(spec, GI_FREE, flex=True) == detect_devices(spec, GI_FREE)


def _start(tmp_path, monkeypatch, cards, **cfg):
    import stt.engines as eng
    import stt.pipeline as pl
    import stt.power as pw
    made = []

    class FakePool:
        def __init__(self, devices, ecfg, *a, **k):
            made.append((list(devices), dict(ecfg)))
            self.workers = {d: d for d in devices}
            self.load_lock = None

        def start(self):
            pass
    monkeypatch.setattr(eng, "gpu_info", lambda *a, **k: dict(GI_FREE))
    monkeypatch.setattr(eng, "resolve_model_path", lambda m: "m")
    monkeypatch.setattr(pl, "WorkerPool", FakePool)
    monkeypatch.setattr(pw, "query", lambda: cards)
    monkeypatch.setattr(pw, "reapply", lambda *a, **k: False)
    p = pl.Pipeline(Config(out_dir=tmp_path, **cfg))
    p.events = lambda e: None
    p.start()
    p.prep.shutdown(wait=False)
    p.fin.shutdown(wait=False)
    return p, made


def test_pipeline_flex_card_with_power_limit(tmp_path, monkeypatch):
    from test_v1917 import _cards
    p, made = _start(tmp_path, monkeypatch, _cards(224, 245))
    assert p.devices == ["cuda:1"] and p.helper_devices == ["cuda:0"] and p._flex_card
    assert made[1][0] == ["cuda:0"] and made[1][1].get("helper")         # ikinci kart yardımcı düzeniyle (esnek)
    assert p._reserve() == 1.0                                          # monitörsüz: 2 GB değil 1 GB
    assert p.flex_state()["card2"] is True


def test_pipeline_flex_without_power_limit_one_card(tmp_path, monkeypatch):
    from test_v1917 import _cards
    p, made = _start(tmp_path, monkeypatch, _cards())                   # güç sınırı yok: yalnız 3090
    assert p.devices == ["cuda:1"] and p.helper_devices == [] and not p._flex_card


def test_pipeline_flex_off_keeps_two_main_cards(tmp_path, monkeypatch):
    from test_v1917 import _cards
    p, made = _start(tmp_path, monkeypatch, _cards(224, 245), gpu_flex=False)
    assert p.devices == ["cuda:0", "cuda:1"] and p.helper_devices == []


def test_monitor_helper_keeps_setting_reserve(tmp_path):
    from stt.pipeline import Pipeline
    p = Pipeline(Config(out_dir=tmp_path, gpus="yardim", helper_reserve_gb=2.5))
    assert p._reserve() == 2.5                                         # monitörlü yardımcı: ayardaki değer


# ---------------------------------------------------------------- esnek tam güç: kademeler
def test_flex_levels_hysteresis():
    from stt.flex import CALM_S, Flex
    f = Flex()
    assert f.update(50, 10, 0) == "" and f.level == 0                   # tek sıçrama yetmez
    assert "kademe 0 → 1" in f.update(50, 10, 10) and f.level == 1
    assert f.update(10, 10, 20) == "" and f.level == 1                  # sakin ama süre dolmadı
    assert f.update(30, 10, 30) == "" and f.level == 1                  # 20-35 arası: ne iner ne çıkar, sayaç sıfır
    f.update(10, 10, 40)
    assert f.update(10, 10, 40 + CALM_S - 1) == "" and f.level == 1
    assert "kademe 1 → 0" in f.update(10, 10, 40 + CALM_S) and f.level == 0
    f.update(80, 10, 200)
    assert "kademe 0 → 2" in f.update(70, 10, 210) and f.level == 2     # doğrudan 2'ye
    f.update(5, 10, 220)
    assert "kademe 2 → 1" in f.update(5, 10, 220 + CALM_S)              # inme kademe kademe
    assert "kademe 1 → 0" in f.update(5, 10, 220 + 2 * CALM_S)


def test_flex_low_ram_and_unknown():
    from stt.flex import Flex
    f = Flex()
    f.update(None, 1.5, 0)
    assert f.update(None, 1.5, 10) and f.level == 1                    # RAM az: işlemci okunamasa da yavaşla
    assert f.update(None, None, 20) == "" and f.level == 1              # hiç ölçülemedi: son durum
    f.update(5, 3.0, 30)
    assert f.update(5, 3.0, 200) == "" and f.level == 1                 # RAM 'sakin' eşiğinin altında: inmez
    g = Flex()
    assert g.update(None, None, 0) == "" and g.level == 0


def test_cpu_sampler_subtracts_our_processes():
    from stt.flex import CpuSampler, descendants
    seq = iter([((100, 1000), {1: 50, 2: 30}), ((700, 2000), {1: 150, 2: 130, 3: 100})])
    cur = {}

    def system():
        cur["v"] = next(seq)
        return cur["v"][0]
    s = CpuSampler(root=1, system=system, tree=lambda r: cur["v"][1])
    assert s.sample() is None                                           # ilk ölçüm
    others, ours = s.sample()
    # makine: 600/1000 = %60 dolu; bizimkiler 100+100+100(yeni ffmpeg) = %30 -> başkaları %30
    assert round(ours) == 30 and round(others) == 30
    assert sorted(descendants(1, {1: 0, 2: 1, 3: 2, 4: 9, 5: 5})) == [1, 2, 3]  # 4 başkasının, 5 kendine döngü


def test_cpu_sampler_reads_this_machine():
    import os
    from stt.flex import CpuSampler
    if os.name == "nt" or not os.path.exists("/proc/stat"):
        return
    s = CpuSampler()
    s.sample()
    sum(i * i for i in range(300000))
    r = s.sample()
    assert r is None or (0 <= r[0] <= 100 and 0 <= r[1] <= 100)


def _flex_pipe(tmp_path, level):
    from test_v199 import _pipe, _ready
    p, job, tasks = _pipe(tmp_path, n=6)
    p.devices = ["cuda:1"]
    p._flex_at = 1e18                    # ölçüm bu testte çalışmasın
    p.flex.level = level
    _ready(p, "cuda:1", "cuda:0")
    return p, job, tasks


def test_flex_level1_holds_second_card(tmp_path):
    from stt.flex import NOTE_L1
    p, job, tasks = _flex_pipe(tmp_path, 1)
    p._h_parked.pop("cuda:0", None)      # ikinci kart açık ve boşta
    p._dispatch()
    assert len(p.pool.put_q) == 1 and p.hpool.put_q == []              # ana kart sürer, ikinci kart yeni iş almaz
    assert p._h_note.get("cuda:0") == NOTE_L1
    p.flex.level = 0
    p._dispatch()
    p._dispatch()
    p._dispatch()
    assert p._h_note.get("cuda:0") != NOTE_L1


def test_flex_level2_rests_main_card(tmp_path):
    import time as t
    p, job, tasks = _flex_pipe(tmp_path, 2)
    p._dispatch()
    assert len(p.pool.put_q) == 1
    tid = p.pool.put_q[0]["id"]
    p._on_msg({"type": "start", "wid": "cuda:1", "tid": tid})
    p._on_msg({"type": "done", "wid": "cuda:1", "tid": tid, "payload": {"segments": [], "language": "en"}, "elapsed": 1})
    assert p._flex_rest_until > t.time()
    p._dispatch()
    assert len(p.pool.put_q) == 1                                       # dinleniyor
    p._flex_rest_until = 0
    p._dispatch()
    assert len(p.pool.put_q) == 2


def test_flex_off_or_night_ignores_level(tmp_path):
    p, job, tasks = _flex_pipe(tmp_path, 2)
    p.cfg.gpu_flex = False
    assert p._flex_level() == 0 and p.flex_state()["note"] == "kapalı"
    p.cfg.gpu_flex, p.cfg.night_mode = True, True
    assert p._flex_level() == 0
    p.cfg.night_mode = False
    assert p._flex_level() == 2
    job.status = "done"
    assert p._flex_level() == 0                                         # iş yok: yavaşlama yok


def test_flex_prep_ahead_limit(tmp_path):
    from stt.flex import PREP_AHEAD_L1
    p, job, tasks = _flex_pipe(tmp_path, 1)
    p._prep_ahead = lambda: PREP_AHEAD_L1
    assert not p._pending_ahead_ok()
    p.flex.level = 0
    assert p._pending_ahead_ok()


def test_flex_tick_measures_and_logs(tmp_path, caplog):
    import logging
    p, job, tasks = _flex_pipe(tmp_path, 0)
    p._flex_at = 0
    vals = iter([(50.0, 10.0), (55.0, 10.0)])
    p._flex_sampler = type("S", (), {"sample": lambda s: next(vals)})()
    p._spawn = lambda fn: fn()
    with caplog.at_level(logging.INFO, logger="stt"):
        p._flex_tick()
        p._flex_at = 0
        p._flex_tick()
    assert p.flex.level == 1 and any("[esnek]" in r.message for r in caplog.records)


def test_power_profiles(tmp_path, monkeypatch):
    import stt.gui as gui
    from stt.power import POWER_PROFILES, profile_of
    assert profile_of(70, 1600) == "guvenli" and profile_of(100, 0) == "tam" and profile_of(75, 1500) == "ozel"
    saved = {}
    monkeypatch.setattr(gui, "save_settings", lambda d: saved.update(d))
    monkeypatch.setattr(gui, "load_config", lambda: Config(out_dir=tmp_path))
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = None
    app._restart = type("E", (), {"set": lambda s: None})()
    monkeypatch.setattr(gui.App, "get_settings", lambda self: {})
    monkeypatch.setattr(gui.App, "_start_api", lambda self: None, raising=False)
    app.save_settings({"power_profile": "tam", "gpu_flex": False})
    assert (saved["gpu_power_pct"], saved["gpu_max_clock"]) == POWER_PROFILES["tam"] and saved["gpu_flex"] is False
    saved.clear()
    app.save_settings({"power_profile": "bozuk"})
    assert "gpu_power_pct" not in saved
