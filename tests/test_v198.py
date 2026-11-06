"""v1.9.8: veri seti eşleştirmesi (docs/TARAMA_2026-11-06.md, A bölümü).

Başlık örnekleri rapordan; fikstürler sahte (FPL'ye bu ortamdan erişilemiyor). Hoca → takım eşlemesi 2026-27
sezonu (premierleague.com listesi, 2026-11-06'da doğrulandı)."""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest

from stt.dataset import classify, find_fixture, match_folder_name, phase_hint, pl_teams, presser_time, rel_path
from stt.match import find_manager, find_teams, looks_like_presser, parse_match

# 1 Arsenal, 7 Chelsea, 14 Man City, 9 Everton, 8 Coventry, 20 Tottenham, 13 Liverpool, 4 Brentford, 11 Hull
FPL = {
    "teams": {"1": {"id": 1, "name": "Arsenal", "short_name": "ARS"}, "7": {"id": 7, "name": "Chelsea", "short_name": "CHE"},
              "14": {"id": 14, "name": "Man City", "short_name": "MCI"}, "9": {"id": 9, "name": "Everton", "short_name": "EVE"},
              "8": {"id": 8, "name": "Coventry", "short_name": "COV"}, "20": {"id": 20, "name": "Spurs", "short_name": "TOT"},
              "13": {"id": 13, "name": "Liverpool", "short_name": "LIV"}, "4": {"id": 4, "name": "Brentford", "short_name": "BRE"},
              "11": {"id": 11, "name": "Hull", "short_name": "HUL"}, "15": {"id": 15, "name": "Man Utd", "short_name": "MUN"},
              "16": {"id": 16, "name": "Nott'm Forest", "short_name": "NFO"}},
    "fixtures": [
        # Pazar 16:30: Arsenal - Chelsea (GW11)
        {"id": 101, "event": 11, "kickoff_time": "2026-11-08T16:30:00Z", "team_h": 1, "team_a": 7, "finished": True,
         "team_h_score": 2, "team_a_score": 1},
        # Çarşamba 20:00: Man City - Arsenal (GW12)
        {"id": 111, "event": 12, "kickoff_time": "2026-11-11T20:00:00Z", "team_h": 14, "team_a": 1, "finished": False},
        {"id": 102, "event": 11, "kickoff_time": "2026-11-08T14:00:00Z", "team_h": 14, "team_a": 15, "finished": True,
         "team_h_score": 1, "team_a_score": 1},
        {"id": 103, "event": 11, "kickoff_time": "2026-11-07T15:00:00Z", "team_h": 9, "team_a": 8, "finished": True,
         "team_h_score": 0, "team_a_score": 0},
        {"id": 104, "event": 11, "kickoff_time": "2026-11-07T17:30:00Z", "team_h": 20, "team_a": 13, "finished": True,
         "team_h_score": 3, "team_a_score": 2},
        {"id": 105, "event": 11, "kickoff_time": "2026-11-09T20:00:00Z", "team_h": 13, "team_a": 4, "finished": False},
    ]}


def _ts(s: str) -> int:
    from datetime import datetime, timezone
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())


def _meta(title, channel="", **info):
    info = {"title": title, "channel": channel, **info}
    return classify(parse_match(info), info, FPL)


# ---------------------------------------------------------------- A1: sözlük kelimeleri takım değil
@pytest.mark.parametrize("title,channel,want", [
    ("Pep Guardiola previews the Manchester derby", "Man City", ("Manchester City", None)),
    ("A nice win for the Gunners | Mikel Arteta reacts", "Arsenal", ("Arsenal", None)),
    ("Sporting director Andrea Berta on the summer window", "Arsenal", ("Arsenal", None)),
    ("Sabah antrenmanı | Okan Buruk", "Galatasaray", ("Galatasaray", None)),
    ("Merseyside derby vs Everton | Andoni Iraola press conference", "Liverpool FC", ("Liverpool", "Everton")),
    ("Derby day! Arteta on the north London derby", "Arsenal", ("Arsenal", None)),
    ("Danish TV: Thomas Frank om kampen", "", (None, None)),
])
def test_common_words_are_not_teams(title, channel, want):
    m = parse_match({"title": title, "channel": channel})
    assert (m["team"], m["away"]) == want


@pytest.mark.parametrize("title,teams", [
    ("Leeds v Derby | Carabao Cup", ["Leeds United", "Derby County"]),
    ("Derby vs Leeds", ["Derby County", "Leeds United"]),
    ("Arsenal vs Sporting CP | Champions League", ["Arsenal", "Sporting CP"]),
    ("Nice vs Arsenal | pre-season friendly", ["Nice", "Arsenal"]),
    ("OGC Nice - Lyon", ["Nice", "Lyon"]),
    ("Qarabağ - Sabah oyunundan sonra mətbuat konfransı", ["Qarabağ", "Sabah"]),
    ("Le Classique: OM - PSG", ["Marseille", "Paris Saint-Germain"]),
])
def test_ambiguous_names_count_with_a_rival(title, teams):
    assert find_teams(title) == teams


def test_opponent_pattern_resolves_ambiguous_club():
    m = parse_match({"title": "PRESS CONFERENCE | Mikel Arteta previews Sporting | Champions League", "channel": "Arsenal"})
    assert (m["team"], m["away"]) == ("Arsenal", "Sporting CP")


def test_nba_spurs_is_not_tottenham():
    assert find_teams("Spurs vs Grizzlies postgame press conference | NBA") == []
    assert find_teams("Spurs vs Arsenal | Premier League") == ["Tottenham", "Arsenal"]


def test_derby_title_finds_the_real_derby_fixture():
    """'Manchester derby' -> rakip Derby County değil; rakipsiz arama Man United maçını bulur."""
    meta = _meta("Pep Guardiola previews the Manchester derby", "Man City", timestamp=_ts("2026-11-06 12:00"))
    assert (meta["gw"], meta["home"], meta["away"], meta["phase"]) == (11, "Manchester City", "Manchester United",
                                                                      "mac-oncesi")


# ---------------------------------------------------------------- A2 / A7: tanınmayan rakip GW'yi silmesin
def test_player_name_is_not_opponent():
    m = parse_match({"title": "Mikel Arteta on Bukayo Saka's fitness | Press conference", "channel": "Arsenal"})
    assert m["away"] is None and m["manager"] == "Mikel Arteta"


def test_opponent_pattern_drops_generic_words():
    m = parse_match({"title": "Mikel Arteta previews Wycombe Clash", "channel": "Arsenal"})
    assert m["away"] == "Wycombe"


def test_unknown_opponent_still_finds_fixture_within_3_days():
    info = {"title": "x", "timestamp": _ts("2026-11-06 12:00")}
    meta = classify({"team": "Arsenal", "home": "Arsenal", "away": "Bukayo Saka's", "manager": "Mikel Arteta"}, info, FPL)
    assert (meta["gw"], meta["home"], meta["away"], meta["opponent"]) == (11, "Arsenal", "Chelsea", "Chelsea")
    info = {"title": "x", "timestamp": _ts("2026-11-03 12:00")}  # maça 5 gün: tahmin edilmez
    meta = classify({"team": "Arsenal", "home": "Arsenal", "away": "Bukayo Saka's"}, info, FPL)
    assert meta["gw"] is None


def test_known_non_pl_opponent_does_not_take_pl_fixture():
    """Bilinen Avrupa takımı (kupa/Avrupa maçı): 2 gün sonraki PL maçına yazılmaz."""
    meta = _meta("Mikel Arteta previews Slavia Prague", "Arsenal", timestamp=_ts("2026-11-06 12:00"))
    assert meta["gw"] is None and meta["opponent"] == "Slavia Prague"


# ---------------------------------------------------------------- A3: maç öncesi / sonrası başlıktan
def test_post_match_next_day_picks_previous_match():
    meta = _meta("Post-Match Press Conference | Mikel Arteta", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert (meta["gw"], meta["home"], meta["away"], meta["phase"]) == (11, "Arsenal", "Chelsea", "mac-sonrasi")


def test_pre_match_picks_next_match():
    meta = _meta("Mikel Arteta's pre-match press conference", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert (meta["gw"], meta["home"], meta["away"], meta["phase"]) == (12, "Manchester City", "Arsenal", "mac-oncesi")


def test_no_hint_monday_presser_is_for_midweek_match():
    meta = _meta("Mikel Arteta press conference", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert (meta["gw"], meta["phase"]) == (12, "mac-oncesi")


def test_upload_date_only_same_day_post_match():
    """X / Instagram / kulüp sitesi yalnız gün verirse (saat 12:00 varsayılıyordu) maç sonrası 'mac-oncesi' oluyordu."""
    meta = _meta("Mikel Arteta post-match press conference", "Arsenal", upload_date="20261108")
    assert (meta["gw"], meta["phase"], meta["match_date"]) == (11, "mac-sonrasi", "2026-11-08")
    meta = _meta("Mikel Arteta press conference", "Arsenal", upload_date="20261108")  # ipucu yok: aynı gün = sonrası
    assert (meta["gw"], meta["phase"]) == (11, "mac-sonrasi")
    meta = _meta("Mikel Arteta press conference", "Arsenal", upload_date="20261107")
    assert (meta["gw"], meta["phase"]) == (11, "mac-oncesi")


def test_post_hint_without_past_match_is_not_forced_onto_next_match():
    """Maç sonrası toplantı ama geride PL maçı yok (kupa maçı): sonraki PL maçına 'mac-oncesi' yazılmaz."""
    meta = _meta("Mikel Arteta post-match press conference", "Arsenal", timestamp=_ts("2026-11-07 10:00"))
    assert meta["gw"] is None and meta["phase"] == "mac-sonrasi"


@pytest.mark.parametrize("title,want", [
    ("Post-Match Press Conference | Mikel Arteta", "mac-sonrasi"),
    ("Arsenal 2-1 Chelsea | Mikel Arteta reacts", "mac-sonrasi"),
    ("Okan Buruk maç sonrası açıklamalar", "mac-sonrasi"),
    ("Mikel Arteta's pre-Chelsea press conference", "mac-oncesi"),
    ("Thomas Frank previews Arsenal clash", "mac-oncesi"),
    ("Okan Buruk'un VfB Stuttgart maçı öncesi açıklamaları", "mac-oncesi"),
    ("Mikel Arteta press conference", None),
    ("Pre-season: Arteta on the 4-3-3 and 2026-11-08 plans", None),
    ("Arteta ahead of Chelsea after Slavia win", "mac-oncesi"),   # yalnız "after" ipucu değil
    ("Arteta on the pre-match and post-match routine", None),        # ikisi birden: saate göre
])
def test_phase_hint(title, want):
    assert phase_hint(title) == want


def test_presser_time_day_only():
    assert presser_time({"upload_date": "20261108"})[1] is True
    assert presser_time({"timestamp": _ts("2026-11-08 19:00"), "upload_date": "20261108"})[1] is False
    assert presser_time({}) == (None, False)


# ---------------------------------------------------------------- A5: açıklamadaki kupa PL'yi ezmesin
def test_cup_in_description_does_not_override_pl_fixture():
    meta = _meta("Mikel Arteta's pre-Chelsea press conference", "Arsenal", timestamp=_ts("2026-11-06 12:00"),
                 description="After Tuesday's Champions League win over Slavia Prague, Mikel Arteta looks ahead.")
    assert (meta["gw"], meta["competition"], meta["phase"]) == (11, "Premier League", "mac-oncesi")


def test_cup_in_title_still_wins():
    meta = _meta("PRESS CONFERENCE | Mikel Arteta previews Chelsea | Carabao Cup", "Arsenal",
                 timestamp=_ts("2026-11-06 12:00"))
    assert meta["gw"] is None and meta["competition"] == "EFL Cup"


def test_cup_only_in_description_without_pl_rival_kept():
    meta = _meta("Mikel Arteta press conference", "Arsenal", timestamp=_ts("2026-11-06 12:00"),
                 description="Mikel Arteta speaks ahead of Wednesday's Champions League clash with Slavia Prague.")
    assert meta["gw"] is None and meta["competition"] == "Champions League"


# ---------------------------------------------------------------- A6: hoca listesi, basın toplantısı kalıbı
@pytest.mark.parametrize("title,channel", [
    ("Thomas Frank previews Arsenal clash", "Tottenham Hotspur"),
    ("Jamie Smith previews Arsenal clash", "Tottenham Hotspur"),   # listede olmayan ad: kulüp kanalı + rakip
    ("Marco Silva reacts to Brighton defeat", "Fulham FC"),
    ("Matthias Jaissle on the Chelsea game", ""),
    ("XABI ALONSO | Pre-Brentford", ""),
    ("Gary O'Neil reacts", "Ipswich Town"),
])
def test_presser_detection_wider(title, channel):
    info = {"title": title, "channel": channel}
    assert looks_like_presser(info, parse_match(info))


@pytest.mark.parametrize("title,channel", [
    ("Arsenal vs Chelsea | Premier League highlights", "Arsenal"),
    ("Derrick Rose on the Knicks", "Bleacher Report"),
    ("Golden State, Portland, Dallas, Washington, Curry vs. Luka | Amerikan Mutfak #11", ""),
    ("AFTV reacts to Arsenal 2-1 Chelsea", "AFTV"),
])
def test_not_a_presser(title, channel):
    info = {"title": title, "channel": channel}
    assert not looks_like_presser(info, parse_match(info))


def test_manager_decides_team_on_neutral_channel():
    """Sky gibi tarafsız kanal: takım başlıktaki ilk ad değil, hocanın takımı (2026-27)."""
    m = parse_match({"title": "Enzo Maresca press conference | Arsenal vs Man City", "channel": "Sky Sports News",
                     "upload_date": "20261110"})
    assert (m["team"], m["home"], m["away"], m["manager"]) == ("Manchester City", "Arsenal", "Manchester City",
                                                              "Enzo Maresca")
    m = parse_match({"title": "Frank Lampard Pre-Match Press Conference | Everton vs Coventry City", "channel": "Sky Sports"})
    assert m["team"] == "Coventry City"
    m = parse_match({"title": "Enzo Maresca press conference | Arsenal", "channel": "Sky Sports News",
                     "upload_date": "20261110"})
    assert (m["team"], m["home"], m["away"]) == ("Manchester City", "Manchester City", "Arsenal")


def test_manager_map_not_used_outside_its_season():
    """2025-26'daki eski video: Maresca o zaman Chelsea'deydi; eşleme uygulanmaz."""
    m = parse_match({"title": "Enzo Maresca press conference | Chelsea vs Arsenal", "channel": "Sky Sports News",
                     "upload_date": "20251201"})
    assert m["team"] == "Chelsea"


def test_club_channel_never_becomes_podcast(tmp_path):
    from stt.config import Config
    from stt.pipeline import Pipeline
    from stt.sources import Item
    pipe = Pipeline(Config(out_dir=tmp_path))
    try:
        item = Item("url", "https://youtu.be/x", title="Jamie Smith previews Arsenal clash", key="u_x")
        assert pipe._speaker_mode(item, {"title": item.title, "channel": "Tottenham Hotspur"},
                                  parse_match({"title": item.title, "channel": "Tottenham Hotspur"})) == "basin"
    finally:
        pipe.close()


# ---------------------------------------------------------------- A8: ad kalıbı
@pytest.mark.parametrize("title,want", [
    ("Kieran McKenna press conference", "Kieran McKenna"),
    ("Gary O'Neil press conference", "Gary O'Neil"),
    ("ERIK TEN HAG PRESS CONFERENCE", "Erik ten Hag"),
    ("JOHN SMITH PRESS CONFERENCE", "John Smith"),
    ("Roberto De Zerbi's post-match press conference", "Roberto De Zerbi"),
    ("De Zerbi on Spurs' win", "Roberto De Zerbi"),
    ("FABIAN HURZELER PRESS CONFERENCE", "Fabian Hürzeler"),
    ("Robin van Persie press conference", "Robin van Persie"),
    ("Mikel Arteta Bukayo Saka", "Mikel Arteta"),
    ("Okan Buruk'un açıklamaları", "Okan Buruk"),
])
def test_manager_name_patterns(title, want):
    assert find_manager(title, find_teams(title)) == want


# ---------------------------------------------------------------- A9: ev / deplasman sırası
def test_home_away_order_only_for_joined_teams():
    m = parse_match({"title": "Andoni Iraola's pre-Brentford press conference | Liverpool FC", "channel": "Liverpool FC"})
    assert (m["home"], m["away"], m["team"]) == ("Liverpool", "Brentford", "Liverpool")
    m = parse_match({"title": "Frank Lampard Pre-Match Press Conference | Everton vs Coventry City",
                     "channel": "Coventry City"})
    assert (m["home"], m["away"], m["team"]) == ("Everton", "Coventry City", "Coventry City")
    m = parse_match({"title": "Everton 0-0 Coventry City | Frank Lampard reacts", "channel": "Coventry City"})
    assert (m["home"], m["away"]) == ("Everton", "Coventry City")


# ---------------------------------------------------------------- A10: yazımlar
def test_spellings():
    assert find_teams("Manchester Utd vs Sheffield Utd") == ["Manchester United", "Sheffield United"]
    assert find_teams("Arsenal vs Union Saint-Gilloise") == ["Arsenal", "Union Saint-Gilloise"]
    assert find_teams("Union St-Gilloise v Arsenal") == ["Union Saint-Gilloise", "Arsenal"]
    assert find_teams("Hull City vs Ipswich Town") == ["Hull City", "Ipswich Town"]
    t = pl_teams(FPL)
    for name in ("Manchester United", "Tottenham", "Nottingham Forest", "Hull City", "Coventry City", "Manchester City"):
        assert name in t, name


def test_fallback_team_list_is_current_season():
    from stt.dataset import PL_FALLBACK
    assert {"Coventry City", "Hull City", "Ipswich Town"} <= set(PL_FALLBACK)
    assert not {"Wolves", "Burnley", "West Ham"} & set(PL_FALLBACK)
    assert len(PL_FALLBACK) == 20


# ---------------------------------------------------------------- A11: index yolları
def test_rel_path_posix():
    assert rel_path(PureWindowsPath(r"C:\out\Premier League\Arsenal\x.txt"), PureWindowsPath(r"C:\out")) == \
        "Premier League/Arsenal/x.txt"


# ---------------------------------------------------------------- A4: FPL verisi yeniden denenir
def test_fpl_retried_after_failure(tmp_path, monkeypatch):
    import stt.pipeline as pl
    from stt.config import Config
    calls = []
    answers = [None, FPL]
    monkeypatch.setattr(pl, "fpl_data", lambda d: calls.append(d) or answers[min(len(calls) - 1, 1)])
    clock = [1000.0]
    monkeypatch.setattr(pl.time, "time", lambda: clock[0])
    pipe = pl.Pipeline(Config(out_dir=tmp_path))
    try:
        assert pipe.fpl() is None
        assert pipe.fpl() is None and len(calls) == 1      # hemen tekrar denenmez (her video 40 sn beklemesin)
        clock[0] += 11 * 60
        assert pipe.fpl() is FPL and len(calls) == 2
        clock[0] += 60
        assert pipe.fpl() is FPL and len(calls) == 2       # bellekte
        clock[0] += 31 * 60
        pipe.fpl()
        assert len(calls) == 3                             # yarım saatte bir disk önbelleği / FPL yeniden okunur
    finally:
        pipe.close()


def test_fpl_fetch_backoff_keeps_stale_cache(tmp_path, monkeypatch):
    """FPL'ye ulaşılamıyorsa eski önbellek döner ve her çağrıda yeniden 2 x 20 sn beklenmez."""
    import os

    import stt.dataset as ds
    cache = tmp_path / "fpl_cache.json"
    cache.write_text(json.dumps(FPL), encoding="utf-8")
    old = cache.stat().st_mtime - 7 * 3600
    os.utime(cache, (old, old))
    tries = []

    def boom(url):
        tries.append(url)
        raise OSError("ağ yok")
    monkeypatch.setattr(ds, "_get_json", boom)
    monkeypatch.setattr(ds, "_fail_until", 0.0)
    assert ds.fpl_data(tmp_path)["fixtures"] == FPL["fixtures"] and len(tries) == 1
    assert ds.fpl_data(tmp_path)["fixtures"] == FPL["fixtures"] and len(tries) == 1


# ---------------------------------------------------------------- A-E: index alanları
def test_fixture_details_in_meta():
    meta = _meta("Post-Match Press Conference | Mikel Arteta", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert (meta["fixture_id"], meta["kickoff_utc"], meta["is_home"], meta["opponent_short"], meta["score"]) == \
        (101, "2026-11-08T16:30:00Z", True, "CHE", "2-1")
    meta = _meta("Mikel Arteta's pre-match press conference", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert (meta["fixture_id"], meta["is_home"], meta["opponent_short"], meta["score"]) == (111, False, "MCI", None)


def test_find_fixture_window_and_phase():
    from datetime import datetime, timezone
    when = datetime(2026, 11, 9, 13, tzinfo=timezone.utc)
    assert find_fixture("Arsenal", None, when, FPL)["id"] == 111
    assert find_fixture("Arsenal", None, when, FPL, phase="mac-sonrasi")["id"] == 101
    assert find_fixture("Arsenal", "Chelsea", when, FPL, phase="mac-oncesi") is None
    assert find_fixture("Arsenal", None, when, FPL, window=0.5) is None
    assert find_fixture("Arsenal", None, when, FPL, window=1)["id"] == 101
    assert match_folder_name(classify({"team": "Arsenal", "home": "Arsenal"}, {"title": "Arteta reacts",
                                       "timestamp": _ts("2026-11-09 13:00")}, FPL)) == "2026-11-08 GW11 Arsenal vs Chelsea"


# ---------------------------------------------------------------- A4: yeniden sınıflandırma (metni yeniden çevirmeden)
def _old_output(root: Path, key: str, title: str, channel: str, ts: int, folder: str, base: str, dataset: dict,
                match: dict | None = None, mode: str | None = "basin", news: bool = False):
    """1.9.7'nin yazdığı gibi bir çıktı (yanlış klasörde) + index satırı + durum kaydı."""
    from stt.outputs import write_all
    from stt.state import State
    d = root / folder
    doc = {"title": title, "source": f"https://youtu.be/{key}", "webpage_url": f"https://youtu.be/{key}", "key": key,
           "duration_s": 12.0, "language": "en", "engine": "whisper", "model": "large-v3",
           "settings": {"diar_model": "pyannote/speaker-diarization-community-1"},
           "match": match or {}, "td_label": "Mikel Arteta", "speaker_mode": mode, "speaker_source": "ses",
           "speakers": {"A": {"label": "Mikel Arteta", "seconds": 8.0}, "B": {"label": "Muhabir", "seconds": 4.0}},
           "source_info": {"title": title, "channel": channel, "timestamp": ts, "id": key, "extractor": "youtube"},
           "segments": [{"id": 0, "start": 0.0, "end": 4.0, "speaker": "Muhabir", "speaker_id": "B", "text": "How is Saka?"},
                        {"id": 1, "start": 4.0, "end": 12.0, "speaker": "Mikel Arteta", "speaker_id": "A",
                         "text": "He is fine and he trained today."}],
           "dataset": dataset}
    paths = write_all(d, base, doc, f"{base} - ozet.md")
    (d / f".{base}.key").write_text(key)
    (d / "mac.json").write_text(json.dumps({"team": dataset.get("team")}), encoding="utf-8")
    if news:
        (d / f"{base} - haber.md").write_text("# Haber", encoding="utf-8")
        (d / f"{base} - haber.json").write_text(json.dumps({"metin": f"{base}.json"}), encoding="utf-8")
    from stt.dataset import index_row, update_index
    update_index(root, index_row(root, key, dataset, doc, paths))
    st = State(root / ".stt")
    st.set(key, source=doc["source"], status="done", title=title, out_dir=str(d),
           info={"json": str(paths["json"]), "diarized": True, "speaker_mode": mode})
    return paths


def test_reclassify_moves_wrong_folder_and_fills_gw(tmp_path):
    from stt.reclassify import reclassify
    from stt.state import State
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "manager": "Mikel Arteta", "is_pl": True,
           "presser_date": "2026-11-06", "match_date": "2026-11-06", "gw": None, "phase": None,
           "home": "Arsenal", "away": "Bukayo Saka's"}
    paths = _old_output(tmp_path, "y_saka", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-11-06 Arsenal vs Bukayo Saka's",
                        "2026-11-06 - Mikel Arteta", old, news=True)
    plan = reclassify(tmp_path, apply=False, data=FPL)
    assert plan["ok"] and plan["moved"] == 1 and plan["gw_found"] == 1
    assert paths["txt"].exists()                                       # deneme turu hiçbir şeyi değiştirmez
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 1
    new = tmp_path / "Premier League" / "Arsenal" / "2026-11-08 GW11 Arsenal vs Chelsea"
    base = "2026-11-06 mac-oncesi - Mikel Arteta"
    assert (new / f"{base}.txt").exists() and (new / f"{base} - haber.md").exists()
    assert json.loads((new / f"{base} - haber.json").read_text("utf-8"))["metin"] == f"{base}.json"
    assert not paths["txt"].parent.exists()                            # eski klasör boşaldı, silindi
    doc = json.loads((new / f"{base}.json").read_text("utf-8"))
    assert doc["dataset"]["gw"] == 11 and doc["dataset"]["opponent"] == "Chelsea" and doc["match"]["away"] is None
    assert doc["segments"][1]["text"] == "He is fine and he trained today."
    assert "GW11" in (new / f"{base}.txt").read_text("utf-8")
    mac = json.loads((new / "mac.json").read_text("utf-8"))
    assert mac["gw"] == 11 and mac["score"] == "2-1" and mac["fpl_fixture"]["id"] == 101
    rows = [json.loads(x) for x in (tmp_path / "index.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["gw"] == 11 and rows[0]["txt"] == \
        f"Premier League/Arsenal/2026-11-08 GW11 Arsenal vs Chelsea/{base}.txt"
    st = State(tmp_path / ".stt").get("y_saka")
    assert st["out_dir"] == str(new) and st["info"]["json"] == str(new / f"{base}.json") and st["status"] == "done"
    again = reclassify(tmp_path, apply=True, data=FPL)
    assert again["moved"] == 0 and again["updated"] == 0 and again["unchanged"] == 1


def test_reclassify_keeps_other_pressers_in_old_folder(tmp_path):
    from stt.reclassify import reclassify
    ds = {"team": "Arsenal", "opponent": "Chelsea", "is_pl": True, "gw": 11, "home": "Arsenal", "away": "Chelsea",
          "presser_date": "2026-11-06", "match_date": "2026-11-08", "phase": "mac-oncesi", "manager": "Mikel Arteta",
          "competition": "Premier League"}
    folder = "Premier League/Arsenal/2026-11-08 GW11 Arsenal vs Chelsea"
    good = _old_output(tmp_path, "y_pre", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                       _ts("2026-11-06 12:00"), folder, "2026-11-06 mac-oncesi - Mikel Arteta", ds)
    # 1.9.7: ertesi gün yüklenen maç sonrası toplantı sonraki maça yazılmıştı
    wrong = {**ds, "gw": 12, "home": "Manchester City", "away": "Arsenal", "opponent": "Manchester City",
             "presser_date": "2026-11-09", "match_date": "2026-11-11", "phase": "mac-oncesi"}
    bad = _old_output(tmp_path, "y_post", "Post-Match Press Conference | Mikel Arteta", "Arsenal",
                      _ts("2026-11-09 13:00"), "Premier League/Arsenal/2026-11-11 GW12 Manchester City vs Arsenal",
                      "2026-11-09 mac-oncesi - Mikel Arteta", wrong)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 1, res
    assert good["txt"].exists()
    assert (good["txt"].parent / "2026-11-09 mac-sonrasi - Mikel Arteta.txt").exists()
    assert not bad["txt"].exists() and not bad["txt"].parent.exists()


def test_reclassify_name_collision_gets_suffix(tmp_path):
    from stt.reclassify import reclassify
    folder = tmp_path / "Premier League" / "Arsenal" / "2026-11-08 GW11 Arsenal vs Chelsea"
    folder.mkdir(parents=True)
    (folder / ".2026-11-06 mac-oncesi - Mikel Arteta.key").write_text("y_other")
    (folder / "2026-11-06 mac-oncesi - Mikel Arteta.txt").write_text("başka video", encoding="utf-8")
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    _old_output(tmp_path, "y_sakaa", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-11-06 Arsenal vs Bukayo Saka's",
                "2026-11-06 - Mikel Arteta", old)
    reclassify(tmp_path, apply=True, data=FPL)
    assert (folder / "2026-11-06 mac-oncesi - Mikel Arteta.txt").read_text("utf-8") == "başka video"
    assert (folder / "2026-11-06 mac-oncesi - Mikel Arteta [sakaa].txt").exists()


def test_reclassify_leaves_podcasts_and_reports_pressers(tmp_path):
    from stt.reclassify import reclassify
    ds = {"team": "Tottenham", "general": True, "presser_date": "2026-11-06"}
    p = _old_output(tmp_path, "y_spurs", "Jamie Smith previews Arsenal clash", "Tottenham Hotspur",
                    _ts("2026-11-06 12:00"), "Podcast ve Film/Jamie Smith previews Arsenal clash",
                    "2026-11-06 - Jamie Smith", ds, mode="herkes")
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert p["txt"].exists() and res["moved"] == 0
    assert [x["title"] for x in res["presser_now"]] == ["Jamie Smith previews Arsenal clash"]


def test_reclassify_needs_fpl_and_skips_other_seasons(tmp_path, monkeypatch):
    import stt.reclassify as rc
    monkeypatch.setattr(rc, "fpl_data", lambda d: None)
    assert not rc.reclassify(tmp_path, apply=True)["ok"]               # FPL yok: GW'ler silinmesin
    ds = {"team": "Arsenal", "opponent": "Chelsea", "is_pl": True, "gw": 30, "presser_date": "2026-03-01",
          "home": "Arsenal", "away": "Chelsea", "match_date": "2026-03-02", "phase": "mac-oncesi"}
    p = _old_output(tmp_path, "y_old", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                    _ts("2026-03-01 12:00"), "Premier League/Arsenal/2026-03-02 GW30 Arsenal vs Chelsea",
                    "2026-03-01 mac-oncesi - Mikel Arteta", ds)
    res = rc.reclassify(tmp_path, apply=True, data=FPL)
    assert res["ok"] and res["other_season"] == 1 and res["moved"] == 0 and p["txt"].exists()


def test_reclassify_missing_or_broken_files(tmp_path):
    from stt.dataset import update_index
    from stt.reclassify import reclassify
    update_index(tmp_path, {"key": "y_gone", "json": "Premier League/Arsenal/x/yok.json", "txt": "x.txt"})
    bad = tmp_path / "Premier League" / "Arsenal" / "y" / "bozuk.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{bozuk", encoding="utf-8")
    update_index(tmp_path, {"key": "y_bad", "json": "Premier League/Arsenal/y/bozuk.json", "txt": "y.txt"})
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["ok"] and len(res["missing"]) == 1 and len(res["errors"]) == 1
    assert len((tmp_path / "index.jsonl").read_text("utf-8").splitlines()) == 2   # satırlar silinmez


def test_reclassify_updates_collection_paths(tmp_path):
    from stt.reclassify import reclassify
    from stt.util import write_jsonl_csv
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    paths = _old_output(tmp_path, "y_col", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-11-06 Arsenal vs Bukayo Saka's",
                        "2026-11-06 - Mikel Arteta", old)
    col = tmp_path / "Koleksiyonlar" / "Arsenal - basın toplantıları"
    col.mkdir(parents=True)
    row = {"date": "2026-11-06", "title": "t", "url": "u", "language": "en", "duration_s": 12.0,
           "txt": str(paths["txt"]), "json": str(paths["json"]), "key": "y_col"}
    write_jsonl_csv(col / "liste.jsonl", col / "liste.csv", [row], list(row))
    (col / "tum_metinler.txt").write_text("eski", encoding="utf-8")
    reclassify(tmp_path, apply=True, data=FPL)
    r = json.loads((col / "liste.jsonl").read_text("utf-8"))
    assert "GW11" in r["txt"] and Path(r["txt"]).exists()
    assert "GW11" in (col / "tum_metinler.txt").read_text("utf-8")


def test_gui_reclassify_dry_run_then_apply(tmp_path, monkeypatch):
    import stt.gui as g
    import stt.reclassify as rc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc, "fpl_data", lambda d: FPL)
    app = g.App()
    app._cfg.out_dir = tmp_path
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    paths = _old_output(tmp_path, "y_gui", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-11-06 Arsenal vs Bukayo Saka's",
                        "2026-11-06 - Mikel Arteta", old)

    class Busy:
        def active(self):
            return True
    app._pipe = Busy()
    r = app.dataset_reclassify(True)
    assert not r["ok"] and "Kuyrukta" in r["error"] and paths["txt"].exists()
    app._pipe = None
    r = app.dataset_reclassify(False)
    assert r["ok"] and r["dry_run"] and r["moved"] == 1 and "taşınacak" in r["text"] and paths["txt"].exists()
    r = app.dataset_reclassify(True)
    assert r["ok"] and r["moved"] == 1 and not paths["txt"].exists()
    ds = app.dataset()
    assert ds["leagues"][0]["teams"][0]["matches"][0]["name"] == "2026-11-08 GW11 Arsenal vs Chelsea"


def test_finalize_writes_new_index_fields_and_posix_paths(tmp_path):
    """Bitirme yolu: FPL verisiyle GW klasörü, yeni index sütunları, açıklamanın başı json'da."""
    from stt.config import Config
    from stt.pipeline import Item, Job, Pipeline
    out = tmp_path / "out"
    pipe = Pipeline(Config(out_dir=out))
    pipe.fpl = lambda: FPL
    try:
        title = "Post-Match Press Conference | Mikel Arteta"
        item = Item("url", "https://www.youtube.com/watch?v=abc", title=title, key="y_abc" + "0" * 20)
        item.extra.update({"diarize": True})
        work = tmp_path / "is"
        (work / "parcalar").mkdir(parents=True)
        segs = [{"start": 0.0, "end": 4.0, "text": "How is Saka?", "words": []},
                {"start": 4.0, "end": 12.0, "text": "He is fine.", "words": []}]
        (work / "parcalar" / "0000.json").write_text(json.dumps({"segments": segs, "language": "en"}), encoding="utf-8")
        (work / "konusmaci.json").write_text(json.dumps({"turns": [[0.0, 4.0, "B"], [4.0, 12.0, "A"]]}),
                                             encoding="utf-8")
        info = {"title": title, "channel": "Arsenal", "duration_s": 12.0, "timestamp": _ts("2026-11-09 13:00"),
                "description": "Mikel Arteta reacts to Sunday's win over Chelsea. " + "x" * 600}
        job = Job(item, work, chunks=[{}], info=info)
        pipe._finalize(job)
        assert job.status == "done", job.error
    finally:
        pipe.close()
    assert job.out_dir == out / "Premier League" / "Arsenal" / "2026-11-08 GW11 Arsenal vs Chelsea"
    row = json.loads((out / "index.jsonl").read_text("utf-8").splitlines()[0])
    assert row["txt"] == "Premier League/Arsenal/2026-11-08 GW11 Arsenal vs Chelsea/2026-11-09 mac-sonrasi - Mikel Arteta.txt"
    assert (row["gw"], row["fixture_id"], row["score"], row["is_home"], row["opponent_short"], row["speaker_role"]) == \
        (11, 101, "2-1", True, "CHE", "hoca")
    assert row["title"] == title and row["phase"] == "mac-sonrasi"
    doc = job.result
    assert doc["description_head"].startswith("Mikel Arteta reacts") and len(doc["description_head"]) == 400
    mac = json.loads((job.out_dir / "mac.json").read_text("utf-8"))
    assert mac["score"] == "2-1" and mac["kickoff_utc"] == "2026-11-08T16:30:00Z"
