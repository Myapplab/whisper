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


def test_unknown_opponent_matching_fixture_within_3_days():
    """Sözlükte olmayan rakip metni ("Manchester", "City") yakındaki PL maçının rakibiyle örtüşürse o maç."""
    info = {"title": "x", "timestamp": _ts("2026-11-06 12:00")}
    meta = classify({"team": "Manchester City", "home": "Manchester City", "away": "Manchester"}, info, FPL)
    assert (meta["gw"], meta["home"], meta["away"]) == (11, "Manchester City", "Manchester United")
    meta = _meta("Pep Guardiola Previews The Manchester Derby", "Man City", timestamp=_ts("2026-11-06 12:00"))
    assert (meta["gw"], meta["away"]) == (11, "Manchester United")
    info = {"title": "x", "timestamp": _ts("2026-11-03 12:00")}  # maça 5 gün: tahmin edilmez
    meta = classify({"team": "Manchester City", "home": "Manchester City", "away": "Manchester"}, info, FPL)
    assert meta["gw"] is None


@pytest.mark.parametrize("title", ["Mikel Arteta previews Wycombe", "Arsenal vs Wycombe | Mikel Arteta press conference",
                                   "Arteta: 'Saka is fine' | Arsenal vs Wycombe"])
def test_unknown_club_is_cup_match_not_next_pl_match(title):
    """Sözlükte olmayan gerçek kulüp (kupa rakibi): iki gün sonraki PL maçına (Man City) yazılmaz."""
    meta = _meta(title, "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert meta["gw"] is None and meta["opponent"] == "Wycombe"


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
    """Açılışta FPL'ye ulaşılamadı: eskiden oturum boyunca GW'siz. Artık 10 dk sonra yeniden denenir (arada ağ
    beklenmez), veri gelince bellekte tutulur, yarım saatte bir disk önbelleğinden yenilenir."""
    import stt.dataset as ds
    import stt.pipeline as pl
    from stt.config import Config
    clock = [1000.0]
    monkeypatch.setattr(pl.time, "time", lambda: clock[0])
    state = {"down": True}
    tries = []
    boot = {"teams": [{"id": int(k), "name": v["name"], "short_name": v["short_name"]} for k, v in FPL["teams"].items()]}

    def get(url):
        tries.append(url)
        if state["down"]:
            raise OSError("ağ yok")
        return boot if "bootstrap" in url else FPL["fixtures"]
    monkeypatch.setattr(ds, "_get_json", get)
    monkeypatch.setattr(ds, "_fail_until", 0.0)
    pipe = pl.Pipeline(Config(out_dir=tmp_path))
    try:
        assert pipe.fpl() is None and len(tries) == 1
        clock[0] += 60
        assert pipe.fpl() is None and len(tries) == 1      # 10 dk dolmadan ağ denenmez (her video beklemesin)
        state["down"] = False
        clock[0] += 10 * 60
        data = pipe.fpl()
        assert data and [f["id"] for f in data["fixtures"]] == [f["id"] for f in FPL["fixtures"]] and len(tries) == 3
        clock[0] += 60
        assert pipe.fpl() is data and len(tries) == 3      # bellekte
        clock[0] += 31 * 60
        assert [f["id"] for f in pipe.fpl()["fixtures"]] == [f["id"] for f in FPL["fixtures"]] and len(tries) == 3  # disk önbelleği
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
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                        "2026-11-06 - Mikel Arteta", old, news=True)
    plan = reclassify(tmp_path, apply=False, data=FPL)
    assert plan["ok"] and plan["moved"] == 1 and plan["gw_found"] == 1
    assert paths["txt"].exists()                                       # deneme turu hiçbir şeyi değiştirmez
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 1
    new = tmp_path / "Premier League" / "Arsenal" / "2026-2027" / "2026-11-08 GW11 Arsenal vs Chelsea"
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
        f"Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea/{base}.txt"
    st = State(tmp_path / ".stt").get("y_saka")
    assert st["out_dir"] == str(new) and st["info"]["json"] == str(new / f"{base}.json") and st["status"] == "done"
    again = reclassify(tmp_path, apply=True, data=FPL)
    assert again["moved"] == 0 and again["updated"] == 0 and again["unchanged"] == 1


def test_reclassify_keeps_other_pressers_in_old_folder(tmp_path):
    from stt.reclassify import reclassify
    ds = {"team": "Arsenal", "opponent": "Chelsea", "is_pl": True, "gw": 11, "home": "Arsenal", "away": "Chelsea",
          "presser_date": "2026-11-06", "match_date": "2026-11-08", "phase": "mac-oncesi", "manager": "Mikel Arteta",
          "competition": "Premier League"}
    folder = "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea"
    good = _old_output(tmp_path, "y_pre", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                       _ts("2026-11-06 12:00"), folder, "2026-11-06 mac-oncesi - Mikel Arteta", ds)
    # 1.9.7: ertesi gün yüklenen maç sonrası toplantı sonraki maça yazılmıştı
    wrong = {**ds, "gw": 12, "home": "Manchester City", "away": "Arsenal", "opponent": "Manchester City",
             "presser_date": "2026-11-09", "match_date": "2026-11-11", "phase": "mac-oncesi"}
    bad = _old_output(tmp_path, "y_post", "Post-Match Press Conference | Mikel Arteta", "Arsenal",
                      _ts("2026-11-09 13:00"), "Premier League/Arsenal/2026-2027/2026-11-11 GW12 Manchester City vs Arsenal",
                      "2026-11-09 mac-oncesi - Mikel Arteta", wrong)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 1, res
    assert good["txt"].exists()
    assert (good["txt"].parent / "2026-11-09 mac-sonrasi - Mikel Arteta.txt").exists()
    assert not bad["txt"].exists() and not bad["txt"].parent.exists()


def test_reclassify_name_collision_gets_suffix(tmp_path):
    from stt.reclassify import reclassify
    folder = tmp_path / "Premier League" / "Arsenal" / "2026-2027" / "2026-11-08 GW11 Arsenal vs Chelsea"
    folder.mkdir(parents=True)
    (folder / ".2026-11-06 mac-oncesi - Mikel Arteta.key").write_text("y_other")
    (folder / "2026-11-06 mac-oncesi - Mikel Arteta.txt").write_text("başka video", encoding="utf-8")
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    _old_output(tmp_path, "y_sakaa", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
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
    monkeypatch.setattr(rc, "fpl_data", lambda d, **k: None)
    assert not rc.reclassify(tmp_path, apply=True)["ok"]               # FPL yok: GW'ler silinmesin
    ds = {"team": "Arsenal", "opponent": "Chelsea", "is_pl": True, "gw": 30, "presser_date": "2026-03-01",
          "home": "Arsenal", "away": "Chelsea", "match_date": "2026-03-02", "phase": "mac-oncesi"}
    p = _old_output(tmp_path, "y_old", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                    _ts("2026-03-01 12:00"), "Premier League/Arsenal/2026-03-02 GW30 Arsenal vs Chelsea",
                    "2026-03-01 mac-oncesi - Mikel Arteta", ds)
    mac = p["txt"].parent / "mac.json"
    mac.write_text(json.dumps({"team": "Arsenal", "score": "3-0", "ek_bilgiler": {"not": "elle"}}), encoding="utf-8")
    res = rc.reclassify(tmp_path, apply=True, data=FPL)
    # 1.9.25: eski sezon yeniden sınıflandırılmaz (fikstürü yok), yalnız sezon klasörüne taşınır
    new = tmp_path / "Premier League" / "Arsenal" / "2025-2026" / "2026-03-02 GW30 Arsenal vs Chelsea"
    assert res["ok"] and res["moved"] == 1 and res["other_season"] == 0, res
    assert (new / "2026-03-01 mac-oncesi - Mikel Arteta.txt").exists() and not p["txt"].exists()
    doc = json.loads((new / "2026-03-01 mac-oncesi - Mikel Arteta.json").read_text("utf-8"))
    assert doc["dataset"]["gw"] == 30 and doc["dataset"]["opponent"] == "Chelsea"   # eski sınıflandırma kalır
    m = json.loads((new / "mac.json").read_text("utf-8"))
    assert m["score"] == "3-0" and m["ek_bilgiler"] == {"not": "elle"} and m["season"] == "2025-2026"
    assert not mac.parent.exists()    # elle eklenen bilgi yeni mac.json'a geçti: boş eski klasör silindi
    row = json.loads((tmp_path / "index.jsonl").read_text("utf-8"))
    assert row["season"] == "2025-2026" and row["txt"].startswith("Premier League/Arsenal/2025-2026/")
    again = rc.reclassify(tmp_path, apply=True, data=FPL)
    assert again["other_season"] == 1 and again["moved"] == 0


def test_reclassify_other_season_without_team_or_elsewhere_untouched(tmp_path):
    from stt.reclassify import reclassify
    ds = {"team": None, "presser_date": "2026-03-01", "match_date": "2026-03-01"}
    p = _old_output(tmp_path, "y_unk", "Mikel Arteta press conference", "Arsenal", _ts("2026-03-01 12:00"),
                    "Diger Takimlar/Bilinmeyen/Mikel Arteta press conference", "2026-03-01 - Mikel Arteta", ds)
    # elle başka bir klasöre konmuş eski sezon kaydı: sezon eklemek dışında yer değiştirmesi gerekirdi, dokunulmaz
    ds2 = {"team": "Arsenal", "is_pl": True, "presser_date": "2026-03-01", "match_date": "2026-03-02",
           "home": "Arsenal", "away": "Chelsea", "opponent": "Chelsea", "gw": 30}
    q = _old_output(tmp_path, "y_else", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                    _ts("2026-03-01 12:00"), "Elle/Arsenal/2026-03-02 GW30 Arsenal vs Chelsea",
                    "2026-03-01 - Mikel Arteta", ds2)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["other_season"] == 2 and res["moved"] == 0 and p["txt"].exists() and q["txt"].exists()


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
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
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
    monkeypatch.setattr(rc, "fpl_data", lambda d, **k: FPL)
    app = g.App()
    app._cfg.out_dir = tmp_path
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    paths = _old_output(tmp_path, "y_gui", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
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
    assert job.out_dir == out / "Premier League" / "Arsenal" / "2026-2027" / "2026-11-08 GW11 Arsenal vs Chelsea"
    row = json.loads((out / "index.jsonl").read_text("utf-8").splitlines()[0])
    assert row["txt"] == "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea/2026-11-09 mac-sonrasi - Mikel Arteta.txt"
    assert (row["gw"], row["fixture_id"], row["score"], row["is_home"], row["opponent_short"], row["speaker_role"]) == \
        (11, 101, "2-1", True, "CHE", "hoca")
    assert row["title"] == title and row["phase"] == "mac-sonrasi"
    doc = job.result
    assert doc["description_head"].startswith("Mikel Arteta reacts") and len(doc["description_head"]) == 400
    mac = json.loads((job.out_dir / "mac.json").read_text("utf-8"))
    assert mac["score"] == "2-1" and mac["kickoff_utc"] == "2026-11-08T16:30:00Z"


# ---------------------------------------------------------------- bağımsız inceleme bulguları
@pytest.mark.parametrize("title", [
    "Mikel Arteta reacts to transfer rumours",
    "Mikel Arteta's reaction to Saka's England call-up",
    "Arteta press conference: Saka out for 2-3 weeks",
    "Arteta reacts to Champions League draw",
    "Okan Buruk: milli ara sonrası hedefimiz belli",
])
def test_phase_hint_no_false_post(title):
    assert phase_hint(title) != "mac-sonrasi"


@pytest.mark.parametrize("title", [
    "Arsenal 2-1 Chelsea: Arteta's verdict",
    "Arteta reacts to Chelsea win",
    "Mikel Arteta reacts to the draw with Chelsea",
    "Mikel Arteta reacts",
    "Fenerbahçe derbisi sonrası Okan Buruk",
    "2-1 win: Arteta on the comeback",
])
def test_phase_hint_real_post(title):
    assert phase_hint(title) == "mac-sonrasi"


def test_injury_timeline_presser_keeps_next_match():
    meta = _meta("Arteta press conference: Saka out for 2-3 weeks", "Arsenal", timestamp=_ts("2026-11-06 12:00"))
    assert (meta["gw"], meta["phase"]) == (11, "mac-oncesi")


@pytest.mark.parametrize("title,channel,desc", [
    ("Arsenal vs Chelsea preview: can we finally win? | Podcast", "The Arsenal Vision Podcast", ""),
    ("Chelsea 2-1 Arsenal | Fan reactions & player ratings", "Chelsea Fan TV", ""),
    ("Every Saka goal vs Chelsea", "Arsenal", ""),
    ("Arsenal x adidas | 2026/27 away kit launch", "Arsenal", "Worn for the first time against Chelsea on Sunday."),
    ("LIVE: Arsenal vs Chelsea | Match day show", "Arsenal", ""),
    ("Liverpool vs Chelsea preview | Blood Red podcast", "Liverpool Echo", ""),
    ("Arsenal vs Chelsea | Fan reactions", "Arsenal Fan TV", ""),
])
def test_club_channel_rule_needs_the_club_itself(title, channel, desc):
    info = {"title": title, "channel": channel, "description": desc}
    assert not looks_like_presser(info, parse_match(info))


@pytest.mark.parametrize("title,want", [
    ("ARNE SLOT'S PRESS CONFERENCE", "Arne Slot"),
    ("XABI ALONSO'S PRESS CONFERENCE", "Xabi Alonso"),
    ("İLHAN PALUT'UN AÇIKLAMALARI", "İlhan Palut"),
    ("Carrick on Mainoo's fitness", "Michael Carrick"),
    ("Howe on Isak", "Eddie Howe"),
    ("Andreas Pereira on his goal", "Andreas Pereira"),
    ("Nuno Mendes on PSG's win", "Nuno Mendes"),
    ("Giuliano Simeone on the derby", "Giuliano Simeone"),
    ("Scott McKenna on Forest", "Scott McKenna"),
    ("Coletiva de Imprensa | Abel Ferreira", "Abel Ferreira"),
    ("Rueda de Prensa | Javier Aguirre", "Javier Aguirre"),
    ("COLETIVA DE IMPRENSA | ABEL FERREIRA", "Abel Ferreira"),
    ("RUEDA DE PRENSA DE JAVIER AGUIRRE", "Javier Aguirre"),
    ("SAKA OUT FOR SIX WEEKS", None),
    ("KEVIN DE BRUYNE PRESS CONFERENCE", "Kevin de Bruyne"),
    ("RÉGIS LE BRIS PRESS CONFERENCE", "Régis Le Bris"),
    ("Nuno on the derby", "Nuno Espírito Santo"),
])
def test_manager_name_review_cases(title, want):
    assert find_manager(title, find_teams(title)) == want


def test_unsafe_surname_accepted_with_own_club():
    """'Alonso' tek başına belirsiz (Marcos Alonso), ama Chelsea kanalında Xabi Alonso."""
    assert find_manager("Alonso on Palmer's fitness", ["Chelsea"]) == "Xabi Alonso"
    assert find_manager("Alonso on Palmer's fitness", []) is None
    info = {"title": "Alonso on Palmer's fitness", "channel": "Chelsea FC"}
    assert looks_like_presser(info, parse_match(info))


@pytest.mark.parametrize("title,channel", [
    ("Carrick on Mainoo's fitness", "Manchester United"),
])
def test_known_short_manager_names_still_pressers(title, channel):
    info = {"title": title, "channel": channel}
    assert looks_like_presser(info, parse_match(info))


@pytest.mark.parametrize("title,channel", [
    ("Nuno Mendes | All goals 2025", "PSG"),
    ("Andreas Pereira skills and goals", "Fulham FC"),
])
def test_namesakes_are_not_managers(title, channel):
    info = {"title": title, "channel": channel}
    assert not looks_like_presser(info, parse_match(info))


@pytest.mark.parametrize("title,teams", [
    ("Zira FK vs Qarabağ", ["Zira", "Qarabağ"]),
    ("Zira FK - Qarabağ oyunundan sonra", ["Zira", "Qarabağ"]),
    ("Sabah FC - Neftçi", ["Sabah", "Neftçi"]),
    ("Sporting Clube de Portugal", ["Sporting CP"]),
    ("Qarabağ - Sabah mətbuat konfransı", ["Qarabağ", "Sabah"]),
    ("Arsenal - Sporting press conference", ["Arsenal", "Sporting CP"]),
    ("Qarabağ - Sabah matçından sonra", ["Qarabağ", "Sabah"]),
    ("Derby - Nice", ["Derby County", "Nice"]),
    ("Leeds 2-0 Derby highlights", ["Leeds United", "Derby County"]),
    ("Arteta: Sporting will be a tough test", ["Sporting CP"]),
    ("Spurs postgame reaction | NBC Sports", ["Tottenham"]),
])
def test_ambiguous_review_cases(title, teams):
    assert find_teams(title) == teams


def test_canon_team_resolves_ambiguous_fpl_names():
    from stt.dataset import canon_team
    assert canon_team("Derby") == "Derby County" and canon_team("Nice") == "Nice"


def test_joined_with_spaced_score():
    m = parse_match({"title": "Everton 0 - 0 Coventry City | Frank Lampard reacts", "channel": "Coventry City"})
    assert (m["home"], m["away"]) == ("Everton", "Coventry City")


def test_season_guard_uses_timestamp_when_no_upload_date():
    """Yerel dosya (yalnız timestamp): 2025'teki Maresca videosu Man City'ye yazılmaz."""
    m = parse_match({"title": "Enzo Maresca press conference | Chelsea", "timestamp": _ts("2025-12-01 12:00")})
    assert m["team"] == "Chelsea"
    m = parse_match({"title": "Enzo Maresca press conference | Chelsea", "timestamp": _ts("2026-11-01 12:00")})
    assert m["team"] == "Manchester City"


def test_fpl_attempts_every_10_minutes_not_20(tmp_path, monkeypatch):
    """pipeline.fpl() ile fpl_data'nın iki ayrı beklemesi birleşip ağ 20 dk'da bir deneniyordu."""
    import stt.dataset as ds
    import stt.pipeline as pl
    from stt.config import Config
    clock = [10_000.0]
    monkeypatch.setattr(ds.time, "time", lambda: clock[0])
    tries = []

    def boom(url):
        tries.append(clock[0])
        clock[0] += 20                      # zaman aşımı
        raise OSError("ağ yok")
    monkeypatch.setattr(ds, "_get_json", boom)
    monkeypatch.setattr(ds, "_fail_until", 0.0)
    pipe = pl.Pipeline(Config(out_dir=tmp_path))
    try:
        while clock[0] < 10_000 + 35 * 60:  # dakikada bir video bitiyor
            pipe.fpl()
            clock[0] += 60
    finally:
        pipe.close()
    gaps = [(b - a) / 60 for a, b in zip(tries, tries[1:])]
    assert len(tries) >= 3 and all(g <= 11.5 for g in gaps), gaps


def test_fpl_force_ignores_backoff(tmp_path, monkeypatch):
    import stt.dataset as ds
    tries = []

    def boom(url):
        tries.append(url)
        raise OSError("x")
    monkeypatch.setattr(ds, "_get_json", boom)
    monkeypatch.setattr(ds, "_fail_until", 0.0)
    assert ds.fpl_data(tmp_path) is None and len(tries) == 1
    assert ds.fpl_data(tmp_path) is None and len(tries) == 1
    assert ds.fpl_data(tmp_path, force=True) is None and len(tries) == 2


def test_reclassify_case_only_rename_keeps_files(tmp_path):
    """Windows'ta (NTFS büyük/küçük harf duyarsız) 'Erik Ten Hag' -> 'Erik ten Hag' yeni dosyaları siliyordu:
    yalnız harf büyüklüğü değişiyorsa kayıt yerinde kalır, eski adıyla."""
    from stt.reclassify import reclassify
    ds = {"team": "Bayer Leverkusen", "manager": "Erik Ten Hag", "presser_date": "2026-11-06",
          "match_date": "2026-11-06", "home": "Bayer Leverkusen"}
    p = _old_output(tmp_path, "y_hag", "Erik Ten Hag press conference", "Bayer Leverkusen", _ts("2026-11-06 12:00"),
                    "Diger Takimlar/Bayer Leverkusen/2026-2027/2026-11-06 Bayer Leverkusen basın toplantısı",
                    "2026-11-06 - Erik Ten Hag", ds)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 0 and not res["errors"], res
    assert p["txt"].exists() and p["json"].exists()
    assert not (p["txt"].parent / "2026-11-06 - Erik ten Hag.txt").exists()
    row = json.loads((tmp_path / "index.jsonl").read_text("utf-8"))
    assert row["json"].endswith("2026-11-06 - Erik Ten Hag.json") and row["manager"] == "Erik ten Hag"


def test_reclassify_old_doc_keeps_cup_from_description(tmp_path):
    """1.9.7 json'unda açıklama yok: açıklamadan doğru bulunmuş kupa maçı PL haftasına taşınmasın."""
    from stt.reclassify import reclassify
    ds = {"team": "Arsenal", "opponent": "Slavia Prague", "competition": "Champions League", "is_pl": True,
          "presser_date": "2026-11-06", "match_date": "2026-11-06", "home": "Arsenal", "away": "Slavia Prague"}
    p = _old_output(tmp_path, "y_cl", "Mikel Arteta press conference", "Arsenal", _ts("2026-11-06 12:00"),
                    "Premier League/Arsenal/2026-2027/2026-11-06 Champions League Arsenal vs Slavia Prague",
                    "2026-11-06 - Mikel Arteta", ds,
                    match={"home": "Arsenal", "away": "Slavia Prague", "team": "Arsenal", "manager": "Mikel Arteta"})
    ds2 = {"team": "Arsenal", "opponent": "Wycombe", "competition": "EFL Cup", "is_pl": True,
           "presser_date": "2026-11-09", "match_date": "2026-11-09", "home": "Arsenal", "away": "Wycombe"}
    p2 = _old_output(tmp_path, "y_cup", "Arsenal vs Wycombe | Mikel Arteta press conference", "Arsenal",
                     _ts("2026-11-09 13:00"), "Premier League/Arsenal/2026-2027/2026-11-09 EFL Cup Arsenal vs Wycombe",
                     "2026-11-09 - Mikel Arteta", ds2,
                     match={"home": "Arsenal", "away": "Wycombe", "team": "Arsenal", "manager": "Mikel Arteta"})
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 0, res["changes"]
    assert p["txt"].exists() and p2["txt"].exists()


def test_reclassify_survives_broken_source_info_and_keeps_extra_data(tmp_path):
    from stt.reclassify import reclassify
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    p = _old_output(tmp_path, "y_ek", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                    "2026-11-06 - Mikel Arteta", old)
    mac = p["txt"].parent / "mac.json"
    mac.write_text(json.dumps({"team": "Arsenal", "ek_bilgiler": {"kadro": ["Saka"]}}), encoding="utf-8")
    # bozuk bir başka kayıt bütün işi durdurmasın
    b = _old_output(tmp_path, "y_bozuk", "x", "Arsenal", _ts("2026-11-06 12:00"), "Premier League/Arsenal/b", "b", {})
    doc = json.loads(b["json"].read_text("utf-8"))
    doc["source_info"] = "bozuk"
    b["json"].write_text(json.dumps(doc), encoding="utf-8")
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] >= 1, res
    assert mac.exists() and json.loads(mac.read_text("utf-8"))["ek_bilgiler"] == {"kadro": ["Saka"]}


def test_reclassify_collections_heal_on_rerun(tmp_path, monkeypatch):
    """Yarıda kalan çalıştırma (koleksiyonlar güncellenmeden kapandı): sonraki çalıştırma yolları tamamlar."""
    import stt.reclassify as rc
    from stt.util import write_jsonl_csv
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    paths = _old_output(tmp_path, "y_heal", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                        _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                        "2026-11-06 - Mikel Arteta", old)
    col = tmp_path / "Koleksiyonlar" / "k"
    col.mkdir(parents=True)
    row = {"date": "2026-11-06", "title": "t", "url": "u", "txt": str(paths["txt"]), "json": str(paths["json"]),
           "key": "y_heal"}
    write_jsonl_csv(col / "liste.jsonl", col / "liste.csv", [row], list(row))
    monkeypatch.setattr(rc, "_fix_collections", lambda *a, **k: None)   # ilk çalıştırma yarıda kaldı
    rc.reclassify(tmp_path, apply=True, data=FPL)
    monkeypatch.undo()
    rc.reclassify(tmp_path, apply=True, data=FPL)
    r = json.loads((col / "liste.jsonl").read_text("utf-8"))
    assert "GW11" in r["txt"] and Path(r["txt"]).exists()


# ---------------------------------------------------------------- ikinci bağımsız inceleme
def test_neutral_channel_manager_club_not_its_own_opponent():
    m = parse_match({"title": "Enzo Maresca press conference", "channel": "Sky Sports News",
                     "timestamp": _ts("2026-11-06 12:00"),
                     "description": "Manchester City manager Enzo Maresca speaks to the media."})
    assert m["team"] == "Manchester City" and m["away"] != "Manchester City"
    meta = classify(m, {"title": "Enzo Maresca press conference", "timestamp": _ts("2026-11-06 12:00")}, FPL)
    assert (meta["gw"], meta["home"], meta["away"]) == (11, "Manchester City", "Manchester United")
    m = parse_match({"title": "Mikel Arteta press conference", "channel": "Sky Sports News",
                     "timestamp": _ts("2026-11-06 12:00"),
                     "description": "Arsenal boss Mikel Arteta previews Sunday's game against Chelsea."})
    assert (m["team"], m["home"], m["away"]) == ("Arsenal", "Arsenal", "Chelsea")


@pytest.mark.parametrize("title,want", [
    ("Maresca Unhappy With Referee", "Enzo Maresca"),
    ("Arteta Hails Saka After Chelsea Win", "Mikel Arteta"),
    ("Arteta Frustrated | Press Conference", "Mikel Arteta"),
    ("Moyes Confident Ahead Of Coventry Clash", "David Moyes"),
    ("Iraola Frustrated After Spurs Defeat", "Andoni Iraola"),
    ("LIVERPOOL: VAN DIJK PRESS CONFERENCE", "Van Dijk"),
    ("PRESS CONFERENCE | DE BRUYNE ON NAPOLI", "De Bruyne"),
    ("TOTTENHAM: DE ZERBI PRESS CONFERENCE", "Roberto De Zerbi"),
])
def test_manager_second_review_cases(title, want):
    assert find_manager(title, find_teams(title)) == want


@pytest.mark.parametrize("title,channel,ts,gw", [
    ("Mikel Arteta previews Boxing Day clash", "Arsenal", "2026-11-06 12:00", 11),
    ("Mikel Arteta Previews North London Derby", "Arsenal", "2026-11-06 12:00", 11),
    ("Mikel Arteta press conference ahead of Gameweek 11", "Arsenal", "2026-11-06 12:00", 11),
    ("Mikel Arteta previews Emirates Stadium clash", "Arsenal", "2026-11-06 12:00", 11),
    ("Andoni Iraola previews Merseyside Derby", "Liverpool FC", "2026-11-08 12:00", 11),
])
def test_generic_words_are_not_an_unknown_opponent(title, channel, ts, gw):
    meta = _meta(title, channel, timestamp=_ts(ts))
    assert meta["gw"] == gw, (meta["opponent"], meta["gw"])


def test_unknown_club_with_generic_word_not_matched_to_namesake():
    """'Exeter City' (kupa rakibi) Manchester City maçı sanılmasın ('city' ortak sözcük)."""
    meta = _meta("Mikel Arteta previews Exeter City", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert meta["gw"] is None and meta["opponent"] == "Exeter City"


@pytest.mark.parametrize("title,want", [
    ("Mikel Arteta reacts to dramatic late win", "mac-sonrasi"),
    ("Mikel Arteta reacts to North London derby win", "mac-sonrasi"),
    ("Mikel Arteta reacts to Arsenal's win over Chelsea", "mac-sonrasi"),
    ("Arteta reacts to FA Cup draw", None),
    ("Mikel Arteta: confidence high after derby win", None),
    ("Mikel Arteta: We must kick on after Chelsea win", None),
    ("Chelsea 25-26 Season Review | Enzo Maresca", None),
    ("Arsenal 2–1 Chelsea | Mikel Arteta", "mac-sonrasi"),
    ("Qarabağ 2:1 Sabah | Qurban Qurbanov", "mac-sonrasi"),
    ("Okan Buruk'un maç sonu açıklamaları", "mac-sonrasi"),
    ("Kasımpaşa deplasmanı sonrası Okan Buruk", "mac-sonrasi"),
    ("Qarabağ - Sabah matçından sonra Qurban Qurbanov", "mac-sonrasi"),
    ("Kickoff 16:30 | Arsenal vs Chelsea preview", "mac-oncesi"),
])
def test_phase_hint_second_review(title, want):
    assert phase_hint(title) == want


@pytest.mark.parametrize("title,teams", [
    ("Nice play! Arteta reacts to Chelsea win", ["Chelsea"]),
    ("Very nice, has to continue | Mikel Arteta press conference", []),
    ("Okan Buruk: Sabah iş gününe başlıyoruz", []),
    ("Qurban Qurbanov: Zira will be a tough test", ["Zira"]),
    ("Zirə - Qarabağ oyunundan sonra mətbuat konfransı", ["Zira", "Qarabağ"]),
    ("Başakşehir - Göztepe maçı öncesi açıklamalar", ["Başakşehir", "Göztepe"]),
])
def test_ambiguous_second_review(title, teams):
    assert find_teams(title) == teams


def test_turkish_presser_and_rams_channel():
    from stt.match import club_of_channel
    assert club_of_channel("Rams Başakşehir FK") == "Başakşehir"
    assert looks_like_presser({"title": "Başakşehir - Göztepe maçı öncesi açıklamalar"})


def test_reclassify_second_run_keeps_neutral_channel_result(tmp_path):
    """1.9.7 kaydı (tarafsız kanal, takımsız): 1. çalıştırma doğru GW; 2. çalıştırma 'Arsenal vs Arsenal' yapıyordu."""
    from stt.reclassify import reclassify
    ds = {"team": None, "presser_date": "2026-11-06", "match_date": "2026-11-06"}
    p = _old_output(tmp_path, "y_n1", "Mikel Arteta press conference", "Sky Sports News", _ts("2026-11-06 12:00"),
                    "Diger Takimlar/Bilinmeyen/Mikel Arteta press conference", "2026-11-06 - Mikel Arteta", ds,
                    match={"home": None, "away": None, "team": None, "manager": "Mikel Arteta"})
    r1 = reclassify(tmp_path, apply=True, data=FPL)
    r2 = reclassify(tmp_path, apply=True, data=FPL)
    r3 = reclassify(tmp_path, apply=True, data=FPL)
    assert r1["moved"] == 1 and r2["moved"] == 0 and r3["moved"] == 0, (r1, r2)
    row = json.loads((tmp_path / "index.jsonl").read_text("utf-8"))
    assert row["gw"] == 11 and row["txt"].startswith("Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea/")
    assert not p["txt"].exists()


@pytest.mark.parametrize("title,channel,old_away", [
    ("Pep Guardiola previews the Manchester derby", "Man City", "Derby County"),
    ("Sporting director Andrea Berta | Mikel Arteta press conference", "Arsenal", "Sporting CP"),
])
def test_reclassify_does_not_resurrect_old_title_false_positive(tmp_path, title, channel, old_away):
    from stt.reclassify import reclassify
    team = "Manchester City" if channel == "Man City" else "Arsenal"
    ds = {"team": team, "opponent": old_away, "is_pl": True, "presser_date": "2026-11-06",
          "match_date": "2026-11-06", "home": team, "away": old_away}
    _old_output(tmp_path, "y_fp", title, channel, _ts("2026-11-06 12:00"),
                f"Premier League/{team}/2026-2027/2026-11-06 {team} vs {old_away}", "2026-11-06 - x", ds,
                match={"home": team, "away": old_away, "team": team})
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 1 and res["gw_found"] == 1, res


def test_reclassify_locked_old_file_is_retried_next_run(tmp_path, monkeypatch):
    """Eski dosya silinemedi (açık): sonraki çalıştırma siler; hata özetinde görünür."""
    import stt.reclassify as rc
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    p = _old_output(tmp_path, "y_lock", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                    "2026-11-06 - Mikel Arteta", old)
    real_unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self == p["txt"]:
            raise PermissionError(13, "The process cannot access the file")
        return real_unlink(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", locked)
    r1 = rc.reclassify(tmp_path, apply=True, data=FPL)
    assert r1["moved"] == 1 and r1["errors"] and p["txt"].exists()
    monkeypatch.setattr(Path, "unlink", real_unlink)
    rc.reclassify(tmp_path, apply=True, data=FPL)
    assert not p["txt"].exists() and not p["txt"].parent.exists()


def test_reclassify_keeps_hand_edited_txt(tmp_path):
    """Not Defteri'nde elle düzeltilmiş .txt yeniden üretilip ezilmesin (taşınır)."""
    from stt.reclassify import reclassify
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    p = _old_output(tmp_path, "y_edit", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                    "2026-11-06 - Mikel Arteta", old)
    edited = p["txt"].read_text("utf-8").replace("he trained today", "he trained TODAY (elle)")
    p["txt"].write_bytes(edited.replace("\n", "\r\n").encode("utf-8"))      # Not Defteri: CRLF
    res = reclassify(tmp_path, apply=True, data=FPL)
    new = tmp_path / "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea/2026-11-06 mac-oncesi - Mikel Arteta.txt"
    assert res["moved"] == 1 and "TODAY (elle)" in new.read_text("utf-8")


def test_reclassify_regenerates_unedited_txt_with_windows_newlines(tmp_path):
    """Windows'ta program .txt'yi CRLF yazar: bu 'elle düzeltilmiş' sayılıp başlık eski kalmasın."""
    from stt.reclassify import reclassify
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    p = _old_output(tmp_path, "y_crlf", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                    "2026-11-06 - Mikel Arteta", old)
    p["txt"].write_bytes(p["txt"].read_text("utf-8").replace("\n", "\r\n").encode("utf-8"))
    reclassify(tmp_path, apply=True, data=FPL)
    new = tmp_path / "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea/2026-11-06 mac-oncesi - Mikel Arteta.txt"
    assert "GW11" in new.read_text("utf-8")


def test_gui_reader_follows_moved_file_and_start_waits(tmp_path, monkeypatch):
    import stt.gui as g
    import stt.reclassify as rc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc, "fpl_data", lambda d, **k: FPL)
    app = g.App()
    app._cfg.out_dir = tmp_path
    old = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
           "home": "Arsenal", "away": "Bukayo Saka's"}
    p = _old_output(tmp_path, "y_rd", "Mikel Arteta on Bukayo Saka's fitness | Press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's",
                    "2026-11-06 - Mikel Arteta", old)
    assert app.dataset_reclassify(True)["moved"] == 1
    assert "He is fine" in app.read_text(str(p["txt"]))          # açık okuyucu eski yolu tutuyor
    assert app.speakers_load(str(p["txt"]))["ok"]
    app._reclassifying = True
    r = app.start("https://youtu.be/abc", [], {})
    assert r["count"] == 0 and r.get("error")
    app._reclassifying = False


# ---------------------------------------------------------------- üçüncü bağımsız inceleme
_SAKA_OLD = {"team": "Arsenal", "opponent": "Bukayo Saka's", "is_pl": True, "presser_date": "2026-11-06",
             "home": "Arsenal", "away": "Bukayo Saka's"}
_SAKA_TITLE = "Mikel Arteta on Bukayo Saka's fitness | Press conference"
_SAKA_FOLDER = "Premier League/Arsenal/2026-2027/2026-11-06 Arsenal vs Bukayo Saka's"
_SAKA_NEW = "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea/2026-11-06 mac-oncesi - Mikel Arteta"


def _lock(monkeypatch, path, op="unlink"):
    real = getattr(Path, op)

    def locked(self, *a, **k):
        if self == path:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, op, locked)


def test_pending_delete_keeps_file_edited_after_move(tmp_path, monkeypatch):
    """Eski .txt açıktı (silinemedi), sonra kullanıcı o eski dosyada düzeltme yapıp kaydetti: silinmemeli."""
    import os
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_e1", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    _lock(monkeypatch, p["txt"])
    reclassify(tmp_path, apply=True, data=FPL)
    monkeypatch.undo()
    p["txt"].write_text(p["txt"].read_text("utf-8") + "\nELLE EKLENDİ", encoding="utf-8")
    st = p["txt"].stat()
    os.utime(p["txt"], ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert p["txt"].exists() and "ELLE EKLENDİ" in p["txt"].read_text("utf-8")
    assert any("değiş" in e["error"] for e in res["errors"]), res["errors"]


def test_pending_delete_does_not_touch_reused_name(tmp_path, monkeypatch):
    """Kullanıcı artık klasörü elle silmiş, aynı ada başka kayıt yazılmış: bekleyen silme ona dokunmamalı."""
    import shutil
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_k1", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    _lock(monkeypatch, p["txt"])
    reclassify(tmp_path, apply=True, data=FPL)
    monkeypatch.undo()
    shutil.rmtree(p["txt"].parent)
    ds = {"team": "Arsenal", "presser_date": "2026-11-06", "home": "Arsenal", "general": True}
    k2 = _old_output(tmp_path, "y_k2", "Some podcast", "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                     "2026-11-06 - Mikel Arteta", ds, mode="herkes")
    reclassify(tmp_path, apply=True, data=FPL)
    assert k2["json"].exists() and k2["txt"].exists()


def test_unreadable_txt_aborts_move(tmp_path, monkeypatch):
    """Eski .txt okunamıyorsa (açık) elle düzeltilmiş mi anlaşılamaz: kayıt bu sefer taşınmaz."""
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_ur", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    _lock(monkeypatch, p["txt"], "read_bytes")
    res = reclassify(tmp_path, apply=True, data=FPL)
    monkeypatch.undo()
    assert res["errors"] and res["moved"] == 0 and p["txt"].exists() and p["json"].exists()
    assert not (tmp_path / f"{_SAKA_NEW}.txt").exists() and not (tmp_path / _SAKA_NEW).parent.exists()
    row = json.loads((tmp_path / "index.jsonl").read_text("utf-8"))
    assert row["json"] == f"{_SAKA_FOLDER}/2026-11-06 - Mikel Arteta.json"


def test_same_place_update_keeps_hand_edited_srt(tmp_path):
    from stt.reclassify import reclassify
    reclassify_dir = None
    p = _old_output(tmp_path, "y_sp", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    reclassify(tmp_path, apply=True, data=FPL)                       # doğru yere taşındı
    srt = tmp_path / f"{_SAKA_NEW}.srt"
    srt.write_text(srt.read_text("utf-8").replace("How is Saka?", "How is Bukayo Saka?"), encoding="utf-8")
    doc = json.loads((tmp_path / f"{_SAKA_NEW}.json").read_text("utf-8"))
    doc["dataset"]["score"] = None                                    # skor sonradan gelecek (yerinde güncelleme)
    doc["dataset"].pop("finished", None)
    (tmp_path / f"{_SAKA_NEW}.json").write_text(json.dumps(doc), encoding="utf-8")
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["updated"] == 1 and res["moved"] == 0, (res, reclassify_dir, p)
    assert "How is Bukayo Saka?" in srt.read_text("utf-8")


def test_edited_txt_gets_new_header_and_keeps_body(tmp_path):
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_hd", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    p["txt"].write_text(p["txt"].read_text("utf-8").replace("trained today", "trained today (elle)"), encoding="utf-8")
    reclassify(tmp_path, apply=True, data=FPL)
    txt = (tmp_path / f"{_SAKA_NEW}.txt").read_text("utf-8")
    head = txt.split("\n\n")[0]
    assert "trained today (elle)" in txt and "GW11" in head and "vs Bukayo Saka's" not in head


def test_malformed_pending_file_does_not_break(tmp_path):
    from stt.reclassify import reclassify
    (tmp_path / ".stt").mkdir()
    (tmp_path / ".stt" / "siniflandirma_bekleyen.json").write_text(
        json.dumps([{"key": "x", "dir": "a", "base": "b"}, "bozuk", {"key": "y"}]), encoding="utf-8")
    p = _old_output(tmp_path, "y_mf", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["ok"] and res["moved"] == 1 and not p["txt"].exists()


@pytest.mark.parametrize("title", ["Mikel Arteta previews Sporting", "Mikel Arteta press conference ahead of Sporting clash"])
def test_neutral_channel_keeps_pattern_opponent(title):
    m = parse_match({"title": title, "channel": "Sky Sports News", "timestamp": _ts("2026-11-09 18:00")})
    assert (m["team"], m["away"]) == ("Arsenal", "Sporting CP")
    meta = classify(m, {"title": title, "timestamp": _ts("2026-11-09 18:00")}, FPL)
    assert meta["gw"] is None


@pytest.mark.parametrize("title,team", [
    ("Sporting Will Be A Tough Test | Mikel Arteta Press Conference", "Sporting CP"),
    ("SPORTING WILL BE A TOUGH TEST | MIKEL ARTETA", "Sporting CP"),
    ("Arteta: 'Sporting Have Improved A Lot'", "Sporting CP"),
])
def test_subject_rule_any_case(title, team):
    assert team in find_teams(title)


@pytest.mark.parametrize("title", ["Mikel Arteta previews Athletic", "Mikel Arteta previews Albion"])
def test_suffix_only_club_names_are_opponents(title):
    meta = _meta(title, "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert meta["gw"] is None and meta["opponent"] in ("Athletic", "Albion")


def test_suffix_name_matching_next_fixture():
    meta = _meta("Mikel Arteta previews City", "Arsenal", timestamp=_ts("2026-11-09 13:00"))
    assert (meta["gw"], meta["home"]) == (12, "Manchester City")


@pytest.mark.parametrize("title", [
    "Mikel Arteta reacts to Bukayo Saka's 3-week injury | Press conference",
    "Arteta reacts to Round of 16 draw",
    "Arteta reacts to the last 16 draw",
    "1:1 with Mikel Arteta | Arsenal",
])
def test_phase_hint_third_review(title):
    assert phase_hint(title) != "mac-sonrasi"


def test_injury_reaction_keeps_gw():
    meta = _meta("Mikel Arteta reacts to Bukayo Saka's 3-week injury | Press conference", "Arsenal",
                 timestamp=_ts("2026-11-06 12:00"))
    assert (meta["gw"], meta["phase"]) == (11, "mac-oncesi")


def test_month_names_are_not_managers():
    assert find_manager("RUEDA DE PRENSA | 8 DE NOVIEMBRE | MANOLO GONZÁLEZ", []) == "Manolo González"
    # ad da olan ay sözcükleri adları bozmasın (Portekizce "março" -> "marco")
    assert find_manager("Marco Rose press conference", []) == "Marco Rose"
    assert find_manager("Solly March on the win", []) == "Solly March"
    assert find_manager("Julio Enciso speaks", []) == "Julio Enciso"


def test_follow_does_not_drop_videos_while_busy(tmp_path, monkeypatch):
    import stt.channel as ch
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    fid = app._follows().add("https://www.youtube.com/@Arsenal", "Arsenal", "basin")["id"]
    monkeypatch.setattr(ch, "check_follow", lambda row, cfg, llm=None: [{"url": "https://youtu.be/new1", "id": "new1"}])
    app._reclassifying = True
    app._check_follows(force=True)
    row = next(r for r in app._follows().all() if r["id"] == fid)
    assert "new1" not in (row.get("seen") or [])
    app._reclassifying = False
    app._check_follows(force=True)
    row = next(r for r in app._follows().all() if r["id"] == fid)
    assert "new1" in (row.get("seen") or [])


def test_gui_busy_guards_and_moved_paths(tmp_path, monkeypatch):
    import stt.gui as g
    import stt.reclassify as rc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc, "fpl_data", lambda d, **k: FPL)
    app = g.App()
    app._cfg.out_dir = tmp_path
    p = _old_output(tmp_path, "y_g", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    _lock(monkeypatch, p["txt"])
    assert app.dataset_reclassify(True)["moved"] == 1
    monkeypatch.undo()
    assert p["txt"].exists()                                    # eski kopya kaldı (açıktı)
    assert app._moved_path(str(p["txt"])) == str(tmp_path / f"{_SAKA_NEW}.txt")   # yine de yeni yer açılır
    app._reclassifying = True
    assert not app.speakers_save(str(p["txt"]), {}).get("ok")
    assert app.resume_pending().get("error")
    app._reclassifying = False


def test_every_known_manager_is_recognised():
    """Listedeki her hoca: tam ad, BÜYÜK HARF, iyelik ekiyle; soyadı tek başına (ortak soyadları hariç);
    PL hocası tarafsız kanalda kendi takımına. ("Jakirović" gibi Orta Avrupa harfleri tanınmıyordu.)"""
    from stt.match import _UNSAFE_SURNAMES, OTHER_MANAGERS, PL_MANAGERS, _norm
    for n in list(PL_MANAGERS) + list(OTHER_MANAGERS):
        for title in (f"{n} press conference", f"{n.upper()} PRESS CONFERENCE", f"{n}'s pre-match press conference",
                      f"{n.upper()}'S POST-MATCH PRESS CONFERENCE", f"Press conference | {n} on the win"):
            assert find_manager(title, find_teams(title)) == n, title
        sur = n.split()[-1]
        want = None if _norm(sur) in _UNSAFE_SURNAMES else n
        assert find_manager(f"{sur} on the win", []) == want, sur
        if n in PL_MANAGERS:
            info = {"title": f"{n} press conference", "channel": "Sky Sports News", "upload_date": "20261106"}
            m = parse_match(info)
            assert m["team"] == PL_MANAGERS[n] and looks_like_presser(info, m), n


# ---------------------------------------------------------------- dördüncü bağımsız inceleme
def test_reclassify_does_not_hold_ui_lock_while_asking_pipeline(tmp_path, monkeypatch):
    """Kilit sırası: Pipeline kilidi altında arayüze olay gönderilir (App._lock); yeniden sınıflandırma App._lock
    tutarken Pipeline kilidini isterse ("Durdur" ile aynı anda) iki iş parçacığı birbirini bekleyip donuyordu."""
    import stt.gui as g
    import stt.reclassify as rc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc, "fpl_data", lambda d, **k: FPL)
    app = g.App()
    app._cfg.out_dir = tmp_path
    seen = []

    class Pipe:
        def active(self):
            seen.append(app._lock.locked())
            return False
    app._pipe = Pipe()
    app.dataset_reclassify(False)
    app._pipe = None
    assert seen and not any(seen)


def test_edited_txt_without_generated_header_kept_as_is(tmp_path):
    """Kullanıcı başlığı silmiş ya da en üste not eklemiş: ilk blok 'başlık' sanılıp atılmasın."""
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_fb", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    txt = p["txt"].read_text("utf-8")
    body = txt.split("\n\n", 1)[1].replace("How is Saka?", "How is Bukayo Saka?")
    p["txt"].write_text(body, encoding="utf-8")                            # başlık silindi, ilk soru düzeltildi
    reclassify(tmp_path, apply=True, data=FPL)
    new = (tmp_path / f"{_SAKA_NEW}.txt").read_text("utf-8")
    assert "How is Bukayo Saka?" in new and "He is fine" in new


def test_note_on_top_of_txt_kept(tmp_path):
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_nt", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    p["txt"].write_text("NOTUM: önemli\n\n" + p["txt"].read_text("utf-8"), encoding="utf-8")
    reclassify(tmp_path, apply=True, data=FPL)
    assert "NOTUM: önemli" in (tmp_path / f"{_SAKA_NEW}.txt").read_text("utf-8")


def test_same_place_update_keeps_edit_when_another_write_fails(tmp_path, monkeypatch):
    """Yerinde güncelleme: özet dosyası açık (yazılamadı) olsa da kullanıcının metin düzeltmesi kaybolmasın."""
    from stt.reclassify import reclassify
    _old_output(tmp_path, "y_pf", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    reclassify(tmp_path, apply=True, data=FPL)
    txt = tmp_path / f"{_SAKA_NEW}.txt"
    txt.write_text(txt.read_text("utf-8").replace("trained today", "trained today (elle)"), encoding="utf-8")
    jp = tmp_path / f"{_SAKA_NEW}.json"
    doc = json.loads(jp.read_text("utf-8"))
    doc["dataset"]["score"] = None
    jp.write_text(json.dumps(doc), encoding="utf-8")
    md = tmp_path / f"{_SAKA_NEW} - ozet.md"
    import stt.util as u
    real = u.replace_retry

    def locked(src, dst, tries=5):
        if Path(dst) == md:
            raise PermissionError(13, "The process cannot access the file")
        return real(src, dst, tries)
    monkeypatch.setattr(u, "replace_retry", locked)
    reclassify(tmp_path, apply=True, data=FPL)
    monkeypatch.undo()
    assert "trained today (elle)" in txt.read_text("utf-8")


def test_record_moved_twice_while_locked_cleans_both(tmp_path, monkeypatch):
    from stt.reclassify import reclassify
    p = _old_output(tmp_path, "y_tw", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    _lock(monkeypatch, p["txt"])
    reclassify(tmp_path, apply=True, data=FPL)                        # 1. taşıma: eski .txt açık kaldı
    jp = tmp_path / f"{_SAKA_NEW}.json"
    doc = json.loads(jp.read_text("utf-8"))
    doc["dataset"]["gw"] = 99                                          # kayıt ikinci kez taşınsın diye bozuluyor
    jp.write_text(json.dumps(doc), encoding="utf-8")
    reclassify(tmp_path, apply=True, data=FPL)
    monkeypatch.undo()
    reclassify(tmp_path, apply=True, data=FPL)
    assert not p["txt"].exists()


def test_open_old_folder_with_other_pressers_stays(tmp_path, monkeypatch):
    import stt.gui as g
    import stt.reclassify as rc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc, "fpl_data", lambda d, **k: FPL)
    opened = []
    monkeypatch.setattr(g, "open_path", lambda p: opened.append(p))
    app = g.App()
    app._cfg.out_dir = tmp_path
    p = _old_output(tmp_path, "y_of", _SAKA_TITLE, "Arsenal", _ts("2026-11-06 12:00"), _SAKA_FOLDER,
                    "2026-11-06 - Mikel Arteta", _SAKA_OLD)
    other = p["txt"].parent / "2026-11-06 - başka.txt"
    other.write_text("başka toplantı", encoding="utf-8")
    assert app.dataset_reclassify(True)["moved"] == 1
    app.open(str(p["txt"].parent))
    assert opened == [str(p["txt"].parent)]


@pytest.mark.parametrize("title,want", [
    ("Fenerbahçe şampiyonluk şansını sürdürdü | basın toplantısı", None),
    ("Sergen Yalçın şampiyonluk için iddialı", "Sergen Yalçın"),
    ("Łukasz Piszczek press conference", "Łukasz Piszczek"),
    ("Niko Kovač press conference", "Niko Kovač"),
])
def test_lowercase_extended_letters_not_names(title, want):
    assert find_manager(title, find_teams(title)) == want


@pytest.mark.parametrize("title,want", [
    ("Mikel Arteta reacts to last night's draw", "mac-sonrasi"),
    ("Mikel Arteta reacts to last weekend's draw", "mac-sonrasi"),
    ("Mikel Arteta reacts to last minute draw", "mac-sonrasi"),
    ("Mikel Arteta reacts to final day draw", "mac-sonrasi"),
    ("Mikel Arteta reacts to 10 man draw", "mac-sonrasi"),
    ("Mikel Arteta reacts to 10-man win", "mac-sonrasi"),
    ("Mikel Arteta reacts to 90th-minute winner", "mac-sonrasi"),
    ("Mikel Arteta reacts to 5 goal thriller", "mac-sonrasi"),
    ("Qarabağ - Sabah 3:0 | Qurban Qurbanov", "mac-sonrasi"),
    ("Qarabağ 3:0 | Mətbuat konfransı", "mac-sonrasi"),
    ("Qarabağ (2:1) Sabah", "mac-sonrasi"),
    ("Arteta reacts to Champions League draw", None),
    ("Arteta reacts to Carabao Cup quarter-final draw", None),
    ("Arteta reacts to Round of 16 draw", None),
    ("Arteta reacts to the last 16 draw", None),
    ("1-1 with Bukayo Saka | Arsenal", None),
])
def test_phase_hint_fourth_review(title, want):
    assert phase_hint(title) == want


@pytest.mark.parametrize("title,team", [
    ("Nice Is The Word | Mikel Arteta press conference", "Nice"),
    ("Being Nice Is Not Enough", "Nice"),
    ("Nice Can Be Better | Enzo Maresca", "Nice"),
    ("Sabah Can Uzun'un açıklamaları", "Sabah"),
])
def test_subject_rule_title_case_false_positives(title, team):
    assert team not in find_teams(title)


def test_resume_pending_busy_keeps_banner(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._pending = [{"uid": "u1", "kind": "url", "source": "https://youtu.be/a", "title": "a", "extra": {}}]
    app._reclassifying = True
    r = app.resume_pending()
    assert r["ok"] is False and r["error"] and app._pending
    app._reclassifying = False


def test_reclassify_season_move_keeps_hand_added_match_info(tmp_path):
    """1.9.25: bu sezonun kaydı yalnız sezon klasörüne taşınırken mac.json'daki ek_bilgiler kaybolmaz."""
    from stt.reclassify import reclassify
    ds = {"team": "Arsenal", "opponent": "Chelsea", "is_pl": True, "gw": 11, "home": "Arsenal", "away": "Chelsea",
          "presser_date": "2026-11-06", "match_date": "2026-11-08", "phase": "mac-oncesi", "manager": "Mikel Arteta",
          "competition": "Premier League"}
    p = _old_output(tmp_path, "y_pre2", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-11-08 GW11 Arsenal vs Chelsea",
                    "2026-11-06 mac-oncesi - Mikel Arteta", ds)
    (p["txt"].parent / "mac.json").write_text(json.dumps({"team": "Arsenal", "ek_bilgiler": {"kadro": [1, 2]}}),
                                              encoding="utf-8")
    plan = reclassify(tmp_path, apply=False, data=FPL)
    assert plan["moved"] == 1 and plan["changes"][0]["to"] == \
        "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea"
    res = reclassify(tmp_path, apply=True, data=FPL)
    new = tmp_path / "Premier League" / "Arsenal" / "2026-2027" / "2026-11-08 GW11 Arsenal vs Chelsea"
    assert res["moved"] == 1 and (new / "2026-11-06 mac-oncesi - Mikel Arteta.txt").exists()
    m = json.loads((new / "mac.json").read_text("utf-8"))
    assert m["ek_bilgiler"] == {"kadro": [1, 2]} and m["gw"] == 11 and m["season"] == "2026-2027"
    assert not p["txt"].parent.exists() and (tmp_path / "Premier League" / "Arsenal").is_dir()


def test_gui_dataset_tree_has_seasons(tmp_path, monkeypatch):
    """1.9.25: arayüz ağacı takım › sezon › maç; eski (sezonsuz) satırlarda sezon tarihten hesaplanır."""
    import stt.gui as g
    from stt.dataset import update_index
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    for key, folder, md, season in [
            ("a", "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea", "2026-11-08", "2026-2027"),
            ("b", "Premier League/Arsenal/2025-2026/2026-03-02 GW30 Arsenal vs Chelsea", "2026-03-02", "2025-2026"),
            ("c", "Premier League/Arsenal/2025-09-14 GW04 Arsenal vs Chelsea", "2025-09-14", None),   # taşınmamış eski
            ("d", "Podcast ve Film/Bir podcast", "2026-11-01", None)]:
        (tmp_path / folder).mkdir(parents=True)
        update_index(tmp_path, {"key": key, "league": "Podcast ve Film" if key == "d" else "Premier League",
                                "team": "Arsenal", "match_date": md, "presser_date": md, "season": season,
                                "txt": f"{folder}/x.txt", "json": f"{folder}/x.json"})
    ds = app.dataset()
    pl = next(x for x in ds["leagues"] if x["league"] == "Premier League")
    tm = pl["teams"][0]
    assert tm["folder"] == str(tmp_path / "Premier League" / "Arsenal") and len(tm["matches"]) == 3
    assert [s["season"] for s in tm["seasons"]] == ["2026-2027", "2025-2026"]
    assert [len(s["matches"]) for s in tm["seasons"]] == [1, 2]
    assert tm["seasons"][0]["folder"] == str(tmp_path / "Premier League" / "Arsenal" / "2026-2027")
    pod = next(x for x in ds["leagues"] if x["league"] == "Podcast ve Film")
    assert pod["teams"][0]["seasons"] == []
