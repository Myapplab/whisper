"""1.9.28: eski sezonlarda (2023-24, 2024-25, 2025-26) maç + GW (uygulamayla gelen FPL sezon sonu fikstürleri),
o sezon PL'de olan (sonra düşen) takım Premier League'de, yeni düzen "<lig>/<sezon>/<takım>/<hoca>/<maç>",
yeniden sınıflandırma eski kayıtları yeni düzene taşır; index.csv yazımı fazladan alanla bozulmaz."""
from __future__ import annotations

import csv
import json
import os

from test_v198 import FPL, _old_output, _ts

import stt.dataset as ds
from stt.dataset import (INDEX_FIELDS, canon_team, classify, data_season, fpl_data, history_seasons, pl_teams,
                         season_data, set_team_manager, target_dir, update_index)
from stt.match import history_manager, is_team, manager_team, parse_match
from stt.util import write_jsonl_csv


# ---------------------------------------------------------------- gömülü geçmiş fikstürler
def test_history_file_has_three_full_seasons():
    h = history_seasons()
    assert {"2023-2024", "2024-2025", "2025-2026"} <= set(h)
    for s, d in h.items():
        assert len(d["teams"]) == 20 and len(d["fixtures"]) == 380, s
        assert data_season(d) == s
        assert all(f["event"] and f["kickoff_time"] and f["finished"] for f in d["fixtures"]), s
        # FPL adları ("Luton", "Sheffield Utd", "Nott'm Forest", "Spurs") sözlükteki takımlara eşleşir
        assert all(is_team(t) for t in pl_teams(d)), s
    assert {"Luton Town", "Sheffield United", "Burnley"} <= set(pl_teams(h["2023-2024"]))
    assert {"Leicester City", "Ipswich Town", "Southampton"} <= set(pl_teams(h["2024-2025"]))
    assert canon_team("Sheffield Utd") == "Sheffield United"


def test_season_data_picks_the_presser_season(tmp_path, monkeypatch):
    assert season_data("2026-11-06", FPL) is FPL                               # güncel sezon: FPL
    assert season_data("2024-01-10", FPL) is history_seasons()["2023-2024"]
    assert data_season(season_data("2024-01-10", FPL)) == "2023-2024"
    assert data_season(season_data("2025-08-20", FPL)) == "2025-2026"
    assert season_data("2022-10-01", FPL) is None                              # fikstürü yok
    assert season_data(None, FPL) is FPL
    # FPL'den indirilen sezon kopyası: gömülü dosyada olmayan sezon (ör. 2026-27 bitince) yine bulunur
    monkeypatch.setattr(ds, "_history", {})
    monkeypatch.setattr(ds, "_get_json", lambda url: {"teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}]}
                        if "bootstrap" in url else [{"id": 1, "event": 1, "kickoff_time": "2026-08-15T14:00:00Z",
                                                     "team_h": 1, "team_a": 1, "finished": True}])
    monkeypatch.setattr(ds, "_fail_until", 0.0)
    fpl_data(tmp_path, force=True)
    assert (tmp_path / "fpl_2026-2027.json").is_file()
    later = {"teams": {}, "fixtures": [{"kickoff_time": "2027-08-14T14:00:00Z"}]}   # FPL artık 2027-28'de
    got = season_data("2026-11-06", later, tmp_path)
    assert got and data_season(got) == "2026-2027"
    assert season_data("2026-11-06", later) is None                            # kopya klasörü verilmezse yok


def test_classify_old_season_finds_match_and_gw(tmp_path):
    """Güncel FPL verisi 2026-27; 2023-24 açılış maçı (Burnley 0-3 Man City, 11 Ağustos 2023) yine bulunur."""
    info = {"title": "Vincent Kompany press conference | Burnley v Man City", "channel": "Burnley FC",
            "timestamp": _ts("2023-08-09 13:00")}
    meta = set_team_manager(tmp_path, classify(parse_match(info), info, FPL, tmp_path / ".stt"))
    assert (meta["team"], meta["is_pl"], meta["gw"], meta["home"], meta["away"], meta["phase"]) == \
        ("Burnley", True, 1, "Burnley", "Manchester City", "mac-oncesi")
    assert meta["fpl_short"] == "BUR" and meta["score"] == "0-3" and meta["team_manager"] == "Vincent Kompany"
    assert target_dir(tmp_path, meta, "x").relative_to(tmp_path).parts == \
        ("Premier League", "2023-2024", "Burnley", "Vincent Kompany", "2023-08-11 GW01 Burnley vs Manchester City")


def test_relegated_team_stays_in_premier_league_for_its_season(tmp_path):
    """Luton 2023-24'te PL'deydi (sonra düştü): "Diğer Takımlar"a değil Premier League/2023-2024'e."""
    info = {"title": "Rob Edwards pre-match press conference | Luton Town v Tottenham", "channel": "Luton Town FC",
            "timestamp": _ts("2023-10-05 12:00")}
    meta = set_team_manager(tmp_path, classify(parse_match(info), info, FPL))
    assert meta["is_pl"] and meta["gw"] == 8 and meta["away"] == "Tottenham" and meta["team_manager"] == "Rob Edwards"
    assert target_dir(tmp_path, meta, "x").relative_to(tmp_path).parts[:4] == \
        ("Premier League", "2023-2024", "Luton Town", "Rob Edwards")
    # ertelenip 3 Ekim'de oynanan GW7 maçı (Luton - Burnley) gerçek tarihiyle
    info = {"title": "Rob Edwards post-match press conference | Luton Town 1-2 Burnley", "channel": "Luton Town FC",
            "timestamp": _ts("2023-10-03 21:00")}
    meta = classify(parse_match(info), info, FPL)
    assert (meta["gw"], meta["match_date"], meta["score"]) == (7, "2023-10-03", "1-2")
    # bu sezon PL'de olmayan takım bu sezonun kaydında "Diğer Takımlar"da kalır
    info = {"title": "Rob Edwards press conference", "channel": "Luton Town FC", "timestamp": _ts("2026-11-05 12:00")}
    assert not classify(parse_match(info), info, FPL)["is_pl"]


def test_old_season_managers_in_history():
    assert history_manager("Liverpool", "2024-01-10") == "Jürgen Klopp"
    assert history_manager("Sheffield United", "2023-11-01") == "Paul Heckingbottom"
    assert history_manager("Sheffield United", "2023-12-10") == "Chris Wilder"
    assert history_manager("Crystal Palace", "2024-01-10") == "Roy Hodgson"
    assert history_manager("Leicester City", "2025-02-01") == "Ruud van Nistelrooy"
    assert history_manager("Southampton", "2025-01-15") == "Ivan Jurić"
    assert history_manager("Chelsea", "2023-11-01") == "Mauricio Pochettino"
    assert history_manager("Wolves", "2023-07-20") == "Julen Lopetegui"
    # tarafsız kanalın eski videosu: hocanın o günkü takımı
    assert manager_team("Jürgen Klopp", "2024-01-10") == "Liverpool"
    assert manager_team("Steve Cooper", "2023-10-01") == "Nottingham Forest"
    assert manager_team("Steve Cooper", "2024-09-01") == "Leicester City"
    m = parse_match({"title": "Jurgen Klopp press conference | Arsenal vs Liverpool", "channel": "Sky Sports News",
                     "upload_date": "20231221"})
    assert m["team"] == "Liverpool"


def test_finalize_old_season_does_not_write_fpl_repo(tmp_path):
    """Eski sezonun GW'si FPL reposunda bu sezonun GW klasörüne (presser.py) yazılmaz; bu sezonun ki yazılır."""
    from stt.config import Config
    from stt.pipeline import Item, Job, Pipeline
    repo = tmp_path / "fpl"
    repo.mkdir()
    (repo / "presser.py").write_text("import sys\n", encoding="utf-8")
    out = tmp_path / "out"
    pipe = Pipeline(Config(out_dir=out, fpl_repo=str(repo), fpl_auto=True))
    pipe.fpl = lambda: FPL
    jobs = []
    try:
        for i, (title, ts) in enumerate([
                ("Mikel Arteta pre-match press conference | Man Utd v Arsenal", _ts("2025-08-15 12:00")),
                ("Mikel Arteta's pre-Chelsea press conference", _ts("2026-11-06 12:00"))]):
            item = Item("url", f"https://www.youtube.com/watch?v=a{i}", title=title, key=f"y_a{i}" + "0" * 20)
            work = tmp_path / f"is{i}"
            (work / "parcalar").mkdir(parents=True)
            (work / "parcalar" / "0000.json").write_text(json.dumps(
                {"segments": [{"start": 0.0, "end": 4.0, "text": "Hello.", "words": []}], "language": "en"}),
                encoding="utf-8")
            job = Job(item, work, chunks=[{}], info={"title": title, "channel": "Arsenal", "duration_s": 4.0,
                                                     "timestamp": ts})
            pipe._finalize(job)
            assert job.status == "done", job.error
            jobs.append(job)
    finally:
        pipe.close()
    old, cur = jobs
    assert old.result["dataset"]["gw"] == 1 and "fpl" not in old.result
    assert old.out_dir.relative_to(out).parts == ("Premier League", "2025-2026", "Arsenal", "Mikel Arteta",
                                                  "2025-08-17 GW01 Manchester United vs Arsenal")
    assert cur.result["fpl"] == {"gw": 11, "team": "ARS"}


# ---------------------------------------------------------------- yeniden sınıflandırma
def test_team_of_recognises_all_layouts(tmp_path):
    from stt.reclassify import _team_of
    r = tmp_path
    pl = r / "Premier League"
    assert _team_of(pl / "Arsenal" / "2026-11-08 x", r) == "Arsenal"                              # eski
    assert _team_of(pl / "Arsenal" / "2025-2026" / "m", r) == "Arsenal"                          # 1.9.25
    assert _team_of(pl / "Arsenal" / "2025-2026 Mikel Arteta" / "m", r) == "Arsenal"             # 1.9.26
    assert _team_of(pl / "2025-2026" / "Arsenal" / "m", r) == "Arsenal"                          # 1.9.28
    assert _team_of(pl / "2025-2026" / "Arsenal" / "Mikel Arteta" / "m", r) == "Arsenal"
    assert _team_of(pl / "Sezon bilinmiyor" / "Galatasaray" / "m", r) == "Galatasaray"
    assert _team_of(r / "Diger Takimlar" / "2026-2027" / "Galatasaray" / "Okan Buruk" / "m", r) == "Galatasaray"
    assert _team_of(pl / "Arsenal" / "2025-2026 yedek" / "m", r) is None                         # elle açılmış
    assert _team_of(pl / "2025-2026" / "Arsenal" / "yedek" / "m", r) is None
    assert _team_of(r / "Podcast ve Film" / "x", r) is None


def test_reclassify_old_seasons_get_gw_and_relegated_team_moves_to_pl(tmp_path):
    """Kullanıcının durumu: 1.9.27 eski sezon kayıtlarını eski düzende, GW'siz bırakmıştı; Luton "Diğer
    Takımlar"daydı. Deneme turu sayıları, sonra uygulama: GW bulunur, yeni düzene taşınır, ek_bilgiler kaybolmaz."""
    from stt.reclassify import reclassify
    luton = _old_output(
        tmp_path, "y_lut", "Rob Edwards pre-match press conference | Luton Town v Tottenham", "Luton Town FC",
        _ts("2023-10-05 12:00"), "Diger Takimlar/Luton Town/2023-2024 Rob Edwards/2023-10-05 Luton Town vs Tottenham",
        "2023-10-05 mac-oncesi - Rob Edwards",
        {"team": "Luton Town", "opponent": "Tottenham", "is_pl": False, "presser_date": "2023-10-05",
         "match_date": "2023-10-05", "home": "Luton Town", "away": "Tottenham", "manager": "Rob Edwards",
         "phase": "mac-oncesi"})
    (luton["txt"].parent / "mac.json").write_text(json.dumps({"team": "Luton Town", "ek_bilgiler": {"not": "x"}}),
                                                  encoding="utf-8")
    ars = _old_output(
        tmp_path, "y_ars", "Mikel Arteta pre-match press conference | Arsenal v Leeds", "Arsenal",
        _ts("2025-08-22 12:00"),
        "Premier League/Arsenal/2025-2026 Mikel Arteta/2025-08-23 GW02 Arsenal vs Leeds United",
        "2025-08-22 mac-oncesi - Mikel Arteta",
        {"team": "Arsenal", "opponent": "Leeds United", "is_pl": True, "gw": 2, "presser_date": "2025-08-22",
         "match_date": "2025-08-23", "home": "Arsenal", "away": "Leeds United", "manager": "Mikel Arteta",
         "phase": "mac-oncesi", "competition": "Premier League"})
    (ars["txt"].parent / "mac.json").write_text(json.dumps({"team": "Arsenal", "ek_bilgiler": {"kadro": [1]}}),
                                                encoding="utf-8")
    plan = reclassify(tmp_path, apply=False, data=FPL)
    assert plan["ok"] and plan["moved"] == 2 and plan["to_pl"] == 1 and plan["gw_found"] == 1, plan
    assert plan["by_season"] == {"2023-2024": 1, "2025-2026": 1}
    assert "Premier League'e geçiyor" in plan["text"] and "2023-2024: 1" in plan["text"]
    assert luton["txt"].exists() and ars["txt"].exists()                       # deneme turu dosyaya dokunmaz

    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 2 and not res["errors"], res
    nl = tmp_path / "Premier League" / "2023-2024" / "Luton Town" / "Rob Edwards" / "2023-10-07 GW08 Luton Town vs Tottenham"
    assert (nl / "2023-10-05 mac-oncesi - Rob Edwards.txt").exists()
    m = json.loads((nl / "mac.json").read_text("utf-8"))
    assert m["gw"] == 8 and m["season"] == "2023-2024" and m["team_manager"] == "Rob Edwards" and m["score"] == "0-1"
    # maç adı değişti (GW eklendi): eski mac.json'daki ek_bilgiler eski klasörde korunur, klasör silinmez
    assert json.loads((luton["txt"].parent / "mac.json").read_text("utf-8"))["ek_bilgiler"] == {"not": "x"}
    assert not luton["txt"].exists()
    na = tmp_path / "Premier League" / "2025-2026" / "Arsenal" / "Mikel Arteta" / "2025-08-23 GW02 Arsenal vs Leeds United"
    assert (na / "2025-08-22 mac-oncesi - Mikel Arteta.txt").exists()
    ma = json.loads((na / "mac.json").read_text("utf-8"))
    assert ma["ek_bilgiler"] == {"kadro": [1]} and ma["score"] == "5-0"         # aynı maç: ek_bilgiler taşındı
    assert not (tmp_path / "Premier League" / "Arsenal").exists()               # boşalan eski takım klasörü
    rows = {r["key"]: r for r in (json.loads(x) for x in (tmp_path / "index.jsonl").read_text("utf-8").splitlines())}
    assert rows["y_lut"]["league"] == "Premier League" and rows["y_lut"]["gw"] == 8
    assert rows["y_lut"]["txt"].startswith("Premier League/2023-2024/Luton Town/Rob Edwards/")
    again = reclassify(tmp_path, apply=True, data=FPL)
    assert again["moved"] == 0 and again["updated"] == 0 and again["unchanged"] == 2, again


# ---------------------------------------------------------------- index.csv
def test_index_csv_tolerates_unknown_and_missing_fields(tmp_path):
    """Kullanıcının günlüğü: "dict contains fields not in fieldnames: 'speaker_role', 'opponent_short' ..." (eski
    1.2.1 kodu yeni alanlı index.jsonl'yi yazarken). index.jsonl'de bilinmeyen alan (başka / yeni sürüm) ya da eksik
    alan (eski sürüm) olsa da index.csv yazılır, sütunlar INDEX_FIELDS."""
    jl = tmp_path / "index.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in [
        {"key": "old", "team": "Arsenal", "league": "Premier League"},                      # 1.2.x satırı
        {"key": "new", "team": "Chelsea", "league": "Premier League", "speaker_role": "hoca", "gelecek_alan": 1},
    ]) + "\n", encoding="utf-8")
    update_index(tmp_path, {"key": "x", "team": "Everton", "league": "Premier League", "season": "2025-2026",
                            "baska_alan": "y"})
    with open(tmp_path / "index.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == INDEX_FIELDS and len(rows) == 3
    assert {r["key"]: r["speaker_role"] for r in rows}["new"] == "hoca"
    # write_jsonl_csv: sütunda olmayan alan hata vermez
    write_jsonl_csv(tmp_path / "l.jsonl", tmp_path / "l.csv", [{"a": 1, "b": 2}], ["a"])
    assert (tmp_path / "l.csv").read_text("utf-8-sig").splitlines() == ["a", "1"]
    assert os.path.getsize(tmp_path / "l.jsonl") > 0
