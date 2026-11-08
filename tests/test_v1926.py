"""1.9.26: veri setinde sezon klasörü o günkü teknik direktörle ("2025-2026 Pep Guardiola"); sezon ortasında ya da
yeni sezonda hoca değişince kayıtlar yeni hocanın klasörüne ("2026-2027 Enzo Maresca")."""
from __future__ import annotations

import json

from test_v198 import FPL, _old_output, _ts

from stt.dataset import index_row, season_folder, set_team_manager, target_dir, team_manager, update_index
from stt.match import (MANAGER_HISTORY, MANAGER_HISTORY_CHECKED, PL_MANAGERS, _known_manager, history_manager,
                       history_team, manager_team, parse_match)


# ---------------------------------------------------------------- hoca geçmişi tablosu
def test_history_manager_by_date():
    assert history_manager("Manchester City", "2026-03-01") == "Pep Guardiola"
    assert history_manager("Manchester City", "2026-09-01") == "Enzo Maresca"
    # Chelsea 2025-26: Maresca -> McFarlane (geçici) -> Rosenior -> McFarlane -> Alonso (2026-27)
    assert [history_manager("Chelsea", d) for d in ("2025-12-20", "2026-01-03", "2026-02-01", "2026-05-10",
                                                    "2026-08-20")] == \
        ["Enzo Maresca", "Calum McFarlane", "Liam Rosenior", "Calum McFarlane", "Xabi Alonso"]
    assert history_manager("Chelsea", "2026-01-06") == "Liam Rosenior"           # başladığı gün yeni hoca
    assert history_manager("Brighton", "2023-05-01") is None                    # tablonun başından önce
    assert history_manager("West Ham", "2026-09-01") is None                    # küme düştü: sonrası bilinmiyor
    assert history_manager("Galatasaray", "2026-09-01") is None
    assert history_manager("Chelsea", None) is None and history_manager("Chelsea", "tarih-yok") is None


def test_history_team_by_date():
    assert history_team("Andoni Iraola", "2025-12-01") == "Bournemouth"
    # yaz arasında atanan hoca 1 Temmuz'dan başlar: Haziran'da hâlâ eski takımının (sezon klasörü eski hocada)
    assert history_team("Andoni Iraola", "2026-06-10") == "Bournemouth"
    assert history_team("Andoni Iraola", "2026-07-10") == "Liverpool"
    assert history_manager("Liverpool", "2026-06-10") == "Arne Slot"
    assert history_team("Enzo Maresca", "2025-12-01") == "Chelsea"
    assert history_team("Enzo Maresca", "2026-03-01") is None                  # işsiz
    assert history_team("Nuno Espírito Santo", "2025-10-01") == "West Ham"


def test_history_table_is_sorted_and_matches_current_list():
    """Tablo tarih sırasında; bu sezonun PL listesi tablonun son girişiyle aynı (biri güncellenip öbürü unutulmasın)."""
    for team, rows in MANAGER_HISTORY.items():
        days = [d for d, _ in rows]
        assert days == sorted(days) and len(set(days)) == len(days), team
        assert not any(d[5:7] == "06" for d in days), (team, "Haziran ataması 1 Temmuz'dan başlamalı")
    for name, team in PL_MANAGERS.items():
        assert history_manager(team, MANAGER_HISTORY_CHECKED) == name, (team, name)
    # tablodaki bilinen hocalar soyadıyla tanınan tam adıyla yazılmış (yazım farkı eşleşmeyi bozmasın)
    for rows in MANAGER_HISTORY.values():
        for _, n in rows:
            k = _known_manager(n.split()) if n else None
            assert k in (None, n), n


def test_manager_team_uses_history_for_old_seasons():
    assert manager_team("Pep Guardiola", "2025-12-01") == "Manchester City"
    assert manager_team("Maresca", "2025-12-01") == "Chelsea"
    assert manager_team("Enzo Maresca", "2026-11-01") == "Manchester City"
    assert manager_team("Matthias Jaissle", "2026-07-15") is None              # henüz Newcastle'da değil
    assert manager_team("Mikel Arteta") == "Arsenal"                           # tarihsiz: bu sezonun listesi
    # tarafsız kanalın eski sezon videosu: takım başlıktaki ilk ad değil, o günkü hocanın takımı
    m = parse_match({"title": "Pep Guardiola press conference | Arsenal vs Man City", "channel": "Sky Sports News",
                     "upload_date": "20251201"})
    assert m["team"] == "Manchester City"
    m = parse_match({"title": "Ange Postecoglou press conference | Arsenal", "channel": "Sky Sports News",
                     "upload_date": "20250920"})
    assert m["team"] == "Nottingham Forest"


# ---------------------------------------------------------------- klasör adı
def _meta(team, day, manager=None, **kw):
    return {"team": team, "is_pl": True, "match_date": day, "presser_date": day, "manager": manager,
            "home": team, "away": "Arsenal" if team != "Arsenal" else "Chelsea", **kw}


def test_season_folder_has_manager_and_changes_with_manager(tmp_path):
    """Kullanıcının örneği: 2025-2026 Manchester City Guardiola, 2026-2027 Manchester City Maresca."""
    def folder(meta):
        return target_dir(tmp_path, set_team_manager(tmp_path, meta), "x").parent.name
    assert folder(_meta("Manchester City", "2026-03-01", "Pep Guardiola")) == "2025-2026 Pep Guardiola"
    assert folder(_meta("Manchester City", "2026-09-12", "Enzo Maresca")) == "2026-2027 Enzo Maresca"
    # sezon ortasında değişim: aynı sezonda ayrı klasörler; oyuncunun toplantısı o günkü hocanın klasörüne
    assert [folder(_meta("Chelsea", d, "Cole Palmer")) for d in ("2025-12-20", "2026-02-01", "2026-05-10")] == \
        ["2025-2026 Enzo Maresca", "2025-2026 Liam Rosenior", "2025-2026 Calum McFarlane"]
    assert folder(_meta("Nottingham Forest", "2025-11-01")) == "2025-2026 Sean Dyche"
    # maç tarihi belirler: maçtan önceki gün eski hocanın toplantısı da o maçın klasörüyle aynı yere
    assert folder(_meta("Chelsea", "2026-01-07", presser_date="2026-01-04")) == "2025-2026 Liam Rosenior"
    # tarih / takım yok: hoca yok
    assert folder({"team": "Galatasaray"}) == "Sezon bilinmiyor"
    assert season_folder("2026-2027", "Gary O'Neil") == "2026-2027 Gary O'Neil"
    assert season_folder("2026-2027", None) == "2026-2027" and season_folder(None, "x") == "Sezon bilinmiyor"


def test_team_manager_from_speaker_and_learned_from_index(tmp_path):
    # tabloda olmayan takım: kulübün kendi kanalında konuşan bilinen hoca
    gs = _meta("Galatasaray", "2026-10-01", "Okan Buruk", is_pl=False)
    assert team_manager(gs, channel="Galatasaray") == "Okan Buruk"
    assert team_manager(gs, channel="Sky Sports") is None                      # tarafsız kanal: emin değil
    # oyuncunun toplantısı, kayıt yokken: bilinmiyor (klasör yalnız sezon)
    assert team_manager(_meta("Galatasaray", "2026-10-05", "Barış Alper Yılmaz", is_pl=False), tmp_path) is None
    # aynı sezonun başka kaydının hocası: o
    update_index(tmp_path, {"key": "g1", "team": "Galatasaray", "league": "Diger", "manager": "Okan Buruk",
                            "team_manager": "Okan Buruk", "match_date": "2026-09-20", "presser_date": "2026-09-19"})
    update_index(tmp_path, {"key": "g0", "team": "Galatasaray", "league": "Diger", "manager": "Okan Buruk",
                            "team_manager": "Okan Buruk", "match_date": "2026-05-20"})   # önceki sezon: sayılmaz
    update_index(tmp_path, {"key": "g9", "team": "Galatasaray", "league": "Diger", "manager": "Diego Simeone",
                            "team_manager": None, "match_date": "2026-10-03"})   # rakip hoca: hocasız satır sayılmaz
    m = _meta("Galatasaray", "2026-10-05", "Barış Alper Yılmaz", is_pl=False)
    assert team_manager(m, tmp_path, "g2") == "Okan Buruk"
    assert team_manager(_meta("Galatasaray", "2027-08-05", "Barış Alper Yılmaz", is_pl=False), tmp_path) is None
    # başka takımın hocası konuşuyor (Sky: "Guardiola on Leverkusen"), kulüp kanalında bile: sayılmaz
    assert team_manager(_meta("Bayer Leverkusen", "2025-12-01", "Pep Guardiola", is_pl=False), tmp_path,
                        channel="Bayer 04 Leverkusen") is None
    # tablodaki takım: oyuncu konuşsa da tablo; podcast / takımsız: yok
    assert team_manager(_meta("Arsenal", "2026-10-01", "Bukayo Saka")) == "Mikel Arteta"
    assert team_manager({**_meta("Arsenal", "2026-10-01"), "general": True}) is None
    assert team_manager({"team": None, "match_date": "2026-10-01"}) is None


def test_other_managers_never_open_fake_folders(tmp_path):
    """İnceleme bulgusu: tarafsız kanalda başka hocanın videosu ("Hansi Flick press conference | Arsenal vs
    Barcelona") tablonun doğrulandığı günden sonra "2026-2027 Hansi Flick" klasörü açıyordu; aynı maçın oyuncu
    toplantısı başka klasöre düşüyordu. Tablodaki takımda her zaman tablo."""
    from stt.dataset import classify
    for day, title, channel in [
            ("20261124", "Hansi Flick press conference | Arsenal vs Barcelona | Champions League", "TNT Sports"),
            ("20261120", "Pep Guardiola on Arsenal's title chances", "Sky Sports News"),
            ("20270110", "Thomas Frank press conference | Arsenal", "Sky Sports News")]:
        info = {"title": title, "channel": channel, "upload_date": day}
        meta = classify(parse_match(info), info, None)
        assert meta["team"] == "Arsenal", title
        assert team_manager(meta, tmp_path, channel=channel) == "Mikel Arteta", title
    # tabloda olmayan takım: tarafsız kanaldaki başka hoca da sayılmaz
    assert team_manager(_meta("Galatasaray", "2026-10-21", "Diego Simeone", is_pl=False), tmp_path,
                        channel="beIN SPORTS") is None
    # Haziran'da (maçsız) yeni hocası açıklanan takımın oyuncu videosu biten sezonun hocasında
    assert team_manager(_meta("Liverpool", "2026-06-10", "Mohamed Salah")) == "Arne Slot"


def test_index_row_and_match_json_have_team_manager(tmp_path):
    from stt.dataset import write_match_json
    meta = set_team_manager(tmp_path, _meta("Manchester City", "2026-03-01", "Rodri"))
    doc = {"segments": [], "title": "t"}
    paths = {"txt": tmp_path / "a.txt", "json": tmp_path / "a.json"}
    row = index_row(tmp_path, "k", meta, doc, paths)
    assert row["team_manager"] == "Pep Guardiola" and row["manager"] == "Rodri" and row["season"] == "2025-2026"
    write_match_json(tmp_path, meta)
    assert json.loads((tmp_path / "mac.json").read_text("utf-8"))["team_manager"] == "Pep Guardiola"


# ---------------------------------------------------------------- yeniden sınıflandırma (eski kayıtlar)
def test_reclassify_moves_old_season_into_manager_folders(tmp_path):
    """1.9.25 düzeni (yalnız sezon klasörü): eski sezon kayıtları o günkü hocanın klasörüne; sezon ortasında
    değişen hoca ayrı klasör. İkinci çalıştırma bir şey değiştirmez."""
    from stt.reclassify import reclassify
    base_ds = {"team": "Chelsea", "is_pl": True, "competition": "Premier League", "home": "Chelsea",
               "away": "Arsenal", "opponent": "Arsenal"}
    a = _old_output(tmp_path, "y_c1", "Enzo Maresca press conference", "Chelsea", _ts("2025-12-19 12:00"),
                    "Premier League/Chelsea/2025-2026/2025-12-20 GW17 Chelsea vs Arsenal",
                    "2025-12-19 mac-oncesi - Enzo Maresca",
                    {**base_ds, "gw": 17, "match_date": "2025-12-20", "presser_date": "2025-12-19",
                     "manager": "Enzo Maresca", "phase": "mac-oncesi"})
    (a["txt"].parent / "mac.json").write_text(json.dumps({"team": "Chelsea", "score": "1-0",
                                                         "ek_bilgiler": {"not": "elle"}}), encoding="utf-8")
    b = _old_output(tmp_path, "y_c2", "Liam Rosenior press conference", "Chelsea", _ts("2026-01-30 12:00"),
                    "Premier League/Chelsea/2025-2026/2026-01-31 GW24 Chelsea vs Arsenal",
                    "2026-01-30 mac-oncesi - Liam Rosenior",
                    {**base_ds, "gw": 24, "match_date": "2026-01-31", "presser_date": "2026-01-30",
                     "manager": "Liam Rosenior", "phase": "mac-oncesi"})
    plan = reclassify(tmp_path, apply=False, data=FPL)
    assert plan["moved"] == 2 and a["txt"].exists() and b["txt"].exists()
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 2 and not res["errors"], res
    ch = tmp_path / "Premier League" / "Chelsea"
    na = ch / "2025-2026 Enzo Maresca" / "2025-12-20 GW17 Chelsea vs Arsenal"
    nb = ch / "2025-2026 Liam Rosenior" / "2026-01-31 GW24 Chelsea vs Arsenal"
    assert (na / "2025-12-19 mac-oncesi - Enzo Maresca.txt").exists()
    assert (nb / "2026-01-30 mac-oncesi - Liam Rosenior.txt").exists()
    assert not (ch / "2025-2026").exists()                                    # boşalan eski sezon klasörü silindi
    m = json.loads((na / "mac.json").read_text("utf-8"))
    assert m["score"] == "1-0" and m["ek_bilgiler"] == {"not": "elle"} and m["team_manager"] == "Enzo Maresca"
    rows = {r["key"]: r for r in (json.loads(x) for x in (tmp_path / "index.jsonl").read_text("utf-8").splitlines())}
    assert rows["y_c1"]["team_manager"] == "Enzo Maresca" and rows["y_c2"]["team_manager"] == "Liam Rosenior"
    assert rows["y_c2"]["txt"].startswith("Premier League/Chelsea/2025-2026 Liam Rosenior/")
    doc = json.loads((nb / "2026-01-30 mac-oncesi - Liam Rosenior.json").read_text("utf-8"))
    assert doc["dataset"]["team_manager"] == "Liam Rosenior" and doc["dataset"]["gw"] == 24
    again = reclassify(tmp_path, apply=True, data=FPL)
    assert again["moved"] == 0 and again["updated"] == 0 and again["other_season"] == 2


def test_reclassify_current_season_from_plain_season_folder(tmp_path):
    """Bu sezonun 1.9.25 kaydı ("Arsenal/2026-2027/...") hocalı klasöre taşınır; takım klasörü kalır."""
    from stt.reclassify import reclassify
    ds = {"team": "Arsenal", "opponent": "Chelsea", "is_pl": True, "gw": 11, "home": "Arsenal", "away": "Chelsea",
          "presser_date": "2026-11-06", "match_date": "2026-11-08", "phase": "mac-oncesi", "manager": "Mikel Arteta",
          "competition": "Premier League"}
    p = _old_output(tmp_path, "y_ars", "Mikel Arteta's pre-Chelsea press conference", "Arsenal",
                    _ts("2026-11-06 12:00"), "Premier League/Arsenal/2026-2027/2026-11-08 GW11 Arsenal vs Chelsea",
                    "2026-11-06 mac-oncesi - Mikel Arteta", ds)
    (p["txt"].parent / "mac.json").write_text(json.dumps({"team": "Arsenal", "ek_bilgiler": {"kadro": [1]}}),
                                              encoding="utf-8")
    res = reclassify(tmp_path, apply=True, data=FPL)
    new = tmp_path / "Premier League" / "Arsenal" / "2026-2027 Mikel Arteta" / "2026-11-08 GW11 Arsenal vs Chelsea"
    assert res["moved"] == 1 and (new / "2026-11-06 mac-oncesi - Mikel Arteta.txt").exists()
    assert json.loads((new / "mac.json").read_text("utf-8"))["ek_bilgiler"] == {"kadro": [1]}
    assert not (tmp_path / "Premier League" / "Arsenal" / "2026-2027").exists()
    assert (tmp_path / "Premier League" / "Arsenal").is_dir()


def test_reclassify_old_season_other_team_folder_not_touched(tmp_path):
    """Eski sezon kaydı başka takımın sezon klasöründe (elle taşınmış): yalnız sezon katmanı değişmiyor, dokunma."""
    from stt.reclassify import reclassify
    ds = {"team": "Chelsea", "is_pl": True, "presser_date": "2026-01-30", "match_date": "2026-01-31",
          "home": "Chelsea", "away": "Arsenal", "opponent": "Arsenal", "gw": 24}
    p = _old_output(tmp_path, "y_x", "Liam Rosenior press conference", "Chelsea", _ts("2026-01-30 12:00"),
                    "Premier League/Arsenal/2025-2026/2026-01-31 GW24 Chelsea vs Arsenal", "2026-01-30 - Liam Rosenior", ds)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 0 and res["other_season"] == 1 and p["txt"].exists()


# ---------------------------------------------------------------- pipeline + arayüz
def test_finalize_puts_presser_into_manager_folder(tmp_path):
    """Bitirme: oyuncunun toplantısı da o günkü hocanın klasörüne ("2025-2026 Liam Rosenior"); index'te team_manager,
    konuşan (manager) ayrı kalır."""
    from stt.config import Config
    from stt.pipeline import Item, Job, Pipeline
    out = tmp_path / "out"
    pipe = Pipeline(Config(out_dir=out))
    pipe.fpl = lambda: FPL
    try:
        title = "Cole Palmer press conference"
        item = Item("url", "https://www.youtube.com/watch?v=cp", title=title, key="y_cp" + "0" * 20)
        item.extra.update({"diarize": True})
        work = tmp_path / "is"
        (work / "parcalar").mkdir(parents=True)
        segs = [{"start": 0.0, "end": 4.0, "text": "How is the team?", "words": []},
                {"start": 4.0, "end": 12.0, "text": "We are fine.", "words": []}]
        (work / "parcalar" / "0000.json").write_text(json.dumps({"segments": segs, "language": "en"}), encoding="utf-8")
        (work / "konusmaci.json").write_text(json.dumps({"turns": [[0.0, 4.0, "B"], [4.0, 12.0, "A"]]}),
                                             encoding="utf-8")
        info = {"title": title, "channel": "Chelsea FC", "duration_s": 12.0, "timestamp": _ts("2026-02-01 13:00")}
        job = Job(item, work, chunks=[{}], info=info)
        pipe._finalize(job)
        assert job.status == "done", job.error
    finally:
        pipe.close()
    assert job.out_dir.parent == out / "Premier League" / "Chelsea" / "2025-2026 Liam Rosenior"
    row = json.loads((out / "index.jsonl").read_text("utf-8").splitlines()[0])
    assert row["team_manager"] == "Liam Rosenior" and row["manager"] == "Cole Palmer" and row["season"] == "2025-2026"
    assert job.result["dataset"]["team_manager"] == "Liam Rosenior"
    assert json.loads((job.out_dir / "mac.json").read_text("utf-8"))["team_manager"] == "Liam Rosenior"


def test_gui_dataset_tree_groups_by_season_and_manager(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    for key, folder, md, season, tm in [
            ("a", "Premier League/Chelsea/2025-2026 Enzo Maresca/2025-12-20 GW17 Chelsea vs Arsenal", "2025-12-20",
             "2025-2026", "Enzo Maresca"),
            ("b", "Premier League/Chelsea/2025-2026 Liam Rosenior/2026-01-31 GW24 Chelsea vs Arsenal", "2026-01-31",
             "2025-2026", "Liam Rosenior"),
            ("c", "Premier League/Chelsea/2026-2027 Xabi Alonso/2026-09-12 GW04 Chelsea vs Arsenal", "2026-09-12",
             "2026-2027", "Xabi Alonso"),
            ("d", "Premier League/Chelsea/2025-2026/2025-09-14 GW04 Chelsea vs Arsenal", "2025-09-14",
             "2025-2026", None)]:                                                        # taşınmamış 1.9.25 kaydı
        (tmp_path / folder).mkdir(parents=True)
        update_index(tmp_path, {"key": key, "league": "Premier League", "team": "Chelsea", "match_date": md,
                                "presser_date": md, "season": season, "team_manager": tm,
                                "txt": f"{folder}/x.txt", "json": f"{folder}/x.json"})
    ds = app.dataset()
    tm = next(x for x in ds["leagues"] if x["league"] == "Premier League")["teams"][0]
    assert [s["label"] for s in tm["seasons"]] == ["2026-2027 · Xabi Alonso", "2025-2026 · Liam Rosenior",
                                                   "2025-2026 · Enzo Maresca", "2025-2026"]
    assert [s["season"] for s in tm["seasons"]] == ["2026-2027", "2025-2026", "2025-2026", "2025-2026"]
    assert tm["seasons"][1]["manager"] == "Liam Rosenior"
    assert tm["seasons"][1]["folder"] == str(tmp_path / "Premier League" / "Chelsea" / "2025-2026 Liam Rosenior")
    assert tm["seasons"][3]["folder"] == str(tmp_path / "Premier League" / "Chelsea" / "2025-2026")


def test_season_dir_detection_and_truncated_match_names():
    from stt.reclassify import _is_season_dir, _same_match_name
    assert _is_season_dir("2025-2026") and _is_season_dir("2025-2026 Pep Guardiola")
    assert _is_season_dir("2026-2027 Gary O'Neil") and _is_season_dir("Sezon bilinmiyor")
    assert _is_season_dir("2025-2026 Okan Buruk")                              # bilinen hoca (takımı tabloda yok)
    assert not _is_season_dir("2025-2026 yedek") and not _is_season_dir("Arsenal")
    full = "2026-03-01 Champions League Arsenal vs Slavia Prague"
    assert _same_match_name(full, full[:30].rstrip(), full)                    # yol sınırı için kısaltılmış yeni ad
    assert _same_match_name(full[:40], full[:30], full)                        # eski ad da kısaltılmıştı
    plain = "2026-03-01 Arsenal vs Chelsea"
    assert not _same_match_name(plain + " (eski ad)", plain, plain)            # elle uzatılmış ad: başka klasör
    assert not _same_match_name(full, full[:30], None)


def test_reclassify_leaves_hand_made_season_folder(tmp_path):
    """Elle açılmış "2025-2026 yedek" klasöründeki eski sezon kaydı oradan çekilmez."""
    from stt.reclassify import reclassify
    ds = {"team": "Chelsea", "is_pl": True, "presser_date": "2026-01-30", "match_date": "2026-01-31",
          "home": "Chelsea", "away": "Arsenal", "opponent": "Arsenal", "gw": 24, "competition": "Premier League"}
    p = _old_output(tmp_path, "y_h", "Liam Rosenior press conference", "Chelsea", _ts("2026-01-30 12:00"),
                    "Premier League/Chelsea/2025-2026 yedek/2026-01-31 GW24 Chelsea vs Arsenal",
                    "2026-01-30 - Liam Rosenior", ds)
    res = reclassify(tmp_path, apply=True, data=FPL)
    assert res["moved"] == 0 and res["other_season"] == 1 and p["txt"].exists()
