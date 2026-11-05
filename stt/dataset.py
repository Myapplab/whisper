"""Veri seti düzeni: lig / takım / maç / basın toplantıları + index.csv.

    cikti/
      Premier League/
        Arsenal/
          2026-11-08 GW11 Arsenal vs Chelsea/
            mac.json                                   <- fikstür bilgisi (FPL), maça ait ek veriler buraya
            2026-11-07 mac-oncesi - Mikel Arteta.txt   <- soru-cevap metni
            2026-11-07 mac-oncesi - Mikel Arteta.json  <- zaman + konuşmacı
            2026-11-07 mac-oncesi - Mikel Arteta.srt
            2026-11-07 mac-oncesi - Mikel Arteta - ozet.md
          2026-11-04 Champions League Arsenal vs Slavia Prague/
      Diger Takimlar/
        Galatasaray/...
      index.csv / index.jsonl                          <- her basın toplantısı bir satır (model eğitimi için)

Maç ve GW, FPL'nin herkese açık fikstür verisinden bulunur (fantasy.premierleague.com/api).
"""
from __future__ import annotations

import csv
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .match import find_teams, safe_filename
from .util import atomic_write_text, log

FPL_BASE = "https://fantasy.premierleague.com/api"
PL_DIR = "Premier League"
OTHER_DIR = "Diger Takimlar"
GENERAL_DIR = "Podcast ve Film"

# FPL'ye ulaşılamazsa: 2025-26 Premier League takımları (kısa kodlarıyla)
PL_FALLBACK = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
    "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Leeds United": "LEE", "Liverpool": "LIV", "Manchester City": "MCI", "Manchester United": "MUN",
    "Newcastle United": "NEW", "Nottingham Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT",
    "West Ham": "WHU", "Wolves": "WOL",
}

COMPETITIONS = [
    (r"champions league|\bucl\b|şampiyonlar ligi|sampiyonlar ligi|çempionlar liqası", "Champions League"),
    (r"europa league|\buel\b|avrupa ligi", "Europa League"),
    (r"conference league|\buecl\b|konferans ligi", "Conference League"),
    (r"\bfa cup\b|emirates fa cup", "FA Cup"),
    (r"carabao|league cup|efl cup", "EFL Cup"),
    (r"premier league|\bepl\b", "Premier League"),
    (r"süper lig|super lig|trendyol", "Süper Lig"),
    (r"premyer liqa", "Premyer Liqa"),
]

_lock = threading.Lock()


# ------------------------------------------------------------------ FPL verisi

def _get_json(url: str) -> dict | list:
    import requests
    r = requests.get(url, timeout=20, headers={"User-Agent": "stt/1.0"})
    r.raise_for_status()
    return r.json()


def fpl_data(cache_dir: Path, max_age_h: float = 6) -> dict | None:
    """{'teams': {fpl_id: {...}}, 'fixtures': [...]} — önbellekli; erişilemezse eski önbellek ya da None."""
    cache = Path(cache_dir) / "fpl_cache.json"
    with _lock:
        if cache.exists() and time.time() - cache.stat().st_mtime < max_age_h * 3600:
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        try:
            boot = _get_json(f"{FPL_BASE}/bootstrap-static/")
            fixtures = _get_json(f"{FPL_BASE}/fixtures/")
            data = {
                "teams": {str(t["id"]): {"id": t["id"], "name": t["name"], "short_name": t["short_name"]}
                          for t in boot["teams"]},
                "fixtures": [{k: f.get(k) for k in ("id", "event", "kickoff_time", "team_h", "team_a",
                                                    "team_h_score", "team_a_score", "finished")} for f in fixtures],
                "fetched": datetime.now(timezone.utc).isoformat(),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(cache, json.dumps(data))
            return data
        except Exception as e:  # noqa: BLE001
            log.warning("FPL fikstür verisi alınamadı (%s); maç/GW eşleştirmesi tarihe göre yapılacak.", str(e)[:120])
            if cache.exists():
                try:
                    return json.loads(cache.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    return None
            return None


def canon_team(name: str) -> str:
    hits = find_teams(name)
    return hits[0] if hits else name


def pl_teams(data: dict | None) -> dict[str, dict]:
    """kanonik ad -> {'short': 'ARS', 'id': 1}"""
    if data and data.get("teams"):
        return {canon_team(t["name"]): {"short": t["short_name"], "id": t["id"]} for t in data["teams"].values()}
    return {k: {"short": v, "id": None} for k, v in PL_FALLBACK.items()}


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def presser_datetime(info: dict) -> datetime | None:
    for k in ("release_timestamp", "timestamp"):
        if info.get(k):
            try:
                return datetime.fromtimestamp(int(info[k]), tz=timezone.utc)
            except (ValueError, OSError):
                pass
    d = info.get("upload_date") or ""
    if re.fullmatch(r"\d{8}", d):
        return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), 12, tzinfo=timezone.utc)
    return None


def find_fixture(team: str, opponent: str | None, when: datetime, data: dict) -> dict | None:
    """Takımın, basın toplantısı zamanına en yakın PL maçı (2 gün önce .. 6 gün sonra)."""
    teams = pl_teams(data)
    if team not in teams or not teams[team].get("id"):
        return None
    tid = teams[team]["id"]
    opp_id = teams.get(opponent or "", {}).get("id")
    best, best_score = None, None
    for f in data.get("fixtures", []):
        if tid not in (f.get("team_h"), f.get("team_a")):
            continue
        ko = _parse_dt(f.get("kickoff_time"))
        if not ko:
            continue
        delta = (ko - when).total_seconds() / 86400
        if delta < -2 or delta > 6:
            continue
        other = f["team_a"] if f["team_h"] == tid else f["team_h"]
        if opp_id and other != opp_id:
            continue
        score = abs(delta) + (0 if delta >= -0.3 else 1.5)  # maç öncesi basın toplantısı daha olası
        if best_score is None or score < best_score:
            best, best_score = f, score
    return best


def competition_from_text(text: str) -> str | None:
    t = text.lower()
    for pat, name in COMPETITIONS:
        if re.search(pat, t):
            return name
    return None


# ------------------------------------------------------------------ sınıflandırma

def classify(match: dict, info: dict, data: dict | None) -> dict:
    """Basın toplantısını lig/takım/maç olarak yerleştir."""
    team = match.get("team") or match.get("home")
    if team:
        opponent = match.get("away") if team == match.get("home") else match.get("home")
    else:
        opponent = match.get("away") or match.get("home")
    when = presser_datetime(info)
    teams = pl_teams(data)
    text = f"{info.get('title') or ''} {(info.get('description') or '')[:300]}"
    comp = competition_from_text(text)
    meta = {"team": team, "opponent": opponent, "manager": match.get("manager"), "competition": comp,
            "is_pl": bool(team and team in teams), "fpl_short": teams.get(team or "", {}).get("short"),
            "presser_date": when.date().isoformat() if when else None, "match_date": None, "gw": None,
            "phase": None, "home": None, "away": None, "fixture": None}
    opp_is_pl = bool(opponent and opponent in teams)
    if not team and opp_is_pl and when and data:
        # takım bilinmiyor ama rakip PL takımı: o tarihteki maçından takımı çıkar ("pre-Chelsea" -> Arsenal)
        fx0 = find_fixture(opponent, None, when, data)
        if fx0:
            by_id = {v["id"]: k for k, v in teams.items()}
            oid = teams[opponent]["id"]
            team = by_id.get(fx0["team_a"] if fx0["team_h"] == oid else fx0["team_h"])
            meta.update(team=team, is_pl=bool(team in teams), fpl_short=teams.get(team or "", {}).get("short"))
    fx = None
    if meta["is_pl"] and when and data and (opponent is None or opp_is_pl) and comp in (None, "Premier League"):
        fx = find_fixture(team, opponent, when, data)
    if fx:
        by_id = {v["id"]: k for k, v in teams.items()}
        ko = _parse_dt(fx["kickoff_time"])
        meta.update(fixture=fx, gw=fx.get("event"), competition="Premier League",
                    home=by_id.get(fx["team_h"]), away=by_id.get(fx["team_a"]),
                    match_date=ko.date().isoformat() if ko else None,
                    phase="mac-oncesi" if ko and ko > when else "mac-sonrasi")
        meta["opponent"] = meta["away"] if meta["home"] == team else meta["home"]
    else:
        meta["home"], meta["away"] = match.get("home"), match.get("away")
        meta["match_date"] = meta["presser_date"]
        title = (info.get("title") or "").lower()
        if re.search(r"post[- ]?match|after|sonrası|sonrasi|sonra\b", title):
            meta["phase"] = "mac-sonrasi"
        elif re.search(r"pre[- ]?match|preview|ahead of|öncesi|oncesi|evvel", title):
            meta["phase"] = "mac-oncesi"
    return meta


def match_folder_name(meta: dict) -> str:
    parts = [meta.get("match_date") or "tarih-yok"]
    if meta.get("gw"):
        parts.append(f"GW{int(meta['gw']):02d}")
    elif meta.get("competition") and meta["competition"] != "Premier League":
        parts.append(meta["competition"])
    if meta.get("home") and meta.get("away"):
        parts.append(f"{meta['home']} vs {meta['away']}")
    elif meta.get("team") and meta.get("opponent"):
        parts.append(f"{meta['team']} vs {meta['opponent']}")
    else:
        parts.append(f"{meta.get('team') or 'Bilinmeyen'} basın toplantısı")
    return safe_filename(" ".join(parts))


def presser_base_name(meta: dict, fallback: str) -> str:
    parts = [meta.get("presser_date") or ""]
    if meta.get("phase"):
        parts.append(meta["phase"])
    name = " ".join(p for p in parts if p)
    who = meta.get("manager") or fallback
    return safe_filename(f"{name} - {who}" if name else who)


def target_dir(root: Path, meta: dict, fallback_title: str) -> Path:
    if meta.get("general"):
        return Path(root) / GENERAL_DIR / safe_filename(fallback_title, 70)
    if not meta.get("team"):
        return Path(root) / OTHER_DIR / "Bilinmeyen" / safe_filename(fallback_title, 70)
    league = PL_DIR if meta.get("is_pl") else OTHER_DIR
    return Path(root) / league / safe_filename(meta["team"]) / match_folder_name(meta)


def write_match_json(folder: Path, meta: dict) -> None:
    p = folder / "mac.json"
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    data.update({k: meta.get(k) for k in ("competition", "match_date", "gw", "home", "away", "team", "opponent")})
    if meta.get("fixture"):
        data["fpl_fixture"] = meta["fixture"]
    data.setdefault("ek_bilgiler", {})  # maça ait başka veriler (kadro, sakatlık, skor...) buraya eklenebilir
    atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=1))


INDEX_FIELDS = ["key", "presser_date", "phase", "league", "team", "opponent", "competition", "gw", "match_date",
                "home", "away", "manager", "language", "duration_s", "questions", "words", "diarized", "url",
                "txt", "json"]


def update_index(root: Path, row: dict) -> None:
    """index.jsonl (anahtar başına tek satır) ve index.csv'yi günceller."""
    root = Path(root)
    with _lock:
        jl = root / "index.jsonl"
        rows: dict[str, dict] = {}
        if jl.exists():
            for line in jl.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    rows[r["key"]] = r
                except Exception:  # noqa: BLE001
                    continue
        rows[row["key"]] = {k: row.get(k) for k in INDEX_FIELDS}
        ordered = sorted(rows.values(), key=lambda r: (r.get("league") or "", r.get("team") or "",
                                                       r.get("match_date") or "", r.get("presser_date") or ""))
        atomic_write_text(jl, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
        tmp = root / "index.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:  # Excel Türkçe karakterleri doğru açsın
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            w.writeheader()
            w.writerows(ordered)
        try:
            tmp.replace(root / "index.csv")
        except PermissionError:  # Excel'de açıksa
            log.warning("index.csv açık (Excel?); güncel tablo index.csv.tmp olarak kaldı.")
