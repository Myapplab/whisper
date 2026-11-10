"""Geçmiş sezonların FPL fikstürlerini uygulamaya gömülü dosyaya yazar: ``stt/assets/fpl_history.json``.

FPL API yalnız güncel sezonu verir. Eski sezonların fikstürü (GW, başlama saati, skor, FPL takım no) vaastav'ın
herkese açık arşivinden alınır: https://github.com/vaastav/Fantasy-Premier-League (data/<sezon>/fixtures.csv,
teams.csv; FPL API'nin sezon sonu kopyası). Biçim ``dataset.fpl_data`` ile aynı ({"teams", "fixtures"}).

Kullanım (geliştirici; sezon bitince yeni sezonu ekleyin):
    python dev/build_fpl_history.py 2023-24 2024-25 2025-26
    python dev/build_fpl_history.py --from-dir <indirilmiş csv klasörü> 2023-24 ...   (dosya adı <sezon>_fixtures.csv)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{name}.csv"
OUT = Path(__file__).resolve().parents[1] / "stt" / "assets" / "fpl_history.json"


def _read(season: str, name: str, src: Path | None) -> list[dict]:
    if src:
        text = (src / f"{season}_{name}.csv").read_text(encoding="utf-8")
    else:
        import requests
        r = requests.get(RAW.format(season=season, name=name), timeout=30, headers={"User-Agent": "stt-dev/1.0"})
        r.raise_for_status()
        text = r.text
    return list(csv.DictReader(io.StringIO(text)))


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def season_key(s: str) -> str:
    """"2023-24" -> "2023-2024" (veri setinin sezon adı)."""
    y = int(s[:4])
    return f"{y}-{y + 1}"


def build(seasons: list[str], src: Path | None = None) -> dict:
    out = {"_kaynak": "vaastav/Fantasy-Premier-League (FPL API sezon sonu kopyası)", "seasons": {}}
    for s in seasons:
        teams = _read(s, "teams", src)
        fixtures = _read(s, "fixtures", src)
        if len(teams) != 20 or len(fixtures) != 380:
            raise SystemExit(f"{s}: beklenmeyen veri ({len(teams)} takım, {len(fixtures)} maç)")
        fx = []
        for f in fixtures:
            fin = str(f.get("finished")).strip().lower() == "true"
            fx.append({"id": _int(f["id"]), "event": _int(f.get("event")), "kickoff_time": f.get("kickoff_time") or None,
                       "team_h": _int(f["team_h"]), "team_a": _int(f["team_a"]),
                       "team_h_score": _int(f.get("team_h_score")) if fin else None,
                       "team_a_score": _int(f.get("team_a_score")) if fin else None, "finished": fin})
        missing = [f["id"] for f in fx if not (f["event"] and f["kickoff_time"] and f["finished"])]
        if missing:
            raise SystemExit(f"{s}: GW / tarih / sonuç eksik maçlar: {missing[:10]}")
        out["seasons"][season_key(s)] = {
            "teams": {str(_int(t["id"])): {"id": _int(t["id"]), "name": t["name"], "short_name": t["short_name"]}
                      for t in teams},
            "fixtures": sorted(fx, key=lambda f: f["id"])}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="+", help="ör. 2023-24 2024-25")
    ap.add_argument("--from-dir", type=Path)
    a = ap.parse_args(argv)
    data = build(a.seasons, a.from_dir)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT} yazıldı: {', '.join(data['seasons'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
