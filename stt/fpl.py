"""FPL reposuna basın toplantısı metnini yaz ve presser.py çalıştır."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .util import atomic_write_text, log

TEAM_NAMES = {
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth", "BRE": "Brentford", "BHA": "Brighton",
    "BUR": "Burnley", "CHE": "Chelsea", "CRY": "Crystal Palace", "EVE": "Everton", "FUL": "Fulham",
    "IPS": "Ipswich", "LEE": "Leeds", "LEI": "Leicester", "LIV": "Liverpool", "MCI": "Manchester City",
    "MUN": "Manchester United", "NEW": "Newcastle", "NFO": "Nottingham Forest", "SHU": "Sheffield United",
    "SOU": "Southampton", "SUN": "Sunderland", "TOT": "Tottenham", "WHU": "West Ham", "WOL": "Wolves",
    "LUT": "Luton", "WAT": "Watford", "NOR": "Norwich", "MID": "Middlesbrough", "COV": "Coventry",
}


def fpl_prompt(team: str) -> str | None:
    name = TEAM_NAMES.get(team.upper())
    return f"{name} press conference, Premier League." if name else None


def fpl_target(repo: Path, gw: int, team: str) -> Path:
    return Path(repo) / "inputs" / "pressers" / f"GW{int(gw):02d}" / f"{team.upper()}.txt"


def python_cmd(explicit: str | None) -> list[str]:
    if explicit:
        return explicit.split()
    if shutil.which("python3") and "WindowsApps" not in (shutil.which("python3") or ""):
        return ["python3"]
    if sys.platform == "win32" and shutil.which("py"):
        return ["py", "-3"]
    return [sys.executable]


def run_fpl(repo: Path, gw: int, team: str, text: str, py: str | None) -> dict:
    repo = Path(repo)
    if not (repo / "presser.py").is_file():
        raise FileNotFoundError(f"FPL reposunda presser.py yok: {repo} (--fpl-repo ya da FPL_REPO ayarlayın)")
    target = fpl_target(repo, gw, team)
    atomic_write_text(target, text)
    rel = target.relative_to(repo).as_posix()
    cmd = python_cmd(py) + ["presser.py", rel]
    log.info("FPL: %s yazıldı -> %s", rel, " ".join(cmd))
    try:  # stdin kapalı + zaman sınırı: presser.py girdi bekler ya da takılırsa sonraki işler beklemesin
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL, timeout=600)
    except subprocess.TimeoutExpired as e:
        log.error("presser.py 10 dakikada bitmedi; durduruldu.")
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return {"file": str(target), "cmd": cmd, "returncode": -1, "stdout": out[-5000:],
                "stderr": "zaman aşımı (600 sn)"}
    except OSError as e:
        log.error("presser.py çalıştırılamadı: %s", e)
        return {"file": str(target), "cmd": cmd, "returncode": -1, "stdout": "", "stderr": str(e)}
    if r.stdout.strip():
        log.info("presser.py çıktı:\n%s", r.stdout.strip()[-3000:])
    if r.returncode != 0:
        log.error("presser.py hata kodu %d:\n%s", r.returncode, r.stderr.strip()[-3000:])
    return {"file": str(target), "cmd": cmd, "returncode": r.returncode,
            "stdout": r.stdout[-5000:], "stderr": r.stderr[-5000:]}
