"""Video başlığından maç adı ve teknik direktör adını çıkarma.

Örnek: "PRESS CONFERENCE 🎙️ | Mikel Arteta previews Slavia Prague | Champions League"
  -> {"home": "Arsenal", "away": "Slavia Prague", "manager": "Mikel Arteta"}
(kanal adı "Arsenal" ise ev sahibi oradan gelir.)
"""
from __future__ import annotations

import re
import unicodedata

# kanonik ad: diğer yazılışlar
TEAMS: dict[str, list[str]] = {
    # Premier League / Championship
    "Arsenal": ["Arsenal FC", "Gunners"],
    "Aston Villa": ["Villa"],
    "Bournemouth": ["AFC Bournemouth"],
    "Brentford": [],
    "Brighton": ["Brighton & Hove Albion", "Brighton and Hove Albion", "Brighton Hove Albion"],
    "Burnley": [],
    "Chelsea": ["Chelsea FC"],
    "Crystal Palace": ["Palace"],
    "Everton": [],
    "Fulham": [],
    "Leeds United": ["Leeds"],
    "Liverpool": ["Liverpool FC", "LFC"],
    "Manchester City": ["Man City", "Man. City", "MCFC"],
    "Manchester United": ["Man United", "Man Utd", "Man. United", "MUFC"],
    "Newcastle United": ["Newcastle", "NUFC"],
    "Nottingham Forest": ["Nott'm Forest", "Forest"],
    "Sunderland": [],
    "Tottenham": ["Tottenham Hotspur", "Spurs", "THFC"],
    "West Ham": ["West Ham United"],
    "Wolves": ["Wolverhampton", "Wolverhampton Wanderers"],
    "Ipswich Town": ["Ipswich"],
    "Leicester City": ["Leicester"],
    "Southampton": [],
    "Sheffield United": ["Sheff Utd"],
    "Sheffield Wednesday": [],
    "Middlesbrough": ["Boro"],
    "Coventry City": ["Coventry"],
    "Norwich City": ["Norwich"],
    "Watford": [],
    "West Brom": ["West Bromwich Albion", "WBA"],
    "Stoke City": ["Stoke"],
    "Hull City": ["Hull"],
    "Birmingham City": ["Birmingham"],
    "Wrexham": [],
    "Millwall": [],
    "Swansea City": ["Swansea"],
    "Cardiff City": ["Cardiff"],
    "Derby County": ["Derby"],
    "Portsmouth": [],
    "Preston": ["Preston North End"],
    "Bristol City": [],
    "Blackburn Rovers": ["Blackburn"],
    "QPR": ["Queens Park Rangers"],
    "Luton Town": ["Luton"],
    "Charlton": ["Charlton Athletic"],
    "Oxford United": [],
    # Avrupa
    "Real Madrid": [], "Barcelona": ["Barca", "Barça", "FC Barcelona"],
    "Atletico Madrid": ["Atlético Madrid", "Atletico de Madrid", "Atlético de Madrid", "Atleti"],
    "Athletic Club": ["Athletic Bilbao"], "Villarreal": [], "Real Sociedad": [], "Real Betis": ["Betis"],
    "Sevilla": [], "Valencia": [], "Girona": [],
    "Bayern Munich": ["Bayern München", "Bayern", "FC Bayern"],
    "Borussia Dortmund": ["Dortmund", "BVB"], "Bayer Leverkusen": ["Leverkusen"], "RB Leipzig": ["Leipzig"],
    "Eintracht Frankfurt": ["Frankfurt"], "Stuttgart": ["VfB Stuttgart", "VfB"], "Freiburg": ["SC Freiburg"],
    "Inter": ["Inter Milan", "Internazionale"], "AC Milan": ["Milan"], "Juventus": ["Juve"], "Napoli": [],
    "Atalanta": [], "Roma": ["AS Roma"], "Lazio": [], "Bologna": [], "Fiorentina": [],
    "Paris Saint-Germain": ["PSG", "Paris SG", "Paris Saint Germain"], "Monaco": ["AS Monaco"],
    "Marseille": ["Olympique de Marseille", "OM"], "Lille": ["LOSC"], "Lyon": ["Olympique Lyonnais"], "Nice": ["OGC Nice"],
    "Benfica": [], "Porto": ["FC Porto"], "Sporting CP": ["Sporting Lisbon", "Sporting"], "Braga": ["SC Braga"],
    "Ajax": [], "PSV": ["PSV Eindhoven"], "Feyenoord": [],
    "Club Brugge": ["Brugge"], "Union Saint-Gilloise": ["Union SG", "Union St-Gilloise"],
    "Celtic": [], "Rangers": [],
    "Galatasaray": [], "Fenerbahçe": ["Fenerbahce"], "Beşiktaş": ["Besiktas"], "Trabzonspor": [],
    "Başakşehir": ["Basaksehir", "İstanbul Başakşehir"], "Göztepe": ["Goztepe"], "Samsunspor": [],
    "Qarabağ": ["Qarabag", "Karabağ", "Karabag"], "Neftçi": ["Neftchi"], "Sabah": ["Sabah FK"], "Zira": [],
    "Slavia Prague": ["Slavia Praha", "Slavia"], "Sparta Prague": ["Sparta Praha"], "Viktoria Plzen": ["Plzen", "Plzeň"],
    "Olympiacos": ["Olympiakos"], "Panathinaikos": [], "PAOK": [],
    "Copenhagen": ["FC Copenhagen", "FC København"], "Midtjylland": ["FC Midtjylland"],
    "Bodø/Glimt": ["Bodo/Glimt", "Bodo Glimt", "Bodø Glimt"], "Kairat": ["Kairat Almaty"], "Pafos": ["Pafos FC"],
    "Shakhtar Donetsk": ["Shakhtar"], "Dynamo Kyiv": ["Dynamo Kiev"], "Dinamo Zagreb": [],
    "Red Star Belgrade": ["Crvena Zvezda", "Red Star"], "Salzburg": ["RB Salzburg", "Red Bull Salzburg"],
    "Sturm Graz": [], "Young Boys": ["BSC Young Boys"], "Basel": ["FC Basel"], "Ferencváros": ["Ferencvaros"],
    "Ludogorets": [], "Maccabi Tel Aviv": [],
}

# Yalnızca soyadı yazılmışsa tam adı tamamlamak için (ipucu; yanlışsa dosya adında soyadı kalır)
MANAGER_HINTS = {
    "arteta": "Mikel Arteta", "slot": "Arne Slot", "guardiola": "Pep Guardiola", "amorim": "Ruben Amorim",
    "maresca": "Enzo Maresca", "emery": "Unai Emery", "howe": "Eddie Howe", "iraola": "Andoni Iraola",
    "glasner": "Oliver Glasner", "moyes": "David Moyes", "farke": "Daniel Farke", "hürzeler": "Fabian Hürzeler",
    "hurzeler": "Fabian Hürzeler", "dyche": "Sean Dyche", "postecoglou": "Ange Postecoglou",
    "rosenior": "Liam Rosenior", "carrick": "Michael Carrick", "ancelotti": "Carlo Ancelotti",
    "flick": "Hansi Flick", "simeone": "Diego Simeone", "kompany": "Vincent Kompany", "mourinho": "José Mourinho",
    "conte": "Antonio Conte", "allegri": "Massimiliano Allegri", "chivu": "Cristian Chivu",
    "spalletti": "Luciano Spalletti", "gasperini": "Gian Piero Gasperini", "tedesco": "Domenico Tedesco",
    "trpisovsky": "Jindřich Trpišovský", "trpišovský": "Jindřich Trpišovský", "gurbanov": "Gurban Gurbanov",
    "nuno": "Nuno Espírito Santo", "pochettino": "Mauricio Pochettino", "tuchel": "Thomas Tuchel",
    "southgate": "Gareth Southgate", "montella": "Vincenzo Montella", "buruk": "Okan Buruk",
}

_NOT_NAME = set("""
press conference conferences presser live full watch preview previews pre post match matchday reaction reacts
champions league premier europa conference cup carabao fa efl uefa super lig süper the a an and of on in at to for
vs v versus head coach boss manager interview highlights sky sports bbc tnt official exclusive news update
monday tuesday wednesday thursday friday saturday sunday today tonight new latest team squad injury injuries
teknik direktor direktorumuz direktorunun hoca hocamiz hocasi baskan baskanimiz kaptan kaptanimiz basin toplantisi
toplanti aciklama aciklamalar aciklamalari aciklamasi mac maci macin oncesi sonrasi sonra once karsilasma
karsilasmasi lig ligi kupa kupasi sampiyonlar avrupa trendyol futbolcumuz oyuncumuz roportaj ozel
bas mesqci mesqcisi metbuat konfrans konfransi oyun oyundan oyunundan evvel sonra qarsilasma
""".split())

_NAME_WORD = r"[A-ZÀ-ÖØ-ÞĞİŞÇÜÖƏ][a-zà-öø-ÿğışçüöə'’\-]+"


_TR = str.maketrans("ıİşŞğĞçÇöÖüÜəƏ", "iIsSgGcCoOuUeE")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.translate(_TR)).encode("ascii", "ignore").decode().lower()


def _alias_index() -> list[tuple[str, str]]:
    pairs = []
    for canon, aliases in TEAMS.items():
        canon = canon.strip()
        for a in [canon] + aliases:
            pairs.append((_norm(a), canon))
    return sorted(set(pairs), key=lambda p: -len(p[0]))  # uzun ad önce ("Inter Milan" > "Milan")


_ALIASES = _alias_index()


def find_teams(text: str) -> list[str]:
    """Metinde geçen takımları, geçiş sırasına göre döner."""
    t = " " + re.sub(r"[^a-z0-9/&' ]+", " ", _norm(text)) + " "
    hits: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for alias, canon in _ALIASES:
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", t):
            a, b = m.span()
            if any(a < y and b > x for x, y in taken):
                continue
            taken.append((a, b))
            hits.append((a, canon))
    out: list[str] = []
    for _, c in sorted(hits):
        if c not in out:
            out.append(c)
    return out


def _titlecase_caps(s: str) -> str:
    # "MIKEL ARTETA PRESS CONFERENCE" -> "Mikel Arteta Press Conference"
    return re.sub(r"\b[A-ZÀ-ÞĞİŞÇÜÖ]{2,}\b", lambda m: m.group(0).capitalize() if len(m.group(0)) > 3 else m.group(0), s)


def _name_runs(title: str, teams: list[str]) -> list[list[str]]:
    """Başlıktaki ardışık 'isim olabilecek' kelime dizileri (2-3 kelime)."""
    t = _titlecase_caps(title)
    team_words = {w for team in teams for w in _norm(team).split()}
    all_team_words = {w for a, _ in _ALIASES for w in a.split()}
    runs, cur = [], []
    for tok in re.findall(r"[\w'’\-]+|[^\w\s]", t):
        base = re.sub(r"['’].*$", "", tok)            # Buruk'un -> Buruk, Slot's -> Slot
        ends = base != tok                              # iyelik eki ismi bitirir
        parts = [_norm(x) for x in base.split("-") if x]
        ok = (re.fullmatch(_NAME_WORD, base) is not None and parts
              and not any(x in _NOT_NAME or x in team_words for x in parts))
        if ok:
            cur.append(base)
        if not ok or ends:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) >= 2 and not all(_norm(w) in all_team_words for w in r):
            out.append(r[:3])
    return out


def find_manager(title: str, teams: list[str]) -> str | None:
    runs = _name_runs(title, teams)
    for r in runs:  # bilinen soyad ile biten aday önce
        if _norm(r[-1]) in MANAGER_HINTS:
            return " ".join(r)
    if runs:
        return " ".join(runs[0])
    for w in re.findall(_NAME_WORD, _titlecase_caps(title)):
        hint = MANAGER_HINTS.get(_norm(re.sub(r"['’].*$", "", w)))
        if hint:
            return hint
    return None


def _opponent_by_pattern(title: str, home: str | None) -> str | None:
    t = _titlecase_caps(title)
    m = re.search(rf"\b(?:previews?|ahead of|before|after|against|vs\.?|v\.?|reacts to|on)\s+((?:{_NAME_WORD}\s?){{1,3}})", t)
    if m:
        cand = m.group(1).strip()
        words = [w for w in cand.split() if _norm(w) not in _NOT_NAME]
        if words and (not home or _norm(" ".join(words)) != _norm(home)):
            return " ".join(words)
    return None


def _is_opponent_mention(title: str, team: str) -> bool:
    t = _norm(title)
    aliases = [a for a, c in _ALIASES if c == team]
    for a in aliases:
        if re.search(r"(?:\bpre[- ]?|\bpost[- ]?|previews?\s+|ahead of\s+|against\s+|before\s+|after\s+|"
                     r"\bvs?\.?\s+|\bface\s+|\bplay\s+)" + re.escape(a) + r"(?![a-z])", t):
            return True
        # Türkçe / Azerice: "Stuttgart maçı öncesi", "Chelsea ile", "Qarabağ oyunundan sonra"
        if re.search(r"(?<![a-z])" + re.escape(a) + r"(?:'\w+)?\s+(?:maci|macinin|macindan|karsilasmasi\w*|ile|"
                     r"oyunu\w*|deplasmani\w*|engeli)\b", t):
            return True
    return False


def parse_match(info: dict) -> dict:
    title = info.get("title") or ""
    channel = info.get("channel") or info.get("uploader") or ""
    desc = (info.get("description") or "")[:400]
    title_teams = find_teams(title)
    chan_teams = find_teams(channel)
    home = chan_teams[0] if chan_teams else (title_teams[0] if title_teams else None)
    away = next((t for t in title_teams if t != home), None)
    if len(title_teams) >= 2 and home in title_teams[:2] and away in title_teams[:2]:
        home, away = title_teams[0], title_teams[1]  # "Everton vs Coventry City" sırası korunur
    if away is None:
        away = next((t for t in find_teams(desc) if t != home), None)
    if away is None:
        away = _opponent_by_pattern(title, home)
    team = chan_teams[0] if chan_teams else home  # basın toplantısının sahibi takım (kanal)
    if not chan_teams and len(title_teams) == 1 and _is_opponent_mention(title, title_teams[0]):
        team, home, away = None, None, title_teams[0]  # "pre-Chelsea", "previews Chelsea": Chelsea rakip
    manager = find_manager(title, title_teams + chan_teams)
    d = info.get("upload_date") or ""
    date = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if re.fullmatch(r"\d{8}", d) else None
    return {"home": home, "away": away, "team": team, "manager": manager, "date": date}


def safe_filename(s: str, maxlen: int = 110) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", s)
    s = re.sub(r"[^\w\s.,'&()\-–]+", " ", s)  # emoji vb.
    s = re.sub(r"\s+", " ", s).strip(" .-")
    return s[:maxlen].rstrip(" .-") or "kayit"


def display_name(match: dict, fallback: str) -> str:
    parts = []
    if match.get("home") and match.get("away"):
        parts.append(f"{match['home']} vs {match['away']}")
    elif match.get("home") or match.get("away"):
        parts.append(match.get("home") or match.get("away"))
    if match.get("manager"):
        parts.append(match["manager"])
    if not parts:
        parts.append(fallback)
    name = " - ".join(parts)
    if match.get("date"):
        name += f" ({match['date']})"
    return safe_filename(name)
