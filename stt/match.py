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
    "Manchester United": ["Man United", "Man Utd", "Man. United", "Manchester Utd", "MUFC"],
    "Newcastle United": ["Newcastle", "NUFC"],
    "Nottingham Forest": ["Nott'm Forest", "Forest"],
    "Sunderland": [],
    "Tottenham": ["Tottenham Hotspur", "Spurs", "THFC"],
    "West Ham": ["West Ham United"],
    "Wolves": ["Wolverhampton", "Wolverhampton Wanderers"],
    "Ipswich Town": ["Ipswich"],
    "Leicester City": ["Leicester"],
    "Southampton": [],
    "Sheffield United": ["Sheff Utd", "Sheffield Utd"],
    "Sheffield Wednesday": [],
    "Middlesbrough": ["Boro"],
    "Coventry City": ["Coventry"],
    "Norwich City": ["Norwich"],
    "Watford": [],
    "West Brom": ["West Bromwich Albion", "WBA"],
    "Stoke City": ["Stoke"],
    "Hull City": ["Hull", "Hull City AFC"],
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
    "Benfica": [], "Porto": ["FC Porto"], "Sporting CP": ["Sporting Lisbon", "Sporting", "Sporting Clube de Portugal"], "Braga": ["SC Braga"],
    "Ajax": [], "PSV": ["PSV Eindhoven"], "Feyenoord": [],
    "Club Brugge": ["Brugge"], "Union Saint-Gilloise": ["Union SG", "Union St-Gilloise"],
    "Celtic": [], "Rangers": [],
    "Galatasaray": [], "Fenerbahçe": ["Fenerbahce"], "Beşiktaş": ["Besiktas"], "Trabzonspor": [],
    "Başakşehir": ["Basaksehir", "İstanbul Başakşehir", "Rams Başakşehir"], "Göztepe": ["Goztepe"], "Samsunspor": [],
    "Qarabağ": ["Qarabag", "Karabağ", "Karabag"], "Neftçi": ["Neftchi"], "Sabah": ["Sabah FK", "Sabah FC"], "Zira": ["Zira FK", "Zirə"],
    "Slavia Prague": ["Slavia Praha", "Slavia"], "Sparta Prague": ["Sparta Praha"], "Viktoria Plzen": ["Plzen", "Plzeň"],
    "Olympiacos": ["Olympiakos"], "Panathinaikos": [], "PAOK": [],
    "Copenhagen": ["FC Copenhagen", "FC København"], "Midtjylland": ["FC Midtjylland"],
    "Bodø/Glimt": ["Bodo/Glimt", "Bodo Glimt", "Bodø Glimt"], "Kairat": ["Kairat Almaty"], "Pafos": ["Pafos FC"],
    "Shakhtar Donetsk": ["Shakhtar"], "Dynamo Kyiv": ["Dynamo Kiev"], "Dinamo Zagreb": [],
    "Red Star Belgrade": ["Crvena Zvezda", "Red Star"], "Salzburg": ["RB Salzburg", "Red Bull Salzburg"],
    "Sturm Graz": [], "Young Boys": ["BSC Young Boys"], "Basel": ["FC Basel"], "Ferencváros": ["Ferencvaros"],
    "Ludogorets": [], "Maccabi Tel Aviv": [],
    # "Inter Miami" -> Inter (Internazionale) sanılmasın
    "Inter Miami": [],
}

# ------------------------------------------------------------------ teknik direktörler
# Premier League 2026-27 hocaları (premierleague.com listesi; 2026-11-06'da doğrulandı, o tarihte sezonda görevden
# alınan yoktu). Hoca değişince / sezon başında güncellenmeli (FPL API hoca vermiyor). Takım eşlemesi yalnız
# MANAGER_SEASON tarihleri içindeki videolarda kullanılır: eski sezonun videosunda hoca başka takımdaydı.
MANAGER_SEASON = ("2026-07-01", "2027-06-30")
PL_MANAGERS = {
    "Mikel Arteta": "Arsenal", "Unai Emery": "Aston Villa", "Marco Rose": "Bournemouth",
    "Keith Andrews": "Brentford", "Fabian Hürzeler": "Brighton", "Xabi Alonso": "Chelsea",
    "Frank Lampard": "Coventry City", "Pierre Sage": "Crystal Palace", "David Moyes": "Everton",
    "Álvaro Arbeloa": "Fulham", "Sergej Jakirović": "Hull City", "Gary O'Neil": "Ipswich Town",
    "Daniel Farke": "Leeds United", "Andoni Iraola": "Liverpool", "Enzo Maresca": "Manchester City",
    "Michael Carrick": "Manchester United", "Matthias Jaissle": "Newcastle United",
    "Oliver Glasner": "Nottingham Forest", "Régis Le Bris": "Sunderland", "Roberto De Zerbi": "Tottenham",
}
# Takımı bu listede tutulmayan bilinen hocalar: yalnız soyadı yazılınca tam ad + "basın toplantısı" tanıma
OTHER_MANAGERS = (
    "Arne Slot", "Pep Guardiola", "Ruben Amorim", "Eddie Howe", "Sean Dyche", "Ange Postecoglou", "Liam Rosenior",
    "Thomas Frank", "Marco Silva", "Kieran McKenna", "Scott Parker", "Rob Edwards", "Igor Tudor", "Graham Potter",
    "Vítor Pereira", "Nuno Espírito Santo", "Erik ten Hag", "Carlo Ancelotti", "Hansi Flick", "Diego Simeone",
    "Vincent Kompany", "José Mourinho", "Antonio Conte", "Massimiliano Allegri", "Cristian Chivu",
    "Luciano Spalletti", "Gian Piero Gasperini", "Domenico Tedesco", "Jindřich Trpišovský", "Gurban Gurbanov",
    "Qurban Qurbanov", "Mauricio Pochettino", "Thomas Tuchel", "Gareth Southgate", "Vincenzo Montella", "Okan Buruk",
)
# Ortak sözcük / çok yaygın soyadı: tek başına yazılınca hoca sayılmaz ("time slot", "Bernardo Silva", "Derrick Rose")
_UNSAFE_SURNAMES = {"slot", "silva", "rose", "sage", "frank", "parker", "edwards", "andrews", "alonso", "tudor",
                    "potter", "flick", "santo"}


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    return unicodedata.normalize("NFKD", s.translate(_TR)).encode("ascii", "ignore").decode().lower()


_TR = str.maketrans("ıİşŞğĞçÇöÖüÜəƏ", "iIsSgGcCoOuUeE")


def _hint_index() -> dict[str, str]:
    """normalize soyadı -> tam ad ("hag" -> "Erik ten Hag"); Premier League listesi öncelikli."""
    out = {_norm(n.split()[-1]): n for n in OTHER_MANAGERS}
    out.update({_norm(n.split()[-1]): n for n in PL_MANAGERS})
    out["nuno"] = "Nuno Espírito Santo"  # adıyla anılır
    return out


MANAGER_HINTS = _hint_index()

_NOT_NAME = set("""
press conference conferences presser live full watch preview previews pre post match matchday reaction reacts react
champions league premier europa conference cup carabao fa efl uefa super lig süper the a an and of on in at to for
vs v versus head coach boss manager interview highlights sky sports bbc tnt official exclusive news update updates
monday tuesday wednesday thursday friday saturday sunday today tonight new latest team squad injury injuries
teknik direktor direktorumuz direktorunun hoca hocamiz hocasi baskan baskanimiz kaptan kaptanimiz basin toplantisi
toplanti aciklama aciklamalar aciklamalari aciklamasi mac maci macin oncesi sonrasi sonra once karsilasma
karsilasmasi lig ligi kupa kupasi sampiyonlar avrupa trendyol futbolcumuz oyuncumuz roportaj ozel
bas mesqci mesqcisi metbuat konfrans konfransi oyun oyundan oyunundan evvel sonra qarsilasma
win wins won defeat loss lose draw victory clash game games fixture tie trip visit test showdown battle derby
final semi quarter round leg week weekend season fans fan says speaks talks reveals explains admits insists
praises confirms discusses fitness transfer transfers window deadline january summer liga laliga serie
bundesliga ligue is it his he we our my their they has have was were be from with into about after before
ahead against why how what who
coletiva imprensa rueda prensa conferencia conferenza stampa pressekonferenz persconferentie presse entrevista
intervista out back set big top all one two three four five six seven eight nine ten more now next last first
ready return returns fit doubt ruled miss misses available injured lineup xi starting bench goal goals assist
assists start starts debut contract deal signing signs sign loan special key title race relegation chances
weeks days months video clip clips every best
""".split())

_UP = "A-ZÀ-ÖØ-ÞĞİŞÇÜÖƏ"
_LO = "a-zà-öø-ÿğışçüöə"
# Ad: büyük harfle başlar; McKenna, MacAllister, O'Neil, D'Alessandro, Jean-Philippe da ad
_NAME_WORD = rf"(?:Mc|Mac|O['’]|D['’])?[{_UP}][{_LO}'’]+(?:-(?:[{_UP}][{_LO}'’]+|[{_LO}'’]+))*"
_PARTICLES = {"van", "von", "der", "den", "de", "di", "da", "du", "del", "della", "le", "la", "ten", "ter",
              "dos", "das"}


def _tnorm(text: str) -> str:
    """Takım araması için metin: küçük harf ASCII, tek boşluk; ayraçlar ('|', ':', boşluklu tire) korunur.
    Kelime içi tire boşluk olur (Saint-Gilloise, pre-Chelsea); takma adlar da aynı işlemden geçer."""
    s = re.sub(r"[–—]", "-", text or "")
    s = re.sub(r"\s+-+\s+", " - ", s)
    s = _norm(s)
    s = re.sub(r"(?<=[a-z0-9])-+(?=[a-z0-9])", " ", s)
    s = re.sub(r"[^a-z0-9/&'|:\- ]+", " ", s)
    return " " + re.sub(r" +", " ", s).strip() + " "


def _alias_norm(a: str) -> str:
    return _tnorm(a).strip()


def _alias_index() -> list[tuple[str, str]]:
    pairs = []
    for canon, aliases in TEAMS.items():
        canon = canon.strip()
        for a in [canon] + aliases:
            pairs.append((_alias_norm(a), canon))
    return sorted(set(pairs), key=lambda p: (-len(p[0]), p[0]))  # uzun ad önce ("Inter Milan" > "Milan")


_ALIASES = _alias_index()
_ALIAS_MAP = {a: c for a, c in _ALIASES}

# Ortak sözcük de olan adlar: tek başına geçince takım sayılmaz ("the Manchester derby", "a nice win",
# "Sporting director", "sabah" = sabah / yarın, "om" = İsveççe/Danca "hakkında"). Niteleyiciyle (Derby County,
# OGC Nice, Sporting CP, Sabah FK) ya da başka bir takıma bağlıyken ("Leeds v Derby", "Qarabağ - Sabah",
# "Derby vs Leeds") takımdır. "OM" yalnız büyük harfle.
_AMBIGUOUS = {"derby", "nice", "sporting", "sabah", "zira", "om"}
_EXACT_CASE = {"om": "OM"}
# Basketbol (NBA) bağlamında "Spurs" = San Antonio Spurs, Tottenham değil
_NBA = re.compile(r"\b(?:nba|wnba|san antonio|wembanyama|grizzlies|lakers|warriors|celtics|knicks|mavericks|nuggets|"
                  r"clippers|timberwolves|pelicans|basketball|basketbol)\b")
_CONNECT_BEFORE = re.compile(r"(?:^|[\s|:])(?:vs|v|versus|against)$")
_SCORE = r"(?:\d+\s*[-:]\s*\d+|\d+\s+\d+)"
_LINK = rf"(?:vs|v|versus|-|{_SCORE})"
# Belirsiz addan sonra gelince "takım - takım" kalıbını doğrulayan sözcükler ("Qarabağ - Sabah oyunundan sonra")
_AMB_AFTER = re.compile(r"(?:$|[|:\-]|\d|(?:mac|matc|oyun|karsilasma|match|game|clash|fixture|tie|press|presser|"
                        r"metbuat|konfrans|basin|toplanti|aciklama|conference|preview|highlight|derbi|derby)\w*)")
# "Sporting will be tough", "Nice are dangerous": belirsiz ad büyük harfle yazılmış özne (önünde tanımlık yoksa).
# Asıl metinde bakılır: "nice play", "Nice, has..." ya da Türkçe "Sabah iş" ("iş" -> "is") özne değil.
_AMB_VERBS = r"(?:are|have|has|will|were|is|was|can|could|would|face|faces|host|hosts|visit|visits)"
_DETERMINERS = {"the", "a", "an", "this", "that", "our", "their", "his", "its", "next", "big", "local", "every", "any",
                "no", "what", "very", "so"}


def _subject_in(text: str, alias: str) -> bool:
    pat = rf"(?<![\w'’])({re.escape(alias[0].upper())}(?i:{re.escape(alias[1:])}))\s+{_AMB_VERBS}\b"
    for m in re.finditer(pat, text or ""):
        prev = re.findall(r"[\w']+", text[:m.start()])
        if not prev or prev[-1].lower() not in _DETERMINERS:
            return True
    return False


def _ambiguous_ok(t: str, a: int, b: int, alias: str, hits: list, cands: list, text: str = "") -> bool:
    """Belirsiz ad (t[a:b]) burada takım mı? ``hits``: kesin takımlar, ``cands``: öbür belirsiz adaylar
    ("Derby - Nice": ikisi de belirsiz, birbirine bağlı)."""
    before, after = t[:a].rstrip(), t[b:].lstrip()
    if _CONNECT_BEFORE.search(before):                       # "Leeds v Derby", "against Sporting"
        return True
    ends = {y for _, y, *_ in hits + cands}
    starts = {x for x, *_ in hits + cands}
    # "<takım> - <ad>" / "<takım> 2-1 <ad>": adın ardında ayraç, maç / basın sözcüğü, skor ya da başlık sonu
    m = re.search(rf"(?:-|{_SCORE})$", before)
    if m and len(t[:m.start()].rstrip()) in ends and _AMB_AFTER.match(after):
        return True
    seg_start = before.strip() in ("", "|", ":") or before.endswith(("|", ":"))
    # "<ad> vs <takım>" başlığın / bölümün başında
    if seg_start:
        m = re.match(_LINK + r"\s+", after)
        if m and b + (len(t[b:]) - len(after)) + m.end() in starts:
            return True
    # "Sporting will be a tough test" (derby hariç: "derby will be..." çoğunlukla yerel maç)
    if alias != "derby" and _subject_in(text, alias):
        return True
    return False


def _team_hits(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """(normalize metin, [(baş, son, kanonik ad)]) geçiş sırasıyla; örtüşen kısa ad atlanır."""
    t = _tnorm(text)
    nba = bool(_NBA.search(t))
    taken: list[tuple[int, int]] = []
    hits: list[tuple[int, int, str]] = []
    amb = []
    for alias, canon in _ALIASES:
        if alias == "spurs" and nba:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", t):
            a, b = m.span()
            if any(a < y and b > x for x, y in taken):
                continue
            if alias in _AMBIGUOUS:
                amb.append((a, b, alias, canon))
                continue
            taken.append((a, b))
            hits.append((a, b, canon))
    amb = [x for x in amb if not any(x[0] < y and x[1] > z for z, y in taken)]
    amb = [x for x in amb if x[2] not in _EXACT_CASE
           or re.search(rf"(?<![A-Za-z0-9]){_EXACT_CASE[x[2]]}(?![A-Za-z0-9])", text)]
    fixed = list(hits)
    for a, b, alias, canon in amb:  # ortak sözcükler: öbür takımlar bulunduktan sonra, bağlama göre
        others = [c for c in amb if c[0] != a]
        if _ambiguous_ok(t, a, b, alias, fixed, others, text):
            hits.append((a, b, canon))
    return t, sorted(hits)


def _uniq(names) -> list[str]:
    out: list[str] = []
    for c in names:
        if c not in out:
            out.append(c)
    return out


def find_teams(text: str) -> list[str]:
    """Metinde geçen takımları, geçiş sırasına göre döner."""
    return _uniq(c for _, _, c in _team_hits(text)[1])


def is_team(name: str | None) -> bool:
    """Tanınan (sözlükteki) takımın kanonik adı mı? Kalıpla yakalanan "Bukayo Saka's" gibi metinler değil."""
    return bool(name) and name in TEAMS


# 2-3 harfli büyük harfli sözcükler arasında kısaltma olarak kalacaklar (gerisi ad: "ERIK TEN HAG" -> "Erik Ten Hag")
_ACRONYMS = {a.upper() for _, al in TEAMS.items() for a in al if a.isupper() and len(a) <= 3} | {
    "FC", "AFC", "CF", "SC", "FK", "SK", "AC", "AS", "RB", "UCL", "UEL", "EPL", "EFL", "FA", "VAR", "BBC", "TNT",
    "GW", "TV", "PL", "UK", "USA", "US", "VS", "NBA", "NFL", "CEO", "MOTD", "ESPN"}


def _titlecase_caps(s: str) -> str:
    # "MIKEL ARTETA PRESS CONFERENCE" -> "Mikel Arteta Press Conference"; "PSG", "FC" gibi kısaltmalar kalır;
    # ad parçacıkları küçük harf ("ERIK TEN HAG" -> "Erik ten Hag", "RUEDA DE PRENSA DE ..." -> "Rueda de Prensa de")
    def fix(m):
        w = m.group(0)
        if len(w) <= 3 and w in _ACRONYMS:
            return w
        prev = s[:m.start()].rstrip()
        if w.lower() in _PARTICLES and prev and prev[-1].isalpha():  # ad içinde; bölüm başında değil ("| DE BRUYNE")
            return w.lower()
        return w.capitalize()
    return re.sub(rf"\b[{_UP}]{{2,}}\b", fix, s)


def _strip_poss(tok: str) -> str:
    """Slot's -> Slot, Buruk'un -> Buruk, SLOT'S -> Slot (büyük harfli başlık), Spurs' -> Spurs;
    O'Neil, D'Alessandro, N'Golo olduğu gibi (kesme işareti ilk harften hemen sonra)."""
    i = max(tok.rfind("'"), tok.rfind("’"))
    if i >= 2 and re.fullmatch(r"\w*", tok[i + 1:]):
        return tok[:i]
    return tok


def _name_runs(title: str, teams: list[str]) -> list[list[str]]:
    """Başlıktaki ardışık 'isim olabilecek' kelime dizileri (tek kelimelikler dahil)."""
    t = _titlecase_caps(title)
    team_words = {w for team in teams for w in _alias_norm(team).split()}
    all_team_words = {w for a, _ in _ALIASES for w in a.split()}
    toks = re.findall(r"[\w'’\-]+|[^\w\s]", t)
    runs, cur = [], []
    for i, tok in enumerate(toks):
        base = _strip_poss(tok)
        ends = base != tok                              # iyelik eki ismi bitirir
        parts = [_norm(x) for x in base.split("-") if x]
        ok = (re.fullmatch(_NAME_WORD, base) is not None and parts
              and not any(x in _NOT_NAME or x in team_words for x in parts))
        if (not ok and cur and tok.lower() in _PARTICLES and i + 1 < len(toks)
                and re.fullmatch(_NAME_WORD, _strip_poss(toks[i + 1]))):
            cur.append(tok)                             # "Erik ten Hag", "Robin van Persie"
            continue
        if ok:
            cur.append(base)
        if not ok or ends:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if not all(_norm(w) in all_team_words for w in r)]


def _known_manager(words: list[str], teams: list[str] | None = None) -> str | None:
    """Kelimeler bilinen bir hoca mı? Tam ad ("Erik ten Hag"), soyadı ("De Zerbi") ya da aynı adla başlayan yazım.
    Aynı soyadlı başka kişi ("Derrick Rose", "Bernardo Silva") değil. Ortak sözcük / çok yaygın soyadı tek başına
    ("Slot", "Alonso") yalnız hocanın takımı ``teams`` içindeyse (kanal / başlık) sayılır."""
    ws = [_norm(w) for w in words if w]
    if not ws:
        return None
    full = MANAGER_HINTS.get(ws[-1])
    if not full:
        return None
    if len(ws) == 1:
        if ws[0] in _UNSAFE_SURNAMES and PL_MANAGERS.get(full) not in (teams or []):
            return None
        return full
    fw = _norm(full).split()
    return full if ws == fw[-len(ws):] or ws[0] == fw[0] else None


def _known_in_run(run: list[str], teams: list[str] | None = None) -> str | None:
    """Çok kelimeli dizide bilinen hoca: iki-üç kelimelik pencere ("Mikel Arteta", "De Zerbi") ya da dizinin başındaki
    soyadı ("Maresca Unhappy With Referee"). Soyadı başka bir adın ardındaysa ("Andreas Pereira") ya da adla anılan
    hocanın adı başka bir soyadıyla geçiyorsa ("Nuno Mendes") başka kişidir."""
    for p, w in enumerate(run):
        key = _norm(w)
        if key in MANAGER_HINTS:
            for s in range(max(0, p - 2), p):
                k = _known_manager(run[s:p + 1])
                if k:
                    return k
            if p == 0 and _norm(MANAGER_HINTS[key].split()[-1]) == key:
                k = _known_manager([w], teams)
                if k:
                    return k
    return None


def find_manager(title: str, teams: list[str]) -> str | None:
    runs = _name_runs(title, teams)
    multi = [r for r in runs if len(r) >= 2]
    for r in multi:  # bilinen hoca önce (yazılışı düzeltilmiş tam adıyla: "FABIAN HURZELER" -> "Fabian Hürzeler")
        k = _known_in_run(r, teams)
        if k:
            return k
    for r in runs:  # tek başına soyadı: "Arteta on Saka", "Carrick on Mainoo's fitness"
        if len(r) == 1:
            k = _known_manager(r, teams)
            if k:
                return k
    if multi:
        r = multi[0][:3]
        while r and r[-1].lower() in _PARTICLES:
            r = r[:-1]
        return " ".join(r)
    return None


def manager_team(name: str | None, date: str | None = None) -> str | None:
    """Bu sezon hocanın takımı (PL listesi). Sezon dışındaki tarihli videoda None."""
    full = _known_manager(name.split()) if name else None
    team = PL_MANAGERS.get(full or "")
    if team and date and not (MANAGER_SEASON[0] <= date <= MANAGER_SEASON[1]):
        return None
    return team


# Rakip adı yerine geçen genel sözcükler: "Boxing Day clash", "North London Derby", "Emirates Stadium",
# "Gameweek 11", "St James' Park". Rakip yalnız bunlardan oluşuyorsa rakip yok sayılır (normal maç araması);
# sözlükte olmayan adla PL maçı eşleştirilirken de sayılmaz ("Exeter City" Manchester City değil).
GENERIC_OPP = set("""
boxing day gameweek matchweek gw round stadium park arena ground road lane emirates anfield etihad trafford old bridge
stamford selhurst goodison craven cottage molineux turf moor st james mary marys north south east west london
merseyside manchester tyne wear midlands black country local city united town county athletic albion wanderers
rovers fc afc the derby clash showdown trip visit test tie fixture game match week weekend home away cup final semi
league big new year christmas festive opener
""".split())


def opp_words(name: str) -> set[str]:
    """Rakip metninin ayırt edici sözcükleri (genel sözcükler ve kısa sözcükler hariç)."""
    return {w for w in _alias_norm(name or "").split() if len(w) > 2 and w not in GENERIC_OPP}


_OPP_RE = re.compile(rf"(?:(?i:\b(?:previews?|ahead\s+of|against|versus|vs)\b\.?)|\bv\b\.?)\s+((?:{_NAME_WORD}\s?){{1,3}})")


def _opponent_by_pattern(title: str, home: str | None) -> str | None:
    """"previews Wycombe", "ahead of Sporting", "vs Union SG": sözlükte olmayan ya da bağlama bağlı rakip.
    "on" bağlacı yok: "Arteta on Bukayo Saka's fitness" rakip değil."""
    t = _titlecase_caps(title)
    for m in _OPP_RE.finditer(t):
        words = [_strip_poss(w) for w in m.group(1).split()]
        words = [w for w in words if w and _norm(w) not in _NOT_NAME]
        if not words:
            continue
        cand = " ".join(words)
        canon = _ALIAS_MAP.get(_alias_norm(cand))
        if canon:
            return None if canon == home else canon
        if home and _alias_norm(cand) == _alias_norm(home):
            return None
        if not opp_words(cand):
            continue  # "Boxing Day", "North London Derby": rakip adı değil
        return cand
    return None


def _is_opponent_mention(title: str, team: str) -> bool:
    t = _tnorm(title)
    aliases = [a for a, c in _ALIASES if c == team]
    for a in aliases:
        if re.search(r"(?:\bpre |\bpost |\bpreviews? |\bahead of |\bagainst |\bbefore |\bafter |\bvs |\bv |\bversus |"
                     r"\bfaces? |\bplay )" + re.escape(a) + r"(?![a-z])", t):
            return True
        # Türkçe / Azerice: "Stuttgart maçı öncesi", "Chelsea ile", "Qarabağ oyunundan sonra"
        if re.search(r"(?<![a-z])" + re.escape(a) + r"(?:'\w+)?\s+(?:maci|macinin|macindan|karsilasmasi\w*|ile|"
                     r"oyunu\w*|deplasmani\w*|engeli)\b", t):
            return True
    return False


_PRESSER_RE = re.compile(
    r"press[ -]?conf|presser|pre[ -]?match|post[ -]?match|basin toplanti|maci? (?:oncesi|sonrasi|sonu)|"
    r"metbuat konfrans|oyun(?:dan|undan)? (?:evvel|sonra)|news conference|media conference|rueda de prensa|"
    r"conferencia de prensa|pressekonferenz|conferenza stampa|conference de presse|persconferentie|coletiva")
# Kulüp kanalında rakip adı geçse de basın toplantısı / röportaj olmayan videolar
_NOT_PRESSER_RE = re.compile(
    r"highlight|extended|\bgoals?\b|behind the scenes|tunnel|unseen|training|antrenman|idman|full match|"
    r"replay|women|u ?21|u ?23|u ?18|academy|vlog|asmr|compilation|skills|matchday|match day|\blive\b|stream|"
    r"watchalong|\bfans?\b|podcast|\bshow\b|\bkit\b|launch|unboxing|documentary|trailer|\btour\b|ticket|"
    r"rewind|classic|on this day|mic'?d|\bbts\b|inside|moments|quiz|challenge|prank|fifa|efootball|ozet|"
    r"\bgol(?:ler)?\b")
# Kulübün kendi kanalı: kanal adında kulüp adından başka yalnız bunlar olabilir ("Liverpool FC", "LFC TV")
_CLUB_CHANNEL_WORDS = {"fc", "afc", "cf", "sc", "fk", "pfk", "sk", "jk", "official", "football", "club", "the", "tv",
                       "hd", "channel", "futbol", "spor", "kulubu", "klubu"}


def club_of_channel(channel: str) -> str | None:
    """Kanal kulübün kendisi mi (Arsenal, Tottenham Hotspur, Qarabağ FK)? Taraftar / gazete / podcast kanalı
    ("Arsenal Fan TV", "Liverpool Echo", "The Arsenal Vision Podcast") değil."""
    t, hits = _team_hits(channel or "")
    teams = _uniq(c for *_, c in hits)
    if len(teams) != 1:
        return None
    rest = t
    for a, b, _ in sorted(hits, reverse=True):
        rest = rest[:a] + " " + rest[b:]
    words = [w for w in re.split(r"[\s|:\-/&']+", rest) if w]
    return teams[0] if all(w in _CLUB_CHANNEL_WORDS or len(w) == 1 for w in words) else None


def looks_like_presser(info: dict, match: dict | None = None) -> bool:
    """Başlık/açıklama basın toplantısına benziyor mu (ya da bilinen bir teknik direktör adı geçiyor mu)?
    find_manager büyük harfli her ad dizisini aday sayar ("Golden State"), bu yüzden yalnız bilinen adlar sayılır.
    Kulüp kanalında (Arsenal, Tottenham Hotspur...) başlıkta tanınan bir rakip varsa da evet ("X previews Arsenal
    clash", "X reacts to Brighton defeat"); özet / antrenman / kadın takımı videoları hariç."""
    m = match or {}
    mgr = m.get("manager") or ""
    title = info.get("title") or ""
    channel = info.get("channel") or info.get("uploader") or ""
    if mgr and _known_manager(mgr.split(), find_teams(f"{title} | {channel}")):
        return True
    text = _norm(f"{title} {(info.get('description') or '')[:300]}")
    if _PRESSER_RE.search(text):
        return True
    club = club_of_channel(channel)
    opp = m.get("away") if m.get("team") in (None, m.get("home")) else m.get("home")
    # rakip başlıkta geçmeli (açıklamadaki rakip yetmez: forma tanıtımı "... against Chelsea on Sunday")
    if club and is_team(opp) and opp != club and (opp in find_teams(title) or opp == _opponent_by_pattern(title, club)) \
            and not _NOT_PRESSER_RE.search(_norm(title)):
        return True
    return False


def _joined(t: str, hits: list) -> bool:
    """Başlıktaki ilk iki farklı takım "A vs B", "A - B", "A 2-1 B" biçiminde mi (sıra ev - deplasman)?"""
    if not hits:
        return False
    a0, b0, c0 = hits[0]
    nxt = next((h for h in hits[1:] if h[2] != c0), None)
    if not nxt:
        return False
    return re.fullmatch(r"\s*(?:" + _LINK + r")?\s*", t[b0:nxt[0]]) is not None


def _ts_date(info: dict) -> str | None:
    """release_timestamp / timestamp -> 'YYYY-MM-DD' (UTC)."""
    from datetime import datetime, timezone
    for k in ("release_timestamp", "timestamp"):
        try:
            if info.get(k):
                return datetime.fromtimestamp(int(info[k]), tz=timezone.utc).date().isoformat()
        except (ValueError, OSError, OverflowError, TypeError):
            pass
    return None


def parse_match(info: dict) -> dict:
    title = info.get("title") or ""
    channel = info.get("channel") or info.get("uploader") or ""
    desc = (info.get("description") or "")[:400]
    t, hits = _team_hits(title)
    title_teams = _uniq(c for _, _, c in hits)
    chan_teams = find_teams(channel)
    home = chan_teams[0] if chan_teams else (title_teams[0] if title_teams else None)
    away = next((x for x in title_teams if x != home), None)
    if len(title_teams) >= 2 and home in title_teams[:2] and away in title_teams[:2] and _joined(t, hits):
        home, away = title_teams[0], title_teams[1]  # "Everton vs Coventry City" sırası korunur
    if away is None:
        away = next((x for x in find_teams(desc) if x != home), None)
    if away is None:
        away = _opponent_by_pattern(title, home)
    team = chan_teams[0] if chan_teams else home  # basın toplantısının sahibi takım (kanal)
    if not chan_teams and len(title_teams) == 1 and _is_opponent_mention(title, title_teams[0]):
        team, home, away = None, None, title_teams[0]  # "pre-Chelsea", "previews Chelsea": Chelsea rakip
    manager = find_manager(title, title_teams + chan_teams)
    d = str(info.get("upload_date") or "")
    date = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if re.fullmatch(r"\d{8}", d) else None
    # tarafsız kanal (Sky, BBC...): toplantının sahibi hocanın takımı, başlıkta ilk yazılan takım değil
    mt = manager_team(manager, date or _ts_date(info))
    if mt and not chan_teams:
        if mt in title_teams:
            team = mt
            if home is None:
                home, away = mt, next((x for x in title_teams if x != mt), away)
        elif len(title_teams) <= 1:
            # rakip: başlıktaki tek takım, yoksa açıklamadaki ilk başka takım (açıklama çoğu zaman hocanın kendi
            # kulübünü de anar: "Manchester City manager Enzo Maresca..."), yoksa kalıpla bulunan
            other = title_teams[0] if title_teams else next((x for x in find_teams(desc) if x != mt), None)
            if other is None and away not in (None, mt) and not is_team(away):
                other = away
            team, home, away = mt, mt, other
    if away is not None and away == home:
        away = None  # takım kendisiyle oynamaz ("Manchester City vs Manchester City")
    return {"home": home, "away": away, "team": team, "manager": manager, "date": date}


def safe_filename(s: str, maxlen: int = 110) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", s)
    s = re.sub(r"[^\w\s.,'&()\-–]+", " ", s)  # emoji vb.
    s = re.sub(r"\s+", " ", s).strip(" .-")
    s = s[:maxlen].rstrip(" .-") or "kayit"
    if s.split(".")[0].strip().upper() in _RESERVED:  # Windows bu adlarla dosya/klasör açamaz (CON, NUL, COM1...)
        s = f"_{s}"
    return s


_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}


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
