"""Veri seti düzeni: lig / sezon / takım / hoca / maç / basın toplantıları + index.csv.

    cikti/
      Premier League/
        2026-2027/                                     <- sezon (1 Temmuz - 30 Haziran) (1.9.28: önce sezon)
          Arsenal/
            Mikel Arteta/                              <- maç günü takımın teknik direktörü (1.9.26); sezon ortasında
                                                          hoca değişirse yeni klasör: Chelsea/Enzo Maresca, Chelsea/Liam Rosenior
              2026-11-08 GW11 Arsenal vs Chelsea/
                mac.json                                   <- fikstür bilgisi (FPL), maça ait ek veriler buraya
                2026-11-07 mac-oncesi - Mikel Arteta.txt   <- soru-cevap metni
                2026-11-07 mac-oncesi - Mikel Arteta.json  <- zaman + konuşmacı
                2026-11-07 mac-oncesi - Mikel Arteta.srt
                2026-11-07 mac-oncesi - Mikel Arteta - ozet.md
              2026-11-04 Champions League Arsenal vs Slavia Prague/
        2023-2024/
          Luton Town/Rob Edwards/...                   <- o sezon PL'de olan (sonra düşen) takım da Premier League'de
      Diger Takimlar/
        2026-2027/Galatasaray/Okan Buruk/...
        Bilinmeyen/<başlık>/                           <- takımı bulunamayan
      index.csv / index.jsonl                          <- her basın toplantısı bir satır (model eğitimi için)

Maç ve GW, FPL'nin herkese açık fikstür verisinden bulunur (fantasy.premierleague.com/api; yalnız güncel sezon).
Eski sezonlar (1.9.28): ``assets/fpl_history.json`` (FPL'nin sezon sonu kopyası, vaastav/Fantasy-Premier-League;
``dev/build_fpl_history.py``) + her sezonun indirilen FPL verisi ``.stt/fpl_<sezon>.json`` olarak saklanır.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath

from .match import (_ALIAS_MAP, _ALIASES, _alias_norm, _known_manager, _norm, club_of_channel, find_teams,
                    history_manager, is_team, manager_team, opp_suffix_words, opp_words, safe_filename)
from .util import atomic_write_text, log, read_jsonl, write_jsonl_csv

FPL_BASE = "https://fantasy.premierleague.com/api"
PL_DIR = "Premier League"
OTHER_DIR = "Diger Takimlar"
GENERAL_DIR = "Podcast ve Film"

# FPL'ye ulaşılamazsa: 2026-27 Premier League takımları (kısa kodlarıyla). 2025-26'dan düşen Wolves, Burnley,
# West Ham; çıkan Coventry, Ipswich, Hull. Sezon başında güncellenmeli.
PL_FALLBACK = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
    "Chelsea": "CHE", "Coventry City": "COV", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Hull City": "HUL", "Ipswich Town": "IPS", "Leeds United": "LEE", "Liverpool": "LIV",
    "Manchester City": "MCI", "Manchester United": "MUN", "Newcastle United": "NEW", "Nottingham Forest": "NFO",
    "Sunderland": "SUN", "Tottenham": "TOT",
}

COMPETITIONS = [
    (r"champions league|\bucl\b|şampiyonlar ligi|sampiyonlar ligi|çempionlar liqası", "Champions League"),
    (r"europa league|\buel\b|avrupa ligi", "Europa League"),
    (r"conference league|\buecl\b|konferans ligi", "Conference League"),
    (r"\bfa cup\b|emirates fa cup", "FA Cup"),
    (r"carabao|league cup|efl cup", "EFL Cup"),
    (r"community shield", "Community Shield"),
    (r"club world cup", "Club World Cup"),
    (r"premier league|\bepl\b", "Premier League"),
    (r"süper lig|super lig|trendyol", "Süper Lig"),
    (r"premyer liqa", "Premyer Liqa"),
]

CUPS = {"Champions League", "Europa League", "Conference League", "FA Cup", "EFL Cup", "Community Shield",
        "Club World Cup"}

_lock = threading.Lock()
FPL_RETRY_S = 600      # FPL'ye ulaşılamadıysa bu süre ağ denenmez (her video 2 x 20 sn beklemesin)
_fail_until = 0.0


# ------------------------------------------------------------------ FPL verisi

def _get_json(url: str) -> dict | list:
    import requests
    r = requests.get(url, timeout=20, headers={"User-Agent": "stt/1.0"})
    r.raise_for_status()
    return r.json()


def _read_cache(cache: Path) -> dict | None:
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def fpl_data(cache_dir: Path, max_age_h: float = 6, force: bool = False) -> dict | None:
    """{'teams': {fpl_id: {...}}, 'fixtures': [...]} — önbellekli; erişilemezse eski önbellek ya da None.
    Erişilemeyince FPL_RETRY_S boyunca yeniden denenmez (eski önbellek / None hemen döner); ``force``: kullanıcı
    elle istedi (yeniden sınıflandır), bekleme süresine bakmadan dene."""
    global _fail_until
    cache = Path(cache_dir) / "fpl_cache.json"
    with _lock:
        if cache.exists() and time.time() - cache.stat().st_mtime < max_age_h * 3600:
            data = _read_cache(cache)
            if data is not None:
                return data
        if time.time() < _fail_until and not force:
            return _read_cache(cache) if cache.exists() else None
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
            _archive(cache.parent, data)
            _fail_until = 0.0
            return data
        except Exception as e:  # noqa: BLE001
            _fail_until = time.time() + FPL_RETRY_S
            old = _read_cache(cache) if cache.exists() else None
            log.warning("FPL fikstür verisi alınamadı (%s); %s", str(e)[:120],
                        "önceki fikstür verisi kullanılıyor." if old else
                        f"maç/GW eşleştirmesi yapılamıyor, {FPL_RETRY_S // 60} dk sonra tekrar denenecek.")
            return old


# ------------------------------------------------------------------ eski sezonlar (1.9.28)

HISTORY_FILE = Path(__file__).resolve().parent / "assets" / "fpl_history.json"
_history: dict | None = None


def history_seasons() -> dict[str, dict]:
    """Uygulamayla gelen eski sezon fikstürleri: {"2023-2024": {"teams", "fixtures"}, ...}. Okunamazsa {}."""
    global _history
    if _history is None:
        try:
            d = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            _history = {k: v for k, v in (d.get("seasons") or {}).items()
                        if isinstance(v, dict) and v.get("teams") and v.get("fixtures")}
        except Exception as e:  # noqa: BLE001
            log.warning("Eski sezon fikstürleri okunamadı (%s): %s", HISTORY_FILE, e)
            _history = {}
    return _history


def data_season(data: dict | None) -> str | None:
    """FPL verisinin sezonu ("2026-2027"): ilk maçın tarihinden."""
    kos = [str(f.get("kickoff_time")) for f in (data or {}).get("fixtures") or [] if f.get("kickoff_time")]
    return season_of({"match_date": min(kos)[:10]}) if kos else None


def _archive_path(cache_dir: Path, season: str) -> Path:
    return Path(cache_dir) / f"fpl_{season}.json"


def _archive(cache_dir: Path, data: dict) -> None:
    """İndirilen FPL verisinin sezonluk kopyası: sezon bitip FPL yeni sezona geçince eski sezonun kayıtları yine
    maç / GW bulsun (uygulamayla gelen geçmiş dosyasında henüz yoksa)."""
    s = data_season(data)
    if not s:
        return
    try:
        atomic_write_text(_archive_path(cache_dir, s), json.dumps(data))
    except OSError as e:
        log.warning("FPL sezon kopyası yazılamadı: %s", e)


def season_data(when: datetime | str | None, data: dict | None, cache_dir: Path | None = None) -> dict | None:
    """Basın toplantısı tarihinin sezonuna ait fikstür verisi. ``data``: güncel FPL verisi. Sıra: güncel veri o
    sezonunsa o; yoksa uygulamayla gelen geçmiş (sezon sonu, tam); yoksa ``cache_dir``'deki sezon kopyası. Hiçbiri
    yoksa None (o sezonun fikstürü elde yok). Tarih yoksa ``data``."""
    if when is None:
        return data
    day = when if isinstance(when, str) else when.date().isoformat()
    s = season_of({"match_date": day[:10]})
    if not s:
        return data
    if data and data_season(data) == s:
        return data
    hist = history_seasons().get(s)
    if hist:
        return hist
    if cache_dir is not None:
        p = _archive_path(cache_dir, s)
        if p.is_file():
            d = _read_cache(p)
            if isinstance(d, dict) and d.get("fixtures") and data_season(d) == s:
                return d
    return None


def canon_team(name: str) -> str:
    """FPL takım adı -> sözlükteki kanonik ad. Ad tam bir takma adsa (belirsiz "Derby" dahil) doğrudan."""
    exact = _ALIAS_MAP.get(_alias_norm(name or ""))
    if exact:
        return exact
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


def presser_time(info: dict) -> tuple[datetime | None, bool]:
    """(basın toplantısı zamanı, yalnız gün mü). yt-dlp saat vermeyip yalnız upload_date verirse (bazı X / kulüp
    sitesi videoları) gün 12:00 UTC alınır; o zaman maç öncesi / sonrası saatle değil günle karşılaştırılır."""
    for k in ("release_timestamp", "timestamp"):
        if info.get(k):
            try:
                return datetime.fromtimestamp(int(info[k]), tz=timezone.utc), False
            except (ValueError, OSError, OverflowError, TypeError):
                pass
    d = str(info.get("upload_date") or "")
    if re.fullmatch(r"\d{8}", d):
        try:
            return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), 12, tzinfo=timezone.utc), True
        except ValueError:
            pass
    return None, False


def presser_datetime(info: dict) -> datetime | None:
    return presser_time(info)[0]


# Başlıktan maç evresi: "Post-Match", "Arteta reacts", "reacts to dramatic late win", takımlı skor "Arsenal 2-1 Chelsea",
# "maç sonrası / maç sonu" / "pre-match", "previews", "pre-Chelsea", "öncesi". İpucu SAYILMAYANLAR: "after" ("after the
# international break", "kick on after Chelsea win": çoğu zaman sonraki maçın öncesi), "reacts to transfer rumours",
# "reaction to call-up", "Champions League / FA Cup draw" (kura), "out for 2-3 weeks", "25-26 season", Türkçe tek
# başına "sonrası" ("milli ara sonrası").
_RESULT = (r"(?:win|wins|won|winner|defeat|loss|lost|victory|beat|result|game|match|thrashing|thriller|comeback|"
           r"equaliser|equalizer|drew)")
_SCORE_TXT = (r"(?<![\d-])\d{1,2} ?- ?\d{1,2}(?![\d-])(?! ?(?:weeks?|wks?|days?|months?|years?|seasons?|games?|"
              r"matches|mins?|minutes?|hafta\w*|gun\w*|ay\b|yil\w*|sezon\w*|mac\w*|dakika\w*|hefte\w*|movsum\w*|"
              r"with\b))")
_REACT_TO = r"\breact(?:s|ion)? to (?:the |our |their |his |a )?(?:[\w'-]+ ){0,3}"
_POST_RE = re.compile(
    r"post[ -]?(?:match|game)|"
    rf"\breact(?:s|ion)?\b(?! to\b)|{_REACT_TO}(?:{_SCORE_TXT}|{_RESULT}\b)|"
    r"(?:mac|matc|karsilasma|oyun|derbi|deplasman|galibiyet|yenilgi|beraberlik|maglubiyet)\w* "
    r"(?:sonrasi|sonra|ardindan)\b|\bmac\w* sonu\b")
# "reacts to Chelsea draw / last night's draw" maç sonucu; "reacts to Champions League / FA Cup / last 16 /
# quarter-final draw" kura (kura sözcüğü "draw"ın hemen önünde)
_POST_DRAW_RE = re.compile(rf"{_REACT_TO}draw\b")
_CUP_DRAW_RE = re.compile(r"\b(?:league|cup|group(?: stage)?|knockout(?: stage| round)?|round of \d+|last (?:16|32|64)|"
                          r"quarter[- ]?finals?|semi[- ]?finals?|finals?|play[- ]?offs?|kura|europa|conference) draw\b")
_PRE_RE = re.compile(r"pre[ -]?(?:match|game)|\bpreviews?\b|\bahead of\b|\bbefore (?:the )?(?:game|match)\b|"
                     r"\bpre[ -](?!season)[a-z]|\boncesi\b|\bevvel\b|build[ -]?up")
# skor: "Arsenal 2-1 Chelsea", "2-1 win", "Qarabağ 2:1 Sabah", "(2:1)"; "2-3 weeks", "4-3-3", "25-26 season",
# tarih, saat (15:00), "1:1 with Arteta" / "1-1 with Saka" (birebir röportaj) değil
_SCORE_RE = re.compile(rf"{_SCORE_TXT}|(?<![\d:])\d:\d(?![\d:])(?!\s*with\b)")


def phase_hint(title: str) -> str | None:
    """Başlık maç öncesi mi sonrası mı diyor? İkisi birden ya da hiçbiri: None (saate göre karar verilir).
    Skor ancak başlıkta takım adı ya da sonuç sözcüğü de varsa maç sonrası sayılır."""
    t = _norm(re.sub(r"[–—]", "-", title or ""))
    post = bool(_POST_RE.search(t)) or bool(_POST_DRAW_RE.search(t) and not _CUP_DRAW_RE.search(t)) or bool(
        _SCORE_RE.search(t) and (find_teams(title) or re.search(rf"\b(?:{_RESULT}|draw)\b", t)))
    pre = bool(_PRE_RE.search(t))
    if post == pre:
        return None
    return "mac-sonrasi" if post else "mac-oncesi"


def _same_club(name: str, team: str | None) -> bool:
    """Sözlükte olmayan rakip metni o takımın bir adıyla örtüşüyor mu? Ayırt edici sözcük varsa yalnız o sayılır
    ("Exeter City" Manchester City değil); yalnız kulüp eki varsa ek ("previews City" -> Manchester City)."""
    if not team:
        return False
    team_words = {w for a, c in _ALIASES if c == team for w in a.split()}
    own = opp_words(name) or opp_suffix_words(name)
    return bool(own & team_words)


def find_fixture(team: str, opponent: str | None, when: datetime, data: dict, phase: str | None = None,
                 day_only: bool = False, window: float | None = None) -> dict | None:
    """Takımın, basın toplantısı zamanına en yakın PL maçı (2 gün önce .. 6 gün sonra; ``window`` verilirse ±window).

    ``phase``: başlıktaki ipucu. "mac-sonrasi" yalnız oynanmış, "mac-oncesi" yalnız oynanacak maçlar (eskiden ertesi
    gün yüklenen maç sonrası toplantı bir sonraki maça yazılıyordu). ``day_only``: zaman yalnız gün olarak biliniyor;
    aynı gün = maç sonrası sayılır (maç günü toplantısız saat bilgisi yoksa çoğunlukla maçtan sonradır)."""
    teams = pl_teams(data)
    if team not in teams or not teams[team].get("id"):
        return None
    tid = teams[team]["id"]
    opp_id = teams.get(opponent or "", {}).get("id")
    lo, hi = (-window, window) if window is not None else (-2, 6)
    best, best_score = None, None
    for f in data.get("fixtures", []):
        if tid not in (f.get("team_h"), f.get("team_a")):
            continue
        ko = _parse_dt(f.get("kickoff_time"))
        if not ko:
            continue
        other = f["team_a"] if f["team_h"] == tid else f["team_h"]
        if opp_id and other != opp_id:
            continue
        if day_only:
            d = (ko.date() - when.date()).days
            if d < lo or d > hi or (phase == "mac-sonrasi" and d > 0) or (phase == "mac-oncesi" and d < 0):
                continue
            score = -d if phase == "mac-sonrasi" else d if phase == "mac-oncesi" else \
                0 if d == 0 else abs(d) + (0 if d > 0 else 1.5)
        else:
            d = (ko - when).total_seconds() / 86400
            if d < lo or d > hi or (phase == "mac-sonrasi" and d > 0) or (phase == "mac-oncesi" and d <= 0):
                continue
            score = -d if phase == "mac-sonrasi" else d if phase == "mac-oncesi" else \
                abs(d) + (0 if d >= -0.3 else 1.5)  # maç öncesi basın toplantısı daha olası
        if best_score is None or score < best_score:
            best, best_score = f, score
    return best


def competitions_in(text: str) -> list[str]:
    t = (text or "").lower()
    return [name for pat, name in COMPETITIONS if re.search(pat, t)]


def competition_from_text(text: str) -> str | None:
    found = competitions_in(text)
    return found[0] if found else None


FIXTURE_FIELDS = ("fixture_id", "kickoff_utc", "is_home", "opponent_short", "score", "finished")


def fixture_details(fx: dict, teams: dict, team: str | None) -> dict:
    """Veri seti için maç alanları: FPL maç no, başlama (UTC), ev sahibi mi, rakip kısa adı, skor (ev-deplasman)."""
    tid = teams.get(team or "", {}).get("id")
    is_home = (fx.get("team_h") == tid) if tid else None
    opp_id = (fx.get("team_a") if is_home else fx.get("team_h")) if tid else None
    hs, as_ = fx.get("team_h_score"), fx.get("team_a_score")
    return {"fixture_id": fx.get("id"), "kickoff_utc": fx.get("kickoff_time"), "is_home": is_home,
            "opponent_short": next((v["short"] for v in teams.values() if v.get("id") == opp_id), None),
            "score": f"{hs}-{as_}" if fx.get("finished") and hs is not None and as_ is not None else None,
            "finished": bool(fx.get("finished"))}


# ------------------------------------------------------------------ sınıflandırma

def classify(match: dict, info: dict, data: dict | None, cache_dir: Path | None = None) -> dict:
    """Basın toplantısını lig/takım/maç olarak yerleştir.

    - ``data``: güncel FPL verisi. Toplantı başka sezondansa o sezonun fikstürü kullanılır (``season_data``; 1.9.28):
      eski sezonda da maç + GW bulunur, o sezon PL'de olan (sonra düşen) takım Premier League'e gider. O sezonun
      verisi hiç yoksa güncel veri (maç bulunmaz; eski davranış).

    - Yarışma önce başlıktan; başlıkta yoksa açıklamadan, ama açıklamadaki kupa adı (kulüpler önceki/sonraki kupa
      maçını rutin anar) o rakiple PL maçı varsa PL'yi ezmez.
    - Rakip sözlükte olmayan bir metinse ("Bukayo Saka's", "City"), rakipsiz ±3 gün içindeki PL maçı aranır;
      bilinen PL dışı takımsa (Slavia Prague) kupa / Avrupa maçıdır, PL maçına yazılmaz.
    - Maç öncesi / sonrası başlıktaki ipucuyla seçilir (``phase_hint``)."""
    team = match.get("team") or match.get("home")
    if team:
        opponent = match.get("away") if team == match.get("home") else match.get("home")
    else:
        opponent = match.get("away") or match.get("home")
    when, day_only = presser_time(info)
    data = season_data(when, data, cache_dir) or data
    teams = pl_teams(data)
    title = info.get("title") or ""
    comp_title = competition_from_text(title)
    desc_comps = [] if comp_title else competitions_in((info.get("description") or "")[:300])
    comp = comp_title or (desc_comps[0] if desc_comps else None)
    hint = phase_hint(title)
    meta = {"team": team, "opponent": opponent, "manager": match.get("manager"), "competition": comp,
            "is_pl": bool(team and team in teams), "fpl_short": teams.get(team or "", {}).get("short"),
            "presser_date": when.date().isoformat() if when else None, "match_date": None, "gw": None,
            "phase": None, "home": None, "away": None, "fixture": None, **{k: None for k in FIXTURE_FIELDS}}
    kw = {"phase": hint, "day_only": day_only}
    opp_is_pl = bool(opponent and opponent in teams)
    if not team and opp_is_pl and when and data:
        # takım bilinmiyor ama rakip PL takımı: o tarihteki maçından takımı çıkar ("pre-Chelsea" -> Arsenal)
        fx0 = find_fixture(opponent, None, when, data, **kw)
        if fx0:
            by_id = {v["id"]: k for k, v in teams.items()}
            oid = teams[opponent]["id"]
            team = by_id.get(fx0["team_a"] if fx0["team_h"] == oid else fx0["team_h"])
            meta.update(team=team, is_pl=bool(team in teams), fpl_short=teams.get(team or "", {}).get("short"))
    fx = None
    if meta["is_pl"] and when and data:
        cup_in_title = comp_title not in (None, "Premier League")
        desc_cup_only = any(c in CUPS for c in desc_comps) and "Premier League" not in desc_comps
        if cup_in_title:
            pass                                                   # "| Carabao Cup": kupa maçı
        elif opponent is None or opp_is_pl:
            if not desc_cup_only or opp_is_pl:                     # açıklamadaki kupa, o rakiple PL maçını ezmez
                fx = find_fixture(team, opponent, when, data, **kw)
        elif not is_team(opponent) or comp_title == "Premier League":
            # sözlükte olmayan rakip: ±3 gündeki PL maçı yalnız rakibi o adla örtüşüyorsa ("Manchester" ->
            # Manchester United). Örtüşmüyorsa (Wycombe) sözlükte olmayan bir kulüple kupa maçıdır.
            if not desc_cup_only:
                fx = find_fixture(team, None, when, data, window=3, **kw)
                if fx:
                    by_id = {v["id"]: k for k, v in teams.items()}
                    other = by_id.get(fx["team_a"] if fx["team_h"] == teams[team]["id"] else fx["team_h"])
                    vague = not (opp_words(str(opponent)) or opp_suffix_words(str(opponent)))  # "Manchester"
                    if not (comp_title == "Premier League" or vague or _same_club(str(opponent), other)):
                        fx = None
                if fx:
                    log.info("Rakip sözlükte yok (%s); %s maçı kullanıldı.", str(opponent)[:60], fx.get("id"))
    if fx:
        by_id = {v["id"]: k for k, v in teams.items()}
        ko = _parse_dt(fx["kickoff_time"])
        if hint:
            phase = hint
        elif day_only and ko:
            phase = "mac-oncesi" if (ko.date() - when.date()).days > 0 else "mac-sonrasi"
        else:
            phase = "mac-oncesi" if ko and ko > when else "mac-sonrasi"
        meta.update(fixture=fx, gw=fx.get("event"), competition="Premier League",
                    home=by_id.get(fx["team_h"]), away=by_id.get(fx["team_a"]),
                    match_date=ko.date().isoformat() if ko else None, phase=phase)
        meta["opponent"] = meta["away"] if meta["home"] == meta["team"] else meta["home"]
        meta.update(fixture_details(fx, teams, meta["team"]))
    else:
        meta["home"], meta["away"] = match.get("home"), match.get("away")
        meta["match_date"] = meta["presser_date"]
        meta["phase"] = hint
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


SEASON_START_MONTH = 7          # sezon 1 Temmuz'da başlar (hazırlık maçları yeni sezona, Mayıs finali eskisine)
NO_SEASON = "Sezon bilinmiyor"  # tarihi olmayan kayıt


def season_of(meta: dict) -> str | None:
    """Kaydın sezonu: "2026-2027". Maç tarihi (yoksa basın toplantısı tarihi) 1 Temmuz'dan önceyse önceki sezon.
    Aynı maçın toplantıları aynı sezon klasörüne düşsün diye önce maç tarihi. Tarih yoksa None."""
    for k in ("match_date", "presser_date"):
        m = re.fullmatch(r"(\d{4})-(\d{2})-\d{2}", str(meta.get(k) or ""))
        if m and 1 <= int(m.group(2)) <= 12:
            y = int(m.group(1)) - (int(m.group(2)) < SEASON_START_MONTH)
            return f"{y}-{y + 1}"
    return None


def _meta_day(meta: dict) -> str | None:
    """season_of ile aynı gün: maç tarihi, yoksa basın toplantısı tarihi (aynı maçın toplantıları aynı klasöre)."""
    for k in ("match_date", "presser_date"):
        d = str(meta.get(k) or "")
        if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])-\d{2}", d):
            return d
    return None


def _speaker_manager(name, team: str, day: str, channel_team: str | None = None) -> str | None:
    """Konuşan, o gün bu takımın hocası olarak bilinen biri mi: listelere göre o gün bu takımda, ya da takımı
    bilinmeyen bilinen bir hoca (Okan Buruk) ve video kulübün kendi kanalından. Tarafsız kanalda başka hocanın
    videosu ("Hansi Flick press conference | Arsenal vs Barcelona", "Guardiola on Arsenal") sayılmaz; oyuncu /
    bilinmeyen ad: None."""
    if not name:
        return None
    full = _known_manager(str(name).split(), [team])
    if not full:
        return None
    mt = manager_team(full, day)
    return full if mt == team or (mt is None and channel_team == team) else None


_learn_cache: dict = {}


def _index_rows(root: Path) -> list[dict]:
    p = Path(root) / "index.jsonl"
    try:
        st = p.stat()
    except OSError:
        return []
    sig = (str(p), st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino)
    if _learn_cache.get("sig") != sig:
        _learn_cache.update(sig=sig, rows=read_jsonl(p))
    return _learn_cache["rows"]


def _learned_manager(root: Path, team: str, day: str, key: str | None) -> str | None:
    """Geçmiş tablosunda olmayan takım: aynı takımın aynı sezondaki başka kayıtlarının hocası (index'teki
    ``team_manager``; yalnız güvenilir kaynaktan yazılır) — oyuncunun / yardımcının toplantısı hocanın klasörüne.
    Önce o günden önceki en yakın kayıt, yoksa sonraki."""
    season = season_of({"match_date": day})
    before, after = None, None
    for r in _index_rows(root):
        if r.get("team") != team or r.get("key") == key or r.get("league") == GENERAL_DIR:
            continue
        rd = _meta_day(r)
        if not rd or season_of({"match_date": rd}) != season:
            continue
        who = r.get("team_manager")
        if not who:
            continue
        if rd <= day:
            if before is None or rd > before[0]:
                before = (rd, who)
        elif after is None or rd < after[0]:
            after = (rd, who)
    return (before or after or (None, None))[1]


def team_manager(meta: dict, root: Path | None = None, key: str | None = None,
                 channel: str | None = None) -> str | None:
    """Maç günü takımın teknik direktörü (sezon klasörünün adı: "2025-2026 Pep Guardiola"). Sıra:
    1. hoca geçmişi tablosu (match.MANAGER_HISTORY; geçici hocalar dahil). Tabloya henüz girilmemiş değişiklikte
       eski hoca kalır (konuşana göre değiştirilmez: tarafsız kanalda başka hocanın videosu sahte klasör açardı,
       aynı maçın oyuncu toplantısı da ayrı klasöre düşerdi);
    2. tabloda yoksa konuşan, o takımın hocası olarak bilinen kişi (``_speaker_manager``; ``channel``: video kanalı);
    3. ``root`` verilirse aynı takımın aynı sezondaki başka kayıtlarının hocası.
    Bilinmiyorsa None (klasör yalnız sezon)."""
    team = meta.get("team")
    day = _meta_day(meta)
    if not team or meta.get("general") or not day:
        return None
    hist = history_manager(team, day)
    if hist:
        return hist
    speaker = _speaker_manager(meta.get("manager"), team, day, club_of_channel(channel or ""))
    if speaker:
        return speaker
    return _learned_manager(root, team, day, key) if root is not None else None


def set_team_manager(root: Path, meta: dict, key: str | None = None, channel: str | None = None) -> dict:
    """``meta['team_manager']``'ı hesaplar (bitirmede ve yeniden sınıflandırmada, klasör hesaplanmadan önce)."""
    meta["team_manager"] = team_manager(meta, root, key, channel)
    return meta


def season_folder(season: str | None, manager: str | None = None) -> str:
    """1.9.26 düzeninin sezon klasörü ("2025-2026 Pep Guardiola"; hoca bilinmiyorsa "2025-2026"). 1.9.28'den beri
    yalnız eski klasörleri tanımak için (yeni düzen: ``season_dir`` + ``manager_dir``)."""
    if not season or season == NO_SEASON:
        return NO_SEASON
    return safe_filename(f"{season} {manager}") if manager else season


def season_dir(meta: dict) -> str:
    """Ligin altındaki sezon klasörü: "2025-2026"; tarihsiz kayıtta "Sezon bilinmiyor"."""
    return season_of(meta) or NO_SEASON


def manager_dir(meta: dict) -> str | None:
    """Takımın altındaki hoca klasörü (maç günkü teknik direktör; ``team_manager``). Bilinmiyorsa None: maç
    klasörü doğrudan takımın altında."""
    tm = meta.get("team_manager")
    return safe_filename(str(tm)) if tm and season_of(meta) else None


def season_dir_name(meta: dict) -> str:
    return season_folder(season_of(meta), meta.get("team_manager"))


def rel_path(p, root) -> str:
    """Veri seti köküne göre yol, her sistemde "/" ile (index.csv/jsonl'yi okuyan betik Windows'a bağlı kalmasın)."""
    return PurePath(p).relative_to(PurePath(root)).as_posix()


def target_dir(root: Path, meta: dict, fallback_title: str) -> Path:
    if meta.get("general"):
        return Path(root) / GENERAL_DIR / safe_filename(fallback_title, 70)
    if not meta.get("team"):
        return Path(root) / OTHER_DIR / "Bilinmeyen" / safe_filename(fallback_title, 70)
    league = PL_DIR if meta.get("is_pl") else OTHER_DIR
    team_dir = Path(root) / league / season_dir(meta) / safe_filename(meta["team"])
    mdir = manager_dir(meta)
    return (team_dir / mdir if mdir else team_dir) / match_folder_name(meta)


MAX_PATH = 250  # Windows yol sınırı (260) altında kalmak için


def out_paths(root: Path, meta: dict, title: str, key: str, create: bool = True,
              base: str | None = None) -> tuple[Path, str]:
    """Çıktı klasörü + dosya adı kökü. Windows yol sınırı gözetilir; aynı klasörde aynı adı başka video almışsa
    " [abcdef]" eki. ``create=False``: yalnız hesapla (yeniden sınıflandırmanın deneme turu). ``base``: dosya adı
    hesaplanmaz, bu kullanılır (yalnız sezon klasörüne taşınan eski kayıt adını korur)."""
    d = target_dir(root, meta, title)
    base = base or presser_base_name(meta, title)
    # Windows 260 karakter yol sınırı: en uzun dosya "<base> - ozet.md.tmp"
    budget = MAX_PATH - len(str(d)) - len(" - ozet.md.tmp") - 12  # 12: olası " [abcdef]" eki
    if budget < 20:  # klasör adı çok uzun (ör. video başlığı): kısalt
        d = d.parent / d.name[:max(20, len(d.name) - (20 - budget))].rstrip(" .-")
        budget = MAX_PATH - len(str(d)) - len(" - ozet.md.tmp") - 12
    if len(base) > budget:
        base = base[:max(20, budget)].rstrip(" .-")
    marker = d / f".{base}.key"
    if marker.exists() and marker.read_text().strip() != key:
        base = f"{base} [{key[2:8]}]"
        marker = d / f".{base}.key"
    if create:
        d.mkdir(parents=True, exist_ok=True)
        marker.write_text(key)
    return d, base


def write_match_json(folder: Path, meta: dict) -> None:
    p = folder / "mac.json"
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    if not isinstance(data, dict):
        data = {}
    data.update({k: meta.get(k) for k in ("competition", "match_date", "gw", "home", "away", "team", "opponent",
                                          "fpl_short", "fixture_id", "kickoff_utc", "is_home", "opponent_short",
                                          "score")})
    data["season"] = season_of(meta)
    data["team_manager"] = meta.get("team_manager")
    if meta.get("fixture"):
        data["fpl_fixture"] = meta["fixture"]
    data.setdefault("ek_bilgiler", {})  # maça ait başka veriler (kadro, sakatlık, skor...) buraya eklenebilir
    atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=1))


INDEX_FIELDS = ["key", "presser_date", "phase", "league", "team", "opponent", "competition", "gw", "match_date",
                "home", "away", "manager", "language", "duration_s", "questions", "words", "diarized", "url",
                "txt", "json",
                # 1.9.8: FPL modeli için (eski satırlarda boş)
                "fixture_id", "kickoff_utc", "is_home", "fpl_short", "opponent_short", "score", "speaker_role",
                "speaker_source", "diar_model", "title",
                # 1.9.25: sezon ("2026-2027"; klasör düzeniyle aynı)
                "season",
                # 1.9.26: maç günü takımın teknik direktörü (sezon klasörü "<sezon> <hoca>"); konuşan "manager"dır
                "team_manager"]


def league_of(meta: dict) -> str:
    return GENERAL_DIR if meta.get("general") else PL_DIR if meta.get("is_pl") else "Diger"


def index_row(root: Path, key: str, meta: dict, doc: dict, paths: dict) -> dict:
    """index.csv / index.jsonl satırı (bitirmede ve yeniden sınıflandırmada aynı)."""
    from .outputs import qa_blocks
    n_q = 0
    if doc.get("speakers") and doc.get("speaker_mode") == "basin":
        n_q = sum(1 for h, _, _ in qa_blocks(doc["segments"], doc.get("td_label")) if h.startswith("Soru"))
    mgr = meta.get("manager")
    return {
        **{k: meta.get(k) for k in ("presser_date", "phase", "team", "opponent", "competition", "gw", "match_date",
                                    "home", "away", "manager", "fixture_id", "kickoff_utc", "is_home", "fpl_short",
                                    "opponent_short", "score")},
        "key": key, "league": league_of(meta),
        "language": doc.get("language"), "duration_s": doc.get("duration_s"), "questions": n_q,
        "words": sum(len((s.get("text") or "").split()) for s in doc.get("segments") or []),
        "diarized": bool(doc.get("speakers")), "url": doc.get("webpage_url") or doc.get("source"),
        "txt": rel_path(paths["txt"], root), "json": rel_path(paths["json"], root),
        # konuşan bilinen bir teknik direktör mü (değilse oyuncu / listede olmayan hoca olabilir)
        "speaker_role": "hoca" if mgr and _known_manager(str(mgr).split()) else None,
        "speaker_source": doc.get("speaker_source"),
        "diar_model": (doc.get("settings") or {}).get("diar_model"), "title": doc.get("title"),
        "season": season_of(meta) if meta.get("team") and not meta.get("general") else None,
        "team_manager": meta.get("team_manager") if meta.get("team") and not meta.get("general") else None}


def update_index(root: Path, row: dict) -> None:
    """index.jsonl (anahtar başına tek satır) ve index.csv'yi günceller."""
    root = Path(root)
    with _lock:
        jl = root / "index.jsonl"
        rows = {r["key"]: r for r in read_jsonl(jl) if r.get("key")}
        rows[row["key"]] = {k: row.get(k) for k in INDEX_FIELDS}
        ordered = sorted(rows.values(), key=lambda r: (r.get("league") or "", r.get("team") or "",
                                                       r.get("match_date") or "", r.get("presser_date") or ""))
        write_jsonl_csv(jl, root / "index.csv", ordered, INDEX_FIELDS)
