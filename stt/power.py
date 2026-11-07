"""İki ekran kartı için güvenli düzen: güç ve saat sınırı (1.9.17).

Kullanıcıda iki ayrı sorun vardı:
- monitörlü karta (3080) iş verilince ekran kararıyor, açılmıyordu (1.9.13-1.9.15) -> çözüm: monitör kablosu anakarttaki
  (Intel) çıkışa; o zaman iki NVIDIA kart da "monitör yok" olur, "Önerilen" ikisini de kullanır (engines.choose_gpus);
- yalnız 3090 yükteyken bile bilgisayar kendiliğinden yeniden başlıyordu (1.9.16; mavi ekran kaydı yok = çoğu zaman güç
  kaynağının ekran kartının milisaniyelik güç sıçramalarında kendini kapatması, ya da hız aşırtma).
İki kart birlikte tam yükte bu sıçramalar toplanır. Bu yüzden iki kart birlikte yalnız güç sınırı (varsayılanın
``gpu_power_pct`` %'si) ve en yüksek saat sınırı (``gpu_max_clock`` MHz; boost sıçramalarını keser) uygulanmışken
çalışır (``safe_dual``). Sınır nvidia-smi ile konur (yönetici izni ister: Windows bir kez sorar), yeniden başlatmada
silindiği için Windows görev zamanlayıcısına "STT Guvenli Guc" görevi yazılır (her açılışta SYSTEM olarak uygular).

Görev XML ile kaydedilir, yükseltilmiş komutlar cmd.exe satırıdır (PowerShell betiği yok: Smart App Control betikleri
kısıtlayabilir). Görev yalnız yönetici yazabilen yerlerdeki programları çalıştırır (System32 / Program Files).
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from xml.sax.saxutils import escape

from .util import log, popen_kw

TASK = "STT Guvenli Guc"
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{8,64}$")
SLACK_W = 3.0  # sürücü sınırı tam sayıya yuvarlayabiliyor
OC_TOOLS = ("msiafterburner", "afterburner", "precisionx", "precision x1", "gputweak", "evga precision", "thundermaster",
            "aorus engine", "giga")
FIELDS = ("power.limit", "power.default_limit", "power.min_limit", "power.max_limit", "power.draw", "temperature.gpu",
          "clocks.gr")


def _f(x: str) -> float | None:
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None  # "[N/A]", "[Not Supported]"
    return v if v == v else None


def parse_query(rows: list[list[str]]) -> dict[int, dict]:
    """nvidia-smi satırları -> {index: {uuid, name, limit, default, min, max, draw, temp, clock}}."""
    out: dict[int, dict] = {}
    n = len(FIELDS)
    for r in rows:
        if len(r) < n + 3:
            continue
        try:
            i = int(r[0])
        except ValueError:
            continue
        vals = [_f(x) for x in r[-n:]]
        name = ",".join(r[2:-n]).strip().replace("NVIDIA GeForce ", "")
        out[i] = {"index": i, "uuid": r[1].strip(), "name": name, "limit": vals[0], "default": vals[1], "min": vals[2],
                  "max": vals[3], "draw": vals[4], "temp": vals[5], "clock": vals[6]}
    return out


def query() -> dict[int, dict]:
    """Kartların güç / ısı durumu (nvidia-smi; ana süreçte CUDA açılmaz). nvidia-smi yoksa {}."""
    from .engines import _smi
    return parse_query(_smi("index,uuid,name," + ",".join(FIELDS)))


def target_w(card: dict, pct: int) -> int | None:
    """Kartın hedef güç sınırı (W): varsayılanın pct %'si, kartın izin verdiği aralıkta."""
    d = card.get("default")
    if not d:
        return None
    w = d * max(1, int(pct)) / 100
    if card.get("min"):
        w = max(w, card["min"])
    if card.get("max"):
        w = min(w, card["max"])
    return int(round(w))


def status(cfg, cards: dict | None = None) -> dict:
    """{"supported": bool, "applied": True/False/None, "cards": [...]} — applied: kart(lar)ın şu anki sınırı hedefte
    ya da altında mı (Afterburner daha düşük koyduysa da olur). None = bilinmiyor (nvidia-smi yok / alan yok)."""
    cards = query() if cards is None else cards
    rows, known = [], []
    for i, c in sorted(cards.items()):
        t = target_w(c, cfg.gpu_power_pct)
        ok = None if (t is None or c.get("limit") is None) else c["limit"] <= t + SLACK_W
        rows.append({**c, "target": t, "ok": ok})
        if ok is not None:
            known.append(ok)
    return {"supported": bool(known), "applied": (all(known) if known else None), "cards": rows}


def restrict(devices: list[str], helpers: list[str], gi: dict, cfg, cards: dict | None = None) -> tuple[list, list, str]:
    """İki (ya da daha çok) kart birlikte kullanılacaksa ve güç sınırı uygulanmamışsa tek karta (monitörsüz, belleği en
    büyük) indir. Sınır bilinemiyorsa (nvidia-smi alanı yok) dokunma: uygulanamayan bir şart kartı sonsuza dek kapatmasın."""
    used = sorted({int(d.split(":")[1]) for d in list(devices) + list(helpers) if d.startswith("cuda:")})
    if len(used) < 2 or not getattr(cfg, "safe_dual", True):
        return devices, helpers, ""
    st = status(cfg, cards)
    bad = [c for c in st["cards"] if c["index"] in used and c["ok"] is False]
    if not bad:
        return devices, helpers, ""
    main = [d for d in devices if d.startswith("cuda:")]
    disp = {i for i, g in (gi or {}).items() if g.get("display")}
    pool = [d for d in main if int(d.split(":")[1]) not in disp] or main
    keep = max(pool, key=lambda d: ((gi or {}).get(int(d.split(":")[1]), {}).get("gb", 0), -int(d.split(":")[1])))
    names = ", ".join(c["name"] for c in bad)
    why = (f"İki ekran kartı birlikte çalışacaktı ama güvenli güç ayarı uygulanmamış ({names}). Bilgisayar yeniden "
           f"başlamasın diye şimdilik yalnız {(gi or {}).get(int(keep.split(':')[1]), {}).get('name', keep)} kullanılıyor. "
           "Ayarlar › Ekran kartı › 'Güvenli güç ayarını uygula'ya basın (Windows bir kez izin sorar); sonra iki kart "
           "birlikte çalışır.")
    return [keep], [], why


def card_line(c: dict) -> str:
    """Günlük için: 'RTX 3090 64°C 238/245 W 1590 MHz'."""
    parts = [c.get("name") or f"kart {c.get('index')}"]
    if c.get("temp") is not None:
        parts.append(f"{c['temp']:.0f}°C")
    if c.get("draw") is not None:
        parts.append(f"{c['draw']:.0f}/{c['limit']:.0f} W" if c.get("limit") else f"{c['draw']:.0f} W")
    if c.get("clock") is not None:
        parts.append(f"{c['clock']:.0f} MHz")
    return " ".join(parts)


def oc_tools() -> list[str] | None:
    """Açık hız aşırtma programları (MSI Afterburner...). Windows değilse None."""
    if os.name != "nt":
        return None
    try:
        r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace",
                           timeout=20, **popen_kw())
    except Exception:  # noqa: BLE001
        return None
    found = []
    for line in r.stdout.splitlines():
        name = line.split(",")[0].strip().strip('"')
        low = name.lower()
        if any(t in low for t in OC_TOOLS) and name not in found:
            found.append(name)
    return found


# ------------------------------------------------------------------ yükseltilmiş uygulama (Windows)
def safe_exe(path: str | None) -> bool:
    """SYSTEM görevi yalnız yönetici yazabilen yerdeki programı çalıştırsın (kullanıcı klasöründeki bir dosya her
    açılışta SYSTEM yetkisiyle çalışırsa güvenlik açığı olur)."""
    if not path:
        return False
    p = os.path.normcase(os.path.abspath(path))
    roots = [os.environ.get("SystemRoot") or r"C:\Windows", os.environ.get("ProgramFiles") or r"C:\Program Files",
             os.environ.get("ProgramW6432") or r"C:\Program Files"]
    return any(p.startswith(os.path.normcase(os.path.abspath(r)) + os.sep) for r in roots if r)


def _q(s: str) -> str:
    return '"' + s + '"'


def plan(cards: dict, pct: int, clock: int) -> list[tuple[str, str]]:
    """[(anahtar, nvidia-smi argümanları)]: önce bütün güç sınırları, sonra saat sınırları."""
    out = []
    ok = [c for _, c in sorted(cards.items()) if UUID_RE.match(c.get("uuid") or "")]
    for c in ok:
        w = target_w(c, pct)
        if w:
            out.append((f"pl@{c['index']}", f"-i {c['uuid']} -pl {w}"))
    if clock:
        for c in ok:
            out.append((f"lgc@{c['index']}", f"-i {c['uuid']} -lgc 210,{int(clock)}"))
    return out


def undo_plan(cards: dict) -> list[tuple[str, str]]:
    out = []
    for _, c in sorted(cards.items()):
        if not UUID_RE.match(c.get("uuid") or ""):
            continue
        if c.get("default"):
            out.append((f"pl@{c['index']}", f"-i {c['uuid']} -pl {int(round(c['default']))}"))
        out.append((f"rgc@{c['index']}", f"-i {c['uuid']} -rgc"))
    return out


def task_xml(smi: str, cmds: list[tuple[str, str]], cmd_exe: str) -> str:
    """Her açılışta (1 dk sonra) ve her oturum açılışında SYSTEM olarak sınırları yeniden uygulayan görev.
    Tek eylem (cmd /s /c a & b): bir komut başarısız olsa da ötekiler çalışır."""
    line = " & ".join(f"{_q(smi)} {a}" for _, a in cmds)
    return ('<?xml version="1.0" encoding="UTF-16"?>\n'
            '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
            '  <RegistrationInfo><Description>STT: ekran kartlari icin guc ve saat siniri (bilgisayar yeniden '
            'baslamasin). Kaldirmak icin: STT Ayarlar &gt; Ekran karti &gt; Geri al.</Description></RegistrationInfo>\n'
            '  <Triggers>\n'
            '    <BootTrigger><Enabled>true</Enabled><Delay>PT1M</Delay></BootTrigger>\n'
            '    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>\n'
            '  </Triggers>\n'
            '  <Principals><Principal id="Author"><UserId>S-1-5-18</UserId><RunLevel>HighestAvailable</RunLevel>'
            '</Principal></Principals>\n'
            '  <Settings>\n'
            '    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n'
            '    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n'
            '    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n'
            '    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>\n'
            '    <Enabled>true</Enabled>\n'
            '  </Settings>\n'
            '  <Actions Context="Author">\n'
            f'    <Exec><Command>{escape(cmd_exe)}</Command><Arguments>{escape("/d /s /c " + _q(line))}</Arguments></Exec>\n'
            '  </Actions>\n'
            '</Task>\n')


def elevated_line(smi: str, cmds: list[tuple[str, str]], res: str, tail: list[tuple[str, str]]) -> str:
    """cmd.exe /s /c "..." içi: her komutun sonucu sonuç dosyasına 'STT_OK anahtar' / 'STT_FAIL anahtar'.
    tail: (anahtar, tam komut) — schtasks gibi nvidia-smi dışı komutlar."""
    r = _q(res)
    parts = [f'(type nul >{r})']
    for key, cmd in [(k, f"{_q(smi)} {a}") for k, a in cmds] + list(tail):
        parts.append(f"({cmd} >>{r} 2>&1 && (echo STT_OK {key})>>{r} || (echo STT_FAIL {key})>>{r})")
    return " & ".join(parts)


def parse_result(text: str) -> dict:
    """Sonuç dosyası -> {"ok": [anahtar], "fail": [anahtar], "out": son satırlar}."""
    ok, fail, other = [], [], []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("STT_OK "):
            ok.append(s[7:].strip())
        elif s.startswith("STT_FAIL "):
            fail.append(s[9:].strip())
        elif s:
            other.append(s)
    return {"ok": ok, "fail": fail, "out": " | ".join(other[-6:])[:600]}


def _work_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(Path.home())
    return Path(base) / "stt"


def run_elevated(params: str, timeout: float = 300.0) -> int:
    """cmd.exe'yi yönetici olarak çalıştır (Windows izin penceresi). Dönüş: çıkış kodu; 1223 = kullanıcı izin vermedi."""
    import ctypes
    from ctypes import wintypes

    class SEI(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong), ("hwnd", wintypes.HWND),
                    ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
                    ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
                    ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
                    ("dwHotKey", wintypes.DWORD), ("hIconOrMonitor", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]

    try:
        ctypes.windll.ole32.CoInitializeEx(None, 0x2 | 0x4)  # STA + OLE1DDE kapalı (ShellExecuteEx önerisi)
    except Exception:  # noqa: BLE001
        pass
    sei = SEI()
    sei.cbSize = ctypes.sizeof(SEI)
    sei.fMask = 0x40 | 0x100  # SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC
    sei.lpVerb = "runas"
    sei.lpFile = os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", "cmd.exe")
    sei.lpParameters = params
    sei.nShow = 0  # SW_HIDE
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SEI)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    if not shell32.ShellExecuteExW(ctypes.byref(sei)):
        return int(ctypes.get_last_error() or 1)
    if not sei.hProcess:
        return 0
    try:
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject(sei.hProcess, int(timeout * 1000))
        code = wintypes.DWORD(0)
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess(sei.hProcess, ctypes.byref(code))
        return int(code.value)
    finally:
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle(sei.hProcess)


def _bad_path(p: str) -> bool:
    return any(ch in p for ch in '%"^\r\n')  # cmd satırında güvenle tırnaklanamaz


def apply(cfg, undo: bool = False, runner=None, cards: dict | None = None) -> dict:
    """Güç + saat sınırını şimdi uygula ve her açılışta uygulayan görevi kaydet (undo: varsayılana dön, görevi sil).
    Dönüş: {"ok", "error", "status"}. runner: testler için run_elevated yerine."""
    import shutil
    if os.name != "nt" and runner is None:
        return {"ok": False, "error": "Yalnız Windows'ta yapılabilir."}
    smi = shutil.which("nvidia-smi") or ""
    if not safe_exe(smi):
        return {"ok": False, "error": "nvidia-smi bulunamadı (ya da beklenmedik bir klasörde: " + (smi or "yok")
                + "). NVIDIA sürücüsünü yeniden kurun."}
    cards = query() if cards is None else cards
    if not cards:
        return {"ok": False, "error": "Ekran kartları okunamadı (nvidia-smi)."}
    wd = _work_dir()
    try:
        wd.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"ok": False, "error": f"Klasör açılamadı: {e}"}
    res, xml = str(wd / "guvenli_guc_sonuc.txt"), str(wd / "guvenli_guc_gorev.xml")
    if _bad_path(res) or _bad_path(smi):
        return {"ok": False, "error": "Klasör adında desteklenmeyen karakter var: " + res}
    sysroot = os.environ.get("SystemRoot") or r"C:\Windows"
    cmd_exe = os.path.join(sysroot, "System32", "cmd.exe")
    schtasks = _q(os.path.join(sysroot, "System32", "schtasks.exe"))
    if undo:
        cmds = undo_plan(cards)
        tail = [("task", f'{schtasks} /Delete /TN {_q(TASK)} /F')]
    else:
        clock = int(cfg.gpu_max_clock or 0)
        cmds = plan(cards, cfg.gpu_power_pct, clock)
        if not any(k.startswith("pl@") for k, _ in cmds):
            return {"ok": False, "error": "Kartların güç sınırı okunamadı (sürücü desteklemiyor olabilir)."}
        try:
            Path(xml).write_text(task_xml(smi, cmds, cmd_exe), encoding="utf-16")
        except OSError as e:
            return {"ok": False, "error": f"Görev dosyası yazılamadı: {e}"}
        tail = [("task", f'{schtasks} /Create /TN {_q(TASK)} /XML {_q(xml)} /F')]
    try:
        Path(res).unlink()
    except OSError:
        pass
    params = "/d /s /c " + _q(elevated_line(smi, cmds, res, tail))
    log.info("[güç] %s: %s", "geri alınıyor" if undo else "uygulanıyor", "; ".join(a for _, a in cmds))
    code = (runner or run_elevated)(params)
    if code == 1223:
        return {"ok": False, "error": "Windows'un sorduğu izne 'Evet' denmedi; hiçbir şey değişmedi."}
    try:
        out = parse_result(Path(res).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        out = {"ok": [], "fail": [], "out": ""}
    if not out["ok"] and not out["fail"]:
        return {"ok": False, "error": f"Komutlar çalışmadı (çıkış kodu {code})."}
    time.sleep(0.5)
    st = status(cfg) if runner is None else status(cfg, cards)
    pl_fail = [k for k in out["fail"] if k.startswith("pl@")]
    msgs = []
    if pl_fail:
        msgs.append("güç sınırı konamadı: " + ", ".join(pl_fail))
    if any(k.startswith("lgc@") for k in out["fail"]):
        msgs.append("saat sınırı desteklenmedi (yalnız güç sınırı geçerli)")
    if "task" in out["fail"]:
        msgs.append("her açılışta uygulayan görev " + ("silinemedi" if undo else "kaydedilemedi"))
    log.info("[güç] sonuç: tamam=%s hata=%s %s", out["ok"], out["fail"], out["out"][:300])
    ok = not pl_fail and "task" not in out["fail"]
    return {"ok": ok, "error": "; ".join(msgs) + (f" ({out['out'][:200]})" if msgs and out["out"] else ""),
            "warn": "; ".join(msgs) if ok else "", "status": st}


# ------------------------------------------------------------------ kontrol listesi (Ayarlar / Kontrol)
def safe_layout(cfg, gi: dict | None = None, cards: dict | None = None, adapters=..., tools=...) -> dict:
    """Ayarlar'daki 'İki kart güvenli düzen' listesi: [{"key", "ok": True/False/None, "text"}] + "ready"."""
    from .engines import gpu_info, igpu_advice, windows_adapters
    gi = gpu_info() if gi is None else gi
    cards = query() if cards is None else cards
    items = []
    adv = igpu_advice(windows_adapters() if adapters is ... else adapters, gi)
    free = [g["name"] for _, g in sorted(gi.items()) if g.get("display") is False]
    if adv is None:
        items.append({"key": "monitor", "ok": None, "text": "Monitör bağlantısı bilinmiyor (yalnız Windows'ta bakılır)."})
    else:
        items.append({"key": "monitor", "ok": adv[0], "text": adv[1]})
    st = status(cfg, cards)
    if not st["supported"]:
        items.append({"key": "power", "ok": None, "text": "Güç sınırı okunamadı (nvidia-smi yok ya da desteklemiyor)."})
    else:
        parts = [f"{c['name']} {c['limit']:.0f} W (hedef {c['target']} W)" for c in st["cards"] if c["ok"] is not None]
        if st["applied"]:
            items.append({"key": "power", "ok": True, "text": "Güvenli güç ayarı uygulanmış: " + ", ".join(parts)
                          + (f"; saat en çok {cfg.gpu_max_clock} MHz" if cfg.gpu_max_clock else "")
                          + ". Her açılışta kendiliğinden uygulanır."})
        else:
            items.append({"key": "power", "ok": False, "text": "Güvenli güç ayarı uygulanmamış: " + ", ".join(parts)
                          + ". Aşağıdaki düğmeye basın (Windows bir kez izin sorar). Kartlar biraz yavaşlar (~%10-15), "
                          "güç sıçramaları ve yeniden başlamalar kesilir."})
    t = oc_tools() if tools is ... else tools
    if t is None:
        items.append({"key": "oc", "ok": None, "text": "Hız aşırtma programı denetlenemedi."})
    elif t:
        items.append({"key": "oc", "ok": False, "text": f"Açık hız aşırtma programı: {', '.join(t)}. İçinde Reset'e, sonra "
                      "Save'e basın; ayarlardan 'Apply overclocking at system startup' (başlangıçta uygula) kapalı olsun. "
                      "Çekirdek / bellek hız aşırtması (+143 / +1441) ekran kartı hatası ve yeniden başlama yapar."})
    else:
        items.append({"key": "oc", "ok": True, "text": "Açık hız aşırtma programı yok."})
    mode = (cfg.gpus or "auto").strip().lower()
    if mode == "auto":
        items.append({"key": "mode", "ok": True, "text": "Ekran kartı kullanımı: Önerilen (monitörsüz kartların hepsi)."})
    else:
        items.append({"key": "mode", "ok": False, "text": "Ekran kartı kullanımı 'Önerilen' değil: güvenli düzen için "
                      "'Önerilen'i seçin (monitörlü karta iş verilmez, monitörsüz kartların hepsi çalışır)."})
    ready = all(i["ok"] for i in items if i["key"] in ("monitor", "power", "mode")) and len(free) >= 2
    if ready:
        summary = "Hazır: " + " + ".join(free) + " birlikte, güvenli güçle çalışır; monitör anakartta, ekran kararmaz."
    elif len(gi) >= 2 and len(free) < 2:
        summary = ("Şu an tek kart çalışır (monitör bir NVIDIA kartına bağlı). İkisinin birlikte çalışması için önce 1. "
                   "adım: monitör kablosu anakarttaki çıkışa.")
    else:
        summary = "Eksik adımlar var (✗)."
    return {"items": items, "ready": ready, "summary": summary, "cards": st["cards"]}
