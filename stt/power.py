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
            "aorusengine", "aorus engine")
# görevi standart kullanıcı da başlatabilsin (Microsoft "elevated task model"): uygulama güç sınırı iş sürerken kalkarsa
# (uyku dönüşü, sürücü sıfırlanması, Afterburner) izin sormadan yeniden uygular. Görevi değiştirmek yalnız yönetici / SYSTEM.
TASK_SDDL = "D:(A;;FA;;;BA)(A;;FA;;;SY)(A;;GRGX;;;AU)"
RESUME_QUERY = ("<QueryList><Query Id=\"0\" Path=\"System\"><Select Path=\"System\">*[System[Provider["
                "@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]</Select></Query></QueryList>")
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
    roots = [system_dir(), os.environ.get("SystemRoot") or r"C:\Windows",
             os.environ.get("ProgramFiles") or r"C:\Program Files", os.environ.get("ProgramW6432") or r"C:\Program Files"]
    return any(p.startswith(os.path.normcase(os.path.abspath(r)) + os.sep) for r in roots if r)


def system_dir() -> str:
    """Windows System32 (ortam değişkenine güvenmeden: GetSystemDirectoryW)."""
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.kernel32.GetSystemDirectoryW(buf, 260):
                return buf.value
        except Exception:  # noqa: BLE001
            pass
    return os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32")


def smi_path() -> str:
    """nvidia-smi: önce System32'deki (DCH sürücüleri oraya koyar, sürücü güncellemesinde yeri değişmez), yoksa PATH'teki
    (yalnız yönetici yazabilen bir yerdeyse)."""
    import shutil
    fixed = os.path.join(system_dir(), "nvidia-smi.exe")
    if os.name == "nt" and os.path.isfile(fixed):
        return fixed
    p = shutil.which("nvidia-smi") or ""
    return p if safe_exe(p) else ""


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
            'baslamasin). Kaldirmak icin: STT Ayarlar &gt; Ekran karti &gt; Geri al.</Description>'
            f'<SecurityDescriptor>{TASK_SDDL}</SecurityDescriptor></RegistrationInfo>\n'
            '  <Triggers>\n'
            '    <BootTrigger><Enabled>true</Enabled><Delay>PT1M</Delay></BootTrigger>\n'
            '    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>\n'
            f'    <EventTrigger><Enabled>true</Enabled><Subscription>{escape(RESUME_QUERY)}</Subscription>'
            '<Delay>PT20S</Delay></EventTrigger>\n'
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
    tail: (anahtar, tam komut) — schtasks gibi nvidia-smi dışı komutlar. Dosyaya tek yönlendirme: ShellExecuteEx
    ~2048 karakterden uzun satırı kesiyor (inceleme: yol her komutta 3 kez geçince 2 kartta ~1900 karakterdi)."""
    parts = [f"({cmd} && echo STT_OK {key}|| echo STT_FAIL {key})"
             for key, cmd in [(k, f"{_q(smi)} {a}") for k, a in cmds] + list(tail)]
    return "(" + " & ".join(parts) + f") >{_q(res)} 2>&1"


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
    try:  # pencere verilmezse izin penceresi öne gelmeyip görev çubuğunda yanıp sönebiliyor
        sei.hwnd = ctypes.windll.user32.GetForegroundWindow()
    except Exception:  # noqa: BLE001
        pass
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
    return any(ch in p for ch in '%"^!\r\n')  # cmd satırında güvenle tırnaklanamaz


MAX_PARAMS = 1900  # ShellExecuteEx satır sınırı (~2048) altında kal


def reapply(timeout: float = 20.0) -> bool:
    """Kayıtlı görevi izin sormadan çalıştır (görev standart kullanıcıya 'başlat' izni verir). Görev yoksa / izin yoksa
    False. Sonucu sınırı yeniden okuyarak anlayın (görev arka planda birkaç saniye sürer)."""
    if os.name != "nt":
        return False
    try:
        r = subprocess.run([os.path.join(system_dir(), "schtasks.exe"), "/Run", "/TN", TASK], capture_output=True,
                           text=True, errors="replace", timeout=timeout, **popen_kw())
    except Exception:  # noqa: BLE001
        return False
    if r.returncode != 0:
        log.info("[güç] görev çalıştırılamadı: %s", (r.stderr or r.stdout or "").strip()[:200])
        return False
    log.info("[güç] güç sınırı kalkmıştı: görev yeniden çalıştırıldı")
    return True


def restrict_or_reapply(devices: list[str], helpers: list[str], gi: dict, cfg, cards: dict | None = None,
                        wait_s: float = 8.0) -> tuple[list, list, str, dict]:
    """``restrict``; sınır yoksa önce kayıtlı görevi çalıştırıp (hızlı başlatma / uyku / Afterburner sınırı silmiş
    olabilir) yeniden dener. Dönüş: (devices, helpers, sebep, kartlar)."""
    cards = query() if cards is None else cards
    d, h, why = restrict(devices, helpers, gi, cfg, cards)
    if why and reapply():
        time.sleep(wait_s)
        cards = query() or cards
        d, h, why = restrict(devices, helpers, gi, cfg, cards)
    return d, h, why, cards


def apply(cfg, undo: bool = False, runner=None, cards: dict | None = None) -> dict:
    """Güç + saat sınırını şimdi uygula ve her açılışta uygulayan görevi kaydet (undo: varsayılana dön, görevi sil).
    Dönüş: {"ok", "error", "status"}. runner: testler için run_elevated yerine."""
    if os.name != "nt" and runner is None:
        return {"ok": False, "error": "Yalnız Windows'ta yapılabilir."}
    smi = smi_path()
    if not smi:
        return {"ok": False, "error": "nvidia-smi bulunamadı (ya da beklenmedik bir klasörde). NVIDIA sürücüsünü "
                "yeniden kurun."}
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
    sysdir = system_dir()
    cmd_exe = os.path.join(sysdir, "cmd.exe")
    schtasks = _q(os.path.join(sysdir, "schtasks.exe"))
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
    params = "/d /v:off /s /c " + _q(elevated_line(smi, cmds, res, tail))
    if len(params) > MAX_PARAMS:
        return {"ok": False, "error": f"Komut satırı çok uzun ({len(params)} karakter; kart sayısı / klasör yolu)."}
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
    st = status(cfg)  # yeniden oku: komut "başarılı" deyip sınırı değiştirmemiş olabilir
    keys = [k for k, _ in cmds] + ["task"]
    missing = [k for k in keys if k not in out["ok"] and k not in out["fail"]]  # yarıda kaldı / zaman aşımı
    pl_fail = [k for k in out["fail"] if k.startswith("pl@") or k.startswith("rgc@")] + \
        [k for k in missing if k != "task" and not k.startswith("lgc@")]
    msgs = []
    if pl_fail:
        msgs.append("güç sınırı " + ("kaldırılamadı" if undo else "konamadı") + ": " + ", ".join(pl_fail))
    if any(k.startswith("lgc@") for k in out["fail"] + missing):
        msgs.append("saat sınırı desteklenmedi (yalnız güç sınırı geçerli)")
    task_bad = "task" in out["fail"] or "task" in missing
    if task_bad and not undo:  # geri almada görev hiç kaydedilmemiş olabilir: sorun değil
        msgs.append("her açılışta uygulayan görev kaydedilemedi")
    if not undo and not pl_fail and st["supported"] and st["applied"] is False:
        msgs.append("komut başarılı göründü ama kartların sınırı değişmedi")
        pl_fail = ["status"]
    log.info("[güç] sonuç: tamam=%s hata=%s eksik=%s %s", out["ok"], out["fail"], missing, out["out"][:300])
    ok = not pl_fail and not (task_bad and not undo)
    err = "; ".join(msgs) + (f" ({out['out'][:200]})" if msgs and out["out"] else "")
    if not ok and not undo and pl_fail:
        err += (". Olmazsa aynı işi MSI Afterburner'da yapabilirsiniz: Reset, Power Limit %" + str(cfg.gpu_power_pct)
                + " ya da altı, Save, 'Apply at system startup' açık (uygulama bunu da kabul eder).")
    return {"ok": ok, "error": err, "warn": "; ".join(msgs) if ok else "", "status": st}


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
    elif len(gi) < 2:
        summary = "Tek ekran kartı var: iki kart düzeni gerekmez (güç sınırı yine de yeniden başlamaları önler)."
    elif len(free) < 2:
        summary = ("Monitör bir NVIDIA kartına bağlıyken iki kart güvenle birlikte çalışamaz (monitörlü karta yük ekranı "
                   "karartıyordu). Önce 1. adım: monitör kablosu anakarttaki çıkışa.")
    else:
        summary = "Eksik adımlar var (✗)."
    return {"items": items, "ready": ready, "summary": summary, "cards": st["cards"], "n": len(gi)}
