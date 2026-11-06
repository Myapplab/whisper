"""Ekran sürücüsü çökmeleri (Windows olay günlüğü) ve yardımcı kart koruması.

Kullanıcıda (1.9.13 sonrası) iki kez ekran tamamen karardı, monitör "HDMI sinyali yok" dedi. Bu, ekran kartı
sürücüsünün takılıp sıfırlanması (TDR) ya da çökmesidir; Windows bunu olay günlüğüne yazar:
- System / nvlddmkm (ör. 13, 14, 153 "sürücü takıldı / SM hatası")
- System / Display 4101 ("ekran sürücüsü yanıt vermedi ve kurtarıldı")
- Application / Windows Error Reporting 1001 "LiveKernelEvent" 141 / 117 (video motoru zaman aşımı)
- System / Kernel-Power 41, EventLog 6008 (bilgisayar beklenmedik kapandı: donup yeniden başlatıldı / güç)
Muhtemel sebepler: monitörlü karta (3080) ağır CUDA yükü (yardımcı mod), ekran kartı hız aşırtması (MSI Afterburner
çekirdek / bellek +), güç kaynağı. Yardımcı kart açıkken sürücü çöktüyse yardımcı mod kendiliğinden kapatılır.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .util import log, popen_kw

PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$s = (Get-Date).AddDays(-%DAYS%)
$a = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm','Display','Microsoft-Windows-Kernel-Power','EventLog'; StartTime=$s} |
  Where-Object { $_.ProviderName -eq 'nvlddmkm' -or ($_.ProviderName -eq 'Display' -and $_.Id -eq 4101) -or
                 ($_.ProviderName -eq 'Microsoft-Windows-Kernel-Power' -and $_.Id -eq 41) -or ($_.ProviderName -eq 'EventLog' -and $_.Id -eq 6008) })
$b = @(Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName='Windows Error Reporting'; Id=1001; StartTime=$s} |
  Where-Object { $_.Message -match 'LiveKernelEvent' })
@($a + $b) | Select-Object -First 100 | ForEach-Object {
  $m = if ($_.Message) { ($_.Message -replace '\s+', ' ') } else { '' }
  [pscustomobject]@{ t = ([DateTimeOffset]$_.TimeCreated).ToUnixTimeSeconds(); id = $_.Id; p = $_.ProviderName;
                     m = $m.Substring(0, [Math]::Min(240, $m.Length)) }
} | ConvertTo-Json -Compress
"""


def classify(ev: dict) -> str:
    """'driver' (ekran sürücüsü takıldı / çöktü) | 'reset' (bilgisayar beklenmedik kapandı) | ''."""
    p, i, m = str(ev.get("p") or ""), int(ev.get("id") or 0), str(ev.get("m") or "")
    if p == "nvlddmkm" or (p == "Display" and i == 4101):
        return "driver"
    if p == "Windows Error Reporting" and "LiveKernelEvent" in m:
        return "driver" if any(c in m for c in ("141", "117", "1a8", "VIDEO", "Video")) else ""
    if (p.endswith("Kernel-Power") and i == 41) or (p == "EventLog" and i == 6008):
        return "reset"
    return ""


def parse_events(text: str) -> list[dict]:
    """PowerShell ConvertTo-Json çıktısı (tek nesne ya da liste) -> [{'t','id','p','m','kind'}] (eskiden yeniye)."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    rows = data if isinstance(data, list) else [data]
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            r = {"t": int(r.get("t")), "id": int(r.get("id") or 0), "p": str(r.get("p") or ""), "m": str(r.get("m") or "")}
        except (TypeError, ValueError):
            continue
        r["kind"] = classify(r)
        if r["kind"]:
            out.append(r)
    out.sort(key=lambda r: r["t"])
    return out


def display_events(days: int = 7) -> list[dict]:
    """Son ``days`` gündeki ekran sürücüsü çökmeleri / beklenmedik kapanmalar. Windows değilse ya da okunamazsa []."""
    if os.name != "nt":
        return []
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS.replace("%DAYS%", str(int(days)))],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, **popen_kw())
    except Exception as e:  # noqa: BLE001
        log.debug("olay günlüğü okunamadı: %s", e)
        return []
    return parse_events(r.stdout)


def fmt_time(t: int) -> str:
    return time.strftime("%d.%m %H:%M", time.localtime(t))


def describe(events: list[dict]) -> str:
    drv = [e for e in events if e["kind"] == "driver"]
    rst = [e for e in events if e["kind"] == "reset"]
    parts = []
    if drv:
        parts.append(f"ekran sürücüsü {len(drv)} kez takıldı/çöktü (" + ", ".join(fmt_time(e["t"]) for e in drv[-5:]) + ")")
    if rst:
        parts.append(f"bilgisayar {len(rst)} kez beklenmedik kapandı (" + ", ".join(fmt_time(e["t"]) for e in rst[-5:]) + ")")
    return "; ".join(parts)


# ------------------------------------------------------------------ yardımcı kart koruması
def _mark_path(state_dir: Path) -> Path:
    return Path(state_dir) / "yardimci_acik.json"


def _read(state_dir: Path) -> dict | None:
    try:
        data = json.loads(_mark_path(state_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _read_windows(state_dir: Path) -> list[list]:
    wins = (_read(state_dir) or {}).get("windows")
    return [w for w in wins if isinstance(w, list) and len(w) == 2] if isinstance(wins, list) else []


def _write_windows(state_dir: Path, wins: list[list], since=...) -> None:
    """since: kullanıcının modu seçtiği / korumanın sıfırlandığı an; None = bilinmiyor (eski sürüm: son
    LEGACY_DAYS günün çökmeleri de sayılır). Verilmezse dosyadaki korunur."""
    try:
        if since is ...:
            since = (_read(state_dir) or {}).get("since")
        p = _mark_path(state_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"windows": wins[-50:], "since": since}), encoding="utf-8")
    except OSError:
        log.debug("yardımcı kart zamanı yazılamadı", exc_info=True)


def mark_helper_open(state_dir: Path) -> None:
    """Yardımcı kart (monitörlü) iş için açıldı: [açılış, kapanış] aralığı başlar (kapanış boş = hâlâ açık).
    Yalnız bu aralıklardaki çökmeler sayılır (inceleme: STT kapalıyken oyunun çökerttiği sürücü de sayılıyordu)."""
    wins = _read_windows(state_dir)
    if wins and wins[-1][1] is None:
        return  # zaten açık
    wins.append([int(time.time()), None])
    _write_windows(state_dir, wins)


def mark_helper_closed(state_dir: Path) -> None:
    """Yardımcı kartta açık işçi kalmadı (ya da uygulama düzgün kapandı): açık aralığı kapat. Bilgisayar donup
    kapanırsa bu yazılmaz: aralık açık kalır ve sonraki açılıştaki çökme / beklenmedik kapanma sayılır."""
    wins = _read_windows(state_dir)
    if wins and wins[-1][1] is None:
        wins[-1][1] = int(time.time())
        _write_windows(state_dir, wins)


LEGACY_DAYS = 3  # işaret dosyası hiç yoksa (1.9.14 öncesi sürümle kullanıldı) son bu kadar gündeki çökmeler sayılır


def helper_crash(state_dir: Path, events: list[dict]) -> dict | None:
    """Monitörlü kart iş yaparken (bir aralığın içinde, 2 dk pay) olan son ekran sürücüsü çökmesi / beklenmedik
    kapanma. Kapanışı yazılmamış aralık (bilgisayar o sırada kapandı) bir sonraki açılışa dek sayılır.
    İşaret dosyası HİÇ yoksa (kullanıcı 1.9.13'te yardımcıyla 5 kez üst üste kararma yaşadı; o sürüm işaret yazmıyordu)
    son ``LEGACY_DAYS`` gündeki çökmeler sayılır. Kullanıcı modu yeniden seçince (``clear_mark``) boş dosya yazılır:
    eski çökmeler bir daha sayılmaz."""
    data = _read(state_dir)
    hits: list[dict] = []
    if data is None or data.get("since") is None:
        # eski sürüm / ilk kez: motor bu oturumda dosyayı oluşturmuş olsa da (aralık yazıldı) karar korunur
        hits += [e for e in events if e["t"] >= time.time() - LEGACY_DAYS * 86400]
    wins = _read_windows(state_dir)
    hits += [e for e in events for a, b in wins if a - 60 <= e["t"] <= (b if b is not None else time.time()) + 120]
    return max(hits, key=lambda e: e["t"]) if hits else None


def clear_mark(state_dir: Path, keep_open: bool = False) -> None:
    """Yeni başlangıç: eski aralıklar ve "son 3 gün" kuralı silinir (since = şimdi). keep_open: monitörlü kart hâlâ
    çalışıyorsa açık aralık yeni başlangıçla sürer (ikinci kararma da kaydedilsin)."""
    now = int(time.time())
    _write_windows(state_dir, [[now, None]] if keep_open else [], since=now)


def display_card_used(gpus: str | None, info: dict) -> bool:
    """Bu kart ayarı monitörün bağlı olduğu ekran kartına iş veriyor mu (yardımcı, hepsi, elle kart numarası)."""
    from .pipeline import detect_devices, helper_devices
    # display None (bilinmiyor: aynı adlı iki kart, sürücü alanı desteklemiyor) = monitörlü olabilir
    disp = {i for i, g in (info or {}).items() if g.get("display") is not False}
    if not disp:
        return False
    used = detect_devices(gpus or "auto", info) + helper_devices(gpus or "auto", info)
    return any(d.startswith("cuda:") and int(d.split(":")[1]) in disp for d in used)
