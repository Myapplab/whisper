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


def mark_helper_open(state_dir: Path) -> None:
    """Yardımcı kart (monitörlü) iş için açıldı: zamanı yaz (sürücü çökerse ilişki kurulabilsin)."""
    try:
        p = _mark_path(state_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        old = {}
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        p.write_text(json.dumps({"first": old.get("first") or int(time.time()), "last": int(time.time())}),
                     encoding="utf-8")
    except OSError:
        log.debug("yardımcı kart zamanı yazılamadı", exc_info=True)


def helper_crash(state_dir: Path, events: list[dict]) -> dict | None:
    """Yardımcı kart ilk açıldıktan sonra ekran sürücüsü çöktüyse / bilgisayar kapandıysa o olay (en yenisi)."""
    try:
        mark = json.loads(_mark_path(state_dir).read_text(encoding="utf-8"))
        since = int(mark.get("first") or 0)
    except Exception:  # noqa: BLE001
        return None
    hits = [e for e in events if e["t"] >= since - 60]
    return hits[-1] if hits else None


def clear_mark(state_dir: Path) -> None:
    try:
        _mark_path(state_dir).unlink()
    except OSError:
        pass
