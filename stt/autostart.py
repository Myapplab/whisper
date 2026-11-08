"""Windows açılınca STT'yi başlatma (1.9.27, Ayarlar > "Otomatik çalışma (ajan)").

Hermes gibi bir ajan linkleri yerel API'ye gönderir; STT kapalıysa (bilgisayar yeniden başladı, Windows güncellemesi)
gönderemez. Kayıt: HKCU\\...\\Run değeri (yönetici izni gerekmez; Görev Yöneticisi > Başlangıç uygulamaları'nda görünür,
kullanıcı oradan kapatabilir: ``StartupApproved``). Komut venv'deki pythonw.exe ile ``-m stt.gui --baslangic``: paket
kurulumda düzenlenebilir kurulu (``pip install -e .``), çalışma klasörüne bağlı değil; pencere küçültülmüş açılır.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
VALUE = "STT Basin Odasi"
ARG = "--baslangic"
REG = None  # testler sahte kayıt defteri koyar (get / set / delete)


def _pythonw() -> str:
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


def command() -> str:
    return f'"{_pythonw()}" -m stt.gui {ARG}'


class _WinReg:
    """HKEY_CURRENT_USER altında okuma / yazma (winreg)."""

    def get(self, key: str, name: str):
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                return winreg.QueryValueEx(k, name)[0]
        except OSError:
            return None

    def set(self, key: str, name: str, value: str) -> None:
        import winreg
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)

    def delete(self, key: str, name: str) -> None:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, name)
        except FileNotFoundError:
            pass


def _reg():
    if REG is not None:
        return REG
    return _WinReg() if os.name == "nt" else None


def status() -> dict:
    """{"supported", "enabled" (kayıt var), "current" (bu kurulumun komutu), "approved" (Görev Yöneticisi'nde
    başlangıçta açık mı; None = hiç dokunulmamış = açık), "command"}."""
    r = _reg()
    if r is None:
        return {"supported": False, "enabled": False, "current": False, "approved": None, "command": ""}
    try:
        cur = r.get(RUN_KEY, VALUE)
        appr = r.get(APPROVED_KEY, VALUE)
    except Exception:  # noqa: BLE001
        cur, appr = None, None
    approved = None
    if isinstance(appr, (bytes, bytearray)) and appr:
        approved = not (appr[0] & 1)  # 02 / 06 = açık, 03 / 07 = kullanıcı Görev Yöneticisi'nde kapatmış
    return {"supported": True, "enabled": bool(cur), "current": cur == command(), "approved": approved,
            "command": cur or ""}


def set_enabled(on: bool) -> str:
    """Kaydı yaz / sil. Dönüş: hata metni ya da ""."""
    r = _reg()
    if r is None:
        return "Windows açılışında başlatma yalnız Windows'ta yapılabilir."
    try:
        if on:
            r.set(RUN_KEY, VALUE, command())
        else:
            r.delete(RUN_KEY, VALUE)
    except OSError as e:
        return f"Windows açılış kaydı {'yazılamadı' if on else 'silinemedi'}: {e}"
    return ""


def ensure() -> None:
    """Ajan modu açıkken kayıt eksik ya da eski kurulum yolundaysa yeniden yaz. Kullanıcı Görev Yöneticisi'nde
    kapattıysa dokunulmaz (onun kararı)."""
    st = status()
    if st["supported"] and st["approved"] is not False and not st["current"]:
        set_enabled(True)
