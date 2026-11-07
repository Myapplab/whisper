"""1.9.20: Intel sürücüsü kuruldu ama monitör hâlâ 3080'de görünüyor (kullanıcı, 2026-11-07)."""
from stt.engines import igpu_advice, mark_display

GI_MON = {0: {"name": "RTX 3080", "gb": 10.0, "display": True}, 1: {"name": "RTX 3090", "gb": 24.0, "display": False}}


def test_driver_installed_cable_still_on_nvidia():
    """Intel sürücüsü artık kurulu (ad görünüyor) ama masaüstüne bağlı değil: kablo hâlâ 3080'de.
    İleti sürücünün tamam olduğunu ve kalan tek adımın kablo olduğunu açıkça söylemeli."""
    ok, txt = igpu_advice([("Intel(R) UHD Graphics 630", False), ("NVIDIA GeForce RTX 3080", True),
                           ("NVIDIA GeForce RTX 3090", False)], GI_MON)
    assert not ok
    assert "sürücüsü kurulu ✓" in txt and "Intel(R) UHD Graphics 630" in txt
    assert "kablo" in txt and "RTX 3080" in txt and "anakarttaki" in txt
    assert "kapatın" in txt  # kablo kapalıyken değiştirilsin


def test_monitor_on_intel_and_nvidia_too():
    """Intel masaüstünde ama 3080 de masaüstünde: ikinci monitör / TV / ikinci kablo 3080'de."""
    gi = {0: {"name": "RTX 3080", "gb": 10.0, "display": True}, 1: {"name": "RTX 3090", "gb": 24.0, "display": False}}
    ok, txt = igpu_advice([("Intel(R) UHD Graphics 630", True), ("NVIDIA GeForce RTX 3080", True),
                           ("NVIDIA GeForce RTX 3090", False)], gi)
    assert not ok and "ikinci" in txt and "RTX 3080" in txt


def test_windows_list_wins_when_desktop_on_intel():
    """Masaüstünü Intel çiziyor ve 3080 Windows'un listesinde yok: nvidia-smi 'Enabled' dese de 3080 boş sayılır
    (yoksa kablo anakarta takılmışken bile 3080 hiç kullanılmıyordu)."""
    info = {0: {"name": "RTX 3080", "gb": 10, "display": True}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}
    mark_display(info, ["Intel(R) UHD Graphics 630"])
    assert info[0]["display"] is False and info[1]["display"] is False


def test_nvidia_enabled_kept_without_intel_desktop():
    """Masaüstü listesi Intel içermiyorsa eski temkinli davranış: nvidia-smi 'Enabled' korunur."""
    info = {0: {"name": "RTX 3080", "gb": 10, "display": True}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}
    mark_display(info, ["NVIDIA GeForce RTX 3090"])
    assert info[0]["display"] is True and info[1]["display"] is True
    info = {0: {"name": "RTX 3080", "gb": 10, "display": True}, 1: {"name": "RTX 3090", "gb": 24, "display": False}}
    mark_display(info, ["Intel(R) UHD Graphics 630", "NVIDIA GeForce RTX 3080"])  # iki monitör: 3080'de de var
    assert info[0]["display"] is True and info[1]["display"] is False
