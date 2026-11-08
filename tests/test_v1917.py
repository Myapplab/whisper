"""1.9.17: iki kart güvenli düzen (güç + saat sınırı, her açılışta görev; sınır yokken iki kart birlikte çalışmaz)."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import stt.power as pw
from stt.config import Config, normalize

U0 = "GPU-11111111-2222-3333-4444-555555555555"
U1 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GI_FREE = {0: {"name": "RTX 3080", "gb": 10.0, "display": False}, 1: {"name": "RTX 3090", "gb": 24.0, "display": False}}
GI_MON = {0: {"name": "RTX 3080", "gb": 10.0, "display": True}, 1: {"name": "RTX 3090", "gb": 24.0, "display": False}}


def _cards(l0=320.0, l1=350.0):
    rows = [["0", U0, "NVIDIA GeForce RTX 3080", f"{l0}", "320.00", "100.00", "340.00", "35.1", "45", "210"],
            ["1", U1, "NVIDIA GeForce RTX 3090", f"{l1}", "350.00", "100.00", "400.00", "[N/A]", "51", "1590"]]
    return pw.parse_query(rows)


def test_parse_query_and_targets():
    c = _cards()
    assert c[0]["name"] == "RTX 3080" and c[0]["uuid"] == U0 and c[1]["draw"] is None and c[1]["clock"] == 1590
    assert pw.target_w(c[0], 70) == 224 and pw.target_w(c[1], 70) == 245
    assert pw.target_w({"default": 150, "min": 125, "max": 200}, 50) == 125      # kartın en düşüğünün altına inmez
    assert pw.target_w({"default": None}, 70) is None
    assert pw.parse_query([["x", "y"], ["0", U0, "RTX", "1"]]) == {}             # bozuk satır


def test_config_bounds():
    c = normalize(Config(gpu_power_pct=20, gpu_max_clock=300))
    assert c.gpu_power_pct == 50 and c.gpu_max_clock == 900
    c = normalize(Config(gpu_power_pct=float("nan"), gpu_max_clock=0))
    assert c.gpu_power_pct == 70 and c.gpu_max_clock == 0
    assert Config().safe_dual is True


def test_status_applied():
    cfg = Config()
    assert pw.status(cfg, _cards())["applied"] is False
    assert pw.status(cfg, _cards(224, 245))["applied"] is True
    assert pw.status(cfg, _cards(200, 230))["applied"] is True                   # Afterburner daha düşük koymuş: olur
    na = {0: {**_cards()[0], "default": None}}
    assert pw.status(cfg, na) == {"supported": False, "applied": None, "cards": pw.status(cfg, na)["cards"]}


def test_restrict_two_cards_without_power_limit():
    """Kullanıcıda yalnız 3090 yükteyken bile bilgisayar yeniden başlıyordu: iki kart birlikte yalnız sınır varken."""
    cfg = Config()
    d, h, why = pw.restrict(["cuda:0", "cuda:1"], [], GI_FREE, cfg, _cards())
    assert d == ["cuda:1"] and h == [] and "Güç ayarını uygula" in why and "RTX 3090" in why
    assert pw.restrict(["cuda:0", "cuda:1"], [], GI_FREE, cfg, _cards(224, 245)) == (["cuda:0", "cuda:1"], [], "")
    # yardımcı modu (monitörlü 3080 yardımcı): yardımcı da sayılır
    d, h, why = pw.restrict(["cuda:1"], ["cuda:0"], GI_MON, cfg, _cards())
    assert d == ["cuda:1"] and h == [] and why
    # tek kart: dokunulmaz; sınır okunamıyor: dokunulmaz (kart sonsuza dek kapanmasın); kullanıcı kapattıysa dokunulmaz
    assert pw.restrict(["cuda:1"], [], GI_FREE, cfg, _cards())[2] == ""
    assert pw.restrict(["cuda:0", "cuda:1"], [], GI_FREE, cfg, {})[2] == ""
    assert pw.restrict(["cuda:0", "cuda:1"], [], GI_FREE, Config(safe_dual=False), _cards())[2] == ""
    # yalnız kullanılan kartlar sayılır: 3080 sınırsız ama kullanılmıyor
    assert pw.restrict(["cuda:1"], [], GI_FREE, cfg, _cards(320, 245))[2] == ""


def test_plan_order_and_uuid_check():
    c = _cards()
    p = pw.plan(c, 70, 1600)
    assert [k for k, _ in p] == ["pl@0", "pl@1", "lgc@0", "lgc@1"]             # önce güç, sonra saat
    assert p[0][1] == f"-i {U0} -pl 224" and p[3][1] == f"-i {U1} -lgc 210,1600"
    # saat sınırı yok ("Tam güç"): önceki 1600 MHz kilidi kaldırılır (1.9.21)
    assert [k for k, _ in pw.plan(c, 70, 0)] == ["pl@0", "pl@1", "rgc@0", "rgc@1"]
    bad = {0: {**c[0], "uuid": "GPU-1234 & calc.exe"}}
    assert pw.plan(bad, 70, 1600) == [] and pw.undo_plan(bad) == []            # komut satırına bir şey sızmasın
    assert [a for _, a in pw.undo_plan(c)] == [f"-i {U0} -pl 320", f"-i {U0} -rgc", f"-i {U1} -pl 350", f"-i {U1} -rgc"]


def test_task_xml_valid():
    smi = r"C:\Windows\System32\nvidia-smi.exe"
    x = pw.task_xml(smi, pw.plan(_cards(), 70, 1600), r"C:\Windows\System32\cmd.exe")
    root = ET.fromstring(x.replace('encoding="UTF-16"', ""))
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert root.find("t:Principals/t:Principal/t:UserId", ns).text == "S-1-5-18"
    assert root.find("t:Triggers/t:BootTrigger/t:Delay", ns).text == "PT1M"
    assert root.find("t:Triggers/t:LogonTrigger", ns) is not None
    ex = root.findall("t:Actions/t:Exec", ns)
    assert len(ex) == 1 and ex[0].find("t:Command", ns).text.endswith("cmd.exe")
    args = ex[0].find("t:Arguments", ns).text
    assert args.startswith('/d /s /c ""C:\\Windows\\System32\\nvidia-smi.exe" -i ') and args.count(" & ") == 3
    assert f"-i {U1} -lgc 210,1600" in args
    # standart kullanıcı görevi başlatabilir (iş sürerken sınır kalkınca izinsiz yeniden uygulama), uyku dönüşünde de çalışır
    assert root.find("t:RegistrationInfo/t:SecurityDescriptor", ns).text == pw.TASK_SDDL
    sub = root.find("t:Triggers/t:EventTrigger/t:Subscription", ns).text
    assert "Power-Troubleshooter" in sub and ET.fromstring(sub).tag == "QueryList"


def test_elevated_line_has_no_handle_redirect():
    """cmd'de '0>>' girdi tanıtıcısı sayılır: 'echo STT_OK pl:0>>dosya' sonucu yazmaz."""
    line = pw.elevated_line(r"C:\W\nvidia-smi.exe", pw.plan(_cards(), 70, 1600), r"C:\U\a b\r.txt",
                            [("task", '"C:\\W\\schtasks.exe" /Create /TN "STT Guvenli Guc" /XML "x.xml" /F')])
    assert not re.search(r"\d>", line.replace("2>&1", ""))
    for k in ("pl@0", "pl@1", "lgc@0", "lgc@1", "task"):
        assert f"&& echo STT_OK {k}|| echo STT_FAIL {k})" in line
    assert line.endswith(') >"C:\\U\\a b\\r.txt" 2>&1') and line.count("r.txt") == 1   # tek yönlendirme
    assert line.count("(") == line.count(")")


def test_params_short_enough_for_shellexecute(tmp_path, monkeypatch):
    """İnceleme: ShellExecuteEx ~2048 karakterden uzun satırı kesiyor; uzun kullanıcı adında bile altında kalsın."""
    res = _apply_env(tmp_path / ("K" * 40) / "AppData" / "Local", monkeypatch)
    seen = {}

    def runner(params):
        seen["p"] = params
        return 1223
    pw.apply(Config(), runner=runner, cards=_cards())
    assert seen["p"].startswith("/d /v:off /s /c ") and len(seen["p"]) < 1300, len(seen["p"])
    assert res.parent.is_dir()


def test_parse_result():
    r = pw.parse_result("Power limit for GPU set to 224 W\nSTT_OK pl@0\r\nSTT_FAIL lgc@0\nSTT_OK task\n")
    assert r["ok"] == ["pl@0", "task"] and r["fail"] == ["lgc@0"] and "224 W" in r["out"]


def _apply_env(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda n: r"C:\Windows\System32\nvidia-smi.exe")
    monkeypatch.setattr(pw, "safe_exe", lambda p: True)
    monkeypatch.setattr(pw.time, "sleep", lambda s: None)
    monkeypatch.setattr(pw, "query", lambda: _cards(224, 245))          # uygulamadan sonra okunan sınır
    return tmp_path / "stt" / "guvenli_guc_sonuc.txt"


def test_apply_success_and_partial(tmp_path, monkeypatch):
    res = _apply_env(tmp_path, monkeypatch)
    seen = {}

    def runner(params):
        seen["p"] = params
        res.write_text("STT_OK pl@0\nSTT_OK pl@1\nSTT_FAIL lgc@0\nSTT_FAIL lgc@1\nSTT_OK task\n")
        return 0
    r = pw.apply(Config(), runner=runner, cards=_cards())
    assert r["ok"] and "saat sınırı desteklenmedi" in r["warn"]
    assert seen["p"].startswith('/d /v:off /s /c "((') and "/XML" in seen["p"]
    xml = (tmp_path / "stt" / "guvenli_guc_gorev.xml").read_text(encoding="utf-16")
    assert "-pl 245" in xml and "S-1-5-18" in xml

    def bad(params):
        res.write_text("STT_FAIL pl@0\nSTT_OK pl@1\nSTT_OK task\n")
        return 1
    r = pw.apply(Config(), runner=bad, cards=_cards())
    assert not r["ok"] and "pl@0" in r["error"] and "Afterburner" in r["error"]

    def half(params):                                                    # yarıda kaldı (zaman aşımı)
        res.write_text("STT_OK pl@0\n")
        return 259
    r = pw.apply(Config(), runner=half, cards=_cards())
    assert not r["ok"] and "pl@1" in r["error"] and "görev" in r["error"]

    monkeypatch.setattr(pw, "query", lambda: _cards())                   # "başarılı" ama sınır değişmemiş
    r = pw.apply(Config(), runner=runner, cards=_cards())
    assert not r["ok"] and "değişmedi" in r["error"]


def test_apply_uac_declined_and_undo(tmp_path, monkeypatch):
    res = _apply_env(tmp_path, monkeypatch)
    r = pw.apply(Config(), runner=lambda p: 1223, cards=_cards())
    assert not r["ok"] and "izne" in r["error"]
    r = pw.apply(Config(), runner=lambda p: 0, cards=_cards())          # sonuç dosyası hiç yazılmadı
    assert not r["ok"] and "çalışmadı" in r["error"]
    seen = {}

    def runner(params):
        seen["p"] = params
        res.write_text("STT_OK pl@0\nSTT_OK rgc@0\nSTT_OK pl@1\nSTT_OK rgc@1\nSTT_FAIL task\n")
        return 0
    r = pw.apply(Config(), undo=True, runner=runner, cards=_cards())
    # görev hiç kaydedilmemişti (Afterburner sınırı koymuştu): silinemedi = sorun değil
    assert r["ok"] and "/Delete" in seen["p"] and "-rgc" in seen["p"] and "-pl 350" in seen["p"]


def test_apply_refuses_unsafe_paths(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: None)
    assert "nvidia-smi" in pw.apply(Config(), runner=lambda p: 0, cards=_cards())["error"]
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "a%b"))
    monkeypatch.setattr(shutil, "which", lambda n: r"C:\Windows\System32\nvidia-smi.exe")
    monkeypatch.setattr(pw, "safe_exe", lambda p: True)
    assert "desteklenmeyen" in pw.apply(Config(), runner=lambda p: 0, cards=_cards())["error"]


def test_safe_exe(monkeypatch):
    import os
    monkeypatch.setenv("SystemRoot", "/win")
    monkeypatch.setenv("ProgramFiles", "/pf")
    monkeypatch.setenv("ProgramW6432", "/pf")
    assert pw.safe_exe("/win/System32/nvidia-smi.exe") and pw.safe_exe("/pf/NVIDIA Corporation/NVSMI/nvidia-smi.exe")
    assert not pw.safe_exe("/home/u/nvidia-smi.exe") and not pw.safe_exe("/winx/nvidia-smi.exe") and not pw.safe_exe(None)
    assert os.sep  # (yol ayırıcısı sisteme göre)


def test_safe_layout_ready_and_missing():
    adapters_igpu = [("Intel(R) UHD Graphics 630", True), ("NVIDIA GeForce RTX 3080", False), ("NVIDIA GeForce RTX 3090", False)]
    lay = pw.safe_layout(Config(), gi=GI_FREE, cards=_cards(224, 245), adapters=adapters_igpu, tools=[])
    assert lay["ready"] and "RTX 3080 + RTX 3090" in lay["summary"]
    assert {i["key"]: i["ok"] for i in lay["items"]} == {"monitor": True, "power": True, "oc": True, "mode": True}
    lay = pw.safe_layout(Config(gpus="yardim"), gi=GI_MON, cards=_cards(),
                         adapters=[("Intel(R) UHD Graphics 630", False), ("NVIDIA GeForce RTX 3080", True)],
                         tools=["MSIAfterburner.exe"])
    ok = {i["key"]: i["ok"] for i in lay["items"]}
    assert not lay["ready"] and ok == {"monitor": False, "power": False, "oc": False, "mode": False}
    assert "monitör" in lay["summary"].lower() and "Reset" in [i for i in lay["items"] if i["key"] == "oc"][0]["text"]


def test_pipeline_start_uses_one_card_until_power_set(tmp_path, monkeypatch):
    import stt.engines as eng
    import stt.pipeline as pl
    made = []

    class FakePool:
        def __init__(self, devices, ecfg, *a, **k):
            made.append(list(devices))
            self.workers = {d: d for d in devices}
            self.load_lock = None

        def start(self):
            pass
    monkeypatch.setattr(eng, "gpu_info", lambda *a, **k: dict(GI_FREE))
    monkeypatch.setattr(eng, "resolve_model_path", lambda m: "m")
    monkeypatch.setattr(pl, "WorkerPool", FakePool)
    events = []
    for cards, want in ((_cards(), ["cuda:1"]), (_cards(224, 245), ["cuda:0", "cuda:1"])):
        monkeypatch.setattr(pw, "query", lambda c=cards: c)
        made.clear()
        events.clear()
        p = pl.Pipeline(Config(out_dir=tmp_path, gpu_flex=False))  # esnek düzen (1.9.21) test_v1921'de
        p.events = events.append
        p._memstat = lambda: None                                      # sayfa dosyası uyarısı (1.9.27) bu testin konusu değil
        p.start()
        assert p.devices == want and made[0] == want
        assert any(e["type"] == "notice" for e in events) == (len(want) == 1)
        p.prep.shutdown(wait=False)
        p.fin.shutdown(wait=False)


def test_safe_apply_switches_to_recommended(tmp_path, monkeypatch):
    import threading

    import stt.gui as gui
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path, gpus="yardim")
    app._pipe = None
    app._gpus = dict(GI_FREE)
    app._restart = threading.Event()
    saved = {}
    monkeypatch.setattr(gui, "save_settings", lambda u: saved.update(u))
    monkeypatch.setattr(gui, "load_config", lambda: Config(out_dir=tmp_path, gpus=saved.get("gpus", "yardim")))
    monkeypatch.setattr(gui.App, "get_settings", lambda self: {"gpus": self._cfg.gpus})
    monkeypatch.setattr(gui.App, "safe_status", lambda self: {"ok": True, "items": []})
    monkeypatch.setattr(pw, "apply", lambda cfg, undo=False: {"ok": True, "status": {}})
    r = app.safe_apply(False)
    assert r["ok"] and saved == {"gpus": "auto"} and app._restart.is_set() and r["settings"]["gpus"] == "auto"
    monkeypatch.setattr(pw, "apply", lambda cfg, undo=False: {"ok": False, "error": "x"})
    app._restart.clear()
    assert not app.safe_apply(False)["ok"] and not app._restart.is_set()


def test_restrict_or_reapply_runs_task_first(monkeypatch):
    """Hızlı başlatma / uyku / Afterburner sınırı silmiş olabilir: önce kayıtlı görev izinsiz çalıştırılır."""
    monkeypatch.setattr(pw.time, "sleep", lambda s: None)
    monkeypatch.setattr(pw, "query", lambda: _cards(224, 245))
    monkeypatch.setattr(pw, "reapply", lambda: True)
    d, h, why, cards = pw.restrict_or_reapply(["cuda:0", "cuda:1"], [], GI_FREE, Config(), _cards())
    assert d == ["cuda:0", "cuda:1"] and not why and cards[1]["limit"] == 245
    monkeypatch.setattr(pw, "reapply", lambda: False)                    # görev yok: tek kart
    d, h, why, cards = pw.restrict_or_reapply(["cuda:0", "cuda:1"], [], GI_FREE, Config(), _cards())
    assert d == ["cuda:1"] and why


def test_power_guard_during_work(tmp_path, monkeypatch):
    """İş sürerken sınır kalkarsa (uyku dönüşü / sürücü sıfırlanması): görev izinsiz yeniden çalışır, olmazsa tek kart."""
    import threading

    import stt.gui as gui
    from stt.pipeline import Pipeline

    class Pool:
        def __init__(self):
            self._spec = {"cuda:0": "cuda:0", "cuda:1": "cuda:1"}
            self.stopped = []

        def stop(self, w):
            self.stopped.append(w)
    pipe = Pipeline(Config(out_dir=tmp_path))
    pipe.pool = Pool()
    pipe.devices = ["cuda:0", "cuda:1"]
    pipe.inflight = {"cuda:0": ("t1", 0.0)}
    pipe.tasks = {"t1": {"id": "t1", "kind": "transcribe", "job": "j"}}
    app = gui.App.__new__(gui.App)
    app._cfg = Config(out_dir=tmp_path)
    app._pipe = pipe
    app._gpus = dict(GI_FREE)
    pushed = []
    app._push = pushed.append
    monkeypatch.setattr(gui.time, "sleep", lambda s: None)
    # sınır yerinde: hiçbir şey yapılmaz
    assert app._power_guard(_cards(224, 245)) == "" and not pipe._drop_wids
    # kalktı, görev yeniden uyguladı: tek karta inilmez
    monkeypatch.setattr(pw, "reapply", lambda: True)
    monkeypatch.setattr(pw, "query", lambda: _cards(224, 245))
    assert app._power_guard(_cards()) == "" and not pipe.fatal
    # kalktı, geri konamadı: 3080 işçisi (görevdeydi) kapatılır, görev sıraya döner, 3090 kalır
    monkeypatch.setattr(pw, "reapply", lambda: False)
    monkeypatch.setattr(pw, "query", lambda: _cards())
    assert app._power_guard(_cards())
    assert pipe.fatal == {"cuda:0"} and pipe.pool.stopped == ["cuda:0"] and pushed[-1]["type"] == "notice"
    assert pipe.cards_in_use() == (["cuda:1"], [])
    assert app._power_guard(_cards()) == ""                           # bir kez: artık tek kart
    pipe.prep.shutdown(wait=False)
    pipe.fin.shutdown(wait=False)
    del threading


def test_single_card_has_no_dual_warning():
    lay = pw.safe_layout(Config(), gi={0: {"name": "RTX 3090", "gb": 24.0, "display": True}}, cards=_cards(),
                         adapters=None, tools=[])
    assert not lay["ready"] and "Tek ekran kartı" in lay["summary"] and lay["n"] == 1


def test_igpu_without_intel_driver():
    """Kullanıcıda: BIOS'ta CPU Graphics açıldı, aygıt 'Microsoft Basic Display Adapter' (Intel sürücüsü yok); kablo
    anakarta takılınca Windows açılışta takılmış göründü. Uygulama sürücüyü kurmayı söylemeli, BIOS'u değil."""
    from stt.engines import igpu_advice
    ok, txt = igpu_advice([("Microsoft Basic Display Adapter", False), ("NVIDIA GeForce RTX 3080", True),
                           ("NVIDIA GeForce RTX 3090", False)], GI_MON)
    assert not ok and "sürücüsü kurulu değil" in txt and "BIOS" not in txt
