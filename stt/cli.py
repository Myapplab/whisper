"""Komut satırı.

  stt <link|dosya|klasör|liste.txt> ... [seçenekler]
  stt izle <klasör> [seçenekler]
  stt kontrol [--derin]
  stt olc [klasör] [--ornek] [--hedef 8]
  stt durum
  stt kanal <kanal/liste linki> [--konu basin|roportaj|hepsi] [--kelime "a,b"] [--haric "women"] [--sadece-liste]
  stt serve [--port 8765]          (yerel API: haber sitesi / haber radarı için)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .util import log, setup_logging

SUBCOMMANDS = {"izle": "watch", "watch": "watch", "kontrol": "doctor", "doctor": "doctor", "check": "doctor",
               "olc": "bench", "ölç": "bench", "bench": "bench", "durum": "status", "status": "status",
               "kanal": "channel", "channel": "channel", "liste": "channel", "serve": "serve", "api": "serve",
               "egit": "train", "eğit": "train", "train": "train"}


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("tanıma")
    g.add_argument("--lang", "--dil", dest="lang", help="en / tr / az / auto (varsayılan: otomatik)")
    g.add_argument("--engine", "--motor", dest="engine", choices=["whisper", "openai"])
    g.add_argument("--model", help="faster-whisper modeli ya da yerel klasör (varsayılan large-v3)")
    g.add_argument("--compute-type", dest="compute_type", help="float16 (GPU), int8 (CPU)...")
    g.add_argument("--beam-size", dest="beam_size", type=int)
    g.add_argument("--gpus", help="auto | 0,1 | 0 | cpu")
    g.add_argument("--batch", dest="batch_size", type=int, help="toplu boyut (varsayılan: karta göre; 0 = kapalı)")
    g.add_argument("--prompt", help="başlangıç istemi (isimler, terimler)")
    g.add_argument("--no-context", dest="no_context", action="store_true",
                   help="condition_on_previous_text=False (tekrar döngüsü olursa)")
    g.add_argument("--split-min", dest="split_min", type=float, help="bundan uzun dosyaları böl (dk, 8)")
    g.add_argument("--chunk-min", dest="chunk_min", type=float, help="hedef parça uzunluğu (dk, 5)")
    g.add_argument("--openai-model", dest="openai_model", help="varsayılan gpt-transcribe")
    g = p.add_argument_group("konuşmacı")
    g.add_argument("--diarize", "--konusmaci", dest="diarize", action="store_true", default=None,
                   help="pyannote ile konuşmacı ayır")
    g.add_argument("--hf-token", dest="hf_token")
    g.add_argument("--speakers", dest="speakers", type=int, help="konuşmacı sayısı biliniyorsa")
    g.add_argument("--mod", "--speaker-mode", dest="speaker_mode", choices=("auto", "basin", "herkes"),
                   help="auto (varsayılan) | basin: teknik direktör + muhabir | herkes: podcast/film, Konuşmacı 1..N")
    g.add_argument("--td-label", dest="td_label")
    g.add_argument("--canli", "--live", dest="live", action="store_true", help="canlı yayını anlık çevir")
    g.add_argument("--haber", "--news", dest="news", action="store_true",
                   help="yapay zekâ ile haber taslağı (stt.toml: llm_provider + anahtar)")
    g.add_argument("--reporter-label", dest="reporter_label")
    g = p.add_argument_group("indirme")
    g.add_argument("--cookies-browser", dest="cookies_browser", help="chrome | firefox | edge | none")
    g.add_argument("--cookies", dest="cookies_file", help="cookies.txt (Netscape biçimi)")
    g = p.add_argument_group("çıktı / durum")
    g.add_argument("-o", "--out", dest="out_dir", help="çıktı klasörü (varsayılan ./cikti)")
    g.add_argument("--force", "--yeniden", dest="force", action="store_true", default=None,
                   help="daha önce işlenmiş olsa da yeniden işle")
    g.add_argument("--keep-audio", dest="keep_audio", action="store_true", default=None, help="ara wav'ı silme")
    g.add_argument("-v", "--verbose", action="store_true")
    g = p.add_argument_group("FPL")
    g.add_argument("--fpl", nargs=2, metavar=("GW", "TAKIM"), help="ör. --fpl 11 ARS")
    g.add_argument("--fpl-repo", dest="fpl_repo")
    g.add_argument("--fpl-python", dest="fpl_python", help='presser.py için python komutu (varsayılan "python3")')


def _apply(cfg: Config, a: argparse.Namespace) -> Config:
    for name in ("lang", "engine", "model", "compute_type", "beam_size", "batch_size", "gpus", "prompt", "split_min",
                 "chunk_min", "openai_model", "diarize", "hf_token", "td_label", "reporter_label", "speaker_mode",
                 "cookies_browser", "cookies_file", "out_dir", "force", "keep_audio", "fpl_repo", "fpl_python"):
        v = getattr(a, name, None)
        if v is not None:
            setattr(cfg, name, Path(v) if name in ("out_dir", "fpl_repo") else v)
    if cfg.lang and cfg.lang.lower() in ("auto", "otomatik", "oto"):
        cfg.lang = None
    if cfg.cookies_browser and cfg.cookies_browser.lower() in ("none", "yok", "off"):
        cfg.cookies_browser = None
    if getattr(a, "no_context", False):
        cfg.condition_on_previous_text = False
    if getattr(a, "speakers", None):
        cfg.min_speakers = cfg.max_speakers = a.speakers
    if getattr(a, "fpl", None):
        gw, team = a.fpl
        if not gw.upper().lstrip("GW").isdigit():
            raise SystemExit(f"--fpl GW sayı olmalı: {gw}")
        cfg.fpl_gw, cfg.fpl_team = int(gw.upper().lstrip("GW")), team.upper()
        if not cfg.prompt:
            from .fpl import fpl_prompt
            cfg.prompt = fpl_prompt(cfg.fpl_team)
    return cfg


def _run_pipeline(cfg: Config, items=None, watch_dir: Path | None = None) -> int:
    from .pipeline import Pipeline

    if cfg.diarize and not cfg.hf_token:
        log.warning("Konuşmacı ayırma için HF token yok (--hf-token, HF_TOKEN ya da stt.toml); "
                    "konuşmacı ayırma OLMADAN devam ediliyor.")
        cfg.diarize = False
    if cfg.fpl_gw is not None and not cfg.fpl_repo:
        log.error("--fpl için FPL repo yolu gerekli: --fpl-repo, FPL_REPO ya da stt.toml fpl_repo")
        return 2
    if cfg.fpl_gw is not None and not (Path(cfg.fpl_repo) / "presser.py").is_file():
        log.error("FPL reposunda presser.py yok: %s", cfg.fpl_repo)
        return 2
    if items and any(it.extra.get("fpl") for it in items):
        if not cfg.fpl_repo or not (Path(cfg.fpl_repo) / "presser.py").is_file():
            log.warning("Listede FPL bilgisi var ama FPL repo (presser.py) bulunamadı; FPL adımı atlanacak.")
            for it in items:
                it.extra.pop("fpl", None)
    if cfg.fpl_gw is not None and items is not None and len(items) > 1:
        log.error("--fpl tek bir girdiyle kullanılır (%d öğe verildi).", len(items))
        return 2
    pipe = Pipeline(cfg)
    try:
        pipe.start()
        if watch_dir:
            from .watch import watch
            watch(watch_dir, pipe, cfg.watch_interval)
        else:
            for it in items:
                pipe.submit(it)
            pipe.wait_all()
    except KeyboardInterrupt:
        log.warning("Durduruldu. Biten parçalar kaydedildi; aynı komutla kaldığı yerden devam eder.")
        return 130
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    finally:
        pipe.close()
    done = [j for j in pipe.results if j.status == "done"]
    skipped = [j for j in pipe.results if j.status in ("skipped", "drm", "skipped_noaudio")]
    failed = [j for j in pipe.results if j.status == "failed"]
    print()
    print(f"Özet: {len(done)} bitti, {len(skipped)} atlandı, {len(failed)} hata. Çıktı: {pipe.out_root}")
    for j in done:
        doc = j.result or {}
        if doc.get("speakers"):
            n = len(doc["speakers"])
            note = (f"konuşmacı ayrıldı: {n} kişi (herkes ayrı)" if doc.get("speaker_mode") == "herkes"
                    else f"konuşmacı ayrıldı: {doc.get('td_label')} + {n - 1} muhabir sesi")
        elif cfg.diarize:
            note = f"KONUŞMACI AYRILAMADI: {doc.get('diarization_error') or j.diar_error or '?'}"
        else:
            note = "konuşmacı ayırma kapalı"
        print(f"  ✓ {j.out_dir}\n      {note}")
    for j in skipped:
        print(f"  - {j.item.source}  ({j.status}{': ' + j.error if j.error else ''})")
    for j in failed:
        print(f"  ✗ {j.item.source}: {j.error}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    from .util import prepare_env
    prepare_env()
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config()
    sub = SUBCOMMANDS.get(argv[0].lower()) if argv else None

    if sub == "doctor":
        p = argparse.ArgumentParser(prog="stt kontrol", description="Kurulumu kontrol et")
        p.add_argument("--derin", "--deep", action="store_true", help="modeli her GPU'da yükleyip dene")
        _common(p)
        a = p.parse_args(argv[1:])
        setup_logging(a.verbose)
        from .doctor import run_doctor
        return run_doctor(_apply(cfg, a), deep=a.derin)

    if sub == "bench":
        p = argparse.ArgumentParser(prog="stt olc", description="WER ölçümü (ses + referans .txt)")
        p.add_argument("klasor", nargs="?", help="ses dosyaları + aynı adlı .txt (ya da .ref.txt) referanslar")
        p.add_argument("--ornek", action="store_true", help="3 İngilizce örnek kaydı indirip ölç")
        p.add_argument("--hedef", type=float, default=8.0, help="hedef WER yüzdesi (8)")
        _common(p)
        a = p.parse_args(argv[1:])
        setup_logging(a.verbose)
        from .bench import run_bench
        return run_bench(_apply(cfg, a), Path(a.klasor) if a.klasor else None, a.ornek, a.hedef)

    if sub == "train":
        p = argparse.ArgumentParser(prog="stt egit", description="Konuşmacı ayırmayı düzeltilmiş videolarla eğit")
        p.add_argument("--kart", help="cuda:1 gibi (varsayılan: monitörsüz en büyük kart)")
        p.add_argument("-o", "--out", dest="out_dir")
        a = p.parse_args(argv[1:])
        setup_logging()
        if a.out_dir:
            cfg.out_dir = Path(a.out_dir)
        import json as _json

        from .tune import pick_device, run_tune
        from .workers import isolate_gpu
        dev = a.kart or pick_device()
        if not dev:
            print("HATA: Uygun ekran kartı yok (eğitim işlemcide yapılmaz)", flush=True)
            return 1
        try:
            res = run_tune(cfg, isolate_gpu(dev), progress=lambda t: print("ILERLEME: " + t, flush=True))
        except Exception as e:  # noqa: BLE001
            log.exception("Eğitim yapılamadı")
            print(f"HATA: {e}", flush=True)
            return 1
        print("SONUC: " + _json.dumps(res, ensure_ascii=False), flush=True)
        return 0

    if sub == "status":
        p = argparse.ArgumentParser(prog="stt durum")
        p.add_argument("-o", "--out", dest="out_dir")
        a = p.parse_args(argv[1:])
        setup_logging()
        if a.out_dir:
            cfg.out_dir = Path(a.out_dir)
        from .state import State
        rows = State(Path(cfg.out_dir) / ".stt").all()
        for r in rows:
            print(f"{r['status']:<16} {r['title'][:60]:<60} {r['out_dir'] or r['error'][:80]}")
        print(f"{len(rows)} kayıt")
        return 0

    if sub == "channel":
        p = argparse.ArgumentParser(prog="stt kanal", description="Kanal / oynatma listesindeki videoları konuya göre işle")
        p.add_argument("link")
        p.add_argument("--konu", default="basin", help="basin (basın toplantısı) | roportaj | hepsi")
        p.add_argument("--kelime", help="ek anahtar kelimeler, virgülle")
        p.add_argument("--haric", help="başlıkta geçerse alma (ör. women,u21)")
        p.add_argument("--en-fazla", dest="limit", type=int, default=None, help="en fazla kaç video listelensin")
        p.add_argument("--sadece-liste", dest="only_list", action="store_true", help="yalnızca listele, işleme")
        _common(p)
        a = p.parse_args(argv[1:])
        setup_logging(a.verbose)
        cfg = _apply(cfg, a)
        from .channel import list_videos, select
        from .sources import Item
        data = list_videos(a.link, cfg, limit=a.limit)
        items = select(data["items"], a.konu, a.kelime, a.haric)
        hit = [e for e in items if e["match"]]
        for e in items:
            print(f"{'✓' if e['match'] else ' '} {e.get('date') or '          '}  {e['title'][:90]}")
        print(f"{len(hit)}/{len(items)} video konuya uyuyor ({data.get('channel') or data.get('title')})")
        if a.only_list or not hit:
            return 0
        from .channel import collection_name
        name = collection_name(data.get("channel") or data.get("title") or "kanal", a.konu, a.kelime)
        its = []
        for e in hit:
            it = Item("url", e["url"], title=e["title"])
            it.extra["collection"] = name
            if a.news:
                it.extra["llm_news"] = True
            its.append(it)
        return _run_pipeline(cfg, its)

    if sub == "serve":
        p = argparse.ArgumentParser(prog="stt serve", description="Yerel API (http://127.0.0.1:PORT/api/status)")
        p.add_argument("--port", type=int, default=None)
        _common(p)
        a = p.parse_args(argv[1:])
        setup_logging(a.verbose)
        cfg = _apply(cfg, a)
        cfg.api_port = a.port or cfg.api_port or 8765
        import threading
        import time

        from .gui import App
        app = App()
        app._cfg = cfg
        app._start_api()
        threading.Thread(target=app._runner, name="motor", daemon=True).start()
        threading.Thread(target=app._follow_loop, name="takip", daemon=True).start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            from .gui import _shutdown
            _shutdown(app)
        return 0

    if sub == "watch":
        p = argparse.ArgumentParser(prog="stt izle", description="Klasörü izle, yeni dosyaları işle")
        p.add_argument("klasor")
        p.add_argument("--aralik", type=float, default=None, help="tarama aralığı (sn)")
        _common(p)
        a = p.parse_args(argv[1:])
        setup_logging(a.verbose)
        cfg = _apply(cfg, a)
        if a.aralik:
            cfg.watch_interval = a.aralik
        d = Path(a.klasor)
        if not d.is_dir():
            log.error("Klasör yok: %s", d)
            return 2
        return _run_pipeline(cfg, watch_dir=d)

    p = argparse.ArgumentParser(
        prog="stt", description=f"stt {__version__} - link/video/ses -> metin (faster-whisper, çoklu GPU)",
        epilog="Alt komutlar: stt izle KLASOR | stt kontrol | stt olc [KLASOR] | stt durum")
    p.add_argument("girdiler", nargs="*", help="link, dosya, klasör ya da .txt link listesi")
    p.add_argument("--izle", "--watch", dest="watch", help="izleme klasörü")
    _common(p)
    a = p.parse_args(argv)
    setup_logging(a.verbose)
    cfg = _apply(cfg, a)
    if a.watch:
        return _run_pipeline(cfg, watch_dir=Path(a.watch))
    if not a.girdiler:
        p.print_help()
        return 2
    from .sources import expand_inputs
    items = expand_inputs(a.girdiler)
    if not items:
        log.error("İşlenecek bir şey bulunamadı.")
        return 2
    for it in items:
        if a.live:
            it.extra["live"] = True
        if a.news:
            it.extra["llm_news"] = True
    log.info("%d öğe işlenecek", len(items))
    return _run_pipeline(cfg, items)
