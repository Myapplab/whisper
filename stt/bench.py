"""Kelime hata oranı (WER) ölçümü: klasördeki ses + referans metinlerle (jiwer)."""
from __future__ import annotations

import copy
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from .config import TOOL_DIR
from .sources import Item, iter_media_files
from .util import fmt_clock, log

SAMPLE_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
              "sherpa-onnx-whisper-tiny.en.tar.bz2")


def download_samples(dest: Path) -> Path:
    """3 İngilizce LibriSpeech kaydı + referans (GitHub, ~118 MB arşivden yalnızca wav'lar çıkarılır)."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    if len(list(dest.glob("*.wav"))) >= 3 and len(list(dest.glob("*.txt"))) >= 3:
        return dest
    log.info("Örnek kayıtlar indiriliyor: %s", SAMPLE_URL)
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "a.tar.bz2"
        with requests.get(SAMPLE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(arc, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
        with tarfile.open(arc, "r:bz2") as t:
            for m in t.getmembers():
                if "/test_wavs/" in m.name and m.isfile():
                    name = Path(m.name).name
                    data = t.extractfile(m).read()  # type: ignore[union-attr]
                    if name == "trans.txt":
                        for line in data.decode().splitlines():
                            if line.strip():
                                fn, txt = line.split(maxsplit=1)
                                (dest / f"{Path(fn).stem}.txt").write_text(txt.strip() + "\n", encoding="utf-8")
                    elif name.endswith(".wav"):
                        (dest / name).write_bytes(data)
    return dest


def find_pairs(folder: Path) -> list[tuple[Path, str]]:
    mapping: dict[str, str] = {}
    for mf in ("trans.txt", "referans.tsv", "refs.tsv"):
        p = folder / mf
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    parts = re.split(r"\t|\s+", line.strip(), maxsplit=1)
                    if len(parts) == 2:
                        mapping[Path(parts[0]).stem] = parts[1]
    pairs = []
    for audio in iter_media_files(folder):
        ref = None
        for cand in (audio.with_suffix(".ref.txt"), audio.with_name(audio.stem + ".referans.txt"),
                     audio.with_suffix(".txt")):
            if cand.exists():
                ref = cand.read_text(encoding="utf-8-sig")
                break
        if ref is None:
            ref = mapping.get(audio.stem)
        if ref is None:
            log.warning("Referans yok, atlandı: %s", audio.name)
            continue
        pairs.append((audio, ref))
    return pairs


def normalizer(lang: str | None):
    try:
        if lang in (None, "en"):
            from whisper_normalizer.english import EnglishTextNormalizer
            return EnglishTextNormalizer()
        from whisper_normalizer.basic import BasicTextNormalizer
        basic = BasicTextNormalizer()
        if lang in ("tr", "az"):
            return lambda s: basic(s.replace("I", "ı").replace("İ", "i").lower())
        return basic
    except Exception:  # noqa: BLE001
        return lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.lower())).strip()


def run_bench(cfg, folder: Path | None, sample: bool, target: float) -> int:
    import jiwer

    from .pipeline import Pipeline

    if sample or folder is None:
        folder = download_samples(TOOL_DIR / "ornekler" / "librispeech")
    pairs = find_pairs(folder)
    if not pairs:
        log.error("Klasörde ses + referans çifti yok: %s", folder)
        return 2
    lang = cfg.lang or "en"
    c = copy.copy(cfg)
    c.out_dir = Path(tempfile.mkdtemp(prefix="stt_olcum_"))
    c.force, c.diarize, c.fpl_gw, c.fpl_team, c.lang = True, False, None, None, cfg.lang or "en"
    pipe = Pipeline(c)
    pipe.start()
    jobs = {}
    try:
        for audio, _ in pairs:
            it = Item("file", str(audio.resolve()), title=audio.stem)
            it.compute_key()
            jobs[it.key] = audio
            pipe.submit(it)
        pipe.wait_all()
    finally:
        pipe.close()
    norm = normalizer(lang)
    rows, refs, hyps = [], [], []
    by_audio = {j.item.source: j for j in pipe.jobs.values()}
    for audio, ref in pairs:
        job = by_audio.get(str(audio.resolve()))
        if not job or not job.result:
            rows.append({"dosya": audio.name, "hata": job.error if job else "işlenmedi"})
            continue
        hyp = " ".join(s["text"] for s in job.result["segments"])
        r, h = norm(ref), norm(hyp)
        if not r.strip():
            continue
        refs.append(r)
        hyps.append(h)
        o = jiwer.process_words(r, h)
        rows.append({"dosya": audio.name, "sure": job.result["duration_s"], "wer": o.wer, "sub": o.substitutions,
                     "del": o.deletions, "ins": o.insertions, "ref_kelime": len(r.split()), "ref": r, "hyp": h})
    total = jiwer.wer(refs, hyps) if refs else 1.0
    lines = [f"# WER ölçümü ({cfg.engine} / {cfg.model if cfg.engine == 'whisper' else cfg.openai_model}, dil={lang})",
             "", "| Dosya | Süre | Kelime | WER | S/D/I |", "|---|---|---|---|---|"]
    for r in rows:
        if "wer" in r:
            lines.append(f"| {r['dosya']} | {fmt_clock(r['sure'])} | {r['ref_kelime']} | %{100 * r['wer']:.2f} | "
                         f"{r['sub']}/{r['del']}/{r['ins']} |")
        else:
            lines.append(f"| {r['dosya']} | - | - | HATA: {r['hata']} | |")
    ok = total < target / 100
    lines += ["", f"**Toplam WER: %{100 * total:.2f}** (hedef < %{target:g}) -> {'GEÇTİ' if ok else 'KALDI'}", ""]
    for r in rows:
        if "wer" in r and r["wer"] > 0:
            lines += [f"### {r['dosya']}", "", f"- REF: {r['ref']}", f"- HYP: {r['hyp']}", ""]
    report = "\n".join(lines)
    print("\n" + report)
    (folder / "olcum.md").write_text(report, encoding="utf-8")
    (folder / "olcum.json").write_text(json.dumps({"wer": total, "target": target / 100, "rows": rows},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Rapor: %s", folder / "olcum.md")
    shutil.rmtree(c.out_dir, ignore_errors=True)
    return 0 if ok else 1
