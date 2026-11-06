"""Konuşmacı ayırmayı kullanıcının düzelttiği videolarla "eğitme" (ayar arama).

Kullanıcı okuyucudaki "Konuşmacılar" ekranında bir videonun konuşmacılarını düzeltip kaydeder
(``<out>/.stt/duzeltme/<key>.json`` = doğru cevap). Bu modül o videoların sesini pyannote'tan bir kez
geçirir (bölütleme + ses imzaları önbellekte kalır), sonra kümeleme ayarlarının (kaç kişi var kararı) birçok
değerini dener ve doğru cevaba en yakın sonucu veren ayarı ``<out>/.stt/konusmaci_ayar.json``'a yazar.
Sonraki her konuşmacı ayırma bu ayarla yapılır (model başına ayrı).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from .util import log

TUNED_FILE = "konusmaci_ayar.json"
SR = 16000


# ---------------------------------------------------------------- ölçüm
def _frames(turns, n: int, step: float) -> list[set]:
    out: list[set] = [set() for _ in range(n)]
    for a, b, lab in turns:
        for i in range(max(0, int(a / step)), min(n, int(b / step) + 1)):
            out[i].add(str(lab))
    return out


def speaker_error(ref, hyp, step: float = 0.1, collar: float = 0.5) -> tuple[float, float]:
    """Konuşmacı hatası: doğru cevapta konuşma olan sürenin ne kadarında program yanlış kişiyi (ya da kimseyi)
    vermiş. Kişiler en iyi eşleşmeyle eşlenir (programın "SPEAKER_01"i = kullanıcının "Ahmet"i).
    Konuşmacı değişimlerinin ``collar`` sn yakını sayılmaz (doğru cevabın zamanı da satır düzeyinde kaba).
    Döner: (hatalı sn, toplam sn)."""
    end = max([t[1] for t in ref] + [t[1] for t in hyp] + [0.0])
    n = int(end / step) + 2
    rf, hf = _frames(ref, n, step), _frames(hyp, n, step)
    skip = set()
    c = int(collar / step)
    for i in range(1, n):
        if rf[i] != rf[i - 1]:
            skip.update(range(max(0, i - c), min(n, i + c + 1)))
    over: dict = {}
    for i in range(n):
        if i in skip or len(rf[i]) != 1:
            continue
        r = next(iter(rf[i]))
        for h in hf[i]:
            over[(r, h)] = over.get((r, h), 0) + 1
    mapping, used_r, used_h = {}, set(), set()
    for (r, h), _ in sorted(over.items(), key=lambda kv: -kv[1]):  # açgözlü en iyi eşleşme
        if r not in used_r and h not in used_h:
            mapping[h] = r
            used_r.add(r)
            used_h.add(h)
    bad = total = 0
    for i in range(n):
        if i in skip or len(rf[i]) != 1:
            continue
        total += 1
        r = next(iter(rf[i]))
        if r not in {mapping.get(h) for h in hf[i]}:
            bad += 1
    return bad * step, total * step


def measure(state_dir: Path) -> dict:
    """Programın son çıktısı (ayırma + kelime eşleme) düzeltilmiş videolarda ne kadar doğru? Yeniden çalıştırmaz:
    düzeltmeden önce saklanan çıktı (``hyp``) ile düzeltilmiş hâl karşılaştırılır. Döner: {videos, minutes, error, per}."""
    per, bad, tot = [], 0.0, 0.0
    groups: dict = {}
    for p in sorted((Path(state_dir) / "duzeltme").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not d.get("hyp") or not d.get("turns"):
            continue
        b, t = speaker_error(d["turns"], d["hyp"])
        if t <= 0:
            continue
        setup = setup_label(d.get("diar"))
        per.append({"title": (d.get("title") or d.get("key") or "")[:70], "error": round(100 * b / t, 1),
                    "minutes": round(t / 60, 1), "setup": setup})
        g = groups.setdefault(setup, [0.0, 0.0, 0])
        g[0], g[1], g[2] = g[0] + b, g[1] + t, g[2] + 1
        bad, tot = bad + b, tot + t
    return {"videos": len(per), "minutes": round(tot / 60, 1), "error": round(100 * bad / tot, 1) if tot else None,
            "per": per,
            "groups": {k: {"videos": n, "minutes": round(t / 60, 1), "error": round(100 * b / t, 1)}
                       for k, (b, t, n) in groups.items()}}


def setup_label(diar: dict | None) -> str:
    """Ayırma ayarının kısa adı (A/B karşılaştırması): 'speaker-diarization-community-1 · ses yükseltme kapalı'."""
    diar = diar or {}
    if not diar:
        return "ayar bilinmiyor (1.9.4 öncesi)"
    lab = str(diar.get("model") or "?").split("/")[-1]
    if "agc" in diar:
        lab += " · ses yükseltme " + ("açık" if diar["agc"] else "kapalı")
    return lab


# ---------------------------------------------------------------- ayar dosyası
def load_tuned(state_dir: Path) -> dict:
    """{model: {"params": {"clustering.threshold": .., ...}, ...}}"""
    p = Path(state_dir) / TUNED_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def save_tuned(state_dir: Path, model: str, params: dict, info: dict) -> Path:
    data = load_tuned(state_dir)
    data[model] = {"params": params, **info}
    p = Path(state_dir) / TUNED_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def flat_params(nested: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (nested or {}).items():
        if isinstance(v, dict):
            out.update(flat_params(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def apply_params(pipe, flat: dict) -> dict:
    """Düz {"clustering.threshold": 0.6} ayarlarını pyannote pipeline'ına uygula (bilinmeyenler atlanır)."""
    cur = pipe.parameters(instantiated=True)
    done = {}
    for key, val in (flat or {}).items():
        node, parts = cur, key.split(".")
        for p in parts[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and parts[-1] in node:
            node[parts[-1]] = type(node[parts[-1]])(val) if isinstance(node[parts[-1]], (int, float)) else val
            done[key] = node[parts[-1]]
    if done:
        pipe.instantiate(cur)
    return done


def candidates(base: dict) -> list[dict]:
    """Denenecek ayarlar. community-1 (VBx: threshold, Fa) ve 3.1 (AHC: threshold) için."""
    th = base.get("clustering.threshold")
    grid: dict = {}
    if th is not None:
        th = float(th)
        lo, hi = (0.4, 0.9) if "clustering.Fa" in base else (max(0.3, th - 0.25), min(1.2, th + 0.25))
        vals = sorted({round(th, 3)} | {round(lo + i * 0.05, 3) for i in range(int((hi - lo) / 0.05) + 1)})
        grid["clustering.threshold"] = vals
    if "clustering.Fa" in base:
        grid["clustering.Fa"] = sorted({float(base["clustering.Fa"]), 0.03, 0.15})
    if "segmentation.min_duration_off" in base:
        grid["segmentation.min_duration_off"] = sorted({float(base["segmentation.min_duration_off"]), 0.0, 0.5})
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


# ---------------------------------------------------------------- eğitim
def references(state_dir: Path) -> list[dict]:
    """Düzeltilmiş (en az 2 kişili) videolar."""
    out = []
    for p in sorted((Path(state_dir) / "duzeltme").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if len({t[2] for t in d.get("turns") or []}) >= 2 and d.get("source"):
            out.append(d)
    return out


def _audio_for(ref: dict, state_dir: Path, cfg) -> Path:
    """Videonun sesi: ``.stt/egitim/<key>/ses.wav`` (yoksa kaynaktan bir kez yeniden indirilir / çevrilir)."""
    from .fetch import prepare_audio
    from .sources import Item
    src = str(ref["source"])
    kind = "url" if src.lower().startswith(("http://", "https://")) else "file"
    if kind == "file" and not Path(src).exists():
        raise RuntimeError(f"Kaynak dosya bulunamadı: {src}")
    item = Item(kind, src, title=ref.get("title") or "", key=ref.get("key") or "")
    wav, _ = prepare_audio(item, Path(state_dir) / "egitim" / (ref.get("key") or "x")[:18], cfg)
    return wav


def _hyp(out) -> list[list]:
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    return [[seg.start, seg.end, str(spk)] for seg, _, spk in ann.itertracks(yield_label=True)]


def pick_device() -> str | None:
    """Monitör bağlı olmayan en büyük ekran kartı (monitörlü karta kendiliğinden yük verilmez)."""
    from .engines import gpu_info
    gi = gpu_info()
    free = sorted((i for i, g in gi.items() if g.get("display") is not True), key=lambda i: -gi[i].get("gb", 0))
    return f"cuda:{free[0]}" if free else None


def run_tune(cfg, device: str = "cuda:0", progress=lambda text: None) -> dict:
    """Düzeltilmiş videolarla en iyi kümeleme ayarını bul ve kaydet. Döner: özet (GUI / CLI için)."""
    from .audio import load_wav
    from .engines import Diarizer
    state_dir = Path(cfg.out_dir).resolve() / ".stt"
    refs = references(state_dir)
    if not refs:
        raise RuntimeError("Düzeltilmiş video yok. Önce bir videonun metnini açıp 'Konuşmacılar' ekranında "
                           "konuşmacıları düzeltin ve kaydedin.")
    t0 = time.time()
    progress(f"Konuşmacı modeli yükleniyor ({len(refs)} düzeltilmiş video)")
    from .config import diar_agc_on
    d = Diarizer(cfg.diar_model, cfg.hf_token, device, agc=diar_agc_on(cfg), batch=cfg.diar_batch,
                 params=load_tuned(state_dir), cpu_ok=False)
    files = []
    for i, ref in enumerate(refs, 1):
        progress(f"Ses hazırlanıyor {i}/{len(refs)}: {(ref.get('title') or '')[:60]}")
        audio = load_wav(str(_audio_for(ref, state_dir, cfg)))
        mode = ref.get("speaker_mode") or "herkes"
        music = (cfg.diar_music or "auto").lower()
        if music == "on" or (music == "auto" and mode != "basin"):
            audio = d._vocals(audio)
        if d.agc:
            from .audio import agc
            agc(audio)
        files.append(({"waveform": d.torch.from_numpy(audio)[None], "sample_rate": SR, "uri": str(i)}, ref))
    pipe = d.pipe
    base = {k: v for k, v in flat_params(pipe.parameters(instantiated=True)).items()
            if k in ("clustering.threshold", "clustering.Fa", "segmentation.min_duration_off")}

    def score(params: dict) -> tuple[float, float]:
        apply_params(pipe, params)
        bad = tot = 0.0
        for f, ref in files:
            # birleştirilmemiş satırlar (1.9.4+ düzeltmeleri); eski kayıtta yalnız birleşik turlar var
            b, t = speaker_error(ref.get("segs") or ref["turns"], _hyp(pipe(f)))
            bad, tot = bad + b, tot + t
        return bad, max(tot, 1e-9)

    pipe.training = True  # bölütleme + ses imzaları ilk denemeden sonra önbellekte: yalnız kümeleme tekrarlanır
    try:
        progress("İlk ölçüm (şu anki ayar); ses imzaları çıkarılıyor")
        b0, tot = score(base)
        best, best_err = dict(base), b0 / tot
        cands = candidates(base)
        for i, c in enumerate(cands, 1):
            b, t = score(c)
            if b / t < best_err - 1e-6:
                best, best_err = dict(c), b / t
            if i % 5 == 0 or i == len(cands):
                progress(f"Ayar deneniyor {i}/{len(cands)} · şu an %{b0 / tot * 100:.0f} → en iyi %{best_err * 100:.0f}")
    finally:
        pipe.training = False
    before = b0 / tot
    res = {"model": d.model_used, "before": round(before * 100, 1), "after": round(best_err * 100, 1),
           "videos": len(refs), "minutes": round(tot / 60, 1), "params": best, "elapsed_s": round(time.time() - t0)}
    if best_err < before - 0.005:
        save_tuned(state_dir, d.model_used, best, {k: res[k] for k in ("before", "after", "videos", "minutes")})
        res["saved"] = True
        log.info("Konuşmacı ayarı eğitildi (%s): hata %%%.0f → %%%.0f, %s", d.model_used, before * 100,
                 best_err * 100, best)
    else:
        res["saved"] = False
        log.info("Konuşmacı ayarı: şu anki ayar zaten en iyisi (hata %%%.0f)", before * 100)
    return res
