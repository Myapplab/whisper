"""İzleme klasörü: içine düşen yeni medya dosyaları / .txt link listeleri otomatik işlenir."""
from __future__ import annotations

import time
from pathlib import Path

from .sources import expand_inputs
from .util import MEDIA_EXTS, PARTIAL_EXTS, log


def watch(folder: Path, pipe, interval: float = 5.0, stable_polls: int = 2) -> None:
    folder = folder.resolve()
    out_root = Path(pipe.out_root).resolve()
    seen: dict[Path, tuple] = {}
    stable: dict[Path, int] = {}
    handled: set[tuple] = set()
    log.info("İzleniyor: %s (her %.0f sn). Durdurmak için Ctrl+C.", folder, interval)
    while True:
        try:
            for p in folder.rglob("*"):
                try:
                    if not p.is_file():
                        continue
                except OSError:
                    continue
                rp = p.resolve()
                if out_root in rp.parents or any(x.startswith(".") for x in rp.relative_to(folder).parts):
                    continue
                ext = p.suffix.lower()
                if ext in PARTIAL_EXTS or (ext not in MEDIA_EXTS and ext != ".txt"):
                    continue
                try:
                    st = p.stat()
                except OSError:  # bu arada silindi / taşındı
                    continue
                sig = (st.st_size, st.st_mtime_ns)
                if seen.get(rp) == sig:
                    stable[rp] = stable.get(rp, 0) + 1
                else:
                    seen[rp], stable[rp] = sig, 0
                    continue
                if stable[rp] >= stable_polls - 1 and (rp, sig) not in handled:
                    try:  # dosya hâlâ başka bir program tarafından yazılıyor mu?
                        with open(rp, "rb"):
                            pass
                    except OSError:
                        continue
                    handled.add((rp, sig))
                    log.info("Yeni dosya: %s", rp.name)
                    for it in expand_inputs([str(rp)]):
                        pipe.submit(it)
            end = time.time() + interval
            while time.time() < end:
                pipe.pump(0.5)
        except KeyboardInterrupt:
            raise
