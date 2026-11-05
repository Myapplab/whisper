"""Canlı yayını doğrudan metne çevirme (YouTube / Twitch / Kick / Dailymotion / ok.ru canlı, m3u8 adresi...).

Yöntem: ffmpeg yayını 16 kHz mono PCM olarak okur; ~20 sn'lik parçalar sessiz bir anda kesilip öncelikli olarak
GPU işçilerine gider; her parça bitince metin sırasıyla 'canlı' dosyasına eklenir ve arayüzde görünür.
Yayın kesilirse yeniden bağlanılır. Yayın bitince (ya da 'Canlıyı bitir'e basılınca) bütün kayıt tek dosyada
birleşir; istenirse konuşmacı ayırma yapılır ve normal çıktılar (txt/srt/json/özet) yazılır.
"""
from __future__ import annotations

import os
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

from .audio import SR
from .util import log

BYTES_PER_S = SR * 2


def resolve_stream(url: str, cfg) -> tuple[str, dict, dict]:
    """Canlı yayının ffmpeg'in okuyabileceği adresi: (adres, başlıklar, bilgi). Dosya / doğrudan m3u8 da olur."""
    if os.path.exists(url) or url.lower().split("?")[0].endswith((".m3u8", ".mpd")):
        from urllib.parse import urlsplit
        title = Path(url).stem if os.path.exists(url) else f"Canlı yayın {urlsplit(url).hostname or ''}".strip()
        return url, {}, {"title": title, "webpage_url": url}
    from .fetch import ytdlp_info
    info = ytdlp_info(url, cfg.cookies_browser, extra={"format": "bestaudio/best", "noplaylist": True})
    if info.get("entries"):
        info = next((e for e in info["entries"] if e), {})
    fmts = info.get("requested_formats") or []
    stream = info.get("url") or (fmts[-1].get("url") if fmts else None)
    if not stream:
        raise RuntimeError("Canlı yayın adresi bulunamadı")
    headers = dict(info.get("http_headers") or (fmts[-1].get("http_headers") if fmts else {}) or {})
    return stream, headers, info


def cut_point(buf: np.ndarray, lo: int, hi: int, win: int = SR // 5) -> int:
    """[lo, hi) arasında en sessiz 0.2 sn'lik pencerenin ortası (cümle ortasından kesmemek için)."""
    hi = min(hi, len(buf))
    if hi - lo <= win:
        return hi
    seg = buf[lo:hi].astype(np.float32) / 32768.0
    n = (len(seg) // win) * win
    e = (seg[:n].reshape(-1, win) ** 2).mean(axis=1)
    return lo + int(np.argmin(e)) * win + win // 2


def write_wav(path: Path, pcm: np.ndarray) -> None:
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype(np.int16).tobytes())
    os.replace(tmp, path)


def raw_to_wav(raw: Path, dst: Path) -> None:
    """Ham PCM'i parça parça WAV'a kopyala (6 saatlik kayıt da belleğe sığsın diye)."""
    tmp = dst.with_suffix(".part")
    with wave.open(str(tmp), "wb") as w, open(raw, "rb") as f:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            w.writeframes(b[: len(b) - len(b) % 2])
    os.replace(tmp, dst)


class LiveRecorder(threading.Thread):
    """Bir canlı yayını okuyup parçalar üretir. on_chunk(idx, wav, offset, dur), on_end(wav_tam, sebep)."""

    def __init__(self, source: str, workdir: Path, cfg, on_chunk, on_end, readrate: float | None = None,
                 on_info=None):
        super().__init__(name="canli", daemon=True)
        self.source, self.work, self.cfg = source, Path(workdir), cfg
        self.on_chunk, self.on_end, self.on_info = on_chunk, on_end, on_info
        self.readrate = readrate
        self.chunk_s = max(5.0, float(getattr(cfg, "live_chunk_s", 20.0)))
        self.max_s = float(getattr(cfg, "live_max_h", 6.0)) * 3600
        self._stop = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.offset = 0.0          # gönderilen parçaların toplam süresi
        self.idx = 0
        self.reason = ""
        self.raw = self.work / "canli.raw"

    def stop(self, reason: str = "durduruldu") -> None:
        self.reason = self.reason or reason
        self._stop.set()
        p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    def _ffmpeg(self, url: str, headers: dict) -> subprocess.Popen:
        from .fetch import ffmpeg_bin
        cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error"]
        if os.path.exists(url):
            cmd += ["-readrate", str(self.readrate or 1.0)]
        elif url.startswith("http"):
            if headers:
                cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
            cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
                    "-rw_timeout", "30000000"]
        cmd += ["-i", url, "-vn", "-sn", "-dn", "-map", "0:a:0?", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
        kw = {"creationflags": 0x00004000 | 0x08000000} if os.name == "nt" else {}
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)

    def _emit(self, pcm: np.ndarray) -> None:
        if len(pcm) < SR // 2:
            return
        p = self.work / "parca" / f"{self.idx:04d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        write_wav(p, pcm)
        dur = len(pcm) / SR
        self.on_chunk(self.idx, p, self.offset, dur)
        self.idx += 1
        self.offset += dur

    def run(self) -> None:  # noqa: C901
        self.work.mkdir(parents=True, exist_ok=True)
        buf = np.zeros(0, dtype=np.int16)
        fails = 0
        rawf = open(self.raw, "ab")  # noqa: SIM115
        try:
            while not self._stop.is_set():
                try:
                    url, headers, info = resolve_stream(self.source, self.cfg)
                    if self.on_info and info:
                        self.on_info(info)
                        self.on_info = None
                    if self.idx and info.get("live_status") in ("was_live", "post_live", "not_live"):
                        self.reason = "yayın bitti"  # artık kayıt (VOD): baştan kaydetme
                        break
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    log.warning("Canlı yayın adresi alınamadı (%d/8): %s", fails, str(e)[:200])
                    if fails >= 8 or self._stop.wait(min(60, 5 * fails)):
                        self.reason = self.reason or f"yayın açılamadı: {str(e)[:160]}"
                        break
                    continue
                self.proc = self._ffmpeg(url, headers)
                got = 0
                block = BYTES_PER_S // 5
                carry = b""
                while not self._stop.is_set():
                    data = self.proc.stdout.read(block) if self.proc.stdout else b""
                    if not data:
                        break
                    data = carry + data
                    if len(data) % 2:
                        carry, data = data[-1:], data[:-1]
                    else:
                        carry = b""
                    got += len(data)
                    rawf.write(data)
                    buf = np.concatenate([buf, np.frombuffer(data, dtype=np.int16)])
                    c = int(self.chunk_s * SR)
                    if len(buf) >= int(c * 1.25):
                        k = cut_point(buf, int(c * 0.6), int(c * 1.25))
                        self._emit(buf[:k])
                        buf = buf[k:]
                    if self.offset + len(buf) / SR >= self.max_s:
                        self.reason = f"en fazla {self.max_s / 3600:.0f} saat"
                        self._stop.set()
                err = ""
                if self.proc.poll() is None:
                    self.proc.terminate()
                try:
                    self.proc.wait(10)
                    err = (self.proc.stderr.read() or b"").decode("utf-8", "replace")[-300:] if self.proc.stderr else ""
                except Exception:  # noqa: BLE001
                    pass
                if self._stop.is_set():
                    break
                if os.path.exists(self.source):
                    self.reason = "dosya bitti"
                    break
                if self.proc.returncode == 0 and got > 0 and not err.strip():
                    self.reason = "yayın bitti"  # ffmpeg akışın sonuna temiz ulaştı (#EXT-X-ENDLIST)
                    break
                # akış koptu: yayın bitti mi, yoksa bağlantı mı gitti? Bir kaç kez yeniden bağlan.
                fails = fails + 1 if got < BYTES_PER_S * 5 else 1
                log.warning("Canlı yayın akışı kesildi (%s); yeniden bağlanılıyor (%d/8)", err.strip()[:160] or "bitti?", fails)
                if fails >= 8 or self._stop.wait(min(30, 3 * fails)):
                    self.reason = self.reason or "yayın bitti"
                    break
        except Exception as e:  # noqa: BLE001
            log.exception("Canlı kayıt hatası")
            self.reason = f"hata: {e}"
        finally:
            try:
                self._emit(buf)
            except Exception:  # noqa: BLE001
                log.exception("son parça yazılamadı")
            rawf.close()
            full = self.work / "ses.wav"
            try:
                raw_to_wav(self.raw, full)
                self.raw.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                log.exception("canlı kayıt birleştirilemedi")
            self.on_end(full, self.reason or "bitti")
