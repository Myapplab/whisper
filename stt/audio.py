"""WAV okuma ve uzun dosyaları VAD (Silero) ile sessizlik noktalarından parçalama."""
from __future__ import annotations

import wave

import numpy as np

SR = 16000


def load_wav(path, start: float | None = None, end: float | None = None) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        a = 0 if start is None else max(0, int(start * SR))
        b = n if end is None else min(n, int(end * SR))
        w.setpos(a)
        raw = w.readframes(max(0, b - a))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=500, speech_pad_ms=200, max_speech_duration_s=60)
    ts = get_speech_timestamps(audio, vad_options=opts, sampling_rate=SR)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def group_chunks(speech: list[tuple[float, float]], duration: float, target_s: float,
                 min_s: float | None = None) -> list[dict]:
    """Konuşma bölgelerini ~target_s uzunlukta parçalara toplar.

    Kesimler iki konuşma bölgesi arasındaki sessizliğin ortasından yapılır; parçalar tüm
    zaman çizgisini boşluksuz kaplar (VAD bir şeyi kaçırsa bile ses kaybolmaz).
    Her parça: {start, end, speech_start, speech_end}.
    """
    if duration <= 0:
        return []
    if not speech:
        return [{"start": 0.0, "end": duration, "speech_start": 0.0, "speech_end": duration}]
    min_s = target_s * 0.5 if min_s is None else min_s
    groups: list[list[tuple[float, float]]] = [[speech[0]]]
    for seg in speech[1:]:
        cur = groups[-1]
        if seg[1] - cur[0][0] > target_s and cur[-1][1] - cur[0][0] >= min_s:
            groups.append([seg])
        else:
            cur.append(seg)
    # son parça çok kısaysa öncekine kat
    if len(groups) > 1 and groups[-1][-1][1] - groups[-1][0][0] < min_s * 0.5:
        last = groups.pop()
        groups[-1].extend(last)
    chunks = []
    for i, g in enumerate(groups):
        start = 0.0 if i == 0 else (groups[i - 1][-1][1] + g[0][0]) / 2
        end = duration if i == len(groups) - 1 else (g[-1][1] + groups[i + 1][0][0]) / 2
        chunks.append({"start": round(start, 3), "end": round(end, 3),
                       "speech_start": round(g[0][0], 3), "speech_end": round(g[-1][1], 3)})
    return chunks


def plan_chunks(wav_path, duration: float, *, split_s: float, target_s: float) -> list[dict]:
    if duration <= split_s:
        return [{"start": 0.0, "end": duration, "speech_start": 0.0, "speech_end": duration}]
    audio = load_wav(wav_path)
    return group_chunks(speech_segments(audio), duration, target_s)
