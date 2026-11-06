"""v1.9.6 testleri:
- İlk cümlenin sonraki konuşmacının çok cümleli bloğuna yutulmaması (merge.py)
- Konuşmacılar ekranında satır bölme, metin düzeltme, kaçırılan satır ekleme, aynı anda konuşma (voices.py, gui.py)
- Üst üste (aynı anda) konuşulan saniyelerin ses imzasından çıkarılması (tune.py)
- 'Kaydet ve eğit' ile çıkarılmış temiz ses imzasının düz 'Kaydet' ile silinmemesi (voices.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stt.merge import assign_speakers
from stt.outputs import to_srt, to_txt, write_all
from stt.tune import _solo_spans, voice_embeddings
from stt.voices import (
    VoiceLibrary,
    apply_corrections,
    learn_from_doc,
    save_doc_voices,
    spans_fp,
    split_segment,
)


def _w(start: float, end: float, word: str) -> dict:
    return {"start": start, "end": end, "word": " " + word}


# ------------------------------------------------------------------ 1. merge.py: ilk cümle yutulmasın

def test_first_sentence_with_tail_drift_not_swallowed_into_second_speaker():
    """Konuşmacı 1 bir kısa cümle (0.9 sn) söylüyor, son kelimesi (0.2 sn) Konuşmacı 2'ye kayıyor;
    ardından Konuşmacı 2 art arda 5 cümle söylüyor. İlk cümle Konuşmacı 1'de kalmalı."""
    segs = [{
        "id": 0, "start": 0.0, "end": 12.0,
        "text": "Sorunuz nedir acaba. İlk cevabım bu. İkinci cümlem devam ediyor. "
                "Üçüncü cümlem burada. Dördüncü cümlem de var. Beşinci cümlem bitti.",
        "words": [
            _w(0.0, 0.45, "Sorunuz"), _w(0.45, 0.90, "nedir"), _w(0.90, 1.10, "acaba."),
            _w(1.20, 1.80, "İlk"), _w(1.80, 2.40, "cevabım"), _w(2.40, 3.00, "bu."),
            _w(3.10, 3.80, "İkinci"), _w(3.80, 4.50, "cümlem"), _w(4.50, 5.10, "devam"), _w(5.10, 5.70, "ediyor."),
            _w(5.80, 6.50, "Üçüncü"), _w(6.50, 7.20, "cümlem"), _w(7.20, 7.90, "burada."),
            _w(8.00, 8.60, "Dördüncü"), _w(8.60, 9.20, "cümlem"), _w(9.20, 9.60, "de"), _w(9.60, 10.10, "var."),
            _w(10.20, 10.80, "Beşinci"), _w(10.80, 11.40, "cümlem"), _w(11.40, 12.00, "bitti."),
        ],
    }]
    turns = [
        (0.0, 0.90, "SPEAKER_00"),   # Konuşmacı 1'in ilk cümlesi (son kelime 0.90-1.10 arası SPEAKER_01'e kaymış)
        (0.90, 12.0, "SPEAKER_01"),  # Konuşmacı 2'nin 5 cümlesi
    ]
    out = assign_speakers(segs, turns)
    assert len(out) == 2
    assert out[0]["text"] == "Sorunuz nedir acaba."
    assert out[0]["speaker_id"] == "SPEAKER_00"
    assert out[1]["speaker_id"] == "SPEAKER_01"
    assert out[1]["text"].startswith("İlk cevabım bu.")


def test_first_sentence_joined_by_comma_split_from_second_speaker_five_sentences():
    """Whisper ilk konuşmacının cümlesini nokta yerine virgülle ikinci konuşmacının 5 cümlesine bağlamışsa,
    virgülde konuşmacı değiştiği için cümleler ayrılmalı ve ilk cümle Konuşmacı 1'de kalmalı."""
    segs = [{
        "id": 0, "start": 0.0, "end": 11.0,
        "text": "Bunu sormak istiyorum, tabii hemen anlatayım. İkinci cümle. Üçüncü cümle. Dördüncü cümle. Beşinci cümle.",
        "words": [
            _w(0.0, 0.35, "Bunu"), _w(0.35, 0.70, "sormak"), _w(0.70, 1.10, "istiyorum,"),
            _w(1.20, 1.70, "tabii"), _w(1.70, 2.20, "hemen"), _w(2.20, 2.80, "anlatayım."),
            _w(3.0, 3.8, "İkinci"), _w(3.8, 4.6, "cümle."),
            _w(4.8, 5.6, "Üçüncü"), _w(5.6, 6.4, "cümle."),
            _w(6.6, 7.4, "Dördüncü"), _w(7.4, 8.2, "cümle."),
            _w(8.4, 9.5, "Beşinci"), _w(9.5, 11.0, "cümle."),
        ],
    }]
    turns = [
        (0.0, 1.15, "SPEAKER_00"),
        (1.15, 11.0, "SPEAKER_01"),
    ]
    out = assign_speakers(segs, turns)
    assert out[0]["text"] == "Bunu sormak istiyorum,"
    assert out[0]["speaker_id"] == "SPEAKER_00"
    assert out[1]["text"].startswith("tabii hemen anlatayım.")
    assert all(s["speaker_id"] == "SPEAKER_01" for s in out[1:])


def test_first_sentence_without_punctuation_preserved_on_speaker_turn():
    """Whisper noktalama koymadan Konuşmacı 1'in ilk cümlesini (2 kelime, 0.9 sn) Konuşmacı 2'nin
    uzun cümlesine yapıştırmışsa bile başlangıçtaki konuşmacı korunmalı ve bölünmeli."""
    segs = [{
        "id": 0, "start": 0.0, "end": 5.0,
        "text": "Çok teşekkürler bugün maçta çok iyi oynadık ve kazandık.",
        "words": [
            _w(0.0, 0.45, "Çok"), _w(0.45, 0.95, "teşekkürler"),
            _w(1.05, 1.50, "bugün"), _w(1.50, 2.00, "maçta"), _w(2.00, 2.50, "çok"),
            _w(2.50, 3.00, "iyi"), _w(3.00, 3.80, "oynadık"), _w(3.80, 4.20, "ve"), _w(4.20, 5.00, "kazandık."),
        ],
    }]
    turns = [
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 5.0, "SPEAKER_01"),
    ]
    out = assign_speakers(segs, turns)
    assert len(out) == 2
    assert out[0]["text"] == "Çok teşekkürler"
    assert out[0]["speaker_id"] == "SPEAKER_00"
    assert out[1]["speaker_id"] == "SPEAKER_01"


def test_mid_sentence_blip_not_falsely_split():
    """Aynı konuşmacı konuşurken cümlenin ortasında 0.6 sn'lik kısa bir akustik sapma olursa
    cümle bölünmemeli, tek konuşmacıda kalmalı."""
    segs = [{
        "id": 0, "start": 0.0, "end": 5.0,
        "text": "Bugün sahada oyuncularımız gerçekten çok büyük mücadele ortaya koydu.",
        "words": [
            _w(0.0, 0.6, "Bugün"), _w(0.6, 1.2, "sahada"), _w(1.2, 2.0, "oyuncularımız"),
            _w(2.0, 2.3, "gerçekten"), _w(2.3, 2.6, "çok"),
            _w(2.6, 3.2, "büyük"), _w(3.2, 4.0, "mücadele"), _w(4.0, 4.5, "ortaya"), _w(4.5, 5.0, "koydu."),
        ],
    }]
    turns = [
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 2.6, "SPEAKER_01"),  # 0.6 sn sapma
        (2.6, 5.0, "SPEAKER_00"),
    ]
    out = assign_speakers(segs, turns)
    assert len(out) == 1
    assert out[0]["speaker_id"] == "SPEAKER_00"


# ------------------------------------------------------------------ 2. voices.py: bölme, metin düzeltme, aynı anda konuşma

def test_split_segment_first_sentence_and_cursor():
    seg = {
        "id": 0, "start": 10.0, "end": 20.0, "speaker_id": "SPEAKER_01",
        "text": "İlk cümle benim. İkinci cümle onun. Üçüncü cümle de onun.",
        "words": [
            _w(10.0, 10.5, "İlk"), _w(10.5, 11.0, "cümle"), _w(11.0, 11.8, "benim."),
            _w(12.2, 13.0, "İkinci"), _w(13.0, 14.0, "cümle"), _w(14.0, 15.0, "onun."),
            _w(15.5, 17.0, "Üçüncü"), _w(17.0, 18.5, "cümle"), _w(18.5, 19.2, "de"), _w(19.2, 20.0, "onun."),
        ],
    }
    # İmleç verilmezse ilk cümle sonundan böler
    p1, p2 = split_segment(seg)
    assert p1["text"] == "İlk cümle benim."
    assert p2["text"] == "İkinci cümle onun. Üçüncü cümle de onun."
    assert p1["start"] == 10.0 and p1["end"] == 11.8
    assert p2["start"] == 12.2 and p2["end"] == 20.0
    assert len(p1["words"]) == 3 and len(p2["words"]) == 7

    # İmleç ikinci cümle sonuna konursa oradan böler
    pos = seg["text"].index("Üçüncü")
    q1, q2 = split_segment(seg, pos=pos)
    assert q1["text"] == "İlk cümle benim. İkinci cümle onun."
    assert q2["text"] == "Üçüncü cümle de onun."


def test_apply_corrections_edit_split_add_and_simultaneous(tmp_path: Path):
    """Konuşmacılar ekranında:
    1) İlk cümlesi Konuşmacı 2'ye yapışmış satırı bölüp ilkini Konuşmacı 1'e verme,
    2) Yanlış anlaşılan kelimeyi düzeltme (.srt'de de güncellenmeli),
    3) Kaçırılan sözü (+ Satır) ekleme,
    4) İki kişinin aynı anda söylediği sözü (👥 Aynı anda) ekleme."""
    doc = {
        "key": "vid1", "title": "Test", "speaker_mode": "herkes", "td_label": None,
        "speakers": {
            "SPEAKER_00": {"label": "Muhabir", "seconds": 2.0},
            "SPEAKER_01": {"label": "Hoca", "seconds": 10.0},
        },
        "segments": [
            {
                "id": 0, "start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_01", "speaker": "Hoca",
                "text": "Hocam nasılsınız? İyiyim teşekkür ederim. YanlışKelime burada.",
                "words": [
                    _w(0.0, 0.8, "Hocam"), _w(0.8, 1.8, "nasılsınız?"),
                    _w(2.0, 3.0, "İyiyim"), _w(3.0, 4.5, "teşekkür"), _w(4.5, 6.0, "ederim."),
                    _w(6.2, 8.0, "YanlışKelime"), _w(8.0, 10.0, "burada."),
                ],
            },
        ],
    }
    edited_segments = [
        # 1. Bölünen ilk cümle -> SPEAKER_00
        {"id": 0, "orig_id": 0, "start": 0.0, "end": 1.8, "sid": "SPEAKER_00", "text": "Hocam nasılsınız?"},
        # 2. Bölünen kalan kısım + yanlış kelime düzeltildi -> SPEAKER_01
        {"id": 1, "orig_id": 0, "start": 2.0, "end": 10.0, "sid": "SPEAKER_01",
         "text": "İyiyim teşekkür ederim. DoğruKelime burada."},
        # 3. 10.0-11.5 arasında aynı anda söylenen kelime (her iki konuşmacı)
        {"id": 2, "orig_id": None, "start": 10.0, "end": 11.5, "sid": "SPEAKER_00", "text": "Kesinlikle."},
        {"id": 3, "orig_id": None, "start": 10.0, "end": 11.5, "sid": "SPEAKER_01", "text": "Kesinlikle."},
        # 4. Kaçırılan satır eklendi
        {"id": 4, "orig_id": None, "start": 11.5, "end": 13.0, "sid": "SPEAKER_00", "text": "Kaçırılan soru."},
    ]
    apply_corrections(doc, labels={"SPEAKER_00": "Ali", "SPEAKER_01": "Fatih Terim"}, segments=edited_segments)
    assert len(doc["segments"]) == 5
    assert doc["segments"][0]["speaker"] == "Ali"
    assert doc["segments"][0]["text"] == "Hocam nasılsınız?"
    assert doc["segments"][1]["speaker"] == "Fatih Terim"
    assert "DoğruKelime" in doc["segments"][1]["text"]
    # .srt çıktısında eski 'YanlışKelime' kalmamalı, düzeltilmiş 'DoğruKelime' ve yeni satırlar yer almalı
    srt = to_srt(doc["segments"], diarized=True)
    assert "YanlışKelime" not in srt
    assert "DoğruKelime" in srt
    assert "Kaçırılan soru." in srt
    txt = to_txt(doc)
    assert "Ali" in txt and "Fatih Terim" in txt
    assert "Kaçırılan soru." in txt
    write_all(tmp_path, "test_out", doc, "test_out - ozet.md")
    assert "DoğruKelime" in (tmp_path / "test_out.srt").read_text(encoding="utf-8")


# ------------------------------------------------------------------ 3. tune.py: aynı anda konuşma ses imzasını bozmasın

def test_solo_spans_and_voice_embeddings_exclude_overlapping_speech():
    """İki kişinin aynı anda konuştuğu [2.0, 4.0] aralığı tek kişilik ses imzasından çıkarılmalı."""
    import types

    ref_turns = [
        (0.0, 2.0, "Ali"),
        (2.0, 4.0, "Ali"),       # üst üste
        (2.0, 4.0, "Veli"),      # üst üste
        (4.0, 12.0, "Ali"),
        (12.0, 22.0, "Veli"),
    ]
    ali_solo = _solo_spans(ref_turns, "Ali")
    veli_solo = _solo_spans(ref_turns, "Veli")
    assert ali_solo == [(4.0, 12.0), (0.0, 2.0)]
    assert veli_solo == [(12.0, 22.0)]

    # [2, 4] aralığında genlik 0.9 (üst üste konuşma gürültüsü), yalnız konuşulan yerlerde Ali=0.1, Veli=0.5
    audio = np.concatenate([
        np.full(2 * 16000, 0.1, dtype=np.float32),
        np.full(2 * 16000, 0.9, dtype=np.float32),
        np.full(8 * 16000, 0.1, dtype=np.float32),
        np.full(10 * 16000, 0.5, dtype=np.float32),
    ])

    class DummyEmbModel:
        min_num_samples = 8000

        def __call__(self, x):
            x = np.asarray(x)
            return np.array([[float(np.abs(x).mean()), 1.0, 0.0]])

    fake_torch = types.SimpleNamespace(from_numpy=lambda a: np.asarray(a))
    embs = voice_embeddings(DummyEmbModel(), audio, ref_turns, ["Ali", "Veli"], torch=fake_torch)
    assert set(embs.keys()) == {"Ali", "Veli"}
    assert abs(embs["Ali"][0][0] - 0.1) < 1e-6
    assert abs(embs["Veli"][0][0] - 0.5) < 1e-6


# ------------------------------------------------------------------ 4. voices.py: 'Kaydet ve eğit' imzası düz 'Kaydet'te silinmesin

def test_plain_save_preserves_trained_duzeltme_voice_signature(tmp_path: Path):
    """'Kaydet ve eğit' ile çıkarılmış temiz ses imzası (src='duzeltme') varken,
    kullanıcı aynı videoda tekrar düz 'Kaydet'e bastığında küme saflığı < %90 olsa bile
    o konuşmacının 'duzeltme' imzası silinmemeli ve 'öğrenilmedi' uyarısı vermemeli."""
    lib = VoiceLibrary(tmp_path)
    clean_vec = [1.0, 0.0, 0.0, 0.0]
    noisy_vec = [0.0, 1.0, 0.0, 0.0]
    # 1) Eğitimden ('duzeltme') temiz imza kaydedilmiş
    # (1.9.7: imza çıkarıldığı satırların parmak iziyle saklanır; satırlar aynı kaldıkça korunur)
    fp = spans_fp([[5.0, 10.0, "Fatih Terim"]], "Fatih Terim")
    lib.replace_video("vid1", "wespeaker", [("Fatih Terim", clean_vec, True, "", fp)], src="duzeltme")

    # 2) Videonun ilk ayırma kümesi (SPEAKER_01) %50 karışık (yarısı başka konuşmacıya taşınmış)
    save_doc_voices(tmp_path, "vid1", "wespeaker", {"SPEAKER_01": noisy_vec},
                    turns=[[0.0, 10.0, "SPEAKER_01"]])
    doc = {
        "key": "vid1", "speaker_mode": "basin", "td_label": "Fatih Terim", "corrected": True,
        "speakers": {
            "SPEAKER_00": {"label": "Muhabir", "seconds": 5.0},
            "SPEAKER_01": {"label": "Fatih Terim", "seconds": 5.0},
        },
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "speaker_id": "SPEAKER_00", "speaker": "Muhabir", "text": "Soru"},
            {"id": 1, "start": 5.0, "end": 10.0, "speaker_id": "SPEAKER_01", "speaker": "Fatih Terim", "text": "Cevap"},
        ],
    }
    skipped = []
    learned = learn_from_doc(tmp_path, doc, ignore=("Muhabir",), skipped=skipped)
    assert "Fatih Terim" in learned
    assert "Fatih Terim" not in skipped
    saved_embs = lib.load()["voices"]["Fatih Terim"]["embs"]
    assert len(saved_embs) == 1
    assert saved_embs[0]["src"] == "duzeltme"


# ------------------------------------------------------------------ 5. gui.py: speakers_load & speakers_save uçtan uca

def test_gui_speakers_load_and_save_with_segments(tmp_path: Path, monkeypatch):
    import stt.gui as g

    monkeypatch.chdir(tmp_path)
    app = g.App()
    app._cfg.out_dir = tmp_path
    doc = {
        "key": "k1", "title": "Basın Toplantısı", "speaker_mode": "basin", "td_label": "Teknik direktör",
        "speakers": {
            "SPEAKER_00": {"label": "Muhabir", "seconds": 3.0},
            "SPEAKER_01": {"label": "Teknik direktör", "seconds": 8.0},
        },
        "segments": [
            {
                "id": 0, "start": 0.0, "end": 8.0, "speaker_id": "SPEAKER_01", "speaker": "Teknik direktör",
                "text": "İlk cümle soru. İkinci cümle cevap.",
                "words": [
                    _w(0.0, 1.0, "İlk"), _w(1.0, 2.0, "cümle"), _w(2.0, 3.0, "soru."),
                    _w(3.5, 5.0, "İkinci"), _w(5.0, 6.5, "cümle"), _w(6.5, 8.0, "cevap."),
                ],
            }
        ],
    }
    jp = tmp_path / "ornek.json"
    jp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    loaded = app.speakers_load(str(jp))
    assert loaded["ok"] is True
    assert len(loaded["segments"][0]["words"]) == 6

    res = app.speakers_save(str(jp), {
        "labels": {"SPEAKER_00": "Ahmet", "SPEAKER_01": "Okan Buruk"},
        "moves": {},
        "td": "SPEAKER_01",
        "segments": [
            {"id": 0, "orig_id": 0, "start": 0.0, "end": 3.0, "sid": "SPEAKER_00", "text": "İlk cümle soru."},
            {"id": 1, "orig_id": 0, "start": 3.5, "end": 8.0, "sid": "SPEAKER_01", "text": "İkinci cümle düzeltilmiş cevap."},
        ],
    })
    assert res["ok"] is True
    updated = json.loads(jp.read_text(encoding="utf-8"))
    assert len(updated["segments"]) == 2
    assert updated["segments"][0]["speaker"] == "Ahmet"
    assert updated["segments"][1]["speaker"] == "Okan Buruk"
    assert "düzeltilmiş" in (tmp_path / "ornek.txt").read_text(encoding="utf-8")
    assert "düzeltilmiş" in (tmp_path / "ornek.srt").read_text(encoding="utf-8")
