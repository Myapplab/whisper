# Geliştirme araçları

GPU'suz / HuggingFace'e erişimi olmayan bir ortamda (ör. bulut oturumu) uçtan uca test için.

## Küçük test modeli (CPU)

```bash
mkdir -p models && cd models
curl -sL https://github.com/rhasspy/models/releases/download/v1.0/asr_faster-whisper-small-int8.tar.gz | tar xz
cd ..
# tokenizer.json ve alignment_heads (word_timestamps için) eksik: openai-whisper paketinden üret
python -m venv /tmp/tokvenv && /tmp/tokvenv/bin/pip install -q tiktoken transformers numpy && /tmp/tokvenv/bin/pip install -q --no-deps openai-whisper
/tmp/tokvenv/bin/python -I dev/make_tokenizer.py models/small-int8/tokenizer.json
/tmp/tokvenv/bin/python -I dev/add_alignment_heads.py models/small-int8/config.json
python -m stt <dosya> --model models/small-int8 --gpus cpu,cpu --lang en -o /tmp/cikti
```

Örnek ses: `https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/tests/data/physicsworks.wav`,
`https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac`.
WER örnekleri: `python -m stt olc --ornek --model models/small-int8 --gpus cpu` (sherpa-onnx test_wavs, GitHub).

## Arayüz testi (pywebview olmadan)

```bash
pip install -e .   # bir kez
STT_OUT=/tmp/gui_out STT_MODEL=models/small-int8 STT_GPUS=cpu,cpu python -I dev/gui_bridge.py &   # http://127.0.0.1:8799
STT_TEST_DIR=/tmp/gui_in node dev/gui_drive.js     # link yapıştır → Başlat → bitişi bekle → ekran görüntüleri
node dev/gui_screens.js                            # açık/koyu tema, okuyucu, ayarlar, 1024px
```

FPL API'ye erişim yoksa örnek fikstür: `tests/test_core.py` içindeki `FPL_SAMPLE`'ı
`<out>/.stt/fpl_cache.json` olarak yazın.

## Kurulum programı

```bash
sudo apt-get install -y nsis
./installer/build.sh   # -> dist/STT-Kurulum.exe
```
