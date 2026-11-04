#!/usr/bin/env bash
# STT-Kurulum.exe üretir (Linux'ta: apt-get install nsis). Çıktı: dist/STT-Kurulum.exe
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
python3 - "$ROOT" "$WORK/staged" <<'PY'
import shutil, sys
src, dst = sys.argv[1], sys.argv[2]
ign = shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache", ".git", "installer", "dev", "dist",
                             "ornekler", "cikti", ".venv", "stt.toml")
shutil.copytree(src, dst, ignore=ign)
shutil.copy(f"{src}/installer/kur.ps1", f"{dst}/kur.ps1")
PY
cp "$ROOT/installer/stt.nsi" "$WORK/"
(cd "$WORK" && makensis -V2 stt.nsi)
mkdir -p "$ROOT/dist"
cp "$WORK/STT-Kurulum.exe" "$ROOT/dist/"
echo "dist/STT-Kurulum.exe hazır"
