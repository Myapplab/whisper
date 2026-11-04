"""Kurulum programının görsellerini üretir (Pillow gerekir): installer/welcome.bmp, installer/header.bmp.

    python3 installer/make_images.py

Görseller 2x çözünürlükte üretilir; NSIS bunları pencereye sığdırır (yüksek DPI ekranlarda net görünür).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ICON = HERE.parent / "stt" / "assets" / "stt.ico"
BG = (15, 19, 24)          # arayüzdeki koyu zemin (#0f1318)
PANEL = (28, 34, 43)
ACCENT = (249, 115, 22)    # turuncu vurgu (#f97316)
INK = (230, 233, 238)
MUTED = (154, 163, 174)


def font(size: int, bold: bool = False):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"])
    for n in names:
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "C:/Windows/Fonts"):
            p = Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def icon(size: int) -> Image.Image:
    im = Image.open(ICON)
    im.size = (256, 256)
    return im.convert("RGBA").resize((size, size), Image.LANCZOS)


def welcome():
    w, h = 328, 628  # 164x314'ün 2 katı
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    # alttan yukarı hafif ışık + ince vurgu çizgisi
    for y in range(h):
        t = y / h
        c = tuple(int(BG[i] + (PANEL[i] - BG[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c)
    d.rectangle([w - 6, 0, w, h], fill=ACCENT)
    ic = icon(132)
    im.paste(ic, ((w - 132) // 2, 120), ic)
    d.text((w // 2, 300), "STT", font=font(64, True), fill=INK, anchor="mm")
    d.text((w // 2, 352), "Basın Odası", font=font(30), fill=ACCENT, anchor="mm")
    y = 440
    for line in ("Basın toplantısı videoları", "→ soru-cevap metni"):
        d.text((w // 2, y), line, font=font(19), fill=MUTED, anchor="mm")
        y += 30
    im.save(HERE / "welcome.bmp")


def header():
    w, h = 300, 114  # 150x57'nin 2 katı
    im = Image.new("RGB", (w, h), (255, 255, 255))
    ic = icon(84)
    im.paste(ic, (w - 84 - 22, (h - 84) // 2), ic)
    im.save(HERE / "header.bmp")


if __name__ == "__main__":
    welcome()
    header()
    print("welcome.bmp, header.bmp hazır")
