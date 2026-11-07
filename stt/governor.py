"""Gece modu (1.9.19): uzun kuyruk (200+ video) için yavaş ama dengeli çalışma, kendini ısıya göre ayarlar.

Kullanıcı: "uyurken 200'den fazla video; bilgisayar çökmesin, GPU az harcasın, yavaş ama stabil; MSI Afterburner gibi ama
uygulama kendi ayarlasın". Afterburner'ın yaptığı fan eğrisi / hız aşırtma yazılımdan güvenle yapılamaz (NVIDIA fan
denetimi yönetici + belgelenmemiş arayüz ister; güç / saat sınırı zaten ``stt/power.py``da). Burada yapılan, kartın
*ne kadar süre* çalıştığını ayarlamak (görev oranı):

- Her görevden (5 dk'lık metin parçası ~10 sn, konuşmacı ayırma 1-2 dk) sonra kart ``rest`` saniye dinlenir. Isı hedefin
  üstündeyse mola uzar, altındaysa kısalır (en az ``REST_MIN``): kart hedef ısının çevresinde kalır.
- Isı ``hedef + HOT_OVER`` derecesine çıkarsa "soğuma molası": yeni görev verilmez, ``hedef - COOL_UNDER``'a inince
  sürer. Mola ``PAUSE_MAX_S``'den uzun sürerse (ör. odanın sıcaklığı, fan durmuş) en uzun molayla devam edilir; iş
  sonsuza dek durmaz.
- Isı okunamıyorsa (nvidia-smi yok) her görevden sonra sabit ``REST_BLIND`` sn mola.

Karar yalnız ısıya göre: güç sıçramalarını güç / saat sınırı keser (``power.py``), bu modül ortalama yükü ve ısıyı
düşürür (fan, kart ömrü, uzun süre kararlılık). Saf mantık: zaman ve ısı dışarıdan verilir (test edilebilir).
"""
from __future__ import annotations

REST_MIN = 4.0        # gece modunda her görevden sonra en az bu kadar mola (sn)
REST_MAX = 90.0       # en uzun mola (sn)
REST_BLIND = 10.0     # ısı okunamıyorsa sabit mola (sn)
HOT_OVER = 8          # hedef + bu kadar derece: soğuma molası
COOL_UNDER = 6        # soğuma molası hedef - bu kadar dereceye inince biter
PAUSE_MAX_S = 900.0   # soğuma molası en çok bu kadar (sonra en uzun molayla devam)
SAMPLE_S = 20.0       # ısı ölçümü aralığı (nvidia-smi; i3'te sık çalışmasın)
NIGHT_PREP_AHEAD = 3  # gece modunda önden hazırlanan video (disk, işlemci)


class Governor:
    """Isıya göre görev arası mola. ``update(temp, now)`` ölçümde, ``rest`` görev bitince kullanılır."""

    def __init__(self, target: int = 72):
        self.target = int(target)
        self.rest = REST_MIN
        self.paused = False
        self.paused_at: float | None = None
        self.temp: float | None = None
        self.forced = False      # soğuma molası çok uzadı: en uzun molayla devam ediliyor

    @property
    def hot(self) -> int:
        return self.target + HOT_OVER

    @property
    def cool(self) -> int:
        return self.target - COOL_UNDER

    def set_target(self, target: int) -> None:
        self.target = int(target)

    def update(self, temp: float | None, now: float) -> str:
        """Yeni ısı ölçümü. Dönüş: günlüğe yazılacak durum değişikliği ("" = değişiklik yok)."""
        self.temp = temp
        if temp is None:
            was = self.paused
            self.paused, self.paused_at, self.forced = False, None, False
            self.rest = REST_BLIND
            return "ısı okunamıyor: soğuma molası bitti, her görevden sonra sabit mola" if was else ""
        if self.paused:
            if temp <= self.cool:
                self.paused, self.paused_at, self.forced = False, None, False
                self.rest = max(self.rest, REST_MIN * 2)
                return f"soğuma molası bitti ({temp:.0f}°C); iş sürüyor, görev arası {self.rest:.0f} sn"
            if self.paused_at is not None and now - self.paused_at >= PAUSE_MAX_S and not self.forced:
                self.forced = True
                self.rest = REST_MAX
                return (f"kart {PAUSE_MAX_S / 60:.0f} dakikadır soğumadı ({temp:.0f}°C; oda sıcak / fan?): en uzun "
                        f"molayla ({REST_MAX:.0f} sn) devam ediliyor")
            return ""
        if temp >= self.hot:
            self.paused, self.paused_at, self.forced = True, now, False
            self.rest = min(REST_MAX, max(self.rest * 2, REST_MIN * 3))
            return f"kart ısındı ({temp:.0f}°C ≥ {self.hot}°C): soğuma molası, {self.cool}°C'ye inince sürecek"
        old = self.rest
        if temp > self.target:
            self.rest = min(REST_MAX, self.rest * 1.5 + 2)
        elif temp < self.target - 4:
            self.rest = max(REST_MIN, self.rest * 0.8)
        if abs(self.rest - old) >= 5 or (self.rest == REST_MIN) != (old == REST_MIN):
            return f"{temp:.0f}°C (hedef {self.target}°C): görev arası mola {old:.0f} → {self.rest:.0f} sn"
        return ""

    def holding(self) -> bool:
        """Yeni görev verilmesin mi (soğuma molası; çok uzadıysa artık tutmaz)."""
        return self.paused and not self.forced

    def note(self) -> str:
        t = f"{self.temp:.0f}°C" if self.temp is not None else "ısı okunamıyor"
        if self.holding():
            return f"soğuma molası · {t} → {self.cool}°C"
        return f"{t} · görev arası {self.rest:.0f} sn mola"
