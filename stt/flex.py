"""Esnek tam güç (1.9.21): kartlar tam güçte çalışır, başka programlar (tarayıcı, oyun, Windows) işlemciyi ya da RAM'i
kullanmaya başlayınca iş kademeli yavaşlar, onlar bırakınca yeniden tam güce çıkar.

Kullanıcı: "iki GPU da max oranda çalışsın, aynı zamanda normal browser ve diğer şeylere güç yetsin; diğer şeyler güç
tüketimi artarken azalt, sonra çoğalt; esnek olsun". Monitör anakarttaki (Intel) çıkışta: tarayıcı ekranı Intel'le
çizer, NVIDIA kartları değil. Kullanıcıda darboğaz işlemci (i3-10100, 4 çekirdek): iki kart tam yükteyken işçiler,
ffmpeg, VAD ve konuşmacı ayırmanın kümelemesi işlemciyi doldurur, tarayıcı takılır. Bu yüzden ölçüt *başka
programların* işlemci payı (makinenin tamamının %'si; bizim süreçlerimiz = uygulama + bütün alt süreçleri, çıkarılır)
ve boş RAM.

Kademeler (``Flex.level``):
  0 tam güç: iki kart da sürekli iş alır.
  1 hafif: ikinci (esnek / yardımcı) kart yeni iş almaz, önden en çok ``PREP_AHEAD_L1`` video hazırlanır (indirme,
    ffmpeg). Tarayıcı ve sekmeler için işlemci boşalır; ana kart (3090) sürer.
  2 güçlü: ayrıca ana kart her görevden sonra ``REST_L2`` sn dinlenir.
Yükselme hızlı (``UP_SAMPLES`` ardışık ölçüm, ~20 sn), inme yavaş ve kademeli (başka programlar ``CALM_S`` sn sakin
kalınca bir kademe): kısa sıçramalarda kartlar açılıp kapanmasın. Ölçülemiyorsa (eski Windows / izin yok) kademe 0.

İşlemci önceliği zaten düşük (işçiler BELOW_NORMAL, yardımcı IDLE): Windows anlık paylaşımı yapar; bu modül işin
miktarını da azaltır. ``Flex`` saf mantıktır (test edilebilir); ölçüm ``CpuSampler``da.

1.9.27 (esnek güç 2): kullanıcıda esnek düzen açıkken "uygulamaları kapattı, kendisi kapandı, bilgisayar kastı, işler
silindi". Sebep sanal bellek (commit = RAM + sayfa dosyası; kullanıcıda ~32 GB): 3-4 ağır süreç + tarayıcı onu
dolduruyor, Windows programları kapatıyordu; eskiden yalnız işlemciye ve boş RAM'e bakılıyordu. Artık:
  - boş sanal bellek < ``COMMIT_LOW_GB`` ya da boş RAM < ``RAM_LOW_GB``: HEMEN (tek ölçüm) kademe 1, ikinci kart boşta
    kalınca çabuk kapanır (``ram_low``; belleği boşalır);
  - < ``COMMIT_CRIT_GB`` / < ``RAM_CRIT_GB``: kademe 2 + ``mem_crit``: yeni video indirilmez (esnek kapalıyken de);
  - bilgisayar başında kimse yokken (``AWAY_S`` fare / klavye yok) başka programların işlemci yükü yavaşlatmaz (tam
    hız); bellek koruması sürer. Kullanıcı dönünce eski esnek davranış.
"""
from __future__ import annotations

import os

SAMPLE_S = 10.0       # ölçüm aralığı (sn)
UP1_CPU = 35.0        # başka programlar makinenin bu kadar %'sini kullanınca kademe 1
UP2_CPU = 60.0        # ... bu kadar %'sini kullanınca kademe 2
CALM_CPU = 20.0       # bunun altında "sakin"
RAM_LOW_GB = 2.0      # boş RAM bundan azsa en az kademe 1
RAM_CRIT_GB = 1.0     # boş RAM bundan azsa kademe 2 + yeni indirme yok
RAM_OK_GB = 3.5       # boş RAM bunun üstündeyse "sakin" sayılabilir
COMMIT_LOW_GB = 4.0   # boş sanal bellek bundan azsa en az kademe 1 (ikinci kart bırakır)
COMMIT_CRIT_GB = 2.0  # boş sanal bellek bundan azsa kademe 2 + yeni indirme yok
COMMIT_OK_GB = 6.0    # boş sanal bellek bunun üstündeyse "sakin" sayılabilir
AWAY_S = 300.0        # bu kadar sn fare / klavye kullanılmadıysa bilgisayar başında kimse yok
UP_SAMPLES = 2        # işlemci yüzünden yükselme için ardışık ölçüm (tek sıçrama yetmez; bellek beklemez)
CALM_S = 60.0         # bir kademe inmek için bu kadar sn sakin
REST_L2 = 8.0         # kademe 2: ana kart her görevden sonra bu kadar dinlenir (sn)
PREP_AHEAD_L1 = 2     # kademe 1-2: önden hazırlanan video
NOTE_L1 = "esnek: bilgisayar kullanılıyor, bu kart bekliyor"


class Flex:
    """Başka programların yüküne ve belleğe göre kademe. ``update(others_pct, ram_free_gb, now, commit_free_gb,
    idle_s)`` her ölçümde çağrılır."""

    def __init__(self):
        self.level = 0
        self.others: float | None = None
        self.ram: float | None = None
        self.commit: float | None = None
        self.idle: float | None = None
        self.away = False        # bilgisayar başında kimse yok (AWAY_S'dir fare / klavye yok)
        self.ram_low = False     # son ölçümde RAM ya da sanal bellek az mıydı (bekleyen ikinci kart hemen kapatılır)
        self.mem_crit = False    # bellek kritik: yeni video indirilmez / hazırlanmaz
        self._up = 0
        self._calm_since: float | None = None

    def update(self, others: float | None, ram_free: float | None, now: float, commit_free: float | None = None,
               idle_s: float | None = None) -> str:
        """Dönüş: günlüğe yazılacak değişiklik ("" = değişiklik yok)."""
        self.others, self.ram, self.commit, self.idle = others, ram_free, commit_free, idle_s
        self.away = idle_s is not None and idle_s >= AWAY_S
        if others is None and ram_free is None and commit_free is None:
            return ""  # ölçülemedi: son durum sürer
        low = (ram_free is not None and ram_free < RAM_LOW_GB) or (commit_free is not None and commit_free < COMMIT_LOW_GB)
        crit = (ram_free is not None and ram_free < RAM_CRIT_GB) or \
            (commit_free is not None and commit_free < COMMIT_CRIT_GB)
        self.ram_low, self.mem_crit = low, crit
        cpu = 0.0 if self.away else (others or 0.0)  # kimse yokken başka programlar için işlemci bırakmaya gerek yok
        mem_want = 2 if crit else 1 if low else 0
        want = max(mem_want, 2 if cpu >= UP2_CPU else 1 if cpu >= UP1_CPU else 0)
        if want > self.level:
            self._calm_since = None
            if mem_want >= want:  # bellek beklemez: dolunca Windows programları kapatıyordu
                old, self.level, self._up = self.level, want, 0
                return f"bellek azaldı ({self._what()}): kademe {old} → {self.level} ({self.describe()})"
            self._up += 1
            if self._up >= UP_SAMPLES:
                old, self.level, self._up = self.level, want, 0
                return f"başka programlar yükte ({self._what()}): kademe {old} → {self.level} ({self.describe()})"
            return ""
        self._up = 0
        mem_ok = (ram_free is None or ram_free >= RAM_OK_GB) and (commit_free is None or commit_free >= COMMIT_OK_GB)
        if self.level and self.away and mem_ok:  # kimse yok, bellek yeterli: beklemeden tam güç
            old, self.level, self._calm_since = self.level, 0, None
            return f"bilgisayar başında kimse yok ({self._what()}): kademe {old} → 0 ({self.describe()})"
        calm = cpu < CALM_CPU and mem_ok
        if self.level == 0 or not calm:
            self._calm_since = None
            return ""
        if self._calm_since is None:
            self._calm_since = now
            return ""
        if now - self._calm_since >= CALM_S:
            old = self.level
            self.level -= 1
            self._calm_since = now  # bir sonraki kademe için yeniden sakin süre
            return f"bilgisayar sakinleşti ({self._what()}): kademe {old} → {self.level} ({self.describe()})"
        return ""

    def _what(self) -> str:
        parts = []
        if self.others is not None:
            parts.append(f"işlemci %{self.others:.0f}")
        if self.ram is not None:
            parts.append(f"boş RAM {self.ram:.1f} GB")
        if self.commit is not None:
            parts.append(f"boş sanal bellek {self.commit:.1f} GB")
        if self.away:
            parts.append("bilgisayar başında kimse yok")
        return ", ".join(parts) or "ölçülemedi"

    def describe(self) -> str:
        d = ("tam güç", "ikinci kart bekliyor", f"ikinci kart bekliyor, ana kart her işten sonra {REST_L2:.0f} sn "
                                                "dinleniyor")[self.level]
        return d + ("; bellek kritik: yeni video indirilmiyor" if self.mem_crit else "")

    def note(self) -> str:
        """Arayüz için kısa durum."""
        if self.level == 0:
            # ölçüm her 10 sn: değer arayüzde bayatlamasın diye yazılmaz (yalnız kademe değişince olay)
            return "tam güç" + (" · başında kimse yok" if self.away else "")
        return f"yavaşladı ({self._what()}): {self.describe()}"


def idle_from_ticks(now_ms: int, last_ms: int) -> float:
    """GetTickCount / GetLastInputInfo (32 bit ms, 49,7 günde bir başa döner) farkı, sn."""
    return ((int(now_ms) - int(last_ms)) & 0xFFFFFFFF) / 1000.0


def user_idle_s() -> float | None:
    """Klavye / fare en son kaç sn önce kullanıldı (bu oturumda). Windows değilse ya da okunamazsa None (= kullanıcı
    var sayılır; eski davranış)."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class LII(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
        lii = LII()
        lii.cbSize = ctypes.sizeof(LII)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return None
        return idle_from_ticks(ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF, lii.dwTime)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ ölçüm
def _win_system_times():
    import ctypes
    from ctypes import wintypes
    idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
        return None
    f = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa: E731
    total = f(kern) + f(user)  # çekirdek süresi boşta süreyi de içerir
    return total - f(idle), total


class _WinTree:
    """Kök süreç + bütün alt süreçleri (işçiler, ffmpeg, WebView2, nvidia-smi): {anahtar: çekirdek+kullanıcı süresi}.
    Görülen her sürecin tanıtıcısı açık tutulur: süreç iki ölçüm arasında kapanırsa (ffmpeg dönüştürmesi bitti) son
    süresi yine okunur ve bir kez daha sayılır, sonra bırakılır (yoksa bizim işimiz "başka programlar" sanılırdı).
    Tanıtıcı açıkken pid başka sürece verilmez. Anahtar (pid, oluşturma zamanı)."""

    def __init__(self):
        self.handles: dict[int, int] = {}

    def __call__(self, root: int) -> dict | None:
        import ctypes
        from ctypes import wintypes

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snap or snap == wintypes.HANDLE(-1).value:
            return None
        parent: dict[int, int] = {}
        try:
            pe = PE()
            pe.dwSize = ctypes.sizeof(PE)
            ok = k32.Process32FirstW(wintypes.HANDLE(snap), ctypes.byref(pe))
            while ok:
                parent[pe.th32ProcessID] = pe.th32ParentProcessID
                ok = k32.Process32NextW(wintypes.HANDLE(snap), ctypes.byref(pe))
        finally:
            k32.CloseHandle(wintypes.HANDLE(snap))
        live = set(descendants(root, parent))
        for pid in live - set(self.handles):
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if h:
                self.handles[pid] = h
        out: dict = {}
        ft = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa: E731
        for pid, h in list(self.handles.items()):
            c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
            if k32.GetProcessTimes(wintypes.HANDLE(h), ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
                out[(pid, ft(c))] = ft(kt) + ft(ut)
            if pid not in live:  # kapandı (ya da artık bizim değil): son süresi bu ölçümde sayıldı, bırak
                k32.CloseHandle(wintypes.HANDLE(h))
                del self.handles[pid]
        return out


def _proc_system_times():
    with open("/proc/stat", encoding="ascii") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    total = sum(vals[:8])  # user nice system idle iowait irq softirq steal
    return total - vals[3] - (vals[4] if len(vals) > 4 else 0), total


def _proc_tree_times(root: int) -> dict[int, int] | None:
    parent, times = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", encoding="utf-8", errors="replace") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            parent[int(d)] = int(rest[1])
            # utime + stime + beklenen (kapanmış) çocukların süresi cutime + cstime (jiffy; /proc/stat ile aynı birim):
            # iki ölçüm arasında kapanan ffmpeg ebeveynin cutime'ına geçer
            times[int(d)] = (int(rest[11]) + int(rest[12]) + int(rest[13]) + int(rest[14]), int(rest[19]))
        except (OSError, IndexError, ValueError):
            continue
    return {(p, times[p][1]): times[p][0] for p in descendants(root, parent) if p in times}


def descendants(root: int, parent: dict[int, int]) -> list[int]:
    """root + bütün alt süreçleri (parent: {pid: ebeveyn pid}). Döngüye karşı korumalı."""
    kids: dict[int, list[int]] = {}
    for pid, pp in parent.items():
        if pid != pp:
            kids.setdefault(pp, []).append(pid)
    out, todo, seen = [], [root], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        todo += kids.get(p, [])
    return out


class CpuSampler:
    """İki ölçüm arasında: (başka programların işlemci payı %, bizim payımız %) — makinenin tamamının yüzdesi.
    İlk ölçümde ve okunamazsa None."""

    def __init__(self, root: int | None = None, system=None, tree=None):
        self.root = root or os.getpid()
        nt = os.name == "nt"
        self._system = system or (_win_system_times if nt else _proc_system_times)
        self._tree = tree or (_WinTree() if nt else _proc_tree_times)
        self._prev = None

    def reset(self) -> None:
        """İş bitti: sonraki ölçüm baştan (uzun aradan sonraki ilk fark yanlış olmasın)."""
        self._prev = None

    def sample(self) -> tuple[float, float] | None:
        try:
            st = self._system()
            procs = self._tree(self.root)
        except Exception:  # noqa: BLE001
            return None
        if st is None or procs is None:
            return None
        prev, self._prev = self._prev, (st, procs)
        if prev is None:
            return None
        dtot = st[1] - prev[0][1]
        if dtot <= 0:
            return None
        busy = max(0.0, (st[0] - prev[0][0]) / dtot * 100)
        # anahtar (pid, başlangıç): yeni süreç (ör. ffmpeg) bütün süresiyle sayılır, pid'i yeniden kullanan süreç karışmaz
        ours = sum(max(0, t - prev[1].get(key, 0)) for key, t in procs.items()) / dtot * 100
        ours = min(busy, ours)
        return max(0.0, busy - ours), ours

