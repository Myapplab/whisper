# Esnek güç 2 + ajanla otomatik çalışma (1.9.27) — uygulama planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** STT, Hermes'in göndereceği binlerce linki günlerce kendi başına, belleği taşırmadan ve hiçbir işi kaybetmeden işlesin.

**Architecture:** Bellek koruması `stt/flex.py`'deki `Flex` kademelerine eklenir (saf mantık), `Pipeline` onu uygular.
Kalıcı kuyruk (`State.queue`) tek doğruluk kaynağı olur; motor ölürse App (`gui.py`) geri çekilmeyle yeniden açar ve
kuyruktan sürdürür. Ajan modu = API + Windows açılışı (`stt/autostart.py`) + kendiliğinden sürdürme. Hermes tarafı
ortak istemci `scripts/stt_client.py`.

**Tech Stack:** Python 3.11 (Windows), ctypes (GetLastInputInfo, GlobalMemoryStatusEx), winreg, http.server, pytest; Hermes betikleri Python 3.12 + urllib.

**Spec:** `docs/superpowers/specs/2026-11-08-esnek-guc-ajan-design.md`

## Global Constraints
- Yanıtlar / kullanıcı metinleri Türkçe; günlük önekleri `[esnek]`, `[verim]`, `[motor]`, `[ajan]`.
- Monitörlü karta kendiliğinden yük yok; konuşmacı ayırma yalnız GPU'da; `diar_cpu` false kalır.
- Anahtar / token repoya yazılmaz. `installer/guncelle.ps1` ASCII, `kur.ps1` UTF-8 BOM (dokunulmuyor).
- GUI kuralı: olaylar yalnız `App._push`; `App`'in `_` ile başlamayan metotları JS'e açık.
- Bütün testler geçmeden push yok (`python -m pytest -q tests`). Commit mesajında model adı yok.
- Sürüm 1.9.27 (`stt/__init__.py`); `pyproject.toml` değişmez (güncellemede pip çalışmasın).

## Review Focus
- Motor kapanırken / işçiler düşerken kalıcı kuyruktan iş silinmesi → Task 1-2 testleri.
- Aynı işin iki kez sıraya girmesi (inbox + kuyruktan sürdürme) → Task 2 `_resume` testi.
- Bellek hiç toparlanmazsa iş sonsuza dek durmasın: yalnız indirme durur, çalışan işler biter → Task 3 testi.
- Hermes'in 503 / ulaşılamadı / kuyruk dolu durumlarında linki "gönderildi" saymaması → Task 7 testleri.
- Windows'a özgü çağrılar (winreg, user32) Linux'ta ve hata verince sessizce "bilinmiyor" dönsün → Task 3, 5 testleri.

---

### Task 1: Kalıcı kuyruk sağlamlaştırma
**Files:** Modify `stt/state.py`, `stt/pipeline.py` (`emit`, `_prepare`, `kill`, `close`), `stt/gui.py` (`_enqueue`, `cancel_all`, `cancel`). Test: `tests/test_v1927.py`.
**Interfaces (Produces):** `State.queue_add` UPSERT (`added`, `tries`, `started` korunur); `State.queue_started(uid, key)`,
`State.queue_unstart_all()`, `State.queue_bump(uid) -> int` (tries+1, started=NULL), `State.queue_count() -> int`;
`queue_all()` satırlarında `key`, `started`, `tries`.
- [ ] Testler: `test_queue_upsert_keeps_added_and_tries`, `test_queue_migrates_old_schema`,
  `test_closing_pipeline_keeps_waiting_items`, `test_enqueue_persists_before_engine`, `test_cancel_all_clears_queue`.
- [ ] Uygula: eski tabloya `ALTER TABLE queue ADD COLUMN key TEXT / started REAL / tries INTEGER DEFAULT 0`; `emit`:
  `_closing` iken failed/skipped kuyruktan silmez; `_prepare`: sıra beklerken kapanış → sessiz dönüş, işe başlayınca
  `queue_started`; `kill`/`close` → `queue_unstart_all` (temiz kapanış deneme sayılmaz); `App._enqueue` → `queue_add`;
  `cancel_all` → `queue_clear` + `_pending=[]`; `cancel` → `queue_remove`.
- [ ] `python -m pytest -q tests` → hepsi geçer. Commit.

### Task 2: Motor durunca işler sürsün
**Files:** `stt/pipeline.py` (`_health`), `stt/gui.py` (`_runner`, `_resume`, `resume_pending`, `_after_engine_start`), `stt/gui.html` ("interrupted"). Test: `tests/test_v1927.py`.
**Interfaces:** `Job.status` yeni değer `"interrupted"`; olay `{"type": "interrupted", uid, error}`; `gui.ENGINE_RETRY_S = (60, 180, 600, 1800, 3600)`,
`gui.MAX_RESUME_TRIES = 3`; `App._resume(rows: list[dict], why: str) -> int`.
- [ ] Testler: `test_all_workers_dead_interrupts_without_deleting`, `test_resume_skips_inbox_pending_and_fails_poison`,
  `test_runner_restarts_engine_and_resumes`.
- [ ] Uygula: `_health` hepsi ölünce işleri `interrupted` (state "interrupted", kuyrukta kalır) + `RuntimeError`;
  `_runner` hata → `_wait_recover(n)` (`_restart` ya da süre dolunca, iş varsa ya da ajan modundaysa), motor ≥ 30 dk
  çalıştıysa sayaç sıfırlanır; yeniden açılınca kuyrukta olup inbox'ta / `_pending`'de olmayanlar `_resume`;
  `_resume`: `started` doluysa `queue_bump`, ≥ 3 → kalıcı hata ("üst üste 3 kez durdu").
- [ ] Testler + tam test. Commit.

### Task 3: Esnek güç 2 (bellek + kullanıcı yokken)
**Files:** `stt/flex.py`, `stt/pipeline.py` (`_flex_tick`, `_pending_ahead_ok`, `_helper_check`, `_helper_diar_dispatch`, `start`). Test: `tests/test_v1927.py`.
**Interfaces:** `Flex.update(others, ram_free, now, commit_free=None, idle_s=None) -> str`; öznitelikler `ram_low`
(RAM ya da sanal bellek az), `mem_crit`, `away`, `commit`; sabitler `COMMIT_LOW_GB=4.0`, `COMMIT_CRIT_GB=2.0`,
`COMMIT_OK_GB=6.0`, `RAM_CRIT_GB=1.0`, `AWAY_S=300.0`; `flex.user_idle_s() -> float | None`;
`pipeline.HELPER_OPEN_RAM_GB=3.0`, `HELPER_OPEN_COMMIT_GB=6.0`, `PAGEFILE_DUAL_GB=40.0`; `Pipeline._memstat`, `Pipeline._idle`.
- [ ] Testler: `test_flex_memory_escalates_at_once`, `test_flex_away_ignores_cpu_not_memory`,
  `test_flex_calm_needs_commit`, `test_user_idle_unknown_off_windows`, `test_mem_crit_pauses_prep_even_flex_off`,
  `test_helper_not_opened_when_memory_low`, `test_hdiar_not_opened_when_memory_low`, `test_pagefile_notice_two_cards`.
- [ ] Uygula; bellek kaynaklı yükselişte saatte en çok bir bildirim (sayfa dosyası ipucu, sınır < 48 GB ise).
- [ ] Testler + tam test. Commit.

### Task 4: Uzun çalışma: işçi yenileme + verim
**Files:** `stt/config.py` (`recycle_min`), `stt/pipeline.py` (`_recycle_tick`, `stats`, `_stats_tick`). Test: `tests/test_v1927.py`.
**Interfaces:** `Config.recycle_min: float = 180.0` (0 kapalı, 20-1440); `Pipeline.stats() -> dict`
(`done_1h, failed_1h, audio_min_1h, done_total, failed_total`).
- [ ] Testler: `test_recycle_runs_outside_night_mode`, `test_recycle_parks_old_idle_helper`, `test_stats_and_hourly_line`.
- [ ] Uygula (`_night_recycle` → `_recycle(now, mins)`; pump adımı `_recycle_tick`). Testler. Commit.

### Task 5: Ajan modu, Windows açılışı, kuyruk doluyken güncelleme
**Files:** Create `stt/autostart.py`; modify `stt/config.py` (`auto_mode`), `stt/gpuguard.py` (`note_unclean_end`), `stt/gui.py`, `stt/gui.html`. Test: `tests/test_v1927.py`.
**Interfaces:** `autostart.VALUE = "STT Basin Odasi"`, `autostart.command() -> str`, `autostart.status() -> dict`
(`supported, enabled, approved, command`), `autostart.set_enabled(on: bool) -> str` (hata metni ya da ""),
`autostart.ensure() -> None`; `gpuguard.note_unclean_end(state_dir, now=None) -> int` (son 24 saatteki kirli kapanış
sayısı); `App.set_auto_mode(on) -> dict`; `App._auto_paused: str`.
- [ ] Testler: `test_autostart_toggle_with_fake_registry`, `test_set_auto_mode_opens_api_and_autostart`,
  `test_auto_resume_on_first_start`, `test_crash_loop_pauses_auto_resume`, `test_update_while_busy_keeps_queue`,
  `test_gui_html_script_parses` (node varsa).
- [ ] Uygula: `--baslangic` → küçültülmüş pencere; ajan modunda açılışta `ensure`; arayüz: Ayarlar'da düğme + durum,
  "pending" olayı, 300+ işte kart sınırı.
- [ ] Testler. Commit.

### Task 6: Yerel API güvenlik + doğru durum kodları + status / lookup
**Files:** `stt/api.py`, `stt/gui.py` (`api_status`, `lookup`, `retryable_error`). Test: `tests/test_v1927.py`.
**Interfaces:** `/api/jobs` 200 `{ok, count, uids}` / 400 / 422 / 503 `{error, retry: true}`; `/api/status` ek alanlar
`queue {total, working, waiting}`, `accepting`, `busy`, `auto_mode`, `paused`, `flex`, `memory`, `notices`, `stats`;
`POST /api/lookup {"urls": [...]}` ve `GET /api/lookup?url=` → `{"items": [{url, key, status, title, out_dir, txt, json, error, retryable, updated}]}`.
- [ ] Testler: `test_api_rejects_browser_and_rebinding`, `test_api_jobs_busy_is_503`, `test_api_jobs_bad_input_4xx`,
  `test_api_status_queue_fields`, `test_api_lookup_statuses`.
- [ ] Uygula. Testler. Commit.

### Task 7: Hermes tarafı (`/home/user/hermes-agent-`)
**Files:** Create `skills/fpl/press-conference-finder/scripts/stt_client.py`, `tests/test_stt_client.py`; modify `find_press_conferences.py`, `backfill.py`, `config.example.json`, `SKILL.md`, `README.md`, `HANDOFF.md`.
**Interfaces:** `SttBusy`, `SttUnreachable`; `ensure_running(stt) -> dict`, `room(status, stt) -> int`,
`send(stt, url) -> str`, `lookup(stt, urls) -> list[dict]`, `report(stt, state) -> list[str]`.
- [ ] Testler (sahte STT sunucusu): 503 → gönderildi sayılmaz; ulaşılamaz → başlat + bekle; kuyruk dolu → gönderim sınırlı;
  rapor: biten işaretlenir, geçici hata bir kez yeniden gönderilir, kalıcı hata raporlanır.
- [ ] Uygula. Testler. Commit + push (hermes main).

### Task 8: Belgeler, sürüm, inceleme, push
- [ ] `stt/__init__.py` 1.9.27; `docs/GELISTIRME.md` "1.9.27"; `docs/YAPILACAKLAR.md` Son durum + SIRADAKİ (eski sezon +
  `Premier League/<sezon>/<takım>`); `CLAUDE.md` haritası (`autostart.py`, ajan modu).
- [ ] Tam test; bağımsız inceleme ajanı; bulguları testle düzelt; push (dal + main).
