"""Yapay zekâ adımları (isteğe bağlı): haber taslağı, özet, alıntılar, soru-cevap ayırma, konu filtresi.

Sağlayıcılar:
  gemini  - Google AI Studio anahtarı (GEMINI_API_KEY). Model boşsa hesabın erişebildiği en yeni 'flash' seçilir.
  claude  - Anthropic anahtarı (ANTHROPIC_API_KEY). Varsayılan claude-haiku-5-5 (ucuz, hızlı);
            daha iyi yazım için llm_model = "claude-sonnet-5-5".
  openai  - OPENAI_API_KEY.

Harcama: her çağrının token sayısı .stt/llm_kullanim.json'a yazılır; aylık tahmini harcama llm_budget_usd'yi
geçerse yapay zekâ adımları atlanır (metin yine yazılır). Fiyatlar TAHMİNİDİR; kesin tutar sağlayıcı panelindedir.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from .util import atomic_write_text, log

CLAUDE_DEFAULT = "claude-haiku-5-5"
OPENAI_DEFAULT = "gpt-5-mini"
# $ / 1M token (girdi, çıktı) — tahmin; bilinmeyen model için temkinli (yüksek) değer
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "flash-lite": (0.10, 0.40), "flash": (0.30, 2.50), "pro": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.40),
}


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


def _price(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    if m in PRICES:
        return PRICES[m]
    for k in ("flash-lite", "flash", "pro"):
        if k in m and "gemini" in m:
            return PRICES[k]
    return (3.0, 15.0)


class Usage:
    """Aylık token/harcama kaydı (tahmini)."""

    _lock = threading.Lock()

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "llm_kullanim.json"

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def month(self) -> dict:
        return self._load().get(time.strftime("%Y-%m"), {"usd": 0.0, "calls": 0, "in": 0, "out": 0})

    def add(self, model: str, tin: int, tout: int) -> float:
        pin, pout = _price(model)
        usd = tin / 1e6 * pin + tout / 1e6 * pout
        with self._lock:
            d = self._load()
            m = d.setdefault(time.strftime("%Y-%m"), {"usd": 0.0, "calls": 0, "in": 0, "out": 0})
            m["usd"] = round(m["usd"] + usd, 6)
            m["calls"] += 1
            m["in"] += tin
            m["out"] += tout
            atomic_write_text(self.path, json.dumps(d, indent=1))  # yarım yazılırsa sınır sıfırlanmasın
        return usd


class LLM:
    def __init__(self, cfg, state_dir: Path):
        self.cfg = cfg
        self.provider = cfg.llm_provider
        self.usage = Usage(state_dir)
        self._model = cfg.llm_model
        self._client = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------- durum
    def key(self, provider: str | None = None) -> str | None:
        """Sağlayıcının anahtarı. GEMINI_API_KEY / ANTHROPIC_API_KEY ortam değişkenleri config'e zaten okunur."""
        provider = provider or self.provider
        return {"gemini": self.cfg.gemini_api_key, "claude": self.cfg.anthropic_api_key,
                "openai": os.environ.get("OPENAI_API_KEY")}.get(provider or "")

    def available(self) -> bool:
        return bool(self.provider and self.key())

    def budget_left(self) -> float:
        return float(self.cfg.llm_budget_usd or 0) - self.usage.month()["usd"]

    def status(self) -> dict:
        m = self.usage.month()
        return {"provider": self.provider, "ready": self.available(), "model": self._model,
                "month_usd": round(m["usd"], 4), "calls": m["calls"], "budget": self.cfg.llm_budget_usd}

    # ---------------------------------------------------------- çağrı
    def complete(self, system: str, prompt: str, max_tokens: int = 4000, want_json: bool = False) -> str:
        if not self.available():
            raise LLMError("Yapay zekâ ayarlı değil (Ayarlar > Yapay zekâ).")
        if self.budget_left() <= 0:
            raise BudgetExceeded(f"Aylık yapay zekâ sınırı doldu (${self.cfg.llm_budget_usd}).")
        if want_json:
            system += "\nYanıtı YALNIZCA geçerli JSON olarak ver; açıklama ya da kod bloğu ekleme."
        fn = {"gemini": self._gemini, "claude": self._claude, "openai": self._openai}[self.provider]
        last: Exception | None = None
        for attempt in range(3):
            try:
                text, model, tin, tout = fn(system, prompt, max_tokens, want_json)
                usd = self.usage.add(model, tin, tout)
                log.info("Yapay zekâ (%s): %d+%d token, ~$%.4f", model, tin, tout, usd)
                return text
            except LLMError:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e).lower()
                if not any(x in msg for x in ("429", "rate", "overload", "500", "502", "503", "timeout", "unavailable")):
                    break
                time.sleep(5 * (attempt + 1))
        raise LLMError(f"Yapay zekâ çağrısı başarısız: {str(last)[:300]}")

    def complete_json(self, system: str, prompt: str, max_tokens: int = 4000):
        return parse_json(self.complete(system, prompt, max_tokens, want_json=True))

    # Gemini (REST: ek paket gerekmez)
    def gemini_model(self) -> str:
        if self._model:
            return self._model
        with self._lock:
            if self._model:
                return self._model
            import requests
            try:
                r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                                 headers={"x-goog-api-key": self.key()}, params={"pageSize": 200}, timeout=30)
                r.raise_for_status()
                names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                         if "generateContent" in (m.get("supportedGenerationMethods") or [])]
            except Exception as e:  # noqa: BLE001
                log.info("Gemini modelleri listelenemedi (%s); gemini-2.5-flash deneniyor", e)
                names = []
            self._model = pick_gemini(names) or "gemini-2.5-flash"
            log.info("Gemini modeli: %s", self._model)
            return self._model

    def _gemini(self, system, prompt, max_tokens, want_json):
        import requests
        model = self.gemini_model()
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3}}
        if want_json:
            body["generationConfig"]["responseMimeType"] = "application/json"
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          headers={"x-goog-api-key": self.key(), "Content-Type": "application/json"},
                          json=body, timeout=300)
        if r.status_code in (400, 401, 403):
            raise LLMError(f"Gemini reddetti (HTTP {r.status_code}): {r.text[:300]}")
        r.raise_for_status()
        d = r.json()
        cands = d.get("candidates") or []
        if not cands:
            raise LLMError(f"Gemini yanıt vermedi: {json.dumps(d.get('promptFeedback') or d)[:200]}")
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])
                       if not p.get("thought"))
        um = d.get("usageMetadata") or {}
        tout = int(um.get("candidatesTokenCount", 0)) + int(um.get("thoughtsTokenCount", 0))
        return text, model, int(um.get("promptTokenCount", 0)), tout

    # Claude (resmi anthropic SDK)
    def _claude(self, system, prompt, max_tokens, want_json):
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.key(), max_retries=3, timeout=600)
        model = self._model or CLAUDE_DEFAULT
        kw = dict(model=model, max_tokens=max(max_tokens, 8000), system=system,
                  messages=[{"role": "user", "content": prompt}], output_config={"effort": "low"})
        try:
            if model.startswith("claude-sonnet-5-5"):
                # sunucu tarafı yedek: güvenlik sınıflandırıcısı reddederse başka model devam eder
                resp = self._client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                         fallbacks="default", **kw)
            else:
                resp = self._client.messages.create(**kw)
        except anthropic.AuthenticationError:
            raise LLMError("Claude API anahtarı geçersiz.") from None
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"Claude API izni yok: {e.message}") from None
        except anthropic.NotFoundError:
            raise LLMError(f"Claude modeli bulunamadı: {model}") from None
        except anthropic.BadRequestError as e:
            raise LLMError(f"Claude isteği reddedildi: {e.message}") from None
        if resp.stop_reason == "refusal":
            raise LLMError("Claude bu metni işlemeyi reddetti.")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, model, resp.usage.input_tokens, resp.usage.output_tokens

    def _openai(self, system, prompt, max_tokens, want_json):
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(max_retries=3, timeout=600)
        model = self._model or OPENAI_DEFAULT
        kw = {}
        if want_json:
            kw["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens * 2, **kw)
        u = resp.usage
        return resp.choices[0].message.content or "", model, u.prompt_tokens, u.completion_tokens


def pick_gemini(names: list[str]) -> str | None:
    """Erişilebilen modellerden en yeni kararlı 'flash' (lite/görsel/ses/canlı olmayan)."""
    def ver(n: str) -> tuple:
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (float(m.group(1)) if m else 0.0,)
    bad = ("lite", "image", "tts", "audio", "live", "embedding", "thinking", "exp", "8b", "native", "robotics",
           "computer")
    cands = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(b in n for b in bad)]
    if not cands:
        return None
    stable = [n for n in cands if "preview" not in n]
    pool = stable or cands
    return max(pool, key=lambda n: (ver(n), "latest" not in n, -len(n)))


def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
        if m:
            return json.loads(m.group(1))
        raise


# ------------------------------------------------------------------ görevler

NEWS_SYSTEM = (
    "Sen deneyimli bir spor/haber editörüsün. Sana bir video dökümü (transkript) verilecek. "
    "Yalnızca dökümde geçen bilgiyi kullan; uydurma isim, skor, tarih ekleme. Alıntıları kelimesi kelimesine al. "
    "Türkçe yaz.")


def news_package(llm: LLM, doc: dict, transcript: str, lang_out: str = "Türkçe") -> dict:
    """Haber taslağı + özet + alıntılar + etiketler. Döner: dict (JSON)."""
    meta = {k: doc.get(k) for k in ("title", "webpage_url", "language", "duration_s")}
    ds = doc.get("dataset") or {}
    meta.update({k: ds.get(k) for k in ("team", "opponent", "competition", "gw", "phase", "presser_date", "manager")
                 if ds.get(k)})
    prompt = (
        f"Video bilgisi: {json.dumps(meta, ensure_ascii=False)}\n\n"
        f"Döküm:\n{transcript[:180_000]}\n\n"
        "Şu JSON'u üret:\n"
        '{"baslik": "haber başlığı (en çok 90 karakter)", "spot": "1-2 cümlelik giriş", '
        '"haber": "4-8 paragraflık haber metni (' + lang_out + ')", '
        '"ozet": ["en önemli 5-8 madde"], '
        '"alintilar": [{"kisi": "kim söyledi", "soz": "orijinal dilde birebir alıntı", "ceviri": "Türkçe çevirisi"}], '
        '"kisiler": ["geçen oyuncu/kişi adları"], "takimlar": ["geçen takımlar"], '
        '"sakatlik_kadro": ["sakatlık, ceza, kadro, rotasyon bilgileri (FPL için önemli)"], '
        '"etiketler": ["5-10 etiket"]}')
    return llm.complete_json(NEWS_SYSTEM, prompt, max_tokens=6000)


def news_markdown(pkg: dict, doc: dict) -> str:
    out = [f"# {pkg.get('baslik') or doc.get('title') or 'Haber'}", ""]
    if pkg.get("spot"):
        out += [f"**{pkg['spot']}**", ""]
    if pkg.get("haber"):
        out += [str(pkg["haber"]).strip(), ""]
    if pkg.get("ozet"):
        out += ["## Özet", ""] + [f"- {x}" for x in pkg["ozet"]] + [""]
    if pkg.get("sakatlik_kadro"):
        out += ["## Sakatlık / kadro", ""] + [f"- {x}" for x in pkg["sakatlik_kadro"]] + [""]
    if pkg.get("alintilar"):
        out += ["## Alıntılar", ""]
        for q in pkg["alintilar"]:
            if isinstance(q, dict):
                line = f"> \"{q.get('soz', '')}\" — {q.get('kisi', '?')}"
                if q.get("ceviri") and q.get("ceviri") != q.get("soz"):
                    line += f"\n>\n> _{q['ceviri']}_"
                out += [line, ""]
    if pkg.get("etiketler"):
        out += ["Etiketler: " + ", ".join(map(str, pkg["etiketler"])), ""]
    src = doc.get("webpage_url") or doc.get("source")
    if src:
        out += [f"Kaynak: {src}", ""]
    out += ["_Bu metin yapay zekâ ile dökümden hazırlanmış bir TASLAKTIR; yayından önce kontrol edin._", ""]
    return "\n".join(out)


QA_SYSTEM = (
    "Bir basın toplantısı dökümünü cümle cümle alacaksın. Her cümlenin teknik direktöre/oyuncuya mı (cevap) "
    "yoksa salondaki bir muhabire mi (soru) ait olduğunu belirle. Muhabir cümleleri genelde kısa sorulardır; "
    "'Thank you', 'Next question' gibi moderatör cümleleri muhabir sayılır.")


def _pick(llm: LLM, system: str, items: list[str], ask: str, key: str, step: int, max_tokens: int) -> list[bool]:
    """Numaralı listeyi parça parça yapay zekâya sor; dönen numaralar True olur."""
    out: list[bool] = []
    for a in range(0, len(items), step):
        part = items[a:a + step]
        numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(part))
        res = llm.complete_json(system, ask.replace("{items}", numbered), max_tokens=max_tokens)
        picked = res.get(key) if isinstance(res, dict) else res
        idx = {int(x) for x in picked or [] if str(x).isdigit()}
        out += [i in idx for i in range(len(part))]
    return out


def label_qa(llm: LLM, sentences: list[str]) -> list[bool]:
    """Her cümle için True = muhabir/soru. Konuşmacı ayırma başarısız olunca kullanılır."""
    return _pick(llm, QA_SYSTEM, sentences, '{items}\n\nJSON: {"muhabir": [muhabire ait cümle numaraları]}',
                 "muhabir", 400, 3000)


TOPIC_SYSTEM = "Video başlıklarını bir konuya göre süzüyorsun. Emin değilsen dahil etme."


def filter_titles(llm: LLM, titles: list[str], topic: str) -> list[bool]:
    ask = f"Konu: {topic}\n\nBaşlıklar:\n{{items}}\n\n" + '{"uygun": [konuya uyan başlık numaraları]} JSON ver.'
    return _pick(llm, TOPIC_SYSTEM, titles, ask, "uygun", 300, 2000)


def write_news(llm: LLM, json_path, doc: dict | None = None) -> dict:
    """'<ad> - haber.md' ve '<ad> - haber.json' yazar. Döner: {"news": Path, "news_json": Path}."""
    from .outputs import to_txt
    json_path = Path(json_path)
    doc = doc or json.loads(json_path.read_text(encoding="utf-8"))
    pkg = news_package(llm, doc, to_txt(doc))
    base = str(json_path.with_suffix(""))
    md, js = Path(base + " - haber.md"), Path(base + " - haber.json")
    md.write_text(news_markdown(pkg, doc), encoding="utf-8")
    js.write_text(json.dumps({**pkg, "kaynak": doc.get("webpage_url") or doc.get("source"), "metin": json_path.name},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Haber taslağı: %s", md)
    return {"news": md, "news_json": js}
