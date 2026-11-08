"""1.9.27: esnek güç 2 (bellek + kullanıcı yokken) ve ajanla (Hermes) otomatik çalışma; işler asla kaybolmasın."""
import time

from stt.config import Config
from stt.sources import Item


# ---------------------------------------------------------------- 1. kalıcı kuyruk
def test_queue_upsert_keeps_added_and_tries(tmp_path):
    from stt.state import State
    st = State(tmp_path / ".stt")
    st.queue_add("a", "url", "https://x/1", "t1", {"lang": "en", "_gen": 3})
    st.queue_add("b", "url", "https://x/2", "t2", {})
    first = {r["uid"]: r for r in st.queue_all()}
    assert first["a"]["extra"] == {"lang": "en"} and first["a"]["tries"] == 0 and first["a"]["started"] is None
    st.queue_started("a", "u_key_a")
    assert st.queue_bump("a") == 1                                      # kirli kesinti sayıldı, started temizlendi
    st.queue_started("a", "u_key_a")
    time.sleep(0.01)
    st.queue_add("a", "url", "https://x/1", "yeni başlık", {"lang": "tr"})   # yeniden ekleme (motora verilirken)
    rows = st.queue_all()
    assert [r["uid"] for r in rows] == ["a", "b"]                       # sıra korunur
    a = rows[0]
    assert a["added"] == first["a"]["added"] and a["tries"] == 1 and a["started"] and a["key"] == "u_key_a"
    assert a["title"] == "yeni başlık" and a["extra"] == {"lang": "tr"}
    st.queue_unstart_all()
    assert all(r["started"] is None for r in st.queue_all())
    assert st.queue_count() == 2


def test_queue_migrates_old_schema(tmp_path):
    import sqlite3

    from stt.state import State
    d = tmp_path / ".stt"
    d.mkdir()
    db = sqlite3.connect(d / "durum.sqlite")
    db.execute("CREATE TABLE queue(uid TEXT PRIMARY KEY, kind TEXT, source TEXT, title TEXT, extra TEXT, added REAL)")
    db.execute("INSERT INTO queue VALUES('old','url','https://x/9','eski','{}',1.0)")
    db.commit()
    db.close()
    st = State(d)                                                       # 1.9.26 kuyruğu: sütunlar eklenir, iş kalır
    r = st.queue_all()[0]
    assert r["uid"] == "old" and r["tries"] == 0 and r["started"] is None and r["key"] is None
    st.queue_started("old", "u_k")
    assert st.queue_bump("old") == 1
    State(d)                                                            # ikinci açılış: yeniden eklemeye çalışmaz


def test_closing_pipeline_keeps_waiting_items(tmp_path):
    from stt.pipeline import Pipeline
    p = Pipeline(Config(out_dir=tmp_path))
    ev = []
    p.events = ev.append
    it = Item("url", "https://www.youtube.com/watch?v=abcdefghijk", title="t")
    it.extra["uid"] = "w1"
    p.state.queue_add("w1", "url", it.source, "t", {})
    p._closing = True                                                   # motor kapanıyor (hata / yeniden açılış)
    p._prepare(it)
    assert [r["uid"] for r in p.state.queue_all()] == ["w1"]            # sıra bekleyen iş silinmez, sonra sürer
    assert not any(e["type"] == "failed" for e in ev)
    p.emit("failed", it, error="iç hata")                              # kapanırken düşen iş de silinmez
    assert p.state.queue_count() == 1
    p.emit("done", it)                                                  # biten iş her zaman çıkar
    assert p.state.queue_count() == 0
    p._closing = False
    p.state.queue_add("w1", "url", it.source, "t", {})
    it.extra["_cancel"] = True                                          # kullanıcının "Durdur"u: silinir
    p._prepare(it)
    assert p.state.queue_count() == 0 and any(e["type"] == "failed" and e.get("cancelled") for e in ev)
    p.close()


def test_prepare_marks_queue_started(tmp_path, monkeypatch):
    import stt.pipeline as pl
    p = pl.Pipeline(Config(out_dir=tmp_path))
    p.events = lambda e: None
    it = Item("url", "https://www.youtube.com/watch?v=abcdefghijk", title="t")
    it.extra["uid"] = "s1"
    p.state.queue_add("s1", "url", it.source, "t", {})
    seen = {}

    def fake_prepare(item, work, cfg):
        seen.update(p.state.queue_all()[0])
        raise pl.SkipItem("ses yok")
    monkeypatch.setattr(pl, "prepare_audio", fake_prepare)
    p._prepare(it)
    assert seen["started"] and seen["key"] == it.key                     # iş başladı: kirli kesintide deneme sayılır
    assert p.state.queue_count() == 0                                   # kalıcı hata: kuyruktan çıktı
    p.close()


def test_clean_close_does_not_count_as_try(tmp_path):
    from stt.pipeline import Pipeline
    p = Pipeline(Config(out_dir=tmp_path))
    p.state.queue_add("k1", "url", "https://x/1", "t", {})
    p.state.queue_started("k1", "u_x")
    p.close()                                                           # motor düzgün kapandı (ayar değişti)
    assert p.state.queue_all()[0]["started"] is None
    q = Pipeline(Config(out_dir=tmp_path))
    q.state.queue_started("k1", "u_x")
    q.kill()                                                            # uygulama kapandı / güncelleme
    assert q.state.queue_all()[0]["started"] is None
    q.close()


def test_enqueue_persists_before_engine(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    r = app.start("https://www.youtube.com/watch?v=abcdefghijk", [], {"lang": "en", "diarize": True})
    rows = app._state().queue_all()
    assert [x["uid"] for x in rows] == r["uids"] and rows[0]["extra"]["lang"] == "en"
    app2 = g.App()                                                      # motor hiç çalışmadan uygulama kapandı
    assert [x["uid"] for x in app2._pending] == r["uids"]


def test_cancel_all_clears_queue(tmp_path, monkeypatch):
    import stt.gui as g
    monkeypatch.chdir(tmp_path)
    app = g.App()
    r = app.start("https://www.youtube.com/watch?v=abcdefghijk\nhttps://www.youtube.com/watch?v=bbbbbbbbbbb", [], {})
    assert app._state().queue_count() == 2
    assert app.cancel(r["uids"][0])["ok"]
    assert [x["uid"] for x in app._state().queue_all()] == [r["uids"][1]]
    app._pending = [{"uid": "eski"}]
    app.cancel_all()                                                    # "Durdur": hiçbir şey sonra sürmesin
    assert app._state().queue_count() == 0 and app._pending == []
