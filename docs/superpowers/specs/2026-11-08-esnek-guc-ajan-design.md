# Esnek güç 2 + ajanla otomatik çalışma (1.9.27) — tasarım

Kullanıcı onayı: 2026-11-08 ("Başla"). Kullanıcının sözleri: monitör Intel HD 630'da, oda soğuk (ısı sorunu yok);
"esnek güç modülünü bu iş için geliştir, sorunsuz çalışsın"; Hermes ajanı linkleri otomatik çekiyor, STT ajanla
otomatik çalışsın (`Myapplab/hermes-agent-`). Yaşanan sorun (soru cevabı): "esnek güç açıkken uygulamaları kapattı,
kendisi kapandı, bilgisayar kastı, işler durdu, sıradaki işlerin hepsi silindi".

## Kök neden (kodla doğrulandı)
- Esnek düzende 3-4 ağır süreç (3090 metin + ayırma, 3080 metin ya da ayırma). `flex.py` işlemciye ve boş RAM'e
  bakıyor, **sanal belleğe (commit) bakmıyor**; ikinci kart açılırken yalnız kartın belleğine bakılıyor. Kullanıcıda
  sanal bellek sınırı ~32 GB (1.8.7 ölçümü: RAM 24 + sayfa dosyası ~8). Commit dolunca Windows programları kapatır.
- İşlerin silinmesi: (a) `_health` bütün işçiler düşünce işleri "failed" sayıp `emit` ile kalıcı kuyruktan siliyor;
  (b) motor kapanırken (`close`) sıra bekleyen hazırlık "Durduruldu" olayıyla kalıcı kuyruktan siliniyor;
  (c) motor çalışmıyorken gelen işler yalnız bellekteki `_inbox`'ta (diske yazılmıyor).
- Hermes linki kaybediyor: STT meşgulken (eğitim / yeniden sınıflandırma) `/api/jobs` HTTP 200 + `ok: false`
  dönüyor; Hermes "gönderildi" sayıyor.

## Kararlar
1. **Bellek bilinçli esnek güç** (`flex.py`): boş sanal bellek < 4 GB ya da boş RAM < 2 GB → hemen kademe 1 (ikinci kart
   yeni iş almaz, boşta 20 sn sonra kapanır); sanal bellek < 2 GB ya da RAM < 1 GB → kademe 2 + yeni indirme durur.
   Sakin = işlemci < %20, RAM ≥ 3.5, sanal bellek ≥ 6 GB. Bellek koruması esnek kapalı / gece modunda da (yalnız
   indirme durdurma). İkinci kart (metin ya da ayırma) yalnız boş RAM ≥ 3.5 + kartın payı ve boş sanal bellek ≥ 6 +
   kartın payı iken açılır (pay ölçülür; inceleme sonrası: aksi hâlde aç-kapa döngüsü), bellek yüzünden kapanınca
   15-60 dk açılmaz.
2. **Kullanıcı yokken tam hız**: 5 dk fare / klavye yok (`GetLastInputInfo`) → işlemci yüzünden yavaşlama yok (bellek
   koruması sürer). Kullanıcı dönünce eski esnek davranış.
3. **Uzun çalışma**: işçiler `recycle_min` (180) dakikada bir boştayken yenilenir (gece modunda `night_recycle_min`);
   saatte bir `[verim]` günlük satırı (biten / hata / ses dakikası / en düşük boş bellek / kademe süreleri). İki kart +
   sanal bellek sınırı < 40 GB → bir kez sayfa dosyası uyarısı.
4. **İşler kaybolmaz**: iş eklenir eklenmez kalıcı kuyruğa; motor kapanırken / işçiler düşünce işler silinmez
   ("interrupted"); motor geri çekilmeli kendiliğinden açılır (60, 180, 600, 1800, 3600 sn) ve işler sürer; aynı işte
   3 kez kirli kesinti → "hata" (sonsuz döngü yok). Kullanıcının "Durdur"u kuyruğu temizler.
5. **Ajan modu** (`auto_mode`, Ayarlar'da tek düğme): Yerel API (8765), Windows açılınca küçültülmüş başlatma (HKCU Run,
   `--baslangic`), açılışta yarım işler kendiliğinden sürer. Son 24 saatte ≥ 2 kirli kapanış → kendiliğinden sürdürmez,
   API yeni iş almaz (503), kullanıcıya bildirir.
6. **API**: Host / Origin / Sec-Fetch-Site denetimi (web sayfaları iş ekleyemez); `/api/jobs` meşgulken 503, kanal linki
   422, boş 400; `/api/status` kuyruk / kabul / bellek / esnek / son bildirimler / verim; `/api/lookup` link → durum
   (done / failed / waiting / working / cancelled / interrupted / unknown, `retryable`).
7. **Kuyruk doluyken Güncelle**: motor durdurulur (kuyruk diskte), güncellenir, yeniden açılınca işler sürer.
8. **Arayüz**: 300'den çok işte çalışanlar + hatalar + son bitenler + sıradakilerin başı gösterilir.
9. **Hermes** (`stt_client.py`): STT kapalıysa açar ve API'yi bekler; `max_queue` (40) doluysa göndermez; 503'te linki
   "gönderildi" saymaz; gönderilenlerin sonucunu `/api/lookup` ile sorar, Telegram'a yazar, geçici hatayı bir kez
   yeniden gönderir.

Sonraki ayrı adım (kullanıcı istedi): eski sezonlar (GW / maç) + düzen `Premier League/<sezon>/<takım>/...`.
