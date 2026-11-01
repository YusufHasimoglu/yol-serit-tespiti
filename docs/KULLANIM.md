# Şerit Analizi — akıllı sahne analizi ve profesyonel arayüz

## Otomatik tanıma

1. Görüntü açıldığında `lanevision/scene.py` ışık seviyesini (gece/gündüz/alacakaranlık) ve olası yol eğriliğini (viraj/düz/belirsiz) **görüntü işleme tabanlı sezgisellerle** tahmin eder.
2. Yapay zekâ (U-Net) modelinin eşik değeri ışık koşuluna uyarlanır. İki çizgi bulunduğunda geometrik tutarlılık kontrolü yapılır.
3. U-Net iki şerit adayı üretip geometrisi geçerliyse tercih edilir. Aksi durumda eğrisel renk takibi ve son çare Hough değerlendirilir.
4. Geometri güvenilir değilse arayüz **belirsiz** gösterebilir. Kötü ışık, bariyer ve yanıltıcı yol işaretleri nedeniyle bu analiz kusursuz değildir.
5. Görsel yüklenince **otomatik analiz** varsayılan açıktır; sol panelden kapatılabilir. Video için çıktı yolu seçildikten sonra kare kare analiz yapılır.

**Önemli ayrım:** Öğrenilmiş model *şerit segmentasyonu* yapar; gece/gündüz ve yol eğriliği seçici ise henüz **denetimli eğitilmiş bir sınıflandırıcı değil**, ışık/çizgi geometrisi analizidir. Bu iki başarı ölçütü birbirine karıştırılmamalıdır.

## Kurulum ve çalıştırma

```bash
python -m pip install -r requirements-ai.txt
python app.py
```

Windows için `baslat_ai_windows.bat` kullanılabilir. `modeller/serit_unet_demo.pt` model dosyası yerinde olmalıdır. UI ekranında `AI HAZIR` yazısı kütüphane ve ağırlık dosyasının varlığını gösterir; çıkarsama sırasında ortaya çıkan bir yükleme hatası halinde arayüz hata gösterebilir ve otomatik yöntem yedek algoritmaya döner.

## Gerçek yol başarısı neden halen sınırlı?

Model 256×144 giriş boyutunda küçük bir U-Net'tir. Sentetik görüntülerle eğitim yaptı; gerçek yol dağılımında genelleştirme yapması doğrulanmamıştır. Gece, ıslak zemin, kamera parlaması, keskin viraj, farklı bakış açıları ve kesik yol çizgileri için **gerçek etiketli eğitim verisi** gereklidir.

## Devam eden eğitim

`egit_senaryo.py` kontrol edilmiş yeni aday ağrlıklar üretir. GitHub Actions `.github/workflows/weekly-train.yml` haftalık veya elle tetiklenebilir. Bu iş akışı **sadece aday model ve metrik raporu Artifact olarak üretir; doğruluğu kanıtlanmamış modeli otomatik olarak ana modele yüklemez.** GitHub'ın zamanlayıcı hizmeti, depo aktifliği, Actions ayarları ve işlem kotasına bağlıdır; sürekli çalışan bir sunucu değildir.

Adayın kabulü için sabit sentetik doğrulama senaryolarında ortalama IoU kazanımı ve senaryo bazında en fazla `%5` düşüş sınırı kontrol edilir. Ayrıca ana modele geçirilmeden önce ayrı, daha önce görülmemiş **gerçek** yol test kümesi ile değerlendirme yapılmalıdır. GitHub'da Actions > Weekly lane-model research training > Run workflow menüsünden elle başlatabilirsiniz.

## Sentetik veri deneyleri

- Eğitim: 5 senaryodan 60'ar örnek = 300 yeni sentetik görsel, önceki modelden 4 ek epoch.
- Sentetik sabit doğrulama: 5 senaryodan 14'er örnek.
- Ortalama IoU: önce 0.4382, sonra 0.7046.
- Gerçek yol dış test başarısı: **henüz ölçülmedi**.

> Bu yazılım yalnızca üniversite araştırma/demonstrasyon amaçlıdır; gerçek araç kontrolü için kullanılmaz.