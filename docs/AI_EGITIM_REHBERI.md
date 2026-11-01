# ŞERİT GÖRÜ — Yapay Zekâ Eğitimi ve Değerlendirme Rehberi

## İçerik ve gerçek durum

Projeye **PyTorch ile eğitilebilen küçük bir U-Net benzeri iki sınıflı segmentasyon modeli** eklendi. İki kanal sırasıyla *sol* ve *sağ* şerit piksellerini gösterir. Varsayılan model dosyası: `modeller/serit_unet_demo.pt`.

**Model gerçekten eğitildi**, ancak bu yalnızca **prototip/demonstrasyon** modelidir. 120 sentetik sahne oluşturuldu, bunlardan 18'i ayrı sentetik doğrulama için kullanıldı. Ayrıca kullanıcının arayüz ekran görüntüsünün yalnızca yol bölümü elle çizilmiş kaba etiketle **eğitime** eklendi. Son kaydedilen ağırlıklar 8. turdan, sentetik doğrulama **IoU ≈ 0,6524**. Bu rakam **gerçek yol fotoğraflarında doğruluk değildir**; tek gerçek fotoğraf üzerinde yapılan gösterim eğitim verisi içerdiği için bağımsız test sayılmaz. Model birçok başka yolda hata yapabilir.

## 1. Kurulum

Python 3.10+ önerilir (PyTorch'un desteklediği sürümlerden biri olmalı).

```bash
python -m pip install -r requirements-ai.txt
python app.py
```

Windows için `baslat_ai_windows.bat` dosyasını açabilirsiniz. Başlangıç `.bat` dosyası PyTorch yüklemez; yalnızca klasik/gelişmiş mod çalışır, otomatik mod buna düşer.

## 2. Yöntem seçimi

- **Otomatik (AI + eğrisel yedek):** Model dosyası varsa AI ile çalışır; iki çizgi bulunamazsa veya geometri çok ters görünürse piksel renk takip algoritmasına; gerekirse Hough'a geçer.
- **Yapay zekâ (U-Net):** Yalnızca AI maskelerinden çizgi çıkarır. Öğrenilmiş model yoksa *hata* verir, sahte AI sonucu gösterilmez.
- **Gelişmiş eğrisel takip:** Beyaz/sarı çizgiler, hareketli pencere takibi, polinom uydurma; PyTorch gerekli değil.
- **Klasik Canny + Hough:** Önceki projedeki algoritma; yöntemler arası karşılaştırma için korunmuştur.

AI sonuçlarında `09 AI olasilik haritasi` (kırmızı=sol / yeşil=sağ) ve `10 AI sinif maskeleri` işlenme görselleri kaydedilebilir.

## 3. Gerçek yol fotoğrafını etiketleme

```bash
python etiketle.py --image C:/videolarim/yol1.jpg --name yol_001
```

Açılan pencerede **1** sol şerit (kırmızı), **2** sağ şerit (yeşil), **sol tık** nokta koyma, **U** geri alma, **S** kaydetme, **Esc** çıkıştır. Virajlar için bol nokta kullanın. Eğitimde otomatik kullanılacak dosyalar:

```text
data/gercek/images/yol_001.png
 data/gercek/masks/yol_001_mask.png
```

Arka plan siyah, sol çizgi kırmızı (BGR kanal R), sağ çizgi yeşil (G) olmalıdır. **Yanlış etiketler modeli daha kötü yapabilir.** Eklediğiniz 1 fotoğrafı eğitim + test için tekrar kullanmayın. Birbirine çok benzeyen ardışık video karelerini eğitim/doğrulama arasında ayırmayın; aynı videonun tüm kareleri tek grupta kalmalı.

## 4. Modeli yeniden eğitme

Örnek (120 sentetik + kendi etiketli gerçek fotoğraflarınız):

```bash
python egit.py --synthetic 120 --epochs 12 --batch 8 --out modeller/serit_unet_demo.pt
```

Daha önce eğitilmiş ağırlıklarla devam:

```bash
python egit.py --synthetic 120 --epochs 5 --batch 8 --resume
```

CUDA destekli PyTorch sürümü kurulmuşsa otomatik CUDA seçilir, değilse CPU ile eğitir. RTX 3050 Ti 4 GB için **`--batch 2` veya `--batch 4`** kullanmanız daha güvenli olabilir. Tüketilen VRAM sürüme ve görüntü boyutuna bağlıdır.

**Gerçek veriyle yüksek başarı istiyorsanız** en az farklı hava, aydınlatma, viraj ve kamera açılarından, birbirinden farklı yüzlerce doğru etiketlenmiş görüntü toplamanız gerekir. Buradaki sayı deney planı önerisidir, garanti değildir.

## 5. TuSimple etiketli gerçek veri kümesi

Resmi format, görüntü yolu için `raw_file`, y koordinatları için `h_samples`, her şerit için de x koordinatlarını tutan `lanes` alanlarını kullanır; **-2** o noktada etiket yok demektir.

Kaynak ve indirme açıklaması: https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection

Verileri kendi kullanım şartlarına uygun şekilde edindikten sonra:

```bash
python egit.py --synthetic 0 --tusimple-root C:/TuSimple --tusimple-json C:/TuSimple/label_data_0313.json --epochs 20 --batch 4 --out modeller/serit_unet_gercek.pt
```

TuSimple yükleyici aracın merkezine en yakın sağ/sol iki şeridi basitçe seçer. Yön değiştirme, çok şeritli yol ve hatalı/eksik etiketler için elle kontrol yapın. Resmi `label_data_*.json` sürümünüzün doğru veri yolunu kullanın. **TuSimple verileri ve ağırlıkları proje ZIP'ine dahil değildir.**

## 6. Bağımsız test (IoU)

**Ayrı bir yol/gün/kamera çekiminden** gelen test görüntülerinizi `data/test/images/` ve `data/test/masks/` klasörlerine, yukarıdaki kurallarla kaydedin. Ardından:

```bash
python degerlendir.py --folder data/test --weights modeller/serit_unet_demo.pt --out test_sonuclari.json
```

Çıktı: sol IoU, sağ IoU, ortalama IoU. Sınıflar piksel seviyesinde ölçülür; TuSimple'ın resmi şerit başarı metriğine eşdeğer değildir. Sonuçlar ancak **eğitime katılmamış** test setinde raporlanmalıdır.

## 7. Sınırlamalar

- Sentetik eğitim görselleri fotogerçekçi değil; bir fotoğrafla aşırı uyum olabilir.
- Bulanık, gece, yağmur, kar, silinmiş çizgiler veya şerit birleşmelerinde hata olabilir.
- Merkez sapması yalnızca **piksel** cinsindendir. Kamera kalibrasyonu olmadan metreye çevrilmez.
- Otomatik modda sonuçta kullanılan gerçek yöntem alt bölümde gösterilir.
- Bu bir **eğitim ve araştırma** yazılımıdır; gerçek araç güvenlik sistemi değildir.