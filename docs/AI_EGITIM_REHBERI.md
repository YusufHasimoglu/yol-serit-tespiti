# ŞERİT GÖRÜ — Yapay Zekâ Eğitimi ve Değerlendirme Rehberi

## İçerik ve gerçek durum

Projede iki uyumlu U-Net biçimi vardır: önceki **iki kanallı** model sol/sağ şerit piksellerini, gerçek BDD100K etiketleriyle eğitilen **dört kanallı** model ise *sol düz*, *sol kesik*, *sağ düz* ve *sağ kesik* olasılıklarını verir. Varsayılan model dosyası: `modeller/serit_unet_demo.pt`.

Varsayılan iki kanallı model, prototip/demonstrasyon ağırlığıdır; sentetik doğrulama başarısı gerçek yol doğruluğu değildir. 100 gerçek BDD100K video ana karesiyle dört kanallı düz/kesik aday model de gerçekten eğitildi, ancak bağımsız doğrulama sonucu uygulama için yeterli olmadığından varsayılan modelin yerine geçirilmedi. Ölçümler, veri kapsamı ve video sınırı için [BDD100 eğitim sonucu](BDD100_EGITIM_SONUCU.md) belgesine bakın.

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

En kolay yol, uygulamada özgün fotoğrafı açtıktan sonra **EĞİTİM İÇİN ETİKETLE** düğmesine basmaktır. Açılan pencerede ekran görüntüsü değil özgün dosya kullanılır. Alternatif olarak komut satırından:

```bash
python etiketle.py --image C:/videolarim/yol1.jpg --name yol_001 --style-aware
```

Açılan pencerede **1** sol düz, **2** sol kesik, **3** sağ düz, **4** sağ kesik seçilir. Her tarafta yalnızca gerçek ego-şerit sınırını ve en az iki noktayı işaretleyin. Ardından **5** ile iki sınır arasındaki sürülebilir alanın çokgenini çizebilirsiniz; bu adımı atlarsanız alan iki çizgiden otomatik oluşturulur. **Sol tık** nokta koyar, **U** geri alır, **S** kaydeder, **Esc** çıkar. Virajlar için bol nokta kullanın. Eğitimde otomatik kullanılacak dosyalar:

```text
data/gercek/images/yol_001.png
 data/gercek/masks/yol_001_mask.png
```

Çizgi maskesi kayıpsız PNG olarak 1=sol düz, 2=sol kesik, 3=sağ düz, 4=sağ kesik sınıflarını taşır. Alan maskesi de ayrı olarak `data/gercek/areas/` altında saklanır; mevcut çizgi modeli çizgi maskesiyle eğitilir, alan maskesi ise görsel kontrol ve sonraki alan-segmentasyonu geliştirmesi içindir. **Yanlış etiketler modeli daha kötü yapabilir.** Eklediğiniz 1 fotoğrafı eğitim + test için tekrar kullanmayın. Birbirine çok benzeyen ardışık video karelerini eğitim/doğrulama arasında ayırmayın; aynı videonun tüm kareleri tek grupta kalmalı.

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

Düz/kesik türlerini de öğrenen aday model için, en az birkaç düz ve birkaç kesik etiket topladıktan sonra ayrı aday dosyası üretin:

```bash
python egit.py --style-aware --synthetic 240 --epochs 15 --batch 4 --out modeller/serit_unet_kullanici_candidate.pt
```

Aday ağırlığı, bağımsız test görsellerinde değerlendirilmeden varsayılan modelin yerine geçirilmez.

**Gerçek veriyle yüksek başarı istiyorsanız** en az farklı hava, aydınlatma, viraj ve kamera açılarından, birbirinden farklı yüzlerce doğru etiketlenmiş görüntü toplamanız gerekir. Buradaki sayı deney planı önerisidir, garanti değildir.

## 5. BDD100K ile düz/kesik gerçek veri eğitimi

BDD100K'nin şerit işaretleme etiketleri paralel şeritler için `laneStyle: solid|dashed` alanını içerir. Görüntü ana karesi, kaynak video klibinin 10. saniyesine karşılık gelir. Önce veriyi resmî kullanım koşullarına uygun biçimde yalnızca yerel diske indirin; ham görüntüleri, videoları ve etiketleri Git'e eklemeyin.

100 farklı kaynak videodan gelen, hem düz hem kesik şerit içeren ana kareleri seçmek için:

```bash
python hazirla_bdd100.py --labels data/external/bdd100k/labels/100k/train --out data/external/bdd100k/bdd100_manifest.json --count 100
```

Ardından 80/20 kaynak-kare eğitim/doğrulama bölmesiyle dört kanallı modeli eğitin:

```bash
python egit.py --synthetic 0 --bdd-root data/external/bdd100k --bdd-json data/external/bdd100k/labels/100k/train --bdd-manifest data/external/bdd100k/bdd100_manifest.json --limit 100 --epochs 18 --batch 4 --out modeller/serit_unet_demo.pt
```

Bu komuttaki 100 görüntünün her biri farklı BDD100K video klibinin etiketli ana karesidir. Tek-kare U-Net, etiketlenmemiş video karelerini denetimli eğitim örneği saymaz; gerçek video klipleri ayrıca zaman tutarlılığı deneyi için kullanılmalıdır. Eğitim raporundaki doğrulama IoU yalnızca ayrılmış 20 ana kareyi kapsar, sürüş güvenliği garantisi değildir.

## 6. TuSimple konum-etkiletli gerçek veri kümesi

Resmi format, görüntü yolu için `raw_file`, y koordinatları için `h_samples`, her şerit için de x koordinatlarını tutan `lanes` alanlarını kullanır; **-2** o noktada etiket yok demektir.

Kaynak ve indirme açıklaması: https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection

Verileri kendi kullanım şartlarına uygun şekilde edindikten sonra:

```bash
python egit.py --synthetic 0 --tusimple-root C:/TuSimple --tusimple-json C:/TuSimple/label_data_0313.json --epochs 20 --batch 4 --out modeller/serit_unet_gercek.pt
```

TuSimple yükleyici aracın merkezine en yakın sağ/sol iki şeridi basitçe seçer. Yön değiştirme, çok şeritli yol ve hatalı/eksik etiketler için elle kontrol yapın. Resmi `label_data_*.json` sürümünüzün doğru veri yolunu kullanın. **TuSimple verileri ve ağırlıkları proje ZIP'ine dahil değildir.**

## 7. Bağımsız test (IoU)

**Ayrı bir yol/gün/kamera çekiminden** gelen test görüntülerinizi `data/test/images/` ve `data/test/masks/` klasörlerine, yukarıdaki kurallarla kaydedin. Ardından:

```bash
python degerlendir.py --folder data/test --weights modeller/serit_unet_demo.pt --out test_sonuclari.json
```

Çıktı: sol IoU, sağ IoU, ortalama IoU. Sınıflar piksel seviyesinde ölçülür; TuSimple'ın resmi şerit başarı metriğine eşdeğer değildir. Sonuçlar ancak **eğitime katılmamış** test setinde raporlanmalıdır.

## 8. Sınırlamalar

- Sentetik eğitim görselleri fotogerçekçi değil; bir fotoğrafla aşırı uyum olabilir.
- Bulanık, gece, yağmur, kar, silinmiş çizgiler veya şerit birleşmelerinde hata olabilir.
- Merkez sapması yalnızca **piksel** cinsindendir. Kamera kalibrasyonu olmadan metreye çevrilmez.
- Otomatik modda sonuçta kullanılan gerçek yöntem alt bölümde gösterilir.
- 100 video ana karesi, farklı koşulları kapsayan başlangıç veri kümesidir; yağmur, gece, silinmiş boya ve Türkiye'ye özgü yol işaretleri için ayrıca bağımsız, doğru etiketli test gerekir.
- Bu bir **eğitim ve araştırma** yazılımıdır; gerçek araç güvenlik sistemi değildir.
