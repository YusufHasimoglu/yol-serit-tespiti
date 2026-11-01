# PROJE RAPORU TASLAĞI

**Proje:** Yol Görüntülerinden Şerit Çizgilerinin Tespiti  
**Alan:** Görüntü İşleme ve Bilgisayarlı Görü  
**Öğrenci:** [Ad Soyad]  
**Danışman:** [Akademik danışman]  
**Tarih:** [Teslim tarihi]

> Bu metin doğrudan düzenlenebilecek bir akademik taslaktır. Gerçek görüntülerde yapılmamış deneyler tamamlanmış gibi sunulmamalıdır.

## Özet

Bu çalışmada, ön kamera perspektifinde elde edilen yol görüntülerinden sağ ve sol şerit sınırlarını tespit etmek amacıyla klasik görüntü işleme tekniklerine dayanan bir prototip geliştirilmiştir. Sistem Python ve OpenCV kullanılarak hazırlanmış; görüntüler gri tona dönüştürülmüş, Gaussian filtre ile gürültü azaltılmış, Canny yöntemi ile kenarlar çıkarılmış ve ilgi bölgesi maskesi uygulanmıştır. Ardından olasılıksal Hough dönüşümü ile doğru parçaları elde edilmiş, eğim ve konumlarına göre sağ/sol olarak ayrılmış ve ağırlıklı doğrusal yaklaşımla şerit sınırları tahmin edilmiştir. Prototip, fotoğraf ve video işleme, ara aşamaları görüntüleme ve çıktıları kaydetme özelliklerine sahiptir. İlk işlevsel doğrulama sentetik test verisiyle yapılmıştır. Gerçek yol verisi üzerindeki başarının belirlenmesi için manuel etiketlenmiş örneklerle ayrı bir deney gerçekleştirilmesi gerekmektedir.

**Anahtar kelimeler:** Görüntü işleme, şerit tespiti, OpenCV, Canny, Hough dönüşümü, bilgisayarlı görü.

## 1. Giriş

Şerit çizgilerinin algılanması, sürücü destek sistemlerinin temel bilgisayarlı görü problemlerinden biridir. Kamera görüntüleri, ışık ve hava koşulları, kamera perspektifi ve yol çizgilerinin yapısı nedeniyle önemli değişkenlikler içerir. Bu projede düşük donanım maliyetiyle anlaşılabilir sonuç verebilen klasik görüntü işleme teknikleriyle bir prototip hazırlanmıştır. Çalışma güvenlik-kritik bir sürüş uygulaması değildir; eğitim ve araştırma amacına yöneliktir.

## 2. Problem Tanımı ve Hedefler

**Girdi:** Önden çekilmiş yol görüntüsü veya video.  
**Çıktı:** Sağ-sol tahmini şerit doğruları, uygun hâllerde renklendirilmiş alan, işlem basamakları ve basit görüntü istatistikleri.

Geliştirilen sistem, farklı görüntü işleme adımlarını görünür kılarak hatanın hangi aşamada oluştuğunu incelemeyi kolaylaştırır. Video üzerinde kareler arası doğrusal yumuşatma uygulanır.

## 3. Kullanılan Teknolojiler

- **Python:** Yazılımın genel akışı, dosya ve parametre yönetimi
- **OpenCV:** Filtreleme, Canny, Hough, çizim, video işleme
- **NumPy:** Matris hesapları, ağırlıklı doğru yaklaştırımı
- **Tkinter ve Pillow:** Masaüstü arayüz ve görüntü önizleme
- **Pytest:** Birim ve bütünleşik testler

## 4. Yöntem

### 4.1. Gri Tonlama ve Gürültü Azaltma

RGB/BGR görüntü tek kanallı gri görüntüye dönüştürülür. Daha sonra 5×5 Gaussian filtresi uygulanır. Bu adım küçük parlaklık değişikliklerinin oluşturduğu kenar gürültüsünü sınırlar.

### 4.2. Canny Kenar Tespiti

Bulanıklaştırılan görüntüde Canny kenar çıkarımı uygulanır. Varsayılan alt ve üst eşikler sırasıyla 55 ve 160'tır. Kenar maskesi, şerit işaretlerine ait olabilecek yüksek gradyanlı pikselleri içerir.

### 4.3. ROI (İlgi Bölgesi)

Yolun ön kamerada genel olarak görüntünün alt bölümünde yer aldığı varsayılır. Üst sınırı görüntü yüksekliğinin yaklaşık %59'unda bulunan dörtgen maske kullanılarak gereksiz kenarlar elenir.

### 4.4. Hough Doğru Tespiti

Olasılıksal Hough dönüşümü (`cv2.HoughLinesP`) uygulandıktan sonra bulunan doğru parçalarının eğimi hesaplanır. Görüntünün sol yarısındaki negatif eğimli parçalar sol, sağ yarısındaki pozitif eğimli parçalar sağ aday olarak sınıflandırılır. Varsayılan eşikler çekim türüne göre ayarlanabilir.

### 4.5. Çizgi Uydurma ve Görselleştirme

Her taraftan elde edilen çizgi uçlarına, noktalara segment uzunluğunun karekökü ile ağırlık verilerek `x = a·y + b` biçiminde doğru uydurulur. Çizgilerin görünür bölümünün uçları hesaplanır. İki geçerli sınır varsa araları şeffaf yeşil ile doldurulur; sınırlar sarı renkle işaretlenir.

Görüntü merkezinin yatay koordinatı `W / 2`, algılanan yol merkezinin alt sınırdaki koordinatı `(x_sol + x_sağ) / 2` olarak alınır. Görsel merkez sapması:

`d_px = W / 2 − (x_sol + x_sağ) / 2`

Bu değer yalnızca **görüntü düzleminde piksel cinsinden bir farktır**. Gerçek mesafe veya tehlike ölçümü sayılmaz.

### 4.6. Video İşleme

Kareler birer birer işlenir. Yeni karede bulunabilen çizgilerle önceki karedeki çizgiler doğrusal olarak yumuşatılır. İlgili önceki çizgi kaybolursa eski çizgi körlemesine görüntüye taşınmaz. Sonuç MP4'e kaydedilir, her karenin istatistikleri CSV, özet veriler JSON olarak yazılır.

## 5. Mimari

```text
Kullanıcı -> app.py / gui.py
                |
                v
           core.py
   (ROI, Canny, Hough, Fit)
                |
                v
         processing.py
       (Görsel / Video)
                |
                v
       PNG, MP4, CSV, JSON
```

Arayüzde uzun video işlemleri ayrı bir iş parçacığında çalışır. Ana iş parçacığı kullanıcı etkileşimini sürdürür; ilerleme bilgisi kuyruktan alınır.

## 6. Test ve Örnek Bulgular

**Sentetik testler:** Paket içindeki 1280×720 çözünürlüklü düz yol görüntüsünde iki sınır tespit edildi; görüntü merkezi sapması yaklaşık **−1,5 piksel** oldu. Düşük ışık için sentetik örnekte de her iki çizgi bulundu. 12 FPS ve 36 kareden oluşan sentetik video örneğinde **36 karenin tamamında** iki çizgi tespit edildi. Otomatik test süitindeki **8 test** başarıyla geçti.

**Yorum:** Bu sayılar yalnızca aynı araçla üretilmiş basit sentetik örneklerde algoritmanın teknik olarak çalıştığını kanıtlar. **Gerçek görüntülerdeki algılama doğruluğu henüz ölçülmemiştir.** Başarı yüzdesi, gerçek görüntüdeki şerit etiketlerine göre ayrıca hesaplanmalıdır.

### Eklenmesi gereken gerçek veri deneyi

| Test grubu | Görüntü adedi | Doğru tespit | Yanlış tespit | Açıklama |
|---|---:|---:|---:|---|
| Gündüz, düz yol | [ ] | [ ] | [ ] | [ ] |
| Virajlı yol | [ ] | [ ] | [ ] | [ ] |
| Gece / düşük ışık | [ ] | [ ] | [ ] | [ ] |
| Silik / kesikli çizgiler | [ ] | [ ] | [ ] | [ ] |
| TOPLAM | [ ] | [ ] | [ ] | [ ] |

Hangi ölçümün *doğru tespit* sayıldığını önceden tanımlayın. Örnek: y = 0,75H ve 0,90H satırlarında insan etiketine göre yatay hata ortalaması 20 pikselin altındaysa isabetli kabul edilebilir. Bu eşik, kamera çözünürlüğüne göre uyarlanmalıdır.

## 7. Sınırlamalar ve Gelecekteki Çalışmalar

Hough yönteminin doğrusal çizgi varsayımı keskin virajlar, karmaşık yol işaretleri ve gölgelere karşı kırılganlık oluşturur. ROI koordinatları kamera yerine bağımlıdır. Ayrıca algılanan çizgilerin bir şerit sınırı olduğu kesin değildir. Gelecek çalışmalarda perspektif dönüşümü, eğri çizgi için polinom uydurma, kameranın kalibrasyonu, gerçek veri etiketleriyle deneyler ve öğrenme tabanlı segmentasyon modelleri değerlendirilebilir.

## 8. Sonuç

Bu proje, görüntüler üzerinde aşamalı şerit çizgisi tespiti gerçekleştiren ve sonuçları görselleştirebilen bir eğitim prototipi sunmuştur. Fotoğraf/video uygulaması, açıklanabilir işlem adımları ve raporlama araçları oluşturulmuştur. Gerçek ortam performansının değerlendirilmesi için çeşitlendirilmiş ve manuel doğrulanmış veri kümesiyle ek deney zorunludur.

## Kaynakça (başlangıç, teslim öncesi kurum yazım kurallarına göre düzenleyin)

1. OpenCV. *Canny Edge Detection*. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
2. OpenCV. *Hough Line Transform*. https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
3. TuSimple. *Lane Detection Benchmark*. https://github.com/TuSimple/tusimple-benchmark

## Ekler

- Ek A: `ornek_sonuclar/serit_sonuc.png` — sentetik görsel çıktısı
- Ek B: `ornek_sonuclar/serit_sonuc_adimlar/` — 8 görüntü işleme aşaması
- Ek C: `ornek_sonuclar/ornek_video_sonuc_istatistik.csv` — video kare istatistikleri
- Ek D: `ornek_sonuclar/arayuz_onizleme.png` — uygulama arayüzü
