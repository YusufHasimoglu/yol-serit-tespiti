# Gerçek Yol Görüntüleriyle Deney Planı

## Amaç

Sentetik örneklerle yapılan ilk fonksiyon testlerini, manuel doğrulanmış gerçek görüntüler üzerinde güvenilir deneylerle genişletmek.

## Görüntü toplama

- Her koşul için en az 10 yol görüntüsü: açık gündüz, gölge, viraj, düşük ışık, silik çizgi.
- Kamera yüksekliği/yönü ve çözünürlüğü mümkün olduğunca not edilmeli.
- İzin/mahremiyet: kendi çekimleriniz veya lisansı uygun açık veriler kullanılmalı; görünür kişi/plaka kayıtlarını uygun şekilde ele alın.
- Aynı videodan birbirine çok benzeyen 100 kareyi bağımsız örnekmiş gibi saymayın.

## Etiketleme

İki kişinin görüntülerdeki geçerli sol ve sağ şerit sınırlarını en az iki yatay noktada işaretlemesi tercih edilir. Sol/sağ çizginin görünmediği kareler ayrıca işaretlenir. Bölümler kıyaslanabilir olacak şekilde seçilmelidir.

## Ölçütler

1. **Şerit görünürlük tespiti:** İnsan tarafından görünür olduğu doğrulanan çizgilerden kaç tanesi algoritma tarafından bulunmuştur?
2. **Yanlış pozitif oranı:** Şerit yokken başka yol işaretlerinin şerit sayılması.
3. **Konumsal hata:** İlgili satırda tahmin edilen x koordinatı ile etiket x koordinatının farkı (mutlak piksel hatası).
4. **Video titremesi:** Sabit sahnede alt uç x koordinatının kareler arası değişimi.
5. **İşleme süresi:** Bir kareyi işlemek için geçen ortalama süre ve kullanılan makine özellikleri.

Kameranın çözünürlüğüne göre bir tolerans belirlenmeli ve raporda açıkça yazılmalıdır. 'İki şerit var' statüsü tek başına doğruluk ölçüsü değildir.

## Deney akışı

1. Görüntüleri numaralandırın ve etiketleyin.
2. İlk varsayılan parametrelerle sonuçları kaydedin.
3. Belirlenen parametre aralığıyla deneyler yapın ve kullanılacak ayarları bir doğrulama alt kümesinde seçin.
4. Ayrı tutulan test alt kümesini son kez değerlendirin.
5. Hatalı örnekleri, gölge/viraj/silinti gibi nedenlere ayırın.
6. Görsel tablo, yöntem görselleri ve gerçek ölçümleri rapora ekleyin.

## Teslim kontrol listesi

- [ ] Kaynak kodlar ve kütüphane listesi
- [ ] Çalıştırma talimatları ve örnek video/görsel
- [ ] Ön işleme / ROI / Hough / final çıktı görselleri
- [ ] Farklı koşullardan gerçek test verileri
- [ ] Manuel etiket ve açık değerlendirme ölçütü
- [ ] Başarı / hata tablosu ve sınırlamalar
- [ ] Programın canlı gösterimi, kısa proje sunumu
