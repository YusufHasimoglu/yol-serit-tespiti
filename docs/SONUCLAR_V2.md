# Şerit Analizi — Ön Test Sonuçları (01.11.2026)

## Model eğitimi

- Model: küçük U-Net, iki kanallı (sol ve sağ çizgi) piksel segmentasyonu, 256×144 giriş.
- Kaynak: 120 **sentetik** görüntü ve kullanıcının paylaştığı ekran görüntüsünün yol bölümünden türetilmiş **1 gerçek fotoğraf etiketi**.
- Doğrulama: 18 farklı sentetik örnek (eğitime dahil edilmedi).
- Son iyi checkpoint: 8. epoch, **sentetik doğrulama IoU ≈ 0.6524**.
- Eğitim CPU üzerinde yapıldı; weights dosyası eklendi.

### Önemli yorum

Sentetik doğrulama IoU'su, gerçek dünya başarı oranı değildir. Tek gerçek örnek eğitime katılmıştır ve demo için kullanılmıştır; bu nedenle gerçek görüntü üzerinde yapılan görsel test bağımsız performans değerlendirmesi değildir.

## Kullanıcı örneği (tekil ve eğitime dahil örnek)

- Kaynak: `ornekler/kullanici_yolu.png` (kullanıcının uygulama ekran görüntüsünden kırpıldı).
- Önceki Canny–Hough modu: sol ve sağ çizgi için **tespit yok**.
- Yeni eğrisel renk takibi: **iki çizgi adayı** bulundu.
- Yeni U-Net: **iki çizgi adayı** bulundu ve görüntü üzerinde çizildi.
- Nitel görsel: `ornek_sonuclar/once_sonra_v2.png`.
- Bu, *bir örnekteki görüntü kalitesi gözlemidir*; tespit doğruluğu yüzdesi değildir.

## Sentetik örnek video

- `ornekler/ornek_yol_video.mp4`: 36 kare, 12 FPS.
- Yeni otomatik mod: **36/36 karede iki çizgi adayı** buldu.
- CSV ve JSON raporları `ornek_sonuclar` klasöründe. Bu ölçüm, bağımsız gerçek yol doğruluğu değil, *algoritmanın çizgi tespit bildirimi oranıdır*.

## Sonraki gerçek bilimsel değerlendirme

1. Farklı yollardan etiketli görüntüler elde edin (örn. TuSimple). Video bazında eğitim/doğrulama/test ayrımı yapın.
2. Gerçek veride baseline Canny–Hough, eğrisel takip, AI ve otomatik modu kıyaslayın.
3. AI için sol IoU, sağ IoU, ortalama IoU; şerit çizgileri için nokta mesafesi/çizgi F1 gibi kriterler tanımlayın.
4. Gece, yağmur, viraj, kavşak ve silinmiş çizgiler için hata örneklerini ayrıca inceleyin.
5. Gerçek test sonuçlarını raporlamadan 'doğruluk %90+' gibi iddialar kullanmayın.

**Uyarı:** Bu çalışma akademik prototiptir, gerçek sürüşte kullanılmaz.