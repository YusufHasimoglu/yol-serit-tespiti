# Şerit Görü — Gerçek Görüntü Deneyi ve Çizgi Devamlılığı

## Ne eklendi?

- Kullanıcının sağladığı **7 gerçek yol görüntüsü** gözle incelendi ve yaklaşık **ego-sürüş-şeridi** sol/sağ sınırları işaretlendi.
- Yolun ortasında bulunan **kesikli çizgi** de sürüş şeridinin sol sınırı olabilir; bariyer ve en soldaki yol kenarıyla karıştırılmaması için yeni geometrik izleme algoritması (`lanevision/continuity.py`) eklendi.
- Kesikli şeritlerin boyalı kısımları gözlenirken, **boşlukların olası devamı** kesikli turuncu çizgi ile sunulur. Bu çizgi doğrudan gözlem değildir.
- 'Kesikli + sürekli çizgi takibi' adlı ayrı bir yöntem eklendi. 'Akıllı otomatik' modunda da yeni algoritma bir adaydır.
- Görsel destek sayısı grafik arayüzünde gösterilir. İki sınır algılandı mesajı bunların doğru olduğu veya sürüş güvenliği için yeterli olduğu anlamına gelmez.

## Yeniden eğitim — gerçekleşen deney

- Yaklaşık elle çizilmiş maskeler: **5 eğitim** görüntüsü, **2 ayrı tutulan kontrol** görüntüsü.
- Gün batımı, viraj, düşük ışık ve sarı yol boyası içeren fotoğraflar bulunuyor.
- Önceki U-Net kontrol noktası ile **5 ek epoch** çalıştırıldı; fotoğraflara parlaklık, bulanıklık ve far parlaması benzeri artırmalar uygulandı. Sentetik örnekler de karıştırıldı.
- **Kontrol görüntülerinde iyileşme yok**: ortalama mask IoU yaklaşık **0,0071** kaldı (iki görüntü ile hesaplandığı için bilimsel genelleme yapılamaz).
- Kalite kontrolü başarısız olduğu için yeni ağırlıklar ana modelin üzerine yazılmadı; deneysel ağırlık `modeller/real_candidate.pt`, rapor `modeller/gercek_egitim_raporu.json`.

## Etiketler

- `data/gercek/annotations.json`: 7 görüntünün normalleştirilmiş, yaklaşık sol ve sağ çizgi noktaları + varsa kesikli boya aralıkları.
- `data/gercek/images`: kullanıcının örnek fotoğrafları; yalnızca kişisel eğitim/test paketi içinde tutulmalıdır.
- `data/gercek/masks`: aynı görüntülerin yaklaşık kırmızı/yeşil eğitim maskeleri.
- Bu etiketler elle yaklaşık hazırlanmıştır; değerlendirmede **doğrulanmış ground truth** kabul edilmemelidir.
- Üçüncü taraf fotoğraflar ve filigranlar, lisans doğrulaması yapılmadan herkese açık depoda dağıtılmamalıdır.

## Nasıl çalıştırılır?

```bash
python -m pip install -r requirements-ai.txt
python app.py
```

Arayüzde `Kesikli + sürekli çizgi takibi` seçeneğiyle yeni algoritmayı bağımsız test edin. `Akıllı otomatik` yöntemi farklı algoritmalara geçebilir.

Yeni görüntüler üzerinde alternatif eğitim (kalite kontrolünden geçerse isteğe bağlı ağırlık güncellemesi):

```bash
python egit_gercek.py --epochs 10 --real-repeats 20 --synthetic-per-scenario 20
# Yalnızca iyileşme varsa yayınlama işlemi için:
python egit_gercek.py --epochs 10 --real-repeats 20 --promote
```

*Uyarı:* Küçük ve benzer görüntülerle çok sayıda epoch yapmak ezberleme sorununu artırabilir. Daha çok **bağımsız etiketli gerçek video/kare** ve zor negatif görüntü gerekir. Gerçek araç kontrolünde kullanılmaz.