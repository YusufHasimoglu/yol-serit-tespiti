# Şerit Görü — Akıllı Yol ve Şerit Analizi

Python/OpenCV ve PyTorch ile geliştirilmiş Türkçe arayüzlü, araştırma amaçlı **şerit çizgisi tespit projesi**. Tek geliştirme deposu üzerinden sürdürülür; kaynak kodu, testler ve belgeler burada tutulur.

## Özellikler

- Fotoğraf ve video açma, sonucu ve ara adımları kaydetme.
- Gündüz, gece, alacakaranlık ve viraj profili için **görüntüden tahmini sahne analizi**.
- Sol/sağ sürüş şeridi sınırlarına odaklanan **kesikli ve sürekli çizgi takibi**, eksik kesikli boyanın olası devamını farklı gösterme.
- U-Net segmentasyon, eğrisel renk takibi, klasik Canny–Hough ve otomatik seçim.
- Eğitim, etiketleme, değerlendirme, sentetik veri üretimi ve testler.
- Haftalık GitHub Actions araştırma eğitim işi; aday ağırlıklar otomatik olarak ana modelin yerine geçirilmez.

**Dikkat:** 'Bulundu' ifadesi geometrik bir aday bulunduğu anlamına gelir; doğruluk garantisi değildir. Sentetik verideki sonuçlar gerçek yol başarısı değildir. Gerçek araç sürüş/kontrol sistemi olarak kullanmayın.

## Windows masaüstü kurulumu

Git yüklü olmalı: https://git-scm.com/download/win
Python 3.10+ sürümünü kurun: https://www.python.org/downloads/windows/

PowerShell açın ve şu komutları çalıştırın (OneDrive masaüstü için de uygun):

```powershell
$desktop = [Environment]::GetFolderPath('Desktop')
$project = Join-Path $desktop 'Serit-Goru'
if (-not (Test-Path $project)) {
    git clone https://github.com/YusufHasimoglu/yol-serit-tespiti.git $project
}
Set-Location $project
.\guncelle_ve_baslat.bat
```

Her sonraki çalıştırmada masaüstündeki `Serit-Goru` klasöründen `guncelle_ve_baslat.bat` dosyasını açın. Bu dosya `git pull --ff-only` ile yeni commit'leri alır, gerekli Python paketlerini yükler ve uygulamayı başlatır. **Yerelde değiştirdiğiniz dosyalar varsa güncelleme çakışabilir**; kayıpsız çözüm için değişiklikleri commit veya yedekleyin.

## Geliştirici komutları

```bash
python -m pip install -r requirements-ai.txt
python app.py
python -m pytest -q
python egit_senaryo.py --help
python egit_gercek.py --help
python etiketle.py --help
```

Gerçek görüntülerle eğitim için fotoğrafları `data/gercek/images/`, etiket maskelerini `data/gercek/masks/` klasörüne yerleştirin. GitHub'a fotoğrafların ham kopyaları **konulmaz**. `data/gercek/annotations.json` yaklaşık çizgi işaretlerini içerir; bunlar kesin yer gerçeği değildir. Eğitim veri kümesi 5 örnek, ayrı kontrol kümesi 2 örnek içerdiğinden değerlendirme çok sınırlıdır. Denenen yeni U-Net kontrol noktası yeterli ilerleme göstermedi; mevcut model ağırlığı korundu.

Ayrıntılar: [Kullanım ve arayüz](docs/KULLANIM.md) · [Gerçek veri deneyi](docs/GERCEK_VERI_DENEYI.md) · [AI eğitim rehberi](docs/AI_EGITIM_REHBERI.md) · [Deney planı](docs/DENEY_PLANI.md).