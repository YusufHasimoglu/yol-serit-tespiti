# Şerit Analizi

**Akıllı Yol ve Şerit Tespiti**

Python/OpenCV ve PyTorch ile geliştirilmiş Türkçe arayüzlü, araştırma amaçlı **şerit çizgisi tespit projesi**. Tek geliştirme deposu üzerinden sürdürülür; kaynak kodu, testler ve belgeler burada tutulur.

## Özellikler

- Fotoğraf ve video açma, sonucu ve ara adımları kaydetme.
- Gündüz, gece, alacakaranlık ve viraj profili için **görüntüden tahmini sahne analizi**.
- Sol/sağ sürüş şeridi sınırlarına odaklanan **kesikli ve sürekli çizgi takibi**, eksik kesikli boyanın olası devamını farklı gösterme.
- Düz/kesik ayrımlı dört kanallı U-Net segmentasyon, UFLD V2, YOLOPv2 şerit + sürülebilir-yol maskesi, eğrisel renk takibi, klasik Canny–Hough ve otomatik seçim.
- Eğitim, etiketleme, değerlendirme, sentetik veri üretimi ve testler.
- Haftalık GitHub Actions araştırma eğitim işi; aday ağırlıklar otomatik olarak ana modelin yerine geçirilmez.

**Dikkat:** 'Bulundu' ifadesi geometrik bir aday bulunduğu anlamına gelir; doğruluk garantisi değildir. Sentetik verideki sonuçlar gerçek yol başarısı değildir. Gerçek araç sürüş/kontrol sistemi olarak kullanmayın.

## Windows masaüstü kurulumu

Git yüklü olmalı: https://git-scm.com/download/win
Python 3.10+ sürümünü kurun: https://www.python.org/downloads/windows/

PowerShell açın ve şu komutları çalıştırın (OneDrive masaüstü için de uygun):

```powershell
$desktop = [Environment]::GetFolderPath('Desktop')
$project = Join-Path $desktop 'Serit-Analizi'
if (-not (Test-Path $project)) {
    git clone https://github.com/YusufHasimoglu/yol-serit-tespiti.git $project
}
Set-Location $project
.\guncelle_ve_baslat.bat
```

Her sonraki çalıştırmada masaüstündeki `Serit-Analizi` klasöründen `guncelle_ve_baslat.bat` dosyasını açın. Bu dosya `git pull --ff-only` ile yeni commit'leri alır, gerekli Python paketlerini yükler ve uygulamayı başlatır. **Yerelde değiştirdiğiniz dosyalar varsa güncelleme çakışabilir**; kayıpsız çözüm için değişiklikleri commit veya yedekleyin.

Mevcut masaüstü projenizi yeniden klonlamanız gerekmez. Klasör adı farklı olsa da `guncelle_ve_baslat.bat` dosyası aynı depodan günceller. Klasörün adını değiştirmek isterseniz uygulamayı kapatıp Windows Gezgini'nde yeniden adlandırabilirsiniz.

## Geliştirici komutları

```bash
python -m pip install -r requirements-ai.txt
python app.py
python -m pytest -q
python egit_senaryo.py --help
python egit_gercek.py --help
python etiketle.py --help
```

Gerçek görüntülerle eğitim için uygulamadaki **EĞİTİM İÇİN ETİKETLE** düğmesini kullanın. Özgün fotoğraf yerelde `data/gercek/images/`, düz/kesik sınıflı maske ise `data/gercek/masks/` klasörüne kaydedilir. GitHub'a fotoğrafların veya üçüncü taraf veri kümelerinin ham kopyaları **konulmaz**. `hazirla_bdd100.py`, resmî BDD100K etiketlerinden düz/kesik sınıfları bulunan 100 farklı video ana karesini yerelde seçer; `egit.py --bdd-root ... --bdd-json ... --bdd-manifest ...` bu karelerle dört kanallı modeli eğitir. Ayrıntılar için eğitim rehberine bakın.

Ayrıntılar: [Kullanım ve arayüz](docs/KULLANIM.md) · [Gerçek veri deneyi](docs/GERCEK_VERI_DENEYI.md) · [AI eğitim rehberi](docs/AI_EGITIM_REHBERI.md) · [Deney planı](docs/DENEY_PLANI.md).

## UFLD V2 derin şerit modeli

Uygulamadaki **“Yapay zekâ (UFLD V2 derin şerit)”** seçeneği, UFLD V2'nin
satır-çapa temelli şerit algılayıcısını OpenCV ile yerelde çalıştırır. Eğrisel
ve kesikli çizgilerde klasik çizgi buluculara göre daha dayanıklı bir ikinci
yöntem sağlar. Modelin kendisi büyük olduğundan GitHub deposuna eklenmez;
uygulamanın `modeller/ufldv2_lane_detection.onnx` yolunda yerelde bulunur.
Bu model çizgi türünü (düz/kesik) söylemez; tür etiketi için kullanıcı
çizimleri ve dört kanallı U-Net kullanılmaya devam eder.

Model henüz yoksa, proje klasöründe aşağıdaki komutu bir kez çalıştırın:

```bash
python indir_ufld_v2.py
```

İndirme yaklaşık 368 MB'tır ve yalnızca kendi bilgisayarınıza kaydedilir.

## YOLOPv2 şerit ve sürülebilir-yol modeli

**“Gelişmiş AI (YOLOPv2 şerit + yol maskesi)”** seçeneği, BDD100K için
eğitilmiş YOLOPv2'nin şerit boyası ve sürülebilir-yol maskelerini birlikte
kullanır. Model maskesi, mevcut fiziksel boya ve şerit-geometrisi kontrolünü
daraltır; uygun bir çift çıkmazsa uygulama uydurma çizgi üretmek yerine
fiziksel boya yedeğine döner. Düz/kesik ayrımı yine gerçek boya boşluklarından
ölçülür.

Model henüz yoksa proje klasöründe bir kez çalıştırın:

```bash
python indir_yolopv2.py
```

İndirme yaklaşık 150 MB'tır ve `modeller/yolopv2_bdd100k.pt` olarak yalnızca
yerel bilgisayarınıza kaydedilir; GitHub'a yüklenmez.

### Kullanıcı düzeltmeleriyle geliştirme

Uygulamadaki **“YOLOPv2 GERİ BİLDİRİMİNİ EĞİT”** düğmesi; sizin doğru/hatalı
olarak kaydedip çizdiğiniz fotoğraf ve video karelerinden ayrı bir yerel
düzeltici model üretir. Bu katman YOLOPv2'nin şerit ve sürülebilir-yol
maskesine eklenir; sol/sağ ile düz/kesik ayrımını yerel görüntülerinize göre
iyileştirir. Resmî YOLOPv2 ağırlığı değişmez. En az 24 tam etiketli örnek
gerekir. İsterseniz aynı eğitimi komut satırından da başlatabilirsiniz:

```bash
python egit_yolopv2_geri_bildirim.py
```

Eğitilen ağırlık ve raporu yalnızca `modeller/` klasöründe bilgisayarınızda
kalır; GitHub'a yüklenmez.

## Yerel şerit örnek medyası

`hazirla_yol_serit_gorselleri.py`, mevcut yerel BDD100K verisinden 10 farklı
şeritli yol görselini `ornek_medya/gorseller/` klasörüne kopyalar. Video için
mevcut BDD100K indirici, aynı dizindeki `ornek_medya/videolar/` hedefine
çalıştırılabilir:

```bash
python hazirla_yol_serit_gorselleri.py
python indir_bdd_video_ornekleri.py --manifest data/external/bdd100k/bdd100_manifest.json --out ornek_medya/videolar --count 10
```

Bu örnekler yerelde tutulur ve GitHub'a yüklenmez. BDD100K kullanım koşulları
veri setinin resmî lisansına tabidir.
