# Yapay zekâ modu neden çalışmıyor?

Arayüzde **`AI modu: pip install -r requirements-ai.txt`** ve **`Kullanılan yöntem: Klasik Hough (yedek)`** yazıyorsa AI modeli çalışmıyordur. Klasik algoritmanın hatalı çizgileri AI başarısı gibi yorumlanmamalıdır.

## Windows kurulumu

Proje klasöründe terminal açın:

```bat
py -m pip install -r requirements-ai.txt
py -c "import torch; print('PyTorch:', torch.__version__)"
py app.py
```

Eğer `py` komutu yoksa `python` kullanın. Uygulamayı çalıştırdığınız Python ortamıyla kütüphane yüklediğiniz ortamın **aynı** olmasına dikkat edin.

`requirements-ai.txt` kurulumu yeterli değildir: eğitilmiş ağırlık dosyası da modelin beklediği yerde olmalıdır. Kaynak kod deposundan indirdiyseniz tam AI V2 paketindeki `modeller/serit_unet_demo.pt` dosyasını `modeller/` dizinine yerleştirin. Eğitim yapmadan bu ağırlık dosyası mevcut değilse AI çıkarımı yapılamaz.

## Değerlendirme

1. Arayüzde doğrudan **Yapay zekâ (U-Net)** modunu seçin. Hata çıkarsa mesajı kaydedin; otomatik yedek sonuçları ile karıştırmayın.
2. İşlem adımlarında **09 AI olasılık haritası** ve **10 AI sınıf maskeleri** çıktılarını inceleyin.
3. Çizgiler bariz yanlışsa gerçek yol görüntülerini ve doğru sol/sağ çizgi etiketlerini ayrı bir veri kümesinde toplayın.
4. Eğitim verisinden tamamen ayrı test kümesinde sınıf bazlı IoU, precision, recall ve görsel hata örneklerini raporlayın.
5. Başlangıç eğitiminde kullanılan sentetik doğrulama puanlarını gerçek yol başarısı diye sunmayın.

**Not:** Bu uygulama araştırma/öğrenme amaçlıdır; araç kontrolü için güvenlik onayı yoktur.
