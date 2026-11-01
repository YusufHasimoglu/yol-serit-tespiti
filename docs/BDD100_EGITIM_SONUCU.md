# BDD100K düz/kesik eğitim deneyi

Bu deneyde BDD100K'nin resmî şerit etiketlerinden, **100 farklı kaynak video**nun 10. saniyedeki etiketli ana kareleri seçildi. Her seçilen kayıtta uygulamanın sol ve sağ ego-şerit hedefi oluşturulabildi ve seçilmiş ikilide hem `solid` hem `dashed` türü yer aldı. Ham görüntüler, video arşivleri, etiketler ve aday ağırlıklar yalnızca `data/external/` ile `modeller/` altında yerelde tutulur; bu depoya eklenmez.

## Eğitim

- Çıkışlar: sol-düz, sol-kesik, sağ-düz, sağ-kesik
- Eğitim/doğrulama: 72 / 28 farklı video ana karesi
- Tur sayısı: 20 (CPU, batch 16)
- En iyi eğitim içi doğrulama IoU: 0,0224 (19. tur, 0,42 eşiği)
- Ayrı eşik taramasındaki en iyi ortalama IoU: 0,1266 (0,70 eşiği)
- Ayrı eşik taramasında sol-kesik IoU: 0,0000

Bu skorlar adayın özellikle kesik sol şeritte güvenilir olmadığını gösterir. Bu nedenle `serit_unet_bdd100_candidate.pt` yalnızca yerel bir araştırma adayı olarak saklanır; uygulamanın varsayılan `serit_unet_demo.pt` ağırlığı bu deneyle değiştirilmemiştir. Dört kanallı model, BDD100K yükleyicisi, değerlendirme aracı ve uygulama bağları depoda bulunur; daha büyük, videoya göre ayrılmış ve bağımsız test edilmiş bir veri kümesi yeterli sonuca ulaşırsa aday güvenle yükseltilebilir.

## Video sınırı

BDD100K ana kareleri gerçekten farklı video kliplerinden gelir; tek-kare U-Net için denetimli hedef bu ana karelerde vardır. Kaynak videoların etiketlenmiş ek kareleri verilmediğinden, etiketlenmemiş 100 videoyu denetimli eğitim örneği olarak saymak doğru değildir. `indir_bdd_video_ornekleri.py` arşivin tamamını indirmeden seçili videoları alma aracını sağlar; kaynak sunucunun düşük aktarım hızı nedeniyle bu deneyde 100 ham video indirilip ek eğitim yapılmış olarak raporlanmaz.
