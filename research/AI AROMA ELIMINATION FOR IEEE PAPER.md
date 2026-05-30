# Destek Sınırlı Klinik Koşullar Altında Sepsis Dozaj Politikaları İçin Orkestre Edilmiş Bir Çevrimdışı Pekiştirmeli Öğrenme İş Akışı

**Yazar:** Furkan Nezih Üzmez, 230202073

## Özet

Bu çalışma, MIMIC-IV veri seti üzerinde sepsis dozaj politikalarının yeniden üretilebilir biçimde değerlendirilmesi için orkestre edilmiş bir çevrimdışı pekiştirmeli öğrenme (offline reinforcement learning) iş akışı sunmaktadır. Temel risk, modelin tarihsel veride zayıf temsil edilen sıvı-vazopresör kombinasyonlarını güvenilir sanmasıdır. Bu nedenle iş akışı; Sepsis-3 kohort seçimini, 72 saatlik klinik pencereyi, 62 boyutlu durum vektörünü, $5\times5$ ayrık aksiyon uzayını, seyrek ödül (sparse reward) ve SOFA-biçimli ödül (SOFA-shaped reward) tasarımlarını, yalnız eğitim verisine dayalı ön işleme (train-only preprocessing), hasta düzeyi bölme (patient-level split), örtük Q-öğrenme (implicit Q-learning, IQL) taramasını, çok kriterli finalist seçimini ve çoklu tohum doğrulamasını aynı protokol içinde tutar. Aşama 1'de iki ödül, üç öğrenme oranı (learning rate) rejimi ve üç ekspektil-sıcaklık (expectile-temperature) ayarından oluşan 18 konfigürasyon değerlendirilir; Aşama 2'de altı finalist üç tohumla yeniden ölçülür. Final aday `iql_sofa_shaped_conservative_safe` konfigürasyonudur. Üç tohum ortalamasında uydurulmuş Q-değerlendirmesi (fitted Q evaluation, FQE)=2.848, ağırlıklı önem örneklemesi (weighted importance sampling, WIS)=8.203, bootstrap WIS 95% güven aralığı=4.963--10.817, etkin örneklem büyüklüğü (effective sample size, ESS)=29.410, destek kütlesi=0.991 ve klinisyen tam-kutu uyumu (exact-bin agreement)=0.414 elde edilmiştir. ESS 50 eşiğinin altında kaldığı için bu sonuç klinik üstünlük iddiası değil, destek sınırı açıkça belirtilmiş retrospektif politika dışı değerlendirme (off-policy evaluation) kanıtıdır.

**Anahtar Kelimeler:** çevrimdışı pekiştirmeli öğrenme, örtük Q-öğrenme (implicit Q-learning), sepsis, MIMIC-IV, politika dışı değerlendirme, yeniden üretilebilirlik, klinik karar desteği

# Giriş

Sepsis resüsitasyonunda aynı hasta birkaç saat içinde farklı sıvı ve vazopresör dozlarına ihtiyaç duyabilir. Bu kararlar gecikmeli mortalite sonuçlarına bağlanır ve daha hasta bireylerin daha agresif tedavi alması karıştırıcılık (confounding) üretir. Çevrimdışı pekiştirmeli öğrenme bu nedenle uygundur: model yalnızca geçmiş yoğun bakım yörüngelerini (trajectories) kullanır, hastalar üzerinde çevrimiçi keşif (online exploration) yapmaz. Aynı özellik yöntemi kırılgan da yapar; politika desteklenmeyen aksiyonlara kayarsa, uydurulmuş Q-değerlendirmesi (FQE) veya ağırlıklı önem örneklemesi (WIS) gibi politika dışı tahminleyiciler (off-policy estimators) sayısal olarak düzgün görünse bile klinik olarak güvenilmez hale gelir.

Bu proje, MIMIC-IV üzerinde yatak başı karar aracı değil, denetlenebilir bir retrospektif değerlendirme laboratuvarı kurar. Model sınıfı modelsiz (model-free), politika dışı (off-policy) ve çevrimdışı pekiştirmeli öğrenmedir. Muhafazakar Q-öğrenme (conservative Q-learning, CQL), davranış kısıtlı Q-öğrenme (batch-constrained Q-learning, BCQ) ve örtük Q-öğrenme aileleri tasarım aşamasında değerlendirilmiştir. Final rapor IQL’ye odaklanır; çünkü IQL veri dışı aksiyon maksimumu almadan politika çıkarır. Kodun MPS ve CUDA üzerinde aynı eğitim mantığıyla çalışması istenir; donanım farkı deney kararına dönüşmemelidir.

Klinik zaman çizelgesi dört saatlik karar pencerelerine ayrılır; her karar, vazopresör ve intravenöz sıvı şiddetinin $5\times5$ kombinasyonlarından biridir. Final deney iki ödül varyantını aynı terminal değerlendirme altında karşılaştırır, finalistleri yalnız doğrulama kanıtıyla seçer ve raporlanabilir çıktı paketi (artifact bundle) üretir. Denetim kapıları, deterministik ızgara genişletmesi (grid expansion), çoklu tohum birleştirmesi ve grafik raporlama bu nedenle yardımcı ayrıntı değil, sonucun geçerlilik koşuludur.

# İlgili Çalışmalar

Çevrimdışı pekiştirmeli öğrenme, politika öğrenimini sabit bir veri kümesinin davranış desteği içinde tutmaya çalışır. IQL bu amaçla veri dışı aksiyonları maksimize etmez; ekspektil değer öğrenimi (expectile value learning) ve avantaj ağırlıklı davranış klonlama (advantage-weighted behavioral cloning) kullanır . CQL ise düşük destekli aksiyonların Q değerini açıkça cezalandırır . Bu ceza ekstrapolasyon hatası (extrapolation error) riskine karşı güçlüdür, fakat yanlış ayarlandığında klinikte nadir ama gerekli kurtarma müdahalelerini de bastırabilir. IQL’nin tercih edilmesi bu yüzden bilinçli bir takastır: destek dışına taşma azalır, ancak veri dağılımında hiç görülmeyen tedavi stratejileri keşfedilemez.

Sepsis RL literatüründe iki çalışma bu tasarımı doğrudan çerçeveler. AI Clinician, sıvı ve vazopresör dozlarını ayrık tedavi kutuları olarak modellemiştir; Raghu ve arkadaşları aynı problemi sürekli durum temsilleriyle ele almıştır . Bu çalışmalar klinik karar desteği için yol açıcı olsa da, sağlıkta RL rehberleri gözlemsel politikaların confounding, destek eksikliği, estimator varyansı ve prospektif doğrulama eksikliği nedeniyle tedavi önerisi sayılamayacağını belirtir . Bu rapor bu sınırı korur: amaç klinik üstünlük iddiası değil, yeniden üretilebilir çevrimdışı değerlendirme (offline evaluation) kanıtıdır. Veri kaynağı kimliksizleştirilmiş (de-identified) kritik bakım kayıtları sunan MIMIC-IV’tür ; aksiyon eksenlerinin sıvı ve vazopresör olarak seçilmesi güncel sepsis yönetiminde bu iki müdahalenin merkezi rolüyle uyumludur .

# Kuramsal Arka Plan ve Algoritmik Konumlandırma

Çevrimdışı pekiştirmeli öğrenmede (offline RL) politika iyileştirmesi, veri desteği dışına taşmadığı sürece anlamlıdır. Klasik politika dışı değer öğrenimi (off-policy value learning) bir sonraki durumda $\max_{a'} Q(s',a')$ sorgusu yaptığında, veri setinde hiç görülmemiş aksiyonlara değer atar. Fonksiyon yaklaştırıcı bu bölgelerde eğitim sinyali almadığı için ekstrapolasyon hatası oluşur; hata Bellman yedeklemeleriyle büyür ve klinik veride güvenilmez doz önerilerine dönüşebilir.

Literatürde bu hataya verilen yanıtlar üç grupta toplanabilir. CQL gibi açık aksiyon-uzayı düzenlileştirmesi (explicit action-space regularization) yöntemleri veri dışı aksiyonların Q değerini düşürür ve kötümser bir alt sınır kurar . IQL gibi örtük ve veri-içi yöntemler (implicit, in-sample methods) gözlenmemiş aksiyonu sorgulamaz; veri içinde yüksek değerli görünen aksiyonların üst ekspektilini (expectile) öğrenir . Durum-uzayı manifoldu düzenlileştirmesi (state-space manifold regularization) ise cezayı aksiyonlardan durum manifoldunun dışına taşır. Bu rapor IQL hattını seçer; çünkü klinik uygulamada ek ceza katsayısı kalibrasyonundan önce destek içinde kalma ve yorumlanabilir doğrulama protokolü daha kritik kabul edilmiştir.

Çok adımlı değer yayılımı (multi-step value propagation) bu çalışmanın doğrudan deney değişkeni değil, sonraki araştırma eksenidir. Tek adımlı yedeklemeler seyrek terminal mortalite (sparse terminal mortality) sinyalini 18 adımlık yörünge (trajectory) boyunca yavaş taşır; Peng’in $Q(\lambda)$ gibi uygunluk izi (eligibility trace) yaklaşımları bu gecikmeyi azaltabilir. Bu raporda önce destek güvenliği ve sızıntısız değerlendirme (leakage-free evaluation) sabitlenmiştir. Multi-step operatör aynı anda değiştirilseydi, final IQL sonucunun reward/LR/expectile seçimi mi yoksa yedekleme operatörü mü nedeniyle değiştiği ayrıştırılamazdı.

# Uçtan Uca Proje Mimarisi

Proje, katı bağımlılık zinciriyle ilerleyen on ana adımdan oluşur. Bu yapı, veri hazırlama hatalarının final politika değerlendirmesine taşınmasını engellemek için bilinçli olarak aşamalı tasarlanmıştır.

<div class="table*">

| Adım | Bileşen                                       | İçerik                                                                                                                                                                                                                            |
|:-----|:----------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Kohort seçimi                                 | MIMIC-IV içinden Sepsis-3 kriterlerine uyan yetişkin yoğun bakım ünitesi (ICU) hastalarının seçilmesi; geçersiz onset, kısa yoğun bakım ünitesi kalışı, eksik temel demografi ve tekrar yatışların yönetilmesi.                   |
| 2    | Zaman çizelgesi                               | Sepsis onset’inden 24 saat önce başlayıp 48 saat sonrasına uzanan 72 saatlik pencerenin 4 saatlik karar adımlarına bölünmesi.                                                                                                     |
| 3    | Sıfır sızıntı (zero leakage) bölme (split)    | Hastaların eğitim/doğrulama/test (train/validation/test) olarak hasta-düzeyinde ayrılması ve ön işleme uyarlama işlemlerinin (preprocessing fit) yalnız eğitim verisi (train) üzerinde yapılması.                                 |
| 4    | Durum (state) çıkarımı                        | Yaşamsal bulgu (vital), laboratuvar, tedavi geçmişi, demografi, eksiklik (missingness) ve türetilmiş değişkenlerden 62 boyutlu durum vektörü üretilmesi.                                                                          |
| 5    | Aksiyon/ödül (action/reward)                  | Vazopresör ve IV sıvı dozlarının 25 ayrık aksiyona dönüştürülmesi; seyrek ve SOFA-biçimli ödüllerin (sparse and SOFA-shaped rewards) üretilmesi.                                                                                  |
| 6    | Geçiş/başlangıç çizgisi (transition/baseline) | Tekrar oynatma verisinin (replay data) $(s_t,a_t,r_t,s_{t+1},done_t)$ formatında yazılması; klinisyen (clinician), tedavisiz (no-treatment) ve davranış klonlama (behavior cloning) başlangıç çizgilerinin (baselines) kurulması. |
| 7–8  | IQL eğitimi                                   | Çevrimdışı veride (offline data) destek dışına çıkmadan ekspektil değer öğrenimi ve avantaj ağırlıklı aktör güncellemesi (advantage-weighted actor update) ile IQL politikalarının eğitilmesi.                                    |
| 9    | Politika dışı değerlendirme (OPE) ve güvenlik | FQE, WIS, ESS, bootstrap güven aralığı (confidence interval, CI), destek kütlesi, klinisyen uyumu (clinician agreement), ısı haritası (heatmap) ve güvenlik teşhislerinin (diagnostics) hesaplanması.                             |
| 10   | Final tarama (final sweep)                    | 18 konfigürasyonlu IQL taraması, finalist seçimi, çoklu tohum doğrulaması ve IEEE rapor çıktılarının (artifacts) üretilmesi.                                                                                                      |

</div>

Bu mimariyi tek bir eğitim betiği (training script) olarak ele almıyoruz. Kohort tanımı, veri sözleşmesi, model eğitimi, politika dışı değerlendirme, güvenlik analizi ve raporlama çıktıları aynı deney protokolüne bağlanır. Bu bağ kurulmazsa final IQL skoru üretilebilir görünür; ancak bölme sızıntısı (split leakage), farklı ön işleme (preprocessing) ya da hatalı geçiş indeksleme (transition indexing) model başarısı sanılabilir.

# Kohort, Zaman ve Split Tasarımı

## Kohort Seçimi

Kohort MIMIC-IV üzerinde Sepsis-3 kriterlerine göre oluşturulur . Dahil edilen popülasyon 18 yaş ve üzeri yetişkin, doğrudan yoğun bakım ünitesi izlemine sahip ve sepsis tanımıyla uyumlu hastalardır. Çocuk hastalar farklı doz ve fizyoloji rejimlerine sahip oldukları için dışarıda bırakılır; bu karar verilmeseydi model tek bir aksiyon-doz anlamını heterojen yaş grupları için öğrenmeye zorlanırdı. Yoğun bakım ünitesi dışı veya yoğun gözlem verisi olmayan hastalar da dışlanır; çünkü 4 saatlik karar pencereleri için yeterli fizyolojik zaman serisi kurulamaz.

Dışlama kuralları model performansını temiz göstermek için değil, karar penceresini klinik olarak tutarlı tutmak için uygulanır. Sepsis onset zamanı bulunamayan hastalar, 4 saatten kısa yoğun bakım ünitesi kalışı olanlar ve yaş/cinsiyet gibi temel demografisi eksik olanlar çıkarılır. Tekrar yatan hastalarda yalnız ilk yatış kullanılır. Aynı hastanın bir yatışı eğitim setine (train set), sonraki yatışı test setine düşerse model kronik profili dolaylı olarak ezberleyebilir; bu nedenle filtreleme sonucu ve dışlanan hasta sayıları denetim çıktılarında (audit artifacts) saklanır.

## Zaman Çizelgesi

Her yoğun bakım ünitesi kalışı için ‘sepsis_onset_time’ hesaplanır. Analiz penceresi onset’ten 24 saat önce başlayıp 48 saat sonrasına kadar devam eder. Bu 72 saatlik pencere 4 saatlik karar adımlarına ayrıldığında tipik epizot (episode) yaklaşık 18 zaman adımı (timestep) içerir. Dört saatlik pencere, literatürdeki sepsis RL kurulumlarıyla karşılaştırılabilirlik sağlar ve klinik ölçümlerin düzensiz zamanlanmasını yönetilebilir bir karar frekansına indirger . Daha kısa pencere seçilseydi ölçüm eksikliği ve gürültülü aksiyon tekrarları artacak, davranış politikası olasılıkları daha seyrek gözlenen mikro kararlar üzerinde tahmin edilecekti. Daha uzun pencere seçilseydi de sıvı ve vazopresör değişimlerinin akut fizyolojik etkileri ortalamaya karışacak, ardışık karar yapısı zayıflayacaktı.

## Split Stratejisi

Varsayılan split hasta düzeyinde eğitim %70, doğrulama %15 ve test %15 olarak yapılır. Bölme tohumu (split seed) 42 olarak sabitlenir. Eğitim, doğrulama ve test özne kümelerinin (subject sets) kesişimi boş olmalıdır. Test seti yalnız final raporlama için kullanılır; hiperparametre (hyperparameter), kontrol noktası (checkpoint) veya metrik tasarımı test sonucuna göre değiştirilmez. Bu kural ihlal edilseydi final test metrikleri gerçek ayrılmış (held-out) kanıt olmaktan çıkar, doğrulamanın uzantısı haline gelirdi.

# MDP Formülasyonu

Problem sonlu ufuklu bir Markov karar süreci olarak modellenir: $$M=(S,A,P,R,\gamma).$$ Burada $S$ hasta durum uzayını, $A$ ayrık tedavi aksiyonlarını, $P$ geçiş dinamiklerini, $R$ ödül fonksiyonunu ve $\gamma$ iskonto faktörünü (discount factor) temsil eder. Her geçiş şu biçimdedir: $$(s_t,a_t,r_t,s_{t+1},done_t).$$

## Durum Temsili

Durum vektörü (state vector) hastanın o andaki klinik durumunu özetleyen 62 sürekli/ikili özellikten oluşur. Başlıca gruplar kalp hızı, MAP, sistolik/diyastolik tansiyon, solunum ve SpO2 gibi yaşamsal bulgular (vital signs); kan gazı, elektrolit, renal, hepatik, koagülasyon ve hemogram laboratuvarları; kümülatif sıvı, vazopresör maruziyeti ve idrar çıkışı gibi tedavi sinyalleri; yaş ve cinsiyet gibi demografik değişkenler; PaO2/FiO2, shock index ve onset sonrası saat gibi türetilmiş değişkenlerdir.

Her 4 saatlik pencerede birden fazla ölçüm varsa özelliğe göre ‘last’, ‘mean’, ‘min’, ‘max’, ‘sum’ veya ‘cumulative’ agregasyon kuralı uygulanır. Örneğin bazı yaşamsal bulgularda son ölçüm klinik karar anına daha yakın bilgi taşırken, toplam sıvı veya kümülatif vazopresör maruziyeti pencere içi tedavi yükünü temsil eder. Tek bir agregasyon kuralı tüm değişkenlere dayatılsaydı klinik anlamı farklı sinyaller aynı istatistiksel kalıba sıkıştırılmış olurdu.

Eksik veriler için sıralı strateji uygulanır: epizot içinde forward-fill, eğitim bölmesi (train split) üzerinden hesaplanan medyan, klinik normal değer fallback ve gerekli durumda eksiklik göstergesi (missingness flag). Missingness göstergeleri eklenmeseydi model, ölçülmeyen laboratuvarı normal değerle karıştırabilir ve hekimin ölçüm kararıyla taşınan risk bilgisini kaybedebilirdi. Yalnız eğitim medyanı (train-only median) kullanılmasaydı doğrulama/test dağılım bilgisi ön işleme üzerinden eğitime sızardı.

## Aksiyon Uzayı (Action Space)

Aksiyon uzayı 25 ayrık tedavi kararından oluşur. Her aksiyon, vazopresör dozu ve IV sıvı miktarının $5\times5$ kombinasyonudur: $$action\_id = vaso\_bin \times 5 + fluid\_bin.$$ Bin 0 tedavi yok anlamına gelir; sıfır dışı (non-zero) dozlar eğitim bölmesi üzerindeki çeyrekliklere göre Q1–Q4 olarak ayrılır. Sıfır doz (zero dose) ayrı tutulur, çünkü hiç tedavi vermemek küçük ama sıfır dışı tedaviyle aynı klinik anlamı taşımaz. Bin eşikleri yalnız eğitim bölmesi üzerinde öğrenilir ve doğrulama/test üzerinde dondurulmuş şekilde uygulanır. Eşikler tüm veri üzerinde hesaplansaydı test doz dağılımı aksiyon kodlamasına sızardı.

<div id="tab:action_grid">

|           |           |     |     |     |     |
|:---------:|:---------:|:---:|:---:|:---:|:---:|
|           | Sıvı bini |     |     |     |     |
| Vazo bini |     0     |  1  |  2  |  3  |  4  |
|     0     |     0     |  1  |  2  |  3  |  4  |
|     1     |     5     |  6  |  7  |  8  |  9  |
|     2     |    10     | 11  | 12  | 13  | 14  |
|     3     |    15     | 16  | 17  | 18  | 19  |
|     4     |    20     | 21  | 22  | 23  | 24  |

Vazopresör ve IV sıvı aksiyon grid’i.

</div>

## Ödül Tasarımı (Reward Design)

İki ödül varyantı kullanılır. Seyrek ödül yalnız terminal mortalite/sağkalım fayda değerini (utility) kullanır: 90 gün yaşayan hasta için $+15$, 90 gün içinde ölüm için $-15$. Bu seçenek klinik nihai sonucu doğrudan hedeflediği için yorumlanması kolaydır, fakat uzun ufuklu ve seyrek geri bildirimli olduğundan değer öğrenmesi yüksek varyanslı hale gelebilir.

SOFA-biçimli ödül terminal faydayı (terminal utility) korur ve ara adımlarda SOFA değişimine küçük bir şekillendirme sinyali (shaping signal) ekler: $$sofa\_shaping = -0.025 \times (SOFA_{current}-SOFA_{previous}).$$ SOFA artışı kötüleşme olarak negatif, SOFA azalması iyileşme olarak pozitif sinyal üretir. SOFA’nın sepsis organ disfonksiyonu tanımı ve klinik şiddet takibiyle ilişkili olması bu ara sinyali klinik olarak gerekçelendirir . SOFA şekillendirmesi (SOFA shaping) kullanılmasaydı eğitim yalnız terminal sonuca dayanacak ve ara fizyolojik iyileşmeleri ayırt etmek zorlaşacaktı. Tersine, shaping ağırlığı çok güçlü seçilseydi ajan mortalite yerine kısa vadeli skor oynamalarını optimize edebilirdi; bu nedenle terminal fayda korunmuş ve seçim ortak terminal değerlendirmeye bağlanmıştır.

## İskonto Faktörü

İskonto faktörü $\gamma=0.99$ olarak sabitlenir. Bu seçim, sepsis tedavisinde kısa vadeli hemodinamik kararların uzun vadeli mortalite sonucuyla ilişkili olması nedeniyle uygundur. Daha küçük bir $\gamma$ kısa vadeli SOFA değişimlerini fazla öne çıkarabilir, $\gamma=1$ ise sonlu ufukta mümkün olsa da FQE ve WIS tahminlerinde geç dönem terminal sinyallerine daha hassas bir varyans profili doğurabilirdi.

# Ön İşleme ve Sızıntısız Sözleşmeler (Preprocessing and Leakage-Free Contracts)

Temel kural ‘eğitime uydur, her yerde dönüştür (fit on train, transform everywhere)’ şeklindedir. Ölçekleyici (scaler) parametreleri, atayıcı (imputer) medyanları, kırpma sınırı (clip bound) veya yüzdelik (percentile) eşikleri, aksiyon bini çeyreklik eşikleri (action-bin quartiles), davranış politikası tahminleyicisi (behavior policy estimator), FQE tahminleyicisi (FQE estimator) ve normalleştirme (normalization) sabitleri eğitim verisi üzerinde uydurulur (fit). Doğrulama ve test üzerinde yalnız eğitim verisinde uydurulmuş dönüşümler (transforms) uygulanır.

Fizyolojik olarak makul olmayan değerler geçerli aralık (valid range) dışında filtrelenir veya kırpılır (clip). Kırpma (clipping) ve normalleştirme parametreleri eğitim bölmesi üzerinde uydurulur. Bu tercih yapılmasaydı doğrulama/test dağılımının uç değer bilgisi eğitim öncesi veri dönüşümlerine yansıyabilir ve raporlanan performans fazla iyimser olurdu.

## Aşama 0 Ön Tarama Denetimi

Ön tarama denetimi (presweep audit), eğitim başlamadan önce veri sözleşmelerini test eder. Sağlıkta RL için bu kapı zorunludur; küçük bir bölme sızıntısı veya sonuç sızıntısı (outcome leakage), gözlemsel veride gerçek politika kalitesinden çok daha büyük görünen kazançlar üretebilir . Denetim seyrek tekrar oynatma (sparse replay) üretimini, seyrek ve SOFA-biçimli tekrar oynatma (SOFA-shaped replay) dosyalarının aynı bölme/ön işleme/aksiyon bini/geçiş (split/preprocessing/action bin/transition) indeksleme/terminal sonuç (terminal outcome) tanımlarını kullanmasını, aksiyon binlerinin (action bins) yalnız eğitim verisi (train-only) fit edilmesini, atama/ölçekleme/kırpma/normalleştirme (imputation/scaling/clipping/normalization) istatistiklerinin yalnız eğitim verisi kalmasını, doğrulama FQE’sinin ortak terminal sağkalım/ölüm ödülü (terminal survival/death reward) ile çalışmasını, zamansal hizalamayı (temporal alignment), hasta çakışması bulunmamasını ve mortalite/taburculuk (mortality/discharge) gibi gelecek sonuç bilgisinin (future outcome) durum özelliklerine (state features) sızmamasını kontrol eder.

<div id="tab:audit">

| Kanıt kalemi             |   Eğitim | Doğrulama |  Test |
|:-------------------------|---------:|----------:|------:|
| Split manifest satırı    |    12063 |      2585 |  2585 |
| SOFA replay satırı       |   140635 |     30539 | 30046 |
| Seyrek replay satırı     |   140635 |     30539 | 30046 |
| Split leakage            | Başarılı |           |       |
| Ortak replay sözleşmesi  | Başarılı |           |       |
| Zamansal hizalama        | Başarılı |           |       |
| Outcome leakage taraması | Başarılı |           |       |

Ön tarama denetim özeti (presweep audit summary).

</div>

Bu adım atlanmış olsaydı Aşama 2 sonucu sayısal olarak üretilebilir görünse bile, aynı hastanın farklı bölmelere (splits) düşmesi veya terminal bilginin özelliklere sızması nedeniyle metodolojik olarak geçersiz hale gelebilirdi.

# Başlangıç Çizgisi Politikaları (Baseline Policies)

Final test karşılaştırması üç referans politika içerir. Klinisyen tekrar oynatımı (clinician replay), tekrar oynatma veri setindeki (replay dataset) gözlenen klinisyen aksiyonlarını değerlendirir ve davranış verisinin kendisini referans alır. Tedavisiz politika (no-treatment policy), uygun yerlerde vazopresör yok/sıvı yok (no-vasopressor/no-fluid) bin’ini seçen basit kontrol politikasıdır. Davranış klonlama ise klinisyen aksiyonlarını gözetimli öğrenme (supervised learning) ile taklit eder. Bu başlangıç çizgileri eklenmeseydi IQL çıktısı yalnız kendi politika dışı değerlendirme skoru ile raporlanır, davranış politikasına ve basit kontrol stratejilerine göre bağlamı kaybolurdu.

# IQL Algoritması

Örtük Q-öğrenme, veri seti dışındaki (out-of-dataset) aksiyonları maksimize etmeden politika çıkarır . Model üç parça öğrenir: eleştirmen (critic), değer işlevi (value function) ve aktör (actor). Eleştirmen $Q_\theta(s,a)$ durum-aksiyon değerini (state-action value) tahmin eder. Değer işlevi, klasik maksimum operatörü (max operator) yerine ekspektil regresyonu (expectile regression) ile uydurulur. Ekspektil düştükçe değer hedefi davranış verisine yaklaşır; ekspektil yükseldikçe veri içindeki daha yüksek değerli aksiyonlara ağırlık verir.

Değer amacı (value objective) asimetrik ekspektil kaybı (expectile loss) ile yazılır: $$L_V(\psi)=\mathbb{E}_{(s,a)\sim D}\left[L_2^\tau(Q_\theta(s,a)-V_\psi(s))\right],$$ $$L_2^\tau(u)=|\tau-\mathbb{I}(u<0)|u^2.$$ $\tau$ 1’e yaklaştıkça değer fonksiyonu davranış verisi içinde daha iyi görünen aksiyonların üst ekspektiline (upper expectile) yaklaşır. Q fonksiyonu ise veri dışı aksiyon maksimumu yerine öğrenilen durum-değer hedefiyle (state-value target) güncellenir: $$L_Q(\theta)=\mathbb{E}_{D}\left[(r+\gamma V_\psi(s')-Q_\theta(s,a))^2\right].$$

Aktör, veri seti içindeki (in-dataset) aksiyonları avantaja (advantage) göre ağırlıklandırarak öğrenir: $$A(s,a)=Q_\theta(s,a)-V_\psi(s),$$ $$L_\pi(\phi)=\mathbb{E}_{D}\left[\exp(\beta A(s,a))\log \pi_\phi(a|s)\right].$$ Burada $\beta$ ya da sıcaklık (temperature) politika çıkarımının agresifliğini kontrol eder. Düşük sıcaklık davranış politikasına daha yakın ve konservatif politika çıkarır; yüksek sıcaklık potansiyel getiriyi artırabilir fakat düşük destekli aksiyonlara kayma riskini de büyütür. Bu nedenle güvenli (safe), referans (baseline) ve iyimser (optimistic) ayarları birlikte denenmiştir.

# IQL Final Tarama Tasarımı

Final tarama iki ödül varyantı, üç öğrenme oranı (learning-rate) rejimi ve üç IQL ayarını on sekiz Aşama 1 adayına genişletir. Öğrenme oranı rejimleri (learning-rate regimes) aktör, eleştirmen ve değer (value) optimizasyonunu ayrı olarak kontrol eder. Konservatif öğrenme oranı rejimi finalde öne çıkarılmıştır; çünkü çevrimdışı RL’de agresif eleştirmen güncellemeleri (critic updates) veri dışı Q tahminlerini büyütebilir ve politika çıkarımı bu hataları çoğaltabilir . Referans rejim (baseline regime) yalnızca referans çapa (baseline anchor) olarak tutulmuştur; hiç denenmeseydi konservatif seçimin gerçekten gerekli olup olmadığı gözlenemezdi. Aktör-konservatif rejim ise politika güncellemelerini yavaşlatmanın, eleştirmen/değer (critic/value) öğrenmesini tamamen yavaşlatmadan destek uyumunu artırıp artırmadığını test eder.

<div class="table*">

| Boyut     | Seçenek                                | Değerler                                                                    |
|:----------|:---------------------------------------|:----------------------------------------------------------------------------|
| Ödül      | seyrek                                 | terminal sağkalım/ölüm faydası (terminal survival/death utility)            |
| Ödül      | SOFA-biçimli                           | terminal fayda + SOFA değişim şekillendirmesi (shaping)                     |
| LR rejimi | konservatif (conservative)             | aktör $3\times10^{-5}$, eleştirmen $1\times10^{-4}$, değer $1\times10^{-4}$ |
| LR rejimi | referans                               | aktör $1\times10^{-4}$, eleştirmen $3\times10^{-4}$, değer $3\times10^{-4}$ |
| LR rejimi | aktör-konservatif (actor-conservative) | aktör $3\times10^{-5}$, eleştirmen $3\times10^{-4}$, değer $3\times10^{-4}$ |
| IQL ayarı | güvenli                                | ekspektil 0.7, sıcaklık 1.0                                                 |
| IQL ayarı | referans                               | ekspektil 0.7, sıcaklık 3.0                                                 |
| IQL ayarı | iyimser                                | ekspektil 0.8, sıcaklık 3.0                                                 |

</div>

<div id="tab:fixed_params">

| Parametre                                               | Değer        |
|:--------------------------------------------------------|:-------------|
| Algoritma                                               | IQL          |
| Epoch                                                   | 200          |
| Yığın boyutu (batch size)                               | 256          |
| İskonto (discount)                                      | 0.99         |
| Politika gizli katman boyutları (policy hidden sizes)   | \[256, 256\] |
| Değer gizli katman boyutları (value hidden sizes)       | \[256, 256\] |
| Eleştirmen gizli katman boyutları (critic hidden sizes) | \[256, 256\] |
| Polyak tau                                              | 0.005        |
| Hedef güncelleme sıklığı (target update freq)           | 10           |
| Gradyan kırpma (grad clip)                              | 10.0         |
| Cihaz (device)                                          | auto         |

Sabit eğitim parametreleri.

</div>

Sabit parametreler korunmuştur; böylece final tarama yalnız ödül (reward), öğrenme oranı rejimi, ekspektil ve sıcaklık etkisini ölçer. Bu parametreler de serbest bırakılsaydı 18 konfigürasyonun etkisi yorumlanamaz, model başarısının mimari mi yoksa ödül/LR/IQL (reward/LR/IQL) ayarı kaynaklı mı olduğu belirsizleşirdi.

<div class="algorithm">

<div class="algorithmic">

Tekrar oynatma, bölme ve ön işleme (replay, split, preprocessing) ve zamansal sözleşmeleri doğrula Ödül, öğrenme oranı ve IQL ayarlarını genişlet Tohum (seed) 42 ile politikayı eğit Doğrulama metriklerini ve destek teşhislerini hesapla Güvenlik veya destek kapılarını geçemeyen adayları işaretle Bileşik skor (composite), ödül dengesi, güvenlik ve çeşitlilik slotlarıyla finalist seç Tohum 123 ve 456 ile politikayı eğit veya yeniden kullan Çoklu tohum değerlendirme kanıtını topla Tekrarlanan tohum kanıtıyla final konfigürasyonu seç Başlangıç çizgisi karşılaştırmaları (baseline comparisons), grafikler ve yeniden üretilebilirlik çıktıları üret

</div>

</div>

# Tohum Stratejisi (Tohum Strategy) ve Karar Kuralı

Tek tohum (single seed), çevrimdışı RL için zayıf kanıttır. Başlatma (initialization), mini-yığın (minibatch) sırası ve stokastik optimizasyon (stochastic optimization) aynı konfigürasyonun FQE/WIS değerini değiştirebilir. Aşama 1’de 18 konfigürasyon tohum 42 ile taranır; ardından altı finalist tohum 123 ve 456 ile yeniden değerlendirilir. Böylece tüm $2\times3\times3\times3=54$ çalıştırma (run) yerine 30 çalıştırma çalıştırılır. Bu tasarım hesaplama maliyetini düşürür, fakat final adayları üç tohum üzerinden raporladığı için tohum varyansı (tohum variance) görünür kalır.

Hiperparametre seçimi test set üzerinden yapılmaz. Seçim doğrulama bölmesi üzerindeki FQE, WIS, ESS, klinisyen uyumu, aksiyon dağılımı (action distribution) makullüğü, düşük destekli (low-support) aksiyon uyarıları, ödül eğrisi (reward curve) ve eğitim kararlılığı (training stability) sinyallerine dayanır. Tek bir metrikle karar verilmez. Örneğin yüksek WIS ama çok düşük ESS varsa bu sonuç güvenilir kabul edilmez.

Bir konfigürasyon final aday olmadan önce doğrulama performansı rekabetçi olmalı, ESS çok düşük olmamalı, FQE sonucu WIS ile tamamen çelişmemeli, aksiyon dağılımı klinik olarak makul olmalı, düşük destekli aksiyon oranı kabul edilebilir düzeyde olmalı, farklı tohumlarda (seeds) performans tamamen kararsız olmamalı ve test set seçim sürecinde kullanılmamış olmalıdır.

# Aşama 1 Finalist Seçim Sonuçları

Aşama 1 seçici altı finalist üretmiştir. En yüksek iki aday SOFA-biçimli ödül ve konservatif öğrenme oranı kullanırken, diğer slotlar seyrek ödül temsilini, güvenlik-destek dengesini, referans çapa seçimini ve çeşitlilik tamamlama (diversity backfill) kuralını korur. Tablo <a href="#tab:finalists" data-reference-type="ref" data-reference="tab:finalists">[tab:finalists]</a>, seçilen adayları ve ana doğrulama teşhislerini gösterir.

<div class="table*">

| Sıra | Yuva (slot)                       | Ödül   | LR          | IQL      |   FQE |   WIS | Destek |  Uyum |
|-----:|:----------------------------------|:-------|:------------|:---------|------:|------:|-------:|------:|
|    1 | en iyi bileşik (top composite)    | SOFA   | konservatif | güvenli  | 3.169 | 8.616 |  0.965 | 0.418 |
|    2 | en iyi bileşik                    | SOFA   | konservatif | referans | 2.280 | 7.009 |  0.973 | 0.401 |
|    3 | en iyi seyrek (best sparse)       | sparse | konservatif | güvenli  | 1.954 | 8.740 |  0.971 | 0.411 |
|    4 | güvenlik desteği (safety support) | sparse | referans    | referans | 1.421 | 6.768 |  0.973 | 0.410 |
|    5 | referans çapa                     | sparse | referans    | güvenli  | 2.345 | 8.125 |  0.963 | 0.416 |
|    6 | çeşitlilik tamamlama              | sparse | konservatif | referans | 2.012 | 7.782 |  0.969 | 0.407 |

</div>

Finalist havuzu (finalist pool) tek bir skora göre seçilmez. Klinik çevrimdışı öğrenmede (offline learning) yüksek değer tahmini (value estimate), düşük davranış desteğiyle birleştiğinde güvenilir değildir . Yalnız en yüksek FQE veya WIS kullanılsaydı, düşük ESS ya da zayıf klinisyen uyumu taşıyan bir politika Aşama 2’ye geçebilirdi. Çeşitlilik slotları da bu yüzden korunur: Aşama 2 yalnız SOFA-konservatif adayları tekrar etmez; seyrek ödülün ve referans ayarların (baseline settings) sınırlarını da gösterir.

<figure id="fig:fqe_support">
<img src="figures/fqe_vs_support.png" />
<figcaption>Doğrulama değer tahmini ile davranış desteği teşhislerinin birlikte gösterimi. Yüksek değer yalnız yeterli destekle birlikte anlamlı kabul edilmelidir.</figcaption>
</figure>

# Aşama 2 ve Baseline Değerlendirmesi

Aşama 2, altı finalist ve üç tohumdan oluşan on sekiz satırlık bir tekrarlı-tohum özeti üretmiştir. Her finalist için tohum 42, 123 ve 456 kontrol noktaları (checkpoints) test tekrar oynatma (test tekrar oynatma verisi (test replay)) üzerinde yeniden değerlendirilmiştir. Tek tohumlu seçim yapılmış olsaydı rastgele başlatma, mini-yığın sırası veya erken eğitim dalgalanması final politika kararını belirleyebilirdi. Üç tohum, hesaplama maliyetini sınırlı tutarken bu varyansı görünür yapar; daha fazla tohum istatistiksel güveni artırırdı, fakat final tarama maliyetini doğrusal olarak büyütürdü.

Final skor, FQE, WIS, ESS, destek kütlesi, klinisyen uyumu, düşük destekli cezası ve güvenlik bayraklarını üç tohum ortalamasıyla birleştirir. FQE fonksiyon yaklaştırmaya dayalı değer tahmini verir; WIS davranış politikasına göre ağırlıklandırılmış gözlemsel kanıt sağlar. İkisini birlikte kullanmak, tek tahminleyicinin (estimator) yanlılık-varyans profilini final kararına dönüştürmemek içindir . ESS ve destek kütlesi olmasaydı, birkaç yüksek ağırlıklı yörünge yüksek WIS değerini güvenilir gösterebilirdi. Klinisyen uyumu olmasaydı politika davranış verisinden gereğinden fazla uzaklaşabilirdi. Bu kurala göre final konfigürasyon SOFA-biçimli (SOFA-shaped), konservatif öğrenme oranı ve güvenli IQL (safe IQL) ayarına sahip ‘iql_sofa_shaped_conservative_safe’ modelidir.

<div class="table*">

| Sıra | Konfigürasyon               | Ayar  |  Skor |   FQE |   WIS |   ESS | Destek |  Uyum |
|-----:|:----------------------------|:------|------:|------:|------:|------:|-------:|------:|
|    1 | SOFA konservatif güvenli    | final | 5.858 | 2.848 | 8.203 | 29.41 |  0.991 | 0.414 |
|    2 | SOFA konservatif referans   | aday  | 5.288 | 2.366 | 8.286 | 26.12 |  0.990 | 0.404 |
|    3 | sparse konservatif güvenli  | aday  | 5.262 | 2.316 | 8.169 | 31.32 |  0.990 | 0.411 |
|    4 | sparse konservatif referans | aday  | 5.033 | 2.050 | 8.536 | 26.10 |  0.990 | 0.401 |
|    5 | sparse baseline güvenli     | aday  | 4.940 | 2.063 | 7.882 | 31.31 |  0.991 | 0.418 |
|    6 | sparse baseline referans    | aday  | 4.755 | 1.535 | 9.755 | 18.96 |  0.991 | 0.409 |

</div>

<div id="tab:baseline">

| Metrik                         |        Sonuç |
|:-------------------------------|-------------:|
| FQE                            |        2.848 |
| WIS                            |        8.203 |
| WIS 95% CI                     | 4.963–10.817 |
| ESS                            |       29.410 |
| Davranışsal destek kütlesi     |        0.991 |
| Klinisyen tam-kutu uyumu       |        0.414 |
| Klinisyen uyuşmazlığı          |        0.586 |
| Desteklenmeyen aksiyon sapması |        0.009 |

Kazanan konfigürasyonun test teşhisleri (test diagnostics). WIS güven aralığı 50 hasta-düzeyi bootstrap yeniden örnekleme (bootstrap resample) ile hesaplanmıştır.

</div>

<figure id="fig:baseline_comparison">
<img src="figures/baseline_comparison.png" />
<figcaption>Final seçili IQL politikası ve referans başlangıç çizgisi gösterimi (baseline display). Seçili IQL değeri gerçek Aşama 2 değerlendirmesinden, referans başlangıç çizgisi çubukları (baseline bars) ise rapor çıktısında (artifact) kullanılan karşılaştırma gösteriminden gelir.</figcaption>
</figure>

<figure id="fig:seed_variance">
<img src="figures/seed_variance.png" />
<figcaption>Aşama 2 finalistleri için tekrarlı-tohum FQE değişkenliği. Daha düşük değer, aynı konfigürasyonun tohumlar arasında daha stabil olduğunu gösterir.</figcaption>
</figure>

# Aksiyon ve Güvenlik Teşhisleri (Action and Safety Diagnostics)

Değer metriği (value metric) tek başına klinik politika güvenliği göstermez. Bir politika iyi FQE üretebilir, fakat gözlemsel veride nadir görülen sıvı-vazopresör kombinasyonlarına kayabilir. Bu yüzden aksiyon ısı haritası (action heatmap), destek kütlesi (support mass), düşük destek oranı (low-support rate), klinisyen uyumu ve aksiyon entropisi (action entropy) aynı anda okunur.

<figure id="fig:action_heatmap">
<img src="figures/action_heatmap.png" />
<figcaption>Final politika analizi için üretilmiş aksiyon dağılımı ısı haritası. Tedavi gridi vazopresör ve sıvı şiddetinin ayrık kombinasyonlarını temsil eder.</figcaption>
</figure>

<figure id="fig:bootstrap_ci">
<img src="figures/bootstrap_ci.png" />
<figcaption>Final seçili IQL politikasının üç tohum için hasta-düzeyi bootstrap WIS güven aralıkları. Güven aralıkları 50 bootstrap yeniden örnekleme ile hesaplanmıştır ve düşük ESS nedeniyle temkinli yorumlanmalıdır.</figcaption>
</figure>

Teşhis okuması (diagnostic reading) somut sorulara dayanır: politika sürekli yüksek sıvı veya yüksek vazopresör mü öneriyor, ‘tedavisiz (no treatment)’ kutusuna çöküyor mu, düşük destekli aksiyonları sık seçiyor mu, yüksek riskli alt gruplarda uyarılar artıyor mu, ödül varyantı (reward variant) değişince aksiyon dağılımı dramatik biçimde değişiyor mu? Protokol bu sorular için klinisyen aksiyon dağılımı, IQL politika aksiyon dağılımı, fark ısı haritası (difference heatmap), düşük destekli aksiyon oranı (action rate), alt gruba göre klinisyen uyumu (clinician agreement by subgroup), yüksek riskli alt grup politika aksiyon sıklığı (high-risk subgroup policy action frequency), eğitim kaybı eğrileri (training loss curves), doğrulama metrik eğrileri, ödül varyantı kutu grafiği, LR rejimi karşılaştırması ve ekspektil/sıcaklık karşılaştırması (expectile/temperature comparison) grafiklerini çıktı olarak üretir.

Aşama 2 finalistlerinde destek kütlesi 0.990–0.991, klinisyen tam-kutu (exact-bin) uyumu 0.401–0.418 aralığındadır. Kazanan politikanın tam-kutu uyumu 0.414’tür; yani kararların 0.586’lık kısmı klinisyen davranışıyla birebir aynı değildir. Buna rağmen desteklenmeyen aksiyon sapması 0.009’da kalır. Bu ayrım önemlidir: politika klinisyeni kopyalamaz, fakat çoğu sapma veri desteği bulunan komşu tedavi yörüngelerinde kalır. ESS ise tüm finalistlerde 50 eşiğinin altındadır. Bu uyarı saklanırsa rapor, az sayıda etkili yörüngenin taşıdığı WIS tahminini kesin sonuç gibi sunar. Bu nedenle final karar “klinik üstünlük” değil, destek kısıtı belirtilmiş retrospektif OPE sonucudur.

# Yeniden Üretilebilirlik (Reproducibility) ve Teknoloji Yığını

İş akışı protokoldeki ana boşlukları kapatır: final IQL tarama çalıştırıcısı (IQL sweep runner), 18 konfigürasyonlu ızgara (grid), iş akışı (workflow) seviyesinde orkestrasyon, ön tarama denetimi, final-altı seçim mantığı, tekrarlı-tohum Aşama 2 birleştirmesi, başlangıç çizgisi karşılaştırması (baseline comparison) ve final grafik raporları. Çıktı seti (artifact set) ön tarama denetimi sonucunu, Aşama 1 seçim tablosunu, Aşama 2 tohum özetini (tohum summary), final metrik paketi (final metrics bundle)’ı, başlangıç çizgisi karşılaştırmasını ve beş grafik çıktısını içerir. Aşama 2 değerlendirmesi 18 finalist-tohum (finalist-seed) satırı olarak saklanır.

Teknoloji yığını (technology stack) Python ve ‘uv’ tabanlı ortam yönetimi, yüksek hızlı ‘Polars’ ve ‘PyArrow’ tabanlı Parquet veri dönüşümü, ‘PyTorch’ ve özelleştirilmiş ‘d3rlpy’ tabanlı RL eğitimi, ‘Hydra’ konfigürasyon yönetimi, ‘MLflow’ deney takibi ve IEEE LaTeX raporlamasından oluşur. Cihaz-agnostic tasarım sayesinde eğitim kodu MPS, CUDA veya otomatik cihaz seçimiyle çalışabilir. Bu altyapı ayrıştırılmasaydı deney sonuçları belirli bir donanıma veya elle değiştirilmiş konfigürasyonlara bağımlı hale gelirdi.

Her çalıştırma için git commit özeti (git commit hash), konfigürasyon dosyası (config file), ödül varyantı, LR rejimi, ekspektil, sıcaklık, tohum, veri seti yolu (dataset path), metadata yolu (metadata path), bölme manifestosu tohumu (split manifest seed), aksiyon bini çıktısı (action bin artifact), eğitim günlükleri (training logs), kontrol noktası yolu (checkpoint path), değerlendirme çıktısı (evaluation output) ve figür yolları (figure paths) kaydedilmelidir. Çalıştırma isimlendirme (run naming) şablonu ödül, LR rejimi, IQL ayarı ve tohum bilgisini birlikte taşıyan ‘iql_reward_lr_setting_seed’ biçimindedir. Örneğin SOFA-biçimli, actor-conservative, baseline ve tohum 42 bilgileri aynı çalıştırma adında saklanır. Bu sözleşme tutulmasaydı sonradan hangi kontrol noktasının hangi veri ve konfigürasyonla üretildiği izlenemezdi.

# Sınırlar ve Etik Değerlendirme

Bu çalışma retrospektif ve gözlemseldir (observational); öğrenilen politika yatak başı tedavi önerisi değildir. Durum vektörü gizli klinik şiddeti, kontrendikasyonları, hekim niyetini ve ölçülmeyen karıştırıcılık etkilerini tam kapsamaz. Gerçek klinik süreç kısmi gözlemlenebilir Markov karar sürecine (partially observable Markov decision process, POMDP)’ye daha yakındır; model hekimin tüm yatak başı bilgisini görmez. Klinisyen aksiyonları (clinician actions) da karıştırıcılık taşır: daha ağır hastalara daha agresif vazopresör verilmesi, tedaviyi zararlı gösterebilir veya yanlış fayda atfı üretebilir.

MIMIC-IV verisi PhysioNet credentialing, CITI training ve Data Use Agreement kapsamındaki yükümlülüklere tabidir; hasta verileri kimliksizleştirilmiş olsa da raporlama akademik ve etik sınırlar içinde kalmalıdır. FQE ve WIS prospektif klinik kanıt (prospective clinical evidence) değildir; fonksiyon yaklaşımı, davranış politikası (behavior policy) tahmini, destek varsayımları ve varyans profiline (variance profile) bağlıdır. ESS değerinin 50 eşiğinin altında kalması ve bootstrap yeniden örnekleme (bootstrap resampling) sayısının sınırlı olması nedeniyle final politika sonucu klinik performans kanıtı olarak yorumlanmamalıdır.

# Gelecek Araştırma Yönleri

Bu çalışmanın sonraki adımı daha büyük bir ızgara çalıştırmak değil, belirsizliği politika güncellemesine sokmaktır. İlk yön dinamik ekspektil kalibrasyonu (dynamic expectile calibration)dır. Yoğun destekli bölgelerde $\tau$ yükseltilerek daha agresif politika iyileştirme yapılabilir; seyrek ve belirsiz bölgelerde $\tau$ 0.5’e yaklaştırılarak davranış klonlamasına yakın, daha güvenli güncelleme seçilebilir.

İkinci yön, seyrek terminal mortalite sinyalini daha hızlı taşımak için uyarlanabilir çok adımlı iz harmanlama (adaptive multi-step trace blending) yaklaşımıdır. Peng’in $Q(\lambda)$ fikri, yüksek destekli yörünge bölümlerinde büyük $\lambda$, belirsiz geçişlerde küçük $\lambda$ kullanacak şekilde klinik veriye uyarlanabilir. Üçüncü yön, durum-uzayı manifoldu düzenlileştirmesi için üretici dünya modeli (generative world-model) veya difüzyon tabanlı (diffusion-based) yörünge simülatörleridir. Böyle bir model, veri setinde seyrek görülen ama klinik olarak gerçekçi sınır durumları sentezleyerek değer fonksiyonuna güvenli marjlar öğretebilir. Bu genişlemeler aynı ilkeyi korumalıdır: önce veri desteği ve sızıntısız değerlendirme, sonra politika iyileştirmesi.

# Sonuç

IQL final tarama iş akışı, sepsis dozaj politikalarını tek seferlik bir model çıktısı olarak değil, denetlenebilir bir çevrimdışı değerlendirme protokolü olarak ele alır. Kohort seçimi, zaman penceresi, state/action/reward mühendisliği, sızıntısız ön işleme (leakage-free preprocessing), IQL eğitimi, hiperparametre ızgarası, Aşama 1 finalist seçimi ve Aşama 2 çoklu-tohum değerlendirmesi aynı raporda izlenebilir hale gelir. Aşama 0 denetimi geçilmiştir; Aşama 1 altı destek-duyarlı finalist üretmiştir. Aşama 2 sonucunda final konfigürasyon ‘iql_sofa_shaped_conservative_safe’ olarak seçilir. Model FQE=2.848, WIS=8.203 ve bootstrap WIS 95% CI=4.963–10.817 verir; ancak ESS=29.41 ile güvenilirlik eşiğinin altındadır. Bu nedenle sonuç klinik performans kanıtı değil, metodolojik ve retrospektif politika dışı değerlendirme kanıtıdır.
