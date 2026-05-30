1

# Destek Sınırlı Klinik Koşullar Altında Sepsis Dozaj Politikaları İçin Orkestre Edilmiş Bir Çevrimdışı Pekiştirmeli Öğrenme İş Akışı

Medikal Bilişim Kıdemli Yapay Zeka Araştırmacısı

Abstract—Bu çalışma, MIMIC-IV veri seti üzerinde klinik sepsis dozaj politikalarının yeniden üretilebilir şekilde değerlendirilmesi amacıyla orkestre edilmiş bir çevrimdışı pekiştirmeli öğrenme (offline reinforcement learning) çerçevesi sunmaktadır. Yoğun bakım ünitesindeki terapötik müdahaleleri dört saatlik aralıklarla sonlu ufuklu bir Markov karar süreci olarak modelleyerek, intravenöz sıvı ve vazopresör dozajlarının ortak kombinasyonunu temsil eden 25 boyutlu bir aksiyon uzayı tanımlıyoruz. Doğrulama aşamasında aşırı uyum (overfitting) ve veri sızıntısı (data leakage) gibi kronikleşen sorunları çözmek amacıyla, hasta düzeyinde katı izolasyonu, zamansal yörünge hizalamasını ve veri bütünlüğü sınırlandırmalarını doğrulayan bir Aşama 0 Ön Tarama Denetim Çerçevesi (Presweep Auditing Framework) geliştirilmiştir. Aşama 1, ödül yapılarını, öğrenme oranlarını ve ekspektil-sıcaklık sınırlarını birbirinden ayıran 18 farklı konfigürasyondan oluşan bir hiperparametre taraması yürütür. Sadece tekil metrik sıralamalarına güvenmek yerine, Aşama 2 çoklu tohum doğrulaması için altı farklı finalist politikayı belirlemek üzere çok kriterli bir seçim protokolü uygulamaktayız. SOFA skoruna göre şekillendirilmiş bir ödül işlevi, muhafazakar öğrenme oranları ve güvenli örtük Q-öğrenme (safe implicit Q-learning) ekspektilleri kullanan konfigürasyon ('iql\_sofa\_shaped\_conservative\_safe'), 2.848 ortalama Uydurulmuş Q-Değerlendirmesi (FQE), 8.203 Ağırlıklı Önem Örneklemesi (WIS) değeri (%95 bootstrap güven aralığı: 4.963–10.817) ve 0.991 destek kütlesi üreterek en kararlı politika olarak öne çıkmaktadır. Önemli bir husus olarak, etkin örneklem büyüklüğü (ESS) standart güvenilirlik eşiğinin altına düştüğü için (29.410 < 50), bu bulguları klinik üstünlük kılavuzları olarak değil, kesin bir şekilde veri desteğiyle sınırlandırılmış politika dışı retrospektif değerlendirme (retrospective off-policy evaluation) kanıtları olarak sunuyoruz.

Index Terms—Çevrimdışı pekiştirmeli öğrenme, örtük Qöğrenme, sepsis, MIMIC-IV, politika dışı değerlendirme, klinik karar destek sistemleri.

#### I. Giriş

Sepsis, yoğun bakım ünitelerinde (YBÜ) akut organ yetmezliğine yol açan ve vücudun enfeksiyona karşı geliştirdiği kontrolsüz sistemik yanıtla karakterize edilen birincil mortalite nedenidir.[1] Resüsitasyon protokolleri genellikle intravenöz sıvılar ve vazopresörlerin karmaşık ve ardışık dozajlanmasını gerektirir; bu tedavilerin hastanın değişen fizyolojik profiline göre dinamik olarak ayarlanması şarttır.[2] Ancak, fizyolojik heterojenlik, tedaviye verilen yanıtların gecikmeli ortaya çıkması ve klinisyen kararlarındaki karıştırıcı değişkenler (confounding) nedeniyle optimal tedavi yörüngelerini belirlemek oldukça güçtür. Pekiştirmeli öğrenme (reinforcement learning), dinamik tedavi rejimlerini keşfetmede önemli bir potansiyele sahip olsa da, kritik bakım süreçlerinde gerçek zamanlı (online)

keşif süreçleri etik açıdan kabul edilemez.[3, 4] Bu engel, politika optimizasyonunun tamamen MIMIC-IV gibi retrospektif gözlemsel veri tabanlarına dayanmasını gerektiren çevrimdışı pekiştirmeli öğrenmeyi (offline reinforcement learning) zorunlu kılmaktadır.[4, 5]

Çevrimdışı pekiştirmeli öğrenme etik avantajlarına rağmen, dağılım kayması (distribution shift), dağılım dışı (out-of-distribution) değer ekstrapolasyonu ve doğrulama aşamasında aşırı uyum gibi ciddi teknik engellerle karşı karşıyadır.[6] Politika optimizasyonu sabit veri setleri üzerinde gerçekleştirildiğinde, standart değer tabanlı algoritmalar, eğitim veri setinde zayıf temsil edilen aksiyonların durum-aksiyon değerlerini (Q-değerlerini) aşırı tahmin etme eğilimindedir.[6] Bu çalışma, matematiksel güvenliği sağlamak, yeniden üretilebilir kohort tanımları oluşturmak ve katı klinik veri desteği sınırlamaları altında çevrimdışı örtük Q-öğrenme (implicit Q-learning) politikalarının çok kriterli değerlendirilmesini mümkün kılmak amacıyla orkestre edilmiş bir iş akışı sunmaktadır.[5]

## II. İlgili Çalışmalar ve Arka Plan

Mevcut literatürü taksonomik bir yaklaşımla ele alıyoruz. Sepsis tedavisinde klinik karar verme süreçleri, tarihsel olarak ayrık pekiştirmeli öğrenme paradigmalarıyla modellenmiştir. Komorowski ve meslektaşları, "AI Clinician" modelini tabular bir Markov karar süreci kullanarak gerçeklemiş ve optimal dozaj stratejilerinin doğrudan retrospektif elektronik sağlık kayıtlarından türetilebileceğini göstermişlerdir.[3] Bu formülasyon, Raghu ve arkadaşları tarafından derin sinir ağları kullanılarak sürekli durumlara genişletilmiş ve hasta fizyolojisi daha yüksek bir doğrulukla yakalanmıştır.[7] Bu algoritmik gelişmelere rağmen, Gottesman ve arkadaşları tarafından formüle edilen sağlık hizmetlerinde pekiştirmeli öğrenme kılavuzları, gözlemsel pekiştirmeli öğrenmenin karıştırıcı etkenlere, tahminci varyansına ve değerlendirme yanlılığına karşı son derece savunmasız olduğu konusunda uyarıda bulunmakta ve sistemik bir doğrulama yapılmadan doğrudan klinik uygulamaya geçirilmesini engellemektedir.[4]

Dağılım kaymasına karşı geliştirilen algoritmik önlemler, açık düzenlileştirme (explicit regularization) ve örtük değer öğrenimi (implicit value learning) olarak ikiye ayrılabilir. Muhafazakar Q-Öğrenme (Conservative Q-Learning - CQL), dağılım dışı aksiyonların durumaksiyon değerlerini açıkça cezalandırarak eleştirmeni (critic) veri setinde düşük desteğe sahip bölgeleri eksik tahmin

etmeye zorlar.[8] Teorik olarak güçlü olmasına rağmen, CQL manuel olarak kalibre edilmesi gereken ağır bir düzenlileştirme parametresi gerektirir.[8] Hatalı kalibrasyon riski, yüksek düzeyde heterojenlik gösteren hasta popülasyonlarında nadir fakat klinik açıdan havati olan kurtarma müdahalelerini agresif bir şekilde bastırabilir. Buna karşılık, Örtük Q-Öğrenme (Implicit Q-Learning - IQL), ekspektil regresyonu yardımıyla örtük bir değer işlevi oluşturarak dağılım dışı aksiyon sorgulamalarından tamamen kaçınır.[9] Değer işlevini yalnızca veri setinde mevcut olan aksiyonlar üzerinden değerlendiren IQL, gözlemlenmemiş geçişleri parametrelestirme veva düzenlilestirme ihtivacını ortadan kaldırarak ekstrapolasyon hatalarını azaltır. [6] Bu strateji. tarihsel dağılımda bulunmayan klinik volları kesfetmeme ödünlemesini kabul eder ve güvenli resüsitasyon kılavuzlarıyla uyumluluk gösterir.[1, 5]

#### III. Matematiksel Cerçeve ve Sepsis MDP Formülasyonu

Klinik karar süreci,  $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)$  bileşenlerinden oluşan sonlu ufuklu, iskonto edilmiş bir Markov karar süreci olarak modellenmiştir.[5, 10] Bir hastanın yoğun bakım kalış süresi, dört saatlık ayrık karar pencerelerine bölünmüştür.[5] Bu aralık, bilinçli bir klinik tavizdir. Daha kısa aralıklar gürültülü fizyolojik dalgalanmalara ve seyrek ölçümlere yol açarken; daha uzun aralıklar akut fizyolojik yanıtları ortalamaya katarak dinamik ilaç ayarlamalarının klinik etkisini gölgelemektedir.

Durum vektörü  $s \in \mathcal{S}$ ; demografik verileri, yaşamsal bulguları, laboratuvar ölçümlerini, tedavi maruziyetlerini ve eksik veri göstergelerini (missingness indicators) içerir.[5] Eksik veri göstergelerinin modele dahil edilmesi analitik açıdan temel bir gereksinimdir. Klinisyenler laboratuvar testlerini hastanın fizyolojik durumundaki kötüleşmeye yanıt olarak istedikleri için, bir ölçümün eksik olması rastgele gerçekleşmez. Eğer bir model, eksik veri göstergesini kaydetmeden normal fizyolojik değerler atarsa (imputation), ajan ölçülmeyen parametreleri sağlıklı durumlarla karıştırır. Bu durum, hekimin belirli fizyolojik yolları izleme yönündeki bilinçli kararında saklı olan örtük risk sinyallerinin kaybolmasına vol acar.[5]

Aksiyon uzayı  $\mathcal{A}$ , ortak intravenöz sıvı ve vazopresör dozaj yoğunluklarını eşleyen  $5\times 5$  boyutlarında ayrık bir ızgara şeklinde yapılandırılmıştır.[5] İskonto faktörünü, intermediate fizyolojik geçişlere olan duyarlılığı korurken terminal hayatta kalma sinyalini muhafaza etmek amacıyla  $\gamma=0.99$  olarak sabitliyoruz.[5] Daha düşük bir iskonto faktörü, ajanın aşırı derecede kısa vadeli organ skoru iyileşmelerine odaklanmasına neden olarak, gecikmeli kardiyovasküler yetmezliğe yol açabilecek agresif tedavileri teşvik edebilir.

Durum-değer işlevi  $V_{\psi}(s)$ , veri seti üzerinde ekspektil regresyonu ile tahmin edilir:

$$L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_2^{\tau} \left( Q_{\theta}(s,a) - V_{\psi}(s) \right) \right] \tag{1}$$

Burada  $L_2^{\tau}(u) = |\tau - \mathbb{I}(u < 0)|u^2$  asimetrik kaybı temsil eder.[6, 9]  $\tau \in (0.5, 1.0)$  parametresi değer iyimserliğini

düzenler.[9] Durum-aksiyon değer işlevi, ortalama karesel Bellman hatası üzerinden eğitilir:

$$L_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma V_{\psi}(s') - Q_{\theta}(s,a) \right)^2 \right]$$
 (2)

Hedef politika, avantaj ağırlıklı regresyon (advantageweighted regression) yoluyla çıkarılır:

$$L_{\pi}(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp \left( \beta \left( Q_{\theta}(s,a) - V_{\psi}(s) \right) \right) \log \pi_{\phi}(a|s) \right]$$
(3)

Burada  $\beta$  parametresi politika çıkarım sıcaklığını kontrol eder.[9]

İki farklı ödül formülasyonunu karşılaştırıyoruz. Seyrek ödül (sparse reward), hastanın taburculuk durumuna göre ikili bir değer atayarak terminal hayatta kalma faydasını doğrudan yansıtır.[5] Klinik faydayı doğrudan temsil etse de, bu formülasyon uzun karar ufuklarında kredi atama (credit assignment) problemini zorlaştırır. Bunu hafifletmek amacıyla, SOFA-şekilli ödül fonksiyonu, Ardışık Organ Yetmezliği Değerlendirmesi (SOFA) skorundaki adım adım değişimleri dahil eder [5]:

$$R_{\text{SOFA}}(s_t, a_t) = R_{\text{sparse}}(s_t, a_t) + \lambda \left( \text{SOFA}_{t-1} - \text{SOFA}_t \right)$$
(4)

Burada  $\lambda$  dengeleyici bir katsayıdır. SOFA takibi ara geri bildirimler sağlasa da, güçlü ödül şekillendirme (reward shaping) süreçleri, ajanın nihai hayatta kalma yerine geçici organ skoru iyileşmelerini optimize ettiği patolojik döngülere yol açma riski taşır.

#### IV. Veri Bütünlüğü ve Ön Tarama Denetimi

Sıkı veri protokolleri oluşturmak amacıyla, analiz süreclerinden önce veri seti özetlerini doğrulayan bir Aşama 0 Ön Tarama Denetim Cerçevesi (Phase 0 Pre-sweep Auditing Framework) tanımlıyoruz.[5] Retrospektif EHR analizleri, eğitim ve doğrulama setleri arasındaki çakışan hasta yörüngelerinin yapay olarak yüksek değerlendirme skorlarına yol açtığı bölüntü kontaminasyonuna (split contamination) karşı son derece hassastır.[4] Denetim katmanı, hasta kohortlarının hasta düzeyinde izole edildiğini kontrol eder, durum-aksiyon-sonraki durum geçişlerinin zamansal hizalamasını doğrular ve veri sözlesmelerini zorunlu kılar. En önemlisi, özniteliklerin terminal taburculuk durumuna dair herhangi bir gösterge içermediğini doğrulamak için kapsamlı bir sonuc sızıntısı (outcome leakage) taraması gerçekleştirir. Tablo I, Aşama 0 denetim parametrelerini detaylandırmaktadır.

TABLE I Aşama 0 Ön Tarama Denetimi ve Veri Protokolü Doğrulaması

| Kohort Bölüntüleri (Hasta Sayısı) 12,063 2,585 2,5 SOFA-Biçimli Deneyim Geçişleri 140,635 30,539 30, Seyrek Ödüllü Deneyim Geçişleri 140,635 30,539 30, Bölüntü Sızıntı Denetimi Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başarılı Başı | Denetim Parametresi                                                                                                      | Eğitim Seti                                | Doğrulama Seti                           | Test                     |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|------------------------------------------|--------------------------|
| DOING DIZINGSI DOZINGANGSI DASAHI DASAHI DAS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | SOFA-Biçimli Deneyim Geçişleri<br>Seyrek Ödüllü Deneyim Geçişleri<br>Bölüntü Sızıntı Denetimi<br>Ortak Deneyim Protokolü | 140,635<br>140,635<br>Başarılı<br>Başarılı | 30,539<br>30,539<br>Başarılı<br>Başarılı | 30,<br>30,<br>Baş<br>Baş |

Doğrulanmış veri setleri kullanılarak Aşama 1 kapsamında 18 adaylı bir hiperparametre taraması gerçekleştirilir (Tablo II). Değer öğrenimi, politika öğrenimi ve ödül gösterimlerini birbirinden bağımsız hale getiriyoruz. Muhafazakar öğrenme oranları, referans ve aktör-muhafazakar rejimlerle birlikte değerlendirilmektedir. Muhafazakar öğrenme oranı kritik öneme sahiptir: eleştirmenin hızlı güncellemelerle dağılım dışı Q-değeri aşırı tahminlerini yaymasını engeller ve politika çıkarımını stabilize eder.[11] Güvenli, referans ve iyimser IQL ayarları, politika çıkarım güvenliğinin sınırlarını haritalamak için ekspektil  $\tau$  ve avantaj sıcaklığı  $\beta$  değerlerini çeşitlendirir.[5, 11]

TABLE II IQL Hiperparametre Tarama Izgarası (Aşama 1)

| Hiperparametre Boyutu            | Değerlendirilen Ayarlar u                       |
|----------------------------------|-------------------------------------------------|
| Ödül Formülasyonu                | Seyrek, SOFA-Biçimli d                          |
| Öğrenme Oranı Rejimi             | Muhafazakar, Referans, Aktör-Muha               |
| IQL Parametre Kurulumu           | Güvenli ( $\tau = 0.7, \beta = 0.5$ ), Referans |
|                                  | $i_{yimser} (\tau = 0.9, \beta = 2.0) $ et      |
| Aşama 1 Referans Tohumu          | 42                                              |
| Aşama 2 Doğrulama Tohumlar       | 1 123, 456                                      |
| Sabit İskonto Faktörü $(\gamma)$ | 0.99                                            |
| Aksiyon Uzayı Boyutları          | 25 Avrik Tedavi Kutusu (5 × 5 Orta              |

## V. Çok Kriterli Seçim ve Sonuçların Tartışılması

Klinik uygulamalarda, politikaları yalnızca en yüksek FQE veya WIS skorlarına göre seçmek tehlikelidir.[4] Yüksek WIS tahminleri, az sayıda yörünge tarafından domine edilebilir; bu durum tahminci varyansını artırır ve etkin örneklem büyüklüğünü (ESS) düşürür.[12] Bu sorunu çözmek için, dengeli bir finalist havuzu oluşturmak üzere altı adet önceden tanımlanmış metrik yuvası genelinde çok kriterli bir seçim protokolü uyguluyoruz (Tablo III). Bu seçim, aday setini yapısal olarak farklı modellerle beslemek için çeşitlilik odaklı tahsisatı (diversity-driven allocation) içerir ve seyrek ödül varyantları ile muhafazakar konfigürasyonların Aşama 2'de değerlendirilmesini garanti altına alır.

Aşama 2, değerlendirme varyansını kantitatif olarak belirlemek amacıyla altı finalisti üç rastgele tohum (42, 123, 456) üzerinde değerlendirir (Tablo IV). SOFAşekilli ödül, muhafazakar öğrenme oranı ve güvenli IQL ekspektil-sıcaklık ayarını birleştiren konfigürasyon ('iql\_sofa\_shaped\_conservative\_safe'), 5.858 bileşik skor, 2.848 ortalama FQE ve 8.203 ortalama WIS değerlerine ulaşarak optimal politika olarak öne çıkmaktadır.

## VI. Aksiyon ve Güvenlik Teşhisleri

Klinik karar politikalarını değerlendirmek, değer tahmininin ötesinde kapsamlı teşhisler gerektirir.[4] Kazanan politikanın test performansı Tablo V'te özetlenmiştir.

Politika yüksek davranışsal destek kütlesi (0.991) ve 0.414 klinisyen tam kutu uyumu sergilese de, etkin örneklem büyüklüğü (ESS) 29.410'dur ve bu değer klinik güvenilirlik eşiği olan 50'nin altındadır.[5] 29.410 düzeyindeki bir

ESS değeri, politikanın WIS tahminlerinin dar bir yörünge kümesi tarafından domine edildiğini ve bunun sonucunda 4.963 ila 10.817 arasında geniş bir %95 bootstrap güven aralığı oluştuğunu göstermektedir. Bu yüksek tahminci varyansı nedeniyle, elde edilen bulgular standart tedaviye göre klinik bir üstünlük iddiası olarak yorumlanmamalıdır. Bunun yerine, destek sınırlı retrospektif koşullar altında politika dışı optimizasyon kararlılığının bir kanıtı olarak değerlendirilmelidir.

Ayrıca, tam kutu uyumu 0.414 seviyesinde olmasına rağmen, desteklenmeyen aksiyon sapması (non-support action divergence) son derece düşüktür (0.009). Bu durum, ajanın klinisyenin tam tercihiyle uyuşmadığı anlarda (durumların %58.6'sında klinik bir kaymayı temsil eder), yine de ampirik veriler tarafından güçlü şekilde desteklenen alternatif aksiyonlar önerdiğini göstermektedir. Politika, düzensiz veya dağılım dışı müdahaleler önermek yerine, veri desteğiyle uyumlu komşu yörüngeleri seçmektedir. Bu muhafazakar davranış, güvenli IQL konfigürasyonunun gerçek dünya klifir keri sınırlamaları altında ekstrapolasyon hatalarını sınırlamı sınırlamaları altında ekstrapolasyon hatalarını etkili bir şekilde sınırlandırdığını doğrulamaktadır.

#### VII. Sonuç

Ortak Buganlışma, sepsis tedavi politikası tasarımında yeniden üretilebilirliği ve veri güvenliğini zorunlu kılan orkestre edilmiş bir çevrimdışı pekiştirmeli öğrenme iş akışı sunmaktadır. Geliştirilen Aşama 0 denetim çerçevesi, çok kriterli seçim protokolü ve çoklu tohum doğrulaması sayesinde, klinik pekiştirmeli öğrenmenin temel hata modları hafifletilmiştir. 'iql\_sofa\_shaped\_conservative\_safe' konfigürasyonunun seçilmesi, muhafazakar öğrenme güncellemelerinin güvenli ekspektil sınırlarıyla birleştirilmesinin kararlı değer tahminleri sağladığını doğrulamaktadır. En önemlisi, ESS sınırlamasının açık bir şekilde raporlanması, şeffaf ve risk bilincine sahip klinik politika raporlaması için metodolojik bir şablon sunmaktadır.

TABLE III Çok Kriterli Sınırlamalar Altında Aşama 1 En İyi Altı Adayın Seçimi

| Derece | Yuva Ataması      | Ödül   | Öğrenme Oranı | IQL Ayarları | FQE Değeri | WIS Değeri | Destek Kütlesi | Klinisyen Uyumu |
|--------|-------------------|--------|---------------|--------------|------------|------------|----------------|-----------------|
| 1      | En İyi Bileşik    | SOFA   | Muhafazakar   | Güvenli      | 3.169      | 8.616      | 0.965          | 0.418           |
| 2      | En İyi Bileşik    | SOFA   | Muhafazakar   | Referans     | 2.280      | 7.009      | 0.973          | 0.401           |
| 3      | En İyi Seyrek     | Seyrek | Muhafazakar   | Güvenli      | 1.954      | 8.740      | 0.971          | 0.411           |
| 4      | Güvenlik Desteği  | Seyrek | Referans      | Referans     | 1.421      | 6.768      | 0.973          | 0.410           |
| 5      | Referans Çapa     | Seyrek | Referans      | Güvenli      | 2.345      | 8.125      | 0.963          | 0.416           |
| 6      | Çeşitlilik Odaklı | Seyrek | Muhafazakar   | Referans     | 2.012      | 7.782      | 0.969          | 0.407           |

TABLE IV Aşama 2 Çok Tohumlu Finalist Değerlendirmesi ve Birleştirilmiş Teşhisler

| Derece | Konfigürasyon Adayı         | Tür   | Skor  | FQE Ort. | WIS Ort. | ESS    | Destek Kütlesi | Klinisyen Uyumu |
|--------|-----------------------------|-------|-------|----------|----------|--------|----------------|-----------------|
| 1      | SOFA Muhafazakar Güvenli    | Final | 5.858 | 2.848    | 8.203    | 29.410 | 0.991          | 0.414           |
| 2      | SOFA Muhafazakar Referans   | Aday  | 5.288 | 2.366    | 8.286    | 26.120 | 0.990          | 0.404           |
| 3      | Seyrek Muhafazakar Güvenli  | Aday  | 5.262 | 2.316    | 8.169    | 31.320 | 0.990          | 0.411           |
| 4      | Seyrek Muhafazakar Referans | Aday  | 5.033 | 2.050    | 8.536    | 26.100 | 0.990          | 0.401           |
| 5      | Seyrek Referans Güvenli     | Aday  | 4.940 | 2.063    | 7.882    | 31.310 | 0.991          | 0.418           |
| 6      | Seyrek Referans Referans    | Aday  | 4.755 | 1.535    | 9.755    | 18.960 | 0.991          | 0.409           |

TABLE V Kazanan Konfigürasyonun Test Teşhisleri

| Değerlendirme Metriği                                  | Deneysel Sonuç |
|--------------------------------------------------------|----------------|
| Ortalama Uydurulmuş Q-Değerlendirmesi (FQE)            | 2.848          |
| Ağırlıklı Önem Örneklemesi (WIS)                       | 8.203          |
| WIS %95 Bootstrap Güven Aralığı                        | 4.963 – 10.817 |
| Etkin Örneklem Büyüklüğü (ESS)                         | 29.410         |
| Davranışsal Destek Kütlesi<br>Klinisyen Tam Kutu Uyumu | 0.991<br>0.414 |
| Klinisyen Uyuşmazlığı (Kaydırılmış Aksiyonlar)         | 0.586          |
| Desteklenmeyen Aksiyon Sapması                         | 0.009          |