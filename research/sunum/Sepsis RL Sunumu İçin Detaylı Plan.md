# **Destek Sınırlı Klinik Koşullar Altında Sepsis Dozaj Politikaları İçin Orkestre Edilmiş Bir Çevrimdışı Pekiştirmeli Öğrenme İş Akışı: Akademik Sunum Tasarımı ve Metodolojik Rapor**

## **Klinik Karar Analitiğinde Çevrimdışı Pekiştirmeli Öğrenmenin Rolü ve İstatistiksel Bariyerler**

Yoğun bakım ünitelerinde sepsis resüsitasyonu, hastaların intravenöz sıvı ve vazopresör ihtiyaçlarının saatlik olarak dinamik bir şekilde değiştiği, mortalite oranları son derece yüksek bir klinik tabloyu temsil etmektedir.1 Klinik kararların gecikmeli mortalite sonuçlarıyla doğrudan ilişkili olması ve ağır durumdaki hastaların doğal olarak daha erken ve agresif müdahalelere maruz kalması, gözlemsel verilerde "endikasyon karıştırıcılığı" (*confounding by indication*) adı verilen kronik bir saptırmaya yol açmaktadır.1 Bu klinik senaryoda, doğrudan hastalar üzerinde yeni ve bilinmeyen tedavi stratejilerini denemeye odaklanan çevrimiçi keşif (*online exploration*) süreçleri tamamen etik dışı ve tehlikelidir.1 Bu kısıt, araştırmacıları yalnızca geçmiş yoğun bakım yörüngelerini içeren retrospektif elektronik sağlık kayıtları üzerinden politika öğrenimini mümkün kılan çevrimdışı pekiştirmeli öğrenme (*offline reinforcement learning*) yöntemlerine yöneltmektedir.1  
Ancak çevrimdışı pekiştirmeli öğrenmede, öğrenilen politikanın veri kümesindeki tarihsel klinisyen davranışlarından saptığı durumlarda, uydurulmuş Q-değerlendirmesi (*Fitted Q-Evaluation \- FQE*) veya ağırlıklı önem örneklemesi (*Weighted Importance Sampling \- WIS*) gibi politika dışı değerlendirme (*off-policy evaluation \- OPE*) yöntemleri yapay bir iyimserlik gösterebilir.1 Gerçekte ise model, veride hiç temsil edilmemiş güvensiz aksiyonları asılsız bir şekilde yüksek değerli atfederek "dışdeğer biçimleme hatasına" (*extrapolation error*) düşmüş olabilir.1 Bu raporda, MIMIC-IV veritabanı üzerinde kurgulanan sepsis resüsitasyon iş akışının, klinik karar vericiler ve akademik kurullar için ne derece güvenli ve istatistiksel olarak denetlenebilir olduğu, veri sınırları ve algoritmik takaslar üzerinden analiz edilmektedir.1

## **Projenin Uçtan Uca Mimari Verileri ve Sözleşmeleri**

İş akışının tüm adımları, veri hazırlama aşamasındaki küçük bir sızıntının veya hatalı eşleştirmenin nihai model üzerinde yapay performans artışlarına yol açmasını engellemek üzere tasarlanmıştır.1 Bu doğrultuda kurgulanan on aşamalı uçtan uca mimari, Tablo I üzerinde detaylandırılmıştır.

### **Tablo I: Projenin On Aşamalı Uçtan Uca Mimarisi**

| Adım | Bileşen | Metodolojik İçerik ve Güvenlik Güvencesi |
| :---- | :---- | :---- |
| **1** | Kohort Seçimi | MIMIC-IV üzerinden Sepsis-3 kriterlerine uyan 18 yaş üzeri yetişkin hastaların seçimi; geçersiz onset zamanı olanlar, ICU kalış süresi 4 saatten kısa olanlar, demografik verisi eksik olanlar ve tekrar yatan hastaların elenmesi.1 |
| **2** | Zaman Çizelgesi | Sepsis onset'inden 24 saat önce başlayıp 48 saat sonrasına uzanan 72 saatlik akut klinik gözlem penceresinin 4 saatlik karar adımlarına bölünmesi (\~18 zaman adımlı epizotlar).1 |
| **3** | Sıfır Sızıntılı Bölme | Hastaların eğitim (%70), doğrulama (%15) ve test (%15) olarak hasta düzeyinde ayrılması; tüm ön işleme ve normalleştirme parametrelerinin (fit) yalnızca eğitim verisinden öğrenilmesi.1 |
| **4** | Durum Çıkarımı | Yaşamsal bulgular, laboratuvarlar, kümülatif tedavi geçmişi, demografi ve hekimin test isteme sıklığını yansıtan eksiklik göstergelerinden (*missingness flags*) oluşan 62 boyutlu durum vektörü üretilmesi.1 |
| **5** | Aksiyon/Ödül | Vazopresör ve IV sıvı dozlarının 25 ayrık aksiyona (![][image1] grid) dönüştürülmesi; seyrek (terminal mortalite) ve SOFA-biçimli şekillendirme ödüllerinin üretilmesi.1 |
| **6** | Geçiş/Başlangıç Çizgisi | Replay verisinin (![][image2]) formatında yazılması; klinisyen tekrar oynatımı (*clinician replay*), tedavisiz kontrol ve davranış klonlama (*behavior cloning*) başlangıç çizgilerinin kurulması.1 |
| **7-8** | IQL Eğitimi | Veri desteği dışına çıkmadan asimetrik ekspektil değer öğrenimi ve avantaj ağırlıklı aktör güncellemeleriyle Implicit Q-Learning (IQL) modellerinin eğitilmesi.1 |
| **9** | OPE ve Güvenlik | FQE, WIS, ESS, bootstrap güven aralıkları (%95 CI), davranışsal destek kütlesi, klinisyen tam-kutu uyumu ve düşük destekli aksiyon teşhislerinin hesaplanması.1 |
| **10** | Final Tarama | 18 farklı IQL konfigürasyonunu içeren Aşama 1 taraması, finalistlerin seçimi, 3 farklı rastgele tohumla (*seed*) Aşama 2 doğrulaması ve IEEE raporlama çıktılarının üretilmesi.1 |

Bu mimaride, aksiyon uzayının klinisyen davranışlarına ve klinik kılavuzlara uygun şekilde kodlanması büyük önem arz etmektedir.1 Bu amaçla vazopresör ve intravenöz sıvı dozları, Tablo II'de gösterilen ![][image1] boyutlu ayrık bir karar matrisi üzerinde tanımlanmıştır.

### **Tablo II: Vazopresör ve IV Sıvı Aksiyon Gridi**

| Vazopresör Bini / Sıvı Bini | Bin 0 (Tedavi Yok) | Bin 1 (Q1 Doz) | Bin 2 (Q2 Doz) | Bin 3 (Q3 Doz) | Bin 4 (Q4 Doz) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Bin 0 (Tedavi Yok)** | Aksiyon 0 | Aksiyon 1 | Aksiyon 2 | Aksiyon 3 | Aksiyon 4 |
| **Bin 1 (Q1 Doz)** | Aksiyon 5 | Aksiyon 6 | Aksiyon 7 | Aksiyon 8 | Aksiyon 9 |
| **Bin 2 (Q2 Doz)** | Aksiyon 10 | Aksiyon 11 | Aksiyon 12 | Aksiyon 13 | Aksiyon 14 |
| **Bin 3 (Q3 Doz)** | Aksiyon 15 | Aksiyon 16 | Aksiyon 17 | Aksiyon 18 | Aksiyon 19 |
| **Bin 4 (Q4 Doz)** | Aksiyon 20 | Aksiyon 21 | Aksiyon 22 | Aksiyon 23 | Aksiyon 24 |

Ön tarama denetimi (*presweep audit*), eğitim süreci başlamadan önce veri bütünlüğünü ve sızıntısızlık sözleşmelerini güvence altına alan en önemli kontrol kapısıdır.1 Tablo III'te gösterilen denetim özet verileri, eğitim, doğrulama ve test bölmeleri arasında hasta çakışması bulunmadığını ve gelecek mortalite bilgisinin durum özelliklerine sızmadığını doğrulamaktadır.1

### **Tablo III: Ön Tarama Denetim Özeti (Presweep Audit Summary)**

| Kanıt Kalemi | Eğitim (Train) | Doğrulama (Val) | Test | Denetim Durumu |
| :---- | :---- | :---- | :---- | :---- |
| **Split Manifest Satırı** | 12.063 hasta | 2.585 hasta | 2.585 hasta | Başarılı (Kesişim Boş) |
| **SOFA Replay Satırı** | 140.635 geçiş | 30.539 geçiş | 30.046 geçiş | Başarılı (Tutarlı) |
| **Seyrek Replay Satırı** | 140.635 geçiş | 30.539 geçiş | 30.046 geçiş | Başarılı (Tutarlı) |
| **Split Leakage Kontrolü** | Yok | Yok | Yok | Başarılı (Sızıntı Yok) |
| **Outcome Leakage Taraması** | Yok | Yok | Yok | Başarılı (Gelecek Bilgisi Yok) |

## **Slayt Stratejisi ve Görsel İçerik Tasarımı (Maksimum 7 Slayt)**

Akademik kurulun ve ilgili öğretim üyesinin klinik güvenlik, veri kısıtları ve model teşhisleri konusundaki hassasiyetini karşılamak üzere tasarlanan 7 slaytlık sunum planı aşağıda sunulmuştur.

### **Slayt 1: Sepsis Karar Analitiği ve Çevrimdışı RL Zorunluluğu**

* **Slayt Başlığı:** Sepsis Resüsitasyonunda Karar Analitiği ve Çevrimdışı RL Zorunluluğu  
* **Slayt Maddeleri:**  
  * **Klinik Tehdit:** Sepsis, dünya genelinde yılda 11 milyon ölüme yol açan ve organ disfonksiyonuyla seyreden yüksek mortaliteli bir tablodur.1  
  * **Endikasyon Karıştırıcılığı (Confounding by Indication):** Ağır hastaların daha agresif tedavileri daha erken alması, gözlemsel verilerde saptırıcı korelasyonlar yaratarak nedensel analizi zorlaştırır.1  
  * **Çevrimdışı RL Tercihi:** Çevrimiçi deneme-yanılma (online exploration) klinik ve etik olarak imkansızdır; tek seçenek tarihsel yoğun bakım kayıtlarıdır.1  
  * **Metodolojik Hedef:** Yapay zeka başarılarını abartmadan, veri sınırları ve klinik kısıtlar altında denetlenebilir ve sızıntısız bir değerlendirme protokolü kurmak.1  
* **Görsel Öneri:** MIMIC-IV veri tabanından başlayan, dışlama kriterlerini uygulayan ve 72 saatlik analiz pencerelerine bölünen hasta kohort akış şeması.1

### **Slayt 2: MDP Tasarımı ve Aşama 0 Ön Tarama Denetimi**

* **Slayt Başlığı:** Markov Karar Süreci (MDP) Formülasyonu ve Sızıntısızlık Denetimi  
* **Slayt Maddeleri:**  
  * **62 Boyutlu Durum Temsili:** Fizyolojik sinyaller, kümülatif tedavi geçmişi ve hekimin karar anındaki şüphesini yansıtan eksiklik göstergeleri (*missingness flags*).1  
  * **25 Ayrık Aksiyon Gridi:** IV sıvı ve vazopresör dozajlarının çeyrekliklere göre ayrıştırıldığı ![][image1] ayrık karar matrisi.1  
  * **Aşama 0 Ön Tarama Denetimi (Presweep Audit):** Sonuç sızıntısını ve veri bölme hatalarını eğitimden önce bloke eden sıkı veri sözleşmeleri.1  
  * **Sıfır Sızıntı İlkesi:** "Eğitime uydur, her yerde dönüştür" protokolü uyarınca, ön işleme parametrelerinin doğrulama ve test kümelerine sızmasının engellenmesi.1  
* **Görsel Öneri:** Slaytta Tablo II (Aksiyon Gridi) ve Tablo III (Ön Tarama Denetim Tablosu) yan yana konumlandırılacaktır.1

### **Slayt 3: Algoritmik Güvenlik Takası: Neden IQL?**

* **Slayt Başlığı:** Algoritmik Güvenlik Takası: Neden CQL Değil de IQL?  
* **Slayt Maddeleri:**  
  * **Muhafazakar Q-Öğrenme (CQL) Riski:** CQL veri dışı aksiyonları açıkça cezalandırarak alt sınır kurar; ancak aşırı cezalandırma, nadir fakat hayat kurtarıcı acil durum tedavilerini (*rescue therapies*) baskılama riski barındırır.1  
  * **Örtük Q-Öğrenme (IQL) Güvenliği:** Veri setinde hiç görülmemiş aksiyonları asla sorgulamayarak dışdeğer biçimleme hatasını (*extrapolation error*) tamamen engeller.1  
  * **Klinik Takas:** Radikal tedavi keşfetme yeteneğinden bilinçli olarak vazgeçilmesi pahasına, hastayı tamamen tarihsel veri desteği (*support mass*) içinde tutma güvencesi.1  
  * **Asimetrik Ekspektil Regresyonu:** Değer fonksiyonunun ![][image3] ekspektil kaybıyla uydurularak aktör ve eleştirmen güncellemelerinin istikrarlı biçimde ayrıştırılması.1  
* **Görsel Öneri:** IQL'in asimetrik ekspektil kayıp fonksiyonunun (![][image4]) matematiksel eğrisini ve veri içi supremum değer tahmin mantığını gösteren grafik şema.1

### **Slayt 4: Deneysel Sonuçlar ve En Dengeli Adayın Seçimi**

* **Slayt Başlığı:** Deneysel Sonuçlar ve Kazanan Konfigürasyonun Belirlenmesi  
* **Slayt Maddeleri:**  
  * **Sistematik Tarama:** Aşama 1 kapsamında 18 farklı konfigürasyon taranmış; Aşama 2'de en iyi 6 finalist 3 farklı rastgele tohumla test edilmiştir.1  
  * **Kazanan Konfigürasyon:** iql\_sofa\_shaped\_conservative\_safe modelinin en yüksek bileşik skora ulaşarak birinci seçilmesi.1  
  * **Ödül Tasarımının Etkisi:** Ara adımlarda SOFA değişimlerini cezalandıran şekillendirme ödülü, yalnızca terminal mortaliteyi hedefleyen seyrek ödüllere göre üstün fizyolojik gidişat sağlamıştır.1  
  * **Konservatif LR Etkisi:** Yavaşlatılmış öğrenme oranları, eleştirmen ağındaki yapay Q-değeri şişmelerini engellemiştir.1  
* **Görsel Öneri:** Slaytta Tablo VII (Aşama 2 Üç-Tohum Değerlendirmesi) gösterilecek ve kazanan modelin satırı vurgulanacaktır.1

### **Slayt 5: Kritik Görsel Güvenlik ve Politika Dışı Değerlendirme (OPE) Analizleri**

* **Slayt Başlığı:** Güvenlik Teşhisleri: FQE vs. Davranış Desteği ve WIS vs. ESS Analizi  
* **Slayt Maddeleri:**  
  * **Değer ve Destek İlişkisi:** Değer tahminlerinin (FQE) yalnızca yeterli davranışsal destek kütlesi (*support mass*) içeren güvenli bölgelerde anlamlı kabul edilmesi.1  
  * **Davranış Klonlama (BC) İllüzyonu:** BC modelinin yüksek WIS skoru üretmesine rağmen, ESS değerinin 1.6'ya çökmesi nedeniyle istatistiksel açıdan geçersiz olması.1  
  * **Klinisyenle Güvenli Sapma:** Modelin klinisyenle tam-kutu uyumu %41.4 düzeyindedir; hekimden saptığı durumlarda ise "desteklenmeyen aksiyon sapması" %0.9 gibi yok denecek kadar az bir seviyede kalmıştır.1  
* **Görsel Öneri:** Doğrulama değer tahmini ile destek teşhislerini bir arada gösteren Şekil 1 (FQE vs Support Mass) ve Şekil 2 (WIS vs ESS Diagnostics) grafikleri.1

### **Slayt 6: İstatistiksel Sınırlar ve ESS Engeli**

* **Slayt Başlığı:** İstatistiksel Bariyerler ve Klinik Üstünlük İddiasının Reddedilmesi  
* **Slayt Maddeleri:**  
  * **Düşük ESS Gerçeği:** Tüm finalist modellerde Etkin Örneklem Büyüklüğünün (ESS \= 29.410) güvenilirlik eşiği olan 50'nin altında kalması.1  
  * **İstatistiksel Sonuç:** Az sayıda yüksek ağırlıklı yörüngenin değer tahminlerini domine etmesi nedeniyle politika değerlendirme varyansının çok yüksek olması.1  
  * **Klinik Üstünlük İddiasının Engeli:** Bu istatistiksel sınır, model için prospektif bir klinik üstünlük iddiası kurulmasını kesin olarak engeller.1  
  * **POMDP Gerçeği:** Yoğun bakımdaki karar süreçlerinin kısmen gözlemlenebilir olması; durum vektörünün hekimin yatak başındaki tüm sezgi ve kısıtlarını kapsayamaması.1  
* **Görsel Öneri:** Üç farklı tohum için hasta düzeyinde elde edilen geniş bootstrap WIS %95 güven aralığı dağılım grafiği (Şekil 7).1

### **Slayt 7: Gelecek Çalışmalar ve Dinamik Güvenlik Katmanları**

* **Slayt Başlığı:** Gelecek Çalışmalar: Dinamik Ekspektiller ve Çalışma Zamanı Filtreleri  
* **Slayt Maddeleri:**  
  * **Dinamik Ekspektil Kalibrasyonu:** Veri desteğinin yoğun olduğu bölgelerde daha agresif iyileştirme (![][image5]), seyrek bölgelerde ise muhafazakar davranış klonlama (![][image6]).1  
  * **Çalışma Zamanı Koruyucu Filtreler (Runtime Safety Filters):** Surviving Sepsis Campaign klinik kılavuz ihlallerini canlı akışta %97 oranında azaltan koruyucu katmanların sisteme entegrasyonu.1  
  * **Çok Adımlı İz Harmanlama:** Seyrek terminal ödül sinyalini yörünge boyunca daha kararlı yaymak amacıyla uyarlanabilir Peng'in ![][image7] entegrasyonu.1  
  * **Metodolojik Katkı:** Çalışma, yatak başı tedavi önermek yerine, retrospektif politika değerlendirme süreçleri için katı ve sızdırmaz bir metodolojik standart sunar.1  
* **Görsel Öneri:** IQL algoritmasına dinamik ekspektil kalibrasyonu ve çalışma zamanı güvenlik filtresi (Safe Actions Filter) ekleyen gelecek mimari akış şeması.1

## **Sistematik Hiperparametre Taraması ve Final Değerlendirme Verileri**

Modelin en başarılı adayını belirlemek üzere kurgulanan Aşama 1 ve Aşama 2 deneysel süreçlerine ait tüm nicel veriler, Tablo IV, Tablo VII, Tablo IX ve Tablo X üzerinde ayrıntılı bir biçimde sunulmuştur.1

### **Tablo IV: Final IQL Hiperparametre Izgarası (Hyperparameter Grid)**

| Hiperparametre | Sınıflandırma | Atanan Değerler ve Algoritmik Karşılıkları |
| :---- | :---- | :---- |
| **Ödül Tasarımı** | Seyrek (Sparse) | Terminal 90 günlük sağkalım için \+15, ölüm için \-15 fayda değeri.1 |
| **Ödül Tasarımı** | SOFA-Biçimli | Terminal faydaya ek olarak ara adımlarda: ![][image8].1 |
| **LR Rejimi** | Konservatif | Aktör öğrenme hızı: ![][image9], Eleştirmen öğrenme hızı: ![][image10], Değer öğrenme hızı: ![][image10].1 |
| **LR Rejimi** | Referans (Baseline) | Aktör öğrenme hızı: ![][image10], Eleştirmen öğrenme hızı: ![][image9], Değer öğrenme hızı: ![][image9].1 |
| **LR Rejimi** | Aktör-Konservatif | Aktör öğrenme hızı: ![][image9], Eleştirmen öğrenme hızı: ![][image9], Değer öğrenme hızı: ![][image9].1 |
| **IQL Ayarı** | Güvenli (Safe) | Ekspektil değeri (![][image3]): 0.7, Politika sıcaklık katsayısı (![][image11]): 1.0 (Konservatif politika çıkarımı).1 |
| **IQL Ayarı** | Referans | Ekspektil değeri (![][image3]): 0.7, Politika sıcaklık katsayısı (![][image11]): 3.0 (Dengeli politika çıkarımı).1 |
| **IQL Ayarı** | İyimser (Optimistic) | Ekspektil değeri (![][image3]): 0.8, Politika sıcaklık katsayısı (![][image11]): 3.0 (Daha agresif politika iyileştirmesi).1 |

Bu hiperparametre uzayının Aşama 1 ön elemesinden geçen finalistler, Aşama 2'de üç farklı rastgele tohum (seed 42, 123, 456\) ile yeniden eğitilmiş ve test kümesi üzerinde değerlendirilmiştir.1 Elde edilen uç-tohum birleşik seçim sonuçları Tablo VII'de yer almaktadır.

### **Tablo VII: Aşama 2 Üç-Tohum Finalist Değerlendirmesi**

| Sıra | Konfigürasyon | Ayar Sınıfı | Bileşik Seçim Skoru | Ortalama FQE | Ortalama WIS | Ortalama ESS | Davranışsal Destek | Klinisyen Uyumu |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **SOFA\_konservatif\_güvenli** | **Finalist** | **5.858** | **2.848** | **8.203** | **29.410** | **0.991** | **0.414** |
| **2** | SOFA\_konservatif\_referans | Aday | 5.288 | 2.366 | 8.286 | 26.120 | 0.990 | 0.404 |
| **3** | sparse\_konservatif\_güvenli | Aday | 5.262 | 2.316 | 8.169 | 31.320 | 0.990 | 0.411 |
| **4** | sparse\_konservatif\_referans | Aday | 5.033 | 2.050 | 8.536 | 26.100 | 0.990 | 0.401 |
| **5** | sparse\_baseline\_güvenli | Aday | 4.940 | 2.063 | 7.882 | 31.310 | 0.991 | 0.418 |
| **6** | sparse\_baseline\_referans | Aday | 4.755 | 1.535 | 9.755 | 18.960 | 0.991 | 0.409 |

Eğitilen modellerin kalitesini ölçmek adına, test aşamasında elde edilen sonuçlar üç farklı referans başlangıç çizgisiyle (*baseline*) karşılaştırılmıştır.1 Tablo IX'daki karşılaştırma verileri, IQL modelinin istatistiksel olarak nerede konumlandığını netleştirmektedir.

### **Tablo IX: Referans Politika Teşhis Karşılaştırması**

| Değerlendirilen Politika Sınıfı | Tahminlenen Değer (WIS) | Etkin Örneklem Büyüklüğü (ESS) | Klinisyen Tam-Kutu Uyumu | Klinik ve İstatistiksel Güvenilirlik Durumu |
| :---- | :---- | :---- | :---- | :---- |
| **Klinisyen Tekrar Oynatımı** | 8.774 | 2585.0 | 1.000 | Tarihsel veri tabanındaki hekim davranışının doğrudan kendisidir.1 |
| **Tedavisiz Kontrol Politikası** | 7.464 | 34.2 | 0.354 | Sıvı ve vazopresör verilmemesini öneren, yüksek mortalite riskli baz kontrol.1 |
| **Davranış Klonlama (BC)** | 9.194 | 1.6 | 0.405 | WIS değeri yapay olarak çok yüksek görünmektedir; ancak ESS 1.6'ya çöktüğü için tamamen geçersizdir.1 |
| **Seçili IQL Politikası** | 8.203 | 29.4 | 0.414 | Destek sınırlarında kalmayı başaran, kabul edilebilir uyuma sahip en kararlı model.1 |

Seçilen kazanan konfigürasyonun nihai test teşhislerinin ayrıntılı dökümü, istatistiksel güven aralıklarıyla birlikte Tablo X üzerinde sunulmuştur.1

### **Tablo X: Kazanan Konfigürasyonun (SOFA\_konservatif\_güvenli) Nihai Test Teşhisleri**

| Metrik Teşhis Kalemi | Ölçülen Değer | Klinik ve Metodolojik Yorumu |
| :---- | :---- | :---- |
| **Uydurulmuş Q-Değerlendirmesi (FQE)** | 2.848 | Modelin durum-değer fonksiyonu (![][image12]) üzerinden tahmin ettiği ortalama başlangıç performansı.1 |
| **Ağırlıklı Önem Örneklemesi (WIS)** | 8.203 | Modelin davranış politikasından saptığı yörüngelerdeki ağırlıklandırılmış performans tahmini.1 |
| **Bootstrap WIS %95 Güven Aralığı** | 4.963 \- 10.817 | 50 hasta düzeyinde bootstrap yeniden örnekleme ile elde edilen geniş ve temkinli aralık.1 |
| **Etkin Örneklem Büyüklüğü (ESS)** | 29.410 | Güvenilirlik sınırı olan 50 eşiğinin altındadır; yüksek varyansı ve klinik üstünlük iddiası kurulamayacağını gösterir.1 |
| **Davranışsal Destek Kütlesi (Support Mass)** | 0.991 | Kararların %99.1'inin tarihsel klinisyen veri desteği içinde yer aldığını kanıtlayan kritik güvenlik göstergesi.1 |
| **Klinisyen Tam-Kutu Uyumu** | 0.414 | Modelin seçtiği aksiyonların hekim kararlarıyla birebir örtüşme oranı.1 |
| **Klinisyen Uyuşmazlık Oranı** | 0.586 | Modelin hekimden farklı bir dozaj veya sıvı miktarı önerdiği kararların oranı.1 |
| **Desteklenmeyen Aksiyon Sapması** | 0.009 | Modelin hekimden saptığı durumlarda dahi veri desteği bulunmayan tehlikeli bölgelere sapma olasılığının binde 9 olduğunu gösterir.1 |

## **Akademik Proje Sunumu Kelime Kelime Konuşma Metni (Script)**

### **Giriş ve Slayt 1: Sepsis Karar Analitiği ve Çevrimdışı RL Zorunluluğu**

Sayın hocalarım, değerli jüri üyeleri. Bugün sizlere, klinik yapay zeka çalışmalarındaki en büyük metodolojik engellerden biri olan veri desteği yetersizliği ve politika dışı değerlendirme riskleri altında kurguladığımız, sepsis resüsitasyonuna yönelik çevrimdışı pekiştirmeli öğrenme iş akışımızı sunacağım.1  
Yoğun bakım ünitelerinde sepsis tedavisi, dinamik hemodinamik parametrelere bağlı olarak intravenöz sıvı ve vazopresör dozajlarının sürekli ayarlanmasını gerektiren, hata payı olmayan, yüksek mortaliteli bir süreci temsil etmektedir.1 Ancak retrospektif klinik verilerle çalışırken karşımıza "Endikasyon Karıştırıcılığı" ya da literatürdeki adıyla *Confounding by Indication* dediğimiz çok büyük bir istatistiksel engel çıkar.1 Klinik pratikte en ağır durumdaki hastalar, kaçınılmaz olarak en erken ve en agresif tedavilere maruz kalırlar.1 Bu durum, ham veride agresif tedaviler ile ölüm oranları arasında yapay bir ilişki varmış gibi görünmesine neden olur ve standart istatistiksel analizleri saptırır.1  
İşte bu yüzden, gerçek klinik süreçlerde çevrimiçi bir deneme-yanılma, yani *online exploration* yapmak tamamen etik dışı ve tehlikeli olduğu için, çevrimdışı pekiştirmeli öğrenmeye yönelmek zorundayız.1 Ancak çevrimdışı RL'de de en büyük risk, modelin geçmiş veri dağılımında hiç temsil edilmemiş ya da çok az temsil edilmiş tehlikeli dozaj kombinasyonlarına asılsız şekilde yüksek değerler atfetmesidir.1 Biz buna "dışdeğer biçimleme hatası" yani *extrapolation error* diyoruz.1 Sunulan bu çalışma, yatak başı bir karar destek sistemi sunmaktan ziyade, bu hataları sistematik olarak engelleyen, denetlenebilir ve sızıntısız bir retrospektif politika değerlendirme metodolojisi kurmayı amaçlamaktadır.1

### **Slayt 2: MDP Tasarımı ve Aşama 0 Ön Tarama Denetimi**

Projenin temelini oluşturan Markov Karar Süreci formülasyonumuz ve veri sızıntısını önleyen katı Aşama 0 denetimimiz bu slaytta özetlenmiştir.1 MIMIC-IV veri seti üzerinde Sepsis-3 kriterlerine uyan 35.239 yoğun bakım yatışını kapsayan kohortumuzu oluşturduktan sonra, hastanın durumunu 62 boyutlu sürekli ve ikili özellikten oluşan bir vektörle modelledik.1 Bu vektörün en kritik bileşenlerinden biri "eksiklik göstergeleridir", yani *missingness flags*.1 Hekimin belirli bir laboratuvar testini isteme sıklığı, aslında onun yatak başındaki klinik şüphe düzeyini gösteren gizli bir sinyaldir; bu göstergeleri durum uzayına ekleyerek hekim niyetini durum uzayına dahil etmiş olduk.1  
Aksiyon uzayımızı, literatürle uyumlu şekilde, intravenöz sıvı ve vazopresör dozajlarının ![][image1] ayrık kombinasyonundan oluşan 25 benzersiz kararla modelledik.1 "Sıfır" tedavisiz durumunu ayrı bir bin olarak koruduk, çünkü tedavi uygulamamak ile küçük doz uygulamak klinik olarak tamamen farklıdır.1  
Metodolojik güvenliğimiz için en kritik adım ise Aşama 0 olarak adlandırdığımız "Ön Tarama Denetimi", yani *presweep audit* sürecidir.1 Veri normalleştirme parametreleri, atama medyanları ve aksiyon çeyreklik eşikleri kesinlikle test verisi dahil edilmeden, yalnızca eğitim verisi üzerinde uydurulmuştur.1 Bu "eğitime uydur, her yerde dönüştür" kuralı sayesinde, klinik yapay zeka çalışmalarında sıkça karşılaşılan ve modelleri yapay olarak başarılı gösteren sonuç sızıntısı, yani *outcome leakage* riskini tamamen ortadan kaldırmış bulunuyoruz.1

### **Slayt 3: Algoritmik Güvenlik Takası: Neden IQL?**

Şimdi, projemizin kalbini oluşturan en önemli algoritmik tercihten bahsetmek istiyorum: Neden Conservative Q-Learning yani CQL değil de Implicit Q-Learning yani IQL algoritmasını seçtik? 1  
Literatürde extrapolation hatasını engellemek için kullanılan CQL gibi yöntemler, veri setinde gözlenmeyen aksiyonların Q-değerlerini açıkça cezalandırır ve kötümser bir alt sınır kurar.1 Kulağa güvenli gelse de, bu yöntemin klinikte çok büyük bir riski vardır.1 Sepsis hastalarında, çok nadir uygulanan ama hayat kurtaran özel kurtarma tedavileri, yani *rescue therapies*, veri setinde çok düşük frekansta bulunurlar.1 CQL'in ceza katsayısı eğer milimetrik olarak kalibre edilmezse, model bu seyrek ama kritik başarılı tedavileri "veri dışı/güvensiz" olarak algılayıp tamamen baskılayabilir.1  
Buna karşın IQL, asimetrik ekspektil kaybı olan ![][image13] ifadesi sayesinde veri kümesi dışındaki hiçbir aksiyonu sorgulamaz ve değerini tahmin etmeye çalışmaz.1 Yalnızca veri desteği içinde kalan, yani klinisyenlerin geçmişte gerçekten uyguladığı aksiyonlar arasından en iyi olanların üst ekspektilini öğrenir.1 Bu tercih, klinik açıdan son derece bilinçli bir takastır.1 Modelimiz, veri kümesinde hiç görülmemiş radikal veya mucizevi tedavi stratejilerini asla keşfedemeyecektir.1 Ancak bunun karşılığında, hastayı kesinlikle veri desteği dışındaki bilinmez ve güvensiz bir dozaj bölgesine sürüklemeyeceğinin garantisini vermektedir.1

### **Slayt 4: Deneysel Sonuçlar ve Kazanan Konfigürasyonun Belirlenmesi**

18 farklı hiperparametre konfigürasyonunu taranıp, en başarılı finalistleri üç farklı rastgele tohumla yeniden değerlendirdikten sonra elde ettiğimiz nihai sonuçlar bu tabloda sunulmuştur.1  
Elde edilen sonuçlara göre kazanan konfigürasyonumuz net bir biçimde iql\_sofa\_shaped\_conservative\_safe olmuştur.1 Bu model, terminal hayatta kalma ödülünün yanı sıra ara adımlarda hastanın SOFA skorundaki değişimleri takip eden bir şekillendirme ödülüyle eğitilmiştir.1 Tablodan da görülebileceği üzere, SOFA şekillendirmeli modelimiz, sadece terminal mortaliteye odaklanan seyrek ödüllü alternatiflerine kıyasla çok daha dengeli bir fizyolojik optimizasyon sağlamıştır.1  
Ayrıca modelimizde "konservatif" bir öğrenme hızı rejimi kullandık.1 Çevrimdışı pekiştirmeli öğrenmede aktör ve eleştirmen güncellemelerini yavaşlatmak, yapay Q-değeri şişmelerini engeller ve farklı rastgele başlatma tohumlarında performansın istikrarlı kalmasını sağlar.1 Üç farklı tohumun ortalamasında ulaştığımız değerler; FQE için 2.848, WIS için ise 8.203'tür.1 Ancak bu sayıların klinik olarak ne anlama geldiğini sorgulamak ve arkasındaki istatistiksel sınırları görmek zorundayız.1

### **Slayt 5: Kritik Görsel Güvenlik ve Politika Dışı Değerlendirme (OPE) Analizleri**

Sadece yüksek bir OPE skoruna bakarak modelin başarılı olduğunu iddia etmek en sık yapılan klinik yapay zeka hatasıdır.1 Bu nedenle geliştirdiğimiz güvenlik ve teşhis grafiklerini incelememiz kritik önem taşımaktadır.1  
Slayttaki "FQE ve Davranış Desteği" grafiğimiz, yüksek değer tahminlerinin yalnızca güçlü bir veri kütlesiyle yani *support mass* ile desteklendiğinde geçerli olduğunu doğrulamaktadır.1 Diğer teşhis grafiğimiz ise daha çarpıcı bir gerçeği ortaya koymaktadır: Davranış Klonlama modelimizin WIS skoru 9.194 gibi çok yüksek bir seviyede görünmektedir.1 Ancak bu modelin Etkin Örneklem Büyüklüğü yani ESS değeri sadece 1.6'dır\! 1 Bir başka deyişle, önem örneklemesi ağırlıkları tek bir hasta yörüngesi üzerine çökmüş durumdadır; bu tahmin istatistiksel olarak tamamen geçersiz ve kararsızdır.1  
Seçtiğimiz IQL modelimiz ise %41.4 oranında klinisyenle tam-kutu uyumu sergilemektedir.1 Yani kararların %58.6'sında hekimden farklı dozlar önermektedir.1 Ancak en kritik güvenlik teşhisimiz şudur: modelin "desteklenmeyen aksiyon sapması" yani *unsupported action rate* sadece 0.009, yani binde dokuzdur.1 Bu ampirik kanıt, modelin klinisyenden saptığı durumlarda bile kesinlikle tehlikeli ve veri desteği olmayan uzak bölgelere kaçmadığını, hekim davranışının hemen komşuluğundaki güvenli ve klinik olarak desteklenen diğer dozaj alternatiflerini seçtiğini göstermektedir.1

### **Slayt 6: İstatistiksel Sınırlar ve ESS Engeli**

Şimdi, bu çalışmanın en dürüst ve en önemli kısmına, yani istatistiksel bariyerimize geliyoruz.1 Sunumu dinleyen siz değerli hocalarımızın en çok önem verdiği bu sınırları net bir şekilde çizmek istiyoruz: Bu çalışmada elde edilen hiçbir ampirik sonuç, modelimizin klinisyenlerden daha üstün bir tedavi uyguladığı şeklinde yorumlanamaz ve bir klinik üstünlük iddiası kurulamaz.1  
Bunun en somut sebebi, modelimizin "Etkin Örneklem Büyüklüğü" yani ESS değerinin 29.410 ile istatistiksel güvenilirlik eşiği olan 50'nin altında kalmış olmasıdır.1 Yoğun bakımda ardışık kararlar uzadıkça, modelin önerdiği politika ile hekimin izlediği politikanın uyuştuğu yörünge sayısı geometrik olarak azalır.1 ESS'in 50'nin altına düşmesi, bootstrap ile hesapladığımız %95 güven aralığının genişlemesine ve varyansın çok yüksek olmasına yol açar.1  
İkinci olarak, sepsis gibi kompleks bir sendromda ortam doğası gereği bir POMDP'dir, yani kısmen gözlemlenebilirdir.1 Biz durum vektörümüze ne kadar değişken eklersek ekleyelim; hekimin yatak başındaki anlık gözlemlerini, hastanın o andaki fiziksel muayene detaylarını veya veritabanına kaydedilmeyen kontrendikasyonları modelimizin tam olarak algılaması mümkün değildir.1 Dolayısıyla, istatistiksel ve metodolojik sınırlarını dürüstçe bildiren muhafazakar bir araştırmacı yaklaşımıyla; bu sonuçları bir klinik üstünlük belgesi olarak değil, gelecekteki klinik yapay zeka çalışmaları için sızıntısız ve güvenli bir değerlendirme protokolü kanıtı olarak sunuyoruz.1

### **Slayt 7: Gelecek Çalışmalar ve Dinamik Güvenlik Katmanları**

Gelecek çalışmalarımızda bu istatistiksel sınırları aşmak adına üç temel yenilikçi doğrultu belirledik.1  
İlk olarak, belirsizliğe göre adapte olan "Dinamik Ekspektil Kalibrasyonu" üzerinde çalışacağız.1 Veri desteğinin yoğun olduğu güvenli bölgelerde ![][image3] parametresini yükseltilerek daha agresif bir iyileştirme yaparken; verinin azaldığı yüksek riskli bölgelerde ![][image6] düzeyine çekilerek muhafazakar davranış klonlama moduna geçilmesini sağlayacağız.1  
İkinci olarak, Surviving Sepsis Campaign klinik kılavuzlarını ve tıbbi kısıtları Lagrangian optimizasyonu yoluyla modelin eğitim sürecine entegre eden "CPQ-IQL" tabanlı kısıtlı pekiştirmeli öğrenme modellerini ve canlı akışta klinik sınır ihlallerini bloke eden "çalışma zamanı koruyucu filtrelerini" yani *runtime safety filters* mimarimize dahil etmeyi hedefliyoruz.1 Literatür, bu tür dinamik kısıt filtrelerinin hayatta kalma oranlarından ödün vermeden klinik kısıt ihlallerini %97 oranında azaltabildiğini göstermektedir.10  
Son olarak, seyrek ödül sinyalini ardışık karar adımlarına daha hızlı yayabilmek adına ampirik olarak Peng'in çok adımlı uygunluk izi yani *eligibility traces* harmanlama yöntemlerini test edeceğiz.1  
Özetlemek gerekirse; bu çalışma, sepsis dozaj politikalarını tek seferlik ve abartılmış bir model başarısı olarak değil; sınırlarını, veri kısıtlarını ve istatistiksel bariyerlerini titizlikle raporlayan, denetlenebilir bir klinik değerlendirme protokolü olarak literatüre sunmaktadır.1 Beni dinlediğiniz için çok teşekkür eder, sorularınızı yanıtlamaktan onur duyarım.1

## **Metodolojik Tuzaklar ve "Neyden Bahsedilmeli / Bahsedilmemeli" Rehberi**

Klinik karar analitiğinde pekiştirmeli öğrenme modelleri sunulurken, jürinin ve ilgili öğretim üyesinin akademik güvenilirlik algısını zedeleyebilecek bazı yaygın hatalardan kaçınmak gerekmektedir.1 Bu amaçla kurgulanan taktiksel rehber aşağıda sunulmuştur.

### **Sunumda Asla Söylenmemesi Gereken Metodolojik Hatalar ve İllüzyonlar**

* **"Modelimiz klinisyenlerden daha yüksek hayatta kalma oranı başarmıştır / klinisyeni yenmiştir" ifadesi kesinlikle kullanılmamalıdır:** Retrospektif bir veri kümesinde ESS değeri 50'nin altında olan ve OPE varyansı bu derece yüksek olan hiçbir model için prospektif bir "klinik üstünlük" iddiası kurulamaz.1 Bu yöndeki bir iddia, jüri tarafından doğrudan "metodolojik cehalet" olarak yorumlanacaktır.1  
* **"Davranış Klonlama (BC) modelimiz en yüksek WIS skoruna sahip olduğu için en iyi modeldir" denilmemelidir:** Davranış klonlama modelinin WIS skoru 9.194 gibi yüksek görünse de, ESS değeri 1.6'dır.1 Yani bu tahmin, önem örneklemesi ağırlıklarının tek bir hasta yörüngesi üzerine çökmesiyle (*weight collapse*) oluşmuş sahte ve istatistiksel olarak geçersiz bir iyimserliktir.1 Bunu bir başarı gibi sunmak en büyük metodolojik tuzaktır.1  
* **"MDP kullanarak gözlemsel verilerdeki karıştırıcı (confounding) problemini tamamen çözdük" denilmemelidir:** Confounding by indication (endikasyon karıştırıcılığı) gözlemsel sağlık verilerinde hiçbir zaman tamamen çözülemez.1 Yoğun bakım ortamı her zaman bir POMDP (Kısmen Gözlemlenebilir MDP) karakteristiği taşır; hekimin tüm yatak başı gözlemleri ve klinik sezgileri veri tabanında kayıtlı değildir.1  
* **"IQL modelimiz yatak başında gerçek zamanlı hasta tedavisi için tamamen hazırdır" ifadesinden kaçınılmalıdır:** Model retrospektif ve gözlemsel bir veri analizi aracıdır; asla prospektif bir klinik onay veya yatak başı karar destek sistemi olarak sunulmamalıdır.1

### **Klinik Güvenliği ve Yetkinliği Vurgulamak İçin Kesinlikle Kullanılması Gereken Anahtar Terimler**

* **Endikasyon Karıştırıcılığı (Confounding by Indication):** Ağır hastaların daha fazla ilaç alması nedeniyle veride ortaya çıkan nedensel saptırmayı açıklamak ve bunu MDP içindeki durum tanımlarıyla (missingness flags, kümülatif ilaç geçmişi) nasıl hafifletmeye çalıştığımızı belirtmek için kesinlikle kullanılmalıdır.1  
* **Dışdeğer Biçimleme Hatası (Extrapolation Error):** Çevrimdışı RL modellerinin, veri havuzunda hiç gözlenmemiş güvensiz aksiyonlara hatalı değerler ataması riskini tanımlamak için kullanılmalıdır.1  
* **Davranışsal Destek Kütlesi (Behavioral Support Mass):** IQL'in veri desteği içinde kalma ve extrapolation hatasını sorgusuz engelleme mekanizmasını açıklamak için bu terim kullanılmalıdır.1  
* **Etkin Örneklem Büyüklüğü (Effective Sample Size \- ESS):** Politika dışı değerlendirmenin istatistiksel güvenilirliğini sorgulamak ve jüriye "verilerimizin ve modelimizin sınırlarını tam olarak biliyoruz" mesajını vermek için en kritik kozdur.1 ESS'in 50'nin altında kalmasını bir "başarısızlık" olarak değil, "retrospektif verinin dürüst istatistiksel sınırı" olarak savunmak gerekir.1

## **Akademik Jüri Tarafından Gelebilecek 3 Ölümcül Soru ve Defansif Cevapları**

### **Soru 1: "ESS (Etkin Örneklem Büyüklüğü) değerinin 29.410 gibi kritik bir eşik olan 50'nin altında kalması, bu çalışmanın OPE (Off-Policy Evaluation) sonuçlarını tamamen geçersiz kılmaz mı? Bu kadar düşük bir ESS ile elde edilen WIS tahminlerine klinik olarak nasıl güvenebiliriz?"**

**Defansif ve Bilimsel Cevap:** "Hocam, son derece haklısınız ve bu çalışmanın en hassas olduğu sınırlılık noktasına parmak bastınız.1 Klinik pekiştirmeli öğrenmede ardışık karar adımları uzadıkça, yeni bir politikanın hekimin geçmişte izlediği yörüngelerle birebir uyuşma ihtimali geometrik olarak azalır; bu durum önem örneklemesi (*importance sampling*) ağırlıklarında kaçınılmaz bir varyans patlamasına ve dolayısıyla ESS değerinin düşmesine yol açar.1  
Bu düşük ESS değeri nedeniyle, çalışmamızda kesinlikle 'modelimiz hekimlerden daha başarılıdır' şeklinde bir klinik üstünlük iddiası kurmuyoruz.1 Ancak bu durum OPE sonuçlarımızı tamamen geçersiz kılmaz, aksine bize tahminlerimizin 'güven sınırlarını' dürüstçe raporlama zorunluluğu getirir.1 Bu varyansı yönetebilmek adına, tek bir WIS noktasıyla yetinmeyip 50 hasta düzeyinde bootstrap yeniden örnekleme yaparak %95 Güven Aralığını (4.963 \- 10.817) açıkça raporladık.1  
Ayrıca, Davranış Klonlama (BC) modelinin ESS değerinin 1.6'ya çökmesine rağmen WIS değerinin yapay bir şekilde 9.194 gibi yüksek çıkmasını da bu analizle ifşa etmiş olduk.1 ESS değerinin 50'nin altında kalması, retrospektif klinik verilerle yapılan sequential OPE çalışmalarının kronik ve yapısal bir gerçeğidir.1 Bu çalışma, bu sınırı saklamak yerine dürüstçe raporlayan ve gelecek çalışmalarda bu varyansı düşürmek için dinamik ekspektil gibi yeni metodolojik güvenlik kilitleri öneren bir şeffaflık standardı getirmektedir.1"

### **Soru 2: "Neden muhafazakar bir yaklaşım olan CQL (Conservative Q-Learning) yerine IQL tercih edildi? CQL'in Q-değerlerini doğrudan cezalandırarak kurduğu kötümser alt sınır klinik açıdan daha güvenli bir bariyer oluşturmaz mıydı?"**

**Defansif ve Bilimsel Cevap:** "Hocam, ilk bakışta CQL'in Q-değerlerini dışarıdan açık bir ceza terimiyle baskılayarak kötümser bir alt sınır (*pessimistic lower bound*) kurması çok daha güvenli görünebilir.1 Ancak klinik açıdan yaklaştığımızda, CQL'in bu açık cezalandırma mekanizmasının çok kritik bir yan etkisi vardır.1  
Yoğun bakımda sepsis hastalarına uygulanan bazı hayat kurtarıcı acil müdahaleler (*rescue therapies*), veri kümesinde çok düşük bir yoğunluğa sahiptir, yani veri desteği oldukça zayıftır.1 CQL'de ceza katsayısını (![][image14]) belirlemek son derece hassas bir kalibrasyon gerektirir.1 Eğer bu katsayıyı en ufak bir şekilde fazla seçerseniz, CQL algoritması bu nadir ama başarılı olan hayati klinisyen müdahalelerini 'veri dışı/güvensiz' olarak algılayıp Q-değerlerini yapay olarak ezebilir ve bu hayat kurtarıcı tedavileri tamamen baskılayabilir.1  
IQL ise Q-değerlerini yapay bir ceza parametresiyle manipüle etmez.1 Asimetrik ekspektil kaybı kullanarak, yalnızca veri desteği içinde kalan aksiyonlar arasından en iyi olanların değerini öğrenir.1 Yani IQL, veri kümesinde hiç görülmemiş radikal bir tedaviyi asla keşfetmeyeceğini baştan kabul eder (ki bu klinik güvenlik için bir avantajdır); ancak bunun karşılığında, klinisyenlerin başarıyla uyguladığı nadir kurtarma tedavilerini asılsız cezalarla yok etmez.1 Bu nedenle, ampirik ceza katsayısı kalibrasyonuna bağımlı kalmadan, veri desteği içinde kalmayı garanti eden IQL modelini klinik olarak çok daha güvenli bulduğumuz için tercih ettik.1"

### **Soru 3: "Gözlemsel elektronik sağlık kayıtlarındaki 'Endikasyon Karıştırıcılığı' (Confounding by Indication) problemini MDP formülasyonunuzda gerçekten çözebildiniz mi? Gözlemlenmeyen karıştırıcılar (unobserved confounders) nedeniyle modelin ağır hastaları cezalandırıp, hafif hastaların kendiliğinden iyileşmesini kendi başarısı olarak görmediğini nasıl garanti ediyorsunuz?"**

**Defansif ve Bilimsel Cevap:** "Hocam, bu soru retrospektif sağlık verileriyle çalışan tüm nedensel çıkarım ve pekiştirmeli öğrenme modellerinin en büyük ontolojik çıkmazıdır.1 Bu probleme net bir cevap vermek gerekirse: Hayır, gözlemsel verilerdeki 'gözlemlenemeyen karıştırıcıları' (*unobserved confounders*) MDP formülasyonuyla yüzde yüz çözmek teorik olarak mümkün değildir; çünkü yoğun bakım ortamı doğası gereği bir POMDP'dir (Kısmen Gözlemlenebilir Markov Karar Süreci).1 Hekimin yatak başındaki anlık gözlemleri, hastanın ailesiyle yaptığı görüşmeler veya kayda geçmeyen klinik sezgileri her zaman gizli birer değişken (*latent variables*) olarak kalır.1  
Ancak bu yapısal karıştırıcılığın etkisini minimize etmek ve modelin hafif hastaları kayırmasını engellemek adına MDP tasarımımızda üç kritik savunma hattı kurduk 1:

1. **Missingness Göstergeleri (Eksiklik Bayrakları):** Modelimize laboratuvar testlerinin değerleriyle birlikte, o testlerin istenme sıklıklarını da durum değişkeni olarak verdik.1 Hekimin test isteme sıklığı, onun yatak başındaki klinik şüphe düzeyinin ve dolayısıyla 'gözlemlenemeyen niyetinin' çok güçlü bir istatistiksel vekilidir.1  
2. **Kümeler Arası Tedavi Yükü (Cumulative Treatment History):** Hastanın geçmiş 4 saatlik pencerelerdeki kümülatif sıvı ve vazopresör maruziyetini durum vektörüne dahil ettik.1 Bu sayede model, hastanın o anki fizyolojisinin arkasındaki 'tedavi yükünü' görerek, daha önce agresif tedaviye maruz kalmış ağır hastalar ile kendi kendine stabil olan hafif hastaları birbirinden ayırt edebilmektedir.1  
3. **SOFA Şekillendirmeli Ödül Fonksiyonu:** Modelimizi yalnızca nihai mortalite (+15 / \-15) ile eğitmedik.1 Eğer öyle yapsaydık, model hafif hastaların kendiliğinden kurtulmasını kolayca sömürebilirdi.1 Bunun yerine, ara adımlarda hastanın organ disfonksiyonu düzeyindeki kötüleşmeleri cezalandıran SOFA değişim ödülünü ekledik.1 Bu sayede model, hastanın hayatta kalıp kalmamasından ziyade, ara adımlardaki fizyolojik kötüleşme veya iyileşme eğilimlerini optimize etmek zorunda kalmıştır.1

Dolayısıyla karıştırıcılık etkisini tamamen sıfırlayamasak da, bu üç metodolojik bariyer sayesinde modelin fizyolojik gidişata sadık kalmasını sağladık.1"

#### **Alıntılanan çalışmalar**

1. iql\_final\_ieee\_report.pdf  
2. A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU \- arXiv, erişim tarihi Mayıs 31, 2026, [https://arxiv.org/pdf/2510.24500](https://arxiv.org/pdf/2510.24500)  
3. Guidelines for reinforcement learning in healthcare \- SciSpace, erişim tarihi Mayıs 31, 2026, [https://scispace.com/pdf/guidelines-for-reinforcement-learning-in-healthcare-4cs15j5tt2.pdf](https://scispace.com/pdf/guidelines-for-reinforcement-learning-in-healthcare-4cs15j5tt2.pdf)  
4. Federated Offline Reinforcement Learning \- PMC \- NIH, erişim tarihi Mayıs 31, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12916147/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12916147/)  
5. Implicit Q-Learning \- Matthew Landers, erişim tarihi Mayıs 31, 2026, [https://mattlanders.net/implicit-q-learning.html](https://mattlanders.net/implicit-q-learning.html)  
6. MIMIC-Sepsis: A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU \- arXiv, erişim tarihi Mayıs 31, 2026, [https://arxiv.org/html/2510.24500v1](https://arxiv.org/html/2510.24500v1)  
7. Implicit Q-Learning for Offline Reinforcement Learning in Blood Glucose Management: A Cross-Dataset Evaluation Study \- Scilight Press, erişim tarihi Mayıs 31, 2026, [https://media.sciltp.com/articles/2602003161/2602003161.pdf](https://media.sciltp.com/articles/2602003161/2602003161.pdf)  
8. Conservative Q-learning Guide 2025 | ShadeCoder, erişim tarihi Mayıs 31, 2026, [https://www.shadecoder.com/topics/conservative-q-learning-a-comprehensive-guide-for-2025](https://www.shadecoder.com/topics/conservative-q-learning-a-comprehensive-guide-for-2025)  
9. AlignIQL: Policy Alignment in Implicit Q-Learning through Constrained Optimization \- arXiv, erişim tarihi Mayıs 31, 2026, [https://arxiv.org/pdf/2405.18187](https://arxiv.org/pdf/2405.18187)  
10. Safe Offline Reinforcement Learning for Sepsis Treatment: A Two-Stage Framework Combining Constraint-Aware Learning with Runtime Safety Filtering \- Scilight Press, erişim tarihi Mayıs 31, 2026, [https://www.sciltp.com/journals/tai/articles/2602003157](https://www.sciltp.com/journals/tai/articles/2602003157)  
11. Conservative Q-Learning: A Robust Offline RL Approach \- Emergent Mind, erişim tarihi Mayıs 31, 2026, [https://www.emergentmind.com/topics/conservative-q-learning-cql](https://www.emergentmind.com/topics/conservative-q-learning-cql)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAACEElEQVR4Xu2WT0gVURSHf5ZFYYbSJluEZhIRERiEoYkoIuiiJJGyTUHbQNCFlEgoblq5EytIE1yLurDQilwpqBAJFoVFm4hWEdHCsN/hzDzunOf84UG2cD744M25Z947b+695w6QkrI7KKfd9DQ9QI/TO/Spm7QDDNAqWkiP0Fa6GMgwNNAt4y/a6Cb9Y/Yiuwax102y1NHv9BN9T5/QCmd8p/hN39HPdBoJHtwlOmqD/4GPNhBHDXIv/Bx0j4RxmNbbYAgfbCCOajpFR+gcXaNdgYxwyqAb6IQdIIfoPHRGk7BBe+hz+pYOQ/94KBfpV2hXEUroN3rfT4jhDF2hpU7sILSAFicWhzSENu/zPvrCMy+TYShA9hMbg26WoyYexnlo8cfofjpD2wMZ8Zw117egneWKiUciPVVu6rADEciSW6KT9GZwKCf8Nv3QDvgs0GVoL/Xpg94kB1NS8qFTu06LzVgcg9DledKJ1UJrkNnLQordpD+g69LnAfSmG04sij10gnZCz4WX0BMwKa+gvyez5tPsxR45sQCyNi+Y2Cz9CT1645DN85jec2JN9BmCDyOKIWTPrnQYKVy+a1suQ6ejyLu+Sv/Q25mMaORH+20Q+q4hbVY6RBzyfrRKT3nX0uGk041nMkK4Rt/QL9AvkOKTIDte1mcY1+ldGwyhkr6G9nM5RWUG3X2XkpKSkhLkL1V1YMeGpxesAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI8AAAAXCAYAAAAyVhy9AAAFQElEQVR4Xu2YB4glRRCGy5xzjneo6J2CIKJndlXMGDArhlNRUcw5K4pZzCcmFM8sZjHngAFFxSyCrhHBiAFz+L+tbl+/3hdnbtE7+oOfnanu7dfTXV1VM2aFQqFQKBQKhUKhUCgUemNd6RPpb+mFrK1Q6Mp85s5zVt5QKHRjJ3PnWS9vKBS6cZ30kzRD3jBSrC3dL90qPSbt0txci22lh83HvlY6V7qqqUc1Rkv3Se9Lu0vHSo9KL9mkO3UzSxdId0h3Bc3Z1OO/ZXrpBOkR6R7pHOlL871M4TlONN/bO6UXpf2TdtqekR6Q5pFOlW4Ktnul2RtdmxkrfSstHO4Z6KNGcy1ONy/g4tg7mIfUM/7tUZ1bpMWlG6S/pIOkVaWfpUuSfnWYKF0YrheRPjZ31P8DOMTT0kPWiDJXmq8vaxFZUHrD/OBOF2ysG/22k1aQJkhLBtt70iqh36zBdnC4H8bh0o/SGtK00r7S+LSD2Nn6D4Obm/8w/xsZF2wbJTZYQlons3ViRunmcP2WNd4sOAjYlwv3wLiMXwXW5WppfnPnuVRaIGnvd94pRLBDcmMfXCz9YR6BI0ear++YcD+19Ly5088UOwUIGGQD9n8la+wXmSKyaLClzrNbcj0U4jm5dPrB/KThRJF5rVoOfdl88TkhkaOk3809OuUi84jXL2wk8z46b0h4XVorN/bIU+bjo9escSIjnea9lLR8bgxw0u+WPs0begRn/k16PLOTttOsER3ilMQG7Cl2UliEZ2GfSYURoiz9iE6RV5PrIfg+cJ70tnnn8UnbVua5sh84VTgkOTSFh3suswEbvGZu7IH4ZrFi3hAgf1dx/AjPcah5imSzBptaO897dRseYVMGrLrzsCc8N4cxQjQmZV+R2NhT+uVOT7rCfkBie8da79e7me2XeHGb9EHSwCITLfYM9+TQD80LLK5TCGm7WiOPprDoTI6aJzKL+cOdGa4pmjc2P4HYrw/3ERaDEEnObgcp5Wvz8JzDppPOBs3HHp20dRub2omwvkdiO868foNO845QBtRxnpWlTXJjYEvz9d0gsZFBsG1j/tv7mRfQ2PIinwKb2oZ9gJieWLMItj+lI8z9ghQHX8QO31jzxyTCHGFvjsRGmBpI7iNU4fzgMXlD4BXp8nBNvr3dvD+bRjjcK7RtIT0ZrlMOM+9PQdgOHDs+VCvOl07OjdZ97APN64Rlwj1plhSW5v52847UcZ6ppO/N54gj55Cucdwdw/1i0pvm/SmGrzGfO+mGMiG+fTIuaXZQWjbYgD3hf9NaESfEtrT5euwd7ESzIbY2XwBeRQlZN0qjYqP5KxqTTOuWCE7znfn/toJikrBHXuWUrmZe5PEqzbeIWFdxOk4L1ymkUk7/53lDgOjxlbRZ3pDAb62fG6372Dw3r6q88jJ/Tuo+TT2Gz3sh86gaRUQi3aU2NjkyYO2dB/htHCh9c0rZUHrWfP0nmm886/2E+WeLCE5O0czhfdB8znMn7UBUzcuJucwzDv9DeyQvvNtCOOaHgbeNHE4kC1QHJshCkP4oBHP4llMF5obj85fap1XdU3Vs6DbvXiLPZ7kxY1Pzt9/JkuPNw9Rs0klZG+BcdR6OMEpY5aQzDt8aUsjVnKoqsHmkTuDUT5O0QZ2xu80benGedpEvQkkxNjdOLowzD588RFoHAZvBV01e++pACqM2IYXmsOmkmCowX15lz7bWr8x1xoZO84ZOzsOhJB38Kl1mrQtj0iDpfoqEhd8+N05C2Hw2fiQYybEjY6z5+0i/UNRTrBYKUw7/APcRI570AOtjAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAAdklEQVR4XmNgGAWDFsgB8UEg/o+G/wHxXyBeBVPICcQXgHgdENcC8XwgPg9l10CxBUxxGRAnwzgMEEXlSHy84AQQe6IL4gJEKxZmgHhICl0CG4gC4tfogrjAEiDeiy6IDTAxQEztR5fABniB+AUQO6JLjAL6AgAt8hcR+4dUVwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAaCAYAAABctMd+AAABSklEQVR4Xu2UPyhGURiHXyEyUJQMdjIYlKJQUlYZv5XBaCFWBsokxCBJRotsBnVRVmIwiWJXMlD+PW/n3Dr37X5XuV9Z7lNP95z3957T/c797hUpKIA6PMTvwC9cCpssw3iDL+IW6PUWj8ImmMVBP97DqiD7lWNxm3fbwNCFa7aYRS2+4qMNUpjBSVvMYkDcXe/YIIVT7LXFLBbEbV6ygUHP+Q7rbZDFhbgn32qDvDThB17aoBKMizuSFRt45rHdjydwS9wvXcXquKkc2qybj9oAGvHcj/UvGmENNuADzvmsLPqA3sUtsGzgtB/34xO2+PkunvlxKh3i7joy9TbcxDdsNlnMCe7boqKv8RU+i9tcrzq/xnv89PWDeIFBj0jXdNogL3p8kbhjqij6Em1jj5+PBVlulnEK+3AI15Px3xmR5LdcXUx0FPwLP1sYQgQl1EKdAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAACWklEQVR4Xu2XO2gUURSGj69CUFBUMBISfHZiYaMSkEhIFLH3UYmNohYiqIVCDCYWYmy0SIiPwkZFUfCBhboqgmChIIitICqoYC2+/p8zm735d2d3ZmeFBO4HH2T+c5Pde+fOmRuzSCQSmRrMgP3wNXwB78NV4YAUOOYNnK+FqchZ88nMSa73ws9w0fiI2myGf+u4rzJ0ctMOf8IdQTbNfBEGg6wWnOQv+A1+gh8Tv8MPcF5laGs5DEfhMi00yX7zu7Za8hJ8J5lyBm6TjAt4F26UvOWshw/hZbhcankZM1+ETsnvwD9wtuQhJ+FCyQ7CU5JV0QGfWvXzww/8Da9XhjaEq/3Iii3GPfPPb5P8RpLn2XEr4TM4UwshXFU2oFvwhPmXZ0fmz8cT142Pzs4m+MSaWwz+Hie7WPJrSb5G8no8hls0VI7APcE1J380uC5Kj/kuuwSXSi2NkrVmEXgj2BD5us3FS8uwck3AV9dbeFoLNUh7HPhYMl8heRq34QMNG7HAvAcs0UJB+KWvmG/NromlmoyYT1Z3Dhsj83qNscxc89fsBS00Yif8qmEB2MD4GJTMH4us8GDEya6VnCfH95KlsdX8b/RL3pCr5p29KLyDF+Fz2Cu1LHAn8sCzK8hmmR+AhoKMHX+7VfcOwl7HRcjV36ab74JzWshBp/mhia8k9oAi8NjMt1T5hHfM/MQY/k9wwHyiN4OszLB57ZAW6sFn6Avs1kIGeCfOm2/7vomlpmFHHzBvpq/MT3zaI9j9f5jfdYW7iDtngxb+F7utdZOPRCKRSCQyOfgH4LB6zzASvDAAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAACLElEQVR4Xu2XTUgVURiGv/xZCBpKBCZiFJG0iKA2QlCmSNSqoIXVStoEuowUKogkd1K7UAMXuikxiNJFUF0FoVQ0KMLACFeFuWgVEf29b99wO/PdmVEJ7jBwHnjgnvecM/fOuXe+c66Ix+PxZINSeB0uwhk4Cfe6AxLohU2wCm6Dp+HL0IiM0A9fwcqgfRF+hNvzI6Lh4v2O8Ko7KAvUw+/wrJNtEV2Em04Wxzf4Dq7AR7At3J0NOkW/vf0mz8G3JovivQ2yyF3RRdhp8ofwF6wwuWXZBhuhAU5J4XPEN/wJ7/8bWhQmRN9/h8nHgny3yS0fYA98At/AO3BraISBq8oC9ABeg8OiFZmvWUwoK20cLFQLotfYqM2cmMBz0ZutNfm9ID9gcstXeCZ4XQ6fBbKuRHIZXnDavPlup50GOfm/RbC1pEN03imTx/ICnrBhkYl7HPhYMt9j8vVoFZ03aDui4MGCNaDOdhSZAdEPvcvkLIzMkwojt9BVCS/UEdF5j50slnPwsw3XgTVhFs5vwqN/Z8bDgxE/9CGT8+S4ZDJLTnTuYSc7GWRDThbLKHxqwxTgL/EHPO9kLHBrsM/JymC7hGvHbXjJaRPuFFyE4yYvoET0V3DLdqQEj83cpaqDNm+EJ8aa/AiRLtGbG3cybvec1xi098FPcCQ/IgH+2eDgY7YjJfgf4AZ8DedEn2dbI1rgF9EdzuUgnBY9L/D0eEX0eh6Px+PxeDyb4g/3bIJSkl+NCwAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAACq0lEQVR4Xu2XWahOYRSGlzEzGRKFTKGEyI2SUnIhKRQXuECKZCopJMotJZELIpSxKDLfGBJFUW5EHJIxERHJ8L7Wv89Z+23/39kdzt3/1HNx3vXtvb9/f9M+ZjVqNCsd4XnYRwslWAeXaNjcnIULNAQL4Qd42vxHFdEKXoOTtdBcLIJXNAwsh7/hUi0EhsO3sIsWGqMn3ACvwkvwOrwDN8N2Dc3qaQufw+laCLAT38zvmeIiXK9himXwvXnnOod8MHwI78FuIScz4SvYUnLlFPwJ+2ohMBu+M58iSVrAPfA7nCq1jCnmw7pD8sPwpGRFzDW/fqUWAr3N24zXgrLVvOEaLQQ49GzzTPJHcK1kRXDBfYE3tSDw/sn7jTMfqjrYJl/KwVH4Ab+GrBP8BWeELMVR8/YDtBC4DA9oGDlmjQ8V4Spmu8chG1jJJoUsxU7z9qk3yOl1RsMMvrWP5jcZITWFWxfb7Q7ZmEo2OmTVYCdPmC/qu1KL7LXEtGlv/kB2mp1PccN8axoaMna0TIdXw/vm85gd4jVDci0aYP2WhpFP8LWGwkTzh6yQvF8lT00JTjVuVZw+JNttNta3yMMpwf24KofMF113LVToCp/ALVoAHcwfXm3R8YTjQo1HLvfYN/BByCJcdOxTVbiR18FtcBDcB4/D7XAYvABnZY0LeGrFi4jfENwRdFTILvMfOlIL5ttaanv9Sw/zDnPu3DbffjhneePsOO5lxdvRfis+ODisvL6IUfCz+a4RyQ6OCZKXgm8mnmr8u2jazIEvrfGjuQw8mpt8r1Xmw9MfToMH8+V6WptPi9THT1k4Kps0LEt29lMunLH5co7F5ovlX+DBxLerH1el4YX8OnsB50mtiCNwvoYl4c7BNfM/Rqk0PITOWdP/RUp93NeoUcQfAnmLZtw/pz4AAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAWCAYAAACbiSE3AAABeUlEQVR4Xu2WuyuHURjHH9dyvwwSi9tGSRYlJUJRYqBs/AMyGciijEoZGX7lUjZJilzKICSZMDBYbMoihfA5nTfO7+T3ukxO7/nUZzjf5/Sr53nfnt8r4vF4PH+jGw/wBM9xAlPibkSEVjzDwuDcjm8493EjQsRENz8UnJPwAV/kc0CRYUr0MPqM7BFfMcvIIoF6E4qMc43o4WwbWSJqsdIODXKxxQ5dIQ838BarrNpXlOMRVtgFyMYdbLILLrCIN3iPDVYtjGo8xTIjy8At7DIyJ+nEZxy2CyHUix5ICabjOvbH3XCYfdEL9DdvSCMe4yoOxpfcQT3VOiuLiV6i01YeRiru4iUWWDUnKBX9PfGExUa+JHoYs0YWRjIu4wg24x7mmBdcQA1ANX2HmUZ+GOS9RpYI9dc8j+NG1oGbohepU6zgAuYH5zbR+2JNdKPfMYOTdih6kOo30uzCf0Ztf9XMFV7jBY7Kz5roEf0Fm4gBHLNDj8fjcZ13Wbc9BvHIdb4AAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAWCAYAAACbiSE3AAABF0lEQVR4Xu2Wr0tDYRSGj+IEhwpWizqjwWBTBLEsWLQIa1abeWuC3T/AbBZxTAUV4wxmg3axWBUG+hyuyLdT7u4sHu554Cnv+3HDy/0lEgRB8HfmbFA2qriK53hhulKxj2/Yxp6UfIyUD4kxfik6xjIu2jBhGjdt6IWiYyxgF2u2gEm8wXVbeKHoGMoSPuJ8kk3gNW4lmTuGGUNZkWyQWRyX7Bq7fSccomPoV2UY1vABz3Cvv/KJjtGx4YCM4S0+4YzpXKJjXNpwAEbxFA9wA+9wKj3gER3jyoY5jOAJtpKsLtl19EXqEr3NP/HeFjkc46ENYUey3/uKLf4z+vl7wXf8+vEVnyX/2d/GIxsmNLBpwyAIAu98A4qOLxAfQVDMAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAA9klEQVR4Xt3SMUhCURSH8VMOgUODNDo1OApNgYLQ2CLNgrZWS0FTu7gJgpOTSzppQugiFLgIDkI1NxTNDToqYt/t+h7HA29q6w+/4X1c4b2LIv9neyhjhCnusb91YrMYHpBXrYO6eg53g4JpFUxMkzgGNrInNG08R9G0LGZImS5tJPGCBd7xjYw+FOwRO+Lf+RZneMW1PuR2iKqN4n/0ZWNJtq8r2AXW4j8+XAMJHTarYSn+/sMN9YPaG551OMAcuzqyU6xwoqP76g9cqpbGJ65U+527hSO0MBb/B+rhWB8K1rchau5KujZGzb38nY1RcwdzNv55PwZNJ91rJaTuAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAWCAYAAAAfD8YZAAAAvUlEQVR4XmNgGAV0AYxAzATjnAXir0D8H4j/APEDILaDSULBBgaIPAi/A+JUZMlgqMRCZEE0cBKII9EFQcCAAaJ5L7oEFIQAcSe6IAzwM0A030GXAAJBIN4JxGzoEsgA5JefDEiBAQXTGSAuwwtWM0Bst4TyuYB4CRAzw1XgASA/gTRHQPlNQKyCkMYP0hkgmsuB2AyIU1Cl8QNXBojm+UDcjSZHECgxQDQ/B2IJNDmCABQwoJTmjy4xCqgAAOXvIY7SS93DAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,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>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAbCAYAAACnZAX6AAAAxElEQVR4XmNgGAUjBngD8X4oPgvEjUDMhCTPiMQGgxogfgrEmlC+CBC/AuJaKL8MiD2hbDAIBeL/DBCbkMFsIH4NxMxAvBeI2WASIIEHQPwQJoAEQLaADIsA4snIEmRpMoNKzEQWhIICBojcVSAWQ5YIh0okIwtCQQ4DRC4PXcIKKhGALgEEdQwQOZBrUAAoHs4D8SwkMXEGiB9WMyAMDAFiBSQ1DLJAvAGIjwDxDiBeDsTWULlJQHwLiBcxYIncUTB4AQBczSfTPPQZ5QAAAABJRU5ErkJggg==>