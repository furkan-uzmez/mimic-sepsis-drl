# MIMIC-IV Sepsis Çevrimdışı Pekiştirmeli Öğrenme (Offline RL) Projesi: Tam Kapsamlı Rapor

Bu doküman, MIMIC-IV veri seti üzerinde sepsis tedavisi (IV sıvı ve vazopressör kararları) için geliştirilen sıfır sızıntılı (zero-leakage), yeniden üretilebilir Çevrimdışı Pekiştirmeli Öğrenme (Offline RL) arayüzü ve deney sürecinin baştan sona tüm detaylarını adım adım özetlemektedir.

## 1. Projenin Amacı ve Temel Çerçevesi
Bu projenin amacı, doktorların canlı hastalar üzerinde deneme yapmasının (online RL) etik ve pratik olarak imkansız olduğu yoğun bakım (ICU) ortamı için, geçmiş tıbbi kayıtları (MIMIC-IV) kullanan güvenilir bir klinik karar destek değerlendirme altyapısı (benchmark) kurmaktır. 

*   **Model:** Model-free, Off-policy, Offline RL (CQL, BCQ, IQL).
*   **Donanım:** Apple Silicon (Metal/MPS) ve NVIDIA (CUDA) üzerinde aynı kod tabanıyla donanım bağımsız (device-agnostic) çalışma.
*   **Klinik Sınır:** Proje yatak başı (bedside) canlı bir tedavi aracı değil, geriye dönük (retrospektif) offline politika değerlendirme (OPE) sistemidir.

---

## 2. Adım Adım Yapılanlar (10 Aşamalı Mimari)

Proje, katı bir bağımlılık zinciri içinde 10 ana faza (Phase) bölünmüş ve tamamlanmıştır:

### Adım 1: Kohort (Hasta Grubu) Tanımı ve Seçimi
*   MIMIC-IV veritabanından **Sepsis-3** kriterlerine uyan hastalar çekilmiştir. Dâhil edilme (inclusion) ve dışlanma (exclusion) kuralları, hem tıbbi geçerliliği sağlamak hem de yapay zekanın kopya çekmesini (data leakage) engellemek için şeffafça kodlanmıştır:
    *   **Dâhil Edilenler (Inclusion):** Sadece 18 yaş ve üzeri yetişkinler (çocukların ilaç dozajları farklı olduğu için modelin kafası karışmasın diye), kesintisiz 4 saatlik yoğun veri takibi yapılabilmesi için sadece doğrudan Yoğun Bakım (ICU) hastaları ve medikal olarak "Sepsis-3" tanısı konmuş hastalar seçilmiştir.
    *   **Dışlananlar (Exclusion):** Sepsise yakalandığı kesin saat (onset time) veri tabanında bulunamayanlar, 4 saatten kısa süre yoğun bakımda kalıp modelin öğrenebileceği bir süreç sunmayanlar ve yaşı/cinsiyeti kaydedilmemiş hastalar veriden atılmıştır. **En kritik dışlanma kuralı ise "Tekrar Yatan Hastalar (Readmissions)"dır;** eğer hasta aylar sonra tekrar yoğun bakıma yattıysa sadece ilk yatışı alınmış, ikincisi silinmiştir. Çünkü aynı hastanın bir yatışı Eğitim (Train) setine diğer yatışı Test setine düşerse model hastanın anatomisini ezberleyerek hile yapabilir.
*   Bu filtrelere takılıp dışlanan hastalar için arkada sayılarla belgelenmiş şeffaf bir `audit.json` raporu üretilmiştir.

### Adım 2: Zaman Çizelgesi ve Sepsis Başlangıç (Onset) Tespiti
*   Her bir ICU kalışı için kesin bir `sepsis_onset_time` (sepsis başlangıç zamanı) hesaplanmıştır.
*   Hastanın yoğun bakımdaki süreci, sepsis teşhisinden **24 saat öncesi ile 48 saat sonrası** arasındaki 72 saatlik bir pencereye hapsedilmiş ve **4'er saatlik karar adımlarına (timestep)** bölünerek bir Markov Karar Sürecine (MDP) dönüştürülmüştür. 
*   **"Ortalama bir hasta 18 karar adımından oluşmaktadır"** ifadesi buradan gelir: 72 saatlik toplam gözlem süresi 4 saatlik pencerelere bölündüğünde (72 / 4) toplam 18 adet zaman/karar adımı (timestep) elde edilir. Her adımda hastanın o anki verileri ölçülür ve model bir sonraki adım için sıvı/ilaç dozunu seçer.

### Adım 3: Veri Sızıntısı Koruması (Zero-Leakage Boundaries)
*   Hastalar rastgele `Train (%70)`, `Validation (%15)` ve `Test (%15)` setlerine ayrılmıştır.
*   Aynı hastanın verisinin farklı setlere düşmesi engellenmiştir (patient-level split).
*   Bu adım, imputation (eksik veri doldurma) ve ölçeklendirme (scaling) işlemlerinde test setinin istatistiklerinin eğitime sızmasını engellemek için projenin en kritik güvenlik bariyeridir.

### Adım 4: Durum (State) Vektörü Çıkarımı
*   Her 4 saatlik adım için hastanın o anki durumunu özetleyen **62 boyutlu sürekli (continuous) bir durum vektörü** oluşturulmuştur.
*   **Agregasyon (Özetleme):** 4 saatlik süre boyunca ölçülen onlarca veri (örn. tansiyon); klinik anlama göre *son ölçüm (last), ortalama (mean), minimum (min), maksimum (max)* veya *toplam (sum)* kurallarıyla tek bir sayıya indirgenmiştir.
*   **Özellikler:** Kalp hızı, tansiyon (MAP vb.) gibi vital bulgular; kan gazı, laktat gibi laboratuvar değerleri; demografik bilgiler ve kümülatif tedavi geçmişi.
*   **Eksik Veri (Missingness):** Eksik veriler önce hasta bazında ileri doğru (forward-fill) doldurulmuş, yetmezse *sadece Train setinden* elde edilen medyan değerler (fallback) kullanılmıştır.

### Adım 5: Aksiyon (Action) ve Ödül (Reward) Mühendisliği
*   **Aksiyonlar:** Vazopressör dozu ve İntravenöz (IV) sıvı hacmi **5x5'lik bir matrise** oturtulmuştur. Model toplamda **25 farklı ayrık aksiyondan** (0: tedavi yok ... 24: maksimum doz) birini seçmektedir.
    *   **Sıfır Doz Hassasiyeti:** "Hiç ilaç/sıvı vermeme" durumu (Doz 0), düşük dozlarla karıştırılmaması için kasıtlı olarak kendine ait bir sınıfa (Bin 0) konulmuştur. Geri kalan dozlar ise yine sadece Train setinde hesaplanan çeyreklik dilimler (Q25, Q50, Q75) ile 4 eşit sınıfa ayrılmıştır.
*   **Ödüller:** Modeller iki farklı ödül fonksiyonu ile eğitilmektedir:
    *   *Sparse:* Yalnızca hasta taburcu olduğunda yaşadıysa +15, öldüyse -15 (terminal utility).
    *   *SOFA-Shaped:* Terminal ödüle ek olarak, organ yetmezlik skoru (SOFA) arttıkça negatif, düştükçe pozitif ara ödüller (shaping) eklenen versiyon.
*   Gelecek ödüllerin bugüne etkisi (Discount factor, $\gamma$) klinik stabilite için **0.99** olarak sabitlenmiştir.

### Adım 6: Geçiş Veri Seti (Transition Dataset) ve Temel Çizgiler (Baselines)
*   Tüm veriler, RL modellerinin yutabileceği `(s_t, a_t, r_t, s_t+1, done)` formatında Parquet dosyalarına dönüştürülmüştür.
*   Karşılaştırma için; gerçek klinisyen davranışı, "hiç tedavi vermeme (no-treatment)" ve "sadece doktoru taklit etme (Behavior Cloning)" baselineları oluşturulmuştur.

### Adım 7 & 8: IQL (Implicit Q-Learning) Modelinin Eğitimi
*   Offline RL modelleri, ellerindeki veri kümesi dışında yeni eylemler denemezler (Böylece gerçek hayatta hiç görmedikleri dozları deneyip "Extrapolation error" yaratma riskine karşı korunurlar).
*   Projede özellikle **IQL (Implicit Q-Learning)** algoritmasına odaklanılmıştır. IQL'in temel mantığı, veride olmayan aksiyonları (örneğin aşırı yüksek bir doz) hayal edip değer biçmek yerine, doğrudan eldeki doktor davranışları içindeki en faydalı (avantajlı) kararları öğrenmesidir.
*   Model altyapısı cihaz-bağımsız (device-agnostic) tasarlanmıştır; yani eğitim kodları değiştirilmeden hem Apple Silicon (MPS / Mac) hem de NVIDIA (CUDA / GPU) üzerinde sorunsuz şekilde çalışabilmektedir.

### Adım 9: Değerlendirme, Güvenlik ve Raporlama
*   Modellerin doğruluğu **Offline Policy Evaluation (OPE)** metrikleriyle (FQE, WIS, ESS) değerlendirilmiştir. OPE sonuçları sadece tek bir sayı olarak verilmemiş, tıbbi istatistik kuralları gereği **Hasta Bazlı Bootstrap %95 Güven Aralıkları (CI)** hesaplanarak raporlanmıştır.
*   Modelin doktordan çok uzaklaşıp saçma sapan dozlar önermesini tespit etmek için *Support Mass* ve *Low-Support Rate* gibi güvenlik teşhis araçları (diagnostics) sisteme gömülmüştür. Ayrıca yüksek riskli alt hasta grupları da izlenmiştir.

### Adım 10: Nihai Çoklu-Tohum (Multi-Seed) Tarama ve IEEE Raporu (Final Sweep)
*   **IQL Protokolü:** 2 ödül varyantı × 3 Learning Rate rejimi × 3 IQL hiperparametresi olmak üzere toplam **18 farklı eğitim ayarı** test edilmiştir.
*   **Finalist Seçimi (Güvenlik Odaklı):** Modeller seçilirken test setine **asla bakılmamış**, seçimler Validation setinde "çok kriterli bir puanlama" ile yapılıp 6 finalist seçilmiştir. Bu seçimde sırf skoru yüksek diye güvensiz modellere izin verilmemiştir; *en dengeli, doktora en çok uyan (clinician agreement) ve en iyi desteklenen (support)* modeller listeye dâhil edilerek çeşitlilik korunmuştur.
*   **Tohum (Seed) Stratejisi ile Sağlamlık Testi:** Seçilen bu 6 modelin başarısının "şans eseri" olmadığını kanıtlamak için, modeller Stage 2'de farklı rastgelelik çekirdekleriyle (Seed 123, Seed 456 vb.) baştan eğitilip, sonuçların kararlılığı (varyans) hesaplanmıştır.
*   **IEEE Raporunun (iql_final_ieee_report.pdf) Çıktısı:** Rapor, sızıntı denetimlerinin %100 başarıyla geçildiğini, model seçiminin çeşitlilik korunarak yapıldığını kanıtlamaktadır. Ancak raporda metodolojik dürüstlük gereği, IQL için nihai politikanın test setindeki FQE ve WIS metriklerinin entegrasyon eksiği nedeniyle *henüz sayısal olarak hesaplanamadığı (H.D. - Hesaplanamayan Değer)* açıkça beyan edilmiştir. 

---

## 3. Teknoloji Yığını ve Altyapı
*   **Python Yönetimi:** `uv`
*   **Veri Transformasyonu:** Yüksek hızlı `Polars` ve `PyArrow` (Parquet).
*   **Derin Öğrenme / RL:** `PyTorch` ve özelleştirilmiş d3rlpy.
*   **Deney Takibi:** `Hydra` (Config) ve `MLflow`.
*   **Raporlama:** Bootstrap CI tabloları, aksiyon ısı haritaları (action heatmaps) ve IEEE LaTeX şablonu.

## 4. Sonuç ve Sınırlar
Bu sistem, verilerin doğru hazırlanmasından model seçimine kadar "end-to-end" (uçtan uca) çalışan devasa bir Sepsis yapay zeka deney laboratuvarıdır. Ancak modellerin OPE sonuçları yalnızca metodolojik iddialardır, prospektif bir klinik kanıtı (canlı hastada çalışır iddiası) sunmaz. Sepsis süreci doğası gereği tam olarak gözlemlenemeyen (POMDP) bir süreçtir ve gizli klinik faktörler her zaman modelin dışında kalır.
