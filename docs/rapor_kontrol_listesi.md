# BLM446 Final Rapor Kontrol Listesi

> Hoca geri bildirimi: `docs/degerlendirme.md` (Doç. Dr. A. Burak İNNER, 14 Mayıs 2026)
> Proje önerisi: `proje_onerisi_cql/230202066_proje_onerisi_cql.pdf`
> Konu: **Onaylandı** ✅

---

## Zorunlu Maddeler (Final Teslim Öncesi)

### 1. γ (discount factor) gerekçesi

Öneri metninde γ belirtilmemiş. Raporda şu gerekçeyle açıklanmalı:

> Sepsis episode'ları tipik olarak 18 adım (~72 saat, 4 saatlik pencereler). γ = 0.99 seçildiğinde
> 0.99^18 ≈ 0.83 olur, yani terminal ödül (90 günlük sağkalım) son adımda yaklaşık %83 ağırlıkla
> karara etki eder. Bu değer, uzun vadeli sonucu önemserken ara adımlardaki SOFA tabanlı
> shaping reward'ın da öğrenmeye katkı vermesine izin verir.

**Konum:** Metod bölümü, MDP tanımı / hiperparametreler alt başlığı.

---

### 2. Gözlemlenebilirlik / POMDP tartışması

Klinik kayıtlarda eksik veri ve ölçülemeyen gizli değişkenler MDP'yi kısmi gözlemlenebilir (POMDP) hale getirebilir. Raporda şu noktalara değinilmeli:

- Ölçülemeyen enflamasyon belirteçleri (örn. sitokin seviyeleri) state vektöründe yok
- Eksik lab değerleri forward-fill + median imputasyon ile dolduruluyor, bu bir yaklaşıklıktır
- Hastanın gerçek fizyolojik durumu ile state vektörü arasında bilgi kaybı olabilir
- Bu durum OPE tahminlerini etkileyebilir; sonuçlar bu farkındalıkla yorumlanmalıdır

**Konum:** Limitations / Tartışma bölümü.

---

### 3. Veri erişim süreci ve etik bölümü

MIMIC-IV veri tabanına erişim süreci raporda belgelenmelidir:

- **Veri kaynağı:** MIMIC-IV v3.1, PhysioNet (physionet.org)
- **Eğitim:** CITI Program "Data or Specimens Only Research" sertifikası
- **Sözleşme:** PhysioNet Veri Kullanım Sözleşmesi (DUA) imzalanmıştır
- **Etik:** MIMIC-IV HIPAA uyumlu, kimliksizleştirilmiş (de-identified) veridir; hasta mahremiyeti korunur
- **Kurum onayı:** Retrospektif çalışma, kurum etik kurulu gerektirmeyebilir (PhysioNet DUA kapsamında)

**Konum:** Kendi başlığı altında "Etik ve Veri Erişimi" bölümü.

---

### 4. Referans listesi (en az 10, IEEE formatı)

Hocanın belirttiği öncelikli 5 kaynak + 5 ek kaynak:

| # | Kaynak | IEEE Formatı |
|---|---|---|---|
| 1 | Kumar et al. 2020 (CQL) | A. Kumar, A. Zhou, G. Tucker, and S. Levine, "Conservative Q-Learning for Offline Reinforcement Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. |
| 2 | Fujimoto et al. 2019 (BCQ) | S. Fujimoto, D. Meger, and D. Precup, "Off-Policy Deep Reinforcement Learning without Exploration," in *International Conference on Machine Learning (ICML)*, 2019. |
| 3 | Kostrikov et al. 2022 (IQL) | I. Kostrikov, A. Nair, and S. Levine, "Offline Reinforcement Learning with Implicit Q-Learning," in *International Conference on Learning Representations (ICLR)*, 2022. |
| 4 | Komorowski et al. 2018 (AI Clinician) | M. Komorowski, L. A. Celi, O. Badawi, A. C. Gordon, and A. A. Faisal, "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care," *Nature Medicine*, vol. 24, pp. 1716-1720, 2018. |
| 5 | Levine et al. 2020 (Offline RL) | S. Levine, A. Kumar, G. Tucker, and J. Fu, "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems," *arXiv preprint arXiv:2005.01643*, 2020. |
| 6 | Tang & Wiens 2021 (Model Selection) | S. Tang and J. Wiens, "Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings," *arXiv preprint arXiv:2107.11003*, 2021. |
| 7 | Tu et al. 2025 (Sepsis Safe RL) | B. Tu et al., "Offline Safe Reinforcement Learning for Sepsis Treatment: Tackling Variable-Length Episodes with Sparse Rewards," *Human-Centric Intelligent Systems*, Springer, 2025. |
| 8 | McCoubrey et al. 2026 (Temporal) | J. McCoubrey et al., "Off by a beat: the effects of temporal misalignment in reinforcement learning for sepsis treatment," *npj Digital Medicine*, Nature, 2026. |
| 9 | Johnson et al. 2023 (MIMIC-IV) | A. Johnson, L. Bulgarelli, T. Pollard, S. Horng, L. A. Celi, and R. Mark, "MIMIC-IV (version 3.1)," *PhysioNet*, 2023. |
| 10 | Vincent et al. 1996 (SOFA) | J.-L. Vincent et al., "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure," *Intensive Care Medicine*, vol. 22, pp. 707-710, 1996. |
| 11 | Nambiar et al. 2023 | A. Nambiar et al., "Deep offline reinforcement learning for real-world treatment optimization applications," *arXiv preprint arXiv:2302.07549*, 2023. |
| 12 | Gottesman et al. 2019 | O. Gottesman et al., "Guidelines for reinforcement learning in healthcare," *Nature Medicine*, vol. 25, pp. 16-18, 2019. |

**Konum:** Referanslar bölümü.

---

## Opsiyonel Maddeler (Raporu Güçlendirir)

### 5. Sayısal eşikli hipotez

Giriş veya metod bölümüne eklenebilir:

> **Hipotez:** CQL, klinisyen baseline politikasına göre WIS normalize skorunda
> en az 0.1 mutlak iyileşme sağlar.

Bu, "CQL daha iyi" demek yerine ölçülebilir, test edilebilir bir iddia ortaya koyar.

---

### 6. Plan B / risk yönetimi

Metod bölümüne eklenebilir:

> CQL α=5 ile 50.000 eğitim adımı sonunda klinisyen baseline'a göre anlamlı
> bir iyileşme göstermezse, IQL alternatif algoritma olarak değerlendirilecektir.

Bu, projenin tek bir algoritmaya körü körüne bağlı olmadığını, başarısızlık durumunda
bir yedek plan olduğunu gösterir.

---

## Hocanın Genel Yorumu

> "Önerinizin yöntem seçimi, MDP formülasyonu ve karşılaştırma planı projeyi yürütmek için
> yeterli temeli sunmaktadır. Konunuz onaylanmıştır; uygulamaya geçebilirsiniz."

**Yani:** Kod/pipeline/implementasyon tarafında hiçbir eksik yok. Yukarıdaki maddelerin tamamı
rapor yazımıyla ilgili. Phase 10 planı (10-01-PLAN.md) bu raporu üretmek için tüm teknik
altyapıyı sağlayacak.

---

## Metodolojik Notlar (Raporda Dikkat Edilecek)

> Aşağıdaki atıfları raporun metod bölümünde kullanabilirsin.

### Two-Stage Model Selection (Tang & Wiens 2021)

Tang & Wiens (2021), "Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings" [1] çalışmasında, healthcare offline RL'de validation performansını doğrudan online ölçemediğimiz için OPE/FQE tabanlı seçim yapılması gerektiğini inceler. FQE'nin aday policy ranking'de en güçlü yöntem olduğunu, ancak compute maliyetinin yüksek olduğunu; bu nedenle iki-aşamalı (WIS pruning → FQE ranking) yaklaşımın en iyi accuracy/compute dengesini sağladığını gösterir.

**Raporda kullanılacak:** Çalışmamızda Tang & Wiens (2021)'in iki-aşamalı model seçim protokolünü uyguladık. Stage 1'de tüm (reward, lr, alpha) kombinasyonları tek seed ile taranmış, Stage 2'de validation FQE'si en yüksek 6 konfigürasyon 5 seed ile doğrulanmıştır. Tüm model seçimi validation split üzerinde, held-out test set kullanılmadan yapılmıştır.

**Atıf:** [1] Tang, S., & Wiens, J. (2021). Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings. *arXiv preprint arXiv:2107.11003*.

**Rapor bölümü:** Deneysel Kurulum / Model Seçimi.

---

### CQL ve Alpha Seçimi (Kumar et al. 2020)

Kumar et al. (2020), "Conservative Q-Learning for Offline Reinforcement Learning" [2] çalışmasında CQL'nin temel amacını şöyle tanımlar: offline data dışındaki (OOD) aksiyonlar için aşırı iyimser Q değer tahminlerini bastırmak ve conservative Q-function öğrenerek policy value için daha güvenli (lower-bound yönlü) tahmin üretmek. CQL'nin α (conservative penalty) parametresi bu bastırmanın şiddetini kontrol eder.

2025 sepsis CQL çalışması (Tu et al., "Offline Safe RL for Sepsis Treatment") [3] ise CQL'nin nadir görülen klinik tedavilerin beklenen reward'unu düşük tahmin ederek daha güvenli öneriler üretmeye çalıştığını ve α için {0.05, 0.1, 0.5, 1, 2} grid'ini tarayıp en iyi sonucu α = 0.05 ile aldığını raporlar. Bu bulgular doğrultusunda çalışmamızda α ∈ {0.05, 0.1, 0.5, 1.0} grid'i taranmıştır.

**Raporda kullanılacak:** CQL'nin conservative penalty katsayısı α, offline dağılım dışı aksiyonların değerini bastırarak daha güvenli politika öğrenmeyi sağlar (Kumar et al., 2020). Sepsis tedavisi gibi yüksek riskli klinik alanlarda, Tu et al. (2025)'in de gösterdiği gibi, küçük α değerleri (≤ 0.1) nadir tedavileri daha temkinli değerlendirerek klinik olarak daha güvenilir öneriler üretebilir.

**Atıf:**
- [2] Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS 2020*.
- [3] Tu, B., et al. (2025). Offline Safe Reinforcement Learning for Sepsis Treatment: Tackling Variable-Length Episodes with Sparse Rewards. *Human-Centric Intelligent Systems*, Springer.

**Rapor bölümü:** Metod / CQL Algoritması ve Hiperparametre Seçimi.

---

### Temporal Alignment (McCoubrey et al. 2026, Nature)

McCoubrey et al. (2026), "Off by a beat: the effects of temporal misalignment in reinforcement learning for sepsis treatment" [4] çalışması, *npj Digital Medicine* (Nature) dergisinde yayımlanmıştır ve sepsis RL'de temporal alignment hatasının hâlâ az fark edilen ama son derece ciddi bir metodolojik failure mode olduğunu gösterir. State-action-next_state indekslemesinde 1-adım kayma bile FQE/WIS metrikleri iyi görünse dahi politikanın hatalı öğrenilmesine ve sonuçların metodolojik olarak geçersiz olmasına yol açabilir.

**Raporda kullanılacak:** Temporal alignment, sepsis RL'de en kritik veri sızıntısı (leakage) riskidir. McCoubrey et al. (2026)'in gösterdiği gibi, state-action-next_state indekslemesindeki küçük kaymalar OPE metriklerini tamamen geçersiz kılabilir. Pipeline'ımız, her transition'da state vektörünün aksiyon penceresinden önceki son gözlemlerden oluşturulduğunu, next_state'in aksiyon penceresi sonrası gözlemleri içerdiğini ve 90 günlük mortalite bilgisinin state'e sızmadığını garanti edecek şekilde denetlenmiştir.

**Atıf:** [4] McCoubrey, J., et al. (2026). Off by a beat: the effects of temporal misalignment in reinforcement learning for sepsis treatment. *npj Digital Medicine*, Nature.

**Rapor bölümü:** Metod / Veri Pipeline ve Leakage Kontrolleri.

---

### Stage 1 / Stage 2 Ayrımı

Raporda sonuçlar sunulurken:

- **Stage 1 (24 config, single seed):** "Exploratory screening" olarak sun. "Confirmed" deme.
- **Stage 2 (6 config, 5 seed):** "Multi-seed confirmation" olarak sun. Asıl sonuçlar bunlar.
- **Final test:** Sadece seçilen tek final policy + baselines. Bir kerelik, held-out test set üzerinde.

Tabloları ayır: Table 1A (Stage 1 screening), Table 1B (Stage 2 confirmation), Table 1C (Final test).

---

### Common Evaluation Reward

Shaped ve sparse reward ile eğitilen modeller **kendi training reward'ları ile FQE yapılırsa kıyaslanamaz.** Farklı reward ölçekleri ve anlamları var. Değerlendirme her zaman **ortak terminal survival utility** (±15 survived/died) üzerinden yapılmalı. Shaped/sparse sadece training içindir. Raporda bunu metod bölümünde açıkça belirt.

---

### Best Checkpoint Selection

Final checkpoint epoch 200 otomatik seçilmez. Her run'da epoch 100, 140, 160, 180, 200 checkpoint'leri arasından **validation FQE'si en yüksek olan** seçilir. Pilot run'da en iyi epoch 140'tı.

---

### FQE Hyperparametreleri

Raporda FQE için şunları belirt:
- FQE network mimarisi (hidden layers, activation)
- FQE learning rate ve epoch sayısı
- FQE discount (γ=0.99 veya γ=1 evaluation)
- Kaç FQE initialization kullanıldığı (ensemble?)
- FQE'nin hangi split üzerinde hesaplandığı (validation)

---

### Data Leakage Kontrolleri

Raporda şu kontrollerin yapıldığını belirt:
- Patient-level split: hiçbir hasta train/val/test'te tekrar etmez
- Action bins, scaler, imputer sadece train split'ten fit edildi
- SOFA onset hesaplaması future bilgi kullanır ama policy input'una future bilgi verilmez
- 90-day mortality state feature olarak kullanılmaz
- Temporal alignment: state-action-next_state indekslemesi denetlendi (bkz. McCoubrey et al. 2026)
