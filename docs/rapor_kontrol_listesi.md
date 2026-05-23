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
|---|---|---|
| 1 | Kumar et al. 2020 (CQL) | A. Kumar, A. Zhou, G. Tucker, and S. Levine, "Conservative Q-Learning for Offline Reinforcement Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. |
| 2 | Fujimoto et al. 2021 (BCQ) | S. Fujimoto, D. Meger, and D. Precup, "Off-Policy Deep Reinforcement Learning without Exploration," in *International Conference on Machine Learning (ICML)*, 2019. |
| 3 | Kostrikov et al. 2021 (IQL) | I. Kostrikov, A. Nair, and S. Levine, "Offline Reinforcement Learning with Implicit Q-Learning," in *International Conference on Learning Representations (ICLR)*, 2022. |
| 4 | Komorowski et al. 2018 (AI Clinician) | M. Komorowski, L. A. Celi, O. Badawi, A. C. Gordon, and A. A. Faisal, "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care," *Nature Medicine*, vol. 24, pp. 1716-1720, 2018. |
| 5 | Levine et al. 2020 (Offline RL) | S. Levine, A. Kumar, G. Tucker, and J. Fu, "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems," *arXiv preprint arXiv:2005.01643*, 2020. |
| 6-10 | Ek kaynaklar | OPE literatürü (Tang & Wiens 2021, Voloshin et al. 2021), sepsis RL review (Pappada et al. 2023), d3rlpy (Seno & Imai 2022), MIMIC-IV (Johnson et al. 2023), SOFA skoru (Vincent et al. 1996) |

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
