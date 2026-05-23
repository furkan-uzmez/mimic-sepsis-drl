# Pipeline ve RL Konumlandırması

Bu doküman, bu repodaki pipeline'ın RL literatüründeki yerini kısa ve net biçimde
anlatır: MDP nerede tanımlanıyor, CQL/BCQ/IQL hangi sınıfa giriyor, sistem
on-policy mi off-policy mi, online mı offline mı, model-free mi model-based mi,
value-based mi policy-based mi?

## Kısa Cevap

Bu repo:

- **offline RL** çalışır
- **off-policy** veri kullanır
- **model-free** algoritmalar uygular
- ana RL tarafında ağırlıklı olarak **value-based** yöntemler kullanır
- karar verme problemi olarak bir **MDP** kurar
- CQL/BCQ/IQL ise bu MDP üstünde öğrenen **algoritma katmanıdır**

Kısacası:

> Önce klinik veriden bir **MDP + offline replay dataset** üretiyoruz.  
> Sonra bunun üstünde **offline, off-policy, model-free** RL algoritmaları
> olan **CQL, BCQ ve IQL** eğitiyoruz.

---

## 1. Pipeline'da MDP Nerede Başlıyor?

Bu projede MDP, ham MIMIC-IV kayıtlarının karar problemi formatına çevrilmesiyle
kurulur.

MDP bileşenleri:

- **State (durum)**:
  hastanın 4 saatlik penceredeki klinik durumu; vital bulgular, laboratuvarlar,
  türetilmiş klinik özellikler ve eksiklik göstergeleri ile oluşturulan sürekli
  özellik vektörü
- **Action (eylem)**:
  tedavi kararı; bu projede vazopressör ve IV sıvı dozları ayrıklaştırılarak
  toplam **25 ayrık aksiyon**
- **Reward (ödül)**:
  terminal mortalite ödülü ve ara shaping bileşenleri
- **Transition (geçiş)**:
  bir durumdan seçilen eylem sonrası bir sonraki 4 saatlik duruma geçiş
- **Episode**:
  onset etrafında tanımlanan çok adımlı hasta trajektorisi

Bu yüzden MDP; `cohort -> onset -> episode grid -> split -> state/action/reward -> transitions/replay`
akışının sonunda somut hale gelir.

Pratik olarak MDP ile en çok ilişkili katmanlar:

- `src/mimic_sepsis_rl/mdp/*`
- `src/mimic_sepsis_rl/cli/build_transitions.py`
- `data/replay/*.parquet`

Önemli ayrım:

- **MDP**, problem tanımıdır
- **CQL / BCQ / IQL**, bu problem üstünde çalışan öğrenme algoritmalarıdır

Yani **CQL MDP'nin bir parçası değil**, MDP'den üretilen offline dataset üzerinde
eğitilen ajan yöntemidir.

---

## 2. Bu Sistem Online RL mi, Offline RL mi?

Bu repo açık biçimde **offline RL** sistemidir.

Bunun nedeni:

- ajan environment ile canlı etkileşime girmez
- yeni veri toplayarak öğrenmez
- sadece geçmiş klinisyen kararlarından oluşan sabit dataset ile eğitilir
- eğitim verisi `replay_train.parquet` gibi önceden donmuş artifaktlardan gelir

Bu nedenle buradaki eğitim:

- online exploration yapmaz
- rollout toplayarak policy güncellemez
- retrospective, logged clinical trajectories üzerinden öğrenir

Başka deyişle:

- **online RL**: ajan veri toplarken öğrenir
- **offline RL**: ajan sabit geçmiş veriden öğrenir

Bu repo ikinci gruptadır.

---

## 3. On-Policy mi, Off-Policy mi?

Bu repo pratikte **off-policy** kurulumdadır.

Neden?

- veri, öğrenilen ajan tarafından üretilmemiştir
- veri, **behavior policy** olarak klinisyen kararlarından gelmiştir
- öğrenilen hedef politika, logged clinician policy'den farklı olabilir

Yani:

- davranış politikası: klinisyen
- hedef politika: CQL / BCQ / IQL ile öğrenilen politika

Bu, klasik off-policy senaryodur.

Ek olarak offline RL neredeyse her zaman off-policy karakter taşır; çünkü veri,
şu an optimize ettiğimiz policy'nin kendi etkileşimlerinden gelmez.

---

## 4. Model-Free mi, Model-Based mi?

Bu repodaki RL eğitim tarafı **model-free**'dir.

Yani algoritmalar:

- ortam dinamiklerini ayrı bir geçiş modeli olarak öğrenmez
- `P(s'|s,a)` veya reward model üzerinden planlama yapmaz
- doğrudan value/policy nesnelerini offline veriden optimize eder

Bu nedenle:

- **CQL**: model-free
- **BCQ**: model-free
- **IQL**: model-free

Eğer önce hasta dinamiğini tahmin eden bir dünya modeli öğrenip sonra onun üstünde
planlama yapsaydık bu daha çok model-based tarafa yaklaşırdı. Bu repoda öyle bir
tasarım yok.

---

## 5. Value-Based mi, Policy-Based mi?

Bu repo ağırlıklı olarak **value-based** offline RL çizgisindedir.

### CQL

CQL bu projede açık biçimde **value-based**'dir.

Sebep:

- Q-network öğrenir
- her state için tüm aksiyonların `Q(s,a)` değerini üretir
- politika, tipik olarak bu Q değerlerinden `argmax_a Q(s,a)` ile türetilir

Yani CQL tarafında doğrudan öğrenilen ana nesne policy değil, **Q-value**'dır.
Policy bu Q fonksiyonundan çıkar.

### BCQ

BCQ ayrık aksiyon bağlamında yine büyük ölçüde **value-based** düşünülür.

- destek dışı aksiyonları sınırlamaya çalışır
- Q tahminini kullanır
- hedef politika seçimini value yapısı yönlendirir

### IQL

IQL biraz daha hibrit görünür ama bu repodaki kullanım bağlamında hâlâ
**value-learning merkezli offline RL** ailesindedir.

- value / advantage benzeri nesneler öğrenir
- doğrudan online policy gradient tarzı bir kurulum değildir
- davranış verisinden policy improvement yapar

Bu yüzden dokümantasyon düzeyinde en doğru kısa ifade şudur:

> Bu repo, çoğunlukla **value-based offline RL** yöntemleri uygular.

---

## 6. CQL Tam Olarak Nereye Giriyor?

CQL'nin bu repodaki doğru sınıflandırması:

- **offline RL**
- **off-policy**
- **model-free**
- **value-based**
- **discrete-action** RL

Kod düzeyinde CQL:

- eğitim katmanında yer alır: [src/mimic_sepsis_rl/training/cql.py](/Users/enesdemir/Documents/mimic-sepsis/src/mimic_sepsis_rl/training/cql.py)
- ortak deney yüzeyine kayıtlıdır: [src/mimic_sepsis_rl/training/registry.py](/Users/enesdemir/Documents/mimic-sepsis/src/mimic_sepsis_rl/training/registry.py)
- offline replay contract ile çalışır: [src/mimic_sepsis_rl/training/experiment_runner.py](/Users/enesdemir/Documents/mimic-sepsis/src/mimic_sepsis_rl/training/experiment_runner.py)

Matematiksel olarak CQL, standart TD/Bellman kaybına konservatif bir düzenleme ekler:

`L = L_TD + alpha * E[logsumexp_a Q(s,a) - Q(s,a_data)]`

Bunun amacı:

- veri desteği dışındaki aksiyonların Q değerlerini şişirmemek
- offline RL'de çok kritik olan extrapolation error riskini azaltmak

Yani CQL'nin projedeki rolü:

- MDP'den üretilen sabit replay verisini alır
- clinician behavior'dan öğrenir
- destek dışı aksiyonlara karşı daha temkinli bir Q-policy üretir

---

## 7. "Policy" Olarak Nasılız?

Bu repoda policy üretimi şu şekilde düşünülmelidir:

1. Klinik veriden state-action-reward-transition kayıtları çıkarılır
2. Bunlardan offline replay dataset üretilir
3. CQL / BCQ / IQL bu sabit dataset üstünde eğitilir
4. Eğitimin sonunda bir target policy elde edilir
5. Bu policy online servise çıkmaz; held-out veride OPE ve safety kontrolleriyle değerlendirilir

Buradaki policy:

- **retrospective learned policy**'dir
- bedside deployment policy'si değildir
- clinician policy'nin yerine geçen canlı karar verici değildir

Bu nedenle en doğru ifade:

> Bu repo, klinisyen davranışından öğrenen ve sonrasında retrospective olarak
> değerlendirilen **offline target policy**'ler üretir.

---

## 8. Pipeline Katmanları

Projeyi katmanlı düşünürsek:

### A. Klinik veri hazırlama katmanı

- cohort seçimi
- onset tespiti
- episode grid oluşturma
- split üretimi

Bu katman henüz RL algoritması değildir; veri problem uzayını hazırlar.

### B. MDP kurulum katmanı

- state extraction
- action discretization
- reward computation
- transition / replay dataset üretimi

Burada klinik veri, RL problemine çevrilir.

### C. Offline RL eğitim katmanı

- CQL
- BCQ
- IQL

Burada artık MDP üstünde öğrenme yapılır.

### D. Değerlendirme katmanı

- OPE
- ESS / WIS / FQE
- clinician agreement
- support / safety kontrolleri

Bu katman policy'nin retrospective performansını ölçer.

---

## 9. En Net Özet Tablosu

| Soru | Bu repo için cevap |
|---|---|
| Problem tipi | MDP |
| Veri tipi | Logged retrospective clinical trajectories |
| RL rejimi | Offline RL |
| Veri-politika ilişkisi | Off-policy |
| Ortam modeli öğreniliyor mu? | Hayır, model-free |
| Ana öğrenme stili | Ağırlıklı olarak value-based |
| CQL hangi sınıfta? | Offline + off-policy + model-free + value-based |
| BCQ hangi sınıfta? | Offline + off-policy + model-free + value-based ağırlıklı |
| IQL hangi sınıfta? | Offline + off-policy + model-free + value-learning merkezli |
| Policy canlı mı? | Hayır, retrospective target policy |

---

## 10. Tek Cümlelik Sunum Metni

Bu pipeline, MIMIC-IV sepsis verisini önce bir **discrete-action clinical MDP**'ye
çeviren, ardından bu sabit logged dataset üzerinde **offline, off-policy,
model-free ve ağırlıklı olarak value-based** RL algoritmaları olan **CQL, BCQ ve
IQL** ile policy öğrenen bir araştırma sistemidir.
