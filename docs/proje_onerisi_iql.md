# Proje Önerisi — MIMIC Sepsis Offline RL (IQL Seçimi)

## 1. Proje önerisi
Bu projede amaç, **MIMIC-IV sepsis kohortu** üzerinde klinisyenlerin geçmiş tedavi kararlarından öğrenen bir **offline reinforcement learning** ajanı geliştirmektir. Ajanın görevi, her 4 saatlik klinik durumda hastaya uygulanabilecek **vazopressör + IV sıvı** kombinasyonlarından uygun olanı seçen bir politika öğrenmektir.

Bu çalışma **prospektif klinik kullanım** için değil, **retrospektif politika öğrenme ve karşılaştırma** amacıyla tasarlanmıştır.

Projenin çekirdek değeri:
- **klinik olarak makul**
- **veri sızıntısına dayanıklı**
- **yeniden üretilebilir**
bir offline RL benchmark'ı oluşturmaktır.

---

## 2. Hangi çevreyi kullanacağız?
Klasik anlamda canlı bir simülatör ortamı değil, **MIMIC-IV tabanlı offline klinik karar ortamı** kullanılacaktır.

Çevre şu şekilde tanımlanır:
- Her hasta trajektorisi bir **episode** olarak ele alınır.
- Zaman adımı **4 saat**tir.
- Episode penceresi onset etrafında tanımlanır.
- Eğitim, canlı etkileşimden değil, daha önce kaydedilmiş klinisyen kararlarından oluşan **replay dataset** üzerinden yapılır.

Yani bu proje **online environment interaction** değil, **fixed logged dataset environment** kullanır.

Pipeline özeti:

```text
raw data
-> cohort
-> onset
-> episodes
-> state
-> action
-> reward
-> transitions / replay
-> offline RL training
```

---

## 3. State (s), Action (a), Reward (r), Probability (P)

### State (s)
State, hastanın 4 saatlik zaman penceresindeki klinik durumunu temsil eden **sürekli özellik vektörüdür**.

Mevcut replay metadata'sına göre eğitim kontratı:
- **state_dim = 62**
- Durum değişkenleri; vital bulgular, laboratuvarlar, kümülatif tedavi bilgileri, yaş/kilo gibi sabit değişkenler ve eksiklik göstergelerini içerir.

Örnek state bileşenleri:
- heart rate
- MAP, SBP, DBP
- respiratory rate, temperature, SpO2
- GCS
- lactate, creatinine, BUN, bilirubin
- INR, PTT, WBC, platelets
- cumulative IV fluid
- cumulative vasopressor dose
- urine output
- age, weight
- PF ratio, shock index
- hours since onset
- missingness indicator'ları

### Action (a)
Action, tedavi kararını temsil eder.

Bu projede eylem uzayı:
- **5 vazopressör seviyesi × 5 IV sıvı seviyesi = 25 ayrık aksiyon**
- `action_id = vaso_bin × 5 + fluid_bin`

Dolayısıyla:
- **n_actions = 25**
- Her aksiyon, klinik olarak yorumlanabilir bir tedavi kombinasyonudur.

### Reward (r)
Reward fonksiyonu iki ana parçadan oluşur:
- **terminal reward**: 90 günlük mortaliteye göre
  - survived: **+15**
  - died: **-15**
- **intermediate shaping**: özellikle **SOFA değişimi** üzerinden

Varsayılan ödül varyantı:
- **`sofa_shaped`**

Yani ödül, sadece sonuca değil, klinik gidişatın iyileşip kötüleşmesine de duyarlıdır.

### Probability (P)
MDP içindeki geçiş olasılığı:
- **P(s'|s,a)**

Ancak bu projede bu olasılık **ayrı bir geçiş modeli olarak öğrenilmiyor**. Çünkü seçilen yaklaşım **model-free**.

Burada P kavramsal olarak vardır; yani:
- aynı klinik durum ve aynı tedavi altında
- farklı hastalar farklı sonraki durumlara geçebilir
- dolayısıyla geçişler tek bir sabit sonuca değil, bir **olasılık dağılımına** karşılık gelir

Ek not:
- Reward kontratı deterministik tanımlanmıştır; aynı episode verisi ve aynı reward config aynı reward dizisini üretir.
- Buna karşılık hasta geçiş dinamiği yine de stokastik yorumlanmalıdır.

---

## 4. Model-free mi, model-based mi? Neden?
Bu projede **model-free** bir yöntem kullanıyoruz.

### Neden model-free?
Çünkü:
- elimizde canlı etkileşimli bir ortam yok
- ayrı bir hasta dinamiği modeli (`P(s'|s,a)`) öğrenip onun üstünde planlama yapmıyoruz
- sabit replay verisi üzerinden doğrudan **value / policy** öğreniyoruz
- repoda doğrudan desteklenen algoritmalar da bu sınıfta: **CQL, BCQ, IQL**

### Neden model-based seçmedik?
Model-based yaklaşım için:
- önce güvenilir bir hasta geçiş modeli kurmak gerekir
- klinik veride gizli değişkenler ve ölçüm eksikleri nedeniyle bu çok zordur
- yanlış öğrenilmiş transition modeli, tedavi önerilerini yanıltabilir

Bu nedenle mevcut proje kapsamı için **model-free offline RL**, daha gerçekçi ve uygulanabilir seçimdir.

Ayrıca `.planning` kararlarına göre ilk tasarım zaten:
- **offline**
- **off-policy**
- **model-free**
- **discrete 25-action**
olarak sabitlenmiştir.

---

## 5. Deterministik mi, stokastik mi? Neden stokastik seçmeliyiz?
Bu problem **stokastik** olarak ele alınmalıdır.

### Neden stokastik?
Çünkü:
- aynı klinik state altında aynı tedavi verildiğinde her hastanın yanıtı aynı olmaz
- hasta fizyolojisi tam gözlenebilir değildir
- kayıt gürültüsü, ölçüm eksikliği ve biyolojik değişkenlik vardır
- ICU süreci doğal olarak belirsizlik içerir

Yani gerçekçi olan yaklaşım:
- **reward hesabı deterministik olabilir**
- ama **environment transition yapısı stokastiktir**

### Neden deterministik varsaymamalıyız?
Çünkü deterministik varsayım:
- klinik belirsizliği küçümser
- aynı aksiyonun her hastada aynı sonucu doğurduğunu varsayar
- offline medical RL için gerçekçi olmayan aşırı basitleştirme olur

Bu yüzden problem tanımı açısından **stokastik MDP** daha doğrudur.

---

## 6. Eylem uzayı ayrık mı, sürekli mi?
Bu projede **discrete action space** seçilmiştir.

### Seçim
- **Ayrık eylem uzayı**
- Toplam **25 aksiyon**

### Neden discrete action space?
Çünkü:
- vazopressör ve sıvı dozları klinik olarak anlamlı bin'lere ayrılmıştır
- offline veride destek bölgelerini kontrol etmek daha kolaydır
- OPE ve policy karşılaştırması daha şeffaf olur
- CQL / BCQ / IQL'nin repo içindeki implementasyonları bu 25-action sözleşmesiyle uyumludur

### Continuous action space neden seçilmedi?
Sürekli aksiyon uzayı teorik olarak daha esnek olsa da:
- doz yoğunluğu desteği seyrek olabilir
- offline veride extrapolation riski büyür
- değerlendirme ve klinik yorumlama zorlaşır
- mevcut repo kontratı sürekli değil, **25-action discrete grid** üzerine kuruludur

Bu nedenle bu proje için **ayrık aksiyon uzayı daha uygulanabilir** seçimdir.

Bu karar `.planning/PROJECT.md` ve `.planning/REQUIREMENTS.md` ile de uyumludur; continuous-action kontrol ilk kapsam dışında bırakılmıştır.

---

## 7. Hangi yöntemi kullanmayı planlıyorum?
Benim seçtiğim yöntem: **IQL (Implicit Q-Learning)**

### Neden IQL?
Bizdeki üç ana aday algoritma:
- **CQL**
- **BCQ**
- **IQL**

Bu proje için **IQL'yi seçiyorum**, çünkü:

1. **Offline RL ile doğal uyumlu**
   - IQL, sabit logged dataset üzerinde öğrenmek için tasarlanmıştır.
   - Online exploration gerektirmez.

2. **Off-policy ve model-free yapıya tam uyumlu**
   - Projemizin veri üretim biçimi klinisyen davranış politikası üzerinden geliyor.
   - IQL bu yapıda doğrudan uygulanabilir.

3. **Discrete 25-action yapısına repo içinde zaten uyarlanmış durumda**
   - Repoda `src/mimic_sepsis_rl/training/iql.py` implementasyonu var.
   - `configs/training/iql.yaml` hazır.
   - `experiment_runner` içinde kayıtlı.

4. **Davranış dağılımından çok kopmadan policy improvement yapabilmesi önemli**
   - IQL, expectile value learning + advantage-weighted behavioral cloning yaklaşımıyla çalışır.
   - Bu, özellikle klinik logged data gibi güvenliğin önemli olduğu senaryolarda avantajlıdır.

5. **BCQ'ya göre daha esnek, CQL'ye göre daha dengeli bir tercih**
   - BCQ, behavior-support maskesiyle daha kısıtlayıcı olabilir.
   - CQL ise aşırı konservatif olduğunda faydalı aksiyonları da bastırabilir.
   - IQL, veri desteğinden tamamen kopmadan iyileştirme yapabildiği için bu proje için iyi bir orta yol sunar.

### Bu yöntem gerçekten uygulanabilir mi?
Evet.

Repo içi uygulanabilirlik kanıtı:
- `configs/training/iql.yaml` mevcut
- `src/mimic_sepsis_rl/training/iql.py` mevcut
- `src/mimic_sepsis_rl/training/registry.py` içinde kayıtlı
- `uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql --describe` başarılı
- `uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql --dry-run` başarılı

Dry-run doğrulaması, IQL eğitim akışının mevcut proje altyapısında çalıştığını gösterir.

Ek repo bağlamı:
- Phase 7: CQL referans eğitim hattı tamamlanmış
- Phase 8: BCQ ve IQL karşılaştırma hattı tamamlanmış
- Mevcut proje odağı: **Phase 9 evaluation / safety / reproducibility**

Yani IQL seçimi yalnızca teorik değil, proje yol haritası açısından da doğru aşamaya oturuyor.

---

## 8. Hangi yöntemleri araştırdım ve neden IQL seçtim?

### Repo içindeki doğrudan adaylar
Bu projede gerçekçi ve uygulanabilir üç aday:
- **CQL**
- **BCQ**
- **IQL**

### CQL neden seçilmedi?
CQL çok güçlü bir offline RL adaydır; özellikle OOD aksiyonlara karşı konservatif davranır.

Ama bu projede tek başına ilk tercih olarak seçmeme nedenim:
- fazla konservatif kalıp değerleri gereğinden fazla aşağı çekebilmesi
- bazı faydalı ama az görülen aksiyonları da bastırabilmesi

### BCQ neden seçilmedi?
BCQ da offline RL için uygundur.

Ancak:
- behavior-support eşiğine duyarlıdır
- mask tabanlı seçim bazı durumlarda fazla sınırlayıcı olabilir
- discrete klinik aksiyonlarda IQL kadar dengeli bir iyileştirme profili sunmayabilir

### IQL neden seçildi?
IQL'yi seçtim çünkü:
- offline veriye uygun
- off-policy yapıya uygun
- model-free
- discrete 25-action yapıya uygun
- repoda hazır implementasyonu var
- klinik davranış verisinden tamamen kopmadan politika iyileştirmesi yapabiliyor

Kısacası:
> **Bu proje için IQL, teorik uyum + pratik uygulanabilirlik + mevcut repo desteği açısından en dengeli seçimdir.**

---

## 9. PPO, DDPG, A3C, TD3 gibi yöntemleri neden seçmedim?
Bu yöntemleri de araştırdım; ancak mevcut proje kurgusu açısından uygun değiller.

### PPO
- genellikle **on-policy** çalışır
- yeni rollout toplamak ister
- sabit offline dataset ile verimli kullanım için doğal seçim değildir

### A3C
- online ve etkileşimli environment varsayar
- paralel actor mantığına dayanır
- retrospektif klinik replay verisi için uygun değildir

### DDPG
- ağırlıklı olarak **continuous action space** için tasarlanmıştır
- bizim eylem uzayımız ayrık 25-action grid
- bu yüzden mevcut problem formuna doğrudan uymaz

### TD3
- DDPG'nin geliştirilmiş versiyonu gibi düşünülebilir
- yine esas olarak **continuous control** içindir
- mevcut discrete clinical action contract ile doğal uyumlu değildir

### Sonuç
Bu yüzden mevcut proje için en mantıklı algoritma havuzu:
- **CQL / BCQ / IQL**

ve bu üçlü içinde benim seçimim:
- **IQL**

Buradaki seçim sadece literatür tercihi değil, aynı zamanda proje kapsamı tercihi:
- PPO ve A3C, offline logged-data kurulumuna doğal uymaz
- DDPG ve TD3, continuous-action beklentisi taşır
- proje gereksinimleri ise açık biçimde **discrete-action offline RL** ister

---

## 10. Değerlendirme ve başarı ölçütü
IQL seçmek tek başına yeterli değildir; politika aşağıdaki ortak değerlendirme yüzeyinde incelenecektir:

### Karşılaştırılacak baseline'lar
- **Clinician replay baseline**
- **No-treatment baseline**
- **Behavior cloning baseline**

### OPE / güvenlik değerlendirmeleri
- **WIS**
- **ESS**
- **FQE**
- clinician sanity review
- action-frequency heatmaps
- subgroup analysis
- support-aware warnings

Bu nokta önemlidir:
> Projede başarı sadece “eğitim loss'u düştü” demek değildir; IQL politikası aynı data contract üzerinde baseline'lara ve OPE metriklerine karşı anlamlı şekilde değerlendirilecektir.

---

## 11. Kapsam dışı kalanlar
`.planning` dosyalarına göre aşağıdakiler bu önerinin kapsamı dışındadır:
- canlı bedside deployment
- prospektif klinik etki iddiası
- continuous-action control
- model-based RL
- sepsis dışı kohortlar

Bu yüzden öneri metni özellikle **retrospektif araştırma sistemi** çerçevesinde yazılmıştır.


---

## 12. Nihai karar özeti

| Başlık | Seçim |
|---|---|
| Problem tipi | Offline RL |
| Veri rejimi | Off-policy |
| Çevre | MIMIC-IV tabanlı sabit replay environment |
| State uzayı | Sürekli klinik özellik vektörü |
| Action uzayı | Ayrık, 25 aksiyon |
| Reward | Terminal mortalite + SOFA shaping |
| Transition yapısı | Stokastik |
| Yöntem sınıfı | Model-free |
| Aday algoritmalar | CQL, BCQ, IQL |
| Seçilen algoritma | **IQL** |
| Seçim nedeni | Offline veriye uygun, discrete action ile uyumlu, repo içinde uygulanabilir ve dengeli |

---

## 13. Kısa savunma cümlesi
Bu projede **offline, off-policy, model-free ve discrete-action** bir klinik karar problemi kurduğumuz için; **PPO/A3C gibi on-policy**, **DDPG/TD3 gibi continuous-control** yöntemler yerine, **offline RL için tasarlanmış CQL/BCQ/IQL** ailesine odaklandım. Bu üç aday içinde **IQL'yi**, hem teorik uyumu hem de mevcut repo içinde hazır ve doğrulanmış biçimde uygulanabiliyor olması nedeniyle seçtim.
