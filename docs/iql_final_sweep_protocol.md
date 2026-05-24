# IQL Final Hyperparameter Sweep Protocol

## 1. Problem Tanimi

Bu projenin amaci, MIMIC-IV sepsis kohortunda klinisyen tedavi kararlarini ve hasta sonuclarini kullanarak retrospektif bir offline reinforcement learning (Offline RL) politikasi egitmektir. Ortam gercek hastalar uzerinde aktif deneme yapmaya izin vermedigi icin ajan yalnizca kaydedilmis klinik veriden ogrenir. Bu nedenle amac, sepsis tedavisinde IV sivi ve vazopressor kararlarini temsil eden ayrik aksiyonlar icin, veri destegi disina cikmadan klinik olarak makul bir politika ogrenmektir.

Calisma prospektif klinik karar sistemi degildir. Elde edilen politika ve metrikler sadece retrospektif analiz, model karsilastirmasi ve offline policy evaluation kapsaminda yorumlanmalidir.

## 2. MDP Formulasyonu

Problem sonlu ufuklu bir Markov Decision Process (MDP) olarak modellenir:

```text
M = (S, A, P, R, gamma)
```

Burada:

- `S`: Hasta durum uzayi
- `A`: Ayrik tedavi aksiyonlari
- `P`: Durum gecis dinamikleri
- `R`: Odul fonksiyonu
- `gamma`: Discount factor

### 2.1 Zaman Adimi

Her hasta epizodu 4 saatlik klinik pencerelere ayrilir. Her pencere bir karar adimi olarak kabul edilir. Analiz penceresi sepsis onset'inden 24 saat once baslayip 48 saat sonrasina kadar devam eder; bu nedenle tipik bir episode yaklasik 18 karar adimi icerir. Episode uzunlugu veri uygunluguna gore degisebilir. Bu nedenle `t` anindaki durum, ilgili 4 saatlik pencereye kadar gozlenen klinik bilgiyi temsil eder.

```text
s_t -> a_t -> r_t -> s_{t+1}
```

### 2.2 State

State vektoru hastanin o andaki klinik durumunu ozetler. Projede state temsili 62 surekli/ikili klinik ozellikten olusur ve MIMIC-IV tablolarindan turetilen vital bulgular, laboratuvarlar, tedavi gecmisi, demografik degiskenler, missingness indicator'lari ve turetilmis ozellikleri kapsar.

Baslica state gruplari:

- Vital bulgular: kalp hizi, MAP, sistolik/diyastolik tansiyon, solunum, SpO2 vb.
- Laboratuvarlar: kan gazi, elektrolitler, renal/hepatik/coagulation belirtecleri, hemogram
- Tedavi sinyalleri: kümülatif sivi, vazopressor maruziyeti, idrar cikisi
- Demografi: yas, cinsiyet
- Turetilmis degiskenler: PaO2/FiO2, shock index, onset sonrasi saat

Her 4 saatlik pencerede birden fazla olcum varsa ozellige gore `last`, `mean`, `min`, `max`, `sum` veya `cumulative` agregasyon kurali uygulanir. Eksik veriler once episode icinde forward-fill ile, bu mumkun degilse train split uzerinden hesaplanan medyanlarla doldurulur. Bu train-only medyan kullanimi leakage engellemek icin kritiktir. State temsili retrospektif modelleme icin bir yaklasimdir; hekimin yatak basi yargisi, gizli siddet, tedavi niyeti ve olculmeyen klinik degiskenler tam olarak temsil edilemez.

### 2.3 Action

Aksiyon uzayi 25 ayrik tedavi kararindan olusur. Her aksiyon, vazopressor dozu ve IV sivi miktarinin 5 x 5 kombinasyonudur.

```text
action_id = vaso_bin * 5 + fluid_bin
```

Vazopressor ve sivi binleri:

```text
0: tedavi yok
1: non-zero Q1
2: non-zero Q2
3: non-zero Q3
4: non-zero Q4
```

Aksiyon grid'i:

```text
          fluid_bin -> 0   1   2   3   4
vaso_bin
0                     0   1   2   3   4
1                     5   6   7   8   9
2                    10  11  12  13  14
3                    15  16  17  18  19
4                    20  21  22  23  24
```

Bin esikleri sadece train split uzerindeki non-zero dozlardan ogrenilir. Validation ve test splitlerinde ayni dondurulmus bin esikleri kullanilir.

### 2.4 Reward

Projede iki ana reward varyanti final sweep icin kullanilacaktir:

```text
sparse
sofa_shaped
```

`sparse` reward yalniz terminal mortalite sonucunu kullanir:

```text
survived 90 days: +15
 died within 90 days: -15
```

`sofa_shaped` reward terminal odulu korur ve ara adimlarda SOFA degisimine gore kucuk bir shaping sinyali ekler:

```text
sofa_shaping = sofa_delta_weight * (SOFA_current - SOFA_previous)
```

Varsayilan SOFA agirligi:

```text
sofa_delta_weight = -0.025
```

Buna gore SOFA artisi kotulesme olarak negatif odul, SOFA azalisi iyilesme olarak pozitif odul uretir. Terminal odulun buyuk tutulmasi, mortalite sinyalinin shaping tarafindan bastirilmamasini saglar.

Model secimi ve final raporlama icin sparse ve SOFA-shaped reward ile egitilen politikalar ayni terminal survival/death utility altinda degerlendirilmelidir. Boylece shaped ve sparse modeller farkli reward olcekleriyle secilmez; reward varyanti yalnizca egitim sinyali olarak yorumlanir.

### 2.5 Transition

Transition dataset su formdaki tuple'lardan olusur:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

- `s_t`: 4 saatlik pencere sonunda hasta durumu
- `a_t`: ayni pencerede klinisyen tarafindan uygulanmis ayrik tedavi aksiyonu
- `r_t`: secilen reward varyantina gore hesaplanan odul
- `s_{t+1}`: bir sonraki pencerenin state vektoru
- `done_t`: episode sonu gostergesi

Offline RL algoritmasi bu kayitli gecisleri kullanir; egitim sirasinda yeni aksiyon deneyimi toplanmaz.

### 2.6 Discount Factor

Discount factor gelecek odullerin bugunku karar icin ne kadar onemli oldugunu belirler.

Projede sabit deger:

```text
gamma = 0.99
```

Bu secim klinik olarak uygundur cunku sepsis tedavisinde kisa vadeli hemodinamik kararlar uzun vadeli mortalite sonucuyla iliskilidir. `gamma=0.99`, terminal mortalite sinyalini episode basindaki kararlara da tasirken ara adimlarin tamamen baskin olmasini engeller.

## 3. Dataset ve Cohort

Dataset MIMIC-IV tabanli sepsis kohortundan uretilir. Pipeline genel olarak su sirayi izler:

1. Sepsis kohortunun secilmesi
2. ICU stay/episode tanimlarinin uretilmesi
3. 4 saatlik episode grid'inin kurulmasi
4. State feature'larinin cikarilmasi
5. Action binlerinin olusturulmasi
6. Reward hesaplanmasi
7. Transition datasetinin yazilmasi
8. Offline RL egitimi ve degerlendirmesi

Egitim icin varsayilan replay dataset yolu:

```text
data/replay/replay_train.parquet
```

Metadata:

```text
data/replay/replay_train_meta.json
```

Sparse replay dataset final sweep baslamadan once uretilmis olmalidir. Sparse ve SOFA-shaped replay datasetleri yalnizca reward tanimi acisindan farkli olmali; hasta splitleri, feature preprocessing, action bin esikleri, transition indexing ve terminal outcome tanimlari ayni kalmalidir. Bu esdegerlik saglanmadan reward varyantlari arasinda yapilan karsilastirma guvenilir kabul edilmemelidir.

## 4. Preprocessing

Preprocessing asamalari leakage-safe olacak sekilde tasarlanir.

### 4.1 Feature Aggregation

Her 4 saatlik penceredeki ham olcumler klinik ozellige gore agregasyon kuraliyla tek degere indirilir:

- `last`: son olcum
- `mean`: pencere ortalamasi
- `min` / `max`: klinik olarak riskli minimum/maksimum deger
- `sum`: pencere toplam miktari
- `cumulative`: episode basindan itibaren kümülatif toplam

### 4.2 Missing Data

Eksik veriler icin sirali politika:

1. Episode icinde forward-fill
2. Train split uzerinden hesaplanan medyan
3. Klinik normal deger fallback
4. Gerekli durumlarda missingness flag

Train medyan dosyasi:

```text
data/processed/features/train_medians.json
```

### 4.3 Clipping ve Normalization

Fizyolojik olarak makul olmayan degerler valid range disinda filtrelenir veya clip edilir. Clip ve normalization parametreleri train split uzerinde fit edilir, validation/test uzerinde sadece uygulanir.

### 4.4 Action Binning

Vazopressor ve IV sivi esikleri train split uzerindeki non-zero degerlerin quartile'lariyla belirlenir. Zero treatment ayri bir bin olarak tutulur.

Bu onemlidir cunku `no treatment`, kucuk ama non-zero tedavi ile ayni klinik anlama gelmez.

## 5. Split Stratejisi

Split patient-level yapilir. Ayni hasta farkli splitlere dusmez.

Varsayilan split:

```text
train: 70%
validation: 15%
test: 15%
seed: 42
```

Manifest dosyalari:

```text
data/splits/train_manifest.parquet
data/splits/validation_manifest.parquet
data/splits/test_manifest.parquet
data/splits/split_summary.json
```

Split kurali:

```text
train_subjects ∩ validation_subjects = empty
train_subjects ∩ test_subjects = empty
validation_subjects ∩ test_subjects = empty
```

## 6. Leakage-Free Pipeline

Temel kural:

```text
Fit on train. Transform everywhere.
```

Train uzerinde fit edilmesi gerekenler:

- Scaler parametreleri
- Imputer medyan/ortalama degerleri
- Clip bound veya percentile esikleri
- Action bin quartile esikleri
- Behavior policy estimator
- FQE estimator
- Normalization sabitleri

Validation/test uzerinde yapilacaklar:

- Train'de fit edilmis transformlari uygulamak
- Hyperparameter secimi icin validation metriklerini hesaplamak
- Final raporlama icin test metriklerini hesaplamak

Test split yalniz final raporlama icin kullanilmalidir. Hyperparameter secimi test setine bakilarak yapilmamalidir.

### 6.1 Stage 0 Pre-sweep Audit

Final IQL sweep baslamadan once asagidaki audit tamamlanmalidir:

1. Sparse replay datasetini uret.
2. Sparse ve SOFA-shaped replay datasetlerinde ayni patient split, preprocessing, action bin, transition indexing ve terminal outcome tanimlari kullanildigini dogrula.
3. Action binlerinin yalniz train split uzerinde fit edildigini dogrula.
4. Imputation, scaling, clipping ve normalization istatistiklerinin yalniz train split uzerinde fit edildigini dogrula.
5. Validation FQE'nin ortak terminal survival/death evaluation reward ile calistigini dogrula.
6. Temporal alignment audit'i calistir: `s_t`, `a_t`, `s_{t+1}`, `r_t` ve terminal reward ayni transition mantigina bagli olmali.
7. Hicbir hastanin birden fazla split'te bulunmadigini dogrula.
8. Mortality, discharge status veya gelecek outcome bilgisinin state feature'larina sizmadigini dogrula.

Bu audit tamamlanmadan Stage 1 sonuclari metodolojik olarak raporlanmamalidir.

## 7. IQL Algoritmasi

Implicit Q-Learning (IQL), offline RL icin gelistirilmis bir algoritmadir. Temel fikri, dataset disindaki aksiyonlari dogrudan maksimize ederek extrapolation error uretmek yerine, kayitli davranis verisi icinde avantaj agirlikli politika iyilestirmesi yapmaktir.

IQL uc bilesen kullanir:

1. Q-function / critic
2. Value function
3. Actor / policy

### 7.1 Critic

Critic, state-action degerini tahmin eder:

```text
Q(s, a)
```

### 7.2 Value Function ve Expectile Regression

IQL'de value function klasik max operator yerine expectile regression ile ogrenilir. Expectile parametresi, value hedefinin Q dagiliminin hangi bolgesine yaklasacagini belirler.

```text
expectile = 0.7 veya 0.8
```

- Daha dusuk expectile: daha konservatif value tahmini
- Daha yuksek expectile: daha optimistic value tahmini

Offline klinik veride cok agresif optimism riskli olabilir. Bu nedenle ana sweep'te `0.7` ve kontrollu sekilde `0.8` denenir.

### 7.3 Advantage-Weighted Actor Update

Actor, dataset icindeki aksiyonlari advantage'a gore agirliklandirarak ogrenir:

```text
A(s, a) = Q(s, a) - V(s)
weight = exp(A(s, a) * temperature)
```

Projede temperature arttikca actor advantage'a daha agresif tepki verir.

- Dusuk temperature: davranis politikasina daha yakin, daha konservatif
- Yuksek temperature: yuksek avantajli aksiyonlara daha agresif kayma

Final sweep'te `temperature=1.0` ve `temperature=3.0` kullanilir.

## 8. Fixed Parameters

Final sweep boyunca sabit tutulacak parametreler:

```yaml
algorithm: iql
n_epochs: 200
batch_size: 256
gamma: 0.99
policy_hidden_sizes: [256, 256]
value_hidden_sizes: [256, 256]
critic_hidden_sizes: [256, 256]
polyak_tau: 0.005
target_update_freq: 10
grad_clip: 10.0
device: auto
```

Bu parametrelerin sabit tutulmasi, final sweep'in yalnizca reward, learning-rate rejimi, expectile ve temperature etkisini olcmesini saglar.

## 9. Denenecek Hyperparameter Sweep

Final IQL sweep:

```yaml
reward_variants:
  - sparse
  - sofa_shaped

lr_regimes:
  conservative:
    actor_lr: 0.00003
    critic_lr: 0.0001
    value_lr: 0.0001

  baseline:
    actor_lr: 0.0001
    critic_lr: 0.0003
    value_lr: 0.0003

  actor_conservative:
    actor_lr: 0.00003
    critic_lr: 0.0003
    value_lr: 0.0003

iql_settings:
  safe:
    expectile: 0.7
    temperature: 1.0

  baseline:
    expectile: 0.7
    temperature: 3.0

  optimistic:
    expectile: 0.8
    temperature: 3.0
```

Toplam config sayisi:

```text
2 reward variants x 3 LR regimes x 3 IQL settings = 18 configs
```

## 10. Seed Stratejisi

Final rapor icin tek seed yeterli degildir. Offline RL egitiminde initialization, mini-batch sirasi ve stochastic optimizasyon sonucu etkileyebilir.

Onerilen pratik uygulama:

```text
Stage 1:
  18 configs
  seed: 42

Stage 2:
  top 5 configs
  seeds: 123, 456

Final reporting:
  top 5 configs each with 3 seeds total
```

Bu durumda toplam run sayisi:

```text
18 + (5 x 2) = 28 run
```

Compute biraz daha rahatsa `top 6` secilebilir:

```text
18 + (6 x 2) = 30 run
```

Bu tasarim 54 run'lik tam grid'e gore daha ekonomiktir; yine de finalist configler 3 seed ile dogrulandigi icin final raporda seed varyansi ve stabilite analizi raporlanabilir.

## 11. Hyperparameter Secim Kriteri

Hyperparameter secimi test set uzerinden yapilmaz. Secim validation split uzerindeki asagidaki sinyallerle yapilir:

- Validation OPE skoru
- FQE degeri
- WIS ve ESS birlikte
- Clinician agreement
- Action distribution makullugu
- Low-support action uyarilari
- Reward curve ve training stability

Tek bir metrikle karar verilmemelidir. Ornegin yuksek WIS ama cok dusuk ESS varsa bu sonuc guvenilir kabul edilmemelidir.

## 12. Evaluation

Final model secimi validation split uzerinde yapilir. Test split yalnizca final config secildikten sonra bir kez kullanilir; test sonucuna bakarak config, checkpoint veya metrik tasarimi degistirilmemelidir.

Raporlanacak ana metrikler:

- FQE: Fitted Q Evaluation, ortak terminal survival/death utility ile
- Patient-level bootstrap 95% confidence interval
- WIS: Weighted Importance Sampling
- ESS: Effective Sample Size
- Behavior support mass
- Low-support action rate
- Clinician exact-bin agreement
- Clinician adjacent-bin agreement
- Policy action frequency
- Behavior support diagnostics
- Subgroup analysis

Final test karsilastirmasi asagidaki baseline'lari icermelidir:

1. Clinician replay: replay datasetindeki gozlenen klinisyen aksiyonlarini degerlendirir.
2. No-treatment policy: uygun yerlerde no-vasopressor/no-fluid bin'ini secen basit kontrol politikasi.
3. Behavior cloning: klinisyen aksiyonlarini supervised learning ile taklit eden davranis politikasi.

WIS ve ESS diagnostik metrik olarak yorumlanmalidir. Learned policy klinisyen davranisindan cok uzaklastiginda WIS yuksek varyansli olabilir; bu durumda FQE, ESS, support mass ve low-support rate birlikte raporlanmalidir.

Klinik yorum icin gerekli kontroller:

- Politika surekli asiri sivi veya asiri vazopressor oneriyor mu?
- `no treatment` aksiyonuna anormal sekilde collapse ediyor mu?
- Cok dusuk davranis destegi olan aksiyonlari sik seciyor mu?
- High-risk subgroup'larda guvenlik uyarilari artiyor mu?
- Reward variant degisince politika dramatik sekilde degisiyor mu?

## 13. Grafikler ve Raporlanacak Artefaktlar

Final raporda olmasi gereken grafikler:

1. Training loss curves
   - actor loss
   - critic loss
   - value loss

2. Validation metric curves
   - FQE over epoch/checkpoint
   - estimated return
   - ESS

3. Hyperparameter comparison plots
   - reward variant bazli boxplot
   - LR regime bazli karsilastirma
   - expectile/temperature setting bazli karsilastirma

4. Action heatmaps
   - clinician action distribution
   - IQL policy action distribution
   - difference heatmap

5. Seed variability plots
   - mean +/- std
   - per-seed scatter

6. Safety/support plots
   - low-support action rate
   - clinician agreement by subgroup
   - high-risk subgroup policy action frequency

## 14. Karar Kurali

Bir config final aday olarak secilmeden once su kosullari saglamalidir:

- Validation performansi diger configlere gore rekabetci olmali
- ESS cok dusuk olmamali
- FQE sonucu WIS ile tamamen celismemeli
- Action distribution klinik olarak makul olmali
- Low-support action orani kabul edilebilir duzeyde olmali
- Farkli seedlerde performans tamamen kararsiz olmamali
- Test set secim surecinde kullanilmamis olmali

Onerilen nihai secim bicimi:

```text
1. Stage 1 validation sonuclarindan top 4-6 config sec
2. Bu configleri 3 seed'e tamamla
3. Validation uzerinden final 1-2 config sec
4. Test set uzerinde yalniz bu final configleri raporla
```

## 15. Reproducibility Checklist

Her run icin kaydedilmesi gerekenler:

- Git commit hash
- Config dosyasi
- Reward variant
- LR regime
- Expectile
- Temperature
- Seed
- Dataset path ve metadata path
- Split manifest seed
- Action bin artifact
- Training logs
- Checkpoint path
- Evaluation output
- Figure paths

Run isimlendirme onerisi:

```text
iql_{reward}_{lr_regime}_{iql_setting}_seed{seed}
```

Ornek:

```text
iql_sofa_shaped_actor_conservative_baseline_seed42
```

## 16. Limitations ve Etik Notlar

Bu calisma klinik deployment calismasi degildir. Learned IQL politikasi gercek hasta tedavisi icin onerilmez; sonuclar yalniz retrospektif offline RL/OPE deneyi olarak yorumlanmalidir.

Ana sinirlar:

- Klinik surec gercekte POMDP'dir; state vektoru hekimin tam klinik bilgisini ve olculmeyen siddet/kontrendikasyon sinyallerini kapsamaz.
- Veri retrospektif ve observational oldugu icin clinician action'lari confounding icerir.
- Offline RL politikasi davranis verisinde zayif desteklenen aksiyonlara kayabilir; support mass ve low-support action rate bu nedenle zorunlu raporlanir.
- FQE ve WIS prospektif klinik kanit degildir; fonksiyon yaklasimi, behavior policy tahmini ve variance varsayimlarina baglidir.
- MIMIC-IV verisi yalniz PhysioNet credentialing, CITI training ve Data Use Agreement kapsaminda kullanilmalidir; hasta verileri de-identified olsa da raporlama akademik ve etik sinirlar icinde kalmalidir.

## 17. Sonuc

Bu protokol, CQL final sweep'e benzer sekilde IQL icin sistematik ve savunulabilir bir hiperparametre taramasi tanimlar. CQL'deki `cql_alpha` yerine IQL'de `expectile` ve `temperature` birlikte taranir. Learning rate ise IQL'in actor, critic ve value bilesenleri ayri oldugu icin tek sayi yerine LR regime olarak ele alinir.

Onerilen final sweep:

```text
2 reward variants x 3 LR regimes x 3 IQL settings x 3 seeds = 54 runs
```

Compute kisitli durumda once 18 config tek seed ile taranip, umut vadeden configler farkli seedlerle dogrulanabilir. Tum pipeline boyunca ana prensip, train-only fit ve held-out test setin yalniz final raporlama icin kullanilmasidir.
