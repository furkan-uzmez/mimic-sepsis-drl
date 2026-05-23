# MIMIC Sepsis Offline RL — Konuşma Notları

Bu doküman, konuşmamız boyunca netleştirdiğimiz ana kavramları toplu ve sade
şekilde özetler. Amaç, pipeline'ın ne yaptığını ve CQL'nin replay verisiyle
nasıl öğrendiğini kısa ama doğru biçimde anlatmaktır.

## 1. Genel Resim

Bu proje:

- MIMIC-IV içinden uygun sepsis ICU hastalarını seçer
- her hasta için bir sepsis **onset** zamanı belirler
- onset etrafında episode oluşturur
- episode'u 4 saatlik step'lere böler
- her step için:
  - state üretir
  - o step'te klinisyenin yaptığı action'ı çıkarır
  - reward hesaplar
- sonra bunlardan replay dataset üretir
- en sonda CQL / BCQ / IQL gibi offline RL algoritmalarıyla öğrenme yapar

Kısacası:

```text
raw data
-> cohort
-> onset
-> episodes
-> features (state)
-> actions
-> rewards
-> transitions / replay
-> CQL training
```

## 2. Bu Sistem Hangi RL Türüne Giriyor?

Bu repo:

- **offline RL**
- **off-policy**
- **model-free**
- ağırlıklı olarak **value-based**

### Neden offline RL?

Çünkü ajan yeni veri toplamıyor. Sadece geçmişte klinisyenlerin verdiği
kararlar ve onların sonuçlarından öğreniyor.

### Neden off-policy?

Çünkü veri, öğrenilen policy tarafından değil, **clinician behavior policy**
tarafından üretilmiş.

### Neden model-free?

Çünkü ayrı bir environment modeli ya da hasta dinamiği modeli öğrenilip planlama
yapılmıyor. Doğrudan value function öğreniliyor.

## 3. Cohort Nedir?

`cohort`, çalışmaya giren hasta/stay grubudur.

Temel seçim mantığı:

- yetişkin hasta (`>= 18`)
- ICU stay olmak zorunda
- minimum ICU kalış süresi `>= 4 saat`
- eksik kritik demografi olmamalı
- aynı hastanın tekrar yatışları çıkarılır
- pratikte ilk uygun ICU stay tutulur

Önemli fikir:

> Cohort, “hangi hastalar bu çalışmaya giriyor?” sorusunun cevabıdır.

## 4. Onset Nedir?

`onset`, sepsis sürecini zaman ekseninde sabitlediğimiz referans zamandır.

Bu şu işe yarar:

- her hasta için ortak bir referans an belirlenir
- episode bunun etrafında kurulur

Bu projede episode mantığı kabaca:

- başlangıç: `onset - 24 saat`
- merkez: `onset`
- bitiş: `onset + 48 saat`

Önemli fikir:

> Onset, episode'un kendisi değil; episode'u kurmak için kullandığımız zaman referansıdır.

## 5. Episodes ve Episode Steps

### `episodes`

Episode düzeyi üst kayıttır.

Burada tipik olarak:

- hangi stay'e ait olduğu
- onset zamanı
- episode başlangıç ve bitiş zamanı
- step sayısı
- truncation bilgisi

### `episode_steps`

Episode içindeki 4 saatlik adımlardır.

Bunlar:

- iterate edilen gerçek zaman adımlarıdır
- state/action/reward bunların üstüne oturur

Bunu şöyle düşünebilirsin:

```python
for episode in episodes:
    for step in episode_steps_of_episode:
        ...
```

Önemli fikir:

> `episodes` üst seviye bölüm kaydıdır, `episode_steps` ise bölümün içindeki 4 saatlik sayfalardır.

## 6. State Nasıl Oluşuyor?

State üretimi `features` katmanında oluyor.

Mantık:

- `episode_steps` bize hangi 4 saatlik pencereye bakacağımızı söyler
- sonra raw MIMIC tabloları o pencereye göre filtrelenir
- ölçümlerden feature çıkarılır
- eksik değerler işlenir
- türetilmiş feature'lar eklenir

Yani:

```text
episode_steps zaman penceresi
-> raw MIMIC tables
-> feature extraction
-> state vector
```

### State kaynakları

Base feature'lar şu tablolardan gelir:

- `chartevents`
- `labevents`
- `inputevents`
- `outputevents`
- `patients`

### Türetilmiş feature örnekleri

- `pf_ratio = pao2 / fio2_vent`
- `shock_index = heart_rate / sbp`
- `hours_since_onset`

### Missingness flag

Bazı feature'lar için `_missing` kolonları eklenir.

Bu flag şunu söyler:

- ölçüm gerçekten var mıydı?
- yoksa bu değer imputasyonla mı dolduruldu?

Önemli fikir:

> State, `episodes` içinden doğrudan çıkmıyor; `episode_steps` ile tanımlanan zaman pencereleri kullanılarak raw MIMIC tablolardan üretiliyor.

## 7. Action Nedir?

Action, her step'te klinisyenin gerçekten yaptığı tedavinin discretize edilmiş
halidir.

Bu projede:

- action space toplam `25` discrete action
- vazopressör ve IV fluid kombinasyonlarından geliyor

Çok önemli not:

> `a_t` random seçilmiyor.
> `a_t`, veride gerçekten yapılmış clinician action.

Ama:

> `a_t`'nin en iyi action olduğu garanti değil.

Yani elimizde şu bilgi var:

> “Bu durumda klinisyen bunu yaptı.”

Ama şu bilgi yok:

> “Bu durumda kesin en iyi yapılması gereken action buydu.”

## 8. Reward Nasıl Hesaplanıyor?

Reward, step-level hesaplanıyor.

Temel form:

```text
reward_t = terminal_t + sofa_shaping_t + lactate_shaping_t + map_shaping_t
```

Varsayılan reward variant:

- `sofa_shaped`

Bu da pratikte şu anlama gelir:

- ara step'lerde küçük shaping
- final step'te büyük terminal reward

### Terminal reward

Sadece son step'te uygulanır.

- 90 gün yaşadıysa: `+15`
- 90 gün içinde öldüyse: `-15`

### SOFA-delta shaping

Ara step'lerde klinik kötüleşme/iyileşmeyi yansıtır.

Form:

```text
sofa_shaping = sofa_delta_weight * (SOFA_current - SOFA_previous)
```

Varsayılan:

- `sofa_delta_weight = -0.025`

Buna göre:

- SOFA artarsa negatif reward
- SOFA azalırsa pozitif reward

Önemli fikir:

> Reward, “bu action optimaldi” etiketi değildir.
> Reward, gözlenen klinik geçişin outcome sinyalidir.

## 9. Replay Nedir?

State, action ve reward hazır olduktan sonra bunlar replay üretiminde kullanılır.

Replay'in temel yapısı:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

Burada:

- `s_t`: mevcut state
- `a_t`: clinician action
- `r_t`: hesaplanan reward
- `s_{t+1}`: sonraki step'in state'i
- `done_t`: episode bitti mi?

Önemli fikir:

> Replay, training'in doğrudan kullandığı son eğitim formatıdır.

## 10. Training Döngüsünde Episodes Kullanılıyor mu?

Preprocessing tarafında:

- `episodes`
- `episode_steps`

kullanılıyor.

Ama asıl training başladığında model artık genelde doğrudan bunlarla çalışmaz.

Training tarafında esas kullanılan:

- `replay_train.parquet`
- transition batch'leri

Yani:

- `episodes` ve `episode_steps` önceki katmanda işini yapmış olur
- training loop hazır replay verisinden sample alır

Önemli fikir:

> Training loop'un ana girdisi artık episode tablosu değil, replay dataset'tir.

## 11. CQL Replay ile Nasıl Çalışıyor?

CQL, replay'den mini-batch'ler çeker:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

Sonra:

1. `s_t` için tüm action'ların Q değerlerini üretir
2. veride gözlenen action `a_t` için ilgili Q değerini alır
3. `s_{t+1}` üzerinden Bellman target hesaplar
4. TD loss hesaplar
5. CQL conservative penalty ekler
6. backprop ile network ağırlıklarını günceller

## 12. Q-function Nedir?

Teoride:

```text
Q(s,a)
```

Pratikte bu projede:

> Q-function küçük bir neural network ile temsil ediliyor.

Bu network:

- input: state vector
- output: her action için bir Q değeri

Yani:

```text
Q_theta(s_t) -> [Q(s_t,1), Q(s_t,2), ..., Q(s_t,25)]
```

Çok önemli not:

> Bu 25 değerin hepsi veriden doğrudan okunmuyor.
> Network bunları tahmin ediyor.

## 13. Burada Nasıl Bir NN Var?

CQL içinde kullanılan ağ basit bir MLP:

```text
state_dim -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(25)
```

Yani:

- 2 hidden layer
- her biri 256 nöron
- çıktı 25 action için 25 Q değeri

Bu yüzden burada:

> Q-function = küçük MLP

## 14. Q Değerleri Başta Nasıl Başlıyor?

Klasik tabular Q-learning gibi tüm hücreler `0` ile başlamıyor.

Burada:

- neural network random weight'lerle initialize ediliyor
- dolayısıyla Q çıktıları başta küçük / rastgele tahminler oluyor

Önemli fikir:

> Random olan reward değil; başlangıçtaki network tahminleri.

## 15. Bellman Target Nedir?

Observed transition'dan target hesaplanır:

```text
target = r_t + gamma * max_a Q_target(s_{t+1}, a)
```

Burada:

- `r_t`: veriden gelen gerçek observed reward
- `max_a Q_target(s_{t+1}, a)`: modelin next state için yaptığı tahmin

Yani Bellman target:

- bir kısmı gerçek veri
- bir kısmı model tahmini

Önemli fikir:

> `r_t` tek başına yeterli değil; Q-learning uzun vadeli toplam değeri öğrenmek istediği için future value da ekleniyor.

## 16. Neden Sadece Observed Action Üzerinden Update Var?

Veride doğrudan bildiğimiz şey:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

Yani aynı `s_t` için sadece gözlenen clinician action'ın sonucunu biliyoruz.

Başka bir action verilseydi ne olurdu, çoğu zaman bilmiyoruz.

Bu yüzden:

- doğrudan supervision observed `a_t` üzerinden geliyor
- diğer action'ların değerleri network genellemesiyle öğreniliyor

Önemli fikir:

> Diğer action'lara random reward verilmiyor.
> Sadece doğrudan gözlenen action için reward var.

## 17. CQL'nin Ekstra Farkı Nedir?

Normal Q-learning tüm action'lara Q atamaya çalışırken,
offline setting'de veri dışı action'lara aşırı iyimser davranabilir.

CQL bunu azaltmak için conservative penalty ekler.

Kabaca:

```text
L = TD loss + alpha * CQL penalty
```

Amaç:

- veride olmayan / az desteklenen action'ların Q'su çok yükselmesin

Önemli fikir:

> CQL diğer action'ları gerçekten denemez; onlar için değer tahmini yapar ve bu tahminin fazla iyimser olmasını bastırır.

## 18. Backprop Neden Var?

Çünkü gerçek `Q(s,a)` fonksiyonunu bilmiyoruz.

Elimizde:

- modelin tahmini
- Bellman target

var.

Aradaki fark loss oluşturur.

Backprop da bu loss'u küçültmek için kullanılır.

Yani:

> Backprop, Q-network tahminlerini Bellman hedeflerine yaklaştırmak için vardır.

## 19. Outer Loop / Inner Loop Sezgisi

FrozenLake benzetmesiyle:

- bir oyun = bir episode
- oyun içindeki hamleler = step'ler
- tüm oyunlar = tüm patient episode'ları

Ama bu projede fark:

- yeni oyun üretmiyoruz
- geçmişte yaşanmış hasta trajektorilerini replay olarak kullanıyoruz

Kavramsal olarak:

- outer loop = episode
- inner loop = step

Training implementasyonu açısından ise çoğu zaman:

- replay buffer
- mini-batch transition sample

ile çalışılıyor.

## 20. Epoch Nedir?

Offline RL training'de aynı replay dataset birden fazla kez kullanılır.

Bu projede CQL default olarak:

- `n_epochs = 200`

Yani aynı train replay dataset üzerinde çok kez geçiliyor.

Önemli fikir:

> Aynı hastaları tekrar tekrar görmek, yeni veri üretmek için değil, Q-function tahminlerini daha iyi hale getirmek içindir.

## 21. En Önemli Düzeltmeler

Konuşma boyunca karışma ihtimali yüksek noktaların kısa özeti:

### Doğru

- `a_t` clinician action'dır
- reward observed transition'dan gelir
- replay training'in ana girdisidir
- Q-function burada küçük bir NN'dir
- Bellman target içinde hem gerçek reward hem model tahmini vardır

### Yanlış anlaşılmaması gerekenler

- diğer action'lara random reward verilmiyor
- target veride hazır duran tek bir kolon değil
- finalde elimizde clinician action listesi değil, öğrenilmiş Q-function ve policy var
- CQL diğer action'ları gerçek environment'da deneyip reward toplamıyor

## 22. Tek Paragraflık En Temiz Özet

Bu pipeline, MIMIC-IV sepsis hastalarını seçip her hasta için sepsis onset zamanını
belirledikten sonra onset etrafında 4 saatlik episode step'leri oluşturur.
Bu step'lerden raw MIMIC tabloları kullanılarak state vector'ler üretilir,
klinisyenin yaptığı tedavi action'ı discretize edilir ve klinik gidişat ile
90 günlük outcome'a göre reward hesaplanır. Daha sonra bu bilgiler
`(s_t, a_t, r_t, s_{t+1}, done_t)` formatında replay dataset'e çevrilir.
CQL, bu replay verisinden küçük bir Q-network kullanarak her action için değer
tahmin etmeyi öğrenir; bunu yaparken observed transition'lardan Bellman target
hesaplar, veri dışı action'lara aşırı iyimser davranmamak için konservatif ceza
ekler ve backprop ile ağ ağırlıklarını günceller.
