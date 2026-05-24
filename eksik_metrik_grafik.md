# Eksik Metrik ve Grafikler

Bu not, MIMIC sepsis offline DRL/IQL-CQL projesinde yayın/tez için gerekli metrik ve grafiklerin mevcut durumunu özetler.

## Minimum Yayın/Tez Figür Seti Durumu

| No | İstenen metrik/grafik | Projede var mı? | Durum |
|---:|---|---|---|
| 1 | FQE/WIS/ESS comparison table/plot | Kısmen | FQE/WIS/ESS altyapısı var ama IQL run çıktılarında tablo/plot yok. |
| 2 | FQE vs low-support scatter | Yok | FQE + low-support birlikte raporlanmıyor. |
| 3 | Clinician vs learned policy action heatmaps | Kısmen | Heatmap fonksiyonları var; mevcut IQL run'da sadece dataset/action heatmap görünüyor, learned policy karşılaştırması yok. |
| 4 | Delta action heatmap | Kısmen | Kod altyapısı var ama IQL output'ta üretilmemiş. |
| 5 | Subgroup safety plot | Kısmen | Subgroup altyapısı var ama high-risk subgroup tanımı/plot otomatik yok. |
| 6 | Seed variance plot | Yok | Tek IQL baseline run var gibi; multi-seed aggregation/plot yok. |
| 7 | Training diagnostics curves | Var | `epoch_metrics.png`, `step_metrics.png`, `q_diagnostics.png` var. |
| 8 | Advantage weight clipping/histogram | Yok | IQL trainer clipping yapıyor ama clipping ratio/histogram loglanmıyor. |
| 9 | Bootstrap CI plot | Kısmen | Bootstrap CI altyapısı var ama mevcut IQL raporunda plot yok. |
| 10 | Example trajectory review | Var | `scripts/evaluate_iql_sweep.py` CSV inceleme tablosu ve trajectory plot uretir. |
| 11 | Episode return distribution/profile | Var | IQL evaluation episode return histogram + sorted return profile uretir. |
| 12 | Support mass vs policy action frequency | Var | IQL secilen aksiyonlar icin behavior support ve policy frequency plot uretir. |
| 13 | Metric correlation matrix | Var | FQE/WIS/ESS/low-support/agreement korelasyon matrisi uretir. |
| 14 | Clinical time series + reward decomposition | Var | PDF Sekil 3 karsiligi olarak IQL mock/real episode uzerinden SOFA proxy + reward decomposition plot uretir. |
| 15 | Cumulative episode reward curve | Var | PDF Sekil 5 karsiligi olarak cumulative discounted episode reward + rolling trend plot uretir. |
| 16 | Exact vs adjacent-bin clinician agreement | Var | PDF Sekil 7 karsiligi olarak exact ve komsu-bin agreement oranlarini uretir. |
| 17 | Pareto frontier plot | Var | FQE yuksek + low-support dusuk + ESS/agreement makul model secim trade-off grafigi uretir. |
| 18 | OPE metric ranking plot | Var | Checkpointleri FQE/WIS/ESS/support/agreement normalized skorlarina gore siralar. |
| 19 | Action deviation severity plot | Var | Clinician-policy vaso/fluid bin sapmasini exact/adjacent/moderate/large ve delta heatmap olarak gosterir. |
| 20 | High-risk-only action heatmap | Var | Sadece high_risk subgroup icin clinician/IQL/delta aksiyon heatmap uretir. |
| 21 | IQL training diagnostics 2x3 panel | Var | Critic/Q loss, value/expectile loss, actor loss, Q/V/advantage, advantage weight mean ve clip fraction mock paneli uretir. |

## En Kritik 7 Şey Açısından Durum

| Kritik madde | Senin projede durum |
|---|---|
| Validation FQE | Kısmen var: `evaluation/ope.py` içinde altyapı var, ama IQL baseline output'ta gerçek FQE sonucu yok. |
| WIS + ESS | Kısmen var: hesaplama fonksiyonu var, ama mevcut IQL metrics içinde yok. |
| Low-support action rate | Kısmen/var: `evaluation/safety.py` içinde var, ama IQL run raporuna bağlanmamış. |
| Clinician vs policy action heatmap | Kısmen var: fonksiyon var, ama learned policy için üretilmiş çıktı yok. |
| High-risk subgroup safety | Kısmen var: subgroup altyapısı var, ama high-risk klinik tanım ve otomatik plot yok. |
| Advantage weight clipping ratio | Yok. |
| Seed variance | Yok veya çok eksik: multi-seed IQL sweep/aggregation yok. |

## Net Cevap

Şu anda gerçekten hazır olan tek şey:

```text
Training diagnostics curves
```

Kısmen hazır ama rapora bağlanmamış olanlar:

```text
FQE
WIS/ESS
low-support action rate
clinician/policy heatmap
delta heatmap
subgroup safety
bootstrap CI
```

Eksik olanlar:

```text
Bunlar artik eklendi:
FQE vs low-support scatter
seed variance plot
advantage weight clipping ratio/histogram
example trajectory review CSV + plot
multi-seed comparison table
IQL sweep evaluation report
episode return distribution/profile
support mass vs policy action frequency
metric correlation matrix
clinical time series + reward decomposition
cumulative episode reward curve
exact vs adjacent-bin clinician agreement
pareto frontier plot
OPE metric ranking plot
action deviation severity plot
high-risk-only action heatmap
IQL training diagnostics 2x3 panel
```

## Öncelikli Yapılacaklar

1. `iql.py` içine advantage weight logları eklendi:

```text
adv_weight_clip_fraction
adv_weight_mean
adv_weight_max_raw
```

2. IQL run'ları için tek bir `evaluate_iql_sweep.py` script'i yazıldı. Bu script şunları üretir:

```text
FQE
WIS
ESS
low-support action rate
clinician vs policy heatmap
delta action heatmap
subgroup safety
bootstrap CI
seed variance
FQE vs low-support scatter
multi-seed comparison table
episode return distribution/profile
support mass vs policy action frequency
metric correlation matrix
example trajectory plot
clinical time series + reward decomposition
cumulative episode reward curve
exact vs adjacent-bin clinician agreement
pareto frontier plot
OPE metric ranking plot
action deviation severity plot
high-risk-only action heatmap
IQL training diagnostics 2x3 panel
```

## Önerilen Karar Mantığı

Final policy seçimi sadece en yüksek FQE'ye göre yapılmamalı. Seçilecek model şu koşulları birlikte sağlamalı:

```text
FQE yüksek veya clinician baseline'dan iyi
WIS aynı yönde destekliyor
ESS tamamen çökmemiş
low-support action rate düşük
clinician action distribution'dan aşırı kopmamış
high-risk subgroup davranışı klinik olarak makul
advantage weight clipping oranı aşırı değil
seed variance düşük
```

## Kısa Öncelik Sırası

1. Advantage weight clipping metriklerini IQL trainer'a ekle.
2. IQL evaluation/sweep rapor script'i yaz.
3. FQE/WIS/ESS + safety metriklerini tek tabloda topla.
4. Heatmap ve scatter plotları üret.
5. Multi-seed aggregation ve seed variance plot ekle.
6. Example trajectory review için birkaç hasta seçip clinician vs RL action çiz.
