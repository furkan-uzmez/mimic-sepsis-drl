# IQL Grafik Katalogu

Bu dosya `runs/iql/iql_baseline/` icindeki gercek IQL baseline run grafiklerini ve `runs/iql/iql_mock_evaluation/` icindeki mock evaluation grafiklerini aciklar. Mock grafikler rapor yapisini ve kod yolunu dogrulamak icindir; bilimsel yorum icin gercek replay/checkpoint ciktilariyla yeniden uretilmelidir.

## Klasorler

| Klasor | Veri tipi | Amac |
|---|---|---|
| `runs/iql/iql_baseline/` | Gercek IQL egitim run'i | Egitim stabilitesi, loss/value tanilari ve dataset/episode reward artefactleri |
| `runs/iql/iql_mock_evaluation/` | Deterministik mock veri | Eksik tez/yayin grafiklerinin kod yolunu, dosya uretimini ve rapor formatini test etmek |

## Gercek IQL Baseline Grafikleri

| Grafik | Kullandigi metrikler | Ne ise yarar? | Rapor notu |
|---|---|---|---|
| `runs/iql/iql_baseline/epoch_metrics.png` | `actor_loss_mean`, `critic_loss_mean`, `value_loss_mean`, `total_loss_mean`, `advantage_mean` | Epoch bazinda IQL optimizasyon dinamiklerini gosterir. Loss'larin patlayip patlamadigini ve egitimin genel trendini izlemek icindir. | Onemli ama mevcut hali tek eksende cok metrik cizdigi icin okunmasi zor; tezde yeniden subplot'lara bolunmus hali tercih edilmeli. |
| `runs/iql/iql_baseline/step_metrics.png` | `actor_loss`, `critic_loss`, `value_loss`, `total_loss`, `mean_q_dataset`, `mean_v_dataset`, `advantage_std` | Step bazinda ham egitim sinyallerini gosterir. Ani patlama, NaN, diverjans veya asiri gürültü kontrolu icin kullanilir. | Destekleyici debug grafigi. Ham ve kalabalik oldugu icin ana rapor figuru olarak kullanilmamali; smoothing ve ayri paneller gerekir. |
| `runs/iql/iql_baseline/q_diagnostics.png` | `mean_q_dataset`, `mean_v_dataset`, `advantage_mean`, `advantage_std` | Q ve V tahminlerinin egitim boyunca nasil oturdugunu, advantage dagiliminin genisleyip genislemedigini gosterir. | Egitim stabilitesi icin faydali. `advantage_mean` surekli negatifse value-Q olcegi yorumlanmali. |
| `runs/iql/iql_baseline/dataset_action_heatmap.png` | Dataset aksiyon frekanslari, 25 ayrik action bin'i, vaso/fluid bin decoder | Davranis politikasinin yani klinisyen/dataset aksiyon dagilimini gosterir. Veri desteginin hangi action bolgelerinde yogunlastigini anlamaya yarar. | Tek basina learned policy'yi gostermez; policy karsilastirmasi icin `iql_action_heatmaps.png` daha gucludur. |
| `runs/iql/iql_baseline/episode_reward_curve.png` | Episode return veya episode reward sirasi | Egitim/run seviyesinde episode odul seyrini gosterir. Reward sinyalinde buyuk anomali var mi diye bakilir. | Ana basari kaniti degil; OPE/FQE/WIS ile desteklenmeli. |
| `runs/iql/iql_baseline/episode_reward_distribution.png` | Episode return dagilimi | Odul dagiliminin merkezi, yayilimi ve outlier davranisini gosterir. | Destekleyici analiz icin uygun; model secimi icin tek basina yeterli degil. |
| `runs/iql/iql_baseline/episode_reward_rolling_curve.png` | Episode return rolling mean/trend | Episode reward trendinin daha okunur hareketli ortalamasini verir. | Egitim analizi veya appendix icin yararli. |

## Mock IQL Evaluation Grafikleri

| Grafik | Kullandigi metrikler | Ne ise yarar? | Rapor notu |
|---|---|---|---|
| `runs/iql/iql_mock_evaluation/iql_action_heatmaps.png` | `clinician_action`, `policy_action`, action decoder, normalized action matrix | Clinician aksiyon dagilimi, IQL policy aksiyon dagilimi ve delta heatmap'i yan yana gosterir. Policy klinisyenden hangi vaso/fluid bolgelerinde ayriliyor sorusunu cevaplar. | Klinik makulluk ve action-distribution safety icin ana grafiklerden biridir. |
| `runs/iql/iql_mock_evaluation/iql_fqe_vs_low_support.png` | `fqe_mean`, `low_support_action_rate`, `seed`, `epoch` | Her checkpoint icin deger tahmini ile veri destegi riskini ayni grafikte gosterir. | Yuksek FQE ama yuksek low-support olan modelleri elemek icin faydali. |
| `runs/iql/iql_mock_evaluation/iql_seed_variance.png` | `fqe_mean`, `seed`, `epoch` | Checkpoint/seed seviyesinde FQE oynakligini gosterir. | Multi-seed stabilite argumani icin destekleyici. |
| `runs/iql/iql_mock_evaluation/iql_bootstrap_ci.png` | `wis_mean`, `wis_lower`, `wis_upper` | WIS icin bootstrap confidence interval hata cubuklarini gosterir. | OPE belirsizligini raporlamak icin onemli. |
| `runs/iql/iql_mock_evaluation/iql_subgroup_safety.png` | `subgroup`, `agreement`, `low_support` | Standard/high-risk gibi subgroup'larda agreement ve low-support oranlarini karsilastirir. | High-risk hastalarda policy guvenligi icin hizli ozet. |
| `runs/iql/iql_mock_evaluation/iql_advantage_weight_histogram.png` | `support_prob` uzerinden proxy advantage weight, clipped weight | Advantage weight dagilimi ve clipping davranisini gosterir. | IQL actor update'inin asiri agirliklara dayanip dayanmadigini anlamaya yarar; mock'ta proxy oldugu belirtilmeli. |
| `runs/iql/iql_mock_evaluation/iql_episode_return_distribution.png` | Episode discounted return | Episode return histogrami ve sorted return profile gosterir. | Outcome/reward dagilimini ozetler; OPE metrikleriyle birlikte yorumlanmali. |
| `runs/iql/iql_mock_evaluation/iql_support_action_frequency.png` | `support_prob`, `support_count`, `policy_action` frekansi | IQL'nin sectigi aksiyonlarin dataset behavior support'u ve policy action frekanslarini gosterir. | Low-support riskini action seviyesinde aciklar. |
| `runs/iql/iql_mock_evaluation/iql_metric_correlation.png` | `fqe_mean`, `wis_mean`, `ess`, `low_support_action_rate`, `clinician_agreement` | OPE ve safety metriklerinin birbirleriyle korelasyonunu gosterir. | Hangi metriklerin ayni yonde hareket ettigini anlamaya yarar; az checkpoint varsa yorum sinirli olur. |
| `runs/iql/iql_mock_evaluation/iql_example_trajectory_plot.png` | `episode_id`, `step_index`, `clinician_action`, `policy_action`, `low_support` | Ornek bir hasta trajectory'sinde clinician ve IQL action kararlarini zaman boyunca karsilastirir. | Manuel klinik inceleme icin iyi bir vaka secimi grafigi. |
| `runs/iql/iql_mock_evaluation/iql_reward_decomposition.png` | Mock SOFA proxy, step reward, intermediate reward, terminal reward, cumulative reward | Klinik zaman serisi + reward decomposition paneli uretir. | PDF Sekil 3 karsiligi. Mock'ta SOFA proxy kullanir; gercek veriyle daha anlamli olur. |
| `runs/iql/iql_mock_evaluation/iql_cumulative_episode_rewards.png` | Episode discounted return, cumulative return, rolling mean | Episode reward trendini ve kümülatif odul seyrini gosterir. | PDF Sekil 5 karsiligi. Egitim/episode odul davranisini ozetler. |
| `runs/iql/iql_mock_evaluation/iql_clinician_agreement.png` | Exact match, adjacent-bin match, subgroup | Clinician ile IQL action'lari arasinda tam eslesme ve komsu-bin eslesme oranlarini gosterir. | PDF Sekil 7 karsiligi. Klinik makulluk icin cok onemli. |
| `runs/iql/iql_mock_evaluation/iql_pareto_frontier.png` | `low_support_action_rate`, `fqe_mean`, `ess`, `clinician_agreement` | FQE yuksek, low-support dusuk, ESS/agreement makul olan checkpointleri secmek icin trade-off grafigi sunar. | Final model secimini tek FQE'ye degil safety-performance dengesine dayandirmak icin en guclu grafiklerden biri. |
| `runs/iql/iql_mock_evaluation/iql_ope_metric_ranking.png` | Normalize `FQE`, `WIS`, `ESS`, support skoru, agreement skoru | Checkpointleri birden cok OPE/safety kriterine gore yan yana siralar. | "Neden bu checkpoint?" sorusuna gorsel cevap verir. |
| `runs/iql/iql_mock_evaluation/iql_action_deviation_severity.png` | Vaso/fluid bin delta, exact/adjacent/moderate/large sapma | Policy action'in clinician action'dan ne kadar uzaklastigini ozetler. | Heatmap'ten daha okunur bir klinik sapma ciddiyeti grafigidir. |
| `runs/iql/iql_mock_evaluation/iql_high_risk_action_heatmaps.png` | `subgroup == high_risk`, clinician/policy action matrix, delta matrix | Sadece high-risk hastalarda clinician-IQL action farkini heatmap olarak gosterir. | Shock/severe hasta davranisini savunmak icin faydali opsiyonel ama guclu grafik. |
| `runs/iql/iql_mock_evaluation/iql_training_diagnostics.png` | Mock `critic_loss`, `value_loss`, `actor_loss`, `mean_q`, `mean_v`, `advantage`, `adv_weight_mean`, `adv_clip_fraction` | IQL egitim tanilarini 2x3 panelde gosterir: Critic/Q loss, Value/Expectile loss, Actor loss, Q/V/Advantage, Advantage Weight Mean, Advantage Clip Fraction. | CQL training diagnostics grafiginin IQL karsiligi. Mock oldugu icin format/rapor yerlesimi icin kullanilmali; gercek run log'lariyla yeniden uretilmesi daha dogru. |

## Tablo ve Ozet Dosyalari

| Dosya | Icerik | Ne ise yarar? |
|---|---|---|
| `runs/iql/iql_mock_evaluation/iql_metrics_summary.csv` | Checkpoint bazinda `FQE`, `WIS`, `ESS`, low-support, agreement, high-risk metrikleri | Rapor tablolarinin ve ranking/Pareto grafiklerinin kaynak tablosu |
| `runs/iql/iql_mock_evaluation/iql_seed_variance.csv` | FQE mean/std/min/max ve checkpoint sayisi | Seed/checkpoint stabilite ozeti |
| `runs/iql/iql_mock_evaluation/iql_example_trajectory_review.csv` | Manuel review icin secilmis row-level action kararları | Vaka incelemesi ve klinik sanity-check icin |
| `runs/iql/iql_mock_evaluation/iql_evaluation_summary.json` | Tum mock evaluation sonucu ve artifact path'leri | Pipeline'in tek dosyalik manifest/ozeti |
| `runs/iql/iql_baseline/metrics_summary.json` | Gercek baseline run metriklerinin final/mean/min/max ozeti | Egitim metriklerinin sayisal denetimi |
| `runs/iql/iql_baseline/artifact_index.json` | Gercek baseline run artifact path'leri | Hangi grafigin nerede oldugunu izlemek icin |

## Hangi Grafikler Ana Rapor Icin Daha Onemli?

Ana metin veya tez savunmasi icin en guclu grafikler:

1. `iql_action_heatmaps.png` - clinician vs IQL action davranisi.
2. `iql_pareto_frontier.png` - model secimi icin FQE/support/ESS/agreement trade-off'u.
3. `iql_ope_metric_ranking.png` - checkpointleri coklu metrikle siralama.
4. `iql_bootstrap_ci.png` - WIS belirsizligi.
5. `iql_clinician_agreement.png` - exact ve adjacent-bin klinik uyum.
6. `iql_action_deviation_severity.png` - klinisyen kararindan sapma ciddiyeti.
7. `iql_high_risk_action_heatmaps.png` - high-risk hasta alt grubunda action guvenligi.

Egitim tanisi veya appendix icin daha uygun grafikler:

1. `iql_training_diagnostics.png`
2. `epoch_metrics.png`
3. `step_metrics.png`
4. `q_diagnostics.png`
5. `episode_reward_curve.png`
6. `episode_reward_distribution.png`
7. `episode_reward_rolling_curve.png`

## Onemli Yorum Notlari

- Mock grafikler bilimsel sonuc degil, kod ve rapor formatinin smoke test'idir.
- Gercek rapor icin `scripts/evaluate_iql_sweep.py` ayni artifact setini gercek `replay_test.parquet` ve gercek IQL checkpointleriyle yeniden uretmelidir.
- `epoch_metrics.png` ve `step_metrics.png` mevcut halleriyle kalabalik oldugu icin ana figür olarak onerilmez; `iql_training_diagnostics.png` benzeri ayri panel duzeni daha uygundur.
- Calibration/reliability grafigi bu katalogda yoktur; gercek episode-level predicted value ve observed return hesaplanmadan mock ile yorumlamak guvenilir degildir.
