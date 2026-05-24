# IQL Final Sweep Eksiklikleri

Bu dosya `docs/iql_final_sweep_protocol.md` protokolunun mevcut repo koduyla karsilastirilmasi sonucu gorulen eksikleri ozetler.

## Kontrol Ozeti

Kontrol edilen komut:

```bash
uv run python -m pytest tests/training/test_iql_metrics.py tests/baselines/test_baselines.py tests/evaluation/test_bootstrap.py -q
```

Sonuc:

```text
31 passed
```

Not: sistemde `python` komutu yok; testler `uv run python` ile calisiyor.

## Hazir Olanlar

- IQL trainer mevcut: `src/mimic_sepsis_rl/training/iql.py`
  - Actor / critic / value network var.
  - `expectile`, `temperature`, ayri `actor_lr`, `critic_lr`, `value_lr` destekleniyor.
  - Advantage weight clipping metrikleri loglaniyor: `adv_weight_clip_fraction`, `adv_weight_mean`, `adv_weight_max_raw`.
- Baseline IQL config mevcut: `configs/training/iql.yaml`
  - `gamma=0.99`, `batch_size=256`, `n_epochs=200`, hidden sizes, LR, expectile ve temperature tanimli.
- Sparse ve SOFA-shaped reward altyapisi mevcut:
  - `src/mimic_sepsis_rl/mdp/reward_models.py`
  - `src/mimic_sepsis_rl/cli/build_transitions.py`
- Replay/transition builder sparse reward uretebiliyor.
- IQL checkpoint evaluator / grafik uretici mevcut: `scripts/evaluate_iql_sweep.py`
  - Mock mode mevcut.
  - Grafik ve CSV artifact uretimi mevcut.
- Clinician, no-treatment ve behavior cloning baseline modulleri/testleri mevcut.
- Patient-level bootstrap altyapisi mevcut ve ilgili testler geciyor.

## Eksik / Tam Hazir Olmayanlar

### 1. IQL sweep runner yok

Eksik dosya:

```text
scripts/run_iql_sweep.py
```

Protokoldeki su akisi otomatik calistiracak script henuz yok:

```text
Stage 1: 18 configs x seed 42
Stage 2: final 6 configs x seeds 123, 456
```

CQL icin benzer script var: `scripts/run_cql_sweep.py`. IQL karsiligi yazilmali.

### 2. 18-config grid generator yok

Final IQL grid su kombinasyonlardan olusuyor:

```text
2 reward variants x 3 LR regimes x 3 IQL settings = 18 configs
```

Bu grid'i otomatik genisleten, her config icin gecici training YAML/manifest olusturan ve run isimlerini standartlastiran kod hazir degil.

### 3. IQL final akisini kapsayan Snakefile yok

Mevcut repo kokunde `Snakefile` var; ancak CQL sweep odakli ve `runs/cql_sweep`, `scripts/run_cql_sweep.py`, `scripts/evaluate_cql_sweep.py` hedeflerini kullaniyor. Bu nedenle IQL final sweep protokolunu kapsayan Snakefile hazir degil.

Snakefile yalniz 18 config'i denememeli; veri artifact uretiminden final rapora kadar tum final IQL akisini dosya tabanli DAG olarak kapsamalidir.

Kapsanacak uc-uca akis:

```text
1. Gerekli processed cohort/episode/split girdilerini kontrol et
2. Sparse replay datasetini uret: data/replay_sparse/replay_{split}.parquet
3. SOFA-shaped replay datasetini uret: data/replay/replay_{split}.parquet
4. Stage 0 pre-sweep audit'i calistir
5. Sparse ve SOFA replay artifactlerinin ayni patient split, preprocessing, action bin, transition indexing ve terminal outcome tanimlarini kullandigini dogrula
6. Action binlerinin yalniz train split uzerinde fit edildigini dogrula
7. Imputation/scaling/clipping/normalization istatistiklerinin yalniz train split uzerinde fit edildigini dogrula
8. Temporal alignment audit'i calistir: s_t, a_t, s_{t+1}, r_t ve terminal reward ayni transition mantigina bagli olmali
9. Split leakage audit'i calistir: hicbir hasta birden fazla split'te bulunmamali
10. Outcome leakage audit'i calistir: mortality, discharge status veya gelecek outcome state feature'larina sizmamali
11. 18-config IQL grid manifestini uret
12. Stage 1: 18 config'in tamamini seed 42 ile egit
13. Stage 1 checkpointlerini ortak terminal survival/death evaluation reward ile validation FQE/WIS/ESS/support metrikleri uzerinden degerlendir
14. Final-6 config secimini ve secim gerekcesini manifest/CSV olarak uret
15. Stage 2: final-6 config'i seed 123 ve 456 ile tekrar egit; seed 42 run'larini yeniden kullan
16. Stage 2 finalist sonuclarini 3 seedlik ozet manifestte topla
17. Final selected IQL checkpoint'i test split uzerinde baseline'larla karsilastir
18. Final CSV/JSON/figure/report artifactlerini uret
```

Beklenen ana Snakemake hedefleri:

```text
rule all:
  results/iql_final/final_report.md
  results/iql_final/final_metrics.json
  results/iql_final/final_comparison.csv
  results/iql_final/stage1/selection/top5_configs.csv
  results/iql_final/stage2/finalists_manifest.json
  results/iql_final/audit/presweep_audit.json
```

Snakefile, her adim icin beklenen input/output dosyalarini acik tanimlamali. Boylece eksik output varsa sadece ilgili rule tekrar calismali, tum sweep bastan baslamamali. Uzun egitim adimlari icin rule'lar config/seed bazinda ayrilmali ve `uv run python ...` komutlari kullanilmalidir.

Test / mock calistirma zorunlulugu:

```text
1. Snakefile icin mock veya mini synthetic data modu olmali.
2. Gercek MIMIC artifactleri olmadan DAG'in yapisal olarak calistigi dogrulanmali.
3. `snakemake -n` dry-run basarili olmali.
4. Mock modda kucuk synthetic replay/split/checkpoint artifactleri uretilip `rule all` tamamlanmali.
5. Uzun egitim yerine mock IQL checkpoint/evaluation ciktilari ureten hizli rule veya script kullanilmali.
6. CI/test komutu olarak en az su dogrulanmali:
   - `uv run snakemake -n --cores 1`
   - `uv run snakemake --cores 1 --config mock=true`
```

Bu mock calistirma, gercek sweep baslamadan once Snakefile'in input/output baglantilarini, rule isimlerini, config genisletmesini, stage bagimliliklarini ve final artifact hedeflerini dogrulamak icin zorunlu kabul edilmeli.

Kullanilacak skill'ler:

```text
.agent/skills/snakemake-mlops-workflows/SKILL.md
.agent/skills/snakemake-workflow-engine/SKILL.md
.agent/skills/uv-python-workflows/SKILL.md
.agent/skills/medical-dl-experiment-reporting/SKILL.md
.agent/skills/python-observability/SKILL.md
```

### 4. Stage 1 runner yok

Eksik islev:

```text
18 config'i seed 42 ile egit
checkpointleri kaydet
validation uzerinde degerlendir
Stage 1 manifest uret
```

Mevcut IQL trainer tek config calistirabiliyor; Stage 1 toplu calistirma orchestration'i yok.

### 5. Stage 1 validation evaluator yok

Eksik islev:

```text
Her Stage 1 checkpointini validation split uzerinde degerlendir
best checkpoint'i validation FQE ile sec
final 6 config'i belirle
```

Mevcut `scripts/evaluate_iql_sweep.py` checkpointleri okuyup artifact uretir, fakat protokoldeki Stage 1 model-selection pipeline'ini tam karsilamaz.

### 6. Final-6 selector yok

Eksik islev:

```text
Validation metriklerinden final 6 config sec
secim gerekcesini manifest/CSV olarak yaz
Stage 2 icin config listesini uret
```

Secim yalniz tek metrikle veya sadece top-6 composite skorla yapilmamali; FQE, WIS, ESS, support mass, low-support rate, clinician agreement, action entropy, diversity ve safety sinyalleri birlikte dikkate alinmali.

Final 6 secim kurali:

```text
Final 6 = gate gecenler arasindan
  2 config: composite_score en yuksek adaylar
  1 config: en iyi sparse reward adayi
  1 config: en iyi sofa_shaped reward adayi
  1 config: en guvenli/stabil support profiline sahip aday
  1 config: baseline anchor veya diversity icin en anlamli aday
```

Hard gate kurallari:

```text
FQE_mean finite
FQE_95CI_lower >= clinician_FQE_lower - tolerance
ESS >= 50
low_support_rate <= 0.20
support_mass >= 0.80
clinician_agreement >= 0.20
action_entropy >= min_entropy
severe_safety_flags == 0
```

Kritik kural:

```text
Yuksek FQE ama dusuk support = finalist degil.
Hard gate'i gecemeyen hicbir config Stage 2'ye alinmaz.
```

Composite score yalniz Stage 1 siralama icin kullanilmali; final sonuc iddiasi Stage 2'de 3 seed ortalamasi ve bootstrap CI ile yapilmalidir.

Composite score onerisi:

```text
score =
  + 0.45 * normalized(FQE_mean)
  + 0.15 * normalized(FQE_95CI_lower)
  + 0.15 * normalized(WIS_mean)
  + 0.10 * normalized(ESS)
  + 0.10 * normalized(support_mass)
  - 0.15 * normalized(low_support_rate)
  - 0.10 * normalized(minor_safety_flags or safety_risk_score)
```

Safety/support slot skoru:

```text
safety_support_score =
  + normalized(ESS)
  + normalized(support_mass)
  + normalized(clinician_agreement)
  + normalized(action_entropy)
  - normalized(low_support_rate)
  - normalized(minor_safety_flags or safety_risk_score)
```

Safety/support slot icin ek sart:

```text
FQE_mean >= median(FQE_mean of gated configs)
```

Baseline anchor tercihi:

```yaml
reward_variant: sparse
lr_regime: baseline
iql_setting: safe
```

Baseline anchor sadece su sartlarda secilmeli:

```text
hard gate'leri geciyor
FQE_mean >= gated_median_FQE - tolerance
```

Gecmiyorsa baseline anchor zorlanmamali; gate gecenler icinde baseline'a en yakin, stabil ve sade config secilmeli.

Diversity kurallari, hard gate sonrasi mumkunse uygulanmali:

```text
En az 2 sparse
En az 2 sofa_shaped
En az 1 safe IQL setting
En az 1 baseline IQL setting
Optimistic sadece support/safety gate'leri temiz gecerse
Ayni reward + LR + IQL ailesinden 3'ten fazla config olmasin
```

Duplicate slot olursa ayni config iki kez yazilmaz; bosalan slot, diversity'yi iyilestiren siradaki en yuksek composite_score adayi ile doldurulur.

Rapor metni icin karar cumlesi:

```text
Stage 2 finalistleri yalnizca validation performansina gore degil, safety/support gate'lerini gecen adaylar arasindan gated composite ranking ile secilmistir. Ilk iki aday composite_score'a gore en yuksek modellerdir. Kalan adaylar reward ablation dengesini, klinik destek kapsamını, action-distribution stabilitesini ve baseline karsilastirilabilirligini koruyacak sekilde secilmistir. Boylece Stage 2'ye hem performans acisindan umut vadeden hem de klinik offline RL acisindan guvenilir ve yorumlanabilir 6 konfigurasyon tasinmistir.
```

### 7. Stage 2 extra-seed runner yok

Eksik islev:

```text
Final 6 config'i seed 123 ve 456 ile tekrar egit
seed 42 run'ini yeniden kullan veya gerekirse rerun et
her finalist config icin toplam 3 seedlik sonuc uret
```

### 8. Gercek FQE entegrasyonu eksik

Mevcut `scripts/evaluate_iql_sweep.py` icindeki `_fqe_actor_score`, gercek Fitted Q Evaluation degil; policy logits/action scores uzerinden ortalama hesaplayan FQE-style bir proxy.

Eksik islev:

```text
IQL policy icin validation/test FQE fit et
ortak terminal survival/death utility kullan
checkpoint/model selection icin gercek FQE skorunu raporla
```

### 9. FQE bootstrap entegrasyonu eksik

Mevcut IQL evaluator WIS bootstrap uretiyor, fakat FQE bootstrap kullanmiyor.

Eksik islev:

```text
patient-level bootstrap FQE 95% CI
validation ve final test raporunda FQE CI
```

### 10. Common terminal utility model selection'a bagli degil

Protokolde sparse ve SOFA-shaped reward ile egitilen politikalarin final secimde ayni terminal survival/death utility altinda degerlendirilmesi gerekiyor.

Eksik islev:

```text
reward-trained policy -> common terminal evaluation reward -> validation FQE/WIS/ESS
```

Bu baglanti IQL Stage 1/Stage 2 pipeline'inda net olarak hazir degil.

### 11. CUDA/GPU zorunlu calisma ve dogrulama yok

Final IQL sweep CPU fallback ile sessizce calismamali. Egitim ve evaluation pipeline'i CUDA destekli GPU kullanacak sekilde ayarlanmali ve bunu baslamadan once kesin olarak dogrulamalidir.

Eksik islev:

```text
CUDA kullanilabilirligini dogrula: torch.cuda.is_available() == True
GPU cihaz adini, CUDA runtime surumunu ve PyTorch CUDA build bilgisini logla
IQL trainer device secimini acikca cuda olarak ayarla
CPU fallback'i final sweep icin hata say
Her run manifestine device, gpu_name, cuda_version, torch_version yaz
Snakemake/runner basinda GPU preflight check calistir
```

Zorunlu kabul kriteri:

```text
Final IQL sweep, CUDA/GPU dogrulamasi gecmeden Stage 1 veya Stage 2 egitimine baslamamalidir.
GPU yoksa veya PyTorch CUDA surumu/driver uyumsuzsa script acik hata ile durmalidir.
Mock mode haricinde CPU fallback kabul edilmemelidir.
```

Minimum dogrulama:

```bash
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU bulunamadi veya kullanilamiyor'
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
PY
```

Not: mevcut test calistirmasinda PyTorch CUDA driver uyari mesaji uretti; bu final sweep oncesi giderilmeli ve GPU dogrulamasi yesil olmadan gercek egitim baslatilmamalidir.

### 12. Stage 0 audit script'i yok

Testlerde split leakage, episode grid ve preprocessing kontrolleri mevcut; ancak protokoldeki Stage 0 Pre-sweep Audit'i repo artifactleri uzerinde tek komutla calistiran script yok.

Eksik script ornegi:

```text
scripts/audit_iql_presweep.py
```

Bu script sunlari kontrol etmeli:

- sparse replay dataset var mi?
- SOFA-shaped replay dataset var mi?
- patient splitler ayni mi?
- feature preprocessing ayni mi?
- action bin esikleri ayni mi?
- transition indexing ayni mi?
- terminal outcome tanimlari ayni mi?
- train-only preprocessing artifactleri var mi?
- split leakage var mi?
- mortality/discharge/future outcome state feature'larina sizmis mi?

### 13. Gercek data artifactleri checkout'ta yok

Eksik artifactler:

```text
data/replay/replay_train.parquet
data/replay/replay_train_meta.json
data/replay_sparse/replay_train.parquet
data/replay_sparse/replay_train_meta.json
data/splits/train_manifest.parquet
data/splits/validation_manifest.parquet
data/splits/test_manifest.parquet
data/splits/split_summary.json
```

Bu dosyalar olmadan final IQL sweep gercek veri uzerinde baslatilamaz.

### 14. Final test + baseline comparison pipeline yok

Baseline modulleri var, ancak final selected IQL checkpoint ile su karsilastirmayi tek rapor halinde yapan script hazir degil:

```text
selected IQL policy
clinician replay
no-treatment policy
behavior cloning
```

Eksik islev:

```text
final test set uzerinde tek seferlik degerlendirme
FQE/WIS/ESS/support/agreement/action heatmap karsilastirmasi
CSV/JSON/figure raporu
```

## 4 Phase Uygulama Plani

Bu eksikler tek session'da uygulanmamali. En guvenli yol, artifact kontratlari ve testleri net olacak sekilde 4 ayri phase halinde ilerlemektir.

### Phase 1 — Data Artifact + Pre-sweep Audit

Amac: Final sweep baslamadan once replay datasetlerinin, splitlerin ve leakage kontrollerinin guvenli oldugunu kanitlamak.

Kapsam:

1. Gercek replay/split artifactlerini uret ve dogrula.
2. Sparse replay datasetini uret: `data/replay_sparse/replay_{split}.parquet`.
3. SOFA-shaped replay datasetini uret: `data/replay/replay_{split}.parquet`.
4. `scripts/audit_iql_presweep.py` yaz.
5. Sparse ve SOFA-shaped replay datasetlerinde ayni patient split, preprocessing, action bin, transition indexing ve terminal outcome tanimlari kullanildigini dogrula.
6. Action binlerinin yalniz train split uzerinde fit edildigini dogrula.
7. Imputation, scaling, clipping ve normalization istatistiklerinin yalniz train split uzerinde fit edildigini dogrula.
8. Temporal alignment audit'i calistir: `s_t`, `a_t`, `s_{t+1}`, `r_t` ve terminal reward ayni transition mantigina bagli olmali.
9. Hicbir hastanin birden fazla split'te bulunmadigini dogrula.
10. Mortality, discharge status veya gelecek outcome bilgisinin state feature'larina sizmadigini dogrula.

Ciktilar:

```text
data/replay/replay_train.parquet
data/replay/replay_validation.parquet
data/replay/replay_test.parquet
data/replay_sparse/replay_train.parquet
data/replay_sparse/replay_validation.parquet
data/replay_sparse/replay_test.parquet
data/splits/train_manifest.parquet
data/splits/validation_manifest.parquet
data/splits/test_manifest.parquet
data/splits/split_summary.json
results/iql_final/audit/presweep_audit.json
```

Minimum dogrulama:

```bash
uv run python -m pytest tests/datasets/test_build_transitions_cli.py tests/data/test_split_manifests.py tests/mdp/test_preprocessing.py tests/datasets/test_transitions.py -q
uv run python scripts/audit_iql_presweep.py
```

### Phase 2 — IQL Sweep Runner + Grid + Mock Snakefile

Amac: 18-config IQL Stage 1 ve final-6 Stage 2 akisini deterministik, test edilebilir ve mock modda hizli calisabilir hale getirmek.

Kapsam:

1. `scripts/run_iql_sweep.py` yaz.
2. 18-config grid generator ekle.
3. Her config icin gecici training YAML/manifest uret.
4. Stage 1 seed-42 runner ve manifest uretimini ekle.
5. Stage 2 extra-seed runner iskeletini ekle.
6. CUDA/GPU preflight check ekle; mock mode haricinde CPU fallback'i hata say.
7. IQL trainer ve sweep runner device ayarini acikca `cuda` kullanacak sekilde bagla.
8. Her run manifestine `device`, `gpu_name`, `cuda_version`, `torch_version` alanlarini yaz.
9. En bastan butun adimlari kapsayan `Snakefile` olustur.
10. Snakefile icin mock veya mini synthetic data modu ekle.
11. Gercek MIMIC artifactleri olmadan DAG'in yapisal olarak calistigini dogrula.
12. Uzun egitim yerine mock IQL checkpoint/evaluation ciktilari ureten hizli rule veya script ekle.

Ciktilar:

```text
scripts/run_iql_sweep.py
Snakefile
results/iql_final/stage1/grid_manifest.json
results/iql_final/stage1/runs/{config_id}/seed_42/checkpoint.pt
results/iql_final/stage1/runs/{config_id}/seed_42/train_metrics.json
results/iql_final/stage2/finalists_manifest.json
```

Minimum dogrulama:

```bash
uv run python -m pytest tests/training/test_iql_metrics.py -q
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU bulunamadi veya kullanilamiyor'
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
PY
uv run snakemake -n --cores 1
uv run snakemake --cores 1 --config mock=true
```

### Phase 3 — Real Evaluation + Final-6 Selection

Amac: Model selection'i proxy skor yerine ortak terminal survival/death evaluation reward ile calisan gercek validation FQE/OPE katmanina baglamak.

Kapsam:

1. Gercek FQE entegrasyonunu ekle.
2. FQE bootstrap entegrasyonunu ekle.
3. Common terminal survival/death utility ile model selection'i zorunlu hale getir.
4. Validation evaluator'i Stage 1 checkpointleri icin calistir.
5. Final-6 selector ekle.
6. Hard gate, composite score, safety/support score, diversity ve baseline anchor kurallarini uygula.
7. Secim gerekcesini manifest/CSV olarak yaz.

Ciktilar:

```text
results/iql_final/stage1/evaluation/{config_id}/validation_metrics.json
results/iql_final/stage1/evaluation/{config_id}/fqe_bootstrap.json
results/iql_final/stage1/selection/final6_configs.csv
results/iql_final/stage1/selection/final6_manifest.json
results/iql_final/stage1/selection/selection_rationale.md
```

Minimum dogrulama:

```bash
uv run python -m pytest tests/evaluation/test_bootstrap.py tests/evaluation/test_ope_pipeline.py tests/evaluation/test_safety_checks.py -q
uv run python scripts/evaluate_iql_sweep.py --stage 1 --mock
```

### Phase 4 — Stage 2 Confirmation + Final Report Bundle

Amac: Final-6 adaylarini 3 seed ile dogrulamak, final selected IQL checkpoint'i baseline'larla karsilastirmak ve raporlanabilir artifact bundle uretmek.

Kapsam:

1. Final 6 config'i seed 123 ve 456 ile tekrar egit; seed 42 run'ini yeniden kullan.
2. Her finalist config icin toplam 3 seedlik sonuc uret.
3. Stage 2 finalist sonuclarini ozet manifestte topla.
4. Final selected IQL checkpoint'i belirle.
5. Final test + baseline comparison script'i yaz.
6. Selected IQL policy, clinician replay, no-treatment policy ve behavior cloning baseline'larini ayni test split uzerinde karsilastir.
7. FQE/WIS/ESS/support/agreement/action heatmap karsilastirmasi uret.
8. CSV/JSON/figure/report bundle uret.
9. Snakefile `rule all` final hedeflerinin tamamlandigini dogrula.

Ciktilar:

```text
results/iql_final/stage2/finalists_manifest.json
results/iql_final/stage2/seed_summary.csv
results/iql_final/final_metrics.json
results/iql_final/final_comparison.csv
results/iql_final/final_report.md
results/iql_final/figures/fqe_vs_support.png
results/iql_final/figures/seed_variance.png
results/iql_final/figures/action_heatmap.png
results/iql_final/figures/baseline_comparison.png
results/iql_final/figures/bootstrap_ci.png
```

Minimum dogrulama:

```bash
uv run python -m pytest tests/baselines/test_baselines.py tests/evaluation/test_bootstrap.py tests/reporting/test_offline_rl_reporting.py -q
uv run snakemake -n --cores 1
uv run snakemake --cores 1 --config mock=true
```

## Phase Bagimliliklari

```text
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
```

Kurallar:

- Phase 2 gercek veri uzerinde baslamadan once Phase 1 audit gecmeli.
- Phase 3 baslamadan once Stage 1 checkpoint/output kontratlari Phase 2'de netlesmeli.
- Phase 4 baslamadan once Final-6 selector Phase 3'te manifest uretmeli.
- Snakefile mock mode Phase 2'de gecmeden gercek sweep baslatilmamali.
- Final iddia sadece Phase 4'te 3 seed ve CI ile raporlanmali.

## Kisa Sonuc

Mevcut repo IQL egitimi, temel config, reward variant altyapisi, mock evaluation artifactleri, baseline modulleri ve bootstrap testleri acisindan iyi durumda. Ancak `docs/iql_final_sweep_protocol.md` dosyasindaki final deney protokolunu tek komutla, Stage 1/Stage 2 seklinde ve gercek FQE tabanli model selection ile calistirmak icin orchestration ve evaluation katmani henuz tamamlanmamis durumda.
