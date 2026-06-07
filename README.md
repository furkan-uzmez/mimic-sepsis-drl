# MIMIC Sepsis IQL Offline RL

MIMIC-IV uzerinde Sepsis-3 tabanli yogun bakim kohortunu 4 saatlik MDP adimlarina donusturen ve Implicit Q-Learning (IQL) ile retrospektif offline RL politika ogrenimi yapan arastirma kod tabanidir.

Bu repo klinik karar destek sistemi degildir. Amaci hasta verisiyle canli etkilesim kurmadan, kayitli klinisyen kararlarindan veri sizintisina karsi korumali ve yeniden uretilebilir bir IQL benchmark akisi saglamaktir.

## Icerik

- [Ozellikler](#ozellikler)
- [Gereksinimler](#gereksinimler)
- [Veri ve Guvenlik Notlari](#veri-ve-guvenlik-notlari)
- [Kurulum](#kurulum)
- [Hizli Baslangic](#hizli-baslangic)
- [Tam Veri Pipeline'i](#tam-veri-pipelinei)
- [IQL Egitimi](#iql-egitimi)
- [IQL Sonuclari](#iql-sonuclari)
- [Dokumantasyon](#dokumantasyon)
- [Sorun Giderme](#sorun-giderme)
- [Gelistirme](#gelistirme)
- [Atiflar](#atiflar)

## Ozellikler

- Sepsis-3 kriterlerine gore yetiskin ICU hasta kohortu uretimi.
- Onset -24 saat ile +48 saat araliginda 4 saatlik episode grid olusturma.
- `state_dim = 62` klinik ozellik ve 25 ayrik tedavi aksiyonu.
- Aksiyonlarin 5 vazopressor bin'i x 5 IV fluid bin'i olarak yorumlanabilir sekilde kodlanmasi.
- Hasta seviyesinde train/validation/test ayrimi ile leakage-safe preprocessing.
- IQL icin value, critic ve actor egitimi; expectile regression ve advantage-weighted actor update akisi.
- Sparse ve SOFA-shaped reward varyantlariyla IQL sweep, finalist secimi ve OPE/safety degerlendirmesi.
- `uv`, Hydra, MLflow, Polars, PyArrow, scikit-learn, PyTorch ve d3rlpy tabanli tekrarlanabilir calisma ortami.

## Gereksinimler

| Gereksinim | Surum / Not |
| --- | --- |
| Python | `>=3.12` |
| Paket yoneticisi | `uv` |
| Ham veri | MIMIC-IV v3.1 dosyalari |
| Egitim runtime'i | PyTorch `2.6.0`, varsayilan index CUDA 12.4 wheel'leri |
| Opsiyonel araclar | `snakemake`, `pytest`, `ruff`, `jupyter` dev grubunda |

Bagimlilikler `pyproject.toml` ve `uv.lock` ile sabitlenir. Kurulumda manuel paket listesi yerine `uv sync` kullanin.

## Veri ve Guvenlik Notlari

- Bu repo hasta verisi barindirmaz.
- MIMIC-IV kullanimi icin PhysioNet uzerinden yetkili erisim ve gerekli CITI egitimi gerekir.
- Ham veri su dizinde beklenir:

```text
data/raw/physionet.org/files/mimiciv/3.1
```

- `data/raw/` icerigini immutably kabul edin; ham dosyalari kodla degistirmeyin.
- Uretilen artifaktlar `data/processed/`, `data/splits/`, `data/replay/`, `results/iql_final/`, `runs/` ve `checkpoints/` altinda tutulur.
- IQL politikasi yalnizca retrospektif arastirma ve benchmark amaciyla yorumlanmalidir.

## Kurulum

```bash
git clone <repo-url>
cd mimic-sepsis-drl
uv sync
```

Kurulumu dogrulamak icin:

```bash
uv run python -m mimic_sepsis_rl.training.device --self-check
```

Basari sinyali: Python ortami acilir, PyTorch cihazi raporlanir ve self-check hata vermeden tamamlanir.

## Hizli Baslangic

Ham veri ve daha once uretilmis kohort/onset/episode/split dosyalari hazirsa IQL icin minimum akisi calistirin:

```bash
uv run python -m mimic_sepsis_rl.cli.build_transitions
uv run python -m mimic_sepsis_rl.training.device --self-check
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql --describe
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql --dry-run
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql
```

Beklenen basari sinyali: replay dosyalari okunur, IQL config'i raporlanir, dry-run hata vermez ve egitim artifaktlari IQL run dizinlerine yazilir.

## Tam Veri Pipeline'i

Asagidaki sira, ham MIMIC-IV dosyalarindan IQL replay veri setine kadar yeniden uretilebilir veri akisini verir.

### 1. Kohortu uret

```bash
uv run python -m mimic_sepsis_rl.cli.build_cohort \
  --config configs/cohort/default.yaml \
  --emit-audit
```

Beklenen ciktilar:

- `data/processed/cohort/cohort.parquet`
- `data/processed/cohort/excluded.parquet`
- `data/processed/cohort/audit.json`

### 2. Sepsis onset uret

```bash
uv run python -m mimic_sepsis_rl.data.onset \
  --config configs/onset/default.yaml
```

Beklenen ciktilar:

- `data/processed/onset/onset_assignments.parquet`
- `data/processed/onset/onset_candidates.parquet`
- `data/processed/onset/unusable_episodes.parquet`
- `data/processed/onset/onset_audit.json`

### 3. Episode grid uret

```bash
uv run python -m mimic_sepsis_rl.cli.build_episode_grid
```

Beklenen ciktilar:

- `data/processed/episodes/episodes.parquet`
- `data/processed/episodes/episode_steps.parquet`
- `data/processed/episodes/grid_audit.json`

### 4. Train / validation / test split uret

```bash
uv run python -m mimic_sepsis_rl.data.splits \
  --config configs/splits/default.yaml \
  --source-episode-set data/processed/episodes/episodes.parquet
```

Beklenen ciktilar:

- `data/splits/train_manifest.parquet`
- `data/splits/validation_manifest.parquet`
- `data/splits/test_manifest.parquet`
- `data/splits/split_summary.json`

### 5. State / action / reward / replay dataset uret

```bash
uv run python -m mimic_sepsis_rl.cli.build_transitions
```

Beklenen ana ciktilar:

- `data/processed/features/state_vectors/state_table_raw.parquet`
- `data/processed/features/state_vectors/state_table_normalized.parquet`
- `data/processed/features/train_medians.json`
- `data/processed/features/state_vectors/preprocessing_artifacts.json`
- `data/processed/actions/action_bins.json`
- `data/processed/actions/step_actions.parquet`
- `data/processed/rewards/reward_config.json`
- `data/processed/rewards/step_rewards.parquet`
- `data/replay/replay_train.parquet`
- `data/replay/replay_train_meta.json`
- `data/replay/replay_validation.parquet`
- `data/replay/replay_validation_meta.json`
- `data/replay/replay_test.parquet`
- `data/replay/replay_test_meta.json`

## IQL Egitimi

IQL egitiminden once hedef konfigurasyonu inceleyin:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm iql \
  --describe
```

Dosya yollarini ve konfigurasyonu yan etkisiz sekilde dogrulayin:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm iql \
  --dry-run
```

Egitimi baslatin:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm iql
```

IQL degerlendirmesinde tek loss degeriyle karar vermeyin. Model secimi FQE/WIS, ESS, support mass, clinician agreement, low-support rate ve safety flag'leri birlikte yorumlanarak yapilmalidir.

## IQL Sonuclari

| Cikti | Link | Not |
| --- | --- | --- |
| Final rapor | [results/iql_final/final_report.md](results/iql_final/final_report.md) | Stage 2 secilen checkpoint ve baseline karsilastirmasi |
| Final metrikler | [results/iql_final/final_metrics.json](results/iql_final/final_metrics.json) | Secilen IQL run ozeti |
| Final karsilastirma | [results/iql_final/final_comparison.csv](results/iql_final/final_comparison.csv) | Baseline ve selected IQL tablo verisi |
| Pre-sweep audit | [results/iql_final/audit/presweep_audit.json](results/iql_final/audit/presweep_audit.json) | Veri sizintisi ve pipeline audit sonucu |
| Stage 1 manifest | [results/iql_final/stage1/stage1_manifest.json](results/iql_final/stage1/stage1_manifest.json) | Ilk sweep manifesti |
| Stage 2 summary | [results/iql_final/stage2/stage2_summary.json](results/iql_final/stage2/stage2_summary.json) | Tekrarlanan seed finalist ozeti |
| Secim gerekcesi | [results/iql_final/stage1/selection/selection_rationale.md](results/iql_final/stage1/selection/selection_rationale.md) | Finalist secim notlari |
| Grafik katalogu | [docs/iql_graphics_catalog.md](docs/iql_graphics_catalog.md) | IQL grafiklerinin ne anlattigi |

Final Stage 2 raporundaki secili konfigurasyon: `iql_sofa_shaped_conservative_safe`. Raporlanan metrikler FQE 2.848, WIS 8.203, WIS 95% CI 4.963-10.817, ESS 29.4 ve support mass 0.991 seklindedir.

Ana gorseller:

- [results/iql_final/figures/fqe_vs_support.png](results/iql_final/figures/fqe_vs_support.png)
- [results/iql_final/figures/seed_variance.png](results/iql_final/figures/seed_variance.png)
- [results/iql_final/figures/action_heatmap.png](results/iql_final/figures/action_heatmap.png)
- [results/iql_final/figures/baseline_comparison.png](results/iql_final/figures/baseline_comparison.png)
- [results/iql_final/figures/bootstrap_ci.png](results/iql_final/figures/bootstrap_ci.png)

## Dokumantasyon

| Kategori | Dokuman | Link |
| --- | --- | --- |
| IQL protocol | Final hyperparameter sweep protocol | [docs/iql_final_sweep_protocol.md](docs/iql_final_sweep_protocol.md) |
| IQL graphics | Grafik katalogu | [docs/iql_graphics_catalog.md](docs/iql_graphics_catalog.md) |
| IQL proposal | Proje onerisi | [docs/proje_onerisi_iql.md](docs/proje_onerisi_iql.md) |
| Cohort | Cohort selection rules | [docs/cohort_selection.md](docs/cohort_selection.md) |
| Features | Feature dictionary | [docs/feature_dictionary.md](docs/feature_dictionary.md) |
| Actions | Action mapping and discretization | [docs/action_mapping.md](docs/action_mapping.md) |
| Rewards | Reward specification | [docs/reward_spec.md](docs/reward_spec.md) |
| Training | Pipeline and RL positioning | [docs/pipeline_rl_positioning.md](docs/pipeline_rl_positioning.md) |
| Evaluation | Evaluation protocol | [docs/evaluation_protocol.md](docs/evaluation_protocol.md) |
| Reproducibility | Reproducibility guide | [docs/reproducibility.md](docs/reproducibility.md) |
| Safety | Leakage boundaries | [docs/leakage_boundaries.md](docs/leakage_boundaries.md) |

## Sorun Giderme

| Belirti | Olasi neden | Cozum |
| --- | --- | --- |
| `data/raw/physionet.org/files/mimiciv/3.1` bulunamiyor | Ham MIMIC-IV dosyalari indirilmemis veya farkli yerde | PhysioNet erisiminizi dogrulayin ve dosyalari beklenen dizine yerlestirin. |
| IQL dry-run replay dosyasi bulamiyor | `build_transitions` calismadi veya ara pipeline eksik | `Tam Veri Pipeline'i` bolumundeki sirayi takip edin. |
| IQL metrikleri tutarsiz gorunuyor | Farkli reward, seed, split veya preprocessing kullanildi | `results/iql_final/audit/presweep_audit.json` ve `docs/iql_final_sweep_protocol.md` dosyalarini kontrol edin. |
| Yuksek FQE ama dusuk support | Politika veri destegi zayif aksiyonlara kayiyor olabilir | FQE'yi ESS, support mass, low-support rate ve clinician agreement ile birlikte yorumlayin. |
| PyTorch cihaz hatasi | CUDA/MPS ortam uyumsuzlugu veya yanlis wheel | `uv run python -m mimic_sepsis_rl.training.device --self-check` komutuyla runtime'i dogrulayin. |

## Gelistirme

Gelistirme bagimliliklarini kurmak icin:

```bash
uv sync --group dev
```

Testleri calistirin:

```bash
uv run pytest
```

Kod kalitesi kontrolu:

```bash
uv run ruff check .
```

Pipeline otomasyonu icin `Snakefile` ve `scripts/` dizinini inceleyin.

## Atiflar

MIMIC-IV veri seti ile uretilen calismalarda asagidaki kaynaklari referans verin.

**MIMIC-IV Dataset**

> Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

**MIMIC-IV Publication**

> Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

**PhysioNet Standard Citation**

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215-e220. RRID:SCR_007345.
