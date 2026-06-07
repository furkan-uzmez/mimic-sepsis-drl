# MIMIC Sepsis Offline RL

MIMIC-IV uzerinde Sepsis-3 tabanli yogun bakim kohortunu 4 saatlik MDP adimlarina donusturen ve CQL, BCQ, IQL gibi cevrimdisi pekistirmeli ogrenme algoritmalarini veri sizintisina karsi korumali sekilde karsilastiran arastirma kod tabanidir.

Bu repo klinik karar destek sistemi degildir. Amaci retrospektif veride yeniden uretilebilir veri hazirlama, model egitimi ve Offline Policy Evaluation (OPE) denemeleri icin denetlenebilir bir benchmark ortami saglamaktir.

## Icerik

- [Ozellikler](#ozellikler)
- [Gereksinimler](#gereksinimler)
- [Veri ve Guvenlik Notlari](#veri-ve-guvenlik-notlari)
- [Kurulum](#kurulum)
- [Hizli Baslangic](#hizli-baslangic)
- [Tam Pipeline](#tam-pipeline)
- [Egitim ve Deneyler](#egitim-ve-deneyler)
- [Ciktilar ve Raporlar](#ciktilar-ve-raporlar)
- [Dokumantasyon](#dokumantasyon)
- [Sorun Giderme](#sorun-giderme)
- [Gelistirme](#gelistirme)
- [Atiflar](#atiflar)

## Ozellikler

- Sepsis-3 kriterlerine gore yetiskin ICU hasta kohortu uretimi.
- Onset -24 saat ile +48 saat araliginda 4 saatlik episode grid olusturma.
- Surekli hasta durum vektorleri, IV fluid ve vazopressor dozlarindan turetilen 25 ayrik eylem.
- Hasta seviyesinde train/validation/test ayrimi ile scaling ve imputation hesaplarini egitim setiyle sinirlama.
- CQL, BCQ ve IQL icin ayni replay veri seti uzerinde karsilastirilabilir egitim akisi.
- `uv`, Hydra, MLflow, Polars, PyArrow, scikit-learn, PyTorch ve d3rlpy tabanli tekrarlanabilir calisma ortami.
- CUDA odakli PyTorch kurulumu ve runtime self-check komutu.

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
- Uretilen artifaktlar `data/processed/`, `data/splits/`, `data/replay/`, `runs/`, `checkpoints/` ve `results/` altinda tutulur.
- Klinik kullanim icin degil; yalnizca retrospektif arastirma ve benchmark amaclidir.

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

Ham veri ve daha once uretilmis kohort/onset/episode/split dosyalari hazirsa minimum akisi calistirin:

```bash
uv run python -m mimic_sepsis_rl.cli.build_transitions
uv run python -m mimic_sepsis_rl.training.device --self-check
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql --describe
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql --dry-run
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql
```

BCQ veya IQL icin son uc komutta `--algorithm cql` yerine `--algorithm bcq` ya da `--algorithm iql` kullanin.

## Tam Pipeline

Asagidaki sira, ham MIMIC-IV dosyalarindan replay veri setine ve model egitimine kadar yeniden uretilebilir calisma akisini verir.

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

## Egitim ve Deneyler

Once hedef algoritmanin deney konfigurasyonunu inceleyin:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm cql \
  --describe
```

Ardindan dosya yollarini ve konfigurasyonu yan etkisiz sekilde dogrulayin:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner \
  --algorithm cql \
  --dry-run
```

Egitimi baslatin:

```bash
uv run python -m mimic_sepsis_rl.training.experiment_runner --algorithm cql
```

Desteklenen algoritmalar:

| Algoritma | Komut degeri | Not |
| --- | --- | --- |
| Conservative Q-Learning | `cql` | Referans rapor hazir |
| Batch-Constrained Q-learning | `bcq` | Ayni replay ciktilarini kullanir |
| Implicit Q-Learning | `iql` | Ayni deney runner akisini kullanir |

## Ciktilar ve Raporlar

| Run | Report | Metrics | Final manifest | Visuals |
| --- | --- | --- | --- | --- |
| CQL latest run | [docs/cql_run_report.md](docs/cql_run_report.md) | [runs/cql/cql_reference_metrics.jsonl](runs/cql/cql_reference_metrics.jsonl) | [checkpoints/cql/cql_epoch0200_step0110000_manifest.json](checkpoints/cql/cql_epoch0200_step0110000_manifest.json) | [docs/assets/cql-run](docs/assets/cql-run) |

Sonuclarin hangi scriptlerden geldigini takip etmek icin `docs/reproducibility.md` ve ilgili training dokumanlarini kullanin.

## Dokumantasyon

| Kategori | Dokuman | Link |
| --- | --- | --- |
| Cohort | Cohort selection rules | [docs/cohort_selection.md](docs/cohort_selection.md) |
| Features | Feature dictionary | [docs/feature_dictionary.md](docs/feature_dictionary.md) |
| Actions | Action mapping and discretization | [docs/action_mapping.md](docs/action_mapping.md) |
| Rewards | Reward specification | [docs/reward_spec.md](docs/reward_spec.md) |
| Training | CQL training reference | [docs/cql_training.md](docs/cql_training.md) |
| Training | Pipeline and RL positioning | [docs/pipeline_rl_positioning.md](docs/pipeline_rl_positioning.md) |
| Benchmarks | Baseline benchmarks | [docs/baseline_benchmarks.md](docs/baseline_benchmarks.md) |
| Evaluation | Evaluation protocol | [docs/evaluation_protocol.md](docs/evaluation_protocol.md) |
| Comparison | Model comparison envelope | [docs/model_comparison.md](docs/model_comparison.md) |
| Reproducibility | Reproducibility guide | [docs/reproducibility.md](docs/reproducibility.md) |
| Safety | Leakage boundaries | [docs/leakage_boundaries.md](docs/leakage_boundaries.md) |

## Sorun Giderme

| Belirti | Olasi neden | Cozum |
| --- | --- | --- |
| `data/raw/physionet.org/files/mimiciv/3.1` bulunamiyor | Ham MIMIC-IV dosyalari indirilmemis veya farkli yerde | PhysioNet erisiminizi dogrulayin ve dosyalari beklenen dizine yerlestirin. |
| Pipeline ara dosya bulamiyor | Onceki pipeline adimi calismadi veya farkli cikti dizini kullanildi | `Tam Pipeline` bolumundeki sirayi bastan takip edin. |
| Split veya preprocessing sonuclari tekrarlanamiyor | Farkli config, seed veya lockfile kullanildi | `configs/` dosyalarini, `uv.lock` dosyasini ve `docs/reproducibility.md` notlarini kontrol edin. |
| PyTorch cihaz hatasi | CUDA/MPS ortam uyumsuzlugu veya yanlis wheel | `uv run python -m mimic_sepsis_rl.training.device --self-check` komutuyla runtime'i dogrulayin. |
| MIMIC-IV erisim hatasi | PhysioNet yetkisi, CITI sertifikasi veya veri yolu eksik | Yetki durumunu PhysioNet'te kontrol edin; repo hasta verisi saglamaz. |

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
