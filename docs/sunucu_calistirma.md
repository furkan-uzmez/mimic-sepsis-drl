# MIMIC Sepsis Offline RL — Sunucuda Baştan Sona Çalıştırma

> Son güncelleme: 2026-05-23
> Sunucudaki durum: `data/raw/physionet.org/files/mimiciv/3.1/` var, başka hiçbir şey yok.

---

## 0. Sunucu Ön Hazırlık

```bash
# Projeyi sunucuya klonla (ya da scp/rsync ile at)
git clone <repo-url> mimic-sepsis
cd mimic-sepsis

# MIMIC raw verileri (zaten attıysan kontrol et)
ls data/raw/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz
ls data/raw/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz
ls data/raw/physionet.org/files/mimiciv/3.1/icu/chartevents.csv.gz

# Snakemake kur (yoksa)
pip install snakemake
# veya: uv pip install snakemake
```

---

## 1. Tek Komutla Hepsini Çalıştır (Snakemake)

```bash
cd /path/to/mimic-sepsis

# Önce ne çalışacak gör (dry-run)
snakemake -j1 --dry-run all

# Hepsini çalıştır (~7-8 saat)
snakemake -j1 all
```

Snakemake otomatik olarak şu sırayla çalıştırır:

| Adım | Kural | Süre | Çıktı |
|---|---|---|---|
| 1 | `cohort` | 5 dk | `data/processed/cohort/cohort.parquet` |
| 2 | `onset` | 10 dk | `data/processed/onset/onset_assignments.parquet` |
| 3 | `episodes` | 5 dk | `data/processed/episodes/episodes.parquet` |
| 4 | `splits` | 1 dk | `data/splits/*_manifest.parquet` |
| 5 | `shaped_replay` | 15 dk | `data/replay/replay_train.parquet` |
| 6 | `sparse_replay` | 15 dk | `data/replay_sparse/replay_train.parquet` |
| 7 | `stage1_sweep` | ~3h | 24 CQL run |
| 8 | `stage1_eval` | 10 dk | FQE ranking |
| 9 | `stage2_sweep` | ~3h | 24 CQL run |
| 10 | `final_eval` | 15 dk | Bootstrap CI |
| 11 | `report` | 5 dk | 8 figür + 5 tablo |
| **Toplam** | | **~7h** | |

---

## 2. Adım Adım Çalıştırma (Snakemake'siz)

Eğer Snakemake kullanmak istemezsen, elle de çalıştırabilirsin:

```bash
cd /path/to/mimic-sepsis

# === Phase 1: Cohort ===
python -m mimic_sepsis_rl.cli.build_cohort
# → data/processed/cohort/cohort.parquet

# === Phase 2: Onset ===
python -m mimic_sepsis_rl.data.onset
# → data/processed/onset/onset_assignments.parquet

# === Phase 3: Episode Grid ===
python -m mimic_sepsis_rl.cli.build_episode_grid
# → data/processed/episodes/episodes.parquet
# → data/processed/episodes/episode_steps.parquet

# === Phase 4: Splits ===
python -m mimic_sepsis_rl.data.splits --config configs/splits/default.yaml
# → data/splits/train_manifest.parquet
# → data/splits/validation_manifest.parquet
# → data/splits/test_manifest.parquet

# === Phase 5-6: Shaped Replay ===
python -m mimic_sepsis_rl.cli.build_transitions --reward-variant sofa_shaped
# → data/replay/replay_train.parquet
# → data/replay/replay_validation.parquet
# → data/replay/replay_test.parquet

# === Phase 5-6: Sparse Replay ===
python -m mimic_sepsis_rl.cli.build_transitions \
    --reward-variant sparse \
    --output-dir data/replay_sparse/
# → data/replay_sparse/replay_train.parquet
# → data/replay_sparse/replay_validation.parquet
# → data/replay_sparse/replay_test.parquet

# === Phase 10: Stage 1 Sweep (24 run, ~3h) ===
python scripts/run_cql_sweep.py --stage 1

# === Phase 10: Stage 1 Evaluation ===
python scripts/evaluate_cql_sweep.py --stage 1
# → runs/cql_sweep/stage1_evaluation.json (top 6 configs)

# === Phase 10: Stage 2 Sweep (24 run, ~3h) ===
python scripts/run_cql_sweep.py \
    --stage 2 \
    --stage1-eval runs/cql_sweep/stage1_evaluation.json

# === Phase 10: Final Evaluation ===
python scripts/evaluate_cql_sweep.py --stage final
# → runs/cql_sweep/evaluation_summary.json

# === Phase 10: Report ===
python scripts/generate_report_figures.py
# → docs/assets/report/fig1..8.png
# → docs/assets/report/table1..5.csv
# → docs/cql_project_report.md
```

---

## 3. Snakemake Parsiyel Çalıştırma

```bash
# Sadece preprocessing (Phase 1-6)
snakemake -j4 shaped_replay sparse_replay

# Sadece Stage 1 sweep + eval
snakemake -j1 stage1_eval

# Sadece report (öncesinde her şey varsa)
snakemake -j1 report

# Tek bir rule'u zorla yeniden çalıştır
snakemake -j1 --force cohort

# Tüm sweep çıktılarını sil (disk boşalt)
snakemake -j1 clean_sweep
```

---

## 4. Sunucudan Lokale Dosya Transferi

Sunucuda her şey bittikten sonra, lokale şu klasörleri çek:

```bash
# Lokal makinede:
cd /Users/enesdemir/Documents/mimic-sepsis

# 1. Sparse replay dataset (lokalde yok)
rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/data/replay_sparse/ \
    data/replay_sparse/

# 2. Phase 10 sweep çıktıları (evaluation sonuçları + manifestler)
rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/runs/cql_sweep/ \
    runs/cql_sweep/

# 3. CQL checkpoint'leri (raporun ana verisi)
rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/checkpoints/cql_sweep/ \
    checkpoints/cql_sweep/

# 4. Figür ve tablolar (raporun kendisi)
rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/docs/assets/report/ \
    docs/assets/report/

# 5. Draft rapor
rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/docs/cql_project_report.md \
    docs/cql_project_report.md

# 6. Processed intermediate'lar (opsiyonel, sadece audit/tekrar lazımsa)
rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/data/processed/ \
    data/processed/

rsync -avz --progress \
    sunucu:/path/to/mimic-sepsis/data/splits/ \
    data/splits/
```

**Minimum indirilmesi gerekenler (rapor için yeterli):**

| Klasör/Dosya | Neden |
|---|---|
| `data/replay_sparse/` | Sparse replay dataset |
| `runs/cql_sweep/` | Evaluation sonuçları, manifestler |
| `checkpoints/cql_sweep/` | Eğitilmiş CQL modelleri |
| `docs/assets/report/` | Figür ve tablolar |
| `docs/cql_project_report.md` | Draft rapor |

**Opsiyonel (tekrar audit/çalıştırma gerekirse):**

| Klasör/Dosya | Neden |
|---|---|
| `data/processed/` | Preprocessing artifact'ları |
| `data/splits/` | Split manifestleri |
| `data/replay/` | Shaped replay (lokalde Furkan'dan var zaten) |

---

## 5. Lokale İndirdikten Sonra

```bash
cd /Users/enesdemir/Documents/mimic-sepsis

# Dosyaların geldiğini kontrol et
ls data/replay_sparse/replay_train.parquet
ls runs/cql_sweep/evaluation_summary.json
ls docs/assets/report/fig*.png | wc -l   # → 8
ls docs/assets/report/table*.csv | wc -l  # → 5

# Gerekirse lokaldede figürleri re-generate et (transfer edilen evaluation JSON'larla)
python scripts/generate_report_figures.py

# Raporu aç
open docs/cql_project_report.md
```

---

## 6. Sorun Giderme

### `ModuleNotFoundError: No module named 'polars'` (veya benzeri)
```bash
uv sync                    # proje dependency'lerini kur
# veya
uv pip install -e ".[dev]"
```

### `FileNotFoundError: data/processed/cohort/cohort.parquet`
Önceki adım fail olmuş. Snakemake kullanıyorsan otomatik halleder. Elle çalıştırıyorsan sırayı takip et.

### GPU OOM (CUDA out of memory)
```bash
# CPU'da çalıştır (daha yavaş ama garantili)
export MIMIC_RL_DEVICE=cpu
snakemake -j1 all
```

### Stage 1/2 eğitim çok uzun sürüyorsa
```bash
# Sadece hızlı test için limit-stays ile az hasta kullan:
python -m mimic_sepsis_rl.cli.build_transitions \
    --reward-variant sofa_shaped \
    --limit-stays 500
```

### Disk alanı yetmiyorsa
```bash
# Her CQL checkpoint ~1 GB. 48 run × 10 checkpoint = ~500 GB olabilir.
# keep_last_n=0 tüm checkpointleri saklıyor.
# Disk azsa, sweep scriptinde keep_last_n=3 yap:

# scripts/run_cql_sweep.py içinde:
# cfg["checkpoint"]["keep_last_n"] = 3  # 0 yerine
```
