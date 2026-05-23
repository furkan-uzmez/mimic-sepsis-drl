# Phase 10 — Çalıştırma Sırası

> Son güncelleme: 2026-05-23
> Protokol: Tang & Wiens 2021 two-stage offline hyperparameter selection
> Tahmini toplam süre: ~6.5 saat (48 CQL run + evaluation)

---

## Öncesi: Kontrol

```bash
cd /Users/enesdemir/Documents/mimic-sepsis

# Shaped replay dataset (Furkan'dan hazır)
ls -lh data/replay/replay_train.parquet

# Kod modülleri
ls src/mimic_sepsis_rl/evaluation/bootstrap.py
ls scripts/run_cql_sweep.py
ls scripts/evaluate_cql_sweep.py
ls scripts/generate_report_figures.py
```

---

## Adım 1: Sparse Replay Dataset Oluşturma

~15 dakika.

```bash
uv run python -m mimic_sepsis_rl.cli.build_transitions \
    --reward-variant sparse \
    --output-dir data/replay_sparse/
```

**Kontrol:** `ls -lh data/replay_sparse/replay_train.parquet`

---

## Adım 2: Stage 1 — Broad Screen

~3 saat. 1 seed × 2 reward × 3 lr × 4 alpha = 24 CQL run'ı.

```bash
uv run python scripts/run_cql_sweep.py --stage 1
```

Çalışanlar:
- Clinician, no-treatment, behavior cloning baselines
- 24 CQL config: tüm (reward, lr, alpha) kombinasyonları, seed=42
- Her run 200 epoch eğitir, checkpoint'leri epoch 100/140/160/180/200'de kaydeder

**Kontrol:** `cat runs/cql_sweep/stage1_manifest.json | python3 -m json.tool | head`

---

## Adım 3: Stage 1 Evaluation (FQE ranking)

~10 dakika. 24 checkpoint'i **ortak terminal survival utility** FQE ile değerlendir.
En iyi 6 config'i seç. Best checkpoint epoch 200 değil, **en yüksek validation FQE'ye göre** seçilir.

```bash
uv run python scripts/evaluate_cql_sweep.py --stage 1
```

**Çıktı:** `runs/cql_sweep/stage1_evaluation.json` — top_configs + rankings.

---

## Adım 4: Stage 2 — Multi-Seed Confirmation

~3 saat. En iyi 6 config × 4 ek seed (123, 456, 789, 1024) = 24 run.

```bash
uv run python scripts/run_cql_sweep.py \
    --stage 2 \
    --stage1-eval runs/cql_sweep/stage1_evaluation.json
```

**Kontrol:** `cat runs/cql_sweep/stage2_manifest.json | python3 -m json.tool | head`

---

## Adım 5: Final Evaluation + Bootstrap CI

~15 dakika. Tüm checkpoint'ler (24 stage1 + 24 stage2 = 48) ortak terminal utility FQE ile değerlendirilir.
Her config için bootstrap CI hesaplanır.

```bash
uv run python scripts/evaluate_cql_sweep.py
```

**Kontrol:** `cat runs/cql_sweep/evaluation_summary.json | python3 -m json.tool | head`

---

## Adım 6: Figür ve Tablo Üretimi

~5 dakika. 8 figür + 3 tablo.

```bash
uv run python scripts/generate_report_figures.py
```

**Kontrol:**
```bash
ls docs/assets/report/fig*.png | wc -l   # → 8
ls docs/assets/report/table*.csv | wc -l  # → 3
```

---

## Sonrası: Rapor Yazımı

- `docs/assets/report/` → 8 figür, 3 tablo
- `docs/cql_project_report.md` → Draft rapor
- `docs/rapor_kontrol_listesi.md` → Hocanın maddeleri + metodolojik notlar

---

## Özet: Run Sayıları

| Aşama | Run | Süre |
|---|---|---|
| Sparse dataset build | 0 | ~15 dk |
| Stage 1 training | 24 | ~3h |
| Stage 1 evaluation | 0 | ~10 dk |
| Stage 2 training | 24 | ~3h |
| Final evaluation | 0 | ~15 dk |
| Figures | 0 | ~5 dk |
| **Toplam** | **48** | **~6.5h** |

---

## Parametre Grid'i

| Parametre | Değerler |
|---|---|
| Reward variant | shaped, sparse |
| Learning rate | 1e-4, 3e-4, 1e-3 |
| CQL alpha | 0.05, 0.1, 0.5, 1.0 |
| Seeds (Stage 1) | 42 |
| Seeds (Stage 2) | 123, 456, 789, 1024 |
| **Sabitler** | γ=0.99, hidden=[256,256], batch=256, epochs=200 |
