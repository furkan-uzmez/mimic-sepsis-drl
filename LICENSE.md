# License And Use Terms

This repository contains research software, documentation, derived data artifacts, and model release files for retrospective offline reinforcement learning on MIMIC-IV sepsis cohorts. Different parts of the repository are subject to different terms.

This file is a project-level licensing declaration. It is not legal advice. Review upstream MIMIC-IV, PhysioNet, institutional, and model release requirements before redistribution or public release.

## Component License Summary

| Component | Paths | Terms |
| --- | --- | --- |
| Source code and scripts | `src/`, `scripts/`, `configs/`, `tests/`, `Snakefile`, `pyproject.toml` | All rights reserved unless a separate open-source license is added. |
| Documentation | `README.md`, `docs/` | All rights reserved unless a separate documentation license is added. |
| Derived replay/split artifacts | `data/replay/`, `data/splits/` | Subject to MIMIC-IV and PhysioNet data use terms. Not independently relicensed here. |
| Training checkpoints and generated results | `checkpoints/`, `results/`, `runs/` | Research artifacts only. Not for clinical, commercial, or production use. |
| Hugging Face model release bundle | `hf_model_repo/mimic-sepsis-iql/` | See `hf_model_repo/mimic-sepsis-iql/LICENSE.md`. |

## Research-Only Restrictions

This project is intended for academic and retrospective research use only. It must not be used for diagnosis, treatment, triage, clinical decision support, live patient care, or automated medical decision-making.

The models, checkpoints, policies, metrics, and evaluation results have not been prospectively validated and are not evidence of clinical safety or efficacy.

## MIMIC-IV And Data Terms

This repository does not grant access to MIMIC-IV and does not include raw patient data. Use of MIMIC-IV is governed by PhysioNet credentialing, data use agreements, required training, and applicable institutional requirements.

Any derived cohort, split, replay, feature, action, reward, or evaluation artifact remains subject to those upstream data terms and must not be redistributed unless the user has verified that redistribution is permitted.

## No Implied Open-Source License

Public availability of this repository does not grant permission to reuse, modify, redistribute, sublicense, or commercially exploit the code, data artifacts, model weights, or documentation beyond rights provided by platform terms and applicable law.

If an open-source license is later added for the code or documentation, it will apply only to the components explicitly identified by that license and will not override MIMIC-IV, PhysioNet, model, clinical, privacy, or institutional restrictions.

## No Warranty

All materials are provided as-is, without warranty of any kind. Users are responsible for verifying legal, ethical, privacy, clinical, and institutional permissions before use, redistribution, or publication.
