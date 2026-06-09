# License

This repository uses a component-based license model because it combines source code, documentation, MIMIC-IV-derived artifacts, experiment outputs, and model release files.

This file is a project-level licensing declaration. It is not legal advice. Review upstream MIMIC-IV, PhysioNet, institutional, privacy, and model release requirements before redistribution or public release.

## Component License Summary

| Component | Paths | License / Terms |
| --- | --- | --- |
| Source code and scripts | `src/`, `scripts/`, `configs/`, `tests/`, `Snakefile`, `pyproject.toml` | MIT License |
| Documentation | `README.md`, `README.tr.md`, `docs/` | MIT License |
| Derived replay/split/data artifacts | `data/replay/`, `data/splits/`, `data/processed/` | Subject to MIMIC-IV and PhysioNet data use terms. Not independently relicensed here. |
| Raw MIMIC-IV data | `data/raw/` | Not included in this repository. Governed by PhysioNet/MIMIC-IV terms. |

## MIT License For Code And Documentation

Copyright (c) 2026 Furkan Nezih Üzmez (furkan-uzmez)

Permission is hereby granted, free of charge, to any person obtaining a copy
of the source code and documentation files covered by this license (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Research And Clinical Use Notice

This project is intended for academic and retrospective research use. It is not a clinical decision support system and must not be used for diagnosis, treatment, triage, live patient care, or automated medical decision-making.

The models, checkpoints, policies, metrics, and evaluation results have not been prospectively validated and are not evidence of clinical safety or efficacy.

## MIMIC-IV And Data Terms

This repository does not grant access to MIMIC-IV and does not include raw patient data. Use of MIMIC-IV is governed by PhysioNet credentialing, data use agreements, required training, and applicable institutional requirements.

Any derived cohort, split, replay, feature, action, reward, or evaluation artifact remains subject to those upstream data terms and must not be redistributed unless the user has verified that redistribution is permitted.

## No Warranty For Non-Code Artifacts

All non-code research artifacts are provided as-is for retrospective research context only, without warranty of any kind. Users are responsible for verifying legal, ethical, privacy, clinical, and institutional permissions before use, redistribution, or publication.
