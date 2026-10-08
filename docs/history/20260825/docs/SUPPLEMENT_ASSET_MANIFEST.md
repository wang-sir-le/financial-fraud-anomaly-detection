# Supplement asset manifest

This repository-facing map distinguishes material included directly in the public package from outputs that must be regenerated locally. It does not claim that excluded transaction-level or large derived artifacts are distributed.

| Supplement section | Publicly included material | Locally generated or non-redistributed material |
| --- | --- | --- |
| S1 Dataset provenance | `DATA_ACCESS.md`, `dataset_provenance.bib`, descriptive material in `SUPPLEMENT.md` | Raw PaySim and IEEE-CIS files |
| S2 Temporal protocols | `src/fraudx/robustness.py`, `src/fraudx/ieee_cis_temporal_protocol.py`, portable configuration | Generated Fold profiles and temporal-protocol JSON/CSV outputs |
| S3 Entity history and leakage controls | `src/fraudx/features.py`, `src/fraudx/ieee_cis_history.py`, corresponding tests | Processed feature Parquet files and detailed run audits |
| S4 Models, parameters, seeds, preprocessing | `configs/paysim_frozen.yaml`, `src/fraudx/models.py`, `src/fraudx/ieee_cis_experiment.py` | Generated feature inventories, run metadata, and checkpoints |
| S5 Capacity, cost, Bootstrap | `src/fraudx/metrics.py`, `src/fraudx/timeblock_bootstrap.py`, `src/fraudx/ieee_cis_timeblock_bootstrap.py` | Prediction ledgers and large Bootstrap replicate files |
| S6 PaySim calibration and sensitivity | `src/fraudx/calibration.py`, `src/fraudx/calibration_metrics.py`, the frozen summary in `SUPPLEMENT.md` | Calibration point tables and generated figures |
| S7 Reproducibility inventory | `pyproject.toml`, `requirements.txt`, `PUBLIC_RELEASE_MANIFEST.csv`, `PUBLIC_RELEASE_AUDIT.md` | Local execution logs and internal manuscript-freeze audit records |

Generated outputs are excluded because they can be large, may contain transaction-level information, or belong to internal submission quality assurance. Their absence does not change the protocol definitions in the Supplement.
