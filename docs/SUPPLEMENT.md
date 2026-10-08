<!-- Historical August 2026 base material. -->

This document describes the original base release. For the October 8, 2026 additions, use [UPDATES_20261008.md](UPDATES_20261008.md) and the current stage READMEs. It is not an updated manuscript submission or full extension-bundle manifest.

# Supplementary Materials (venue-neutral draft)

This document assembles existing frozen protocols, configurations, audits, and tables. It reports no new analysis and does not supersede the frozen main-text evidence.

## S1. Dataset provenance and descriptive details

PaySim is a synthetic mobile-money transaction simulation introduced by Lopez-Rojas, Elmir, and Axelsson \cite{LopezRojasEtAl2016PaySim}. IEEE-CIS is the anonymized fraud-detection competition dataset released in 2019 by the IEEE Computational Intelligence Society with data provided by Vesta Corporation \cite{IEEECIS2019FraudDetection}.

| Dataset | Data setting | Transactions | Fraud prevalence | Time variable | Behavioral entity used in the primary history analysis |
| --- | --- | ---: | ---: | --- | --- |
| PaySim | synthetic mobile-money simulation | 6,362,620 | 0.129082% | simulation step (1–743) | recipient (`nameDest`) |
| IEEE-CIS | real-world-derived anonymized transactions | 590,540 | 3.499001% | relative `TransactionDT` ordering variable | Entity A (`card1`), treated only as an anonymized card-related identifier |

The datasets differ in provenance, transaction semantics, time representation, entity meaning, and operational capacity implementation. They were therefore evaluated under separate frozen protocols and used for directional external validation rather than exact replication.

## S2. Dataset-specific temporal protocols

### S2.1 PaySim

PaySim used three expanding-window rolling folds over complete simulation steps. Transactions sharing a step stayed in one split. Validation was ordered before Test in every Fold. The Validation period determined an empirical threshold under a maximum alert rate of 3%; that threshold was then transferred unchanged to Test.

| Fold | Split | Step start | Step end | Transactions | Fraud cases | Fraud rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Train | 1 | 205 | 2,538,675 | 2,294 | 0.090362% |
| 1 | Validation | 206 | 238 | 639,263 | 401 | 0.062728% |
| 1 | Test | 239 | 281 | 642,661 | 498 | 0.077490% |
| 2 | Train | 1 | 281 | 3,820,599 | 3,193 | 0.083573% |
| 2 | Validation | 282 | 323 | 642,988 | 450 | 0.069986% |
| 2 | Test | 324 | 354 | 605,510 | 312 | 0.051527% |
| 3 | Train | 1 | 354 | 5,069,097 | 3,955 | 0.078022% |
| 3 | Validation | 355 | 398 | 656,800 | 494 | 0.075213% |
| 3 | Test | 399 | 743 | 636,723 | 3,764 | 0.591152% |

### S2.2 IEEE-CIS

IEEE-CIS used expanding windows over complete `TransactionDT` groups. Each Fold satisfied `max(Train) < min(Validation)` and `max(Validation) < min(Test)`, and no timestamp group crossed a split. Evaluation reapplied an exact Test top-3% rule with `ceil(0.03 × N)` alerts; it did not use the PaySim threshold-transfer rule.

| Fold | Split | TransactionDT start | TransactionDT end | Transactions |
| --- | --- | ---: | ---: | ---: |
| 1 | Train | 86400 | 5592303 | 236,216 |
| 1 | Validation | 5592304 | 7306520 | 59,054 |
| 1 | Test | 7306535 | 8745772 | 59,054 |
| 2 | Train | 86400 | 8745772 | 354,324 |
| 2 | Validation | 8745798 | 10437996 | 59,054 |
| 2 | Test | 10438003 | 12192842 | 59,054 |
| 3 | Train | 86400 | 12192842 | 472,432 |
| 3 | Validation | 12192900 | 13990904 | 59,054 |
| 3 | Test | 13990941 | 15811131 | 59,054 |

## S3. Entity-history definitions and leakage controls

For PaySim, history was keyed by recipient (`nameDest`) over frozen 1-, 6-, and 24-step windows. The retained summaries covered prior recipient frequency, origin diversity, amount behavior and ratios, and time since the preceding recipient transaction. SQL range windows ended at one step before the current transaction. The separate (`nameOrig`, `nameDest`) Pair candidate was retained as a sensitivity result; the frozen audit found no repeated pairs in this PaySim file and did not generalize that dataset-specific degeneration beyond PaySim.

For IEEE-CIS, Entity A was `card1`; Entity C combined `card1`, `card2`, `card3`, `card5`, and `card6`. These fields are anonymized card-related attributes, not verified customer or account identifiers. Entity C was unavailable when any component was missing, with no component or global imputation during history construction. The frozen history summaries were: previous_count, previous_amount_mean, previous_amount_std, amount_to_previous_mean, time_since_last, previous_unique_product_count. Only transactions with `historical TransactionDT < current TransactionDT` contributed history. Same-timestamp transactions did not contribute to one another. Validation could use Train and strictly earlier Validation transactions; Test could use Train, Validation, and strictly earlier Test transactions. Labels were excluded from history construction.

The recorded leakage checks covered first-entity state, same-timestamp isolation, strict past-only ordering, missing-component handling, stable row order, label independence, input-order invariance, prefix invariance, and future-append invariance. All checks in the frozen IEEE-CIS leakage audit passed.

## S4. Model schemes, parameters, seeds, and preprocessing

All schemes used LightGBM gradient-boosted decision trees. The shared core configuration used 150 trees, learning rate 0.05, 31 leaves, row subsampling 0.8, and feature subsampling 0.8. IEEE-CIS additionally recorded `max_depth=-1`, `min_child_samples=20`, `subsample_freq=1`, zero L1/L2 regularization, deterministic training, and column-wise construction. The five frozen training seeds were 42, 52, 62, 72, 82.

| Dataset | Scheme | Features | Positive-class weight | Frozen role |
| --- | --- | --- | ---: | --- |
| PaySim | Raw weight-1 | Raw transaction features | 1 | Baseline |
| PaySim | Recipient history weight-1 | Raw plus recipient history | 1 | History contrast |
| PaySim | Recipient history weight-2 | Raw plus recipient history | 2 | Complete framework comparison |
| PaySim | Recipient history weight-5 | Raw plus recipient history | 5 | Weight sensitivity |
| PaySim | Recipient-plus-Pair history weight-1 | Raw plus recipient and Pair history | 1 | Pair sensitivity |
| IEEE-CIS | M0 | Raw | 1 | raw transaction baseline |
| IEEE-CIS | M1 | Raw + Entity A history | 1 | history contribution |
| IEEE-CIS | M2 | Raw + Entity A history | 2 | pre-specified primary framework |
| IEEE-CIS | M3 | Raw + Entity C history | 2 | entity-granularity sensitivity analysis |

For IEEE-CIS, M2 remained the pre-specified Primary Scheme and M3 remained the Entity Granularity Sensitivity Scheme. The frozen contrasts were C01 = M1 − M0, C12 = M2 − M1, C02 = M2 − M0, and C23 = M3 − M2. Numeric IEEE-CIS inputs were converted to float32 with infinities mapped to missing values and native LightGBM missing-value handling. Categorical vocabularies were learned from Train only, with separate missing and unseen codes; target encoding was not used. The complete feature-column inventory and configuration objects are indexed in the asset manifest rather than reproduced as hundreds of column names here.

## S5. Capacity, analytical cost, and paired Bootstrap protocols

PaySim used a Validation-determined empirical threshold constrained by an alert rate of at most 3%, then transferred the threshold unchanged to Test. IEEE-CIS instead ranked each Test set directly and flagged exactly `ceil(0.03 × N)` records, sorting raw probability descending and breaking ties by ascending `TransactionID`. These are intentionally different capacity implementations.

Both datasets used the analytical scenario `C_FP=1` and `C_FN=100`. These values were not observed monetary losses or a universal optimum. For IEEE-CIS exact fixed capacity, cost reduction and the change in captured true positives are algebraically linked by `Cost Reduction = 101 × ΔTP`.

Both Bootstrap analyses resampled paired frozen prediction records in overlapping, non-circular moving time blocks for 5,000 replicates per Fold, with no retraining, prediction generation, recalibration, or new threshold search. PaySim sampled unique Test-step positions (Fold 1: T=43, L=4, Fold 2: T=31, L=3, Fold 3: T=345, L=7) and compared Raw weight-1 with recipient-history weight-2. IEEE-CIS sampled unique `TransactionDT` groups (Fold 1: T=57052, L=38, Fold 2: T=57630, L=39, Fold 3: T=57717, L=39) for C02 (M2 − M0), sharing each sampled sequence across both models and all five seeds before reapplying exact capacity. Fold-specific uncertainty was primary; equal-Fold Overall summaries were secondary. These procedures quantify within-window uncertainty and are not tests of cross-Fold heterogeneity or uncertainty over arbitrary unseen future regimes.

## S6. PaySim calibration and sensitivity materials

Calibration was evaluated systematically only on PaySim. Uncalibrated probabilities were compared with fold/seed-specific Platt scaling and isotonic regression fitted within the frozen chronological Validation design. Reliability metrics comprised Brier score, 10-bin expected calibration error under uniform and quantile binning, and log loss. The aggregate frozen reliability table is reproduced below.

| calibration_method | test_brier_mean | test_ece_uniform_10_mean | test_ece_quantile_10_mean | test_log_loss_mean | manuscript_role |
| --- | --- | --- | --- | --- | --- |
| Isotonic | 0.0029659253597535667 | 0.002708105987341169 | 0.0014751442747769333 | 0.013250848637041901 | Supplementary candidate |
| Platt | 0.0032320274684972 | 0.0029642459531613566 | 0.0019247890381295333 | 0.015881213045028233 | Supplementary candidate |
| Uncalibrated | 0.0053034758452229665 | 0.005227069111403567 | 0.0041924290380176005 | 0.11127434260769538 | Supplementary candidate |

The frozen analysis observed improved aggregate reliability metrics for Platt and isotonic calibration relative to uncalibrated probabilities. However, Platt scaling produced no Validation or Test decision disagreements under the frozen empirical capacity procedure and no business-cost change. Reliability, ranking, and downstream decision behavior were therefore retained as separate evaluation dimensions. The auxiliary fixed probability-threshold analysis remained explicitly supplementary.

## S7. Reproducibility inventory

| Software | PaySim recorded version | IEEE-CIS recorded version |
| --- | --- | --- |
| duckdb | 1.5.5 | not recorded in this asset |
| fraudx | 0.1.0 | not recorded in this asset |
| lightgbm | 4.7.0 | 4.7.0 |
| numpy | 2.4.4 | 2.4.4 |
| pandas | 3.0.2 | 3.0.2 |
| pyarrow | not recorded in this asset | 25.0.1 |
| python | not recorded in this asset | 3.11.9 |
| scikit-learn | 1.8.0 | not recorded in this asset |
| scikit_learn | not recorded in this asset | 1.8.0 |

The principal recorded execution entrypoints are `python -m fraudx.cli --config configs/optimized_first_round.yaml --rolling` for the PaySim rolling experiment, `python build_ieee_temporal_protocol.py` for the IEEE-CIS temporal protocol, `python run_ieee_cis_experiment.py` for the frozen M0–M3 phase, `python analyze_ieee_cis_results.py` for the read-only result analysis, and `python run_ieee_cis_timeblock_bootstrap.py` for the frozen paired time-block analysis. Exact input and output hashes, configurations, protocol files, feature inventories, audit tables, prediction-ledger hashes, and stage metadata are indexed in `supplement_asset_manifest.csv`.

This inventory describes existing local artifacts and entrypoints only. It makes no commitment about public code release, repository location, or data redistribution; those statements remain pending researcher confirmation in `data_code_availability_pending_template.md`.
