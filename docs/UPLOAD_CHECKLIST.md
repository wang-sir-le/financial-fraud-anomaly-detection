<!-- Historical August 2026 base material. -->

This document describes the original base release. For the October 8, 2026 additions, use [UPDATES_20261008.md](UPDATES_20261008.md) and the current stage READMEs. It is not an updated manuscript submission or full extension-bundle manifest.

# Public upload checklist

## Already organized in this candidate

- [x] Portable README without local absolute paths.
- [x] Core PaySim and IEEE-CIS research code.
- [x] Main scientific entrypoints and public-code tests.
- [x] Portable PaySim frozen configuration.
- [x] Python dependency declarations.
- [x] Dataset provenance and non-redistribution guidance.
- [x] Venue-neutral Supplement.
- [x] Git exclusions for datasets, predictions, models, caches, and generated outputs.
- [x] Internal manuscript-freeze and submission-audit code excluded.

## Owner decisions or checks required before pushing

- [ ] Confirm the target GitHub repository visibility is `Public`.
- [x] Add the MIT License.
- [x] Use `wang-sir-le` as the temporary public author identifier.
- [ ] Replace the temporary author identifier if a different publication name is preferred.
- [ ] Review the repository history for previously committed datasets, credentials, or large derived artifacts; `.gitignore` does not remove history.
- [ ] Run the validation commands recorded in `PUBLIC_RELEASE_AUDIT.md`.
- [ ] Decide whether compact aggregate result tables should be added in a later release.
- [ ] Tag a release only after the uploaded tree matches the audited manifest.

## Do not upload

- Raw PaySim or IEEE-CIS files.
- Processed transaction-level Parquet/CSV files.
- Prediction ledgers, fitted models, checkpoints, or Bootstrap replicate files.
- Local `.env` files, tokens, credentials, virtual environments, or absolute-path configuration.
- Internal manuscript-freeze scripts and issue/hash audit generators unless separately justified.
