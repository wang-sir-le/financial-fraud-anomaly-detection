# Reproducibility scope

## Included

- Leakage-controlled PaySim preprocessing and entity-history construction.
- PaySim rolling temporal evaluation, repeated-seed analysis, calibration analysis, prediction reconstruction, and paired moving time-block Bootstrap implementation.
- IEEE-CIS data audit, Entity A/C history construction, temporal protocol builder, frozen M0-M3 experiment, directional result analysis, and paired moving time-block Bootstrap implementation.
- Portable experiment configuration, dependency declarations, and public-code tests.
- Venue-neutral Supplement describing the frozen scientific protocols.

## Deliberately excluded

- PaySim and IEEE-CIS raw data or redistributed copies.
- Transaction-level processed features or prediction ledgers.
- Fitted model objects, checkpoints, large Bootstrap replicate files, and local caches.
- Local absolute paths, credentials, editor files, and virtual environments.
- Manuscript drafting, citation-mapping, freeze-management, issue-registry, and internal quality-assurance generators.

## Protocol boundaries

PaySim and IEEE-CIS use different capacity implementations. PaySim transfers a validation-selected threshold constrained to at most a 3% alert rate. IEEE-CIS applies an exact Test top-3% ranking rule. Calibration was systematically evaluated only on PaySim. Bootstrap intervals quantify paired uncertainty within the frozen future windows; they are not a formal test of cross-Fold temporal heterogeneity.

## Expected reconstruction order

1. Obtain datasets through their authoritative access routes.
2. Install the environment from `pyproject.toml` or `requirements.txt`.
3. Run dataset validation and strictly causal feature construction.
4. Generate frozen temporal splits.
5. Execute the fixed model comparisons and seeds.
6. Run result-integrity checks before any downstream uncertainty analysis.
7. Run the paired time-block Bootstrap from the accepted prediction ledger.

All generated artifacts should remain outside version control unless they are compact, non-sensitive, and explicitly reviewed for release.
