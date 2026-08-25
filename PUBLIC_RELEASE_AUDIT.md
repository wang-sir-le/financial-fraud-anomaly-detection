# Public release candidate audit

## Outcome

The directory is organized as a GitHub upload candidate for the scientific reproduction code. No remote push, repository-history rewrite, release tag, or Data/Code Availability statement was performed.

## Scope controls

- Original experiment code and frozen manuscript assets were not overwritten.
- Twenty-seven files under `src/fraudx/` are byte-identical to their audited source versions.
- A portable PaySim configuration, repository documentation, Git exclusions, and one orchestration-only PaySim Bootstrap entrypoint were added in the derived package.
- Manuscript drafting, citation mapping, submission-freeze automation, internal issue registries, and internal hash-audit generators were excluded.
- Raw data, processed transaction-level data, prediction ledgers, fitted models, checkpoints, and Bootstrap replicate files were excluded.

## Validation results

| Check | Result |
| --- | --- |
| Ruff | PASS; no findings |
| MyPy | PASS; 34 source files checked |
| Bandit | PASS for release gate; 0 medium/high issues, 8 existing low-severity findings |
| Public test suite | PASS; 113 passed, 8 skipped |
| Skipped-test rationale | Frozen-output integration tests require locally regenerated assets that are intentionally not redistributed |
| Candidate-source test | PASS; tests were rerun with the candidate `src/` forced on `PYTHONPATH` |
| Synthetic smoke test | PASS; used only for execution validation and produced no paper evidence |
| Command entrypoints | PASS; PaySim and IEEE-CIS public entrypoints parsed successfully |
| Absolute local path scan | PASS; 0 matches |
| Credential assignment pattern scan | PASS; 0 matches |
| Prohibited data/model artifact scan | PASS; 0 files |
| File size scan | PASS; no file exceeded 5 MB before manifest generation |

## Remaining release actions

The owner confirmed that the target repository is public and selected the MIT License. `wang-sir-le` is recorded as the temporary public author identifier. Before a tagged release, the uploaded tree must be verified against the manifest and checksums, the existing Git history must be inspected for previously committed data, credentials, or large artifacts, and the temporary author identifier should be replaced if a different publication name is preferred.
