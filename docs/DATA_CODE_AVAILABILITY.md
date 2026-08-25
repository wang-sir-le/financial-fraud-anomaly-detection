# Data and code availability policy

## Confirmed policy

The repository owner confirmed that the target GitHub repository is public and selected the MIT License. The current public author identifier is `wang-sir-le`, pending replacement if a preferred publication name is later supplied.

The public code package covers the main PaySim and IEEE-CIS scientific workflows: preprocessing, leakage-controlled entity-history construction, temporal protocol generation, LightGBM experiments, evaluation, PaySim calibration, and paired time-block Bootstrap analyses. It includes portable configuration, dependency declarations, execution entrypoints, tests, and the venue-neutral Supplement. Internal manuscript-freeze, citation-management, issue-registry, and submission-audit automation are outside the public research-code scope.

Neither PaySim nor IEEE-CIS is redistributed. PaySim users are directed to the original dataset paper and an authorized acquisition source. IEEE-CIS users are directed to the official Kaggle competition page and must obtain the data under the applicable platform access conditions.

The execution environment is specified through `pyproject.toml` and `requirements.txt`. The repository provides the main scientific entrypoints but does not promise distribution of every internal quality-assurance script.

## Remote synchronization gate

The owner plans to synchronize the audited release candidate before manuscript submission. The package has not yet been synchronized to the remote repository. The manuscript must not state that the complete reproduction package is available at the repository URL until the uploaded tree and `checksums.sha256` have been verified against `PUBLIC_RELEASE_MANIFEST.csv`. If the selected venue uses double-blind review, the repository reference must follow that venue's anonymity policy.

## Candidate manuscript statement after successful synchronization

> The code supporting the principal PaySim and IEEE-CIS workflows is available under the MIT License at https://github.com/wang-sir-le/financial-fraud-anomaly-detection. The repository includes preprocessing, leakage-controlled history-feature construction, temporal validation, model evaluation, calibration and paired time-block Bootstrap code, together with configuration and environment specifications. Raw PaySim and IEEE-CIS data are not redistributed. PaySim access is documented through its original source, whereas IEEE-CIS must be obtained from the official Kaggle competition page under the applicable access conditions.

This statement becomes active only after remote-content verification.
