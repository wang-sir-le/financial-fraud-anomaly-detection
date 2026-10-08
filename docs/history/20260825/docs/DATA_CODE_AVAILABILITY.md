# Data and code availability policy

## Confirmed policy

The repository owner confirmed that the target GitHub repository is public and selected the MIT License. The current public author identifier is `wang-sir-le`, pending replacement if a preferred publication name is later supplied.

The public code package covers the main PaySim and IEEE-CIS scientific workflows: preprocessing, leakage-controlled entity-history construction, temporal protocol generation, LightGBM experiments, evaluation, PaySim calibration, and paired time-block Bootstrap analyses. It includes portable configuration, dependency declarations, execution entrypoints, tests, and the venue-neutral Supplement. Internal manuscript-freeze, citation-management, issue-registry, and submission-audit automation are outside the public research-code scope.

Neither PaySim nor IEEE-CIS is redistributed. PaySim users are directed to the original dataset paper and an authorized acquisition source. IEEE-CIS users are directed to the official Kaggle competition page and must obtain the data under the applicable platform access conditions.

The execution environment is specified through `pyproject.toml` and `requirements.txt`. The repository provides the main scientific entrypoints but does not promise distribution of every internal quality-assurance script.

## Verified remote synchronization status

The audited scientific reproduction package has been synchronized to the public GitHub repository through an ordinary commit to `main`. The original verified root commit is `dc3d374eab1d1ecfd39dcb1a849da6d04236e27b`. At the synchronization-verification stage, the remote tree contained all 72 approved repository files, and all 72 matched the approved local candidate by byte-derived SHA-256. Anonymous repository cloning and access to representative package resources were also verified. Raw PaySim and IEEE-CIS data remain excluded from the repository.

Remote publication and integrity verification establish that the approved research assets are publicly accessible and byte-consistent with the audited candidate. They do not establish that an independent external researcher has completed end-to-end reproduction, and they do not guarantee execution on every system.

`PUBLIC_RELEASE_AUDIT.md` is retained as a historical pre-release candidate-audit snapshot. Statements in that file describing the absence of a remote push refer to the earlier candidate-audit stage and do not describe the repository's current synchronization status.

## Candidate venue-neutral manuscript statement after verified synchronization

> The code supporting the principal PaySim and IEEE-CIS workflows is available under the MIT License at https://github.com/wang-sir-le/financial-fraud-anomaly-detection. The repository includes preprocessing, leakage-controlled history-feature construction, temporal validation, model evaluation, calibration and paired time-block Bootstrap code, together with configuration and environment specifications. Raw PaySim and IEEE-CIS data are not redistributed. PaySim access is documented through its original source, whereas IEEE-CIS must be obtained from the official Kaggle competition page under the applicable access conditions.

This is a candidate venue-neutral manuscript statement. Its insertion into a submission remains subject to a separate CR-004 Availability activation decision and the selected venue's anonymity policy. It is not active in the manuscript at this stage.
