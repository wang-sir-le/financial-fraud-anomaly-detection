"""Independent exhaustive decision checks for the portable training route."""

import numpy as np

from fraudx.paysim_deterministic.training import decision_metrics, threshold_grid


def test_grid_matches_exhaustive_tied_scores_and_no_alert() -> None:
    rng = np.random.default_rng(19092026)
    for _ in range(30):
        labels = np.r_[0, 1, rng.integers(0, 2, 48)]
        scores = rng.integers(0, 9, len(labels)) / 8
        capacities, costs = [.01, .03, .1, 1.0], [1, 10, 100]
        grid = threshold_grid(labels, scores, capacities, costs)
        candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
        for (q, c), result in grid.items():
            feasible = []
            for threshold in candidates:
                report = decision_metrics(labels, scores >= threshold, c)
                if report["alert_rate"] <= q:
                    feasible.append(((report["cost"], -report["tp"], report["alerts"],
                                      -threshold), threshold))
            expected = min(feasible)[1]
            assert result["threshold"] == expected
