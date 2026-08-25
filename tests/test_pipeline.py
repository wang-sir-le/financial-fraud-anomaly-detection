from pathlib import Path

from fraudx.config import load_config
from fraudx.pipeline import run_experiment


def test_smoke_pipeline_writes_research_artifacts(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)
    raw = dict(config.raw)
    raw["output_dir"] = str(tmp_path)
    object.__setattr__(config, "output_dir", tmp_path)
    object.__setattr__(config, "raw", raw)
    results = run_experiment(config, smoke=True)
    assert not results.empty
    assert (tmp_path / "tables" / "ablation_results.csv").exists()
    assert (tmp_path / "tables" / "validation_selected_test_results.csv").exists()
    assert (tmp_path / "tables" / "fixed_model_cost_sensitivity.csv").exists()
    assert (tmp_path / "tables" / "model_ranking_results.csv").exists()
    assert (tmp_path / "metadata" / "split_profile.json").exists()
    assert (tmp_path / "metadata" / "model_metadata.json").exists()
    assert (tmp_path / "metadata" / "run_summary.json").exists()
