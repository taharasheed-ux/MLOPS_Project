from experiments.run_noise_ablation import (
    LABEL_FLIP_MODES,
    build_ablation_config,
    make_policy_config,
)
from src.utils import load_yaml


def _base_config():
    return {
        "batch_size": 10,
        "batches": {
            "batch_1": {
                "perturbations": {
                    "covariate_shift": {"AGEP": {"method": "shift", "value": 3}},
                    "conditional_shift": [
                        {
                            "condition": {
                                "feature": "WKHP",
                                "operator": ">=",
                                "value": 50,
                            },
                            "flip_probability": 0.2,
                        }
                    ],
                    "feature_importance_shift": {"WKHP": {"noise_std_factor": 0.2}},
                }
            }
        },
    }


def test_clean_temporal_ablation_removes_all_perturbations():
    config = build_ablation_config(_base_config(), "clean_temporal")

    assert config["batches"]["batch_1"]["perturbations"] == {}


def test_covariate_feature_ablation_removes_label_flips_only():
    config = build_ablation_config(_base_config(), "covariate_feature_shift")
    perturbations = config["batches"]["batch_1"]["perturbations"]

    assert "conditional_shift" not in perturbations
    assert "covariate_shift" in perturbations
    assert "feature_importance_shift" in perturbations


def test_label_flip_ablation_keeps_only_conditional_shift():
    config = build_ablation_config(_base_config(), "label_flip_only")
    perturbations = config["batches"]["batch_1"]["perturbations"]

    assert list(perturbations.keys()) == ["conditional_shift"]


def test_make_policy_config_overrides_anchor_without_mutating_base():
    base = {
        "retraining_policy": {"severity_threshold": 0.1},
        "retraining_data": {"anchor_fraction": 0.1, "window_batches": 3},
    }

    config = make_policy_config(base, anchor_fraction=0.0, window_batches=2)

    assert config["retraining_data"]["anchor_fraction"] == 0.0
    assert config["retraining_data"]["window_batches"] == 2
    assert base["retraining_data"]["anchor_fraction"] == 0.1


def test_label_flip_modes_are_explicit_stress_test_modes():
    assert LABEL_FLIP_MODES == {"label_flip_only", "full_with_label_flips"}


def test_main_acs_config_excludes_label_flips_but_stress_config_keeps_them():
    main_config = load_yaml("drift_config_acs.yaml")
    stress_config = load_yaml("drift_config_acs_label_flip_stress.yaml")

    main_perturbations = [
        batch.get("perturbations", {})
        for batch in main_config["batches"].values()
    ]
    stress_perturbations = [
        batch.get("perturbations", {})
        for batch in stress_config["batches"].values()
    ]

    assert all("conditional_shift" not in perturb for perturb in main_perturbations)
    assert any("conditional_shift" in perturb for perturb in stress_perturbations)
