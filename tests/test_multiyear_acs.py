import pandas as pd


def test_get_acs_years_prefers_multiyear_list():
    from src.data_processing import _get_acs_years

    years = _get_acs_years({"survey_years": ["2018", "2016", "2017"]})
    assert years == ["2016", "2017", "2018"]


def test_resolve_acs_temporal_split_uses_earliest_years_for_training():
    from src.data_processing import _resolve_acs_temporal_split

    enabled, train_years, eval_years = _resolve_acs_temporal_split(
        {
            "survey_years": ["2016", "2017", "2018"],
            "temporal_split": {
                "enabled": True,
                "initial_train_years": 1,
            },
        }
    )

    assert enabled is True
    assert train_years == {"2016"}
    assert eval_years == {"2017", "2018"}


def test_temporal_drift_simulator_sources_batches_by_year():
    from src.drift_simulation import DriftSimulator

    test_df = pd.DataFrame(
        {
            "AGEP": [30, 31, 32, 45, 46, 47],
            "WKHP": [40, 41, 39, 50, 51, 49],
            "COW": [1, 1, 2, 1, 2, 2],
            "SCHL": [16, 16, 17, 18, 18, 17],
            "MAR": [1, 1, 2, 2, 3, 3],
            "OCCP": [1000, 1001, 1002, 2000, 2001, 2002],
            "POBP": [10, 10, 11, 20, 20, 21],
            "RELP": [1, 2, 1, 2, 1, 2],
            "SEX": [1, 2, 1, 2, 1, 2],
            "RAC1P": [1, 1, 2, 2, 1, 2],
            "income": ["<=50K", "<=50K", ">50K", "<=50K", ">50K", ">50K"],
            "DATA_YEAR": ["2017", "2017", "2017", "2018", "2018", "2018"],
        }
    )

    config = {
        "batch_size": 2,
        "source_mode": "temporal_years",
        "temporal_year_column": "DATA_YEAR",
        "batches": {
            "batch_1": {"description": "year 2017"},
            "batch_2": {"description": "year 2017"},
            "batch_3": {"description": "year 2018"},
            "batch_4": {"description": "year 2018"},
        },
    }

    simulator = DriftSimulator(test_df=test_df, config=config, seed=42)
    batches = simulator.generate_batches()

    assert set(batches["batch_1"]["DATA_YEAR"]) == {"2017"}
    assert set(batches["batch_2"]["DATA_YEAR"]) == {"2017"}
    assert set(batches["batch_3"]["DATA_YEAR"]) == {"2018"}
    assert set(batches["batch_4"]["DATA_YEAR"]) == {"2018"}
