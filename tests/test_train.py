from pathlib import Path

import pandas as pd
import pytest

from prodml import train


def test_training_pipeline_creates_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training pipeline should create a model artifact."""

    sample_df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 12:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 14:00:00",
                    "2024-01-01 15:00:00",
                    "2024-01-01 16:00:00",
                    "2024-01-01 17:00:00",
                    "2024-01-01 18:00:00",
                    "2024-01-01 19:00:00",
                ]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:10:00",
                    "2024-01-01 11:12:00",
                    "2024-01-01 12:08:00",
                    "2024-01-01 13:15:00",
                    "2024-01-01 14:20:00",
                    "2024-01-01 15:11:00",
                    "2024-01-01 16:09:00",
                    "2024-01-01 17:18:00",
                    "2024-01-01 18:13:00",
                    "2024-01-01 19:16:00",
                ]
            ),
            "PULocationID": [74, 75, 74, 75, 76, 74, 76, 75, 74, 76],
            "DOLocationID": [236, 236, 237, 237, 236, 238, 238, 236, 237, 238],
            "trip_distance": [2.5, 3.0, 1.8, 4.2, 5.0, 2.1, 3.7, 4.8, 2.9, 3.3],
        }
    )

    model_path = tmp_path / "model.pkl"

    monkeypatch.setattr(
        train,
        "download_data",
        lambda: tmp_path / "fake.parquet",
    )

    monkeypatch.setattr(
        train,
        "load_data",
        lambda path: sample_df,
    )

    monkeypatch.setattr(
        train,
        "MODEL_PATH",
        model_path,
    )

    train.main()

    assert model_path.exists()
    assert model_path.stat().st_size > 0
