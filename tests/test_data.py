from pathlib import Path

import pandas as pd
import pytest

from prodml.data import download_data, load_data, prepare_data


def test_prepare_data_filters_invalid_durations() -> None:
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 12:00:00",
                ]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:30",
                    "2024-01-01 11:10:00",
                    "2024-01-01 16:00:00",
                ]
            ),
        }
    )

    result = prepare_data(df)

    assert len(result) == 1
    assert result.iloc[0]["duration"] == 10.0


def test_load_data_reads_parquet(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"

    expected = pd.DataFrame(
        {
            "PULocationID": [74],
            "DOLocationID": [236],
            "trip_distance": [2.5],
        }
    )

    expected.to_parquet(path)

    result = load_data(path)

    pd.testing.assert_frame_equal(result, expected)


def test_download_data_skips_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "existing.parquet"
    path.write_text("already here")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("urlretrieve should not be called")

    monkeypatch.setattr(
        "prodml.data.urlretrieve",
        fail_if_called,
    )

    result = download_data(
        url="https://example.com/data.parquet",
        path=path,
    )

    assert result == path


def test_download_data_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data" / "new.parquet"

    def fake_download(url: str, filename: Path) -> None:
        filename.write_text("downloaded")

    monkeypatch.setattr(
        "prodml.data.urlretrieve",
        fake_download,
    )

    result = download_data(
        url="https://example.com/data.parquet",
        path=path,
    )

    assert result == path
    assert path.exists()
