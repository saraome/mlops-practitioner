import pandas as pd
import pytest

from prodml.features import add_features, prepare_feature_dicts


def test_add_features_creates_pu_do() -> None:
    """PU_DO should combine pickup and dropoff location IDs."""

    df = pd.DataFrame(
        {
            "PULocationID": [74],
            "DOLocationID": [236],
            "trip_distance": [2.5],
        }
    )

    result = add_features(df)

    assert result.loc[0, "PU_DO"] == "74_236"


@pytest.mark.parametrize(
    ("pickup", "dropoff", "distance", "expected_pu_do"),
    [
        (74, 236, 0.0, "74_236"),
        (999, 888, 2.5, "999_888"),
    ],
)
def test_feature_edge_cases(
    pickup: int,
    dropoff: int,
    distance: float,
    expected_pu_do: str,
) -> None:
    """Feature engineering should preserve valid edge-case values."""

    df = pd.DataFrame(
        {
            "PULocationID": [pickup],
            "DOLocationID": [dropoff],
            "trip_distance": [distance],
        }
    )

    result = add_features(df)
    feature_dicts = prepare_feature_dicts(result)

    assert result.loc[0, "PU_DO"] == expected_pu_do
    assert feature_dicts[0]["trip_distance"] == distance


def test_missing_pickup_category_raises_error() -> None:
    """Missing pickup category should fail explicitly."""

    df = pd.DataFrame(
        {
            "DOLocationID": [236],
            "trip_distance": [2.5],
        }
    )

    with pytest.raises(KeyError):
        add_features(df)
