from typing import cast

import pandas as pd

from prodml.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features."""
    df = df.copy()

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    return df


def prepare_feature_dicts(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Convert model features into dictionaries for DictVectorizer."""
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    return cast(
        list[dict[str, object]],
        df[features].to_dict(orient="records"),
    )
