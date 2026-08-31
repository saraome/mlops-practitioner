from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from prodml.config import DATA_PATH, DATA_URL, DURATION_MAX, DURATION_MIN


def download_data(
    url: str = DATA_URL,
    path: Path = DATA_PATH,
) -> Path:
    """Download the dataset if it does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        urlretrieve(url, path)

    return path


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the taxi trip dataset from a Parquet file."""
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trip duration and keep trips between 1 and 180 minutes."""
    df = df.copy()

    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[(df["duration"] >= DURATION_MIN) & (df["duration"] <= DURATION_MAX)].copy()

    return df
