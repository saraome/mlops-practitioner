import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet"
)

DATA_PATH = PROJECT_ROOT / "data" / "green_tripdata_2024-01.parquet"
MODEL_PATH = Path(
    os.getenv(
        "PRODML_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "model.pkl"),
    )
)

DURATION_MIN = 1
DURATION_MAX = 180

TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["PU_DO"]
NUMERICAL_FEATURES = ["trip_distance"]
