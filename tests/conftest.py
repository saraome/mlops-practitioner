from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.predict import DurationPredictor


@pytest.fixture
def sample_features() -> dict[str, object]:
    return {
        "PU_DO": "74_236",
        "trip_distance": 2.5,
    }


@pytest.fixture(scope="session")
def trained_model() -> DurationPredictor:
    return DurationPredictor.load()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client
