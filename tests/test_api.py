from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_endpoint(client: TestClient) -> None:
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert "model_version" in data
    assert "training_date" in data
    assert "feature_names" in data
    assert "framework" in data
    assert "artifact_hash" in data


def test_predict_endpoint(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "PU_DO": "74_236",
            "trip_distance": 2.5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["prediction"], float)
    assert data["model_version"] == "0.1.0"
    assert "correlation_id" in data
    assert "latency_ms" in data


def test_correlation_id_matches_header(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "PU_DO": "74_236",
            "trip_distance": 2.5,
        },
    )

    data = response.json()

    assert response.headers["X-Request-ID"] == data["correlation_id"]


def test_batch_prediction(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json=[
            {
                "PU_DO": "74_236",
                "trip_distance": 2.5,
            },
            {
                "PU_DO": "75_236",
                "trip_distance": 5.0,
            },
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert all("prediction" in item for item in data)


def test_invalid_trip_distance_returns_422(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={
            "PU_DO": "74_236",
            "trip_distance": -5,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "Invalid request data"
