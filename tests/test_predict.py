import pytest

from prodml.predict import DurationPredictor


def test_predict_one_returns_sane_float(
    trained_model: DurationPredictor,
    sample_features: dict[str, object],
) -> None:
    """Prediction should be a float in a reasonable duration range."""

    prediction = trained_model.predict_one(sample_features)

    assert isinstance(prediction, float)
    assert 0 < prediction < 180


def test_prediction_is_deterministic(
    trained_model: DurationPredictor,
    sample_features: dict[str, object],
) -> None:
    """The same input should produce the same prediction."""

    first_prediction = trained_model.predict_one(sample_features)
    second_prediction = trained_model.predict_one(sample_features)

    assert first_prediction == pytest.approx(second_prediction)


def test_predict_batch(
    trained_model: DurationPredictor,
) -> None:
    """Batch prediction should return one result per input."""

    features = [
        {
            "PU_DO": "74_236",
            "trip_distance": 2.5,
        },
        {
            "PU_DO": "75_236",
            "trip_distance": 5.0,
        },
    ]

    predictions = trained_model.predict_batch(features)

    assert len(predictions) == 2
    assert all(isinstance(prediction, float) for prediction in predictions)


def test_unseen_pu_do_still_returns_prediction(
    trained_model: DurationPredictor,
) -> None:
    """An unseen PU_DO category should not crash prediction."""

    features = {
        "PU_DO": "999_888",
        "trip_distance": 2.5,
    }

    prediction = trained_model.predict_one(features)

    assert isinstance(prediction, float)
