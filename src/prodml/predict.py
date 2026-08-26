import logging
import pickle
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

from prodml.config import MODEL_PATH

logger = logging.getLogger("prodml.predict")

R = TypeVar("R")


def timed(func: Callable[..., R]) -> Callable[..., R]:
    """Measure and log function execution time."""

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> R:
        start_time = perf_counter()

        result = func(*args, **kwargs)

        elapsed_ms = (perf_counter() - start_time) * 1000

        logger.info(
            "%s took %.2f ms",
            func.__name__,
            elapsed_ms,
        )

        return result

    return wrapper


class DurationPredictor:
    """Load the trained model and make duration predictions."""

    def __init__(
        self,
        vectorizer: Any,
        model: Any,
        metadata: dict[str, object],
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DurationPredictor":
        """Load the vectorizer and model from disk."""

        try:
            with path.open("rb") as f:
                artifact = pickle.load(f)
        except Exception:
            logger.exception("Failed to load model from %s", path)
            raise

        logger.info("Model loaded from %s", path)

        return cls(
            vectorizer=artifact["vectorizer"],
            model=artifact["model"],
            metadata=artifact.get("metadata", {}),
        )

    @timed
    def predict_one(self, features: dict[str, object]) -> float:
        """Predict duration for one trip."""

        logger.debug("Prediction features: %s", features)

        trip_distance = features.get("trip_distance")

        if isinstance(trip_distance, (int, float)) and trip_distance > 100:
            logger.warning(
                "Input outside training range: trip_distance=%s",
                trip_distance,
            )

        X = self.vectorizer.transform([features])
        prediction = self.model.predict(X)[0]
        logger.info("Prediction served: %.3f minutes", prediction)
        return float(prediction)

    def predict_batch(
        self,
        features: list[dict[str, object]],
    ) -> list[float]:
        """Predict duration for multiple trips."""

        X = self.vectorizer.transform(features)
        predictions = self.model.predict(X)

        return [float(prediction) for prediction in predictions]
