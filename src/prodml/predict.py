import logging
import pickle
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

from prodml.config import MODEL_PATH

logger = logging.getLogger(__name__)

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

    def __init__(self, vectorizer: Any, model: Any) -> None:
        self.vectorizer = vectorizer
        self.model = model

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DurationPredictor":
        """Load the vectorizer and model from disk."""

        with path.open("rb") as f:
            artifact = pickle.load(f)

        return cls(
            vectorizer=artifact["vectorizer"],
            model=artifact["model"],
        )

    @timed
    def predict_one(self, features: dict[str, object]) -> float:
        """Predict duration for one trip."""

        X = self.vectorizer.transform([features])
        prediction = self.model.predict(X)[0]

        return float(prediction)

    def predict_batch(
        self,
        features: list[dict[str, object]],
    ) -> list[float]:
        """Predict duration for multiple trips."""

        X = self.vectorizer.transform(features)
        predictions = self.model.predict(X)

        return [float(prediction) for prediction in predictions]
