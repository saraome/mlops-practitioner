import pickle
from pathlib import Path
from typing import Any

from prodml.config import MODEL_PATH


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the trained model and vectorizer."""
    with path.open("rb") as f:
        artifact = pickle.load(f)

    return artifact


def predict_duration(
    features: dict[str, object],
    path: Path = MODEL_PATH,
) -> float:
    """Predict taxi trip duration in minutes."""
    artifact = load_model(path)

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    X = vectorizer.transform([features])
    prediction = model.predict(X)[0]

    return float(prediction)
