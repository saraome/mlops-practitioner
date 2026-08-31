import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import MODEL_PATH, PROJECT_ROOT


def test_pickle_and_onnx_predictions_match() -> None:
    """Pickle and ONNX models should produce nearly identical predictions."""

    with MODEL_PATH.open("rb") as file:
        artifact = pickle.load(file)

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    features = [
        {
            "PU_DO": "74_236",
            "trip_distance": 2.5,
        }
    ]

    x = vectorizer.transform(features)

    pickle_prediction = model.predict(x)

    onnx_path = PROJECT_ROOT / "models" / "model.onnx"

    session = ort.InferenceSession(str(onnx_path))

    input_name = session.get_inputs()[0].name

    onnx_prediction = session.run(
        None,
        {
            input_name: x.toarray().astype(np.float32),
        },
    )[0].ravel()

    assert np.allclose(
        pickle_prediction,
        onnx_prediction,
        atol=1e-4,
    )
