import logging
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split

from prodml.config import MODEL_PATH, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE
from prodml.data import load_data, prepare_data
from prodml.features import add_features, prepare_feature_dicts
from prodml.logging_conf import setup_logging

logger = logging.getLogger("prodml.export")
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"


def export_to_onnx(
    pickle_path: Path = MODEL_PATH,
    onnx_path: Path = ONNX_MODEL_PATH,
) -> None:
    """Export the trained scikit-learn model to ONNX."""

    with pickle_path.open("rb") as f:
        artifact = pickle.load(f)

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    n_features = len(vectorizer.feature_names_)

    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with onnx_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())


def check_parity(
    pickle_path: Path = MODEL_PATH,
    onnx_path: Path = ONNX_MODEL_PATH,
) -> None:
    """Compare Pickle and ONNX predictions on 500 validation rows."""

    with pickle_path.open("rb") as f:
        artifact = pickle.load(f)

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    # Recreate the same validation split used during training
    df = add_features(prepare_data(load_data()))

    _, df_val = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    df_sample = df_val.iloc[:500]

    feature_dicts = prepare_feature_dicts(df_sample)

    X = vectorizer.transform(feature_dicts)

    # Pickle / scikit-learn predictions
    pickle_predictions = model.predict(X)

    # ONNX Runtime expects a dense float32 tensor
    X_onnx = X.toarray().astype(np.float32)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    onnx_predictions = session.run(
        None,
        {input_name: X_onnx},
    )[0].ravel()

    max_difference = np.max(np.abs(pickle_predictions - onnx_predictions))

    logger.info(
        "Maximum prediction difference: %.8f",
        max_difference,
    )

    assert np.allclose(
        pickle_predictions,
        onnx_predictions,
        atol=1e-4,
    )

    logger.info("Pickle and ONNX predictions match")


def benchmark_inference(
    pickle_path: Path = MODEL_PATH,
    onnx_path: Path = ONNX_MODEL_PATH,
) -> None:
    """Benchmark Pickle and ONNX inference latency on 500 validation rows."""

    with pickle_path.open("rb") as f:
        artifact = pickle.load(f)

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    df = add_features(prepare_data(load_data()))

    _, df_val = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    df_sample = df_val.iloc[:500]

    feature_dicts = prepare_feature_dicts(df_sample)

    X = vectorizer.transform(feature_dicts)
    X_onnx = X.toarray().astype(np.float32)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        model.predict(X[:1])
        session.run(
            None,
            {input_name: X_onnx[:1]},
        )

    pickle_latencies = []
    onnx_latencies = []

    for i in range(500):
        start = perf_counter()
        model.predict(X[i : i + 1])
        pickle_latencies.append((perf_counter() - start) * 1000)

        start = perf_counter()
        session.run(
            None,
            {input_name: X_onnx[i : i + 1]},
        )
        onnx_latencies.append((perf_counter() - start) * 1000)

    pickle_mean = np.mean(pickle_latencies)
    pickle_p95 = np.percentile(pickle_latencies, 95)

    onnx_mean = np.mean(onnx_latencies)
    onnx_p95 = np.percentile(onnx_latencies, 95)

    logger.info(
        "Pickle latency: mean=%.4f ms, p95=%.4f ms",
        pickle_mean,
        pickle_p95,
    )

    logger.info(
        "ONNX latency: mean=%.4f ms, p95=%.4f ms",
        onnx_mean,
        onnx_p95,
    )


def main() -> None:
    setup_logging()
    export_to_onnx()
    check_parity()
    benchmark_inference()


if __name__ == "__main__":
    main()
