# Module 1 Report

## Baseline Model Metrics

- Validation RMSE: 7.966 minutes
- Validation MAE: 4.395 minutes
- Validation R² = 0.417

## Serialization Comparison

The trained model was serialized using both Pickle and ONNX.

Prediction parity was evaluated using 500 validation samples.

### Prediction Parity

```text
Maximum prediction difference: 0.00000237
Tolerance: 0.0001
Result: Passed
```

The ONNX predictions closely match the original scikit-learn model predictions.

### Inference Latency

The benchmark measures model inference after feature transformation.

| Format | Mean Latency | P95 Latency |
| --- | ---: | ---: |
| Pickle / scikit-learn | 0.2649 ms | 0.3630 ms |
| ONNX Runtime | 0.0351 ms | 0.0626 ms |

In this benchmark, ONNX Runtime provided lower inference latency than the Pickle/scikit-learn model.

### Model Size

| Format | File Size |
| --- | ---: |
| Pickle | 141 KB |
| ONNX | 23 KB |

The ONNX model artifact is considerably smaller than the Pickle artifact.

### Serialization Format Trade-offs

| Format | Human Readable | Cross-language | Schema Enforcement | Safe for Untrusted Files |
| --- | --- | --- | --- | --- |
| JSON | Yes | Yes | No, unless validated separately | Generally safer as data |
| Protobuf | No | Yes | Yes | Safer than executable serialization |
| Pickle | No | Python-focused | No | No |
| ONNX | No | Yes | Yes, model graph format | More suitable for portable model inference |

Pickle is convenient for Python and preserves the scikit-learn objects directly, including the fitted `DictVectorizer`. However, Pickle files should only be loaded from trusted sources because unpickling malicious files can execute arbitrary code.

ONNX provides a portable model representation, smaller model size, and faster inference in the benchmark performed for this project.

For the current service, ONNX is the preferred format for model inference, while the fitted `DictVectorizer` is still loaded from the trusted Pickle artifact for feature transformation.

## API Service

The trained taxi-duration model is exposed through a FastAPI service.

### Implemented Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Confirms that the API and model are ready |
| `/metadata` | GET | Returns model metadata and artifact hash |
| `/predict` | POST | Returns a prediction for one trip |
| `/predict/batch` | POST | Returns predictions for multiple trips |

The model is loaded once at application startup using FastAPI lifespan management.

The `/metadata` endpoint returns:

- model version
- training date
- feature names
- framework
- artifact SHA-256 hash

Prediction responses include:

- predicted trip duration
- model version
- correlation ID
- prediction latency in milliseconds

Pydantic validation is used for request validation. Invalid inputs such as a negative `trip_distance` return HTTP `422 Unprocessable Entity`.

Each request receives a correlation ID that is included in:

- structured application logs
- the API response
- the `X-Request-ID` response header

---

## Automated Testing

A pytest-based automated test suite was added to validate the application pipeline.

The test suite covers:

- feature engineering
- edge cases
- single prediction
- batch prediction
- unseen `PU_DO` values
- API health and metadata endpoints
- API prediction endpoints
- request validation
- correlation ID propagation
- data loading and preparation
- training pipeline execution
- Pickle / ONNX prediction parity

Shared pytest fixtures are defined in `tests/conftest.py`.

`pytest.mark.parametrize` is used for edge-case testing, and `monkeypatch` is used to isolate tests from external downloads and production files.

### Coverage Result

The project enforces a minimum coverage gate of 70%.

Final measured coverage:

```text
Total coverage: 71.56%
Required coverage: 70%
Status: PASSED
```

Key coverage results included:

```text
api/main.py       96%
api/schemas.py   100%
data.py          100%
features.py      100%
logging_conf.py  100%
predict.py        91%
train.py          98%
```

The ONNX export utility was not directly covered by the final coverage run, but ONNX parity was validated separately through serialization tests.

---

## Containerization

The FastAPI prediction service was containerized using Docker.

### Docker Design

The Docker implementation includes:

- multi-stage build
- Python 3.12 slim base image
- non-root runtime user (`appuser`)
- application health check
- model path configured through `PRODML_MODEL_PATH`
- port `8000` exposed for the FastAPI service

The container health endpoint was tested successfully:

```text
GET /health -> {"status":"ok"}
```

The runtime user was also verified:

```text
appuser
```

### Docker Ignore Comparison

The image was built before and after adding `.dockerignore`.

| Build | Content Size |
| --- | ---: |
| Without `.dockerignore` | 246 MB |
| With `.dockerignore` | 246 MB |

The final image size remained unchanged because the Dockerfile already copies only selected project files. However, `.dockerignore` reduces unnecessary build context by excluding files such as:

- `.venv`
- dataset files
- notebooks
- tests
- caches
- Git metadata

### Docker Compose

A `docker-compose.yml` file was added to simplify local execution.

The service can be started with:

```bash
docker compose up -d
```

and stopped with:

```bash
docker compose down
```

The Docker Compose deployment was tested successfully using the `/health` endpoint.

### Docker Hub Publishing

The image was published to Docker Hub using both:

```text
0.1.0
latest
```

The versioned tag provides a reproducible release, while `latest` identifies the most recent published image.

---

## Module 1 Completion Summary

Module 1 now includes:

- reproducible data preparation
- baseline model training and evaluation
- installable Python package structure
- structured JSON logging
- model serialization comparison
- ONNX export and parity validation
- FastAPI prediction service
- request validation and correlation IDs
- automated pytest test suite
- coverage gate above 70%
- multi-stage Docker image
- non-root runtime execution
- Docker Compose support
- Docker Hub publishing

The module is ready for final review and Pull Request preparation.
