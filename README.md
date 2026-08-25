# MLOps Practitioner

A hands-on MLOps project that transforms a machine learning notebook into a reproducible, maintainable, and production-oriented Python application.

The project currently predicts NYC Green Taxi trip duration in minutes.

The work starts with a baseline Jupyter Notebook and gradually applies software engineering and MLOps practices while preserving the original machine learning behavior.

---

## Project Goal

The goal of this project is to predict taxi trip duration in minutes using NYC Green Taxi trip data.

The project starts with a baseline machine learning model developed in a Jupyter Notebook.

The same model logic is then refactored into a structured Python package without changing the baseline model behavior.

---

## Dataset

The project uses one month of the NYC TLC Green Taxi Trip Record Data.

- Dataset: Green Taxi Trip Records
- Month: January 2024
- File format: Parquet
- Source: NYC Taxi & Limousine Commission

Dataset URL:

```text
https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet
```

The dataset is downloaded automatically if it is not already available locally.

The `data/` directory is excluded from Git because the raw dataset should not be stored in the repository.

---

## Machine Learning Baseline

The prediction target is:

```text
Trip duration in minutes
```

Trip duration is calculated from the pickup and drop-off timestamps.

Trips outside the following duration range are excluded:

```text
1 to 180 minutes
```

### Features

The model uses two features:

- `PU_DO`: a combined pickup and drop-off location identifier
- `trip_distance`

The `PU_DO` categorical feature is transformed using:

```text
DictVectorizer
```

### Model

The baseline model is:

```text
LinearRegression
```

### Train / Validation Split

The dataset is split using:

```text
test_size = 0.2
random_state = 42
```

The `DictVectorizer` is fitted only on the training dataset and then used to transform the validation dataset.

---

## Baseline Results

The baseline validation results are:

```text
Validation RMSE: 7.966 minutes
Validation MAE: 4.395 minutes
R²: 0.417
```

After refactoring the notebook into a Python package, the validation results remained:

```text
Validation RMSE: 7.966 minutes
Validation MAE: 4.395 minutes
R²: 0.417
```

Therefore, the refactoring preserved the original model behavior.

---

## Project Structure

```text
mlops-practitioner/
├── data/
│   └── green_tripdata_2024-01.parquet
│
├── models/
│   ├── baseline.pkl
│   ├── model.pkl
│   └── model.onnx
│
├── notebooks/
│   └── 00-baseline.ipynb
│
├── reports/
│   └── module-1.md
│
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Python Package

The notebook logic has been refactored into the `prodml` Python package.

### `config.py`

Contains shared project configuration such as:

- project paths
- dataset URL
- dataset path
- model path
- minimum and maximum trip duration
- train / validation split settings
- categorical and numerical feature names

### `data.py`

Responsible for:

- downloading the dataset
- loading the Parquet dataset
- calculating trip duration
- filtering trip durations between 1 and 180 minutes

### `features.py`

Responsible for feature engineering.

It creates the `PU_DO` feature and converts the required model features into dictionaries that can be processed by `DictVectorizer`.

### `train.py`

Responsible for:

- loading and preparing the dataset
- creating the model features
- splitting the dataset into training and validation sets
- fitting the `DictVectorizer`
- training the `LinearRegression` model
- calculating RMSE, MAE, and R²
- saving the trained vectorizer and model

The trained Pickle artifact is saved as:

```text
models/model.pkl
```

### `predict.py`

Provides the prediction interface through the:

```python
DurationPredictor
```

class.

The class supports:

```python
DurationPredictor.load()
DurationPredictor.predict_one()
DurationPredictor.predict_batch()
```

The trained model is loaded once and can then be reused for single or batch predictions.

A custom `@timed` decorator is also used to measure prediction execution latency.

### `logging_conf.py`

Provides structured JSON logging for the application.

### `export.py`

Responsible for:

- exporting the trained scikit-learn regression model to ONNX
- comparing Pickle and ONNX predictions
- checking prediction parity
- benchmarking inference latency

---

## Environment Setup

The project uses:

```text
Python 3.12
```

and `uv` for Python dependency and environment management.

Synchronize the project environment using:

```bash
uv sync
```

If the virtual environment needs to be activated manually:

```bash
source .venv/bin/activate
```

---

## Train the Model

The model can be trained using:

```bash
python -m prodml.train
```

The project also provides the following command-line entry point:

```bash
prodml-train
```

Training creates:

```text
models/model.pkl
```

Expected validation metrics are approximately:

```text
RMSE: 7.966
MAE: 4.395
R²: 0.417
```

---

## Making Predictions

Load the predictor:

```python
from prodml.predict import DurationPredictor

predictor = DurationPredictor.load()
```

### Single Prediction

```python
features = {
    "PU_DO": "74_236",
    "trip_distance": 2.5,
}

prediction = predictor.predict_one(features)

print(prediction)
```

Example prediction:

```text
12.4679...
```

### Batch Prediction

The same model can also predict multiple trips:

```python
features = [
    {
        "PU_DO": "74_236",
        "trip_distance": 2.5,
    },
    {
        "PU_DO": "74_236",
        "trip_distance": 5.0,
    },
]

predictions = predictor.predict_batch(features)

print(predictions)
```

---

## Structured Logging

The project uses structured JSON logging instead of `print()` statements inside the application source code.

Each log record contains:

- `timestamp`
- `level`
- `logger`
- `message`
- `correlation_id`

Example:

```json
{
  "timestamp": "2026-08-25T17:39:16.187384+00:00",
  "level": "INFO",
  "logger": "prodml.train",
  "message": "Validation RMSE: 7.966",
  "correlation_id": "-"
}
```

The project uses different logging levels for different situations.

### DEBUG

Used for detailed prediction input information.

Example:

```text
Prediction features
```

### INFO

Used for normal application events such as:

- model loading
- model metrics
- predictions
- prediction latency
- model export

### WARNING

Used when an input is outside the expected training range.

For example:

```text
trip_distance > 100
```

### ERROR

Used when an operation such as model loading fails.

The logging configuration also prepares a `correlation_id` context that will later be connected to individual API requests.

---

## Code Quality

The project currently uses several development tools to maintain code quality.

### Ruff

Ruff is used for linting and import organization.

Run:

```bash
uv run ruff check src/prodml
```

Automatic fixes can be applied using:

```bash
uv run ruff check src/prodml --fix
```

### Black

Black is used for code formatting.

Check formatting using:

```bash
uv run black --check src/prodml
```

Format the code using:

```bash
uv run black src/prodml
```

### mypy

mypy is used for static type checking.

Run:

```bash
uv run mypy src/prodml
```

### pre-commit

The project uses pre-commit hooks to automatically run code-quality checks before Git commits.

Run all configured hooks manually using:

```bash
uv run pre-commit run --all-files
```

---

## Model Serialization

The project currently evaluates two model serialization approaches:

- Pickle
- ONNX

The original trained artifact is stored as:

```text
models/model.pkl
```

The regression model is also exported to:

```text
models/model.onnx
```

Run the ONNX export and serialization benchmark using:

```bash
python -m prodml.export
```

---

## Pickle and ONNX Prediction Parity

Prediction parity was evaluated using 500 validation samples.

The maximum difference between the scikit-learn predictions and ONNX predictions was:

```text
Maximum prediction difference: 0.00000237
```

The required tolerance was:

```text
0.0001
```

Result:

```text
Passed
```

The ONNX model therefore produces predictions that closely match the original scikit-learn regression model.

---

## Serialization Benchmark

The inference benchmark was performed after feature transformation so that the comparison focuses on model inference.

The most recent benchmark produced:

| Format | Mean Latency | P95 Latency | Model Size |
| --- | ---: | ---: | ---: |
| Pickle / scikit-learn | 0.2649 ms | 0.3630 ms | 141 KB |
| ONNX Runtime | 0.0351 ms | 0.0626 ms | 23 KB |

In this benchmark, ONNX Runtime provided:

- lower inference latency
- lower P95 latency
- a smaller model artifact
- predictions equivalent to the original scikit-learn model within the required tolerance

Latency values may vary slightly between runs depending on system load.

---

## Serialization Trade-offs

### Pickle

Advantages:

- simple to use with Python
- directly preserves fitted scikit-learn objects
- currently stores both the fitted `DictVectorizer` and the trained regression model

Disadvantages:

- mainly Python-focused
- not suitable for loading files from untrusted sources
- malicious Pickle files may execute arbitrary code during deserialization

### ONNX

Advantages:

- portable model representation
- supports inference outside the original Python/scikit-learn environment
- smaller model artifact in this project
- faster model inference in the current benchmark

In the current implementation, only the regression model is exported to ONNX.

The fitted `DictVectorizer` is still loaded from the trusted Pickle artifact and is used to transform the raw features before ONNX inference.

The current inference flow is therefore:

```text
Raw features
     ↓
Fitted DictVectorizer
     ↓
Numerical feature vector
     ↓
ONNX model
     ↓
Trip duration prediction
```

---

## Current Progress

The following work has been completed:

- baseline Jupyter Notebook
- data preparation
- feature engineering
- baseline model training
- model evaluation
- Python package refactoring
- reproducible dataset download
- editable package installation
- single prediction interface
- batch prediction interface
- custom prediction timing decorator
- type hints
- Ruff linting
- Black formatting
- mypy type checking
- pre-commit hooks
- structured JSON logging
- logging levels
- Pickle model serialization
- ONNX model export
- Pickle / ONNX prediction parity testing
- Pickle / ONNX latency benchmarking
- serialization format comparison

---

## Git Workflow

Development is performed on a dedicated feature branch rather than directly on `main`.

The current Module 1 branch is:

```text
module-1-packaging
```

Changes are saved using small logical commits.

The completed module will later be submitted through one Pull Request according to the project collaboration workflow.