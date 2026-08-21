# MLOps Practitioner

A hands-on MLOps project for transforming a machine learning notebook into a reproducible and maintainable Python package.

The project currently focuses on predicting NYC Green Taxi trip duration in minutes.

## Project Goal

The goal of this project is to predict taxi trip duration using a baseline machine learning model, then gradually refactor the notebook into a structured Python package without changing the original model behavior.

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

The dataset is downloaded automatically if it is not available locally.

The `data/` directory is excluded from Git.

## Machine Learning Baseline

The prediction target is trip duration in minutes.

The model uses the following features:

- `PU_DO`: combination of pickup and drop-off location IDs
- `trip_distance`

The categorical feature is transformed using `DictVectorizer`.

The baseline model is:

```text
LinearRegression
```

Trips with durations outside the range of 1 to 180 minutes are excluded.

The dataset is split into training and validation sets using:

```text
test_size = 0.2
random_state = 42
```

## Baseline Results

Validation results:

```text
RMSE: 7.966 minutes
MAE: 4.395 minutes
R²: 0.417
```

The refactored Python package reproduces the same validation MAE as the original notebook.

## Project Structure

```text
mlops-practitioner/
├── data/
├── models/
│   ├── baseline.pkl
│   └── model.pkl
├── notebooks/
│   └── 00-baseline.ipynb
├── reports/
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── predict.py
│       └── train.py
├── tests/
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Package Components

### `config.py`

Contains project configuration such as:

- dataset URL and path
- model path
- duration limits
- train/validation split settings
- feature names

### `data.py`

Responsible for:

- downloading the dataset
- loading the Parquet file
- calculating trip duration
- filtering trip durations

### `features.py`

Responsible for creating model features and preparing feature dictionaries for `DictVectorizer`.

### `train.py`

Responsible for:

- train/validation split
- fitting `DictVectorizer`
- training `LinearRegression`
- evaluating the model
- saving the trained model

### `predict.py`

Provides the `DurationPredictor` interface.

It supports:

```python
DurationPredictor.load()
DurationPredictor.predict_one()
DurationPredictor.predict_batch()
```

A custom `@timed` decorator is used to measure prediction execution time.

## Environment Setup

The project uses Python 3.12 and `uv` for dependency management.

Install and synchronize dependencies:

```bash
uv sync
```

Activate the environment if needed:

```bash
source .venv/bin/activate
```

## Train the Model

Run:

```bash
python -m prodml.train
```

or:

```bash
prodml-train
```

The trained model is saved to:

```text
models/model.pkl
```

## Make a Prediction

Example:

```python
from prodml.predict import DurationPredictor

predictor = DurationPredictor.load()

features = {
    "PU_DO": "74_236",
    "trip_distance": 2.5,
}

prediction = predictor.predict_one(features)

print(prediction)
```

Batch prediction is also supported with:

```python
predictor.predict_batch(...)
```

## Code Quality

The project currently uses:

- Ruff
- Black
- mypy
- pre-commit

Run Ruff:

```bash
ruff check src
```

Check formatting:

```bash
black --check src
```

Run type checking:

```bash
mypy src/prodml
```

Run all pre-commit hooks:

```bash
pre-commit run --all-files
```

## Git Workflow

Development is performed on a feature branch rather than directly on `main`.

Current Module 1 branch:

```text
module-1-packaging
```

Work is saved using small logical commits before opening one Pull Request for the completed module.