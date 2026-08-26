import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import MODEL_PATH
from prodml.logging_conf import correlation_id_var, setup_logging
from prodml.predict import DurationPredictor

logger = logging.getLogger("prodml.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once when the API starts."""

    setup_logging()

    logger.info("Loading prediction model")

    app.state.predictor = DurationPredictor.load()
    app.state.artifact_hash = calculate_artifact_hash()

    logger.info("Prediction model loaded")

    yield

    logger.info("API shutting down")


def calculate_artifact_hash() -> str:
    """Calculate SHA-256 hash of the model artifact."""

    sha256 = hashlib.sha256()

    with MODEL_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


app = FastAPI(
    title="NYC Taxi Duration Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a clean response for invalid request data."""

    logger.warning(
        "Validation error: %s",
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request data",
            "details": exc.errors(),
            "correlation_id": correlation_id_var.get(),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected errors without exposing internal details."""

    logger.exception(
        "Unexpected API error",
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": correlation_id_var.get(),
        },
    )


@app.middleware("http")
async def add_correlation_id(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """Attach a correlation ID to every request."""

    correlation_id = request.headers.get(
        "X-Request-ID",
        str(uuid4()),
    )

    token = correlation_id_var.set(correlation_id)

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = correlation_id

        logger.info(
            "Request completed: %s %s status=%s",
            request.method,
            request.url.path,
            response.status_code,
        )

        return response

    finally:
        correlation_id_var.reset(token)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Check that the API and prediction model are ready."""

    predictor = getattr(
        request.app.state,
        "predictor",
        None,
    )

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    return {"status": "ok"}


@app.get("/metadata")
def metadata(request: Request) -> dict[str, object]:
    """Return information about the loaded model."""

    predictor = request.app.state.predictor

    return {
        **predictor.metadata,
        "artifact_hash": request.app.state.artifact_hash,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Predict trip duration for one trip."""

    predictor: DurationPredictor = request.app.state.predictor

    features = payload.model_dump()

    start_time = perf_counter()

    prediction = predictor.predict_one(features)

    latency_ms = (perf_counter() - start_time) * 1000

    model_version = str(
        predictor.metadata.get(
            "model_version",
            "unknown",
        )
    )

    return PredictionResponse(
        prediction=prediction,
        model_version=model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Predict trip duration for multiple trips."""

    predictor: DurationPredictor = request.app.state.predictor

    items = payload.root

    features = [item.model_dump() for item in items]

    start_time = perf_counter()

    predictions = predictor.predict_batch(features)

    total_latency_ms = (perf_counter() - start_time) * 1000

    model_version = str(
        predictor.metadata.get(
            "model_version",
            "unknown",
        )
    )

    correlation_id = correlation_id_var.get()

    responses = [
        PredictionResponse(
            prediction=prediction,
            model_version=model_version,
            correlation_id=correlation_id,
            latency_ms=total_latency_ms,
        )
        for prediction in predictions
    ]

    return BatchPredictionResponse(root=responses)
