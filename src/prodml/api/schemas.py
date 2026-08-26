from pydantic import BaseModel, ConfigDict, Field, RootModel


class PredictionRequest(BaseModel):
    """Input data for a single prediction."""

    PU_DO: str = Field(
        min_length=3,
        examples=["74_236"],
    )

    trip_distance: float = Field(
        gt=0,
        lt=200,
        examples=[2.5],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PU_DO": "74_236",
                "trip_distance": 2.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response returned for a single prediction."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float = Field(ge=0)


class BatchPredictionRequest(RootModel[list[PredictionRequest]]):
    """Input data for batch prediction."""


class BatchPredictionResponse(RootModel[list[PredictionResponse]]):
    """Response returned for batch prediction."""
