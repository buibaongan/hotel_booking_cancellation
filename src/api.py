import configparser
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.preprocessing import ColumnCleaner


DEFAULT_MODEL_PATH = "models/best_model.pkl"
DEFAULT_PREPROCESSOR_PATH = "reports/preprocessor.joblib"
DEFAULT_CONFIG_PATH = "config.ini"


class PredictionRequest(BaseModel):
    booking: Dict[str, Any] = Field(
        ...,
        description="One booking record. Use raw booking fields when a preprocessor is available, otherwise processed model features.",
    )


class BatchPredictionRequest(BaseModel):
    bookings: List[Dict[str, Any]] = Field(..., description="List of booking records.")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: Optional[float]
    risk_level: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class ModelService:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.config_path = config_path
        self.model = self._load_required_artifact(model_path, "model")
        self.preprocessor = self._load_optional_artifact(preprocessor_path)
        self.expected_features = self._resolve_expected_features()

    def predict_records(self, records: List[Dict[str, Any]], processed: bool = False) -> List[PredictionResponse]:
        if not records:
            raise HTTPException(status_code=400, detail="At least one booking record is required.")

        frame = pd.DataFrame(records)
        features = self._prepare_features(frame, processed=processed)
        predictions = self.model.predict(features)
        probabilities = self._predict_probabilities(features)

        responses = []
        for index, prediction in enumerate(predictions):
            prediction_int = int(prediction)
            probability = probabilities[index] if probabilities is not None else None
            responses.append(
                PredictionResponse(
                    prediction=prediction_int,
                    label="Canceled" if prediction_int == 1 else "Not canceled",
                    probability=probability,
                    risk_level=self._risk_level(probability, prediction_int),
                )
            )
        return responses

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "preprocessor_path": self.preprocessor_path if self.preprocessor is not None else None,
            "preprocessor_loaded": self.preprocessor is not None,
            "expected_feature_count": len(self.expected_features),
            "expected_features": self.expected_features,
        }

    def _prepare_features(self, frame: pd.DataFrame, processed: bool) -> pd.DataFrame:
        if self.preprocessor is not None and not processed:
            self._validate_raw_features(frame)
            return self.preprocessor.transform(frame)

        non_numeric = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        if non_numeric:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Raw categorical fields require reports/preprocessor.joblib. "
                    "Either provide the saved preprocessor or send processed numeric features with processed=true. "
                    f"Non-numeric fields received: {non_numeric}"
                ),
            )

        features = frame.copy()
        if self.expected_features:
            missing_features = [feature for feature in self.expected_features if feature not in features.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Processed requests must include every model feature. "
                        "Call /metadata for the expected feature list. "
                        f"Missing features: {missing_features}"
                    ),
                )
            features = features.reindex(columns=self.expected_features, fill_value=0)
        return features.astype(float)

    def _validate_raw_features(self, frame: pd.DataFrame) -> None:
        required_columns = getattr(self.preprocessor, "required_raw_columns_", None)
        if not required_columns:
            return
        received_columns = {ColumnCleaner.clean_name(column) for column in frame.columns}
        missing_columns = [column for column in required_columns if column not in received_columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Raw requests must include the columns used when the preprocessor was fitted. "
                    f"Missing columns: {missing_columns}"
                ),
            )

    def _predict_probabilities(self, features: pd.DataFrame) -> Optional[List[float]]:
        if not hasattr(self.model, "predict_proba"):
            return None
        probabilities = self.model.predict_proba(features)
        if probabilities.shape[1] == 1:
            return [float(value) for value in probabilities[:, 0]]
        return [float(value) for value in probabilities[:, 1]]

    def _resolve_expected_features(self) -> List[str]:
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        if self.preprocessor is not None and getattr(self.preprocessor, "feature_columns_", None):
            return list(self.preprocessor.feature_columns_)

        config = configparser.ConfigParser()
        config.read(self.config_path, encoding="utf-8")
        if config.has_option("DATA", "features"):
            return [feature.strip() for feature in config["DATA"]["features"].split(",") if feature.strip()]
        return []

    @staticmethod
    def _risk_level(probability: Optional[float], prediction: int) -> str:
        if probability is None:
            return "High" if prediction == 1 else "Low"
        if probability >= 0.7:
            return "High"
        if probability >= 0.4:
            return "Medium"
        return "Low"

    @staticmethod
    def _load_required_artifact(path: str, name: str) -> Any:
        if not os.path.exists(path):
            raise RuntimeError(f"Required {name} artifact not found: {path}")
        return joblib.load(path)

    @staticmethod
    def _load_optional_artifact(path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        return joblib.load(path)


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    return ModelService(
        model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        preprocessor_path=os.getenv("PREPROCESSOR_PATH", DEFAULT_PREPROCESSOR_PATH),
        config_path=os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH),
    )


app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    version="1.0.0",
    description="Predicts whether a hotel booking is likely to be canceled.",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        service = get_service()
        return {"status": "ok", "model_loaded": service.model is not None}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    try:
        return get_service().metadata()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    processed: bool = Query(False, description="Set true when sending already processed numeric model features."),
) -> PredictionResponse:
    try:
        return get_service().predict_records([request.booking], processed=processed)[0]
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    processed: bool = Query(False, description="Set true when sending already processed numeric model features."),
) -> BatchPredictionResponse:
    try:
        predictions = get_service().predict_records(request.bookings, processed=processed)
        return BatchPredictionResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
