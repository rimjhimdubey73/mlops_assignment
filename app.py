"""FastAPI service for the packaged heart-disease classifier."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from mlflow import sklearn as mlflow_sklearn
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from src.data_preprocessing import COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = COLUMNS[:-1]
CLASS_LABELS = {0: "no_disease", 1: "disease_present"}
logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO").upper(),
	format="%(message)s",
)
request_logger = logging.getLogger("heart_disease_api")


class HeartDiseaseInput(BaseModel):
	model_config = ConfigDict(extra="forbid")

	age: float | None
	sex: float | None
	cp: float | None
	trestbps: float | None
	chol: float | None
	fbs: float | None
	restecg: float | None
	thalach: float | None
	exang: float | None
	oldpeak: float | None
	slope: float | None
	ca: float | None
	thal: float | None


class PredictionResponse(BaseModel):
	prediction: Literal[0, 1]
	label: Literal["no_disease", "disease_present"]
	confidence: float = Field(ge=0.0, le=1.0)
	probabilities: dict[str, float]


def resolve_model_path() -> Path:
	"""Resolve an explicit model path or the version written by training."""
	explicit_path = os.getenv("MODEL_PATH")
	if explicit_path:
		model_path = Path(explicit_path).expanduser()
		if not model_path.is_absolute():
			model_path = PROJECT_ROOT / model_path
		model_path = model_path.resolve()
	else:
		model_root = Path(
			os.getenv("MODEL_ROOT", str(PROJECT_ROOT / "models"))
		).expanduser()
		if not model_root.is_absolute():
			model_root = PROJECT_ROOT / model_root
		model_root = model_root.resolve()
		pointer_path = model_root / "current_model.txt"
		if not pointer_path.is_file():
			raise FileNotFoundError(f"Model pointer not found: {pointer_path}")
		model_name = pointer_path.read_text(encoding="utf-8").strip()
		if not model_name or Path(model_name).name != model_name:
			raise ValueError("current_model.txt must contain a model directory name")
		model_path = (model_root / model_name).resolve()
		if model_path.parent != model_root:
			raise ValueError("The model pointer must refer to a directory under MODEL_ROOT")

	if not (model_path / "MLmodel").is_file():
		raise FileNotFoundError(f"MLflow model metadata not found under {model_path}")
	return model_path


def load_serving_model():
	return mlflow_sklearn.load_model(str(resolve_model_path()))


@asynccontextmanager
async def lifespan(application: FastAPI):
	application.state.model = load_serving_model()
	yield


app = FastAPI(
	title="Heart Disease Prediction API",
	version="1.0.0",
	lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def log_api_request(request: Request, call_next):
	request_id = uuid.uuid4().hex
	started_at = time.perf_counter()
	try:
		response = await call_next(request)
	except Exception:
		request_logger.exception(
			json.dumps(
				{
					"event": "api_request_failed",
					"request_id": request_id,
					"method": request.method,
					"path": request.url.path,
					"duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
				}
			)
		)
		raise

	response.headers["X-Request-ID"] = request_id
	request_logger.info(
		json.dumps(
			{
				"event": "api_request",
				"request_id": request_id,
				"method": request.method,
				"path": request.url.path,
				"status_code": response.status_code,
				"duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
			}
		)
	)
	return response


@app.get("/health")
def health(request: Request) -> dict[str, str]:
	return {"status": "ok", "model_loaded": str(hasattr(request.app.state, "model")).lower()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseInput, request: Request) -> PredictionResponse:
	model = request.app.state.model
	features = pd.DataFrame(
		[{column: getattr(payload, column) for column in FEATURE_COLUMNS}],
		columns=FEATURE_COLUMNS,
	)
	prediction = int(model.predict(features)[0])
	class_probabilities = {
		int(class_value): float(probability)
		for class_value, probability in zip(
			model.classes_, model.predict_proba(features)[0], strict=True
		)
	}
	probabilities = {
		CLASS_LABELS[class_value]: class_probabilities[class_value]
		for class_value in CLASS_LABELS
	}
	return PredictionResponse(
		prediction=prediction,
		label=CLASS_LABELS[prediction],
		confidence=class_probabilities[prediction],
		probabilities=probabilities,
	)