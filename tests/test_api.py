import json
import logging

import numpy as np
from fastapi.testclient import TestClient

import app as serving_app


class StubModel:
	classes_ = np.array([0, 1])

	def predict(self, features):
		return np.array([1])

	def predict_proba(self, features):
		return np.array([[0.2, 0.8]])


def sample_payload() -> dict[str, float]:
	return {
		"age": 63,
		"sex": 1,
		"cp": 1,
		"trestbps": 145,
		"chol": 233,
		"fbs": 1,
		"restecg": 2,
		"thalach": 150,
		"exang": 0,
		"oldpeak": 2.3,
		"slope": 3,
		"ca": 0,
		"thal": 6,
	}


def test_predict_returns_class_and_confidence(monkeypatch) -> None:
	monkeypatch.setattr(serving_app, "load_serving_model", lambda: StubModel())
	with TestClient(serving_app.app) as client:
		response = client.post("/predict", json=sample_payload())

	assert response.status_code == 200
	assert response.json() == {
		"prediction": 1,
		"label": "disease_present",
		"confidence": 0.8,
		"probabilities": {"no_disease": 0.2, "disease_present": 0.8},
	}


def test_predict_requires_all_feature_fields(monkeypatch) -> None:
	monkeypatch.setattr(serving_app, "load_serving_model", lambda: StubModel())
	payload = sample_payload()
	del payload["age"]
	with TestClient(serving_app.app) as client:
		response = client.post("/predict", json=payload)

	assert response.status_code == 422


def test_metrics_endpoint_exposes_request_counters(monkeypatch) -> None:
	monkeypatch.setattr(serving_app, "load_serving_model", lambda: StubModel())
	with TestClient(serving_app.app) as client:
		client.get("/health")
		response = client.get("/metrics")

	assert response.status_code == 200
	assert "http_requests_total" in response.text


def test_request_logs_include_status_and_request_id_not_payload(
	monkeypatch,
	caplog,
) -> None:
	monkeypatch.setattr(serving_app, "load_serving_model", lambda: StubModel())
	caplog.set_level(logging.INFO, logger="heart_disease_api")
	payload = sample_payload()
	with TestClient(serving_app.app) as client:
		response = client.post("/predict", json=payload)

	log_entry = next(
		record.message
		for record in caplog.records
		if record.name == "heart_disease_api" and '"event": "api_request"' in record.message
	)
	log_data = json.loads(log_entry)
	assert log_data["status_code"] == 200
	assert response.headers["x-request-id"] == log_data["request_id"]
	assert str(payload["chol"]) not in log_entry