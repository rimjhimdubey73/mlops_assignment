"""Clean and one-hot encode the UCI Cleveland heart-disease dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"

COLUMNS = [
	"age",
	"sex",
	"cp",
	"trestbps",
	"chol",
	"fbs",
	"restecg",
	"thalach",
	"exang",
	"oldpeak",
	"slope",
	"ca",
	"thal",
	"num",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
	"sex",
	"cp",
	"fbs",
	"restecg",
	"exang",
	"slope",
	"ca",
	"thal",
]


def read_raw_data(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
	"""Read the headerless source file and convert its '?' markers to NA."""
	return pd.read_csv(
		path,
		header=None,
		names=COLUMNS,
		na_values="?",
		skipinitialspace=True,
	)


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
	"""Impute missing features, retain categorical codes, and binarize the target."""
	data = raw_data.copy()
	for column in COLUMNS:
		data[column] = pd.to_numeric(data[column], errors="coerce")

	data = data.dropna(subset=["num"])
	if data.empty:
		raise ValueError("No rows with a valid target remain after loading the source data")
	if not data["num"].isin([0, 1, 2, 3, 4]).all():
		raise ValueError("The Cleveland target must contain values from 0 through 4")

	for column in NUMERIC_FEATURES:
		median = data[column].median()
		if pd.isna(median):
			raise ValueError(f"Cannot impute {column}: the column contains no numeric values")
		data[column] = data[column].fillna(median)

	for column in CATEGORICAL_FEATURES:
		modes = data[column].mode(dropna=True)
		if modes.empty:
			raise ValueError(f"Cannot impute {column}: the column contains no valid categories")
		data[column] = data[column].fillna(modes.iloc[0]).astype("int64")

	data["target"] = (data.pop("num") > 0).astype("int64")
	return data


def encode_features(clean_data_frame: pd.DataFrame) -> pd.DataFrame:
	"""One-hot encode categorical predictors while leaving continuous values intact."""
	encoded = pd.get_dummies(
		clean_data_frame,
		columns=CATEGORICAL_FEATURES,
		prefix=CATEGORICAL_FEATURES,
		dtype="int64",
	)
	target = encoded.pop("target")
	encoded["target"] = target
	return encoded


def save_processed_data(data: pd.DataFrame, path: Path = DEFAULT_PROCESSED_PATH) -> Path:
	"""Write encoded, model-ready data as a CSV and return its path."""
	path.parent.mkdir(parents=True, exist_ok=True)
	data.to_csv(path, index=False)
	return path


def preprocess_dataset(
	raw_path: Path = DEFAULT_RAW_PATH,
	output_path: Path = DEFAULT_PROCESSED_PATH,
) -> pd.DataFrame:
	"""Load, clean, encode, and save the dataset using the default project schema."""
	cleaned = clean_data(read_raw_data(raw_path))
	encoded = encode_features(cleaned)
	save_processed_data(encoded, output_path)
	return encoded


def main() -> None:
	processed = preprocess_dataset()
	print(
		f"Saved {len(processed)} cleaned rows and {processed.shape[1] - 1} encoded predictors "
		f"to {DEFAULT_PROCESSED_PATH}"
	)


if __name__ == "__main__":
	main()
