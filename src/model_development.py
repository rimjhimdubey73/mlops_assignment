"""Tune and compare two classifiers using nested stratified cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import (
	CATEGORICAL_FEATURES,
	DEFAULT_RAW_PATH,
	NUMERIC_FEATURES,
	read_raw_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT_NAME = "heart-disease-model-development"
EDA_FIGURES = [
	PROJECT_ROOT / "reports" / "figures" / "class_balance.png",
	PROJECT_ROOT / "reports" / "figures" / "feature_histograms.png",
	PROJECT_ROOT / "reports" / "figures" / "correlation_heatmap.png",
]
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
SCORING = {
	"accuracy": "accuracy",
	"precision": "precision",
	"recall": "recall",
	"roc_auc": "roc_auc",
}


def build_preprocessor() -> ColumnTransformer:
	"""Impute, scale continuous features, and impute/one-hot encode categories."""
	numeric_pipeline = Pipeline(
		[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	return ColumnTransformer(
		[
			("numeric", numeric_pipeline, NUMERIC_FEATURES),
			("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
		]
	)


def build_model_searches() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
	"""Return model pipelines and compact, explicit hyperparameter search spaces."""
	preprocessor = build_preprocessor()
	return {
		"Logistic Regression": (
			Pipeline(
				[
					("preprocessor", preprocessor),
					(
						"classifier",
						LogisticRegression(
							max_iter=2000,
							solver="liblinear",
							random_state=RANDOM_STATE,
						),
					),
				]
			),
			{
				"classifier__C": [0.1, 1.0, 10.0],
				"classifier__class_weight": [None, "balanced"],
			},
		),
		"Random Forest": (
			Pipeline(
				[
					("preprocessor", preprocessor),
					(
						"classifier",
						RandomForestClassifier(
							n_estimators=200,
							random_state=RANDOM_STATE,
							n_jobs=1,
						),
					),
				]
			),
			{
				"classifier__max_depth": [None, 6],
				"classifier__min_samples_leaf": [1, 3],
				"classifier__class_weight": [None, "balanced"],
			},
		),
	}


def _log_tracking_runs(
	comparison: pd.DataFrame,
	search_spaces: dict[str, dict[str, list[object]]],
	fitted_models: dict[str, Pipeline],
	features: pd.DataFrame,
	results_path: Path,
	raw_path: Path,
	tracking_uri: str,
	experiment_name: str,
	row_count: int,
) -> None:
	"""Log candidate results and package the outer-CV-selected full pipeline."""
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)
	selected_model = str(comparison.iloc[0]["model"])

	for _, result in comparison.iterrows():
		model_name = str(result["model"])
		best_params = json.loads(result["best_params_full_data_search"])
		fold_params = json.loads(result["outer_fold_best_params"])
		with mlflow.start_run(run_name=model_name):
			mlflow.set_tags(
				{
					"model_family": model_name,
					"selected_by_outer_roc_auc": str(model_name == selected_model).lower(),
				}
			)
			mlflow.log_params(
				{
					"model": model_name,
					"tuning_metric": "roc_auc",
					"outer_cv": f"{OUTER_FOLDS}-fold stratified",
					"inner_cv": f"{INNER_FOLDS}-fold stratified",
					"random_state": RANDOM_STATE,
					"training_rows": row_count,
					"raw_dataset": raw_path.name,
				}
			)
			for parameter_name, parameter_value in best_params.items():
				mlflow.log_param(f"best_{parameter_name}", parameter_value)
			mlflow.log_metrics(
				{
					f"{metric_name}_{statistic}": float(result[f"{metric_name}_{statistic}"])
					for metric_name in SCORING
					for statistic in ("mean", "std")
				}
			)
			mlflow.log_dict(
				{
					"search_space": search_spaces[model_name],
					"best_params_full_data_search": best_params,
					"best_params_by_outer_fold": fold_params,
					"selection_rule": "highest mean outer-fold ROC-AUC",
				},
				"parameters/tuning_details.json",
			)
			mlflow.log_artifact(str(results_path), artifact_path="comparison")
			for figure_path in EDA_FIGURES:
				if figure_path.is_file():
					mlflow.log_artifact(str(figure_path), artifact_path="eda_figures")

			metric_names = ["accuracy", "precision", "recall", "roc_auc"]
			metric_labels = ["Accuracy", "Precision", "Recall", "ROC-AUC"]
			metric_means = [float(result[f"{name}_mean"]) for name in metric_names]
			metric_stds = [float(result[f"{name}_std"]) for name in metric_names]
			figure, axis = plt.subplots(figsize=(7.5, 4.5))
			bars = axis.bar(
				metric_labels,
				metric_means,
				yerr=metric_stds,
				capsize=4,
				color=["#367A72", "#D07B56", "#7B9E9A", "#C65F45"],
			)
			axis.bar_label(bars, fmt="%.3f", padding=4)
			axis.set_ylim(0, 1.05)
			axis.set_ylabel("Outer-fold mean (error bars: standard deviation)")
			axis.set_title(f"{model_name} | Nested CV Metrics")
			axis.grid(axis="y", alpha=0.25)
			axis.set_axisbelow(True)
			figure.tight_layout()
			mlflow.log_figure(figure, "plots/outer_cv_metrics.png")
			plt.close(figure)

			if model_name == selected_model:
				final_model = fitted_models[model_name]
				input_example = features.head(5)
				model_signature = infer_signature(
					input_example,
					final_model.predict(input_example),
				)
				model_requirements = mlflow_sklearn.get_default_pip_requirements() + [
					f"mlflow=={mlflow.__version__}",
					f"numpy=={np.__version__}",
					f"pandas=={pd.__version__}",
				]
				model_info = mlflow_sklearn.log_model(
					sk_model=final_model,
					name="heart_disease_pipeline",
					skops_trusted_types=["numpy.dtype"],
					input_example=input_example,
					signature=model_signature,
					pip_requirements=model_requirements,
				)
				local_model_path = (
					PROJECT_ROOT / "models" / model_info.model_uri.rsplit("/", 1)[-1]
				)
				mlflow_sklearn.save_model(
					sk_model=final_model,
					path=str(local_model_path),
					skops_trusted_types=["numpy.dtype"],
					input_example=input_example,
					signature=model_signature,
					pip_requirements=model_requirements,
				)
				(PROJECT_ROOT / "models" / "current_model.txt").write_text(
					f"{local_model_path.name}\n",
					encoding="utf-8",
				)
				mlflow.set_tag("model_uri", model_info.model_uri)
				mlflow.set_tag(
					"local_model_path",
					local_model_path.relative_to(PROJECT_ROOT).as_posix(),
				)
				mlflow.log_dict(
					{
						"model_uri": model_info.model_uri,
						"local_model_path": local_model_path.relative_to(
							PROJECT_ROOT
						).as_posix(),
						"input_features": list(features.columns),
						"prediction_target": "target (0=no disease, 1=disease present)",
						"preprocessing": "median imputation and standard scaling for numeric features; "
						"most-frequent imputation and one-hot encoding for categorical features",
						"model_environment": model_requirements,
					},
					"model/serving_contract.json",
				)
				print(f"Packaged selected pipeline as MLflow model: {model_info.model_uri}")
				print(
					"Saved standalone model package: "
					f"{local_model_path.relative_to(PROJECT_ROOT).as_posix()}"
				)


def run_model_comparison(
	raw_path: Path = DEFAULT_RAW_PATH,
	results_path: Path = DEFAULT_RESULTS_PATH,
	tracking_uri: str = DEFAULT_TRACKING_URI,
	experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> pd.DataFrame:
	"""Tune both pipelines, save CV metrics, and log one MLflow run per candidate."""
	raw_data = read_raw_data(raw_path)
	features = raw_data.drop(columns="num")
	target = (raw_data["num"] > 0).astype("int64")
	if target.nunique() != 2:
		raise ValueError("Model development requires both binary target classes")

	outer_cv = StratifiedKFold(
		n_splits=OUTER_FOLDS,
		shuffle=True,
		random_state=RANDOM_STATE,
	)
	inner_cv = StratifiedKFold(
		n_splits=INNER_FOLDS,
		shuffle=True,
		random_state=RANDOM_STATE,
	)
	results: list[dict[str, object]] = []
	fitted_models: dict[str, Pipeline] = {}

	model_searches = build_model_searches()
	search_spaces = {
		model_name: parameter_grid
		for model_name, (_, parameter_grid) in model_searches.items()
	}
	for model_name, (pipeline, parameter_grid) in model_searches.items():
		search = GridSearchCV(
			estimator=pipeline,
			param_grid=parameter_grid,
			scoring="roc_auc",
			cv=inner_cv,
			n_jobs=1,
			refit=True,
			error_score="raise",
		)
		outer_scores = cross_validate(
			search,
			features,
			target,
			cv=outer_cv,
			scoring=SCORING,
			return_estimator=True,
			n_jobs=1,
			error_score="raise",
		)
		final_search = GridSearchCV(
			estimator=pipeline,
			param_grid=parameter_grid,
			scoring="roc_auc",
			cv=inner_cv,
			n_jobs=1,
			refit=True,
			error_score="raise",
		).fit(features, target)
		fitted_models[model_name] = final_search.best_estimator_

		row: dict[str, object] = {
			"model": model_name,
			"tuning_metric": "roc_auc",
			"best_params_full_data_search": json.dumps(final_search.best_params_, sort_keys=True),
			"outer_fold_best_params": json.dumps(
				[fold_search.best_params_ for fold_search in outer_scores["estimator"]],
				sort_keys=True,
			),
		}
		for metric_name in SCORING:
			fold_scores = outer_scores[f"test_{metric_name}"]
			row[f"{metric_name}_mean"] = float(fold_scores.mean())
			row[f"{metric_name}_std"] = float(fold_scores.std(ddof=1))
		results.append(row)

	comparison = pd.DataFrame(results).sort_values(
		"roc_auc_mean", ascending=False, ignore_index=True
	)
	results_path.parent.mkdir(parents=True, exist_ok=True)
	comparison.to_csv(results_path, index=False)
	_log_tracking_runs(
		comparison,
		search_spaces,
		fitted_models,
		features,
		results_path,
		raw_path,
		tracking_uri,
		experiment_name,
		len(target),
	)
	return comparison


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--raw-data",
		type=Path,
		default=DEFAULT_RAW_PATH,
		help="path to processed.cleveland.data from Setup 1",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_RESULTS_PATH,
		help="path for the cross-validation comparison CSV",
	)
	parser.add_argument(
		"--tracking-uri",
		default=DEFAULT_TRACKING_URI,
		help="MLflow tracking URI (default: local SQLite database)",
	)
	parser.add_argument(
		"--experiment-name",
		default=DEFAULT_EXPERIMENT_NAME,
		help="MLflow experiment name",
	)
	args = parser.parse_args()
	comparison = run_model_comparison(
		args.raw_data,
		args.output,
		args.tracking_uri,
		args.experiment_name,
	)
	print(comparison.to_string(index=False))
	print(f"\nSaved cross-validation results to {args.output}")
	print(f"Selected by mean outer-fold ROC-AUC: {comparison.iloc[0]['model']}")


if __name__ == "__main__":
	main()