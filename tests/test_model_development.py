import numpy as np
import pandas as pd

from src.model_development import build_model_searches


def make_features() -> pd.DataFrame:
	row_ids = np.arange(12)
	return pd.DataFrame(
		{
			"age": 30 + row_ids * 3,
			"sex": row_ids % 2,
			"cp": row_ids % 4 + 1,
			"trestbps": 110 + row_ids * 3,
			"chol": 180 + row_ids * 7,
			"fbs": row_ids % 2,
			"restecg": row_ids % 3,
			"thalach": 190 - row_ids * 5,
			"exang": row_ids % 2,
			"oldpeak": row_ids / 10,
			"slope": row_ids % 3 + 1,
			"ca": [0, 1, 2, 3, np.nan, 1, 2, 3, 0, 1, 2, 3],
			"thal": [3, np.nan, 7, 3, 6, 7, 3, 6, 7, 3, 6, 7],
		}
	)


def test_both_model_pipelines_fit_and_predict_with_raw_features() -> None:
	features = make_features()
	target = pd.Series([0] * 6 + [1] * 6)
	model_searches = build_model_searches()

	assert set(model_searches) == {"Logistic Regression", "Random Forest"}
	for pipeline, parameter_grid in model_searches.values():
		assert "classifier__class_weight" in parameter_grid
		pipeline.fit(features, target)
		predictions = pipeline.predict(features.assign(cp=99))

		assert len(predictions) == len(features)
		assert set(predictions).issubset({0, 1})