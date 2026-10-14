import numpy as np
import pandas as pd

from src.data_preprocessing import COLUMNS, clean_data, encode_features


def make_raw_frame() -> pd.DataFrame:
	return pd.DataFrame(
		[
			[40, 1, 1, 120, 200, 0, 0, 180, 0, 0.0, 1, 0, 3, 0],
			[60, 0, 2, 140, 240, 1, 1, 150, 1, 1.0, 2, 2, 7, 1],
			[np.nan, 1, 2, 130, 220, 0, 2, 160, 0, 0.5, 2, np.nan, np.nan, 2],
			[50, 0, 3, 125, 210, 0, 0, 170, 0, 0.2, 1, 2, 3, 4],
		],
		columns=COLUMNS,
	)


def test_clean_data_imputes_features_and_binarizes_target() -> None:
	raw_data = make_raw_frame()

	cleaned = clean_data(raw_data)

	assert cleaned.loc[2, "age"] == 50
	assert cleaned.loc[2, "ca"] == 2
	assert cleaned.loc[2, "thal"] == 3
	assert cleaned["target"].tolist() == [0, 1, 1, 1]
	assert not cleaned.isna().any().any()
	assert np.isnan(raw_data.loc[2, "age"])


def test_encode_features_one_hot_encodes_categories() -> None:
	encoded = encode_features(clean_data(make_raw_frame()))

	assert "cp_1" in encoded.columns
	assert "cp_2" in encoded.columns
	assert "thal_7" in encoded.columns
	assert encoded.columns[-1] == "target"
	assert encoded.select_dtypes(exclude="number").empty
	assert encoded["target"].tolist() == [0, 1, 1, 1]