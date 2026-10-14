"""Download the UCI Cleveland heart-disease dataset.

Run from the project root with ``python -m src.data_acquisition``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE_URL = (
	"https://archive.ics.uci.edu/ml/machine-learning-databases/"
	"heart-disease/processed.cleveland.data"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
EXPECTED_COLUMNS = 14
MINIMUM_ROWS = 300


def download_dataset(destination: Path = DEFAULT_DESTINATION) -> Path:
	"""Download and validate the raw Cleveland data before saving it."""
	request = Request(SOURCE_URL, headers={"User-Agent": "mlops-heart-disease-eda/1.0"})
	with urlopen(request, timeout=30) as response:
		content = response.read()

	rows = [row for row in csv.reader(content.decode("utf-8-sig").splitlines()) if row]
	if len(rows) < MINIMUM_ROWS:
		raise ValueError(f"Expected at least {MINIMUM_ROWS} rows, received {len(rows)}")
	if any(len(row) != EXPECTED_COLUMNS for row in rows):
		raise ValueError(f"Expected {EXPECTED_COLUMNS} columns in every row")
	if any(row[-1].strip() not in {"0", "1", "2", "3", "4"} for row in rows):
		raise ValueError("Unexpected disease-severity value in the target column")

	destination.parent.mkdir(parents=True, exist_ok=True)
	destination.write_bytes(content)
	return destination


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_DESTINATION,
		help="destination for the raw dataset (default: data/raw/processed.cleveland.data)",
	)
	args = parser.parse_args()
	saved_path = download_dataset(args.output)
	print(f"Downloaded validated Cleveland dataset to {saved_path}")


if __name__ == "__main__":
	main()
