# MLOps Heart Disease: Setup 1

## Data source

This setup uses the processed Cleveland subset of the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The source has 14 fields and represents disease severity in `num` as values `0` through `4`.

## Reproduce acquisition and cleaning

From this project directory, create and activate a virtual environment, then install the declared dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and validate the raw source, then create the cleaned CSV:

```powershell
python -m src.data_acquisition
python -m src.data_preprocessing
```

The acquisition script saves `data/raw/processed.cleveland.data`. The preprocessing script converts `?` to missing values, imputes continuous features with medians and categorical features with modes, maps `num = 0` to no disease and `num > 0` to disease present, and one-hot encodes categorical predictors. It writes `data/processed/heart_disease_clean.csv`.

## Run the EDA

Open `notebooks/01_eda.ipynb`, select the project's `.venv` Python kernel, and run all cells. The notebook audits the source data and saves `reports/figures/class_balance.png`, `reports/figures/feature_histograms.png`, and `reports/figures/correlation_heatmap.png`.

The notebook's full-dataset imputations are for this EDA artifact. For the later modeling task, fit imputation and encoding steps on training folds only to avoid data leakage.

## Step 2: Feature engineering and model development

Install the updated `requirements.txt`, then run from the project directory:

```powershell
python -m src.model_development
```

The training script reads the raw source rather than the full-dataset-imputed EDA CSV. Each model is a scikit-learn pipeline: numeric features receive median imputation and standard scaling; categorical features receive most-frequent imputation and one-hot encoding. Because preprocessing is inside each pipeline, imputers, scalers, and encoders are refit only on each training fold.

Model comparison uses nested stratified cross-validation: four-fold inner CV tunes hyperparameters by ROC-AUC, while five-fold outer CV estimates accuracy, precision, recall, and ROC-AUC. The outer-fold means and standard deviations, fold-specific tuning choices, and final full-data search parameters are saved to `reports/model_comparison.csv`. Logistic Regression searches `C` in `{0.1, 1, 10}` and class weighting; Random Forest searches maximum depth, minimum leaf size, and class weighting with 200 trees. The candidate with the highest mean outer-fold ROC-AUC is selected for the next assignment stage. These metrics are cross-validation estimates, not an independent external validation result.

## Step 3: Experiment tracking

Running `python -m src.model_development` now logs one MLflow run per classifier to a local SQLite database (`mlflow.db`), with file artifacts under `mlruns/`. Each run records the model and CV configuration, tuning search space, selected parameters, accuracy/precision/recall/ROC-AUC means and standard deviations, and artifacts including the comparison CSV, tuning-details JSON, EDA figures (when present), and an outer-CV metric plot. The run selected by mean outer-fold ROC-AUC is tagged in MLflow. Generated tracking data is ignored by Git.

To browse the experiment locally, run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`.

## Step 4: Model packaging and reproducibility

The Step 2 training command refits each candidate on all available raw rows using its inner-CV-selected hyperparameters, then logs the candidate with the highest mean outer-fold ROC-AUC as an MLflow `heart_disease_pipeline` model. The saved sklearn pipeline includes the fitted imputers, `StandardScaler`, `OneHotEncoder`, and classifier. It uses MLflow's skops serialization with an explicit `numpy.dtype` trusted-type allowlist. MLflow stores the model signature, a raw-feature input example, and pinned runtime requirements alongside the model artifact. The same standalone package is saved in a versioned `models/` directory, so it can be loaded without the tracking database. The CLI prints the MLflow model URI and local model path; both are saved as tags on the selected run.

Load the packaged model with the URI printed by training (or copied from the selected MLflow run):

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models/<printed-model-directory>")
predictions = model.predict(raw_feature_dataframe)
```

Pass a pandas DataFrame with the original 13 predictor columns (without `num` or `target`). The fitted preprocessing pipeline handles missing values and categorical encoding at prediction time. Install the project environment from `requirements.txt` before loading the model.

## Step 5: CI/CD and automated testing

Install the test and lint tools, then run the quality checks locally:

```powershell
python -m pip install -r requirements-dev.txt
ruff check src tests app.py
python -m pytest -q
```

The unit tests cover missing-value imputation, binary target creation, one-hot encoding, and fit/predict behavior for both model pipelines, including missing and unseen categorical values. The GitHub Actions workflow in `.github/workflows/ci.yml` runs linting and tests, acquires the dataset, executes model training with MLflow tracking, and uploads the comparison CSV, model packages, SQLite tracking database, and MLflow artifacts for every workflow run.

## Step 6: Model containerization

After training has written `models/current_model.txt`, build and run the API container from the project directory:

```powershell
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

The FastAPI service loads the versioned sklearn pipeline selected by the pointer. `POST /predict` accepts one JSON object with the original 13 feature names and returns the binary prediction, predicted label, predicted-class probability as confidence, and both class probabilities. `GET /health` reports whether the model loaded. The image uses the model package's pinned requirements and Python 3.14.7, matching the saved model environment.

Example PowerShell request:

```powershell
$body = @{
	age = 63; sex = 1; cp = 1; trestbps = 145; chol = 233; fbs = 1
	restecg = 2; thalach = 150; exang = 0; oldpeak = 2.3; slope = 3
	ca = 0; thal = 6
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
```

The GitHub Actions workflow also builds the image, starts it, and smoke-tests `/predict`. The model input JSON must include all 13 feature keys; values may be `null` where a measurement is missing.

## Step 7: Kubernetes deployment

The Kubernetes manifest in `k8s/heart-disease-api.yaml` creates a one-replica Deployment and a `LoadBalancer` Service in the `heart-disease` namespace. It includes startup, readiness, and liveness probes, resource requests/limits, a non-root read-only container, and temporary storage for `/tmp`.

Prerequisites: Docker Engine and a running Minikube cluster, or Docker Desktop with Kubernetes enabled. The local image must be built from the project root and made available to the cluster:

```powershell
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
kubectl apply -f k8s/heart-disease-api.yaml
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=180s
kubectl get pods,services -n heart-disease
kubectl create secret generic grafana-admin --namespace heart-disease --from-literal=admin-password='REPLACE_WITH_A_STRONG_PASSWORD' --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring.yaml
kubectl rollout status deployment/prometheus -n heart-disease --timeout=180s
kubectl rollout status deployment/grafana -n heart-disease --timeout=180s
```

For Minikube, start `minikube tunnel` in a second terminal and keep it running. Wait for `kubectl get service heart-disease-api -n heart-disease` to show an external address. On Docker Desktop, use the external address reported by the local cluster. Verify the exposed endpoints (replace the address if your cluster reports a different one):

```powershell
$baseUrl = "http://127.0.0.1"
Invoke-RestMethod -Uri "$baseUrl/health"
$body = @{
	age = 63; sex = 1; cp = 1; trestbps = 145; chol = 233; fbs = 1
	restecg = 2; thalach = 150; exang = 0; oldpeak = 2.3; slope = 3
	ca = 0; thal = 6
} | ConvertTo-Json
Invoke-RestMethod -Uri "$baseUrl/predict" -Method Post -ContentType "application/json" -Body $body
```

If the local cluster does not assign a directly reachable LoadBalancer address, use `kubectl port-forward -n heart-disease service/heart-disease-api 8000:80` and change `$baseUrl` to `http://127.0.0.1:8000` for endpoint verification. Use a non-placeholder strong password for Grafana; do not commit the generated Secret. The screenshot evidence checklist is in `reports/screenshots/README.md`. No deployment screenshots are included until a cluster has actually been run.

## Step 8: Monitoring and logging

The API writes one-line JSON access/error logs to standard output. Events include request ID, method, path, status, and latency; request bodies are never logged. Kubernetes makes these available through `kubectl logs -n heart-disease deployment/heart-disease-api`. Prometheus request and latency metrics are exposed at `/metrics`; the CI smoke test checks that endpoint.

`k8s/monitoring.yaml` deploys Prometheus with a 15-second scrape interval and provisions Grafana with a dashboard for request rate, 95th-percentile latency, status-class rates, and API target health. Prometheus and Grafana are ClusterIP-only. To inspect them locally:

```powershell
kubectl port-forward -n heart-disease service/grafana 3000:3000
```

Open `http://127.0.0.1:3000`, sign in as `admin` with the password used to create the `grafana-admin` Secret, and open the **Heart Disease API** dashboard. Generate a few `/health` and `/predict` requests before checking the panels. To query Prometheus directly, use `kubectl port-forward -n heart-disease service/prometheus 9090:9090` and open `http://127.0.0.1:9090`. Monitoring storage uses `emptyDir` for this local demonstration and is not durable across pod replacement.

## Step 9: Final report

See [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md) for the consolidated methods, results, architecture diagram, CI/CD summary, repository status, and deployment evidence checklist. The project has no configured Git remote yet; add the published repository URL before submission. Live cluster screenshots and deployment URLs must be added after deploying on a machine with Docker and Kubernetes available.
