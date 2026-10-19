"""Build the ten-page assignment report PDF from repository artifacts."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
OUTPUT = REPORT_DIR / "FINAL_REPORT.pdf"
SCREENSHOTS = REPORT_DIR / "screenshots"
FIGURES = REPORT_DIR / "figures"

PAGE_W, PAGE_H = A4
MARGIN = 0.58 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor("#173B57"),
        alignment=TA_LEFT,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#476273"),
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#173B57"),
        spaceAfter=9,
    )
)
styles.add(
    ParagraphStyle(
        name="Subhead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#19756F"),
        spaceBefore=5,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.7,
        leading=12,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BulletSmall",
        parent=styles["BodySmall"],
        leftIndent=12,
        firstLineIndent=-8,
        bulletIndent=1,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="CodeSmall",
        fontName="Courier",
        fontSize=7.2,
        leading=9.2,
        leftIndent=5,
        rightIndent=5,
        backColor=colors.HexColor("#F1F5F7"),
        borderColor=colors.HexColor("#D5E0E5"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        parent=styles["BodySmall"],
        fontSize=7.4,
        leading=9,
        textColor=colors.HexColor("#536874"),
        alignment=TA_CENTER,
        spaceAfter=3,
    )
)


def para(text: str, style: str = "BodySmall") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"&#8226; {text}", styles["BulletSmall"])


def code(text: str) -> Preformatted:
    return Preformatted(text.strip(), styles["CodeSmall"], maxLineLength=104)


def image(path: Path, max_width: float, max_height: float) -> Image | None:
    if not path.is_file():
        return None
    result = Image(str(path))
    ratio = min(max_width / result.imageWidth, max_height / result.imageHeight)
    result.drawWidth = result.imageWidth * ratio
    result.drawHeight = result.imageHeight * ratio
    result.hAlign = "CENTER"
    return result


def add_image(story: list, path: Path, caption: str, max_height: float) -> None:
    item = image(path, CONTENT_W, max_height)
    if item is not None:
        story.extend([item, para(caption, "Caption")])


def metrics_table() -> Table:
    data = [
        ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"],
        ["Logistic Regression", "0.858 +/- 0.056", "0.864 +/- 0.060", "0.820 +/- 0.084", "0.920 +/- 0.024"],
        ["Random Forest", "0.828 +/- 0.018", "0.823 +/- 0.059", "0.806 +/- 0.055", "0.916 +/- 0.023"],
    ]
    table = Table(data, colWidths=[1.34 * inch, 1.24 * inch, 1.24 * inch, 1.24 * inch, 1.24 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.3),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F7")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD7DD")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def architecture_diagram() -> Drawing:
    drawing = Drawing(500, 116)
    labels = [
        (8, 59, 93, 40, "UCI data", "acquire / clean"),
        (109, 59, 93, 40, "EDA", "figures / audit"),
        (210, 59, 93, 40, "Nested CV", "tune / compare"),
        (311, 59, 86, 40, "MLflow", "runs / artifacts"),
        (405, 59, 88, 40, "Model", "skops package"),
    ]
    for x, y, width, height, title, subtitle in labels:
        drawing.add(Rect(x, y, width, height, rx=4, ry=4, fillColor=colors.HexColor("#E8F1F4"), strokeColor=colors.HexColor("#46827C"), strokeWidth=1))
        drawing.add(String(x + width / 2, y + 24, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#173B57")))
        drawing.add(String(x + width / 2, y + 10, subtitle, textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#476273")))
    for x1, x2 in [(101, 109), (202, 210), (303, 311), (397, 405)]:
        drawing.add(Line(x1, 79, x2, 79, strokeColor=colors.HexColor("#476273"), strokeWidth=1))
        drawing.add(Polygon([x2, 79, x2 - 4, 82, x2 - 4, 76], fillColor=colors.HexColor("#476273"), strokeColor=colors.HexColor("#476273")))
    drawing.add(Rect(98, 5, 304, 36, rx=4, ry=4, fillColor=colors.HexColor("#F2F6F7"), strokeColor=colors.HexColor("#46827C"), strokeWidth=1))
    drawing.add(String(250, 27, "GitHub Actions: lint -> tests -> train -> build -> smoke test", textAnchor="middle", fontName="Helvetica-Bold", fontSize=7.1, fillColor=colors.HexColor("#173B57")))
    drawing.add(String(250, 14, "Docker API on Minikube -> Prometheus -> Grafana; JSON logs via kubectl", textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#476273")))
    drawing.add(Line(450, 58, 450, 42, strokeColor=colors.HexColor("#476273"), strokeWidth=1))
    drawing.add(Line(450, 42, 402, 42, strokeColor=colors.HexColor("#476273"), strokeWidth=1))
    drawing.add(Polygon([402, 42, 406, 45, 406, 39], fillColor=colors.HexColor("#476273"), strokeColor=colors.HexColor("#476273")))
    return drawing


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D4E0E5"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 0.38 * inch, PAGE_W - MARGIN, PAGE_H - 0.38 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#617681"))
    canvas.drawString(MARGIN, PAGE_H - 0.29 * inch, "AIMLCZG523 | MLOps Assignment 01")
    canvas.drawRightString(PAGE_W - MARGIN, 0.32 * inch, f"Heart Disease Classification | {doc.page}")
    canvas.restoreState()


def build_report() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=0.54 * inch,
        bottomMargin=0.55 * inch,
        title="MLOps Assignment 01 - Heart Disease Prediction",
        author="Rimjhim Dubey (2025AE05367)",
        subject="End-to-end reproducible ML system report",
    )
    story: list = []

    def start_page(title: str, eyebrow: str) -> None:
        story.extend([para(eyebrow.upper(), "Subhead"), para(title, "SectionTitle")])

    start_page("Heart Disease Prediction", "AIMLCZG523 | Assignment 01")
    story.extend(
        [
            para("End-to-end machine learning development, experiment tracking, CI/CD, containerization, local Kubernetes deployment, and monitoring.", "ReportSubtitle"),
            para("Student: Rimjhim Dubey<br/>BITS ID: 2025AE05367<br/>Email: 2025ae05367@wilp.bits-pilani.ac.in<br/>Report date: 19 October 2026<br/>Repository: github.com/rimjhimdubey73/mlops_assignment<br/>Dataset: UCI Heart Disease, processed Cleveland subset", "BodySmall"),
            Spacer(1, 7),
            para("Executive summary", "Subhead"),
            para("This submission builds a binary heart-disease classifier from the processed Cleveland dataset. Numeric and categorical preprocessing is fitted inside stratified cross-validation to avoid leakage. Logistic Regression and Random Forest are compared with nested cross-validation; the Logistic Regression pipeline is selected by mean outer-fold ROC-AUC and packaged with MLflow. A FastAPI service serves predictions and Prometheus metrics, logs structured request metadata, and is deployed with Prometheus and Grafana to a local Minikube cluster.", "BodySmall"),
            para("Verified outcome", "Subhead"),
            bullet("7 tests pass and Ruff reports no issues in the local Python 3.12 environment."),
            bullet("Docker image built successfully; the model-loaded health endpoint and sample prediction returned HTTP 200 in the VM."),
            bullet("API, Prometheus, and Grafana Kubernetes deployments rolled out; Prometheus target was up and the provisioned Grafana dashboard was found."),
            bullet("GitHub Actions screenshot shows a green quality-and-training job with training, container build, smoke test, and artifact upload."),
            Spacer(1, 6),
            para("Scope note", "Subhead"),
            para("This is an educational classifier, not a clinical diagnostic tool. Cross-validation results are estimates on a small processed subset, not evidence of clinical safety or external validity. The local Minikube LoadBalancer has no external IP; access instructions are provided instead of claiming a public production endpoint.", "BodySmall"),
            PageBreak(),
        ]
    )

    start_page("Data acquisition and preparation", "01 | Data")
    story.extend(
        [
            para("Source and target", "Subhead"),
            para("The data acquisition module downloads the processed Cleveland file from the UCI Heart Disease repository and validates its 14-column schema. The original target `num` is an ordinal severity value from 0 to 4. For the assignment's binary classification problem, values 0 are mapped to no disease and values 1 through 4 to disease present.", "BodySmall"),
            para("Cleaning decisions", "Subhead"),
            bullet("The raw file uses `?` for unknown measurements; these values are converted to missing values."),
            bullet("Continuous measurements are median-imputed; categorical measurements are mode-imputed."),
            bullet("Categorical predictors are one-hot encoded. The EDA-only processed CSV has a binary `target` and 28 encoded predictor columns."),
            bullet("The source contains 303 rows, 4 missing `ca` values, and 2 missing `thal` values. Duplicate rows were checked; none were found."),
            para("Reproduction", "Subhead"),
            code("python -m src.data_acquisition\npython -m src.data_preprocessing"),
            para("The raw source is retained separately from the EDA-imputed table. Model training reads the raw features and places all imputation, scaling, and encoding inside sklearn pipelines fitted independently on each training fold.", "BodySmall"),
            Spacer(1, 5),
        ]
    )
    add_image(story, FIGURES / "class_balance.png", "Figure 1. Binary class counts after mapping severity to disease absent/present.", 2.45 * inch)
    story.append(PageBreak())

    start_page("Exploratory analysis", "01 | EDA")
    story.extend(
        [
            para("The notebook audits missingness and class balance, plots the numeric feature distributions, and creates a correlation heatmap from the prepared predictors. EDA imputations are used only to make descriptive figures; the modeling pipeline refits preprocessing in each cross-validation fold to prevent leakage.", "BodySmall"),
            Spacer(1, 5),
        ]
    )
    hist = image(FIGURES / "feature_histograms.png", 3.55 * inch, 4.3 * inch)
    corr = image(FIGURES / "correlation_heatmap.png", 3.55 * inch, 4.3 * inch)
    if hist and corr:
        figure_table = Table([[hist, corr]], colWidths=[3.62 * inch, 3.62 * inch])
        figure_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        story.extend([figure_table, para("Figures 2-3. Numeric distributions (left) and encoded-feature correlation heatmap (right).", "Caption")])
    story.extend(
        [
            Spacer(1, 5),
            para("Interpretation and limits", "Subhead"),
            para("The target classes are moderately balanced: 164 examples (54.13%) without disease and 139 (45.87%) with disease. Age, chest-pain category, maximum heart rate, exercise-induced angina, and ST depression are clinically interpretable inputs in the dataset, but visual association is not causation. The modest sample size and single processed regional subset limit generalization.", "BodySmall"),
            PageBreak(),
        ]
    )

    start_page("Feature engineering and model development", "02 | Modeling")
    story.extend(
        [
            para("Pipeline design", "Subhead"),
            para("Each candidate is a single sklearn Pipeline with a ColumnTransformer. Numeric features receive median imputation followed by StandardScaler. Categorical features receive most-frequent imputation and OneHotEncoder with unknown categories ignored at inference. Because preprocessing is inside the estimator, validation folds do not learn imputation values, scaling statistics, or category vocabularies from held-out data.", "BodySmall"),
            para("Tuning procedure", "Subhead"),
            bullet("Nested stratified cross-validation: 4-fold inner search optimizes ROC-AUC; 5-fold outer CV estimates generalization."),
            bullet("Logistic Regression searches C in {0.1, 1, 10} and unweighted/balanced class weights."),
            bullet("Random Forest uses 200 trees and searches depth, minimum leaf size, and unweighted/balanced class weights."),
            bullet("Accuracy, precision, recall, and ROC-AUC are reported as outer-fold mean +/- sample standard deviation."),
            metrics_table(),
            Spacer(1, 7),
            para("Selection", "Subhead"),
            para("Logistic Regression is selected by mean outer-fold ROC-AUC (0.9195 versus 0.9156). The difference is small relative to fold variability, so the selection is a pragmatic model-selection result, not evidence of a statistically significant advantage. The reported scores are cross-validation estimates; there is no independent external test set.", "BodySmall"),
            code("python -m src.model_development"),
            PageBreak(),
        ]
    )

    start_page("Experiment tracking and model package", "03-04 | MLflow and reproducibility")
    story.extend(
        [
            para("MLflow experiment", "Subhead"),
            para("Training creates one MLflow run for each classifier in the `heart-disease-model-development` experiment using a local SQLite tracking database and file artifact store. Runs record model family, random state, fold configuration, raw dataset name, training-row count, tuned parameters, and metric means and standard deviations. Logged artifacts include the comparison CSV, inner-search details and outer-fold parameter choices, EDA figures when available, and metric plots. The winning run is tagged and includes the logged model URI.", "BodySmall"),
            para("Packaged inference pipeline", "Subhead"),
            para("The selected full-data pipeline is logged as an MLflow sklearn model and saved to the versioned `models/` directory. The package carries fitted preprocessing and classifier objects, an input example, model signature, pinned dependency requirements, and a `current_model.txt` pointer. It uses skops serialization with an explicit trusted-type allowlist. Serving accepts the original 13 raw features so preprocessing is identical to training.", "BodySmall"),
            code("mlflow ui --backend-store-uri sqlite:///mlflow.db\n# Open http://127.0.0.1:5000\n\ncat models/current_model.txt"),
            para("Reproducibility", "Subhead"),
            bullet("Use the declared `requirements.txt` and `requirements-dev.txt` in a clean Python environment."),
            bullet("Acquire source data with the script; do not rely on a manually edited or pre-imputed training table."),
            bullet("Nested CV and model search use a fixed random seed and explicit search grids."),
            bullet("The versioned model package used for serving is produced by the training command and is a small local artifact; retrain to recreate it."),
            para("The experiment database, logs, and model output are generated artifacts and are not all committed. The repository's CI workflow uploads them on each successful run.", "BodySmall"),
            PageBreak(),
        ]
    )

    start_page("Automated tests and CI/CD", "05 | Quality gates")
    story.extend(
        [
            para("Local quality checks", "Subhead"),
            code("python -m pip install -r requirements-dev.txt\nruff check src tests app.py\npython -m pytest -q"),
            para("The current local verification passed: Ruff clean; 7 pytest tests passed. Tests cover data missing-value handling and target conversion, the model pipelines' fit/predict behavior including missing/unseen categories, API schema behavior, metrics exposition, and request logging. One upstream Starlette deprecation warning remains and does not fail the suite.", "BodySmall"),
            para("Hosted workflow", "Subhead"),
            para("The GitHub Actions workflow runs on pushes, pull requests, and manual dispatch. It installs dependencies, lints, tests, downloads data, trains and tracks models, builds the serving Docker image, starts it, exercises health/prediction/metrics endpoints, and uploads model, comparison, MLflow, and screenshot artifacts. Shell failure handling collects API container logs when a smoke test fails.", "BodySmall"),
            para("Evidence", "Subhead"),
        ]
    )
    add_image(story, SCREENSHOTS / "Screenshot (15).png", "Figure 4. Uploaded GitHub Actions job log: quality-and-training completed successfully, including the container build and smoke test.", 4.1 * inch)
    story.extend(
        [
            para("The uploaded run screenshot is genuine hosted CI evidence. The screenshot is useful for demonstrating this run, but future source changes should be followed by another successful run before submission.", "BodySmall"),
            PageBreak(),
        ]
    )

    start_page("API and container verification", "06 | Serving")
    story.extend(
        [
            para("FastAPI service contract", "Subhead"),
            bullet("`GET /health` reports service status and whether the model loaded."),
            bullet("`POST /predict` requires all 13 raw feature keys; values can be null for missing measurements."),
            bullet("Prediction response includes binary class, human-readable label, predicted-class confidence, and both class probabilities."),
            bullet("`GET /metrics` exposes Prometheus request and latency metrics; JSON logs include request ID, method, path, status, and duration without logging request bodies."),
            para("Build and run", "Subhead"),
            code("docker build -t heart-disease-api:latest .\ndocker run --rm -p 8000:8000 heart-disease-api:latest\n\ncurl http://127.0.0.1:8000/health"),
            para("Container verification", "Subhead"),
            para("The Docker image built successfully from the trained package. The documented sample payload returned HTTP 200 with prediction 0 (`no_disease`), confidence 0.5901, and probabilities 0.5901 / 0.4099. The same behavior was verified in the Minikube deployment.", "BodySmall"),
            para("Successful prediction evidence", "Subhead"),
            para("The executed sample request, HTTP 200 status, prediction, confidence, and both probabilities are shown in `reports/screenshots/03-predict-endpoint.png`.", "BodySmall"),
            Spacer(1, 3),
            image(SCREENSHOTS / "03-predict-endpoint.png", CONTENT_W, 1.12 * inch),
            PageBreak(),
        ]
    )

    start_page("Kubernetes deployment", "07 | Local production deployment")
    story.extend(
        [
            para("The image is deployed to Minikube 1.35.0 running Kubernetes 1.32.0. The namespace contains a one-replica FastAPI deployment, a LoadBalancer Service, Prometheus, and Grafana. API probes, non-root execution, read-only root filesystem, resource requests/limits, and ephemeral writable volumes are configured. Prometheus and Grafana are internal ClusterIP services.", "BodySmall"),
            para("Verified rollout", "Subhead"),
            bullet("API, Prometheus, and Grafana deployments each reached one ready replica after Minikube was restarted."),
            bullet("Health and sample prediction returned HTTP 200 through the VM port-forward."),
            bullet("The API Service external IP is `<pending>`; no public cloud endpoint is claimed."),
            code("minikube start --driver=docker --cpus=2 --memory=4096\nminikube kubectl -- rollout status deployment/heart-disease-api -n heart-disease\nminikube kubectl -- get pods,services -n heart-disease -o wide\nminikube kubectl -- port-forward -n heart-disease service/heart-disease-api 8000:80"),
            Spacer(1, 5),
            architecture_diagram(),
            para("Figure 5. Data-to-serving architecture and automated CI path.", "Caption"),
            para("The workload and endpoint screenshots are included in `reports/screenshots/`. Minikube may stop when the VM is idle; run `minikube start` before using the local API.", "BodySmall"),
            PageBreak(),
        ]
    )

    start_page("Monitoring and request logging", "08 | Observability")
    story.extend(
        [
            para("Prometheus scrapes the API `/metrics` endpoint every 15 seconds. The provisioned Grafana dashboard includes request rate, 95th percentile latency, status-class rates, and target health. The API emits single-line JSON access/error logs to stdout; Kubernetes users can inspect them with `minikube kubectl -- logs -n heart-disease deployment/heart-disease-api`.", "BodySmall"),
            para("Verified monitoring", "Subhead"),
            bullet("Prometheus API target `heart-disease-api` reported health `up`."),
            bullet("Grafana 12.1.1 health endpoint returned database status `ok`."),
            bullet("The `Heart Disease API` dashboard was found in Grafana's provisioned dashboard list."),
            bullet("Monitoring uses emptyDir storage for this local demonstration; metrics history is not durable after pod replacement."),
            Spacer(1, 5),
        ]
    )
    add_image(story, SCREENSHOTS / "Screenshot (6).png", "Figure 6. Grafana dashboard showing request rate, p95 latency, successful responses, and API target health.", 4.75 * inch)
    story.extend(
        [
            code("minikube kubectl -- port-forward -n heart-disease service/prometheus 9090:9090\nminikube kubectl -- port-forward -n heart-disease service/grafana 3000:3000"),
            para("The Grafana admin password is stored in a Kubernetes Secret and must not be committed to the repository. Use the secret in the live cluster to sign in; rotate it before leaving the demonstration environment if it has been shared.", "BodySmall"),
            PageBreak(),
        ]
    )

    start_page("Submission checklist", "09 | Deliverables")
    story.extend(
        [
            para("Implemented and verified", "Subhead"),
            bullet("UCI data acquisition, cleaning, EDA figures, preprocessing, two candidate models, nested CV, and documented metrics."),
            bullet("MLflow experiments, logged artifacts, selected serialized pipeline, requirements, and inference API."),
            bullet("Unit tests and Ruff; passing hosted GitHub Actions screenshot; Docker image build and smoke test."),
            bullet("Minikube API and monitoring deployments, endpoint verification, Prometheus scrape, Grafana dashboard, and request logs."),
            bullet("Repository: https://github.com/rimjhimdubey73/mlops_assignment."),
            para("Deployment access and remaining item", "Subhead"),
            bullet("The API is deployed locally on Minikube and has no public URL. Use the README port-forward instructions for access."),
            bullet("The assignment requests a short video recording of the end-to-end pipeline. No video is currently included; record and add one before submission."),
            para("Scope", "Subhead"),
            para("This model is an educational demonstration, not a clinical diagnostic tool. Evaluation uses a small regional dataset and cross-validation rather than an independent external test set.", "BodySmall"),
            para("Full setup commands are in `README.md`; the report is generated with `python reports/build_final_report.py` after installing `requirements-dev.txt`.", "BodySmall"),
        ]
    )

    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_report()
