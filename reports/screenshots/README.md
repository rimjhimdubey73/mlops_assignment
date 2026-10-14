# Deployment Screenshot Checklist

Add genuine screenshots after deploying the image to Minikube, Docker Desktop Kubernetes, or a cloud cluster. Do not submit placeholders as deployment evidence.

- `01-kubernetes-workloads.png`: ready API, Prometheus, and Grafana pods plus the API LoadBalancer Service.
- `02-health-endpoint.png`: successful `/health` response from the deployed service.
- `03-predict-endpoint.png`: successful sample `/predict` response including confidence and probabilities.
- `04-grafana-dashboard.png`: provisioned Heart Disease API dashboard with scraped metrics.
- `05-ci-workflow.png`: successful GitHub Actions run with build, smoke test, and uploaded artifacts.

Record the cluster type, deployment date, and reachable API URL in the final report after capture. The current workspace has no Docker/Kubernetes executable or Git remote, so these screenshots are intentionally absent.
