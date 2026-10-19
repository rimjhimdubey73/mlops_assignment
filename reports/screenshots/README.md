# Deployment Evidence Checklist

Local Minikube deployment verified on 2026-10-19. Access the API through the VM port-forward instructions in the project README.

Present in this folder:

- `01-kubernetes-workloads.png`: cropped from the uploaded live terminal capture; all three pods are `1/1 Running` and the API Service is visible.
- `02-health-endpoint.png`: cropped from the same capture; `/health` returns `model_loaded: true` with HTTP 200.
- `03-predict-endpoint.png`: cropped from the uploaded live terminal capture; sample prediction returned HTTP 200 with confidence and both class probabilities.
- `Screenshot (6).png`: Grafana dashboard with API monitoring panels.
- `Screenshot (7).png` and `Screenshot (10).png` through `Screenshot (14).png`: Swagger UI endpoint/schema views. These do not show successful executed health/prediction requests.
- `Screenshot (9).png`: successful GitHub Actions run summary.
- `Screenshot (15).png`: successful GitHub Actions `quality-and-training` job details.
- `Screenshot (16).png`: original uploaded VS Code capture used to create the focused `01` and `02` evidence crops.
- `Screenshot (17).png`: original uploaded VS Code capture used to create the focused `03` prediction evidence crop.

Still needed for the assignment:


For browser evidence, run `minikube kubectl -- port-forward -n heart-disease service/heart-disease-api 8000:80` in a VM terminal, forward port 8000 in VS Code, then use `/health` or `/docs`. The public Dev Tunnel URL can require GitHub authentication and is not the local deployment URL. See `../FINAL_REPORT.md` for verified results and `../../README.md` for setup and local access instructions.
