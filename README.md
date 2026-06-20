# Workflow-CI — Amri Reza Wahyudin

Kriteria 3 submission: MLflow Project + GitHub Actions CI that retrains the
Adult-income classifier and publishes artifacts (Skilled) + a Docker image
on Docker Hub (Advance).

## Repo layout

```
Workflow-CI/
├── .github/workflows/ci.yml       # GitHub Actions: train + upload + docker push
├── MLProject/
│   ├── MLProject                  # MLflow Project entry-point spec
│   ├── conda.yaml                 # conda env for `mlflow run`
│   ├── modelling.py               # training script (parameterised)
│   ├── src/                       # preprocessing + metrics helpers
│   │   ├── preprocessing.py
│   │   └── metrics.py
│   ├── data/adult_clean.csv       # cleaned dataset (mirror of K1 output)
│   └── DOCKERHUB_LINK.txt         # written by CI after successful push
└── README.md
```

## What the CI does

Trigger: `push` to `main` or `workflow_dispatch`.

1. Set up Python 3.12 + install deps via `uv pip install`.
2. **Basic+Skilled+Advance:** `mlflow run MLProject --no-conda` trains the
   model, logs metrics + 4+ artifacts to DagsHub, and writes the run id
   to `MLProject/ci_run_id.txt`.
3. **Skilled:** uploads `MLProject/mlruns/` + run id file as a GitHub
   Actions artifact (`mlflow-model`).
4. **Advance:**
   - logs into Docker Hub via `docker/login-action@v3` (using repo secrets).
   - `mlflow models build-docker -m runs:/<RUN_ID>/model -n amzadin14/smsml-adult`.
   - `docker push amzadin14/smsml-adult:latest`.
5. Writes the Docker Hub URL to `MLProject/DOCKERHUB_LINK.txt` and
   auto-commits.

## Secrets required (set in repo Settings → Secrets and variables → Actions)

| Secret              | Value                          |
|---------------------|--------------------------------|
| `DAGSHUB_TOKEN`     | DagsHub user token             |
| `DOCKERHUB_USERNAME`| `AmZaDin14`                    |
| `DOCKERHUB_TOKEN`   | Docker Hub access token        |

## Local reproduction

```bash
# From the repo root
cd MLProject
mlflow run . -P data-path=data/adult_clean.csv -P run-name=local
```

This trains the model locally and logs to whatever `MLFLOW_TRACKING_URI`
points to (default: `./mlruns`).
