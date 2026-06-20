# Simple Docker wrapper for the MLflow model.
# Built and pushed to Docker Hub manually (the runner's Docker daemon
# rejects our PAT with "malformed HTTP Authorization header", so we
# build locally and verify the image is on Docker Hub in CI).

FROM python:3.12-slim

WORKDIR /opt/mlflow

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.9.0 \
    pandas \
    numpy \
    gunicorn \
    gevent

# Copy the model artifact (the `model` directory from mlruns/<exp>/<run>/artifacts/)
COPY model /opt/ml/model

ENV MLFLOW_DISABLE_ENV_CREATION=true
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

ENTRYPOINT ["mlflow", "models", "serve", \
    "--model-uri", "/opt/ml/model", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--no-conda"]
