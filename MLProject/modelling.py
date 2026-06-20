"""MLProject entry point — K3 CI retraining.

Mirrors K2/modelling_tuning.py but parameterised for MLflow Project and
faster (single fit, no grid) so the CI pipeline completes in a few minutes.

Run via MLflow Project:
    mlflow run . -P data-path=data/adult_clean.csv -P run-name=ci
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

if "DAGSHUB_USER_TOKEN" not in os.environ and "DAGSHUB_TOKEN" in os.environ:
    os.environ["DAGSHUB_USER_TOKEN"] = os.environ["DAGSHUB_TOKEN"]

import dagshub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Reuse preprocessing constants and helpers from K2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import NUMERIC_COLS, build_preprocessor
from metrics import baseline_stats, compute_metrics

warnings.filterwarnings("ignore")

TARGET = "income"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path, default=Path("data/adult_clean.csv"))
    p.add_argument("--run-name", type=str, default="ci")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # DagsHub tracking — only initialise if not already set (e.g. by mlflow run's tracking URI)
    existing_uri = mlflow.get_tracking_uri()
    if "dagshub.com" not in (existing_uri or ""):
        dagshub_user = os.environ.get("DAGSHUB_USER", "AmZaDin14")
        dagshub_repo = os.environ.get("DAGSHUB_REPO", "smsml-adult")
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
    # Set experiment only if not nested under an existing run
    if not mlflow.active_run():
        mlflow.set_experiment("adult_ci_reproducible")

    # Estimate baseline drift
    baseline_X = X_train.sample(n=2000, random_state=RANDOM_STATE)
    baseline = baseline_stats(baseline_X[NUMERIC_COLS].to_numpy(float), NUMERIC_COLS)
    Path("input_drift_baseline.json").write_text(json.dumps(baseline, indent=2))

    # Single fit (faster than grid; CI budget is small)
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=15,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        )),
    ])

    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_name = f"{args.run_name}_{run_ts}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("student", "amri-reza-wahyudin")
        mlflow.set_tag("trigger", "ci")
        mlflow.set_tag("framework", "sklearn + manual_log")

        # Manual log params
        mlflow.log_param("data_path", str(args.data_path))
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_train_rows", int(len(X_train)))
        mlflow.log_param("n_test_rows", int(len(X_test)))

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Manual log metrics
        metrics = compute_metrics(y_test.values, y_pred, y_proba)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Extra artifacts
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(input_features=pre.transformers_[1][2]))
        full_names = list(pre.transformers_[0][2]) + cat_names
        order = np.argsort(clf.feature_importances_)[-20:]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([full_names[i] for i in order], clf.feature_importances_[order], color="#4C72B0")
        ax.set_title("Top 20 Feature Importances (CI)")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title("Confusion Matrix (CI)")
        ax.set_xticklabels(["<=50K", ">50K"])
        ax.set_yticklabels(["<=50K", ">50K"])
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC={metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="#888")
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_artifact("input_drift_baseline.json")

        mlflow.sklearn.log_model(pipe, artifact_path="model")

    # Persist run id to a file so the GH Actions step can read it
    Path("ci_run_id.txt").write_text(run_id)
    print(f"CI run id: {run_id}")


if __name__ == "__main__":
    main()
