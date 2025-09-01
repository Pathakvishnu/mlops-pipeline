#!/usr/bin/env bash
set -euo pipefail
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 2000
