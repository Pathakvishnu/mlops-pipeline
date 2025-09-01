.ONESHELL:
SHELL := /bin/bash
init:
    python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && dvc init
run:
    dvc repro
serve:
    uvicorn serve.api:app --host 0.0.0.0 --port 8000
mlflow:
    ./mlflow/start_server.sh
