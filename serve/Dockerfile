FROM python:3.11-slim
RUN apt-get update && apt-get install -y openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY serve/api.py /app/serve/api.py
ENV MLFLOW_TRACKING_URI=mlruns
ENV MODEL_NAME=titanic_clf
ENV PREPROCESS_MODEL_NAME=titanic_preprocess
ENV MODEL_URI=models:/titanic_clf/Production
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
