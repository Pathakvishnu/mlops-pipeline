import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, functions as F, types as T

import pandas as pd
import mlflow.pyfunc

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
MODEL_NAME = os.environ.get("MODEL_NAME", "titanic_clf")
PREPROCESS_MODEL_NAME = os.environ.get("PREPROCESS_MODEL_NAME", "titanic_preprocess")


PREPROCESS_MODEL_URI = os.environ.get("PREPROCESS_MODEL_URI", f"models:/{PREPROCESS_MODEL_NAME}/Production")
MODEL_URI = os.environ.get("MODEL_URI", f"models:/{MODEL_NAME}/Production")

app = FastAPI(title="Titanic Classifier API", version="1.0.0")

spark = None


def get_spark():
    """Lazy SparkSession getter"""
    global spark
    if spark is None:
        spark = (
            SparkSession.builder
            .appName("TitanicAPI")
            .getOrCreate()
        )
    return spark

def extract_title(name: str) -> str:
    if name is None:
        return "Unknown"
    import re
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Name: str 
    Ticket: str 
    Cabin: str

@app.on_event("startup")
def load_models():
    global classifier_model
    global preprocess_model
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    spark = get_spark()
    preprocess_model = mlflow.spark.load_model(PREPROCESS_MODEL_URI)
    classifier_model = mlflow.spark.load_model(MODEL_URI)

@app.get("/health")
def health():
    return {"status": "ok", "preprocess_uri":PREPROCESS_MODEL_URI, "model_uri": MODEL_URI}

@app.post("/predict")
def predict(items: list[Passenger]):
    df = pd.DataFrame([x.model_dump() for x in items])
    udf_title = F.udf(extract_title, T.StringType())
    spark = get_spark()
    spark_df = spark.createDataFrame(df)
    spark_df = spark_df.withColumn("Title", udf_title(F.col("Name")))
    spark_df = spark_df.withColumn("FamilySize", F.col("SibSp").cast("int") + F.col("Parch").cast("int") + F.lit(1))
    spark_df = spark_df.withColumn("IsAlone", F.when(F.col("FamilySize")==1, 1).otherwise(0))
    spark_df = spark_df.drop("Cabin","Ticket")  # simplify
    processed_df = preprocess_model.transform(spark_df)
    preds = classifier_model.transform(processed_df)
    result = preds.select("prediction").toPandas().to_dict(orient="records")
    return {"predictions": result}
