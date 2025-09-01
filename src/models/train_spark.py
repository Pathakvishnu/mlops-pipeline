import argparse, os, json, time, yaml
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.utils.logging_utils import get_logger

def main(args):
    # mlflow.set_tracking_uri("http://0.0.0.0:2000")   # or "http://localhost:2000"
    logger = get_logger("train")
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)
    seed = int(params.get("seed", 42))
    model_cfg = params.get("model", {})
    tuning_cfg = params.get("tuning", {})
    registry_cfg = params.get("registry", {})
    model_name = registry_cfg.get("model_name","titanic_clf")

    spark = SparkSession.builder.appName("titanic-train").getOrCreate()
    train = spark.read.parquet(args.train)
    test = spark.read.parquet(args.test)

    if model_cfg.get("name") == "random_forest":
        clf = RandomForestClassifier(featuresCol="features", labelCol="Survived",
                                        numTrees=int(model_cfg.get("rf_num_trees",200)),
                                        maxDepth=int(model_cfg.get("rf_max_depth",8)),
                                        seed=seed)
        grid = ParamGridBuilder()\
            .addGrid(clf.numTrees, [int(x) for x in tuning_cfg["grid"]["rf_num_trees"]])\
            .addGrid(clf.maxDepth, [int(x) for x in tuning_cfg["grid"]["rf_max_depth"]])\
            .build()
    else:
        clf = LogisticRegression(featuresCol="features", labelCol="Survived",
                                    maxIter=int(model_cfg.get("max_iter",50)),
                                    regParam=float(model_cfg.get("reg_param",0.0)),
                                    elasticNetParam=float(model_cfg.get("elastic_net_param",0.0)))
        grid = ParamGridBuilder()\
            .addGrid(clf.regParam, [float(x) for x in tuning_cfg["grid"]["lr_reg_params"]])\
            .addGrid(clf.elasticNetParam, [float(x) for x in tuning_cfg["grid"]["lr_elastic"]])\
            .build()

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Survived", metricName="areaUnderROC")
    use_cv = bool(tuning_cfg.get("use_cv", True))
    trainer = CrossValidator(estimator=clf, estimatorParamMaps=grid, evaluator=evaluator,
                                numFolds=int(tuning_cfg.get("num_folds",3)), seed=seed) if use_cv else clf
    
    mlflow.set_experiment("titanic_spark")
    with mlflow.start_run(run_name="titanic_dataset") as run:
        start = time.time()
        model = trainer.fit(train)
        dur = time.time() - start

        pred = model.transform(test)
        roc_auc = evaluator.evaluate(pred)
        pr_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Survived", metricName="areaUnderPR")
        pr_auc = pr_eval.evaluate(pred)

        mlflow.log_params({"model_name": model_cfg.get("name"), "seed": seed, "use_cv": use_cv})
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("train_seconds", dur)

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/metrics.json","w") as f:
            json.dump({"roc_auc": roc_auc, "pr_auc": pr_auc, "train_seconds": dur}, f, indent=2)

        mlflow.spark.log_model(model.bestModel if use_cv else model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        best_auc, best_ver = -1.0, None
        for v in versions:
            r = client.get_run(v.run_id)
            m = r.data.metrics.get("roc_auc")
            if m is not None and m > best_auc:
                best_auc, best_ver = m, v.version
        if best_ver is not None:
            client.transition_model_version_stage(
                name=model_name, version=best_ver, stage="Production", archive_existing_versions=True
            )
            logger.info("Promoted version %s to Production with roc_auc=%.4f", best_ver, best_auc)

        conf = spark.sparkContext.getConf().getAll()
        with open("artifacts/spark_conf.json","w") as f:
            json.dump({k:v for k,v in conf}, f, indent=2)
        mlflow.log_artifact("artifacts/spark_conf.json")

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--params", required=True)
    main(p.parse_args())
