import json, subprocess, sys, os
DRIFT_FILE = "artifacts/drift_metrics.json"
def main():
    if not os.path.exists(DRIFT_FILE):
        print("No drift file found; skipping retrain."); return
    with open(DRIFT_FILE,"r") as f:
        metrics = json.load(f)
    if metrics.get("drift_detected", False):
        print("Drift detected — retraining pipeline...")
        subprocess.check_call(["dvc", "repro", "preprocess", "train"])
    else:
        print("No significant drift; nothing to do.")
if __name__ == "__main__":
    main()
