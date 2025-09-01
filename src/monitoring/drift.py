import argparse, json, os, yaml
import pandas as pd
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    qs = np.linspace(0, 100, bins+1)
    cuts = np.percentile(expected, qs)
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_perc = e_counts / max(e_counts.sum(), eps)
    a_perc = a_counts / max(a_counts.sum(), eps)
    return float(np.sum((a_perc - e_perc) * np.log((a_perc + eps) / (e_perc + eps))))

def main(args):
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)
    psi_threshold = float(params.get("drift", {}).get("psi_threshold", 0.2))

    with open(args.baseline, "r") as f:
        baseline_stats = json.load(f)

    new_df = pd.read_csv(args.new)
    metrics = {}
    drift_flag = False
    for col, _ in baseline_stats.items():
        if col not in new_df.columns: 
            continue
        exp = new_df[col].dropna().values
        act = new_df[col].dropna().values
        if len(exp)==0 or len(act)==0:
            continue
        val = psi(exp, act, bins=10)
        metrics[f"psi_{col}"] = val
        if val > psi_threshold:
            drift_flag = True

    metrics["drift_detected"] = bool(drift_flag)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--params", required=True)
    main(p.parse_args())
