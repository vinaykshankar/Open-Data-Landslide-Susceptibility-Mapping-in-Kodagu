"""
05_spatial_block_cv.py
=======================
Spatial block cross-validation for LR and RF models.
Implements the block CV strategy described in Section 3.5 of the paper,
which accounts for spatial autocorrelation by holding out geographic
blocks rather than random samples.

Block size: ~10 km × 10 km (tunable)
Folds: 5 spatial folds
Reports: AUC (held-out), Success Rate Curve, Prediction Rate Curve

Input:  data/model_dataset.csv  (must include lat, lon columns)
Output: data/results/spatial_cv_metrics.csv
        data/results/spatial_cv_roc_data.csv

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import pandas as pd
import numpy as np
import os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

DATA_PATH   = "data/model_dataset.csv"
RESULTS_DIR = "data/results"
BLOCK_SIZE_DEG = 0.09   # ≈ 10 km at this latitude
N_FOLDS        = 5
FEATURES = ["slope", "aspect", "elevation", "plan_curv", "profile_curv", "twi"]
TARGET   = "presence"


def assign_blocks(df, block_size):
    """Assign each point to a spatial block grid cell."""
    df = df.copy()
    df["block_col"] = ((df["lon"] - df["lon"].min()) / block_size).astype(int)
    df["block_row"] = ((df["lat"] - df["lat"].min()) / block_size).astype(int)
    df["block_id"]  = df["block_row"].astype(str) + "_" + df["block_col"].astype(str)
    return df


def make_folds(df, n_folds, seed=42):
    """Assign blocks to folds (stratified by presence rate per block)."""
    block_stats = df.groupby("block_id")[TARGET].agg(["sum", "count"])
    block_stats["pres_rate"] = block_stats["sum"] / block_stats["count"]
    blocks = block_stats.index.tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    fold_map = {b: i % n_folds for i, b in enumerate(blocks)}
    df["fold"] = df["block_id"].map(fold_map)
    return df


def run_spatial_cv(df, model_name):
    """Run spatial block CV for one model type."""
    results = []
    all_y, all_proba = [], []

    for fold in range(N_FOLDS):
        train = df[df["fold"] != fold]
        test  = df[df["fold"] == fold]

        if test[TARGET].nunique() < 2:
            print(f"  Fold {fold}: skipped (only one class in test set)")
            continue

        X_tr, y_tr = train[FEATURES].values, train[TARGET].values
        X_te, y_te = test[FEATURES].values,  test[TARGET].values

        if model_name == "LR":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42))
            ])
        else:
            model = RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        results.append({"model": model_name, "fold": fold,
                        "n_train": len(train), "n_test": len(test),
                        "AUC": auc})
        all_y.extend(y_te)
        all_proba.extend(proba)
        print(f"  {model_name} fold {fold}: AUC = {auc:.4f}  "
              f"(train={len(train)}, test={len(test)})")

    mean_auc = np.mean([r["AUC"] for r in results])
    std_auc  = np.std( [r["AUC"] for r in results])
    print(f"  {model_name}  mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # Overall ROC curve across all held-out folds
    fpr, tpr, _ = roc_curve(all_y, all_proba)
    roc_df = pd.DataFrame({"model": model_name, "fpr": fpr, "tpr": tpr})

    return pd.DataFrame(results), roc_df, mean_auc, std_auc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = assign_blocks(df, BLOCK_SIZE_DEG)
    df = make_folds(df, N_FOLDS)

    n_blocks = df["block_id"].nunique()
    print(f"Dataset: {len(df)} points  |  {n_blocks} spatial blocks  |  {N_FOLDS} folds\n")

    all_metrics, all_roc = [], []

    for mname in ["LR", "RF"]:
        print(f"── Spatial Block CV: {mname} ──")
        metrics_df, roc_df, mean_auc, std_auc = run_spatial_cv(df, mname)
        summary = {"model": mname, "mean_AUC": mean_auc, "std_AUC": std_auc,
                   "n_folds": N_FOLDS, "block_size_deg": BLOCK_SIZE_DEG}
        all_metrics.append(metrics_df)
        all_roc.append(roc_df)
        print()

    pd.concat(all_metrics).to_csv(f"{RESULTS_DIR}/spatial_cv_metrics.csv", index=False)
    pd.concat(all_roc).to_csv(   f"{RESULTS_DIR}/spatial_cv_roc_data.csv",  index=False)
    print("Spatial CV results saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
