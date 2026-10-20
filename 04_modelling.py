"""
04_modelling.py
================
Landslide susceptibility modelling using three methods:
  1. Information Value (IV)  — bivariate statistical
  2. Logistic Regression (LR) — multivariate statistical
  3. Random Forest (RF)       — machine learning

Matches the methodology in Section 3.4 of the paper.

Input:  data/model_dataset.csv  (rows = sample points,
        columns = lat, lon, presence, slope, aspect, elevation,
                  plan_curv, profile_curv, twi)
Output: data/results/iv_scores.csv
        data/results/lr_model.pkl
        data/results/rf_model.pkl
        data/results/model_metrics.csv

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import pandas as pd
import numpy as np
import os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

DATA_PATH   = "data/model_dataset.csv"
RESULTS_DIR = "data/results"

FEATURES = ["slope", "aspect", "elevation", "plan_curv", "profile_curv", "twi"]
TARGET   = "presence"

# ── IV bins (from Table 2 in the paper) ───────────────────────────────────
IV_BINS = {
    "slope":     [0, 5, 15, 25, 35, 45, 90],
    "aspect":    [0, 45, 90, 135, 180, 225, 270, 315, 360],
    "elevation": [0, 500, 700, 900, 1100, 1300, 1500, 2000],
    "twi":       [0, 4, 6, 8, 10, 12, 20],
}


def information_value(df, factor, bins, target="presence"):
    """Compute IV score per class and total IV for a factor."""
    df = df.copy()
    df["bin"] = pd.cut(df[factor], bins=bins, include_lowest=True)
    total_p  = df[target].sum()
    total_a  = (1 - df[target]).sum()

    rows = []
    for cls, grp in df.groupby("bin", observed=True):
        n_p = grp[target].sum()
        n_a = (1 - grp[target]).sum()
        pct_p = n_p / total_p if total_p > 0 else 1e-9
        pct_a = n_a / total_a if total_a > 0 else 1e-9
        pct_p = max(pct_p, 1e-9)
        pct_a = max(pct_a, 1e-9)
        woe   = np.log(pct_p / pct_a)
        iv    = (pct_p - pct_a) * woe
        rows.append({"factor": factor, "class": str(cls),
                     "n_presence": n_p, "n_absence": n_a,
                     "pct_p": pct_p, "pct_a": pct_a,
                     "WoE": woe, "IV_class": iv})
    result = pd.DataFrame(rows)
    result["IV_total"] = result["IV_class"].sum()
    return result


def run_iv(df):
    print("── Information Value ──")
    all_iv = []
    for factor, bins in IV_BINS.items():
        iv_df = information_value(df, factor, bins)
        all_iv.append(iv_df)
        print(f"  {factor:20s}  IV = {iv_df['IV_total'].iloc[0]:.4f}")
    return pd.concat(all_iv, ignore_index=True)


def run_lr(X_train, y_train, X_test, y_test):
    print("── Logistic Regression ──")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    print(f"  AUC (random split): {auc:.4f}")
    return pipe, auc


def run_rf(X_train, y_train, X_test, y_test):
    print("── Random Forest ──")
    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    print(f"  AUC (random split): {auc:.4f}")
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("  Feature importances:")
    print(fi.to_string())
    return rf, auc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {len(df)} rows  |  presences: {df[TARGET].sum()}")

    # ── IV ────────────────────────────────────────────────────────────────
    iv_results = run_iv(df)
    iv_results.to_csv(f"{RESULTS_DIR}/iv_scores.csv", index=False)

    # ── Train/test split (random 70/30) ───────────────────────────────────
    from sklearn.model_selection import train_test_split
    X = df[FEATURES].values
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # ── LR ────────────────────────────────────────────────────────────────
    lr_pipe, lr_auc = run_lr(X_tr, y_tr, X_te, y_te)
    with open(f"{RESULTS_DIR}/lr_model.pkl", "wb") as f:
        pickle.dump(lr_pipe, f)

    # ── RF ────────────────────────────────────────────────────────────────
    rf_model, rf_auc = run_rf(X_tr, y_tr, X_te, y_te)
    with open(f"{RESULTS_DIR}/rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # ── Summary ───────────────────────────────────────────────────────────
    metrics = pd.DataFrame([
        {"model": "Logistic Regression", "split": "random_70_30", "AUC": lr_auc},
        {"model": "Random Forest",       "split": "random_70_30", "AUC": rf_auc},
    ])
    metrics.to_csv(f"{RESULTS_DIR}/model_metrics.csv", index=False)
    print("\nMetrics saved to", f"{RESULTS_DIR}/model_metrics.csv")


if __name__ == "__main__":
    main()
