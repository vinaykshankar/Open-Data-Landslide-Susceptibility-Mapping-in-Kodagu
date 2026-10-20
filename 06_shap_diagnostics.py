"""
06_shap_diagnostics.py
=======================
SHAP (SHapley Additive exPlanations) model-attribution diagnostics
for the Random Forest susceptibility model, as reported in Section 4
and Figure 6 of the paper.

Produces:
  - Global feature importance (mean |SHAP|)
  - SHAP beeswarm / summary plot
  - SHAP dependence plots for top-3 features

Input:  data/model_dataset.csv
        data/results/rf_model.pkl   (trained RF from 04_modelling.py)
Output: data/results/shap_values.csv
        figures/shap_summary.png
        figures/shap_dependence_<feature>.png

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import pandas as pd
import numpy as np
import os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError("Install shap:  pip install shap --break-system-packages")

DATA_PATH  = "data/model_dataset.csv"
MODEL_PATH = "data/results/rf_model.pkl"
OUT_DIR    = "data/results"
FIG_DIR    = "figures"

FEATURES = ["slope", "aspect", "elevation", "plan_curv", "profile_curv", "twi"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]

    with open(MODEL_PATH, "rb") as f:
        rf = pickle.load(f)

    print("Computing SHAP values (TreeExplainer) …")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]   # use class=1 (landslide presence)
    else:
        sv = shap_values

    # ── Save raw SHAP values ───────────────────────────────────────────
    shap_df = pd.DataFrame(sv, columns=[f"shap_{f}" for f in FEATURES])
    shap_df.to_csv(f"{OUT_DIR}/shap_values.csv", index=False)
    print(f"SHAP values saved: {OUT_DIR}/shap_values.csv")

    # ── Global importance ──────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
    print("\nGlobal feature importance (mean |SHAP|):")
    print(importance.to_string())

    # ── Summary / beeswarm plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(sv, X, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved: {FIG_DIR}/shap_summary.png")

    # ── Dependence plots for top-3 features ───────────────────────────
    top3 = importance.head(3).index.tolist()
    for feat in top3:
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.dependence_plot(feat, sv, X, ax=ax, show=False)
        plt.tight_layout()
        out = f"{FIG_DIR}/shap_dependence_{feat}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Dependence plot saved: {out}")


if __name__ == "__main__":
    main()
