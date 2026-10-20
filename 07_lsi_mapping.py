"""
07_lsi_mapping.py
==================
Applies the trained LR and RF models pixel-by-pixel across the
terrain-factor rasters to produce the Landslide Susceptibility
Index (LSI) maps, then classifies them into 5 zones:
  Very Low / Low / Moderate / High / Very High

Thresholds are derived from the natural-breaks (Jenks) method or
equal-area quantiles (paper uses quantile classification).

Input:  data/factors/*.tif   (one raster per factor, all co-registered)
        data/results/lr_model.pkl
        data/results/rf_model.pkl
Output: data/results/lsi_lr.tif
        data/results/lsi_rf.tif
        data/results/lsi_rf_classified.tif
        data/results/zone_statistics.csv

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import numpy as np
import rasterio
import pandas as pd
import os, pickle
from rasterio.plot import show

DATA_DIR    = "data/factors"
RESULTS_DIR = "data/results"
FEATURES    = ["slope", "aspect", "elevation", "plan_curv", "profile_curv", "twi"]

# Susceptibility class boundaries (quantile-based, 5 classes)
CLASS_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
CLASS_QUANTILES = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]


def load_factor_stack(factor_names, data_dir):
    """Load all factor rasters and stack into (n_pixels, n_features) array."""
    arrays, profile, shape = [], None, None
    for name in factor_names:
        path = os.path.join(data_dir, f"{name}.tif")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile
                shape   = arr.shape
            arrays.append(arr.ravel())
    stack = np.column_stack(arrays)   # (n_pixels, n_features)
    return stack, profile, shape


def classify_lsi(lsi_flat, quantiles):
    """Assign class 1–5 using quantile thresholds."""
    thresholds = np.nanquantile(lsi_flat, quantiles)
    classified  = np.zeros_like(lsi_flat, dtype=np.uint8)
    for i in range(1, len(quantiles)):
        mask = (lsi_flat >= thresholds[i-1]) & (lsi_flat <= thresholds[i])
        classified[mask] = i
    return classified, thresholds


def zone_statistics(lsi_flat, classified_flat, thresholds):
    rows = []
    for i, label in enumerate(CLASS_LABELS, start=1):
        mask = classified_flat == i
        rows.append({
            "class": i,
            "label": label,
            "lsi_min": thresholds[i-1],
            "lsi_max": thresholds[i],
            "pixel_count": int(mask.sum()),
            "pct_area": 100.0 * mask.sum() / (classified_flat > 0).sum()
        })
    return pd.DataFrame(rows)


def save_raster(array_2d, profile, path, dtype=rasterio.float32):
    profile.update(dtype=dtype, count=1, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array_2d.astype(dtype), 1)
    print(f"  Saved: {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading factor stack …")
    stack, profile, shape = load_factor_stack(FEATURES, DATA_DIR)

    # Mask out nodata pixels
    valid_mask = ~np.any(np.isnan(stack), axis=1)
    X_valid    = stack[valid_mask]
    print(f"  Valid pixels: {valid_mask.sum():,} / {len(valid_mask):,}")

    all_stats = []

    for mname, pkl_path in [("LR", f"{RESULTS_DIR}/lr_model.pkl"),
                             ("RF", f"{RESULTS_DIR}/rf_model.pkl")]:
        print(f"\n── {mname} LSI map ──")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        proba_valid = model.predict_proba(X_valid)[:, 1]

        lsi_flat = np.full(len(valid_mask), np.nan, dtype=np.float32)
        lsi_flat[valid_mask] = proba_valid

        lsi_2d = lsi_flat.reshape(shape)
        save_raster(lsi_2d, profile.copy(), f"{RESULTS_DIR}/lsi_{mname.lower()}.tif")

        # Classify
        classified_flat, thresholds = classify_lsi(
            proba_valid, CLASS_QUANTILES
        )
        classified_2d = np.zeros(len(valid_mask), dtype=np.uint8)
        classified_2d[valid_mask] = classified_flat
        classified_2d = classified_2d.reshape(shape)
        save_raster(classified_2d, profile.copy(),
                    f"{RESULTS_DIR}/lsi_{mname.lower()}_classified.tif",
                    dtype=rasterio.uint8)

        stats = zone_statistics(proba_valid, classified_flat, thresholds)
        stats["model"] = mname
        all_stats.append(stats)
        print(stats.to_string(index=False))

    pd.concat(all_stats).to_csv(f"{RESULTS_DIR}/zone_statistics.csv", index=False)
    print(f"\nZone statistics saved: {RESULTS_DIR}/zone_statistics.csv")


if __name__ == "__main__":
    main()
