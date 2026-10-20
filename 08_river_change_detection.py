"""
08_river_change_detection.py
=============================
Same-season river-channel change detection using Sentinel-2 MNDWI
water masks (2019 and 2024 dry seasons), as described in Section 3.6
and reported in Section 4.3 of the paper.

MNDWI = (Green – SWIR1) / (Green + SWIR1)
Water mask threshold: MNDWI > 0.0

The script compares the two dry-season extents and quantifies:
  - Area gained (2024 > 2019)
  - Area lost   (2019 > 2024)
  - Stable water
  - Net change

Input:  data/mndwi_2019_dry.tif   (Sentinel-2 derived, GEE export)
        data/mndwi_2024_dry.tif
Output: data/results/river_change.tif   (classified change raster)
        data/results/river_change_stats.csv

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import numpy as np
import rasterio
import pandas as pd
import os

MNDWI_2019 = "data/mndwi_2019_dry.tif"
MNDWI_2024 = "data/mndwi_2024_dry.tif"
RESULTS_DIR = "data/results"

MNDWI_THRESHOLD = 0.0   # standard open-water threshold

# Change classes
STABLE_WET  = 1   # water in both years
GAIN        = 2   # water in 2024 only (new or expanded)
LOSS        = 3   # water in 2019 only (dried or shifted)
STABLE_DRY  = 0   # dry in both years


def load_mndwi(path, threshold):
    """Load MNDWI raster and threshold to binary water mask."""
    with rasterio.open(path) as src:
        mndwi   = src.read(1).astype(np.float32)
        profile = src.profile
        nodata  = src.nodata or -9999
        mndwi[mndwi == nodata] = np.nan
    water = (mndwi > threshold).astype(np.uint8)
    return water, profile


def detect_change(w2019, w2024):
    change = np.zeros_like(w2019, dtype=np.uint8)
    change[(w2019 == 1) & (w2024 == 1)] = STABLE_WET
    change[(w2019 == 0) & (w2024 == 1)] = GAIN
    change[(w2019 == 1) & (w2024 == 0)] = LOSS
    return change


def area_stats(change_raster, pixel_area_m2):
    stats = {
        "stable_water_km2": float((change_raster == STABLE_WET).sum() * pixel_area_m2 / 1e6),
        "gain_km2":         float((change_raster == GAIN).sum()        * pixel_area_m2 / 1e6),
        "loss_km2":         float((change_raster == LOSS).sum()        * pixel_area_m2 / 1e6),
    }
    stats["net_change_km2"] = stats["gain_km2"] - stats["loss_km2"]
    return stats


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading MNDWI masks …")
    w2019, profile = load_mndwi(MNDWI_2019, MNDWI_THRESHOLD)
    w2024, _       = load_mndwi(MNDWI_2024, MNDWI_THRESHOLD)

    # Pixel area (assumes metric projection, 10 m Sentinel-2)
    transform  = profile["transform"]
    pixel_m    = abs(transform.a)   # pixel width in metres
    pixel_area = pixel_m ** 2

    print("Detecting change …")
    change = detect_change(w2019, w2024)

    # Save change raster
    out_path = f"{RESULTS_DIR}/river_change.tif"
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(change, 1)
    print(f"Change raster saved: {out_path}")

    # Statistics
    stats = area_stats(change, pixel_area)
    print("\nRiver change statistics:")
    for k, v in stats.items():
        print(f"  {k:25s}: {v:.4f} km²")

    df = pd.DataFrame([stats])
    df["year_early"]  = 2019
    df["year_late"]   = 2024
    df["sensor"]      = "Sentinel-2"
    df["threshold"]   = MNDWI_THRESHOLD
    df["pixel_size_m"] = pixel_m
    df.to_csv(f"{RESULTS_DIR}/river_change_stats.csv", index=False)
    print(f"\nStats saved: {RESULTS_DIR}/river_change_stats.csv")


if __name__ == "__main__":
    main()
