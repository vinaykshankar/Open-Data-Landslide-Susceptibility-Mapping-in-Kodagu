"""
01_inventory_compilation.py
============================
Landslide inventory compilation from Sentinel-2 imagery using GEE-derived
NDVI differencing. This script processes the GEE-exported CSV/GeoJSON output
and applies the detection thresholds described in the paper.

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
Institution: RV Institute of Management, Bengaluru

Detection thresholds (from paper Section 3.2):
  - ΔNDVI > 0.20  (pre–post NDVI difference)
  - Post-NDVI < 0.35
  - Area < 2 ha
  - Patch area ≥ 900 m²
  - Slope ≥ 15°
  - Centroid slope ≥ 15°

Input:  GEE-exported scar polygons (GeoJSON or Shapefile)
Output: Filtered 461-scar inventory (GeoJSON + CSV of centroids)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_SCARS   = "data/raw_scars_gee_export.geojson"   # GEE export
SLOPE_RASTER  = "data/slope_reclass.tif"               # slope in degrees
OUTPUT_SHP    = "data/landslide_inventory_461.gpkg"
OUTPUT_CSV    = "data/landslide_centroids_461.csv"

# ── Thresholds ─────────────────────────────────────────────────────────────
DELTA_NDVI_MIN  = 0.20
POST_NDVI_MAX   = 0.35
AREA_MAX_HA     = 2.0
AREA_MIN_M2     = 900.0
SLOPE_MIN_DEG   = 15.0


def load_scars(path):
    gdf = gpd.read_file(path)
    print(f"Loaded {len(gdf)} raw scar polygons")
    return gdf


def apply_thresholds(gdf):
    """Apply the five detection thresholds from the paper."""
    mask = (
        (gdf["delta_ndvi"]  >  DELTA_NDVI_MIN) &
        (gdf["post_ndvi"]   <  POST_NDVI_MAX)  &
        (gdf["area_m2"]     >= AREA_MIN_M2)    &
        (gdf["area_m2"]     <= AREA_MAX_HA * 10000) &
        (gdf["slope_deg"]   >= SLOPE_MIN_DEG)
    )
    filtered = gdf[mask].copy()
    print(f"After thresholds: {len(filtered)} scars retained")
    return filtered


def compute_centroids(gdf):
    """Extract centroid coordinates (WGS84)."""
    gdf_wgs = gdf.to_crs("EPSG:4326")
    gdf["centroid_lon"] = gdf_wgs.geometry.centroid.x
    gdf["centroid_lat"] = gdf_wgs.geometry.centroid.y
    return gdf


def main():
    os.makedirs("data", exist_ok=True)

    gdf = load_scars(INPUT_SCARS)
    gdf = apply_thresholds(gdf)
    gdf = compute_centroids(gdf)

    # Save inventory
    gdf.to_file(OUTPUT_SHP, driver="GPKG")
    print(f"Inventory saved: {OUTPUT_SHP}")

    # Save centroid CSV
    cols = ["centroid_lat", "centroid_lon", "area_m2", "delta_ndvi",
            "post_ndvi", "slope_deg"]
    gdf[[c for c in cols if c in gdf.columns]].to_csv(OUTPUT_CSV, index=False)
    print(f"Centroids saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
