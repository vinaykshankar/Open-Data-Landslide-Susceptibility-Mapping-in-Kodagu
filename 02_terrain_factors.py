"""
02_terrain_factors.py
======================
Derives terrain conditioning factors from the SRTM 1 arc-second DEM
and reclassifies them into the bins used in Table 2 of the paper.

Factors produced (all at 30 m, UTM Zone 43N / EPSG:32643):
  1. Slope          (degrees, Horn's method)
  2. Aspect         (degrees)
  3. Plan curvature (per 100 m)
  4. Profile curvature
  5. Elevation      (m)
  6. TWI            (Topographic Wetness Index)

Input:  Merged SRTM DEM (data/Kodagu_SRTM_merged.tif)
Output: Individual factor GeoTIFFs in data/factors/

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import generic_filter
import os

DEM_INPUT   = "data/Kodagu_SRTM_merged.tif"
OUT_DIR     = "data/factors"
CELL_SIZE   = 30.0   # metres (UTM 43N)


def save_raster(array, profile, path):
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
    print(f"  Saved: {path}")


def compute_slope_aspect(dem, cellsize):
    """Horn (1981) gradient — same as QGIS/ArcGIS default."""
    dz_dx = generic_filter(dem, lambda x: (x[2]+2*x[5]+x[8] - x[0]-2*x[3]-x[6]) / (8*cellsize),
                           size=3, mode="nearest")
    dz_dy = generic_filter(dem, lambda x: (x[6]+2*x[7]+x[8] - x[0]-2*x[1]-x[2]) / (8*cellsize),
                           size=3, mode="nearest")
    slope  = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx)) % 360
    return slope, aspect


def compute_curvature(dem, cellsize):
    """Plan and profile curvature (Zevenbergen & Thorne 1987)."""
    z = dem
    # 3×3 kernel coefficients
    D = generic_filter(z, lambda x: (x[3]+x[5]-2*x[4]) / (2*cellsize**2), size=3, mode="nearest")
    E = generic_filter(z, lambda x: (x[1]+x[7]-2*x[4]) / (2*cellsize**2), size=3, mode="nearest")
    plan_curv    = -2 * D * 100   # ×100 convention
    profile_curv = -2 * E * 100
    return plan_curv, profile_curv


def compute_twi(slope_deg, flowacc):
    """TWI = ln(A / tan(β));  flowacc in m²."""
    beta = np.radians(np.where(slope_deg < 0.001, 0.001, slope_deg))
    twi  = np.log(np.maximum(flowacc, 1.0) / np.tan(beta))
    return twi


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with rasterio.open(DEM_INPUT) as src:
        dem     = src.read(1).astype(np.float64)
        profile = src.profile
        nodata  = src.nodata or -9999
        dem[dem == nodata] = np.nan

    print("Computing slope and aspect …")
    slope, aspect = compute_slope_aspect(dem, CELL_SIZE)
    save_raster(slope,  profile, f"{OUT_DIR}/slope.tif")
    save_raster(aspect, profile, f"{OUT_DIR}/aspect.tif")

    print("Computing curvature …")
    plan_curv, prof_curv = compute_curvature(dem, CELL_SIZE)
    save_raster(plan_curv, profile, f"{OUT_DIR}/plan_curvature.tif")
    save_raster(prof_curv, profile, f"{OUT_DIR}/profile_curvature.tif")

    print("Saving elevation …")
    save_raster(dem.astype(np.float32), profile, f"{OUT_DIR}/elevation.tif")

    print("TWI: flow accumulation must be computed externally (SAGA/QGIS).")
    print("     If data/flow_accumulation.tif exists, TWI will be derived.")
    flowacc_path = "data/flow_accumulation.tif"
    if os.path.exists(flowacc_path):
        with rasterio.open(flowacc_path) as fsrc:
            flowacc = fsrc.read(1).astype(np.float64)
        flowacc_m2 = flowacc * (CELL_SIZE ** 2)
        twi = compute_twi(slope, flowacc_m2)
        save_raster(twi, profile, f"{OUT_DIR}/twi.tif")
    else:
        print("  Skipping TWI (flow accumulation not found).")

    print("\nAll terrain factors written to:", OUT_DIR)


if __name__ == "__main__":
    main()
