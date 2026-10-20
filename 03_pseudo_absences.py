"""
03_pseudo_absences.py
======================
Generates 457 pseudo-absence points for the susceptibility modelling,
matched 1:1 to the 461 landslide presence points (a small imbalance
arises from excluding pseudo-absences that fall on mapped scars or
within a 500 m buffer).

Strategy (paper Section 3.3):
  - Random sampling across the study area
  - Exclude cells within 500 m of any mapped scar
  - Exclude cells with slope < 5° (flat areas unlikely to fail)
  - Target ratio ≈ 1:1 (457 absences : 461 presences)

Input:  landslide_inventory_461.gpkg, slope.tif, Kodagu boundary
Output: pseudo_absences_457.csv  (lat, lon columns)

Study: Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka
Author: Vinay Kellengere Shankarnarayan
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point, box
import os

INVENTORY_PATH = "data/landslide_inventory_461.gpkg"
SLOPE_PATH     = "data/factors/slope.tif"
BOUNDARY_PATH  = "data/Kodagu.shp"
OUTPUT_CSV     = "data/pseudo_absences_457.csv"

BUFFER_M       = 500       # exclusion buffer around scars (metres)
SLOPE_MIN      = 5.0       # exclude flat cells below this slope
N_TARGET       = 457
RANDOM_SEED    = 42


def sample_candidates(boundary_geom, slope_src, n_candidates=50000):
    """Draw random points within the boundary, extract slope values."""
    bounds = boundary_geom.bounds  # minx, miny, maxx, maxy
    rng    = np.random.default_rng(RANDOM_SEED)

    xs = rng.uniform(bounds[0], bounds[2], n_candidates)
    ys = rng.uniform(bounds[1], bounds[3], n_candidates)

    # Filter to boundary
    pts    = [Point(x, y) for x, y in zip(xs, ys)]
    inside = [boundary_geom.contains(p) for p in pts]
    xs, ys = xs[inside], ys[inside]

    # Extract slope at each point
    coords = list(zip(xs, ys))
    rows, cols = zip(*[slope_src.index(x, y) for x, y in coords])
    slope_vals = slope_src.read(1)[list(rows), list(cols)]

    df = pd.DataFrame({"lon": xs, "lat": ys, "slope": slope_vals})
    return df


def main():
    os.makedirs("data", exist_ok=True)

    # Load boundary
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326").union_all()

    # Load inventory and buffer
    inv = gpd.read_file(INVENTORY_PATH).to_crs("EPSG:32643")
    buffer = inv.geometry.buffer(BUFFER_M).union_all()
    buffer_wgs = gpd.GeoSeries([buffer], crs="EPSG:32643").to_crs("EPSG:4326").iloc[0]

    with rasterio.open(SLOPE_PATH) as slope_src:
        candidates = sample_candidates(boundary, slope_src)

    # Apply exclusions
    candidates = candidates[candidates["slope"] >= SLOPE_MIN]
    cand_geom  = gpd.GeoSeries(
        [Point(r.lon, r.lat) for _, r in candidates.iterrows()],
        crs="EPSG:4326"
    )
    not_buffered = ~cand_geom.intersects(buffer_wgs)
    candidates   = candidates[not_buffered.values].reset_index(drop=True)

    print(f"Valid candidates after exclusions: {len(candidates)}")

    if len(candidates) < N_TARGET:
        raise ValueError(f"Not enough candidates ({len(candidates)} < {N_TARGET}). "
                         "Increase n_candidates or relax exclusion criteria.")

    absences = candidates.sample(n=N_TARGET, random_state=RANDOM_SEED)[["lat", "lon"]]
    absences["presence"] = 0
    absences.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(absences)} pseudo-absences to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
