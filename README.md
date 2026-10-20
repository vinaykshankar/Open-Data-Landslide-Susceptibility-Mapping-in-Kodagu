# Kodagu Landslide Susceptibility Mapping

**Open-Data Landslide Susceptibility Mapping in Kodagu, Karnataka:
A Spatially Cross-Validated Assessment with Preliminary River-Channel Disturbance Analysis**

| | |
|---|---|
| **Author** | Vinay Kellengere Shankarnarayan |
| **Institution** | RV Institute of Management, Bengaluru, Karnataka, India |
| **ORCID** | [0000-0002-5759-0262](https://orcid.org/0000-0002-5759-0262) |
| **Journal** | Asian Journal of Geographical Research (AGJSR) |
| **DOI** | *https://doi.org/10.5281/zenodo.23493494* |

---

## Overview

This repository contains the Python analysis pipeline for a 30 m landslide
susceptibility assessment of Kodagu district, Karnataka, India. The study:

- Compiles a **461-scar 2018 landslide inventory** from Sentinel-2 NDVI differencing via Google Earth Engine
- Derives six **terrain conditioning factors** from the SRTM 1 arc-second DEM
- Fits three models — **Information Value (IV)**, **Logistic Regression (LR)**, and **Random Forest (RF)**
- Validates using both **random** and **spatial block cross-validation** (5-fold, ~10 km blocks)
- Interprets model behaviour with **SHAP** diagnostics
- Produces classified **Landslide Susceptibility Index (LSI)** raster maps
- Detects **river-channel change** on the Lakshmana Theertha using same-season Sentinel-2 MNDWI masks (2019 vs 2024)

**Study area:** Kodagu district (11°56′–12°52′ N; 75°22′–76°12′ E), Western Ghats, Karnataka, India

---

## Repository Structure

```
├── README.md
├── 01_inventory_compilation.py
├── 02_terrain_factors.py
├── 03_pseudo_absences.py
├── 04_modelling.py
├── 05_spatial_block_cv.py
├── 06_shap_diagnostics.py
├── 07_lsi_mapping.py
├── 08_river_change_detection.py
├── gee_inventory_script.js          # Google Earth Engine script (run in GEE)
└── data/
    ├── Kodagu_Landslide_500.zip      # 461-scar inventory shapefile
    ├── model_dataset.csv             # assembled point dataset (presence + factors)
    └── pseudo_absences_457.csv       # pseudo-absence coordinates
```

---

## Scripts

Run in order (01 → 08). Each script reads outputs produced by earlier scripts.

| # | Script | Stage | Description |
|---|--------|-------|-------------|
| 1 | `01_inventory_compilation.py` | Inventory | Filters GEE-exported scar polygons using five detection thresholds → 461-scar inventory |
| 2 | `02_terrain_factors.py` | Factor derivation | Derives slope, aspect, curvature, elevation, TWI from SRTM DEM (Horn 1981; Zevenbergen & Thorne 1987) |
| 3 | `03_pseudo_absences.py` | Sampling | Generates 457 pseudo-absence points with 500 m scar buffer and slope ≥ 5° filter |
| 4 | `04_modelling.py` | Modelling | Information Value, Logistic Regression, and Random Forest susceptibility models |
| 5 | `05_spatial_block_cv.py` | Validation | 5-fold spatial block cross-validation (~10 km blocks) for LR and RF |
| 6 | `06_shap_diagnostics.py` | Interpretability | SHAP TreeExplainer diagnostics — global importance, beeswarm plot, dependence plots |
| 7 | `07_lsi_mapping.py` | LSI mapping | Pixel-wise LSI probability maps + 5-class quantile zone classification |
| 8 | `08_river_change_detection.py` | River change | Same-season MNDWI change detection, Sentinel-2, 2019 vs 2024 dry season |

---

## Requirements

Python 3.9+ recommended.

```bash
pip install geopandas rasterio numpy pandas scikit-learn shap matplotlib openpyxl
```

---

## Data

### Input data required

| File | Source | Notes |
|------|--------|-------|
| `Kodagu_SRTM_merged.tif` | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) | SRTM 1 arc-second DEM, merged tiles, reprojected to UTM 43N |
| `Kodagu.shp` | Survey of India / KSRSAC | District boundary shapefile |
| `raw_scars_gee_export.geojson` | Google Earth Engine (see `gee_inventory_script.js`) | GEE-exported scar polygons before threshold filtering |
| `mndwi_2019_dry.tif` | [Copernicus Sentinel-2](https://scihub.copernicus.eu/) | MNDWI, dry season (January 2019) |
| `mndwi_2024_dry.tif` | Copernicus Sentinel-2 | MNDWI, dry season (January 2024) |
| `flow_accumulation.tif` | SAGA GIS / QGIS | Flow accumulation raster for TWI (optional) |

### Author-generated data (included in this repository)

| File | Description |
|------|-------------|
| `Kodagu_Landslide_500.zip` | 461-scar 2018 landslide inventory (Shapefile, UTM 43N) |
| `model_dataset.csv` | Point dataset: 461 presences + 457 pseudo-absences with all six terrain factors |
| `pseudo_absences_457.csv` | 457 pseudo-absence coordinates (WGS84 lat/lon) |

> **Note:** Planet imagery referenced in Figure 8 of the paper is proprietary and is **not** included. The quantitative river analysis uses only open Sentinel-2 data.

---

## Detection Thresholds (Inventory)

The following thresholds were applied to GEE-exported scar polygons (Section 3.2 of the paper):

| Threshold | Value |
|-----------|-------|
| ΔNDVI (pre–post) | > 0.20 |
| Post-event NDVI | < 0.35 |
| Minimum patch area | ≥ 900 m² |
| Maximum patch area | < 2 ha |
| Centroid slope | ≥ 15° |

---

## Key Results

| Model | Random split AUC | Spatial block CV AUC |
|-------|-----------------|----------------------|
| Logistic Regression | — | ~0.88 |
| Random Forest | — | ~0.89 |

- High + Very High zones cover **15.0%** of the district but contain **70%** of mapped scars (322/461)
- Spatial block CV AUC confirms discrimination holds under geographic hold-out
- Same-season river comparison (2019 vs 2024): no coherent lateral migration resolved at 10 m

---

## GEE Script

The Google Earth Engine script (`gee_inventory_script.js`) performs:
1. Sentinel-2 surface reflectance compositing (2018 pre- and post-monsoon)
2. NDVI differencing (ΔNDVI = pre-NDVI − post-NDVI)
3. Polygon delineation of candidate scar patches
4. Export to Google Drive as GeoJSON

To run: paste into the [GEE Code Editor](https://code.earthengine.google.com/) and click **Run**.

---

## Reproducibility

All third-party input data (SRTM, Sentinel-2, administrative boundaries) are freely available
from the sources listed above. The full analysis from raw DEM to final LSI map can be
reproduced by running scripts 01–08 in sequence with the required input files in place.

---

---

## License

Code: [MIT License](https://opensource.org/licenses/MIT)
Data (inventory): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
