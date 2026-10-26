# Input data: sources, preparation and access

[← Main project README](../README.md)

This directory is the **local workspace for input data**. Git tracks this README but ignores the data files. Original raw inputs, downloaded archives, research-specific extracts and generated raster datasets must be acquired separately. This guide describes the versions and transformations used in the *published model configuration*; it does not imply that every dataset can be redistributed or reconstructed pixel-for-pixel.

## 1. Required files

The filenames below match `GridBuilderConfig` and the additional acceptance modules. Preserve Shapefile sidecars (`.shx`, `.dbf`, `.prj`, possibly `.cpg`) under the same basename.

```text
Input Data/
  README.md
  Raster_shp/
    ger_1km_rectangle.shp              # + Shapefile sidecars
  VG250_GEM.shp                          # + sidecars
  VG250_LAN.shp                          # + sidecars
  de_onshore_wind_holtinger2016_med_100m.tif
  MaStR_Wind24.xlsx                      # sheet: 01012026
  ERA5_data_windspeed_2024.nc
  Zensus2022_Bevoelkerungszahl_1km-Gitter.csv
  Zensus2022_Durchschn_Flaeche_je_Bewohner_1km-Gitter.csv
  Zensus2022_Groesse_des_privaten_Haushalts_1km-Gitter.csv
  RWIGEOREDX_GRIDS_V16_SUF_YEAR_ABS.csv
  RWIGEOREDX_HOUPURC_V16_SUF_YEAR_ABS.xlsx
  Gemeindekoordinaten AuszugGV2QAktuell.xlsx  # sheet: Grossstaedte
  Scenicness_Labi_Netz/
    Scenicness_roth_export.tif
  windOnshoreTurbineData.csv
  windOnshoreTurbinePowerCurves.csv
```

**Optional:** `DEU_capacity-factor_IEC3.tif` (Global Wind Atlas) and `RWIGEOREDX_APPURC_V16_SUF_YEAR_ABS.xlsx` (apartment-purchase prices). The normal final grid configuration has `build_cf_from_gwa=False` and `use_hk_then_wk_before_muni=False`; neither optional file is required for that configuration.

## 2. Source and publication matrix

| # | Input source | Upstream reference | Redistribution / access decision |
|---|---|---|---|
| 1 | EEA 1-km grid | [EEA Reference Grid](https://www.eea.europa.eu/en/datahub/datahubitem-view/3c362237-daa4-45e2-8c16-aaadfb1a003b) | Download; document original cell alignment |
| 2 | BKG VG250 | [BKG VG250](https://gdz.bkg.bund.de/index.php/default/open-data/verwaltungsgebiete-1-250-000-stand-01-01-vg250-01-01.html) | Download; `dl-de/by-2-0`; exact local edition open |
| 3 | GLAES land eligibility | [GLAES GitHub](https://github.com/FZJ-IEK3-VSA/glaes) | Historical generation; exact output version not demonstrated |
| 4 | Global Wind Atlas | [GWA dataset](https://globalwindatlas.info/en/about/dataset) | Optional diagnosis only; original data CC BY 4.0 |
| 5 | MaStR wind stock | [BNetzA MaStR](https://www.marktstammdatenregister.de/MaStR/Datendownload) | Original local author extract retained; `dl-de/by-2-0`; assess public export before uploading |
| 6 | ERA5 hourly single levels | [CDS DOI](https://doi.org/10.24381/cds.adbb2d47) | Official download; Copernicus attribution/license |
| 7–9 | Zensus 2022 | [Zensus portal](https://ergebnisse.zensus2022.de/datenbank/online/themes) | Public Destatis download; `dl-de/by-2-0` |
| 10–12 | RWI-GEO-REDX | [V16 SUF DOI](https://doi.org/10.7807/immo:redx:suf:v16) | Access via FDZ Ruhr; **do not redistribute** |
| 13 | Destatis municipality directory | [Quarterly directory](https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV2QAktuell.html) | Manually processed; share exact extract only after source/rights review |
| 14 | BfN scenic beauty | [Roth et al. 2021](https://doi.org/10.19217/skr597) | Specific BfN/VGeoBund follow-on-product license; attribution required |
| 15 | Turbines and power curves | [Pöstges & Weber 2023](https://doi.org/10.1016/j.eneco.2023.106534) | Upload intended with paper citation; **no new open license assigned** |

The numeric labels 10–12 refer to REDX `GRIDS`, `HOUPURC` and optional `APPURC` variants, not three different data providers.

## 3. Preparation notes by dataset

### EEA reference grid (1)

Use the EEA **1-km polygon** grid for Germany in **ETRS89-LAEA (EPSG:3035)**, following Hermann Peifer's 2011 EEA reference-grid specification. Store the working Shapefile as `Raster_shp/ger_1km_rectangle.shp`. The model uses a normalized `idm` in `x_y` form and reconstructs `x_km`/`y_km`. An alternative grid with merely the same nominal cell size is not interchangeable without checking its origin, geometry and ID transformation.

### BKG administrative boundaries, VG250 (2)

Obtain municipality (`VG250_GEM`) and state (`VG250_LAN`) polygon layers from the same VG250 edition. The builder assigns the municipality code (`AGS_0` or `AGS`) and state using point-in-polygon joins of EEA grid-cell centers; for municipalities it applies the `GF == 4` filter if that field is available. It automatically reprojects as needed.

**Attribution template** (insert the actual year of the last data retrieval): `© BKG (year) dl-de/by-2-0, Datenquellen: https://sgx.geodatenzentrum.de/web_public/gdz/datenquellen/datenquellen_vg_nuts.pdf`. 

### GLAES land-eligibility TIFF (3)

`de_onshore_wind_holtinger2016_med_100m.tif` is a **research-generated** 100-m eligibility raster. The grid builder does **not** execute GLAES: it aggregates the already-generated TIFF onto the EEA 1-km grid as `pot_share` and `pot_area_km2`.

[GLAES v1.5.0 Zenodo software release](https://doi.org/10.5281/zenodo.20837257)

**Actual implemented `Holtinger2016_med` exclusions:**

| Criterion | Applied rule |
|---|---|
| Elevation; slope | ≥1,750 m; ≥8.5° |
| Settlements | ≤1,200 m buffer |
| Industry/mining; railways; main/secondary roads | ≤300 m buffers |
| Airports; power lines; national/protected parks | ≤5,100 m; ≤250 m; ≤2,000 m buffers |
| Rivers; protected habitat/bird/biosphere/landscape/monument | Direct overlap / zero buffer |
| Lakes | ≤1,750 m buffer |

**MIT license covers GLAES software**.

### Global Wind Atlas (4; optional)

The optional `DEU_capacity-factor_IEC3.tif` was downloaded through the GWA web application. Its capacity-factor diagnostic `cf_gwa` is not used in the paper's final LCOE, turbine selection or ERA5 yield calculations; `build_cf_from_gwa=False` is the distributed default. The GWA app's datasets are generally [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless otherwise indicated. Cite Davis et al. (2023), DOI [10.1175/BAMS-D-21-0075.1](https://doi.org/10.1175/BAMS-D-21-0075.1), and the specific dataset release if using the optional raster.

### Market Master Data Register: MaStR (5)

`MaStR_Wind24.xlsx`, sheet **`01012026`**, is the author's original working Excel export from the BNetzA register, manually reduced from publicly available information. The user reports that the original download was filtered to operating turbines (`In Betrieb`); the final working file does not itself retain the operating-status field to independently verify that filter. Do not present it as independently verified.

The supplied workbook was inspected: the model sheet has **27,587 unique MaStR identifiers**, **27,586 records with positive valid capacity** and **67.442 GW** before grid assignment; the latest commissioning date is **31 December 2025**. The workbook also has an older `01012025` sheet. The installed-stock capacity after spatial assignment is approximately **67.3 GW**.

The MaStR public-data license is **Datenlizenz Deutschland – Namensnennung 2.0**.

### ERA5 hourly winds, meteorological year 2024 (6)

Download [ERA5 hourly data on single levels](https://doi.org/10.24381/cds.adbb2d47) via the Copernicus Climate Data Store, specifying:

- Product: **Reanalysis**; period: **1 January–31 December 2024**, every hour (2024 is a leap year with **8,784 hours**).
- Variables: **100m u-component of wind (`u100`)**, **100m v-component (`v100`)**, **10m u-component (`u10`)**, **10m v-component (`v10`)**.
- Area: geographic bounding box covering **all of Germany**, including coastal/northern cells; the exact original box has not been recorded.
- Download format: **NetCDF**; working filename `ERA5_data_windspeed_2024.nc`.

The grid builder loads the four wind components, calculates speeds from vector magnitudes, extrapolates to turbine hub height, interpolates performance curves and computes `net_flh_era5`/`cf_era5`. The final paper yield inputs are **ERA5-based**; do not substitute a GWA capacity-factor raster. The model's 8,760-hour normalization convention should be documented when comparing calendar-year capacity factors.

### Zensus 2022 gridded statistics (7–9)

Download the official **1-km** grid CSVs using the [Zensus 2022 results portal](https://ergebnisse.zensus2022.de/datenbank/online/themes) and the Destatis topic/data-download pages; do not accidentally select 100-m or 10-km editions. Keep the filenames in §1 and the original grid identifiers so the builder can map them to EEA `idm`.

| Dataset / CSV prefix | Required column | Usage |
|---|---|---|
| `Zensus2022_Bevoelkerungszahl_1km-Gitter` | `Einwohner` | Population (`pop_2022`) |
| `Zensus2022_Durchschn_Flaeche_je_Bewohner_1km-Gitter` | `durchschnFlaechejeBew` | Floor area per inhabitant; multiplied by population |
| `Zensus2022_Groesse_des_privaten_Haushalts_1km-Gitter` | `Insgesamt_Haushalte` | Total number of private households (not mean household size) |

**License:** Datenlizenz Deutschland – Namensnennung 2.0 (`dl-de/by-2-0`); credit `Statistische Ämter des Bundes und der Länder, Zensus 2022` and disclose any edits.

### RWI-GEO-REDX **V16 SUF** (10–12)

The exact input edition is **RWI-GEO-REDX V16, Scientific Use File**, DOI [10.7807/immo:redx:suf:v16](https://doi.org/10.7807/immo:redx:suf:v16), not V17. The data are obtained through **FDZ Ruhr** subject to the provider's scientific-use terms; they are **not open input files** and must not be copied into public GitHub/Zenodo releases.

`RWIGEOREDX_GRIDS_V16_SUF_YEAR_ABS.csv` provides the gridded regional price information; `RWIGEOREDX_HOUPURC_V16_SUF_YEAR_ABS.xlsx` supplies house-purchase prices (`Munic_RegionEff_abs_yearly`). `RWIGEOREDX_APPURC_V16_SUF_YEAR_ABS.xlsx` is present in configuration but **not read in the normal case** (`use_hk_then_wk_before_muni=False`). The builder uses price records from 2015 onward, with temporal adjustment to **2025**, minimum observation checks and configured regional fallbacks. Even if original property data are not visibly included in an exported cost curve, redistribution rights for derived spatial outputs must be reviewed separately.

### Destatis municipality-center coordinates (13)

`Gemeindekoordinaten AuszugGV2QAktuell.xlsx` is a manually processed extract of the [Destatis quarterly municipality directory](https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV2QAktuell.html). The 'Frondel module' reads worksheet **`Grossstaedte`**, columns **`Name`, `Einwohner`, `Lon`, `Lat`**. Cities in that list determine the urban/rural distance classification (10-km distance criterion). The list was manually assembled.

### BfN / Roth landscape scenic beauty (14)

Original raster: `Labi_Netz.gdb/schoenheit`, **2021**, **1 km** resolution, beauty rating **1–9**. The delivered working GeoTIFF is `Scenicness_Labi_Netz/Scenicness_roth_export.tif`. The BfN product note cites Roth et al. (2021), *Entwicklung eines Bewertungsmodells zum Landschaftsbild beim Stromnetzausbau*, DOI [10.19217/skr597](https://doi.org/10.19217/skr597).

**Licensing:** the GeoTIFF is governed by the **specific BfN data license for VGeoBund-derived products**. Credit: `Bundesamt für Naturschutz (2021), erstellt unter Verwendung von Geobasisdaten © GeoBasis-DE / BKG (2021)`.

### Turbine specifications and performance curves (15)

The two inputs `windOnshoreTurbineData.csv` and `windOnshoreTurbinePowerCurves.csv` originate from research at the University of Duisburg-Essen associated with:

Pöstges, A., & Weber, C. (2023). *Identifying key elements for adequate simplifications of investment choices – The case of wind energy expansion*. **Energy Economics, 120**, 106534. [DOI: 10.1016/j.eneco.2023.106534](https://doi.org/10.1016/j.eneco.2023.106534).

The article describes eight representative turbine types and reports parameters in Table 2. The current model uses `turbineType6` and `turbineType7`, including wind-speed–power relationships from the original CSVs. The complete numeric power curves are not tabulated in the article.




For the exact model command, software environment, scenario descriptions and carefully filtered result export, return to the [main README](../README.md).
