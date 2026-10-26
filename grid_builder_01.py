# -*- coding: utf-8 -*-
"""Build the German 1-km model grid and ERA5-based wind yields.

The builder creates or extends three successive datasets:
- ``grid_m``: spatial reference cells and attributes;
- ``grid_de``: cells assigned to Germany and its administrative units;
- ``grid_de_era5``: model-ready grid with 2024 ERA5 energy yields.

It can be called from the central ``main.py`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin


# ============================================================
# Config
# ============================================================

@dataclass
class GridBuilderConfig:
    root: Path = Path(__file__).resolve().parent

    # --------------------------------------------------------
    # pipeline switches
    # --------------------------------------------------------
    build_grid_m: bool = True
    save_grid_m: bool = True
    save_grid_de: bool = True
    save_grid_de_era5: bool = True

    build_potential_from_tif: bool = True
    build_existing_from_mastr: bool = True
    build_cf_from_gwa: bool = False  # Optional GWA diagnostic; never used for final ERA5-based LCOE
    build_era5_yields: bool = True

    # if True, only keep selected columns from config output lists
    reduce_outputs_to_relevant_columns: bool = False

    # --------------------------------------------------------
    # outputs
    # --------------------------------------------------------
    out_dir_name: str = "Output"
    grid_m_parquet_name: str = "grid_m.parquet"
    grid_de_parquet_name: str = "grid_de.parquet"
    grid_de_era5_parquet_name: str = "grid_de_era5.parquet"

    # --------------------------------------------------------
    # base spatial inputs
    # --------------------------------------------------------
    grid_shp: str = r"Input Data\Raster_shp\ger_1km_rectangle.shp"
    vg250_gem: str = r"Input Data\VG250_GEM.shp"
    vg250_lan: str = r"Input Data\VG250_LAN.shp"

    # --------------------------------------------------------
    # REDX inputs
    # --------------------------------------------------------
    redx_grid_csv: str = r"Input Data\RWIGEOREDX_GRIDS_V16_SUF_YEAR_ABS.csv"
    redx_muni_hk_xlsx: str = r"Input Data\RWIGEOREDX_HOUPURC_V16_SUF_YEAR_ABS.xlsx"
    redx_muni_wk_xlsx: str = r"Input Data\RWIGEOREDX_APPURC_V16_SUF_YEAR_ABS.xlsx"
    redx_muni_sheet: str = "Munic_RegionEff_abs_yearly"
    redx_muni_sheet_wk: str = "Munic_RegionEff_abs_yearly"

    # REDX settings
    start_year: int = 2025
    min_year: int = 2015
    min_nobs: int = 2
    primary_segment: str = "HK"
    secondary_segment: str = "WK"
    use_hk_then_wk_before_muni: bool = False

    # temporal indexing
    target_price_year: int = 2025
    use_temporal_indexing: bool = True

    # --------------------------------------------------------
    # Zensus
    # --------------------------------------------------------
    zensus_pop_csv: str = r"Input Data\Zensus2022_Bevoelkerungszahl_1km-Gitter.csv"
    zensus_area_pc_csv: str = r"Input Data\Zensus2022_Durchschn_Flaeche_je_Bewohner_1km-Gitter.csv"
    zensus_households_csv: str = r"Input Data\Zensus2022_Groesse_des_privaten_Haushalts_1km-Gitter.csv"

    # --------------------------------------------------------
    # GLAES / GWA / MaStR
    # --------------------------------------------------------
    glaes_tif: Path = Path(__file__).resolve().parent / 'Input Data' / 'de_onshore_wind_holtinger2016_med_100m.tif'
    gwa_cf_tif: str = r"Input Data\DEU_capacity-factor_IEC3.tif"
    mastr_wind_xlsx: str = r"Input Data\MaStR_Wind24.xlsx"
    mastr_wind_sheet: str = "01012026"

    # --------------------------------------------------------
    # ERA5 / turbine inputs
    # --------------------------------------------------------
    era5_files: tuple[str, ...] = (
        r"Input Data\ERA5_data_windspeed_2024.nc",
    )
    turbine_data_csv: str = r"Input Data\windOnshoreTurbineData.csv"
    power_curves_csv: str = r"Input Data\windOnshoreTurbinePowerCurves.csv"

    # turbine names
    low_wind_turbine: str = "turbineType6"
    high_wind_turbine: str = "turbineType7"

    # mixing range
    mix_ws_low: float = 6.5
    mix_ws_high: float = 8.0
    mix_ws_mid: float = 7.25

    # global ERA5 correction on wind speed
    era5_wind_speed_factor: float = 1.0

    # technical production adjustments
    technical_availability: float = 0.90
    capacity_factor_scale: float = 0.93
    # --------------------------------------------------------
    # Existing fleet yield calibration
    # --------------------------------------------------------
    # Used only to approximate generation from the existing turbine fleet.
    # New-build placement and LCOE calculations still use net_flh_era5.
    existing_fleet_flh_scale: float = 0.78

    # coastal proxy
    north_coast_lat_deg: float = 54.50

    # ERA5 options
    use_only_potential_cells_for_era5: bool = False
    print_diagnostics: bool = True

    # --------------------------------------------------------
    # relevant output variables for later calculations
    # --------------------------------------------------------
    output_keep_geometry: bool = True

    output_id_cols: tuple[str, ...] = (
        "idm",
        "x_km",
        "y_km",
        "gid2019",
        "nuts3_id",
        "state_id",
        "state_name",
        "in_de",
    )

    output_price_cols: tuple[str, ...] = (
        "redx_grid_raw_eur_per_m2",
        "redx_grid_raw_nobs",
        "redx_grid_raw_year",
        "redx_grid_eur_per_m2",
        "redx_grid_nobs",
        "redx_grid_year",
        "redx_grid_source_year",
        "redx_grid_index_source",
        "redx_muni_eur_per_m2",
        "redx_muni_nobs",
        "redx_muni_year",
        "redx_muni_source_year",
        "redx_muni_index_source",
        "price_eur_per_m2_used",
        "price_year_used",
        "price_nobs_used",
        "price_source",
        "price_source_year",
        "price_index_source",
    )

    output_demography_cols: tuple[str, ...] = (
        "pop_2022",
        "households_total",
        "wohnflaeche_je_bewohner_m2",
        "area_total_m2",
    )
    
    output_potential_cols: tuple[str, ...] = (
        "pot_share",
        "pot_area_km2",
        "pot_area_m2",
    )

    output_existing_cols: tuple[str, ...] = (
        "n_existing",
        "existing_gw",
    )

    output_wind_resource_cols: tuple[str, ...] = (
        "cf_gwa",
        "era5_lat",
        "era5_lon",
        "dist_sea_km_proxy",
        "era5_ws10_mean_corr",
        "era5_ws100_mean",
        "era5_ws100_mean_corr",
        "share_type6_era5",
        "share_type7_era5",
        "turbine_type_era5",
        "hubheight_era5",
        "diameter_era5",
        "rated_power_kw_era5",
        "rated_power_mw_era5",
        "costs_era5",
        "mean_ws_hub_era5",
        "gross_mwh_era5",
        "gross_flh_era5",
        "cf_era5",
        "yield_correction_factor_era5",
        "net_mwh_era5",
        "net_flh_era5",
        "existing_fleet_flh_scale",
        "net_flh_existing_era5",
        "cf_existing_era5",        
    )


# ============================================================
# Public runner
# ============================================================

def run_grid_builder(cfg: GridBuilderConfig) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    out_dir = cfg.root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    path_grid_m = out_dir / cfg.grid_m_parquet_name
    path_grid_de = out_dir / cfg.grid_de_parquet_name
    path_grid_de_era5 = out_dir / cfg.grid_de_era5_parquet_name

    # --------------------------------------------------------
    # build / load grid_m
    # --------------------------------------------------------
    if (not cfg.build_grid_m) and path_grid_m.exists():
        print(f"[LOAD] grid_m <- {path_grid_m}")
        grid_m = gpd.read_parquet(path_grid_m)
    else:
        print("[BUILD] Building grid_m from scratch...")
        grid_m = build_grid_m(cfg)

    # --------------------------------------------------------
    # optional GWA
    # --------------------------------------------------------
    if cfg.build_cf_from_gwa and "cf_gwa" not in grid_m.columns:
        print(f"[GWA] Aggregating CF -> 1km: {cfg.gwa_cf_tif}")
        cf_df = aggregate_mean_tif_to_1km_sumweight(
            grid_m, cfg.root / cfg.gwa_cf_tif, out_col="cf_gwa"
        )
        grid_m = grid_m.merge(cf_df, on="idm", how="left")

        grid_de_tmp = grid_m.loc[grid_m["in_de"]].copy()
        print(
            f"[GWA] cf_gwa coverage (DE): {grid_de_tmp['cf_gwa'].notna().mean():.3f} "
            f"| mean={grid_de_tmp['cf_gwa'].mean():.3f}"
        )

    # --------------------------------------------------------
    # optional GLAES
    # --------------------------------------------------------
    if cfg.build_potential_from_tif and "pot_share" not in grid_m.columns:
        print(f"[POT] Aggregating GLAES TIF -> 1km: {cfg.glaes_tif}")
        pot_df = aggregate_eligibility_tif_to_1km(grid_m, cfg.glaes_tif)
        grid_m = grid_m.merge(pot_df, on="idm", how="left")

        grid_de_tmp = grid_m.loc[grid_m["in_de"]].copy()
        eligible_km2 = grid_de_tmp["pot_area_km2"].fillna(0).sum()
        total_km2 = len(grid_de_tmp) * 1.0
        print(
            f"[POT] Eligible share (DE land cells): {eligible_km2 / total_km2:.4f}  "
            f"(eligible={eligible_km2:,.1f} km², total={total_km2:,.1f} km²)"
        )

    # --------------------------------------------------------
    # optional MaStR
    # --------------------------------------------------------
    if cfg.build_existing_from_mastr and "n_existing" not in grid_m.columns:
        print(
            f"[MaStR] Counting existing turbines per 1km cell: "
            f"{cfg.mastr_wind_xlsx} / {cfg.mastr_wind_sheet}"
        )
        grid_m = attach_mastr_existing_turbines(
            grid_m, cfg.root / cfg.mastr_wind_xlsx, cfg.mastr_wind_sheet
        )
        print(f"[MaStR] Total turbines counted: {int(grid_m['n_existing'].sum()):,}")
        print(f"[MaStR] Total installed capacity (GW): {grid_m['existing_gw'].sum():.3f}")
        print(f"[MaStR] Cells with >=1 turbine: {int((grid_m['n_existing'] > 0).sum()):,}")

    # --------------------------------------------------------
    # derive grid_de
    # --------------------------------------------------------
    grid_de = grid_m.loc[grid_m["in_de"]].copy()

    print(f"[DONE] grid_m rows: {len(grid_m):,} | grid_de rows: {len(grid_de):,}")
    print(f"[DONE] Price coverage (used): {grid_de['price_eur_per_m2_used'].notna().mean():.3f}")
    print(f"[DONE] Zensus pop coverage: {grid_de['pop_2022'].notna().mean():.3f}")

    if cfg.reduce_outputs_to_relevant_columns:
        grid_m = select_relevant_output_columns(grid_m, cfg, stage="grid_m")
        grid_de = select_relevant_output_columns(grid_de, cfg, stage="grid_de")

    if cfg.save_grid_m:
        print(f"[SAVE] grid_m -> {path_grid_m}")
        grid_m.to_parquet(path_grid_m, index=False)

    if cfg.save_grid_de:
        print(f"[SAVE] grid_de -> {path_grid_de}")
        grid_de.to_parquet(path_grid_de, index=False)

    # --------------------------------------------------------
    # ERA5 extension
    # --------------------------------------------------------
    if cfg.build_era5_yields:
        print("[ERA5] Extending grid_de with ERA5 yields")
        grid_de_era5 = extend_grid_de_with_era5(grid_de, cfg)

        if cfg.reduce_outputs_to_relevant_columns:
            grid_de_era5 = select_relevant_output_columns(grid_de_era5, cfg, stage="grid_de_era5")

        if cfg.save_grid_de_era5:
            print(f"[SAVE] grid_de_era5 -> {path_grid_de_era5}")
            grid_de_era5.to_parquet(path_grid_de_era5, index=False)
    else:
        grid_de_era5 = grid_de.copy()

    print("[DONE] Grid builder module finished.")
    return grid_m, grid_de, grid_de_era5


# ============================================================
# Output column handling
# ============================================================

def get_relevant_output_columns(cfg: GridBuilderConfig) -> list[str]:
    cols = []
    cols.extend(cfg.output_id_cols)
    cols.extend(cfg.output_price_cols)
    cols.extend(cfg.output_demography_cols)
    cols.extend(cfg.output_potential_cols)
    cols.extend(cfg.output_existing_cols)
    cols.extend(cfg.output_wind_resource_cols)
    return list(dict.fromkeys(cols))


def select_relevant_output_columns(
    gdf: gpd.GeoDataFrame,
    cfg: GridBuilderConfig,
    stage: str,
) -> gpd.GeoDataFrame:
    keep = [c for c in get_relevant_output_columns(cfg) if c in gdf.columns]
    if cfg.output_keep_geometry and "geometry" in gdf.columns:
        keep = keep + ["geometry"]
    keep = list(dict.fromkeys(keep))
    print(f"[OUTPUT] {stage}: keeping {len(keep)} relevant columns")
    return gdf[keep].copy()


# ============================================================
# Grid build
# ============================================================

def build_grid_m(cfg: GridBuilderConfig) -> gpd.GeoDataFrame:
    years = list(range(cfg.start_year, cfg.min_year - 1, -1))
    pcols = [f"pindex{y}" for y in years]
    ncols = [f"NOBS{y}" for y in years]
    years_arr = np.array(years, dtype=float)

    grid_m = load_base_grid(cfg)
    print(f"[GRID] Loaded 1km grid: {len(grid_m):,} cells | CRS: {grid_m.crs}")

    grid_m = attach_gid_and_mask(cfg, grid_m)
    print(f"[VG250] in_de share: {grid_m['in_de'].mean():.3f}")

    grid_m = attach_state(cfg, grid_m)
    print(f"[VG250] State coverage: {grid_m['state_name'].notna().mean():.3f}")

    grid_m = attach_redx_grid_prices(cfg, grid_m, pcols, ncols, years_arr)
    print(f"[REDX] Raw grid price coverage: {grid_m['redx_grid_raw_eur_per_m2'].notna().mean():.3f}")

    grid_m, muni_ratio_lookup, nuts3_ratio_lookup = attach_redx_muni_prices(
        cfg, grid_m, pcols, ncols, years_arr
    )

    if cfg.use_temporal_indexing:
        grid_m = temporal_index_grid_prices(
            cfg=cfg,
            grid_m=grid_m,
            muni_ratio_lookup=muni_ratio_lookup,
            nuts3_ratio_lookup=nuts3_ratio_lookup,
        )
    else:
        grid_m["redx_grid_eur_per_m2"] = grid_m["redx_grid_raw_eur_per_m2"]
        grid_m["redx_grid_nobs"] = grid_m["redx_grid_raw_nobs"]
        grid_m["redx_grid_year"] = grid_m["redx_grid_raw_year"]
        grid_m["redx_grid_source_year"] = grid_m["redx_grid_raw_year"]
        grid_m["redx_grid_index_source"] = np.where(
            grid_m["redx_grid_raw_eur_per_m2"].notna(), "RAW", np.nan
        )

    print(f"[REDX] Indexed grid price coverage: {grid_m['redx_grid_eur_per_m2'].notna().mean():.3f}")
    print(f"[REDX] Indexed muni price coverage: {grid_m['redx_muni_eur_per_m2'].notna().mean():.3f}")

    grid_m = build_final_price_used(grid_m)
    print(f"[REDX] Final price coverage (used): {grid_m['price_eur_per_m2_used'].notna().mean():.3f}")

    grid_m = attach_zensus(cfg, grid_m)
    print(
        f"[ZENSUS] pop coverage: {grid_m['pop_2022'].notna().mean():.3f} | "
        f"area_total_m2 coverage: {grid_m['area_total_m2'].notna().mean():.3f}"
    )

    return grid_m


# ============================================================
# Base grid / VG250
# ============================================================

def load_base_grid(cfg: GridBuilderConfig) -> gpd.GeoDataFrame:
    grid = gpd.read_file(cfg.root / cfg.grid_shp)[["idm", "geometry"]].copy()
    tmp = grid["idm"].astype(str).str.split("_", expand=True)
    grid["x_km"] = pd.to_numeric(tmp[0], errors="coerce")
    grid["y_km"] = pd.to_numeric(tmp[1], errors="coerce")
    return grid[["idm", "x_km", "y_km", "geometry"]].copy()


def attach_gid_and_mask(cfg: GridBuilderConfig, grid_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gem = gpd.read_file(cfg.root / cfg.vg250_gem)
    if "GF" in gem.columns:
        gem = gem.loc[gem["GF"] == 4].copy()

    ags_col = "AGS_0" if "AGS_0" in gem.columns else "AGS"
    gem["gid2019"] = gem[ags_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)

    if gem.crs != grid_m.crs:
        gem = gem.to_crs(grid_m.crs)

    grid_cent = grid_m[["idm", "geometry"]].copy()
    grid_cent["geometry"] = grid_cent.geometry.centroid

    joined = gpd.sjoin(
        grid_cent,
        gem[["gid2019", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    grid_m = grid_m.merge(joined[["idm", "gid2019"]], on="idm", how="left")
    grid_m["in_de"] = grid_m["gid2019"].notna()
    grid_m["nuts3_id"] = build_nuts3_from_gid(grid_m["gid2019"])
    return grid_m


def attach_state(cfg: GridBuilderConfig, grid_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lan = gpd.read_file(cfg.root / cfg.vg250_lan)

    if lan.crs != grid_m.crs:
        lan = lan.to_crs(grid_m.crs)

    name_col = "GEN" if "GEN" in lan.columns else ("NAME" if "NAME" in lan.columns else None)
    code_col = None
    for cand in ["RS", "AGS", "AGS_0", "LAN", "ID", "KRS", "SCHLUESSEL"]:
        if cand in lan.columns:
            code_col = cand
            break

    keep = ["geometry"]
    if name_col:
        keep.append(name_col)
    if code_col:
        keep.append(code_col)
    lan = lan[keep].copy()

    cent = grid_m[["idm", "geometry"]].copy()
    cent["geometry"] = cent.geometry.centroid

    joined = gpd.sjoin(
        cent,
        lan,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    out = grid_m.merge(joined.drop(columns="geometry"), on="idm", how="left")

    if name_col:
        out = out.rename(columns={name_col: "state_name"})
    else:
        out["state_name"] = np.nan

    if code_col:
        codes = out[code_col].astype(str).str.replace(r"\D", "", regex=True)
        out["state_id"] = codes.str.slice(0, 2)
        out = out.drop(columns=[code_col])
    else:
        out["state_id"] = np.nan

    return out


# ============================================================
# REDX helpers
# ============================================================

def latest_valid_from_wide(
    df: pd.DataFrame,
    pcols: list[str],
    ncols: list[str],
    years_arr: np.ndarray,
    min_nobs: int,
    out_prefix: str,
) -> pd.DataFrame:
    P = df[pcols].to_numpy(dtype=float)
    N = df[ncols].to_numpy(dtype=float)

    valid = np.isfinite(P) & np.isfinite(N) & (N >= float(min_nobs))
    has_any = valid.any(axis=1)
    idx = valid.argmax(axis=1)
    row = np.arange(len(df))

    out = df.copy()
    out[f"{out_prefix}_eur_per_m2_raw"] = np.where(has_any, P[row, idx], np.nan)
    out[f"{out_prefix}_nobs_raw"] = np.where(has_any, N[row, idx], np.nan)
    out[f"{out_prefix}_year_raw"] = np.where(has_any, years_arr[idx], np.nan)
    return out


def wide_to_long_prices(
    df: pd.DataFrame,
    id_col: str,
    pcols: list[str],
    ncols: list[str],
) -> pd.DataFrame:
    price_long = df[[id_col] + pcols].melt(
        id_vars=id_col, var_name="pcol", value_name="price"
    )
    nobs_long = df[[id_col] + ncols].melt(
        id_vars=id_col, var_name="ncol", value_name="nobs"
    )

    price_long["year"] = price_long["pcol"].str.replace("pindex", "", regex=False).astype(int)
    nobs_long["year"] = nobs_long["ncol"].str.replace("NOBS", "", regex=False).astype(int)

    out = price_long[[id_col, "year", "price"]].merge(
        nobs_long[[id_col, "year", "nobs"]],
        on=[id_col, "year"],
        how="left",
    )
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["nobs"] = pd.to_numeric(out["nobs"], errors="coerce")
    return out


def build_nuts3_from_gid(gid: pd.Series) -> pd.Series:
    gid_str = gid.astype("string")
    gid_digits = gid_str.str.replace(r"\D", "", regex=True).str.zfill(8)
    out = gid_digits.str.slice(0, 5)
    out = out.where(gid.notna(), pd.NA)
    return out


def compute_group_year_index(
    long_df: pd.DataFrame,
    group_col: str,
    min_nobs: int,
) -> pd.DataFrame:
    tmp = long_df.copy()
    tmp = tmp.loc[
        tmp[group_col].notna() &
        tmp["price"].notna() &
        tmp["nobs"].notna() &
        (tmp["nobs"] >= float(min_nobs))
    ].copy()

    if tmp.empty:
        return pd.DataFrame(columns=[group_col, "year", "grp_price"])

    agg = (
        tmp.assign(w_price=tmp["price"] * tmp["nobs"])
        .groupby([group_col, "year"], as_index=False)
        .agg(
            w_price=("w_price", "sum"),
            w=("nobs", "sum"),
        )
    )
    agg["grp_price"] = np.where(agg["w"] > 0, agg["w_price"] / agg["w"], np.nan)
    return agg[[group_col, "year", "grp_price"]].copy()


def build_group_ratio_lookup(
    group_year_df: pd.DataFrame,
    group_col: str,
    target_year: int,
) -> pd.DataFrame:
    tgt = group_year_df.loc[
        group_year_df["year"] == target_year, [group_col, "grp_price"]
    ].rename(columns={"grp_price": "grp_price_target"})

    out = group_year_df.merge(tgt, on=group_col, how="left")
    out["ratio_to_target"] = np.where(
        out["grp_price"].notna() &
        out["grp_price_target"].notna() &
        (out["grp_price"] > 0),
        out["grp_price_target"] / out["grp_price"],
        np.nan,
    )
    return out[[group_col, "year", "ratio_to_target"]].copy()


def apply_temporal_indexing(
    base_df: pd.DataFrame,
    value_col_raw: str,
    year_col_raw: str,
    nobs_col_raw: str,
    gid_col: str,
    nuts3_col: str,
    muni_ratio_lookup: pd.DataFrame,
    nuts3_ratio_lookup: pd.DataFrame,
    target_year: int,
    out_prefix: str,
) -> pd.DataFrame:
    out = base_df.copy()

    yr = pd.to_numeric(out[year_col_raw], errors="coerce")
    out["_src_year_int"] = yr.astype("Int64")

    muni_lu = muni_ratio_lookup.rename(columns={
        "gid2019": gid_col,
        "year": "_src_year_int",
        "ratio_to_target": "_ratio_muni",
    })
    out = out.merge(muni_lu, on=[gid_col, "_src_year_int"], how="left")

    nuts3_lu = nuts3_ratio_lookup.rename(columns={
        "nuts3_id": nuts3_col,
        "year": "_src_year_int",
        "ratio_to_target": "_ratio_nuts3",
    })
    out = out.merge(nuts3_lu, on=[nuts3_col, "_src_year_int"], how="left")

    ratio = out["_ratio_muni"].where(out["_ratio_muni"].notna(), out["_ratio_nuts3"])
    source = np.where(
        out["_ratio_muni"].notna(),
        "MUNI_INDEX",
        np.where(out["_ratio_nuts3"].notna(), "NUTS3_INDEX", "RAW"),
    )

    effective_ratio = ratio.where(ratio.notna(), 1.0)

    out[f"{out_prefix}_eur_per_m2"] = np.where(
        out[value_col_raw].notna(),
        out[value_col_raw] * effective_ratio,
        np.nan,
    )
    out[f"{out_prefix}_year"] = np.where(
        out[value_col_raw].notna(),
        float(target_year),
        np.nan,
    )
    out[f"{out_prefix}_nobs"] = out[nobs_col_raw]
    out[f"{out_prefix}_source_year"] = out[year_col_raw]

    idx_source = pd.Series(source, index=out.index, dtype="object")
    idx_source = idx_source.where(out[value_col_raw].notna(), None)
    out[f"{out_prefix}_index_source"] = idx_source

    out = out.drop(columns=["_src_year_int", "_ratio_muni", "_ratio_nuts3"], errors="ignore")
    return out


# ============================================================
# REDX grid / municipality prices
# ============================================================

def attach_redx_grid_prices(
    cfg: GridBuilderConfig,
    grid_m: gpd.GeoDataFrame,
    pcols: list[str],
    ncols: list[str],
    years_arr: np.ndarray,
) -> gpd.GeoDataFrame:
    usecols = ["grid", "housing_type"] + pcols + ncols
    redx = pd.read_csv(cfg.root / cfg.redx_grid_csv, usecols=usecols)
    redx["grid"] = redx["grid"].astype(str)

    for c in pcols + ncols:
        redx[c] = pd.to_numeric(redx[c], errors="coerce")

    hk = redx.loc[redx["housing_type"] == cfg.primary_segment].copy()
    hk = latest_valid_from_wide(hk, pcols, ncols, years_arr, cfg.min_nobs, "redx_grid_hk_raw")
    out = hk[[
        "grid",
        "redx_grid_hk_raw_eur_per_m2_raw",
        "redx_grid_hk_raw_nobs_raw",
        "redx_grid_hk_raw_year_raw",
    ]].rename(columns={
        "grid": "idm",
        "redx_grid_hk_raw_eur_per_m2_raw": "redx_grid_raw_eur_per_m2",
        "redx_grid_hk_raw_nobs_raw": "redx_grid_raw_nobs",
        "redx_grid_hk_raw_year_raw": "redx_grid_raw_year",
    })

    if cfg.use_hk_then_wk_before_muni:
        wk = redx.loc[redx["housing_type"] == cfg.secondary_segment].copy()
        wk = latest_valid_from_wide(wk, pcols, ncols, years_arr, cfg.min_nobs, "redx_grid_wk_raw")
        wk = wk[[
            "grid",
            "redx_grid_wk_raw_eur_per_m2_raw",
            "redx_grid_wk_raw_nobs_raw",
            "redx_grid_wk_raw_year_raw",
        ]].rename(columns={
            "grid": "idm",
            "redx_grid_wk_raw_eur_per_m2_raw": "redx_grid_wk_raw_eur_per_m2",
            "redx_grid_wk_raw_nobs_raw": "redx_grid_wk_raw_nobs",
            "redx_grid_wk_raw_year_raw": "redx_grid_wk_raw_year",
        })

        out = out.merge(wk, on="idm", how="left")

        m = out["redx_grid_raw_eur_per_m2"].isna() & out["redx_grid_wk_raw_eur_per_m2"].notna()
        out.loc[m, "redx_grid_raw_eur_per_m2"] = out.loc[m, "redx_grid_wk_raw_eur_per_m2"]
        out.loc[m, "redx_grid_raw_nobs"] = out.loc[m, "redx_grid_wk_raw_nobs"]
        out.loc[m, "redx_grid_raw_year"] = out.loc[m, "redx_grid_wk_raw_year"]

    grid_m = grid_m.merge(
        out[["idm", "redx_grid_raw_eur_per_m2", "redx_grid_raw_nobs", "redx_grid_raw_year"]],
        on="idm",
        how="left",
    )
    return grid_m


def attach_redx_muni_prices(
    cfg: GridBuilderConfig,
    grid_m: gpd.GeoDataFrame,
    pcols: list[str],
    ncols: list[str],
    years_arr: np.ndarray,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    hk = pd.read_excel(cfg.root / cfg.redx_muni_hk_xlsx, sheet_name=cfg.redx_muni_sheet)
    hk["gid2019"] = hk["gid2019"].astype(str).str.zfill(8)

    for c in pcols + ncols:
        hk[c] = pd.to_numeric(hk[c], errors="coerce")

    hk_raw = latest_valid_from_wide(hk, pcols, ncols, years_arr, cfg.min_nobs, "redx_muni_raw")

    muni = hk_raw[[
        "gid2019",
        "redx_muni_raw_eur_per_m2_raw",
        "redx_muni_raw_nobs_raw",
        "redx_muni_raw_year_raw",
    ]].rename(columns={
        "redx_muni_raw_eur_per_m2_raw": "redx_muni_raw_eur_per_m2",
        "redx_muni_raw_nobs_raw": "redx_muni_raw_nobs",
        "redx_muni_raw_year_raw": "redx_muni_raw_year",
    })

    if cfg.use_hk_then_wk_before_muni:
        wk = pd.read_excel(cfg.root / cfg.redx_muni_wk_xlsx, sheet_name=cfg.redx_muni_sheet_wk)
        wk["gid2019"] = wk["gid2019"].astype(str).str.zfill(8)

        for c in pcols + ncols:
            wk[c] = pd.to_numeric(wk[c], errors="coerce")

        wk_raw = latest_valid_from_wide(wk, pcols, ncols, years_arr, cfg.min_nobs, "redx_muni_wk_raw")
        wk_raw = wk_raw[[
            "gid2019",
            "redx_muni_wk_raw_eur_per_m2_raw",
            "redx_muni_wk_raw_nobs_raw",
            "redx_muni_wk_raw_year_raw",
        ]].rename(columns={
            "redx_muni_wk_raw_eur_per_m2_raw": "redx_muni_wk_raw_eur_per_m2",
            "redx_muni_wk_raw_nobs_raw": "redx_muni_wk_raw_nobs",
            "redx_muni_wk_raw_year_raw": "redx_muni_wk_raw_year",
        })

        muni = muni.merge(wk_raw, on="gid2019", how="left")
        m = muni["redx_muni_raw_eur_per_m2"].isna() & muni["redx_muni_wk_raw_eur_per_m2"].notna()
        muni.loc[m, "redx_muni_raw_eur_per_m2"] = muni.loc[m, "redx_muni_wk_raw_eur_per_m2"]
        muni.loc[m, "redx_muni_raw_nobs"] = muni.loc[m, "redx_muni_wk_raw_nobs"]
        muni.loc[m, "redx_muni_raw_year"] = muni.loc[m, "redx_muni_wk_raw_year"]

    muni_long = wide_to_long_prices(hk, id_col="gid2019", pcols=pcols, ncols=ncols)
    muni_long["nuts3_id"] = build_nuts3_from_gid(muni_long["gid2019"])

    muni_group_year = compute_group_year_index(
        muni_long, group_col="gid2019", min_nobs=cfg.min_nobs
    )
    muni_ratio_lookup = build_group_ratio_lookup(
        muni_group_year,
        group_col="gid2019",
        target_year=cfg.target_price_year,
    )

    nuts3_group_year = compute_group_year_index(
        muni_long, group_col="nuts3_id", min_nobs=cfg.min_nobs
    )
    nuts3_ratio_lookup = build_group_ratio_lookup(
        nuts3_group_year,
        group_col="nuts3_id",
        target_year=cfg.target_price_year,
    )

    muni["nuts3_id"] = build_nuts3_from_gid(muni["gid2019"])

    muni = apply_temporal_indexing(
        base_df=muni,
        value_col_raw="redx_muni_raw_eur_per_m2",
        year_col_raw="redx_muni_raw_year",
        nobs_col_raw="redx_muni_raw_nobs",
        gid_col="gid2019",
        nuts3_col="nuts3_id",
        muni_ratio_lookup=muni_ratio_lookup,
        nuts3_ratio_lookup=nuts3_ratio_lookup,
        target_year=cfg.target_price_year,
        out_prefix="redx_muni",
    )

    muni = muni[[
        "gid2019",
        "redx_muni_eur_per_m2",
        "redx_muni_nobs",
        "redx_muni_year",
        "redx_muni_source_year",
        "redx_muni_index_source",
    ]].copy()

    grid_m = grid_m.merge(muni, on="gid2019", how="left")
    return grid_m, muni_ratio_lookup, nuts3_ratio_lookup


def temporal_index_grid_prices(
    cfg: GridBuilderConfig,
    grid_m: gpd.GeoDataFrame,
    muni_ratio_lookup: pd.DataFrame,
    nuts3_ratio_lookup: pd.DataFrame,
) -> gpd.GeoDataFrame:
    grid_m = apply_temporal_indexing(
        base_df=grid_m,
        value_col_raw="redx_grid_raw_eur_per_m2",
        year_col_raw="redx_grid_raw_year",
        nobs_col_raw="redx_grid_raw_nobs",
        gid_col="gid2019",
        nuts3_col="nuts3_id",
        muni_ratio_lookup=muni_ratio_lookup,
        nuts3_ratio_lookup=nuts3_ratio_lookup,
        target_year=cfg.target_price_year,
        out_prefix="redx_grid",
    )
    return grid_m


def build_final_price_used(grid_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_m["price_source"] = pd.Series(index=grid_m.index, dtype="object")
    grid_m["price_index_source"] = pd.Series(index=grid_m.index, dtype="object")
    grid_m["price_source_year"] = np.nan

    grid_m["price_eur_per_m2_used"] = grid_m["redx_grid_eur_per_m2"]
    grid_m["price_year_used"] = grid_m["redx_grid_year"]
    grid_m["price_nobs_used"] = grid_m["redx_grid_nobs"]
    grid_m["price_source_year"] = grid_m["redx_grid_source_year"]
    grid_m["price_index_source"] = grid_m["redx_grid_index_source"]
    grid_m.loc[grid_m["price_eur_per_m2_used"].notna(), "price_source"] = "REDX_GRID"

    fb = grid_m["price_eur_per_m2_used"].isna() & grid_m["redx_muni_eur_per_m2"].notna()
    grid_m.loc[fb, "price_eur_per_m2_used"] = grid_m.loc[fb, "redx_muni_eur_per_m2"]
    grid_m.loc[fb, "price_year_used"] = grid_m.loc[fb, "redx_muni_year"]
    grid_m.loc[fb, "price_nobs_used"] = grid_m.loc[fb, "redx_muni_nobs"]
    grid_m.loc[fb, "price_source_year"] = grid_m.loc[fb, "redx_muni_source_year"]
    grid_m.loc[fb, "price_index_source"] = grid_m.loc[fb, "redx_muni_index_source"]
    grid_m.loc[fb, "price_source"] = "REDX_MUNI"

    return grid_m


# ============================================================
# Zensus
# ============================================================

def attach_zensus(cfg: GridBuilderConfig, grid_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pop = pd.read_csv(cfg.root / cfg.zensus_pop_csv, sep=";", decimal=",")
    area_pc = pd.read_csv(cfg.root / cfg.zensus_area_pc_csv, sep=";", decimal=",")
    hh = pd.read_csv(cfg.root / cfg.zensus_households_csv, sep=";", decimal=",")

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------
    x = pd.to_numeric(pop["x_mp_1km"], errors="coerce")
    y = pd.to_numeric(pop["y_mp_1km"], errors="coerce")
    pop["idm"] = (
        ((x - 500) // 1000).astype("Int64").astype(str)
        + "_"
        + (((y - 500) // 1000).astype("Int64").astype(str))
    )
    pop_t = pop[["idm", "Einwohner"]].rename(columns={"Einwohner": "pop_2022"}).copy()
    pop_t["pop_2022"] = pd.to_numeric(pop_t["pop_2022"], errors="coerce")

    # --------------------------------------------------------
    # Living area per inhabitant
    # --------------------------------------------------------
    x2 = pd.to_numeric(area_pc["x_mp_1km"], errors="coerce")
    y2 = pd.to_numeric(area_pc["y_mp_1km"], errors="coerce")
    area_pc["idm"] = (
        ((x2 - 500) // 1000).astype("Int64").astype(str)
        + "_"
        + (((y2 - 500) // 1000).astype("Int64").astype(str))
    )
    area_t = area_pc[["idm", "durchschnFlaechejeBew"]].rename(
        columns={"durchschnFlaechejeBew": "wohnflaeche_je_bewohner_m2"}
    ).copy()
    area_t["wohnflaeche_je_bewohner_m2"] = pd.to_numeric(
        area_t["wohnflaeche_je_bewohner_m2"], errors="coerce"
    )

    # --------------------------------------------------------
    # Households total
    # --------------------------------------------------------
    x3 = pd.to_numeric(hh["x_mp_1km"], errors="coerce")
    y3 = pd.to_numeric(hh["y_mp_1km"], errors="coerce")
    hh["idm"] = (
        ((x3 - 500) // 1000).astype("Int64").astype(str)
        + "_"
        + (((y3 - 500) // 1000).astype("Int64").astype(str))
    )
    hh_t = hh[["idm", "Insgesamt_Haushalte"]].rename(
        columns={"Insgesamt_Haushalte": "households_total"}
    ).copy()
    hh_t["households_total"] = pd.to_numeric(hh_t["households_total"], errors="coerce")

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------
    grid_m = (
        grid_m
        .merge(pop_t, on="idm", how="left")
        .merge(area_t, on="idm", how="left")
        .merge(hh_t, on="idm", how="left")
    )

    grid_m["area_total_m2"] = (
        grid_m["pop_2022"].astype(float) * grid_m["wohnflaeche_je_bewohner_m2"].astype(float)
    )

    return grid_m


# ============================================================
# Raster helpers
# ============================================================

def build_1km_template_from_grid_m(grid_m: gpd.GeoDataFrame):
    xmin_km = int(np.nanmin(grid_m["x_km"].to_numpy()))
    xmax_km = int(np.nanmax(grid_m["x_km"].to_numpy()))
    ymin_km = int(np.nanmin(grid_m["y_km"].to_numpy()))
    ymax_km = int(np.nanmax(grid_m["y_km"].to_numpy()))

    width = xmax_km - xmin_km + 1
    height = ymax_km - ymin_km + 1

    left = xmin_km * 1000.0
    top = (ymax_km + 1) * 1000.0

    dst_transform = from_origin(left, top, 1000.0, 1000.0)
    dst_shape = (height, width)
    return xmin_km, ymin_km, ymax_km, dst_transform, dst_shape


def aggregate_mean_tif_to_1km_sumweight(
    grid_m: gpd.GeoDataFrame,
    tif_path: Path,
    out_col: str,
) -> pd.DataFrame:
    xmin_km, _, ymax_km, dst_transform, dst_shape = build_1km_template_from_grid_m(grid_m)
    dst_crs = grid_m.crs

    with rasterio.open(tif_path) as src:
        src_arr = src.read(1).astype("float32", copy=False)

        if src.nodata is None or (isinstance(src.nodata, float) and np.isnan(src.nodata)):
            valid = np.isfinite(src_arr)
        else:
            valid = np.isfinite(src_arr) & (src_arr != float(src.nodata))

        val = np.where(valid, src_arr, 0.0).astype("float32", copy=False)
        w = valid.astype("float32")

        val_sum = np.zeros(dst_shape, dtype="float32")
        w_sum = np.zeros(dst_shape, dtype="float32")

        reproject(
            source=val,
            destination=val_sum,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.sum,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

        reproject(
            source=w,
            destination=w_sum,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.sum,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

    mean = np.where(w_sum > 0, val_sum / w_sum, np.nan).astype("float32")

    ny, nx = mean.shape
    rr, cc = np.meshgrid(np.arange(ny, dtype=int), np.arange(nx, dtype=int), indexing="ij")
    x_km = (xmin_km + cc).astype(int)
    y_km = (ymax_km - rr).astype(int)

    return pd.DataFrame(
        {
            "idm": (x_km.astype(str) + "_" + y_km.astype(str)).ravel(),
            out_col: mean.ravel().astype(float),
        }
    )


def aggregate_eligibility_tif_to_1km(grid_m: gpd.GeoDataFrame, tif_path: Path) -> pd.DataFrame:
    xmin_km, _, ymax_km, dst_transform, dst_shape = build_1km_template_from_grid_m(grid_m)
    dst_crs = grid_m.crs

    with rasterio.open(tif_path) as src:
        src_arr = src.read(1).astype("float32", copy=False)
        nodata = src.nodata
        valid = (src_arr != nodata) if nodata is not None else np.isfinite(src_arr)

        elig = np.where(valid, src_arr, 0.0)
        elig = np.clip(elig, 0.0, 100.0) / 100.0
        weight = valid.astype("float32")

        elig_sum = np.zeros(dst_shape, dtype="float32")
        w_sum = np.zeros(dst_shape, dtype="float32")

        reproject(
            source=elig,
            destination=elig_sum,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.sum,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

        reproject(
            source=weight,
            destination=w_sum,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.sum,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

    pot_share = np.where(w_sum > 0, elig_sum / w_sum, 0.0).astype("float32")
    pot_share = np.clip(pot_share, 0.0, 1.0)

    ny, nx = pot_share.shape
    rr, cc = np.meshgrid(np.arange(ny, dtype=int), np.arange(nx, dtype=int), indexing="ij")
    x_km = (xmin_km + cc).astype(int)
    y_km = (ymax_km - rr).astype(int)

    out = pd.DataFrame({
        "idm": (x_km.astype(str) + "_" + y_km.astype(str)).ravel(),
        "pot_share": pot_share.ravel().astype(float),
    })
    out["pot_area_km2"] = out["pot_share"] * 1.0
    out["pot_area_m2"] = out["pot_share"] * 1_000_000.0
    return out


# ============================================================
# MaStR
# ============================================================

def attach_mastr_existing_turbines(
    grid_m: gpd.GeoDataFrame,
    xlsx_path: Path,
    sheet_name: str,
    netpower_unit: str = "kW",
) -> gpd.GeoDataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    lat = df["lat"].astype(str).str.replace(",", ".", regex=False).str.strip()
    lon = df["lon"].astype(str).str.replace(",", ".", regex=False).str.strip()
    df["lat"] = pd.to_numeric(lat, errors="coerce")
    df["lon"] = pd.to_numeric(lon, errors="coerce")

    p = df["netPower"].astype(str).str.replace(",", ".", regex=False).str.strip()
    df["netPower"] = pd.to_numeric(p, errors="coerce")

    df = df.dropna(subset=["lat", "lon", "netPower"]).copy()

    if netpower_unit == "kW":
        df["netPower_GW"] = df["netPower"] / 1_000_000.0
    elif netpower_unit == "MW":
        df["netPower_GW"] = df["netPower"] / 1_000.0
    else:
        df["netPower_GW"] = df["netPower"]

    pts = gpd.GeoDataFrame(
        df[["netPower_GW"]].copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(grid_m.crs)

    joined = gpd.sjoin(
        pts,
        grid_m[["idm", "geometry"]],
        how="inner",
        predicate="within",
    )

    agg = (
        joined.groupby("idm", as_index=False)
        .agg(
            n_existing=("netPower_GW", "size"),
            existing_gw=("netPower_GW", "sum"),
        )
    )

    grid_m = grid_m.merge(agg, on="idm", how="left")
    grid_m["n_existing"] = grid_m["n_existing"].fillna(0).astype(int)
    grid_m["existing_gw"] = grid_m["existing_gw"].fillna(0.0).astype(float)

    return grid_m


# ============================================================
# ERA5 extension main
# ============================================================

def extend_grid_de_with_era5(grid: gpd.GeoDataFrame, cfg: GridBuilderConfig) -> gpd.GeoDataFrame:
    if cfg.use_only_potential_cells_for_era5 and "pot_share" in grid.columns:
        before = len(grid)
        grid_work = grid.loc[grid["pot_share"].fillna(0) > 0].copy()
        print(f"[FILTER] pot_share > 0: {len(grid_work):,} / {before:,} cells")
    else:
        grid_work = grid.copy()

    print("[GRID] prepare centroids")
    grid_pts = prepare_grid_points(grid_work, cfg)

    print("[TURBINES] load metadata")
    turbine_meta = load_turbine_metadata(cfg.root / cfg.turbine_data_csv)

    print("[POWER CURVES] load power curves")
    power_curves = load_power_curves(cfg.root / cfg.power_curves_csv)

    unique_pts = (
        grid_pts[["era5_lat", "era5_lon"]]
        .drop_duplicates()
        .sort_values(["era5_lat", "era5_lon"])
        .reset_index(drop=True)
    )
    print(f"[ERA5] unique weather points: {len(unique_pts):,}")

    print("[ERA5] compute point-level results (fast mode)")
    point_results = compute_results_for_unique_era5_points_fast(
        unique_pts=unique_pts,
        cfg=cfg,
        turbine_meta=turbine_meta,
        power_curves=power_curves,
    )

    print("[MERGE] join weather-point results to cells")
    grid_pts = grid_pts.merge(point_results, on=["era5_lat", "era5_lon"], how="left")

    print("[SELECT] compute turbine shares")
    shares = compute_turbine_shares(grid_pts["era5_ws100_mean_corr"], cfg)
    grid_pts = pd.concat([grid_pts, shares], axis=1)

    grid_pts["turbine_type_era5"] = np.where(
        grid_pts["share_type7_era5"] >= 1.0,
        cfg.high_wind_turbine,
        np.where(
            grid_pts["share_type6_era5"] >= 1.0,
            cfg.low_wind_turbine,
            "mix_type6_type7",
        )
    )

    print("[FINALIZE] build mixed outputs")
    grid_pts = select_final_outputs(grid_pts, cfg)

    keep_cols = [
        "idm",
        "era5_lat",
        "era5_lon",
        "dist_sea_km_proxy",
        "era5_ws10_mean_corr",
        "era5_ws100_mean",
        "era5_ws100_mean_corr",
        "share_type6_era5",
        "share_type7_era5",
        "turbine_type_era5",
        "hubheight_era5",
        "diameter_era5",
        "rated_power_kw_era5",
        "rated_power_mw_era5",
        "costs_era5",
        "mean_ws_hub_era5",
        "gross_mwh_era5",
        "gross_flh_era5",
        "cf_era5",
        "yield_correction_factor_era5",
        "net_mwh_era5",
        "net_flh_era5",
        "existing_fleet_flh_scale",
        "net_flh_existing_era5",
        "cf_existing_era5",
    ]
    out = grid.merge(grid_pts[keep_cols], on="idm", how="left")

    if cfg.print_diagnostics:
        print_diagnostic_summary(out)

    return out


# ============================================================
# ERA5 grid prep
# ============================================================

def prepare_grid_points(grid: gpd.GeoDataFrame, cfg: GridBuilderConfig) -> gpd.GeoDataFrame:
    pts = grid[["idm", "geometry"]].copy()
    pts["geometry"] = pts.geometry.centroid

    pts_ll = pts.to_crs("EPSG:4326").copy()
    pts_ll["lon"] = pts_ll.geometry.x
    pts_ll["lat"] = pts_ll.geometry.y

    pts_ll["era5_lat"] = np.round(pts_ll["lat"] / 0.25) * 0.25
    pts_ll["era5_lon"] = np.round(pts_ll["lon"] / 0.25) * 0.25

    pts_ll["dist_sea_km_proxy"] = np.maximum(
        0.0,
        (cfg.north_coast_lat_deg - pts_ll["lat"]) * 111.32
    )

    return pts_ll[["idm", "geometry", "lat", "lon", "era5_lat", "era5_lon", "dist_sea_km_proxy"]].copy()


# ============================================================
# CSV loading
# ============================================================

def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=",")
    return df


def load_turbine_metadata(path: Path) -> pd.DataFrame:
    df = read_csv_auto_sep(path)

    required = ["turbineType", "hubHeight", "diameter", "power", "costs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in turbine data: {missing}. Available: {list(df.columns)}")

    cols = ["turbineType", "hubHeight", "diameter", "power", "costs"]
    if "variant" in df.columns:
        cols.append("variant")

    out = df[cols].copy()
    out["turbineType"] = out["turbineType"].astype(str).str.strip()
    out["hubHeight"] = pd.to_numeric(out["hubHeight"], errors="coerce")
    out["diameter"] = pd.to_numeric(out["diameter"], errors="coerce")
    out["power"] = pd.to_numeric(out["power"], errors="coerce")
    out["costs"] = pd.to_numeric(out["costs"], errors="coerce")

    out = out.dropna(subset=["turbineType", "hubHeight", "diameter", "power"]).drop_duplicates("turbineType")
    return out


def load_power_curves(path: Path) -> pd.DataFrame:
    df = read_csv_auto_sep(path)

    required = ["turbineType", "windspeed", "production"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in power curves: {missing}. Available: {list(df.columns)}")

    out = df[["turbineType", "windspeed", "production"]].copy()
    out["turbineType"] = out["turbineType"].astype(str).str.strip()
    out["windspeed"] = pd.to_numeric(out["windspeed"], errors="coerce")
    out["production"] = pd.to_numeric(out["production"], errors="coerce")

    out = out.dropna(subset=["turbineType", "windspeed", "production"])
    return out


# ============================================================
# ERA5 dataset utilities
# ============================================================

def open_era5_dataset(path: Path) -> xr.Dataset:
    last_err = None
    for engine in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not open ERA5 file: {path}\n"
        f"Last error: {last_err}\n"
        f"Install netcdf4 or h5netcdf."
    )


def get_coord_name(ds: xr.Dataset, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in ds.coords:
            return c
    for c in candidates:
        if c in ds.dims:
            return c
    return None


def nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return np.abs(values[:, None] - targets[None, :]).argmin(axis=0)


def load_era5_arrays(path: Path) -> dict:
    ds = open_era5_dataset(path)

    lat_name = get_coord_name(ds, ["latitude", "lat"])
    lon_name = get_coord_name(ds, ["longitude", "lon"])

    if lat_name is None or lon_name is None:
        ds.close()
        raise ValueError(f"Could not identify lat/lon in {path}. Available: {list(ds.coords)}")

    required_vars = ["u100", "v100", "u10", "v10"]
    missing_vars = [v for v in required_vars if v not in ds.data_vars]
    if missing_vars:
        ds.close()
        raise ValueError(f"Missing ERA5 variables in {path}: {missing_vars}. Available: {list(ds.data_vars)}")

    u100 = ds["u100"].values
    v100 = ds["v100"].values
    u10 = ds["u10"].values
    v10 = ds["v10"].values
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    dims = ds["u100"].dims

    ds.close()

    return {
        "u100": u100,
        "v100": v100,
        "u10": u10,
        "v10": v10,
        "lat_vals": lat_vals,
        "lon_vals": lon_vals,
        "dims": dims,
    }


def extract_ws_series_for_many_points(
    u100: np.ndarray,
    v100: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    dims: tuple,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dims = tuple(dims)

    time_candidates = {"time", "valid_time"}
    lat_candidates = {"latitude", "lat"}
    lon_candidates = {"longitude", "lon"}

    time_pos = next((i for i, d in enumerate(dims) if d in time_candidates), None)
    lat_pos = next((i for i, d in enumerate(dims) if d in lat_candidates), None)
    lon_pos = next((i for i, d in enumerate(dims) if d in lon_candidates), None)

    if time_pos is None or lat_pos is None or lon_pos is None:
        raise ValueError(f"Could not identify time/lat/lon among dimensions: {dims}")

    if len(dims) != 3:
        raise ValueError(f"Expected 3D ERA5 variables, received: dims={dims}")

    n_points = len(lat_idx)

    def _extract(arr: np.ndarray) -> np.ndarray:
        arr_std = np.moveaxis(arr, [time_pos, lat_pos, lon_pos], [0, 1, 2])
        out = arr_std[:, lat_idx, lon_idx]
        if out.ndim != 2 or out.shape[1] != n_points:
            raise ValueError(f"Unexpected extraction shape: {out.shape}, dims={dims}")
        return out

    u100_pts = _extract(u100)
    v100_pts = _extract(v100)
    u10_pts = _extract(u10)
    v10_pts = _extract(v10)

    ws100_all = np.sqrt(u100_pts ** 2 + v100_pts ** 2).astype(float)
    ws10_all = np.sqrt(u10_pts ** 2 + v10_pts ** 2).astype(float)

    return ws100_all, ws10_all


# ============================================================
# Fast ERA5 point-level modeling
# ============================================================

def compute_results_for_unique_era5_points_fast(
    unique_pts: pd.DataFrame,
    cfg: GridBuilderConfig,
    turbine_meta: pd.DataFrame,
    power_curves: pd.DataFrame,
) -> pd.DataFrame:
    turbine_low = get_turbine_row(turbine_meta, cfg.low_wind_turbine)
    turbine_high = get_turbine_row(turbine_meta, cfg.high_wind_turbine)

    per_file_ws100 = []
    per_file_ws10 = []

    unique_pts = unique_pts.copy()

    for file_idx, era5_rel_path in enumerate(cfg.era5_files, start=1):
        era5_path = cfg.root / era5_rel_path
        print(f"[ERA5] load file {file_idx}/{len(cfg.era5_files)}: {era5_path.name}")

        era = load_era5_arrays(era5_path)

        lat_idx = nearest_indices(era["lat_vals"], unique_pts["era5_lat"].to_numpy(dtype=float))
        lon_idx = nearest_indices(era["lon_vals"], unique_pts["era5_lon"].to_numpy(dtype=float))

        ws100_all, ws10_all = extract_ws_series_for_many_points(
            u100=era["u100"],
            v100=era["v100"],
            u10=era["u10"],
            v10=era["v10"],
            dims=era["dims"],
            lat_idx=lat_idx,
            lon_idx=lon_idx,
        )

        if len(cfg.era5_files) > 1:
            ws100_all = np.sort(ws100_all, axis=0)[::-1, :]
            ws10_all = np.sort(ws10_all, axis=0)[::-1, :]

        per_file_ws100.append(ws100_all)
        per_file_ws10.append(ws10_all)

    ws100_mean_all = np.mean(np.stack(per_file_ws100, axis=0), axis=0)
    ws10_mean_all = np.mean(np.stack(per_file_ws10, axis=0), axis=0)

    ws100_corr_all = ws100_mean_all * cfg.era5_wind_speed_factor
    ws10_corr_all = ws10_mean_all * cfg.era5_wind_speed_factor

    mean_ws100 = np.nanmean(ws100_mean_all, axis=0)
    mean_ws100_corr = np.nanmean(ws100_corr_all, axis=0)
    mean_ws10_corr = np.nanmean(ws10_corr_all, axis=0)

    rows = []
    n_points = unique_pts.shape[0]

    for i in range(n_points):
        ws100_corr = ws100_corr_all[:, i]
        ws10_corr = ws10_corr_all[:, i]

        low_res = calc_turbine_production_for_point(
            ws10=ws10_corr,
            ws100=ws100_corr,
            turbine_row=turbine_low,
            power_curves=power_curves,
            tech_avail=cfg.technical_availability,
            capfactor=cfg.capacity_factor_scale,
        )

        high_res = calc_turbine_production_for_point(
            ws10=ws10_corr,
            ws100=ws100_corr,
            turbine_row=turbine_high,
            power_curves=power_curves,
            tech_avail=cfg.technical_availability,
            capfactor=cfg.capacity_factor_scale,
        )

        rows.append({
            "era5_lat": float(unique_pts.iloc[i]["era5_lat"]),
            "era5_lon": float(unique_pts.iloc[i]["era5_lon"]),
            "era5_ws10_mean_corr": float(mean_ws10_corr[i]),
            "era5_ws100_mean": float(mean_ws100[i]),
            "era5_ws100_mean_corr": float(mean_ws100_corr[i]),

            "hubheight_low": low_res["hubHeight"],
            "diameter_low": low_res["diameter"],
            "rated_power_kw_low": low_res["power_kw"],
            "rated_power_mw_low": low_res["power_mw"],
            "costs_low": low_res["costs"],
            "mean_ws_hub_low": low_res["mean_ws_hub"],
            "gross_mwh_low": low_res["gross_mwh"],
            "gross_flh_low": low_res["gross_flh"],
            "cf_low": low_res["cf"],

            "hubheight_high": high_res["hubHeight"],
            "diameter_high": high_res["diameter"],
            "rated_power_kw_high": high_res["power_kw"],
            "rated_power_mw_high": high_res["power_mw"],
            "costs_high": high_res["costs"],
            "mean_ws_hub_high": high_res["mean_ws_hub"],
            "gross_mwh_high": high_res["gross_mwh"],
            "gross_flh_high": high_res["gross_flh"],
            "cf_high": high_res["cf"],
        })

        if (i + 1) % 100 == 0 or (i + 1) == n_points:
            print(f"[ERA5] modeled {i+1:,}/{n_points:,} points")

    return pd.DataFrame(rows)


# ============================================================
# Turbine production
# ============================================================

def get_turbine_row(turbine_meta: pd.DataFrame, turbine_type: str) -> pd.Series:
    sub = turbine_meta.loc[turbine_meta["turbineType"] == turbine_type]
    if sub.empty:
        raise ValueError(f"{turbine_type} not found in windOnshoreTurbineData.csv.")
    return sub.iloc[0].copy()


def calc_turbine_production_for_point(
    ws10: np.ndarray,
    ws100: np.ndarray,
    turbine_row: pd.Series,
    power_curves: pd.DataFrame,
    tech_avail: float,
    capfactor: float,
) -> dict:
    turbine_type = str(turbine_row["turbineType"])
    hub_height = float(turbine_row["hubHeight"])
    diameter = float(turbine_row["diameter"])
    power_kw = float(turbine_row["power"])
    power_mw = power_kw / 1000.0
    costs = float(turbine_row["costs"]) if "costs" in turbine_row.index and pd.notna(turbine_row["costs"]) else np.nan

    eps = 1e-6
    ws10_safe = np.maximum(ws10, eps)
    ws100_safe = np.maximum(ws100, eps)

    alpha = np.log(ws100_safe / ws10_safe) / np.log(100.0 / 10.0)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)

    ws_hub = ws10_safe * ((hub_height / 10.0) ** alpha)

    xsample = np.array([0, 3, 4.5, 7, 11, 15, 25, 35], dtype=float)
    values = np.array([0, 0, 0.035, 0.035, 0.025, 0.015, 0, 0], dtype=float)
    scale_factor = np.interp(ws_hub, xsample, values, left=values[0], right=values[-1])

    ws_hub_scaled = (1.0 - scale_factor) * ws_hub

    pc = power_curves.loc[power_curves["turbineType"] == turbine_type].copy()
    if pc.empty:
        raise ValueError(f"No power curve found for {turbine_type}.")

    pc = pc.sort_values("windspeed")
    x = pc["windspeed"].to_numpy(dtype=float)
    y = pc["production"].to_numpy(dtype=float)

    y_max = np.nanmax(y)
    if not np.isfinite(y_max) or y_max <= 0:
        raise ValueError(f"Invalid power curve for {turbine_type}")

    power_out_kw = np.interp(ws_hub_scaled, x, y, left=0.0, right=0.0)
    power_out_kw = power_out_kw / y_max * power_kw * tech_avail * capfactor

    gross_mwh = float(np.nansum(power_out_kw) / 1000.0)
    gross_flh = gross_mwh / power_mw if power_mw > 0 else np.nan
    cf = gross_flh / len(power_out_kw) if len(power_out_kw) > 0 else np.nan

    return {
        "hubHeight": hub_height,
        "diameter": diameter,
        "power_kw": power_kw,
        "power_mw": power_mw,
        "costs": costs,
        "mean_ws_hub": float(np.nanmean(ws_hub)),
        "gross_mwh": gross_mwh,
        "gross_flh": gross_flh,
        "cf": cf,
    }


# ============================================================
# Mixing logic
# ============================================================

def compute_turbine_shares(ws_mean_corr: pd.Series, cfg: GridBuilderConfig) -> pd.DataFrame:
    ws = pd.to_numeric(ws_mean_corr, errors="coerce").astype(float)

    lo = float(cfg.mix_ws_low)
    hi = float(cfg.mix_ws_high)
    width = hi - lo

    if width <= 0:
        raise ValueError("mix_ws_high must be greater than mix_ws_low.")

    share_high = (ws - lo) / width
    share_high = np.clip(share_high, 0.0, 1.0)
    share_low = 1.0 - share_high

    return pd.DataFrame(
        {
            "share_type6_era5": share_low,
            "share_type7_era5": share_high,
        },
        index=ws_mean_corr.index,
    )


def select_final_outputs(grid_pts: pd.DataFrame, cfg: GridBuilderConfig) -> gpd.GeoDataFrame:
    out = grid_pts.copy()

    share_low = out["share_type6_era5"].astype(float)
    share_high = out["share_type7_era5"].astype(float)

    out["hubheight_era5"] = share_low * out["hubheight_low"] + share_high * out["hubheight_high"]
    out["diameter_era5"] = share_low * out["diameter_low"] + share_high * out["diameter_high"]
    out["rated_power_kw_era5"] = share_low * out["rated_power_kw_low"] + share_high * out["rated_power_kw_high"]
    out["rated_power_mw_era5"] = share_low * out["rated_power_mw_low"] + share_high * out["rated_power_mw_high"]
    out["costs_era5"] = share_low * out["costs_low"] + share_high * out["costs_high"]
    out["mean_ws_hub_era5"] = share_low * out["mean_ws_hub_low"] + share_high * out["mean_ws_hub_high"]

    out["gross_mwh_era5"] = share_low * out["gross_mwh_low"] + share_high * out["gross_mwh_high"]
    out["gross_flh_era5"] = share_low * out["gross_flh_low"] + share_high * out["gross_flh_high"]
    out["cf_era5"] = share_low * out["cf_low"] + share_high * out["cf_high"]

    out["yield_correction_factor_era5"] = compute_yield_correction_factor(
        dist_sea_km=out["dist_sea_km_proxy"].astype(float).to_numpy(),
        hub_height=out["hubheight_era5"].astype(float).to_numpy(),
        diameter=out["diameter_era5"].astype(float).to_numpy(),
        capacity_kw=out["rated_power_kw_era5"].astype(float).to_numpy(),
        flh_actual=out["gross_flh_era5"].astype(float).to_numpy(),
    )

    out["net_mwh_era5"] = out["gross_mwh_era5"] * out["yield_correction_factor_era5"]
    out["net_flh_era5"] = out["gross_flh_era5"] * out["yield_correction_factor_era5"]
    
    # Correct capacity factor consistently to net FLH
    out["cf_era5"] = out["net_flh_era5"] / 8760.0
    
    # Existing-fleet yield proxy:
    # This is used only for existing generation offsets, not for future new-build ranking.
    out["existing_fleet_flh_scale"] = float(cfg.existing_fleet_flh_scale)
    out["net_flh_existing_era5"] = out["net_flh_era5"] * float(cfg.existing_fleet_flh_scale)
    out["cf_existing_era5"] = out["net_flh_existing_era5"] / 8760.0

    return out


# ============================================================
# Yield correction
# ============================================================

def compute_yield_correction_factor(
    dist_sea_km: np.ndarray,
    hub_height: np.ndarray,
    diameter: np.ndarray,
    capacity_kw: np.ndarray,
    flh_actual: np.ndarray,
) -> np.ndarray:
    flh_deviation = (
        844.67225
        - 1.5175 * dist_sea_km
        - 1.87725 * hub_height
        - 2.2045 * diameter
        + 0.09875 * capacity_kw
    )

    flh_corrected = flh_actual - flh_deviation

    with np.errstate(divide="ignore", invalid="ignore"):
        correction_factor = flh_corrected / flh_actual

    correction_factor = np.where(np.isfinite(correction_factor), correction_factor, np.nan)
    correction_factor = np.clip(correction_factor, 0.0, 2.0)
    return correction_factor


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostic_summary(df: pd.DataFrame) -> None:
    print("[CHECK] summary")

    for c in [
        "share_type6_era5",
        "share_type7_era5",
        "era5_ws100_mean_corr",
        "gross_flh_era5",
        "net_flh_era5",
        "gross_mwh_era5",
        "net_mwh_era5",
        "cf_era5",
        "costs_era5",
    ]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            print(
                f"  {c}: "
                f"n={s.notna().sum():,}, "
                f"mean={s.mean():.3f}, "
                f"p50={s.median():.3f}, "
                f"min={s.min():.3f}, "
                f"max={s.max():.3f}"
            )

    if "turbine_type_era5" in df.columns:
        print("[CHECK] dominant type counts")
        print(df["turbine_type_era5"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    cfg = GridBuilderConfig()
    grid_m, grid_de, grid_de_era5 = run_grid_builder(cfg)