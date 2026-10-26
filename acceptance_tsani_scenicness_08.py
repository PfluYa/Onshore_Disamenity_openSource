# -*- coding: utf-8 -*-
"""Scenic-beauty exclusion scenario for German onshore wind.

Excludes candidate turbine sites whose mapped scenic-beauty score is at
least ``scenicness_threshold``. The active config defaults to 5.0.
The historical raster handling is intentionally preserved so that published
model outputs remain unchanged; see ``Input Data/README.md`` for the
known, small boundary-mask limitation.

Inputs: ``grid_de_era5.parquet`` and the BfN scenic-beauty GeoTIFF.
Outputs: placement cells, cost curves, state reports and scenario metadata.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

from acceptance_tsani_proxy_07 import (
    TsaniProxySharedConfig,
    add_remaining_potential_columns_variable_turbine,
    allocate_iterative_min_base_pv_lcoe,
    add_lcoe_metrics_pv_base_only,
    build_state_targets_b2045,
    build_state_additions_df,
    plot_map_in_de,
    plot_base_lcoe_curves,
)


@dataclass
class TsaniScenicnessConfig:
    root: Path = Path(__file__).resolve().parent

    input_grid_de_era5_parquet: Optional[Path] = None
    output_dir: Path = Path(__file__).resolve().parent / 'Output' / 'placing_tsani_scenicness'

    scenicness_tif: Path = Path(__file__).resolve().parent / 'Input Data' / 'Scenicness_Labi_Netz' / 'Scenicness_roth_export.tif'

    scenicness_threshold: float = 5.0
    exclude_if_scenicness_ge_threshold: bool = True

    drop_unused_columns_at_start: bool = True

    state_id_col: str = "state_id"
    existing_gw_col: str = "existing_gw"

    flh_col: str = "net_flh_era5"
    existing_flh_col: str = "net_flh_existing_era5"
    cf_col: str = "cf_era5"
    turbine_power_mw_col: str = "rated_power_mw_era5"
    turbine_power_kw_col: str = "rated_power_kw_era5"
    turbine_cost_col: str = "costs_era5"

    flh_fill: float = 0.0
    cf_fill: float = 0.0
    turbine_power_mw_fill: float = 0.0
    turbine_power_kw_fill: float = 0.0
    turbine_cost_fill: float = 0.0

    plot_exclusion_map: bool = True
    plot_curves: bool = True
    debug_every_steps: int = 1000

    keep_cols_initial: Tuple[str, ...] = (
        "idm",
        "x_km",
        "y_km",
        "geometry",
        "in_de",
        "gid2019",
        "nuts3_id",
        "state_id",
        "state_name",
        "pop_2022",
        "pot_share",
        "pot_area_km2",
        "pot_area_m2",
        "n_existing",
        "existing_gw",
        "share_type6_era5",
        "share_type7_era5",
        "turbine_type_era5",
        "rated_power_kw_era5",
        "rated_power_mw_era5",
        "costs_era5",
        "gross_flh_era5",
        "net_flh_era5",
        "cf_era5",
        "yield_correction_factor_era5",
        "net_flh_existing_era5",
        "cf_existing_era5",
        "existing_fleet_flh_scale",
    )


def build_output_suffix(
    *,
    use_state_targets: bool,
    build_limit_mode: str,
    scenicness_threshold: float,
) -> str:
    regional = f"with_states_{build_limit_mode}" if use_state_targets else f"national_{build_limit_mode}"
    thr_txt = str(scenicness_threshold).replace(".", "p")
    return f"tsani_scenicness_ge{thr_txt}_{regional}"


def build_1km_template_from_grid(gdf: gpd.GeoDataFrame):
    xmin_km = int(np.nanmin(gdf["x_km"].to_numpy()))
    xmax_km = int(np.nanmax(gdf["x_km"].to_numpy()))
    ymin_km = int(np.nanmin(gdf["y_km"].to_numpy()))
    ymax_km = int(np.nanmax(gdf["y_km"].to_numpy()))

    width = xmax_km - xmin_km + 1
    height = ymax_km - ymin_km + 1

    left = xmin_km * 1000.0
    top = (ymax_km + 1) * 1000.0

    dst_transform = from_origin(left, top, 1000.0, 1000.0)
    dst_shape = (height, width)

    return xmin_km, ymax_km, dst_transform, dst_shape


def aggregate_scenicness_to_1km(
    gdf: gpd.GeoDataFrame,
    tif_path: Path,
    out_col: str = "scenicness_mean",
) -> pd.DataFrame:
    xmin_km, ymax_km, dst_transform, dst_shape = build_1km_template_from_grid(gdf)

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32", copy=False)

        if src.nodata is None:
            valid = np.isfinite(arr)
        else:
            valid = np.isfinite(arr) & (arr != float(src.nodata))

        val = np.where(valid, arr, 0.0).astype("float32")
        weight = valid.astype("float32")

        val_sum = np.zeros(dst_shape, dtype="float32")
        w_sum = np.zeros(dst_shape, dtype="float32")

        reproject(
            source=val,
            destination=val_sum,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=gdf.crs,
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
            dst_crs=gdf.crs,
            resampling=Resampling.sum,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

    scenic = np.where(w_sum > 0, val_sum / w_sum, np.nan).astype("float32")

    ny, nx = scenic.shape
    rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    x_km = (xmin_km + cc).astype(int)
    y_km = (ymax_km - rr).astype(int)

    return pd.DataFrame({
        "idm": (x_km.astype(str) + "_" + y_km.astype(str)).ravel(),
        out_col: scenic.ravel().astype(float),
    })


def validate_config(cfg: TsaniScenicnessConfig, shared: TsaniProxySharedConfig) -> None:
    if cfg.input_grid_de_era5_parquet is None:
        raise ValueError("cfg.input_grid_de_era5_parquet must be set.")

    if not cfg.scenicness_tif.exists():
        raise FileNotFoundError(f"Scenicness TIF not found: {cfg.scenicness_tif}")

    if shared.build_limit_mode not in {"total_gw", "exhaustion"}:
        raise ValueError("shared.build_limit_mode must be 'total_gw' or 'exhaustion'")

    if shared.build_limit_mode == "total_gw" and shared.target_total_gw is None:
        raise ValueError("shared.target_total_gw must be set when build_limit_mode == 'total_gw'")

    if cfg.scenicness_threshold < 0:
        raise ValueError("scenicness_threshold must be >= 0")


def run_tsani_scenicness(
    cfg: TsaniScenicnessConfig,
    shared: TsaniProxySharedConfig,
) -> Dict[str, Any]:
    validate_config(cfg, shared)

    print("[RUN] Tsani-proxy scenicness exclusion model")
    print(f"[MODEL] exclude scenicness >= {cfg.scenicness_threshold:.2f}")
    print(f"[LOAD] {cfg.input_grid_de_era5_parquet}")

    gdf = gpd.read_parquet(cfg.input_grid_de_era5_parquet)
    print(f"[INFO] rows: {len(gdf):,}")
    print(f"[CRS] {gdf.crs}")

    if cfg.drop_unused_columns_at_start:
        keep = [c for c in cfg.keep_cols_initial if c in gdf.columns]
        if "geometry" in gdf.columns and "geometry" not in keep:
            keep.append("geometry")
        gdf = gdf[keep].copy()
        print(f"[CLEAN] kept {len(keep)} columns")

    if gdf["idm"].duplicated().any():
        raise ValueError(f"Duplicate idm in input grid: {int(gdf['idm'].duplicated().sum()):,}")

    print(f"[SCENICNESS] aggregate TIF -> 1km: {cfg.scenicness_tif}")
    scenic_df = aggregate_scenicness_to_1km(
        gdf=gdf,
        tif_path=cfg.scenicness_tif,
        out_col="scenicness_mean",
    )

    gdf = gdf.merge(scenic_df, on="idm", how="left")

    gdf["is_excluded_by_scenicness"] = (
        pd.to_numeric(gdf["scenicness_mean"], errors="coerce")
        .fillna(-np.inf)
        >= float(cfg.scenicness_threshold)
    ).astype(int)

    if not cfg.exclude_if_scenicness_ge_threshold:
        gdf["is_excluded_by_scenicness"] = 0

    print(f"[SCENICNESS] coverage: {gdf['scenicness_mean'].notna().mean():.3f}")
    print(f"[SCENICNESS] mean:     {gdf['scenicness_mean'].mean():.3f}")
    print(f"[SCENICNESS] excluded cells share: {gdf['is_excluded_by_scenicness'].mean():.3f}")

    gdf["flh_net"] = (
        pd.to_numeric(gdf[cfg.flh_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.flh_fill)
        .astype(float)
    )
    
    if cfg.existing_flh_col in gdf.columns:
        gdf["flh_existing"] = (
            pd.to_numeric(gdf[cfg.existing_flh_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(cfg.flh_fill)
            .astype(float)
        )
        print(f"[EXISTING FLH] using calibrated existing FLH column: {cfg.existing_flh_col}")
    else:
        gdf["flh_existing"] = gdf["flh_net"].astype(float)
        print(f"[EXISTING FLH] column {cfg.existing_flh_col!r} not found -> using flh_net")    
    
    gdf["cf"] = (
        pd.to_numeric(gdf[cfg.cf_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.cf_fill)
        .astype(float)
    )
    gdf["turbine_power_mw"] = (
        pd.to_numeric(gdf[cfg.turbine_power_mw_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.turbine_power_mw_fill)
        .astype(float)
    )
    gdf["turbine_power_kw"] = (
        pd.to_numeric(gdf[cfg.turbine_power_kw_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.turbine_power_kw_fill)
        .astype(float)
    )
    gdf["turbine_cost_input"] = (
        pd.to_numeric(gdf[cfg.turbine_cost_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.turbine_cost_fill)
        .astype(float)
    )

    if "n_new_max" not in gdf.columns:
        gdf = add_remaining_potential_columns_variable_turbine(
            gdf=gdf,
            pot_mw_per_km2=shared.pot_mw_per_km2,
            turbine_power_mw_col="turbine_power_mw",
            existing_gw_col=cfg.existing_gw_col,
            existing_proxy_mw_per_turbine=shared.existing_proxy_mw_per_turbine,
        )

    if cfg.exclude_if_scenicness_ge_threshold:
        n_before = int(gdf["n_new_max"].sum())
        gdf.loc[gdf["is_excluded_by_scenicness"] == 1, "n_new_max"] = 0
        n_after = int(gdf["n_new_max"].sum())
        print(f"[SCENICNESS] excluded potential turbines: {n_before - n_after:,}")

    gdf["existing_gw_cell"] = (
        pd.to_numeric(gdf[cfg.existing_gw_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    gdf["new_build_gw_max_cell"] = (
        pd.to_numeric(gdf["n_new_max"], errors="coerce").fillna(0.0).astype(float)
        * pd.to_numeric(gdf["turbine_power_mw"], errors="coerce").fillna(0.0).astype(float)
        / 1000.0
    )

    gdf["existing_twh_per_year_cell"] = gdf["existing_gw_cell"] * 1000.0 * gdf["flh_existing"] / 1e6
    gdf["new_build_twh_per_year_max_cell"] = gdf["new_build_gw_max_cell"] * 1000.0 * gdf["flh_net"] / 1e6

    existing_total_gw = float(gdf["existing_gw_cell"].sum())
    existing_total_twh_per_year = float(gdf["existing_twh_per_year_cell"].sum())

    print("\n[THEORETICAL TECHNICAL POTENTIAL]")
    print(f"Existing capacity:               {existing_total_gw:.2f} GW")
    print(f"Additional new-build potential:  {gdf['new_build_gw_max_cell'].sum():.2f} GW")
    print(f"Total technical potential:       {existing_total_gw + gdf['new_build_gw_max_cell'].sum():.2f} GW")
    print(f"Existing generation:             {existing_total_twh_per_year:.2f} TWh/year")
    print(f"Additional new-build generation: {gdf['new_build_twh_per_year_max_cell'].sum():.2f} TWh/year")

    targets = build_state_targets_b2045()
    targets_df = build_state_additions_df(gdf, targets, cfg.state_id_col, cfg.existing_gw_col)

    if shared.build_limit_mode == "total_gw":
        additional_gw_needed = max(float(shared.target_total_gw) - existing_total_gw, 0.0)
    else:
        additional_gw_needed = None

    placement, curve_raw, state_report, meta = allocate_iterative_min_base_pv_lcoe(
        gdf=gdf,
        targets_df=targets_df,
        state_id_col=cfg.state_id_col,
        n_new_max_col="n_new_max",
        flh_col="flh_net",
        cf_col="cf",
        turbine_power_mw_col="turbine_power_mw",
        turbine_power_kw_col="turbine_power_kw",
        turbine_cost_col="turbine_cost_input",
        use_state_targets=shared.use_state_targets,
        state_upper_slack=shared.state_upper_slack,
        build_limit_mode=shared.build_limit_mode,
        target_new_build_gw=additional_gw_needed,
        eps_rel=shared.eps_rel,
        in_de_col="in_de" if "in_de" in gdf.columns else None,
        debug_every_steps=cfg.debug_every_steps,
        lifetime_yrs=shared.lifetime_yrs,
        wacc=shared.wacc,
        fixed_om_share_of_capex_per_year=shared.fixed_om_share_of_capex_per_year,
    )

    gdf = gdf.merge(placement, on="idm", how="left")
    gdf["n_new"] = gdf["n_new"].fillna(0).astype(int)
    gdf["added_gw"] = gdf["added_gw"].fillna(0.0)
    gdf["added_cost_eur"] = gdf["added_cost_eur"].fillna(0.0)

    if cfg.plot_exclusion_map and "in_de" in gdf.columns:
        plot_map_in_de(
            gdf,
            value_col="is_excluded_by_scenicness",
            mask_col="in_de",
            title=f"Tsani scenicness excluded cells (scenicness >= {cfg.scenicness_threshold:.1f})",
            cmap="viridis",
        )

    curve = add_lcoe_metrics_pv_base_only(curve_raw)
    phase_switch_gw_total = np.nan
    phase_switch_twh_per_year_total = np.nan

    if meta.get("phase_switch_gw") is not None and np.isfinite(meta.get("phase_switch_gw", np.nan)):
        phase_switch_gw_total = existing_total_gw + float(meta["phase_switch_gw"])

    if meta.get("phase_switch_twh_per_year") is not None and np.isfinite(meta.get("phase_switch_twh_per_year", np.nan)):
        phase_switch_twh_per_year_total = existing_total_twh_per_year + float(meta["phase_switch_twh_per_year"])    

    if not curve.empty:
        curve["cum_gw_total_system"] = existing_total_gw + curve["cum_gw"]
        curve["cum_twh_per_year_total_system"] = existing_total_twh_per_year + curve["cum_twh_per_year"]
        curve["existing_total_gw"] = existing_total_gw
        curve["existing_total_twh_per_year"] = existing_total_twh_per_year

        curve["lcoe_total_pv_step_eur_per_mwh"] = curve["lcoe_base_pv_step_eur_per_mwh"]
        curve["lcoe_total_pv_cum_eur_per_mwh"] = curve["lcoe_base_pv_cum_eur_per_mwh"]

    if cfg.plot_curves and not curve.empty:
        plot_base_lcoe_curves(
            curve=curve,
            phase_switch_gw=phase_switch_gw_total,
            phase_switch_twh_per_year=phase_switch_twh_per_year_total,
            x_col_gw="cum_gw_total_system",
            x_col_twh="cum_twh_per_year_total_system",
            title_prefix="Tsani scenicness proxy",
        )

    suffix = build_output_suffix(
        use_state_targets=shared.use_state_targets,
        build_limit_mode=shared.build_limit_mode,
        scenicness_threshold=cfg.scenicness_threshold,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    out_cells = cfg.output_dir / f"placement_cells_iterative_tsani_scenicness_{suffix}.parquet"
    out_curve = cfg.output_dir / f"cost_curve_iterative_tsani_scenicness_{suffix}.parquet"
    out_state = cfg.output_dir / f"state_report_iterative_tsani_scenicness_{suffix}.parquet"
    out_meta = cfg.output_dir / f"meta_tsani_scenicness_{suffix}.parquet"

    print(f"[SAVE] cells -> {out_cells}")
    gdf.to_parquet(out_cells, index=False)

    print(f"[SAVE] curve -> {out_curve}")
    curve.to_parquet(out_curve, index=False)

    print(f"[SAVE] state report -> {out_state}")
    state_report.to_parquet(out_state, index=False)

    meta_df = pd.DataFrame([meta])
    meta_df["scenicness_threshold"] = float(cfg.scenicness_threshold)
    meta_df["n_excluded_cells"] = int(gdf["is_excluded_by_scenicness"].sum())
    meta_df["excluded_cells_share"] = float(gdf["is_excluded_by_scenicness"].mean())

    print(f"[SAVE] meta -> {out_meta}")
    meta_df.to_parquet(out_meta, index=False)

    return {
        "placement_cells": gdf,
        "cost_curve": curve,
        "state_report": state_report,
        "meta": meta_df,
    }


if __name__ == "__main__":
    cfg = TsaniScenicnessConfig()
    shared = TsaniProxySharedConfig()

    if cfg.input_grid_de_era5_parquet is None:
        cfg.input_grid_de_era5_parquet = cfg.root / "Output" / "grid_de_era5.parquet"

    run_tsani_scenicness(cfg, shared)