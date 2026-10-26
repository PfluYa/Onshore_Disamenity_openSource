# -*- coding: utf-8 -*-
"""Population-buffer exclusion scenario for German onshore wind.

Exclude otherwise eligible candidate cells within ``buffer_radius_km`` of
cells whose population density exceeds ``population_density_threshold``.
For the 1-km grid, the number of inhabitants ``pop_2022`` is used as the
population-per-km² proxy. This is a spatial exclusion scenario, not a
monetary disamenity-cost model.

Outputs
-------
- placement_cells_iterative_tsani_proxy_{suffix}.parquet
- cost_curve_iterative_tsani_proxy_{suffix}.parquet
- state_report_iterative_tsani_proxy_{suffix}.parquet
- meta_tsani_proxy_{suffix}.parquet
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any
import heapq

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# ============================================================
# Shared config fallback for standalone use
# ============================================================

@dataclass
class TsaniProxySharedConfig:
    pot_mw_per_km2: float = 25.0
    existing_proxy_mw_per_turbine: float = 2.5

    use_state_targets: bool = True
    state_lower_slack: float = 0.05
    state_upper_slack: float = 0.05

    build_limit_mode: str = "total_gw"
    target_total_gw: Optional[float] = 220.0

    lifetime_yrs: int = 20
    wacc: float = 0.06
    fixed_om_share_of_capex_per_year: float = 0.025

    eps_rel: float = 1e-12


# ============================================================
# Module config only
# ============================================================

@dataclass
class TsaniProxyConfig:
    root: Path = Path(__file__).resolve().parent

    # input / output
    input_grid_de_era5_parquet: Optional[Path] = None
    output_dir: Path = Path(__file__).resolve().parent / 'Output' / 'placing_tsani_proxy'

    # optional cleanup
    drop_unused_columns_at_start: bool = True

    # ids / columns
    state_id_col: str = "state_id"
    existing_gw_col: str = "existing_gw"

    # population density proxy
    population_density_col: str = "pop_2022"
    population_density_threshold: float = 1000.0
    buffer_radius_km: float = 4.0
    use_population_buffer_exclusion: bool = True

    # ERA5-based engineering / energy
    flh_col: str = "net_flh_era5"
    existing_flh_col: str = "net_flh_existing_era5"
    gross_flh_col: str = "gross_flh_era5"
    cf_col: str = "cf_era5"
    turbine_power_mw_col: str = "rated_power_mw_era5"
    turbine_power_kw_col: str = "rated_power_kw_era5"
    turbine_cost_col: str = "costs_era5"
    turbine_type_col: str = "turbine_type_era5"

    # fills
    population_density_fill: float = 0.0
    flh_fill: float = 0.0
    cf_fill: float = 0.0
    turbine_power_mw_fill: float = 0.0
    turbine_power_kw_fill: float = 0.0
    turbine_cost_fill: float = 0.0

    # plotting
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


# ============================================================
# Output suffix helper
# ============================================================

def build_output_suffix(
    *,
    use_state_targets: bool,
    build_limit_mode: str,
    population_density_threshold: float,
    buffer_radius_km: float,
) -> str:
    regional = f"with_states_{build_limit_mode}" if use_state_targets else f"national_{build_limit_mode}"
    thr_txt = str(int(round(population_density_threshold)))
    rad_txt = str(int(round(buffer_radius_km)))
    return f"tsani_proxy_pop{thr_txt}_r{rad_txt}_{regional}"


# ============================================================
# Public runner
# ============================================================

def run_tsani_proxy(
    cfg: TsaniProxyConfig,
    shared: TsaniProxySharedConfig,
) -> Dict[str, Any]:
    validate_config(cfg, shared)

    if cfg.input_grid_de_era5_parquet is None:
        raise ValueError("cfg.input_grid_de_era5_parquet must be set.")

    print("[RUN] Tsani-proxy exclusion model")
    print(f"[MODEL] threshold = {cfg.population_density_threshold:.1f} people/km²")
    print(f"[MODEL] buffer radius = {cfg.buffer_radius_km:.1f} km")
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

    # --------------------------------------------------------
    # Base variables
    # --------------------------------------------------------
    gdf["population_density_people_per_km2"] = (
        pd.to_numeric(gdf[cfg.population_density_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.population_density_fill)
        .astype(float)
    )

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

    # --------------------------------------------------------
    # Tsani-proxy exclusion flags
    # --------------------------------------------------------
    gdf["is_sensitive_population_cell"] = (
        gdf["population_density_people_per_km2"] > float(cfg.population_density_threshold)
    ).astype(int)

    gdf = add_distance_to_sensitive_population_cells(
        gdf=gdf,
        sensitive_col="is_sensitive_population_cell",
        out_dist_col="dist_to_sensitive_population_km",
    )
    
    print("[CHECK] after sensitive-distance join")
    print(f"  rows:           {len(gdf):,}")
    print(f"  unique idm:     {gdf['idm'].nunique():,}")
    print(f"  duplicated idm: {gdf['idm'].duplicated().sum():,}")
    
    if gdf["idm"].duplicated().any():
        raise ValueError("Duplicate idm after sensitive-distance join.")
    

    if cfg.use_population_buffer_exclusion:
        gdf["is_excluded_by_population_buffer"] = (
            pd.to_numeric(gdf["dist_to_sensitive_population_km"], errors="coerce")
            .fillna(np.inf)
            <= float(cfg.buffer_radius_km)
        ).astype(int)
    else:
        gdf["is_excluded_by_population_buffer"] = 0

    print(f"[TSANI-PROXY] sensitive cells share: {gdf['is_sensitive_population_cell'].mean():.3f}")
    print(f"[TSANI-PROXY] excluded cells share:  {gdf['is_excluded_by_population_buffer'].mean():.3f}")

    if "n_new_max" not in gdf.columns:
        gdf = add_remaining_potential_columns_variable_turbine(
            gdf=gdf,
            pot_mw_per_km2=shared.pot_mw_per_km2,
            turbine_power_mw_col="turbine_power_mw",
            existing_gw_col=cfg.existing_gw_col,
            existing_proxy_mw_per_turbine=shared.existing_proxy_mw_per_turbine,
        )

    if cfg.use_population_buffer_exclusion:
        n_before = int(gdf["n_new_max"].sum())
        gdf.loc[gdf["is_excluded_by_population_buffer"] == 1, "n_new_max"] = 0
        n_after = int(gdf["n_new_max"].sum())
        print(f"[TSANI-PROXY] excluded potential turbines: {n_before - n_after:,}")

    # --------------------------------------------------------
    # Theoretical technical potential
    # --------------------------------------------------------
    gdf["existing_gw_cell"] = (
        pd.to_numeric(gdf[cfg.existing_gw_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    gdf["new_build_gw_max_cell"] = (
        pd.to_numeric(gdf["n_new_max"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        * pd.to_numeric(gdf["turbine_power_mw"], errors="coerce").fillna(0.0).astype(float)
        / 1000.0
    )

    theoretical_existing_gw = float(gdf["existing_gw_cell"].sum())
    theoretical_new_build_gw = float(gdf["new_build_gw_max_cell"].sum())
    theoretical_total_gw = theoretical_existing_gw + theoretical_new_build_gw

    gdf["existing_twh_per_year_cell"] = gdf["existing_gw_cell"] * 1000.0 * gdf["flh_existing"] / 1e6
    gdf["new_build_twh_per_year_max_cell"] = gdf["new_build_gw_max_cell"] * 1000.0 * gdf["flh_net"] / 1e6

    theoretical_existing_twh_per_year = float(gdf["existing_twh_per_year_cell"].sum())
    theoretical_new_build_twh_per_year = float(gdf["new_build_twh_per_year_max_cell"].sum())
    theoretical_total_twh_per_year = theoretical_existing_twh_per_year + theoretical_new_build_twh_per_year

    print("\n[THEORETICAL TECHNICAL POTENTIAL]")
    print(f"Existing capacity:               {theoretical_existing_gw:.2f} GW")
    print(f"Additional new-build potential:  {theoretical_new_build_gw:.2f} GW")
    print(f"Total technical potential:       {theoretical_total_gw:.2f} GW")
    print(f"Existing generation:             {theoretical_existing_twh_per_year:.2f} TWh/year")
    print(f"Additional new-build generation: {theoretical_new_build_twh_per_year:.2f} TWh/year")
    print(f"Total technical generation:      {theoretical_total_twh_per_year:.2f} TWh/year")

    existing_total_gw = float(gdf["existing_gw_cell"].sum())
    existing_total_twh_per_year = float(gdf["existing_twh_per_year_cell"].sum())

    print(f"[EXISTING] Total existing GW: {existing_total_gw:.2f}")
    print(f"[EXISTING] Total existing TWh/year: {existing_total_twh_per_year:.2f}")

    # --------------------------------------------------------
    # State targets
    # --------------------------------------------------------
    targets = build_state_targets_b2045()
    targets_df = build_state_additions_df(gdf, targets, cfg.state_id_col, cfg.existing_gw_col)
    print("[STATE] add_gw by state:")
    print(targets_df[["state_id", "target_gw", "existing_gw", "add_gw"]].to_string(index=False))
    print(f"[TARGET] Total additional GW (sum add_gw): {float(targets_df['add_gw'].sum()):.2f}")

    if shared.build_limit_mode == "total_gw":
        additional_gw_needed = max(float(shared.target_total_gw) - existing_total_gw, 0.0)
        print(f"[BUILD LIMIT] total system target = {shared.target_total_gw:.2f} GW")
        print(f"[BUILD LIMIT] additional GW to build = {additional_gw_needed:.2f}")
    else:
        additional_gw_needed = np.nan
        print("[BUILD LIMIT] build until technical exhaustion")

    # --------------------------------------------------------
    # Placement
    # --------------------------------------------------------
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
        target_new_build_gw=additional_gw_needed if shared.build_limit_mode == "total_gw" else None,
        eps_rel=shared.eps_rel,
        in_de_col="in_de" if "in_de" in gdf.columns else None,
        debug_every_steps=cfg.debug_every_steps,
        lifetime_yrs=shared.lifetime_yrs,
        wacc=shared.wacc,
        fixed_om_share_of_capex_per_year=shared.fixed_om_share_of_capex_per_year,
    )

    print("[STATE-REPORT] built vs corridor:")
    print(
        state_report[
            ["state_id", "add_gw", "target_gw_phaseA", "max_gw", "built_gw", "gap_to_target_gw_phaseA"]
        ].to_string(index=False)
    )

    gdf = gdf.merge(placement, on="idm", how="left")
    gdf["n_new"] = gdf["n_new"].fillna(0).astype(int)
    gdf["added_gw"] = gdf["added_gw"].fillna(0.0)
    gdf["added_cost_eur"] = gdf["added_cost_eur"].fillna(0.0)

    if cfg.plot_exclusion_map and "in_de" in gdf.columns:
        plot_map_in_de(
            gdf,
            value_col="is_excluded_by_population_buffer",
            mask_col="in_de",
            title=f"Tsani-proxy excluded cells (pop>{cfg.population_density_threshold:.0f}, r={cfg.buffer_radius_km:.0f} km)",
            cmap="viridis",
        )

    # --------------------------------------------------------
    # Curves
    # --------------------------------------------------------
    curve = add_lcoe_metrics_pv_base_only(curve_raw)

    if not curve.empty:
        curve["cum_gw_total_system"] = existing_total_gw + curve["cum_gw"]
        curve["cum_twh_per_year_total_system"] = existing_total_twh_per_year + curve["cum_twh_per_year"]
        curve["existing_total_gw"] = existing_total_gw
        curve["existing_total_twh_per_year"] = existing_total_twh_per_year

        curve["lcoe_total_pv_step_eur_per_mwh"] = curve["lcoe_base_pv_step_eur_per_mwh"]
        curve["lcoe_total_pv_cum_eur_per_mwh"] = curve["lcoe_base_pv_cum_eur_per_mwh"]

    phase_switch_gw_total = np.nan
    phase_switch_twh_per_year_total = np.nan

    if meta.get("phase_switch_gw") is not None and np.isfinite(meta.get("phase_switch_gw", np.nan)):
        phase_switch_gw_total = existing_total_gw + float(meta["phase_switch_gw"])

    if meta.get("phase_switch_twh_per_year") is not None and np.isfinite(meta.get("phase_switch_twh_per_year", np.nan)):
        phase_switch_twh_per_year_total = existing_total_twh_per_year + float(meta["phase_switch_twh_per_year"])

    if cfg.plot_curves and not curve.empty:
        plot_base_lcoe_curves(
            curve=curve,
            phase_switch_gw=phase_switch_gw_total,
            phase_switch_twh_per_year=phase_switch_twh_per_year_total,
            x_col_gw="cum_gw_total_system",
            x_col_twh="cum_twh_per_year_total_system",
            title_prefix="Tsani-proxy",
        )

    print(f"[RESULT] Total new turbines: {int(gdf['n_new'].sum()):,}")
    print(f"[RESULT] Total added GW: {gdf['added_gw'].sum():.2f}")
    print(f"[RESULT] Total base cost (EUR PV): {gdf['added_cost_eur'].sum():.3e}")

    suffix = build_output_suffix(
        use_state_targets=shared.use_state_targets,
        build_limit_mode=shared.build_limit_mode,
        population_density_threshold=cfg.population_density_threshold,
        buffer_radius_km=cfg.buffer_radius_km,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    out_cells = cfg.output_dir / f"placement_cells_iterative_tsani_proxy_{suffix}.parquet"
    out_curve = cfg.output_dir / f"cost_curve_iterative_tsani_proxy_{suffix}.parquet"
    out_state = cfg.output_dir / f"state_report_iterative_tsani_proxy_{suffix}.parquet"
    out_meta = cfg.output_dir / f"meta_tsani_proxy_{suffix}.parquet"

    print(f"[SAVE] cells -> {out_cells}")
    gdf.to_parquet(out_cells, index=False)

    print(f"[SAVE] curve -> {out_curve}")
    curve.to_parquet(out_curve, index=False)

    print(f"[SAVE] state report -> {out_state}")
    state_report.to_parquet(out_state, index=False)

    meta_df = pd.DataFrame([meta])
    meta_df["population_density_threshold"] = float(cfg.population_density_threshold)
    meta_df["buffer_radius_km"] = float(cfg.buffer_radius_km)
    meta_df["n_sensitive_cells"] = int(gdf["is_sensitive_population_cell"].sum())
    meta_df["n_excluded_cells"] = int(gdf["is_excluded_by_population_buffer"].sum())

    print(f"[SAVE] meta -> {out_meta}")
    meta_df.to_parquet(out_meta, index=False)

    return {
        "placement_cells": gdf,
        "cost_curve": curve,
        "state_report": state_report,
        "meta": meta_df,
    }


# ============================================================
# Validation
# ============================================================

def validate_config(cfg: TsaniProxyConfig, shared: TsaniProxySharedConfig) -> None:
    allowed_build = {"total_gw", "exhaustion"}
    if shared.build_limit_mode not in allowed_build:
        raise ValueError(f"shared.build_limit_mode must be one of {allowed_build}, got {shared.build_limit_mode!r}")

    if shared.build_limit_mode == "total_gw":
        if shared.target_total_gw is None:
            raise ValueError("shared.target_total_gw must be set when build_limit_mode == 'total_gw'")
        if float(shared.target_total_gw) < 0:
            raise ValueError("shared.target_total_gw must be >= 0")

    if not (0.0 <= shared.state_lower_slack < 1.0):
        raise ValueError("shared.state_lower_slack must be in [0,1)")

    if shared.state_upper_slack < 0.0:
        raise ValueError("shared.state_upper_slack must be >= 0")

    if cfg.population_density_threshold < 0:
        raise ValueError("population_density_threshold must be >= 0")

    if cfg.buffer_radius_km < 0:
        raise ValueError("buffer_radius_km must be >= 0")


# ============================================================
# Targets
# ============================================================

def build_state_targets_b2045() -> pd.DataFrame:
    bl_to_id = {
        "SH": "01", "HH": "02", "NI": "03", "HB": "04", "NW": "05", "HE": "06", "RP": "07", "BW": "08",
        "BY": "09", "SL": "10", "BE": "11", "BB": "12", "MV": "13", "SN": "14", "ST": "15", "TH": "16",
    }
    targets_bl = {
        "BW": 10.4, "BY": 13.1, "BE": 0.0, "BB": 20.5, "HB": 0.2, "HH": 0.1, "HE": 8.8, "MV": 10.8,
        "NI": 26.5, "NW": 18.6, "RP": 9.8, "SL": 1.2, "SN": 4.8, "ST": 11.7, "SH": 15.6, "TH": 7.7,
    }
    df = pd.DataFrame({
        "state_id": [bl_to_id[k] for k in targets_bl.keys()],
        "target_gw": list(targets_bl.values()),
    })
    return df.sort_values("state_id").reset_index(drop=True)


def build_state_additions_df(
    gdf: gpd.GeoDataFrame,
    targets_df: pd.DataFrame,
    state_id_col: str,
    existing_gw_col: str,
) -> pd.DataFrame:
    existing = (
        gdf.groupby(state_id_col, as_index=False)[existing_gw_col]
        .sum()
        .rename(columns={existing_gw_col: "existing_gw"})
    )
    out = targets_df.merge(existing, on="state_id", how="left")
    out["existing_gw"] = out["existing_gw"].fillna(0.0).astype(float)
    out["add_gw"] = (out["target_gw"].astype(float) - out["existing_gw"]).clip(lower=0.0)
    return out


# ============================================================
# Tsani-proxy distance helper
# ============================================================

def add_distance_to_sensitive_population_cells(
    gdf: gpd.GeoDataFrame,
    sensitive_col: str,
    out_dist_col: str,
) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if out.crs is None:
        raise ValueError("gdf.crs is None. Need projected CRS for distance calculations.")

    if out["idm"].duplicated().any():
        n_dup = int(out["idm"].duplicated().sum())
        raise ValueError(f"Duplicate idm before distance calculation: {n_dup:,}")

    sensitive = out.loc[out[sensitive_col].astype(float) > 0.5, ["idm", "geometry"]].copy()

    if sensitive.empty:
        out[out_dist_col] = np.inf
        return out

    tmp = gpd.sjoin_nearest(
        out[["idm", "geometry"]].copy(),
        sensitive[["geometry"]].copy(),
        how="left",
        distance_col="_dist_sensitive_m",
    )

    # IMPORTANT:
    # sjoin_nearest can return multiple rows per idm if several sensitive cells
    # have exactly the same nearest distance.
    # Therefore collapse back to one row per idm.
    tmp["_dist_sensitive_m"] = pd.to_numeric(tmp["_dist_sensitive_m"], errors="coerce")

    dist_min = (
        tmp.groupby("idm", as_index=False)["_dist_sensitive_m"]
        .min()
    )

    out = out.merge(dist_min, on="idm", how="left")

    if out["idm"].duplicated().any():
        n_dup = int(out["idm"].duplicated().sum())
        raise ValueError(f"Duplicate idm after distance calculation: {n_dup:,}")

    out[out_dist_col] = out["_dist_sensitive_m"] / 1000.0
    out[out_dist_col] = out[out_dist_col].fillna(np.inf)
    out = out.drop(columns=["_dist_sensitive_m"], errors="ignore")

    return out

# ============================================================
# PV helpers
# ============================================================

def pv_annuity_factor(wacc: float, lifetime_yrs: int) -> float:
    r = float(wacc)
    L = int(lifetime_yrs)

    if L <= 0:
        raise ValueError("lifetime_yrs must be > 0")

    if abs(r) < 1e-15:
        return float(L)

    return float((1.0 - (1.0 + r) ** (-L)) / r)


# ============================================================
# Index map + arrays
# ============================================================

def build_index_maps(
    gdf: gpd.GeoDataFrame,
    x_col: str = "x_km",
    y_col: str = "y_km",
) -> Tuple[Dict[str, Tuple[int, int]], List[str], int, int, int, int]:
    df = gdf[[x_col, y_col, "idm"]].copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    x = df[x_col].astype(int).to_numpy()
    y = df[y_col].astype(int).to_numpy()

    xmin, xmax = int(x.min()), int(x.max())
    ymin, ymax = int(y.min()), int(y.max())
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1

    rr = (y - ymin).astype(int)
    cc = (x - xmin).astype(int)

    idm_to_rc: Dict[str, Tuple[int, int]] = {}
    rc_to_idm = np.full((ny, nx), "", dtype=object)

    for _idm, r, c in zip(df["idm"].astype(str).to_numpy(), rr, cc):
        idm_to_rc[_idm] = (int(r), int(c))
        rc_to_idm[r, c] = _idm

    return idm_to_rc, list(rc_to_idm.ravel()), xmin, ymin, ny, nx


def gdf_to_array_numeric(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    x_col: str = "x_km",
    y_col: str = "y_km",
) -> Tuple[np.ndarray, int, int]:
    df = gdf[[x_col, y_col, value_col]].copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    x = df[x_col].astype(int).to_numpy()
    y = df[y_col].astype(int).to_numpy()

    xmin, xmax = int(x.min()), int(x.max())
    ymin, ymax = int(y.min()), int(y.max())
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1

    arr = np.full((ny, nx), np.nan, dtype=np.float64)
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    rr = (y - ymin).astype(int)
    cc = (x - xmin).astype(int)
    arr[rr, cc] = vals

    return arr, xmin, ymin


def gdf_to_array(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    x_col: str = "x_km",
    y_col: str = "y_km",
) -> Tuple[np.ndarray, Tuple[int, int, int, int], int, int]:
    arr, xmin, ymin = gdf_to_array_numeric(gdf, value_col=value_col, x_col=x_col, y_col=y_col)
    df = gdf[[x_col, y_col]].copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    x = df[x_col].astype(int).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    extent = (int(x.min()), int(x.max()), int(y.min()), int(y.max()))
    return arr, extent, xmin, ymin


# ============================================================
# Engineering arrays
# ============================================================

def build_cell_engineering_arrays(
    gdf: gpd.GeoDataFrame,
    *,
    turbine_power_mw_col: str,
    turbine_power_kw_col: str,
    turbine_cost_col: str,
    flh_col: str,
    x_col: str = "x_km",
    y_col: str = "y_km",
    fixed_om_share_of_capex_per_year: float = 0.025,
    lifetime_yrs: int = 20,
    wacc: float = 0.06,
) -> Dict[str, np.ndarray]:
    power_mw, _, _ = gdf_to_array_numeric(gdf, turbine_power_mw_col, x_col=x_col, y_col=y_col)
    power_kw, _, _ = gdf_to_array_numeric(gdf, turbine_power_kw_col, x_col=x_col, y_col=y_col)
    cost_in, _, _ = gdf_to_array_numeric(gdf, turbine_cost_col, x_col=x_col, y_col=y_col)
    flh_net, _, _ = gdf_to_array_numeric(gdf, flh_col, x_col=x_col, y_col=y_col)

    power_mw = np.nan_to_num(power_mw, nan=0.0)
    power_kw = np.nan_to_num(power_kw, nan=0.0)
    cost_in = np.nan_to_num(cost_in, nan=0.0)
    flh_net = np.nan_to_num(flh_net, nan=0.0)

    pv_factor = pv_annuity_factor(wacc=wacc, lifetime_yrs=lifetime_yrs)

    capex_eur_per_turbine = power_kw * cost_in
    pv_fom_eur_per_turbine = capex_eur_per_turbine * float(fixed_om_share_of_capex_per_year) * pv_factor
    pv_base_eur_per_turbine = capex_eur_per_turbine + pv_fom_eur_per_turbine

    annual_mwh_per_turbine = power_mw * flh_net
    pv_energy_mwh_per_turbine = annual_mwh_per_turbine * pv_factor

    return {
        "power_mw": power_mw,
        "power_kw": power_kw,
        "flh_net": flh_net,
        "capex_eur_per_turbine": capex_eur_per_turbine,
        "pv_fom_eur_per_turbine": pv_fom_eur_per_turbine,
        "pv_base_eur_per_turbine": pv_base_eur_per_turbine,
        "annual_mwh_per_turbine": annual_mwh_per_turbine,
        "pv_energy_mwh_per_turbine": pv_energy_mwh_per_turbine,
    }


# ============================================================
# Allocation
# ============================================================

def allocate_iterative_min_base_pv_lcoe(
    gdf: gpd.GeoDataFrame,
    targets_df: pd.DataFrame,
    *,
    state_id_col: str,
    n_new_max_col: str,
    flh_col: str,
    cf_col: str,
    turbine_power_mw_col: str,
    turbine_power_kw_col: str,
    turbine_cost_col: str,
    use_state_targets: bool,
    state_upper_slack: float = 0.0,
    build_limit_mode: str = "total_gw",
    target_new_build_gw: Optional[float] = None,
    eps_rel: float = 1e-12,
    in_de_col: Optional[str] = "in_de",
    debug_every_steps: int = 0,
    lifetime_yrs: int = 20,
    wacc: float = 0.06,
    fixed_om_share_of_capex_per_year: float = 0.025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    idm_to_rc, rc_to_idm_flat, _xmin, _ymin, ny, nx = build_index_maps(gdf)

    cf, _, _ = gdf_to_array_numeric(gdf, value_col=cf_col)
    cf = np.nan_to_num(cf, nan=0.0).astype(np.float64)

    nmax, _, _ = gdf_to_array_numeric(gdf, value_col=n_new_max_col)
    nmax = np.nan_to_num(nmax, nan=0.0)
    nmax = np.floor(nmax).astype(int)

    eng = build_cell_engineering_arrays(
        gdf=gdf,
        turbine_power_mw_col=turbine_power_mw_col,
        turbine_power_kw_col=turbine_power_kw_col,
        turbine_cost_col=turbine_cost_col,
        flh_col=flh_col,
        fixed_om_share_of_capex_per_year=fixed_om_share_of_capex_per_year,
        lifetime_yrs=lifetime_yrs,
        wacc=wacc,
    )

    power_mw = eng["power_mw"]
    flh_net_arr = eng["flh_net"]
    pv_base_eur_per_turbine = eng["pv_base_eur_per_turbine"]
    pv_energy_mwh_per_turbine = eng["pv_energy_mwh_per_turbine"]
    annual_mwh_per_turbine = eng["annual_mwh_per_turbine"]

    state = np.full((ny, nx), "", dtype=object)
    for _idm, sid in zip(gdf["idm"].astype(str).to_numpy(), gdf[state_id_col].astype(str).to_numpy()):
        rc = idm_to_rc.get(_idm)
        if rc is not None:
            r0, c0 = rc
            state[r0, c0] = str(sid)

    if in_de_col is not None and in_de_col in gdf.columns:
        in_de, _, _ = gdf_to_array_numeric(gdf, value_col=in_de_col)
        in_de_mask = np.isfinite(in_de) & (in_de > 0.5)
    else:
        in_de_mask = np.ones((ny, nx), dtype=bool)

    target_gw_by_state: Dict[str, float] = {}
    max_gw_by_state: Dict[str, float] = {}
    built_gw_by_state: Dict[str, float] = {}

    for _, row in targets_df.iterrows():
        sid = str(row["state_id"])
        add_gw = float(row["add_gw"])
        target_gw_by_state[sid] = max(0.0, add_gw)
        max_gw_by_state[sid] = max(0.0, (1.0 + float(state_upper_slack)) * add_gw)
        built_gw_by_state[sid] = 0.0

    def all_state_targets_met() -> bool:
        if not use_state_targets:
            return True
        for sid, target in target_gw_by_state.items():
            if built_gw_by_state.get(sid, 0.0) + eps_rel < target:
                return False
        return True

    def reached_global_build_limit(built_total_gw: float) -> bool:
        if build_limit_mode == "exhaustion":
            return False
        if target_new_build_gw is None:
            return False
        return built_total_gw + eps_rel >= float(target_new_build_gw)

    def marginal_base_pv_lcoe_score(r0: int, c0: int) -> float:
        pw = float(power_mw[r0, c0])
        pv_base = float(pv_base_eur_per_turbine[r0, c0])
        pv_energy = float(pv_energy_mwh_per_turbine[r0, c0])

        if pw <= 0.0 or pv_base <= 0.0 or pv_energy <= 0.0:
            return float("inf")

        return float(pv_base / pv_energy)

    def can_build_here(r0: int, c0: int, free_national_phase: bool) -> bool:
        if not in_de_mask[r0, c0]:
            return False
        if nmax[r0, c0] <= 0:
            return False
        if power_mw[r0, c0] <= 0:
            return False
        if pv_energy_mwh_per_turbine[r0, c0] <= 0:
            return False

        if not use_state_targets:
            return True

        if free_national_phase:
            return True

        sid = state[r0, c0]
        if not sid:
            return False

        take_gw = float(power_mw[r0, c0]) / 1000.0
        return built_gw_by_state.get(sid, 0.0) + take_gw <= max_gw_by_state.get(sid, 0.0) + eps_rel

    def exists_feasible_state_phase_option() -> bool:
        rr, cc = np.where(in_de_mask & (nmax > 0) & (power_mw > 0) & (pv_energy_mwh_per_turbine > 0))
        for r0, c0 in zip(rr, cc):
            if can_build_here(int(r0), int(c0), free_national_phase=False):
                return True
        return False

    n_new = np.zeros((ny, nx), dtype=int)
    added_cost = np.zeros((ny, nx), dtype=np.float64)
    curve_rows: List[Dict[str, float]] = []

    heap: List[Tuple[float, int, int]] = []
    rr, cc = np.where(in_de_mask & (nmax > 0) & (power_mw > 0) & (pv_energy_mwh_per_turbine > 0))
    for r0, c0 in zip(rr, cc):
        s = marginal_base_pv_lcoe_score(int(r0), int(c0))
        heap.append((s, int(r0), int(c0)))
    heapq.heapify(heap)
    deferred_items: List[Tuple[int, int]] = []
    deferred_seen = set()  

    free_national_phase = not use_state_targets
    phase_switch_gw = np.nan
    phase_switch_twh_per_year = np.nan

    built_total_gw = 0.0
    built_twh_total = 0.0
    steps = 0

    while heap or (use_state_targets and (not free_national_phase) and deferred_items):
        if not heap and use_state_targets and (not free_national_phase) and deferred_items:
            free_national_phase = True
            phase_switch_gw = built_total_gw
            phase_switch_twh_per_year = built_twh_total

            print(
                f"[PHASE] heap exhausted in state phase; switching to free national expansion at "
                f"{built_total_gw:.2f} GW and {built_twh_total:.2f} TWh/year"
            )

            n_reinserted = 0
            for rr0, cc0 in deferred_items:
                if (
                    nmax[rr0, cc0] > 0
                    and in_de_mask[rr0, cc0]
                    and power_mw[rr0, cc0] > 0
                    and pv_energy_mwh_per_turbine[rr0, cc0] > 0
                ):
                    s = marginal_base_pv_lcoe_score(rr0, cc0)
                    heapq.heappush(heap, (s, rr0, cc0))
                    n_reinserted += 1

            deferred_items = []
            deferred_seen.clear()

            print(f"[PHASE] reinserted deferred locations: {n_reinserted:,}")        
        if reached_global_build_limit(built_total_gw):
            break

        if use_state_targets and (not free_national_phase):
            if all_state_targets_met() or (not exists_feasible_state_phase_option()):
                free_national_phase = True
                phase_switch_gw = built_total_gw
                phase_switch_twh_per_year = built_twh_total

                print(
                    f"[PHASE] switching to free national expansion at "
                    f"{built_total_gw:.2f} GW and {built_twh_total:.2f} TWh/year"
                )

                n_reinserted = 0
                for rr0, cc0 in deferred_items:
                    if (
                        nmax[rr0, cc0] > 0
                        and in_de_mask[rr0, cc0]
                        and power_mw[rr0, cc0] > 0
                        and pv_energy_mwh_per_turbine[rr0, cc0] > 0
                    ):
                        s = marginal_base_pv_lcoe_score(rr0, cc0)
                        heapq.heappush(heap, (s, rr0, cc0))
                        n_reinserted += 1

                deferred_items = []
                deferred_seen.clear()

                print(f"[PHASE] reinserted deferred locations: {n_reinserted:,}")

        score_now, r0, c0 = heapq.heappop(heap)

        if not can_build_here(r0, c0, free_national_phase=free_national_phase):
            if use_state_targets and (not free_national_phase):
                key = (int(r0), int(c0))
                if key not in deferred_seen and nmax[r0, c0] > 0:
                    deferred_items.append(key)
                    deferred_seen.add(key)
            continue

        sid = state[r0, c0]
        take_mw = float(power_mw[r0, c0])
        take_gw = take_mw / 1000.0
        take_mwh_per_year = float(annual_mwh_per_turbine[r0, c0])
        take_twh_per_year = take_mwh_per_year / 1e6
        base_cost = float(pv_base_eur_per_turbine[r0, c0])
        pv_energy_step = float(pv_energy_mwh_per_turbine[r0, c0])

        if build_limit_mode == "total_gw" and target_new_build_gw is not None:
            if built_total_gw + take_gw > float(target_new_build_gw) + eps_rel:
                break

        n_new[r0, c0] += 1
        nmax[r0, c0] -= 1
        added_cost[r0, c0] += base_cost

        built_total_gw += take_gw
        built_twh_total += take_twh_per_year
        if sid:
            built_gw_by_state[sid] = built_gw_by_state.get(sid, 0.0) + take_gw

        steps += 1
        if debug_every_steps and (steps % int(debug_every_steps) == 0):
            phase_name = "free" if free_national_phase else "state_targets"
            print(
                f"[ITER] step={steps:,} phase={phase_name} "
                f"built_total_gw={built_total_gw:.2f} built_total_twh={built_twh_total:.2f} heap={len(heap):,}"
            )

        curve_rows.append({
            "phase": "free_after_state_targets" if (use_state_targets and free_national_phase) else ("national" if not use_state_targets else "state_target_phase"),
            "state_id": sid,
            "idm": rc_to_idm_flat[r0 * nx + c0],
            "take_turbines": 1,
            "take_mw": take_mw,
            "take_gw": take_gw,
            "take_mwh_per_year": take_mwh_per_year,
            "take_twh_per_year": take_twh_per_year,
            "score_base_pv_lcoe": float(score_now),
            "cf": float(cf[r0, c0]),
            "flh_net": float(flh_net_arr[r0, c0]),
            "pv_base_cost_eur_step": base_cost,
            "pv_energy_mwh_step": pv_energy_step,
            "built_gw_total_after": float(built_total_gw),
            "built_gw_state_after": float(built_gw_by_state.get(sid, 0.0)) if sid else 0.0,
        })

        if nmax[r0, c0] > 0:
            heapq.heappush(heap, (score_now, r0, c0))

    placement = pd.DataFrame({
        "idm": np.array(rc_to_idm_flat, dtype=object),
        "n_new": n_new.ravel().astype(int),
        "added_gw": np.where(power_mw.ravel() > 0, n_new.ravel().astype(float) * power_mw.ravel() / 1000.0, 0.0),
        "added_cost_eur": added_cost.ravel().astype(float),
    })
    placement = placement[placement["idm"] != ""].reset_index(drop=True)

    if curve_rows:
        curve = pd.DataFrame(curve_rows)
        curve["step"] = np.arange(1, len(curve) + 1)
        curve["cum_gw"] = curve["take_gw"].cumsum()
        curve["cum_twh_per_year"] = curve["take_twh_per_year"].cumsum()
        curve["cum_base_cost_eur"] = curve["pv_base_cost_eur_step"].cumsum()
        curve["cum_pv_energy_mwh"] = curve["pv_energy_mwh_step"].cumsum()
        curve["avg_base_cost_eur_per_mw"] = curve["cum_base_cost_eur"] / np.maximum(curve["cum_gw"] * 1000.0, 1e-12)
    else:
        curve = pd.DataFrame()

    rep = targets_df.copy()
    rep["target_gw_phaseA"] = rep["add_gw"].astype(float)
    rep["max_gw"] = rep["add_gw"].astype(float) * (1.0 + float(state_upper_slack))
    rep["built_gw"] = rep["state_id"].astype(str).map(lambda s: built_gw_by_state.get(s, 0.0)).astype(float)
    rep["gap_to_target_gw_phaseA"] = (rep["target_gw_phaseA"] - rep["built_gw"]).clip(lower=0.0)

    meta = {
        "phase_switch_gw": float(phase_switch_gw) if np.isfinite(phase_switch_gw) else np.nan,
        "phase_switch_twh_per_year": float(phase_switch_twh_per_year) if np.isfinite(phase_switch_twh_per_year) else np.nan,
        "built_total_gw": float(built_total_gw),
        "built_total_twh_per_year": float(built_twh_total),
    }

    return placement, curve, rep, meta


# ============================================================
# Remaining potential
# ============================================================

def add_remaining_potential_columns_variable_turbine(
    gdf: gpd.GeoDataFrame,
    pot_mw_per_km2: float,
    turbine_power_mw_col: str,
    existing_gw_col: str,
    existing_proxy_mw_per_turbine: float,
) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g["pot_mw"] = g["pot_area_km2"].astype(float) * float(pot_mw_per_km2)

    if existing_gw_col in g.columns:
        g["existing_mw"] = g[existing_gw_col].astype(float) * 1000.0
    else:
        g["existing_mw"] = g["n_existing"].astype(float) * float(existing_proxy_mw_per_turbine)

    g["remain_mw"] = (g["pot_mw"] - g["existing_mw"]).clip(lower=0.0)

    pw = pd.to_numeric(g[turbine_power_mw_col], errors="coerce").astype(float)
    g["n_new_max"] = np.where(
        pw > 0,
        np.floor(g["remain_mw"] / pw),
        0,
    ).astype(int)
    return g


# ============================================================
# LCOE metrics
# ============================================================

def add_lcoe_metrics_pv_base_only(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    if curve.empty:
        return curve

    if "step" not in curve.columns:
        curve["step"] = np.arange(1, len(curve) + 1)

    if "cum_gw" not in curve.columns:
        curve["cum_gw"] = curve["take_gw"].cumsum()

    if "cum_twh_per_year" not in curve.columns:
        curve["cum_twh_per_year"] = curve["take_twh_per_year"].cumsum()

    if "cum_base_cost_eur" not in curve.columns:
        curve["cum_base_cost_eur"] = curve["pv_base_cost_eur_step"].cumsum()

    if "cum_pv_energy_mwh" not in curve.columns:
        curve["cum_pv_energy_mwh"] = curve["pv_energy_mwh_step"].cumsum()

    curve["lcoe_base_pv_step_eur_per_mwh"] = (
        curve["pv_base_cost_eur_step"] /
        np.maximum(curve["pv_energy_mwh_step"], 1e-12)
    )

    curve["lcoe_base_pv_cum_eur_per_mwh"] = (
        curve["cum_base_cost_eur"] /
        np.maximum(curve["cum_pv_energy_mwh"], 1e-12)
    )

    return curve


# ============================================================
# Plot helpers
# ============================================================

def plot_map_in_de(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    mask_col: str = "in_de",
    title: str = "",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    value_arr, _extent, _xmin, _ymin = gdf_to_array(gdf, value_col)
    mask_arr, _extent2, _xmin2, _ymin2 = gdf_to_array(gdf, mask_col)

    mask = np.isfinite(mask_arr) & (mask_arr.astype(float) > 0.5)
    data = np.where(mask, value_arr.astype(float), np.nan)

    flat = data[np.isfinite(data)]
    if flat.size == 0:
        print("[PLOT] no finite values inside Germany")
        return

    plt.figure(figsize=figsize)
    plt.imshow(data, origin="lower", cmap=cmap, interpolation="nearest")
    plt.title(title or value_col)
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_base_lcoe_curves(
    curve: pd.DataFrame,
    phase_switch_gw: Optional[float] = None,
    phase_switch_twh_per_year: Optional[float] = None,
    x_col_gw: str = "cum_gw",
    x_col_twh: str = "cum_twh_per_year",
    title_prefix: str = "",
) -> None:
    if curve.empty:
        print("[PLOT] curve is empty")
        return

    prefix = f"{title_prefix} — " if title_prefix else ""

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve[x_col_gw], curve["lcoe_base_pv_cum_eur_per_mwh"], label="Cumulative Base PV-LCOE")
    if phase_switch_gw is not None and np.isfinite(phase_switch_gw):
        ax.axvline(phase_switch_gw, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative buildout incl. existing capacity (GW)")
    ax.set_ylabel("EUR / MWh")
    ax.set_title(f"{prefix}Cumulative Base PV-LCOE vs Capacity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve[x_col_gw], curve["lcoe_base_pv_step_eur_per_mwh"], label="Marginal Base PV-LCOE")
    if phase_switch_gw is not None and np.isfinite(phase_switch_gw):
        ax.axvline(phase_switch_gw, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative buildout incl. existing capacity (GW)")
    ax.set_ylabel("EUR / MWh")
    ax.set_title(f"{prefix}Marginal Base PV-LCOE vs Capacity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve[x_col_twh], curve["lcoe_base_pv_cum_eur_per_mwh"], label="Cumulative Base PV-LCOE")
    if phase_switch_twh_per_year is not None and np.isfinite(phase_switch_twh_per_year):
        ax.axvline(phase_switch_twh_per_year, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative annual generation incl. existing generation (TWh/year)")
    ax.set_ylabel("EUR / MWh")
    ax.set_title(f"{prefix}Cumulative Base PV-LCOE vs Annual Generation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve[x_col_twh], curve["lcoe_base_pv_step_eur_per_mwh"], label="Marginal Base PV-LCOE")
    if phase_switch_twh_per_year is not None and np.isfinite(phase_switch_twh_per_year):
        ax.axvline(phase_switch_twh_per_year, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative annual generation incl. existing generation (TWh/year)")
    ax.set_ylabel("EUR / MWh")
    ax.set_title(f"{prefix}Marginal Base PV-LCOE vs Annual Generation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = TsaniProxyConfig()
    shared = TsaniProxySharedConfig()

    if cfg.input_grid_de_era5_parquet is None:
        cfg.input_grid_de_era5_parquet = cfg.root / "Output" / "grid_de_era5.parquet"

    run_tsani_proxy(cfg, shared)