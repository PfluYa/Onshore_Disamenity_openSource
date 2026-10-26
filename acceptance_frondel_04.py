# -*- coding: utf-8 -*-
"""Hedonic-price onshore-wind disamenity model (Frondel specification).

Places turbines on a German 1-km grid using a greedy allocation with ERA5-
based site-specific technical costs. Residential-property disamenities are
calculated using the distance to the *nearest* turbine, rather than summing
independent exposure contributions from all turbines.

Shared assumptions (capacity density, existing stock, federal-state targets,
slack, overall expansion target, lifetime, WACC and fixed O&M share) are
supplied to ``run_frondel(cfg, shared)``. This module retains the allocation
mode, existing-stock handling, temporal decay, urban classification,
distance specification and numerical fallback settings.
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
from matplotlib.colors import Normalize, LogNorm


# ============================================================
# Module config only
# ============================================================

@dataclass
class FrondelConfig:
    root: Path = Path(__file__).resolve().parent

    # input / output
    input_grid_de_era5_parquet: Optional[Path] = None
    output_dir: Path = Path(__file__).resolve().parent / 'Output' / 'placing_frondel_adjusted'

    # optional cleanup
    drop_unused_columns_at_start: bool = True

    # allocation mode, centrally set from main.py
    allocation_mode: str = "base_only"   # "base_only" or "total_lcoe"

    # temporal decay
    use_temporal_decay: bool = False
    temporal_profile: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
    temporal_tail: float = 0.0

    # urban dummy
    cities_excel: Optional[Path] = None
    cities_sheet: str = "Grossstaedte"
    urban_radius_km: float = 10.0
    urban_col: str = "is_urban_city10km"

    # ids / columns
    state_id_col: str = "state_id"
    existing_gw_col: str = "existing_gw"

    # existing stock switch stays module-specific
    use_existing_stock: bool = True

    # ERA5-based energy / turbine columns
    flh_col: str = "net_flh_era5"
    existing_flh_col: str = "net_flh_existing_era5"
    gross_flh_col: str = "gross_flh_era5"
    cf_col: str = "cf_era5"
    turbine_power_mw_col: str = "rated_power_mw_era5"
    turbine_power_kw_col: str = "rated_power_kw_era5"
    turbine_cost_col: str = "costs_era5"
    turbine_type_col: str = "turbine_type_era5"

    # value proxy for Frondel costs
    property_price_col: str = "price_eur_per_m2_used"
    area_total_col: str = "area_total_m2"

    # Frondel horizon
    frondel_max_distance_km: int = 9

    # numerics / fallback fills
    eps_rel: float = 1e-12
    flh_fill: float = 0.0
    cf_fill: float = 0.0
    turbine_power_mw_fill: float = 0.0
    turbine_power_kw_fill: float = 0.0
    turbine_cost_fill: float = 0.0

    # plotting
    plot_mc_map: bool = True
    plot_mc_col: str = "_added_cost_cell"
    log_scale: bool = True
    clip_q: Tuple[float, float] = (0.02, 0.98)
    plot_curves: bool = True
    debug_every_steps: int = 500

    # relevant columns to keep early
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
        "price_eur_per_m2_used",
        "price_source",
        "price_source_year",
        "price_index_source",
        "pop_2022",
        "wohnflaeche_je_bewohner_m2",
        "area_total_m2",
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
# Optional standalone shared config
# ============================================================

@dataclass
class FrondelSharedConfig:
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


# ============================================================
# Helper
# ============================================================

def build_output_suffix(
    use_state_targets: bool,
    build_limit_mode: str,
    allocation_mode: str,
    use_existing_stock: bool,
) -> str:
    regional_suffix = f"with_states_{build_limit_mode}" if use_state_targets else f"national_{build_limit_mode}"
    existing_suffix = "with_existing" if use_existing_stock else "without_existing"
    return f"{regional_suffix}_{allocation_mode}_{existing_suffix}"


# ============================================================
# Public runner
# ============================================================

def run_frondel(cfg: FrondelConfig, shared: Any) -> Dict[str, Any]:
    validate_module_config(cfg)
    validate_shared_config(shared)

    if cfg.input_grid_de_era5_parquet is None:
        raise ValueError("cfg.input_grid_de_era5_parquet must be set.")

    if cfg.cities_excel is None:
        cfg.cities_excel = cfg.root / "Input Data" / "Gemeindekoordinaten AuszugGV2QAktuell.xlsx"

    print("[RUN] Greedy placement with Frondel nearest-turbine acceptance costs")
    print(f"[MODE] allocation_mode = {cfg.allocation_mode}")
    print(f"[EXISTING STOCK] use_existing_stock = {cfg.use_existing_stock}")
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

    print(f"[LOAD] cities -> {cfg.cities_excel} | sheet={cfg.cities_sheet}")
    cities_df = load_grossstaedte_excel(cfg.cities_excel, cfg.cities_sheet)
    print(f"[INFO] cities rows: {len(cities_df):,}")

    print(f"[URBAN] Building urban dummy with radius={cfg.urban_radius_km:.1f} km...")
    gdf = add_urban_dummy_from_city_centers(
        gdf=gdf,
        cities_df=cities_df,
        radius_km=cfg.urban_radius_km,
        out_col=cfg.urban_col,
    )
    print(f"[URBAN] share urban: {gdf[cfg.urban_col].mean():.3f}")

    # --------------------------------------------------------
    # Base variables
    # --------------------------------------------------------
    gdf["V_eur"] = (
        pd.to_numeric(gdf[cfg.property_price_col], errors="coerce").fillna(0.0).astype(float) *
        pd.to_numeric(gdf[cfg.area_total_col], errors="coerce").fillna(0.0).astype(float)
    )
    gdf["V_eur"] = gdf["V_eur"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
            use_existing_stock=cfg.use_existing_stock,
        )

    # --------------------------------------------------------
    # Theoretical technical potential
    # --------------------------------------------------------
    if cfg.use_existing_stock:
        gdf["existing_gw_cell"] = (
            pd.to_numeric(gdf[cfg.existing_gw_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    else:
        gdf["existing_gw_cell"] = 0.0

    gdf["new_build_gw_max_cell"] = (
        pd.to_numeric(gdf["n_new_max"], errors="coerce").fillna(0.0).astype(float) *
        pd.to_numeric(gdf["turbine_power_mw"], errors="coerce").fillna(0.0).astype(float) / 1000.0
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
    print(f"Existing capacity:              {theoretical_existing_gw:.2f} GW")
    print(f"Additional new-build potential: {theoretical_new_build_gw:.2f} GW")
    print(f"Total technical potential:      {theoretical_total_gw:.2f} GW")
    print(f"Existing generation:            {theoretical_existing_twh_per_year:.2f} TWh/year")
    print(f"Additional new-build generation:{theoretical_new_build_twh_per_year:.2f} TWh/year")
    print(f"Total technical generation:     {theoretical_total_twh_per_year:.2f} TWh/year")

    existing_total_gw = theoretical_existing_gw
    existing_total_twh_per_year = theoretical_existing_twh_per_year

    print(f"[EXISTING] Total existing GW: {existing_total_gw:.2f}")
    print(f"[EXISTING] Total existing TWh/year: {existing_total_twh_per_year:.2f}")

    # --------------------------------------------------------
    # Frondel loss fractions
    # --------------------------------------------------------
    if cfg.use_temporal_decay:
        temporal_decay_factor = compute_temporal_decay_factor(
            profile=cfg.temporal_profile,
            tail=cfg.temporal_tail,
            lifetime_yrs=shared.lifetime_yrs,
            wacc=shared.wacc,
        )
        print(f"[TEMPORAL] decay factor = {temporal_decay_factor:.3f}")
    else:
        temporal_decay_factor = 1.0
        print("[TEMPORAL] disabled -> decay factor = 1.000")

    rural_loss_fracs, urban_loss_fracs = build_frondel_loss_fraction_tables(
        temporal_decay_factor=temporal_decay_factor
    )
    print(f"[FRONDEL] max rural loss frac = {rural_loss_fracs.max():.4f}")
    print(f"[FRONDEL] max urban loss frac = {urban_loss_fracs.max():.4f}")

    # --------------------------------------------------------
    # State targets
    # --------------------------------------------------------
    targets = build_state_targets_b2045()
    targets_df = build_state_additions_df(
        gdf=gdf,
        targets_df=targets,
        state_id_col=cfg.state_id_col,
        existing_gw_col=cfg.existing_gw_col,
        use_existing_stock=cfg.use_existing_stock,
    )
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
    placement, curve_raw, state_report, meta = allocate_iterative_frondel_nearest(
        gdf=gdf,
        targets_df=targets_df,
        state_id_col=cfg.state_id_col,
        n_new_max_col="n_new_max",
        flh_col="flh_net",
        cf_col="cf",
        turbine_power_mw_col="turbine_power_mw",
        turbine_power_kw_col="turbine_power_kw",
        turbine_cost_col="turbine_cost_input",
        value_col="V_eur",
        urban_col=cfg.urban_col,
        rural_loss_fracs=rural_loss_fracs,
        urban_loss_fracs=urban_loss_fracs,
        frondel_max_distance_km=cfg.frondel_max_distance_km,
        allocation_mode=cfg.allocation_mode,
        use_state_targets=shared.use_state_targets,
        state_lower_slack=shared.state_lower_slack,
        state_upper_slack=shared.state_upper_slack,
        build_limit_mode=shared.build_limit_mode,
        target_new_build_gw=additional_gw_needed if shared.build_limit_mode == "total_gw" else None,
        eps_rel=cfg.eps_rel,
        in_de_col="in_de" if "in_de" in gdf.columns else None,
        existing_gw_col=cfg.existing_gw_col,
        existing_proxy_mw_per_turbine=shared.existing_proxy_mw_per_turbine,
        use_existing_stock=cfg.use_existing_stock,
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

    if cfg.plot_mc_map and "in_de" in gdf.columns:
        gdf[cfg.plot_mc_col] = gdf["added_cost_eur"].astype(float)
        plot_mc_in_de(
            gdf,
            mc_col=cfg.plot_mc_col,
            mask_col="in_de",
            title=f"Acceptance cost per cell (EUR PV) — Frondel nearest [{cfg.allocation_mode}]",
            log_scale=cfg.log_scale,
            clip_quantiles=cfg.clip_q,
        )

    # --------------------------------------------------------
    # Curves
    # --------------------------------------------------------
    curve_cum = add_lcoe_metrics_pv_cumulative_variable(curve_raw)
    curve_step = add_lcoe_metrics_pv_marginal_variable(curve_raw)

    if not curve_cum.empty:
        curve_cum["cum_gw_total_system"] = existing_total_gw + curve_cum["cum_gw"]
        curve_cum["cum_twh_per_year_total_system"] = existing_total_twh_per_year + curve_cum["cum_twh_per_year"]
        curve_cum["existing_total_gw"] = existing_total_gw
        curve_cum["existing_total_twh_per_year"] = existing_total_twh_per_year

    if not curve_step.empty:
        curve_step["cum_gw_total_system"] = existing_total_gw + curve_step["cum_gw"]
        curve_step["cum_twh_per_year_total_system"] = existing_total_twh_per_year + curve_step["cum_twh_per_year"]
        curve_step["existing_total_gw"] = existing_total_gw
        curve_step["existing_total_twh_per_year"] = existing_total_twh_per_year

    curve_out = curve_step.copy()

    if not curve_step.empty and not curve_cum.empty:
        cum_cols_to_add = [
            c for c in [
                "step",
                "pv_base_cost_eur_cum",
                "pv_acc_cost_eur_cum",
                "pv_total_cost_eur_cum",
                "lcoe_base_pv_cum_eur_per_mwh",
                "lcoe_acc_pv_cum_eur_per_mwh",
                "lcoe_total_pv_cum_eur_per_mwh",
                "acc_share_pv_cum",
            ]
            if c in curve_cum.columns
        ]

        curve_out = curve_out.merge(
            curve_cum[cum_cols_to_add],
            on="step",
            how="left",
            suffixes=("", "_cumdup"),
        )

    phase_switch_gw_total = np.nan
    phase_switch_twh_per_year_total = np.nan
    if meta.get("phase_switch_gw") is not None and np.isfinite(meta.get("phase_switch_gw", np.nan)):
        phase_switch_gw_total = existing_total_gw + float(meta["phase_switch_gw"])
    if meta.get("phase_switch_twh_per_year") is not None and np.isfinite(meta.get("phase_switch_twh_per_year", np.nan)):
        phase_switch_twh_per_year_total = existing_total_twh_per_year + float(meta["phase_switch_twh_per_year"])

    if cfg.plot_curves and not curve_cum.empty and not curve_step.empty:
        plot_frondel_curves(
            curve_cum=curve_cum,
            curve_step=curve_step,
            phase_switch_gw=phase_switch_gw_total,
            phase_switch_twh_per_year=phase_switch_twh_per_year_total,
        )

    if not curve_step.empty:
        print("\n[MARGINAL PV-LCOE SUMMARY]")
        print(f"Base start:      {curve_step['lcoe_base_pv_step_eur_per_mwh'].iloc[0]:.1f} €/MWh")
        print(f"Total end:       {curve_step['lcoe_total_pv_step_eur_per_mwh'].iloc[-1]:.1f} €/MWh")

    if not curve_cum.empty:
        print("\n[CUMULATIVE PV-LCOE SUMMARY]")
        print(f"Base start:      {curve_cum['lcoe_base_pv_cum_eur_per_mwh'].iloc[0]:.1f} €/MWh")
        print(f"Acceptance end:  {curve_cum['lcoe_acc_pv_cum_eur_per_mwh'].iloc[-1]:.1f} €/MWh")
        print(f"Total end:       {curve_cum['lcoe_total_pv_cum_eur_per_mwh'].iloc[-1]:.1f} €/MWh")
        print(f"Acc share end:   {curve_cum['acc_share_pv_cum'].iloc[-1]*100:.1f}%")
        print(f"Cumulative NEW GW end:         {curve_cum['cum_gw'].iloc[-1]:.2f}")
        print(f"Cumulative TOTAL GW end:       {curve_cum['cum_gw_total_system'].iloc[-1]:.2f}")
        print(f"Cumulative NEW TWh/year end:   {curve_cum['cum_twh_per_year'].iloc[-1]:.2f}")
        print(f"Cumulative TOTAL TWh/year end: {curve_cum['cum_twh_per_year_total_system'].iloc[-1]:.2f}")

    print(f"[RESULT] Total new turbines: {int(gdf['n_new'].sum()):,}")
    print(f"[RESULT] Total added GW: {gdf['added_gw'].sum():.2f}")
    print(f"[RESULT] Total acceptance cost (EUR PV): {gdf['added_cost_eur'].sum():.3e}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = build_output_suffix(
        use_state_targets=shared.use_state_targets,
        build_limit_mode=shared.build_limit_mode,
        allocation_mode=cfg.allocation_mode,
        use_existing_stock=cfg.use_existing_stock,
    )

    out_cells = cfg.output_dir / f"placement_cells_iterative_frondel_nearest_{suffix}.parquet"
    out_curve = cfg.output_dir / f"cost_curve_iterative_frondel_nearest_{suffix}.parquet"
    out_state = cfg.output_dir / f"state_report_iterative_frondel_nearest_{suffix}.parquet"
    out_meta = cfg.output_dir / f"meta_frondel_nearest_{suffix}.parquet"

    print(f"[SAVE] cells -> {out_cells}")
    gdf.to_parquet(out_cells, index=False)

    print(f"[SAVE] curve -> {out_curve}")
    curve_out.to_parquet(out_curve, index=False)

    print(f"[SAVE] state report -> {out_state}")
    state_report.to_parquet(out_state, index=False)

    meta_df = pd.DataFrame([meta])
    print(f"[SAVE] meta -> {out_meta}")
    meta_df.to_parquet(out_meta, index=False)

    return {
        "placement_cells": gdf,
        "cost_curve": curve_out,
        "state_report": state_report,
        "meta": meta_df,
    }

# ============================================================
# Validation
# ============================================================

def validate_module_config(cfg: FrondelConfig) -> None:
    if cfg.allocation_mode not in {"total_lcoe", "base_only"}:
        raise ValueError("allocation_mode must be 'total_lcoe' or 'base_only'")

    if cfg.frondel_max_distance_km <= 0:
        raise ValueError("frondel_max_distance_km must be > 0")

    if cfg.use_temporal_decay:
        if len(cfg.temporal_profile) == 0:
            raise ValueError("temporal_profile must not be empty when use_temporal_decay=True")
        if any((x < 0.0 or x > 1.0) for x in cfg.temporal_profile):
            raise ValueError("All entries in temporal_profile must be in [0,1]")
        if not (0.0 <= cfg.temporal_tail <= 1.0):
            raise ValueError("temporal_tail must be in [0,1]")


def validate_shared_config(shared: Any) -> None:
    required_attrs = [
        "pot_mw_per_km2",
        "existing_proxy_mw_per_turbine",
        "use_state_targets",
        "state_lower_slack",
        "state_upper_slack",
        "build_limit_mode",
        "target_total_gw",
        "lifetime_yrs",
        "wacc",
        "fixed_om_share_of_capex_per_year",
    ]
    missing = [a for a in required_attrs if not hasattr(shared, a)]
    if missing:
        raise ValueError(f"shared config missing attributes: {missing}")

    if shared.build_limit_mode not in {"total_gw", "exhaustion"}:
        raise ValueError("shared.build_limit_mode must be 'total_gw' or 'exhaustion'")

    if shared.build_limit_mode == "total_gw":
        if shared.target_total_gw is None:
            raise ValueError("shared.target_total_gw must be set when build_limit_mode == 'total_gw'")
        if float(shared.target_total_gw) < 0:
            raise ValueError("shared.target_total_gw must be >= 0")

    if not (0.0 <= float(shared.state_lower_slack) < 1.0):
        raise ValueError("shared.state_lower_slack must be in [0,1)")

    if float(shared.state_upper_slack) < 0.0:
        raise ValueError("shared.state_upper_slack must be >= 0")

    if int(shared.lifetime_yrs) <= 0:
        raise ValueError("shared.lifetime_yrs must be > 0")


# ============================================================
# PV metrics
# ============================================================

def add_lcoe_metrics_pv_cumulative_variable(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    if curve.empty:
        return curve

    if "cum_gw" not in curve.columns:
        curve["cum_gw"] = curve["take_gw"].cumsum()

    if "cum_twh_per_year" not in curve.columns:
        curve["cum_twh_per_year"] = curve["take_twh_per_year"].cumsum()

    pv_energy_step = curve["pv_energy_mwh_step"].to_numpy(dtype=float)
    pv_energy_cum = np.cumsum(pv_energy_step)

    pv_base_step = curve["pv_base_cost_eur_step"].to_numpy(dtype=float)
    pv_base_cum = np.cumsum(pv_base_step)

    pv_acc_step = curve["take_cost_eur"].to_numpy(dtype=float)
    pv_acc_cum = np.cumsum(pv_acc_step)

    pv_total_cum = pv_base_cum + pv_acc_cum

    curve["pv_base_cost_eur_cum"] = pv_base_cum
    curve["pv_acc_cost_eur_cum"] = pv_acc_cum
    curve["pv_total_cost_eur_cum"] = pv_total_cum

    curve["lcoe_base_pv_cum_eur_per_mwh"] = pv_base_cum / np.maximum(pv_energy_cum, 1e-12)
    curve["lcoe_acc_pv_cum_eur_per_mwh"] = pv_acc_cum / np.maximum(pv_energy_cum, 1e-12)
    curve["lcoe_total_pv_cum_eur_per_mwh"] = pv_total_cum / np.maximum(pv_energy_cum, 1e-12)
    curve["acc_share_pv_cum"] = curve["lcoe_acc_pv_cum_eur_per_mwh"] / np.maximum(curve["lcoe_total_pv_cum_eur_per_mwh"], 1e-12)
    return curve


def add_lcoe_metrics_pv_marginal_variable(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    if curve.empty:
        return curve

    if "cum_gw" not in curve.columns:
        curve["cum_gw"] = curve["take_gw"].cumsum()

    if "cum_twh_per_year" not in curve.columns:
        curve["cum_twh_per_year"] = curve["take_twh_per_year"].cumsum()

    curve["lcoe_base_pv_step_eur_per_mwh"] = curve["pv_base_cost_eur_step"] / np.maximum(curve["pv_energy_mwh_step"], 1e-12)
    curve["lcoe_acc_pv_step_eur_per_mwh"] = curve["take_cost_eur"] / np.maximum(curve["pv_energy_mwh_step"], 1e-12)
    curve["lcoe_total_pv_step_eur_per_mwh"] = (curve["pv_base_cost_eur_step"] + curve["take_cost_eur"]) / np.maximum(curve["pv_energy_mwh_step"], 1e-12)
    return curve


# ============================================================
# Temporal decay helper
# ============================================================

def compute_temporal_decay_factor(
    profile: Tuple[float, ...],
    tail: float,
    lifetime_yrs: int,
    wacc: float,
) -> float:
    r = float(wacc)

    pv_full = 0.0
    for t in range(1, lifetime_yrs + 1):
        pv_full += 1.0 / ((1.0 + r) ** t)

    pv_temp = 0.0
    for t in range(1, lifetime_yrs + 1):
        if t <= len(profile):
            lam = float(profile[t - 1])
        else:
            lam = float(tail)
        pv_temp += lam / ((1.0 + r) ** t)

    if pv_full <= 0:
        return 1.0

    return float(pv_temp / pv_full)


# ============================================================
# German number parsing
# ============================================================

def _to_float_de(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_int_de(x) -> int:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace(".", "").replace(",", "")
    try:
        return int(s)
    except ValueError:
        return np.nan


def load_grossstaedte_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    needed = {"Name", "Einwohner", "Lon", "Lat"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Expected columns {needed} in Excel sheet '{sheet_name}', got: {list(df.columns)}")

    out = df[["Name", "Einwohner", "Lon", "Lat"]].copy()
    out["Einwohner"] = out["Einwohner"].map(_to_int_de).astype("Int64")
    out["Lon"] = out["Lon"].map(_to_float_de)
    out["Lat"] = out["Lat"].map(_to_float_de)
    out = out.dropna(subset=["Lon", "Lat"]).reset_index(drop=True)
    return out


# ============================================================
# Urban dummy
# ============================================================

def add_urban_dummy_from_city_centers(
    gdf: gpd.GeoDataFrame,
    cities_df: pd.DataFrame,
    radius_km: float,
    out_col: str,
) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("gdf.crs is None. Need a projected CRS for distance calculations.")

    cities_gdf = gpd.GeoDataFrame(
        cities_df.copy(),
        geometry=gpd.points_from_xy(cities_df["Lon"], cities_df["Lat"]),
        crs="EPSG:4326",
    ).to_crs(gdf.crs)

    tmp = gpd.sjoin_nearest(
        gdf[["idm", "geometry"]].copy(),
        cities_gdf[["geometry"]].copy(),
        how="left",
        distance_col="_dist_to_city_m",
    )

    out = gdf.merge(tmp[["idm", "_dist_to_city_m"]], on="idm", how="left")
    out[out_col] = (out["_dist_to_city_m"].astype(float) <= float(radius_km) * 1000.0).astype(int)
    return out


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
    use_existing_stock: bool = True,
) -> pd.DataFrame:
    if use_existing_stock:
        existing = (
            gdf.groupby(state_id_col, as_index=False)[existing_gw_col]
            .sum()
            .rename(columns={existing_gw_col: "existing_gw"})
        )
    else:
        existing = (
            gdf[[state_id_col]]
            .drop_duplicates()
            .rename(columns={state_id_col: "state_id"})
        )
        existing["existing_gw"] = 0.0

    out = targets_df.merge(existing, on="state_id", how="left")
    out["existing_gw"] = out["existing_gw"].fillna(0.0).astype(float)
    out["add_gw"] = (out["target_gw"].astype(float) - out["existing_gw"]).clip(lower=0.0)
    return out


# ============================================================
# Frondel tables
# ============================================================

def frondel_band_betas_rural_tableA3() -> np.ndarray:
    return np.array([-0.0841, -0.0697, -0.0610, -0.0492, -0.0486, -0.0325, -0.0251, -0.00971], dtype=float)


def frondel_band_betas_urban_interaction_tableA3() -> np.ndarray:
    return np.array([0.0681, 0.0595, 0.0442, 0.0363, 0.0424, 0.0239, 0.0199, 0.00563], dtype=float)


def beta_to_loss_frac(beta: float) -> float:
    return float(max(0.0, 1.0 - np.exp(beta)))


def build_frondel_loss_fraction_tables(
    temporal_decay_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    betas_rural = frondel_band_betas_rural_tableA3()
    betas_urban = betas_rural + frondel_band_betas_urban_interaction_tableA3()

    rural = np.array([beta_to_loss_frac(b) for b in betas_rural], dtype=float)
    urban = np.array([beta_to_loss_frac(b) for b in betas_urban], dtype=float)

    rural = np.concatenate([rural, np.array([0.0], dtype=float)])
    urban = np.concatenate([urban, np.array([0.0], dtype=float)])

    rural *= float(temporal_decay_factor)
    urban *= float(temporal_decay_factor)

    return rural, urban


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

def pv_annuity_factor(wacc: float, lifetime_yrs: int) -> float:
    r = float(wacc)
    L = int(lifetime_yrs)

    if L <= 0:
        raise ValueError("lifetime_yrs must be > 0")

    if abs(r) < 1e-15:
        return float(L)

    return float((1.0 - (1.0 + r) ** (-L)) / r)


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
# Nearest-turbine Frondel helpers
# ============================================================

def build_distance_band_kernel(max_distance_km: int) -> Dict[str, np.ndarray]:
    R = int(max_distance_km)
    yy, xx = np.mgrid[-R:R+1, -R:R+1]
    dist = np.sqrt(xx**2 + yy**2)

    inside = dist < float(max_distance_km)
    band_idx = np.floor(dist).astype(int)
    band_idx = np.where(inside, band_idx, -1)

    return {
        "dist": dist.astype(np.float64),
        "inside": inside.astype(bool),
        "band_idx": band_idx.astype(int),
    }


def existing_turbine_presence_array(
    gdf: gpd.GeoDataFrame,
    *,
    existing_gw_col: str,
    existing_proxy_mw_per_turbine: float,
    use_existing_stock: bool,
    x_col: str = "x_km",
    y_col: str = "y_km",
) -> np.ndarray:
    if not use_existing_stock:
        arr, _, _ = gdf_to_array_numeric(
            gdf.assign(_zero_existing_proxy=0.0),
            value_col="_zero_existing_proxy",
            x_col=x_col,
            y_col=y_col,
        )
        arr = np.nan_to_num(arr, nan=0.0)
        return np.zeros_like(arr, dtype=bool)

    if "n_existing" in gdf.columns:
        tmp = pd.to_numeric(gdf["n_existing"], errors="coerce").fillna(0.0).astype(float)
        count = np.where(tmp > 0, np.maximum(1, np.round(tmp)).astype(int), 0)
    else:
        gw = pd.to_numeric(gdf[existing_gw_col], errors="coerce").fillna(0.0).astype(float)
        mw = gw * 1000.0
        count = np.where(
            mw > 0.0,
            np.maximum(1, np.round(mw / float(existing_proxy_mw_per_turbine))).astype(int),
            0,
        )

    tmp_gdf = gdf[["x_km", "y_km"]].copy()
    tmp_gdf["existing_count_proxy"] = count
    arr, _, _ = gdf_to_array_numeric(tmp_gdf, value_col="existing_count_proxy", x_col=x_col, y_col=y_col)
    arr = np.nan_to_num(arr, nan=0.0)
    return (arr > 0.5).astype(bool)


def initialize_nearest_exposure_from_existing(
    *,
    residential_mask: np.ndarray,
    in_de_mask: np.ndarray,
    value_arr: np.ndarray,
    urban_mask: np.ndarray,
    existing_turbine_mask: np.ndarray,
    rural_loss_fracs: np.ndarray,
    urban_loss_fracs: np.ndarray,
    dist_kernel: np.ndarray,
    inside_kernel: np.ndarray,
    band_idx_kernel: np.ndarray,
    max_distance_km: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = value_arr.shape
    current_nearest_dist = np.full((ny, nx), np.inf, dtype=np.float64)
    current_loss_frac = np.zeros((ny, nx), dtype=np.float64)
    current_loss_eur = np.zeros((ny, nx), dtype=np.float64)

    rr, cc = np.where(existing_turbine_mask & in_de_mask)
    if len(rr) == 0:
        return current_nearest_dist, current_loss_frac, current_loss_eur

    R = int(max_distance_km)

    for r0, c0 in zip(rr, cc):
        _apply_nearest_turbine_local(
            current_nearest_dist=current_nearest_dist,
            current_loss_frac=current_loss_frac,
            current_loss_eur=current_loss_eur,
            value_arr=value_arr,
            urban_mask=urban_mask,
            residential_mask=residential_mask,
            in_de_mask=in_de_mask,
            turbine_r=int(r0),
            turbine_c=int(c0),
            rural_loss_fracs=rural_loss_fracs,
            urban_loss_fracs=urban_loss_fracs,
            dist_kernel=dist_kernel,
            inside_kernel=inside_kernel,
            band_idx_kernel=band_idx_kernel,
            max_distance_km=R,
            commit=True,
        )

    return current_nearest_dist, current_loss_frac, current_loss_eur


def _apply_nearest_turbine_local(
    *,
    current_nearest_dist: np.ndarray,
    current_loss_frac: np.ndarray,
    current_loss_eur: np.ndarray,
    value_arr: np.ndarray,
    urban_mask: np.ndarray,
    residential_mask: np.ndarray,
    in_de_mask: np.ndarray,
    turbine_r: int,
    turbine_c: int,
    rural_loss_fracs: np.ndarray,
    urban_loss_fracs: np.ndarray,
    dist_kernel: np.ndarray,
    inside_kernel: np.ndarray,
    band_idx_kernel: np.ndarray,
    max_distance_km: int,
    commit: bool,
) -> Tuple[float, Dict[str, np.ndarray]]:
    ny, nx = current_nearest_dist.shape
    R = int(max_distance_km)

    r0 = max(0, turbine_r - R)
    r1 = min(ny, turbine_r + R + 1)
    c0 = max(0, turbine_c - R)
    c1 = min(nx, turbine_c + R + 1)

    rs = slice(r0, r1)
    cs = slice(c0, c1)

    kr0 = r0 - (turbine_r - R)
    kr1 = kr0 + (r1 - r0)
    kc0 = c0 - (turbine_c - R)
    kc1 = kc0 + (c1 - c0)

    dist_patch = dist_kernel[kr0:kr1, kc0:kc1]
    inside_patch = inside_kernel[kr0:kr1, kc0:kc1]
    band_patch = band_idx_kernel[kr0:kr1, kc0:kc1]

    res_patch = residential_mask[rs, cs]
    de_patch = in_de_mask[rs, cs]
    eligible = res_patch & de_patch & inside_patch

    if not np.any(eligible):
        return 0.0, {}

    cur_dist_patch = current_nearest_dist[rs, cs]
    cur_loss_patch = current_loss_frac[rs, cs]
    val_patch = value_arr[rs, cs]
    urb_patch = urban_mask[rs, cs]

    closer = eligible & (dist_patch + 1e-12 < cur_dist_patch)
    if not np.any(closer):
        return 0.0, {}

    band_safe = np.where(band_patch >= 0, band_patch, 0)
    new_loss_frac_all = np.where(
        urb_patch,
        urban_loss_fracs[band_safe],
        rural_loss_fracs[band_safe],
    )
    new_loss_frac_all = np.where(inside_patch, new_loss_frac_all, 0.0)

    delta_loss_frac = np.where(closer, np.maximum(new_loss_frac_all - cur_loss_patch, 0.0), 0.0)
    added_cost = float(np.sum(val_patch * delta_loss_frac))

    if commit:
        current_nearest_dist[rs, cs] = np.where(closer, dist_patch, cur_dist_patch)
        updated_loss_frac = np.where(closer, np.maximum(cur_loss_patch, new_loss_frac_all), cur_loss_patch)
        current_loss_frac[rs, cs] = updated_loss_frac
        current_loss_eur[rs, cs] = val_patch * updated_loss_frac

    return added_cost, {
        "added_cost": added_cost,
        "n_improved_cells": int(np.count_nonzero(closer)),
        "n_cost_cells": int(np.count_nonzero(delta_loss_frac > 0)),
    }


def marginal_nearest_turbine_cost(
    *,
    current_nearest_dist: np.ndarray,
    current_loss_frac: np.ndarray,
    value_arr: np.ndarray,
    urban_mask: np.ndarray,
    residential_mask: np.ndarray,
    in_de_mask: np.ndarray,
    turbine_r: int,
    turbine_c: int,
    rural_loss_fracs: np.ndarray,
    urban_loss_fracs: np.ndarray,
    dist_kernel: np.ndarray,
    inside_kernel: np.ndarray,
    band_idx_kernel: np.ndarray,
    max_distance_km: int,
) -> float:
    cost, _ = _apply_nearest_turbine_local(
        current_nearest_dist=current_nearest_dist,
        current_loss_frac=current_loss_frac,
        current_loss_eur=np.zeros_like(current_loss_frac),
        value_arr=value_arr,
        urban_mask=urban_mask,
        residential_mask=residential_mask,
        in_de_mask=in_de_mask,
        turbine_r=turbine_r,
        turbine_c=turbine_c,
        rural_loss_fracs=rural_loss_fracs,
        urban_loss_fracs=urban_loss_fracs,
        dist_kernel=dist_kernel,
        inside_kernel=inside_kernel,
        band_idx_kernel=band_idx_kernel,
        max_distance_km=max_distance_km,
        commit=False,
    )
    return float(cost)


def apply_new_turbine_nearest(
    *,
    current_nearest_dist: np.ndarray,
    current_loss_frac: np.ndarray,
    current_loss_eur: np.ndarray,
    value_arr: np.ndarray,
    urban_mask: np.ndarray,
    residential_mask: np.ndarray,
    in_de_mask: np.ndarray,
    turbine_r: int,
    turbine_c: int,
    rural_loss_fracs: np.ndarray,
    urban_loss_fracs: np.ndarray,
    dist_kernel: np.ndarray,
    inside_kernel: np.ndarray,
    band_idx_kernel: np.ndarray,
    max_distance_km: int,
) -> float:
    cost, _ = _apply_nearest_turbine_local(
        current_nearest_dist=current_nearest_dist,
        current_loss_frac=current_loss_frac,
        current_loss_eur=current_loss_eur,
        value_arr=value_arr,
        urban_mask=urban_mask,
        residential_mask=residential_mask,
        in_de_mask=in_de_mask,
        turbine_r=turbine_r,
        turbine_c=turbine_c,
        rural_loss_fracs=rural_loss_fracs,
        urban_loss_fracs=urban_loss_fracs,
        dist_kernel=dist_kernel,
        inside_kernel=inside_kernel,
        band_idx_kernel=band_idx_kernel,
        max_distance_km=max_distance_km,
        commit=True,
    )
    return float(cost)


# ============================================================
# Allocation
# ============================================================

def allocate_iterative_frondel_nearest(
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
    value_col: str,
    urban_col: str,
    rural_loss_fracs: np.ndarray,
    urban_loss_fracs: np.ndarray,
    frondel_max_distance_km: int,
    allocation_mode: str,
    use_state_targets: bool,
    state_lower_slack: float = 0.0,
    state_upper_slack: float = 0.0,
    build_limit_mode: str = "total_gw",
    target_new_build_gw: Optional[float] = None,
    eps_rel: float = 1e-12,
    in_de_col: Optional[str] = "in_de",
    existing_gw_col: str = "existing_gw",
    existing_proxy_mw_per_turbine: float = 2.5,
    use_existing_stock: bool = True,
    debug_every_steps: int = 0,
    lifetime_yrs: int = 20,
    wacc: float = 0.06,
    fixed_om_share_of_capex_per_year: float = 0.025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    idm_to_rc, rc_to_idm_flat, _xmin, _ymin, ny, nx = build_index_maps(gdf)

    value_arr, _, _ = gdf_to_array_numeric(gdf, value_col=value_col)
    value_arr = np.nan_to_num(value_arr, nan=0.0).astype(np.float64)

    U, _, _ = gdf_to_array_numeric(gdf, value_col=urban_col)
    U = np.nan_to_num(U, nan=0.0).astype(np.float64)
    urban_mask = U > 0.5

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

    residential_mask = in_de_mask & np.isfinite(value_arr) & (value_arr > 0.0)

    existing_turbine_mask = existing_turbine_presence_array(
        gdf,
        existing_gw_col=existing_gw_col,
        existing_proxy_mw_per_turbine=existing_proxy_mw_per_turbine,
        use_existing_stock=use_existing_stock,
    )

    kernel = build_distance_band_kernel(frondel_max_distance_km)
    dist_kernel = kernel["dist"]
    inside_kernel = kernel["inside"]
    band_idx_kernel = kernel["band_idx"]

    current_nearest_dist, current_loss_frac, current_loss_eur = initialize_nearest_exposure_from_existing(
        residential_mask=residential_mask,
        in_de_mask=in_de_mask,
        value_arr=value_arr,
        urban_mask=urban_mask,
        existing_turbine_mask=existing_turbine_mask,
        rural_loss_fracs=rural_loss_fracs,
        urban_loss_fracs=urban_loss_fracs,
        dist_kernel=dist_kernel,
        inside_kernel=inside_kernel,
        band_idx_kernel=band_idx_kernel,
        max_distance_km=frondel_max_distance_km,
    )

    print(f"[EXISTING TURBINES] occupied cells with existing turbines: {int(existing_turbine_mask.sum()):,}")
    print(f"[INITIAL] current loss stock from existing turbines (not booked to new build): {float(current_loss_eur.sum()):.3e} EUR PV-equivalent")

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

    def score_base(r0: int, c0: int) -> float:
        pw = float(power_mw[r0, c0])
        pv_base = float(pv_base_eur_per_turbine[r0, c0])
        pv_energy = float(pv_energy_mwh_per_turbine[r0, c0])
        if pw <= 0.0 or pv_base <= 0.0 or pv_energy <= 0.0:
            return float("inf")
        return float(pv_base / pv_energy)

    def marginal_acceptance_cost(r0: int, c0: int) -> float:
        return marginal_nearest_turbine_cost(
            current_nearest_dist=current_nearest_dist,
            current_loss_frac=current_loss_frac,
            value_arr=value_arr,
            urban_mask=urban_mask,
            residential_mask=residential_mask,
            in_de_mask=in_de_mask,
            turbine_r=r0,
            turbine_c=c0,
            rural_loss_fracs=rural_loss_fracs,
            urban_loss_fracs=urban_loss_fracs,
            dist_kernel=dist_kernel,
            inside_kernel=inside_kernel,
            band_idx_kernel=band_idx_kernel,
            max_distance_km=frondel_max_distance_km,
        )

    def score_total(r0: int, c0: int) -> float:
        pw = float(power_mw[r0, c0])
        pv_base = float(pv_base_eur_per_turbine[r0, c0])
        pv_energy = float(pv_energy_mwh_per_turbine[r0, c0])
        if pw <= 0.0 or pv_base <= 0.0 or pv_energy <= 0.0:
            return float("inf")
        pv_acc = float(marginal_acceptance_cost(r0, c0))
        return float((pv_base + pv_acc) / pv_energy)

    def score_active(r0: int, c0: int) -> float:
        if allocation_mode == "total_lcoe":
            return score_total(r0, c0)
        if allocation_mode == "base_only":
            return score_base(r0, c0)
        raise ValueError("allocation_mode must be 'total_lcoe' or 'base_only'")

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

    # Exact dynamic scores: a newly installed nearest turbine can LOWER the
    # marginal exposure cost of a *different* candidate. Merely refreshing the
    # heap top on pop is insufficient, because the currently best candidate may
    # still have a stale and overly HIGH heap key. Maintain versioned scores and
    # explicitly refresh sites sharing an affected residential exposure cell.
    heap: List[Tuple[float, int, int, int]] = []
    heap_version = np.zeros((ny, nx), dtype=np.int64)
    cached_score = np.full((ny, nx), np.inf, dtype=np.float64)
    heap_present = np.zeros((ny, nx), dtype=bool)
    rr, cc = np.where(in_de_mask & (nmax > 0) & (power_mw > 0) & (pv_energy_mwh_per_turbine > 0))
    initial_sites = int(len(rr))
    for r0, c0 in zip(rr, cc):
        s = score_active(int(r0), int(c0))
        cached_score[r0, c0] = s
        heap_present[r0, c0] = True
        heap.append((s, int(r0), int(c0), 0))
    heapq.heapify(heap)
    deferred_items: List[Tuple[int, int]] = []
    deferred_seen = set()

    def insert_current_score(r0: int, c0: int) -> None:
        # Called only when the location is newly activated/reinserted.
        s = score_active(r0, c0)
        cached_score[r0, c0] = s
        heap_version[r0, c0] += 1
        heap_present[r0, c0] = True
        heapq.heappush(heap, (s, r0, c0, int(heap_version[r0, c0])))

    def refresh_candidates_sharing_changed_homes(
        turbine_r: int, turbine_c: int,
        old_distances: np.ndarray, old_r0: int, old_c0: int,
    ) -> None:
        # Only distances / registered losses at residential cells whose nearest
        # turbine changed can alter another location's marginal score. A candidate
        # may be affected iff it is within R km of at least one such cell.
        if allocation_mode != 'total_lcoe':
            return
        r1 = old_r0 + old_distances.shape[0]
        c1 = old_c0 + old_distances.shape[1]
        changed = (
            residential_mask[old_r0:r1, old_c0:c1]
            & (current_nearest_dist[old_r0:r1, old_c0:c1] + 1e-12 < old_distances)
        )
        changed_r, changed_c = np.where(changed)
        if len(changed_r) == 0:
            return
        R = int(frondel_max_distance_km)
        # Local union of candidate footprints, bounded by ±2R from new turbine.
        ar0 = max(0, turbine_r - 2 * R)
        ar1 = min(ny, turbine_r + 2 * R + 1)
        ac0 = max(0, turbine_c - 2 * R)
        ac1 = min(nx, turbine_c + 2 * R + 1)
        affected = np.zeros((ar1-ar0, ac1-ac0), dtype=bool)
        for hr, hc in zip(changed_r + old_r0, changed_c + old_c0):
            br0 = max(ar0, int(hr) - R)
            br1 = min(ar1, int(hr) + R + 1)
            bc0 = max(ac0, int(hc) - R)
            bc1 = min(ac1, int(hc) + R + 1)
            kr0 = br0 - (int(hr) - R)
            kc0 = bc0 - (int(hc) - R)
            affected[br0-ar0:br1-ar0, bc0-ac0:bc1-ac0] |= inside_kernel[
                kr0:kr0+(br1-br0), kc0:kc0+(bc1-bc0)
            ]
        affected &= heap_present[ar0:ar1, ac0:ac1]
        for dr, dc in zip(*np.where(affected)):
            r0, c0 = int(ar0 + dr), int(ac0 + dc)
            s = score_active(r0, c0)
            if abs(s - cached_score[r0, c0]) > 1e-9:
                cached_score[r0, c0] = s
                heap_version[r0, c0] += 1
                heapq.heappush(heap, (s, r0, c0, int(heap_version[r0, c0])))

    def maybe_compact_heap() -> None:
        # Old heap entries are invalidated by versions. Keep memory bounded.
        if len(heap) <= max(250_000, 3 * initial_sites):
            return
        heap[:] = [(float(cached_score[r0, c0]), int(r0), int(c0), int(heap_version[r0, c0]))
                   for r0, c0 in zip(*np.where(heap_present))]
        heapq.heapify(heap)

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
                    insert_current_score(rr0, cc0)
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
                        insert_current_score(rr0, cc0)
                        n_reinserted += 1

                deferred_items = []
                deferred_seen.clear()

                print(f"[PHASE] reinserted deferred locations: {n_reinserted:,}")
                
        _, r0, c0, stamp_version = heapq.heappop(heap)
        # Ignore invalid entries from an earlier dynamic-score revision.
        if (not heap_present[r0, c0]) or (stamp_version != heap_version[r0, c0]):
            continue
        heap_present[r0, c0] = False

        if not can_build_here(r0, c0, free_national_phase=free_national_phase):
            if use_state_targets and (not free_national_phase):
                key = (int(r0), int(c0))
                if key not in deferred_seen and nmax[r0, c0] > 0:
                    deferred_items.append(key)
                    deferred_seen.add(key)
            continue

        take_mw = float(power_mw[r0, c0])
        take_gw = take_mw / 1000.0
        take_mwh_per_year = float(annual_mwh_per_turbine[r0, c0])
        take_twh_per_year = take_mwh_per_year / 1e6

        if build_limit_mode == "total_gw" and target_new_build_gw is not None:
            if built_total_gw + take_gw > float(target_new_build_gw) + eps_rel:
                break

        score_now = score_active(r0, c0)
        if abs(score_now - cached_score[r0, c0]) > 1e-9:
            insert_current_score(r0, c0)
            continue

        sid = state[r0, c0]
        acc_cost_now = float(marginal_acceptance_cost(r0, c0))

        # Snapshot affected residential distances before adding the turbine.
        R = int(frondel_max_distance_km)
        old_r0 = max(0, r0 - R)
        old_r1 = min(ny, r0 + R + 1)
        old_c0 = max(0, c0 - R)
        old_c1 = min(nx, c0 + R + 1)
        old_distances = current_nearest_dist[old_r0:old_r1, old_c0:old_c1].copy()

        cost_eur = apply_new_turbine_nearest(
            current_nearest_dist=current_nearest_dist,
            current_loss_frac=current_loss_frac,
            current_loss_eur=current_loss_eur,
            value_arr=value_arr,
            urban_mask=urban_mask,
            residential_mask=residential_mask,
            in_de_mask=in_de_mask,
            turbine_r=r0,
            turbine_c=c0,
            rural_loss_fracs=rural_loss_fracs,
            urban_loss_fracs=urban_loss_fracs,
            dist_kernel=dist_kernel,
            inside_kernel=inside_kernel,
            band_idx_kernel=band_idx_kernel,
            max_distance_km=frondel_max_distance_km,
        )

        if abs(cost_eur - acc_cost_now) > 1e-6 * max(1.0, abs(cost_eur), abs(acc_cost_now)):
            acc_cost_now = float(cost_eur)

        n_new[r0, c0] += 1
        nmax[r0, c0] -= 1
        added_cost[r0, c0] += cost_eur

        built_total_gw += take_gw
        built_twh_total += take_twh_per_year
        if sid:
            built_gw_by_state[sid] = built_gw_by_state.get(sid, 0.0) + take_gw

        steps += 1
        if debug_every_steps and (steps % int(debug_every_steps) == 0):
            phase_name = "free" if free_national_phase else "state_targets"
            print(
                f"[ITER] step={steps:,} phase={phase_name} "
                f"built_total_gw={built_total_gw:.2f} built_total_twh={built_twh_total:.2f} "
                f"heap={len(heap):,} current_loss_stock={float(current_loss_eur.sum()):.3e}"
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
            "take_cost_eur": float(cost_eur),
            "mc_total_at_pick": float(acc_cost_now),
            "score_active": float(score_now),
            "score_base_pv_lcoe": float(score_base(r0, c0)),
            "score_total_pv_lcoe_at_pick": float(
                (float(pv_base_eur_per_turbine[r0, c0]) + float(acc_cost_now)) / max(float(pv_energy_mwh_per_turbine[r0, c0]), 1e-12)
            ),
            "cf": float(cf[r0, c0]),
            "flh_net": float(flh_net_arr[r0, c0]),
            "pv_base_cost_eur_step": float(pv_base_eur_per_turbine[r0, c0]),
            "pv_energy_mwh_step": float(pv_energy_mwh_per_turbine[r0, c0]),
            "built_gw_total_after": float(built_total_gw),
            "built_gw_state_after": float(built_gw_by_state.get(sid, 0.0)) if sid else 0.0,
        })

        refresh_candidates_sharing_changed_homes(r0, c0, old_distances, old_r0, old_c0)
        if nmax[r0, c0] > 0:
            insert_current_score(r0, c0)
        if steps % 512 == 0:
            maybe_compact_heap()

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
        curve["cum_cost_eur"] = curve["take_cost_eur"].cumsum()
        curve["avg_cost_eur_per_mw"] = curve["cum_cost_eur"] / np.maximum(curve["cum_gw"] * 1000.0, 1e-12)
    else:
        curve = pd.DataFrame()

    rep = targets_df.copy()
    rep["target_gw_phaseA"] = rep["add_gw"].astype(float)
    rep["max_gw"] = rep["add_gw"].astype(float) * (1.0 + float(state_upper_slack))
    rep["built_gw"] = rep["state_id"].astype(str).map(lambda s: built_gw_by_state.get(s, 0.0)).astype(float)
    rep["gap_to_target_gw_phaseA"] = (rep["target_gw_phaseA"] - rep["built_gw"]).clip(lower=0.0)

    if use_state_targets:
        miss = rep[rep["gap_to_target_gw_phaseA"] > 1e-9]
        if len(miss) > 0:
            print("[WARN] Some states did NOT fully reach target_gw in phase A before free expansion:")
            print(miss[["state_id", "target_gw_phaseA", "built_gw", "gap_to_target_gw_phaseA"]].to_string(index=False))

    meta = {
        "phase_switch_gw": float(phase_switch_gw) if np.isfinite(phase_switch_gw) else np.nan,
        "phase_switch_twh_per_year": float(phase_switch_twh_per_year) if np.isfinite(phase_switch_twh_per_year) else np.nan,
        "built_total_gw": float(built_total_gw),
        "built_total_twh_per_year": float(built_twh_total),
    }

    return placement, curve, rep, meta


# ============================================================
# Remaining potential with variable turbine size
# ============================================================

def add_remaining_potential_columns_variable_turbine(
    gdf: gpd.GeoDataFrame,
    pot_mw_per_km2: float,
    turbine_power_mw_col: str,
    existing_gw_col: str,
    existing_proxy_mw_per_turbine: float,
    use_existing_stock: bool = True,
) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g["pot_mw"] = g["pot_area_km2"].astype(float) * float(pot_mw_per_km2)

    if use_existing_stock:
        if existing_gw_col in g.columns:
            g["existing_mw"] = g[existing_gw_col].astype(float) * 1000.0
        else:
            g["existing_mw"] = g["n_existing"].astype(float) * float(existing_proxy_mw_per_turbine)
    else:
        g["existing_mw"] = 0.0

    g["remain_mw"] = (g["pot_mw"] - g["existing_mw"]).clip(lower=0.0)

    pw = pd.to_numeric(g[turbine_power_mw_col], errors="coerce").astype(float)
    g["n_new_max"] = np.where(
        pw > 0,
        np.floor(g["remain_mw"] / pw),
        0,
    ).astype(int)
    return g


# ============================================================
# Plot helpers
# ============================================================

def plot_mc_in_de(
    gdf: gpd.GeoDataFrame,
    mc_col: str,
    mask_col: str = "in_de",
    title: str = "",
    log_scale: bool = True,
    clip_quantiles: Tuple[float, float] = (0.02, 0.98),
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    mc_arr, _extent, _xmin, _ymin = gdf_to_array(gdf, mc_col)
    mask_arr, _extent2, _xmin2, _ymin2 = gdf_to_array(gdf, mask_col)

    mask = np.isfinite(mask_arr) & (mask_arr.astype(float) > 0.5)
    data = np.where(mask, mc_arr.astype(float), np.nan)

    flat = data[np.isfinite(data)]
    if flat.size == 0:
        print("[PLOT] no finite values inside Germany")
        return

    if log_scale:
        flat_pos = flat[flat > 0]
        if flat_pos.size == 0:
            print("[PLOT] no positive values inside Germany for log scale")
            return
        qlo, qhi = clip_quantiles
        vmin = float(np.quantile(flat_pos, qlo))
        vmax = float(np.quantile(flat_pos, qhi))
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        plot_data = np.where(data > 0, data, np.nan)
    else:
        qlo, qhi = clip_quantiles
        vmin = float(np.quantile(flat, qlo))
        vmax = float(np.quantile(flat, qhi))
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        plot_data = data

    plt.figure(figsize=figsize)
    plt.imshow(plot_data, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    plt.title(title or mc_col)
    plt.axis("off")
    plt.colorbar(label="EUR")
    plt.tight_layout()
    plt.show()


def plot_frondel_curves(
    curve_cum: pd.DataFrame,
    curve_step: pd.DataFrame,
    phase_switch_gw: Optional[float] = None,
    phase_switch_twh_per_year: Optional[float] = None,
) -> None:
    if curve_cum.empty or curve_step.empty:
        print("[PLOT] curve is empty")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve_cum["cum_gw_total_system"], curve_cum["lcoe_base_pv_cum_eur_per_mwh"], label="Base")
    ax.plot(curve_cum["cum_gw_total_system"], curve_cum["lcoe_total_pv_cum_eur_per_mwh"], label="Total")
    if phase_switch_gw is not None and np.isfinite(phase_switch_gw):
        ax.axvline(phase_switch_gw, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative buildout incl. existing capacity (GW)")
    ax.set_ylabel("EUR / MWh (PV-consistent)")
    ax.set_title("Cumulative system cost in €/MWh — Frondel nearest")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve_step["cum_gw_total_system"], curve_step["lcoe_base_pv_step_eur_per_mwh"], label="Base")
    ax.plot(curve_step["cum_gw_total_system"], curve_step["lcoe_total_pv_step_eur_per_mwh"], label="Total")
    if phase_switch_gw is not None and np.isfinite(phase_switch_gw):
        ax.axvline(phase_switch_gw, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative buildout incl. existing capacity (GW)")
    ax.set_ylabel("EUR / MWh (PV-consistent)")
    ax.set_title("Marginal system cost in €/MWh — Frondel nearest")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve_cum["cum_twh_per_year_total_system"], curve_cum["lcoe_base_pv_cum_eur_per_mwh"], label="Base")
    ax.plot(curve_cum["cum_twh_per_year_total_system"], curve_cum["lcoe_total_pv_cum_eur_per_mwh"], label="Total")
    if phase_switch_twh_per_year is not None and np.isfinite(phase_switch_twh_per_year):
        ax.axvline(phase_switch_twh_per_year, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative annual generation incl. existing generation (TWh/year)")
    ax.set_ylabel("EUR / MWh (PV-consistent)")
    ax.set_title("Cumulative system cost in €/MWh — Frondel nearest")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve_step["cum_twh_per_year_total_system"], curve_step["lcoe_base_pv_step_eur_per_mwh"], label="Base")
    ax.plot(curve_step["cum_twh_per_year_total_system"], curve_step["lcoe_total_pv_step_eur_per_mwh"], label="Total")
    if phase_switch_twh_per_year is not None and np.isfinite(phase_switch_twh_per_year):
        ax.axvline(phase_switch_twh_per_year, linestyle="--", alpha=0.7, label="Free expansion starts")
    ax.set_xlabel("Cumulative annual generation incl. existing generation (TWh/year)")
    ax.set_ylabel("EUR / MWh (PV-consistent)")
    ax.set_title("Marginal system cost in €/MWh — Frondel nearest")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = FrondelConfig()
    shared = FrondelSharedConfig()

    if cfg.input_grid_de_era5_parquet is None:
        cfg.input_grid_de_era5_parquet = cfg.root / "Output" / "grid_de_era5.parquet"

    run_frondel(cfg, shared)