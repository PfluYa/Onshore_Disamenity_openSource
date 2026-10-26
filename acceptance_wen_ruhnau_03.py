# -*- coding: utf-8 -*-
"""Onshore-wind disamenity valuation based on Wen et al. / Ruhnau.

Supports ``allocation_mode='base_only'`` and ``'total_lcoe'`` and evaluates
both the lower and upper WTP scenarios. Common model assumptions (capacity
density, existing-stock representation, state targets and slack, lifetime,
WACC and O&M cost share) are supplied via ``run_wen_ruhnau(cfg, shared)``.
Only method-specific settings are maintained in this module.
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
# Module config only
# ============================================================

@dataclass
class WenRuhnauConfig:
    root: Path = Path(__file__).resolve().parent

    # input / output
    input_grid_de_era5_parquet: Optional[Path] = None
    output_dir: Path = Path(__file__).resolve().parent / 'Output' / 'placing_wen_ruhnau'

    # optional cleanup
    drop_unused_columns_at_start: bool = True

    # central mode, typically set from main.py
    allocation_mode: str = "base_only"   # "total_lcoe" or "base_only"

    # ids / columns
    state_id_col: str = "state_id"
    existing_gw_col: str = "existing_gw"
    population_col: str = "pop_2022"

    # Wen / Ruhnau parameters stay in module
    wen_scale_lower: float = 1.0
    wen_scale_upper: float = 10.0
    wen_coeff: float = 3.6
    wen_zero_distance_km: float = 4.0
    min_distance_km: float = 0.2

    # ERA5-based engineering / energy columns
    flh_col: str = "net_flh_era5"
    existing_flh_col: str = "net_flh_existing_era5"
    gross_flh_col: str = "gross_flh_era5"
    cf_col: str = "cf_era5"
    turbine_power_mw_col: str = "rated_power_mw_era5"
    turbine_power_kw_col: str = "rated_power_kw_era5"
    turbine_cost_col: str = "costs_era5"
    turbine_type_col: str = "turbine_type_era5"

    # numerics / fallback fills stay in module
    eps_rel: float = 1e-12
    population_fill: float = 0.0
    flh_fill: float = 0.0
    cf_fill: float = 0.0
    turbine_power_mw_fill: float = 0.0
    turbine_power_kw_fill: float = 0.0
    turbine_cost_fill: float = 0.0

    # plotting
    plot_wtp_kernel: bool = True
    plot_curve_lcoe: bool = True
    debug_every_steps: int = 1000

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
        "pop_2022",
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
class WenRuhnauSharedConfig:
    pot_mw_per_km2: float = 25.0
    existing_proxy_mw_per_turbine: float = 2.5

    use_state_targets: bool = True
    state_lower_slack: float = 0.05
    state_upper_slack: float = 0.05

    lifetime_yrs: int = 20
    wacc: float = 0.06
    fixed_om_share_of_capex_per_year: float = 0.025


# ============================================================
# Helper
# ============================================================

def build_output_suffix(use_state_targets: bool, allocation_mode: str) -> str:
    suffix_core = "with_states" if use_state_targets else "national"
    return f"{suffix_core}_{allocation_mode}"


# ============================================================
# Public runner
# ============================================================

def run_wen_ruhnau(cfg: WenRuhnauConfig, shared: Any) -> Dict[str, Any]:
    validate_module_config(cfg)
    validate_shared_config(shared)

    if cfg.input_grid_de_era5_parquet is None:
        raise ValueError("cfg.input_grid_de_era5_parquet must be set.")

    print("[RUN] Greedy placement")
    print(f"[MODE] allocation_mode = {cfg.allocation_mode}")
    print("[MODEL] Wen / Ruhnau: WTP(d) = scale * 3.6 * ln(4 / d), 0 at 4 km")
    print(f"[MODEL] lower scale = {cfg.wen_scale_lower:.3f}")
    print(f"[MODEL] upper scale = {cfg.wen_scale_upper:.3f}")

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

    gdf["population"] = (
        pd.to_numeric(gdf[cfg.population_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cfg.population_fill)
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

    if "n_new_max" not in gdf.columns:
        gdf = add_remaining_potential_columns_variable_turbine(
            gdf=gdf,
            pot_mw_per_km2=shared.pot_mw_per_km2,
            turbine_power_mw_col="turbine_power_mw",
            existing_gw_col=cfg.existing_gw_col,
            existing_proxy_mw_per_turbine=shared.existing_proxy_mw_per_turbine,
        )

    gdf["existing_gw_cell"] = (
        pd.to_numeric(gdf[cfg.existing_gw_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    existing_total_gw = float(gdf["existing_gw_cell"].sum())
    gdf["existing_mwh_per_year_cell"] = gdf["existing_gw_cell"] * 1000.0 * gdf["flh_existing"]
    gdf["existing_twh_per_year_cell"] = gdf["existing_mwh_per_year_cell"] / 1e6
    existing_total_twh_per_year = float(gdf["existing_twh_per_year_cell"].sum())

    print(f"[EXISTING] Total existing GW: {existing_total_gw:.2f}")
    print(f"[EXISTING] Total existing TWh/year: {existing_total_twh_per_year:.2f}")

    targets = build_state_targets_b2045()
    targets_df = build_state_additions_df(gdf, targets, cfg.state_id_col, cfg.existing_gw_col)
    print("[STATE] add_gw by state:")
    print(targets_df[["state_id", "target_gw", "existing_gw", "add_gw"]].to_string(index=False))
    print(f"[TARGET] Total additional GW (sum add_gw): {float(targets_df['add_gw'].sum()):.2f}")

    scenario_results: Dict[str, Dict[str, Any]] = {}

    for scenario_name, scale in {
        "lower": cfg.wen_scale_lower,
        "upper": cfg.wen_scale_upper,
    }.items():
        print("\n" + "=" * 70)
        print(f"[SCENARIO] {scenario_name.upper()} | scale = {scale:.3f}")
        print("=" * 70)

        gdf_s = gdf.copy()

        wen_kernel = build_wen_kernel(cfg, scale=scale)

        if cfg.plot_wtp_kernel:
            plot_kernel(
                wen_kernel,
                title=f"Wen / Ruhnau WTP kernel [{scenario_name}] [EUR/person/turbine/year]"
            )

        annual_acc_df = build_annual_acceptance_cost_df(
            gdf=gdf_s,
            population_col="population",
            kernel=wen_kernel,
            radius_km=int(cfg.wen_zero_distance_km),
        )
        gdf_s = gdf_s.merge(annual_acc_df, on="idm", how="left")
        gdf_s["annual_acceptance_cost_eur_per_turbine"] = (
            gdf_s["annual_acceptance_cost_eur_per_turbine"].fillna(0.0)
        )

        if cfg.allocation_mode == "total_lcoe":
            placement, curve_raw, state_report = allocate_global_iterative_min_total_pv_lcoe_wen_ruhnau(
                gdf=gdf_s,
                targets_df=targets_df,
                state_id_col=cfg.state_id_col,
                n_new_max_col="n_new_max",
                annual_acceptance_cost_col="annual_acceptance_cost_eur_per_turbine",
                flh_col="flh_net",
                cf_col="cf",
                turbine_power_mw_col="turbine_power_mw",
                turbine_power_kw_col="turbine_power_kw",
                turbine_cost_col="turbine_cost_input",
                enforce_state_constraints=shared.use_state_targets,
                state_lower_slack=shared.state_lower_slack,
                state_upper_slack=shared.state_upper_slack,
                eps_rel=cfg.eps_rel,
                in_de_col="in_de" if "in_de" in gdf_s.columns else None,
                debug_every_steps=cfg.debug_every_steps,
                lifetime_yrs=shared.lifetime_yrs,
                wacc=shared.wacc,
                fixed_om_share_of_capex_per_year=shared.fixed_om_share_of_capex_per_year,
            )
        elif cfg.allocation_mode == "base_only":
            placement, curve_raw, state_report = allocate_global_iterative_min_base_pv_lcoe_with_expost_acceptance(
                gdf=gdf_s,
                targets_df=targets_df,
                state_id_col=cfg.state_id_col,
                n_new_max_col="n_new_max",
                annual_acceptance_cost_col="annual_acceptance_cost_eur_per_turbine",
                flh_col="flh_net",
                cf_col="cf",
                turbine_power_mw_col="turbine_power_mw",
                turbine_power_kw_col="turbine_power_kw",
                turbine_cost_col="turbine_cost_input",
                enforce_state_constraints=shared.use_state_targets,
                state_lower_slack=shared.state_lower_slack,
                state_upper_slack=shared.state_upper_slack,
                eps_rel=cfg.eps_rel,
                in_de_col="in_de" if "in_de" in gdf_s.columns else None,
                debug_every_steps=cfg.debug_every_steps,
                lifetime_yrs=shared.lifetime_yrs,
                wacc=shared.wacc,
                fixed_om_share_of_capex_per_year=shared.fixed_om_share_of_capex_per_year,
            )
        else:
            raise ValueError("allocation_mode must be 'total_lcoe' or 'base_only'")

        print("[STATE-REPORT] built vs corridor:")
        print(
            state_report[
                ["state_id", "add_gw", "min_gw", "max_gw", "built_gw", "gap_to_min_gw"]
            ].to_string(index=False)
        )

        gdf_s = gdf_s.merge(placement, on="idm", how="left")
        gdf_s["n_new"] = gdf_s["n_new"].fillna(0).astype(int)
        gdf_s["added_gw"] = gdf_s["added_gw"].fillna(0.0)
        gdf_s["added_cost_eur"] = gdf_s["added_cost_eur"].fillna(0.0)

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

        scenario_results[scenario_name] = {
            "scale": scale,
            "gdf": gdf_s,
            "curve_raw": curve_raw,
            "curve_cum": curve_cum,
            "curve_step": curve_step,
            "curve_out": curve_out,
            "state_report": state_report,
        }

    curve_cum_lower = scenario_results["lower"]["curve_cum"]
    curve_cum_upper = scenario_results["upper"]["curve_cum"]
    curve_step_lower = scenario_results["lower"]["curve_step"]
    curve_step_upper = scenario_results["upper"]["curve_step"]

    if cfg.plot_curve_lcoe and not curve_step_lower.empty and not curve_step_upper.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            curve_step_lower["cum_gw_total_system"],
            curve_step_lower["lcoe_base_pv_step_eur_per_mwh"],
            label="Base",
            linewidth=2.0,
        )
        ax.plot(
            curve_step_lower["cum_gw_total_system"],
            curve_step_lower["lcoe_total_pv_step_eur_per_mwh"],
            label="Total — lower",
            linewidth=2.0,
            linestyle="--",
        )
        ax.plot(
            curve_step_upper["cum_gw_total_system"],
            curve_step_upper["lcoe_total_pv_step_eur_per_mwh"],
            label="Total — upper",
            linewidth=2.0,
            linestyle="-",
        )
        ax.set_xlabel("Cumulative buildout incl. existing capacity (GW)")
        ax.set_ylabel("EUR / MWh (PV-consistent)")
        ax.set_title(f"Marginal system cost in €/MWh — Wen/Ruhnau lower vs upper [{cfg.allocation_mode}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    if cfg.plot_curve_lcoe and not curve_cum_lower.empty and not curve_cum_upper.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            curve_cum_lower["cum_gw_total_system"],
            curve_cum_lower["lcoe_base_pv_cum_eur_per_mwh"],
            label="Base",
            linewidth=2.0,
        )
        ax.plot(
            curve_cum_lower["cum_gw_total_system"],
            curve_cum_lower["lcoe_total_pv_cum_eur_per_mwh"],
            label="Total — lower",
            linewidth=2.0,
            linestyle="--",
        )
        ax.plot(
            curve_cum_upper["cum_gw_total_system"],
            curve_cum_upper["lcoe_total_pv_cum_eur_per_mwh"],
            label="Total — upper",
            linewidth=2.0,
            linestyle="-",
        )
        ax.set_xlabel("Cumulative buildout incl. existing capacity (GW)")
        ax.set_ylabel("EUR / MWh (PV-consistent)")
        ax.set_title(f"Cumulative system cost in €/MWh — Wen/Ruhnau lower vs upper [{cfg.allocation_mode}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = build_output_suffix(shared.use_state_targets, cfg.allocation_mode)

    for scenario_name, res in scenario_results.items():
        out_cells = cfg.output_dir / f"placement_cells_iterative_wen_ruhnau_{scenario_name}_{suffix}.parquet"
        out_curve = cfg.output_dir / f"cost_curve_iterative_wen_ruhnau_{scenario_name}_{suffix}.parquet"

        print(f"[SAVE] cells ({scenario_name}) -> {out_cells}")
        res["gdf"].to_parquet(out_cells, index=False)

        print(f"[SAVE] curve ({scenario_name}) -> {out_curve}")
        res["curve_out"].to_parquet(out_curve, index=False)

    return {
        "placement_lower": scenario_results["lower"]["gdf"],
        "placement_upper": scenario_results["upper"]["gdf"],
        "curve_lower": scenario_results["lower"]["curve_out"],
        "curve_upper": scenario_results["upper"]["curve_out"],
    }


# ============================================================
# Validation
# ============================================================

def validate_module_config(cfg: WenRuhnauConfig) -> None:
    if cfg.allocation_mode not in {"base_only", "total_lcoe"}:
        raise ValueError("allocation_mode must be 'base_only' or 'total_lcoe'")


def validate_shared_config(shared: Any) -> None:
    required_attrs = [
        "pot_mw_per_km2",
        "existing_proxy_mw_per_turbine",
        "use_state_targets",
        "state_lower_slack",
        "state_upper_slack",
        "lifetime_yrs",
        "wacc",
        "fixed_om_share_of_capex_per_year",
    ]
    missing = [a for a in required_attrs if not hasattr(shared, a)]
    if missing:
        raise ValueError(f"shared config missing attributes: {missing}")

    if not (0.0 <= float(shared.state_lower_slack) < 1.0):
        raise ValueError("state_lower_slack must be in [0,1)")

    if float(shared.state_upper_slack) < 0.0:
        raise ValueError("state_upper_slack must be >= 0")

    if int(shared.lifetime_yrs) <= 0:
        raise ValueError("lifetime_yrs must be > 0")


# ============================================================
# Wen / Ruhnau functions
# ============================================================

def wen_ruhnau_wtp(distance_km: np.ndarray, cfg: WenRuhnauConfig, scale: float) -> np.ndarray:
    d_raw = np.asarray(distance_km, dtype=float)
    d = np.maximum(d_raw, float(cfg.min_distance_km))

    wtp = float(scale) * float(cfg.wen_coeff) * np.log(float(cfg.wen_zero_distance_km) / d)
    wtp = np.where(d_raw < float(cfg.wen_zero_distance_km), wtp, 0.0)
    wtp = np.maximum(wtp, 0.0)
    return wtp


def build_wen_kernel(cfg: WenRuhnauConfig, scale: float) -> np.ndarray:
    R = int(cfg.wen_zero_distance_km)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    dist = np.sqrt(xx**2 + yy**2).astype(float)

    kernel = np.zeros((2 * R + 1, 2 * R + 1), dtype=float)
    inside = dist < float(cfg.wen_zero_distance_km)
    kernel[inside] = wen_ruhnau_wtp(dist[inside], cfg, scale=scale)
    return kernel


# ============================================================
# PV metrics
# ============================================================

def add_lcoe_metrics_pv_cumulative_variable(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    if curve.empty:
        return curve

    if "step" not in curve.columns:
        curve["step"] = np.arange(1, len(curve) + 1)

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

    if "step" not in curve.columns:
        curve["step"] = np.arange(1, len(curve) + 1)

    if "cum_gw" not in curve.columns:
        curve["cum_gw"] = curve["take_gw"].cumsum()

    if "cum_twh_per_year" not in curve.columns:
        curve["cum_twh_per_year"] = curve["take_twh_per_year"].cumsum()

    curve["lcoe_base_pv_step_eur_per_mwh"] = curve["pv_base_cost_eur_step"] / np.maximum(curve["pv_energy_mwh_step"], 1e-12)
    curve["lcoe_acc_pv_step_eur_per_mwh"] = curve["take_cost_eur"] / np.maximum(curve["pv_energy_mwh_step"], 1e-12)
    curve["lcoe_total_pv_step_eur_per_mwh"] = (curve["pv_base_cost_eur_step"] + curve["take_cost_eur"]) / np.maximum(curve["pv_energy_mwh_step"], 1e-12)

    return curve


# ============================================================
# State targets
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
# Arrays / grid helpers
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


def fft_conv_full(arr: np.ndarray, kernel_small: np.ndarray, R: int) -> np.ndarray:
    ny, nx = arr.shape
    ks = kernel_small.shape[0]

    Ap = np.zeros((ny + 2 * R, nx + 2 * R), dtype=np.float64)
    Ap[R:R + ny, R:R + nx] = arr.astype(np.float64)

    Kp = np.zeros_like(Ap)
    k0 = np.fft.ifftshift(kernel_small.astype(np.float64))
    Kp[:ks, :ks] = k0

    conv = np.fft.ifft2(np.fft.fft2(Ap) * np.fft.fft2(Kp)).real
    return conv[R:R + ny, R:R + nx]


# ============================================================
# Acceptance cost map
# ============================================================

def build_annual_acceptance_cost_df(
    gdf: gpd.GeoDataFrame,
    population_col: str,
    kernel: np.ndarray,
    radius_km: int,
) -> pd.DataFrame:
    pop_arr, xmin, ymin = gdf_to_array_numeric(gdf, value_col=population_col)
    pop_arr = np.nan_to_num(pop_arr, nan=0.0).astype(np.float64)

    annual_acc = fft_conv_full(pop_arr, kernel, R=radius_km)

    ny, nx = annual_acc.shape
    rr, cc = np.meshgrid(np.arange(ny, dtype=int), np.arange(nx, dtype=int), indexing="ij")
    x_km = (xmin + cc).astype(int)
    y_km = (ymin + rr).astype(int)

    out = pd.DataFrame({
        "idm": (x_km.astype(str) + "_" + y_km.astype(str)).ravel(),
        "annual_acceptance_cost_eur_per_turbine": annual_acc.ravel().astype(float),
    })
    return out


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
    fixed_om_share_of_capex_per_year: float,
    lifetime_yrs: int,
    wacc: float,
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
        "pv_factor": np.full_like(power_mw, pv_factor, dtype=float),
    }


# ============================================================
# Allocation
# ============================================================

def allocate_global_iterative_min_total_pv_lcoe_wen_ruhnau(
    gdf: gpd.GeoDataFrame,
    targets_df: pd.DataFrame,
    *,
    state_id_col: str,
    n_new_max_col: str,
    annual_acceptance_cost_col: str,
    flh_col: str,
    cf_col: str,
    turbine_power_mw_col: str,
    turbine_power_kw_col: str,
    turbine_cost_col: str,
    enforce_state_constraints: bool,
    state_lower_slack: float = 0.0,
    state_upper_slack: float = 0.0,
    eps_rel: float = 1e-12,
    in_de_col: Optional[str] = "in_de",
    debug_every_steps: int = 0,
    lifetime_yrs: int = 20,
    wacc: float = 0.06,
    fixed_om_share_of_capex_per_year: float = 0.025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    idm_to_rc, rc_to_idm_flat, _xmin, _ymin, ny, nx = build_index_maps(gdf)

    cf, _, _ = gdf_to_array_numeric(gdf, value_col=cf_col)
    cf = np.nan_to_num(cf, nan=0.0).astype(np.float64)

    nmax, _, _ = gdf_to_array_numeric(gdf, value_col=n_new_max_col)
    nmax = np.nan_to_num(nmax, nan=0.0)
    nmax = np.floor(nmax).astype(int)

    annual_acc, _, _ = gdf_to_array_numeric(gdf, value_col=annual_acceptance_cost_col)
    annual_acc = np.nan_to_num(annual_acc, nan=0.0).astype(np.float64)

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
    pv_factor_arr = eng["pv_factor"]
    pv_base_eur_per_turbine = eng["pv_base_eur_per_turbine"]
    pv_energy_mwh_per_turbine = eng["pv_energy_mwh_per_turbine"]

    pv_acceptance_eur_per_turbine = annual_acc * pv_factor_arr

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

    min_gw_by_state: Dict[str, float] = {}
    max_gw_by_state: Dict[str, float] = {}
    built_gw_by_state: Dict[str, float] = {}

    for _, row in targets_df.iterrows():
        sid = str(row["state_id"])
        add_gw = float(row["add_gw"])
        if add_gw <= 0:
            min_gw_by_state[sid] = 0.0
            max_gw_by_state[sid] = 0.0
        else:
            min_gw_by_state[sid] = (1.0 - float(state_lower_slack)) * add_gw
            max_gw_by_state[sid] = (1.0 + float(state_upper_slack)) * add_gw
        built_gw_by_state[sid] = 0.0

    national_max_gw = float(targets_df["add_gw"].sum()) * (1.0 + float(state_upper_slack))
    built_total_gw = 0.0
    free_national_phase = not enforce_state_constraints
    phase_switch_gw = np.nan
    phase_switch_twh_per_year = np.nan    

    def state_allows_more(sid: str) -> bool:
        if not enforce_state_constraints:
            return True
        return built_gw_by_state.get(sid, 0.0) + 1e-12 < max_gw_by_state.get(sid, 0.0)
    def all_state_min_targets_met() -> bool:
        if not enforce_state_constraints:
            return True
        for sid, min_gw in min_gw_by_state.items():
            if built_gw_by_state.get(sid, 0.0) + eps_rel < min_gw:
                return False
        return True    

    def marginal_total_pv_lcoe_score(r0: int, c0: int) -> float:
        pw = float(power_mw[r0, c0])
        pv_base = float(pv_base_eur_per_turbine[r0, c0])
        pv_acc = float(pv_acceptance_eur_per_turbine[r0, c0])
        pv_energy = float(pv_energy_mwh_per_turbine[r0, c0])

        if pw <= 0.0 or pv_base <= 0.0 or pv_energy <= 0.0:
            return float("inf")

        return float((pv_base + pv_acc) / pv_energy)

    n_new = np.zeros((ny, nx), dtype=int)
    added_cost = np.zeros((ny, nx), dtype=np.float64)
    curve_rows: List[Dict[str, float]] = []

    heap: List[Tuple[float, int, int]] = []
    rr, cc = np.where(in_de_mask & (nmax > 0) & (power_mw > 0) & (pv_energy_mwh_per_turbine > 0))
    for r0, c0 in zip(rr, cc):
        heap.append((marginal_total_pv_lcoe_score(int(r0), int(c0)), int(r0), int(c0)))
    heapq.heapify(heap)

    deferred_items: List[Tuple[int, int]] = []
    deferred_seen = set()

    steps = 0
    while heap or (enforce_state_constraints and (not free_national_phase) and deferred_items):
        if not heap and enforce_state_constraints and (not free_national_phase) and deferred_items:
           free_national_phase = True
           phase_switch_gw = built_total_gw

           print(
               f"[PHASE] heap exhausted in state phase; switching to free national expansion at "
               f"{built_total_gw:.2f} GW"
           )

           n_reinserted = 0
           for rr0, cc0 in deferred_items:
               if (
                   nmax[rr0, cc0] > 0
                   and in_de_mask[rr0, cc0]
                   and power_mw[rr0, cc0] > 0
                   and pv_energy_mwh_per_turbine[rr0, cc0] > 0
               ):
                   s = marginal_total_pv_lcoe_score(rr0, cc0)
                   heapq.heappush(heap, (s, rr0, cc0))
                   n_reinserted += 1

           deferred_items = []
           deferred_seen.clear()

           print(f"[PHASE] reinserted deferred locations: {n_reinserted:,}")

        if enforce_state_constraints and (not free_national_phase) and all_state_min_targets_met():
           free_national_phase = True
           phase_switch_gw = built_total_gw

           print(f"[PHASE] switching to free national expansion at {built_total_gw:.2f} GW")

           n_reinserted = 0
           for rr0, cc0 in deferred_items:
               if (
                   nmax[rr0, cc0] > 0
                   and in_de_mask[rr0, cc0]
                   and power_mw[rr0, cc0] > 0
                   and pv_energy_mwh_per_turbine[rr0, cc0] > 0
               ):
                   s = marginal_total_pv_lcoe_score(rr0, cc0)
                   heapq.heappush(heap, (s, rr0, cc0))
                   n_reinserted += 1

           deferred_items = []
           deferred_seen.clear()

           print(f"[PHASE] reinserted deferred locations: {n_reinserted:,}")
           
        score_now, r0, c0 = heapq.heappop(heap)

        if (not in_de_mask[r0, c0]) or (nmax[r0, c0] <= 0):
            continue

        sid = state[r0, c0]
        take_gw = float(power_mw[r0, c0]) / 1000.0
        if take_gw <= 0:
            continue

        if (not enforce_state_constraints) and (built_total_gw + take_gw > national_max_gw + 1e-12):
            break

        if enforce_state_constraints and (not free_national_phase):
            if (not sid) or (not state_allows_more(sid)):
                key = (int(r0), int(c0))
                if key not in deferred_seen and nmax[r0, c0] > 0:
                    deferred_items.append(key)
                    deferred_seen.add(key)
                continue

            gw_left_state = max_gw_by_state[sid] - built_gw_by_state[sid]
            if gw_left_state + 1e-12 < take_gw:
                key = (int(r0), int(c0))
                if key not in deferred_seen and nmax[r0, c0] > 0:
                    deferred_items.append(key)
                    deferred_seen.add(key)
                continue

        pv_acc = float(pv_acceptance_eur_per_turbine[r0, c0])

        n_new[r0, c0] += 1
        nmax[r0, c0] -= 1
        added_cost[r0, c0] += pv_acc

        built_total_gw += take_gw
        if sid:
            built_gw_by_state[sid] = built_gw_by_state.get(sid, 0.0) + take_gw

        steps += 1
        if debug_every_steps and (steps % int(debug_every_steps) == 0):
            print(f"[ITER] step={steps:,} built_total_gw={built_total_gw:.2f} heap={len(heap):,}")

        take_mw = float(power_mw[r0, c0])
        take_mwh_per_year = take_mw * float(flh_net_arr[r0, c0])
        take_twh_per_year = take_mwh_per_year / 1e6

        curve_rows.append({
            "state_id": sid,
            "idm": rc_to_idm_flat[r0 * nx + c0],
            "take_turbines": 1,
            "take_mw": take_mw,
            "take_gw": take_gw,
            "take_mwh_per_year": take_mwh_per_year,
            "take_twh_per_year": take_twh_per_year,
            "take_cost_eur": pv_acc,
            "score_total_pv_lcoe": float(score_now),
            "annual_acceptance_cost_eur": float(annual_acc[r0, c0]),
            "cf": float(cf[r0, c0]),
            "flh_net": float(flh_net_arr[r0, c0]),
            "pv_base_cost_eur_step": float(pv_base_eur_per_turbine[r0, c0]),
            "pv_energy_mwh_step": float(pv_energy_mwh_per_turbine[r0, c0]),
        })

        if nmax[r0, c0] > 0:
            if (not enforce_state_constraints) or free_national_phase or (sid and state_allows_more(sid)):
                heapq.heappush(heap, (score_now, r0, c0))
            elif enforce_state_constraints and (not free_national_phase):
                key = (int(r0), int(c0))
                if key not in deferred_seen:
                    deferred_items.append(key)
                    deferred_seen.add(key)

    placement = pd.DataFrame({
        "idm": np.array(rc_to_idm_flat, dtype=object),
        "n_new": n_new.ravel().astype(int),
        "added_gw": np.where(power_mw.ravel() > 0, n_new.ravel().astype(float) * power_mw.ravel() / 1000.0, 0.0),
        "added_cost_eur": added_cost.ravel().astype(float),
    })
    placement = placement[placement["idm"] != ""].reset_index(drop=True)

    curve = pd.DataFrame(curve_rows) if curve_rows else pd.DataFrame()

    rep = targets_df.copy()
    rep["min_gw"] = rep["add_gw"].astype(float) * (1.0 - float(state_lower_slack))
    rep["max_gw"] = rep["add_gw"].astype(float) * (1.0 + float(state_upper_slack))
    rep["built_gw"] = rep["state_id"].astype(str).map(lambda s: built_gw_by_state.get(s, 0.0)).astype(float)
    rep["gap_to_min_gw"] = (rep["min_gw"] - rep["built_gw"]).clip(lower=0.0)

    return placement, curve, rep


def allocate_global_iterative_min_base_pv_lcoe_with_expost_acceptance(
    gdf: gpd.GeoDataFrame,
    targets_df: pd.DataFrame,
    *,
    state_id_col: str,
    n_new_max_col: str,
    annual_acceptance_cost_col: str,
    flh_col: str,
    cf_col: str,
    turbine_power_mw_col: str,
    turbine_power_kw_col: str,
    turbine_cost_col: str,
    enforce_state_constraints: bool,
    state_lower_slack: float = 0.0,
    state_upper_slack: float = 0.0,
    eps_rel: float = 1e-12,
    in_de_col: Optional[str] = "in_de",
    debug_every_steps: int = 0,
    lifetime_yrs: int = 20,
    wacc: float = 0.06,
    fixed_om_share_of_capex_per_year: float = 0.025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    idm_to_rc, rc_to_idm_flat, _xmin, _ymin, ny, nx = build_index_maps(gdf)

    cf, _, _ = gdf_to_array_numeric(gdf, value_col=cf_col)
    cf = np.nan_to_num(cf, nan=0.0).astype(np.float64)

    nmax, _, _ = gdf_to_array_numeric(gdf, value_col=n_new_max_col)
    nmax = np.nan_to_num(nmax, nan=0.0)
    nmax = np.floor(nmax).astype(int)

    annual_acc, _, _ = gdf_to_array_numeric(gdf, value_col=annual_acceptance_cost_col)
    annual_acc = np.nan_to_num(annual_acc, nan=0.0).astype(np.float64)

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
    pv_factor_arr = eng["pv_factor"]
    pv_base_eur_per_turbine = eng["pv_base_eur_per_turbine"]
    pv_energy_mwh_per_turbine = eng["pv_energy_mwh_per_turbine"]

    pv_acceptance_eur_per_turbine = annual_acc * pv_factor_arr

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

    min_gw_by_state: Dict[str, float] = {}
    max_gw_by_state: Dict[str, float] = {}
    built_gw_by_state: Dict[str, float] = {}

    for _, row in targets_df.iterrows():
        sid = str(row["state_id"])
        add_gw = float(row["add_gw"])
        if add_gw <= 0:
            min_gw_by_state[sid] = 0.0
            max_gw_by_state[sid] = 0.0
        else:
            min_gw_by_state[sid] = (1.0 - float(state_lower_slack)) * add_gw
            max_gw_by_state[sid] = (1.0 + float(state_upper_slack)) * add_gw
        built_gw_by_state[sid] = 0.0

    national_max_gw = float(targets_df["add_gw"].sum()) * (1.0 + float(state_upper_slack))
    built_total_gw = 0.0
    free_national_phase = not enforce_state_constraints
    phase_switch_gw = np.nan
    phase_switch_twh_per_year = np.nan    

    def state_allows_more(sid: str) -> bool:
        if not enforce_state_constraints:
            return True
        return built_gw_by_state.get(sid, 0.0) + 1e-12 < max_gw_by_state.get(sid, 0.0)
    def all_state_min_targets_met() -> bool:
        if not enforce_state_constraints:
            return True
        for sid, min_gw in min_gw_by_state.items():
            if built_gw_by_state.get(sid, 0.0) + eps_rel < min_gw:
                return False
        return True    

    def marginal_base_pv_lcoe_score(r0: int, c0: int) -> float:
        pw = float(power_mw[r0, c0])
        pv_base = float(pv_base_eur_per_turbine[r0, c0])
        pv_energy = float(pv_energy_mwh_per_turbine[r0, c0])

        if pw <= 0.0 or pv_base <= 0.0 or pv_energy <= 0.0:
            return float("inf")
        return float(pv_base / pv_energy)

    n_new = np.zeros((ny, nx), dtype=int)
    added_cost = np.zeros((ny, nx), dtype=np.float64)
    curve_rows: List[Dict[str, float]] = []

    heap: List[Tuple[float, int, int]] = []
    rr, cc = np.where(in_de_mask & (nmax > 0) & (power_mw > 0) & (pv_energy_mwh_per_turbine > 0))
    for r0, c0 in zip(rr, cc):
        heap.append((marginal_base_pv_lcoe_score(int(r0), int(c0)), int(r0), int(c0)))
    heapq.heapify(heap)

    deferred_items: List[Tuple[int, int]] = []
    deferred_seen = set()

    steps = 0
    while heap or (enforce_state_constraints and (not free_national_phase) and deferred_items):
        if not heap and enforce_state_constraints and (not free_national_phase) and deferred_items:
            free_national_phase = True
            phase_switch_gw = built_total_gw

            print(
                f"[PHASE] heap exhausted in state phase; switching to free national expansion at "
                f"{built_total_gw:.2f} GW"
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

        if enforce_state_constraints and (not free_national_phase) and all_state_min_targets_met():
            free_national_phase = True
            phase_switch_gw = built_total_gw

            print(f"[PHASE] switching to free national expansion at {built_total_gw:.2f} GW")

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

        if (not in_de_mask[r0, c0]) or (nmax[r0, c0] <= 0):
            continue

        sid = state[r0, c0]
        take_gw = float(power_mw[r0, c0]) / 1000.0
        if take_gw <= 0:
            continue

        if (not enforce_state_constraints) and (built_total_gw + take_gw > national_max_gw + 1e-12):
            break

        if enforce_state_constraints and (not free_national_phase):
            if (not sid) or (not state_allows_more(sid)):
                key = (int(r0), int(c0))
                if key not in deferred_seen and nmax[r0, c0] > 0:
                    deferred_items.append(key)
                    deferred_seen.add(key)
                continue

            gw_left_state = max_gw_by_state[sid] - built_gw_by_state[sid]
            if gw_left_state + 1e-12 < take_gw:
                key = (int(r0), int(c0))
                if key not in deferred_seen and nmax[r0, c0] > 0:
                    deferred_items.append(key)
                    deferred_seen.add(key)
                continue

        pv_acc = float(pv_acceptance_eur_per_turbine[r0, c0])
        pv_base = float(pv_base_eur_per_turbine[r0, c0])

        n_new[r0, c0] += 1
        nmax[r0, c0] -= 1
        added_cost[r0, c0] += pv_base

        built_total_gw += take_gw
        if sid:
            built_gw_by_state[sid] = built_gw_by_state.get(sid, 0.0) + take_gw

        steps += 1
        if debug_every_steps and (steps % int(debug_every_steps) == 0):
            print(f"[ITER] step={steps:,} built_total_gw={built_total_gw:.2f} heap={len(heap):,}")

        take_mw = float(power_mw[r0, c0])
        take_mwh_per_year = take_mw * float(flh_net_arr[r0, c0])
        take_twh_per_year = take_mwh_per_year / 1e6

        curve_rows.append({
            "state_id": sid,
            "idm": rc_to_idm_flat[r0 * nx + c0],
            "take_turbines": 1,
            "take_mw": take_mw,
            "take_gw": take_gw,
            "take_mwh_per_year": take_mwh_per_year,
            "take_twh_per_year": take_twh_per_year,
            "take_cost_eur": pv_acc,
            "score_base_pv_lcoe": float(score_now),
            "annual_acceptance_cost_eur": float(annual_acc[r0, c0]),
            "cf": float(cf[r0, c0]),
            "flh_net": float(flh_net_arr[r0, c0]),
            "pv_base_cost_eur_step": pv_base,
            "pv_energy_mwh_step": float(pv_energy_mwh_per_turbine[r0, c0]),
        })

        if nmax[r0, c0] > 0:
            if (not enforce_state_constraints) or free_national_phase or (sid and state_allows_more(sid)):
                heapq.heappush(heap, (score_now, r0, c0))
            elif enforce_state_constraints and (not free_national_phase):
                key = (int(r0), int(c0))
                if key not in deferred_seen:
                    deferred_items.append(key)
                    deferred_seen.add(key)

    placement = pd.DataFrame({
        "idm": np.array(rc_to_idm_flat, dtype=object),
        "n_new": n_new.ravel().astype(int),
        "added_gw": np.where(power_mw.ravel() > 0, n_new.ravel().astype(float) * power_mw.ravel() / 1000.0, 0.0),
        "added_cost_eur": added_cost.ravel().astype(float),
    })
    placement = placement[placement["idm"] != ""].reset_index(drop=True)

    curve = pd.DataFrame(curve_rows) if curve_rows else pd.DataFrame()

    rep = targets_df.copy()
    rep["min_gw"] = rep["add_gw"].astype(float) * (1.0 - float(state_lower_slack))
    rep["max_gw"] = rep["add_gw"].astype(float) * (1.0 + float(state_upper_slack))
    rep["built_gw"] = rep["state_id"].astype(str).map(lambda s: built_gw_by_state.get(s, 0.0)).astype(float)
    rep["gap_to_min_gw"] = (rep["min_gw"] - rep["built_gw"]).clip(lower=0.0)

    return placement, curve, rep


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
    g["n_new_max"] = np.where(pw > 0, np.floor(g["remain_mw"] / pw), 0).astype(int)
    return g


# ============================================================
# Plot helpers
# ============================================================

def plot_kernel(kernel: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(kernel, origin="lower")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = WenRuhnauConfig()
    shared = WenRuhnauSharedConfig()

    if cfg.input_grid_de_era5_parquet is None:
        cfg.input_grid_de_era5_parquet = cfg.root / "Output" / "grid_de_era5.parquet"

    run_wen_ruhnau(cfg, shared)