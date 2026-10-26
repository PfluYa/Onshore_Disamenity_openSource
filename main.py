# -*- coding: utf-8 -*-
"""Unified Onshore Disamenity pipeline — 8 paper scenarios.

Models: Base, WTP-Low, WTP-High, HedPrice-F (Frondel 2019),
HedPrice-H (Heuer & Sommer 2026 A10 conservative), LifeSat, ExclPop, ExclSce.

Default runs the previous four combinations of regional restrictions and
siting rules (base_only / total_lcoe), with 50% state upper slack and
220 GW system target. The paper comparison point is 160 GW.

Edit only the USER SETTINGS section for normal runs.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from grid_builder_01 import GridBuilderConfig, run_grid_builder
from base_lcoe_02 import BaseLCOEConfig, run_base_lcoe
from acceptance_wen_ruhnau_03 import WenRuhnauConfig, run_wen_ruhnau
from acceptance_frondel_04 import FrondelConfig, run_frondel
from acceptance_heuer_09 import HeuerConfig, run_heuer
from acceptance_krekel_zerrahn_06 import KrekelZerrahnConfig, run_krekel_zerrahn
from acceptance_tsani_proxy_07 import TsaniProxyConfig, run_tsani_proxy
from acceptance_tsani_scenicness_08 import TsaniScenicnessConfig, run_tsani_scenicness
from compare_lcoe_05 import CompareLCOEConfig, run_compare_lcoe


# ===========================================================================
# USER SETTINGS: all model-wide assumptions and module switches in one place
# ===========================================================================
ROOT = Path(__file__).resolve().parent
GRID_INPUT = ROOT / "Output_Main" / "grid_de_era5.parquet"
REBUILD_GRID = False       # True ONLY when underlying input data changed.

RUN_ALLOCATION_MODES = ("base_only", "total_lcoe")
RUN_STATE_MODES = (True, False)  # True: federal-state targets; False: national
STATE_UPPER_SLACK = 0.50
STATE_LOWER_SLACK = 0.05
TARGET_TOTAL_GW = 220.0    # length of cost curve incl. existing stock
REPORT_AT_GW = 160.0      # paper evaluation point

# Enable all previous seven scenarios PLUS HedPrice-H.
RUN_MODELS = {
    "Base": True,
    "WTP": True,          # WTP-Low and WTP-High in one module
    "HedPrice-F": True,
    "HedPrice-H": True,
    "LifeSat": True,
    "ExclPop": True,
    "ExclSce": True,
}

# Optional plotting and logging; comparative figures are saved automatically.
SHOW_COMPARISON_PLOTS = False
SAVE_COMPARISON_PLOTS = True
DEBUG_EVERY_STEPS = 1000

# The Frondel and Heuer modules both use the existing-stock configuration.
USE_EXISTING_STOCK = True

# ===========================================================================


@dataclass
class SharedModelConfig:
    pot_mw_per_km2: float = 25.0
    existing_proxy_mw_per_turbine: float = 2.5
    use_state_targets: bool = True
    state_lower_slack: float = STATE_LOWER_SLACK
    state_upper_slack: float = STATE_UPPER_SLACK
    build_limit_mode: str = "total_gw"
    target_total_gw: Optional[float] = TARGET_TOTAL_GW
    lifetime_yrs: int = 20
    wacc: float = 0.06
    fixed_om_share_of_capex_per_year: float = 0.025
    eps_rel: float = 1e-12


def scenario_folder(mode: str, states: bool) -> Path:
    region = "with_states" if states else "national"
    upper = int(round(100 * STATE_UPPER_SLACK))
    return ROOT / f"Output_Main_{region}_{mode}_upperSlack{upper}"


def run_input_builder() -> Path:
    """Use the existing ERA5 input by default; build only on explicit request."""
    if REBUILD_GRID:
        gb = GridBuilderConfig()
        gb.root = ROOT
        gb.out_dir_name = "Output_Main"
        gb.glaes_tif = ROOT / "Input Data" / "de_onshore_wind_holtinger2016_med_100m.tif"
        print("[GRID] Rebuilding input raster and ERA5 inputs...")
        grid_m, grid_de, grid_de_era5 = run_grid_builder(gb)
        del grid_m, grid_de, grid_de_era5
        gc.collect()
    if not GRID_INPUT.exists():
        raise FileNotFoundError(
            f"Missing base grid: {GRID_INPUT}\n"
            "Create it via grid_builder_01.py or change GRID_INPUT in main.py."
        )
    print(f"[GRID] Using common input for ALL scenarios: {GRID_INPUT}")
    return GRID_INPUT


def model_at_target(label: str, curve: pd.DataFrame, target: float,
                    technical_baseline: float) -> dict:
    """Select nearest achieved capacity, preserving existing output semantics."""
    if curve is None or curve.empty:
        return {}
    x = pd.to_numeric(curve["cum_gw_total_system"], errors="coerce")
    valid = np.isfinite(x)
    if not valid.any():
        return {}
    if x.max() < target - 0.25:
        print(f"[WARN] {label}: maximum buildout {x.max():.2f} GW < {target:.2f} GW")
        return {}
    row = curve.loc[(x - target).abs().idxmin()]
    get = lambda k: float(row.get(k, np.nan))
    return {
        "model": label,
        "requested_capacity_gw": target,
        "achieved_capacity_gw": get("cum_gw_total_system"),
        "cumulative_total_lcoe_eur_mwh": get("lcoe_total_pv_cum_eur_per_mwh"),
        "cumulative_technical_lcoe_eur_mwh": get("lcoe_base_pv_cum_eur_per_mwh"),
        "cumulative_disamenity_lcoe_eur_mwh": get("lcoe_acc_pv_cum_eur_per_mwh"),
        "premium_over_base_eur_mwh": get("lcoe_total_pv_cum_eur_per_mwh") - technical_baseline,
        "total_generation_twh_per_year": get("cum_twh_per_year_total_system"),
        "build_steps": len(curve),
    }


def run_scenario(grid: Path, mode: str, states: bool) -> pd.DataFrame:
    """Run selected modules once each; persist unchanged model-specific outputs."""
    if mode not in ("base_only", "total_lcoe"):
        raise ValueError(mode)
    parent = scenario_folder(mode, states)
    parent.mkdir(parents=True, exist_ok=True)
    shared = SharedModelConfig(use_state_targets=states)
    curves = {}
    print("\n" + "=" * 86)
    print(f"[SCENARIO] {parent.name}: {mode}, state targets={states}, "
          f"upper slack={STATE_UPPER_SLACK:.2f}, total target={TARGET_TOTAL_GW:.1f} GW")
    print("=" * 86)

    if RUN_MODELS["Base"]:
        cfg = BaseLCOEConfig(input_grid_de_era5_parquet=grid,
                            output_dir=parent / "placing_base_only_clean_controls")
        cfg.plot_base_lcoe_map = False
        cfg.plot_curves = False
        cfg.debug_every_steps = DEBUG_EVERY_STEPS
        res = run_base_lcoe(cfg, shared)
        curves["Base"] = res["cost_curve"]
        del res
        gc.collect()

    if RUN_MODELS["WTP"]:
        cfg = WenRuhnauConfig(input_grid_de_era5_parquet=grid,
                              output_dir=parent / "placing_wen_ruhnau",
                              allocation_mode=mode)
        cfg.plot_wtp_kernel = False
        cfg.plot_curve_lcoe = False
        cfg.debug_every_steps = DEBUG_EVERY_STEPS
        res = run_wen_ruhnau(cfg, shared)
        curves["WTP-Low"] = res["curve_lower"]
        curves["WTP-High"] = res["curve_upper"]
        del res
        gc.collect()

    if RUN_MODELS["HedPrice-F"]:
        cfg = FrondelConfig(input_grid_de_era5_parquet=grid,
                            output_dir=parent / ("placing_frondel_adjusted_heapfix"
                                                 if mode == "total_lcoe"
                                                 else "placing_frondel_adjusted"),
                            allocation_mode=mode,
                            use_existing_stock=USE_EXISTING_STOCK)
        cfg.root = ROOT
        cfg.cities_excel = ROOT / "Input Data" / "Gemeindekoordinaten AuszugGV2QAktuell.xlsx"
        cfg.plot_mc_map = False
        cfg.plot_curves = False
        cfg.debug_every_steps = DEBUG_EVERY_STEPS
        res = run_frondel(cfg, shared)
        curves["HedPrice-F"] = res["cost_curve"]
        del res
        gc.collect()

    if RUN_MODELS["HedPrice-H"]:
        cfg = HeuerConfig(input_grid_de_era5_parquet=grid,
                          output_dir=parent / "placing_heuer_joint_conservative",
                          specification="joint_conservative",
                          allocation_mode=mode,
                          radius_km=3.0,
                          joint_cell_assignment="intersection_nearest_band",
                          use_existing_stock=USE_EXISTING_STOCK,
                          debug_every_steps=DEBUG_EVERY_STEPS)
        res = run_heuer(cfg, shared)
        curves["HedPrice-H"] = res["cost_curve"]
        del res
        gc.collect()

    if RUN_MODELS["LifeSat"]:
        cfg = KrekelZerrahnConfig(input_grid_de_era5_parquet=grid,
                                  output_dir=parent / "placing_krekel_zerrahn",
                                  allocation_mode=mode)
        cfg.plot_kernel = False
        cfg.plot_curve_lcoe = False
        cfg.debug_every_steps = DEBUG_EVERY_STEPS
        res = run_krekel_zerrahn(cfg, shared)
        curves["LifeSat"] = res["cost_curve"]
        del res
        gc.collect()

    if RUN_MODELS["ExclPop"]:
        cfg = TsaniProxyConfig(input_grid_de_era5_parquet=grid,
                               output_dir=parent / "placing_tsani_proxy",
                               population_density_threshold=1000.0,
                               buffer_radius_km=4.0)
        cfg.plot_exclusion_map = False
        cfg.plot_curves = False
        cfg.debug_every_steps = DEBUG_EVERY_STEPS
        res = run_tsani_proxy(cfg, shared)
        curves["ExclPop"] = res["cost_curve"]
        del res
        gc.collect()

    if RUN_MODELS["ExclSce"]:
        cfg = TsaniScenicnessConfig(input_grid_de_era5_parquet=grid,
                                    output_dir=parent / "placing_tsani_scenicness",
                                    scenicness_threshold=5.0)
        cfg.root = ROOT
        cfg.scenicness_tif = ROOT / "Input Data" / "Scenicness_Labi_Netz" / "Scenicness_roth_export.tif"
        cfg.plot_exclusion_map = False
        cfg.plot_curves = False
        cfg.debug_every_steps = DEBUG_EVERY_STEPS
        res = run_tsani_scenicness(cfg, shared)
        curves["ExclSce"] = res["cost_curve"]
        del res
        gc.collect()

    comp = CompareLCOEConfig(root=ROOT, output_dir=parent / "compare_lcoe",
                             save_plots=SAVE_COMPARISON_PLOTS,
                             show_plots=SHOW_COMPARISON_PLOTS)
    run_compare_lcoe(comp,
        base_curve=curves.get("Base"),
        wen_lower_curve=curves.get("WTP-Low"),
        wen_upper_curve=curves.get("WTP-High"),
        frondel_curve=curves.get("HedPrice-F"),
        heuer_curve=curves.get("HedPrice-H"),
        krekel_zerrahn_curve=curves.get("LifeSat"),
        tsani_proxy_curve=curves.get("ExclPop"),
        tsani_scenicness_curve=curves.get("ExclSce"))

    if "Base" in curves:
        bx = pd.to_numeric(curves["Base"]["cum_gw_total_system"], errors="coerce")
        baseline_row = curves["Base"].loc[(bx - REPORT_AT_GW).abs().idxmin()]
        baseline = float(baseline_row["lcoe_base_pv_cum_eur_per_mwh"])
    else:
        baseline = float("nan")
    report_rows = [model_at_target(name, curve, REPORT_AT_GW, baseline)
                   for name, curve in curves.items()]
    report = pd.DataFrame([row for row in report_rows if row])
    if not report.empty:
        report.insert(0, "allocation_mode", mode)
        report.insert(0, "use_state_targets", states)
        report.to_csv(parent / "comparison_160GW.csv", index=False)
        print("\n[SUMMARY] 160 GW:\n" + report.to_string(index=False))
    return report


def main() -> pd.DataFrame:
    grid = run_input_builder()
    reports = []
    for states in RUN_STATE_MODES:
        for mode in RUN_ALLOCATION_MODES:
            reports.append(run_scenario(grid, mode, states))
    summary = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    if not summary.empty:
        out_dir = ROOT / f"Output_Results_upperSlack{int(round(100 * STATE_UPPER_SLACK))}"
        out_dir.mkdir(parents=True, exist_ok=True)
        outfile = out_dir / "model_comparison_all_scenarios_160GW.csv"
        summary.to_csv(outfile, index=False)
        print(f"\n[ALL DONE] Eight models across four scenario settings -> {outfile}")
    return summary


if __name__ == "__main__":
    results = main()
