# -*- coding: utf-8 -*-
"""Figures 1–7 and paper result tables for the unified Onshore_Disamenity_SOffie repo.

Read-only post-processing. Does NOT rerun turbine allocation.

Source: CleanRepo main.py (Oct 2026); eight paper scenarios:
Base, WTP-Low, WTP-High, HedPrice-F, HedPrice-H, LifeSat, ExclPop, ExclSce.

Main case: federal-state targets, +50% upper corridor, total-LCOE siting.
Allocation comparison: same state targets, base-only vs total-LCOE.
Regional sensitivity: national vs state-specific total-LCOE siting.

Run from Spyder:  %runfile <project>/results_script_paper_all_figures.py --wdir

IMPORTANT (Oct 2026): HedPrice-F extended LCOE must be read from the
validated placing_frondel_adjusted_heapfix output. Standard allocation
(base_only) still uses the original placing_frondel_adjusted output.
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable


# ============================================================================
# USER SETTINGS -- adjust ROOT if necessary; NO model computations are launched
# ============================================================================
ROOT = Path(__file__).resolve().parent
STATE_UPPER_SLACK_PERCENT = 50
TARGET_GW = 160.0
PLOT_MIN_GW = 65.0
PLOT_MAX_GW = 180.0
MARGINAL_ROLLING_TURBINES = 300
USE_EXISTING_STOCK = True

# True = create appendix figure/table comparing state constraints vs national.
MAKE_REGIONAL_SENSITIVITY = True
# True = additional figures: decomposition at 160 GW and focused hedonic comparison.
MAKE_EXTRA_FIGURES = True
# Paper Figures 6–7: maps from 160 GW turbine placement sequences.
MAKE_PAPER_MAPS = True
# Optional administrative border can take longer on full 1-km raster.
DRAW_GERMANY_OUTLINE = True
# Display existing fleet in Figure 7 using subtle grey dots.
SHOW_EXISTING_STOCK_ON_MAPS = True
# The site-level LCOE color scale is clipped only for visualization.
MAP_LCOE_Q_LOWER = 0.01
MAP_LCOE_Q_UPPER = 0.99
# Figure 6 symmetric scale: manual value in MW or None for automatic.
MAP_REALLOCATION_LIMIT_MW = None
# True displays each figure in Spyder; False saves PDFs/PNGs and closes the figure.
SHOW_FIGURES = False
FIGURE_DPI = 300

OUT_DIR = ROOT / f"Output_Results_upperSlack{STATE_UPPER_SLACK_PERCENT}" / "Paper_Figures_FrondelFixed"
# ============================================================================

MODEL_ORDER = (
    "Base", "WTP-Low", "WTP-High", "HedPrice-F", "HedPrice-H",
    "LifeSat", "ExclPop", "ExclSce",
)
MONETARY_MODELS = ("WTP-Low", "WTP-High", "HedPrice-F", "HedPrice-H", "LifeSat")
EXCLUSION_MODELS = ("ExclPop", "ExclSce")

# Group-related hues and distinct line styles; consistent across all figures.
STYLE = {
    "Base":       dict(color="#343943", ls="-",  lw=2.8, zorder=5),
    "WTP-Low":    dict(color="#E9A35A", ls="-",  lw=2.1, zorder=3),
    "WTP-High":   dict(color="#BE6718", ls="--", lw=2.3, zorder=3),
    "HedPrice-F": dict(color="#11649A", ls="-",  lw=2.7, zorder=6),
    "HedPrice-H": dict(color="#199FB6", ls="--", lw=2.7, zorder=6),
    "LifeSat":    dict(color="#8356A6", ls="-",  lw=2.2, zorder=4),
    "ExclPop":    dict(color="#63806D", ls="-.", lw=2.2, zorder=2),
    "ExclSce":    dict(color="#898584", ls=":",  lw=2.6, zorder=2),
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "hatch.linewidth": 0.95,
})

# The patterns are intentionally tied to the cleaned main.py's exact output
# names. We NEVER silently take the first of several files or switch variants.
FOLDERS = {
    "Base": "placing_base_only_clean_controls",
    "WTP-Low": "placing_wen_ruhnau",
    "WTP-High": "placing_wen_ruhnau",
    "HedPrice-F": "placing_frondel_adjusted",
    "HedPrice-H": "placing_heuer_joint_conservative",
    "LifeSat": "placing_krekel_zerrahn",
    "ExclPop": "placing_tsani_proxy",
    "ExclSce": "placing_tsani_scenicness",
}


def output_parent(states: bool, allocation_mode: str) -> Path:
    region = "with_states" if states else "national"
    return ROOT / f"Output_Main_{region}_{allocation_mode}_upperSlack{STATE_UPPER_SLACK_PERCENT}"


def curve_filename(model: str, states: bool, mode: str) -> str:
    region = "with_states" if states else "national"
    base_suffix = f"{region}_total_gw"
    existing_suffix = "with_existing" if USE_EXISTING_STOCK else "without_existing"
    if model == "Base":
        return f"cost_curve_iterative_base_pv_{base_suffix}.parquet"
    if model == "WTP-Low":
        return f"cost_curve_iterative_wen_ruhnau_lower_{region}_{mode}.parquet"
    if model == "WTP-High":
        return f"cost_curve_iterative_wen_ruhnau_upper_{region}_{mode}.parquet"
    if model == "HedPrice-F":
        return f"cost_curve_iterative_frondel_nearest_{base_suffix}_{mode}_{existing_suffix}.parquet"
    if model == "HedPrice-H":
        return f"cost_curve_iterative_heuer_joint_conservative_{base_suffix}_{mode}_{existing_suffix}.parquet"
    if model == "LifeSat":
        return f"cost_curve_iterative_krekel_zerrahn_{region}_{mode}.parquet"
    if model == "ExclPop":
        return f"cost_curve_iterative_tsani_proxy_tsani_proxy_pop1000_r4_{base_suffix}.parquet"
    if model == "ExclSce":
        return f"cost_curve_iterative_tsani_scenicness_tsani_scenicness_ge5p0_{base_suffix}.parquet"
    raise ValueError(f"Unknown model: {model}")


def curve_path(model: str, states: bool, mode: str) -> Path:
    # Frondel's corrected dynamic greedy ranking was rerun ONLY in total_lcoe
    # mode and deliberately saved to a separate subfolder. The base_only run
    # is unchanged and must still be read from its original output folder.
    # NEVER fall back to the old total_lcoe output if the corrected file is
    # missing: that would silently reproduce the invalid paper results.
    if model == "HedPrice-F" and mode == "total_lcoe":
        folder = "placing_frondel_adjusted_heapfix"
    else:
        folder = FOLDERS[model]
    return output_parent(states, mode) / folder / curve_filename(model, states, mode)


def heuer_meta_path(states: bool, mode: str) -> Path:
    name = curve_filename("HedPrice-H", states, mode).replace("cost_curve_iterative_", "meta_")
    return output_parent(states, mode) / FOLDERS["HedPrice-H"] / name


def validate_heuer_meta(states: bool, mode: str) -> None:
    path = heuer_meta_path(states, mode)
    if not path.is_file():
        raise FileNotFoundError(f"HedPrice-H metadata missing; cannot confirm A10 conservative model: {path}")
    meta = pd.read_parquet(path)
    if meta.empty:
        raise ValueError(f"Empty HedPrice-H metadata: {path}")
    row = meta.iloc[0]
    expectations = {
        "specification": "joint_conservative",
        "algorithm_version": "joint_a10_heap_reinsert_v2",
        "cell_exposure_policy": "intersection_nearest_band",
        "allocation_mode": mode,
    }
    for field, value in expectations.items():
        if field not in meta.columns or str(row[field]) != value:
            raise ValueError(
                f"WRONG HedPrice-H run in {path}: expected {field}={value!r}; "
                f"got {row.get(field, '<missing>')!r}. "
                "Use results from the new CleanRepo main.py."
            )
    if "use_existing_stock" in meta.columns:
        if str(row["use_existing_stock"]).lower() != str(USE_EXISTING_STOCK).lower():
            raise ValueError(f"HedPrice-H stock configuration mismatch: {path}")
    print(f"[PASS] HedPrice-H A10 intersection / heap v2 / {mode}")


def first_col(df: pd.DataFrame, names: tuple[str, ...]):
    return next((n for n in names if n in df.columns), None)


def numeric(df: pd.DataFrame, names: tuple[str, ...], *, required: bool = True):
    name = first_col(df, names)
    if name is None:
        if required:
            raise KeyError(f"Missing columns {names}; columns are {list(df.columns)}")
        return None
    return pd.to_numeric(df[name], errors="coerce")


def standardize(df: pd.DataFrame, model: str, source: Path) -> pd.DataFrame:
    """Use saved PV-LCOE metrics without modifying their economics/denominators."""
    if df.empty:
        raise ValueError(f"Empty curve: {source}")
    out = pd.DataFrame(index=df.index)
    out["capacity_gw"] = numeric(df, ("cum_gw_total_system",))
    out["generation_twh"] = numeric(df, ("cum_twh_per_year_total_system",))
    out["technical_lcoe"] = numeric(df, (
        "lcoe_base_pv_cum_eur_per_mwh", "cumulative_base_lcoe_eur_per_mwh"))
    out["total_lcoe"] = numeric(df, (
        "lcoe_total_pv_cum_eur_per_mwh", "cumulative_total_lcoe_eur_per_mwh"))
    dis = numeric(df, (
        "lcoe_acc_pv_cum_eur_per_mwh", "lcoe_disamenity_pv_cum_eur_per_mwh",
        "cumulative_disamenity_adder_eur_per_mwh"), required=False)
    if dis is None:
        if model in MONETARY_MODELS:
            raise KeyError(f"Missing explicit disamenity component for {model}: {source}")
        dis = pd.Series(0.0, index=df.index)
    out["disamenity_lcoe"] = dis
    marg = numeric(df, ("lcoe_total_pv_step_eur_per_mwh", "marginal_total_lcoe_eur_per_mwh"), required=False)
    if marg is None:
        raise KeyError(f"Missing marginal LCOE series for {model}: {source}")
    out["marginal_lcoe"] = marg
    out["step"] = numeric(df, ("step",), required=False)
    if out["step"] is None or out["step"].isna().all():
        out["step"] = np.arange(1, len(df) + 1)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["capacity_gw", "technical_lcoe", "total_lcoe", "generation_twh"])
    out = out.sort_values("capacity_gw", kind="stable").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No valid cost-curve rows for {model}: {source}")
    if out["marginal_lcoe"].notna().sum() == 0:
        raise ValueError(f"No valid marginal-LCOE data for {model}: {source}")
    if out["capacity_gw"].duplicated().any():
        raise ValueError(f"Duplicate cumulative capacity points for {model}: {source}")
    if float(out["capacity_gw"].max()) < TARGET_GW - 0.01:
        raise ValueError(f"{model} reaches only {out['capacity_gw'].max():.2f} GW, < {TARGET_GW} GW")
    err = (out["technical_lcoe"] + out["disamenity_lcoe"] - out["total_lcoe"]).abs()
    max_err = float(err.max())
    if max_err > 0.02:
        raise ValueError(f"{model}: PV cost components do not sum to total LCOE "
                         f"(max deviation {max_err:.5f} EUR/MWh): {source}")
    out["marginal_smooth"] = out["marginal_lcoe"].rolling(
        window=MARGINAL_ROLLING_TURBINES, center=True, min_periods=1
    ).median()
    return out


def load_curve(model: str, states: bool, mode: str,
               manifest: list[dict]) -> pd.DataFrame:
    path = curve_path(model, states, mode)
    if not path.is_file():
        raise FileNotFoundError(
            f"Output missing for {model} (states={states}, mode={mode}):\n  {path}\n"
            "Please check that the new main.py completed this scenario."
        )
    if model == "HedPrice-H":
        validate_heuer_meta(states, mode)
    print(f"[LOAD] {'states' if states else 'national':8s} {mode:10s} {model:11s} <- {path.name}")
    df = standardize(pd.read_parquet(path), model, path)
    manifest.append({
        "model": model, "state_targets": states, "allocation_mode": mode,
        "source_path": str(path), "rows": len(df),
        "max_capacity_gw": float(df["capacity_gw"].max()),
    })
    return df


def load_set(states: bool, mode: str, models, manifest: list[dict]):
    return {model: load_curve(model, states, mode, manifest) for model in models}


def target_row(df: pd.DataFrame, model: str) -> pd.Series:
    ix = (df["capacity_gw"] - TARGET_GW).abs().idxmin()
    r = df.loc[ix]
    difference = abs(float(r["capacity_gw"]) - TARGET_GW)
    if difference > 0.05:
        raise ValueError(f"{model}: nearest capacity differs by {difference:.4f} GW from target")
    return r


def technical_reference(curves: dict[str, pd.DataFrame]) -> float:
    return float(target_row(curves["Base"], "Base")["total_lcoe"])


def interpolate_base(base: pd.DataFrame, x):
    x = np.asarray(x, dtype=float)
    return np.interp(x, base["capacity_gw"], base["total_lcoe"])


def plot_segment(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["capacity_gw"] >= PLOT_MIN_GW) &
              (df["capacity_gw"] <= PLOT_MAX_GW)]


def init_figure(figsize=(9.4, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="y", color="#d6dbe0", alpha=0.7, lw=0.65, zorder=0)
    ax.tick_params(which="both", length=4, color="#59606a")
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim(PLOT_MIN_GW, PLOT_MAX_GW)
    return fig, ax


def target_marker(ax):
    ax.axvline(TARGET_GW, color="#8d9197", ls=(0, (3, 3)), lw=1.2,
               alpha=0.8, zorder=1)


def draw(ax, model: str, x, y, **kwargs):
    style = dict(STYLE[model])
    style.update(kwargs)
    style.setdefault("label", model)
    return ax.plot(x, y, **style)[0]


def xlabel(ax):
    ax.set_xlabel("Total installed onshore wind capacity incl. existing turbines (GW)")


def legend_below(fig, ax, ncol=4):
    h, labels = ax.get_legend_handles_labels()
    if h:
        # Use figure-level legend to keep all data area unobstructed.
        fig.legend(h, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(ncol, len(h)), frameon=False, columnspacing=1.65,
                   handlelength=2.6)
        fig.subplots_adjust(left=0.105, right=0.98, top=0.97, bottom=0.22)
    else:
        fig.subplots_adjust(left=0.105, right=0.98, top=0.97, bottom=0.16)


def save_figure(fig, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{basename}.{ext}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.13)
        print(f"[SAVE] {path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def fig1_baseline(curves):
    fig, ax = init_figure()
    d = plot_segment(curves["Base"])
    draw(ax, "Base", d["capacity_gw"], d["total_lcoe"], label="Technical baseline")
    target_marker(ax)
    xlabel(ax)
    ax.set_ylabel("Total LCOE (EUR/MWh)")
    legend_below(fig, ax, ncol=1)
    save_figure(fig, "fig1_technical_baseline")


def fig2_total_lcoe(curves):
    fig, ax = init_figure(figsize=(10.3, 5.9))
    for model in MODEL_ORDER:
        d = plot_segment(curves[model])
        draw(ax, model, d["capacity_gw"], d["total_lcoe"])
    target_marker(ax)
    xlabel(ax)
    ax.set_ylabel("Total LCOE (EUR/MWh)")
    legend_below(fig, ax, ncol=4)
    save_figure(fig, "fig2_total_lcoe_all_models")


def fig3_premiums(curves):
    fig, ax = init_figure(figsize=(10.3, 5.9))
    base = curves["Base"]
    for model in MODEL_ORDER[1:]:
        d = plot_segment(curves[model])
        y = d["total_lcoe"].to_numpy() - interpolate_base(base, d["capacity_gw"])
        draw(ax, model, d["capacity_gw"], y)
    target_marker(ax)
    ax.axhline(0, color="#535962", lw=1.1, alpha=0.8)
    xlabel(ax)
    ax.set_ylabel("LCOE premium over technical baseline (EUR/MWh)")
    legend_below(fig, ax, ncol=4)
    save_figure(fig, "fig3_lcoe_premiums")


def summary_table(curves):
    base_lcoe = technical_reference(curves)
    base_generation = float(target_row(curves["Base"], "Base")["generation_twh"])
    result = []
    for model in MODEL_ORDER:
        r = target_row(curves[model], model)
        total = float(r["total_lcoe"])
        result.append({
            "model": model,
            "achieved_capacity_gw": float(r["capacity_gw"]),
            "total_lcoe_eur_mwh": total,
            "premium_over_baseline_eur_mwh": total - base_lcoe,
            "premium_percent": 100 * (total / base_lcoe - 1),
            "technical_component_eur_mwh": float(r["technical_lcoe"]),
            "disamenity_component_eur_mwh": float(r["disamenity_lcoe"]),
            "total_generation_twh_per_year": float(r["generation_twh"]),
            "generation_difference_to_base_twh_per_year": (
                float(r["generation_twh"]) - base_generation),
            "marginal_lcoe_at_target_eur_mwh": float(r["marginal_lcoe"]),
        })
    return pd.DataFrame(result).sort_values("total_lcoe_eur_mwh").reset_index(drop=True)


def allocation_table(base_only, extended):
    base_reference = technical_reference(extended)
    rows = []
    for model in MONETARY_MODELS:
        b = target_row(base_only[model], model + " base-only")
        e = target_row(extended[model], model + " total-LCOE")
        rows.append({
            "model": model,
            "standard_total_lcoe_eur_mwh": float(b["total_lcoe"]),
            "extended_total_lcoe_eur_mwh": float(e["total_lcoe"]),
            "delta_extended_minus_standard_eur_mwh": float(e["total_lcoe"] - b["total_lcoe"]),
            "standard_premium_eur_mwh": float(b["total_lcoe"] - base_reference),
            "extended_premium_eur_mwh": float(e["total_lcoe"] - base_reference),
            "standard_technical_eur_mwh": float(b["technical_lcoe"]),
            "extended_technical_eur_mwh": float(e["technical_lcoe"]),
            "standard_disamenity_eur_mwh": float(b["disamenity_lcoe"]),
            "extended_disamenity_eur_mwh": float(e["disamenity_lcoe"]),
            "standard_generation_twh": float(b["generation_twh"]),
            "extended_generation_twh": float(e["generation_twh"]),
            "generation_delta_extended_minus_standard_twh": float(
                e["generation_twh"] - b["generation_twh"]),
        })
    return pd.DataFrame(rows)


def fig4_allocation_effects(df):
    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    y = np.arange(len(df))
    h = 0.33
    colors = [STYLE[m]["color"] for m in df["model"]]
    # Light colored faces + opaque colored hatch lines: strong separation
    # even when printed in grayscale or resized for the journal.
    light_colors = [(*to_rgb(c), 0.29) for c in colors]
    ax.barh(y - h/2, df["standard_premium_eur_mwh"], height=h,
            facecolor=light_colors, edgecolor=colors, lw=1.15,
            hatch="////", label="Standard LCOE allocation")
    ax.barh(y + h/2, df["extended_premium_eur_mwh"], height=h,
            color=colors, alpha=0.96, edgecolor="#30343a", lw=0.4,
            label="Extended LCOE allocation")
    ax.set_yticks(y, df["model"])
    ax.invert_yaxis()
    ax.axvline(0, color="#42464c", lw=1)
    ax.grid(axis="x", alpha=0.24, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"LCOE premium at {TARGET_GW:g} GW (EUR/MWh)")
    legend_below(fig, ax, ncol=2)
    save_figure(fig, "fig4_allocation_standard_vs_extended")


def fig5_marginal(curves):
    fig, ax = init_figure(figsize=(10.3, 5.9))
    for model in MODEL_ORDER:
        d = plot_segment(curves[model])
        draw(ax, model, d["capacity_gw"], d["marginal_smooth"])
    target_marker(ax)
    xlabel(ax)
    ax.set_ylabel(f"Marginal LCOE (EUR/MWh); {MARGINAL_ROLLING_TURBINES}-turbine rolling median")
    legend_below(fig, ax, ncol=4)
    save_figure(fig, "fig5_smoothed_marginal_lcoe")


def regional_table(states, national):
    state_base = technical_reference(states)
    national_base = technical_reference(national)
    rows = []
    for model in MODEL_ORDER:
        s = target_row(states[model], model + " states")
        n = target_row(national[model], model + " national")
        p_s = float(s["total_lcoe"] - state_base)
        p_n = float(n["total_lcoe"] - national_base)
        rows.append({
            "model": model,
            "lcoe_states_eur_mwh": float(s["total_lcoe"]),
            "lcoe_national_eur_mwh": float(n["total_lcoe"]),
            "premium_states_eur_mwh": p_s,
            "premium_national_eur_mwh": p_n,
            "premium_national_minus_states_eur_mwh": p_n - p_s,
            "generation_states_twh": float(s["generation_twh"]),
            "generation_national_twh": float(n["generation_twh"]),
        })
    return pd.DataFrame(rows)


def fig_appendix_regional(df):
    d = df[df["model"] != "Base"].sort_values("premium_states_eur_mwh")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(d))
    h = 0.34
    ax.barh(y - h / 2, d["premium_states_eur_mwh"], h, color="#527da1",
            label="With federal-state targets")
    ax.barh(y + h / 2, d["premium_national_eur_mwh"], h, color="#a3b0bb",
            label="National allocation")
    ax.set_yticks(y, d["model"])
    ax.invert_yaxis()
    ax.axvline(0, color="#53575c", lw=1)
    ax.grid(axis="x", alpha=0.24)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"LCOE premium at {TARGET_GW:g} GW (EUR/MWh), relative to respective baseline")
    legend_below(fig, ax, ncol=2)
    save_figure(fig, "appendix_regional_state_vs_national")


def fig_extra_decomposition(summary):
    """Stack technical siting penalty and remaining explicit disamenity, not total base LCOE."""
    data = summary[summary["model"] != "Base"].copy()
    base_lcoe = float(summary.loc[summary["model"] == "Base", "total_lcoe_eur_mwh"].iloc[0])
    data["technical_penalty"] = data["technical_component_eur_mwh"] - base_lcoe
    data = data.sort_values("premium_over_baseline_eur_mwh")
    if (data["technical_penalty"] < -0.005).any():
        warnings.warn("Negative technical penalties: inspect stacked decomposition before interpretation.")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    ax.barh(y, data["technical_penalty"], color="#758b9d",
            label="Technical siting penalty")
    ax.barh(y, data["disamenity_component_eur_mwh"],
            left=data["technical_penalty"], color="#c2875c",
            label="Remaining explicit disamenity")
    ax.set_yticks(y, data["model"])
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22)
    ax.axvline(0, lw=1, color="#50565f")
    ax.set_xlabel(f"Decomposition of LCOE premium at {TARGET_GW:g} GW (EUR/MWh)")
    legend_below(fig, ax, ncol=2)
    save_figure(fig, "extra_premium_decomposition_160GW")


def fig_extra_hedonic(curves):
    fig, ax = init_figure(figsize=(9.3, 5.2))
    base = curves["Base"]
    for model in ("HedPrice-F", "HedPrice-H"):
        d = plot_segment(curves[model])
        premium = d["total_lcoe"].to_numpy() - interpolate_base(base, d["capacity_gw"])
        draw(ax, model, d["capacity_gw"], premium)
    target_marker(ax)
    xlabel(ax)
    ax.set_ylabel("LCOE premium over technical baseline (EUR/MWh)")
    legend_below(fig, ax, ncol=2)
    save_figure(fig, "extra_hedprice_F_vs_H")


def save_table(df: pd.DataFrame, name: str) -> None:
    path = OUT_DIR / name
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"[SAVE] {path}")



# ==========================================================================
# FIGURES 6 + 7: GEOGRAPHIC PAPER MAPS AT EXACT TARGET STEP
# ==========================================================================
# IMPORTANT: placement_cells_* contains the FULL final (often 220 GW) run.
# Never use its final added_gw column for a 160 GW paper map. Instead
# reconstruct selected new-build capacity up to the target step from the
# chronological cost_curve_* (which retains idm / step / take_gw).


def paper_map_files():
    """Exact inputs for paper maps: baseline, standard F, corrected extended F."""
    base_curve = curve_path("Base", states=True, mode="total_lcoe")
    standard_curve = curve_path("HedPrice-F", states=True, mode="base_only")
    extended_curve = curve_path("HedPrice-F", states=True, mode="total_lcoe")
    base_cells = base_curve.with_name(base_curve.name.replace("cost_curve_", "placement_cells_", 1))
    fixed_cells = extended_curve.with_name(extended_curve.name.replace("cost_curve_", "placement_cells_", 1))
    for path in (base_curve, standard_curve, extended_curve, base_cells, fixed_cells):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot create paper map: missing file\n  {path}")
    # Check COMPLETE directory names, not individual words within Path.parts.
    # A folder named "placing_frondel_adjusted_heapfix" is a single path part;
    # therefore testing `"heapfix" in path.parts` would fail incorrectly.
    expected_fixed_dir = "placing_frondel_adjusted_heapfix"
    expected_standard_dir = "placing_frondel_adjusted"
    if (extended_curve.parent.name.lower() != expected_fixed_dir
            or fixed_cells.parent != extended_curve.parent):
        raise ValueError(
            "HedPrice-F extended paper maps MUST use corrected heapfix output: "
            f"{expected_fixed_dir}. Got: {extended_curve}"
        )
    if standard_curve.parent.name.lower() != expected_standard_dir:
        raise ValueError(
            "HedPrice-F standard map must use unchanged base_only output: "
            f"{expected_standard_dir}. Got: {standard_curve}"
        )
    return base_curve, standard_curve, extended_curve, base_cells, fixed_cells


def reconstruction_at_target(cost_curve_path: Path) -> tuple[pd.DataFrame, dict]:
    """Recover true 160-GW allocations from the ordered turbine placements."""
    raw = pd.read_parquet(cost_curve_path)
    required = {"idm", "step", "take_gw", "cum_gw_total_system", "cum_gw"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"Cannot reconstruct 160 GW map from {cost_curve_path}: missing {missing}")

    raw = raw.copy()
    raw["step"] = pd.to_numeric(raw["step"], errors="raise")
    raw["take_gw"] = pd.to_numeric(raw["take_gw"], errors="raise")
    raw["cum_gw_total_system"] = pd.to_numeric(raw["cum_gw_total_system"], errors="raise")
    raw["cum_gw"] = pd.to_numeric(raw["cum_gw"], errors="raise")
    raw = raw.sort_values("step", kind="stable").reset_index(drop=True)
    if raw["step"].duplicated().any() or (raw["take_gw"] <= 0).any():
        raise ValueError("Invalid turbine placement sequence: duplicate steps or nonpositive capacity")

    pos = int(np.argmin(np.abs(raw["cum_gw_total_system"].to_numpy() - TARGET_GW)))
    picked = raw.iloc[:pos+1].copy()
    end_row = raw.iloc[pos]
    difference_gw = abs(float(end_row["cum_gw_total_system"]) - TARGET_GW)
    if difference_gw > 0.05:
        raise ValueError(f"Map target mismatch: reached {end_row['cum_gw_total_system']} GW, target {TARGET_GW}")

    result = picked.groupby("idm", as_index=False, sort=False)["take_gw"].sum()
    result = result.rename(columns={"take_gw": "selected_added_gw"})
    observed = float(result["selected_added_gw"].sum())
    expected = float(end_row["cum_gw"])
    if not np.isclose(observed, expected, atol=5e-5, rtol=0):
        raise ValueError(f"Map capacity validation failed: {observed:.6f} versus {expected:.6f} GW")

    info = {
        "source": str(cost_curve_path),
        "target_step": int(end_row["step"]),
        "installed_gw": float(end_row["cum_gw_total_system"]),
        "new_build_gw": observed,
        "selected_cells": len(result),
    }
    print(f"[PASS] Target map: step {info['target_step']:,}, "
          f"installed {info['installed_gw']:.6f} GW, "
          f"new build {info['new_build_gw']:.6f} GW, {len(result):,} selected cells")
    return result, info


def load_paper_map_data() -> tuple[object, dict]:
    """Prepare both comparisons: fig6 standard F vs fixed F; fig7 Base vs fixed F."""
    import geopandas as gpd

    base_curve, standard_curve, fixed_curve, base_cells_path, fixed_cells_path = paper_map_files()
    selected_base, info_base = reconstruction_at_target(base_curve)
    selected_standard, info_standard = reconstruction_at_target(standard_curve)
    selected_fixed, info_fixed = reconstruction_at_target(fixed_curve)
    print(f"[LOAD MAP GRID] {base_cells_path}")
    grid = gpd.read_parquet(base_cells_path)
    if not isinstance(grid, gpd.GeoDataFrame):
        grid = gpd.GeoDataFrame(grid, geometry="geometry")
    needed = {"idm", "geometry", "base_lcoe_cell_eur_per_mwh"}
    missing = needed.difference(grid.columns)
    if missing:
        raise KeyError(f"Missing base grid fields {missing} in {base_cells_path}")
    if grid["idm"].duplicated().any():
        raise ValueError("Grid idm values are not unique")
    if grid.crs is None:
        raise ValueError("Grid CRS undefined; cannot safely draw maps")

    print(f"[LOAD CHECKED FRONDEL CELLS] {fixed_cells_path}")
    fixed_cells = gpd.read_parquet(fixed_cells_path)
    if "idm" not in fixed_cells.columns:
        raise KeyError("Corrected HedPrice-F placement_cells missing idm")
    known_fixed_ids = set(fixed_cells["idm"])
    if not set(selected_fixed["idm"]).issubset(known_fixed_ids):
        raise ValueError("Corrected Frondel cost curve and placement cells mismatch")
    del fixed_cells

    if "in_de" in grid.columns:
        grid = grid.loc[pd.to_numeric(grid["in_de"], errors="coerce").fillna(0) > 0.5].copy()
    grid = grid[[x for x in ("idm", "geometry", "base_lcoe_cell_eur_per_mwh",
                             "existing_gw_cell", "existing_gw") if x in grid.columns]].copy()
    grid_crs = grid.crs
    for selected, output_col in (
        (selected_base, "baseline_added_gw"),
        (selected_standard, "standard_added_gw"),
        (selected_fixed, "extended_added_gw"),
    ):
        grid = grid.merge(selected.rename(columns={"selected_added_gw": output_col}),
                          on="idm", how="left", validate="one_to_one")
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=grid_crs)
    for col in ("baseline_added_gw", "standard_added_gw", "extended_added_gw"):
        grid[col] = grid[col].fillna(0).astype(float)
    # Fig6 compares the SAME HedPrice-F model under standard vs extended siting.
    grid["delta_added_mw"] = 1000 * (grid["extended_added_gw"] - grid["standard_added_gw"])
    grid["technical_site_lcoe"] = pd.to_numeric(
        grid["base_lcoe_cell_eur_per_mwh"], errors="coerce")
    if "existing_gw_cell" in grid.columns:
        grid["existing_for_map"] = pd.to_numeric(grid["existing_gw_cell"], errors="coerce").fillna(0)
    elif "existing_gw" in grid.columns:
        grid["existing_for_map"] = pd.to_numeric(grid["existing_gw"], errors="coerce").fillna(0)
    else:
        grid["existing_for_map"] = 0.0
        warnings.warn("Existing wind capacity column not found; gray points unavailable for Figure 7")

    for col, info, scenario in (
        ("baseline_added_gw", info_base, "Technical baseline"),
        ("standard_added_gw", info_standard, "HedPrice-F standard"),
        ("extended_added_gw", info_fixed, "HedPrice-F extended"),
    ):
        if abs(float(grid[col].sum()) - info["new_build_gw"]) > 0.05:
            raise ValueError(f"Germany map grid misses >0.05 GW of {scenario} selected capacity")
    print(f"[PASS] Fig6 reallocation: {grid['delta_added_mw'].gt(0).sum():,} increased cells, "
          f"{grid['delta_added_mw'].lt(0).sum():,} decreased cells, "
          f"shifted {grid.loc[grid['delta_added_mw']>0, 'delta_added_mw'].sum()/1000:.3f} GW")
    return grid, {"baseline": info_base, "standard_hedprice_f": info_standard,
                  "extended_hedprice_f": info_fixed}


def germany_outline(grid):
    """Use a common country outline; calculate once for Figures 6 and 7."""
    if not DRAW_GERMANY_OUTLINE:
        return None
    try:
        from shapely import union_all
        # The 1 km grid is a coverage of non-overlapping squares; a union
        # gives a useful outline without external boundary files.
        return union_all(grid.geometry.values)
    except Exception as exc:
        warnings.warn(f"Cannot calculate map outline ({exc}); drawing cells without outline")
        return None


def draw_map_background(ax, grid, outline=None):
    """Low-contrast background; no map titles/legends inside data region."""
    # Points are faster than plotting >300k polygon patches three times.
    centroids = grid.geometry.centroid
    ax.scatter(centroids.x.to_numpy(), centroids.y.to_numpy(), s=0.28,
               c="#e9e9e9", alpha=0.73, marker="s", linewidths=0,
               rasterized=True, zorder=1)
    if outline is not None:
        try:
            import geopandas as gpd
            gpd.GeoSeries([outline.boundary], crs=grid.crs).plot(
                ax=ax, color="#bcbcbc", linewidth=0.35, alpha=0.65, zorder=4)
        except Exception as exc:
            warnings.warn(f"Map boundary unavailable: {exc}")
    ax.set_aspect("equal")
    ax.set_axis_off()


def truncated_diverging_cmap():
    cmap = plt.get_cmap("RdBu_r")
    return LinearSegmentedColormap.from_list(
        "RdBu_r_paper", cmap(np.linspace(0.12, 0.88, 256)))


def fig6_spatial_reallocation(grid, outline=None):
    """Corrected HedPrice-F extended minus HedPrice-F standard at 160 GW."""
    from matplotlib.colors import TwoSlopeNorm
    changed = grid.loc[grid["delta_added_mw"].abs() > 1e-7].copy()
    if changed.empty:
        raise ValueError("Figure 6: no sites differ between allocation modes")

    absvals = changed["delta_added_mw"].abs()
    vmax = (float(MAP_REALLOCATION_LIMIT_MW)
            if MAP_REALLOCATION_LIMIT_MW is not None else float(absvals.quantile(0.99)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(absvals.max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig = plt.figure(figsize=(9.0, 8.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.035], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    draw_map_background(ax, grid, outline)
    changed.plot(ax=ax, column="delta_added_mw", cmap=truncated_diverging_cmap(),
                 norm=norm, linewidth=0, edgecolor="none", zorder=3, legend=False)
    sm = ScalarMappable(norm=norm, cmap=truncated_diverging_cmap()); sm.set_array([])
    bar = fig.colorbar(sm, cax=cax, extend="both" if (absvals > vmax).any() else "neither")
    bar.set_label("Δ added capacity (MW per cell)\nExtended minus standard LCOE allocation", fontsize=10)
    bar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.97, bottom=0.03)
    save_figure(fig, "fig6_spatial_reallocation_hedprice_F")


def fig7_technical_site_lcoe(grid, outline=None):
    """Technical site LCOE for baseline versus corrected HedPrice-F at 160 GW."""
    def selected(added_col):
        return grid.loc[(grid[added_col] > 0) & grid["technical_site_lcoe"].notna()].copy()
    left = selected("baseline_added_gw")
    right = selected("extended_added_gw")
    if left.empty or right.empty:
        raise ValueError("Figure 7: no selected locations or missing technical LCOE")
    values = pd.concat([left["technical_site_lcoe"], right["technical_site_lcoe"]])
    vmin = float(values.quantile(MAP_LCOE_Q_LOWER))
    vmax = float(values.quantile(MAP_LCOE_Q_UPPER))
    if vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    fig = plt.figure(figsize=(13.0, 6.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.04], wspace=0.035)
    axes = [fig.add_subplot(gs[0, k]) for k in range(2)]
    cax = fig.add_subplot(gs[0, 2])
    existing = grid.loc[grid["existing_for_map"] > 0].copy()
    if not existing.empty:
        existing["geometry"] = existing.geometry.centroid

    for ax, subset, caption in (
        (axes[0], left, "Technical baseline"),
        (axes[1], right, "Extended LCOE allocation (HedPrice-F)"),
    ):
        draw_map_background(ax, grid, outline)
        if SHOW_EXISTING_STOCK_ON_MAPS and not existing.empty:
            existing.plot(ax=ax, color="#9a9a9a", markersize=0.8,
                          alpha=0.23, linewidth=0, zorder=2)
        subset.plot(ax=ax, column="technical_site_lcoe", cmap=cmap,
                    norm=norm, linewidth=0, edgecolor="none", zorder=3)
        ax.set_title(caption, fontsize=11.5, pad=8)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    bar = fig.colorbar(sm, cax=cax, extend="both")
    bar.set_label("Technical LCOE of selected cells (EUR/MWh)", fontsize=10)
    bar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.015, right=0.945, top=0.92, bottom=0.035)
    save_figure(fig, "fig7_technical_site_lcoe_baseline_vs_hedprice_F")


def create_paper_maps():
    grid, info = load_paper_map_data()
    outline = germany_outline(grid)
    fig6_spatial_reallocation(grid, outline)
    fig7_technical_site_lcoe(grid, outline)
    map_info = pd.DataFrame([
        {"scenario": "Technical baseline", **info["baseline"]},
        {"scenario": "HedPrice-F standard LCOE allocation", **info["standard_hedprice_f"]},
        {"scenario": "HedPrice-F extended LCOE allocation (heapfix)", **info["extended_hedprice_f"]},
    ])
    save_table(map_info, "map_inputs_160GW.csv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    print("\n=== PAPER RESULTS: 8 MODELS / 160 GW / UPPER SLACK 50% ===")
    print("Reading existing Parquet results only; the allocation models are NOT run.")
    print(f"ROOT:   {ROOT}")
    print(f"OUTPUT: {OUT_DIR}")
    print("[CHECK] HedPrice-F extended: corrected heapfix outputs ONLY")
    print("[CHECK] HedPrice-F standard: unchanged original base_only outputs")

    # Main paper case: disamenity-aware ranking with state targets.
    ext = load_set(states=True, mode="total_lcoe", models=MODEL_ORDER, manifest=manifest)
    main_summary = summary_table(ext)
    print("\n[MAIN PAPER TABLE: 160 GW]\n", main_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save_table(main_summary, "table_160GW_extended_states.csv")
    fig1_baseline(ext)
    fig2_total_lcoe(ext)
    fig3_premiums(ext)
    fig5_marginal(ext)

    # Both types of siting are implemented for the FIVE monetary model scenarios.
    std = load_set(states=True, mode="base_only", models=MONETARY_MODELS, manifest=manifest)
    allocation = allocation_table(std, ext)
    print("\n[ALLOCATION EFFECTS: 160 GW]\n", allocation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save_table(allocation, "table_allocation_effects_160GW.csv")
    fig4_allocation_effects(allocation)

    if MAKE_PAPER_MAPS:
        create_paper_maps()

    if MAKE_REGIONAL_SENSITIVITY:
        nat = load_set(states=False, mode="total_lcoe", models=MODEL_ORDER, manifest=manifest)
        regional = regional_table(ext, nat)
        print("\n[REGIONAL SENSITIVITY: 160 GW]\n", regional.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        save_table(regional, "table_regional_sensitivity_160GW.csv")
        fig_appendix_regional(regional)

    if MAKE_EXTRA_FIGURES:
        fig_extra_decomposition(main_summary)
        fig_extra_hedonic(ext)

    save_table(pd.DataFrame(manifest), "inputs_manifest.csv")
    print(f"\n[DONE] All figures (PNG+PDF) and tables saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
