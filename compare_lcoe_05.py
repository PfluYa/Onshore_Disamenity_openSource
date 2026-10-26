# -*- coding: utf-8 -*-
"""Compare baseline and disamenity-inclusive LCOE curves.

Includes Base, WTP-Low, WTP-High, HedPrice-F, LifeSat, ExclPop and ExclSce;
HedPrice-H is also supported by the comparison configuration. The module
supports historical result files in which some baseline/total columns are
missing: where possible, missing columns can be reconstructed from their
counterparts. Existing-fleet offsets can optionally be aligned to Base.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paper labels and reproducible default colors (Frondel and Heuer separated).
MODEL_STYLE = {
    "Base": dict(color="#1f4e79", linewidth=2.8, linestyle="-"),
    "WTP-Low": dict(color="#f28e2b", linewidth=2.2, linestyle="-"),
    "WTP-High": dict(color="#c66a00", linewidth=2.2, linestyle="--"),
    "HedPrice-F": dict(color="#2ca02c", linewidth=2.4, linestyle="-"),
    "HedPrice-H": dict(color="#008b8b", linewidth=2.4, linestyle="-"),
    "LifeSat": dict(color="#8a60bf", linewidth=2.2, linestyle="-"),
    "ExclPop": dict(color="#8c564b", linewidth=2.2, linestyle="--"),
    "ExclSce": dict(color="#e377c2", linewidth=2.2, linestyle=":"),
}


# ============================================================
# Config
# ============================================================

@dataclass
class CompareLCOEConfig:
    root: Path = Path(__file__).resolve().parent
    output_dir: Path = Path(__file__).resolve().parent / 'Output_Main' / 'compare_lcoe'

    save_plots: bool = True
    show_plots: bool = True
    save_combined_curve_table: bool = True

    align_existing_offsets_to_base: bool = True

    include_base_lcoe: bool = True
    include_wen_ruhnau: bool = True
    include_frondel: bool = True
    include_heuer: bool = True
    include_krekel_zerrahn: bool = True
    include_tsani_proxy: bool = True
    include_tsani_scenicness: bool = True

    wen_compare_lower: bool = True
    wen_compare_upper: bool = True

    plot_capacity_axis: bool = True
    plot_generation_axis: bool = True

    plot_cumulative: bool = True
    plot_marginal: bool = True

    figure_dpi: int = 180


# ============================================================
# Existing result loader
# ============================================================

def _load_df_if_exists(path: Path, label: str) -> Optional[pd.DataFrame]:
    if path.exists():
        print(f"[LOAD] {label} <- {path}")
        return pd.read_parquet(path)
    print(f"[WARN] missing {label}: {path}")
    return None


def load_compare_inputs_from_existing(
    cfg: CompareLCOEConfig,
    shared: Any,
    allocation_mode: str,
    use_frondel_existing_stock: bool,
    base_dir: Path,
    base_subdir: str,
    wen_subdir: str,
    frondel_subdir: str,
    krekel_zerrahn_subdir: Optional[str] = None,
    tsani_proxy_subdir: Optional[str] = None,
    tsani_scenicness_subdir: Optional[str] = None,
    tsani_population_density_threshold: float = 500.0,
    tsani_buffer_radius_km: float = 8.0,
    scenicness_threshold: float = 7.0,
) -> Dict[str, Optional[pd.DataFrame]]:

    use_state_targets = bool(shared.use_state_targets)
    build_limit_mode = str(shared.build_limit_mode)

    base_suffix = (
        f"with_states_{build_limit_mode}"
        if use_state_targets
        else f"national_{build_limit_mode}"
    )

    wen_suffix = (
        f"with_states_{allocation_mode}"
        if use_state_targets
        else f"national_{allocation_mode}"
    )

    frondel_regional = (
        f"with_states_{build_limit_mode}"
        if use_state_targets
        else f"national_{build_limit_mode}"
    )
    frondel_existing = "with_existing" if use_frondel_existing_stock else "without_existing"
    frondel_suffix = f"{frondel_regional}_{allocation_mode}_{frondel_existing}"

    out = {
        "base_curve": _load_df_if_exists(
            base_dir / base_subdir / f"cost_curve_iterative_base_pv_{base_suffix}.parquet",
            "base curve",
        ),
        "wen_lower_curve": _load_df_if_exists(
            base_dir / wen_subdir / f"cost_curve_iterative_wen_ruhnau_lower_{wen_suffix}.parquet",
            "wen lower curve",
        ),
        "wen_upper_curve": _load_df_if_exists(
            base_dir / wen_subdir / f"cost_curve_iterative_wen_ruhnau_upper_{wen_suffix}.parquet",
            "wen upper curve",
        ),
        "frondel_curve": _load_df_if_exists(
            base_dir / frondel_subdir / f"cost_curve_iterative_frondel_nearest_{frondel_suffix}.parquet",
            "frondel curve",
        ),
    }

    if krekel_zerrahn_subdir is not None:
        kz_suffix = base_suffix
        out["krekel_zerrahn_curve"] = _load_df_if_exists(
            base_dir / krekel_zerrahn_subdir / f"cost_curve_iterative_krekel_zerrahn_{kz_suffix}.parquet",
            "krekel zerrahn curve",
        )
    else:
        out["krekel_zerrahn_curve"] = None

    if tsani_proxy_subdir is not None:
        regional = (
            f"with_states_{build_limit_mode}"
            if use_state_targets
            else f"national_{build_limit_mode}"
        )
        thr = str(int(round(tsani_population_density_threshold)))
        rad = str(int(round(tsani_buffer_radius_km)))
        tsani_suffix = f"tsani_proxy_pop{thr}_r{rad}_{regional}"

        out["tsani_proxy_curve"] = _load_df_if_exists(
            base_dir / tsani_proxy_subdir / f"cost_curve_iterative_tsani_proxy_{tsani_suffix}.parquet",
            "tsani proxy curve",
        )
    else:
        out["tsani_proxy_curve"] = None

    if tsani_scenicness_subdir is not None:
        regional = (
            f"with_states_{build_limit_mode}"
            if use_state_targets
            else f"national_{build_limit_mode}"
        )
        thr_txt = str(scenicness_threshold).replace(".", "p")
        scenic_suffix = f"tsani_scenicness_ge{thr_txt}_{regional}"

        out["tsani_scenicness_curve"] = _load_df_if_exists(
            base_dir / tsani_scenicness_subdir / f"cost_curve_iterative_tsani_scenicness_{scenic_suffix}.parquet",
            "tsani scenicness curve",
        )
    else:
        out["tsani_scenicness_curve"] = None

    return out


# ============================================================
# Standardization
# ============================================================

def _ensure_curve(df: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
    if df is None:
        print(f"[WARN] curve {name} is None")
        return None
    if len(df) == 0:
        print(f"[WARN] curve {name} is empty")
        return None
    return df.copy()


def _standardize_curve_columns(
    df: Optional[pd.DataFrame],
    name: str,
) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    out = df.copy()

    if "step" not in out.columns:
        out["step"] = np.arange(1, len(out) + 1)

    if "cum_gw" not in out.columns and "take_gw" in out.columns:
        out["cum_gw"] = out["take_gw"].cumsum()

    if "cum_twh_per_year" not in out.columns and "take_twh_per_year" in out.columns:
        out["cum_twh_per_year"] = out["take_twh_per_year"].cumsum()

    if "cum_gw_total_system" not in out.columns and "cum_gw" in out.columns:
        if "existing_total_gw" in out.columns:
            out["cum_gw_total_system"] = float(out["existing_total_gw"].iloc[0]) + out["cum_gw"]
        else:
            out["cum_gw_total_system"] = out["cum_gw"]

    if "cum_twh_per_year_total_system" not in out.columns and "cum_twh_per_year" in out.columns:
        if "existing_total_twh_per_year" in out.columns:
            out["cum_twh_per_year_total_system"] = float(out["existing_total_twh_per_year"].iloc[0]) + out["cum_twh_per_year"]
        else:
            out["cum_twh_per_year_total_system"] = out["cum_twh_per_year"]

    # Build cumulative cost/energy columns if possible
    if "cum_pv_energy_mwh" not in out.columns and "pv_energy_mwh_step" in out.columns:
        out["cum_pv_energy_mwh"] = out["pv_energy_mwh_step"].cumsum()

    if "cum_base_cost_eur" not in out.columns and "pv_base_cost_eur_step" in out.columns:
        out["cum_base_cost_eur"] = out["pv_base_cost_eur_step"].cumsum()

    if "cum_acc_cost_eur" not in out.columns and "take_cost_eur" in out.columns:
        out["cum_acc_cost_eur"] = out["take_cost_eur"].cumsum()

    # Base LCOE
    if "lcoe_base_pv_step_eur_per_mwh" not in out.columns:
        if {"pv_base_cost_eur_step", "pv_energy_mwh_step"}.issubset(out.columns):
            out["lcoe_base_pv_step_eur_per_mwh"] = (
                out["pv_base_cost_eur_step"] / np.maximum(out["pv_energy_mwh_step"], 1e-12)
            )
        elif "lcoe_total_pv_step_eur_per_mwh" in out.columns:
            out["lcoe_base_pv_step_eur_per_mwh"] = out["lcoe_total_pv_step_eur_per_mwh"]

    if "lcoe_base_pv_cum_eur_per_mwh" not in out.columns:
        if {"cum_base_cost_eur", "cum_pv_energy_mwh"}.issubset(out.columns):
            out["lcoe_base_pv_cum_eur_per_mwh"] = (
                out["cum_base_cost_eur"] / np.maximum(out["cum_pv_energy_mwh"], 1e-12)
            )
        elif "lcoe_total_pv_cum_eur_per_mwh" in out.columns:
            out["lcoe_base_pv_cum_eur_per_mwh"] = out["lcoe_total_pv_cum_eur_per_mwh"]

    # Total LCOE
    if "lcoe_total_pv_step_eur_per_mwh" not in out.columns:
        if {"pv_base_cost_eur_step", "take_cost_eur", "pv_energy_mwh_step"}.issubset(out.columns):
            out["lcoe_total_pv_step_eur_per_mwh"] = (
                (out["pv_base_cost_eur_step"] + out["take_cost_eur"])
                / np.maximum(out["pv_energy_mwh_step"], 1e-12)
            )
        elif "lcoe_base_pv_step_eur_per_mwh" in out.columns:
            out["lcoe_total_pv_step_eur_per_mwh"] = out["lcoe_base_pv_step_eur_per_mwh"]

    if "lcoe_total_pv_cum_eur_per_mwh" not in out.columns:
        if {"cum_base_cost_eur", "cum_acc_cost_eur", "cum_pv_energy_mwh"}.issubset(out.columns):
            out["lcoe_total_pv_cum_eur_per_mwh"] = (
                (out["cum_base_cost_eur"] + out["cum_acc_cost_eur"])
                / np.maximum(out["cum_pv_energy_mwh"], 1e-12)
            )
        elif "lcoe_base_pv_cum_eur_per_mwh" in out.columns:
            out["lcoe_total_pv_cum_eur_per_mwh"] = out["lcoe_base_pv_cum_eur_per_mwh"]

    needed = [
        "cum_gw_total_system",
        "cum_twh_per_year_total_system",
        "lcoe_total_pv_cum_eur_per_mwh",
        "lcoe_total_pv_step_eur_per_mwh",
    ]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        print(f"[WARN] curve {name} missing columns after standardization: {missing}")
        return None

    return out


def _align_existing_offsets_to_base(
    base_curve: Optional[pd.DataFrame],
    curves: Dict[str, Optional[pd.DataFrame]],
) -> Dict[str, Optional[pd.DataFrame]]:
    if base_curve is None or base_curve.empty:
        return curves

    if "existing_total_gw" not in base_curve.columns or "existing_total_twh_per_year" not in base_curve.columns:
        return curves

    ref_existing_gw = float(base_curve["existing_total_gw"].iloc[0])
    ref_existing_twh = float(base_curve["existing_total_twh_per_year"].iloc[0])

    out = {}

    for name, df in curves.items():
        if df is None or df.empty:
            out[name] = df
            continue

        d = df.copy()

        if "cum_gw" in d.columns:
            d["cum_gw_total_system"] = ref_existing_gw + d["cum_gw"]
            d["existing_total_gw"] = ref_existing_gw

        if "cum_twh_per_year" in d.columns:
            d["cum_twh_per_year_total_system"] = ref_existing_twh + d["cum_twh_per_year"]
            d["existing_total_twh_per_year"] = ref_existing_twh

        out[name] = d

    return out


# ============================================================
# Plot construction
# ============================================================

def _build_curve_dict(
    cfg: CompareLCOEConfig,
    *,
    base_curve: Optional[pd.DataFrame],
    wen_lower_curve: Optional[pd.DataFrame],
    wen_upper_curve: Optional[pd.DataFrame],
    frondel_curve: Optional[pd.DataFrame],
    heuer_curve: Optional[pd.DataFrame],
    krekel_zerrahn_curve: Optional[pd.DataFrame],
    tsani_proxy_curve: Optional[pd.DataFrame],
    tsani_scenicness_curve: Optional[pd.DataFrame],
) -> Dict[str, pd.DataFrame]:

    curves: Dict[str, pd.DataFrame] = {}

    if cfg.include_base_lcoe and base_curve is not None:
        curves["Base"] = base_curve

    if cfg.include_wen_ruhnau and cfg.wen_compare_lower and wen_lower_curve is not None:
        curves["WTP-Low"] = wen_lower_curve

    if cfg.include_wen_ruhnau and cfg.wen_compare_upper and wen_upper_curve is not None:
        curves["WTP-High"] = wen_upper_curve

    if cfg.include_frondel and frondel_curve is not None:
        curves["HedPrice-F"] = frondel_curve

    if cfg.include_heuer and heuer_curve is not None:
        curves["HedPrice-H"] = heuer_curve

    if cfg.include_krekel_zerrahn and krekel_zerrahn_curve is not None:
        curves["LifeSat"] = krekel_zerrahn_curve

    if cfg.include_tsani_proxy and tsani_proxy_curve is not None:
        curves["ExclPop"] = tsani_proxy_curve

    if cfg.include_tsani_scenicness and tsani_scenicness_curve is not None:
        curves["ExclSce"] = tsani_scenicness_curve

    return curves


def _plot_model_comparison(
    curves: Dict[str, pd.DataFrame],
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Optional[Path],
    show: bool,
    dpi: int,
) -> None:
    if not curves:
        print(f"[PLOT] no curves for {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for label, df in curves.items():
        if x_col not in df.columns or y_col not in df.columns:
            print(f"[WARN] skip {label}: missing {x_col} or {y_col}")
            continue

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        m = x.notna() & y.notna()

        if m.sum() == 0:
            print(f"[WARN] skip {label}: no finite data")
            continue

        ax.plot(x[m], y[m], label=label, **MODEL_STYLE.get(label, dict(linewidth=2.2)))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        print(f"[SAVE] plot -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _combined_curve_table(curves: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    keep_cols = [
        "step",
        "cum_gw",
        "cum_gw_total_system",
        "cum_twh_per_year",
        "cum_twh_per_year_total_system",
        "lcoe_base_pv_step_eur_per_mwh",
        "lcoe_total_pv_step_eur_per_mwh",
        "lcoe_base_pv_cum_eur_per_mwh",
        "lcoe_total_pv_cum_eur_per_mwh",
    ]

    for model, df in curves.items():
        sub = df[[c for c in keep_cols if c in df.columns]].copy()
        sub.insert(0, "model", model)
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Runner
# ============================================================

def run_compare_lcoe(
    cfg: CompareLCOEConfig,
    *,
    base_curve: Optional[pd.DataFrame] = None,
    wen_lower_curve: Optional[pd.DataFrame] = None,
    wen_upper_curve: Optional[pd.DataFrame] = None,
    frondel_curve: Optional[pd.DataFrame] = None,
    heuer_curve: Optional[pd.DataFrame] = None,
    krekel_zerrahn_curve: Optional[pd.DataFrame] = None,
    tsani_proxy_curve: Optional[pd.DataFrame] = None,
    tsani_scenicness_curve: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    base_curve = _standardize_curve_columns(_ensure_curve(base_curve, "base"), "base")
    wen_lower_curve = _standardize_curve_columns(_ensure_curve(wen_lower_curve, "wen_lower"), "wen_lower")
    wen_upper_curve = _standardize_curve_columns(_ensure_curve(wen_upper_curve, "wen_upper"), "wen_upper")
    frondel_curve = _standardize_curve_columns(_ensure_curve(frondel_curve, "frondel"), "frondel")
    heuer_curve = _standardize_curve_columns(_ensure_curve(heuer_curve, "heuer"), "heuer")
    krekel_zerrahn_curve = _standardize_curve_columns(_ensure_curve(krekel_zerrahn_curve, "krekel_zerrahn"), "krekel_zerrahn")
    tsani_proxy_curve = _standardize_curve_columns(_ensure_curve(tsani_proxy_curve, "tsani_proxy"), "tsani_proxy")
    tsani_scenicness_curve = _standardize_curve_columns(_ensure_curve(tsani_scenicness_curve, "tsani_scenicness"), "tsani_scenicness")

    if cfg.align_existing_offsets_to_base:
        aligned = _align_existing_offsets_to_base(
            base_curve=base_curve,
            curves={
                "base_curve": base_curve,
                "wen_lower_curve": wen_lower_curve,
                "wen_upper_curve": wen_upper_curve,
                "frondel_curve": frondel_curve,
                "heuer_curve": heuer_curve,
                "krekel_zerrahn_curve": krekel_zerrahn_curve,
                "tsani_proxy_curve": tsani_proxy_curve,
                "tsani_scenicness_curve": tsani_scenicness_curve,
            },
        )

        base_curve = aligned["base_curve"]
        wen_lower_curve = aligned["wen_lower_curve"]
        wen_upper_curve = aligned["wen_upper_curve"]
        frondel_curve = aligned["frondel_curve"]
        heuer_curve = aligned["heuer_curve"]
        krekel_zerrahn_curve = aligned["krekel_zerrahn_curve"]
        tsani_proxy_curve = aligned["tsani_proxy_curve"]
        tsani_scenicness_curve = aligned["tsani_scenicness_curve"]

    curves = _build_curve_dict(
        cfg,
        base_curve=base_curve,
        wen_lower_curve=wen_lower_curve,
        wen_upper_curve=wen_upper_curve,
        frondel_curve=frondel_curve,
        heuer_curve=heuer_curve,
        krekel_zerrahn_curve=krekel_zerrahn_curve,
        tsani_proxy_curve=tsani_proxy_curve,
        tsani_scenicness_curve=tsani_scenicness_curve,
    )

    print("[COMPARE] included curves:")
    for k, v in curves.items():
        print(f"  - {k}: {len(v):,} rows")

    if cfg.plot_cumulative and cfg.plot_capacity_axis:
        _plot_model_comparison(
            curves,
            x_col="cum_gw_total_system",
            y_col="lcoe_total_pv_cum_eur_per_mwh",
            title="Cumulative total system cost in €/MWh — model comparison",
            xlabel="Cumulative buildout incl. existing capacity (GW)",
            ylabel="EUR / MWh",
            out_path=cfg.output_dir / "compare_cumulative_total_lcoe_vs_capacity.png" if cfg.save_plots else None,
            show=cfg.show_plots,
            dpi=cfg.figure_dpi,
        )

    if cfg.plot_cumulative and cfg.plot_generation_axis:
        _plot_model_comparison(
            curves,
            x_col="cum_twh_per_year_total_system",
            y_col="lcoe_total_pv_cum_eur_per_mwh",
            title="Cumulative total system cost in €/MWh — model comparison",
            xlabel="Cumulative annual generation incl. existing generation (TWh/year)",
            ylabel="EUR / MWh",
            out_path=cfg.output_dir / "compare_cumulative_total_lcoe_vs_generation.png" if cfg.save_plots else None,
            show=cfg.show_plots,
            dpi=cfg.figure_dpi,
        )

    if cfg.plot_marginal and cfg.plot_capacity_axis:
        _plot_model_comparison(
            curves,
            x_col="cum_gw_total_system",
            y_col="lcoe_total_pv_step_eur_per_mwh",
            title="Marginal total system cost in €/MWh — model comparison",
            xlabel="Cumulative buildout incl. existing capacity (GW)",
            ylabel="EUR / MWh",
            out_path=cfg.output_dir / "compare_marginal_total_lcoe_vs_capacity.png" if cfg.save_plots else None,
            show=cfg.show_plots,
            dpi=cfg.figure_dpi,
        )

    if cfg.plot_marginal and cfg.plot_generation_axis:
        _plot_model_comparison(
            curves,
            x_col="cum_twh_per_year_total_system",
            y_col="lcoe_total_pv_step_eur_per_mwh",
            title="Marginal total system cost in €/MWh — model comparison",
            xlabel="Cumulative annual generation incl. existing generation (TWh/year)",
            ylabel="EUR / MWh",
            out_path=cfg.output_dir / "compare_marginal_total_lcoe_vs_generation.png" if cfg.save_plots else None,
            show=cfg.show_plots,
            dpi=cfg.figure_dpi,
        )

    combined = _combined_curve_table(curves)

    if cfg.save_combined_curve_table and not combined.empty:
        out_table = cfg.output_dir / "combined_lcoe_curves.parquet"
        combined.to_parquet(out_table, index=False)
        print(f"[SAVE] combined curves -> {out_table}")

    return {
        "curves": curves,
        "combined_curve_table": combined,
        "base_curve": base_curve,
        "wen_lower_curve": wen_lower_curve,
        "wen_upper_curve": wen_upper_curve,
        "frondel_curve": frondel_curve,
        "heuer_curve": heuer_curve,
        "krekel_zerrahn_curve": krekel_zerrahn_curve,
        "tsani_proxy_curve": tsani_proxy_curve,
        "tsani_scenicness_curve": tsani_scenicness_curve,
    }