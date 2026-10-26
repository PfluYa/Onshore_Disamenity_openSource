# -*- coding: utf-8 -*-
"""Heuer & Sommer (2026) — hedonic wind-disamenity scenarios.

Three separate model transfers (NOT estimation results provided by the authors):
  distance : a first turbine within 2 km causes a 2.1 log-point percent
             property-price reduction; subsequent turbines add no further loss.
             Extending the significant 1–2 km coefficient to 0–1 km is an
             explicit modelling assumption; there is no significant 0–1 km
             point estimate in the paper.
  intensity: treatment-intensity coefficients from Appendix Table A9 are
             converted to a monotone, saturating function by weighted NLS
             (or optionally monotonized by inverse-SE-weighted PAVA).
             Exposure is counted within 2 km. Effects plateau after 10
             turbines because the empirical last category is >=10.

  joint_conservative: directly uses *jointly estimated* Appendix A10 point
             coefficients for distance bands and count bins 1–3 vs >=4.
             Near band 0–1 km is conservatively assigned the 1–2 km loss;
             all property value of a 1-km residential cell is considered
             affected when ANY part of that cell intersects a distance
             band (minimum turbine-center-to-residential-cell distance).
             Unsignificant negative 2–3km point estimates are retained.
             Effects from distinct bands enter additively on the log-price
             scale, consistent with A10 Eq. (3). No empirical coefficient is
             inflated by a confidence interval. This is a modelling transfer.

All PV costs are *one-time capitalized* residential property-price losses.
Existing turbines initialize exposure, but existing property losses are not
booked as new-build costs. Old distance/intensity scenarios remain center-based;
the joint-conservative scenario uses cell-intersection exposure.

Unlike a lazy re-evaluation heap, this module updates ALL affected candidate
scores whenever a turbine is built, so each step uses the current true score.
The existing wind modules remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import heapq

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import convolve
from scipy.optimize import least_squares

from acceptance_frondel_04 import (
    build_index_maps,
    gdf_to_array_numeric,
    build_cell_engineering_arrays,
    add_remaining_potential_columns_variable_turbine,
    build_state_targets_b2045,
    build_state_additions_df,
    add_lcoe_metrics_pv_cumulative_variable,
    add_lcoe_metrics_pv_marginal_variable,
)


# Appendix Table A7, column (1): ln(price), 0-1 / 1-2 / 2-3 km.
HEUER_DISTANCE_BETAS = np.array([+0.014, -0.021, -0.012])
HEUER_DISTANCE_SE = np.array([0.017, 0.009, 0.008])
# Appendix Table A9, column (1): 1, 2, 3, 4, 5-9, >=10 turbines.
HEUER_INTENSITY_BETAS = np.array([-0.017, -0.020, -0.019, -0.032, -0.021, -0.033])
HEUER_INTENSITY_SE = np.array([0.011, 0.012, 0.011, 0.013, 0.014, 0.023])
HEUER_INTENSITY_LABELS = ("1", "2", "3", "4", "5-9", ">=10")

# Appendix Table A10 (model (3), log prices): first category 1–3 turbines,
# second >=4 turbines PER distance band (verified in main-paper Section 2.3).
# 0–1 km has +0.014 (SE 0.017), no separate count coefficients. We do NOT
# treat this nonsignificant positive estimate as an amenity: instead transfer
# the 1–2-km coefficients to the nearer band (a conservative modelling choice).
HEUER_JOINT_A10_ORIGINAL = {
    "0-1": {"reported_beta": +0.014, "reported_se": 0.017},
    "1-2": {"1-3": -0.020, ">=4": -0.021, "se_1-3": 0.009, "se_>=4": 0.011},
    "2-3": {"1-3": -0.012, ">=4": -0.013, "se_1-3": 0.008, "se_>=4": 0.011},
}
# [band, count category 0/1-3/>=4] -> log price coefficient
HEUER_JOINT_BETAS = np.array([
    [0., -0.020, -0.021],   # 0–1 km: conservative transfer from 1–2 km
    [0., -0.020, -0.021],   # 1–2 km: directly from A10
    [0., -0.012, -0.013],   # 2–3 km: negative but not significant
], dtype=float)



@dataclass
class HeuerConfig:
    input_grid_de_era5_parquet: Optional[Path] = None
    output_dir: Path = Path("Output_Main/placing_heuer")
    specification: str = "intensity"  # "distance", "intensity", "joint_conservative"
    allocation_mode: str = "total_lcoe"  # "total_lcoe" or "base_only"
    radius_km: float = 2.0
    use_existing_stock: bool = True
    state_id_col: str = "state_id"
    existing_gw_col: str = "existing_gw"
    existing_count_col: str = "n_existing"
    price_col: str = "price_eur_per_m2_used"
    floor_area_col: str = "area_total_m2"
    flh_col: str = "net_flh_era5"
    existing_flh_col: str = "net_flh_existing_era5"
    cf_col: str = "cf_era5"
    turbine_power_mw_col: str = "rated_power_mw_era5"
    turbine_power_kw_col: str = "rated_power_kw_era5"
    turbine_cost_col: str = "costs_era5"
    debug_every_steps: int = 1000
    # The model of Table A7 has no significant 0-1-km estimate: we explicitly
    # carry the significant 1-2-km loss over to the closer distance band.
    close_distance_policy: str = "extend_1to2_to_0to1"
    intensity_method: str = "saturating"  # relevant to old intensity only
    joint_cell_assignment: str = "intersection_nearest_band"


def weighted_isotonic_increasing(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators projection to nondecreasing loss coefficients."""
    vals = np.asarray(y, float)
    w = np.asarray(weights, float)
    if vals.ndim != 1 or vals.size != w.size or np.any(w <= 0):
        raise ValueError("Invalid isotonic inputs")
    blocks: List[dict] = []
    for i in range(vals.size):
        blocks.append({"start": i, "end": i + 1, "w": float(w[i]), "sum": float(w[i] * vals[i])})
        while len(blocks) > 1 and blocks[-2]["sum"] / blocks[-2]["w"] > blocks[-1]["sum"] / blocks[-1]["w"]:
            b = blocks.pop()
            a = blocks[-1]
            a["end"] = b["end"]
            a["sum"] += b["sum"]
            a["w"] += b["w"]
    result = np.zeros(vals.size, dtype=float)
    for b in blocks:
        result[b["start"]:b["end"]] = b["sum"] / b["w"]
    return result


def fit_saturating_intensity() -> Tuple[float, float]:
    """Inverse-SE-weighted least squares on the six A9 log-loss coefficients.

    We model log loss b(n)=a*(1-exp(-k*n)). Treatment bin 5-9 is assigned
    midpoint 7 and >=10 is assigned 10 (both explicit interpolation choices).
    The function is concave, saturating and avoids jumps between bins.
    """
    x = np.array([1., 2., 3., 4., 7., 10.])
    y = -HEUER_INTENSITY_BETAS
    se = HEUER_INTENSITY_SE
    res = least_squares(
        lambda p: (p[0] * (1. - np.exp(-p[1] * x)) - y) / se,
        x0=[0.03, 0.6], bounds=([0., 0.], [0.25, 10.]),
    )
    if not res.success:
        raise RuntimeError("Saturating intensity fit did not converge")
    return float(res.x[0]), float(res.x[1])


def build_loss_table(specification: str, intensity_method: str = "saturating") -> pd.DataFrame:
    """Return count->fraction table for n=0..10. n>10 uses n=10."""
    if specification == "distance":
        if HEUER_DISTANCE_BETAS[1] >= 0:
            raise ValueError("Unexpected positive effect in A7 1-2km")
        beta_by_count = np.array([0.] + [-0.021] * 10, float)
    elif specification == "intensity":
        if intensity_method == "saturating":
            # b(n)=a*(1-exp(-k*n)), n=1..10; no extrapolation beyond 10.
            a, k = fit_saturating_intensity()
            beta_by_count = -a * (1. - np.exp(-k * np.arange(11)))
        elif intensity_method == "isotonic":
            empirical_log_losses = -HEUER_INTENSITY_BETAS
            weights = 1 / HEUER_INTENSITY_SE ** 2
            fitted = weighted_isotonic_increasing(empirical_log_losses, weights)
            # Original categories are: 1,2,3,4,5-9,>=10.
            beta_by_count = np.array([0., *(-fitted[:4]), *([-fitted[4]] * 5), -fitted[5]], float)
        else:
            raise ValueError("intensity_method must be saturating or isotonic")
    else:
        raise ValueError("specification must be distance or intensity")
    fractions = 1. - np.exp(beta_by_count)
    if np.any(np.diff(fractions) < -1e-12):
        raise AssertionError("Loss fraction must be nondecreasing")
    return pd.DataFrame({"n_turbines": np.arange(11), "log_price_coefficient": beta_by_count,
                         "loss_fraction": fractions})


def circle_offsets(radius_km: float) -> List[Tuple[int, int]]:
    # Candidate distances measured using 1-km cell-center spacing.
    R = int(np.ceil(radius_km))
    return [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1)
            if dr * dr + dc * dc < radius_km * radius_km - 1e-12]


def count_existing_turbines(gdf: gpd.GeoDataFrame, cfg: HeuerConfig,
                            shared: Any) -> np.ndarray:
    if cfg.use_existing_stock and cfg.existing_count_col in gdf.columns:
        tmp = pd.to_numeric(gdf[cfg.existing_count_col], errors="coerce").fillna(0.).clip(lower=0.)
    elif cfg.use_existing_stock:
        mw = pd.to_numeric(gdf[cfg.existing_gw_col], errors="coerce").fillna(0.).clip(lower=0.) * 1000.
        tmp = (mw / float(shared.existing_proxy_mw_per_turbine)).round().clip(lower=0.)
    else:
        tmp = pd.Series(np.zeros(len(gdf)), index=gdf.index)
    temp = gdf[["x_km", "y_km"]].copy()
    temp["_existing_count"] = tmp.to_numpy(dtype=float)
    arr, _, _ = gdf_to_array_numeric(temp, "_existing_count")
    return np.nan_to_num(arr, nan=0).astype(np.int32)


def marginal_cost_at(r: int, c: int, counts: np.ndarray,
                     property_values: np.ndarray, loss_fracs: np.ndarray,
                     offsets: List[Tuple[int, int]]) -> float:
    ny, nx = counts.shape
    cost = 0.0
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < ny and 0 <= cc < nx:
            v = property_values[rr, cc]
            if v > 0.:
                n = min(10, int(counts[rr, cc]))
                cost += v * (loss_fracs[min(n + 1, 10)] - loss_fracs[n])
    return max(cost, 0.)


def affected_candidates(r: int, c: int, values: np.ndarray,
                        offsets: List[Tuple[int, int]]) -> set[Tuple[int, int]]:
    ny, nx = values.shape
    affected = set()
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < ny and 0 <= cc < nx) or values[rr, cc] <= 0:
            continue
        for dr2, dc2 in offsets:
            cand_r, cand_c = rr + dr2, cc + dc2
            if 0 <= cand_r < ny and 0 <= cand_c < nx:
                affected.add((cand_r, cand_c))
    return affected


def _stamp_row_meta(suffix: str, cfg: HeuerConfig, loss_table: pd.DataFrame) -> Dict[str, Any]:
    return {"model": "heuer_2026", "specification": cfg.specification,
            "intensity_method": cfg.intensity_method,
            "allocation_mode": cfg.allocation_mode, "radius_km": cfg.radius_km,
            "use_existing_stock": cfg.use_existing_stock, "appendix_table": "A7" if cfg.specification == "distance" else "A9",
            "policy_0_1km": cfg.close_distance_policy,
            "coefficients": str(loss_table.log_price_coefficient.round(8).tolist()),
            "source": "Heuer and Sommer (2026), Energy Economics 158, 109318"}


def allocate_iterative_heuer(gdf: gpd.GeoDataFrame, targets_df: pd.DataFrame,
                             cfg: HeuerConfig, shared: Any, target_new_build_gw: Optional[float]):
    mapping, ids, _xmin, _ymin, ny, nx = build_index_maps(gdf)
    vals, _, _ = gdf_to_array_numeric(gdf, "V_eur")
    vals = np.nan_to_num(vals, nan=0.)
    nmax, _, _ = gdf_to_array_numeric(gdf, "n_new_max")
    nmax = np.nan_to_num(nmax, nan=0).astype(np.int32)
    eng = build_cell_engineering_arrays(
        gdf=gdf, turbine_power_mw_col=cfg.turbine_power_mw_col,
        turbine_power_kw_col=cfg.turbine_power_kw_col, turbine_cost_col=cfg.turbine_cost_col,
        flh_col=cfg.flh_col, fixed_om_share_of_capex_per_year=shared.fixed_om_share_of_capex_per_year,
        lifetime_yrs=shared.lifetime_yrs, wacc=shared.wacc)
    power = eng["power_mw"]
    pv_base = eng["pv_base_eur_per_turbine"]
    pv_energy = eng["pv_energy_mwh_per_turbine"]
    ann_energy = eng["annual_mwh_per_turbine"]

    if "in_de" in gdf.columns:
        inde, _, _ = gdf_to_array_numeric(gdf, "in_de")
        inde = np.nan_to_num(inde, nan=0.) > 0.5
    else:
        inde = np.ones((ny, nx), dtype=bool)
    # No exposure cost for missing residential value; retain all true property
    # values even where there is no candidate turbine potential.
    vals[~inde] = 0.0
    existing_count = count_existing_turbines(gdf, cfg, shared)
    offsets = circle_offsets(cfg.radius_km)
    kernel = np.zeros((2 * int(np.ceil(cfg.radius_km)) + 1,) * 2, dtype=np.int32)
    mid = kernel.shape[0] // 2
    for dr, dc in offsets:
        kernel[mid + dr, mid + dc] = 1
    counts = convolve(existing_count, kernel, mode="constant", cval=0).astype(np.int32)
    initial_count_max = int(counts.max())
    table = build_loss_table(cfg.specification, cfg.intensity_method)
    lam = table["loss_fraction"].to_numpy()

    state = np.full((ny, nx), "", dtype="U4")
    for cell, sid in zip(gdf["idm"].astype(str), gdf[cfg.state_id_col].astype(str)):
        rc = mapping.get(cell)
        if rc is not None:
            state[rc] = sid
    cap_targets = dict(zip(targets_df.state_id.astype(str), targets_df.add_gw.astype(float)))
    cap_max = {k: v * (1.0 + float(shared.state_upper_slack)) for k, v in cap_targets.items()}
    built_states = {k: 0.0 for k in cap_targets}
    free_phase = not bool(shared.use_state_targets)
    def targets_met():
        return all(built_states.get(s, 0.) + shared.eps_rel >= v for s, v in cap_targets.items())
    def can_build(r, c):
        if not inde[r, c] or nmax[r, c] <= 0 or power[r, c] <= 0 or pv_energy[r, c] <= 0:
            return False
        if free_phase:
            return True
        sid = state[r, c]
        return sid in cap_max and built_states[sid] + power[r, c] / 1000. <= cap_max[sid] + shared.eps_rel

    valid = inde & (nmax > 0) & (power > 0) & (pv_energy > 0) & (pv_base > 0)
    scores_base = np.divide(pv_base, pv_energy, out=np.full_like(pv_base, np.inf), where=pv_energy > 0)
    # Fully initialized count-based marginal costs: convolution is exact for
    # the local, symmetrical disk kernel, avoids O(grid * neighbor loops).
    delta = lam[np.minimum(counts + 1, 10)] - lam[np.minimum(counts, 10)]
    init_costs = convolve(vals * delta, kernel.astype(float), mode="constant", cval=0)
    init_costs = np.maximum(init_costs, 0.)
    costs = init_costs
    if cfg.allocation_mode == "total_lcoe":
        scores = (pv_base + costs) / np.maximum(pv_energy, 1e-12)
    else:
        scores = scores_base.copy()
    scores[~valid] = np.inf

    versions = np.zeros((ny, nx), dtype=np.int32)
    rr, cc = np.where(valid)
    heap = [(float(scores[r, c]), 0, int(r), int(c)) for r, c in zip(rr, cc)]
    heapq.heapify(heap)
    deferred: set[Tuple[int, int]] = set()
    added_cost = np.zeros((ny, nx), dtype=float)
    new_turbines = np.zeros((ny, nx), dtype=np.int32)
    rows = []
    built_gw = 0.
    built_twh = 0.
    phase_switch_gw = np.nan
    phase_switch_twh = np.nan

    def push(r, c):
        versions[r, c] += 1
        if valid[r, c] and nmax[r, c] > 0:
            new_score = scores_base[r, c]
            if cfg.allocation_mode == "total_lcoe":
                new_score = (pv_base[r, c] + costs[r, c]) / pv_energy[r, c]
            heapq.heappush(heap, (float(new_score), int(versions[r, c]), int(r), int(c)))

    def switch_to_free():
        nonlocal free_phase, phase_switch_gw, phase_switch_twh
        free_phase = True
        phase_switch_gw = built_gw
        phase_switch_twh = built_twh
        for r, c in deferred:
            if nmax[r, c] > 0:
                push(r, c)
        deferred.clear()

    while heap or deferred:
        if shared.build_limit_mode == "total_gw" and target_new_build_gw is not None and built_gw + shared.eps_rel >= target_new_build_gw:
            break
        if not free_phase and targets_met():
            switch_to_free()
        if not heap:
            if not free_phase and deferred:
                switch_to_free()
                continue
            break
        score, stamp, r, c = heapq.heappop(heap)
        if stamp != versions[r, c] or nmax[r, c] <= 0:
            continue
        if not can_build(r, c):
            if not free_phase:
                deferred.add((r, c))
            continue
        take_gw = float(power[r, c]) / 1000.
        if shared.build_limit_mode == "total_gw" and target_new_build_gw is not None and built_gw + take_gw > target_new_build_gw + shared.eps_rel:
            # Preserve the original algorithm's no-overshooting policy.
            break
        take_cost = marginal_cost_at(r, c, counts, vals, lam, offsets)
        if not np.isclose(take_cost, costs[r, c], rtol=1e-9, atol=1e-4):
            raise AssertionError(f"Stale Heuer candidate score at {(r,c)}: {take_cost} vs {costs[r,c]}")
        old_count = int(counts[r, c])
        for dr, dc in offsets:
            rj, cj = r + dr, c + dc
            if 0 <= rj < ny and 0 <= cj < nx:
                counts[rj, cj] += 1
        added_cost[r, c] += take_cost
        new_turbines[r, c] += 1
        nmax[r, c] -= 1
        built_gw += take_gw
        take_mwh = float(ann_energy[r, c])
        built_twh += take_mwh / 1e6
        sid = state[r, c]
        if sid:
            built_states[sid] = built_states.get(sid, 0.) + take_gw
        rows.append(dict(
            phase=("free_after_state_targets" if shared.use_state_targets and free_phase else
                   ("national" if not shared.use_state_targets else "state_target_phase")),
            state_id=sid, idm=ids[r * nx + c], take_turbines=1,
            take_mw=float(power[r, c]), take_gw=take_gw,
            take_mwh_per_year=take_mwh, take_twh_per_year=take_mwh / 1e6,
            take_cost_eur=float(take_cost), mc_total_at_pick=float(take_cost),
            score_active=float(score), score_base_pv_lcoe=float(scores_base[r, c]),
            score_total_pv_lcoe_at_pick=float((pv_base[r, c] + take_cost) / pv_energy[r, c]),
            pv_base_cost_eur_step=float(pv_base[r, c]),
            pv_energy_mwh_step=float(pv_energy[r, c]),
            exposure_count_at_candidate_before=old_count,
            built_gw_total_after=built_gw,
            built_gw_state_after=built_states.get(sid, 0.),
        ))
        # Every candidate that shares a relevant residential exposure cell
        # with the new turbine is repriced immediately (exact greedy priority).
        candidates = affected_candidates(r, c, vals, offsets)
        candidates.add((r, c))
        for cr, ccol in candidates:
            if valid[cr, ccol] and nmax[cr, ccol] > 0:
                costs[cr, ccol] = marginal_cost_at(cr, ccol, counts, vals, lam, offsets)
                push(cr, ccol)
        if cfg.debug_every_steps and len(rows) % cfg.debug_every_steps == 0:
            print(f"[HEUER] {cfg.specification}/{cfg.allocation_mode}: step {len(rows):,}, new GW={built_gw:.2f}, queue={len(heap):,}")

    placement = pd.DataFrame(dict(idm=ids, n_new=new_turbines.ravel(),
             added_gw=(new_turbines * power / 1000.).ravel(),
             added_cost_eur=added_cost.ravel()))
    placement = placement.loc[placement.idm != ""].reset_index(drop=True)
    if rows:
        curve = pd.DataFrame(rows)
        curve["step"] = np.arange(1, len(curve) + 1)
        curve["cum_gw"] = curve.take_gw.cumsum()
        curve["cum_twh_per_year"] = curve.take_twh_per_year.cumsum()
        curve["cum_cost_eur"] = curve.take_cost_eur.cumsum()
    else:
        curve = pd.DataFrame()
    state_rep = targets_df.copy()
    state_rep["target_gw_phaseA"] = state_rep.add_gw.astype(float)
    state_rep["max_gw"] = state_rep.add_gw * (1.0 + shared.state_upper_slack)
    state_rep["built_gw"] = state_rep.state_id.astype(str).map(built_states).fillna(0.)
    state_rep["gap_to_target_gw_phaseA"] = (state_rep.target_gw_phaseA - state_rep.built_gw).clip(lower=0)
    meta = dict(phase_switch_gw=phase_switch_gw, phase_switch_twh_per_year=phase_switch_twh,
                built_total_gw=built_gw, built_total_twh_per_year=built_twh,
                n_turbines=len(rows), initial_count_max=initial_count_max, final_count_max=int(counts.max()),
                initial_stock_turbines=int(existing_count.sum()))
    return placement, curve, state_rep, meta


def run_heuer(cfg: HeuerConfig, shared: Any) -> Dict[str, Any]:
    if cfg.input_grid_de_era5_parquet is None:
        raise ValueError("Heuer input_grid_de_era5_parquet must be set")
    if cfg.specification not in {"distance", "intensity", "joint_conservative"}:
        raise ValueError("Heuer specification must be distance, intensity or joint_conservative")
    if cfg.allocation_mode not in {"base_only", "total_lcoe"}:
        raise ValueError("Heuer allocation_mode must be base_only or total_lcoe")
    if cfg.close_distance_policy != "extend_1to2_to_0to1":
        raise ValueError("Only explicitly documented close-distance policy is implemented")
    expected_radius = 3.0 if cfg.specification == "joint_conservative" else 2.0
    if not np.isclose(cfg.radius_km, expected_radius):
        raise ValueError(f"{cfg.specification} requires radius_km={expected_radius}")
    if cfg.specification == "joint_conservative" and cfg.joint_cell_assignment != "intersection_nearest_band":
        raise ValueError("Joint exposure requires intersection_nearest_band")

    print(f"[HEUER] {cfg.specification} | {cfg.allocation_mode} | input={cfg.input_grid_de_era5_parquet}")
    gdf = gpd.read_parquet(cfg.input_grid_de_era5_parquet)
    required = ["idm", "x_km", "y_km", cfg.price_col, cfg.floor_area_col,
                cfg.existing_gw_col, cfg.flh_col, cfg.turbine_power_mw_col,
                cfg.turbine_power_kw_col, cfg.turbine_cost_col, "pot_area_km2", cfg.state_id_col]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Heuer input missing columns: {missing}")
    if gdf.idm.duplicated().any():
        raise ValueError("Heuer input grid contains duplicate idm")
    gdf["V_eur"] = (pd.to_numeric(gdf[cfg.price_col], errors="coerce").fillna(0).clip(lower=0) *
                    pd.to_numeric(gdf[cfg.floor_area_col], errors="coerce").fillna(0).clip(lower=0))
    if "n_new_max" not in gdf.columns:
        gdf = add_remaining_potential_columns_variable_turbine(
            gdf=gdf, pot_mw_per_km2=shared.pot_mw_per_km2,
            turbine_power_mw_col=cfg.turbine_power_mw_col,
            existing_gw_col=cfg.existing_gw_col,
            existing_proxy_mw_per_turbine=shared.existing_proxy_mw_per_turbine,
            use_existing_stock=cfg.use_existing_stock)
    if cfg.use_existing_stock:
        gdf["existing_gw_cell"] = pd.to_numeric(gdf[cfg.existing_gw_col], errors="coerce").fillna(0.)
    else:
        gdf["existing_gw_cell"] = 0.
    existing_gw = float(gdf.existing_gw_cell.sum())
    if cfg.existing_flh_col in gdf.columns:
        old_flh = pd.to_numeric(gdf[cfg.existing_flh_col], errors="coerce").fillna(0.)
    else:
        old_flh = pd.to_numeric(gdf[cfg.flh_col], errors="coerce").fillna(0.)
    existing_twh = float((gdf.existing_gw_cell * 1000. * old_flh / 1e6).sum())

    targets = build_state_additions_df(gdf, build_state_targets_b2045(),
                cfg.state_id_col, cfg.existing_gw_col, cfg.use_existing_stock)
    if shared.build_limit_mode == "total_gw":
        if shared.target_total_gw is None:
            raise ValueError("Missing target_total_gw")
        new_gw = max(0., float(shared.target_total_gw) - existing_gw)
    else:
        new_gw = None
    if cfg.specification == "joint_conservative":
        placement, curve, state_report, meta = allocate_iterative_heuer_joint(gdf, targets, cfg, shared, new_gw)
    else:
        placement, curve, state_report, meta = allocate_iterative_heuer(gdf, targets, cfg, shared, new_gw)
    if not curve.empty:
        curve = add_lcoe_metrics_pv_marginal_variable(curve)
        curve = add_lcoe_metrics_pv_cumulative_variable(curve)
        curve["cum_gw_total_system"] = curve.cum_gw + existing_gw
        curve["cum_twh_per_year_total_system"] = curve.cum_twh_per_year + existing_twh
        curve["existing_total_gw"] = existing_gw
        curve["existing_total_twh_per_year"] = existing_twh
    out_gdf = gdf.merge(placement, on="idm", how="left", validate="one_to_one")
    for name, val in [("n_new", 0), ("added_gw", 0.), ("added_cost_eur", 0.)]:
        out_gdf[name] = out_gdf[name].fillna(val)
    if cfg.specification == "joint_conservative":
        loss_table = joint_parameter_table()
        meta.update({
            "model": "heuer_2026", "specification": "joint_conservative",
            "algorithm_version": "joint_a10_heap_reinsert_v2",
            "allocation_mode": cfg.allocation_mode, "use_existing_stock": cfg.use_existing_stock,
            "radius_km": 3.0, "appendix_table": "A10",
            "joint_coefficients_log_price": str(HEUER_JOINT_BETAS.tolist()),
            "policy_0_1km": "transfer_A10_1to2_by_count",
            "cell_exposure_policy": cfg.joint_cell_assignment,
            "non_significant_2to3_included": True,
            "source": "Heuer and Sommer (2026), Energy Economics 158, 109318",
        })
    else:
        loss_table = build_loss_table(cfg.specification, cfg.intensity_method)
        meta.update(_stamp_row_meta("", cfg, loss_table))
    meta_df = pd.DataFrame([meta])
    suffix = (f"{'with_states' if shared.use_state_targets else 'national'}_"
              f"{shared.build_limit_mode}_{cfg.allocation_mode}_"
              f"{'with_existing' if cfg.use_existing_stock else 'without_existing'}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"heuer_{cfg.specification}"
    out_gdf.to_parquet(cfg.output_dir / f"placement_cells_iterative_{prefix}_{suffix}.parquet", index=False)
    curve.to_parquet(cfg.output_dir / f"cost_curve_iterative_{prefix}_{suffix}.parquet", index=False)
    state_report.to_parquet(cfg.output_dir / f"state_report_iterative_{prefix}_{suffix}.parquet", index=False)
    meta_df.to_parquet(cfg.output_dir / f"meta_{prefix}_{suffix}.parquet", index=False)
    loss_table.to_csv(cfg.output_dir / f"parameters_{prefix}.csv", index=False)
    print(f"[HEUER] built new={meta['built_total_gw']:.3f} GW / {meta['n_turbines']:,} turbines; saved to {cfg.output_dir}")
    return dict(placement_cells=out_gdf, cost_curve=curve, state_report=state_report,
                meta=meta_df, parameters=loss_table)


# ============================================================
# A10 JOINT distance x intensity, conservative raster exposure
# ============================================================

def joint_parameter_table() -> pd.DataFrame:
    """Export the original A10 and transferred beta values transparently."""
    rows = []
    for band, title in enumerate(("0-1", "1-2", "2-3")):
        source = HEUER_JOINT_A10_ORIGINAL[title]
        for category, idx in (("0", 0), ("1-3", 1), (">=4", 2)):
            original_beta = source.get(category, np.nan)
            original_se = source.get("se_" + category, np.nan)
            if band == 0 and idx > 0:
                original_beta = source["reported_beta"]
                original_se = source["reported_se"]
            rows.append(dict(distance_band_km=title, count_category=category,
                             source_table="A10", a10_reported_beta=original_beta,
                             a10_reported_se=original_se,
                             implemented_log_price_beta=HEUER_JOINT_BETAS[band, idx],
                             transfer_rule=("0-1 band uses 1-2 beta; A10 +0.014 omitted"
                                            if band == 0 else "A10 point estimate"),
                             cell_exposure="whole residential cell if circle intersects"))
    return pd.DataFrame(rows)


def joint_band_offsets(radius_km: float = 3.0) -> Tuple[List[Tuple[int, int]], ...]:
    """Exact cell-intersection assignment; closest band gets each cell.

    Turbine is at CENTER of 1km x 1km candidate cell. Residential property
    may be anywhere inside its 1km x 1km cell. Minimum point-to-square
    distance is used: this overstates exposure for cells only partly covered.
    Each residential cell belongs to exactly one band for each turbine.
    """
    if not np.isclose(radius_km, 3.0):
        raise ValueError("Joint A10 transfer requires a 3-km radius")
    offsets: List[List[Tuple[int, int]]] = [[], [], []]
    R = int(np.ceil(radius_km + 0.5))
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            nearest = np.hypot(max(abs(dr) - 0.5, 0.0),
                               max(abs(dc) - 0.5, 0.0))
            if nearest <= 1.0 + 1e-12:
                offsets[0].append((dr, dc))
            elif nearest <= 2.0 + 1e-12:
                offsets[1].append((dr, dc))
            elif nearest <= 3.0 + 1e-12:
                offsets[2].append((dr, dc))
    return tuple(offsets)


def joint_beta_for_count(band: int, counts: np.ndarray) -> np.ndarray:
    """A10 total effect in log prices, constant inside each count category."""
    cat = np.where(np.asarray(counts) == 0, 0,
                   np.where(np.asarray(counts) <= 3, 1, 2))
    return HEUER_JOINT_BETAS[band, cat]


def joint_beta_step(band: int, counts: np.ndarray) -> np.ndarray:
    """Change in log prices after ONE more turbine in the same band."""
    x = np.asarray(counts)
    return joint_beta_for_count(band, x + 1) - joint_beta_for_count(band, x)


def joint_total_beta(counts: np.ndarray) -> np.ndarray:
    if counts.shape[0] != 3:
        raise ValueError("Joint count array should have first dimension 3")
    return sum(joint_beta_for_count(b, counts[b]) for b in range(3))


def joint_marginal_cost_many(cand_r: np.ndarray, cand_c: np.ndarray,
                             counts: np.ndarray, current_beta: np.ndarray,
                             values: np.ndarray,
                             bands: Tuple[List[Tuple[int, int]], ...]) -> np.ndarray:
    """Exact batch computation: V * [exp(B_old) - exp(B_new)].

    B sums contributions of the three jointly estimated distance bands in
    log-price units. Band-specific candidate additions change their count
    group only on 0->1 and 3->4 transitions. For every new turbine,
    marginal property loss is nonnegative.
    """
    rr = np.asarray(cand_r, dtype=np.intp).ravel()
    cc = np.asarray(cand_c, dtype=np.intp).ravel()
    ny, nx = values.shape
    result = np.zeros(rr.size, dtype=float)
    for b, offsets in enumerate(bands):
        if not offsets:
            continue
        dr = np.array([i for i, j in offsets], dtype=np.intp)
        dc = np.array([j for i, j in offsets], dtype=np.intp)
        rj = rr[:, None] + dr[None, :]
        cj = cc[:, None] + dc[None, :]
        mask = (rj >= 0) & (rj < ny) & (cj >= 0) & (cj < nx)
        rj = np.clip(rj, 0, ny-1)
        cj = np.clip(cj, 0, nx-1)
        incr_beta = joint_beta_step(b, counts[b][rj, cj])
        delta_loss = np.exp(current_beta[rj, cj]) * (-np.expm1(incr_beta))
        result += (values[rj, cj] * delta_loss * mask).sum(axis=1)
    return np.maximum(result, 0.)


def allocate_iterative_heuer_joint(gdf: gpd.GeoDataFrame, targets_df: pd.DataFrame,
                                   cfg: HeuerConfig, shared: Any,
                                   target_new_build_gw: Optional[float]):
    """Exact per-step greedy score, fully updated for dynamic A10 exposures."""
    mapping, ids, xmin, ymin, ny, nx = build_index_maps(gdf)
    values, _, _ = gdf_to_array_numeric(gdf, "V_eur")
    values = np.nan_to_num(values, nan=0.0)
    nmax, _, _ = gdf_to_array_numeric(gdf, "n_new_max")
    nmax = np.nan_to_num(nmax, nan=0.).astype(np.int32)
    eng = build_cell_engineering_arrays(
        gdf=gdf, turbine_power_mw_col=cfg.turbine_power_mw_col,
        turbine_power_kw_col=cfg.turbine_power_kw_col,
        turbine_cost_col=cfg.turbine_cost_col, flh_col=cfg.flh_col,
        fixed_om_share_of_capex_per_year=shared.fixed_om_share_of_capex_per_year,
        lifetime_yrs=shared.lifetime_yrs, wacc=shared.wacc)
    power = eng["power_mw"]
    pv_base = eng["pv_base_eur_per_turbine"]
    pv_energy = eng["pv_energy_mwh_per_turbine"]
    annual_energy = eng["annual_mwh_per_turbine"]
    if "in_de" in gdf.columns:
        inde, _, _ = gdf_to_array_numeric(gdf, "in_de")
        inde = np.nan_to_num(inde, nan=0.) > 0.5
    else:
        inde = np.ones((ny, nx), dtype=bool)
    values[~inde] = 0.0

    offsets = joint_band_offsets(cfg.radius_km)
    kernels = []
    for lst in offsets:
        kernel = np.zeros((9, 9), dtype=np.int32)
        for dr, dc in lst:
            kernel[4 + dr, 4 + dc] = 1
        kernels.append(kernel)
    existing_count = count_existing_turbines(gdf, cfg, shared)
    counts = np.stack([convolve(existing_count, ker, mode="constant", cval=0)
                       for ker in kernels]).astype(np.int32)
    B = joint_total_beta(counts)
    Vexp = values * np.exp(B)
    costs = np.zeros_like(values)
    for b, ker in enumerate(kernels):
        marginal_fraction = -np.expm1(joint_beta_step(b, counts[b]))
        costs += convolve(Vexp * marginal_fraction, ker.astype(float),
                          mode="constant", cval=0.)
    costs = np.maximum(costs, 0.)

    states = np.full((ny, nx), "", dtype="U8")
    for cell_id, sid in zip(gdf["idm"].astype(str), gdf[cfg.state_id_col].astype(str)):
        rc = mapping.get(cell_id)
        if rc is not None:
            states[rc] = sid
    targets = dict(zip(targets_df.state_id.astype(str), targets_df.add_gw.astype(float)))
    maximum = {k: v * (1.0 + float(shared.state_upper_slack)) for k, v in targets.items()}
    built_by_state = {k: 0.0 for k in targets}
    free_phase = not bool(shared.use_state_targets)

    def all_targets_met():
        return all(built_by_state.get(s, 0.) + shared.eps_rel >= v
                   for s, v in targets.items())

    def can_build(r, c):
        if not inde[r, c] or nmax[r, c] <= 0 or power[r, c] <= 0 or pv_energy[r, c] <= 0:
            return False
        if free_phase:
            return True
        sid = states[r, c]
        return sid in maximum and built_by_state[sid] + power[r, c] / 1000. <= maximum[sid] + shared.eps_rel

    valid = inde & (nmax > 0) & (power > 0) & (pv_energy > 0) & (pv_base > 0)
    score_base = np.divide(pv_base, pv_energy, out=np.full_like(pv_base, np.inf), where=pv_energy > 0)
    versions = np.zeros((ny, nx), dtype=np.int32)
    ri, ci = np.where(valid)
    if cfg.allocation_mode == "total_lcoe":
        scores = (pv_base + costs) / np.maximum(pv_energy, 1e-12)
    else:
        scores = score_base
    heap = [(float(scores[r, c]), 0, int(r), int(c)) for r, c in zip(ri, ci)]
    heapq.heapify(heap)
    deferred: set[Tuple[int, int]] = set()
    added_cost = np.zeros((ny, nx), dtype=float)
    new_turbines = np.zeros((ny, nx), dtype=np.int32)
    rows = []
    built_gw = 0.0
    built_twh = 0.0
    phase_switch_gw = np.nan
    phase_switch_twh = np.nan

    def push(r, c):
        versions[r, c] += 1
        if valid[r, c] and nmax[r, c] > 0:
            cost = costs[r, c] if cfg.allocation_mode == "total_lcoe" else 0.
            sc = (pv_base[r, c] + cost) / pv_energy[r, c]
            heapq.heappush(heap, (float(sc), int(versions[r, c]), int(r), int(c)))

    def switch_to_free():
        nonlocal free_phase, phase_switch_gw, phase_switch_twh
        free_phase = True
        phase_switch_gw = built_gw
        phase_switch_twh = built_twh
        for r, c in deferred:
            if nmax[r, c] > 0:
                push(r, c)
        deferred.clear()

    while heap or deferred:
        if (shared.build_limit_mode == "total_gw" and target_new_build_gw is not None
                and built_gw + shared.eps_rel >= target_new_build_gw):
            break
        if not free_phase and all_targets_met():
            switch_to_free()
        if not heap:
            if not free_phase and deferred:
                switch_to_free()
                continue
            break
        score, stamp, r, c = heapq.heappop(heap)
        if stamp != versions[r, c] or nmax[r, c] <= 0:
            continue
        if not can_build(r, c):
            if not free_phase:
                deferred.add((r, c))
            continue
        take_gw = float(power[r, c]) / 1000.
        if (shared.build_limit_mode == "total_gw" and target_new_build_gw is not None
                and built_gw + take_gw > target_new_build_gw + shared.eps_rel):
            break  # original no-overshoot policy
        take_cost = float(joint_marginal_cost_many(np.array([r]), np.array([c]),
                                                  counts, B, values, offsets)[0])
        if cfg.allocation_mode == "total_lcoe" and not np.isclose(
                take_cost, costs[r, c], rtol=1e-9, atol=1e-3):
            raise AssertionError(f"Stale joint cost at {(r,c)}: {take_cost} vs {costs[r,c]}")
        count_before = counts[:, r, c].copy()
        # Update all exposed residential cells, tracking count by distance band.
        for b, lst in enumerate(offsets):
            for dr, dc in lst:
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx:
                    old_count = counts[b, rr, cc]
                    counts[b, rr, cc] += 1
                    B[rr, cc] += (joint_beta_for_count(b, counts[b, rr, cc]) -
                                  joint_beta_for_count(b, old_count))

        added_cost[r, c] += take_cost
        new_turbines[r, c] += 1
        nmax[r, c] -= 1
        built_gw += take_gw
        take_mwh = float(annual_energy[r, c])
        built_twh += take_mwh / 1e6
        sid = states[r, c]
        if sid:
            built_by_state[sid] = built_by_state.get(sid, 0.) + take_gw
        rows.append(dict(
            phase=("free_after_state_targets" if shared.use_state_targets and free_phase
                   else ("national" if not shared.use_state_targets else "state_target_phase")),
            state_id=sid, idm=ids[r * nx + c], take_turbines=1,
            take_mw=float(power[r, c]), take_gw=take_gw,
            take_mwh_per_year=take_mwh, take_twh_per_year=take_mwh / 1e6,
            take_cost_eur=take_cost, mc_total_at_pick=take_cost,
            score_active=float(score), score_base_pv_lcoe=float(score_base[r, c]),
            score_total_pv_lcoe_at_pick=float((pv_base[r, c] + take_cost) / pv_energy[r, c]),
            pv_base_cost_eur_step=float(pv_base[r, c]),
            pv_energy_mwh_step=float(pv_energy[r, c]),
            exposure_count_at_candidate_before=int(count_before.sum()),
            exposure_0_1_before=int(count_before[0]),
            exposure_1_2_before=int(count_before[1]),
            exposure_2_3_before=int(count_before[2]),
            built_gw_total_after=built_gw, built_gw_state_after=built_by_state.get(sid, 0.),
        ))
        if cfg.allocation_mode == "total_lcoe":
            # Only residential cells in the 3-km intersecting disk can change.
            changed_candidates = affected_candidates(r, c, values,
                                 [pair for group in offsets for pair in group])
            changed_candidates.add((r, c))
            cand = [(rr, cc) for rr, cc in changed_candidates
                    if valid[rr, cc] and nmax[rr, cc] > 0]
            if cand:
                crr, ccc = np.array(cand, dtype=np.intp).T
                new_costs = joint_marginal_cost_many(crr, ccc, counts, B, values, offsets)
                for (rr, cc), new_cost in zip(cand, new_costs):
                    cost_changed = not np.isclose(
                        new_cost, costs[rr, cc], rtol=1e-11, atol=1e-5
                    )
                    costs[rr, cc] = new_cost
                    # CRITICAL: the chosen turbine cell was popped from the
                    # priority queue. If it still has capacity, it MUST be
                    # reinserted even when its marginal disamenity cost has
                    # not changed. Other candidate cells only need updating
                    # when their marginal costs have changed.
                    if cost_changed or (rr == r and cc == c):
                        push(rr, cc)
        else:
            # In base-only mode exposure affects reported ex-post costs but
            # never ranking. Reinsert the selected cell at its base LCOE
            # if it can accommodate additional turbines.
            if valid[r, c] and nmax[r, c] > 0:
                push(r, c)
        if cfg.debug_every_steps and len(rows) % cfg.debug_every_steps == 0:
            print(f"[HEUER JOINT] {cfg.allocation_mode}: step {len(rows):,}, new GW={built_gw:.2f}, queue={len(heap):,}")

    placement = pd.DataFrame(dict(idm=ids, n_new=new_turbines.ravel(),
        added_gw=(new_turbines * power / 1000.).ravel(), added_cost_eur=added_cost.ravel()))
    placement = placement.loc[placement.idm != ""].reset_index(drop=True)
    if rows:
        curve = pd.DataFrame(rows)
        curve["step"] = np.arange(1, len(curve) + 1)
        curve["cum_gw"] = curve.take_gw.cumsum()
        curve["cum_twh_per_year"] = curve.take_twh_per_year.cumsum()
        curve["cum_cost_eur"] = curve.take_cost_eur.cumsum()
    else:
        curve = pd.DataFrame()
    state_rep = targets_df.copy()
    state_rep["target_gw_phaseA"] = state_rep.add_gw.astype(float)
    state_rep["max_gw"] = state_rep.add_gw * (1.0 + shared.state_upper_slack)
    state_rep["built_gw"] = state_rep.state_id.astype(str).map(built_by_state).fillna(0.)
    state_rep["gap_to_target_gw_phaseA"] = (state_rep.target_gw_phaseA - state_rep.built_gw).clip(lower=0)
    meta = dict(phase_switch_gw=phase_switch_gw,
                phase_switch_twh_per_year=phase_switch_twh,
                built_total_gw=built_gw, built_total_twh_per_year=built_twh,
                n_turbines=len(rows),
                existing_stock_turbines=int(existing_count.sum()),
                initial_count_max_by_band=str([int(a.max()) for a in
                                           [convolve(existing_count, k, mode="constant", cval=0) for k in kernels]]),
                final_count_max_by_band=str([int(counts[i].max()) for i in range(3)]),
                offset_cells_by_band=str([len(a) for a in offsets]),
                joint_cell_policy="nearest point of residential square; full V_j",
                implementation_betas=str(HEUER_JOINT_BETAS.tolist()))
    return placement, curve, state_rep, meta
