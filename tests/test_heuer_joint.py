# -*- coding: utf-8 -*-
"""Run with `python test_heuer_joint.py`; needs numpy, pandas, geopandas, scipy."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import convolve
from acceptance_heuer_09 import (
    HEUER_JOINT_BETAS, joint_band_offsets, joint_beta_for_count,
    joint_beta_step, joint_total_beta, joint_marginal_cost_many,
    allocate_iterative_heuer_joint, HeuerConfig,
)


def test_parameters_and_bands():
    assert HEUER_JOINT_BETAS.shape == (3, 3)
    assert np.all(HEUER_JOINT_BETAS[:, 1:] <= 0)
    bands = joint_band_offsets()
    assert [len(b) for b in bands] == [9, 12, 24], list(map(len, bands))
    pair_band = {pair: k for k, group in enumerate(bands) for pair in group}
    assert len(pair_band) == 45, 'no double assignment'
    assert pair_band[(0, 0)] == 0
    assert pair_band[(1, 0)] == 0  # cell intersects 1km disk although centre is 1 km
    assert pair_band[(2, 0)] == 1
    assert pair_band[(3, 0)] == 2
    assert (4, 0) not in pair_band
    # category jumps: 0->1 and 3->4, no costs for 1->2 or 2->3
    for b in range(3):
        step = joint_beta_step(b, np.arange(7))
        assert step[0] < 0 and step[3] < 0
        assert np.allclose(step[[1, 2, 4, 5, 6]], 0.)
        assert np.all(step <= 0.)


def test_marginal_exact_telescopes():
    bands = joint_band_offsets()
    ny = nx = 11
    counts = np.zeros((3, ny, nx), dtype=np.int32)
    counts[1, 5, 7] = 3
    counts[2, 5, 8] = 1
    V = np.zeros((ny, nx))
    V[5, 5] = 5e5
    V[5, 7] = 1e6
    V[5, 8] = 2e6
    B = joint_total_beta(counts)
    old_welfare = np.sum(V * (1. - np.exp(B)))
    cand = (5, 5)
    candidate_cost = joint_marginal_cost_many(np.array([cand[0]]), np.array([cand[1]]),
                                               counts, B, V, bands)[0]
    new_counts = counts.copy()
    for b, offs in enumerate(bands):
        for dr, dc in offs:
            rr, cc = cand[0]+dr, cand[1]+dc
            if 0 <= rr < ny and 0 <= cc < nx:
                new_counts[b, rr, cc] += 1
    new_welfare = np.sum(V * (1. - np.exp(joint_total_beta(new_counts))))
    assert np.isclose(candidate_cost, new_welfare - old_welfare, rtol=1e-12)
    assert candidate_cost >= 0.
    # checking that existing turbines do not cause a cost at initial state
    none = np.zeros_like(V)
    assert np.allclose(joint_marginal_cost_many(np.array([5]), np.array([5]),
                                              counts, B, none, bands), 0)


def make_synthetic_grid():
    rows = []
    for y in range(9):
        for x in range(9):
            # three feasible cells for comparable greedy re-ranking
            candidate = (x, y) in {(2, 3), (4, 4), (6, 6), (5, 2)}
            households_value = 1e6 if (2 <= x <= 6 and 2 <= y <= 6) else 0.
            rows.append(dict(idm=f'{x}_{y}', x_km=x, y_km=y,
                             in_de=1, price_eur_per_m2_used=1000.,
                             area_total_m2=households_value / 1000.,
                             existing_gw=0., n_existing=int((x, y) == (2, 2)),
                             n_new_max=int(candidate), pot_area_km2=0.5,
                             state_id='01', net_flh_era5=2500. + 80.*x,
                             rated_power_mw_era5=4., rated_power_kw_era5=4000.,
                             costs_era5=1550., geometry=Point(x*1000., y*1000.)))
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:3035')


def test_small_allocator_exact_greedy():
    gdf = make_synthetic_grid()
    gdf['V_eur'] = gdf.price_eur_per_m2_used * gdf.area_total_m2
    shared = SimpleNamespace(
        pot_mw_per_km2=25., existing_proxy_mw_per_turbine=2.5,
        state_upper_slack=0.5, use_state_targets=False, eps_rel=1e-12,
        build_limit_mode='total_gw', lifetime_yrs=20, wacc=.06,
        fixed_om_share_of_capex_per_year=.025)
    cfg = HeuerConfig(specification='joint_conservative', radius_km=3.0,
                      allocation_mode='total_lcoe', debug_every_steps=0)
    targets = pd.DataFrame({'state_id':['01'], 'add_gw':[1.]})
    placement, curve, report, meta = allocate_iterative_heuer_joint(
        gdf, targets, cfg, shared, .016)
    assert len(curve) == 4
    assert np.isclose(placement.added_gw.sum(), .016)
    assert np.all(curve.take_cost_eur >= 0.)
    assert np.isclose(curve.take_cost_eur.sum(), placement.added_cost_eur.sum())

    # Independent exhaustive cheapest-site calculation at each step,
    # rebuilding counts and evaluating every candidate (no heap).
    bands = joint_band_offsets()
    stock = np.zeros((9, 9), dtype=np.int32)
    stock[2, 2] = 1
    kern = []
    for offs in bands:
        k = np.zeros((9, 9), np.int32)
        for dr, dc in offs: k[4+dr, 4+dc] = 1
        kern.append(k)
    V = np.zeros((9, 9))
    for _, r in gdf.iterrows(): V[int(r.y_km), int(r.x_km)] = r.V_eur
    cs = [(3, 2), (4, 4), (6, 6), (2, 5)]
    chosen = []
    af = (1-1.06**-20)/.06
    for _ in range(4):
        cnt = np.stack([convolve(stock, k, mode='constant', cval=0) for k in kern]); B = joint_total_beta(cnt)
        ranking = []
        for r, c in cs:
            row = gdf.loc[(gdf.y_km==r)&(gdf.x_km==c)].iloc[0]
            pve = row.rated_power_mw_era5*row.net_flh_era5*af
            pvc = row.rated_power_kw_era5*row.costs_era5*(1 + .025*af)
            cost = joint_marginal_cost_many(np.array([r]), np.array([c]), cnt, B, V, bands)[0]
            ranking.append(((pvc+cost)/pve,r,c))
        _, r, c = min(ranking)
        chosen.append(f'{c}_{r}')
        stock[r, c] += 1
        cs.remove((r,c))
    assert curve.idm.tolist() == chosen, (curve.idm.tolist(),chosen)



def test_repeated_placements_and_base_only_invariance():
    """Regression test for re-inserting an already chosen but still feasible site.

    Base-only must reproduce exactly the older Heuer algorithm irrespective of
    exposure coefficients. Total-LCOE must stay globally greedy with repeated
    builds per candidate cell, including zero incremental exposure changes.
    """
    from acceptance_heuer_09 import allocate_iterative_heuer
    gdf = make_synthetic_grid()
    gdf['V_eur'] = gdf.price_eur_per_m2_used * gdf.area_total_m2
    mask = gdf.n_new_max > 0
    gdf.loc[mask, 'n_new_max'] = 3
    shared = SimpleNamespace(
        pot_mw_per_km2=25., existing_proxy_mw_per_turbine=2.5,
        state_upper_slack=.5, use_state_targets=False, eps_rel=1e-12,
        build_limit_mode='total_gw', lifetime_yrs=20, wacc=.06,
        fixed_om_share_of_capex_per_year=.025)
    targets = pd.DataFrame({'state_id':['01'], 'add_gw':[1.]})
    target_gw = .032  # eight 4-MW turbines, versus four candidate cells
    actual = {}
    for mode in ('base_only', 'total_lcoe'):
        cfg = HeuerConfig(specification='joint_conservative', radius_km=3.,
                          allocation_mode=mode, debug_every_steps=0)
        placement, curve, _, _ = allocate_iterative_heuer_joint(
            gdf, targets, cfg, shared, target_gw)
        assert len(curve) == 8, (mode, len(curve))
        assert np.isclose(placement.added_gw.sum(), .032)
        assert (placement.n_new <= 3).all()
        assert (placement.n_new >= 0).all()
        assert (placement.n_new > 1).any(), 'At least one cell must be reused'
        assert np.isclose(curve.take_cost_eur.sum(), placement.added_cost_eur.sum())
        actual[mode] = curve
    # Compare against the older Heuer base-only allocator exactly.
    old_cfg = HeuerConfig(specification='distance', radius_km=2.,
                          allocation_mode='base_only', debug_every_steps=0)
    _, old_curve, _, _ = allocate_iterative_heuer(
        gdf, targets, old_cfg, shared, target_gw)
    assert old_curve.idm.tolist() == actual['base_only'].idm.tolist(), (
        old_curve.idm.tolist(),actual['base_only'].idm.tolist())
    assert np.allclose(old_curve.pv_base_cost_eur_step,
                       actual['base_only'].pv_base_cost_eur_step)
    assert np.allclose(old_curve.pv_energy_mwh_step,
                       actual['base_only'].pv_energy_mwh_step)

    # Independent exhaustive selection at each total-LCOE step, including
    # recently selected cells as long as nmax > 0.
    bands=joint_band_offsets()
    count=np.zeros((9,9),dtype=np.int32)
    count[2,2] = 1
    kernels=[]
    for offs in bands:
        ker=np.zeros((9,9),np.int32)
        for dr,dc in offs: ker[4+dr,4+dc] = 1
        kernels.append(ker)
    V=np.zeros((9,9))
    for _,row in gdf.iterrows():V[int(row.y_km),int(row.x_km)]=row.V_eur
    available={(3,2):3,(4,4):3,(6,6):3,(2,5):3}
    seq=[]
    af=(1-1.06**-20)/.06
    for _ in range(8):
        counts=np.stack([convolve(count,k,mode='constant',cval=0)
                         for k in kernels])
        B=joint_total_beta(counts)
        ranking=[]
        for (r,c),free in available.items():
            if free<=0:continue
            row=gdf.loc[(gdf.y_km==r)&(gdf.x_km==c)].iloc[0]
            pv_e=row.rated_power_mw_era5*row.net_flh_era5*af
            pv_c=row.rated_power_kw_era5*row.costs_era5*(1+.025*af)
            cost=joint_marginal_cost_many(np.array([r]),np.array([c]),
                                           counts,B,V,bands)[0]
            ranking.append(((pv_c+cost)/pv_e,r,c))
        _,r,c=min(ranking)
        seq.append(f'{c}_{r}')
        count[r,c]+=1
        available[(r,c)]-=1
    assert actual['total_lcoe'].idm.tolist() == seq, (
        actual['total_lcoe'].idm.tolist(),seq)


def test_single_cell_reuse():
    """A candidate that is still available is never lost from the heap."""
    grid = make_synthetic_grid()
    grid['V_eur'] = grid.price_eur_per_m2_used * grid.area_total_m2
    grid.loc[grid.n_new_max > 0, 'n_new_max'] = 0
    grid.loc[grid.idm == '4_4', 'n_new_max'] = 4
    shared=SimpleNamespace(pot_mw_per_km2=25.,existing_proxy_mw_per_turbine=2.5,
        state_upper_slack=.5,use_state_targets=False,eps_rel=1e-12,
        build_limit_mode='total_gw',lifetime_yrs=20,wacc=.06,
        fixed_om_share_of_capex_per_year=.025)
    cfg=HeuerConfig(specification='joint_conservative',radius_km=3.,debug_every_steps=0)
    targets=pd.DataFrame({'state_id':['01'],'add_gw':[1.]})
    for mode in ('base_only','total_lcoe'):
        cfg.allocation_mode=mode
        _,curve,_,_=allocate_iterative_heuer_joint(grid,targets,cfg,shared,.016)
        assert curve.idm.tolist() == ['4_4']*4, (mode,curve.idm.tolist())

if __name__ == '__main__':
    test_parameters_and_bands()
    test_marginal_exact_telescopes()
    test_small_allocator_exact_greedy()
    test_repeated_placements_and_base_only_invariance()
    test_single_cell_reuse()
    print('PASS: coefficients, raster exposure, telescoping PV losses, exact greedy and repeat-cell regression tests')
