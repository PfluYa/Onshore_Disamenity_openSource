"""Check the real Frondel allocation function against independent full rescoring."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import acceptance_frondel_04 as F


def synthetic(seed, n=9):
    rng=np.random.default_rng(seed)
    rows=[]
    for y in range(n):
        for x in range(n):
            nmax=int(rng.choice([0,0,0,0,1,1,1,2]))
            urban=int(rng.random()<0.32)
            V=float(rng.choice([0.,0.,0.,0.,0.,10e6,20e6,35e6]))
            existing=float(rng.choice([0.,0.,0.,0.,0.,0.,0.004]))
            rows.append(dict(idm=f'{y}-{x}',x_km=x,y_km=y,
                             state_id='01' if x<n//2 else '02',
                             n_new_max=nmax,flh_net=float(rng.uniform(1700,3400)),
                             cf=0.3,turbine_power_mw=4.,turbine_power_kw=4000.,
                             turbine_cost_input=float(rng.uniform(800,1400)),
                             V_eur=V,is_urban_city10km=urban,in_de=1,existing_gw=existing))
    return pd.DataFrame(rows)


def oracle(gdf, use_existing_stock, max_steps=None):
    val,_,_=F.gdf_to_array_numeric(gdf,'V_eur')
    urb,_,_=F.gdf_to_array_numeric(gdf,'is_urban_city10km')
    nmax,_,_=F.gdf_to_array_numeric(gdf,'n_new_max')
    eng=F.build_cell_engineering_arrays(gdf,turbine_power_mw_col='turbine_power_mw',
        turbine_power_kw_col='turbine_power_kw',turbine_cost_col='turbine_cost_input',
        flh_col='flh_net')
    existing=F.existing_turbine_presence_array(gdf,existing_gw_col='existing_gw',
        existing_proxy_mw_per_turbine=2.5,use_existing_stock=use_existing_stock)
    lr,lu=F.build_frondel_loss_fraction_tables()
    kernel=F.build_distance_band_kernel(9)
    kwargs=dict(rural_loss_fracs=lr,urban_loss_fracs=lu,dist_kernel=kernel['dist'],
        inside_kernel=kernel['inside'],band_idx_kernel=kernel['band_idx'],max_distance_km=9)
    res=val>0
    inside=np.ones(val.shape,dtype=bool)
    D,L,E=F.initialize_nearest_exposure_from_existing(
        residential_mask=res,in_de_mask=inside,value_arr=val,urban_mask=urb>0.5,
        existing_turbine_mask=existing,**kwargs)
    common=dict(current_nearest_dist=D,current_loss_frac=L,value_arr=val,
        urban_mask=urb>0.5,residential_mask=res,in_de_mask=inside,**kwargs)
    ids={tuple(r[['y_km','x_km']].astype(int)):r['idm'] for _,r in gdf.iterrows()}
    res_order=[]
    costs=[]
    steps=0
    phase_a = True
    st = {str(row.idm):str(row.state_id) for _,row in gdf.iterrows()}
    by_state={'01':0., '02':0.}
    while np.any(nmax>0):
        if phase_a and all(v >= .008-1e-11 for v in by_state.values()):
            phase_a=False
        candidates=[]
        for y,x in zip(*np.where(nmax>0)):
            en=eng['pv_energy_mwh_per_turbine'][y,x]
            tech=eng['pv_base_eur_per_turbine'][y,x]
            if en<=0: continue
            if phase_a and by_state[st[ids[y,x]]] + .004 > .012 + 1e-11: continue
            c=F.marginal_nearest_turbine_cost(turbine_r=y,turbine_c=x,**common)
            candidates.append(((tech+c)/en,int(y),int(x)))
        if not candidates:
            if phase_a:
                phase_a=False
                continue
            break
        best,y,x=min(candidates)
        score=best
        realized=F.apply_new_turbine_nearest(turbine_r=y,turbine_c=x,
            current_loss_eur=E,**common)
        nmax[y,x]-=1
        by_state[st[ids[y,x]]] += .004
        res_order.append(ids[y,x])
        costs.append(score)
        steps+=1
        if max_steps and steps>=max_steps: break
    return res_order, costs


def actual(gdf, use_existing_stock, allocation_mode='total_lcoe'):
    targets=pd.DataFrame([dict(state_id=s,target_gw=.008,add_gw=.008) for s in ['01','02']])
    lr,lu=F.build_frondel_loss_fraction_tables()
    placement,curve,report,meta=F.allocate_iterative_frondel_nearest(
        gdf,targets,state_id_col='state_id',n_new_max_col='n_new_max',
        flh_col='flh_net',cf_col='cf',turbine_power_mw_col='turbine_power_mw',
        turbine_power_kw_col='turbine_power_kw',turbine_cost_col='turbine_cost_input',
        value_col='V_eur',urban_col='is_urban_city10km',
        rural_loss_fracs=lr,urban_loss_fracs=lu,frondel_max_distance_km=9,
        allocation_mode=allocation_mode,use_state_targets=True,
        state_upper_slack=.50,
        build_limit_mode='exhaustion',use_existing_stock=use_existing_stock)
    return curve['idm'].tolist(),curve['score_total_pv_lcoe_at_pick'].tolist()

def run_tests():
    for seed in (10, 34, 2026):
        gdf = synthetic(seed, 9)
        for use_existing in (False, True):
            expected, ex_scores = oracle(gdf, use_existing)
            observed, ob_scores = actual(gdf, use_existing)
            assert observed == expected, (seed, use_existing, "state allocation order differs")
            assert np.allclose(ob_scores, ex_scores, rtol=1e-12, atol=1e-10)
            print(f'PASS states=True, seed={seed}, existing={use_existing}, placements={len(observed)}')
    print('PASS exact state-constrained greedy selection against brute-force rescoring')

if __name__ == '__main__':
    run_tests()
