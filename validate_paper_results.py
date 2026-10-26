# -*- coding: utf-8 -*-
"""Cross-check model outputs against original 2026-10-22 paper reference values.

Only meaningful for the SAME original input grid and model assumptions.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'Output_Results_upperSlack50' / 'Paper_Figures_FrondelFixed'
REFERENCE = {'Base':64.297121,'HedPrice-F':65.970942,'HedPrice-H':66.034363}
TOL = 0.06  # EUR/MWh, protects against stale/incorrect runs without floating-point brittleness


def validate_table() -> None:
    path = RESULTS/'table_160GW_extended_states.csv'
    if not path.is_file(): raise FileNotFoundError(path)
    df = pd.read_csv(path).set_index('model')
    col = 'total_lcoe_eur_mwh'
    for k, ref in REFERENCE.items():
        actual = float(df.loc[k,col]); gap=abs(actual-ref)
        print(f'[{"PASS" if gap<=TOL else "FAIL"}] {k}: {actual:.6f} EUR/MWh; reference {ref:.6f}; difference {gap:.6f}')
        if gap>TOL: raise AssertionError(f'{k} deviates > {TOL} EUR/MWh')
    f=float(df.loc['HedPrice-F','total_lcoe_eur_mwh'])
    h=float(df.loc['HedPrice-H','total_lcoe_eur_mwh'])
    assert f<h, 'Expected corrected HedPrice-F below HedPrice-H at 160GW'


def validate_siting() -> None:
    p=RESULTS/'table_allocation_effects_160GW.csv'
    if not p.is_file(): raise FileNotFoundError(p)
    df=pd.read_csv(p).set_index('model')
    f=df.loc['HedPrice-F']
    improvement=float(f['standard_total_lcoe_eur_mwh']-f['extended_total_lcoe_eur_mwh'])
    print(f'[CHECK] HedPrice-F avoided Total LCOE vs standard allocation: {improvement:.6f} EUR/MWh')
    assert abs(improvement-2.023819)<TOL, 'Expected the corrected heapfix improvement ~2.02 EUR/MWh'
    print('[PASS] Allocation effect agrees with corrected 2026-10-22 reference')


if __name__=='__main__':
    try:
        validate_table();validate_siting()
    except Exception as e:
        print(f'[FAIL] Reference validation: {type(e).__name__}: {e}')
        sys.exit(1)
    print('[PASS] Reference table validation finished')
