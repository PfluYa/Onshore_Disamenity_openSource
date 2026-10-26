"""Create a curated local Zenodo results package from the finished paper outputs.

The export contains publication figures, aggregate CSV tables and a reconstructed
160-GW HedPrice-F spatial-reallocation summary. Optionally export *numeric-only*
cost curves as CSV and Parquet, clipped to the common total-capacity horizon.

Raw inputs, original placement-cell/grid Parquets, geometries, IDs, exposure and
real-estate data are never written into this package. No network upload occurs.

Usage (Windows terminal):
    python prepare_zenodo_results.py --replace
    python prepare_zenodo_results.py --include-curves --replace
    python prepare_zenodo_results.py --include-curves --include-national-curves --replace
    python prepare_zenodo_results.py --include-curves --replace --make-zip

Spyder IPython console:
    %runfile C:/Users/.../Onshore_Disamenity_SOffie/prepare_zenodo_results.py --args "--include-curves --replace"
"""
from __future__ import annotations

import argparse
from datetime import date, timezone, datetime
import hashlib
import shutil
import zipfile
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Output_Results_upperSlack50' / 'Paper_Figures_FrondelFixed'
EXPORT = ROOT / 'Zenodo_Results_Export'
MARKER = '.created_by_prepare_zenodo_results'
CURVE_MAX_GW = 220.0  # 160 GW is the paper's target; 220 GW is its common curve horizon.
TARGET_GW = 160.0

FIGURE_NAMES = (
    'fig1_technical_baseline',
    'fig2_total_lcoe_all_models',
    'fig3_lcoe_premiums',
    'fig4_allocation_standard_vs_extended',
    'fig5_smoothed_marginal_lcoe',
    'fig6_spatial_reallocation_hedprice_F',
    'fig7_technical_site_lcoe_baseline_vs_hedprice_F',
)
EXTRA_NAMES = (
    'extra_hedprice_F_vs_H', 'extra_premium_decomposition_160GW',
    'appendix_regional_state_vs_national',
)
TABLE_NAMES = (
    'table_160GW_extended_states.csv',
    'table_allocation_effects_160GW.csv',
    'table_regional_sensitivity_160GW.csv',
    'map_inputs_160GW.csv',
)
CURVE_FIELDS = (
    'step', 'capacity_gw', 'generation_twh', 'technical_lcoe',
    'disamenity_lcoe', 'total_lcoe', 'marginal_lcoe',
)


def prepare_directory(folder: Path, replace: bool) -> None:
    if folder.exists():
        if not replace:
            raise SystemExit(
                f'[STOP] Export already exists: {folder}\n'
                'Inspect/back up first; use --replace to regenerate this generated directory.'
            )
        if not (folder / MARKER).is_file():
            raise SystemExit(f'[STOP] No generated-export marker in {folder}; will not delete it.')
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / MARKER).write_text(
        'Generated locally by prepare_zenodo_results.py. Not part of the data deposit.\n',
        encoding='utf-8',
    )


def copy_paper_files(dest: Path) -> tuple[int, int]:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f'[STOP] No paper figures directory: {SOURCE}; run figures first.')
    (dest / 'figures').mkdir()
    (dest / 'tables').mkdir()
    figure_count = 0
    for base in FIGURE_NAMES + EXTRA_NAMES:
        for ext in ('.png', '.pdf'):
            src = SOURCE / f'{base}{ext}'
            if src.is_file():
                shutil.copy2(src, dest / 'figures' / src.name)
                figure_count += 1
            elif base in FIGURE_NAMES:
                raise FileNotFoundError(
                    f'[STOP] Main paper figure missing: {src}. Regenerate figures first.'
                )
    table_count = 0
    for name in TABLE_NAMES:
        src = SOURCE / name
        if src.is_file():
            df = pd.read_csv(src)
            # map_inputs_160GW.csv can contain absolute local paths. Retain only filenames.
            if name == 'map_inputs_160GW.csv' and 'source' in df.columns:
                df['source_file'] = df['source'].fillna('').astype(str).map(
                    lambda value: PureWindowsPath(value.replace('/', '\\')).name
                )
                df = df.drop(columns=['source'])
            prohibited = [c for c in df.columns if any(term in c.lower() for term in (
                'geometry', 'price', 'v_eur', 'address', 'residential_value',
                'local_path', 'source_path', 'idm',
            ))]
            if prohibited:
                raise ValueError(f'[STOP] Review unexpected columns in {name}: {prohibited}')
            df.to_csv(dest / 'tables' / name, index=False, float_format='%.9g')
            table_count += 1
        elif name in TABLE_NAMES[:2]:
            raise FileNotFoundError(f'[STOP] Required summary table missing: {src}')
    return figure_count, table_count


def compute_spatial_reallocation(paper) -> dict:
    """Derive 160-GW aggregate directly from chronological raw siting curves.

    Only the per-cell total MW from each model is held in memory temporarily.
    No identifiers or coordinates are included in the returned summary or export.
    """
    _, standard_curve, extended_curve, _, _ = paper.paper_map_files()
    std, std_info = paper.reconstruction_at_target(standard_curve)
    ext, ext_info = paper.reconstruction_at_target(extended_curve)
    for name, info in [('standard', std_info), ('extended', ext_info)]:
        if abs(info['installed_gw'] - TARGET_GW) > 0.05:
            raise ValueError(f'{name}: invalid 160 GW map reconstruction')

    joined = std.rename(columns={'selected_added_gw': 'standard_gw'}).merge(
        ext.rename(columns={'selected_added_gw': 'extended_gw'}),
        on='idm', how='outer', validate='one_to_one',
    ).fillna(0.)
    # Both source tables are grouped by idm in reconstruction_at_target().
    delta = joined['extended_gw'].to_numpy(dtype=float) - joined['standard_gw'].to_numpy(dtype=float)
    gain = float(delta[delta > 0].sum())
    loss = float(delta[delta < 0].sum())
    net = float(delta.sum())
    stotal = float(std_info['new_build_gw'])
    etotal = float(ext_info['new_build_gw'])
    if abs(net - (etotal - stotal)) > 1e-8:
        raise ValueError(f'[STOP] Reallocation accounting inconsistent: net={net}, difference={etotal-stotal}')
    if abs(net) > 0.01:  # 0.01 GW (10 MW): permits discrete nearest-160-GW steps
        raise ValueError(f'[STOP] Nonconserving spatial comparison: net={net:.6f} GW')
    if gain < 1 or loss > -1 or stotal < 1:
        raise ValueError('[STOP] Suspicious spatial reallocation summary')

    out = {
        'model': 'HedPrice-F',
        'allocation_comparison': 'extended minus standard',
        'capacity_target_gw': TARGET_GW,
        'standard_installed_total_gw': float(std_info['installed_gw']),
        'extended_installed_total_gw': float(ext_info['installed_gw']),
        'standard_new_build_gw': stotal,
        'extended_new_build_gw': etotal,
        'increased_capacity_gw': gain,
        'decreased_capacity_gw': loss,  # signed, negative
        'net_capacity_difference_gw': net,
        'reallocated_capacity_gw': gain,  # no double-counting of positive and negative
        'reallocated_share_new_build_percent': 100. * gain / stotal,
        'increased_cells': int((delta > 0).sum()),
        'decreased_cells': int((delta < 0).sum()),
    }
    return out


def export_spatial_reallocation(dest: Path) -> dict:
    import results_script_paper_all_figures as paper
    summary = compute_spatial_reallocation(paper)
    file = dest / 'tables' / 'table_spatial_reallocation_160GW.csv'
    pd.DataFrame([summary]).to_csv(file, index=False, float_format='%.12g')
    print(f"[PASS] Spatial reallocation at 160 GW: "
          f"+{summary['increased_capacity_gw']:.6f} / "
          f"{summary['decreased_capacity_gw']:.6f} GW; "
          f"net {summary['net_capacity_difference_gw']:.6f} GW; "
          f"{summary['reallocated_share_new_build_percent']:.2f}% "
          f"({summary['increased_cells']:,} gained, "
          f"{summary['decreased_cells']:,} lost cells)")
    return summary


def clip_numeric_curve(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """Limit all released scenarios to the common 220-GW *total* fleet horizon."""
    data = data.loc[:, CURVE_FIELDS].copy()
    if set(data.columns) != set(CURVE_FIELDS) or data.empty:
        raise ValueError(f'Unexpected numeric curve schema: {name}')
    if data.select_dtypes(exclude='number').shape[1] != 0:
        raise ValueError(f'Non-numeric data found: {name}')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite curve values: {name}')
    if (data['capacity_gw'].diff().dropna() <= 0).any():
        raise ValueError(f'Capacity is not strictly increasing: {name}')
    if abs(float(data.iloc[(data['capacity_gw'] - TARGET_GW).abs().argmin()]['capacity_gw']) - TARGET_GW) > 0.05:
        raise ValueError(f'No valid 160 GW reference in curve: {name}')
    data = data.loc[data['capacity_gw'] <= CURVE_MAX_GW + 1e-10].copy()
    if data.empty or float(data['capacity_gw'].max()) < CURVE_MAX_GW - 0.05:
        raise ValueError(f'Curve does not reach common 220 GW horizon: {name}')
    return data.reset_index(drop=True)


def export_numeric_curves(dest: Path, national: bool) -> int:
    """Only allowlisted, non-spatial columns are ever copied from source Parquets."""
    import results_script_paper_all_figures as paper
    out = dest / 'numeric_curves'
    out.mkdir()
    cases = [(True, 'total_lcoe', paper.MODEL_ORDER),
             (True, 'base_only', paper.MONETARY_MODELS)]
    if national:
        cases.extend([(False, 'total_lcoe', paper.MODEL_ORDER),
                      (False, 'base_only', paper.MONETARY_MODELS)])
    count = 0
    for states, allocation, models in cases:
        for model in models:
            data = paper.load_curve(model, states, allocation, manifest=[])
            name = f'{"states" if states else "national"}__{allocation}__{model.replace("-", "_")}'
            before = len(data)
            data = clip_numeric_curve(data, name)
            data.to_csv(out / f'{name}.csv', index=False, float_format='%.12g')
            data.to_parquet(out / f'{name}.parquet', index=False)
            print(f'[CURVE] {name}: {len(data):,} steps, '
                  f'{data["capacity_gw"].max():.6f} GW (removed {before-len(data):,} post-220GW rows)')
            count += 1
    return count


def add_readme(dest: Path, figure_files: int, table_files: int, curve_series: int,
               spatial: dict) -> None:
    text = f'''# Supplementary results: German onshore wind disamenities and cost-potential curves

Companion dataset for *Internalizing local disamenities in onshore wind cost-potential curves: Evidence from Germany*.

**Author:** Yannik Pflugfelder, University of Duisburg-Essen, Germany.  
**Dataset generated:** {date.today().isoformat()} using the corrected October 2026 model workflow.  
**Status:** author-curated result files; a public Zenodo DOI and the associated paper DOI are not yet assigned. Update the Zenodo record with the permanent identifiers when available.  
**Scope:** numerical results and figures only; not a complete distribution of the model's restricted source inputs.

## Files

- `figures/`: {figure_files} files (seven numbered paper figures and up to three supplementary plots, as PNG/PDF).
- `tables/`: {table_files} CSV files with 160-GW scenario results, siting-allocation comparison, regional sensitivity where available, and the **independently reconstructed aggregate HedPrice-F spatial reallocation**.
- `numeric_curves/`: {curve_series} scenario/allocation series, each published in **CSV and Parquet** when selected during export. If absent, regenerate with `--include-curves`.
- `manifest_sha256.csv`: SHA-256 hashes and byte sizes of the release contents, excluding the manifest and an internal exporter marker. Verify files using their hashes.

### Main quantities and conventions

- Wind resource basis: **ERA5 wind year 2024**; model grid: **1 km × 1 km** covering Germany.
- Modelled fixed existing stock: approximately **67.3 GW**; reference point **160 GW total installed capacity**, including existing turbines; published numeric curves are truncated to a **common maximum of 220 GW total installed capacity** (ending at the last feasible discrete turbine placement not exceeding 220 GW). The underlying WTP model files may go further; this release intentionally does not.
- Costs: **EUR/MWh**; total, technical and explicit disamenity LCOE are computed for **newly installed turbines only**. Annual generation includes existing turbines.
- Main configuration: federal-state deployment targets with a 50% upper corridor during the state-target phase. **`total_lcoe`** means disamenity-aware extended allocation; **`base_only`** means technical-ranking standard allocation with disamenities evaluated ex post. Optional `national` curves represent allocations without federal-state target constraints.
- Corrected HedPrice-F extended runs use `placing_frondel_adjusted_heapfix/`. HedPrice-H uses its joint distance/intensity A10 model. All numerical curves retain unmodified results *up to 220 GW*: truncation is the only change.

**Seven numeric curve columns:** `step` = sequential turbine placement; `capacity_gw` = total installed GW including stock; `generation_twh` = annual electricity generation (TWh/year), including existing stock; `technical_lcoe`, `disamenity_lcoe`, `total_lcoe` = portfolio-average present-value LCOE for newly installed turbines (EUR/MWh); `marginal_lcoe` = step-specific marginal turbine LCOE (EUR/MWh). No geography, raster-cell identifiers, housing values or other REDX input variables are included.

### Spatial reallocation summary (HedPrice-F at 160 GW)

`tables/table_spatial_reallocation_160GW.csv` is computed **afresh from the selected placement steps in the original standard and corrected extended cost-curve files**, grouped by raster cell, and only the aggregate differences are released. Individual cell identifiers and per-cell capacities are not exported. Positive and negative GW are *signed* differences in the selected new-build capacity: a location gaining capacity has a positive delta; a location losing capacity has a negative delta.

- Gaining cells: **{spatial['increased_cells']:,}**, receiving **+{spatial['increased_capacity_gw']:.6f} GW**.
- Losing cells: **{spatial['decreased_cells']:,}**, yielding **{spatial['decreased_capacity_gw']:.6f} GW**.
- Net difference: **{spatial['net_capacity_difference_gw']:.6f} GW** (small offset from discrete turbine-step matching at approximately 160 GW).
- Reallocated capacity: **{spatial['reallocated_capacity_gw']:.6f} GW**, or **{spatial['reallocated_share_new_build_percent']:.2f}%** of the standard allocation's **{spatial['standard_new_build_gw']:.6f} GW** of new-build capacity. This counts capacity moved only once (positive changes, not sum of absolute changes).

## Provenance, rights and reproduction

The reproducible model source code, input-file names, source download/access instructions, citation requirements, licenses, and model-run commands are in the **single README.md of the companion software repository** (`prepare_zenodo_results.py`, `run_final.py`, `results_script_paper_all_figures.py`). A full rerun requires the same source editions and their permitted access conditions; the exact input editions should be recorded when archiving the software release.

**Restricted raw data are not provided.** The underlying model uses RWI-GEO-REDX Scientific Use File (SUF) residential property-price data subject to its provider's contractual terms. Original housing grids, geospatial rasters and per-cell `placement_cells_*.parquet` datasets are excluded. Numerical curves are non-spatial aggregate model outputs, but absence of raw REDX columns is **not** itself a legal authorization for public sharing. The author must ensure that release of derived results complies with the applicable agreements and third-party map/figure-source terms. Verify the input REDX edition (grid builder file names V16 versus manuscript V17 citation) before assigning a public license.

**License:** To be selected by the copyright holder in the Zenodo deposit interface after reviewing rights (e.g., CC BY 4.0 for rights-cleared author-generated results). The mere inclusion of this README does not grant rights to redistribute third-party inputs. DOI and final citation metadata will be recorded in the Zenodo landing page; they are not invented here.

For questions about source inputs or access to restricted data, consult the companion software repository's central README and the RWI FDZ Ruhr data access information.
'''
    (dest / 'README.md').write_text(text, encoding='utf-8')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def write_manifest(dest: Path) -> None:
    rows = []
    for path in sorted(dest.rglob('*')):
        if path.is_file() and path.name not in (MARKER, 'manifest_sha256.csv'):
            rows.append({'path': path.relative_to(dest).as_posix(),
                         'size_bytes': path.stat().st_size,
                         'sha256': sha256(path)})
    pd.DataFrame(rows).to_csv(dest / 'manifest_sha256.csv', index=False)


def create_upload_zip(dest: Path) -> Path:
    """Create a release ZIP without the exporter marker; stored outside export folder."""
    path = dest.with_name('Zenodo_Results_Upload.zip')
    with zipfile.ZipFile(path, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(dest.rglob('*')):
            if file.is_file() and file.name != MARKER:
                z.write(file, arcname=file.relative_to(dest).as_posix())
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-curves', action='store_true',
                   help='Export only normalized numeric cost curves (CSV + Parquet) in addition to figs/tables.')
    p.add_argument('--include-national-curves', action='store_true',
                   help='Additionally export national allocation curves (requires --include-curves).')
    p.add_argument('--replace', action='store_true', help='Replace an export previously created by this script.')
    p.add_argument('--make-zip', action='store_true',
                   help='Create a companion ZIP for manual deposit, excluding the internal marker file.')
    opts = p.parse_args()
    if opts.include_national_curves and not opts.include_curves:
        p.error('--include-national-curves requires --include-curves')
    if not SOURCE.is_dir():
        raise SystemExit(f'[STOP] Missing paper output: {SOURCE}\nRun python run_final.py figures first.')
    prepare_directory(EXPORT, replace=opts.replace)
    try:
        f, t = copy_paper_files(EXPORT)
        spatial = export_spatial_reallocation(EXPORT)
        t += 1
        c = export_numeric_curves(EXPORT, opts.include_national_curves) if opts.include_curves else 0
        add_readme(EXPORT, f, t, c, spatial)
        write_manifest(EXPORT)
    except Exception:
        print(f'[WARNING] Export incomplete; inspect or discard manually: {EXPORT}')
        raise
    print(f'[PASS] Curated Zenodo results exported: {EXPORT}')
    print(f'[CONTENTS] {f} figure files, {t} aggregate tables, {c} numeric curve series')
    if opts.make_zip:
        archive = create_upload_zip(EXPORT)
        print(f'[ZIP] Created for manual review and Zenodo deposit: {archive}')
    print('[NOTE] Export is LOCAL ONLY; check dataset rights, exact input version and license before public deposit.')


if __name__ == '__main__':
    main()
