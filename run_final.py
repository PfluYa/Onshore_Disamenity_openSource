"""Central entry point for final research reproduction.

IMPORTANT: Running this file without arguments (e.g. Spyder's Run button)
starts a full grid -> models -> figures -> validation recalculation and permits
overwriting existing outputs. Back up the project first.

Explicit commands:
 python run_final.py check
 python run_final.py tests
 python run_final.py all --force-overwrite             # rebuild grid only if missing
 python run_final.py all --force-overwrite --rebuild-grid  # also rebuild existing grid
 python run_final.py figures
 python run_final.py environment
 python run_final.py locate-grid
 python run_final.py grid
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check() -> bool:
    print(f'[ROOT] {ROOT}')
    expected_scripts = (
        'main.py', 'grid_builder_01.py', 'base_lcoe_02.py',
        'acceptance_wen_ruhnau_03.py', 'acceptance_frondel_04.py',
        'acceptance_heuer_09.py', 'acceptance_krekel_zerrahn_06.py',
        'acceptance_tsani_proxy_07.py', 'acceptance_tsani_scenicness_08.py',
        'compare_lcoe_05.py', 'results_script_paper_all_figures.py',
    )
    missing = [p for p in expected_scripts if not (ROOT / p).is_file()]
    if missing:
        print('[FAIL] Missing scripts:', ', '.join(missing))
        return False
    print('[PASS] All required scripts present')
    grid = ROOT / 'Output_Main' / 'grid_de_era5.parquet'
    if grid.is_file():
        print(f'[PASS] Existing grid: {grid} ({grid.stat().st_size / 1e9:.2f} GB)')
    else:
        print(f'[WARN] Preprocessed input grid not found: {grid}')
        print('[INFO] Synthetic tests and plotting existing outputs do not require this grid.')
        print('[NEXT] Locate an existing grid: python run_final.py locate-grid')
        print('[NEXT] If genuinely absent, rebuild it: python run_final.py grid')
        print('[NEXT] Models/all now build the input grid automatically when missing.')
    input_dir = ROOT / 'Input Data'
    print(f"[{'PASS' if input_dir.is_dir() else 'INFO'}] Raw input directory: {input_dir}")
    for mod in ('numpy','pandas','geopandas','scipy','matplotlib','rasterio','xarray','pyarrow','shapely','openpyxl','netCDF4'):
        try:
            m = importlib.import_module(mod)
            print(f"[PKG] {mod}: {getattr(m,'__version__','installed')}")
        except ImportError as e:
            print(f'[MISSING PKG] {mod}: {e}')
    import main
    import results_script_paper_all_figures as figures
    from acceptance_heuer_09 import HEUER_JOINT_BETAS
    assert main.ROOT == ROOT and figures.ROOT == ROOT
    assert 'heapfix' in (main.ROOT / 'placing_frondel_adjusted_heapfix').name
    assert HEUER_JOINT_BETAS.shape == (3, 3)
    expected_frondel = figures.curve_path('HedPrice-F', states=True, mode='total_lcoe')
    assert expected_frondel.parent.name == 'placing_frondel_adjusted_heapfix'
    print('[PASS] Model/figures use the same project root and corrected Frondel path')
    return not missing



def locate_grid() -> list[Path]:
    """Report plausible preprocessed-grid files; never choose one silently."""
    print(f'[SEARCH] Looking for grid_de_era5*.parquet below {ROOT}')
    import os
    found: list[Path] = []
    for current, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d not in {
            '.git', '__pycache__', '.venv', 'venv', '.pytest_cache',
        }]
        for name in files:
            if name.lower().startswith('grid_de_era5') and name.lower().endswith('.parquet'):
                path = Path(current) / name
                found.append(path)
                print(f'[FOUND] {path} ({path.stat().st_size / 1e9:.3f} GB)')
    if not found:
        print('[NOT FOUND] No grid_de_era5*.parquet found inside this repository.')
        print('[NEXT] Check external disks/backups or regenerate the grid using "python run_final.py grid".')
    else:
        expected = ROOT / 'Output_Main' / 'grid_de_era5.parquet'
        if expected not in found:
            print(f'[NEXT] The model expects: {expected}')
            print('[CAUTION] Verify wind year, REDX release and preprocessing configuration before copying a candidate.')
            print('[NEXT] Copy an identical verified file to the expected path, or set GRID_INPUT in main.py.')
    return found


def rebuild_grid(force_overwrite: bool = False) -> None:
    """Build the input raster from original inputs; never run during ``check``."""
    expected = ROOT / 'Output_Main' / 'grid_de_era5.parquet'
    if expected.is_file() and not force_overwrite:
        raise SystemExit(f'[STOP] Grid already exists: {expected}\n'
                         'Back it up first and use --force-overwrite only if rebuilding is intended.')
    if not (ROOT / 'Input Data').is_dir():
        raise SystemExit('[STOP] Local Input Data/ directory is absent; cannot rebuild the grid.')
    from grid_builder_01 import GridBuilderConfig, run_grid_builder
    cfg = GridBuilderConfig()
    cfg.root = ROOT
    cfg.out_dir_name = 'Output_Main'
    cfg.glaes_tif = ROOT / 'Input Data' / 'de_onshore_wind_holtinger2016_med_100m.tif'
    print('[GRID] Rebuilding the preprocessed input grid using the raw local inputs.', flush=True)
    print('[GRID] Uses ERA5 2024 and the exact raw data file versions in grid_builder_01.py.')
    print('[GRID] Rebuild also writes intermediate grid_m and grid_de files.', flush=True)
    result = run_grid_builder(cfg)
    del result
    if not expected.is_file():
        raise RuntimeError(f'[FAIL] Grid builder returned but output is absent: {expected}')
    print(f'[PASS] Preprocessed grid ready: {expected}')


def environment() -> None:
    print(f'Python: {sys.version}\nPlatform: {platform.platform()}')
    for mod in ('numpy','pandas','geopandas','scipy','matplotlib','rasterio','xarray','pyarrow','shapely','openpyxl','netCDF4'):
        try:
            pkg = importlib.import_module(mod)
            print(f"{mod}=={getattr(pkg,'__version__','unknown')}")
        except ImportError:
            print(f'{mod}: NOT INSTALLED')
    grid = ROOT / 'Output_Main' / 'grid_de_era5.parquet'
    if grid.exists():
        print(f'Grid SHA-256: {sha256_file(grid)}  {grid}')


def tests() -> None:
    files = ['test_frondel_greedy_exact.py','test_frondel_greedy_states.py','test_heuer_joint.py']
    for name in files:
        print(f'\n[TEST] {name}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'tests'/name)], cwd=ROOT, check=True,
                       env={**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')})
    print('[PASS] All synthetic model tests finished')


def existing_outputs() -> list[Path]:
    from main import scenario_folder, RUN_ALLOCATION_MODES, RUN_STATE_MODES
    return [scenario_folder(m, s) for s in RUN_STATE_MODES for m in RUN_ALLOCATION_MODES
            if scenario_folder(m, s).exists()]


def models(force_overwrite: bool, rebuild_grid_input: bool = False) -> None:
    """Run all models, first preparing the grid if absent or explicitly requested.

    Existing scenario folders are checked BEFORE expensive grid preprocessing.
    The standalone main.py retains REBUILD_GRID=False to avoid a double build.
    """
    import main
    occupied = existing_outputs()
    if occupied and not force_overwrite:
        raise SystemExit('Refusing to rewrite existing model output directories:\n  '
                         + '\n  '.join(map(str, occupied))
                         + '\nUse --force-overwrite after backing up local outputs.')

    if rebuild_grid_input:
        print('[GRID] --rebuild-grid requested: recomputing the grid from raw inputs.', flush=True)
        rebuild_grid(force_overwrite=force_overwrite)
    elif not main.GRID_INPUT.is_file():
        print(f'[GRID] Missing {main.GRID_INPUT}; generating it automatically.', flush=True)
        rebuild_grid(force_overwrite=False)
    else:
        print(f'[GRID] Reusing existing input grid: {main.GRID_INPUT}', flush=True)

    if not main.GRID_INPUT.is_file():
        raise RuntimeError('[FAIL] No input grid available after preprocessing: '
                           + str(main.GRID_INPUT))
    if occupied:
        print('[WARNING] Existing model outputs may be overwritten:', flush=True)
        for p in occupied:
            print('  ', p, flush=True)

    # run_final owns grid preprocessing; prevent main.run_input_builder from rebuilding twice.
    if main.REBUILD_GRID:
        print('[GRID] Disabling redundant main.REBUILD_GRID for this run.', flush=True)
        main.REBUILD_GRID = False
    main.main()


def figures() -> None:
    import results_script_paper_all_figures
    results_script_paper_all_figures.main()


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', default='all',
                   choices=['check','locate-grid','grid','tests','models','figures','verify','all','environment'])
    p.add_argument('--force-overwrite', action='store_true',
                   help='Permit overwriting existing outputs (and input grid with --rebuild-grid)')
    p.add_argument('--rebuild-grid', action='store_true',
                   help='With models/all: rebuild the grid even if it already exists')
    a = p.parse_args()
    # Spyder invokes run_final.py without script arguments. Make Run/Runfile
    # reproduce the full paper, as explicitly intended by the project author.
    # Explicit commands such as `check` and `figures` retain their old behavior.
    if len(sys.argv) == 1:
        a.command = 'all'
        a.force_overwrite = True
        a.rebuild_grid = True
        print('[MODE] No arguments: FULL REPRODUCTION (Spyder default).', flush=True)
        print('[MODE] Rebuild grid -> 8 scenarios x 4 configurations -> '
              'Figures 1-7/tables -> 160-GW validation.', flush=True)
        print('[WARNING] Existing grid and model outputs may be overwritten. '
              'Keep a separate backup before running.', flush=True)
    if a.rebuild_grid and a.command not in ('models', 'all'):
        p.error('--rebuild-grid is only supported with models or all')
    if a.command == 'check':
        if not check():
            raise SystemExit(2)
    elif a.command == 'locate-grid': locate_grid()
    elif a.command == 'grid': rebuild_grid(a.force_overwrite)
    elif a.command == 'environment': environment()
    elif a.command == 'tests': tests()
    elif a.command == 'models': models(a.force_overwrite, a.rebuild_grid)
    elif a.command == 'figures': figures()
    elif a.command == 'verify':
        subprocess.run([sys.executable, str(ROOT/'validate_paper_results.py')], cwd=ROOT, check=True)
    elif a.command == 'all':
        models(a.force_overwrite, a.rebuild_grid)
        figures()
        subprocess.run([sys.executable, str(ROOT/'validate_paper_results.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    cli()
