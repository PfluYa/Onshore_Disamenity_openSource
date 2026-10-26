# Internalizing local disamenities in onshore wind cost-potential curves: Evidence from Germany

**Replication code for the accompanying research paper**  
Yannik Pflugfelder · University of Duisburg-Essen · SOffie project

This repository estimates German onshore wind cost-potential curves on a **1 km × 1 km grid**. It compares a technical cost baseline with willingness-to-pay, hedonic-property-price, life-satisfaction and spatial-exclusion approaches to local disamenities. Electricity yields use **hourly ERA5 wind data for 2024**; the modeled existing turbine stock is fixed at the end of **2025**. The main comparison point is **160 GW total installed capacity** (including existing turbines), and the released cost-curve range extends to **220 GW**.

## Repository structure

| Location | Role |
|---|---|
| `run_final.py` | Main entry point: grid, scenarios, figures and validation |
| `grid_builder_01.py` | EEA grid, administrative/geographic inputs and ERA5 yields |
| `main.py` | Common assumptions; eight scenarios in four allocation configurations |
| `base_lcoe_02.py` | Technical baseline (Base) |
| `acceptance_wen_ruhnau_03.py` | WTP-Low and WTP-High |
| `acceptance_frondel_04.py` | HedPrice-F; dynamic nearest-turbine property-price impacts |
| `acceptance_heuer_09.py` | HedPrice-H; joint distance/intensity property-price impacts |
| `acceptance_krekel_zerrahn_06.py` | LifeSat |
| `acceptance_tsani_proxy_07.py` | ExclPop |
| `acceptance_tsani_scenicness_08.py` | ExclSce |
| `compare_lcoe_05.py` | Cost-curve comparisons |
| `results_script_paper_all_figures.py` | Paper figures 1–7 and accompanying result tables |
| `validate_paper_results.py` | Reference-value checks at 160 GW |
| `prepare_zenodo_results.py` | Export curated aggregate results for Zenodo |
| `tests/` | Synthetic allocation and hedonic-exposure tests |
| **[`Input Data/README.md`](Input%20Data/README.md)** | **Exact input filenames, access, preparation, licensing and limitations** |

## Input data at a glance

**Raw input files are not included in the GitHub repository.** Follow the linked [input-data guide](Input%20Data/README.md) to prepare the local `Input Data/` directory. In particular, the complete workflow needs access to restricted **RWI-GEO-REDX V16 Scientific Use Files**. Open datasets and their licenses cannot substitute for those restricted data.

| Source | Purpose | Access / replication strategy |
|---|---|---|
| EEA Reference Grid; BKG VG250 | 1-km geometry and administrative boundaries | Official download; BKG edition must match the original |
| GLAES / Holtinger2016_med | Wind-eligible land area | Historically generated GeoTIFF; upstream method documented; exact regeneration not verified |
| MaStR, Bundesnetzagentur | Existing onshore turbines through 2025 | Author-filtered public export; original Excel retained |
| ERA5 (2024), ECMWF/Copernicus | Hourly wind fields and site yields | Official CDS download (`u100`, `v100`, `u10`, `v10`) |
| Zensus 2022 (three 1-km layers) | Population, residential floor space and households | Destatis open downloads |
| RWI-GEO-REDX V16 SUF | Residential property values | Restricted; obtain from FDZ Ruhr |
| Destatis municipality directory | Urban-centre classification | Author-prepared Excel from official register |
| BfN/Roth (2021) | Scenic-beauty exclusion | BfN GeoTIFF; specific VGeoBund-derived-data conditions |
| Pöstges & Weber (2023) | Turbine parameters and power curves | Original institute CSVs; paper citation; no new open license assigned |
| Global Wind Atlas | Optional capacity-factor diagnostic | Not required for final ERA5 LCOE results |

## Setup and reproduction

Use the research project's working Conda environment if exact replication is required. `requirements.txt` is a package list, not a fully locked environment.

```bash
python -m pip install -r requirements.txt
python run_final.py check
python run_final.py tests
```

Place local inputs in `Input Data/` using the **exact file names and worksheet names** in its README. The model-ready grid is written to `Output_Main/grid_de_era5.parquet` and is **not suitable for public redistribution**: it can contain derived values from restricted property-price inputs.

### Commands

```bash
# Check installation / synthetic model behavior (no model recalculation)
python run_final.py check
python run_final.py tests
python run_final.py environment

# Figures/tables from existing model outputs; no grid rebuild
python run_final.py figures
python run_final.py verify

# Full reproduction, building the grid if missing
python run_final.py all --force-overwrite

# Full reproduction, explicitly rebuilding the input grid
python run_final.py all --force-overwrite --rebuild-grid
```

The full workflow covers four combinations: with/without federal-state targets, and technical-only (`base_only`) versus disamenity-aware (`total_lcoe`) location ranking. The state-constrained configurations use an upper slack of **50%**. HedPrice-F's extended allocation uses the corrected dynamic greedy implementation.

## Citation and rights

Please cite the paper (when published) and the repository's archived software release. Use [`CITATION.cff`](CITATION.cff) for available software citation metadata.

Input datasets are governed by **different licenses and access conditions**; see [`Input Data/README.md`](Input%20Data/README.md). The RWI-GEO-REDX SUF is not publicly redistributable without permission. 
The original Python source code in this repository is licensed under the MIT License; see LICENSE. Third-party datasets and research inputs are subject to their respective licenses and access conditions, as documented in Input Data/README.md.
