# Recovering the magnitude and direction of treatment and temporal differences with limited soil remeasurement

Research code by Yongye Huang and Zhining Wang.

The programs replay the selection of soil specimens for remeasurement under a fixed assay budget. They compare joint selection with rules targeting the direction or magnitude of treatment and temporal differences.

## Reproduce two tables from the paper

1. Extract the paper's supplementary archive, `Online_Resource_Two_Data_and_Code.zip`, into a separate working directory. It provides the measurements, recorded specimen selections, performance summaries and data-source documentation. The archive's `README.md` gives the instructions for all analyses.
2. To use the scripts from this repository, copy `code/`, `data/`, `direction/`, `spectral/` and `expand_selections.py` into that working directory, merging folders and keeping the relative paths. The scripts correspond to the copies supplied in the archive. The two examples below read the deposited summaries directly.
3. Open a terminal in that working directory. With Python 3.12, run

```console
python -m pip install -r requirements.txt
python direction/ultuna_fixed_budget.py --output reproduced/UltunaFixedBudgetTable.tex
python direction/archive_stability.py --output reproduced/ArchiveStabilityTable.tex
```

The first command after installation regenerates Table S68, which compares selection rules at Ultuna with 50 of 100 specimens assayed. The last command regenerates Table S69, which reports differences between selection rules across seven archives. Compare the two output files with their counterparts in the archive's `Supplement/` directory.

## Repository contents

- `code/` implements the finite-archive benchmarks and their diagnostics.
- `direction/` implements the selection rules, the independent replays, and the analyses and figures comparing direction and magnitude.
- `spectral/` contains the spectral calibration and validation analyses.
- `data/source/` contains scripts for preparing source data.
- `expand_selections.py` unpacks the specimen selections supplied in Online Resource 2 for analyses that use individual repeats.
- `ALLOCATION.md` explains the bounded integer allocation used by a comparator. The study protocols accompany the independent replays under `direction/harvard_forest/` and `direction/seasonal_validation/`.
- `ENVIRONMENT.json` records the software versions used for the analyses and figures. Data sources, credits and reuse terms are documented in `DATA_LICENSES.md` in Online Resource 2.
