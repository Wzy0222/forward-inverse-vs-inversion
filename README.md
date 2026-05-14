# Vs inversion figure reproduction code

This repository contains the code and configuration used to reproduce selected manuscript figures from processed result files.

The GitHub code repository does not directly include HDF5 data files or trained model weight files. Download the data and model archive from Zenodo or release assets.

10.5281/zenodo.20174724

After downloading the archive, place the files at these paths:

```text
data/duo_synthetic_result.h5
data/duo_real_result.h5
data/Li_2022_16.h5
data/gmt_data/*.gmt
models/*.pth
```

## Figure mapping

| Figure | Input | Command |
| --- | --- | --- |
| Fig. 5 | `data/duo_synthetic_result.h5` | `bash scripts/reproduce_fig5.sh` |
| Fig. 7 | `data/duo_real_result.h5` | `bash scripts/reproduce_fig7.sh` |
| Fig. 12 / Fig. 13 | paths in `configs/compare_li2022.yaml` | `bash scripts/compare_with_li2022.sh` |

## Environment

Use Python 3.10 or newer. PyGMT also requires GMT 6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a different Python environment is used:

```bash
PYTHON=/path/to/python bash scripts/check_reproducibility.sh
```

## Run commands

```bash
bash scripts/reproduce_fig5.sh
bash scripts/reproduce_fig7.sh
bash scripts/compare_with_li2022.sh
bash scripts/check_reproducibility.sh
```

## Expected outputs

- `outputs/fig5/fig5.png`
- `outputs/fig7/fig7.png`
- `outputs/figures/li2022_slice_comparison.png`
- `outputs/figures/li2022_depthwise_error.png`

Additional PDF files, depth-slice PNG files, and CSV metric tables are written under `outputs/`.

The scripts write under `outputs/` and cache directories inside `outputs/`. They read `data/` and `models/` but do not modify those directories.
