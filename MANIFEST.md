# Manifest

## Code release files

| File | Purpose |
| --- | --- |
| `README.md` | Setup, data placement, and run instructions. |
| `MANIFEST.md` | File inventory for the code release. |
| `requirements.txt` | Python package requirements. |
| `CITATION.cff` | Citation metadata placeholder for the release. |
| `LICENSE` | License placeholder to be finalized before public deposit. |
| `.gitignore` | Prevents data, model weights, outputs, notebooks, patches, and local caches from being committed. |
| `checksums/sha256sums.txt` | Expected checksums for files supplied by the data and model archive. |

## Source files

| File | Purpose |
| --- | --- |
| `src/datasets.py` | HDF5 reading helpers. |
| `src/models.py` | Transformer model definitions matching the experiment architecture. |
| `src/plot_fig5.py` | Reads `duo_synthetic_result.h5` and writes Fig. 5 outputs. |
| `src/plot_fig7.py` | Reads `duo_real_result.h5` and writes Fig. 7 outputs. |
| `src/compare_with_li2022.py` | Compares this-study real-data inversion results with the Li et al. (2022) reference model. |

## Command scripts

| File | Purpose |
| --- | --- |
| `scripts/reproduce_fig5.sh` | Runs `src/plot_fig5.py` with `configs/reproduce_fig5.yaml`. |
| `scripts/reproduce_fig7.sh` | Runs `src/plot_fig7.py` with `configs/reproduce_fig7.yaml`. |
| `scripts/compare_with_li2022.sh` | Runs `src/compare_with_li2022.py` with `configs/compare_li2022.yaml`. |
| `scripts/check_reproducibility.sh` | Runs the three reproduction commands and checks for the expected PNG outputs. |

## Configuration files

| File | Purpose |
| --- | --- |
| `configs/reproduce_fig5.yaml` | Input paths, HDF5 field names, depth grid, and plotting settings for Fig. 5. |
| `configs/reproduce_fig7.yaml` | Input paths, HDF5 field names, depth grid, and plotting settings for Fig. 7. |
| `configs/compare_li2022.yaml` | Input paths, HDF5 field names, comparison settings, and plotting settings for Fig. 12 / Fig. 13. |

## External archive files

The following files are supplied by the data and model archive, not by the GitHub code repository:

- `data/duo_synthetic_result.h5`
- `data/duo_real_result.h5`
- `data/Li_2022_16.h5`
- `data/gmt_data/*.gmt`
- `models/model_85000.pth`
- `models/model_sinv_65000.pth`

## Generated outputs

The scripts create these files under `outputs/`:

- `outputs/fig5/fig5.png`
- `outputs/fig7/fig7.png`
- `outputs/figures/li2022_slice_comparison.png`
- `outputs/figures/li2022_depthwise_error.png`
- additional PDF, depth-slice PNG, and CSV metric files
