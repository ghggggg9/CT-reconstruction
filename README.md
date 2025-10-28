# CT Reconstruction Project

A deep learning project for CT (Computed Tomography) image reconstruction using transformer-based models.

## Overview

This project implements a CT reconstruction model that learns to reconstruct 3D CT volumes from 2D X-ray projections (DRRs - Digitally Reconstructed Radiographs).

## Dataset

### DRR_Final (Included)
The `DRR_Final/` directory contains processed 2D X-ray projection images (107MB total):
- Contains data from LIDC-IDRI patients (LIDC-IDRI-0001 to LIDC-IDRI-1010)
- Each patient folder includes:
  - `drr_ap.png`: Anterior-Posterior view
  - `drr_lat.png`: Lateral view

### LIDC-IDRI Dataset (Not Included)
The original 3D CT volumes from the LIDC-IDRI dataset are **not included** in this repository due to size constraints (43GB).

**To use this project, you need to download the LIDC-IDRI dataset:**
- Official source: [TCIA LIDC-IDRI Collection](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- Place the downloaded dataset in the `LIDC-IDRI/` directory in the project root

## Project Structure

```
.
├── train.py                    # Main training script
├── DRR_Final/                  # 2D X-ray projections (included)
├── LIDC-IDRI/                  # 3D CT volumes (not included - download separately)
├── checkpoints_transformer/    # Model checkpoints (excluded from git)
├── outputs_transformer/        # Training outputs (excluded from git)
└── *.md                        # Documentation files
```

## Getting Started

1. Clone this repository
2. Download the LIDC-IDRI dataset and place it in `LIDC-IDRI/` directory
3. Install dependencies (requirements to be added)
4. Run training: `python train.py`

## Documentation

- [How to Read CT Images Guide](How_to_Read_CT_Images_Guide.md) - Guide for understanding CT image outputs
- [Quick Improvement Guide](Quick_Improvement_Guide.md) - Tips for improving model performance
- [Training Analysis Report](Training_Analysis_Report_5000epochs.md) - Detailed analysis of 5000-epoch training results

## License

[Add license information here]

## Citation

If you use the LIDC-IDRI dataset, please cite:
```
Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., ... & Clarke, L. P. (2011).
The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.
Medical physics, 38(2), 915-931.
```
