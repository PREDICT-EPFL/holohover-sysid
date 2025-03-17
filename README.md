# Holohover - SysID

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.ifacol.2024.08.543-green.svg)](https://doi.org/10.1016/j.ifacol.2024.08.543) [![Preprint](https://img.shields.io/badge/Preprint-arXiv-blue.svg)](https://arxiv.org/abs/2405.09405) [![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40\_180545)-90e3dc.svg)](https://nccr-automation.ch/)

This repository provides the code accompanying the paper [On identifying the non-linear dynamics of a hovercraft using an end-to-end deep learning approach](https://arxiv.org/abs/2405.09405).

## Getting Started

All configuration is done through the `params.json` file. It specifies the data source, learning parameters, and initial model parameters.

### Preprocessing Data

Before learning a new mode, the data has to be preprocessed. To do so, create a new folder in the `experiments` folder, add the recorded mcap bag, change the `data/experiment` field in `params.json`, and run

```bash
python3 run_preprocessing.py
```

### Learning a Model

Change the `params.json` file accordingly and run 

```bash
python3 run_learning.py
```

## Datasets and Models from the Paper

The data used for training is `2023_10_18-11_28_12_sysid_h1_old` and the data used for validation is `2023_11_22-11_57_44_sysid_h1_old`. The relevant models are located in `models/paper` and the RMSE calculation is done in `paper_results.ipynb`. The control experiment and analysis can be found in `control_experiment`.

## Citing our Work

To cite our work in other academic papers, please use the following BibTex entry:
```
@article{schwan2024,
title = {On identifying the non-linear dynamics of a hovercraft using an end-to-end deep learning approach.},
journal = {IFAC-PapersOnLine},
volume = {58},
number = {15},
pages = {289-294},
year = {2024},
note = {20th IFAC Symposium on System Identification SYSID 2024},
issn = {2405-8963},
doi = {https://doi.org/10.1016/j.ifacol.2024.08.543},
url = {https://www.sciencedirect.com/science/article/pii/S2405896324013235},
author = {R. Schwan and N. Schmid and E. Chassaing and K. Samaha and C.N. Jones}
}
```
