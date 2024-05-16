# Holohover - SysID

[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2405.09405-green.svg)](https://doi.org/10.48550/arXiv.2405.09405) [![Preprint](https://img.shields.io/badge/Preprint-arXiv-blue.svg)](https://arxiv.org/abs/2405.09405) [![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40180545)-90e3dc.svg)](https://nccr-automation.ch/)

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
@misc{schwan2024,
author={Schwan, Roland and Schmid, Nicolaj and Chassaing, Etienne and Samaha, Karim and Jones, Colin N.},
title={On identifying the non-linear dynamics of a hovercraft using an end-to-end deep learning approach}, 
year={2024},
eprint = {arXiv:2405.09405},
}
```
