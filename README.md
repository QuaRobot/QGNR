# Quantum Graphon Learning (QGNR)

This repository contains the implementation used in the paper:

**Unveiling the Nature of Graphs through Quantum Graphon Learning**,  
npj Quantum Information, 2025.

The project provides a quantum machine learning version of graphon learning and includes the full training and evaluation pipeline.


## Requirements

This code depends on the following Python packages:

- `torch==2.0.0`  
- `torchquantum==0.1.7`  
- `qiskit==0.38.0`  
- `POT==0.9.4`  
- `einops==0.8.1`  
- `numpy`  
- `pickle`  
- `argparse`  
- `os`  



## Usage

Run training with:

    python train_GNR.py

Common arguments:

- --model  
  Choose `QGNR` (quantum version) or `IGNR` (baseline).

- --mlp_dim_hidden  
  Hidden layer sizes of the MLP (e.g., `20,20,20`).  
  **Note:** This argument is only effective for IGNR.  
  QGNR does not use `mlp_dim_hidden`.

- --n-epoch  
  Number of training epochs.

- --f-name  
  Directory name for saving results.

Example (QGNR):

    python train_GNR.py --model QGNR --f-name qgnr_exp

Example (IGNR):

    python train_GNR.py --model IGNR --mlp_dim_hidden 20,20,20 --f-name ignr_exp

## Dataset

Graph datasets used in this repository are stored in:

    Data/graphs.zip

These datasets are directly taken from the original IGNR implementation  
and are reused here to evaluate the proposed QGNR model. 
The dataset is provided as a zip file and must be extracted before use.

## Acknowledgement

This repository is a quantum machine learning extension based on IGNR: https://github.com/Mishne-Lab/IGNR

