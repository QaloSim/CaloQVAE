# CaloQVAE

Quantum VAEs for Calorimeter shower generation
- ...

## Overview
### Repository Structure


| Directory        | Content    | 
| ------------- |:-------------| 
| `configs/`      | Configuration files | 
| `data/` | Data manager and loader |
| `engine/`  | Training loops. |
| `models/` | Core module, includes definitions of all models.  |
| `notebooks/` | Standalone experimentation notebooks. |
| `paper/`  | Notebook to generate figures reported in [CaloDVAE](https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_145.pdf)
| `sandbox/` | Collection of test scripts and standalone models. |
| `scripts/` | Steering scripts includes one to run - run.py|
| `utils/` | Helper functionalities for core modules (plotting etc.) |

### Input Data

|  Dataset | Location |
| ------------- | ------------- |
| MNIST  | retrieved through torchvision |
| Calorimeter Data (GEANT4 showers, ⟂ to center) | [![DOI](https://zenodo.org/badge/DOI/10.17632/pvn3xc3wy5.1.svg)](https://doi.org/10.17632/pvn3xc3wy5.1)|


## Setup
```
git clone git@github.com:QaloSim/CaloQVAE.git
cd CaloQVAE
```

### Installation
#### Via Virtual Environment and pip
Initial package setup:
```
python3 -m venv venv_divae
source source.me
python3 -m pip install -r requirements.txt
```

### After Installation
After the initial setup, simply navigate to the package directory and run

```
source source.me
```
Sources the virtual environment and appends to `PYTHONPATH`.

## How To...

### ...configure models
We're currently using Hydra for config management. The top-level file is `config.yaml`. For more info on Hydra, click [here](https://hydra.cc/docs/tutorials/intro/)

### ...run models
```
python scripts/run.py
```

### ... run with Slurm submission
It is possible to run on computing clusters with the Slrum submission engine. Hydra has a built-in plugin interfacing the library `submitit`. It is important to use these dependencies:
```
hydra-core==1.1.0
hydra-submitit-launcher==1.1.5
submitit @ https://github.com/facebookincubator/submitit/archive/refs/tags/1.3.0.tar.gz
```
as the default PyPI version does not work on Cedar. A first script is added in the `scripts/` directory and a great starting point. To utitlise the batch submission, simply add the `--multirun` flag to your command line and specify which parameter to loop over like so:
```
python scripts/runSlurm.py --multirun config.myopt=1,2 
```

### References
[1] Jason Rolfe, Discrete Variational Autoencoders,
http://arxiv.org/abs/1609.02200

[2] M. Paganini ([@mickypaganini](https://github.com/mickypaganini)), L. de Oliveira ([@lukedeo](https://github.com/lukedeo)), B. Nachman ([@bnachman](https://github.com/bnachman)), _CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks_ [[`arXiv:1705.02355`](https://arxiv.org/abs/1705.02355)].