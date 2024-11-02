# **Force Field Modeling -- 3BPA**

Implemented based on official implementation of [MACE](https://github.com/ACEsuit/mace).


## Installation

```shell
conda create --name gaunt-mace python=3.10
conda activate gaunt-mace
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==1.7.1 torch_scatter==2.0.9 torch_sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.13.1%2Bcu117.html
pip install e3nn torch_ema
pip install ase==3.22.1 
pip install numpy==1.24.4
cd MACE && pip install . && cd ../
```


## Usage

```shell
bash run_3bpa.sh
```

