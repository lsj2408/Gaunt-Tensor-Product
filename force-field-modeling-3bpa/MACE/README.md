# MACE

Implementation of MACE.

## Installation

To install the dependencies on cpu run following line,

```bash
sbatch ./scripts/setup_env.sh
```
For cuda depdencies, edit the second line of the file from "CUDA"=cpu to the prefered version of cuda (eg. cu102).

## Experiments

All the experiments of the paper can be reproduced from the command line using the parser script. We first give a detail example on how to train a scale shifted MACE on 3BPA configurations at 300K :

```bash
# Run command
python3 ./MACE/scripts/run_train.py \
    --dataset="3bpa" \ #Type of dataset, put rMD17, acac or 3bpa
    --subset="train_300K" \
    --default_dtype="float32"\
    --seed=5 \
    --model="EquivariantRealScaleShiftNonLinearBodyOrderedModel" \ 
    --interaction="RealAgnosticResidualInteractionBlock" \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --device=cuda \
    --max_num_epochs=2000 \
    --patience=256 \
    --name="3bpa_mace_l2" \
    --energy_weight=1.0 \
    --forces_weight=1000.0 \
    --max_ell=3 \
    --hidden_irreps='256x0e + 256x1o + 256x2e' \ #Type of messages, put 256x0e only for invariant MACE
    --r_max=5.0 \
    --num_cutoff_basis=5 \
    --correlation=3 \
    --num_radial_coupling=1 \
    --batch_size=5 \
    --num_interactions=2 \
    --weight_decay=5e-7 \
    --ema \
    --ema_decay=0.99 \
    --scaling='rms_forces_scaling' \
    --restart_latest \
    --amsgrad \
```

## Requirements

* Python >= 3.8
* PyTorch >= 1.8
* PyTorch geometric >= 1.7.1


