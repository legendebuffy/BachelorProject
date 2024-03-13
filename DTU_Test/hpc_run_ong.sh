#!/bin/bash
#BSUB -J EU_coin_s
## 'EU'=Edgebank-unlimited ; 's'=subset

## output files
#BSUB -o ../hpc_runs/Run_%J.out.txt
#BSUB -e ../hpc_runs/Run_%J.err.txt

## GPU
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -n 4

## runtime
#BSUB -W 1:00

## mail when done
#BSUB -N

## since all commands are from cmd's cwd, remember to place cwd in git folder: "BachelorProject"

source ../env_BScP/bin/activate

# Edgebank, u_coin
python edgebank.py --subset True -d tgbl-coin --run run1 --seed 1
