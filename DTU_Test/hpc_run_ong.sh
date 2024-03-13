#!/bin/bash
#BSUB -J Edgebank-u_coin_subset
## ps (pretrain-subset), f (tinetune-nosubset)
#BSUB -o hpc/runs/Run_%J.out.txt
#BSUB -e hpc/runs/Run_%J.err.txt

## GPU
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

## runtime
#BSUB -W 1:00

## specs
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -n 1

## mail when done
#BSUB -N

## since all commands are from xterm's cd,
## remember to place xterm cd in git folder: "ProjectWork2023"

source env_BScP/bin/activate

## SleepEEG -> Epilepsy (n2: 2hr)
python code/main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy --subset False --device cuda
