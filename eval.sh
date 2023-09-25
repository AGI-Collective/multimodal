#!/bin/bash
#BSUB -nnodes 7
#BSUB -W 10
#BSUB -q debug
#BSUB -o /ccs/home/alexisroger/scratch/jobs/magma_pythia70m_out.%J
#BSUB -e /ccs/home/alexisroger/scratch/jobs/magma_pythia70m_err.%J
#BSUB -J magma_pythia70m
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499

#passage killable: 25000
# source /gpfs/alpine/csc499/proj-shared/env_setup/setup.sh
source /gpfs/alpine/csc499/scratch/alexisroger/neox/setup.sh

#source activate /ccs/home/$(whoami)/scratch/miniconda3/envs/magma
source activate /gpfs/alpine/csc499/scratch/alexisroger/neox/miniconda3/envs/magma

# export HF_DATASETS_CACHE=/gpfs/alpine/scratch/$(whoami)/csc499/cache/hugginface
export HF_DATASETS_CACHE=/gpfs/alpine/scratch/$(whoami)/csc499/cache/transformers
export HUGGINGFACE_HUB_CACHE=/gpfs/alpine/scratch/$(whoami)/csc499/cache/transformers
export HF_HOME=/gpfs/alpine/scratch/$(whoami)/csc499/cache/transformers
export TRANSFORMERS_CACHE=/gpfs/alpine/scratch/$(whoami)/csc499/cache/transformers
# export TRANSFORMERS_CACHE=/css/home/$(whoami)/.cache

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/$(whoami)/csc499/cache/torch_extensions

# Write the hostfile for this job
cat $LSB_DJOB_HOSTFILE | sort | uniq | tail -n +2 | sed -e 's/$/ slots=6/' > /ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts
export DLTS_HOSTFILE=/ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts

NNODES=$(wc -l < /ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts)

export WANDB_DIR=/gpfs/alpine/scratch/$(whoami)/csc499/wandb
export WANDB_MODE=dryrun

TRAIN_PATH=/gpfs/alpine/csc499/scratch/alexisroger/neox/lora
cd $TRAIN_PATH

python $TRAIN_PATH/deepy.py $TRAIN_PATH/generate.py \
	--conf_dir $TRAIN_PATH/configs magma_pythia_410M.yml magma_setup.yml text_generation.yml
