#!/bin/bash -x

#SBATCH --account=laionize
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --partition=develbooster
#SBATCH --job-name=hbsmall

# load low-level libraries
module load Stages/2023 GCC/11.3.0  OpenMPI/4.1.4
ml git
source /p/project/ccstdl/gupta6/miniconda3/bin/activate
conda activate multimodal_flash_env

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export LOCAL_WORLD_SIZE=4
export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR

bash /p/project/ccstdl/gupta6/write_hostfile.sh
export DLTS_HOSTFILE=/p/project/ccstdl/gupta6/hostfiles/hosts_$SLURM_JOBID

# GPT_NEOX_HOME="/p/project/laionize/jitsev1_juwelsbooster/open_clip_lr_mod"
GPT_NEOX_HOME="/p/project/ccstdl/gupta6/multimodal"
export PYTHONPATH="$PYTHONPATH:${GPT_NEOX_HOME}/src"
export PYTHONPATH="${PYTHONPATH}:/p/project/ccstdl/gupta6/multimodal"

# export PYTHONPATH="$PYTHONPATH:${HOME}/home/open_clip/src"

cd ${GPT_NEOX_HOME}
# wandb offline

# config_file="/p/project/ccstdl/gupta6/multimodal/configs/local_setup.yml"
# temp_config_file="temp_config_file.json"
# cp $config_file $temp_config_file

# python -c "import json; \
#            config = json.load(open('$temp_config_file', 'r')); \
#            config['master_addr'] = '$MASTER_ADDR'; \
#            json.dump(config, open('$temp_config_file', 'w'));"
wandb online
python ./deepy.py train.py configs/1-3B.yml configs/hummingbird_streaming.yml