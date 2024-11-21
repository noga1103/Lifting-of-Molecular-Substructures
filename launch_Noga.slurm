#! /bin/sh

#SBATCH --job-name=run
#SBATCH --output=./output/%j.out # redirect stdout
#SBATCH --account=gpu-research
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
##SBATCH --mem=16000 # CPU memory (MB)
##SBATCH --cpus-per-gpu=2 
#SBATCH --gpus=1 # GPUs in total
#SBATCH --exclude="rack-omerl-g01,rack-gww-dgx1,n-501,n-301,n-307"

# All the options:
# "geforce_rtx_3090|a100|a5000|a6000|quadro_rtx_8000|tesla_v100" # GPU type. Supported Features: tesla_v100, quadro_rtx_8000, geforce_rtx_3090, titan_xp, geforce_rtx_2080,a100,a5000,a6000

##SBATCH --constraint="a6000|quadro_rtx_8000|tesla_v100|l40s"
##SBATCH --partition=killable

##SBATCH --constraint="l40s"
##SBATCH --partition=killable
##SBATCH --nodelist="n-803"

#SBATCH --constraint="geforce_rtx_3090"
#SBATCH --partition=killable

##SBATCH --constraint="a100" 
##SBATCH --partition=gpu-a100-killable

nvidia-smi

cd /home/dcor/jh1/code/gnn/

PYTHONUNBUFFERED=1
WANDB_API_KEY="5b89f619aac9924356bc9455dcee6df7659defc8"

/home/dcor/jh1/miniforge3/envs/gnn/bin/python train.py configs/param_sweep_3/HNHN_small.json
