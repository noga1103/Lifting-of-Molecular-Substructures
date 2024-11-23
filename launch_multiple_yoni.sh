gnn_3090() {
    local config_file=$1

    sbatch --job-name=run \
           --output=./output/%j.out \
           --account=gpu-research \
           --time=1440 \
           --signal=USR1@120 \
           --nodes=1 \
           --mem=8000 \
           --cpus-per-gpu=1 \
           --gpus=1 \
           --exclude="rack-omerl-g01,n-501,n-301,n-307,n-305" \
           --constraint="geforce_rtx_3090|a100|a5000|a6000|quadro_rtx_8000|tesla_v100" \
           --partition=killable \
           --wrap="nvidia-smi && cd /home/dcor/jh1/code/gnn/ && export WANDB_API_KEY="5b89f619aac9924356bc9455dcee6df7659defc8" && export PYTHONUNBUFFERED=1 && /home/dcor/jh1/miniforge3/envs/gnn/bin/python train.py ${config_file}"
}

for file in configs/still_not_run/*; do
    echo $file
    gnn_3090 $file
done
