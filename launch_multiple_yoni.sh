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
           --exclude="rack-omerl-g01" \
           --constraint="geforce_rtx_3090" \
           --partition=killable \
           --wrap="nvidia-smi && cd /home/dcor/jh1/code/gnn/ && export PYTHONUNBUFFERED=1 && /home/dcor/jh1/miniforge3/envs/gnn/bin/python train.py ${config_file}"
}

for file in configs/param_sweep/*; do
    echo $file
    gnn_3090 $file
done