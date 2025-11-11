# run all experiments using the grid search script
python -u -m survkit.grid_search \
        --group support2_mixture \
        --use_slurm True \
        --use_threads False \
        > logs/support2_mixture.log 2>&1

python -u -m survkit.grid_search \
        --group sepsis_mixture \
        --use_slurm True \
        --use_threads False \
        > logs/sepsis_mixture.log 2>&1

python -u -m survkit.grid_search \
        --group mnist_mixture \
        --use_slurm True \
        --use_threads False \
        --memory 640G \
        --cpu 64 \
        > logs/mnist_mixture.log 2>&1