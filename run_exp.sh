# run all experiments using the grid search script
python -u -m survkit.grid_search \
        --group mnist_mixture \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 1_000 \
        --max_per_gpu 1 \
        --gpu_list 0 \
        > logs/mnist_mixture.log 2>&1

python -u -m survkit.grid_search \
        --group support2_mixture \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 1_000 \
        --max_per_gpu 1 \
        --gpu_list 0 \
        > logs/support2_mixture.log 2>&1

python -u -m survkit.grid_search \
        --group sepsis_mixture \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 1_000 \
        --max_per_gpu 1 \
        --gpu_list 0 \
        > logs/sepsis_mixture.log 2>&1