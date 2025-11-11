python -u -m survkit.grid_search \
        --group mnist_expert_sensitivity \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 1_000 \
        --max_per_gpu 1 \
        > logs/mnist_expert_sensitivity.log 2>&1

python -u -m survkit.grid_search \
        --group support2_expert_sensitivity \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 1_000 \
        --max_per_gpu 1 \
        > logs/support2_expert_sensitivity.log 2>&1

python -u -m survkit.grid_search \
        --group sepsis_expert_sensitivity \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 1_000 \
        --max_per_gpu 1 \
        > logs/sepsis_expert_sensitivity.log 2>&1