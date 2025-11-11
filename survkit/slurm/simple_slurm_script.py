from simple_slurm import Slurm

def insomnia_sbatch(command_str, batch_run_config, job_name, log_file_path, submit=True, depends_on=None,):
    slurm = Slurm(account=batch_run_config.account,
                  job_name=job_name,
                  partition=batch_run_config.partition,
                  qos=batch_run_config.qos,
                  gres=f'gpu:{batch_run_config.gpu}' if batch_run_config.gpu > 0 else 'none',
                  mem=batch_run_config.memory,
                  cpus_per_task=batch_run_config.cpu,
                  time=batch_run_config.time,
                  output=log_file_path,
                  exclude='ins093,ins094,ins086')
    if depends_on:
        dependency_str = 'afterok:' + ':'.join(depends_on)
        slurm.set_dependency(dependency_str)
    slurm.set_shell("/bin/bash")
    # set tmpdir because the default /tmp may be cleared on every job
    # if the cluster doesn't have job-level isolation
    slurm.add_cmd(f'JOB_TMPDIR="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/tm3229/survkit/tmp/$SLURM_JOB_ID"')
    # set trap to clean up tmpdir on exit
    slurm.add_cmd("trap 'rm -rf \"${JOB_TMPDIR}\"' EXIT")
    # create tmpdir
    slurm.add_cmd('mkdir -p $JOB_TMPDIR')
    # set tmpdir
    slurm.add_cmd('export TMPDIR=$JOB_TMPDIR')
    # set wandb data directory for staging artifacts
    slurm.add_cmd('export WANDB_DATA_DIR=$JOB_TMPDIR/wandb_data_dir')
    slurm.add_cmd('export WANDB_CACHE_DIR=$JOB_TMPDIR/wandb_cache_dir')
    # load conda env
    slurm.add_cmd('module load anaconda/2023.09')
    slurm.add_cmd('eval "$(conda shell.bash hook)"')
    slurm.add_cmd('conda activate survkit')
    slurm.add_cmd(command_str)
    if submit:
        job_id = slurm.sbatch()
        return slurm, job_id
    return slurm

# if __name__ == '__main__':
#     from survkit.configs.batch_run import BatchRunConfig
#     batch_run_config = BatchRunConfig(use_slurm=True, partition='short', time='12:00:00', use_threads=False)
#     # check if GPU is available
#     command = ['python -c "import time, torch; print(\'CUDA found: \', torch.cuda.is_available()); time.sleep(10)"']
#     job_name = 'torch_test'
#     log_file_path = './torch_test.log'
#     slurm, job_id = manitou_sbatch(' '.join(command), batch_run_config, job_name, log_file_path)
#     print(f'Launched job {job_name} with SLURM job ID {job_id}')
#     print(f'Slurm script:\n{slurm}')
