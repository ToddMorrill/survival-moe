"""Configuration for batch runs (e.g., when you want to run a grid search using Slurm, etc.). """
from dataclasses import dataclass, field
import logging

from .argparser import ArgParser

@dataclass
class BatchRunConfig:
    """Configuration for batch runs (e.g., when you want to run a grid search using Slurm, etc.)."""
    use_slurm: bool = field(
        default=False,
        metadata={"help": "Set to True to use Slurm, False to run locally"})
    cluster: str = field(
        default='insomnia',
        metadata={
            "help":
            "Slurm cluster to run on. This dicates which sbatch script gets called."
        })
    account: str = field(
        default='zgroup',
        metadata={
            "help":
            "Slurm account to run on."
        })
    partition: str = field(
        default='short', # 'none',
        metadata={
            "help":
            "Slurm cluster partition to run on."
        })
    qos: str = field(
        default='none',
        metadata={
            "help":
            "Slurm quality of service to use. If 'none', will not set qos."
        })
    use_threads: bool = field(default=True, metadata={"help": "If False, each command will run sequentially in a single worker. If True, launch jobs in subprocesses from threads."})
    max_subprocesses: int = field(default=8, metadata={"help": "Maximum number of subprocesses to run in parallel."})
    cpu: int = field(
        default=8, metadata={"help": "Number of CPUs to use per task (Slurm)"})
    gpu: int = field(
        default=1, metadata={"help": "Number of GPUs to use per task (Slurm)"})
    time: str = field(default='12:00:00',
                      metadata={"help": "Maximum running time for Slurm jobs. 5 days for zgroup partition. 12 hours for the short partition: 12:00:00"})
    memory: str = field(
        default='64G', metadata={"help": "CPU memory allocation per task (Slurm)"})
    slurm_name: str = field(
        default=None,
        metadata={
            "help": "User-specified Slurm job name or generated from tasks"
        })
    out_path: str = field(
        default=None,
        metadata={
            "help":
            "User-specified output log file path or generated from tasks and testing"
        })
    gpu_mb_per_job: int = field(
        default=-1,
        # estimate MB required per job, then run as many jobs per GPU as possible
        metadata={"help": "MBs required per job so that we can stack multiple jobs on a single GPU. This is useful when you have powerful GPUs that can handle multiple tasks at once. See slurm_launcher.py for more details. If -1, will not stack jobs."})
    max_per_gpu: int = field(
        default=-1,
        # maximum number of jobs to run on a single GPU, -1 means no limit
        metadata={"help": "Maximum number of jobs to run on a single GPU. This is useful to limit the number of jobs per GPU even if there is enough memory. If -1, will not limit the number of jobs per GPU."})
    gpu_list: list[int] = field(
        default_factory=lambda: [],
        metadata={
            "help":
            "List of GPU indices to use. If empty, will use all available GPUs."
        })
    num_runs: int = field(
        default=-1,
        metadata={
            "help":
            "Number of grid search runs to perform, particularly with large search spaces. -1 means all possible combinations, and a positive integer will randomly subsample the search space and run that many combinations."
        })

    def __post_init__(self):
        # assert that use_slurm and use_subprocess are not both True
        if self.use_slurm and self.use_threads:
            raise ValueError(
                'Only one of use_slurm and use_threads can be set to True.')
        if self.out_path is not None:
            logging.warning("--out_path is deprecated and has no effect. Use --log_dir instead.")

if __name__ == '__main__':
    parser = ArgParser([BatchRunConfig])
    batchrunconfig, unknown_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    print(batchrunconfig)