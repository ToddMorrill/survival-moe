import logging
import os
import shlex
import subprocess
from typing import Dict, List, Optional
import queue
import threading

import gpustat
from prefect import task, flow
from prefect.context import get_run_context
from prefect.logging import get_run_logger
from prefect.cache_policies import NO_CACHE
from survkit.configs.batch_run import BatchRunConfig
from survkit.slurm.simple_slurm_script import insomnia_sbatch

def get_gpu_slots(gpu_mb_per_job, gpu_list, max_per_gpu=-1):
    """Fills up GPUs in the list with jobs until they are full."""
    # check if gpus are available
    if gpustat.gpu_count() == 0:
        return []

    stats = gpustat.GPUStatCollection.new_query()
    free_gpu_slots = []
    for gpu in stats:
        # if gpu.index not in gpu_list and gpu_list is not empty, skip
        if gpu.index not in gpu_list and gpu_list:
            continue
        # if the GPU is busy (more than 10GB in use), skip
        if gpu.memory_used > 8_000:
        # if gpu.utilization > 0:
            continue
        # if -1, don't stack jobs
        if gpu_mb_per_job == -1:
            free_gpu_slots.append(gpu.index)
        else:
            num_slots = gpu.memory_free // gpu_mb_per_job
            if max_per_gpu > 0:
                num_slots = min(num_slots, max_per_gpu)
            for _ in range(num_slots):
                free_gpu_slots.append(gpu.index)
    return free_gpu_slots

def worker(jobs_queue, gpu_id):
    while True:
        command_dict = jobs_queue.get()
        if command_dict is None:  # Check for the sentinel value to stop the worker
            jobs_queue.task_done()
            break

        command = command_dict['command']
        if gpu_id is not None:
            # modify the --device argument to use the GPU id
            # NB: assume the --device argument will be of the form
            # --device cpu or --device cuda or --device cuda:0
            device_idx = [idx for idx, arg in enumerate(command) if arg.startswith('--device')][0] + 1 # +1 to get the value of the argument
            # raise error if device already has a GPU id and it's not the one we want to use
            if ':' in command[device_idx] and command[device_idx].split(':')[1] != gpu_id:
                raise ValueError(f'GPU ID already specified for device: {command[device_idx]}')
            command[device_idx] = command[device_idx] + f':{0}' # since we set CUDA_VISIBLE_DEVICES, we can just use 0

        command_str = shlex.join(command)

        # run the command using subprocess
        with open(command_dict['log_file_path'], 'w') as f:
            f.write(f'Command: {command_str}\n')
            # flush the buffer to ensure the command is written before running program
            f.flush()
            logging.info(f'Running command: {command_str}')

            # catch any exceptions and write to log file, this way if e.g., a job runs out out memory, we can start a new job that might succeed
            try:
                env = os.environ.copy()  # copy current environment
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # set GPU restriction
                subprocess.run(command, check=True, stdout=f, stderr=f, env=env)
            except subprocess.CalledProcessError as e:
                f.write(f'Error: {e}\n')
                logging.error(f'Error running command: {command_str}')

        jobs_queue.task_done()

def local_launch(batch_run_config: BatchRunConfig, commands: list[dict]):
    gpu_slots = get_gpu_slots(batch_run_config.gpu_mb_per_job, batch_run_config.gpu_list, batch_run_config.max_per_gpu)

    # may be a cpu job in which case gpu_slots will be empty and we can fall back to batch_run_config.max_subprocesses
    if not gpu_slots:
        gpu_slots = [None] * min(batch_run_config.max_subprocesses, len(commands))

    # enqueue all the commands
    jobs_queue = queue.Queue()
    for command in commands:
        jobs_queue.put(command)

    # create a worker thread that will fire off subprocesses for each element in gpu_slots
    if batch_run_config.use_threads:
        threads = []
        for gpu_id in gpu_slots:
            thread = threading.Thread(target=worker, args=(jobs_queue, gpu_id))
            thread.start()
            threads.append(thread)

        # Enqueue sentinel values to signal the workers to exit
        for _ in range(len(gpu_slots)):
            jobs_queue.put(None)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    else:
        jobs_queue.put(None)
        worker(jobs_queue, None)

@task(cache_policy=NO_CACHE)
def run_command_task(command_dict: dict, resource_queue: queue.Queue):
    """
    A Prefect task to acquire a compute resource (like a GPU ID),
    run a command, and then release the resource.
    """
    logger = get_run_logger()
    job_name = command_dict['name']
    
    # acquire a resource from the shared queue
    resource_id = resource_queue.get()
    
    try:
        logger.info(f"▶️ Starting job '{job_name}' on resource '{resource_id}'...")
        
        env = os.environ.copy()
        if resource_id is not None:
            # set CUDA_VISIBLE_DEVICES to isolate the GPU for the subprocess
            env["CUDA_VISIBLE_DEVICES"] = str(resource_id)

        command_str = shlex.join(command_dict['command'])
        
        # run the command, logging output to the specified file
        with open(command_dict['log_file_path'], 'w') as log_file:
            log_file.write(f"Running on resource (GPU ID): {resource_id}\n")
            log_file.write(f"Command: {command_str}\n\n")
            log_file.flush()
            
            # prefect automatically captures failures if check=True
            subprocess.run(
                command_dict['command'],
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT, # redirect stderr to stdout (and thus to the log file)
                env=env
            )
            
        logger.info(f"✅ Finished job '{job_name}' successfully.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Job '{job_name}' failed with exit code {e.returncode}.")
        # re-raise the exception to make the Prefect task fail
        raise
    finally:
        # release the resource back to the queue
        resource_queue.put(resource_id)
        logger.info(f"Resource '{resource_id}' released by job '{job_name}'.")

@flow()
def prefect_job_launcher(
    batch_run_config: BatchRunConfig,
    commands: List[Dict],
    depends_on_model_hashes: Optional[Dict] = None
):
    """
    A Prefect flow to manage and execute a DAG of jobs locally.
    """
    logger = get_run_logger()
    run_id = get_run_context().flow_run.id
    print(f"🚀 Flow Run ID: {run_id}")
    logger.info(f"Starting job launch for {len(commands)} commands.")

    # create a queue of available compute resources (GPUs or CPU slots)
    resource_queue = queue.Queue()
    slots = get_gpu_slots(batch_run_config.gpu_mb_per_job, batch_run_config.gpu_list, batch_run_config.max_per_gpu)
    if not slots:
        logger.info(f"No GPUs found or specified. Using {batch_run_config.max_subprocesses} CPU slots.")
        for _ in range(batch_run_config.max_subprocesses):
            resource_queue.put(None) # None represents a CPU slot
    else:
        logger.info(f"Populating resource queue with GPU IDs: {slots}")
        for slot in slots:
            resource_queue.put(slot)

    # set up the dependency graph structure for Prefect
    # assumes commands are already topologically sorted
    futures = {} # maps job name to its future
    for i, command_dict in enumerate(commands):
        job_name = command_dict['name']
        dependencies = depends_on_model_hashes.get(job_name, []) if depends_on_model_hashes else []
        logger.info(f"Job '{job_name}' depends on: {dependencies}")
        
        # get the Prefect future objects for all dependencies
        wait_for = [futures[dep_name] for dep_name in dependencies if dep_name in futures]
        
        # submit the task to the runner
        # Prefect automatically respects the `wait_for` dependencies.
        # the task will only run after its dependencies are met AND
        # the queue has a free worker
        future = run_command_task.submit(
            command_dict,
            resource_queue,
            wait_for=wait_for
        )
        futures[job_name] = future
    
    # wait for all jobs to complete
    logger.info("Waiting for all jobs to complete...")
    for job_name, future in futures.items():
        try:
            future.result()  # this will block until the task is done
            logger.info(f"Job '{job_name}' completed successfully.")
        except Exception as e:
            logger.error(f"Job '{job_name}' failed with error: {e}")
        
    logger.info("All jobs have been submitted. Waiting for completion...")

def slurm_launch(batch_run_config: BatchRunConfig,
                 commands: list[dict],
                 depends_on_model_hashes: Optional[dict] = None):
    # assumes commands are already topologically sorted
    model_hash_to_job_id = {}
    for i, command_dict in enumerate(commands):
        command = command_dict['command']
        model_hash = command_dict['name']
        command_str = shlex.join(command)

        print(f'Launching {command_dict["name"]} with command: {command_str}')
        if batch_run_config.cluster == 'insomnia':
            log_file_path = command_dict['log_file_path']
            job_name = command_dict['name']
            depends_on = depends_on_model_hashes.get(model_hash, []) if depends_on_model_hashes else []
            depends_on_job_ids = [model_hash_to_job_id[dep] for dep in depends_on if dep in model_hash_to_job_id]
            # extend https://github.com/amq92/simple_slurm/blob/a6fcdf6ccd300d9ad63300d152c20177478ae84b/simple_slurm/core.py#L304-L305
            # to support multiple dependencies, not just one
            slurm, job_id = insomnia_sbatch(
                command_str,
                batch_run_config,
                job_name,
                log_file_path,
                depends_on=depends_on_job_ids,)
            model_hash_to_job_id[command_dict['name']] = job_id
        else:
            raise ValueError(
                f'Unknown cluster: {batch_run_config.cluster}')

def launch(
        commands: list[dict],
        batch_run_config: BatchRunConfig,
        depends_on_model_hashes: Optional[list[str]] = None,
):
    print(f'Launching {len(commands)} jobs')

    # if running locally
    if not batch_run_config.use_slurm:
        prefect_job_launcher(batch_run_config, commands, depends_on_model_hashes)
        # local_launch(batch_run_config, commands)
    else:
        slurm_launch(batch_run_config, commands, depends_on_model_hashes)
