# Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads
This repository implements the models specified in the paper Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads. This work was published at the [Machine Learning for Health (ML4H) Symposium 2025](https://ahli.cc/ml4h/) and as a workshop paper at the [Learning from Time Series for Health](https://timeseries4health.github.io/) workshop at NeurIPS 2025.

## Project setup
If installing python packages into a virtual environment, run the following commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If using a conda environment, run the following commands
```bash
conda create -n survkit python=3.13 -y
conda activate survkit
pip install -r requirements.txt
```
**NB:** I've patched the Kaplan-Meier estimator from `torchsurv` to support GPU usage. If you want to use a GPU, you need to set the `device` argument in the `KaplanMeierEstimator` class to your desired device (e.g., "cuda" or "cpu"). Depending on the version of `torchsurv` you install, you may get this fix for free based on the Github issue I raised [here](https://github.com/Novartis/torchsurv/issues/113). If you still have errors, you can manually patch the file `site-packages/torchsurv/stats/kaplan_meier.py` following the changes proposed in the issue.

Log into Wandb to track experiments
```bash
wandb init
```

Also specify your Wandb `entity` in `survkit/configs/wandb.py`.

## Download datasets
Only Sepsis needs to be downloaded manually from [PhysioNet 2019 Sepsis Prediction Dataset](https://physionet.org/content/challenge-2019/1.0.0/). MNIST and SUPPORT2 will be downloaded automatically when you train your first models. You can use the following command to download the Sepsis data to the data directory
```bash
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/ -P data/
```

I would recommend running a single model training run for each of the 3 datasets (see below) to ensure everything is working correctly before running the full experiments.  This will ensure that only a single process is trying to download and preprocess the datasets rather than multiple processes trying to do so at the same time.

## Training a single model
To train a single model, use the `survkit.train` module with the desired experiment arguments. For example, to train a model on the SUPPORT2 dataset, run
```bash
python -m survkit.train --experiment_args args/support2.args
```
MNIST
```bash
python -m survkit.train --experiment_args args/mnist.args
```
Sepsis
```bash
python -m survkit.train --experiment_args args/sepsis.args
```

See all configs in `survkit/configs/train.py`, `survkit/configs/wandb.py`, and `survkit/configs/batch_run.py` for more details on the available arguments.

## Experiments
### Table 1 Experiments
To reproduce the Table 1 results, pick one of the following options based on your environment. All results will be logged to Wandb, which can be downloaded later for analysis. Note that RandomSurvivalForest (RSF) models may use a large amount of memory (~600GB) on the Survival MNIST dataset.

**Server environment** If running on a server (i.e., not on SLURM) use
```bash
nohup ./run_exp.sh > logs/run_exp.log 2>&1 &
```
See `survkit/configs/batch_run.py` for more details on how to gain more parallelism. You can either specify a list of available GPUs on the server using the `--gpu_list` argument or set `--max_subprocesses <n>` to run `<n>` experiments in parallel on your available GPUs. [`prefect`](https://docs.prefect.io/v3/get-started) handles local resource management.

**SLURM environment** To run on SLURM you will first need to add a configuration for your cluster. See `survkit/slurm/simple_slurm_script.py` for an example. Once you have your SLURM configuration set up, you'll need to add a conditional statement that calls your configuration in `survkit/slurm/slurm_launcher.py`'s `slurm_launch` function and specify the `cluster` command line argument when calling `grid_search.py`. After that, you can run the experiments using
```bash
nohup ./run_exp_slurm.sh > logs/run_exp_slurm.log 2>&1 &
```

**Analyzing Results**
After running the experiments, you can analyze the results using the notebook `notebooks/metrics_table.ipynb`.

### Expert sensistivity analysis (Figure 3)
 Train models with different numbers of experts and evaluate their performance.

**Server environment** If running on a server (i.e., not on SLURM) use
```bash
nohup ./run_expert_sensitivity.sh > logs/run_expert_sensitivity.log 2>&1 &
```

**Analyzing Results**
After running the experiments, you can analyze the results using the notebook `notebooks/expert_sensitivity.ipynb`.

### Routing analysis
See `notebooks/mnist_synthetic_clustering_fixed.ipynb`, `notebooks/mnist_synthetic_clustering_personalized.ipynb`, and `notebooks/support2_routing_analysis.ipynb` for more details.
