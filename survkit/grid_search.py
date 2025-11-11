"""This module launches batch runs (e.g., when you want to run a grid search using Slurm, etc.).

Examples:
    $ nohup python -u -m survkit.grid_search \
        --group mnist_mixture \
        --use_slurm False \
        --use_threads True \
        --gpu_mb_per_job 70_000 \
        > logs/mnist_mixture.log 2>&1 &
    
    $ nohup python -u -m survkit.grid_search \
        --group support2_mixture \
        --use_slurm False \
        --use_threads True \
        > logs/support2_mixture.log 2>&1 &
"""
import copy
from dataclasses import fields
from itertools import product
import itertools
import logging
import os
import graphlib

import numpy as np
import pandas as pd
import wandb

from survkit.train import create_model_hash

from .configs import WandbConfig, BatchRunConfig, TrainConfig, ArgParser
from .slurm.slurm_launcher import launch

# set numpy's random seed so shuffling is consistent
np.random.seed(42)

def product_from_dict(params):
    """Create a product from a set of keys"""
    keys, values = zip(*params.items())
    runs = []
    for bundle in product(*values):
        d = dict(zip(keys, bundle))
        runs.append(d)
    return runs


def product_from_dicts(params_1, params_2):
    """Create a product from two param lists of (partial) dictionaries"""
    runs = []
    for param_1, param_2 in product(params_1, params_2):
        merged = {}
        merged.update(param_1)
        merged.update(param_2)
        runs.append(merged)
    return runs


def is_existing_run(run_df, full_run_spec, ignore_keys=[]):
    # there might be some runs that accidentally have config.param columns, which will invalidate the cache hit
    # so remove any columns that not in the set of keys in full_run_spec
    subset_run_df = run_df[[k for k in run_df.columns if k in full_run_spec.keys()]]
    row_filter = []
    for k, v in full_run_spec.items():
        if k in ['log_dir', 'device', 'save_model', 'model_hash'] + ignore_keys:
            # skip because model_hash isn't in the full_run_spec by this point in the script
            continue

        if k not in subset_run_df.columns:
            # set to True if the column is not in the dataframe, which means we don't need to filter on it
            row_filter.append(pd.Series([True] * len(subset_run_df)))
        elif isinstance(v, list):
            row_filter.append(subset_run_df[f'{k}'].apply(lambda x: v == x))
        elif v is None:
            row_filter.append(subset_run_df[f'{k}'].isna())
        else:
            row_filter.append(subset_run_df[f'{k}'] == v)
    already_ran = pd.concat(row_filter, axis=1).all(axis=1).any()
    return already_ran


def support2():
    """SUPPORT2 experiments."""

    # baseline runs, no mixture model
    default_params = {
        'seed': [42, 43, 44, 45, 46],
        'model': ['FFNet'],
        'hidden_dim': [256], # ensure there are more parameters than the mixture model for a fair comparison
        'experiment_args': [
            'args/support2.args',
        ],
    }
    default_runs = product_from_dict(default_params)

    # mixture model runs
    mixture_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [128],
        'mixture_components': [8],
        'mixture_topk': [8],
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixtureMTLR', 'FFNetTimeWarpMoE', 'FFNetMixture'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/support2.args',
        ],
    }
    mixture_runs = product_from_dict(mixture_params)

    # sksurv runs
    sksurv_params = {
        'seed': [42, 43, 44, 45, 46],
        'model': ['SKSurvRF', 'SKSurvCox'],
        'experiment_args': [
            'args/support2.args',
        ],
    }
    sksurv_params_runs = product_from_dict(sksurv_params)

    runs = mixture_runs + default_runs + sksurv_params_runs
    # runs = sksurv_params_runs

    # make a few tweaks to the parameter counts
    modifications = {'FFNet': {'hidden_dim': 176},
                     'FFNetMixture': {'hidden_dim': 208, 'mixture_components': 10, 'mixture_topk': 10, 'lr': 5e-3},
                     'FFNetTimeWarpMoE': {'hidden_dim': 186, 'mixture_components': 10, 'mixture_topk': 10, 'lr': 5e-3},
                     'FFNetMixtureMTLR': {'hidden_dim': 128, 'num_hidden_layers': 1, 'mixture_components': 8, 'mixture_topk': 8},}
    for run in runs:
        modifications_for_model = modifications.get(run.get('model'), {})
        for k, v in modifications_for_model.items():
            run[k] = v

    # modify the SKSurvRF runs
    modifications = {'SKSurvRF': {'rsf_n_estimators': 200, 'rsf_max_features': 'sqrt', 'rsf_min_samples_split': 200},}
    for run in runs:
        modifications_for_model = modifications.get(run.get('model'), {})
        for k, v in modifications_for_model.items():
            run[k] = v

    # pair up by seed
    # give each run a 'depends_on' key that points to the corresponding mixture run
    # index according to order in runs
    depends_on = {}
    # for idx, run in enumerate(mixture_runs):
    #     depends_on[idx] = []
    # for idx, run in enumerate(default_runs):
    #     depends_on[len(default_runs) + idx] = [idx] # no dependencies for mixture runs
    return runs, depends_on

def support2_expert_sensitivity():
    """Vary the number of experts in the mixture models to see how performance varies."""
    # mixture model runs
    fixed_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [208],
        'mixture_components': [2, 5, 10, 20],
        # 'mixture_topk': [8], # below, match mixture_topk to mixture_components
        'group_ent_lambda': [0.01],
        'lr': [5e-3],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixture'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/support2.args',
        ],
    }
    fixed_moe_runs = product_from_dict(fixed_moe_params)

    adjustable_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [186],
        'mixture_components': [2, 5, 10, 20],
        # 'mixture_topk': [8], # below, match mixture_topk to mixture_components
        'group_ent_lambda': [0.01],
        'lr': [5e-3],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetTimeWarpMoE'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/support2.args',
        ],
    }
    adjustable_moe_runs = product_from_dict(adjustable_moe_params)

    personalized_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [128],
        'num_hidden_layers': [1],
        'mixture_components': [2, 4, 8, 16],
        # 'mixture_topk': [8], # below, match mixture_topk to mixture_components
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixtureMTLR'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/support2.args',
        ],
    }
    personalized_moe_runs = product_from_dict(personalized_moe_params)
    runs = fixed_moe_runs + adjustable_moe_runs + personalized_moe_runs

    for run in runs:
        # match mixture_topk to mixture_components for each run
        run['mixture_topk'] = run['mixture_components']

    depends_on = {}
    return runs, depends_on

def sepsis():
    """Sepsis experiments."""

    # baseline runs, no mixture model
    default_params = {
        'seed': [42, 43, 44, 45, 46],
        'model': ['FFNet'],
        'hidden_dim': [256], # ensure there are more parameters than the mixture model for a fair comparison
        'experiment_args': [
            'args/sepsis.args',
        ],
    }
    default_runs = product_from_dict(default_params)

    # mixture model runs
    mixture_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [128],
        'mixture_components': [8],
        'mixture_topk': [8],
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixtureMTLR', 'FFNetTimeWarpMoE', 'FFNetMixture'],
        'learn_temperature': [True],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/sepsis.args',
        ],
    }
    mixture_runs = product_from_dict(mixture_params)

    # sksurv runs
    sksurv_params = {
        'seed': [42, 43, 44, 45, 46],
        'model': ['SKSurvRF', 'SKSurvCox'],
        'experiment_args': [
            'args/sepsis.args',
        ],
    }
    sksurv_params_runs = product_from_dict(sksurv_params)
    runs = mixture_runs + default_runs + sksurv_params_runs
    # runs = sksurv_params_runs

    # make a few tweaks to the parameter counts
    modifications = {'FFNet': {'hidden_dim': 176},
                     'FFNetMixture': {'hidden_dim': 208, 'mixture_components': 10, 'mixture_topk': 10},
                     'FFNetTimeWarpMoE': {'hidden_dim': 186, 'mixture_components': 10, 'mixture_topk': 10},
                     'FFNetMixtureMTLR': {'hidden_dim': 128, 'num_hidden_layers': 1, 'mixture_components': 8, 'mixture_topk': 8},}
    for run in runs:
        modifications_for_model = modifications.get(run.get('model'), {})
        for k, v in modifications_for_model.items():
            run[k] = v

    # modify the SKSurvRF runs
    modifications = {'SKSurvRF': {'rsf_n_estimators': 50, 'rsf_max_features': 'sqrt', 'rsf_min_samples_split': 50},}
    for run in runs:
        modifications_for_model = modifications.get(run.get('model'), {})
        for k, v in modifications_for_model.items():
            run[k] = v
    # pair up by seed
    # give each run a 'depends_on' key that points to the corresponding mixture run
    # index according to order in runs
    depends_on = {}
    # for idx, run in enumerate(mixture_runs):
    #     depends_on[idx] = []
    # for idx, run in enumerate(default_runs):
    #     depends_on[len(default_runs) + idx] = [idx] # no dependencies for mixture runs
    return runs, depends_on

def sepsis_expert_sensitivity():
    """Vary the number of experts in the mixture models to see how performance varies."""
    # mixture model runs
    fixed_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [208],
        'mixture_components': [2, 5, 10, 20],
        # 'mixture_topk': [8], # below, match mixture_topk to mixture_components
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixture'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/sepsis.args',
        ],
    }
    fixed_moe_runs = product_from_dict(fixed_moe_params)

    adjustable_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [186],
        'mixture_components': [2, 5, 10, 20],
        # 'mixture_topk': [8], # below, match mixture_topk to mixture_components
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetTimeWarpMoE'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/sepsis.args',
        ],
    }
    adjustable_moe_runs = product_from_dict(adjustable_moe_params)

    personalized_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [128],
        'num_hidden_layers': [1],
        'mixture_components': [2, 4, 8, 16],
        # 'mixture_topk': [8], # below, match mixture_topk to mixture_components
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixtureMTLR'],
        'logit_temp': [2.0],
        'experiment_args': [
            'args/sepsis.args',
        ],
    }
    personalized_moe_runs = product_from_dict(personalized_moe_params)
    runs = fixed_moe_runs + adjustable_moe_runs + personalized_moe_runs

    for run in runs:
        # match mixture_topk to mixture_components for each run
        run['mixture_topk'] = run['mixture_components']

    depends_on = {}
    return runs, depends_on


def mnist():
    """MNIST experiments."""

    # baseline runs, no mixture model
    default_params = {
        'seed': [42, 43, 44, 45, 46],
        'model': ['FFNet'],
        'hidden_dim': [256], # ensure there are more parameters than the mixture model for a fair comparison
        'time_bins': [100],
        'mnist_censoring_percentage': [0.15],
        'adjust_eval': [True], # adjust for censoring with
        'ipcw': [True], # use IPCW for loss
        'experiment_args': [
            'args/mnist.args',
        ],
    }
    default_runs = product_from_dict(default_params)

    # mixture model runs
    mixture_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [128],
        'time_bins': [100],
        'mnist_censoring_percentage': [0.15],
        'adjust_eval': [True], # adjust for censoring with
        'ipcw': [True], # use IPCW for loss
        'mixture_components': [8],
        'mixture_topk': [8],
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixtureMTLR', 'FFNetTimeWarpMoE', 'FFNetMixture'],
        'logit_temp': [2.0],
        'learn_temperature': [True],
        'experiment_args': [
            'args/mnist.args',
        ],
    }
    mixture_runs = product_from_dict(mixture_params)

    # sksurv runs
    sksurv_params = {
        'seed': [42, 43, 44, 45, 46],
        'model': ['SKSurvCox', 'SKSurvRF',],
        'time_bins': [100],
        'mnist_censoring_percentage': [0.15],
        'adjust_eval': [True], # adjust for censoring with
        'ipcw': [True], # use IPCW for loss
        'experiment_args': [
            'args/mnist.args',
        ],
    }
    sksurv_params_runs = product_from_dict(sksurv_params)

    runs = mixture_runs + default_runs + sksurv_params_runs
    # runs = sksurv_params_runs
    # make a few tweaks to the parameter counts
    modifications = {'FFNet': {'hidden_dim': 176},
                     'FFNetMixture': {'hidden_dim': 208, 'mixture_components': 10, 'mixture_topk': 10},
                     'FFNetTimeWarpMoE': {'hidden_dim': 186, 'mixture_components': 10, 'mixture_topk': 10},
                     'FFNetMixtureMTLR': {'hidden_dim': 160, 'num_hidden_layers': 1, 'mixture_components': 10, 'mixture_topk': 10},}
    for run in runs:
        modifications_for_model = modifications.get(run.get('model'), {})
        for k, v in modifications_for_model.items():
            run[k] = v

    # pair up by seed
    # give each run a 'depends_on' key that points to the corresponding mixture run
    # index according to order in runs
    depends_on = {}
    # for idx, run in enumerate(mixture_runs):
    #     depends_on[idx] = []
    # for idx, run in enumerate(default_runs):
    #     depends_on[len(default_runs) + idx] = [idx] # no dependencies for mixture runs
    return runs, depends_on

def mnist_expert_sensitivity():
    """Vary the number of experts in the mixture models to see how performance varies."""

    # mixture model runs
    fixed_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [208],
        'time_bins': [100],
        'mnist_censoring_percentage': [0.15],
        'adjust_eval': [True], # adjust for censoring with
        'ipcw': [True], # use IPCW for loss
        'mixture_components': [2, 5, 10, 20],
        # 'mixture_topk': [8],
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixture'],
        'logit_temp': [2.0],
        'learn_temperature': [True],
        'experiment_args': [
            'args/mnist.args',
        ],
    }
    fixed_moe_runs = product_from_dict(fixed_moe_params)

    adjustable_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [186],
        'time_bins': [100],
        'mnist_censoring_percentage': [0.15],
        'adjust_eval': [True], # adjust for censoring with
        'ipcw': [True], # use IPCW for loss
        'mixture_components': [2, 5, 10, 20],
        # 'mixture_topk': [8],
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetTimeWarpMoE'],
        'logit_temp': [2.0],
        'learn_temperature': [True],
        'experiment_args': [
            'args/mnist.args',
        ],
    }
    adjustable_moe_runs = product_from_dict(adjustable_moe_params)

    personalized_moe_params = {
        'seed': [42, 43, 44, 45, 46],
        'hidden_dim': [160],
        'num_hidden_layers': [1],
        'time_bins': [100],
        'mnist_censoring_percentage': [0.15],
        'adjust_eval': [True], # adjust for censoring with
        'ipcw': [True], # use IPCW for loss
        'mixture_components': [2, 4, 8, 16],
        # 'mixture_topk': [8],
        'group_ent_lambda': [0.01],
        'instance_ent_lambda': [0.0],
        'model': ['FFNetMixtureMTLR'],
        'logit_temp': [2.0],
        'learn_temperature': [True],
        'experiment_args': [
            'args/mnist.args',
        ],
    }
    personalized_moe_runs = product_from_dict(personalized_moe_params)
    runs = fixed_moe_runs + adjustable_moe_runs + personalized_moe_runs
    for run in runs:
        # match mixture_topk to mixture_components for each run
        run['mixture_topk'] = run['mixture_components']

    depends_on = {}
    return runs, depends_on


def process_full_run_spec(run, wandb_config_keys, config_fields):
    # need to accommodate the experiment_args parameter, which specifies a file filled with arguments
    # the priority of arguments is as follows:
    # 1. command line arguments specified in the run
    # 2. arguments specified in the experiment_args file
    # 3. default arguments in TrainConfig

    # solution: create all the command line arguments and use the parser, which already handles this ordering
    args = []
    for k, v in run.items():
        # filter out wandb_config keys
        if k in wandb_config_keys:
            continue
        # unpack lists
        if isinstance(v, list):
            args.extend([f'--{k}'] + [str(val) for val in v])
        else:
            args.extend([f'--{k}', f'{v}'])

    # use the existing parser to parse the arguments
    parser = ArgParser([TrainConfig, WandbConfig])
    train_config, wandb_config_, unknown_args = parser.parse_args_into_dataclasses(
        args=args,
        return_remaining_strings=True,
        args_file_flag='--experiment_args',
    )

    full_run_spec = {
        **train_config.__dict__,
    }

    # remove any arguments with None values
    full_run_spec = {
        k: v
        for k, v in full_run_spec.items() if v is not None
    }

    # also remove any arguments that are not in the TrainConfig (otherwise, they'll get rejected when sent to the parser)
    full_run_spec = {
        k: v
        for k, v in full_run_spec.items() if k in config_fields
    }

    # also remove any arguments that are empty lists
    full_run_spec = {
        k: v
        for k, v in full_run_spec.items()
        if not (isinstance(v, list) and len(v) == 0)
    }

    # get model hash (can optionally exclude any newly added keys from the hash and not the run parameters, which is useful for new params)
    model_hash = create_model_hash(full_run_spec, exclude_keys=['model_hash'])
    full_run_spec['model_hash'] = model_hash
    return full_run_spec, model_hash

def remove_model_hash_from_depends_on_dict(depends_on_dict, model_hash):
    """Remove the model hash from the depends_on_dict."""
    # loop through the depends_on_dict and remove the model_hash from the dependencies
    for k, v in depends_on_dict.items():
        if model_hash in v:
            v.remove(model_hash)
    # also remove the key itself
    depends_on_dict.pop(model_hash, None)

def preprocess_runs(runs, wandb_config_keys, config_fields, depends_on):
    full_run_specs = []
    model_hashes = []
    for idx, run in enumerate(runs):
        full_run_spec, model_hash = process_full_run_spec(run, wandb_config_keys, config_fields)
        full_run_specs.append(full_run_spec)
        model_hashes.append(model_hash)
    depends_on_model_hashes = {}
    for k, v in depends_on.items():
        model_hash = model_hashes[k]
        dependencies = [model_hashes[i] for i in v]
        depends_on_model_hashes[model_hash] = dependencies
    return full_run_specs,model_hashes,depends_on_model_hashes

def update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on):
    temp_full_run_specs, temp_model_hashes, temp_depends_on_model_hashes = preprocess_runs(
        temp_runs, wandb_config_keys, config_fields, depends_on)
    full_run_specs.extend(temp_full_run_specs)
    model_hashes.extend(temp_model_hashes)
    # merge the temp_depends_on_model_hashes into the main depends_on_model_hashes
    for k, v in temp_depends_on_model_hashes.items():
        if k not in depends_on_model_hashes:
            depends_on_model_hashes[k] = v
        else:
            # merge the lists
            depends_on_model_hashes[k].extend(v)

def main(batch_run_config, wandb_config, train_config):
    logging.basicConfig(level=logging.INFO)

    # create log_dir, if specified
    # default to current working directory
    log_dir = os.getcwd()
    if train_config.log_dir is not None:
        # if group specified, append to log_dir
        if wandb_config.group is not None:
            log_dir = os.path.join(train_config.log_dir, wandb_config.group)
        os.makedirs(log_dir, exist_ok=True)

    # set batch_run_config.slurm_name to group if not specified
    if batch_run_config.slurm_name is None and wandb_config.group is not None:
        batch_run_config.slurm_name = wandb_config.group

    # download the runs from the wandb group so we can avoid repeating runs
    if wandb_config.group is not None:
        api = wandb.Api()
        runs = api.runs(
            f'{wandb_config.entity}/{wandb_config.project}',
            filters={
                "group": wandb_config.group,
                "state": "finished",
            })
        logging.info(f'Found {len(runs)} runs in group {wandb_config.group}.')
    else:
        runs = []
    run_df = pd.DataFrame([run.config for run in runs])
    # fill in run_df with any missing columns with the default values
    # this is handy when new columns are added to the TrainConfig or LIFParameter
    default_train_config = TrainConfig()
    for key, value in {
            **default_train_config.__dict__,
    }.items():
        if key not in run_df.columns:
            if isinstance(value, list):
                run_df[key] = [value] * len(run_df)
            else:
                run_df[key] = value
    # if a column has NaN values, fill them in with the default value
    for key, value in {
            **default_train_config.__dict__,
    }.items():
        if run_df[key].isnull().any():
            if value is not None:
                run_df[key] = run_df[key].fillna(value=value)
    
    # wandb_config keys
    wandb_config_keys = [f.name for f in fields(WandbConfig)]
    # get field names for TrainConfig
    config_fields = [f.name for f in fields(TrainConfig)]

    full_run_specs, model_hashes, depends_on_model_hashes = [], [], {}
    if wandb_config.group in ['mnist_mixture']:
        temp_runs, depends_on = mnist()
        update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on)
    elif wandb_config.group in ['support2_mixture', 'support2_c_ipcw']:
        temp_runs, depends_on = support2()
        update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on)
    elif wandb_config.group == 'sepsis_mixture':
        temp_runs, depends_on = sepsis()
        update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on)

    if wandb_config.group == 'mnist_expert_sensitivity':
        temp_runs, depends_on = mnist_expert_sensitivity()
        update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on)

    if wandb_config.group == 'sepsis_expert_sensitivity':
        temp_runs, depends_on = sepsis_expert_sensitivity()
        update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on)

    if wandb_config.group in ['support2_expert_sensitivity']:
        temp_runs, depends_on = support2_expert_sensitivity()
        update_lists_dicts(wandb_config_keys, config_fields, full_run_specs, model_hashes, depends_on_model_hashes, temp_runs, depends_on)

    # sample a subset of runs
    if batch_run_config.num_runs > 0:
        run_range = np.arange(len(runs))
        run_idxs = np.random.choice(run_range,
                                    size=batch_run_config.num_runs,
                                    replace=False)
        full_run_specs = [full_run_specs[i] for i in run_idxs]
        excluded_run_idxs = set(run_range) - set(run_idxs)
        # loop through the depends_on dict and filter it to only include the runs that are in the sampled runs
        for idx in excluded_run_idxs:
            model_hash = model_hashes[idx]
            # remove the model_hash from the depends_on_model_hashes dict
            remove_model_hash_from_depends_on_dict(depends_on_model_hashes, model_hash)

    commands = []
    for idx, full_run_spec in enumerate(full_run_specs):
        model_hash = full_run_spec['model_hash']
        run_name = model_hash

        # check if run is in the dataframe, which is a combination of train_config and lif_params
        if len(run_df) > 0:
            already_ran = is_existing_run(run_df, full_run_spec, ignore_keys=['model_hash'])
            if already_ran:
                logging.info(f'Run already exists in dataframe: {full_run_spec}')
                # remove the model_hash from the depends_on dict
                remove_model_hash_from_depends_on_dict(depends_on_model_hashes, model_hash)
                continue

        command = [
            'python',
            '-m',
            'survkit.train',
            '--name',
            f'{run_name}',
            '--group',
            f'{wandb_config.group}',
        ]

        # create the final argument list
        args = []
        for k, v in full_run_spec.items():
            # unpack lists
            if isinstance(v, list):
                args.extend([f'--{k}'] + [str(val) for val in v])
            else:
                args.extend([f'--{k}', f'{v}'])
        command.extend(args)

        log_file_path = run_name + '.log'
        if train_config.log_dir is not None:
            log_file_path = os.path.join(log_dir, log_file_path)
        commands.append({
            'command': command,
            'log_file_path': log_file_path,
            'name': run_name
        })

    # topologically sort the commands based on their dependencies
    command_order = list(graphlib.TopologicalSorter(depends_on_model_hashes).static_order())
    # reorder the commands based on the topological sort
    if command_order:
        commands = sorted(commands, key=lambda x: command_order.index(x['name']))
    launch(
        commands=commands,
        batch_run_config=batch_run_config,
        depends_on_model_hashes=depends_on_model_hashes,
    )

if __name__ == '__main__':
    parser = ArgParser([BatchRunConfig, WandbConfig, TrainConfig])
    batch_run_config, wandb_config, train_config, unknown_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    main(batch_run_config, wandb_config, train_config)
