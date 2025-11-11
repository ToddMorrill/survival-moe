from collections import defaultdict
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import wandb

from survkit.configs.train import TrainConfig, Models
from survkit.configs.wandb import WandbConfig
from survkit.data import load_data, mnist_survival_data
from survkit.train import Trainer, get_model
from survkit.utils import get_args, get_colors


def get_runs_df(entity_project,
                group,
                states=['finished'],
                per_page=1000,
                use_cache=True):
    api = wandb.Api()
    runs = api.runs(entity_project,
                    filters={
                        "group": group,
                        "state": {
                            "$in": states
                        }
                    },
                    per_page=per_page)
    run_dicts = []
    for run in tqdm(runs):
        run_dict = {}
        run_dict['name'] = run.name
        run_dict['state'] = run.state
        run_dict['id'] = run.id
        try:
            command = run.metadata.get('command', [''])
        except:
            print(f"Error fetching command for run {run.name}")
            command = ['']
        run_dict['command'] = ' '.join(command)
        run_dict.update(run.summary)
        run_dict.update(run.config)
        run_dicts.append(run_dict)
    df = pd.DataFrame(run_dicts)
    return df


def load_model(model_dir):
    checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
    checkpoint.keys()
    train_config = TrainConfig(**checkpoint['train_config'])
    # load the dataset
    feature_names, embed_map, embed_col_indexes, num_time_bins, train_loader, val_loader, test_loader, metadata = load_data(
        train_config)
    # get the time bins of the model
    time_bins = train_loader.dataset.dataset.time_bins if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.time_bins
    # in some cases, we may have extra feature dimensions that are not accounted for in the feature names
    # sample a point from the training set to determine if there are extra dimensions
    sample = train_loader.dataset.dataset[0] if hasattr(
        train_loader.dataset, 'dataset') else train_loader.dataset[0]
    x = sample[0]
    extra_dims = 0
    if feature_names is not None:
        extra_dims = len(x) - len(
            feature_names)  # always greater than or equal to 0
    model = get_model(train_config,
                      metadata,
                      feature_names,
                      embed_map,
                      embed_col_indexes,
                      num_time_bins,
                      extra_dims=extra_dims)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return_dict = {
        'model': model,
        'train_config': train_config,
        'feature_names': feature_names,
        'embed_map': embed_map,
        'embed_col_indexes': embed_col_indexes,
        'time_bins': time_bins,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata
    }
    return return_dict

def get_model_outputs(model_dict, loader='test'):
    train_loader = model_dict['train_loader']
    val_loader = model_dict['val_loader']
    test_loader = model_dict['test_loader']
    if loader == 'train':
        loader = train_loader
    elif loader == 'val':
        loader = val_loader
    elif loader == 'test':
        loader = test_loader
    else:
        raise ValueError("Loader must be one of 'train', 'val', or 'test'.")
    train_config = model_dict['train_config']
    model = model_dict['model'].to('cuda')
    trainer = Trainer(train_config, model, optimizer=None, time_bins=model_dict['time_bins'])
    cdfs = []
    pmfs = []
    losses = []
    model_outs = []
    ys = []
    topk_expert_scores = []
    expert_scores = []
    expert_output = []
    memory_output = []
    topk_mem_scores = []
    final_logits = []
    t_raws = []
    t_seqs = []
    ts = []
    cs = []
    hidden_states = []
    log_expert_pmfs = []
    for batch in tqdm(loader):
        x, y, c, t, t_seq, t_raw, t_raw_seq, g, index = batch.get('x'), batch.get('y'), batch.get('c'), batch.get('t'), batch.get('t_seq'), batch.get('t_raw'), batch.get('t_raw_seq'), batch.get('g'), batch.get('index')
        with torch.no_grad():
            model_output = model(x.to('cuda'))
        topk_expert_scores_batch = model_output.get('topk_expert_scores', torch.tensor([]))
        expert_scores_batch = model_output.get('expert_scores', torch.tensor([]))
        expert_output_batch = model_output.get('expert_output', torch.tensor([]))
        model_out_batch = model_output.get('model_out', torch.tensor([]))
        final_logits_batch = model_output.get('final_logits', torch.tensor([]))
        log_expert_pmfs_batch = model_output.get('log_expert_pmfs', torch.tensor([]))
        
        topk_expert_scores.append(topk_expert_scores_batch.detach())
        expert_scores.append(expert_scores_batch.detach())
        expert_output.append(expert_output_batch.detach())
        model_outs.append(model_out_batch.detach())
        final_logits.append(final_logits_batch.detach())
        log_expert_pmfs.append(log_expert_pmfs_batch.detach())

        with torch.no_grad():
            cdf = Trainer.discrete_survival_cdf(y_pred=final_logits_batch)
            pmf = Trainer.discrete_survival_pmf(y_pred=final_logits_batch)
            # compute the loss
            loss, _, aux_stats = trainer.loss_function(
                    final_logits_batch,
                    t_seq.to('cuda'),
                    c.to('cuda'),)
            # blow out individual losses from aux_stats
            losses_batch = trainer.recover_loss_order(c.to('cuda'), aux_stats)

        cdfs.append(cdf.detach())
        pmfs.append(pmf.detach())
        losses.append(losses_batch.detach())
        ts.append(t.detach())
        t_seqs.append(t_seq.detach())
        cs.append(c.detach())

        if y is not None:
            ys.append(y.detach())
        if t_raw is not None:
            t_raws.append(t_raw.detach())

    # concatenate all the results
    cdfs = torch.concatenate(cdfs, dim=0)
    pmfs = torch.concatenate(pmfs, dim=0)
    losses = torch.concatenate(losses, dim=0)
    model_outs = torch.concatenate(model_outs, dim=0)
    ts = torch.concatenate(ts, dim=0)
    t_seqs = torch.concatenate(t_seqs, dim=0)
    cs = torch.concatenate(cs, dim=0)
    if ys:
        ys = torch.concatenate(ys, dim=0)
    if t_raws:
        t_raws = torch.concatenate(t_raws, dim=0)
    if log_expert_pmfs:
        log_expert_pmfs = torch.concatenate(log_expert_pmfs, dim=0).detach()

    topk_expert_scores = torch.concatenate(topk_expert_scores, dim=0).detach()
    expert_scores = torch.concatenate(expert_scores, dim=0).detach()
    expert_output = torch.concatenate(expert_output, dim=0).detach()
    final_logits = torch.concatenate(final_logits, dim=0).detach()

    return_dict = {
        'cdfs': cdfs,
        'pmfs': pmfs,
        'losses': losses,
        'y': ys,
        't_raw': t_raws,
        't': ts,
        't_seq': t_seqs,
        'c': cs,
        'topk_expert_scores': topk_expert_scores,
        'expert_scores': expert_scores,
        'expert_output': expert_output,
        'model_outs': model_outs,
        'final_logits': final_logits,
        'log_expert_pmfs': log_expert_pmfs,
    }
    return return_dict

def plot_synth_distribution(args_file, class_idxs2colors, ax=None, first=False, last=False):
    """
    Plots survival distributions. Behaves differently based on whether 'ax' is provided.

    - If ax is None (standalone mode): creates a new figure, plots data, adds all
      labels, titles, a legend, and saves/shows the figure.
    - If ax is not None (component mode): Plots data on the given ax. Only adds
      x/y labels if 'first' or 'last' flags are set. Does not create a legend
      or show the plot.
    """
    ax_was_passed = ax is not None
    if not ax_was_passed:
        # standalone mode: create our own figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        # component mode: get the figure from the provided axes
        fig = ax.get_figure()

    # data loading and core plotting (same for both modes)
    command_string = f'--experiment_args ../args/{args_file} --mode disabled'
    wandb_config, train_config = get_args(command_string)
    print(f'MNIST means: {train_config.mnist_means}')
    print(f'MNIST stds: {train_config.mnist_stds}')
    print(f'MNIST sample pcts: {train_config.mnist_sample_pct}')
    train_loader, val_loader, test_loader, metadata = mnist_survival_data(
        train_config)
    # retrieve 10,000 samples from the train_loader
    num_batches = 10_000 // train_loader.batch_size
    images, targets, censored_flags, event_times, event_times_label_seqs, raw_event_times, raw_event_times_label_seqs = [], [], [], [], [], [], []
    for i, batch in enumerate(train_loader):
        x, y, c, t, t_seq, t_raw, t_raw_seq, g = batch
        images.append(x)
        targets.append(y)
        censored_flags.append(c)
        event_times.append(t)
        event_times_label_seqs.append(t_seq)
        raw_event_times.append(t_raw)
        raw_event_times_label_seqs.append(t_raw_seq)
        if i == num_batches - 1:
            break
    # stack the samples
    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0)
    censored_flags = torch.cat(censored_flags, dim=0)
    event_times = torch.cat(event_times, dim=0)
    event_times_label_seqs = torch.cat(event_times_label_seqs, dim=0)
    raw_event_times = torch.cat(raw_event_times, dim=0)
    raw_event_times_label_seqs = torch.cat(raw_event_times_label_seqs, dim=0)

    classes = [i for i in range(metadata.num_classes)]
    means = train_loader.dataset.dataset.means
    for c in classes:
        class_mask = targets == c
        class_samples = raw_event_times[class_mask]
        sns.kdeplot(class_samples,
                    ax=ax,
                    label=f"Class {c}",
                    color=get_colors()[class_idxs2colors[c]])
        # scale the y_data
        line = ax.lines[-1]
        x_data, y_data_density = line.get_data()
        n_class_samples = len(class_samples)
        scaled_y_data = y_data_density * n_class_samples
        line.set_ydata(scaled_y_data)

    ax.set_xlim(-0.5, max(means) + 2)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)


    # finalize
    if ax_was_passed:
        # component model: only set labels based on flags
        if first:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
        if last:
            ax.set_xlabel("Time")
        # no title, no legend, no saving/showing
    else:
        # standalone model: create a complete, self-contained plot
        ax.set_title(f"Survival Densities for {os.path.basename(args_file)}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        ax.legend(title="Class") # use ax.legend() for a single plot
        args_setting = os.path.splitext(os.path.basename(args_file))[0]
        # save and show the complete figure
        fig.savefig(f"../analysis/survival_densities_{args_setting}.png", bbox_inches='tight')
        fig.savefig(f"../analysis/survival_densities_{args_setting}.svg", bbox_inches='tight')
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # example usage
    wandb_config = WandbConfig()
    entity_project = f'{wandb_config.entity}/{wandb_config.project}'
    group = 'support2'
    df = get_runs_df(entity_project, group)
