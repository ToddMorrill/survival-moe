"""Defines all data related classes and functions.

Examples:
    $ python -m survkit.data
"""
from collections import defaultdict
from dataclasses import dataclass
import os
import pickle
import random

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.distributions import LogNormal
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict

# warnings.filterwarnings("ignore", message="Bins whose width are too small.*")


@dataclass
class DatasetMetadata:
    """Metadata for the dataset."""
    name: str
    num_classes: int
    input_dim: int


def log_normal_samples(n_samples, mean, std):
    """Samples from a log-normal distribution."""
    mean = torch.tensor(float(mean))
    std = torch.tensor(float(std))
    mu = torch.log(mean**2 / torch.sqrt(std**2 + mean**2))
    sigma = torch.sqrt(torch.log(1 + (std**2 / mean**2)))
    return LogNormal(mu, sigma).sample((n_samples, ))

def log_normal_pdf(ts, mean, std):
    """Evaluates the log-normal PDF at the given time points using scipy."""
    # Ensure inputs are numpy arrays for calculations
    ts = np.asarray(ts)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    pdf_values = lognorm.pdf(ts, s=sigma, scale=np.exp(mu), loc=0)
    return pdf_values

def log_normal_cdf(ts, mean, std):
    """Evaluates the log-normal CDF at the given time points using scipy."""
    # Ensure inputs are numpy arrays for calculations
    ts = np.asarray(ts)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    cdf_values = lognorm.cdf(ts, s=sigma, scale=np.exp(mu), loc=0)
    return cdf_values

def generate_label_sequence(raw_event_times, time_bins, floor=False):
    """Finds the nearest time bin for each event time. If floor is True, the
    event time is mapped to the nearest time bin at or before the event
    time. This is the more conservative approach because it flags events
    early. If floor is False, the event time is mapped to the nearest time
    bin at or after the event time. This makes more sense for censored times
    because we know that as of the earlier time bin, the event had not yet
    occurred.
    """
    if floor:
        event_time_bins = torch.bucketize(raw_event_times, time_bins, right=True)
        # minus 1 because torch.bucketize returns 1-indexed bins
        event_time_bins = event_time_bins - 1
        # bump any negative indices to 0
        event_time_bins = torch.clamp(event_time_bins, min=0)
        # look up the actual times
        event_times_nearest_bin = time_bins[event_time_bins]
        # define label as sequences of 0s up until the event time and then event time onwards should be 1
        label_seqs = event_times_nearest_bin[:, None] <= time_bins[None, :]
    else:
        event_time_bins = torch.bucketize(raw_event_times,
                                          time_bins,
                                          right=False)
        # if floor is False, we can get out of bounds indices, so keep track of them
        # and replace them with inf
        out_of_bounds = event_time_bins == time_bins.shape[0]
        event_time_bins = torch.where(out_of_bounds, time_bins.shape[0] - 1,
                                      event_time_bins)
        event_times_nearest_bin = time_bins[event_time_bins]
        # replace out of bounds with inf
        event_times_nearest_bin = torch.where(out_of_bounds, float('inf'),
                                              event_times_nearest_bin)
        # define label as sequences of 0s up until the event time and then event time onwards should be 1
        label_seqs = event_times_nearest_bin[:, None] <= time_bins[None, :]

    event_times_label_seqs = label_seqs.long()
    return event_times_label_seqs


class MNISTSurvival(torchvision.datasets.MNIST):
    """MNIST survival analysis dataset where event times are sampled from a log-normal distribution per class."""

    def __init__(self,
                 config,
                 time_bins=None,
                 means=None,
                 stds=None,
                 split='train',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        # presample the targets for each class
        self.raw_event_times = torch.zeros_like(self.targets,
                                                dtype=torch.float)
        # if means and stds are not provided, use default values
        if means is None:
            # +1 so that the first class has a mean of 1, not 0
            means = [i + 1 for i in range(10)]
        if stds is None:
            stds = [0.5 for _ in range(10)]
        self.means = torch.tensor(means) if isinstance(means, list) else means
        self.stds = torch.tensor(stds) if isinstance(stds, list) else stds
        # if time_bins is not provided, use the number of time bins from the config
        # and set the min and max according to the means
        if time_bins is None:
            time_bin_min = 0.0 # torch.tensor(means).min()
            time_bin_max = torch.tensor(means).max()
            # when floor=false, really want the max time bin to be larger than the max event time
            # because event times after this max time really mean no event, namely, (0, 0, ..., 0)
            # so really pad out the last time bin to be larger than the max event time
            if not self.config.labeling_floor_mode:
                time_bin_max = time_bin_max + 3 * torch.tensor(stds).max()
            time_bins = torch.linspace(time_bin_min, time_bin_max,
                                       config.time_bins)
        self.time_bins = time_bins
        # add a final bin edge to account for the never happens bin
        self.time_bins = torch.cat((self.time_bins,
                                    torch.tensor([self.time_bins[-1] + 1],
                                                 dtype=torch.float32)))

        for i in range(10):
            mask = self.targets == i
            self.raw_event_times[mask] = log_normal_samples(
                mask.sum(), means[i], stds[i])

        # determine the event times to censor
        n_censored = int(config.mnist_censoring_percentage *
                         len(self.raw_event_times))
        self.censored_events = torch.randperm(len(
            self.raw_event_times))[:n_censored]
        # identify approximate minimum times per class as mean - 3*std
        min_times = torch.zeros_like(self.means)
        # min_times = torch.maximum(self.means - 3 * self.stds, torch.tensor(0))
        # sample censored times from a uniform distribution between min_times and the raw_event_times
        offset = min_times[self.targets[self.censored_events]]
        intervals = self.raw_event_times[self.censored_events] - offset
        censoring_times = torch.rand(n_censored) * intervals + offset
        # copy over raw_event_times to event_times so we have a new tensor that can be safely modified
        self.event_times = torch.clone(self.raw_event_times)
        self.event_times[self.censored_events] = censoring_times

        # convert self.censored_events to a boolean tensor
        censored_events = torch.zeros_like(self.targets, dtype=torch.bool)
        censored_events[self.censored_events] = True
        self.censored_events = censored_events

        # possibly quantize the event times according to the number of time_bins
        if self.config.quantize_event_times:
            # since this is synthetic data store keep track of both actual event times and censored event times for evaluation
            self.raw_event_times_label_seqs = generate_label_sequence(
                self.raw_event_times,
                self.time_bins,
                floor=self.config.labeling_floor_mode)
            self.event_times_label_seqs = generate_label_sequence(
                self.event_times,
                self.time_bins,
                floor=self.config.labeling_floor_mode)

        self.groups = None


    def __getitem__(self, index):
        # NB: you can still provide transforms and target_transforms and they will be applied
        image, target = super().__getitem__(index)
        censored = self.censored_events[index]
        event_time = self.event_times[index]
        event_time_label_seq = self.event_times_label_seqs[index]
        # since this is synthetic data, we can return the raw event time for evaluation
        raw_event_time = self.raw_event_times[index]
        raw_event_time_label_seq = self.raw_event_times_label_seqs[index]
        # if the group policy is not None, return the group index
        group_info = None
        if self.groups is not None:
            group_info = self.groups[index]
        return image, target, censored, event_time, event_time_label_seq, raw_event_time, raw_event_time_label_seq, group_info, index


class SubclassSampler(Sampler):
    def __init__(self, dataset, sample_pct, shuffle=True):
        """
        Args:
            dataset: The dataset to sample from. Must have a 'targets' attribute.
            sample_pct (dict): A dictionary where keys are class labels and
                values are the number of samples to draw from that class.
            shuffle (bool): If True, shuffle the iterator over the subsampled
                indices at each epoch.
        """
        self.dataset = dataset
        self.samples_pct = sample_pct
        self.shuffle = shuffle
        self.use_last_iter = False

        self.class_indices = defaultdict(list)
        try:
            # for datasets like torchvision.datasets.MNIST
            for idx, target in enumerate(self.dataset.targets):
                if isinstance(target, torch.Tensor):
                    target = target.item()
                self.class_indices[target].append(idx)
            subset2full = torch.arange(len(self.dataset), dtype=torch.int64)
            # invert the map to go from full2subset
            self.full2subset = subset2full # no inverse to be done
        except AttributeError:
            # for datasets where targets might be stored differently or need to be loaded
            # this might be slower if the dataset is large and not in memory
            print(
                "Warning: 'targets' attribute not found. Iterating through dataset to get targets. This might be slow."
            )
            for idx in range(len(dataset)):
                returned = dataset[idx]  # Assuming dataset returns (data, target, ...)
                target = returned[1]
                if isinstance(target, torch.Tensor):
                    target = target.item()
                self.class_indices[target].append(idx)
            subset2full = dataset.indices # assumes that this is a Subset dataset
            # invert the map to go from full2subset
            self.full2subset = torch.zeros(len(self.dataset.dataset), dtype=torch.int64)
            for i, idx in enumerate(subset2full):
                self.full2subset[idx] = i

        # subsample class indices based on the sample_pct once
        self.subsampled_indices = {}
        self.subset2class = {}
        for class_label, indices in self.class_indices.items():
            self.subsampled_indices[class_label] = np.random.choice(
                indices,
                size=int(len(indices) * sample_pct.get(class_label, 1.0)),
                replace=False)
            for idx in self.subsampled_indices[class_label]:
                self.subset2class[idx] = class_label
        # add a full2class mapping for debugging
        self.full2class = torch.zeros(len(self.dataset.dataset), dtype=torch.int64)
        for idx, class_label in self.subset2class.items():
            full_idx = subset2full[idx]
            self.full2class[full_idx] = class_label
        self.num_samples = sum(len(indices) for indices in self.subsampled_indices.values())

    def set_use_last_iter(self, use_last_iter):
        """Sets whether to use the last iterator or create a new one."""
        self.use_last_iter = use_last_iter
            
    def __iter__(self):
        """Returns an iterator over the subsampled indices."""
        if self.use_last_iter:
            # if using the last iterator, just return the last iterator
            self.use_last_iter = False
            return iter(self.last_iter)
        
        # otherwise, just return the subsampled indices
        indices = []
        for class_label, sampled_indices in self.subsampled_indices.items():
            indices.extend(sampled_indices)

        if self.shuffle:
            random.shuffle(indices)
        self.last_iter = indices
        self.use_last_iter = False
        return iter(indices)

    def __len__(self):
        return self.num_samples

def mnist_survival_collate_fn(batch):
    images, targets, censored_flags, event_times, event_times_label_seqs, raw_event_times, raw_event_times_label_seqs, groups, index = zip(
        *batch)
    # we may or may not have group info
    if groups[0] is None:
        groups = None
    else:
        groups = torch.stack(groups)
    return_dict = {'x': torch.stack(images), 'y': torch.tensor(targets), 'c': torch.stack(
        censored_flags), 't': torch.stack(event_times), 't_seq': torch.stack(
            event_times_label_seqs), 't_raw': torch.stack(raw_event_times), 't_raw_seq': torch.stack(
                raw_event_times_label_seqs), 'g': groups, 'index': torch.tensor(index)}
    return return_dict

def get_dataloaders(train_dataset, val_dataset, test_dataset, train_config, collate_fn, train_sampler=None, shuffle_train=True):
    """Returns the dataloaders for the datasets."""
    generator = torch.Generator().manual_seed(42) # train_config.seed
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=shuffle_train if train_sampler is None else False,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator,
        sampler=train_sampler if train_sampler is not None else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def mnist_survival_data(train_config, shuffle_train=True, metadata_only=False):
    """Returns the MNIST survival dataset."""
    metadata = DatasetMetadata(name='MNIST', num_classes=10, input_dim=28 * 28)
    if metadata_only:
        return metadata

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    if len(train_config.mnist_means) == 1:
        means = [1 + train_config.mnist_means[0] * i for i in range(10)]
    else:
        means = train_config.mnist_means
    if len(train_config.mnist_stds) == 1:
        stds = [train_config.mnist_stds[0]] * 10
    else:
        stds = train_config.mnist_stds
    train_dataset = MNISTSurvival(config=train_config,
                                  means=means,
                                  stds=stds,
                                  root=train_config.data_dir,
                                  train=True,
                                  download=True,
                                  transform=transform)
    val_dataset = MNISTSurvival(config=train_config,
                                means=means,
                                stds=stds,
                                root=train_config.data_dir,
                                train=True,
                                download=True,
                                transform=transform)
    test_dataset = MNISTSurvival(config=train_config,
                                 means=means,
                                 stds=stds,
                                 root=train_config.data_dir,
                                 train=False,
                                 download=True,
                                 transform=transform)

    # subset the train and val datasets
    random_perm = torch.randperm(len(train_dataset))
    train_indices = random_perm[:55_000]
    # val set with 5,000 images
    val_indices = random_perm[55_000:]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_sampler = None
    if train_config.mnist_sample_pct is not None:
        sample_pct = {}
        for idx, pct in enumerate(train_config.mnist_sample_pct):
            sample_pct[idx] = pct
        train_sampler = SubclassSampler(train_dataset, sample_pct=sample_pct, shuffle=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, train_config, mnist_survival_collate_fn, train_sampler=train_sampler, shuffle_train=shuffle_train)
    return train_loader, val_loader, test_loader, metadata

def unpack_uci_dataset(support2):
    """They use this little annoying dotdict thing, which prevents you from
    easily accessing the objections in the dictionary when you try to pickle."""
    support2_dict = {}
    support2_dict['data'] = dict(support2.data)
    support2_dict['metadata'] = dict(support2.metadata)
    support2_dict['variables'] = support2.variables # just a df
    # special support for the metadata field, which also has some dotdicts on it
    if 'additional_info' in support2.metadata:
        support2_dict['metadata']['additional_info'] = dict(
            support2.metadata.additional_info)
    if 'intro_paper' in support2.metadata:
        support2_dict['metadata']['intro_paper'] = dict(
            support2.metadata.intro_paper)
    return support2_dict

def repack_uci_dataset(support2_dict):
    """They use this little annoying dotdict thing, which prevents you from
    easily accessing the objects in the dictionary when you try to pickle."""
    support2 = {'data': dotdict(support2_dict['data']),
        'metadata': dotdict(support2_dict['metadata']),
        'variables': support2_dict['variables']
    }
    # add dotdict to metadata.additional_info and metadata.intro_paper, if available
    if 'additional_info' in support2_dict['metadata']:
        support2['metadata']['additional_info'] = dotdict(support2_dict['metadata']['additional_info'])
    if 'intro_paper' in support2_dict['metadata']:
        support2['metadata']['intro_paper'] = dotdict(support2_dict['metadata']['intro_paper'])
    return dotdict(support2)

def make_numeric_impute_pipeline(fill_strategy="median", fill_value=None):
    """
    Returns a Pipeline that:
    1) Imputes a single numeric column using either a constant fill value or a strategy (mean, median, etc.),
        with add_indicator=True so that the output is shape (n,2).
    2) Scales the first sub-column (the imputed numeric) but leaves the missingness indicator (the second sub-column) unscaled.
    """
    # Step 1: create the SimpleImputer
    if fill_strategy == "constant":
        imputer = SimpleImputer(
            strategy="constant",
            fill_value=fill_value,
        )
        # add_indicator=True) cant't use this because you may get 1 column or you may get 2 columns
        # depending on the presence of missing values, which breaks the next step
        # solution: explicitly use MissingIndicator
    else:
        # e.g., "mean", "median"
        imputer = SimpleImputer(strategy=fill_strategy, )
        # add_indicator=True)

    # Step 2: after imputation, apply StandardScaler to the numeric column.
    # and MissingIndicator to flag the missingness of the original column
    union = FeatureUnion([("imputed", Pipeline([("imputer", imputer), ("scaler", StandardScaler())])),
                          ("indicator", MissingIndicator(features="all"))])

    return union

def build_categorical_categories(df, cat_cols, missing_token="missing"):
    """
    Given a dataframe and a list of categorical columns, produce a dictionary mapping each column name to
    a list of categories for scikit-learn's OrdinalEncoder, ensuring:
    missing_token -> index 0
    real categories -> index >=1
    """
    categories_for_encoder = {}

    for col in cat_cols:
        # convert real NaNs to the 'missing' label
        df[col] = df[col].fillna(missing_token)

        # collect all unique categories
        unique_cats = set(df[col].unique())

        # ensure missing_token is not in the set so we can reorder easily
        unique_cats.discard(missing_token)

        # sort the "real" categories to ensure consistent ordering
        main_cats = sorted(list(unique_cats))

        # construct the final list for that column
        final_cat_list = [missing_token] + main_cats
        categories_for_encoder[col] = final_cat_list

    return categories_for_encoder

def convert_cat_vars_to_str(df, cat_vars, missing_token='missing'):
    # categorical variables need to be converted to a string for downstream processing
    for col in cat_vars:
        notna_mask = df[col].notna()
        # convert the column type first so that pandas doesn't complain
        df[col] = df[col].astype(str)
        # now fill in the missing values with the string "missing"
        df.loc[~notna_mask, col] = missing_token
    return df

def replace_unknown_categories(df,
                                categories_for_encoder,
                                replacement="missing"):
    """
    Given a dataframe and a list of categorical columns, replace any unknown categories with the "missing" category
    """
    # NB: we need to ensure that the test and val datasets have the same categories as the training set
    # if any categories are new, replace them with the "missing" category
    for col in categories_for_encoder.keys():
        # convert real NaNs to the 'missing' label
        df[col] = df[col].fillna(replacement)

        # get the known categories for this column
        known_cats = set(categories_for_encoder[col])

        # replace any unknown categories with the "missing" category
        df[col] = df[col].apply(lambda x: x
                                if x in known_cats else replacement)

    return df

class SUPPORT2(Dataset):
    """SUPPORT2 dataset."""

    def __init__(self, config, split='train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.split = split

        support2 = self.get_support2_data(config.data_dir)

        # data (as pandas dataframes)
        # X = support2.data.features
        # y = support2.data.targets

        # round up all continuous features and categorical features
        vars_df = support2.variables
        feature_mask = vars_df['role'] == 'Feature'
        cont_mask = vars_df['type'] == 'Continuous'
        cat_mask = vars_df['type'] == 'Categorical'
        # for whatever reason glucose, bun, and urine are marked as Integer but they're really continuous
        cont_mask = cont_mask | (vars_df['name'].isin(
            ['glucose', 'bun', 'urine']))
        # hday (day in hospital when patient entered study) is an integer but let's convert it to a float and bucketize by quantiles
        cont_mask = cont_mask | (vars_df['name'] == 'hday')

        # list out variables by type
        cont_vars = vars_df.loc[feature_mask & cont_mask, 'name'].tolist()
        cat_vars = vars_df.loc[feature_mask & cat_mask, 'name'].tolist()
        # bucketization vars
        bucket_vars = ['hday']

        # columns that have recommended "constant" fill values (from Harrell)
        phys_fill_values = {
            "alb": 3.5,  # Serum albumin
            "pafi": 333.3,  # PaO2/FiO2 ratio
            "bili": 1.01,
            "crea": 1.01,
            "bun": 6.51,
            "wblc": 9.0,  # White blood cell count (thousands)
            "urine": 2502
        }
        phys_cols = list(phys_fill_values.keys())

        # get remain continuous cols, which don't have recommended fill values
        cont_vars = [
            col for col in cont_vars
            if col not in phys_cols and col not in bucket_vars
        ]

        # assert that the breakdown of vars collectively recovers all the features
        feature_cols = vars_df.loc[feature_mask, 'name'].tolist()
        assert set(feature_cols) == set(
            cont_vars + cat_vars + phys_cols +
            bucket_vars), "Feature columns do not match original feature list"

        # potentially revise the task to predict a shorter duration
        if config.support2_start_day is not None:
            # filter the dataset to only include patients who survived at least support2_start_day days
            support2.data.original = support2.data.original[
                support2.data.original['d.time'] >= config.support2_start_day]
            # subtract support2_start_day from all d.time values
            support2.data.original['d.time'] = support2.data.original[
                'd.time'] - config.support2_start_day
        
        # potentiall administratively censor at a certain day
        if config.support2_censor_day is not None:
            # censor any event times greater than support2_censor_day
            support2.data.original['death'] = np.where(
                support2.data.original['d.time'] > config.support2_censor_day,
                0, support2.data.original['death'])
            support2.data.original['d.time'] = np.where(
                support2.data.original['d.time'] > config.support2_censor_day,
                config.support2_censor_day, support2.data.original['d.time'])

        # preprocess a few columns across train/val/test splits
        support2.data.original['hday'] = support2.data.original['hday'].astype(
            'float')
        # categorical variables need to be converted to a string for downstream processing
        support2.data.original = convert_cat_vars_to_str(support2.data.original, cat_vars, missing_token='missing')

        # work with original dataset, which has d.time in it, and will be needed for the survival analysis
        test_pct = 0.1
        val_pct = 0.1
        X_train, X_test = train_test_split(support2.data.original,
                                           test_size=test_pct,
                                           random_state=42)
        # want val_pct to be 10% of the original dataset, not 10% of the training set
        adj_val_pct = val_pct / (1 - test_pct)
        X_train, X_val = train_test_split(X_train,
                                          test_size=adj_val_pct,
                                          random_state=42)

        # transform individual columns
        numeric_transformers = []

        # 3.1) Physio columns with custom fill
        for col in phys_cols:
            fill_val = phys_fill_values[col]
            numeric_transformers.append((
                f"impute_{col}",
                make_numeric_impute_pipeline(fill_strategy="constant",
                                                  fill_value=fill_val),
                [col]  # this pipeline acts only on this one column
            ))

        # 3.2) Other numeric columns with median fill
        for col in cont_vars:
            numeric_transformers.append(
                (f"impute_{col}",
                 make_numeric_impute_pipeline(fill_strategy="median"),
                 [col]))

        # Combine into a single ColumnTransformer that processes each numeric column individually
        numeric_preprocessor = ColumnTransformer(
            transformers=numeric_transformers, remainder="drop")

        # handle categorical columns
        self.categories_for_encoder = build_categorical_categories(
            X_train, cat_vars)
        # create the pipeline for categorical columns that first imputes missing values
        # and then encodes the categorical values with OrdinalEncoder
        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="constant",
                                     fill_value="missing")),
            ("encode",
             OrdinalEncoder(categories=list(
                 self.categories_for_encoder.values()), ))
            # since there is no way to train the "unknown" category embedding (nothing is unknown in the training set)
            # we'll just map the unknown category to the missing category
            # but this is a little annoying since sklearn won't let me reuse category 0 for this purpose so make sure that
            # for test datasets, you confer the known categories and replace any unknowns with the missing category
            # before passing to the encoder
            #  handle_unknown="use_encoded_value",
            #  unknown_value=0))
        ])

        # quantize hday with 20 buckets
        # every 5th percentile should map to a bucket
        # Pipeline for the binned column
        binned_pipe = Pipeline([
            (
                "impute", SimpleImputer(strategy="median", )
            ),  # the downstream ColumnTransformer seems to fail when there are no missing values and add_indicator=True)),
            # column transformer to handle the imputed column with Kbins and the indicator as a passthrough
            # ("kbins", kbins_col_transformer)
            (
                "kbins",
                KBinsDiscretizer(n_bins=20,
                                 encode="ordinal",
                                 strategy="quantile"),
            )
        ])

        self.preprocessor = ColumnTransformer(
            [("num", numeric_preprocessor, cont_vars + phys_cols),
             ("cat", cat_pipeline, cat_vars),
             ("binned", binned_pipe, bucket_vars)],
            remainder="drop",
            verbose_feature_names_out=False)

        # fit the preprocessor on the training data
        self.preprocessor.fit(X_train)

        self.X_train = X_train
        self.X_val = replace_unknown_categories(
            X_val, self.categories_for_encoder)
        self.X_test = replace_unknown_categories(
            X_test, self.categories_for_encoder)

        # transform the datasets
        # TODO: really need this init function to live outside of the dataset class probably
        # OR, you could return the preprocessor from the training dataset and other essential state and pass them
        # to a downstream val/test dataset
        self.X_train_transformed = self.preprocessor.transform(X_train)
        self.X_val_transformed = self.preprocessor.transform(X_val)
        self.X_test_transformed = self.preprocessor.transform(X_test)

        # get the feature names from the preprocessor
        feature_names = self.preprocessor.get_feature_names_out()
        # map to column indexes
        self.feature_names = {name: i for i, name in enumerate(feature_names)}
        # retrieve the categorical and quantized columns, which will index into an embedding table
        embed_col_indexes = []
        for col in cat_vars + bucket_vars:
            embed_col_indexes.append(self.feature_names[col])
        self.embed_col_indexes = embed_col_indexes
        # create an embedding map for ALL variables that will index into an embedding table
        embed_map = {}
        for col in self.categories_for_encoder:
            embed_map[col] = self.categories_for_encoder[col]
        for i, col in enumerate(bucket_vars):
            embed_map[col] = self.preprocessor.named_transformers_[
                'binned'].named_steps['kbins'].bin_edges_[i]
        self.embed_map = embed_map

        if config.quantile_event_times:
            # define some time bins for the event times using the KBinsDiscretizer
            event_times_binner = KBinsDiscretizer(n_bins=config.time_bins, encode="ordinal", strategy="quantile")
            # fit the KBinsDiscretizer on the training data
            event_times_binner.fit(self.X_train[['d.time']])
            # get the bin edges
            self.time_bins = torch.tensor(event_times_binner.bin_edges_[0], dtype=torch.float32)
        else:
            # evenly spaced time bins between 0 and max event times in the training set
            min_time = 0.0
            max_time = self.X_train['d.time'].max()
            self.time_bins = torch.linspace(min_time, max_time, config.time_bins)
        # add a final bin edge to account for the never happens bin
        self.time_bins = torch.cat((self.time_bins, torch.tensor([self.time_bins[-1] + 1], dtype=torch.float32)))
        self.train_event_time_label_seqs = generate_label_sequence(
            torch.tensor(self.X_train['d.time'].values),
            self.time_bins,
            floor=self.config.labeling_floor_mode)
        self.val_event_time_label_seqs = generate_label_sequence(
            torch.tensor(self.X_val['d.time'].values),
            self.time_bins,
            floor=self.config.labeling_floor_mode)
        self.test_event_time_label_seqs = generate_label_sequence(
            torch.tensor(self.X_test['d.time'].values),
            self.time_bins,
            floor=self.config.labeling_floor_mode)

        self.groups = None

    @staticmethod
    def get_support2_data(data_dir):
        # check if we have the data downloaded
        cache_filepath = os.path.join(data_dir, 'support2.pickle')
        if os.path.exists(cache_filepath):
            with open(cache_filepath, 'rb') as handle:
                pickle_dict = pickle.load(handle)
                support2 = repack_uci_dataset(pickle_dict)
        else:
            # fetch the data
            support2 = fetch_ucirepo(id=880)
            # ensure the cache dir exists
            os.makedirs(data_dir, exist_ok=True)
            # cache it
            with open(cache_filepath, 'wb') as handle:
                pickle_dict = unpack_uci_dataset(support2)
                pickle.dump(pickle_dict, handle)
        return support2

    def get_label(self, df, index, event_time_label_seqs):
        # Label is a combination of `y['death']` where 0 means the patient was censored and 1 means the patient died (so need to flip this). The time of censoring or death is given by `support2.data.original['d.time']`
        censored = bool(1 - df.iloc[index]['death'])
        event_time = float(df.iloc[index]['d.time'])
        event_time_label_seq = event_time_label_seqs[index]
        return torch.tensor(censored), torch.tensor(event_time), event_time_label_seq

    def __getitem__(self, index):
        # TODO: fix this so each dataset is already split out
        if self.split == 'train':
            x = self.X_train_transformed[index]
            df = self.X_train
            event_time_label_seqs = self.train_event_time_label_seqs
        elif self.split == 'val':
            x = self.X_val_transformed[index]
            df = self.X_val
            event_time_label_seqs = self.val_event_time_label_seqs
        elif self.split == 'test':
            x = self.X_test_transformed[index]
            df = self.X_test
            event_time_label_seqs = self.test_event_time_label_seqs

        censored, event_time, event_time_label_seq = self.get_label(
                df, index, event_time_label_seqs)

        group = None
        if self.groups is not None:
            group = self.groups[index]

        return torch.tensor(x, dtype=torch.float32), censored, event_time, event_time_label_seq, group, index

    def __len__(self):
        if self.split == 'train':
            return len(self.X_train_transformed)
        elif self.split == 'val':
            return len(self.X_val_transformed)
        elif self.split == 'test':
            return len(self.X_test_transformed)

def support2_collate_fn(batch):
    # pattern match the MNIST dataset return signature
    target = None  # no notion of class label here
    raw_event_time = None  # don't have ground truth for ALL instances
    raw_event_time_label_seq = None  # don't have ground truth for ALL instances
    xs, censored_flags, event_times, event_times_label_seqs, groups, index = zip(*batch)
    if groups[0] is None:
        groups = None
    else:
        groups = torch.stack(groups)
    return_dict = {'x': torch.stack(xs), 'y': target, 'c': torch.stack(censored_flags),
                   't': torch.stack(event_times), 't_seq': torch.stack(event_times_label_seqs),
                   't_raw': raw_event_time, 't_raw_seq': raw_event_time_label_seq, 'g': groups, 'index': torch.tensor(index)}
    return return_dict

def support2_data(train_config, shuffle_train=True, metadata_only=False):
    """Returns the SUPPORT2 dataset."""
    # input dim isn't so simple
    metadata = DatasetMetadata(name='SUPPORT2', num_classes=None, input_dim=None)
    if metadata_only:
        return metadata

    train_dataset = SUPPORT2(config=train_config, split='train')
    val_dataset = SUPPORT2(config=train_config, split='val')
    test_dataset = SUPPORT2(config=train_config, split='test')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, train_config, support2_collate_fn, shuffle_train=shuffle_train)
    return train_loader, val_loader, test_loader, metadata


class Sepsis(Dataset):
    """PhysioNet 2019 Sepsis prediction dataset."""

    def __init__(self,
                 config,
                 split='train',
                 include_hospital_unit=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.set_a = os.path.join(
            config.data_dir,
            'physionet.org/files/challenge-2019/1.0.0/training/training_setA')
        self.set_b = os.path.join(
            config.data_dir,
            'physionet.org/files/challenge-2019/1.0.0/training/training_setB')

        # define the feature and label columns
        self.cont_vars = [
            'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
            'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
            'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
            'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
            'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
            'WBC', 'Fibrinogen', 'Platelets', 'Age', ]#'HospAdmTime', 'ICULOS'
        self.cat_vars = ['Gender']
        if include_hospital_unit:
            self.cat_vars += ['Unit1', 'Unit2']
        self.label_col = 'SepsisLabel'

        if self.config.cache_dataset:
            # try to load the dataset from the cache
            cache_filepath = os.path.join(
                config.data_dir,
                f'physionet.org/sepsis_hrs{self.config.num_hours}_1rec{self.config.single_record}.pickle'
            )
            preprocessor_filepath = os.path.join(
                config.data_dir,
                f'physionet.org/preprocessor_hrs{self.config.num_hours}_1rec{self.config.single_record}.pickle'
            )

            if os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as handle:
                    dataset = pickle.load(handle)
                self.data = dataset[
                    f'X_{split}']  # numpy array of shape (n_samples, n_features)
                self.labels = dataset[
                    f'y_{split}']  # numpy array of shape (n_samples, seq_len)
                self.metadata = dataset[
                    'metadata']  # contains variable list, categorical feature mappings, and patient IDs to row idxs
                with open(preprocessor_filepath, 'rb') as handle:
                    self.preprocessor = pickle.load(handle)
            else:
                # preprocess the dataset and save it to the cache
                data_metadata, preprocessor = self.preprocess_dataset()
                # save the dataset (all splits) to the cache
                with open(cache_filepath, 'wb') as handle:
                    pickle.dump(data_metadata, handle)

                # save the preprocessor to the cache
                with open(preprocessor_filepath, 'wb') as handle:
                    pickle.dump(preprocessor, handle)
                # load the dataset for the requested split
                self.data = data_metadata[f'X_{split}']
                self.labels = data_metadata[f'y_{split}']
                self.metadata = data_metadata[
                    'metadata']  # contains variable list, categorical feature mappings, and patient IDs to row idxs
                self.preprocessor = preprocessor
        else:
            # preprocess the dataset but don't save it
            data_metadata, preprocessor = self.preprocess_dataset()
            # load the dataset for the requested split
            self.data = data_metadata[f'X_{split}']
            self.labels = data_metadata[f'y_{split}']
            self.metadata = data_metadata['metadata']
            self.preprocessor = preprocessor

        # unpack a couple of the key attributes from metadata that are used by the trainer and model
        self.feature_names = self.metadata['feature_names']
        self.embed_map = self.metadata['embed_map']
        self.embed_col_indexes = self.metadata['embed_col_indexes']

        # if time_bins isn't specified, fall back to the maximum of 336
        self.max_time_bins = 336
        self.num_time_bins = self.config.time_bins
        if self.config.time_bins == -1:
            self.num_time_bins = self.max_time_bins

        # process the label sequences
        # it's possible to get sepsis multiple times (~1%)
        # sepsis is marked 6 hours in advance and can last for multiple hours
        # assume that if a patient doesn't get sepsis, then they never get it (no censoring)
        # sequences can last up to 2 weeks (336 hours)
        self.onset_times = []
        for label_seq in self.labels:
            if 1 in label_seq:
                onset_time = label_seq.argmax() + 6  # label is 6 hours before sepsis
            else:
                onset_time = self.max_time_bins
            self.onset_times.append(onset_time)

        # assume that once you get it, you always stay "on"
        # need to use FULL event sequence in the case of sequential modeling, because we'll be looking ahead
        full_time_bins = torch.arange(0, self.max_time_bins + 1) # + 1 accounts for the "never get it" case
        self.label_seqs = generate_label_sequence(
            torch.tensor(self.onset_times),
            full_time_bins,
            floor=self.config.labeling_floor_mode)

        # if we're NOT returning single records, then prepare the label sequence for each time step (shift left)
        # for instance if 1s start at index 2 and number of hours in the example is 3, then we have
        # [0, 0, 1]
        # [0, 1, 1]
        # [1, 1, 1]
        # in any case, truncate to the num_time_bins + 1
        if not self.config.single_record:
            # shift the label sequence left by 1 hour for each time step in the data instance
            self.label_seqs = [self.shift_sequence(s, len(self.data[idx])) for idx, s in enumerate(self.label_seqs)]
            # label_seqs have two dimensions
            self.label_seqs = [seq[:, :self.num_time_bins + 1] for seq in self.label_seqs]
            # again need to ensure the last element is always 1
            for idx, seq in enumerate(self.label_seqs):
                self.label_seqs[idx][:, -1] = 1
        else:
            # label_seqs only have 1 dimension
            self.label_seqs = [seq[:self.num_time_bins + 1] for seq in self.label_seqs]
            # again need to ensure the last element is always 1
            for idx, seq in enumerate(self.label_seqs):
                self.label_seqs[idx][-1] = 1

        self.time_bins = torch.arange(0, self.num_time_bins + 1) # + 1 accounts for the "never get it" case
        # store the time_bins as a float
        self.time_bins = self.time_bins.float()

        self.groups = None

    def shift_sequence(self, s, k):
        """
        Given a 1D array s of length num_hours, construct the k x num_hours matrix where
        row i is s shifted i times to the left, and the last bit of s is
        repeated past the end. The exception to this is if the only element that is a 1
        is the last element, in which case, do not shift it.

        s: the raw label sequence
        k: (k-1) is the number of shifts to make and k is the number of rows to return
        
        Example:
        s = [[0, 0, 1, 1], k=2 --> 
             [[0, 1, 1, 1],
              [1, 1, 1, 1]]
        
        s = [[0, 0, 1], k=3 -->
            [[0, 0, 1],
             [0, 0, 1],
             [0, 0, 1]]
        """
        n = len(s) - 1
        # remove the last element from s (which is always 1)
        last_element = s[-1]
        s = s[:-1]
        # create the grid of indices i+j
        i = np.arange(k)[:, None]  # shape (k, 1)
        j = np.arange(n)[None, :]  # shape (1, num_hours)
        idx = i + j # shape (k, num_hours)

        # clip i+j at n-1 so that once we go past the end, we stay at the last element
        idx = np.minimum(idx, n-1)

        # index directly into s
        label_matrix = s[idx]
        # concatenate a 1 column
        label_matrix = np.concatenate((label_matrix, np.full((k, 1), last_element)), axis=1)
        return label_matrix

    def define_splits(self, set_a_dfs, set_b_dfs):
        # define train, val, and test sets
        # take 10% from set_a and set_b for the val and test sets
        val_pct = 0.1
        test_pct = 0.1
        train_pct = 1 - val_pct - test_pct
        # sample the set and shuffle
        set_a_patients = list(set_a_dfs.keys())
        set_b_patients = list(set_b_dfs.keys())
        random.Random(42).shuffle(set_a_patients)
        random.Random(42).shuffle(set_b_patients)
        set_a_train_end_idx = int(train_pct * len(set_a_patients))
        set_b_train_end_idx = int(train_pct * len(set_b_patients))
        set_a_val_end_idx = int((train_pct + val_pct) * len(set_a_patients))
        set_b_val_end_idx = int((train_pct + val_pct) * len(set_b_patients))
        set_a_train_patients = set_a_patients[:set_a_train_end_idx]
        set_b_train_patients = set_b_patients[:set_b_train_end_idx]
        set_a_val_patients = set_a_patients[
            set_a_train_end_idx:set_a_val_end_idx]
        set_b_val_patients = set_b_patients[
            set_b_train_end_idx:set_b_val_end_idx]
        set_a_test_patients = set_a_patients[set_a_val_end_idx:]
        set_b_test_patients = set_b_patients[set_b_val_end_idx:]
        patient_splits = {
            'train': {
                'setA': set_a_train_patients,
                'setB': set_b_train_patients
            },
            'val': {
                'setA': set_a_val_patients,
                'setB': set_b_val_patients
            },
            'test': {
                'setA': set_a_test_patients,
                'setB': set_b_test_patients
            }
        }
        return patient_splits

    def record_patient_id_to_idxs(self,
                                  patient_id_to_row_idxs,
                                  set_dfs,
                                  set_ids,
                                  start_idx=0):
        # record mapping of patient_id to row index
        idx = start_idx
        for patient_id in set_ids:
            idx_start = idx
            idx_end = idx + len(set_dfs[patient_id])
            patient_id_to_row_idxs[patient_id] = (idx_start, idx_end)
            idx = idx_end
        return patient_id_to_row_idxs, idx

    def fit_preprocessor(self, train_df):
        # define a standard scaler preprocessers with median as a fill stategy
        numeric_transformers = []
        for col in self.cont_vars:
            numeric_transformers.append(
                (f"impute_{col}",
                 make_numeric_impute_pipeline(fill_strategy="median"), [col]))

        # combine into a single ColumnTransformer that processes each numeric column individually
        numeric_preprocessor = ColumnTransformer(
            transformers=numeric_transformers, remainder="drop")

        # handle categorical columns
        # convert the categorical variables to strings
        categories_for_encoder = build_categorical_categories(
            train_df, self.cat_vars, missing_token='missing')
        # create the pipeline for categorical columns that first imputes missing values
        # and then encodes the categorical values with OrdinalEncoder
        cat_pipeline = Pipeline([("impute",
                                  SimpleImputer(strategy="constant",
                                                fill_value="missing")),
                                 ("encode",
                                  OrdinalEncoder(categories=list(
                                      categories_for_encoder.values()), ))])

        # combine the numeric and categorical preprocessors
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_preprocessor, self.cont_vars),
            ("cat", cat_pipeline, self.cat_vars)
        ],
                                         remainder="drop",
                                         verbose_feature_names_out=False)
        # fit the preprocessor on the training data
        preprocessor.fit(train_df)
        return preprocessor, categories_for_encoder

    def load_psvs(self, set_dir):
        num_hours = self.config.num_hours if self.config.num_hours > 0 else None
        set_dfs = {}
        set_labels = {}
        for filename in tqdm(os.listdir(set_dir)):
            if filename.endswith('.psv'):
                filepath = os.path.join(set_dir, filename)
                patient_id = filename.split('.')[0]
                df = pd.read_csv(filepath, sep='|')
                # even if we truncate the number of observed hours, if we're doing sequential prediction
                # we can't truncate the labels (because by the final hour (num_hours), we're going to be looking ahead to num_hours + time_bins)
                label_seq = df[self.label_col].values
                # we may limit the number of rows for testing and development
                df = df.iloc[:num_hours]
                df['patient_id'] = patient_id
                set_dfs[patient_id] = df
                set_labels[patient_id] = label_seq
        return set_dfs, set_labels

    def collapse_rows(self, examples, cat_var_idxs):
        collapsed_examples = []
        for x in examples:
            # average row-wise
            agg_x = np.mean(x, axis=0)
            # so far, we've blindly averaged all columns, but we need to do something different for the categorical columns
            # namely, we want the max within each group
            cat_array = x[:, cat_var_idxs]
            max_x = np.max(cat_array, axis=0)

            # now put the two together
            for i, col_idx in enumerate(cat_var_idxs):
                agg_x[col_idx] = max_x[i]

            collapsed_examples.append(agg_x)

        return collapsed_examples

    def breakout_patients(self, transformed_data, patient_id_to_row_idxs):
        examples = []
        for patient_id, (start_idx, end_idx) in patient_id_to_row_idxs.items():
            patient_examples = transformed_data[start_idx:end_idx]
            examples.append(patient_examples)
        return examples

    def record_labels_sequences(self, set_a_labels, set_b_labels, patient_splits, split):
        # record the label sequences for the event times, process on the fly later
        # because we want maximum flexibility in how we aggregate the label
        # or use it in a utility function
        label_seqs = []
        for patient_id in patient_splits[split]['setA']:
            label_seq = set_a_labels[patient_id]
            label_seqs.append(label_seq)
        for patient_id in patient_splits[split]['setB']:
            label_seq = set_b_labels[patient_id]
            label_seqs.append(label_seq)
        return label_seqs

    def get_train_val_test(self):
        set_a_dfs, set_a_labels = self.load_psvs(self.set_a)
        set_b_dfs, set_b_labels = self.load_psvs(self.set_b)
        # define the splits
        patient_splits = self.define_splits(set_a_dfs, set_b_dfs)

        # combine the dataframes into a single dataframes
        train_patient_id_to_row_idxs, train_df = self.merge_set_a_set_b(set_a_dfs, set_b_dfs, patient_splits, split='train')

        val_patient_id_to_row_idxs, val_df = self.merge_set_a_set_b(set_a_dfs, set_b_dfs, patient_splits, split='val')

        test_patient_id_to_row_idxs, test_df = self.merge_set_a_set_b(set_a_dfs, set_b_dfs, patient_splits, split='test')
        returned = {'train_patient_id_to_row_idxs': train_patient_id_to_row_idxs,
                   'train_df': train_df,
                   'val_patient_id_to_row_idxs': val_patient_id_to_row_idxs,
                   'val_df': val_df,
                   'test_patient_id_to_row_idxs': test_patient_id_to_row_idxs,
                   'test_df': test_df,
                   'patient_splits': patient_splits,
                   'setA_labels': set_a_labels,
                   'setB_labels': set_b_labels}
        return returned

    def merge_set_a_set_b(self, set_a_dfs, set_b_dfs, patient_splits, split):
        a_df = pd.concat([set_a_dfs[patient_id] for patient_id in patient_splits[split]['setA']])
        # record the patient_id to row index mapping
        patient_id_to_row_idxs = {}
        idx = 0
        patient_id_to_row_idxs, idx = self.record_patient_id_to_idxs(patient_id_to_row_idxs, set_a_dfs, patient_splits[split]['setA'], start_idx=idx)
        b_df = pd.concat([set_b_dfs[patient_id]for patient_id in patient_splits[split]['setB']])
        patient_id_to_row_idxs, idx = self.record_patient_id_to_idxs(patient_id_to_row_idxs, set_b_dfs, patient_splits[split]['setB'], start_idx=idx)
        df = pd.concat([a_df, b_df])
        return patient_id_to_row_idxs, df

    def preprocess_dataset(self):
        data_metadata_labels = self.get_train_val_test()
        train_patient_id_to_row_idxs, train_df, val_patient_id_to_row_idxs, val_df, test_patient_id_to_row_idxs, test_df, patient_splits = data_metadata_labels['train_patient_id_to_row_idxs'], data_metadata_labels['train_df'], data_metadata_labels['val_patient_id_to_row_idxs'], data_metadata_labels['val_df'], data_metadata_labels['test_patient_id_to_row_idxs'], data_metadata_labels['test_df'], data_metadata_labels['patient_splits']


        # ensure cat_vars are strings
        train_df = convert_cat_vars_to_str(train_df,
                                           self.cat_vars,
                                           missing_token='missing')
        val_df = convert_cat_vars_to_str(val_df,
                                         self.cat_vars,
                                         missing_token='missing')
        test_df = convert_cat_vars_to_str(test_df,
                                          self.cat_vars,
                                          missing_token='missing')

        # fit the preprocessor on the training data
        preprocessor, categories_for_encoder = self.fit_preprocessor(train_df)

        # transform the datasets
        train_transformed = preprocessor.transform(train_df)
        val_transformed = preprocessor.transform(val_df)
        test_transformed = preprocessor.transform(test_df)

        # break the examples into arrays per patient
        train_examples = self.breakout_patients(train_transformed,
                                                train_patient_id_to_row_idxs)
        val_examples = self.breakout_patients(val_transformed,
                                              val_patient_id_to_row_idxs)
        test_examples = self.breakout_patients(test_transformed,
                                               test_patient_id_to_row_idxs)

        # get the feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()
        # map to column indexes
        feature_names = {name: i for i, name in enumerate(feature_names)}

        # retrieve the categorical columns, which will index into an embedding table
        embed_col_indexes = []
        for col in self.cat_vars:
            embed_col_indexes.append(feature_names[col])
        # create an embedding map for ALL variables that will index into an embedding table
        embed_map = {}
        for col in categories_for_encoder:
            embed_map[col] = categories_for_encoder[col]
        embed_col_indexes = embed_col_indexes
        embed_map = embed_map

        train_label_seqs = self.record_labels_sequences(data_metadata_labels['setA_labels'], data_metadata_labels['setB_labels'], patient_splits, split='train')
        val_label_seqs = self.record_labels_sequences(
            data_metadata_labels['setA_labels'], data_metadata_labels['setB_labels'], patient_splits, split='val')
        test_label_seqs = self.record_labels_sequences(
            data_metadata_labels['setA_labels'], data_metadata_labels['setB_labels'], patient_splits, split='test')

        # if we're collapsing to a single record, then average continuous cols and take the max over the categorical cols
        if self.config.single_record:
            # collapse the rows into a single row per patient
            train_examples = self.collapse_rows(train_examples,
                                                embed_col_indexes)
            val_examples = self.collapse_rows(val_examples, embed_col_indexes)
            test_examples = self.collapse_rows(test_examples,
                                               embed_col_indexes)

        # patient_ids
        train_patient_ids = list(train_patient_id_to_row_idxs.keys())
        val_patient_ids = list(val_patient_id_to_row_idxs.keys())
        test_patient_ids = list(test_patient_id_to_row_idxs.keys())

        return {
            'X_train': train_examples,
            'y_train': train_label_seqs,
            'X_val': val_examples,
            'y_val': val_label_seqs,
            'X_test': test_examples,
            'y_test': test_label_seqs,
            'metadata': {
                'patient_splits': patient_splits,
                'train_patient_ids': train_patient_ids,
                'val_patient_ids': val_patient_ids,
                'test_patient_ids': test_patient_ids,
                'feature_names': feature_names,
                'embed_col_indexes': embed_col_indexes,
                'embed_map': embed_map,
                'categories_for_encoder': categories_for_encoder,
                'cat_vars': self.cat_vars,
                'cont_vars': self.cont_vars,
                'label_col': self.label_col
            }
        }, preprocessor

    def __getitem__(self, index):
        x = self.data[index]
        label_seqs = self.label_seqs[index]
        event_time = label_seqs.argmax(dim=-1)
        censored = event_time >= self.num_time_bins # False
        if self.groups is None:
            group = None
        else:
            group = self.groups[index]
        return torch.tensor(x, dtype=torch.float32), censored, event_time, label_seqs, group, index

    def __len__(self):
        return len(self.data)

def sepsis_data(train_config, shuffle_train=True, metadata_only=False):
    """Returns the Sepsis dataset."""
    # input dim isn't so simple
    metadata = DatasetMetadata(name='Sepsis', num_classes=None, input_dim=None)
    if metadata_only:
        return metadata

    train_dataset = Sepsis(config=train_config, split='train')
    val_dataset = Sepsis(config=train_config, split='val')
    test_dataset = Sepsis(config=train_config, split='test')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, train_config, support2_collate_fn, shuffle_train=shuffle_train)
    return train_loader, val_loader, test_loader, metadata


def apply_discretizers(df, discretizers):
    # apply the discretizers to the dataframe
    for col, discretizer in discretizers.items():
        valid_rows = df[col].notna()
        # only apply the discretizer to the valid rows
        df.loc[valid_rows,
               col] = discretizer.transform(df.loc[valid_rows,
                                                   col].values.reshape(-1, 1))
    # fill the NaN values with -1
    df.fillna(-1, inplace=True)
    # increment the values by 1 so that missing values have index position 0
    # in the dowstream embedding table
    for col in discretizers.keys():
        df[col] += 1
    return df


def load_data(train_config, shuffle_train=True, update_seed=False):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    if update_seed:
        # set the seed to the one defined in the config
        random.seed(train_config.seed)
        torch.manual_seed(train_config.seed)
        np.random.seed(train_config.seed)
    feature_names = None
    embed_map = None
    embed_col_indexes = None
    time_bins = None
    if train_config.dataset == "MNIST":
        train_loader, val_loader, test_loader, metadata = mnist_survival_data(
            train_config, shuffle_train=shuffle_train)
    elif train_config.dataset == "SUPPORT2":
        train_loader, val_loader, test_loader, metadata = support2_data(
            train_config, shuffle_train=shuffle_train)
        # retrieve the feature names and embed_maps to know how many input dimensions we have
        feature_names = train_loader.dataset.feature_names
        embed_map = train_loader.dataset.embed_map
        embed_col_indexes = train_loader.dataset.embed_col_indexes
        time_bins = len(train_loader.dataset.time_bins)
    elif train_config.dataset == 'Sepsis':
        train_loader, val_loader, test_loader, metadata = sepsis_data(
            train_config, shuffle_train=shuffle_train)
        # retrieve the feature names and embed_maps to know how many input dimensions we have
        feature_names = train_loader.dataset.feature_names
        embed_map = train_loader.dataset.embed_map
        embed_col_indexes = train_loader.dataset.embed_col_indexes
        time_bins = len(train_loader.dataset.time_bins)

    # set the seed back to the experimentally defined seed
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    return feature_names,embed_map,embed_col_indexes,time_bins,train_loader,val_loader,test_loader,metadata


# if __name__ == "__main__":
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Lambda(lambda x: x.view(-1))
    # ])
    # dataset = MNISTSurvival(root='./data',
    #                         train=True,
    #                         download=True,
    #                         transform=transform)
    # image, target, event_time = dataset[0]
    # train_config = TrainConfig()
    # train_loader, val_loader, test_loader, metadata = mnist_survival_data(
    #     train_config)
    # batch = next(iter(train_loader))

    # train_config = TrainConfig()
    # train_loader, val_loader, test_loader, metadata = support2_data(
    #     train_config)
    # batch = next(iter(train_loader))
    # x, target, censored, event_times, event_times_label_seqs, raw_event_time, raw_event_time_label_seq = batch

    # train_config = TrainConfig(data_dir='./data', cache_dataset=True, num_hours=100, single_record=True)
    # sepsis_dataset = Sepsis(train_config, split='train')