from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Datasets(Enum):
    MNIST = 0
    SUPPORT2 = 1
    Sepsis = 2

class Models(Enum):
    FFNet = 0
    FFNetMixture = 1
    FFNetMixtureMTLR = 2  # Mixture of MTLR heads
    FFNetTimeWarpMoE = 3  # Time-Warped Mixture of Experts
    SKSurvRF = 4  # Random Survival Forest from sksurv
    SKSurvCox = 5  # Cox Proportional Hazards model from sksurv

class TimeWarpFunctions(Enum):
    TwoLogistics = 0
    BetaCDF = 1

class ExpertMetric(Enum):
    Euclidean = 0
    Cosine = 1

class Optimizers(Enum):
    Adam = 0
    AdamW = 1
    SGD = 2

class LossFunctions(Enum):
    DiscreteSurvival = 0
    CrossEntropy = 1

@dataclass
class TrainConfig:
    # NB: any field without a default value specified or Optional will be required

    # optimization hyperparameters
    seed: int = field(default=42, metadata={"help": "Random seed."})
    optimizer: str = field(
        default=Optimizers.Adam.name,
        metadata={
            "help":
            f"The optimizer to use. Options are {list(Optimizers.__members__.keys())}."
        })
    lr: float = field(default=0.001, metadata={"help": "Learning rate."})
    epochs: int = field(default=1,
                        metadata={"help": "Number of training epochs."})

    # dataset hyperparameters
    dataset: str = field(
        default=Datasets.MNIST.name,
        metadata={
            "help":
            f"The dataset to use. Options are {list(Datasets.__members__.keys())}."
        })
    quantile_event_times: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use quantiles of the event times as the time bins. If False, will use uniform bins between 0 and maximum of event times."
        })
    support2_start_day: int = field(
        default=None,
        metadata={
            "help":
            "Start day for SUPPORT2. If None, will use all days. Otherwise, must be a non-negative integer."
        })
    support2_censor_day: int = field(
        default=None,
        metadata={
            "help":
            "Censor day for SUPPORT2. If None, will use all days. Otherwise, must be a positive integer."
        })
    cache_dataset: bool = field(
        default=True, metadata={"help": "Whether to cache the dataset."})
    num_hours: int = field(
        default=-1,
        metadata={
            "help":
            "Maximum number of hours of DATA to use in the SEPSIS dataset. -1 means all hours, otherwise, must be a positive integer. NB: time_bins still controls the number of lookahead hours for forecasting."
        })
    single_record: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use a single record for each patient in the SEPSIS dataset. If False, each hour is a patient record. If True, aggregates (avg. continuous vars, max cat. vars) over num_hours of observations."
        })
    embedding_dimension: int = field(
        default=4,
        metadata={
            "help": "Embedding dimension for each of the categorical features."
        })
    one_hot_embed: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use one-hot encoding for the categorical features instead of learned embeddings."
        })
    batch_size: int = field(default=64, metadata={"help": "Batch size."})
    labeling_floor_mode: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use floor mode labeling scheme. floor=True means that events are labeled as occurring at the nearest time bin at or before the event time. floor=False means that events are labeled as occurring at the nearest time bin at or after the event time. Yu et al. (2011) use floor=False. floor=True is the more conservative approach because you flag events as occurring earlier than they actually do. However, floor=False makes more sense for censored times because we know that as of the earlier time bin, the event had not yet occurred and is appropriately labeled as a 0."
        })
    data_dir: str = field(
        default='./data',
        metadata={
            "help":
            "Directory where datasets are stored. If None, datasets will be downloaded to the current working directory."
        })
    mnist_means: List[float] = field(
        default_factory=lambda: [1.0],
        metadata={
            "help":
            "Means for the log-normal distribution for each class in MNIST. If single value, will be interpreted as the offset between classes with a starting value of 1.0."
        })
    mnist_stds: List[float] = field(
        default_factory=lambda: [0.5],
        metadata={
            "help":
            "Standard deviations for the log-normal distribution for each class in MNIST. If single value, will be used for all classes."
        })
    mnist_sample_pct: List[float] = field(
        default=None,
        metadata={
            "help":
            "Percentage of samples to use for each class in MNIST. If None, will use all samples. Otherwise, a pct must be specified for each class."
        })
    mnist_censoring_percentage: float = field(
        default=0.0,
        metadata={"help": "Percentage of events to censor in MNIST."})

    # model hyperparameters
    model: str = field(
        default=Models.FFNet.name,
        metadata={
            "help":
            f"The model to use. Options are {list(Models.__members__.keys())}."
        })
    time_warp_function: str = field(
        default=TimeWarpFunctions.TwoLogistics.name,
        metadata={
            "help":
            f"The time warp function to use for the FFNetTimeWarpMoE model. Options are {list(TimeWarpFunctions.__members__.keys())}."
        })
    mixture_components: int = field(
        default=10,
        metadata={
            "help":
            "Number of mixture components to use in the FFNetMixture model. Must be greater than 0 if using the FFNetMixture model."
        })
    mixture_topk: int = field(
        default=10,
        metadata={
            "help":
            "Number of top mixture components to use in the FFNetMixture model. Must be greater than 0 and less than or equal to mixture_components."
        })
    expert_metric: str = field(
        default=ExpertMetric.Cosine.name,
        metadata={
            "help":
            f"The metric to use for the expert router in the FFNetMixture or FFNetMemoryMixture model. Options are {list(ExpertMetric.__members__.keys())}."
        })
    learn_temperature: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to learn the temperature parameter in the FFNetMixture model. If True, the temperature will be learned as a model parameter. If False, the temperature will be fixed to the initialized value."
        })
    logit_temp: float = field(
        default=2.0,
        metadata={
            "help":
            "Temperature for the logits in the FFNetMixture or FFNetMemoryMixture model. If learn_temperature is True, this will be a learnable parameter. Otherwise, it will be fixed to this value."
        })
    hidden_dim: int = field(default=128,
                            metadata={"help": "Hidden dimension."})
    moe_value_dim: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "Value dimension for the mixture of experts. If None, will use hidden_dim."
        })
    moe_chunk_value_dim: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to chunk the value dimension into subspaces for each expert in the mixture of experts. If True, will create a separate output layer for each expert that operates on a subspace of the value dimension."
        })
    num_hidden_layers: int = field(
        default=2,
        metadata={
            "help":
            "Number of hidden layers. Does not include the input layer."
        })
    instance_ent_lambda: float = field(
        default=0.0,
        metadata={
            "help":
            "Lambda for the instance entropy loss. 0 means no instance entropy loss."
        })
    group_ent_lambda: float = field(
        default=0.0,
        metadata={
            "help":
            "Lambda for the group entropy loss. 0 means no group entropy loss."
        })
    cosine_anneal_group_ent_lambda: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use cosine annealing for the group entropy loss lambda. If True, the group entropy loss lambda will be annealed from group_ent_lambda to cosine_anneal_min_lambda_multiplier over the course of training."
        })
    cosine_anneal_min_lambda_multiplier: float = field(
        default=0.0,
        metadata={
            "help":
            "Minimum multiplier for the group entropy loss lambda when using cosine annealing. This is multiplied by group_ent_lambda to get the minimum value."
        })
    cosine_anneal_hold_epochs: int = field(
        default=0,
        metadata={
            "help":
            "Number of epochs to hold the group entropy loss lambda at the initial value before starting cosine annealing. 0 means no hold."
        })
    cosine_anneal_epochs: int = field(
        default=2,
        metadata={"help": "Number of epochs to use for cosine annealing."})
    time_bins: int = field(
        default=20,
        metadata={
            "help": "Number of time bins for the discrete survival curve."
        })
    quantize_event_times: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to quantize the event times according to the number of time bins."
        })
    loss: str = field(
        default=LossFunctions.DiscreteSurvival.name,
        metadata={
            "help":
            f"The loss function to use. Options are {list(LossFunctions.__members__.keys())}."
        })
    ipcw: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use IPCW (Inverse Probability of Censoring Weighting) adjustment for the group-based loss function and possibly evaluation metrics (e.g., accuracy, etc.). Mutually exclusive with impute."
        })
    impute: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use imputation for the group-based loss function and possibly evaluation metrics (e.g., accuracy, etc.). Mutually exclusive with IPCW."
        })
    administrative_censoring: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to treat all censored instances as uncensored for the purposes of evaluation metrics (e.g., accuracy, etc.). This is useful in scenarios where we know that the censored instances survived up to the last time bin."
        })
    adjust_eval: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to use IPCW or imputation adjustments for the evaluation metrics (e.g., accuracy, etc.)."
        })
    model_dir: str = field(
        default='./models',
        metadata={
            "help":
            "Directory where model files should be written. If None, models will be written to the current working directory."
        })
    model_hash: str = field(
        default=None,
        metadata={
            "help":
            "Hash of the model. If None, a hash will be generated from the model parameters."
        })
    save_model: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to save the best performing model to disk at the end of each epoch (assuming improvement). If True, the model will be saved to the model_dir directory."
        })
    monitored_val_metric: str = field(
        default='loss',
        metadata={
            "help":
            "Validation metric to monitor at the end of each training epoch for early stopping. Some options are 'loss', 'acc', 'brier_score', 'ece'."
        })
    minimize_val_metric: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to minimize the monitored validation metric. If False, the metric will be maximized."
        })
    early_stopping_patience: int = field(
        default=10,
        metadata={
            "help":
            "Number of epochs to wait for improvement before stopping training."
        })
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate."})

    # random survival forest hyperparameters
    rsf_n_estimators: int = field(
        default=100,
        metadata={"help": "Number of trees in the random survival forest."})
    rsf_min_samples_split: int = field(
        default=100,
        metadata={
            "help":
            "Minimum number of samples required to split an internal node in the random survival forest."
        })
    rsf_max_features: str = field(
        default='sqrt',
        metadata={
            "help":
            "Number of features to consider when looking for the best split in the random survival forest. If 'sqrt', will use the square root of the number of features."
        })

    # Cox model hyperparameters
    cox_alpha: float = field(
        default=0.01,
        metadata={
            "help":
            "Overall strength of the combined L1 and L2 penalties in the Cox model."
        })
    cox_l1_ratio: float = field(
        default=0.01,
        metadata={
            "help":
            "The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty."
        })

    # miscellaneous
    time_bin_pctiles: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75],
        metadata={
            "help":
            "Percentiles of the time bins to evaluate the model on. If None, will use all time bins."
        })
    log_dir: str = field(
        default='./logs',
        metadata={
            "help":
            "Directory where log files should be written. If None, logs will be written to the current working directory."
        })
    device: str = field(
        default="cuda",
        metadata={
            "help": "Device to use for computation (e.g., 'cuda', 'cpu')."
        })

    def __post_init__(self):
        if self.ipcw or self.impute:
            assert self.ipcw != self.impute, "ipcw and impute are mutually exclusive. Please set a maximum of one of them to True."
        if self.num_hours == 0:
            raise ValueError("num_hours must be a positive integer or -1.")
        # if rsf_max_features is an integer, convert it from string to int
        if self.rsf_max_features != 'sqrt':
            try:
                self.rsf_max_features = int(self.rsf_max_features)
            except ValueError:
                raise ValueError(
                    "rsf_max_features must be 'sqrt' or an integer.")
