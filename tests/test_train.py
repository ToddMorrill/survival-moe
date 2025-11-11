import torch
from tqdm import tqdm
from survkit.data import load_data, mnist_survival_data, log_normal_pdf, log_normal_cdf
from survkit.train import Trainer, get_model

from tests.test_utils import get_config


def sample_y_pred_y_true():
    # assume 5 time bins + 1 for no event and batch size 2
    # logits by time bin
    y_pred = torch.tensor([[-0.2, -0.1, 0.15, 0.69, 0.42],
                           [-0.72, -0.15, 0.28, 0.90, 1.2]])
    y = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.float32)
    return y_pred, y

def sample_y_pred_y_true_large():
    # assume 5 time bins + 1 for no event and batch size 11
    # logits by time bin
    y_pred = torch.tensor([[-0.2, -0.1, 0.15, 0.69, 0.42],
                           [-0.72, -0.15, 0.28, 0.90, 1.2],
                           [-0.5, -0.3, 0.1, 0.4, 0.6],
                           [-0.6, -0.2, 0.2, 0.5, 0.7],
                           [-0.8, -0.4, 0.05, 0.3, 0.9],
                           [-1.0, -0.5, 0.1, 0.2, 1.1],
                           [-1.2, -0.6, 0.15, 0.25, 1.3],
                           [-1.4, -0.7, 0.2, 0.3, 1.4],
                           [-1.6, -0.8, 0.25, 0.35, 1.5],
                           [-1.8, -1.0, 0.3, 0.4, 1.6],
                           [-2.0, -1.2, 0.35, 0.45, 1.7]])
    y = torch.tensor([[0, 0, 0, 1, 1],
                      [0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1]
                      ], dtype=torch.float32)
    return y_pred, y

def test_discrete_survival_loss_uncensored():
    _, train_config = get_config("--time_bins 5")
    # create the trainer
    trainer = Trainer(train_config, model=None, optimizer=None, time_bins=5)
    y_pred, y = sample_y_pred_y_true()
    log_numerator = torch.sum(torch.tensor([[0.0, 0.69, 0.42],
                                            [0.28, 0.90, 1.2]]),
                              dim=1)
    reversed_partial_sums = torch.tensor([[
        0.42, 0.42 + 0.69, 0.42 + 0.69 + 0.15, 0.42 + 0.69 + 0.15 - 0.1,
        0.42 + 0.69 + 0.15 - 0.1 - 0.2
    ],
                                          [
                                              1.2, 1.2 + 0.90,
                                              1.2 + 0.90 + 0.28,
                                              1.2 + 0.90 + 0.28 - 0.15,
                                              1.2 + 0.90 + 0.28 - 0.15 - 0.72
                                          ]])
    summed_exp = torch.sum(torch.exp(reversed_partial_sums), dim=1)
    log_denominator = torch.log(summed_exp)
    losses_expected = -(log_numerator - log_denominator)
    losses = trainer.discrete_survival_loss_uncensored(y_pred, y)
    assert torch.allclose(losses, losses_expected)


def test_discrete_survival_loss_censored():
    _, train_config = get_config("--time_bins 5")
    # create the trainer
    trainer = Trainer(train_config, model=None, optimizer=None, time_bins=5)
    y_pred, y = sample_y_pred_y_true()
    # assume the event times are censored at the 3rd and 4th time bins
    y_censored = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
                              dtype=torch.float32)
    numerator_sums = torch.tensor([[
        -float('inf'), -float('inf'), 0.15 + 0.69 + 0.42, 0.69 + 0.42, 0.42
    ], [-float('inf'), -float('inf'), -float('inf'), 0.90 + 1.2, 1.2]])
    log_numerators = torch.logsumexp(numerator_sums, dim=1)
    reversed_partial_sums = torch.tensor([[
        0.42, 0.42 + 0.69, 0.42 + 0.69 + 0.15, 0.42 + 0.69 + 0.15 - 0.1,
        0.42 + 0.69 + 0.15 - 0.1 - 0.2
    ],
                                          [
                                              1.2, 1.2 + 0.90,
                                              1.2 + 0.90 + 0.28,
                                              1.2 + 0.90 + 0.28 - 0.15,
                                              1.2 + 0.90 + 0.28 - 0.15 - 0.72
                                          ]])
    summed_exp = torch.sum(torch.exp(reversed_partial_sums), dim=1)
    log_denominators = torch.log(summed_exp)
    losses_expected = -(log_numerators - log_denominators)
    losses = trainer.discrete_survival_loss_censored(y_pred, y_censored)
    assert torch.allclose(losses, losses_expected)


def test_discrete_survival_pmf():
    _, train_config = get_config("--time_bins 5")
    # create the trainer
    trainer = Trainer(train_config, model=None, optimizer=None, time_bins=5)
    y_pred, _ = sample_y_pred_y_true()
    # order the pmf support by time bins ending with no event (or another way to think of this is "happens later"), i.e.,
    # (1, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1), (0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1), (0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0)
    partial_sums = torch.tensor([[
        -0.2 - 0.1 + 0.15 + 0.69 + 0.42, -0.1 + 0.15 + 0.69 + 0.42,
        0.15 + 0.69 + 0.42, 0.69 + 0.42, 0.42
    ],
                                 [
                                     -0.72 - 0.15 + 0.28 + 0.90 + 1.2,
                                     -0.15 + 0.28 + 0.90 + 1.2,
                                     0.28 + 0.90 + 1.2, 0.90 + 1.2, 1.2
                                 ]])
    # exponentiate
    unnormalized_pmf = torch.exp(partial_sums)
    pmf_expected = unnormalized_pmf / torch.sum(
        unnormalized_pmf, dim=1, keepdim=True)
    pmf = trainer.discrete_survival_pmf(y_pred)
    assert torch.allclose(pmf, pmf_expected)
    # assert that the pmf sums to 1
    assert torch.allclose(torch.sum(pmf, dim=1), torch.tensor(1.0))


def test_discrete_survival_function():
    _, train_config = get_config("--time_bins 5")
    # create the trainer
    trainer = Trainer(train_config, model=None, optimizer=None, time_bins=5)
    y_pred, _ = sample_y_pred_y_true()
    survival_function = trainer.discrete_survival_function(y_pred)
    # check that the survival function is monotonically decreasing
    assert torch.all(torch.diff(survival_function, dim=1) <= 0)
    # check that the survival function is between 0 and 1
    # I encountered a tiny bit of numerical imprecision in computing the discrete survival pmf
    assert torch.all(survival_function >= (0 - 1e-6)) and torch.all(
        survival_function <= 1)
    # check that the diffs are equal to the pmf
    pmf = trainer.discrete_survival_pmf(y_pred)
    # prepend a 1 to the survival function so that the diffs are equal to the pmf
    diffs = torch.diff(torch.flip(survival_function, dims=[1]),
                       append=torch.ones_like(survival_function[:, :1]),
                       dim=1)
    diffs = torch.flip(diffs, dims=[1])
    assert torch.allclose(diffs, pmf)

# def test_ece_equal_mass_binning_with_uncensored_mask():
#     _, train_config = get_config("--time_bins 5")
#     time_bins = torch.arange(train_config.time_bins+1, dtype=torch.float32)
#     # create the trainer
#     trainer = Trainer(train_config, model=None, optimizer=None, time_bins=None)
#     y_pred, y = sample_y_pred_y_true_large()
#     c = torch.tensor([False] * y.shape[0])  # assume all instances are uncensored
#     # convert y_pred to probabilities
#     y_pred = trainer.discrete_survival_pmf(y_pred)
#     # produce the expected inputs to the ece function
#     batch_y_pred_by_decile, batch_y_by_decile, batch_y_pred_deciles_one_hot, _ = trainer.compute_ece_at_ts(
#                     y_pred.to(train_config.device),
#                     y.to(train_config.device),
#                     c.to(train_config.device),
#                     return_decile_avgs=False)

#     returned = trainer.ece_aggregate(batch_y_pred_by_decile,
#                                                                    batch_y_by_decile,
#                                                                    y_pred_deciles_one_hot=None,
#                                                                    uncensored_mask=None,
#                                                                    return_avgs=True)
#     ece_per_time, y_pred_by_bin_mean, y_by_bin_mean, valid_by_bin_count = returned
#     # assert shapes
#     assert ece_per_time.shape == (train_config.time_bins,)
#     assert y_pred_by_bin_mean.shape == (train_config.time_bins, 10)
#     assert y_by_bin_mean.shape == (train_config.time_bins, 10)
#     assert valid_by_bin_count.shape == (train_config.time_bins, 10)

#     # assert y_pred_by_bin_mean and y_by_bin_mean are probabilities
#     assert torch.all(y_pred_by_bin_mean >= 0) and torch.all(y_pred_by_bin_mean <= 1.0000001)
#     assert torch.all(y_by_bin_mean >= 0) and torch.all(y_by_bin_mean <= 1.0000001)
#     # expected valid_by_bin_count should fill up the first 5 bins with 2 instances each, the next with 1, and the rest with 0
#     exp_valid_by_bin_count = torch.tensor([[2., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [2., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [2., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [2., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#         [2., 1., 1., 1., 1., 1., 1., 1., 1., 1.]], dtype=int).to(train_config.device)
#     assert torch.allclose(valid_by_bin_count, exp_valid_by_bin_count)

# def test_ece_equal_mass_binning_MNIST(num_batches=2, time_bins=100):
#     _, train_config = get_config(
#         f"--experiment_args args/mnist_baseline.args --time_bins {time_bins}")
#     feature_names, embed_map, embed_col_indexes, time_bins, train_loader, val_loader, test_loader, metadata = load_data(
#         train_config)
#     time_bins = train_loader.dataset.dataset.time_bins
#     # broadcast to 10 classes for MNIST
#     time_bins = time_bins.unsqueeze(0).repeat(10, 1)  # shape (10, time_bins+1)
#     means = train_loader.dataset.dataset.means[:, None]  # shape (10, 1)
#     stds = train_loader.dataset.dataset.stds[:, None]  # shape (10, 1)
#     # get cdf values for the time bins
#     cdf_values = log_normal_cdf(time_bins, means, stds)
#     cdf_values = torch.tensor(cdf_values)

#     # convert to probabilities
#     t_seqs = []
#     y_preds = []
#     cs = []
#     idx = 0
#     while idx != num_batches:
#         for batch in train_loader:
#             x, y, c, t, t_seq, t_raw, t_raw_seq, g = batch
#             y_pred = cdf_values[y]
#             # y_pred = pmf_values[y]
#             t_seqs.append(t_seq)
#             y_preds.append(y_pred)
#             cs.append(c)
#             idx += 1
#             if idx == num_batches:
#                 break
#         # get a new train_loader with a different seed
#         train_config.seed += 1
#         feature_names, embed_map, embed_col_indexes, _, train_loader, val_loader, test_loader, metadata = load_data(train_config, update_seed=True)
#     t_seqs = torch.cat(t_seqs, dim=0).to(train_config.device)
#     y_preds = torch.cat(y_preds, dim=0).to(train_config.device)
#     cs = torch.cat(cs, dim=0).to(train_config.device)
#     t_seqs = t_seqs.to(torch.float32)
#     y_preds = y_preds.to(torch.float32)

#     # create the trainer
#     trainer = Trainer(train_config,
#                       model=None,
#                       optimizer=None,
#                       time_bins=time_bins)
#     # manually compute the cdf because the procedure used during training will destroy the pmf structure
#     # y_pred_cdf = torch.cumsum(y_preds, dim=1)  # shape (batch_size, time_bins)
#     # break this out into deciles. shape (batch_size, time_bins, 10)
#     deciles = torch.linspace(0, 0.9, 10).to(train_config.device)  # shape (10)
#     y_pred_deciles = torch.bucketize(
#         y_preds, deciles,
#         right=True) - 1  # minus 1 to get 0-indexed buckets
#     # create a 1-hot encoding of the deciles
#     y_pred_deciles_one_hot = torch.nn.functional.one_hot(
#         y_pred_deciles, num_classes=len(
#             deciles)).float()  # shape (batch_size, time_bins, 10)
#     # fill in the probabilities
#     y_pred_by_decile = y_pred_deciles_one_hot * y_preds.unsqueeze(
#         -1)  # shape (batch_size, time_bins, 10)

#     # get the rate of occurrence for each decile
#     y_by_decile = y_pred_deciles_one_hot * t_seqs.unsqueeze(
#         -1)  # shape (batch_size, time_bins, 10)

#     # equal mass binning
#     ece_equal_mass_results = trainer.ece_equal_mass_binning_with_uncensored_mask(
#         y_pred_by_decile,
#         y_by_decile,
#         y_pred_deciles_one_hot,
#         uncensored_mask=None,
#         return_avgs=True)
#     ece_equal_mass_at_ts, y_pred_by_equal_mass_bin_mean, y_by_equal_mass_bin_mean, per_equal_mass_bin_count = ece_equal_mass_results
#     mean_ece = ece_equal_mass_at_ts.mean().item()
#     return mean_ece


# def test_ece_equal_mass_miscal():
#     _, train_config = get_config(
#         "--experiment_args args/mnist_baseline.args --time_bins 30")
#     time_bins = 1
#     trainer = Trainer(train_config,
#                       model=None,
#                       optimizer=None,
#                       time_bins=time_bins)
#     # y_pred for easy labels
#     y_pred = torch.full((100, 1), 0.9)
#     y = torch.zeros(100, 1)

#     # manually compute the cdf because the procedure used during training will destroy the pmf structure
#     y_pred_cdf = torch.cumsum(y_pred, dim=1)  # shape (batch_size, time_bins)
#     # break this out into deciles. shape (batch_size, time_bins, 10)
#     deciles = torch.linspace(0, 0.9, 10)  # shape (10)
#     y_pred_deciles = torch.bucketize(
#         y_pred_cdf, deciles,
#         right=True) - 1  # minus 1 to get 0-indexed buckets
#     # create a 1-hot encoding of the deciles
#     y_pred_deciles_one_hot = torch.nn.functional.one_hot(
#         y_pred_deciles,
#         num_classes=len(deciles)).float()  # shape (batch_size, time_bins, 10)
#     # fill in the probabilities
#     y_pred_by_decile = y_pred_deciles_one_hot * y_pred_cdf.unsqueeze(
#         -1)  # shape (batch_size, time_bins, 10)

#     # get the rate of occurrence for each decile
#     y_by_decile = y_pred_deciles_one_hot * y.unsqueeze(
#         -1)  # shape (batch_size, time_bins, 10)
#     # equal mass binning
#     ece_equal_mass_results = trainer.ece_equal_mass_binning_with_uncensored_mask(
#         y_pred_by_decile.to(train_config.device),
#         y_by_decile.to(train_config.device),
#         y_pred_deciles_one_hot.to(train_config.device),
#         uncensored_mask=None,
#         return_avgs=True)
#     ece_equal_mass_at_ts, y_pred_by_equal_mass_bin_mean, y_by_equal_mass_bin_mean, per_equal_mass_bin_count = ece_equal_mass_results
#     mean_ece = ece_equal_mass_at_ts.mean()
#     assert torch.allclose(mean_ece,
#                           torch.tensor(0.9, dtype=mean_ece.dtype),
#                           atol=1e-7)


# def test_ece_equal_mass_perfect_cal():
#     _, train_config = get_config(
#         "--experiment_args args/mnist_baseline.args --time_bins 30")
#     time_bins = 1
#     trainer = Trainer(train_config,
#                       model=None,
#                       optimizer=None,
#                       time_bins=time_bins)
#     # y_pred for easy labels
#     preds = torch.linspace(0.1, 1.0, 10)  # [0.1, 0.2, ..., 1.0]
#     y_pred = preds.repeat_interleave(10).view(-1, 1)  # Shape (100, 1)

#     # For the 10 samples with pred=0.1, make 1 label=1.
#     # For the 10 samples with pred=0.2, make 2 labels=1, etc.
#     labels = []
#     for i in range(10):
#         num_events = i + 1
#         group_labels = ([0] * (10 - num_events)) + ([1] * num_events)
#         labels.extend(group_labels)
#     y = torch.tensor(labels, dtype=torch.float).view(-1, 1)  # Shape (100, 1)

#     # manually compute the cdf because the procedure used during training will destroy the pmf structure
#     y_pred_cdf = torch.cumsum(y_pred, dim=1)  # shape (batch_size, time_bins)
#     # break this out into deciles. shape (batch_size, time_bins, 10)
#     deciles = torch.linspace(0, 0.9, 10)  # shape (10)
#     y_pred_deciles = torch.bucketize(
#         y_pred_cdf, deciles,
#         right=True) - 1  # minus 1 to get 0-indexed buckets
#     # create a 1-hot encoding of the deciles
#     y_pred_deciles_one_hot = torch.nn.functional.one_hot(
#         y_pred_deciles,
#         num_classes=len(deciles)).float()  # shape (batch_size, time_bins, 10)
#     # fill in the probabilities
#     y_pred_by_decile = y_pred_deciles_one_hot * y_pred_cdf.unsqueeze(
#         -1)  # shape (batch_size, time_bins, 10)

#     # get the rate of occurrence for each decile
#     y_by_decile = y_pred_deciles_one_hot * y.unsqueeze(
#         -1)  # shape (batch_size, time_bins, 10)
#     # equal mass binning
#     ece_equal_mass_results = trainer.ece_equal_mass_binning_with_uncensored_mask(
#         y_pred_by_decile.to(train_config.device),
#         y_by_decile.to(train_config.device),
#         y_pred_deciles_one_hot.to(train_config.device),
#         uncensored_mask=None,
#         return_avgs=True)
#     ece_equal_mass_at_ts, y_pred_by_equal_mass_bin_mean, y_by_equal_mass_bin_mean, per_equal_mass_bin_count = ece_equal_mass_results
#     mean_ece = ece_equal_mass_at_ts.mean()
#     assert torch.allclose(mean_ece,
#                           torch.tensor(0.0, dtype=mean_ece.dtype),
#                           atol=1e-7)
#     assert torch.allclose(y_pred_by_equal_mass_bin_mean,
#                           y_by_equal_mass_bin_mean,
#                           atol=1e-7)
#     assert torch.allclose(per_equal_mass_bin_count,
#                           torch.tensor([10] * 10).view(1, -1).to(train_config.device),
#                           atol=1e-7)
#     assert torch.allclose(y_pred_by_equal_mass_bin_mean,
#                             preds.view(1, -1).to(train_config.device),
#                             atol=1e-7)
#     assert torch.allclose(y_by_equal_mass_bin_mean,
#                             preds.view(1, -1).to(train_config.device),
#                             atol=1e-7)


if __name__ == "__main__":
    # plot mean_ece vs num_batches
    time_bin_vals = [10, 20, 50, 100]
    import matplotlib.pyplot as plt
    num_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3125]
    num_examples = [x * 64 for x in num_batches]
    mean_eces = []
    for time_bins in time_bin_vals:
        mean_ece = []
        for num_batch in tqdm(num_batches):
            mean_ece.append(
                test_ece_equal_mass_binning_MNIST(num_batch, time_bins))
        mean_eces.append(mean_ece)

    for idx, time_bins in enumerate(time_bin_vals):
        print(f"Mean ECE for time bins {time_bins}: {mean_eces[idx]}")
        plt.plot(num_examples, mean_eces[idx], label=f"Time bins: {time_bins}")
    plt.xlabel("Number of examples")
    plt.ylabel("Mean ECE")
    plt.title("Mean ECE vs Number of examples")
    plt.legend()
    plt.savefig("analysis/mean_ece_vs_num_examples.png")
