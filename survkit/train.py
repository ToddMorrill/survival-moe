"""The main entry point for training a model.

Examples:
    $ python -m survkit.train \
        --experiment_args args/sepsis.args
"""
from functools import partial
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from tqdm import tqdm
import warnings

# supress UserWarning: The first time coordinate is not 0. A authentic survival curve should start from 0 with 100% survival probability. Adding 0 to the beginning of the time coordinates and 1 to the beginning of the predicted curves.
#   warnings.warn(zero_pad_msg)
warnings.filterwarnings(
    "ignore",
    message=
    "The first time coordinate is not 0. A authentic survival curve should start from 0 with 100% survival probability. Adding 0 to the beginning of the time coordinates and 1 to the beginning of the predicted curves."
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator
from torchsurv.metrics.cindex import ConcordanceIndex
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from SurvivalEVAL.Evaluations.util import predict_median_st
import matplotlib.pyplot as plt
import wandb

from .model import FFNet
from .configs import Optimizers, Models
from .utils import get_args, get_lambda_multiplier
from .data import load_data, SubclassSampler


class Trainer:
    """Trains a model."""

    def __init__(self,
                 train_config,
                 model,
                 optimizer,
                 time_bins,
                 model_dir=None):
        self.train_config = train_config
        self.device = torch.device(train_config.device)
        self.model = model.to(self.device) if model is not None else None
        self.optimizer = optimizer if optimizer is not None else None
        self.time_bins = time_bins
        self.model_dir = model_dir

    def compute_kaplan_meier(self,
                             loader=None,
                             event_times=None,
                             censored=None,
                             censoring_dist=False,
                             min_estimate=1e-2):
        """Computes the Kaplan-Meier estimate from the specified loader. Optionally
        returns the censoring distribution."""
        if event_times is None or censored is None:
            num_points = get_num_train_data_points(loader)
            event_times = torch.zeros(num_points, dtype=torch.float)
            censored = torch.zeros(num_points, dtype=torch.bool)
            idx = 0
            for i, batch in enumerate(loader):
                x, y, c, t, t_seq, t_raw, t_raw_seq, g, index = batch.get(
                    'x'), batch.get('y'), batch.get('c'), batch.get(
                        't'), batch.get('t_seq'), batch.get(
                            't_raw'), batch.get('t_raw_seq'), batch.get(
                                'g'), batch.get('index')
                event_times[idx:idx + len(t)] = t
                # KaplanMeierEstimator expects True if the event occurred, so we need to invert the censoring indicator
                censored[idx:idx + len(t)] = ~c
                idx += len(t)
        else:
            # need to invert the censoring indicator because KaplanMeierEstimator expects True if the event occurred
            censored = ~censored
            event_times = event_times.to(torch.float)
        km = KaplanMeierEstimator()
        km(censored.to(self.device),
           event_times.to(self.device),
           censoring_dist=censoring_dist,
           device=self.device)
        # adjust the estimate to avoid division by 0
        km.km_est = torch.where(km.km_est < min_estimate, min_estimate,
                                km.km_est)
        return km

    def cross_entropy_loss(self, y_pred, y):
        """Computes the cross entropy loss."""
        return nn.CrossEntropyLoss()(y_pred, y)

    @staticmethod
    def discrete_survival_pmf(y_pred):
        """Generates probabilities for every possible event time including no event."""
        # y_pred is a logit sequence of shape (batch_size, time_bins)
        # want to compute the probabilities for the label sequences
        # (1, 1, 1, ...., 1), (0, 1, 1, ..., 1), (0, 0, 1, ..., 1), ..., (0, 0, 0, ..., 1)
        # where (0, 0, 0, ..., 1) means never happens or later (the last time bin is special)
        # which is how the logits are already ordered
        # reverse the order and take the partial sums
        log_unnormalized_pmf = torch.cumsum(torch.flip(y_pred, dims=[1]),
                                            dim=1)
        # flip the order back
        log_unnormalized_pmf = torch.flip(log_unnormalized_pmf, dims=[1])
        # get probabilities in log space for numerical stability
        log_pmf = log_unnormalized_pmf - torch.logsumexp(
            log_unnormalized_pmf, dim=1, keepdim=True)
        return log_pmf.exp()

    @staticmethod
    def discrete_survival_function(y_pred):
        """Generates survival probabilities for every possible event time."""
        pmf = Trainer.discrete_survival_pmf(y_pred)
        # the survival function is just 1 minus the cumulative sum of the pmf
        return 1 - torch.cumsum(pmf, dim=1)

    @staticmethod
    def discrete_survival_cdf(y_pred):
        """Generates cumulative distribution function for every possible event time."""
        pmf = Trainer.discrete_survival_pmf(y_pred)
        # the cdf is just the cumulative sum of the pmf
        return torch.cumsum(pmf, dim=1)

    @staticmethod
    def cdf_to_pmf(cdf):
        """Converts a cumulative distribution function to a probability mass function."""
        # cdf is a sequence of probabilities for every possible event time
        # so we just take the difference between consecutive elements
        pmf = torch.diff(cdf, dim=1, prepend=torch.zeros_like(cdf[:, :1]))
        return pmf

    def discrete_survival_loss_uncensored(self, y_pred, y):
        """This loss function follows https://papers.nips.cc/paper_files/paper/2011/hash/1019c8091693ef5c5f55970346633f92-Abstract.html
        for uncensored data."""
        # y_pred is a logit sequence of shape (batch_size, time_bins)
        # y is a binary encoding of the event times of shape (batch_size, time_bins)
        log_numerator = torch.sum(y_pred * y, dim=1)
        # denominator accounts for the whole space of possible event times, which means we're going
        # to consider labels (1, 1, 1, ...., 1), (0, 1, 1, ..., 1), (0, 0, 1, ..., 1), ..., (0, 0, 0, ..., 1)
        # where (0, 0, 0, ..., 1) means never happens or later (the last time bin is special)
        # implementation-wise, this is just a sequence of partial sums of the logits followed by exponentiating
        # but we have to reverse the order of the logits because torch.cumsum computes the cumulative sum from the beginning of the sequence
        partial_sums = torch.cumsum(torch.flip(y_pred, dims=[1]), dim=1)
        log_denominator = torch.logsumexp(partial_sums, dim=1)
        losses = -(log_numerator - log_denominator)
        return losses

    def discrete_survival_loss_censored(self, y_pred, y):
        """This loss function follows https://papers.nips.cc/paper_files/paper/2011/hash/1019c8091693ef5c5f55970346633f92-Abstract.html
        for censored data."""
        # y_pred is a logit sequence of shape (batch_size, time_bins)
        # y is a binary encoding of the (censored) event times of shape (batch_size, time_bins)
        # so this is going to be a run of 0s before the censoring event and then 1s after the censoring event
        # so we want all cumulative sums from the end of the sequence up to the censoring event plus the no event case (all 0s)
        flipped_logits = torch.flip(y_pred, dims=[1])
        partial_sums = torch.cumsum(flipped_logits, dim=1)
        # flip the order back
        partial_sums = torch.flip(partial_sums, dims=[1])
        # censoring occurs at different times per example so the easiest thing to do make this a uniform computation per example
        # is to replace pre-censoring time bins with -inf, because then exp evaluates to 0 at these points and then they don't contribute to the sum
        # recall y is a binary encoding of the event times and will be 0s up to the censoring time and 1s after
        # so we can just invert this to find the places where we need to replace with -inf
        y = y.to(torch.bool)
        partial_sums_masked = torch.where(~y, torch.tensor(-float('inf')),
                                          partial_sums)
        # get the log numerator
        log_numerator = torch.logsumexp(partial_sums_masked, dim=1)
        # get the log denominator
        log_denominator = torch.logsumexp(partial_sums, dim=1)
        losses = -(log_numerator - log_denominator)
        return losses

    def group_entropy_vals(self, router_logits):
        grouper_output = torch.softmax(router_logits, dim=1)
        # compute entropy per row
        avg_instance_ent = -torch.sum(
            grouper_output * torch.log(grouper_output + 1e-8),
            dim=-1).mean(dim=0)
        avg_group_prob = grouper_output.mean(dim=0)
        avg_group_ent = -torch.sum(
            avg_group_prob * torch.log(avg_group_prob + 1e-8), dim=-1)
        return avg_instance_ent, avg_group_ent

    def load_balancing_loss(self, router_logits):
        """Alternative to avg_group_ent in group_entropy_vals. This is the loss
        function used for mixture of experts models."""
        if self.train_config.mixture_topk == self.train_config.mixture_components:
            grouper_output = torch.softmax(router_logits, dim=1)
            # compute an entropy-like surrogate of the group probabilities
            avg_routing_prob = grouper_output.mean(dim=0)
            num_components = grouper_output.shape[1]
            loss = num_components * torch.sum(
                avg_routing_prob * avg_routing_prob)
            return loss
        else:
            # load balance over topk components
            num_experts = router_logits.shape[1]
            batch_size = router_logits.shape[0]
            topk = self.train_config.mixture_topk
            _, topk_indices = torch.topk(router_logits, topk, dim=1)
            # fill in a 1 where the expert was selected
            experts_chosen = torch.zeros_like(router_logits).scatter_(
                1, topk_indices, 1)
            # f_i: fraction of examples assigned to expert i
            fraction_routed = experts_chosen.sum(dim=0) / batch_size
            # P_i: average probability assigned to expert i
            # NB: this is over all experts, not just the topk
            avg_prob = torch.softmax(router_logits, dim=1).mean(dim=0)
            return num_experts * torch.sum(fraction_routed * avg_prob)

    def loss_function(self,
                      y_pred,
                      y,
                      c,
                      grouper_output=None,
                      router_logits=None,
                      step=None):
        """Computes the loss function, where c is the censoring indicator.
        Optionally adds the group loss if grouper_output is provided.
        """
        uncensored_y_pred = y_pred[~c]
        uncensored_y = y[~c]
        censored_y_pred = y_pred[c]
        censored_y = y[c]
        auxiliary_stats = {}
        grouper_stats = {}
        if self.train_config.loss == "CrossEntropy":
            loss = self.cross_entropy_loss(y_pred, y)
        elif self.train_config.loss == "DiscreteSurvival":
            uncensored_losses = self.discrete_survival_loss_uncensored(
                uncensored_y_pred, uncensored_y)
            censored_losses = self.discrete_survival_loss_censored(
                censored_y_pred, censored_y)
            # weight individual examples equally
            uncensored_loss = uncensored_losses.sum() / y.size(0)
            censored_loss = censored_losses.sum() / y.size(0)
            auxiliary_stats['uncensored_losses'] = uncensored_losses
            auxiliary_stats['censored_losses'] = censored_losses
            loss = uncensored_loss + censored_loss
            if grouper_output is not None:
                avg_instance_ent, avg_group_ent, router_loss = torch.tensor(
                    0.0, device=self.device), torch.tensor(
                        0.0,
                        device=self.device), torch.tensor(0.0,
                                                          device=self.device)
                if router_logits is not None:
                    avg_instance_ent, avg_group_ent = self.group_entropy_vals(
                        router_logits)
                    router_loss = self.load_balancing_loss(router_logits)

                # add optional entropy regularization terms
                # to encourage sharp group assignments AND use of all groups
                if self.train_config.instance_ent_lambda > 0:
                    # minimizing the entropy of the group assignments per instance means sharper group assignments
                    loss += self.train_config.instance_ent_lambda * avg_instance_ent
                if self.train_config.group_ent_lambda > 0:
                    # maximizing the entropy of the group assignments overall means using all groups
                    group_ent_lambda = self.train_config.group_ent_lambda
                    # maybe anneal the group_ent_lambda as function of the step
                    if self.train_config.cosine_anneal_group_ent_lambda and step is not None:
                        lambda_multiplier = self.get_lambda_multiplier(step)
                        group_ent_lambda *= lambda_multiplier
                    loss += group_ent_lambda * router_loss
                grouper_stats['avg_instance_ent'] = avg_instance_ent.item()
                grouper_stats['avg_group_ent'] = avg_group_ent.item()
        return loss, grouper_stats, auxiliary_stats

    def recover_loss_order(self, censored, aux_stats):
        # uncensored_losses and censored_losses are now out of order relative to the censored mask
        # so we need to actually recover the original order using torch.nonzero
        censored_idxs = torch.nonzero(censored, as_tuple=False).squeeze(1)
        uncensored_idxs = torch.nonzero(~censored, as_tuple=False).squeeze(1)
        # put the losses back in the original order
        losses = torch.zeros_like(censored,
                                  dtype=torch.float,
                                  device=self.device)
        losses[censored_idxs] = aux_stats['censored_losses'].detach()
        losses[uncensored_idxs] = aux_stats['uncensored_losses'].detach()
        return losses

    def predict(self, y_pred, threshold=0.5, return_predicted_time=False):
        """Translate a logit sequence into a sequence of predictions at every point in time.
        """
        cumulative_probs = self.discrete_survival_cdf(y_pred)
        pred_seq = cumulative_probs >= threshold
        if return_predicted_time:
            # compute the predicted time
            # the predicted time is the time bin at which the cumulative probability first exceeds the threshold
            # if no time bin exceeds the threshold, then we predict the last time bin
            predicted_idx = torch.argmax(pred_seq.float(), dim=1)
            predicted_time = self.time_bins[predicted_idx]
            return pred_seq, predicted_time
        return pred_seq

    def predict_with_cost_function(self, y_pred, error='absolute_error'):
        """Predict using equation 8 from Yu (2011)."""
        # matrix of shape time_bins x time_bins to precompute costs
        idxs = torch.arange(self.time_bins.size(0))
        idx_pairs = torch.cartesian_prod(idxs, idxs)
        time_pairs = self.time_bins[idx_pairs]
        # compute the costs
        errors_flattened = self.compute_regression_error(time_pairs[:, 0],
                                                         time_pairs[:, 1],
                                                         error=error,
                                                         average=False)
        costs = errors_flattened.view(self.time_bins.size(0),
                                      self.time_bins.size(0))
        # compute the pmf
        pmf = self.discrete_survival_pmf(y_pred)
        weighted_preds = torch.matmul(pmf, costs)
        # take the argmin
        predicted_idx = torch.argmin(weighted_preds, dim=1)
        # decode the predicted time
        predicted_time = self.time_bins[predicted_idx]
        return predicted_time

    def predict_expected_time(self, y_pred):
        """Predict the expected time of the event using the survival function."""
        # compute the survival function
        s = self.discrete_survival_function(y_pred)
        # compute the expected time
        # E[T] = sum(t * P(T=t))
        expected_time = torch.sum(self.time_bins * s, dim=1)
        return expected_time

    def uncensored_mask(self, y, c):
        """Takes as input y, a batch of label sequences (0s, 1s) and c, a batch
        of boolean censoring indicators and returns a binary mask of shape (batch, time_bins)
        indicating if the status of the event at that point in time is known to us."""
        # define a mask for valid time bins
        # if uncensored, then all time bins are valid
        # if censored, then only the time bins up to the censoring time are valid
        uncensored_mask = torch.ones_like(y, dtype=torch.bool)
        # censored ys are 0s up to the censoring time and 1s after
        # this means if we invert this, we get 1s up to the censoring time and 0s after, which is exactly what we want
        y_censored_inverted = ~(y[c].to(torch.bool))
        uncensored_mask[c] = y_censored_inverted
        return uncensored_mask

    def accuracy_average(self, correct, uncensored_indicator, times):
        """Computes accuracy given a binary mask indicating if the status of the event at that point in time is known to us."""
        # average over the batch dimension to get the accuracy at each point in time
        if self.train_config.adjust_eval and self.train_config.ipcw:
            # accounting for censoring
            # adjust by inverse censoring probabilities
            # first assume that censoring or the event occurs after the time bin
            correct_future_event = correct / self.censoring_probs[None, :]
            # next assume that the event occurred before the time bin and adjust based on whether the event was observed
            # if we predict past the last time bin, we need to use the last bin's censoring probability
            # -2 because the last bin is the "no event" bin
            times_clamped = torch.clamp(times,
                                        max=self.censoring_km.time.max())
            if torch.any(times > self.censoring_km.time.max()):
                print(
                    "Warning: some event times exceed the maximum time in the censoring KM estimate. Clamping to max."
                )
            correct_past_event = (
                correct / self.censoring_km.predict(times_clamped).to(
                    self.device)[:, None]) * uncensored_indicator[:, None]
            # choose which term to use
            time_mask = times_clamped[:, None] <= self.time_bins[
                None, :]  # shape (batch_size, time_bins)
            correct = torch.where(time_mask, correct_past_event,
                                  correct_future_event)
            correct_by_time = correct.sum(dim=0)
            total = correct.shape[0]
        elif self.train_config.adjust_eval and (
                self.train_config.impute
                or self.train_config.administrative_censoring):
            correct_by_time = correct.sum(dim=0)
            # if we've imputed labels, then use the total number of examples
            total = correct.shape[0]
        else:
            time_mask = times[:, None] <= self.time_bins[
                None, :]  # shape (batch_size, time_bins)
            # if the instance is uncensored, then we should consider the correct indicator
            uncensored_mask = torch.where(uncensored_indicator[:, None], True,
                                          time_mask)
            correct_by_time = torch.where(uncensored_mask, correct,
                                          0).sum(dim=0)
            total = uncensored_mask.sum(dim=0)  # shape (time_bins)

        return correct_by_time.float() / total

    def estimate_y_given_c(self, y, y_pred, quantize=False):
        """NB: y is a sequence of 0s and 1s, and for censored instances, the 1s
        begin at the censoring time. Consuming functions should be careful to
        only use imputation for the censored instances/time bins."""
        # impute the missing labels for y_t
        s = self.discrete_survival_function(y_pred)
        c_bin = y.argmax(dim=1)  # shape (batch_size,)
        # impute the missing labels conditioned on the censoring time
        s_c = s[torch.arange(s.shape[0]), c_bin]  # shape (batch_size,)
        y_est = (s_c[:, None] - s) / s_c[:, None]
        if quantize:
            # quantize back to a 0/1 label
            y_est = torch.where(y_est >= 0.5, 1, 0)
        return y_est

    def compute_accuracy_at_ts(
        self,
        y_pred,
        y,
        c,
        t,
        threshold=0.5,
        average=True,
    ):
        """Computes accuracy at each point in time."""
        y_pred = self.predict(y_pred, threshold)
        # uncensored_mask = self.uncensored_mask(y, c)

        if self.train_config.adjust_eval and self.train_config.impute:
            y_est = self.estimate_y_given_c(y, y_pred, quantize=True)
            # backfill y where it was censored
            y = torch.where(c[:, None] & y.to(bool), y_est, y)

        correct = (y_pred == y)

        if average:
            return self.accuracy_average(correct, ~c, t)

        return correct

    def brier_average(self, squared_diffs, uncensored_indicator, times):
        """Computes the Brier score given a binary mask indicating if the status of the event at that point in time is known to us."""
        # average over the batch dimension to get the Brier score at each point in time
        if self.train_config.adjust_eval and self.train_config.ipcw:
            # accounting for censoring
            # adjust by inverse censoring probabilities
            # first assume that censoring or the event occurs after the time bin
            metric_future_event = squared_diffs / self.censoring_probs[None, :]
            # next assume that the event occurred before the time bin and adjust based on whether the event was observed
            # if we predict past the last time bin, we need to use the last bin's censoring probability
            # -2 because the last bin is the "no event" bin
            times_clamped = torch.clamp(times,
                                        max=self.censoring_km.time.max())
            metric_past_event = (
                squared_diffs / self.censoring_km.predict(times_clamped).to(
                    self.device)[:, None]) * uncensored_indicator[:, None]
            # choose which term to use
            time_mask = times_clamped[:, None] <= self.time_bins[
                None, :]  # shape (batch_size, time_bins)
            metric = torch.where(time_mask, metric_past_event,
                                 metric_future_event)
            metric_by_time = metric.sum(dim=0)
            total = metric.shape[0]
        elif self.train_config.adjust_eval and (
                self.train_config.impute
                or self.train_config.administrative_censoring):
            metric_by_time = squared_diffs.sum(dim=0)
            # if we've imputed labels, then use the total number of examples
            total = squared_diffs.shape[0]
        else:
            time_mask = times[:, None] <= self.time_bins[
                None, :]  # shape (batch_size, time_bins)
            # if the instance is uncensored, then we should consider the correct indicator
            uncensored_mask = torch.where(uncensored_indicator[:, None], True,
                                          time_mask)
            metric_by_time = torch.where(uncensored_mask, squared_diffs,
                                         0).sum(dim=0)
            total = uncensored_mask.sum(dim=0)  # shape (time_bins)

        return metric_by_time.float() / total

    def compute_brier_score_at_ts(self, y_pred, y, c, t, average=True):
        """Computes the Brier score at each point in time."""
        y_pred_cdf = self.discrete_survival_cdf(y_pred)
        # uncensored_mask = self.uncensored_mask(y, c)

        if self.train_config.adjust_eval and self.train_config.impute:
            y_est = self.estimate_y_given_c(y, y_pred, quantize=False)
            # backfill y where it was censored
            y = torch.where(c[:, None] & y.to(bool), y_est, y)

        squared_diffs = (y_pred_cdf - y)**2

        if average:
            # average over the batch dimension to get the Brier score at each point in time
            # accounting for censoring
            return self.brier_average(squared_diffs, ~c, t)
        return squared_diffs

    def ece_aggregate(self,
                      y_pred_by_qtile,
                      y_by_qtile,
                      y_pred_qtiles_one_hot,
                      event_indicator,
                      times,
                      return_decile_avgs=False):
        """Computes the calibration error.
        y_pred_by_qtile, y_by_qtile, and y_pred_qtiles_one_hot: shape (batch_size, time_bins, num_bins)
        """
        # handle two use cases: when times are one dimensional and when they are two-dimensional
        if times.dim() == 1:
            # times is a 1D tensor of shape (batch_size,) expand it to (batch_size, time_bins)
            times = times[:, None].expand(-1, self.time_bins.size(0))
        if event_indicator.dim() == 1:
            # event_indicator is a 1D tensor of shape (batch_size,) expand it to (batch_size, time_bins)
            event_indicator = event_indicator[:, None].expand(
                -1, self.time_bins.size(0))

        # no need to adjust predictions by inverse censoring probabilities because they're "observed"
        y_pred_by_qtile_sum = y_pred_by_qtile.sum(
            dim=0)  # shape (time_bins, num_bins)
        # get the count of the probabilities for each quantile
        per_qtile_count = y_pred_qtiles_one_hot.sum(
            dim=0)  # shape (time_bins, num_bins)
        # get the mean of the probabilities for each quantile
        # guard against division by 0 - though a better way to do this is to replace the 0s first in per_qtile_count
        y_pred_by_qtile_mean = torch.where(
            per_qtile_count > 0, y_pred_by_qtile_sum / per_qtile_count,
            0)  # shape (time_bins, num_bins)

        # IPCW-adjusted empirical probabilities
        if self.train_config.adjust_eval and self.train_config.ipcw:
            # identify events and survivors for each time bin t
            # for an event at time t, their true time T_i must be <= t and they must have an event (delta_i=1)
            event_mask = (times <= self.time_bins.unsqueeze(0)
                          ) & event_indicator  # shape (batch_size, time_bins)
            # for a survivor at time t, their true time T_i must be > t
            survivor_mask = times > self.time_bins.unsqueeze(
                0)  # shape (batch_size, time_bins)

            # calculate IPCW weights for events and survivors
            # weight for events: 1 / G(T_i)
            # if we predict past the last time bin, we need to use the last bin's censoring probability
            # -2 because the last bin is the "no event" bin
            # times_clamped = torch.clamp(times, max=self.time_bins[-2])
            times_clamped = torch.clamp(times,
                                        max=self.censoring_km.time.max())
            weight_event = 1.0 / self.censoring_km.predict(
                times_clamped)  # shape (batch_size, time_bins)
            # weight for survivors: 1 / G(t)
            weight_survivor = 1.0 / self.censoring_probs  # self.censoring_probs is G(t) shape (time_bins,)

            # build the numerator and denominator for the empirical probability
            # numerator: sum of weights for people who had an event in each bin
            # we multiply the one-hot bin assignment by the event weight and mask
            numerator = (y_pred_qtiles_one_hot * weight_event.unsqueeze(2) *
                         event_mask.unsqueeze(2)).sum(dim=0)

            # denominator: sum of weights for all evaluable people (events + survivors) in each bin
            denominator_events = numerator  # re-use the numerator calculation
            denominator_survivors = (y_pred_qtiles_one_hot *
                                     weight_survivor.view(1, -1, 1) *
                                     survivor_mask.unsqueeze(2)).sum(dim=0)
            denominator = denominator_events + denominator_survivors

            # calculate the final empirical probability
            y_by_qtile_mean = torch.where(denominator > 0,
                                          numerator / denominator, 0.0)
        elif self.train_config.adjust_eval and (
                self.train_config.impute
                or self.train_config.administrative_censoring):
            # adjust labels by imputation
            # or in the case of administrative censoring, we can treat all censored instances as uncensored because we know that they survived up to the last time bin
            y_by_qtile_sum = y_by_qtile.sum(dim=0)
            # guard against division by 0
            y_by_qtile_mean = torch.where(per_qtile_count > 0,
                                          y_by_qtile_sum / per_qtile_count,
                                          0)  # shape (time_bins, num_bins)
        else:
            # compute the biased average of the labels for each quantile
            time_mask = times <= self.time_bins[
                None, :]  # shape (batch_size, time_bins), 1 if time is before or equal to the time bin
            uncensored_mask = torch.where(
                event_indicator, True,
                time_mask)  # shape (batch_size, time_bins)
            y_by_qtile_masked = torch.where(uncensored_mask[:, :, None],
                                            y_by_qtile, 0)
            y_by_qtile_sum = y_by_qtile_masked.sum(
                dim=0)  # shape (time_bins, num_bins)
            # guard against division by 0
            y_by_qtile_mean = torch.where(per_qtile_count > 0,
                                          y_by_qtile_sum / per_qtile_count, 0)

        diff = torch.abs(y_pred_by_qtile_mean -
                         y_by_qtile_mean)  # shape (time_bins, num_bins)
        # weighted average over the bin dimension to get the expected calibration error at each point in time
        ece_per_time = (diff * per_qtile_count).sum(
            dim=1) / per_qtile_count.sum(dim=1)  # shape (time_bins)
        if return_decile_avgs:
            # return the decile averages as well
            return ece_per_time, y_pred_by_qtile_mean, y_by_qtile_mean, per_qtile_count
        return ece_per_time

    def compute_ece_equal_mass_at_ts(self,
                                     y_pred,
                                     y,
                                     c,
                                     t,
                                     return_avgs=False,
                                     num_bins=10):
        """Computes the calibration error using equal mass binning."""
        y_pred_cdf = self.discrete_survival_cdf(y_pred)

        # batch_size is unlikely to be divisible by num_bins, so do some indexing
        batch_size, time_bins = y_pred.size()
        # determine bin sizes
        base_size = batch_size // num_bins
        remainder = batch_size % num_bins
        bin_sizes = torch.full((num_bins, ), base_size, device=self.device)
        bin_sizes[:remainder] += 1  # shape (num_bins,)

        # assign each example to a bin
        bin_ids = torch.repeat_interleave(
            torch.arange(num_bins, device=self.device),
            bin_sizes)  # shape (batch_size,)
        bin_ids = torch.broadcast_to(
            bin_ids[:, None],
            (batch_size, time_bins))  # shape (batch_size, time_bins)

        # argsort the predicted probabilities for each time bin
        sorted_indices = torch.argsort(
            y_pred_cdf, dim=0, stable=True)  # shape (batch_size, time_bins)
        col_indices = torch.arange(time_bins)[None, :]
        y_preds_sorted = y_pred_cdf[
            sorted_indices, col_indices]  # shape (batch_size, time_bins)
        y_labels_sorted = y[sorted_indices,
                            col_indices]  # shape (batch_size, time_bins)

        # explode the y_preds_sorted and y_labels_sorted to a 1-hot representation to recycle the ece_aggregate function
        y_pred_qtiles_one_hot = torch.nn.functional.one_hot(
            bin_ids, num_classes=num_bins).float(
            )  # shape (batch_size, time_bins, num_bins)
        y_pred_by_qtile = y_preds_sorted[:, :,
                                         None] * y_pred_qtiles_one_hot  # shape (batch_size, time_bins, num_bins)
        y_by_qtile = y_labels_sorted[:, :,
                                     None] * y_pred_qtiles_one_hot  # shape (batch_size, time_bins, num_bins)
        # expand and reorder the times and event indicators
        times = t[:, None].expand(batch_size,
                                  time_bins)  # shape (batch_size, time_bins)
        event_indicator = ~c[:, None].expand(
            batch_size, time_bins)  # shape (batch_size, time_bins)
        times = times[sorted_indices,
                      col_indices]  # shape (batch_size, time_bins)
        event_indicator = event_indicator[
            sorted_indices, col_indices]  # shape (batch_size, time_bins)

        return self.ece_aggregate(y_pred_by_qtile,
                                  y_by_qtile,
                                  y_pred_qtiles_one_hot,
                                  event_indicator,
                                  times,
                                  return_decile_avgs=return_avgs)

    def compute_ece_at_ts(self,
                          y_pred,
                          y,
                          c,
                          t,
                          num_bins=10,
                          return_decile_avgs=False):
        """Computes the calibration error at each point in time."""
        y_pred_cdf = self.discrete_survival_cdf(y_pred)
        # break this out into quantiles. shape (batch_size, time_bins, num_bins)
        endpoint = 1 - (1 / num_bins)
        qtiles = torch.linspace(0, endpoint,
                                num_bins).to(self.device)  # shape (num_bins)
        y_pred_qtiles = torch.bucketize(
            y_pred_cdf, qtiles,
            right=True) - 1  # minus 1 to get 0-indexed buckets
        # create a 1-hot encoding of the quantiles
        y_pred_qtiles_one_hot = torch.nn.functional.one_hot(
            y_pred_qtiles, num_classes=len(
                qtiles)).float()  # shape (batch_size, time_bins, num_bins)
        # fill in the probabilities
        y_pred_by_qtile = y_pred_qtiles_one_hot * y_pred_cdf.unsqueeze(
            -1)  # shape (batch_size, time_bins, num_bins)

        if self.train_config.adjust_eval and self.train_config.impute:
            y_est = self.estimate_y_given_c(y, y_pred, quantize=False)
            # backfill y where it was censored
            y = torch.where(c[:, None] & y.to(bool), y_est, y)

        # get the rate of occurrence for each quantile
        y_by_qtile = y_pred_qtiles_one_hot * y.unsqueeze(
            -1)  # shape (batch_size, time_bins, num_bins)

        # average over the batch dimension to get the ECE at each point in time
        # accounting for censoring
        return self.ece_aggregate(y_pred_by_qtile, y_by_qtile,
                                  y_pred_qtiles_one_hot, ~c, t,
                                  return_decile_avgs)

    def mean_residual_life(self, y_pred, y):
        """Estimate the average residual life given the predicted survival
        function and censoring time. y is a batch of sequence of 0s and 1s, and
        c is a boolean indicating if the example is censored.
        
        NB: this function computes the MRL for all examples, so downstream
        functions should take care to only use the MRL for the censored examples.
        """
        # get the survival function
        s = self.discrete_survival_function(y_pred)
        # compute the mean residual life
        # the integral of s(t) from c to infinity can be approximated by the sum of s(t) from c to the end of the sequence
        # y is a sequence of 0s and 1s, so just multiply elementwise and sum to compute this
        numerator = (s * y).sum(dim=1)  # shape (batch_size,)
        c_bin = y.argmax(dim=1)  # shape (batch_size,)
        denominator = s[torch.arange(s.shape[0]), c_bin]  # shape (batch_size,)
        # guard against division by 0
        denominator = torch.where(denominator > 0, denominator, 1e-8)
        # compute the mean residual life
        mean_residual_life = numerator / denominator
        return mean_residual_life

    def compute_regression_error(self,
                                 t_pred,
                                 t,
                                 y_pred=None,
                                 y=None,
                                 c=None,
                                 error='absolute_error',
                                 average=True):
        """Compute regression-like error functions given a time point estimate. error options are:
        {'absolute_error', 'squared_error', 'relative_error', 'log_error'}.
        
        If c is provided, then some of the event times are censored and we need to exclude them from the error calculation.
        """
        # in the case of censoring, we don't always observe t, so we can impute it with the mean residual life
        if (c is not None
            ) and self.train_config.adjust_eval and self.train_config.impute:
            # actually this won't always work
            # MRL treats all bins as the same size, which won't work if you have variable bin sizes
            # mrl = self.mean_residual_life(y_pred, y)
            # note that in the case of censoring, t is the censoring time
            # we add to that the expected residual life after censoring
            # t = torch.where(c, t + mrl, t)
            # impute y_t
            y_est = self.estimate_y_given_c(y, y_pred, quantize=True)
            # find the time bin at which the event occurred
            t_bin = y_est.argmax(dim=1)  # shape (batch_size,)
            # translate the time bin to the time
            t_est = self.time_bins[t_bin]  # shape (batch_size,)
            # backfill t where it was censored
            t = torch.where(c, t_est, t)

        if error == 'absolute_error':
            errors = torch.abs(t_pred - t)
        elif error == 'squared_error':
            errors = (t_pred - t)**2
        elif error == 'relative_error':
            rel_error = torch.abs((t_pred - t) / t_pred)
            errors = torch.min(rel_error, torch.ones_like(rel_error))
        elif error == 'log_error':
            errors = torch.abs(torch.log(t_pred) - torch.log(t))
        else:
            raise ValueError(f"Unknown error: {error}")

        if (c is not None
            ) and self.train_config.adjust_eval and self.train_config.ipcw:
            # accounting for censoring
            # adjust by inverse censoring probabilities
            # get censoring probabilities at the predicted time
            # if we predict past the last time bin, we need to use the last bin's censoring probability
            # -2 because the last bin is the "no event" bin
            # t_pred_clamped = torch.clamp(t_pred, max=self.time_bins[-2])
            t_pred_clamped = torch.clamp(t_pred,
                                         max=self.censoring_km.time.max())
            censoring_probs_at_pred = self.censoring_km.predict(
                t_pred_clamped).to(self.device)
            # adjust the errors by the censoring probabilities, where censored
            errors = torch.where(c, errors / censoring_probs_at_pred, errors)

        if average:
            # filter out the censored examples if c was provided
            if (c is not None
                ) and self.train_config.adjust_eval and self.train_config.ipcw:
                # only some errors valid
                errors = errors[~c]
            elif (
                    c is not None
            ) and self.train_config.adjust_eval and self.train_config.impute:
                # all errors valid
                errors = errors
            elif (c is not None):
                # only some errors valid
                errors = errors[~c]

            return errors.mean()
        return errors

    def log_metrics(
        self,
        subset,
        error_name,
        loss,
        acc_at_ts,
        brier_score_at_ts,
        ece_at_ts,
        acc,
        brier_score,
        ece,
        absolute_error,
        optimized_absolute_error,
        step,
        pctiles=[0.25, 0.5, 0.75],
        ece_equal_mass_at_ts=None,
        ece_equal_mass=None,
        grouper_stats=None,
        concordance_score=None,
        extra_keys=None,
        use_tqdm=True,
    ):
        """Logs metrics to wandb and print to tqdm"""
        # report the accuracy at the middle of the time bins
        # median_time_idx = int(self.train_config.time_bins * 0.5)
        error_name = error_name.replace('_', ' ').title()
        ece_equal_mass_string = ''
        if ece_equal_mass is not None:
            ece_equal_mass_string = f'ECEEM: {ece_equal_mass:.4f}, '
        if use_tqdm:
            tqdm.write(
                f"[{step}] {subset.title()} Metrics - Loss: {loss:.4f}, Acc: {acc:.4f}, Brier: {brier_score:.4f}, ECE: {ece:.4f}, {ece_equal_mass_string}{error_name}: {absolute_error:,.4f}, {error_name} Optimized: {optimized_absolute_error:,.4f}"
            )
        # log at all specified percentiles
        pctile_idxs = [int(p * self.train_config.time_bins) for p in pctiles]
        pctiles = [f'{int(p * 100)}' for p in pctiles]  # format as percentages
        select_acc_at_ts = {
            f'{subset}_acc@{p}th': acc_at_ts[idx]
            for p, idx in zip(pctiles, pctile_idxs)
        }
        select_brier_score_at_ts = {
            f'{subset}_brier@{p}th': brier_score_at_ts[idx]
            for p, idx in zip(pctiles, pctile_idxs)
        }
        select_ece_at_ts = {
            f'{subset}_ece@{p}th': ece_at_ts[idx]
            for p, idx in zip(pctiles, pctile_idxs)
        }
        if ece_equal_mass_at_ts is not None:
            select_ece_equal_mass_at_ts = {
                f'{subset}_ece_equal_mass@{p}th': ece_equal_mass_at_ts[idx]
                for p, idx in zip(pctiles, pctile_idxs)
            }
        # adjust the key names for the grouper_stats
        subset_grouper_stats = {
            f'{subset}_{k}': v
            for k, v in grouper_stats.items()
        }

        # log to wandb
        log_dict = {
            **select_acc_at_ts,
            **select_brier_score_at_ts,
            **select_ece_at_ts,
            **subset_grouper_stats,
            f'{subset}_acc': acc,
            f'{subset}_brier': brier_score,
            f'{subset}_ece': ece,
            f'{subset}_loss': loss,
            f'{subset}_absolute_error': absolute_error,
            f'{subset}_absolute_error_optimized': optimized_absolute_error,
        }
        if ece_equal_mass_at_ts is not None:
            log_dict.update(select_ece_equal_mass_at_ts)
        if ece_equal_mass is not None:
            log_dict[f'{subset}_ece_equal_mass'] = ece_equal_mass
        if concordance_score is not None:
            log_dict[f'{subset}_concordance'] = concordance_score

        if extra_keys is not None:
            # update the log dict while modifying the key name to include the subset
            extra_keys = {f'{subset}_{k}': v for k, v in extra_keys.items()}
            log_dict.update(extra_keys)

        wandb.log(log_dict, step=step)

    def load_best_model(self):
        # load the best model
        model_dir = os.path.join(self.model_dir, 'best')
        checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
        self.model.load_state_dict(checkpoint['model'])
        completed_steps = checkpoint['completed_steps']
        print(f"Loaded model from {model_dir} at step {completed_steps}")

    def train(self,
              train_loader,
              val_loader,
              error='absolute_error',
              pctiles=[0.25, 0.5, 0.75]):
        """Trains the model for one epoch."""
        # compute the Kaplan-Meier estimate of the censoring distribution
        # so that we can adjust lots of quantities for censoring (MSE, Brier score, group-based loss, etc.)
        self.add_km(train_loader)

        # set up the cosine annealing scheduler, if parameters are provided
        if self.train_config.cosine_anneal_group_ent_lambda:
            hold_steps = self.train_config.cosine_anneal_hold_epochs * len(
                train_loader)
            anneal_steps = self.train_config.cosine_anneal_epochs * len(
                train_loader)
            min_val = self.train_config.cosine_anneal_min_lambda_multiplier
            self.get_lambda_multiplier = partial(get_lambda_multiplier,
                                                 hold_steps=hold_steps,
                                                 anneal_steps=anneal_steps,
                                                 min_val=min_val)

        self.model.train()
        step = 0
        epoch_steps = len(train_loader)
        best_monitored_val_metric = float(
            'inf') if self.train_config.minimize_val_metric else -float('inf')
        early_stopping_patience = self.train_config.early_stopping_patience
        for epoch in range(self.train_config.epochs):
            for idx, batch in tqdm(enumerate(train_loader),
                                   desc=f"Epoch {epoch}",
                                   total=len(train_loader)):
                x, y, c, t, t_seq, t_raw, t_raw_seq, g, index = batch.get(
                    'x'), batch.get('y'), batch.get('c'), batch.get(
                        't'), batch.get('t_seq'), batch.get(
                            't_raw'), batch.get('t_raw_seq'), batch.get(
                                'g'), batch.get('index')
                model_output = self.model(x.to(self.device))
                # we've standardized the model_out return dict, so just pull the relevant keys
                y_pred = model_output.get('final_logits')
                grouper_output = model_output.get('topk_expert_scores')
                router_logits = model_output.get('expert_scores')
                hidden_states = model_output.get('model_out')
                final_hidden_states = model_output.get('final_hidden_states')

                loss, grouper_stats, aux_stats = self.loss_function(
                    y_pred,
                    t_seq.to(self.device),
                    c.to(self.device),
                    grouper_output,
                    router_logits,
                    step=step,
                )

                self.optimizer.zero_grad()
                loss.backward()
                # record the norm of the gradients
                grad_norm = torch.nn.utils.get_total_norm(
                    self.model.parameters(), )
                wandb.log({
                    'grad_norm': grad_norm,
                }, step=step)
                self.optimizer.step()
                # update tqdm description
                if idx % 100 == 0:
                    acc_at_ts = self.compute_accuracy_at_ts(
                        y_pred, t_seq.to(self.device), c.to(self.device),
                        t.to(self.device))
                    brier_score_at_ts = self.compute_brier_score_at_ts(
                        y_pred, t_seq.to(self.device), c.to(self.device),
                        t.to(self.device))
                    ece_at_ts = self.compute_ece_at_ts(y_pred,
                                                       t_seq.to(self.device),
                                                       c.to(self.device),
                                                       t.to(self.device))

                    # aggregrate over time bins
                    acc = torch.mean(acc_at_ts)
                    brier_score = torch.mean(brier_score_at_ts)
                    ece = torch.mean(ece_at_ts)

                    _, point_predictions = self.predict(
                        y_pred, return_predicted_time=True)
                    # now predict using the cost function
                    point_predictions_with_cost_function = self.predict_with_cost_function(
                        y_pred, error=error)
                    absolute_error = self.compute_regression_error(
                        point_predictions,
                        t.to(self.device),
                        y_pred,
                        t_seq.to(self.device),
                        c.to(self.device),
                        error=error)
                    absolute_error_optimized = self.compute_regression_error(
                        point_predictions_with_cost_function,
                        t.to(self.device),
                        y_pred,
                        t_seq.to(self.device),
                        c.to(self.device),
                        error=error)

                    self.log_metrics(
                        subset='train',
                        error_name=error,
                        loss=loss.item(),
                        acc_at_ts=acc_at_ts,
                        brier_score_at_ts=brier_score_at_ts,
                        ece_at_ts=ece_at_ts,
                        acc=acc,
                        brier_score=brier_score,
                        ece=ece,
                        absolute_error=absolute_error,
                        optimized_absolute_error=absolute_error_optimized,
                        step=step,
                        pctiles=pctiles,
                        grouper_stats=grouper_stats)

                    # log some stragglers
                    log_extras = {}
                    # cosine annealing lambda multiplier, if using
                    if self.train_config.cosine_anneal_group_ent_lambda:
                        lambda_multiplier = self.get_lambda_multiplier(step)
                        log_extras[
                            'cosine_anneal_lambda_multiplier'] = lambda_multiplier

                    # monitor the temperature params
                    if self.train_config.model in [
                            Models.FFNetMixture.name,
                    ]:
                        log_extras['logit_temp'] = F.softplus(
                            self.model.logit_temp)

                    wandb.log(log_extras, step=step)

                # end of epoch
                if (step + 1) % epoch_steps == 0:
                    # evaluate the model on the validation set
                    eval_dict = self.evaluate(val_loader,
                                              subset='val',
                                              error=error,
                                              step=step,
                                              pctiles=pctiles)
                    if self.train_config.save_model:
                        monitored_val_metric = eval_dict[
                            self.train_config.monitored_val_metric]
                        better = monitored_val_metric < best_monitored_val_metric if self.train_config.minimize_val_metric else monitored_val_metric > best_monitored_val_metric
                        if better:
                            best_monitored_val_metric = monitored_val_metric
                            # save the model
                            completed_steps = step + 1
                            self.save_model(completed_steps, 'best')
                            early_stopping_patience = self.train_config.early_stopping_patience
                            tqdm.write(
                                f"Saved model after completing {completed_steps} with {self.train_config.monitored_val_metric}: {monitored_val_metric:.4f}"
                            )
                        else:
                            early_stopping_patience -= 1
                            if early_stopping_patience == 0:
                                tqdm.write(
                                    f"Early stopping at step {step} with patience {self.train_config.early_stopping_patience}"
                                )
                                return step + 1

                step += 1
        return step

    def add_km(self, loader=None, event_times=None, censored=None):
        # adjust any small censoring bins because dividing by 0 is bad
        # TODO: do we want to expose 1e-2 as a parameter? Right now it's hardcoded in
        self.censoring_km = self.compute_kaplan_meier(loader=loader,
                                                      event_times=event_times,
                                                      censored=censored,
                                                      censoring_dist=True,
                                                      min_estimate=1e-2)
        # get inverse censoring probabilities for each time bin
        # Kaplan-Meier will complain if you try to predict on points greater than the censoring distribution observed
        # so replace any times greater than the max time
        max_time = self.censoring_km.time.max()
        time_bins = torch.where(self.time_bins > max_time, max_time,
                                self.time_bins)
        self.censoring_probs = self.censoring_km.predict(time_bins)

    def save_model(self, completed_steps, name='last'):
        model_dir = os.path.join(self.model_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        save_dict = {
            'model': self.model.state_dict(),
            'train_config': self.train_config.__dict__,
            'completed_steps': completed_steps
        }
        torch.save(save_dict, os.path.join(model_dir, 'model.pt'))
        # since we won't be resuming training, don't bother saving the optimizer state

    def evaluate(
        self,
        loader,
        subset='val',
        error='absolute_error',
        step=None,
        pctiles=[0.25, 0.5, 0.75],
        log=True,
        train_times=None,
        train_indicators=None,
    ):
        """Evaluates the model on the any subset of the data."""
        self.model.eval()
        with torch.no_grad():
            num_examples = 0
            losses = []
            correct = []
            censored_mask = []
            times = []
            brier_scores = []
            y_preds = []
            t_seqs = []
            y_pred_records = []
            y_pred_expected_time = []
            point_predictions_with_cost_function = []
            absolute_errors = []
            absolute_errors_optimized = []
            grouper_outputs = []
            router_logits = []
            hidden_states = []
            uncensored_losses = []
            censored_losses = []
            survival_functions = []
            for batch in tqdm(loader, desc="Evaluating"):
                x, y, c, t, t_seq, t_raw, t_raw_seq, g, index = batch.get(
                    'x'), batch.get('y'), batch.get('c'), batch.get(
                        't'), batch.get('t_seq'), batch.get(
                            't_raw'), batch.get('t_raw_seq'), batch.get(
                                'g'), batch.get('index')
                model_output = self.model(x.to(self.device))
                # we've standardized the model_out return dict, so just pull the relevant keys
                y_pred_batch = model_output.get('final_logits')
                router_logits_batch = model_output.get('expert_scores')
                hidden_states_batch = model_output.get('model_out')
                hidden_states.append(hidden_states_batch.detach())
                # grouper_outputs.append(grouper_output_batch.detach() if grouper_output_batch is not None else None)
                if router_logits_batch is not None:
                    router_logits.append(router_logits_batch.detach())
                loss, grouper_stats, aux_stats = self.loss_function(
                    y_pred_batch,
                    t_seq.to(self.device),
                    c.to(self.device),
                    step=step,
                )
                uncensored_losses.append(
                    aux_stats['uncensored_losses'].detach())
                censored_losses.append(aux_stats['censored_losses'].detach())
                losses.append(loss.item())
                times.append(t)
                num_examples += x.size(0)
                batch_correct = self.compute_accuracy_at_ts(y_pred_batch,
                                                            t_seq.to(
                                                                self.device),
                                                            c.to(self.device),
                                                            t.to(self.device),
                                                            average=False)
                batch_brier_scores = self.compute_brier_score_at_ts(
                    y_pred_batch,
                    t_seq.to(self.device),
                    c.to(self.device),
                    t.to(self.device),
                    average=False)
                _, batch_point_predictions = self.predict(
                    y_pred_batch, return_predicted_time=True)
                # now predict using the cost function
                batch_point_predictions_with_cost_function = self.predict_with_cost_function(
                    y_pred_batch, error=error)
                batch_expected_time = self.predict_expected_time(y_pred_batch)
                if t_raw is not None:
                    batch_absolute_errors = self.compute_regression_error(
                        batch_point_predictions,
                        t_raw.to(self.device),
                        average=False,
                        error=error)
                    batch_absolute_errors_optimized = self.compute_regression_error(
                        batch_point_predictions_with_cost_function,
                        t_raw.to(self.device),
                        average=False,
                        error=error)
                else:
                    batch_absolute_errors = self.compute_regression_error(
                        batch_point_predictions,
                        t.to(self.device),
                        y_pred_batch,
                        t_seq.to(self.device),
                        c.to(self.device),
                        average=False,
                        error=error)
                    batch_absolute_errors_optimized = self.compute_regression_error(
                        batch_point_predictions_with_cost_function,
                        t.to(self.device),
                        y_pred_batch,
                        t_seq.to(self.device),
                        c.to(self.device),
                        average=False,
                        error=error)

                correct.append(batch_correct)
                censored_mask.append(c.to(self.device))
                brier_scores.append(batch_brier_scores)
                y_preds.append(y_pred_batch.detach())
                t_seqs.append(t_seq.to(self.device))
                y_pred_expected_time.append(batch_expected_time)
                absolute_errors.append(batch_absolute_errors)
                absolute_errors_optimized.append(
                    batch_absolute_errors_optimized)
                y_pred_records.append(
                    self.discrete_survival_cdf(y_pred_batch).detach().cpu())
                point_predictions_with_cost_function.append(
                    batch_point_predictions_with_cost_function)
                if train_times is not None and train_indicators is not None:
                    # compute the survival function for the eval data
                    # this is used to compute the pseudo observations for the regression error
                    batch_survival_function = self.discrete_survival_function(
                        y_pred_batch).detach().cpu()
                    survival_functions.append(batch_survival_function)

        # convert to tensors
        losses = torch.tensor(losses).to(
            self.device)  # keep all computations on the GPU for consistency
        times = torch.concatenate(times).to(self.device)
        correct = torch.concatenate(correct)
        censored_mask = torch.concatenate(censored_mask)
        brier_scores = torch.concatenate(brier_scores)
        y_preds = torch.concatenate(y_preds)
        t_seqs = torch.concatenate(t_seqs)
        y_pred_expected_time = torch.concatenate(y_pred_expected_time)
        absolute_errors = torch.concatenate(absolute_errors)
        absolute_errors_optimized = torch.concatenate(
            absolute_errors_optimized)
        point_predictions_with_cost_function = torch.concatenate(
            point_predictions_with_cost_function)
        y_pred_records = torch.concatenate(y_pred_records)
        hidden_states = torch.cat(hidden_states) if hidden_states else None
        censored_losses = torch.cat(censored_losses)
        uncensored_losses = torch.cat(uncensored_losses)
        if train_times is not None and train_indicators is not None:
            # concatenate the survival functions
            survival_functions = torch.cat(survival_functions).to(self.device)
        eval_metrics_dict = self.compute_eval_metrics(
            y_preds,
            t_seqs,
            losses,
            times,
            correct,
            absolute_errors,
            absolute_errors_optimized,
            brier_scores,
            censored_mask,
            y_pred_expected_time,
            point_predictions_with_cost_function,
            survival_functions=survival_functions,
            train_times=train_times,
            train_indicators=train_indicators,
        )
        avg_loss, acc_at_ts, brier_score_at_ts, ece_at_ts, acc, brier_score, ece, absolute_error, absolute_error_optimized, y_pred_by_decile_mean, y_by_decile_mean, per_decile_count = eval_metrics_dict[
            'avg_loss'], eval_metrics_dict['acc_at_ts'], eval_metrics_dict[
                'brier_score_at_ts'], eval_metrics_dict[
                    'ece_at_ts'], eval_metrics_dict['acc'], eval_metrics_dict[
                        'brier_score'], eval_metrics_dict['ece'], eval_metrics_dict[
                            'absolute_error'], eval_metrics_dict[
                                'absolute_error_optimized'], eval_metrics_dict[
                                    'y_pred_by_decile_mean'], eval_metrics_dict[
                                        'y_by_decile_mean'], eval_metrics_dict[
                                            'per_decile_count']
        ece_equal_mass_at_ts, ece_equal_mass = eval_metrics_dict[
            'ece_equal_mass_at_ts'], eval_metrics_dict['ece_equal_mass']
        y_pred_by_equal_mass_bin_mean, y_by_equal_mass_bin_mean, per_equal_mass_bin_count = eval_metrics_dict[
            'y_pred_by_equal_mass_bin_mean'], eval_metrics_dict[
                'y_by_equal_mass_bin_mean'], eval_metrics_dict[
                    'per_equal_mass_bin_count']
        concordance_score = eval_metrics_dict['concordance_score']
        concordance_ipcw = eval_metrics_dict['c_ipcw']
        extra_keys = {
            'concordance_ipcw': concordance_ipcw,
        }

        if log:
            self.log_metrics(
                subset=subset,
                error_name=error,
                loss=avg_loss,
                acc_at_ts=acc_at_ts,
                brier_score_at_ts=brier_score_at_ts,
                ece_at_ts=ece_at_ts,
                acc=acc,
                brier_score=brier_score,
                ece=ece,
                absolute_error=absolute_error,
                optimized_absolute_error=absolute_error_optimized,
                step=step,
                pctiles=pctiles,
                grouper_stats=grouper_stats,
                ece_equal_mass_at_ts=ece_equal_mass_at_ts,
                ece_equal_mass=ece_equal_mass,
                concordance_score=concordance_score,
                extra_keys=extra_keys,
            )

        self.model.train()
        return {
            'acc_at_ts': acc_at_ts,
            'brier_score_at_ts': brier_score_at_ts,
            'grouper_outputs': grouper_outputs,
            'loss': avg_loss,
            'acc': acc,
            'brier_score': brier_score,
            'ece': ece,
            'absolute_error': absolute_error,
            'absolute_error_optimized': absolute_error_optimized,
            'y_pred_by_decile_mean': y_pred_by_decile_mean,
            'y_by_decile_mean': y_by_decile_mean,
            'per_decile_count': per_decile_count,
            'ece_equal_mass_at_ts': ece_equal_mass_at_ts,
            'ece_equal_mass': ece_equal_mass,
            'y_pred_by_equal_mass_bin_mean': y_pred_by_equal_mass_bin_mean,
            'y_by_equal_mass_bin_mean': y_by_equal_mass_bin_mean,
            'per_equal_mass_bin_count': per_equal_mass_bin_count,
            'y_pred': y_pred_records,
            'concordance_score': concordance_score,
            'concordance_ipcw': concordance_ipcw,
        }

    def compute_eval_metrics(self,
                             y_preds,
                             t_seqs,
                             losses,
                             times,
                             correct,
                             absolute_errors,
                             absolute_errors_optimized,
                             brier_score_at_ts,
                             censored_mask,
                             y_pred_expected_time=None,
                             point_predictions_with_cost_function=None,
                             survival_functions=None,
                             train_times=None,
                             train_indicators=None):
        avg_loss = None
        if losses is not None:
            avg_loss = losses.sum() / len(losses)
        acc_at_ts = self.accuracy_average(correct, ~censored_mask, times)
        brier_score_at_ts = self.brier_average(brier_score_at_ts,
                                               ~censored_mask, times)
        ece_results = self.compute_ece_at_ts(y_preds,
                                             t_seqs,
                                             censored_mask,
                                             times,
                                             return_decile_avgs=True)
        ece_at_ts, y_pred_by_decile_mean, y_by_decile_mean, per_decile_count = ece_results
        # ece using equal mass binning

        ece_equal_mass_results = self.compute_ece_equal_mass_at_ts(
            y_preds, t_seqs, censored_mask, times, return_avgs=True)
        ece_equal_mass_at_ts, y_pred_by_equal_mass_bin_mean, y_by_equal_mass_bin_mean, per_equal_mass_bin_count = ece_equal_mass_results
        # aggregrate over time bins
        acc = torch.mean(acc_at_ts)
        brier_score = torch.mean(brier_score_at_ts)
        ece = torch.mean(ece_at_ts)
        ece_equal_mass = torch.mean(ece_equal_mass_at_ts)
        if self.train_config.adjust_eval and self.train_config.ipcw:
            # with the IPCW approach, we need to filter out censored examples
            absolute_error = absolute_errors[~censored_mask].mean()
            absolute_error_optimized = absolute_errors_optimized[
                ~censored_mask].mean()
        elif self.train_config.adjust_eval and self.train_config.impute:
            # with the imputation approach, we don't need to filter out censored examples
            absolute_error = absolute_errors.mean()
            absolute_error_optimized = absolute_errors_optimized.mean()
        else:
            # with no adjustment, we need to filter out censored examples
            absolute_error = absolute_errors[~censored_mask].mean()
            absolute_error_optimized = absolute_errors_optimized[
                ~censored_mask].mean()

        c_ipcw = np.nan
        # if survival functions are provided, compute the concordance score
        if survival_functions is not None and train_times is not None and train_indicators is not None:
            # compute the concordance score using the survival functions
            # convert the survival functions to a DataFrame with time along the index
            # and the columns as individual survival functions
            # survival_functions has shape (num_samples, num_time_bins)
            # but the expected shape is (num_time_bins, num_samples), hence the transpose
            survival_curves_df = pd.DataFrame(
                survival_functions.T.cpu().numpy(),
                index=self.time_bins.cpu().numpy(),
                columns=range(survival_functions.shape[0]))
            cindex = np.nan
            try:
                evl = LifelinesEvaluator(survival_curves_df, times,
                                         ~censored_mask, train_times,
                                         train_indicators)
                try:
                    cindex, _, _ = evl.concordance()
                except:
                    pass
                try:
                    interpolated_point_predictions = evl.predict_time_from_curve(
                        predict_median_st)

                    # compute IPCW adjusted concordance with torchsurv
                    # ground truth
                    time = times.to(self.device).float()  # observed times X_i
                    event = (~censored_mask).to(self.device)  # True if event

                    # IPCW weights at subject times using KM for the censoring distribution G
                    time_clamped = torch.clamp(
                        time, max=self.censoring_km.time.max())
                    G = self.censoring_km.predict(time_clamped).to(
                        self.device)  # shape (n,)
                    ipcw = event.float() / torch.clamp(G, min=1e-6)

                    # build a scalar risk from survival curves (negative median time with linear interpolation)
                    risk = torch.tensor(-interpolated_point_predictions,
                                        device=self.device,
                                        dtype=torch.float)

                    # compute IPCW-adjusted concordance index
                    cindex_evaluator = ConcordanceIndex()
                    c_ipcw = cindex_evaluator(risk, event, time, weight=ipcw)
                except:
                    c_ipcw = np.nan
            except:
                cindex = np.nan

        return_dict = {
            'avg_loss': avg_loss,
            'acc_at_ts': acc_at_ts,
            'brier_score_at_ts': brier_score_at_ts,
            'ece_at_ts': ece_at_ts,
            'ece_equal_mass_at_ts': ece_equal_mass_at_ts,
            'acc': acc,
            'brier_score': brier_score,
            'ece': ece,
            'ece_equal_mass': ece_equal_mass,
            'absolute_error': absolute_error,
            'absolute_error_optimized': absolute_error_optimized,
            'y_pred_by_decile_mean': y_pred_by_decile_mean,
            'y_by_decile_mean': y_by_decile_mean,
            'per_decile_count': per_decile_count,
            'y_pred_by_equal_mass_bin_mean': y_pred_by_equal_mass_bin_mean,
            'y_by_equal_mass_bin_mean': y_by_equal_mass_bin_mean,
            'per_equal_mass_bin_count': per_equal_mass_bin_count,
            'c_ipcw': c_ipcw,
        }
        if survival_functions is not None and train_times is not None and train_indicators is not None:
            return_dict['concordance_score'] = cindex
        else:
            return_dict['concordance_score'] = None
        return return_dict


def get_model(train_config,
              metadata,
              feature_names=None,
              embed_map=None,
              embed_col_indexes=None,
              time_bins=None,
              extra_dims=0):
    """Returns the model."""
    input_dim = metadata.input_dim
    if feature_names is not None and embed_map is not None:
        # remove the embed_map features from the count of continuous features
        num_continuous_features = len(feature_names) - len(embed_map)
        if not train_config.one_hot_embed:
            categorical_feature_dims = len(
                embed_map) * train_config.embedding_dimension
        else:
            categorical_feature_dims = sum(len(v) for v in embed_map.values())
        input_dim = num_continuous_features + categorical_feature_dims + extra_dims
    time_bins = train_config.time_bins + 1 if time_bins is None else time_bins
    valid_model_names = [model.name for model in Models]
    if train_config.model == Models.FFNet.name:
        return FFNet(train_config,
                     input_dim=input_dim,
                     hidden_dim=train_config.hidden_dim,
                     output_dim=time_bins,
                     num_hidden_layers=train_config.num_hidden_layers,
                     feature_names=feature_names,
                     embed_map=embed_map,
                     embed_col_indexes=embed_col_indexes)
    elif train_config.model in valid_model_names:
        output_dim = train_config.mixture_components
        return FFNet(train_config,
                     input_dim=input_dim,
                     hidden_dim=train_config.hidden_dim,
                     output_dim=output_dim,
                     time_bins=time_bins,
                     num_hidden_layers=train_config.num_hidden_layers,
                     feature_names=feature_names,
                     embed_map=embed_map,
                     embed_col_indexes=embed_col_indexes)
    else:
        raise ValueError(f"Unknown model: {train_config.model}")


def get_optimizer(train_config, model):
    """Returns the optimizer."""
    temp_params = [p for name, p in model.named_parameters() if 'temp' in name]
    base_params = [
        p for name, p in model.named_parameters() if 'temp' not in name
    ]
    # lower learning rate for temperature parameters
    param_groups = [
        {
            'params': base_params
        },
    ]
    if len(temp_params) > 0:
        param_groups.append({
            'params': temp_params,
            'lr': train_config.lr
        })  # * 0.1})

    if train_config.optimizer == Optimizers.Adam.name:
        return torch.optim.Adam(param_groups, lr=train_config.lr, eps=1e-7)
    elif train_config.optimizer == Optimizers.AdamW.name:
        return torch.optim.AdamW(param_groups, lr=train_config.lr)
    elif train_config.optimizer == Optimizers.SGD.name:
        return torch.optim.SGD(param_groups,
                               lr=train_config.lr,
                               momentum=0.9,
                               nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {train_config.optimizer}")


def create_model_hash(config_dict, exclude_keys=['model_hash']):
    # remove any arguments with None values
    config_dict = {k: v for k, v in config_dict.items() if v is not None}

    # also remove any arguments that are empty lists
    config_dict = {
        k: v
        for k, v in config_dict.items()
        if not (isinstance(v, list) and len(v) == 0)
    }

    # convert any numpy arrays to lists
    config_dict = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in config_dict.items()
    }

    # convert any lists to tuples so they can be hashed
    config_dict = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in config_dict.items()
    }

    exclude_keys = set(exclude_keys or [])
    filtered_params = {
        k: v
        for k, v in config_dict.items() if k not in exclude_keys
    }
    param_str = json.dumps(filtered_params, sort_keys=True)
    return hashlib.sha256(param_str.encode('utf-8')).hexdigest()[:8]


def get_loader_data(loader):
    x = []
    t = []
    c = []
    t_seq = []
    for batch in loader:
        x.append(batch.get('x'))
        t.append(batch.get('t'))
        t_seq.append(batch.get('t_seq'))
        c.append(batch.get('c'))

    x = torch.cat(x, dim=0).cpu().numpy()
    t = torch.cat(t, dim=0).cpu().numpy()
    t_seq = torch.cat(t_seq, dim=0).cpu().numpy()
    c = torch.cat(c, dim=0).cpu().numpy()
    return x, t, t_seq, c


def prepare_x_y(embed_col_indexes, x, t, c, encoder):
    if encoder is not None:
        categorical_data = x[:, embed_col_indexes].astype(np.int32)
        one_hot_encoded = encoder.transform(categorical_data)

        continuous_cols = np.setdiff1d(np.arange(x.shape[1]),
                                       embed_col_indexes)
        continuous_data = x[:, continuous_cols]

        x = np.hstack((continuous_data, one_hot_encoded))
    # create y in the expected format
    y = np.empty(dtype=[('event', np.bool), ('time', np.float64)],
                 shape=x.shape[0])
    y['event'] = ~c
    y['time'] = t
    return x, y


def get_survival_functions_df(sksurv_model, x, time_points):
    survival_functions = sksurv_model.predict_survival_function(x)
    # time_points = np.unique(np.concatenate([fn.x for fn in survival_functions]))
    # time_points = np.clip(train_loader.dataset.time_bins.numpy(), a_min=0.0, a_max=train_y['d.time'].max())
    # Create a dictionary to hold the resampled survival curves
    # The keys will be column headers (patient identifiers) and values will be the probabilities
    d = {}
    for i, fn in enumerate(survival_functions):
        # fn(time_points) evaluates the step function at each time point in our common grid
        # ensure the last time point is always within the range of the survival function by using try-except block
        try:
            d[i] = fn(time_points)
        except:
            max_time_point = fn.x[-1]
            d[i] = fn(np.clip(time_points, a_min=None, a_max=max_time_point))

    # Convert the dictionary to a DataFrame and set the time points as the index
    survival_curves_df = pd.DataFrame(d)
    survival_curves_df.index = time_points

    # `survival_curves_df` is now in the format LifelinesEvaluator expects!
    # It has time as the index and each patient as a column.
    # print(survival_curves_df.head())
    return survival_curves_df


def evaluate_split(time_bins,
                   x_eval,
                   y_eval,
                   t_eval,
                   t_seq_eval,
                   c_eval,
                   y_train,
                   model,
                   trainer,
                   subset='val'):
    eval_survival_curves_df = get_survival_functions_df(
        model, x_eval, time_bins)
    eval_time = y_eval['time']
    eval_event = y_eval['event']
    train_time = y_train['time']
    train_event = y_train['event']
    # concordance = model.score(x_eval, y_eval)
    evl = LifelinesEvaluator(eval_survival_curves_df, eval_time, eval_event,
                             train_time, train_event)
    cindex, _, _ = evl.concordance()
    # evaluate IPCW adjusted concordance
    interpolated_point_predictions = evl.predict_time_from_curve(
        predict_median_st)
    # clamp infinite values to max prediction time
    infinite_mask = np.isinf(interpolated_point_predictions)
    interpolated_point_predictions[infinite_mask] = time_bins[-1].item()

    # compute IPCW adjusted concordance with torchsurv
    # ground truth
    time = torch.tensor(t_eval).to('cuda').float()  # observed times X_i
    event = torch.tensor(~c_eval).to('cuda')  # True if event

    # IPCW weights at subject times using *your* KM for the censoring distribution G
    time_clamped = torch.clamp(time, max=trainer.censoring_km.time.max())
    G = trainer.censoring_km.predict(time_clamped).to('cuda')  # shape (n,)
    ipcw = event.float() / torch.clamp(G, min=1e-6)

    # build a scalar risk from survival curves (negative median time with linear interpolation)
    risk = torch.tensor(-interpolated_point_predictions,
                        device='cuda',
                        dtype=torch.float)

    # compute IPCW-adjusted concordance via torchsurv
    cindex_evaluator = ConcordanceIndex()
    c_ipcw = cindex_evaluator(risk, event, time, weight=ipcw)
    brier_score = evl.brier_score_multiple_points(time_bins)
    cdfs = 1 - torch.tensor(eval_survival_curves_df.T.values)
    pmf = trainer.cdf_to_pmf(cdfs).to(torch.float32)
    # convert the pmf to increment logits to compute the loss
    y_pred = FFNet.pmf_to_logits(pmf).to('cuda')
    t_seq_eval = torch.tensor(t_seq_eval).to('cuda')
    c_eval = torch.tensor(c_eval).to('cuda')
    t_eval = torch.tensor(t_eval).to('cuda')
    loss, _, _ = trainer.loss_function(y_pred, t_seq_eval, c_eval)
    # compute accuracy at time bins
    acc_at_ts = trainer.compute_accuracy_at_ts(y_pred,
                                               t_seq_eval,
                                               c_eval,
                                               t_eval,
                                               average=True)
    acc = acc_at_ts.mean()
    brier_score_at_ts = trainer.compute_brier_score_at_ts(y_pred,
                                                          t_seq_eval,
                                                          c_eval,
                                                          t_eval,
                                                          average=True)
    brier_score = brier_score_at_ts.mean()
    ece_at_ts, _, _, _ = trainer.compute_ece_at_ts(y_pred,
                                                   t_seq_eval,
                                                   c_eval,
                                                   t_eval,
                                                   return_decile_avgs=True)
    ece = ece_at_ts.mean()
    _, point_predictions = trainer.predict(y_pred, return_predicted_time=True)

    # now predict using the cost function
    point_predictions_with_cost_function = trainer.predict_with_cost_function(
        y_pred, error='absolute_error')
    absolute_error = trainer.compute_regression_error(point_predictions,
                                                      t_eval,
                                                      y_pred,
                                                      t_seq_eval,
                                                      c_eval,
                                                      error='absolute_error')
    absolute_error_optimized = trainer.compute_regression_error(
        point_predictions_with_cost_function,
        t_eval,
        y_pred,
        t_seq_eval,
        c_eval,
        error='absolute_error')
    # evaluate ECE
    ece_equal_mass_at_ts, y_pred_by_equal_mass_bin_mean, y_by_equal_mass_bin_mean, per_equal_mass_bin_count = trainer.compute_ece_equal_mass_at_ts(
        y_pred, t_seq_eval, c_eval, t_eval, return_avgs=True)
    ece_equal_mass = ece_equal_mass_at_ts.mean()
    eval_dict = dict(subset=subset,
                     error_name='absolute_error',
                     loss=loss,
                     acc_at_ts=acc_at_ts,
                     brier_score_at_ts=brier_score_at_ts,
                     ece_at_ts=ece_at_ts,
                     acc=acc,
                     brier_score=brier_score,
                     ece=ece,
                     absolute_error=absolute_error,
                     optimized_absolute_error=absolute_error_optimized,
                     step=0,
                     pctiles=[0.25, 0.5, 0.75],
                     ece_equal_mass_at_ts=ece_equal_mass_at_ts,
                     ece_equal_mass=ece_equal_mass,
                     grouper_stats={},
                     concordance_score=cindex,
                     extra_keys={
                         'concordance_ipcw': c_ipcw,
                     })
    trainer.log_metrics(use_tqdm=False, **eval_dict)
    return eval_dict


def sksurv_pipeline(train_config, wandb_config, embed_col_indexes,
                    train_loader, val_loader, test_loader):
    train_dataset = train_loader.dataset.dataset if hasattr(
        train_loader.dataset, 'dataset') else train_loader.dataset
    time_bins_orig = train_dataset.time_bins.cpu().numpy()

    # get all the train data (assuming train shuffling has been disabled so this is consistent across runs)
    x_train, t_train, t_seq_train, c_train = get_loader_data(train_loader)
    time_bins = np.clip(time_bins_orig, a_min=0.0,
                        a_max=t_train.max())  # clip for downstream eval

    # expand categorical features to one-hot encoding for sksurv models
    encoder = None
    if embed_col_indexes is not None:
        encoder = OneHotEncoder(sparse_output=False)
        categorical_data = x_train[:, embed_col_indexes].astype(np.int32)
        encoder.fit(categorical_data)

    # shuffle the indices and take 5 chunks for cross-validation
    num_samples = x_train.shape[0]
    indices = np.arange(num_samples)
    # set the seed for consistency
    np.random.seed(42)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, 5)
    # seed mapping to hold out certain chunks
    seed_mapping = {
        42: [0, 1, 2, 3],
        43: [1, 2, 3, 4],
        44: [2, 3, 4, 0],
        45: [3, 4, 0, 1],
        46: [4, 0, 1, 2],
    }
    chosen_indices = np.concatenate(
        [chunks[i] for i in seed_mapping[train_config.seed]])

    x_train = x_train[chosen_indices]
    t_train = t_train[chosen_indices]
    t_seq_train = t_seq_train[chosen_indices]
    c_train = c_train[chosen_indices]

    # create a trainer to compute metrics
    trainer = Trainer(train_config=train_config,
                      model=None,
                      optimizer=None,
                      time_bins=torch.tensor(time_bins_orig).to('cuda'))
    trainer.add_km(loader=None,
                   event_times=torch.tensor(t_train).to('cuda'),
                   censored=torch.tensor(c_train).to('cuda'))

    x_train, y_train = prepare_x_y(embed_col_indexes, x_train, t_train,
                                   c_train, encoder)
    # get val and test data for evaluation
    x_val, t_val, t_seq_val, c_val = get_loader_data(val_loader)
    x_val, y_val = prepare_x_y(embed_col_indexes, x_val, t_val, c_val, encoder)
    x_test, t_test, t_seq_test, c_test = get_loader_data(test_loader)
    x_test, y_test = prepare_x_y(embed_col_indexes, x_test, t_test, c_test,
                                 encoder)

    print(
        f"Training sksurv model: {train_config.model} on {x_train.shape[0]} samples with {x_train.shape[1]} features."
    )
    if train_config.model == Models.SKSurvRF.name:
        # MNIST is particularly memory intensive, so maybe limit the n_jobs
        n_jobs = -1 if train_config.dataset != 'MNIST' else min(64, os.cpu_count())
        # n_jobs = -1
        model = RandomSurvivalForest(
            n_estimators=train_config.rsf_n_estimators,
            max_features=train_config.rsf_max_features,
            min_samples_split=train_config.rsf_min_samples_split,
            n_jobs=n_jobs)
        model.fit(x_train, y_train)
    elif train_config.model == Models.SKSurvCox.name:
        model = CoxnetSurvivalAnalysis(
            fit_baseline_model=True,
            alphas=[train_config.cox_alpha],
            l1_ratio=train_config.cox_l1_ratio,
        )
        model.fit(x_train, y_train)
    else:
        raise ValueError(f"Unknown sksurv model: {train_config.model}")

    print(f"Evaluating sksurv model on validation and test sets.")
    val_metrics = evaluate_split(time_bins,
                                 x_val,
                                 y_val,
                                 t_val,
                                 t_seq_val,
                                 c_val,
                                 y_train,
                                 model,
                                 trainer,
                                 subset='val')
    test_metrics = evaluate_split(time_bins,
                                  x_test,
                                  y_test,
                                  t_test,
                                  t_seq_test,
                                  c_test,
                                  y_train,
                                  model,
                                  trainer,
                                  subset='test')
    return {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }


def pipeline(train_config, wandb_config):
    # wandb setup
    run = wandb.init(name=wandb_config.name,
                     project=wandb_config.project,
                     entity=wandb_config.entity,
                     group=wandb_config.group,
                     config=train_config.__dict__,
                     dir=wandb_config.dir,
                     mode=wandb_config.mode,
                     resume_from=None)
    # create a model hash
    model_hash = create_model_hash(
        train_config.__dict__,
    ) if train_config.model_hash is None else train_config.model_hash
    model_dir = os.path.join(train_config.dataset,
                             f'{model_hash}_{train_config.dataset}')
    model_dir = os.path.join(train_config.model_dir, model_dir)
    model_dir = os.path.abspath(model_dir)
    run.config.update({
        'local_model_dir': model_dir,
        'model_hash': model_hash
    },
                      allow_val_change=True)

    # set the device
    device = torch.device(train_config.device)

    # we're using an sksurv model, don't shuffle the data so that we can do k-fold runs
    shuffle_train = True
    if train_config.model in [Models.SKSurvRF.name, Models.SKSurvCox.name]:
        shuffle_train = False
    # load the dataset
    feature_names, embed_map, embed_col_indexes, time_bins, train_loader, val_loader, test_loader, metadata = load_data(
        train_config, shuffle_train=shuffle_train)

    # another pathway for the sksurv models
    if train_config.model in [Models.SKSurvRF.name, Models.SKSurvCox.name]:
        metrics = sksurv_pipeline(train_config, wandb_config,
                                  embed_col_indexes, train_loader, val_loader,
                                  test_loader)
        exit(0)

    # set the random seed for parameter generation (dataset creation has its own fixed seed so the dataset is always the same)
    # but the dataloader order will be differ under different train_config.seed values
    torch.manual_seed(train_config.seed)

    # in some cases, we may have extra feature dimensions that are not accounted for in the feature names
    # sample a point from the training set to determine if there are extra dimensions
    sample = train_loader.dataset.dataset[0] if hasattr(
        train_loader.dataset, 'dataset') else train_loader.dataset[0]
    x = sample[0]
    extra_dims = 0
    if feature_names is not None:
        extra_dims = len(x) - len(
            feature_names)  # always greater than or equal to 0
    # load the model
    model = get_model(train_config,
                      metadata,
                      feature_names,
                      embed_map,
                      embed_col_indexes,
                      time_bins,
                      extra_dims=extra_dims)
    # compute the number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({'num_model_parameters': num_params}, step=0)
    print(f"Model has {num_params:,} trainable parameters.")
    # compute the number of training data points
    num_train_data_points = get_num_train_data_points(train_loader)
    print(f"Number of training data points: {num_train_data_points:,}")
    wandb.log({'num_train_data_points': num_train_data_points}, step=0)

    # load the optimizer
    optimizer = get_optimizer(train_config, model)
    time_bins = train_loader.dataset.dataset.time_bins if hasattr(
        train_loader.dataset, 'dataset') else train_loader.dataset.time_bins
    # create the trainer
    trainer = Trainer(train_config,
                      model,
                      optimizer,
                      time_bins.to(device),
                      model_dir=model_dir)
    # train the model
    step = trainer.train(train_loader,
                         val_loader,
                         pctiles=train_config.time_bin_pctiles)

    # if we've saved the model, then reload the best model
    if train_config.save_model:
        # load the best model
        model_dir = os.path.join(model_dir, 'best')
        checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
        trainer.model.load_state_dict(checkpoint['model'])
        completed_steps = checkpoint['completed_steps']
        print(f"Loaded model from {model_dir} at step {completed_steps}")
        model = trainer.model

    # round up the train event times and censor indicators
    train_times, train_indicators = get_train_times_indicators(
        device, train_loader)
    # evaluate all groups on the val set
    val_dict = trainer.evaluate(
        val_loader,
        subset='val',
        error='absolute_error',
        step=step,
        pctiles=train_config.time_bin_pctiles,
        log=True,
        train_times=train_times,
        train_indicators=train_indicators,
    )

    # evaluate on the test set once
    test_dict = trainer.evaluate(test_loader,
                                 subset='test',
                                 error='absolute_error',
                                 step=step,
                                 pctiles=train_config.time_bin_pctiles,
                                 log=True,
                                 train_times=train_times,
                                 train_indicators=train_indicators)
    wandb.finish()


def get_train_times_indicators(device, train_loader):
    train_times = []
    train_indicators = []
    for batch in train_loader:
        x, y, c, t, t_seq, t_raw, t_raw_seq, g, index = batch.get(
            'x'), batch.get('y'), batch.get('c'), batch.get('t'), batch.get(
                't_seq'), batch.get('t_raw'), batch.get(
                    't_raw_seq'), batch.get('g'), batch.get('index')
        train_times.append(t)
        train_indicators.append(c)
    train_times = torch.cat(train_times).to(device)
    train_indicators = ~torch.cat(train_indicators).to(device)
    return train_times, train_indicators


def get_num_train_data_points(loader):
    num_train_data_points = len(loader.dataset)
    # loader may be using a subsampler so check the length of that
    if isinstance(loader.sampler,
                  (torch.utils.data.SubsetRandomSampler, SubclassSampler)):
        num_train_data_points = len(loader.sampler)
    return num_train_data_points


if __name__ == "__main__":
    # print the command string
    print(" ".join(sys.argv))
    wandb_config, train_config = get_args()

    pipeline(train_config, wandb_config)
