"""Defines model architectures."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from survkit.configs.train import Models, TimeWarpFunctions


def invert_temp(init_param):
    # convert an initial parameter to a log-space temperature
    # this is the inverse of the softplus function
    init_param = torch.tensor(init_param, )
    return torch.log(torch.exp(init_param) - 1)


class FFNetMixtureMTLR(nn.Module):
    """MoE head where each head is a separate MTLR predictor head."""

    def __init__(self,
                 config,
                 hidden_dim,
                 time_bins,
                 num_heads,
                 value_dim=None):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.time_bins = time_bins
        self.num_heads = num_heads
        if value_dim is None:
            value_dim = hidden_dim
        assert value_dim % num_heads == 0, "value_dim must be divisible by num_heads"
        self.subspace_dim = value_dim // num_heads
        # Q, K, V framework
        self.Q_projection = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.K = nn.Parameter(torch.empty((num_heads, hidden_dim)))
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))
        self.V_projection = nn.Linear(hidden_dim, value_dim)
        # if chunking the the value_dim into num_heads subspaces, then each head will operate on the subspace_dim
        if self.config.moe_chunk_value_dim:
            self.output_layer = nn.Parameter(
                torch.empty((num_heads, self.subspace_dim, time_bins)))
            self.output_layer_bias = nn.Parameter(
                torch.zeros((num_heads, time_bins)))
            self.init_output_layer()
        else:
            self.output_layer = nn.Parameter(
                torch.empty((num_heads, value_dim, time_bins)))
            self.output_layer_bias = nn.Parameter(
                torch.zeros((num_heads, time_bins)))
            self.init_output_layer()

        if config.learn_temperature:
            self.logit_temp = nn.Parameter(invert_temp(config.logit_temp))
        else:
            self.register_buffer('logit_temp', invert_temp(config.logit_temp))

    def init_output_layer(self):
        with torch.no_grad():
            for h in range(self.num_heads):
                Wh = self.output_layer[h]  # (D, T)
                D, T = Wh.shape
                # reinit as if weight were (T, D) for a Linear(T <- D)
                tmp = torch.empty(T, D, device=Wh.device, dtype=Wh.dtype)
                nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
                Wh.copy_(tmp.transpose(0, 1))  # back to (D, T)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tmp)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.output_layer_bias[h], -bound, bound)

    def forward(self, x):
        # project to V
        V = self.V_projection(x)  # (batch_size, value_dim)
        if self.config.moe_chunk_value_dim:
            # reshape V to (batch_size, num_heads, subspace_dim)
            V = V.view(V.shape[0], self.num_heads, self.subspace_dim)
        else:
            # reshape V to (batch_size, num_heads, value_dim)
            V = V[:, None, :].expand(
                V.shape[0], self.num_heads,
                V.shape[1])  # (batch_size, num_heads, value_dim)
        # compute the outputs over time
        # (batch_size, num_heads, subspace_dim) @ (num_heads, subspace_dim, time_bins)
        output = torch.einsum(
            'bhd,hdt->bht', V,
            self.output_layer)  # (batch_size, num_heads, time_bins)
        # add the bias
        output = output + self.output_layer_bias.unsqueeze(
            0)  # (batch_size, num_heads, time_bins)
        # aggregate the outputs using the router weights
        # (batch_size, num_heads) @ (batch_size, num_heads, time_bins)
        # project to Q and query keys
        Q = self.Q_projection(x)  # (batch_size, hidden_dim)
        router_logits = torch.einsum(
            'bd,hd->bh', Q,
            self.K)  # (batch_size, num_heads) # / math.sqrt(self.hidden_dim)
        log_router_weights = torch.log_softmax(
            router_logits / F.softplus(self.logit_temp),
            dim=1)  # (batch_size, num_heads)
        log_experts = FFNet.discrete_survival_log_pmf(
            output)  # (batch_size, num_heads, T)
        # combine in log-space
        log_expert_output = log_router_weights.unsqueeze(2) + log_experts
        log_mix = torch.logsumexp(log_expert_output, dim=1)  # (batch_size, T)
        output = FFNet.log_pmf_to_logits(log_mix)

        return {
            'model_out': x,
            'final_logits': output,
            'topk_expert_scores': router_logits,
            'expert_scores': router_logits,
            'log_expert_pmfs': log_experts,
        }


class SigmoidWarpHead(nn.Module):
    """Per-example, per-expert warp params for a 2-sigmoid monotone map."""

    def __init__(self, hidden_dim, num_experts, a_min=0.01):
        super().__init__()
        self.K = num_experts
        self.a_min = a_min
        # for 2 sigmoids: weights(2) + slopes(2) + centers(2) = 6 per expert
        self.proj = nn.Linear(hidden_dim, num_experts * 6, bias=True)

    def decode(self, raw, B):
        # raw: (B, K*6) -> (B, K, 6)
        M = 2
        raw = raw.view(B, self.K, 3 * M)
        w_raw = raw[..., :M]  # weights over sigmoids
        a_raw = raw[..., M:2 * M]  # slopes
        c_raw = raw[..., 2 * M:3 * M]  # centers
        w = torch.softmax(w_raw, dim=-1)  # (B,K,2), sum to 1
        # a = torch.exp(a_raw) + self.a_min # (B,K,2), strictly > 0
        # a = F.softplus(a_raw) + self.a_min # (B,K,2), strictly > 0
        # slopes (bounded)
        a_min, a_max = self.a_min, 40.0
        a = a_min + (a_max - a_min) * torch.sigmoid(
            a_raw)  # (B,K,2), keeps slopes reasonable
        # make the centers work together
        # ordered centers in (0,1): c1 in (0,1), c2 in (c1,1)
        c1 = torch.sigmoid(c_raw[..., 0])
        c2 = c1 + (1 - c1) * torch.sigmoid(c_raw[..., 1])
        c = torch.stack([c1, c2], dim=-1)  # (B,K,2)
        return w, a, c

    def forward(self, x):
        B = x.shape[0]
        raw = self.proj(x)  # (B, K*6)
        w, a, c = self.decode(raw, B)  # each (B,K,2)
        return w, a, c


def sigmoids(u, w, a, c):  # u: (B,K,1) or (B,K,T) in [0,1]
    # logistic CDFs
    s1 = torch.sigmoid(a[..., 0].unsqueeze(-1) * (u - c[..., 0].unsqueeze(-1)))
    s2 = torch.sigmoid(a[..., 1].unsqueeze(-1) * (u - c[..., 1].unsqueeze(-1)))
    return w[..., 0].unsqueeze(-1) * s1 + w[...,
                                            1].unsqueeze(-1) * s2  # (B,K,?)


@torch.no_grad()
def inv_two_sigmoid(t, w, a, c, iters: int, eps: float):
    """
    t: (B,K,T) in [0,1].  w,a,c: (B,K,2). Returns tau \approx psi(t) in [0,1] with bisection.
    """
    B, K, T = t.shape

    # affine normalization to [0,1] so F(0)=0, F(1)=1 numerically
    F0 = sigmoids(torch.zeros(B, K, 1, device=t.device, dtype=t.dtype), w, a,
                  c)
    F1 = sigmoids(torch.ones(B, K, 1, device=t.device, dtype=t.dtype), w, a, c)
    t_norm = (t - F0) / (F1 - F0 + eps)

    lo = torch.zeros_like(t_norm)
    hi = torch.ones_like(t_norm)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = (sigmoids(mid, w, a, c) - F0) / (F1 - F0 + eps)
        lo = torch.where(val < t_norm, mid, lo)
        hi = torch.where(val >= t_norm, mid, hi)
    tau = 0.5 * (lo + hi)
    return tau


class InvertTwoSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t, w, a, c):
        tau = inv_two_sigmoid(t, w, a, c, 20, 1e-5)  # solver
        ctx.save_for_backward(tau, t, w, a, c)
        return tau

    @staticmethod
    def backward(ctx, grad_tau):
        tau, t, w, a, c = ctx.saved_tensors
        sigma = torch.sigmoid
        sigma_p = lambda z: sigma(z) * (1 - sigma(z))

        z0 = a[..., 0, None] * (tau - c[..., 0, None])
        z1 = a[..., 1, None] * (tau - c[..., 1, None])

        F = w[..., 0, None] * sigma(z0) + w[..., 1, None] * sigma(z1)
        F0 = w[..., 0, None] * sigma(-a[..., 0, None] * c[..., 0, None]) + w[
            ..., 1, None] * sigma(-a[..., 1, None] * c[..., 1, None])
        F1 = w[..., 0, None] * sigma(
            a[..., 0, None] * (1 - c[..., 0, None])) + w[..., 1, None] * sigma(
                a[..., 1, None] * (1 - c[..., 1, None]))
        den = (F1 - F0).clamp_min(1e-8)

        # dFtilde/du
        Fu = (w[..., 0, None] * a[..., 0, None] * sigma_p(z0) +
              w[..., 1, None] * a[..., 1, None] * sigma_p(z1)) / den

        # parameter partials at u=tau
        dFw = torch.stack([sigma(z0), sigma(z1)], dim=-1)  # (B,K,T,2)
        dFa = torch.stack([
            w[..., 0, None] * sigma_p(z0) * (tau - c[..., 0, None]),
            w[..., 1, None] * sigma_p(z1) * (tau - c[..., 1, None])
        ],
                          dim=-1)
        dFc = torch.stack([
            -w[..., 0, None] * a[..., 0, None] * sigma_p(z0),
            -w[..., 1, None] * a[..., 1, None] * sigma_p(z1)
        ],
                          dim=-1)

        # normalization correction for Ftilde
        # dFtilde/dtheta = (dF - dF0 - Ftilde*(dF1 - dF0)) / (F1 - F0)
        Ftilde = (F - F0) / den
        dF0_w = torch.stack([
            sigma(-a[..., 0, None] * c[..., 0, None]),
            sigma(-a[..., 1, None] * c[..., 1, None])
        ],
                            dim=-1)
        dF1_w = torch.stack([
            sigma(a[..., 0, None] * (1 - c[..., 0, None])),
            sigma(a[..., 1, None] * (1 - c[..., 1, None]))
        ],
                            dim=-1)

        dF0_a = torch.stack([
            -w[..., 0, None] * c[..., 0, None] *
            sigma_p(-a[..., 0, None] * c[..., 0, None]), -w[..., 1, None] *
            c[..., 1, None] * sigma_p(-a[..., 1, None] * c[..., 1, None])
        ],
                            dim=-1)
        dF1_a = torch.stack([
            w[..., 0, None] * (1 - c[..., 0, None]) *
            sigma_p(a[..., 0, None] * (1 - c[..., 0, None])), w[..., 1, None] *
            (1 - c[..., 1, None]) * sigma_p(a[..., 1, None] *
                                            (1 - c[..., 1, None]))
        ],
                            dim=-1)

        dF0_c = torch.stack([
            -w[..., 0, None] * a[..., 0, None] *
            sigma_p(-a[..., 0, None] * c[..., 0, None]), -w[..., 1, None] *
            a[..., 1, None] * sigma_p(-a[..., 1, None] * c[..., 1, None])
        ],
                            dim=-1)
        dF1_c = torch.stack([
            -w[..., 0, None] * a[..., 0, None] *
            sigma_p(a[..., 0, None] * (1 - c[..., 0, None])),
            -w[..., 1, None] * a[..., 1, None] * sigma_p(a[..., 1, None] *
                                                         (1 - c[..., 1, None]))
        ],
                            dim=-1)

        dFtilde_w = (dFw - dF0_w - Ftilde.unsqueeze(-1) *
                     (dF1_w - dF0_w)) / den.unsqueeze(-1)
        dFtilde_a = (dFa - dF0_a - Ftilde.unsqueeze(-1) *
                     (dF1_a - dF0_a)) / den.unsqueeze(-1)
        dFtilde_c = (dFc - dF0_c - Ftilde.unsqueeze(-1) *
                     (dF1_c - dF0_c)) / den.unsqueeze(-1)

        # implicit gradient: dtau/dtheta = -(dFtilde/dtheta)/Fu
        scale = (-grad_tau / (Fu + 1e-8)).unsqueeze(-1)
        gw = (dFtilde_w * scale).sum(dim=-2)  # sum over T
        ga = (dFtilde_a * scale).sum(dim=-2)
        gc = (dFtilde_c * scale).sum(dim=-2)

        return None, gw, ga, gc


class TimeWarpMoE(nn.Module):

    def __init__(self, config, hidden_dim, time_bins):
        super().__init__()
        self.config = config
        self.t = nn.Parameter(torch.linspace(0, 1, time_bins),
                              requires_grad=False)
        self.proto = nn.Parameter(
            torch.zeros(config.mixture_components,
                        time_bins))  # shared prototypes (increment logits)
        nn.init.uniform_(self.proto, -0.1, 0.1)
        self.gate = nn.Linear(hidden_dim,
                              config.mixture_components,
                              bias=False)  # π_k(x)
        if config.time_warp_function == TimeWarpFunctions.TwoLogistics.name:
            # warp head (two-sigmoid)
            self.warp = SigmoidWarpHead(hidden_dim,
                                        config.mixture_components,
                                        a_min=0.01)
        elif config.time_warp_function == TimeWarpFunctions.BetaCDF.name:
            self.warp = nn.Linear(hidden_dim,
                                  config.mixture_components * 2,
                                  bias=False)  # (alpha,beta) params
        self.T = time_bins
        self.register_buffer('t_grid',
                             torch.linspace(0., 1., self.T).view(1, 1, self.T))
        if config.learn_temperature:
            self.logit_temp = nn.Parameter(invert_temp(config.logit_temp))
        else:
            self.register_buffer('logit_temp', invert_temp(config.logit_temp))

    def sample_protos_at_tau(self, tau, B, K, T):
        idx = tau * (T - 1)
        idx0 = idx.floor().clamp(max=T - 2).long()
        w = (idx - idx0.float())
        # linear interpolation of proto along time:
        P0 = self.proto.unsqueeze(0).expand(B, K, T).gather(2, idx0)  # (B,K,T)
        P1 = self.proto.unsqueeze(0).expand(B, K,
                                            T).gather(2, idx0 + 1)  # (B,K,T)
        warped = (1 - w) * P0 + w * P1  # (B,K,T)
        return warped

    def forward(self, x):
        B, K, T = x.size(0), self.proto.size(0), self.proto.size(1)
        # gate
        expert_scores = self.gate(x)  # (B,K)
        log_w = F.log_softmax(expert_scores / F.softplus(self.logit_temp),
                              dim=1)  # (B,K)
        # inverse map tau = psi(t)
        t = self.t_grid.expand(B, K, T)  # (B,K,T)
        if self.config.time_warp_function == TimeWarpFunctions.BetaCDF.name:
            warp_out = self.warp(x)  # (B,K,2)
            a = F.softplus(warp_out[..., 0]) + 1e-3  # (B,K), alpha > 0
            b = F.softplus(warp_out[..., 1]) + 1e-3  # (B,K), beta > 0
            # inverse Beta CDF via PyTorch's implementation
            tau = torch.distributions.Beta(
                a.unsqueeze(-1), b.unsqueeze(-1)).icdf(t)  # (B,K,T) in [0,1]
        elif self.config.time_warp_function == TimeWarpFunctions.TwoLogistics.name:
            # warp params
            w, a, c = self.warp(x)  # (B,K,2) each
            tau = InvertTwoSigmoid.apply(t, w, a, c)  # (B,K,T) in [0,1]

        # warp prototypes and mix in PMF space
        inc_logits_warped = self.sample_protos_at_tau(tau, B, K, T)  # (B,K,T)
        log_p_comp = FFNet.discrete_survival_log_pmf(
            inc_logits_warped)  # (B,K,T)
        log_p_mix = torch.logsumexp(log_w.unsqueeze(-1) + log_p_comp,
                                    dim=1)  # (B,T)
        final_logits = FFNet.log_pmf_to_logits(log_p_mix)
        return {
            'final_logits': final_logits,
            'expert_scores': expert_scores,
            'model_out': x,
            'log_expert_pmfs': log_p_comp
        }


class FFNet(nn.Module):
    """A simple feedforward neural network. There are num_hidden_layers + 1 linear layers."""

    def __init__(self,
                 config,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 time_bins=None,
                 num_hidden_layers=2,
                 feature_names=None,
                 embed_map=None,
                 embed_col_indexes=None):
        super().__init__()
        self.config = config
        self.model_type = config.model
        self.feature_names = feature_names
        self.embed_map = embed_map
        self.embed_col_indexes = embed_col_indexes
        # if embed_map is not None, then define embedding layers
        if embed_map is not None:
            if not config.one_hot_embed:
                self.embedding_layers = nn.ModuleList([
                    nn.Embedding(num_embeddings=len(embed_map[col]),
                                 embedding_dim=config.embedding_dimension)
                    for col in embed_map
                ])
            else:
                # one-hot via frozen identity lookups (callable modules)
                self.embedding_layers = nn.ModuleList([
                    nn.Embedding.from_pretrained(
                        torch.eye(len(embed_map[col]), device=config.device),
                        freeze=True  # no gradients, acts like pure one-hot
                    ) for col in embed_map
                ])
        else:
            self.embedding_layers = None

        layers = []
        for i in range(num_hidden_layers):
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            layers.append(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            if i < num_hidden_layers - 1:
                layers.append(nn.ReLU())
        if num_hidden_layers == 0:
            hidden_dim = input_dim  # in this case, the input layer is also the output layer
        self.model = nn.Sequential(*layers)
        self.final_activation = nn.ReLU()

        if self.model_type == Models.FFNetMixtureMTLR.name:
            # define a mixture of MTLR heads
            value_dim = hidden_dim if config.moe_value_dim is None else config.moe_value_dim
            self.mixture_mtlr = FFNetMixtureMTLR(config,
                                                 hidden_dim,
                                                 time_bins,
                                                 config.mixture_components,
                                                 value_dim=value_dim)

        if self.model_type in [Models.FFNetMixture.name]:
            # define a parametric module
            self.expert_router = nn.Linear(hidden_dim,
                                           config.mixture_components,
                                           bias=False)  # k experts
            self.expert_components = nn.Parameter(
                torch.empty((config.mixture_components, time_bins)))
            nn.init.uniform_(self.expert_components, -0.1, 0.1)

        if self.model_type in [Models.FFNetMixture.name]:
            if config.learn_temperature:
                # learnable temperature
                self.logit_temp = nn.Parameter(invert_temp(config.logit_temp))
                self.expert_logit_temp = nn.Parameter(
                    invert_temp(1.0))  # for the expert module
            else:
                # fixed temperature
                self.register_buffer('logit_temp',
                                     invert_temp(config.logit_temp))
                self.register_buffer('expert_logit_temp', invert_temp(1.0))

        if self.model_type == Models.FFNet.name:
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        if self.model_type == Models.FFNetTimeWarpMoE.name:
            self.time_warp_moe = TimeWarpMoE(config, hidden_dim, time_bins)

    def freeze_backbone_parameters(self):
        """Freezes the parameters of the backbone model, which includes the embedding layers, linear layers, but not the mixture components."""
        if self.embedding_layers is not None:
            for embedding_layer in self.embedding_layers:
                for param in embedding_layer.parameters():
                    param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_expert_parameters(self):
        """Freezes the parameters of the expert module, which includes the expert router and the expert components."""
        if self.model_type == Models.FFNetMixtureMTLR.name:
            # freeze the MTLR head parameters
            for param in self.mixture_mtlr.parameters():
                param.requires_grad = False
            return
        if self.model_type == Models.FFNetTimeWarpMoE.name:
            # freeze the TimeWarpMoE parameters
            for param in self.time_warp_moe.parameters():
                param.requires_grad = False
            return
        # otherwise freeze the expert parameters
        for param in self.expert_router.parameters():
            param.requires_grad = False
        self.expert_components.requires_grad = False
        # freeze temperature parameters if they are learnable
        if self.config.learn_temperature:
            for param in [
                    self.logit_temp, self.expert_logit_temp,
            ]:
                param.requires_grad = False

    @staticmethod
    def discrete_survival_log_pmf(y_pred):
        """Generates probabilities for every possible event time including no event."""
        # y_pred is a logit sequence of shape (batch_size, time_bins)
        # want to compute the probabilities for the label sequences
        # (1, 1, 1, ...., 1), (0, 1, 1, ..., 1), (0, 0, 1, ..., 1), ..., (0, 0, 0, ..., 1)
        # where (0, 0, 0, ..., 1) means never happens or later (the last time bin is special)
        # which is how the logits are already ordered
        # reverse the order and take the partial sums
        log_unnormalized_pmf = torch.cumsum(torch.flip(y_pred, dims=[-1]),
                                            dim=-1)
        # flip the order back
        log_unnormalized_pmf = torch.flip(log_unnormalized_pmf, dims=[-1])
        # get probabilities in log space for numerical stability
        log_pmf = log_unnormalized_pmf - torch.logsumexp(
            log_unnormalized_pmf, dim=-1, keepdim=True)
        return log_pmf

    @staticmethod
    def log_pmf_to_logits(log_pmf):
        """
        Inverts the discrete_survival_log_pmf without leaving log-space.
        Args:
            log_pmf (torch.Tensor): A batch of log probability mass functions, shape (B, T).
        Returns:
            torch.Tensor: A valid logit sequence, shape (B, T).
        """
        logits = torch.zeros_like(log_pmf)
        logits[..., :-1] = log_pmf[..., :-1] - log_pmf[..., 1:]
        logits[..., -1] = 0.0
        return logits

    @staticmethod
    def pmf_to_logits(pmf, alpha=0.0):
        """
        Inverts the discrete_survival_pmf function by choosing logit_T = 0.
        
        Args:
            pmf (torch.Tensor): A batch of probability mass functions, shape (B, T).
            alpha (float): Label smoothing parameter, defaults to 0.0.
        
        Returns:
            torch.Tensor: A valid logit sequence, shape (B, T).
        """
        # if alpha is greater than 0, implements label smoothing
        T = pmf.shape[1]
        pmf = (1 - alpha) * pmf + alpha / T
        pmf = torch.clamp(pmf, min=1e-9)
        log_p = torch.log(pmf)
        logits = torch.zeros_like(log_p)
        # for t = 1..T-1 (indices 0..T-2)
        logits[:, :-1] = log_p[:, :-1] - log_p[:, 1:]
        # for t = T, we simply leave the logit as 0 (the initial value)
        return logits

    def expert_module_forward(self, model_out):
        if self.config.expert_metric == 'Euclidean':
            # l2 distance metric between the queries and the keys
            distances = torch.cdist(model_out, self.expert_router.weight,
                                    p=2)  # (batch_size, k)
            expert_scores = -distances / F.softplus(self.logit_temp)
        elif self.config.expert_metric == 'Cosine':
            normalized_expert_keys = self.expert_router.weight  # no normalization needed
            # the matmul and division are differentiable operations in the graph
            expert_scores = torch.matmul(
                model_out, normalized_expert_keys.t()) / F.softplus(
                    self.logit_temp)
        # aggregate top-k experts
        topk_expert_scores, topk_expert_indices = torch.topk(
            expert_scores, self.config.mixture_topk, dim=1)
        topk_expert_scores_scaled = topk_expert_scores / F.softplus(
            self.expert_logit_temp)
        log_expert_weights = torch.log_softmax(topk_expert_scores_scaled,
                                               dim=1)  # (batch_size, k)
        selected_experts = self.expert_components[topk_expert_indices]
        log_selected_experts = self.discrete_survival_log_pmf(
            selected_experts)  # (batch_size, k, T)
        # combine in log-space
        log_expert_output = log_expert_weights.unsqueeze(
            2) + log_selected_experts
        log_mix = torch.logsumexp(log_expert_output, dim=1)  # (batch_size, T)
        expert_output = self.log_pmf_to_logits(log_mix)
        return expert_output, topk_expert_scores, expert_scores

    def forward(self, x):
        if self.embedding_layers is not None:
            # x a tensor where embed_col_indexes are the columns containing the indexes for the embeddings
            # create a list of tensors where each tensor is the result of passing the corresponding column through its embedding layer
            x_embed = [
                embedding_layer(x[:, col].long()) for embedding_layer, col in
                zip(self.embedding_layers, self.embed_col_indexes)
            ]
            # concatenate the tensors in x_embed
            x_embed = torch.cat(x_embed, dim=1)
            # take the remaining columns of x that are not in embed_col_indexes
            continuous_cols = [
                col for col in range(x.shape[1])
                if col not in self.embed_col_indexes
            ]
            x_continuous = x[:, continuous_cols]
            # concatenate x_continuous and x_embed
            x = torch.cat([x_continuous, x_embed], dim=1)
        # pass x through the model
        final_hidden_states = self.model(x)
        model_out = self.final_activation(final_hidden_states)

        if self.model_type == Models.FFNetMixture.name:
            expert_output, topk_expert_scores, expert_scores = self.expert_module_forward(
                model_out)
            return {
                'model_out': model_out,
                'final_logits': expert_output,
                'topk_expert_scores': topk_expert_scores,
                'expert_scores': expert_scores,
                'final_hidden_states': final_hidden_states
            }
        if self.model_type == Models.FFNetMixtureMTLR.name:
            mix_mtlr_dict = self.mixture_mtlr(model_out)
            mix_mtlr_dict['final_hidden_states'] = final_hidden_states
            return mix_mtlr_dict
        if self.model_type == Models.FFNet.name:
            final_logits = self.output_layer(model_out)
            return {
                'model_out': model_out,
                'final_logits': final_logits,
                'final_hidden_states': final_hidden_states
            }
        if self.model_type == Models.FFNetTimeWarpMoE.name:
            time_warp_moe_dict = self.time_warp_moe(model_out)
            time_warp_moe_dict['final_hidden_states'] = final_hidden_states
            return time_warp_moe_dict
