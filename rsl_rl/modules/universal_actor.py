from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch
import torch.nn as nn

from .utils import make_mlp_layers, get_activation


class UniversalActorCfg:
    input_shape: Sequence[int] = MISSING
    hidden_dims: Sequence[int] = (512, 256, 128)
    action_size: int = MISSING

    activation: str = 'elu'


class UniversalActor(nn.Module):
    def __init__(self, cfg: UniversalActorCfg):
        super().__init__()
        activation = get_activation(cfg.activation)

        input_size = math.prod(cfg.input_shape)

        self.actor_backbone = make_mlp_layers(
            (input_size, *cfg.hidden_dims, cfg.action_size),
            activation_func=activation,
            output_activation=False
        )

        self.std = nn.Parameter(torch.zeros(cfg.action_size))
        self.distribution = None

    def forward(self, x: torch.Tensor):
        return self.actor_backbone(x)

    def act(self, x: torch.Tensor, eval_: bool = False, **kwargs) -> torch.Tensor:
        if eval_:
            return self.forward(x)

        mean = self.forward(x)

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.std)
        return self.distribution.sample()

    def reset_std(self, std):
        self.std.data = torch.full_like(self.std.data, fill_value=std)

    def clip_std(self, min_std: float, max_std: float) -> None:
        self.std.data = torch.clamp(self.std.data, min_std, max_std)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return torch.sum(self.distribution.entropy(), dim=-1, keepdim=True)
