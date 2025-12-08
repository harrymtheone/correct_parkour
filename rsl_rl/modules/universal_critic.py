from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Sequence

import torch
from torch import nn

from .utils import make_mlp_layers, get_activation


class UniversalCriticCfg:
    priv_shape: tuple[int, int] = MISSING
    scan_shape: Sequence[int] = MISSING
    foothold_shape: Sequence[int] = MISSING

    hidden_dims: Sequence[int] = (512, 256, 128)

    activation: str = 'elu'


class UniversalCritic(nn.Module):
    def __init__(self, cfg: UniversalCriticCfg):
        super().__init__()
        activation = get_activation(cfg.activation)

        priv_size = math.prod(cfg.priv_shape)
        scan_size = math.prod(cfg.scan_shape)
        self.scan_ndim = len(cfg.scan_shape)
        foothold_size = math.prod(cfg.foothold_shape)
        self.foothold_ndim = len(cfg.scan_shape)

        self.priv_enc = make_mlp_layers(
            (priv_size, 256, 128),
            activation_func=activation,
            output_activation=False,
        )
        self.scan_enc = make_mlp_layers(
            (scan_size, 256, 32),
            activation_func=activation
        )
        self.edge_mask_enc = make_mlp_layers(
            (scan_size, 256, 32),
            activation_func=activation
        )
        self.foothold_enc = make_mlp_layers(
            (foothold_size, 64, 32),
            activation_func=activation
        )
        self.critic = make_mlp_layers(
            (128 + 32 + 32 + 32, *cfg.hidden_dims, 1),
            activation_func=activation,
            output_activation=False
        )

    def forward(
            self,
            priv_his: torch.Tensor,
            scan: torch.Tensor,
            edge_mask: torch.Tensor,
            foothold: torch.Tensor,
    ):
        priv_latent = self.priv_enc(priv_his.flatten(-2))
        scan_enc = self.scan_enc(scan.flatten(-self.scan_ndim))
        edge_enc = self.edge_mask_enc(edge_mask.flatten(-self.scan_ndim))
        foothold_enc = self.foothold_enc(foothold.flatten(-self.foothold_ndim))
        return self.critic(torch.cat([priv_latent, scan_enc, edge_enc, foothold_enc], dim=-1))
