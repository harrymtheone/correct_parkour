from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Sequence

import torch
import torch.nn as nn

from rsl_rl.modules.utils import get_activation, make_mlp_layers, wrapper
from rsl_rl.modules.vae import VAE, VAECfg

gru_hidden_size = 128
encoder_output_size = 3 + 64  # v_t, z_t


class EstimatorCfg:
    prop_size: int = MISSING
    prop_his_len: int = MISSING
    scan_shape: tuple[int, int] = MISSING

    latent_zm_size: int = 16
    latent_z_size: int = 16

    encoder_channel_size: int = 16
    decoder_scan_hidden_dims: Sequence[int] = (64, 128)
    decoder_ot1_hidden_dims: Sequence[int] = (64, 128)
    activation: str = 'elu'


class Estimator(nn.Module):
    def __init__(self, cfg: EstimatorCfg):
        super().__init__()
        self.prop_his_len = cfg.prop_his_len
        self.scan_shape = cfg.scan_shape
        activation = get_activation(cfg.activation)

        if cfg.prop_his_len != 100:
            self.projection = nn.Conv1d(
                cfg.prop_his_len, 100, kernel_size=1, bias=False
            )

        self.prop_his_encoder = nn.Sequential(
            nn.Conv1d(cfg.prop_size, 2 * cfg.encoder_channel_size, kernel_size=4, stride=4),
            activation,
            nn.Conv1d(2 * cfg.encoder_channel_size, 4 * cfg.encoder_channel_size, kernel_size=5, stride=5),
            activation,
            nn.Conv1d(4 * cfg.encoder_channel_size, 8 * cfg.encoder_channel_size, kernel_size=5, stride=1),  # (8 * channel_size, 1)
            activation,
            nn.Flatten()
        )

        vae_input_size = 8 * cfg.encoder_channel_size
        self.vae_vel = VAE(VAECfg(
            input_size=vae_input_size, output_size=3,
        ))
        self.vae_zm = VAE(VAECfg(
            input_size=vae_input_size, output_size=cfg.latent_zm_size,
        ))
        self.vae_z = VAE(VAECfg(
            input_size=vae_input_size, output_size=cfg.latent_z_size,
        ))

        self.decoder_ot1 = make_mlp_layers(
            (3 + cfg.latent_zm_size + cfg.latent_z_size, *cfg.decoder_ot1_hidden_dims, cfg.prop_size),
            activation_func=activation,
            output_activation=False,
        )
        self.decoder_scan = make_mlp_layers(
            (cfg.latent_zm_size, *cfg.decoder_scan_hidden_dims, math.prod(cfg.scan_shape)),
            activation_func=activation,
            output_activation=False,
        )

    def forward(self, prop_his, sample: bool, seq_dim: bool):
        if self.prop_his_len != 100:
            prop_his = self.projection(prop_his)

        prop_his = prop_his.transpose(-1, -2)
        x = wrapper(self.prop_his_encoder, prop_his, seq_dim=seq_dim)

        vel = self.vae_vel(x, sample=sample)
        zm = self.vae_zm(x, sample=sample)
        z = self.vae_z(x, sample=sample)

        ot1 = self.decoder_ot1(torch.cat([vel[0], zm[0], z[0]], dim=-1))
        scan_est = self.decoder_scan(zm[0])

        return vel, zm, z, ot1, scan_est
