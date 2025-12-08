from __future__ import annotations

from dataclasses import MISSING

import torch
from torch import nn


class VAECfg:
    input_size: int = MISSING
    output_size: int = MISSING


class VAE(nn.Module):
    def __init__(self, cfg: VAECfg):
        super().__init__()
        self.mlp_mu = nn.Linear(cfg.input_size, cfg.output_size)
        self.mlp_std = nn.Linear(cfg.input_size, cfg.output_size)
        self.softplus = nn.Softplus()

        # Initialize std head with small weights to start with small std
        nn.init.xavier_uniform_(self.mlp_std.weight, gain=0.01)
        nn.init.constant_(self.mlp_std.bias, 0.0)  # softplus(0) = ln(2) ≈ 0.69

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mlp_mu(x)
        # Predict std directly using softplus (always positive, smooth gradients)
        std = self.softplus(self.mlp_std(x)) + 1e-6
        # Clamp std to reasonable range for numerical stability
        std = torch.clamp(std, min=0.01, max=5.0)

        # Compute logvar for KL loss compatibility (KL loss expects logvar)
        logvar = 2.0 * torch.log(std)

        out = self.reparameterize(mu, std) if sample else mu
        return out, mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return eps * std + mu


# class VAE(nn.Module):
#     def __init__(self, cfg: VAECfg):
#         super().__init__()
#         self.mlp_mu = nn.Linear(cfg.input_size, cfg.output_size)
#         self.mlp_logvar = nn.Linear(cfg.input_size, cfg.output_size)
#
#     def forward(
#             self,
#             x: torch.Tensor,
#             sample: bool = False
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mu = self.mlp_mu(x)
#         logvar = self.mlp_logvar(x)
#         out = self.reparameterize(mu, logvar) if sample else mu
#         return out, mu, logvar
#
#     @staticmethod
#     def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
