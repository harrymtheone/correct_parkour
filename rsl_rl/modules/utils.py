from __future__ import annotations
from typing import Sequence

import torch
from torch import nn


def get_activation(act_name: str):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def make_mlp_layers(
        shape: Sequence[int],
        activation_func: nn.Module,
        output_activation: bool = True,
) -> nn.Module:
    layers = nn.Sequential()

    for l1, l2 in zip(shape[:-1], shape[1:]):
        layers.append(nn.Linear(l1, l2))
        layers.append(activation_func)

    if not output_activation:
        layers.pop(-1)

    return layers


def wrapper(
        module: nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        seq_dim: bool,
):
    if isinstance(module, nn.RNNBase):
        x, hidden = x

        if seq_dim:
            return module(x, hidden)
        else:
            out, hidden = module(x.unsqueeze(0), hidden)
            return out.squeeze(0), hidden

    elif seq_dim:
        n_seq = x.size(0)
        return module(x.flatten(0, 1)).unflatten(0, (n_seq, x.size(1)))
    else:
        return module(x)
