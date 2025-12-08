from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.utils import wrapper, get_activation


class ActorCriticCfg:
    num_actor_obs: int = MISSING
    num_critic_obs: int = MISSING
    num_actions: int = MISSING
    actor_hidden_dims: Sequence[int] = (256, 256, 256)
    critic_hidden_dims: Sequence[int] = (256, 256, 256)
    activation: str = 'elu'
    init_noise_std: float = 1.0


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, cfg: ActorCriticCfg):
        super().__init__()

        activation = get_activation(cfg.activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(cfg.num_actor_obs, cfg.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(cfg.actor_hidden_dims)):
            if l == len(cfg.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(cfg.actor_hidden_dims[l], cfg.num_actions))
            else:
                actor_layers.append(nn.Linear(cfg.actor_hidden_dims[l], cfg.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(cfg.num_critic_obs, cfg.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(cfg.critic_hidden_dims)):
            if l == len(cfg.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(cfg.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(cfg.critic_hidden_dims[l], cfg.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(cfg.init_noise_std * torch.ones(cfg.num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, seq_dim=False):
        mean = wrapper(self.actor, observations, seq_dim=seq_dim)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, seq_dim=False, **kwargs):
        self.update_distribution(observations, seq_dim=seq_dim)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, seq_dim=False, **kwargs):
        value = wrapper(self.critic, critic_observations, seq_dim=seq_dim)
        return value
