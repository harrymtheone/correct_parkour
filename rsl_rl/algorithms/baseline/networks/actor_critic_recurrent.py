from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.utils import wrapper
from .actor_critic import ActorCritic, ActorCriticCfg


class ActorCriticRecurrentCfg(ActorCriticCfg):
    rnn_type: str = 'lstm'
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(self, cfg: ActorCriticRecurrentCfg):
        # Create a modified cfg for parent class with rnn_hidden_size as input dims
        parent_cfg = ActorCriticCfg()
        parent_cfg.num_actor_obs = cfg.rnn_hidden_size
        parent_cfg.num_critic_obs = cfg.rnn_hidden_size
        parent_cfg.num_actions = cfg.num_actions
        parent_cfg.actor_hidden_dims = cfg.actor_hidden_dims
        parent_cfg.critic_hidden_dims = cfg.critic_hidden_dims
        parent_cfg.activation = cfg.activation
        parent_cfg.init_noise_std = cfg.init_noise_std

        super().__init__(parent_cfg)

        self.memory_a = Memory(cfg.num_actor_obs, type=cfg.rnn_type, num_layers=cfg.rnn_num_layers, hidden_size=cfg.rnn_hidden_size)
        self.memory_c = Memory(cfg.num_critic_obs, type=cfg.rnn_type, num_layers=cfg.rnn_num_layers, hidden_size=cfg.rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def init_hidden_states(self, num_envs: int, device: str):
        """Initialize hidden states for actor and critic RNNs.
        
        Returns:
            actor_hidden: Hidden state(s) for actor RNN
            critic_hidden: Hidden state(s) for critic RNN
        """
        actor_hidden = self.memory_a.init_hidden_states(num_envs, device)
        critic_hidden = self.memory_c.init_hidden_states(num_envs, device)
        return actor_hidden, critic_hidden

    def act(self, observations, hidden_states=None, seq_dim=False):
        """Forward pass through actor.
        
        Args:
            observations: Input observations
            hidden_states: Hidden states for RNN (required for both seq_dim modes now)
            seq_dim: Whether input has sequence dimension
            
        Returns:
            actions: Sampled actions
            new_hidden_states: Updated hidden states (only for seq_dim=False)
        """
        output, new_hidden = self.memory_a(observations, hidden_states, seq_dim=seq_dim)
        actions = super().act(output, seq_dim=seq_dim)
        if seq_dim:
            return actions
        else:
            return actions, new_hidden

    def act_inference(self, observations, hidden_states):
        """Inference mode action selection.
        
        Returns:
            actions: Mean actions (no sampling)
            new_hidden_states: Updated hidden states
        """
        output, new_hidden = self.memory_a(observations, hidden_states, seq_dim=False)
        actions = super().act_inference(output)
        return actions, new_hidden

    def evaluate(self, critic_observations, hidden_states=None, seq_dim=False):
        """Forward pass through critic.
        
        Returns:
            values: Value estimates
            new_hidden_states: Updated hidden states (only for seq_dim=False)
        """
        output, new_hidden = self.memory_c(critic_observations, hidden_states, seq_dim=seq_dim)
        values = super().evaluate(output, seq_dim=seq_dim)
        if seq_dim:
            return values
        else:
            return values, new_hidden


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        self.rnn_type = type.lower()
        rnn_cls = nn.GRU if self.rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def init_hidden_states(self, num_envs: int, device: str):
        """Initialize hidden states.
        
        Returns:
            For LSTM: tuple (h, c) each of shape [num_layers, num_envs, hidden_size]
            For GRU: tensor of shape [num_layers, num_envs, hidden_size]
        """
        h = torch.zeros(self.num_layers, num_envs, self.hidden_size, device=device)
        if self.rnn_type == 'lstm':
            c = torch.zeros(self.num_layers, num_envs, self.hidden_size, device=device)
            return (h, c)
        else:
            return h

    def forward(self, input, hidden_states, seq_dim=False):
        """Forward pass through RNN.
        
        Args:
            input: Input tensor
            hidden_states: Hidden states (required)
            seq_dim: Whether input has sequence dimension
            
        Returns:
            output: RNN output
            new_hidden_states: Updated hidden states
        """
        output, new_hidden = wrapper(self.rnn, (input, hidden_states), seq_dim=seq_dim)
        return output, new_hidden
