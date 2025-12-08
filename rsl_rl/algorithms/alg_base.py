from __future__ import annotations

from typing import Any

import torch


class BaseAlgorithm:
    """Base class for all RL algorithms.
    
    Algorithms inheriting from this class work with dict-based observations
    and are compatible with the generic Runner class.
    """

    def act(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Select actions given observations during training.
        
        Args:
            obs: Dictionary of observation tensors
            
        Returns:
            Actions tensor [num_envs, num_actions]
        """
        raise NotImplementedError

    def process_env_step(
            self,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            infos: dict[str, Any],
            next_obs: dict[str, torch.Tensor] = None,
    ):
        """Process environment step results and store transition.
        
        Args:
            rewards: Reward tensor [num_envs]
            dones: Done flags [num_envs]
            infos: Info dict from environment
            next_obs: Next observations dict (optional, for algorithms that need it)
        """
        raise NotImplementedError

    def compute_returns(self, last_obs: dict[str, torch.Tensor]):
        """Compute returns/advantages after rollout collection.
        
        Args:
            last_obs: Final observations for bootstrapping
        """
        raise NotImplementedError

    def update(self, cur_it: int = 0, **kwargs) -> dict[str, float]:
        """Update algorithm (run optimization steps).
        
        Args:
            cur_it: Current iteration number
            
        Returns:
            Dictionary of metrics to log
        """
        raise NotImplementedError

    def reset(self, dones: torch.Tensor):
        """Reset algorithm state for terminated environments.
        
        Args:
            dones: Done flags indicating which envs to reset
        """
        pass  # Default: no-op

    def train(self):
        """Set algorithm to training mode."""
        pass  # Default: no-op

    def eval(self):
        """Set algorithm to evaluation mode."""
        pass  # Default: no-op

    def play_act(self, obs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Select actions during evaluation/inference.
        
        Args:
            obs: Dictionary of observation tensors
            
        Returns:
            Dictionary containing 'actions' and any other outputs
        """
        raise NotImplementedError

    def load(self, loaded_dict: dict[str, Any], load_optimizer: bool = True) -> Any:
        """Load algorithm state from checkpoint.
        
        Args:
            loaded_dict: Checkpoint dictionary
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Any info stored in checkpoint
        """
        raise NotImplementedError

    def save(self) -> dict[str, Any]:
        """Save algorithm state for checkpointing.
        
        Returns:
            Dictionary containing all state to save
        """
        raise NotImplementedError
