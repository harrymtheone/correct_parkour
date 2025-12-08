"""Generic runner for algorithms inheriting from BaseAlgorithm.

This runner works with dict-based observations and any algorithm
that implements the BaseAlgorithm interface.
"""
from __future__ import annotations

import os
import time
import statistics
from collections import deque
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from rsl_rl.algorithms.alg_base import BaseAlgorithm
    from rsl_rl.env import VecEnv


class RunnerCfg:
    """Configuration for the Runner."""
    # Training
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    
    # Checkpointing
    save_interval: int = 50
    
    # Logging
    log_dir: str = None
    experiment_name: str = 'default'
    run_name: str = ''
    
    # Resume training
    resume: bool = False
    resume_path: str = None


class Runner:
    """Generic on-policy runner for BaseAlgorithm-compatible algorithms.
    
    This runner:
    - Works with dict-based observations
    - Delegates storage management to the algorithm
    - Logs metrics returned by algorithm.update()
    - Supports any algorithm inheriting from BaseAlgorithm
    """

    def __init__(
            self,
            env: VecEnv,
            alg: BaseAlgorithm,
            cfg: RunnerCfg,
    ):
        """Initialize the runner.
        
        Args:
            env: Vectorized environment
            alg: Algorithm instance (must inherit from BaseAlgorithm)
            cfg: Runner configuration
        """
        self.env = env
        self.alg = alg
        self.cfg = cfg
        
        # Extract config values
        self.num_steps_per_env = cfg.num_steps_per_env
        self.save_interval = cfg.save_interval
        self.log_dir = cfg.log_dir

        # Logging
        self.writer: SummaryWriter | None = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    @property
    def device(self):
        return self.env.device

    def learn(self, num_learning_iterations: int = None, init_at_random_ep_len: bool = False):
        """Run training loop.
        
        Args:
            num_learning_iterations: Number of training iterations (default: cfg.max_iterations)
            init_at_random_ep_len: Initialize with random episode lengths
        """
        if num_learning_iterations is None:
            num_learning_iterations = self.cfg.max_iterations
        # Initialize tensorboard writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # Optionally randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        # Reset environment and get initial observations
        obs, _ = self.env.reset()
        obs = self._to_device(obs)

        # Set algorithm to training mode
        self.alg.train()

        # Episode tracking
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ==================== Rollout Collection ====================
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Get actions from algorithm
                    actions = self.alg.act(obs)

                    # Step environment
                    next_obs, _, rewards, dones, infos = self.env.step(actions)
                    next_obs = self._to_device(next_obs)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Process step (algorithm stores transition)
                    self.alg.process_env_step(rewards, dones, infos, next_obs)

                    # Reset algorithm state for done environments
                    self.alg.reset(dones)

                    # Update observations
                    obs = next_obs

                    # Book keeping for logging
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            collection_time = time.time() - start

            # ==================== Learning Step ====================
            start = time.time()

            # Compute returns with final observations
            with torch.inference_mode():
                self.alg.compute_returns(obs)

            # Update algorithm and get metrics
            metrics = self.alg.update(cur_it=it)

            learn_time = time.time() - start

            # ==================== Logging ====================
            if self.log_dir is not None:
                self._log(
                    it=it,
                    num_learning_iterations=num_learning_iterations,
                    metrics=metrics,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    ep_infos=ep_infos,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                )

            # Save checkpoint
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def _to_device(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move observations to device."""
        if isinstance(obs, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
        else:
            # Fallback for tensor observations
            return obs.to(self.device)

    def _log(
            self,
            it: int,
            num_learning_iterations: int,
            metrics: dict[str, float],
            collection_time: float,
            learn_time: float,
            ep_infos: list,
            rewbuffer: deque,
            lenbuffer: deque,
            width: int = 80,
            pad: int = 35,
    ):
        """Log training progress."""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        # Log episode info
        ep_string = ''
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(f'Episode/{key}', value, it)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # Log all metrics from algorithm
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, it)

        # Log performance metrics
        self.writer.add_scalar('Perf/total_fps', fps, it)
        self.writer.add_scalar('Perf/collection_time', collection_time, it)
        self.writer.add_scalar('Perf/learning_time', learn_time, it)

        # Log reward/episode length
        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)
            mean_ep_len = statistics.mean(lenbuffer)
            self.writer.add_scalar('Train/mean_reward', mean_reward, it)
            self.writer.add_scalar('Train/mean_episode_length', mean_ep_len, it)
            self.writer.add_scalar('Train/mean_reward/time', mean_reward, self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', mean_ep_len, self.tot_time)

        # Console output
        header = f" \033[1m Learning iteration {it}/{self.current_learning_iteration + num_learning_iterations} \033[0m "

        log_string = f"""{'#' * width}\n{header.center(width, ' ')}\n\n"""
        log_string += f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""

        # Add key metrics to console (limit to avoid clutter)
        key_metrics = ['Loss/value_loss', 'Loss/surrogate_loss', 'Loss/kl_div', 'Train/noise_std']
        for key in key_metrics:
            if key in metrics:
                log_string += f"""{f'{key}:':>{pad}} {metrics[key]:.4f}\n"""

        if len(rewbuffer) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n"""

        log_string += ep_string
        log_string += f"""{'-' * width}\n"""
        log_string += f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
        log_string += f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
        log_string += f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
        log_string += f"""{'ETA:':>{pad}} {self.tot_time / (it + 1) * (num_learning_iterations - it):.1f}s\n"""

        print(log_string)

    def save(self, path: str, infos: dict = None):
        """Save checkpoint.
        
        Args:
            path: Path to save checkpoint
            infos: Additional info to save
        """
        save_dict = self.alg.save()
        save_dict['iter'] = self.current_learning_iteration
        save_dict['infos'] = infos
        torch.save(save_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        """Load checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Info stored in checkpoint
        """
        loaded_dict = torch.load(path, map_location=self.device)
        self.current_learning_iteration = loaded_dict.get('iter', 0)
        return self.alg.load(loaded_dict, load_optimizer)

    def get_inference_policy(self):
        """Get inference function from algorithm.
        
        Returns:
            The play_act method bound to the algorithm
        """
        self.alg.eval()
        return self.alg.play_act


