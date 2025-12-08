from __future__ import annotations

import torch
from torch.distributions import kl_divergence, Normal

from rsl_rl.algorithms.alg_base import BaseAlgorithm
from rsl_rl.modules.universal_actor import UniversalActor, UniversalActorCfg
from rsl_rl.modules.universal_critic import UniversalCritic, UniversalCriticCfg
from rsl_rl.storage import RolloutStorageV2 as RolloutStorage
from rsl_rl.utils.masked_loss import masked_mean, masked_MSE, masked_L1, masked_vae_kl_loss
from .networks import Estimator, EstimatorCfg


class PPODreamWaQ(BaseAlgorithm):
    def __init__(self, task_cfg, env, **kwargs):
        self.env = env

        # PPO parameters
        self.task_cfg = task_cfg
        self.ppo_cfg = task_cfg.ppo
        self.vae_cfg = task_cfg.vae
        self.learning_rate = self.ppo_cfg.learning_rate

        self.cur_it = 0

        self._build_components()

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def _build_components(self):
        # ---------------- Estimator ----------------
        self.estimator = Estimator(
            EstimatorCfg(
                prop_size=56, prop_his_len=100, scan_shape=(32, 16),
            )
        ).to(self.device)

        self.optimizer_vae = torch.optim.Adam(self.estimator.parameters(), lr=1e-3)
        self.scaler_vae = torch.GradScaler(enabled=self.vae_cfg.use_amp)

        # ---------------- Actor Critic ----------------
        self.actor = UniversalActor(
            UniversalActorCfg(
                input_shape=(56 + 3 + 16 + 16,), action_size=15,
            )
        ).to(self.device)
        self.actor.reset_std(self.ppo_cfg.init_noise_std)

        self.critic = UniversalCritic(
            UniversalCriticCfg(
                priv_shape=(50, 66), scan_shape=(32, 16), foothold_shape=(2, 50)
            )
        ).to(self.device)

        self.optimizer_ppo = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=self.learning_rate
        )

        # Rollout Storage
        self.storage = RolloutStorage(self.num_envs, 24, self.device)

    def act(self, obs: dict[str, torch.Tensor], **kwargs):
        """
        Args:
            obs: dict with keys like 'prop_his', 'proprio', 'vel_gt', 'scan_gt',
                 'priv_his', 'scan', 'edge_mask', 'foothold', etc.
        """
        # estimator forward
        vel_est, zm_est, z_est, ot1, scan_est = self.estimator.forward(
            obs['prop_his'], sample=True, seq_dim=False
        )

        # actor forward
        vel, zm, z = vel_est[0], zm_est[0], z_est[0]
        actor_input = torch.cat([obs['proprio'], vel, zm, z], dim=-1)
        actions = self.actor.act(actor_input, eval_=False)

        # critic forward
        values = self.critic.forward(
            obs['priv_his'], obs['scan'], obs['edge_mask'], obs['foothold'],
        )

        # Store observations to storage
        self.storage.add_observations(obs)

        # Store transition elements
        self.storage.add_transition_element('actor_input', actor_input)
        self.storage.add_transition_element('actions', actions)
        self.storage.add_transition_element('values', values)
        self.storage.add_transition_element('actions_log_prob', self.actor.get_actions_log_prob(actions))
        self.storage.add_transition_element('action_mean', self.actor.action_mean)
        self.storage.add_transition_element('action_sigma', self.actor.action_std)

        return actions

    def process_env_step(self, rewards, dones, infos, next_obs: dict[str, torch.Tensor] = None):
        # Store next observations for estimation training
        if next_obs is not None:
            for k, v in next_obs.items():
                self.storage.add_transition_element(f'next_{k}', v)

        rewards = rewards.clone().unsqueeze(1)
        dones_tensor = dones.unsqueeze(1)

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            values = self.storage.storage['values'].buf[self.storage.step]
            if 'reach_goals' in infos:
                bootstrapping = (infos['time_outs'] | infos['reach_goals']).unsqueeze(1).to(self.device)
            else:
                bootstrapping = infos['time_outs'].unsqueeze(1).to(self.device)
            rewards += self.ppo_cfg.gamma * values * bootstrapping

        self.storage.add_transition_element('rewards', rewards)
        self.storage.add_transition_element('dones', dones_tensor)

        # Flush the transition
        self.storage.flush_transition()

    def compute_returns(self, last_obs: dict[str, torch.Tensor]):
        last_values = self.critic.forward(
            last_obs['priv_his'], last_obs['scan'], last_obs['edge_mask'], last_obs['foothold'],
        )
        self.storage.compute_returns(last_values, self.ppo_cfg.gamma, self.ppo_cfg.lam)

    def update(self, cur_it=0, **kwargs):
        update_est = True

        self.cur_it = cur_it
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0

        mean_vel_est_loss = 0
        mean_ot1_est_loss = 0
        mean_scan_est_loss = 0

        mean_vel_kl_loss = 0
        mean_z_kl_loss = 0
        mean_zm_kl_loss = 0
        mean_abs_vel = 0
        mean_abs_z = 0
        mean_abs_zm = 0
        mean_std_vel = 0
        mean_std_z = 0
        mean_std_zm = 0
        mean_snr_vel = 0
        mean_snr_z = 0
        mean_snr_zm = 0
        kl_change = []

        # ########################## PPO update ##########################
        num_ppo_updates = 0
        for batch in self.storage.reccurent_mini_batch_generator(
                self.ppo_cfg.num_mini_batches,
                self.ppo_cfg.num_learning_epochs
        ):
            if torch.any(batch["masks"]):
                num_ppo_updates += 1
            else:
                continue

            ppo_metrics = self.update_ppo(batch)
            mean_kl += ppo_metrics['kl_mean']
            mean_value_loss += ppo_metrics['value_loss']
            mean_surrogate_loss += ppo_metrics['surrogate_loss']
            mean_entropy_loss += ppo_metrics['entropy_loss']
            kl_change.append(ppo_metrics['kl_mean'])

        # ########################## Estimator update ##########################
        num_vae_updates = 0
        if update_est:
            for batch in self.storage.reccurent_mini_batch_generator(
                    24, 1
            ):
                num_vae_updates += 1
                est_metrics = self.update_estimation(batch)
                mean_vel_est_loss += est_metrics['vel_est_loss']
                mean_ot1_est_loss += est_metrics['ot1_est_loss']
                mean_scan_est_loss += est_metrics['scan_est_loss']
                mean_vel_kl_loss += est_metrics['vel_kl_loss']
                mean_z_kl_loss += est_metrics['z_kl_loss']
                mean_zm_kl_loss += est_metrics['zm_kl_loss']
                mean_abs_vel += est_metrics['abs_vel']
                mean_abs_zm += est_metrics['abs_zm']
                mean_abs_z += est_metrics['abs_z']
                mean_std_vel += est_metrics['std_vel']
                mean_std_zm += est_metrics['std_zm']
                mean_std_z += est_metrics['std_z']
                mean_snr_vel += est_metrics['snr_vel']
                mean_snr_zm += est_metrics['snr_zm']
                mean_snr_z += est_metrics['snr_z']

        # ---- PPO ----
        if num_ppo_updates > 0:
            mean_kl /= num_ppo_updates
            mean_value_loss /= num_ppo_updates
            mean_surrogate_loss /= num_ppo_updates
            mean_entropy_loss /= num_ppo_updates
        # ---- VAE ----
        if num_vae_updates > 0:
            mean_vel_est_loss /= num_vae_updates
            mean_ot1_est_loss /= num_vae_updates
            mean_scan_est_loss /= num_vae_updates
            mean_vel_kl_loss /= num_vae_updates
            mean_z_kl_loss /= num_vae_updates
            mean_zm_kl_loss /= num_vae_updates
            mean_abs_vel /= num_vae_updates
            mean_abs_z /= num_vae_updates
            mean_abs_zm /= num_vae_updates
            mean_std_vel /= num_vae_updates
            mean_std_z /= num_vae_updates
            mean_std_zm /= num_vae_updates
            mean_snr_vel /= num_vae_updates
            mean_snr_z /= num_vae_updates
            mean_snr_zm /= num_vae_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        metrics = {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss': mean_value_loss,
            'Loss/kl_div': mean_kl,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Train/noise_std': self.actor.std.mean().item(),
        }

        if update_est:
            metrics.update({
                'VAE/vel_est_loss': mean_vel_est_loss,
                'VAE/ot1_est_loss': mean_ot1_est_loss,
                'VAE/scan_est_loss': mean_scan_est_loss,
                'VAE_KL/vel_kl_loss': mean_vel_kl_loss,
                'VAE_KL/z_kl_loss': mean_z_kl_loss,
                'VAE_KL/zm_kl_loss': mean_zm_kl_loss,
                'VAE_KL/abs_vel': mean_abs_vel,
                'VAE_KL/abs_z': mean_abs_z,
                'VAE_KL/abs_zm': mean_abs_zm,
                'VAE_KL/std_vel': mean_std_vel,
                'VAE_KL/std_z': mean_std_z,
                'VAE_KL/std_zm': mean_std_zm,
                'VAE_KL/SNR_vel': mean_snr_vel,
                'VAE_KL/SNR_z': mean_snr_z,
                'VAE_KL/SNR_zm': mean_snr_zm,
            })

        return metrics

    def update_ppo(self, batch: dict):
        actor_input = batch['actor_input']
        mask = batch['masks']
        actions = batch['actions']
        values = batch['values']
        advantages = batch['advantages']
        returns = batch['returns']
        old_actions_log_prob = batch['actions_log_prob']

        # Forward pass
        self.actor.act(actor_input)

        with torch.no_grad():
            kl_mean = kl_divergence(
                Normal(batch['action_mean'], batch['action_sigma']),
                Normal(self.actor.action_mean, self.actor.action_std)
            ).sum(dim=2, keepdim=True)
            kl_mean = masked_mean(kl_mean, mask).item()

        actions_log_prob = self.actor.get_actions_log_prob(actions)

        # Critic forward using dict-based observations from batch
        evaluation = self.critic.forward(
            batch['priv_his'], batch['scan'], batch['edge_mask'], batch['foothold']
        )

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * ratio.clamp(1.0 - self.ppo_cfg.clip_param, 1.0 + self.ppo_cfg.clip_param)

        # Debug NaN inputs before loss calculation
        if torch.isnan(surrogate).any() or torch.isinf(surrogate).any():
            masked_surr = surrogate[mask]
            if torch.isnan(masked_surr).any() or torch.isinf(masked_surr).any():
                print("\n" + "=" * 30 + " CRASH DEBUG " + "=" * 30)
                print(f"Valid Surrogate Max: {masked_surr.max()}, Min: {masked_surr.min()}")

                masked_adv = advantages[mask]
                print(f"Valid Advantages Max: {masked_adv.max()}, Min: {masked_adv.min()}")
                if torch.isnan(masked_adv).any(): print("!!! NaN in Advantages !!!")

                masked_ratio = ratio[mask]
                print(f"Valid Ratio Max: {masked_ratio.max()}, Min: {masked_ratio.min()}")
                if torch.isnan(masked_ratio).any(): print("!!! NaN in Ratio !!!")

                masked_logp = actions_log_prob[mask]
                masked_old_logp = old_actions_log_prob[mask]
                print(f"Valid LogProb Max: {masked_logp.max()}, Min: {masked_logp.min()}")
                print(f"Valid OldLogProb Max: {masked_old_logp.max()}, Min: {masked_old_logp.min()}")
                if torch.isnan(masked_logp).any(): print("!!! NaN in Current LogProb !!!")
                if torch.isnan(masked_old_logp).any(): print("!!! NaN in Old LogProb !!!")

                print("=" * 73 + "\n")
                raise ValueError(f"NaN/Inf in VALID surrogate steps! Max: {masked_surr.max()}, Min: {masked_surr.min()}")

        surrogate_loss = masked_mean(torch.maximum(surrogate, surrogate_clipped), mask)

        # Value function loss
        if self.ppo_cfg.use_clipped_value_loss:
            value_clipped = values + (evaluation - values).clamp(-self.ppo_cfg.clip_param, self.ppo_cfg.clip_param)
            value_loss = (evaluation - returns).square()
            value_loss_clipped = (value_clipped - returns).square()
            value_loss = masked_mean(torch.maximum(value_loss, value_loss_clipped), mask)
        else:
            value_loss = masked_MSE(evaluation, returns, mask)

        # Entropy loss
        entropy_loss = -masked_mean(self.actor.entropy, mask)

        # Total PPO loss
        total_loss = (surrogate_loss
                      + self.ppo_cfg.value_loss_coef * value_loss
                      + self.ppo_cfg.entropy_coef * entropy_loss)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"NaN/Inf in total_loss. Surr: {surrogate_loss.item()}, Val: {value_loss.item()}, Ent: {entropy_loss.item()}")

        # Use KL to adaptively update learning rate
        if self.ppo_cfg.schedule == 'adaptive' and self.ppo_cfg.desired_kl is not None:
            if kl_mean > self.ppo_cfg.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif self.ppo_cfg.desired_kl / 2.0 > kl_mean > 0.0:
                self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            for param_group in self.optimizer_ppo.param_groups:
                param_group['lr'] = self.learning_rate

        # Gradient step
        self.optimizer_ppo.zero_grad()
        total_loss.backward()

        # Check actor gradients for NaN or Inf
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"NaN or Inf detected in actor gradient: {name}")

        # Check critic gradients for NaN or Inf
        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"NaN or Inf detected in critic gradient: {name}")

        self.optimizer_ppo.step()

        # Check actor parameters for NaN or Inf after update
        for name, param in self.actor.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"NaN or Inf detected in actor parameter after update: {name}")

        self.actor.clip_std(self.ppo_cfg.noise_range[0], self.ppo_cfg.noise_range[1])

        return {
            'kl_mean': kl_mean,
            'value_loss': value_loss.item(),
            'surrogate_loss': surrogate_loss.item(),
            'entropy_loss': -entropy_loss.item(),
        }

    def update_estimation(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.vae_cfg.use_amp):
            mask = batch['masks']

            # Get observations from batch (dict-based)
            prop_his = batch['prop_his']
            vel_gt = batch['vel_gt']
            scan_gt = batch['scan_gt']
            next_proprio = batch['next_proprio']

            vel_est, zm_est, z_est, ot1_est, scan_est = self.estimator.forward(
                prop_his, sample=True, seq_dim=True
            )

            vel, vel_mu, vel_logvar = vel_est
            zm, zm_mu, zm_logvar = zm_est
            z, z_mu, z_logvar = z_est

            # Estimation loss
            vel_est_loss = masked_MSE(vel, vel_gt, mask)
            ot1_est_loss = masked_MSE(ot1_est, next_proprio, mask)
            scan_est_loss = masked_L1(scan_est, scan_gt, mask)

            # KL loss
            vel_kl_loss = masked_vae_kl_loss(vel_mu, vel_logvar, mask)
            zm_kl_loss = masked_vae_kl_loss(zm_mu, zm_logvar, mask)
            z_kl_loss = masked_vae_kl_loss(z_mu, z_logvar, mask)

            std_vel = vel_logvar.exp().sqrt().mean().item()
            std_zm = zm_logvar.exp().sqrt().mean().item()
            std_z = z_logvar.exp().sqrt().mean().item()
            mean_abs_vel = vel_mu.abs().mean().item()
            mean_abs_zm = zm_mu.abs().mean().item()
            mean_abs_z = z_mu.abs().mean().item()

            # Calculate SNR = |mean| / std (avoid division by zero)
            snr_vel = vel_mu.std().mean().item() / (std_vel + 1e-8)
            snr_zm = zm_mu.std().mean().item() / (std_zm + 1e-8)
            snr_z = z_mu.std().mean().item() / (std_z + 1e-8)

        kl_loss = (self.vae_cfg.kl_coef_vel * vel_kl_loss
                   + self.vae_cfg.kl_coef_zm * zm_kl_loss
                   + self.vae_cfg.kl_coef_z * z_kl_loss)

        # Total estimation loss
        total_loss = vel_est_loss + ot1_est_loss + scan_est_loss + kl_loss

        # Gradient step
        self.optimizer_vae.zero_grad()
        self.scaler_vae.scale(total_loss).backward()
        self.scaler_vae.step(self.optimizer_vae)
        self.scaler_vae.update()

        return {
            'vel_est_loss': vel_est_loss.item(),
            'ot1_est_loss': ot1_est_loss.item(),
            'scan_est_loss': scan_est_loss.item(),

            'vel_kl_loss': vel_kl_loss.item(),
            'z_kl_loss': z_kl_loss.item(),
            'zm_kl_loss': zm_kl_loss.item(),

            'abs_vel': mean_abs_vel,
            'abs_zm': mean_abs_zm,
            'abs_z': mean_abs_z,

            'std_vel': std_vel,
            'std_zm': std_zm,
            'std_z': std_z,

            'snr_vel': snr_vel,
            'snr_zm': snr_zm,
            'snr_z': snr_z,
        }

    def play_act(self, obs: dict[str, torch.Tensor], **kwargs):
        # estimator forward
        vel_est, zm_est, z_est, ot1, scan_est = self.estimator.forward(
            obs['prop_his'], sample=False, seq_dim=False
        )

        # actor forward
        vel, zm, z = vel_est[0], zm_est[0], z_est[0]
        actor_input = torch.cat([obs['proprio'], vel, zm, z], dim=-1)

        actions = self.actor.act(actor_input, **kwargs)

        return {'actions': actions, 'recon': scan_est.view(-1, 32, 16)}

    def reset(self, dones):
        return

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            self.optimizer_ppo.load_state_dict(loaded_dict['optimizer_ppo_state_dict'])
            self.optimizer_vae.load_state_dict(loaded_dict['optimizer_vae_state_dict'])

        if not self.ppo_cfg.continue_from_last_std:
            self.actor.reset_std(self.ppo_cfg.init_noise_std)

        return loaded_dict['infos']

    def save(self):
        return {
            'estimator_state_dict': self.estimator.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),

            'optimizer_ppo_state_dict': self.optimizer_ppo.state_dict(),
            'optimizer_vae_state_dict': self.optimizer_vae.state_dict(),
        }
