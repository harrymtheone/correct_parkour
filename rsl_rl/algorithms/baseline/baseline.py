from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms.alg_base import BaseAlgorithm
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorageV2


class PPOBaseline(BaseAlgorithm):
    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Hidden states for recurrent policies (managed here, not in actor_critic)
        self.actor_hidden = None
        self.critic_hidden = None

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageV2(num_envs, num_transitions_per_env, self.device)
        self.num_envs = num_envs
        
        # Initialize hidden states for recurrent policies
        if self.actor_critic.is_recurrent:
            self.actor_hidden, self.critic_hidden = self.actor_critic.init_hidden_states(num_envs, self.device)

    def eval(self):
        self.actor_critic.test()

    def train(self):
        self.actor_critic.train()

    def act(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # Extract observations
        # 'proprio' is the main observation for the actor
        # 'priv_obs' is the privileged observation for the critic (if available)
        actor_obs = obs['proprio']
        critic_obs = obs.get('priv_obs', actor_obs)

        if critic_obs is None:
            critic_obs = actor_obs

        # Store hidden states for recurrent policies before forward pass
        if self.actor_critic.is_recurrent:
            # Each hidden could be a tuple (h, c) for LSTM or single tensor for GRU
            if isinstance(self.actor_hidden, tuple):
                self.storage.add_hidden_states('actor_h', self.actor_hidden[0])
                self.storage.add_hidden_states('actor_c', self.actor_hidden[1])
            else:
                self.storage.add_hidden_states('actor_h', self.actor_hidden)
            
            if isinstance(self.critic_hidden, tuple):
                self.storage.add_hidden_states('critic_h', self.critic_hidden[0])
                self.storage.add_hidden_states('critic_c', self.critic_hidden[1])
            else:
                self.storage.add_hidden_states('critic_h', self.critic_hidden)

            # Forward pass with hidden states, get updated hidden states
            actions, self.actor_hidden = self.actor_critic.act(actor_obs, hidden_states=self.actor_hidden, seq_dim=False)
            values, self.critic_hidden = self.actor_critic.evaluate(critic_obs, hidden_states=self.critic_hidden, seq_dim=False)
            actions = actions.detach()
            values = values.detach()
        else:
            # Non-recurrent: simple forward pass
            actions = self.actor_critic.act(actor_obs).detach()
            values = self.actor_critic.evaluate(critic_obs).detach()

        actions_log_prob = self.actor_critic.get_actions_log_prob(actions).detach()
        action_mean = self.actor_critic.action_mean.detach()
        action_sigma = self.actor_critic.action_std.detach()

        # Store transition elements
        self.storage.add_transition_element('observations', actor_obs)
        self.storage.add_transition_element('critic_observations', critic_obs)
        self.storage.add_transition_element('actions', actions)
        self.storage.add_transition_element('values', values)
        self.storage.add_transition_element('actions_log_prob', actions_log_prob)
        self.storage.add_transition_element('action_mean', action_mean)
        self.storage.add_transition_element('action_sigma', action_sigma)

        return actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        rewards = rewards.clone()
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            values = self.storage.storage['values'].buf[self.storage.step]
            rewards += self.gamma * torch.squeeze(values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.storage.add_transition_element('rewards', rewards.unsqueeze(-1))
        self.storage.add_transition_element('dones', dones.unsqueeze(-1))

        # Flush the transition
        self.storage.flush_transition()

    def reset(self, dones):
        """Reset hidden states for done environments."""
        if not self.actor_critic.is_recurrent:
            return
            
        # Reset hidden states for done environments
        if isinstance(self.actor_hidden, tuple):
            # LSTM: reset both h and c
            self.actor_hidden[0][..., dones, :] = 0.0
            self.actor_hidden[1][..., dones, :] = 0.0
        else:
            # GRU: reset single hidden state
            self.actor_hidden[..., dones, :] = 0.0
            
        if isinstance(self.critic_hidden, tuple):
            self.critic_hidden[0][..., dones, :] = 0.0
            self.critic_hidden[1][..., dones, :] = 0.0
        else:
            self.critic_hidden[..., dones, :] = 0.0

    def compute_returns(self, last_obs: dict[str, torch.Tensor]):
        critic_obs = last_obs.get('priv_obs', last_obs['proprio'])
        if critic_obs is None:
            critic_obs = last_obs['proprio']

        if self.actor_critic.is_recurrent:
            last_values, _ = self.actor_critic.evaluate(critic_obs, hidden_states=self.critic_hidden, seq_dim=False)
            last_values = last_values.detach()
        else:
            last_values = self.actor_critic.evaluate(critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, cur_it: int = 0, **kwargs) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            actions_batch = batch['actions']
            values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_actions_log_prob_batch = batch['actions_log_prob']
            old_mu_batch = batch['action_mean']
            old_sigma_batch = batch['action_sigma']
            
            # Get hidden states for recurrent policies
            if self.actor_critic.is_recurrent:
                actor_h = batch.get('actor_h')
                actor_c = batch.get('actor_c')
                critic_h = batch.get('critic_h')
                critic_c = batch.get('critic_c')
                
                if actor_c is not None:
                    actor_hidden = (actor_h, actor_c)
                    critic_hidden = (critic_h, critic_c)
                else:
                    actor_hidden = actor_h
                    critic_hidden = critic_h
            else:
                actor_hidden = None
                critic_hidden = None

            self.actor_critic.act(obs_batch, hidden_states=actor_hidden, seq_dim=True)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, hidden_states=critic_hidden, seq_dim=True)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    old_dist = Normal(old_mu_batch, old_sigma_batch)
                    new_dist = Normal(mu_batch, sigma_batch)
                    kl = kl_divergence(old_dist, new_dist).sum(dim=-1)
                    kl_mean = kl.mean()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = values_batch + (value_batch - values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return {
            'value_loss': mean_value_loss,
            'surrogate_loss': mean_surrogate_loss,
            'learning_rate': self.learning_rate
        }

    def play_act(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # For inference/playing
        actor_obs = obs['proprio']
        if self.actor_critic.is_recurrent:
            actions, self.actor_hidden = self.actor_critic.act_inference(actor_obs, self.actor_hidden)
            return actions
        else:
            return self.actor_critic.act_inference(actor_obs)

    def save(self) -> dict:
        return {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_optimizer: bool = True):
        self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        return loaded_dict.get('infos', {})
