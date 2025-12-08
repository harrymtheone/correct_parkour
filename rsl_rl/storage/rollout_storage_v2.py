from __future__ import annotations

import torch


class DataBuf:
    def __init__(
            self,
            n_envs: int,
            n_trans_per_env: int,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: str,
    ) -> None:
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        self.device = device
        self.buf = torch.zeros(
            n_trans_per_env, n_envs, *shape,
            dtype=dtype,
            device=self.device
        )

    def set(self, idx: int, value: torch.Tensor):
        self.buf[idx] = value

    def get(self, slice_: int | slice | list):
        return self.buf[slice_]

    def flatten_get(self, idx: int | slice | list | torch.Tensor):
        return self.buf.flatten(0, 1)[idx]


class HiddenBuf:
    def __init__(
            self,
            n_envs: int,
            n_trans_per_env: int,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: str
    ):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        self.device = device
        # hidden states are usually [num_layers, num_envs, hidden_size]
        # buffer will be [n_trans_per_env, num_layers, num_envs, hidden_size]
        self.buf = torch.zeros(
            n_trans_per_env, *shape,
            dtype=dtype,
            device=self.device
        )

    def set(self, idx: int, value: torch.Tensor):
        self.buf[idx].copy_(value)

    def get(self, slice_: int | slice | list):
        return self.buf[slice_]


class RolloutStorageV2:
    def __init__(self, num_envs: int, num_transitions_per_env: int, device: str = 'cpu'):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0
        self.init_done = False

        self.storage: dict[str, DataBuf] = {}
        self.hidden_states_storage: dict[str, HiddenBuf] = {}

        # Core standard buffers that are always expected (pre-allocated for convenience/compatibility)
        # Note: In V2 we might want these to be dynamic too, but keeping them for now as per "keep others the same"
        # However, the prompt asks to call add_... with name and value, implying dynamic storage.
        # We will implement dynamic storage based on the first transition.

    def add_observations(self, obs_dict: dict[str, torch.Tensor]):
        """
        obs_dict: dictionary mapping observation names to tensors
        """
        if not self.init_done:
            for k, v in obs_dict.items():
                self._init_element_storage(k, v)

        for k, v in obs_dict.items():
            if k in self.storage:
                self.storage[k].set(self.step, v)

    def add_hidden_states(
            self,
            name: str,
            hidden_states: torch.Tensor
    ):
        """
        name: identifier for the hidden state
        hidden_states: hidden state tensor [num_layers, num_envs, hidden_size]
        
        For LSTM with (h, c) tuple, call this twice with different names:
            add_hidden_states('actor_h', h)
            add_hidden_states('actor_c', c)
        """
        if not self.init_done:
            if name not in self.hidden_states_storage:
                # Create buffer outside inference mode to allow gradient computation later
                with torch.inference_mode(False):
                    self.hidden_states_storage[name] = HiddenBuf(
                        self.num_envs,
                        self.num_transitions_per_env,
                        hidden_states.shape,
                        hidden_states.dtype,
                        self.device
                    )

        if name in self.hidden_states_storage:
            self.hidden_states_storage[name].set(self.step, hidden_states)

    def add_transition_element(self, name: str, value: torch.Tensor):
        if not self.init_done:
            self._init_element_storage(name, value)

        if name in self.storage:
            self.storage[name].set(self.step, value)

    def flush_transition(self):
        self.init_done = True  # Lock initialization after first flush if not already done
        self.step += 1
        if self.step > self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

    def _init_element_storage(self, name: str, value: torch.Tensor):
        if name not in self.storage:
            # Value shape is expected to be [num_envs, ...]
            # Create buffer outside inference mode to allow gradient computation later
            with torch.inference_mode(False):
                self.storage[name] = DataBuf(
                    self.num_envs,
                    self.num_transitions_per_env,
                    value.shape[1:],
                    value.dtype,
                    self.device
                )

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantage = 0
        # Assuming standard names exist
        values = self.storage['values'].buf
        rewards = self.storage['rewards'].buf
        dones = self.storage['dones'].buf

        if 'returns' not in self.storage:
            self.storage['returns'] = DataBuf(
                self.num_envs, self.num_transitions_per_env, (1,), torch.float, self.device
            )
        if 'advantages' not in self.storage:
            self.storage['advantages'] = DataBuf(
                self.num_envs, self.num_transitions_per_env, (1,), torch.float, self.device
            )

        returns_buf = self.storage['returns'].buf
        advantages_buf = self.storage['advantages'].buf

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns_buf[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages_buf[:] = returns_buf - values
        advantages_buf[:] = (advantages_buf - advantages_buf.mean()) / (advantages_buf.std() + 1e-8)

    def get_statistics(self):
        if 'dones' not in self.storage or 'rewards' not in self.storage:
            return 0.0, 0.0

        done = self.storage['dones'].buf.clone()
        rewards = self.storage['rewards'].buf

        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), rewards.mean()

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Prepare flattened views for all stored data
        flattened_storage = {}
        for k, v in self.storage.items():
            flattened_storage[k] = v.buf.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                batch_dict = {}
                for k, v in flattened_storage.items():
                    batch_dict[k] = v[batch_idx]

                yield batch_dict

    def reccurent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        # Assuming 'dones' is present
        dones = self.storage['dones'].buf
        valid_mask = ~(torch.cumsum(dones.float(), dim=0) > 0).squeeze(-1)

        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                masks_batch = valid_mask[:, start:stop]

                batch_dict = {
                    'masks': masks_batch
                }

                # Slice standard buffers
                for k, v in self.storage.items():
                    # v.buf is [steps, envs, ...]
                    batch_dict[k] = v.buf[:, start:stop]

                # Slice hidden states
                # HiddenBuf stores [steps, num_layers, num_envs, hidden_size]
                # We want [num_layers, batch_envs, hidden_size] at step 0
                for k, v in self.hidden_states_storage.items():
                    # Take step 0
                    batch_dict[k] = v.buf[0, :, start:stop]

                yield batch_dict
