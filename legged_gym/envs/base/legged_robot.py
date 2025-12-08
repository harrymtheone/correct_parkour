import sys
import time

import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, to_torch, get_axis_params, torch_rand_float

from legged_gym.simulator import Simulator
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.math import wrap_to_pi
from rsl_rl.env import VecEnv
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(VecEnv):
    """Legged robot environment for locomotion training."""

    def __init__(self, cfg: LeggedRobotCfg):
        """Initialize the legged robot environment.

        Args:
            cfg: Environment configuration (includes simulator config in cfg.simulator)
        """
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        # Parse config
        self._parse_cfg(cfg)

        # Create simulator from config
        self.sim = Simulator(cfg.simulator)
        self.device = self.sim.device
        self.headless = self.sim.headless

        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions

        # Optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Allocate buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

        # Create simulation environment
        self._create_sim()

        # Setup viewer
        if not self.headless:
            self.sim.create_viewer()
            self.sim.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # Initialize buffers
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def _create_sim(self):
        """Create simulation, ground plane, and environments."""
        # Create ground plane
        self.sim.create_ground_plane(
            self.cfg.terrain.static_friction,
            self.cfg.terrain.dynamic_friction,
            self.cfg.terrain.restitution
        )

        # Load robot asset
        asset_options = {
            'default_dof_drive_mode': self.cfg.asset.default_dof_drive_mode,
            'collapse_fixed_joints': self.cfg.asset.collapse_fixed_joints,
            'replace_cylinder_with_capsule': self.cfg.asset.replace_cylinder_with_capsule,
            'flip_visual_attachments': self.cfg.asset.flip_visual_attachments,
            'fix_base_link': self.cfg.asset.fix_base_link,
            'density': self.cfg.asset.density,
            'angular_damping': self.cfg.asset.angular_damping,
            'linear_damping': self.cfg.asset.linear_damping,
            'max_angular_velocity': self.cfg.asset.max_angular_velocity,
            'max_linear_velocity': self.cfg.asset.max_linear_velocity,
            'armature': self.cfg.asset.armature,
            'thickness': self.cfg.asset.thickness,
            'disable_gravity': self.cfg.asset.disable_gravity,
        }

        asset, num_dof, num_bodies, dof_props, shape_props, body_names, dof_names = \
            self.sim.load_asset(self.cfg.asset.file, asset_options)

        self.num_dof = num_dof
        self.num_bodies = num_bodies
        self.num_dofs = len(dof_names)
        self.dof_names = dof_names

        # Get body indices for feet, penalized contacts, termination contacts
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Compute environment origins (grid layout)
        self._get_env_origins()

        # Initial state
        base_init_state_list = (
                self.cfg.init_state.pos +
                self.cfg.init_state.rot +
                self.cfg.init_state.lin_vel +
                self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        # Create environments
        self.sim.create_envs(
            num_envs=self.num_envs,
            asset=asset,
            env_origins=self.env_origins,
            base_init_state=self.base_init_state,
            actor_name=self.cfg.asset.name,
            self_collisions=self.cfg.asset.self_collisions,
            dof_props=dof_props,
            shape_props=shape_props,
            process_shape_props=self._process_rigid_shape_props,
            process_dof_props=self._process_dof_props,
            process_body_props=self._process_rigid_body_props,
        )

        # Prepare simulation
        self.sim.prepare()

        # Acquire state tensors
        self.sim.acquire_state_tensors(self.num_bodies, self.num_dofs)

        # Find body indices
        self.feet_indices = self.sim.find_body_indices(feet_names)
        self.penalised_contact_indices = self.sim.find_body_indices(penalized_contact_names)
        self.termination_contact_indices = self.sim.find_body_indices(termination_contact_names)

    def _get_env_origins(self):
        """Set up environment origins in a grid layout."""
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(
            torch.arange(num_rows),
            torch.arange(num_cols),
            indexing='ij'
        )
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def step(self, actions):
        """Apply actions, simulate, and compute observations/rewards.

        Args:
            actions: Tensor of shape (num_envs, num_actions)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.sim.set_dof_actuation_force(self.torques)
            self.sim.step()

            if self.cfg.env.test:
                elapsed_time = self.sim.get_elapsed_time()
                sim_time = self.sim.get_sim_time()
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.sim.fetch_results()
            self.sim.refresh_dof_state_tensor()

        self.post_physics_step()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """Process after physics step: compute observations, rewards, resets."""
        self.sim.refresh_state_tensors()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Compute base state
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # Compute observations, rewards, resets
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.obs_buf = self.compute_observations()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        for key in self.obs_buf:
            self.obs_buf[key] = torch.clip(self.obs_buf[key], -clip_obs, clip_obs)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def reset(self):
        """Reset all environments."""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_buf = self.compute_observations()
        return self.obs_buf, self.extras

    def check_termination(self):
        """Check if environments need to be reset."""
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1
        )
        self.reset_buf |= torch.logical_or(
            torch.abs(self.rpy[:, 1]) > 1.0,
            torch.abs(self.rpy[:, 0]) > 0.8
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _reset_idx(self, env_ids):
        """Reset selected environments."""
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # Reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """Compute rewards from reward functions."""
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """Compute observations. Override in subclass."""
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        """Render the simulation."""
        if not self.sim.render(sync_frame_time):
            sys.exit()

    # ==================== Properties for simulator state access ====================

    @property
    def root_states(self) -> torch.Tensor:
        return self.sim.root_states

    @property
    def dof_state(self) -> torch.Tensor:
        return self.sim.dof_state

    @property
    def dof_pos(self) -> torch.Tensor:
        return self.sim.dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self.sim.dof_vel

    @property
    def contact_forces(self) -> torch.Tensor:
        return self.sim.contact_forces

    # ==================== Callbacks ====================

    def _process_rigid_shape_props(self, props, env_id):
        """Callback for modifying rigid shape properties per environment."""
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """Callback for modifying DOF properties per environment."""
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # Soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        """Callback for modifying rigid body properties per environment."""
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        """Callback after physics step for command updates."""
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.step_dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _resample_commands(self, env_ids):
        """Randomly select commands for specified environments."""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions using PD control."""
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Reset DOF positions and velocities."""
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        self.sim.set_dof_state_indexed(self.dof_state, env_ids)

    def _reset_root_states(self, env_ids):
        """Reset root states (position, orientation, velocities)."""
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        self.sim.set_root_state_indexed(self.root_states, env_ids)

    def _push_robots(self):
        """Apply random pushes to robots."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.sim.set_root_state_indexed(self.root_states, push_env_ids)

    def update_command_curriculum(self, env_ids):
        """Update command curriculum based on performance."""
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    # ==================== Initialization ====================

    def _init_buffers(self):
        """Initialize torch tensors for simulation states."""
        # Views into simulator state tensors
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]

        # Initialize data buffers
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., 2), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """Prepare list of reward functions."""
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.step_dt

        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    @property
    def physics_dt(self) -> float:
        """Physics simulation timestep."""
        return self.cfg.simulator.dt

    @property
    def step_dt(self) -> float:
        """Environment step timestep (physics_dt * decimation)."""
        return self.cfg.control.decimation * self.physics_dt

    def _parse_cfg(self, cfg):
        """Parse configuration."""
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.step_dt)

    # ==================== Reward Functions ====================

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        return torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.step_dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.step_dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
            5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1
        )

    def _reward_stand_still(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
