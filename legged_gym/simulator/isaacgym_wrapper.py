from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict


class IsaacGymWrapper:
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.gym = gymapi.acquire_gym()  # noqa
        self.cfg = cfg
        self.sim_device = sim_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless
        self.device = self.sim_device  # alias

        self.sim = None
        self.viewer = None
        self.enable_viewer_sync = True

        # Internal buffers (use properties to access)
        self._root_states = None
        self._dof_state = None
        self._dof_pos = None
        self._dof_vel = None
        self._rigid_body_state = None
        self._contact_forces = None

        # Body/DOF names
        self._body_names: List[str] = []
        self._dof_names: List[str] = []

        # Redirect root support (for robots where base link is not the "root")
        self.redirect_root_to = getattr(cfg.asset, 'redirect_root_to', None)
        self.redirect_root_idx = None

        self.init_done = False

        self._create_sim()
        self._create_ground_plane()
        self._create_envs()
        self.gym.prepare_sim(self.sim)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(class_to_dict(self.cfg.sim), sim_params)

        # Override use_gpu_pipeline based on device
        if self.sim_device == 'cpu':
            sim_params.use_gpu_pipeline = False
        else:
            sim_params.use_gpu_pipeline = True
            sim_params.physx.use_gpu = True

        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self._body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self._dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self._body_names)
        self.num_dofs = len(self._dof_names)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = torch.tensor(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        # Prepare friction randomization
        self.friction_coeffs = None
        if self.cfg.domain_rand.randomize_friction:
            friction_range = self.cfg.domain_rand.friction_range
            num_buckets = 64
            bucket_ids = torch.randint(0, num_buckets, (self.cfg.env.num_envs, 1))
            friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
            self.friction_coeffs = friction_buckets[bucket_ids]

        for i in range(self.cfg.env.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.cfg.env.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Resolve redirect root index
        self._resolve_redirect_root_idx()

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.cfg.env.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.cfg.env.num_envs))
        num_rows = np.ceil(self.cfg.env.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.cfg.env.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.cfg.env.num_envs]
        self.env_origins[:, 2] = 0.

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction and self.friction_coeffs is not None:
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _resolve_redirect_root_idx(self):
        """Resolve the index for redirecting root queries to a specific link."""
        if self.redirect_root_to is not None:
            if self.redirect_root_to in self._body_names:
                self.redirect_root_idx = self._body_names.index(self.redirect_root_to)
                print(f"Redirecting root to {self.redirect_root_to} (index {self.redirect_root_idx})")
            else:
                raise ValueError(f"redirect_root_to '{self.redirect_root_to}' not found in body names: {self._body_names}")

    def init_buffers(self):
        """Acquire and wrap simulation state tensors."""
        # Acquire tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Wrap tensors
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.cfg.env.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.cfg.env.num_envs, self.num_dof, 2)[..., 1]
        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.cfg.env.num_envs, -1, 3)

        # Rigid body state: (num_envs * num_bodies, 13) -> (num_envs, num_bodies, 13)
        # Format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.cfg.env.num_envs, self.num_bodies, 13)

        self.init_done = True

    # ---------------------------------------- State Properties ----------------------------------------

    @property
    def dof_names(self):
        """List of DOF names."""
        return self._dof_names

    @property
    def body_names(self):
        """List of rigid body names."""
        return self._body_names

    # Legacy attribute access (for backward compatibility)
    @property
    def root_states(self):
        """Root states tensor (num_envs, 13): [pos(3), quat(4), lin_vel(3), ang_vel(3)]."""
        return self._root_states

    @property
    def dof_state(self):
        """DOF state tensor."""
        return self._dof_state

    @property
    def dof_pos(self):
        """DOF positions (num_envs, num_dof)."""
        return self._dof_pos

    @property
    def dof_vel(self):
        """DOF velocities (num_envs, num_dof)."""
        return self._dof_vel

    @property
    def base_quat(self):
        """Base quaternion (num_envs, 4) in (x, y, z, w) format."""
        return self._root_states[:, 3:7]

    @property
    def base_pos(self):
        """Base position (num_envs, 3)."""
        return self._root_states[:self.cfg.env.num_envs, 0:3]

    @property
    def contact_forces(self):
        """Contact forces (num_envs, num_bodies, 3)."""
        return self._contact_forces

    # ---------------------------------------- Root State Properties ----------------------------------------

    @property
    def root_pos_w(self):
        """Root position in world frame (num_envs, 3)."""
        if self.redirect_root_idx is None:
            return self._get_root_pos_w()
        return self.link_pos_w[:, self.redirect_root_idx]

    def _get_root_pos_w(self):
        """Get root position from root state tensor."""
        return self._root_states[:, 0:3]

    @property
    def root_quat_w(self):
        """Root quaternion in world frame (num_envs, 4) in (x, y, z, w) format."""
        if self.redirect_root_idx is None:
            return self._get_root_quat_w()
        return self.link_quat_w[:, self.redirect_root_idx]

    def _get_root_quat_w(self):
        """Get root quaternion from root state tensor."""
        return self._root_states[:, 3:7]

    @property
    def root_lin_vel_w(self):
        """Root linear velocity in world frame (num_envs, 3)."""
        if self.redirect_root_idx is None:
            return self._get_root_lin_vel_w()
        return self.link_lin_vel_w[:, self.redirect_root_idx]

    def _get_root_lin_vel_w(self):
        """Get root linear velocity from root state tensor."""
        return self._root_states[:, 7:10]

    @property
    def root_ang_vel_w(self):
        """Root angular velocity in world frame (num_envs, 3)."""
        if self.redirect_root_idx is None:
            return self._get_root_ang_vel_w()
        return self.link_ang_vel_w[:, self.redirect_root_idx]

    def _get_root_ang_vel_w(self):
        """Get root angular velocity from root state tensor."""
        return self._root_states[:, 10:13]

    # ---------------------------------------- Link State Properties ----------------------------------------

    @property
    def link_pos_w(self):
        """Link positions in world frame (num_envs, num_bodies, 3)."""
        return self._rigid_body_state[:, :, 0:3]

    @property
    def link_quat_w(self):
        """Link quaternions in world frame (num_envs, num_bodies, 4) in (x, y, z, w) format."""
        return self._rigid_body_state[:, :, 3:7]

    @property
    def link_lin_vel_w(self):
        """Link linear velocities in world frame (num_envs, num_bodies, 3)."""
        return self._rigid_body_state[:, :, 7:10]

    @property
    def link_ang_vel_w(self):
        """Link angular velocities in world frame (num_envs, num_bodies, 3)."""
        return self._rigid_body_state[:, :, 10:13]

    # ---------------------------------------- Utility Methods ----------------------------------------
    def get_body_ids(self, names: str | Sequence[str]) -> torch.Tensor:
        """Get link indices for the given names using Isaac Gym API."""
        if isinstance(names, str):
            names = (names,)
        link_ids = torch.zeros(len(names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(names):
            link_ids[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
        return link_ids

    def get_dof_ids(self, names: str | Sequence[str]) -> torch.Tensor:
        """Get DOF indices for the given names using Isaac Gym API."""
        if isinstance(names, str):
            names = (names,)
        dof_ids = torch.zeros(len(names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(names):
            dof_ids[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], name)
        return dof_ids

    # ---------------------------------------- Simulation Methods ----------------------------------------

    def prepare_sim(self):
        self.gym.prepare_sim(self.sim)

    def set_camera(self, position, lookat):
        if self.headless:
            return
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Subscribe to keyboard events
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def render(self, sync_frame_time=True):
        if self.headless or self.viewer is None:
            return

        # Check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            import sys
            sys.exit()

        # Check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                import sys
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # Step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def simulate(self):
        self.gym.simulate(self.sim)

    def fetch_results(self, force_sync=True):
        self.gym.fetch_results(self.sim, force_sync)

    def refresh_dof_state_tensor(self):
        self.gym.refresh_dof_state_tensor(self.sim)

    def refresh_actor_root_state_tensor(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def refresh_net_contact_force_tensor(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def refresh_rigid_body_state_tensor(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def set_dof_actuation_force_tensor(self, torques):
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

    def set_dof_state_tensor_indexed(self, dof_state, env_ids_int32):
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def set_actor_root_state_tensor_indexed(self, root_states, env_ids_int32):
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def get_elapsed_time(self):
        return self.gym.get_elapsed_time(self.sim)

    def get_sim_time(self):
        return self.gym.get_sim_time(self.sim)

    @property
    def sim_device_id(self):
        return int(self.sim_device.split(":")[1]) if "cuda" in self.sim_device else 0
