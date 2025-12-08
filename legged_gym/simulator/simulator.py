"""IsaacGym Simulator wrapper class.

This module provides a clean interface for interacting with the IsaacGym simulator,
encapsulating gym/sim management, environment creation, and state tensor access.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import torch_rand_float

from legged_gym import LEGGED_GYM_ROOT_DIR


class SimulatorCfg:
    """Configuration for the IsaacGym simulator."""
    sim_device: str = 'cuda:0'
    headless: bool = True

    # Simulation parameters
    dt: float = 0.005
    substeps: int = 1
    gravity: list = [0., 0., -9.81]
    up_axis: int = 1  # 0 is y, 1 is z
    use_gpu_pipeline: bool = True

    # PhysX parameters
    class physx:
        num_threads: int = 10
        solver_type: int = 1  # 0: pgs, 1: tgs
        num_position_iterations: int = 4
        num_velocity_iterations: int = 0
        contact_offset: float = 0.01
        rest_offset: float = 0.0
        bounce_threshold_velocity: float = 0.5
        max_depenetration_velocity: float = 1.0
        max_gpu_contact_pairs: int = 2 ** 23
        default_buffer_size_multiplier: int = 5
        contact_collection: int = 2


class Simulator:
    """Wrapper class for IsaacGym simulator.
    
    Handles simulation creation, environment setup, and state tensor management.
    """

    def __init__(self, cfg: SimulatorCfg):
        """Initialize the simulator.
        
        Args:
            cfg: Simulator configuration
        """
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.headless = cfg.headless

        # Build sim_params from config
        self.sim_params = self._build_sim_params(cfg)
        self.sim_device = cfg.sim_device

        # Parse device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(cfg.sim_device)

        # Determine compute device
        if sim_device_type == 'cuda' and cfg.use_gpu_pipeline:
            self.device = cfg.sim_device
        else:
            self.device = 'cpu'

        # Graphics device for rendering
        self.graphics_device_id = self.sim_device_id if not cfg.headless else -1

        # Create simulation
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params
        )

        # Viewer (created later if needed)
        self.viewer = None
        self.enable_viewer_sync = True

        # Environment handles
        self.envs = []
        self.actor_handles = []

        # State tensors (initialized after environments are created)
        self._root_states_tensor = None
        self._dof_state_tensor = None
        self._contact_forces_tensor = None
        self._rigid_body_states_tensor = None

        # Wrapped torch tensors
        self._root_states = None

    def _build_sim_params(self, cfg: SimulatorCfg) -> gymapi.SimParams:
        """Build IsaacGym SimParams from config."""
        sim_params = gymapi.SimParams()
        sim_params.dt = cfg.dt
        sim_params.substeps = cfg.substeps
        sim_params.gravity = gymapi.Vec3(*cfg.gravity)
        sim_params.up_axis = gymapi.UP_AXIS_Z if cfg.up_axis == 1 else gymapi.UP_AXIS_Y
        sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline

        # PhysX parameters
        sim_params.physx.use_gpu = cfg.use_gpu_pipeline
        sim_params.physx.num_threads = cfg.physx.num_threads
        sim_params.physx.solver_type = cfg.physx.solver_type
        sim_params.physx.num_position_iterations = cfg.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = cfg.physx.num_velocity_iterations
        sim_params.physx.contact_offset = cfg.physx.contact_offset
        sim_params.physx.rest_offset = cfg.physx.rest_offset
        sim_params.physx.bounce_threshold_velocity = cfg.physx.bounce_threshold_velocity
        sim_params.physx.max_depenetration_velocity = cfg.physx.max_depenetration_velocity
        sim_params.physx.max_gpu_contact_pairs = cfg.physx.max_gpu_contact_pairs
        sim_params.physx.default_buffer_size_multiplier = cfg.physx.default_buffer_size_multiplier

        return sim_params
        self._dof_state = None
        self._contact_forces = None
        self._rigid_body_states = None

        # Environment info
        self.num_envs = 0
        self.num_bodies = 0
        self.num_dofs = 0

    # ==================== Environment Setup ====================

    def create_ground_plane(self, static_friction: float, dynamic_friction: float, restitution: float):
        """Add a ground plane to the simulation.
        
        Args:
            static_friction: Static friction coefficient
            dynamic_friction: Dynamic friction coefficient
            restitution: Restitution coefficient
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = static_friction
        plane_params.dynamic_friction = dynamic_friction
        plane_params.restitution = restitution
        self.gym.add_ground(self.sim, plane_params)

    def load_asset(self, asset_file: str, asset_options: dict) -> tuple:
        """Load a robot asset from file.
        
        Args:
            asset_file: Path to the asset file (URDF/MJCF)
            asset_options: Dictionary of asset options
            
        Returns:
            Tuple of (asset, num_dof, num_bodies, dof_props, shape_props, body_names, dof_names)
        """
        asset_path = asset_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_filename = os.path.basename(asset_path)

        options = gymapi.AssetOptions()
        options.default_dof_drive_mode = asset_options.get('default_dof_drive_mode', gymapi.DOF_MODE_NONE)
        options.collapse_fixed_joints = asset_options.get('collapse_fixed_joints', True)
        options.replace_cylinder_with_capsule = asset_options.get('replace_cylinder_with_capsule', True)
        options.flip_visual_attachments = asset_options.get('flip_visual_attachments', False)
        options.fix_base_link = asset_options.get('fix_base_link', False)
        options.density = asset_options.get('density', 0.001)
        options.angular_damping = asset_options.get('angular_damping', 0.0)
        options.linear_damping = asset_options.get('linear_damping', 0.0)
        options.max_angular_velocity = asset_options.get('max_angular_velocity', 1000.0)
        options.max_linear_velocity = asset_options.get('max_linear_velocity', 1000.0)
        options.armature = asset_options.get('armature', 0.0)
        options.thickness = asset_options.get('thickness', 0.01)
        options.disable_gravity = asset_options.get('disable_gravity', False)

        asset = self.gym.load_asset(self.sim, asset_root, asset_filename, options)

        num_dof = self.gym.get_asset_dof_count(asset)
        num_bodies = self.gym.get_asset_rigid_body_count(asset)
        dof_props = self.gym.get_asset_dof_properties(asset)
        shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        body_names = self.gym.get_asset_rigid_body_names(asset)
        dof_names = self.gym.get_asset_dof_names(asset)

        return asset, num_dof, num_bodies, dof_props, shape_props, body_names, dof_names

    def create_envs(
            self,
            num_envs: int,
            asset,
            env_origins: torch.Tensor,
            base_init_state: torch.Tensor,
            actor_name: str,
            self_collisions: int,
            dof_props,
            shape_props,
            process_shape_props: Callable = None,
            process_dof_props: Callable = None,
            process_body_props: Callable = None,
    ):
        """Create environments with actors.
        
        Args:
            num_envs: Number of environments to create
            asset: Loaded asset
            env_origins: Origins for each environment [num_envs, 3]
            base_init_state: Initial state [13] (pos, quat, lin_vel, ang_vel)
            actor_name: Name for the actors
            self_collisions: Self-collision flags
            dof_props: DOF properties from asset
            shape_props: Shape properties from asset
            process_shape_props: Callback to modify shape props per env
            process_dof_props: Callback to modify DOF props per env
            process_body_props: Callback to modify body props per env
        """
        self.num_envs = num_envs

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*base_init_state[:3])

        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        self.envs = []
        self.actor_handles = []

        for i in range(num_envs):
            # Create environment
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(num_envs))
            )

            # Randomize position slightly
            pos = env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # Process shape properties
            if process_shape_props is not None:
                shape_props = process_shape_props(shape_props, i)
            self.gym.set_asset_rigid_shape_properties(asset, shape_props)

            # Create actor
            actor_handle = self.gym.create_actor(
                env_handle, asset, start_pose, actor_name, i, self_collisions, 0
            )

            # Process DOF properties
            if process_dof_props is not None:
                dof_props = process_dof_props(dof_props, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            # Process body properties
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if process_body_props is not None:
                body_props = process_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def find_body_indices(self, body_names: list[str]) -> torch.Tensor:
        """Find rigid body indices by name.
        
        Args:
            body_names: List of body names to find
            
        Returns:
            Tensor of body indices
        """
        indices = torch.zeros(len(body_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(body_names):
            indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )
        return indices

    def prepare(self):
        """Prepare the simulation after environments are created."""
        self.gym.prepare_sim(self.sim)

    def create_viewer(self):
        """Create viewer for visualization."""
        if self.headless:
            return

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def acquire_state_tensors(self, num_bodies: int, num_dofs: int):
        """Acquire GPU state tensors from the simulation.
        
        Args:
            num_bodies: Number of rigid bodies per environment
            num_dofs: Number of DOFs per environment
        """
        self.num_bodies = num_bodies
        self.num_dofs = num_dofs

        # Acquire tensors
        self._root_states_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._rigid_body_states_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # Wrap as torch tensors
        self._root_states = gymtorch.wrap_tensor(self._root_states_tensor)
        self._dof_state = gymtorch.wrap_tensor(self._dof_state_tensor)
        self._contact_forces = gymtorch.wrap_tensor(self._contact_forces_tensor).view(self.num_envs, -1, 3)
        self._rigid_body_states = gymtorch.wrap_tensor(self._rigid_body_states_tensor)

        # Initial refresh
        self.refresh_state_tensors()

    def refresh_state_tensors(self):
        """Refresh all state tensors from the simulation."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def refresh_rigid_body_state_tensor(self):
        """Refresh rigid body state tensor."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def refresh_dof_state_tensor(self):
        """Refresh DOF state tensor."""
        self.gym.refresh_dof_state_tensor(self.sim)

    # ==================== Properties for State Access ====================

    @property
    def root_states(self) -> torch.Tensor:
        """Root states tensor [num_envs, 13] - (pos[3], quat[4], lin_vel[3], ang_vel[3])."""
        return self._root_states

    @property
    def dof_state(self) -> torch.Tensor:
        """DOF state tensor [num_envs * num_dofs, 2] - (pos, vel)."""
        return self._dof_state

    @property
    def dof_pos(self) -> torch.Tensor:
        """DOF positions [num_envs, num_dofs]."""
        return self._dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]

    @property
    def dof_vel(self) -> torch.Tensor:
        """DOF velocities [num_envs, num_dofs]."""
        return self._dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

    @property
    def contact_forces(self) -> torch.Tensor:
        """Contact forces [num_envs, num_bodies, 3]."""
        return self._contact_forces

    @property
    def rigid_body_states(self) -> torch.Tensor:
        """Rigid body states [num_envs * num_bodies, 13]."""
        return self._rigid_body_states

    @property
    def rigid_body_states_view(self) -> torch.Tensor:
        """Rigid body states reshaped [num_envs, num_bodies, 13]."""
        return self._rigid_body_states.view(self.num_envs, -1, 13)

    # ==================== Simulation Control ====================

    def step(self):
        """Step the simulation forward one timestep."""
        self.gym.simulate(self.sim)

    def fetch_results(self, timeout: bool = True):
        """Fetch simulation results (needed for CPU pipeline)."""
        self.gym.fetch_results(self.sim, timeout)

    def set_dof_actuation_force(self, torques: torch.Tensor):
        """Set DOF actuation forces/torques.
        
        Args:
            torques: Tensor of torques [num_envs, num_dofs]
        """
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

    def set_dof_state_indexed(self, dof_state: torch.Tensor, env_ids: torch.Tensor):
        """Set DOF state for specific environments.
        
        Args:
            dof_state: Full DOF state tensor
            env_ids: Environment indices to update (int32)
        """
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def set_root_state_indexed(self, root_states: torch.Tensor, env_ids: torch.Tensor):
        """Set actor root state for specific environments.
        
        Args:
            root_states: Full root states tensor
            env_ids: Environment indices to update (int32)
        """
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    # ==================== Timing ====================

    def get_elapsed_time(self) -> float:
        """Get elapsed wall-clock time since simulation start."""
        return self.gym.get_elapsed_time(self.sim)

    def get_sim_time(self) -> float:
        """Get current simulation time."""
        return self.gym.get_sim_time(self.sim)

    # ==================== Rendering ====================

    def render(self, sync_frame_time: bool = True) -> bool:
        """Render the simulation.
        
        Args:
            sync_frame_time: Whether to synchronize with real-time
            
        Returns:
            False if viewer was closed, True otherwise
        """
        if self.viewer is None:
            return True

        # Check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            return False

        # Handle keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                return False
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # Fetch results for GPU pipeline
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        # Step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

        return True

    def set_camera(self, position: list, lookat: list):
        """Set camera position and look-at target.
        
        Args:
            position: Camera position [x, y, z]
            lookat: Look-at target [x, y, z]
        """
        if self.viewer is None:
            return
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
