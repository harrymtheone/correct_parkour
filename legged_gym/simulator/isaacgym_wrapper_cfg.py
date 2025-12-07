"""Configuration classes for IsaacGymWrapper."""
from __future__ import annotations

from dataclasses import MISSING
from typing import List, Tuple


class SimCfg:
    """Simulation parameters passed to Isaac Gym."""
    dt: float = 0.005
    substeps: int = 1
    gravity: List[float] = [0.0, 0.0, -9.81]
    up_axis: int = 1  # 0 is y, 1 is z

    class physx:
        num_threads: int = 10
        solver_type: int = 1  # 0: pgs, 1: tgs
        num_position_iterations: int = 4
        num_velocity_iterations: int = 0
        contact_offset: float = 0.01  # [m]
        rest_offset: float = 0.0  # [m]
        bounce_threshold_velocity: float = 0.5  # [m/s]
        max_depenetration_velocity: float = 1.0
        max_gpu_contact_pairs: int = 2**23
        default_buffer_size_multiplier: int = 5
        contact_collection: int = 2  # 0: never, 1: last sub-step, 2: all sub-steps


class TerrainCfg:
    """Terrain/ground plane configuration."""
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


class AssetCfg:
    """Robot asset configuration."""
    file: str = MISSING  # Path to URDF/MJCF file, can use {LEGGED_GYM_ROOT_DIR}
    name: str = "robot"
    
    # Asset loading options
    default_dof_drive_mode: int = 3  # 0: none, 1: pos, 2: vel, 3: effort
    collapse_fixed_joints: bool = True
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    fix_base_link: bool = False
    
    # Physical properties
    density: float = 0.001
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.01
    disable_gravity: bool = False
    
    # Collision
    self_collisions: int = 1  # 1 to disable, 0 to enable
    
    # Optional: redirect root state queries to a specific link
    redirect_root_to: str | None = None
    
    # Contact body names (used by LeggedRobot, not directly by wrapper)
    foot_name: str = "foot"
    penalize_contacts_on: List[str] = []
    terminate_after_contacts_on: List[str] = []


class InitStateCfg:
    """Initial state configuration."""
    pos: List[float] = [0.0, 0.0, 1.0]  # x, y, z [m]
    rot: List[float] = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w quaternion
    lin_vel: List[float] = [0.0, 0.0, 0.0]  # x, y, z [m/s]
    ang_vel: List[float] = [0.0, 0.0, 0.0]  # x, y, z [rad/s]


class EnvCfg:
    """Environment configuration."""
    num_envs: int = MISSING
    env_spacing: float = 3.0  # [m]


class DomainRandCfg:
    """Domain randomization configuration."""
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.25)
    
    randomize_base_mass: bool = False
    added_mass_range: Tuple[float, float] = (-1.0, 1.0)


class RewardsCfg:
    """Rewards configuration (only wrapper-relevant parts)."""
    soft_dof_pos_limit: float = 1.0  # Percentage of limits used for soft limits


class ViewerCfg:
    """Viewer/camera configuration."""
    pos: List[float] = [10.0, 0.0, 6.0]  # Camera position [m]
    lookat: List[float] = [11.0, 5.0, 3.0]  # Camera lookat point [m]


class IsaacGymWrapperCfg:
    """Top-level configuration for IsaacGymWrapper.
    
    This config contains all parameters needed to initialize the Isaac Gym simulation,
    create environments, and configure the viewer.
    
    Example usage:
        cfg = IsaacGymWrapperCfg()
        cfg.asset.file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        cfg.env.num_envs = 4096
        
        wrapper = IsaacGymWrapper(cfg, sim_device, graphics_device_id, headless)
    """
    sim: SimCfg = SimCfg()
    terrain: TerrainCfg = TerrainCfg()
    asset: AssetCfg = AssetCfg()
    init_state: InitStateCfg = InitStateCfg()
    env: EnvCfg = EnvCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()
    rewards: RewardsCfg = RewardsCfg()
    viewer: ViewerCfg = ViewerCfg()

