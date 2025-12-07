from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.utils.task_registry import task_registry
from .base.legged_robot import LeggedRobot

task_registry.register("g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
