from legged_gym.envs.g1.baseline import G1BaselineEnv, G1BaselineCfg

from legged_gym.utils.task_registry import task_registry

task_registry.register("g1_baseline", G1BaselineEnv, G1BaselineCfg)
