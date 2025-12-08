from __future__ import annotations

import os
from datetime import datetime

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from rsl_rl.algorithms import algorithm_dict
from rsl_rl.env import VecEnv
from rsl_rl.runners import runner_dict
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params, init_config


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.task_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, cfg_class: type):
        """Register a task with its config class.
        
        Args:
            name: Name of the task
            task_class: The environment class
            cfg_class: The config class type (not instance)
        """
        self.task_classes[name] = task_class
        self.task_cfgs[name] = cfg_class
    
    def get_task_class(self, name: str) -> type[VecEnv]:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> LeggedRobotCfg:
        """Get initialized config instance for a registered task.
        
        Instantiates the config class and recursively initializes all nested classes.
        """
        cfg = init_config(self.task_cfgs[name])
        return cfg
    
    def make(self, name, args=None, cfg=None, log_root="default"):
        """Creates environment, algorithm, and runner from a registered task name.

        Args:
            name (string): Name of a registered task.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called.
            cfg (LeggedRobotCfg, optional): Config to use. If None, will be loaded from registered task.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging.
                                      Logs will be saved in <log_root>/<date_time>_<run_name>.
                                      Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if no registered task corresponds to 'name'

        Returns:
            Runner: The created runner (contains env and algorithm)
            LeggedRobotCfg: the corresponding config
        """
        if args is None:
            args = get_args()

        # Check if task is registered
        if name not in self.task_classes:
            raise ValueError(f"Task with name: {name} was not registered")

        # Get task class and config
        env_class = self.get_task_class(name)
        if cfg is None:
            cfg = self.get_cfgs(name)

        # Override cfg from args
        cfg, _ = update_cfg_from_args(cfg, None, args)
        set_seed(cfg.seed)

        # Create environment
        sim_params = {"sim": class_to_dict(cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = env_class(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless
        )

        # Setup logging directory
        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + cfg.runner.run_name)
        
        cfg.runner.log_dir = log_dir

        # Create algorithm and runner
        runner_class = runner_dict[cfg.runner.runner_class_name]
        alg_class = algorithm_dict[cfg.runner.algorithm_class_name]

        alg = alg_class(env, cfg.algorithm)
        alg.init_storage(cfg.runner.num_steps_per_env)

        runner = runner_class(env, alg, cfg.runner)

        # Resume from checkpoint if specified
        if cfg.runner.resume:
            resume_path = get_load_path(log_root, load_run=cfg.runner.load_run, checkpoint=cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)

        return runner, cfg


# make global task registry
task_registry = TaskRegistry()
