import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params, init_config
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, cfg: LeggedRobotCfg):
        self.task_classes[name] = task_class
        self.cfgs[name] = init_config(cfg)
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfg(self, name) -> LeggedRobotCfg:
        return self.cfgs[name]
    
    def make_env(self, name, args=None, cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            cfg (Dict, optional): Config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if cfg is None:
            # load config files
            cfg = self.get_cfg(name)
        # override cfg from args (if specified)
        cfg = update_cfg_from_args(cfg, args)
        set_seed(cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, cfg

    def make_alg_runner(self, env, name=None, args=None, cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfg]:
        """ Creates the training algorithm either from a registered name or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            cfg (Dict, optional): Config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'cfg' are provided
            Warning: If both 'name' or 'cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'cfg' must be not None")
            # load config files
            cfg = self.get_cfg(name)
        else:
            if name is not None:
                print(f"'cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        cfg = update_cfg_from_args(cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, cfg.runner.run_name)
        
        cfg_dict = class_to_dict(cfg)
        runner = OnPolicyRunner(env, cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=cfg.runner.load_run, checkpoint=cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, cfg

# make global task registry
task_registry = TaskRegistry()
