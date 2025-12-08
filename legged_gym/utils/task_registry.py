import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, Runner, runner_dict
from rsl_rl.algorithms import algorithm_dict
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params, init_config
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

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
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> LeggedRobotCfg:
        """Get initialized config instance for a registered task.
        
        Instantiates the config class and recursively initializes all nested classes.
        """
        cfg = init_config(self.task_cfgs[name])
        return cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

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
        if env_cfg is None:
            # load config files
            env_cfg = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, cfg=None, log_root="default"):
        """ Creates the training algorithm either from a registered name or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            cfg (LeggedRobotCfg, optional): Config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'cfg' are provided
            Warning: If both 'name' or 'cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            LeggedRobotCfg: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'cfg' must be not None")
            # load config files
            cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        cfg, _ = update_cfg_from_args(cfg, None, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + cfg.runner.run_name)
        
        runner_class_name = getattr(cfg.runner, 'runner_class_name', 'Runner')
        runner_class = runner_dict[runner_class_name]

        # New Runner implementation
        alg_class_name = getattr(cfg.runner, 'algorithm_class_name', 'baseline')
        alg_class = algorithm_dict[alg_class_name]
        
        # Create ActorCritic
        policy_class_name = getattr(cfg.runner, 'policy_class_name', 'ActorCritic')
        if policy_class_name == 'ActorCritic':
            actor_critic_class = ActorCritic
        elif policy_class_name == 'ActorCriticRecurrent':
            actor_critic_class = ActorCriticRecurrent
        else:
            actor_critic_class = eval(policy_class_name)

        if env.num_privileged_obs is not None:
            num_critic_obs = env.num_privileged_obs
        else:
            num_critic_obs = env.num_obs
        
        # Set log_dir in cfg.runner as it's expected by Runner
        cfg.runner.log_dir = log_dir
        
        actor_critic = actor_critic_class(env.num_obs, num_critic_obs, env.num_actions, **class_to_dict(cfg.policy)).to(args.rl_device)
        
        alg = alg_class(actor_critic, device=args.rl_device, **class_to_dict(cfg.algorithm))
        alg.init_storage(env.num_envs, cfg.runner.num_steps_per_env, [env.num_obs], [env.num_privileged_obs], [env.num_actions])
        
        runner = runner_class(env, alg, cfg.runner, device=args.rl_device)
        
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
