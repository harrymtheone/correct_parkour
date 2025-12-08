import inspect
import os
import random
from dataclasses import MISSING

import numpy as np
import torch
from isaacgym import gymutil


def init_config(cfg_class):
    """Initializes a config class by instantiating all nested member classes recursively.
    
    Takes a config class type and returns an instance with all nested class attributes
    converted to instances.
    
    Args:
        cfg_class: A config class type (not an instance)
        
    Returns:
        An instance of cfg_class with all nested classes instantiated
        
    Raises:
        ValueError: If any attribute has the value MISSING
    """
    obj = cfg_class()
    _init_member_classes(obj, path=cfg_class.__name__)
    return obj


def _init_member_classes(obj, path: str):
    """Recursively instantiate all member classes of an object.
    
    Iterates over all attributes, finds classes, instantiates them,
    and recursively does the same for nested classes.
    
    Args:
        obj: The object to process
        path: The current path for error messages
        
    Raises:
        ValueError: If any attribute has the value MISSING
    """
    for key in dir(obj):
        if key.startswith("_"):
            continue
        var = getattr(obj, key)
        attr_path = f"{path}.{key}"
        if var is MISSING:
            raise ValueError(f"Missing required config value at: {attr_path}")
        if inspect.isclass(var):
            i_var = var()
            setattr(obj, key, i_var)
            _init_member_classes(i_var, path=attr_path)
        elif hasattr(var, "__dict__") and not callable(var):
            # If var is an instance (not a class), recurse to check its attributes
            _init_member_classes(var, path=attr_path)


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(cfg, _, args):
    """Update config from command line arguments.
    
    Args:
        cfg: The unified config object
        _: Unused (kept for backward compatibility)
        args: Command line arguments
        
    Returns:
        Tuple of (cfg, None) for backward compatibility
    """
    if cfg is not None:
        # num envs
        if args.num_envs is not None:
            cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            cfg.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg.runner.checkpoint = args.checkpoint

    return cfg, None


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for simulation and RL algorithm (cpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # Use single device for both sim and RL
    args.sim_device = args.device
    args.rl_device = args.device
    return args
