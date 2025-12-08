import os

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry


def play(args):
    cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    cfg.env.num_envs = min(cfg.env.num_envs, 100)
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.push_robots = False
    cfg.env.test = True
    cfg.runner.resume = True

    runner, cfg = task_registry.make(name=args.task, args=args, cfg=cfg)
    policy = runner.get_inference_policy(device=runner.env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    obs = runner.env.get_observations()
    for i in range(10 * int(runner.env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = runner.env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
