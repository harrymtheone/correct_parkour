import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    runner, cfg = task_registry.make(name=args.task, args=args)
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
