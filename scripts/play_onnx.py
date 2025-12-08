import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
import onnxruntime as ort

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Import ONNX model first to check batch size constraints if possible, 
    # but here we assume batch_size=1 for standard exported models.
    # Warning: Standard exported ONNX models have fixed batch size 1.
    # We enforce num_envs = 1 to match the model.
    env_cfg.env.num_envs = 1 
    
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Hardcoded ONNX model path - Updated to user's path
    onnx_model_path = "/home/harry/Downloads/unitree_rl_gym/g1_10000.onnx"
    
    if not os.path.exists(onnx_model_path):
        print(f"Warning: ONNX model not found at {onnx_model_path}")
        # Fallback for testing if file moved
        # onnx_model_path = "path/to/your/model.onnx"
        return
    
    print(f"Loading ONNX model from: {onnx_model_path}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    input_details = ort_session.get_inputs()
    output_details = ort_session.get_outputs()
    input_names = [i.name for i in input_details]
    output_names = [o.name for o in output_details]
    print("Inputs:", input_names)
    print("Outputs:", output_names)
    
    # Determine expected observation shape from model
    # shape is usually [batch_size, obs_dim]
    model_obs_dim = input_details[0].shape[1]
    model_batch_size = input_details[0].shape[0]
    
    print(f"Model expects obs dim: {model_obs_dim}, batch size: {model_batch_size}")
    
    if env.num_obs != model_obs_dim:
        print(f"Warning: Environment obs dim ({env.num_obs}) != Model obs dim ({model_obs_dim}). Will slice observations.")

    # RNN configuration - MUST match the exported model
    # Based on export_onnx.py defaults
    rnn_hidden_size = 64
    rnn_num_layers = 1
    
    # Initialize hidden states
    # ONNX expects numpy arrays
    h = np.zeros((rnn_num_layers, env.num_envs, rnn_hidden_size), dtype=np.float32)
    c = np.zeros((rnn_num_layers, env.num_envs, rnn_hidden_size), dtype=np.float32)

    print("Starting simulation loop...")
    for i in range(10*int(env.max_episode_length)):
        # Prepare input dict
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        
        # Slice observations if dimensions mismatch
        if obs_np.shape[1] > model_obs_dim:
            obs_np = obs_np[:, :model_obs_dim]
        
        input_dict = {}
        # Map inputs based on names found in the model
        for name in input_names:
            if 'obs' in name:
                input_dict[name] = obs_np
            elif 'h_in' in name:
                input_dict[name] = h
            elif 'c_in' in name:
                input_dict[name] = c
        
        # Run inference
        try:
            outputs = ort_session.run(output_names, input_dict)
        except Exception as e:
            print(f"Inference error: {e}")
            break
        
        # Map outputs
        actions_np = None
        for idx, name in enumerate(output_names):
            if 'actions' in name:
                actions_np = outputs[idx]
            elif 'h_out' in name:
                h = outputs[idx]
            elif 'c_out' in name:
                c = outputs[idx]
        
        # Fallback if names don't match expected pattern, assume standard order [actions, h, c]
        if actions_np is None:
            actions_np = outputs[0]
            if len(outputs) > 1: h = outputs[1]
            if len(outputs) > 2: c = outputs[2]

        actions = torch.from_numpy(actions_np).to(env.device)
        
        # Step environment
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # Reset hidden states for done environments
        dones_np = dones.cpu().numpy()
        if np.any(dones_np):
            # Assuming h and c are (num_layers, batch, hidden_size)
            h[:, dones_np, :] = 0.0
            c[:, dones_np, :] = 0.0

if __name__ == '__main__':
    args = get_args()
    play(args)
