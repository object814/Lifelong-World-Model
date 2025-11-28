#!/usr/bin/env python3
# eval.py
# Usage example:
#   python eval.py --configs defaults dmc --logdir /path/to/logdir --checkpoint latest_model.pt --episodes 10 --label 3

import argparse
import copy
import functools
import pathlib
import sys
import os

# Ensure repo dir is importable (so dreamer.py and other modules resolve)
repo_root = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(repo_root))

# same as in dreamer.py
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')

import yaml
import torch
import numpy as np

import dreamer        # the script you ran (provides Dreamer, make_env, make_dataset, count_steps)
import tools          # provides Logger, load_episodes, etc.

to_np = lambda x: x.detach().cpu().numpy()

def load_config_from_names(parser):
  args, remaining = parser.parse_known_args()
  print(remaining)
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  return parser.parse_args(remaining)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True,
                      help='names from configs.yaml (same as training)')
  parser.add_argument('--logdir', required=True,
                      help='logdir used when training (where latest_model.pt sits)')
  parser.add_argument('--checkpoint', default='latest_model.pt',
                      help='checkpoint filename inside logdir (default latest_model.pt)')
  parser.add_argument('--episodes', type=int, default=1,
                      help='how many evaluation episodes to run')
  parser.add_argument('--envs', type=int, default=None,
                      help='how many parallel envs to create (default uses config.envs)')
  parser.add_argument('--label', type=int, default=0,
                      help='label/index for multi-task setups (matches training loop)')
  parser.add_argument('--device', default=None,
                      help='torch device to map checkpoint to (default from config.device)')
  args = parser.parse_args()

  # Load config
  config = load_config_from_names(parser)

  # convert some config entries to python types the same way dreamer.main does
  # dreamer.main eventually sets config.act = getattr(torch.nn, config.act)
  # but many other args are already plain values in configs.yaml
  # We'll ensure a few dependent values are adjusted similar to training:
  logdir = pathlib.Path(args.logdir).expanduser()
  config.traindir = config.traindir or (logdir / 'train_eps')
  config.evaldir  = config.evaldir  or (logdir / 'eval_eps')
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.log_every if hasattr(config, 'log_every') else config.eval_every
  # ensure time_limit exists before dividing (matches training behavior)
  if hasattr(config, 'time_limit'):
    config.time_limit //= config.action_repeat
  config.act = getattr(torch.nn, config.act)
  config.device = args.device or config.device

  # ensure directories exist
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)

  # compute current step count from saved train episodes (same function as training)
  step = dreamer.count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step)

  # load offline episodes if any (same as training)
  train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
  eval_eps = tools.load_episodes(config.evaldir, limit=1)

  # Build envs first (same as training), so we can inspect the action space.
  make = lambda mode: dreamer.make_env(config, logger, mode, train_eps, eval_eps, args.label)
  envs_to_create = args.envs if args.envs is not None else config.envs

  # Create train and eval env lists (training created both; we mirror that).
  train_envs = [make('train') for _ in range(envs_to_create)]
  eval_envs  = [make('eval')  for _ in range(envs_to_create)]

  # Inspect action space and set config.num_actions BEFORE making the agent.
  acts = train_envs[0].action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  # datasets (same batching used during training)
  train_dataset = dreamer.make_dataset(train_eps, config)
  train_dataset_vae = dreamer.make_dataset(train_eps, config, vae=True)
  eval_dataset = dreamer.make_dataset(eval_eps, config)

  # create the agent object (same constructor used during training)
  agent = dreamer.Dreamer(config, logger, train_dataset, train_dataset_vae).to(config.device)
  agent.requires_grad_(requires_grad=False)

  # Load checkpoint
  ckpt_path = (logdir / args.checkpoint)
  if not ckpt_path.exists():
    raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
  print(f'Loading agent checkpoint from {ckpt_path} (map to {config.device})')
  state = torch.load(str(ckpt_path), map_location=config.device)
  # If the checkpoint is a full state_dict as expected, load it
  try:
    agent.load_state_dict(state)
  except Exception as e:
    # fallback: maybe the checkpoint is wrapped (e.g., a dict with "model_state_dict")
    # attempt common keys:
    fallback_keys = ['state_dict', 'model_state_dict']
    found = False
    if isinstance(state, dict):
      for k in fallback_keys:
        if k in state:
          agent.load_state_dict(state[k])
          found = True
          break
    if not found:
      raise RuntimeError(f'Could not load checkpoint into agent: {e}')

  # Optional: load vae file if it exists (training saves agent._wm.vae in vae.pt)
  vae_path = logdir / 'vae.pt'
  if vae_path.exists():
    print(f'Loading VAE checkpoint from {vae_path}')
    try:
      # try assigning directly (works if vae.pt is the saved module)
      agent._wm.vae = torch.load(str(vae_path), map_location=config.device)
    except Exception as e:
      # try loading state_dict if available
      try:
        tmp = torch.load(str(vae_path), map_location=config.device)
        if isinstance(tmp, dict) and 'state_dict' in tmp and hasattr(agent._wm, 'vae'):
          agent._wm.vae.load_state_dict(tmp['state_dict'])
        else:
          print('vae.pt content not directly assignable; skipping detailed load:', e)
      except Exception as e2:
        print('Could not load vae.pt into agent._wm.vae, skipping auto-load:', e2)

  # If the training used multiple labels / tasks, set the current label as needed
  agent._wm.curr_label = args.label

  # Recreate policy_old/value_old if repository expects them elsewhere (optional)
  # The training script copies some internals, but for evaluation the primary need is the actor inside agent.
  try:
    agent._task_behavior.value_old.load_state_dict(agent._task_behavior.value.state_dict())
  except Exception:
    # Not critical for evaluation - continue
    pass

  # Now run evaluation episodes. This uses the same process_episode callback registered in make_env,
  # so it will call logger.scalar('eval_return', ...) and logger.scalar('eval_length', ...) then logger.write().
  eval_policy = functools.partial(agent, training=False)
  print('Running evaluation...')
  manual_label = input(f'Current label/index is {args.label}. Input new label/index or press Enter to keep:')
  if manual_label != '':
    args.label = int(manual_label)
  print(f'Using label/index {args.label} for evaluation.')
  tools.simulate(eval_policy, eval_envs, episodes=args.episodes, label=args.label)

  # Make sure we flush logger explicitly (should have been called via process_episode)
  logger.write()
  print('Evaluation finished; tensorboard logs in:', logdir)

  # cleanup envs
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass

if __name__ == '__main__':
  main()
