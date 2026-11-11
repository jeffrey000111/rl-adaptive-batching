#!/bin/bash
#SBATCH --job-name=rl-batching
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/hpc_extended_%j.log

module load python3/3.12.12
cd ~/rl-adaptive-batching

python3 -c "
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, 'src')
from env.serving_env import InferenceServingEnv
from agents.sac_agent import ContinuousToDiscreteWrapper
from stable_baselines3 import DQN, PPO, SAC
import numpy as np

TIMESTEPS = 500000
PATTERN = 'bursty'
SEED = 42
EVAL_EPS = 20

def evaluate(model, pattern, wrapper=False):
    base = InferenceServingEnv(pattern=pattern, seed=99)
    env = ContinuousToDiscreteWrapper(base) if wrapper else base
    rewards = []
    for ep in range(EVAL_EPS):
        obs, _ = env.reset(seed=ep+500)
        tot = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _, info = env.step(a)
            tot += rew
            if done: break
        rewards.append(tot)
    return float(np.mean(rewards)), float(np.std(rewards))

results = {}

# DQN 500k
print('Training DQN 500k...')
env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
model = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=128, gamma=0.99, policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=SEED, device='cpu')
model.learn(total_timesteps=TIMESTEPS)
m, s = evaluate(model, PATTERN)
results['DQN_500k'] = {'reward': m, 'std': s}
print(f'DQN 500k: {m:.1f} +/- {s:.1f}')

# PPO 500k
print('Training PPO 500k...')
env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
model = PPO('MlpPolicy', env, learning_rate=1e-3, n_steps=1024, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=SEED, device='cpu')
model.learn(total_timesteps=TIMESTEPS)
m, s = evaluate(model, PATTERN)
results['PPO_500k'] = {'reward': m, 'std': s}
print(f'PPO 500k: {m:.1f} +/- {s:.1f}')

# SAC 500k
print('Training SAC 500k...')
base = InferenceServingEnv(pattern=PATTERN, seed=SEED)
env = ContinuousToDiscreteWrapper(base)
model = SAC('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=128, gamma=0.99, tau=0.005, ent_coef='auto', policy_kwargs=dict(net_arch=[32, 32]), verbose=1, seed=SEED, device='cpu')
model.learn(total_timesteps=TIMESTEPS)
m, s = evaluate(model, PATTERN, wrapper=True)
results['SAC_500k'] = {'reward': m, 'std': s}
print(f'SAC 500k: {m:.1f} +/- {s:.1f}')

# Save
os.makedirs('results/hpc_extended', exist_ok=True)
with open('results/hpc_extended/hpc_extended_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Done! Saved to results/hpc_extended/hpc_extended_results.json')
print(json.dumps(results, indent=2))
"
