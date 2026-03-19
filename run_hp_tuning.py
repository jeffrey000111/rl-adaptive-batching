"""
Hyperparameter tuning for PPO and SAC.
Sweeps over learning rates, discount factors, and network sizes.
Saves results to results/hp_tuning/

Usage: python run_hp_tuning.py
"""

import sys, os, json, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.serving_env import InferenceServingEnv
from agents.sac_agent import ContinuousToDiscreteWrapper
from stable_baselines3 import PPO, SAC, DQN
import numpy as np

# ------ config ------
PATTERN = "bursty"          # hardest pattern, most useful for tuning
TIMESTEPS = 20000           # shorter runs for tuning (increase for final)
EVAL_EPS = 5
SEED = 42

LRS = [1e-4, 3e-4, 1e-3]
GAMMAS = [0.95, 0.99]
ARCHS = [[32, 32], [64, 64], [128, 128]]


def eval_model(model, pattern, episodes, use_wrapper=False):
    base = InferenceServingEnv(pattern=pattern, seed=99)
    env = ContinuousToDiscreteWrapper(base) if use_wrapper else base
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep + 200)
        total = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(a)
            total += r
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def tune_ppo():
    print("\n" + "=" * 60)
    print("  PPO HYPERPARAMETER TUNING")
    print("=" * 60)
    results = []

    for lr, gamma, arch in itertools.product(LRS, GAMMAS, ARCHS):
        tag = f"lr={lr}_g={gamma}_arch={arch}"
        print(f"\n  Training PPO: {tag}")
        try:
            env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
            model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma,
                        n_steps=512, batch_size=64, n_epochs=10,
                        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                        policy_kwargs=dict(net_arch=arch),
                        verbose=0, seed=SEED, device="cpu")
            model.learn(total_timesteps=TIMESTEPS)
            mean_r, std_r = eval_model(model, PATTERN, EVAL_EPS)
            result = {"algo": "PPO", "lr": lr, "gamma": gamma,
                      "arch": str(arch), "reward_mean": mean_r, "reward_std": std_r}
            results.append(result)
            print(f"    reward: {mean_r:.1f} +/- {std_r:.1f}")
        except Exception as e:
            print(f"    FAILED: {e}")

    return results


def tune_sac():
    print("\n" + "=" * 60)
    print("  SAC HYPERPARAMETER TUNING")
    print("=" * 60)
    results = []
    ent_coefs = ["auto", 0.1, 0.2]

    for lr, gamma, arch in itertools.product(LRS, GAMMAS, ARCHS):
        for ent in ent_coefs:
            tag = f"lr={lr}_g={gamma}_arch={arch}_ent={ent}"
            print(f"\n  Training SAC: {tag}")
            try:
                base = InferenceServingEnv(pattern=PATTERN, seed=SEED)
                env = ContinuousToDiscreteWrapper(base)
                model = SAC("MlpPolicy", env, learning_rate=lr, gamma=gamma,
                            buffer_size=5000, learning_starts=500,
                            batch_size=64, tau=0.005, ent_coef=ent,
                            policy_kwargs=dict(net_arch=arch),
                            verbose=0, seed=SEED, device="cpu")
                model.learn(total_timesteps=TIMESTEPS)
                mean_r, std_r = eval_model(model, PATTERN, EVAL_EPS, use_wrapper=True)
                result = {"algo": "SAC", "lr": lr, "gamma": gamma,
                          "arch": str(arch), "ent_coef": str(ent),
                          "reward_mean": mean_r, "reward_std": std_r}
                results.append(result)
                print(f"    reward: {mean_r:.1f} +/- {std_r:.1f}")
            except Exception as e:
                print(f"    FAILED: {e}")

    return results


if __name__ == "__main__":
    os.makedirs("results/hp_tuning", exist_ok=True)

    ppo_results = tune_ppo()
    sac_results = tune_sac()

    all_results = ppo_results + sac_results

    with open("results/hp_tuning/tuning_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # print best configs
    if ppo_results:
        best_ppo = max(ppo_results, key=lambda x: x["reward_mean"])
        print(f"\n\nBest PPO: lr={best_ppo['lr']}, gamma={best_ppo['gamma']}, "
              f"arch={best_ppo['arch']}, reward={best_ppo['reward_mean']:.1f}")

    if sac_results:
        best_sac = max(sac_results, key=lambda x: x["reward_mean"])
        print(f"Best SAC: lr={best_sac['lr']}, gamma={best_sac['gamma']}, "
              f"arch={best_sac['arch']}, ent={best_sac.get('ent_coef','auto')}, "
              f"reward={best_sac['reward_mean']:.1f}")

    print(f"\nResults saved to results/hp_tuning/tuning_results.json")
