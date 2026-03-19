"""
Run all baselines + RL agents and compare results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.env.inference_env import InferenceServingEnv
from src.baselines.heuristics import StaticBatcher, TimeoutBatcher, ThresholdBatcher, evaluate_baseline
from src.agents.bandits import EpsilonGreedy, UCB, evaluate_bandit
import json


def run_comparison(traffic="steady"):
    print(f"\n{'='*60}")
    print(f"  Traffic pattern: {traffic}")
    print(f"{'='*60}")

    env = InferenceServingEnv(traffic_pattern=traffic)
    results = []

    # heuristic baselines
    for agent in [StaticBatcher(8), StaticBatcher(16), TimeoutBatcher(), ThresholdBatcher()]:
        r = evaluate_baseline(env, agent)
        results.append(r)
        print(f"  {r['agent']:30s} | lat={r['avg_latency_ms']:7.1f} | p99={r['p99_latency_ms']:7.1f} | slo={r['slo_attainment']:.3f} | thru={r['avg_throughput']:6.0f}")

    # bandit baselines
    for agent in [EpsilonGreedy(0.1), UCB(2.0)]:
        r = evaluate_bandit(env, agent)
        results.append(r)
        print(f"  {r['agent']:30s} | lat={r['avg_latency_ms']:7.1f} | p99={r['p99_latency_ms']:7.1f} | slo={r['slo_attainment']:.3f} | thru={r['avg_throughput']:6.0f}")

    return results


if __name__ == "__main__":
    all_results = {}
    for pattern in ["steady", "bursty", "diurnal"]:
        all_results[pattern] = run_comparison(pattern)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/baseline_comparison.json")
