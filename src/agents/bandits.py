"""
Multi-armed bandit baselines: Epsilon-Greedy and UCB.
Treat each batch size as an arm.
"""
import numpy as np

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

class EpsilonGreedy:
    def __init__(self, epsilon=0.1, n_actions=7):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)

    def predict(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values))

    def update(self, action, reward):
        self.counts[action] += 1
        alpha = 1.0 / self.counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

    def __repr__(self):
        return f"EpsilonGreedy(eps={self.epsilon})"


class UCB:
    def __init__(self, c=2.0, n_actions=7):
        self.c = c
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
        self.total_steps = 0

    def predict(self, obs):
        self.total_steps += 1
        # try each arm at least once
        for a in range(self.n_actions):
            if self.counts[a] == 0:
                return a
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_steps) / (self.counts + 1e-8)
        )
        return int(np.argmax(ucb_values))

    def update(self, action, reward):
        self.counts[action] += 1
        alpha = 1.0 / self.counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

    def __repr__(self):
        return f"UCB(c={self.c})"


def evaluate_bandit(env, agent, n_episodes=20):
    all_metrics = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_latencies = []
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(action, reward)
            done = terminated or truncated
            if info["batch_size"] > 0:
                ep_latencies.append(info["latency_ms"])
        metrics = env.get_metrics()
        metrics["p99_latency"] = np.percentile(ep_latencies, 99) if ep_latencies else 0
        all_metrics.append(metrics)
    return {
        "agent": str(agent),
        "avg_latency_ms": round(np.mean([m["avg_latency_ms"] for m in all_metrics]), 2),
        "p99_latency_ms": round(np.mean([m["p99_latency"] for m in all_metrics]), 2),
        "slo_attainment": round(np.mean([m["slo_attainment"] for m in all_metrics]), 4),
        "avg_throughput": round(np.mean([m["total_served"] for m in all_metrics]), 1),
    }
