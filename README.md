# RL for Adaptive Batching in ML Inference Serving

Reinforcement Learning course project — Spring 2026, SJSU

## Team
- Yumeng Ren (018399628) — Environment design and simulation
- Haoran Jiang (018321927) — DQN and PPO agent implementation
- Brian Lam (014220934) — Baselines, evaluation, and reporting

## What this project is about

ML inference servers need to batch requests before sending them to the GPU. Bigger batches = better GPU utilization but higher latency. We're training an RL agent to pick the right batch size on the fly, instead of relying on fixed heuristics.

## Repo structure

```
├── README.md
├── data/
│   ├── AzureLLMInferenceTrace_code.csv    # Azure trace — code generation (8.8k requests)
│   └── AzureLLMInferenceTrace_conv.csv    # Azure trace — conversation (19.4k requests)
├── notebooks/                              # EDA and experiments
├── src/                                    # Source code
│   ├── env/                                # Gymnasium environment
│   ├── agents/                             # DQN, PPO, bandit baselines
│   └── baselines/                          # Heuristic baselines
└── docs/                                   # Proposal, check-ins, report
```

## Dataset

We use the [Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset) from Microsoft. Two CSV files with ~28k real production requests (timestamps, input tokens, output tokens). We use these to calibrate our simulator's arrival patterns and sequence length distributions.

## Algorithms

- **DQN** — discrete batch size selection
- **PPO** — policy gradient comparison
- **Epsilon-Greedy / UCB** — bandit baselines
- **Heuristics** — static batching, timeout-based, threshold rule
