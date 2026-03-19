# RL for Adaptive Batching in ML Inference Serving

Reinforcement Learning course project — Spring 2026, SJSU

## Team
- Yumeng Ren (018399628) — Environment design and simulation
- Haoran Jiang (018321927) — PPO and SAC agent implementation
- Brian Lam (014220934) — DQN baseline, heuristic baselines, evaluation and reporting

## What this project is about

ML inference servers need to batch requests before sending them to the GPU. Bigger batches = better GPU utilization but higher latency. We're training RL agents to pick the right batch size on the fly, instead of relying on fixed heuristics.

## Setup

```bash
pip install -r requirements.txt
```

## Run experiments

```bash
python run_all.py
```

This trains DQN (baseline), PPO, and SAC across three traffic patterns (steady, bursty, diurnal) and saves results to `results/all_results.json`.

## Repo structure

```
├── README.md
├── requirements.txt
├── run_all.py                              # run all experiments
├── data/
│   ├── AzureLLMInferenceTrace_code.csv
│   └── AzureLLMInferenceTrace_conv.csv
├── notebooks/
│   └── 01_eda_azure_traces.py              # EDA on Azure traces
├── src/
│   ├── env/
│   │   └── serving_env.py                  # Gymnasium environment
│   ├── agents/
│   │   ├── dqn_agent.py                    # DQN (baseline)
│   │   ├── ppo_agent.py                    # PPO (core)
│   │   └── sac_agent.py                    # SAC (core)
│   └── baselines/
│       └── heuristics.py                   # Static, Timeout, Threshold
├── results/                                # experiment outputs
└── docs/                                   # proposal, check-ins, report
```

## Algorithms

**Core comparison (per professor feedback):**
- **PPO** — policy gradient, handles non-stationary traffic well
- **SAC** — combines value-based and actor-critic, entropy-regularized

**Baselines:**
- **DQN** — value-based RL baseline
- **Static Batcher** — always same batch size
- **Timeout Batcher** — big batch if queue is full, small otherwise
- **Threshold Batcher** — scales batch size with queue length

## Dataset

[Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset) — ~28k real production requests used to calibrate our simulator.
