"""Measure BERT-base batch inference latency on the local CPU (post-hoc check).

Tests the *shape* of the processing-time model,
``time = fixed + per_token * batch_size * length``, and in particular its
linear extrapolation above batch size 8. It does not measure the T4
parameters the simulator uses. See the pre-registration, Deviation 4.

Method: Hugging Face ``bert-base-uncased`` encoder (no task head), PyTorch,
inference mode, a fixed number of intra-op threads, random token ids with a
full attention mask. Every configuration is warmed up first; then each round
times every configuration once in a freshly shuffled order (to spread slow
drift such as thermal throttling evenly), and the median over rounds is
reported.

Writes results/real_latency/cpu_bert_base.json.

Usage:
    python scripts/measure_cpu_latency.py
"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/real_latency/cpu_bert_base.json"
MODEL = "bert-base-uncased"
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
LENGTHS = (128, 384)
THREADS = 4
WARMUP = 5
ROUNDS = 20
SEED = 0


def main() -> None:
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    model = AutoModel.from_pretrained(MODEL).eval()
    generator = torch.Generator().manual_seed(SEED)
    inputs = {}
    for batch in BATCH_SIZES:
        for length in LENGTHS:
            ids = torch.randint(1000, 30000, (batch, length), generator=generator)
            inputs[(batch, length)] = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    samples: dict[tuple[int, int], list[float]] = {key: [] for key in inputs}
    order = list(inputs)
    shuffler = random.Random(SEED)
    started = time.time()
    with torch.inference_mode():
        for key in order:
            for _ in range(WARMUP):
                model(**inputs[key])
        for round_index in range(ROUNDS):
            shuffler.shuffle(order)
            for key in order:
                t0 = time.perf_counter()
                model(**inputs[key])
                samples[key].append(time.perf_counter() - t0)
            print(f"round {round_index + 1}/{ROUNDS} done", flush=True)

    record = {
        "exploratory": True,
        "purpose": "check the shape of the processing-time model on a real CPU",
        "model": MODEL,
        "environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "intra_op_threads": THREADS,
            "inter_op_threads": 1,
        },
        "method": {"warmup_per_config": WARMUP, "rounds": ROUNDS, "statistic": "median",
                   "order": "shuffled every round", "seed": SEED},
        "measurements": [
            {
                "batch_size": batch,
                "seq_len": length,
                "median_s": float(np.median(samples[(batch, length)])),
                "p10_s": float(np.percentile(samples[(batch, length)], 10)),
                "p90_s": float(np.percentile(samples[(batch, length)], 90)),
                "samples_s": samples[(batch, length)],
            }
            for batch in BATCH_SIZES
            for length in LENGTHS
        ],
        "wall_seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} in {record['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
