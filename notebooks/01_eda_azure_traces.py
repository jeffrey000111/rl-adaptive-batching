# %% [markdown]
# # Azure LLM Inference Trace - Exploratory Data Analysis
# Quick look at Microsoft's production traces to calibrate our simulator.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# load the two traces
code_df = pd.read_csv("../data/AzureLLMInferenceTrace_code.csv")
conv_df = pd.read_csv("../data/AzureLLMInferenceTrace_conv.csv")

print(f"Code trace: {len(code_df)} requests")
print(f"Conv trace: {len(conv_df)} requests")
print(f"Total: {len(code_df) + len(conv_df)} requests")

# %%
code_df.head()

# %%
conv_df.head()

# %%
# basic stats
print("=== Code Trace ===")
print(code_df[["ContextTokens", "GeneratedTokens"]].describe())
print("\n=== Conversation Trace ===")
print(conv_df[["ContextTokens", "GeneratedTokens"]].describe())

# %%
# distribution of input/output tokens
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(code_df["ContextTokens"], bins=50, alpha=0.7, color="steelblue")
axes[0, 0].set_title("Code: Input Tokens (ContextTokens)")
axes[0, 0].set_xlabel("Tokens")

axes[0, 1].hist(code_df["GeneratedTokens"], bins=50, alpha=0.7, color="coral")
axes[0, 1].set_title("Code: Output Tokens (GeneratedTokens)")
axes[0, 1].set_xlabel("Tokens")

axes[1, 0].hist(conv_df["ContextTokens"], bins=50, alpha=0.7, color="steelblue")
axes[1, 0].set_title("Conversation: Input Tokens")
axes[1, 0].set_xlabel("Tokens")

axes[1, 1].hist(conv_df["GeneratedTokens"], bins=50, alpha=0.7, color="coral")
axes[1, 1].set_title("Conversation: Output Tokens")
axes[1, 1].set_xlabel("Tokens")

plt.tight_layout()
plt.savefig("../notebooks/token_distributions.png", dpi=100)
plt.show()

# %%
# parse timestamps and look at arrival patterns
code_df["TIMESTAMP"] = pd.to_datetime(code_df["TIMESTAMP"])
conv_df["TIMESTAMP"] = pd.to_datetime(conv_df["TIMESTAMP"])

# inter-arrival times
code_df["iat_ms"] = code_df["TIMESTAMP"].diff().dt.total_seconds() * 1000
conv_df["iat_ms"] = conv_df["TIMESTAMP"].diff().dt.total_seconds() * 1000

print("=== Inter-Arrival Times (ms) ===")
print("Code trace:")
print(code_df["iat_ms"].describe())
print("\nConversation trace:")
print(conv_df["iat_ms"].describe())

# %%
# arrival rate over time (requests per second, 1-second windows)
code_df["second"] = code_df["TIMESTAMP"].dt.floor("1s")
code_rate = code_df.groupby("second").size()

conv_df["second"] = conv_df["TIMESTAMP"].dt.floor("1s")
conv_rate = conv_df.groupby("second").size()

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

axes[0].plot(range(len(code_rate)), code_rate.values, alpha=0.7, linewidth=0.5)
axes[0].set_title("Code Trace: Requests per Second")
axes[0].set_ylabel("req/s")

axes[1].plot(range(len(conv_rate)), conv_rate.values, alpha=0.7, linewidth=0.5, color="coral")
axes[1].set_title("Conversation Trace: Requests per Second")
axes[1].set_ylabel("req/s")
axes[1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig("../notebooks/arrival_rates.png", dpi=100)
plt.show()

# %%
# key takeaways for simulator calibration
print("=== Key Numbers for Simulator ===")
print(f"Code trace median input tokens: {code_df['ContextTokens'].median():.0f}")
print(f"Code trace median output tokens: {code_df['GeneratedTokens'].median():.0f}")
print(f"Conv trace median input tokens: {conv_df['ContextTokens'].median():.0f}")
print(f"Conv trace median output tokens: {conv_df['GeneratedTokens'].median():.0f}")
print(f"Code trace mean arrival rate: {code_rate.mean():.1f} req/s")
print(f"Conv trace mean arrival rate: {conv_rate.mean():.1f} req/s")
print(f"Code trace max burst: {code_rate.max()} req/s")
print(f"Conv trace max burst: {conv_rate.max()} req/s")
