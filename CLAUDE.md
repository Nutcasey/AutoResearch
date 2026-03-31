# CLAUDE.md — AutoResearch Project Guide

## Project Overview

AutoResearch is Andrej Karpathy's autonomous AI research framework. The concept: point an AI coding agent at a small but real LLM training setup and let it experiment autonomously. The agent modifies code, trains for 5 minutes, checks if val loss improved, keeps or discards changes, and repeats — indefinitely. You sleep, wake up to ~100 experiments and (hopefully) a better model.

This is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The human writes `program.md` (agent instructions); the agent writes `train.py` (model code).

## Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch 2.9.1 (CUDA 12.8, bf16, `torch.compile`)
- **Attention:** Flash Attention 3 via `kernels` package (Hopper-native or community fallback)
- **Tokenizer:** `rustbpe` (training) + `tiktoken` (runtime), custom 8192-vocab BPE
- **Optimizer:** MuonAdamW — Muon (orthogonalized momentum) for 2D matrix params, AdamW for embeddings/scalars
- **Data:** HuggingFace `climbmix-400b-shuffle` parquet shards
- **Package Manager:** [uv](https://docs.astral.sh/uv/)
- **Other deps:** numpy, pandas, pyarrow, matplotlib, requests

## Directory Structure

```
AutoResearch/
├── prepare.py          # Data download, tokenizer training, dataloader, evaluation (READ-ONLY)
├── train.py            # Model, optimizer, training loop (AGENT MODIFIES THIS)
├── program.md          # Agent instructions / skill file (HUMAN MODIFIES THIS)
├── analysis.ipynb      # Jupyter notebook (likely for analysing experiment results)
├── progress.png        # Teaser image showing experiment progress
├── pyproject.toml      # Dependencies and uv config
├── uv.lock             # Locked dependency versions
├── .python-version     # Python 3.10
├── .gitignore          # Excludes results.tsv, CLAUDE.md, AGENTS.md, __pycache__, .venv, etc.
└── README.md           # Full project documentation
```

**Runtime artifacts (not in repo):**
- `~/.cache/autoresearch/data/` — parquet training shards
- `~/.cache/autoresearch/tokenizer/` — trained BPE tokenizer + token_bytes lookup
- `results.tsv` — experiment log (untracked)
- `run.log` — training output (untracked)

## Key Files

### `prepare.py` (DO NOT MODIFY)
- **Constants:** `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300s`, `EVAL_TOKENS=~20M`, `VOCAB_SIZE=8192`
- **Data prep:** Downloads parquet shards from HuggingFace, trains BPE tokenizer with `rustbpe`
- **Runtime utilities:** `Tokenizer` class, `make_dataloader()` (BOS-aligned best-fit packing), `evaluate_bpb()` (the ground truth metric)
- **Evaluation:** Bits per byte (BPB) — vocab-size-independent, sums per-token cross-entropy in nats, converts to bits/byte. Lower = better.

### `train.py` (AGENT'S PLAYGROUND)
- **GPTConfig:** depth×64 aspect ratio, 128 head dim, GQA-ready, sliding window pattern
- **Architecture:** RMSNorm, RoPE, Flash Attention 3, ReluSquared MLP, value embeddings (ResFormer), per-layer residual/x0 scaling, logit softcapping (15)
- **Optimizer:** MuonAdamW with compiled fused kernels, Nesterov momentum, polar express orthogonalization, NorMuon variance reduction, cautious weight decay
- **Hyperparameters:** All exposed as module-level constants (DEPTH=8, TOTAL_BATCH_SIZE=2^19, various LRs, warmup/warmdown ratios)
- **Training loop:** Time-budgeted (5 min wall clock excluding first 10 warmup steps), progress-based LR schedule, fast-fail on NaN/explosion

### `program.md` (HUMAN'S PLAYGROUND)
- Agent skill file defining the experiment loop protocol
- Setup: create branch, read files, verify data, init results.tsv
- Loop: modify train.py → commit → run → check results → keep/discard → repeat forever
- Logging: TSV format (commit, val_bpb, memory_gb, status, description)

## Build / Run

```bash
# Prerequisites: NVIDIA GPU (tested H100), Python 3.10+, uv

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# One-time data prep (~2 min)
uv run prepare.py
# Optional: download more shards
uv run prepare.py --num-shards 100

# Run a single training experiment (~5 min)
uv run train.py

# Autonomous mode: point your AI agent at program.md and let it go
```

## Architecture

### Model (GPT)
- **Embeddings:** Token embedding (`wte`) + per-layer value embeddings (alternating layers, ResFormer-style with input-dependent gating)
- **Blocks:** Pre-norm (RMSNorm) → CausalSelfAttention → Pre-norm → MLP, with residual scaling (`resid_lambdas`, `x0_lambdas`)
- **Attention:** RoPE positional encoding, QK-norm, Flash Attention 3, configurable sliding window pattern (S=half, L=full context)
- **MLP:** Linear → ReLU² → Linear (4× expansion)
- **Output:** Logit softcapping at 15, cross-entropy loss
- **Default config:** 8 layers, 512 dim (8×64), 4 heads, ~50M params

### Optimizer (MuonAdamW)
- **Matrix params (2D):** Muon — Nesterov momentum → polar express orthogonalization (Newton-Schulz) → NorMuon variance reduction → cautious weight decay
- **Embeddings/scalars:** AdamW with per-group LR scaling (∝ 1/√d_model)
- **LR schedule:** Linear warmup → constant → cosine warmdown to 0
- **All kernels torch.compiled** with `dynamic=False, fullgraph=True`

### Experiment Loop (Autonomous)
1. Establish baseline (run unmodified train.py)
2. Agent proposes change → edits train.py → git commit
3. Train 5 min → extract val_bpb from run.log
4. If improved: keep commit, advance branch
5. If not: git reset, try next idea
6. Log everything to results.tsv
7. Never stop until human intervenes

### Metric
- **val_bpb** (validation bits per byte) — the single metric that matters
- Computed on pinned validation shard (shard_06542)
- Vocab-size-independent so architecture changes are fairly compared

## Important Notes

- **NVIDIA GPU required** — needs CUDA, Flash Attention 3, bf16. H100 tested; see README for smaller GPU guidance.
- **`prepare.py` is read-only** — the evaluation harness, data loading, and constants must not be modified. This is the ground truth.
- **`train.py` is the only file the agent edits** — everything is fair game within it: architecture, hyperparams, optimizer, batch size, model size.
- **Fixed 5-minute time budget** — wall clock training time (excluding startup/compilation warmup of first 10 steps). Makes experiments comparable regardless of changes.
- **`.gitignore` excludes CLAUDE.md** — this file is gitignored by the project. It won't appear in `git status` unless force-added.
- **results.tsv is untracked** — experiment logs stay local, not committed.
- **MFU calculated against H100** — the `H100_BF16_PEAK_FLOPS` constant (989.5 TFLOPS) is hardcoded; MFU % is only meaningful on H100.
- **GC disabled during training** — Python garbage collection is frozen after step 0 to avoid ~500ms stalls, with periodic manual collection every 5000 steps.
- **Smaller GPU tips** in README: use TinyStories dataset, reduce vocab/seq_len/depth/batch_size, use "L" window pattern only.
