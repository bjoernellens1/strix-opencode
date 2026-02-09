# strix-opencode

A reproducible **Strix Halo** dev-stack for **Ubuntu + distrobox** that runs local LLM inference servers and integrates with [OpenCode](https://opencode.ai) via a custom [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) plugin.

Optimized for a **single-user workflow** — interactive latency and stability over multi-user throughput.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Startup Modes](#startup-modes)
- [Model Roles & Endpoints](#model-roles--endpoints)
- [Model Download Notes](#model-download-notes)
- [Environment Variables](#environment-variables)
- [Scripts Reference](#scripts-reference)
- [OpenCode Configuration](#opencode-configuration)
- [oh-my-opencode Plugin](#oh-my-opencode-plugin)
- [Distrobox Notes (Ubuntu)](#distrobox-notes-ubuntu)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This project runs a **4-tier local AI inference stack** on a single AMD Strix Halo system (128 GB shared UMA memory):

| Tier | Role | Backend | Model | Context |
|------|------|---------|-------|---------|
| **1 Orchestrator** | Planning, coordination, delegation | vLLM (GPU, BF16) | Qwen2.5-14B-Instruct | 64K |
| **2 Coder** | Code generation, tool use | vLLM (GPU, BF16) | Qwen3-Coder-30B-A3B-Instruct | 48K |
| **3 Reviewer** | Escalation, deep review, architecture | llama.cpp (CPU) | Llama-3.3-70B Q4_K_M | 32K |
| **0 Utility** | Fast utility tasks, summaries | llama.cpp (CPU) | Qwen2.5-3B Q4_K_M | 8K |

**Key design decisions:**
- **All BF16 on GPU** — no quantization works on gfx1151 (FP8, AWQ, GPTQ all fail; see [DECISIONS.md](DECISIONS.md))
- **CPU tiers via llama.cpp** — 70B reviewer runs INT4 GGUF on CPU without competing for GPU memory
- **Staggered GPU startup** — prevents shared-memory race condition during vLLM memory profiling
- **0.95 total GPU utilization** — tight for single-user; swap file + earlyoom recommended

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  OpenCode  (TUI / agent runtime)                                     │
│  config: opencode/opencode.jsonc                                     │
│  plugin: opencode/oh-my-opencode/                                    │
├─────────────┬──────────────┬──────────────────┬─────────────────────┤
│             │              │                  │                       │
│  Primary    │  Coder       │  Reviewer        │  Utility             │
│  (orch)     │  (subagent)  │  (subagent)      │  (subagent)          │
│             │              │                  │                       │
│ ┌───────────┴┐ ┌──────────┴──┐ ┌─────────────┴──┐ ┌──────────────┐ │
│ │ vLLM :8001 │ │ vLLM :8002  │ │ llama.cpp :8011│ │llama.cpp:8012│ │
│ │ 14B GPU    │ │ 30B GPU     │ │ 70B CPU        │ │ 3B CPU       │ │
│ │ BF16       │ │ BF16        │ │ Q4_K_M GGUF    │ │ Q4_K_M GGUF  │ │
│ └────────────┘ └─────────────┘ └────────────────┘ └──────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
        GPU (0.35)      GPU (0.60)       CPU (16 threads)  CPU (8 threads)
              └──── total: 0.95 ────┘
```

---

## Prerequisites

- **Ubuntu** (22.04+ recommended) with AMD Strix Halo GPU
- **Docker** (with Compose v2) — `docker compose` must work
- **GPU access**: `/dev/kfd` and `/dev/dri` exposed to containers
- **Distrobox** (optional; see [Distrobox Notes](#distrobox-notes-ubuntu))
- **curl** (for the health script)
- **~128 GB swap file** recommended (shared memory system)
- **earlyoom** recommended (`sudo apt install earlyoom`)
- **HuggingFace token** for gated models: set `HF_TOKEN` in `.env`

### System Stability (Recommended)

With 95% GPU utilization on shared UMA memory, system stability measures are important:

```bash
# 1. Enlarge swap to 128 GB
sudo swapoff /swap.img && sudo rm /swap.img
sudo fallocate -l 128G /swap.img && sudo chmod 600 /swap.img
sudo mkswap /swap.img && sudo swapon /swap.img

# 2. Install earlyoom (prevents hard freezes)
sudo apt install earlyoom && sudo systemctl enable --now earlyoom

# 3. Increase swappiness (push inactive pages to swap aggressively)
sudo sysctl vm.swappiness=80
echo 'vm.swappiness=80' | sudo tee /etc/sysctl.d/99-swappiness.conf
```

---

## Repository Structure

```
strix-opencode/
├── README.md                                  # This file
├── DECISIONS.md                               # Architecture decisions, research log, quantization results
├── .env.example                               # Template for environment variables
├── .gitignore                                 # Ignores models/, .env, secrets, caches
├── strix-opencode.md                          # Original build instructions
├── compose/
│   ├── vllm.yml                               # GPU tiers: orchestrator (14B) + coder (30B)
│   ├── cpu.yml                                # CPU tiers: reviewer (70B) + utility (3B)
│   ├── fallback.vulkan-radv.orch.yml          # Legacy: llama.cpp orchestrator (Vulkan RADV)
│   └── fallback.rocm-6.4.4-rocwmma.orch.yml  # Legacy: llama.cpp orchestrator (ROCm rocWMMA)
├── opencode/
│   ├── opencode.jsonc                         # OpenCode provider/agent configuration
│   └── oh-my-opencode/                        # Plugin (git submodule)
├── scripts/
│   ├── up                                     # Start services (gpu | cpu | full | hybrid-*)
│   ├── down                                   # Stop all services
│   ├── health                                 # Check all 4 endpoints
│   └── switch-orch                            # Switch orchestrator (vllm | cloud)
├── models/                                    # GGUF files for llama.cpp (git-ignored)
└── .opencode/
    └── oh-my-opencode.json                    # Template: agent maxTokens for local models
```

---

## Quick Start

```bash
# 1. Clone and enter
git clone --recurse-submodules https://github.com/bjoernellens1/strix-opencode.git
cd strix-opencode

# 2. Set up environment
cp .env.example .env
# Edit .env — set HF_TOKEN for gated model downloads

# 3. Download GGUF models for CPU tiers (reviewer + utility)
mkdir -p models
# Llama-3.3-70B for reviewer:
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
  llama-3.3-70b-instruct.Q4_K_M.gguf --local-dir models
# Qwen2.5-3B for utility:
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct.Q4_K_M.gguf --local-dir models

# 4a. Start GPU tiers only (most common)
./scripts/up gpu

# 4b. Or start all 4 tiers
./scripts/up full

# 5. Verify health
./scripts/health

# 6. Run OpenCode
opencode --config /path/to/strix-opencode/opencode/opencode.jsonc
```

First start is slow — vLLM downloads model weights (~85 GB total for both GPU models). Subsequent starts use the HuggingFace cache.

---

## Startup Modes

| Mode | Command | What starts |
|------|---------|-------------|
| **gpu** (default) | `./scripts/up` or `./scripts/up gpu` | vLLM orchestrator (:8001) + vLLM coder (:8002) |
| **cpu** | `./scripts/up cpu` | llama.cpp reviewer (:8011) + llama.cpp utility (:8012) |
| **full** | `./scripts/up full` | All 4 tiers |
| **hybrid-radv** | `./scripts/up hybrid-radv` | vLLM + llama.cpp orch (Vulkan RADV, legacy) |
| **hybrid-rocm** | `./scripts/up hybrid-rocm` | vLLM + llama.cpp orch (ROCm rocWMMA, legacy) |

### Stopping

```bash
./scripts/down    # Stops all containers across all compose files
```

---

## Model Roles & Endpoints

| Tier | Role | Port | Default Model | Backend |
|------|------|------|---------------|---------|
| 1 | Orchestrator | `http://127.0.0.1:8001/v1` | Qwen/Qwen2.5-14B-Instruct | vLLM (GPU) |
| 2 | Coder | `http://127.0.0.1:8002/v1` | Qwen/Qwen3-Coder-30B-A3B-Instruct | vLLM (GPU) |
| 3 | Reviewer | `http://127.0.0.1:8011/v1` | Llama-3.3-70B Q4_K_M GGUF | llama.cpp (CPU) |
| 0 | Utility | `http://127.0.0.1:8012/v1` | Qwen2.5-3B Q4_K_M GGUF | llama.cpp (CPU) |

All endpoints serve an OpenAI-compatible `/v1` API.

### Cloud Planner (Optional)

You can route the orchestrator to a cloud provider while keeping local coder/reviewer/utility:

```bash
# Set cloud credentials in .env or environment
export OPENAI_API_KEY=sk-...
export CLOUD_PLANNER_MODEL=gpt-5

./scripts/switch-orch cloud
# To switch back:
./scripts/switch-orch vllm
```

---

## Model Download Notes

### vLLM Models (GPU Tiers)

vLLM pulls HuggingFace models **automatically** on first run. Models are cached in `$HF_HOME` (default: `~/.cache/huggingface`).

- Set `HF_TOKEN` in `.env` for gated models
- First download: ~28 GB (14B) + ~57 GB (30B) = **~85 GB total**
- Subsequent starts use the cache

### llama.cpp GGUF Models (CPU Tiers)

GGUF files must be **pre-downloaded** into `./models/`:

```bash
mkdir -p models

# Tier 3 Reviewer: Llama-3.3-70B (~40 GB)
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
  llama-3.3-70b-instruct.Q4_K_M.gguf --local-dir models

# Tier 0 Utility: Qwen2.5-3B (~2 GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct.Q4_K_M.gguf --local-dir models
```

Update `REVIEWER_GGUF` and `UTILITY_GGUF` in `.env` if your filenames differ.

---

## Environment Variables

All configurable values are in `.env` (copied from `.env.example`).

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated models |
| `HF_HOME` | `${HOME}/.cache/huggingface` | HuggingFace model cache |
| `VLLM_CACHE` | `${HOME}/.cache/vllm` | vLLM runtime cache |
| `LLAMA_MODELS_DIR` | `./models` | Directory containing GGUF files |

### Ports

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_PORT` | `8001` | Tier 1: vLLM orchestrator |
| `CODER_PORT` | `8002` | Tier 2: vLLM coder |
| `REVIEWER_PORT` | `8011` | Tier 3: llama.cpp reviewer |
| `UTILITY_PORT` | `8012` | Tier 0: llama.cpp utility |

### GPU Tiers (vLLM)

> **Important:** `ORCH_GPU_UTIL + CODER_GPU_UTIL` must be **<= 1.0**. Both share the same physical GPU memory on Strix Halo.

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_MODEL` | `Qwen/Qwen2.5-14B-Instruct` | Orchestrator model (14B dense) |
| `CODER_MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Coder model (30B MoE) |
| `ORCH_GPU_UTIL` | `0.35` | GPU memory fraction for orchestrator (~44.8 GB) |
| `CODER_GPU_UTIL` | `0.60` | GPU memory fraction for coder (~76.8 GB) |
| `ORCH_MAX_LEN` | `65536` | Orchestrator context window (64K tokens) |
| `CODER_MAX_LEN` | `49152` | Coder context window (48K tokens) |
| `ORCH_KV_CACHE_DTYPE` | `auto` | KV cache dtype (**must be `auto`** on gfx1151) |
| `CODER_KV_CACHE_DTYPE` | `auto` | KV cache dtype (**must be `auto`** on gfx1151) |
| `ORCH_MAX_NUM_SEQS` | `4` | Max concurrent sequences (single-user) |
| `CODER_MAX_NUM_SEQS` | `4` | Max concurrent sequences (single-user) |

### CPU Tiers (llama.cpp)

| Variable | Default | Description |
|----------|---------|-------------|
| `REVIEWER_GGUF` | `llama-3.3-70b-instruct.Q4_K_M.gguf` | GGUF filename for reviewer |
| `REVIEWER_CTX` | `32768` | Reviewer context window (32K) |
| `REVIEWER_THREADS` | `16` | CPU threads for reviewer |
| `UTILITY_GGUF` | `qwen2.5-3b-instruct.Q4_K_M.gguf` | GGUF filename for utility |
| `UTILITY_CTX` | `8192` | Utility context window (8K) |
| `UTILITY_THREADS` | `8` | CPU threads for utility |

### Cloud Planner (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(unset)* | API key for cloud planner |
| `CLOUD_PLANNER_MODEL` | *(unset)* | Model name (e.g., `gpt-5`) |

---

## Scripts Reference

All scripts live in `scripts/` and are executable (`chmod +x`).

### `scripts/up`

Start the inference stack:

```bash
./scripts/up              # Default: GPU tiers only
./scripts/up gpu          # Explicit: GPU tiers only
./scripts/up cpu          # CPU tiers only
./scripts/up full         # All 4 tiers
./scripts/up hybrid-radv  # Legacy: vLLM + Vulkan RADV llama.cpp
./scripts/up hybrid-rocm  # Legacy: vLLM + ROCm llama.cpp
```

On first run, `.env.example` is automatically copied to `.env` if `.env` does not exist.

### `scripts/down`

Stop all services across all compose files:

```bash
./scripts/down
```

### `scripts/health`

Check all 4 endpoints (queries `/v1/models` on each port):

```bash
./scripts/health
```

Outputs the model list from each service. Services that aren't running show "(not responding)" — this is expected when not using `full` mode.

### `scripts/switch-orch`

Switch the OpenCode primary agent backend:

```bash
./scripts/switch-orch vllm   # Use local vLLM orchestrator (:8001)
./scripts/switch-orch cloud  # Use cloud planner (requires OPENAI_API_KEY)
```

This edits `opencode/opencode.jsonc` in-place. Coder, reviewer, and utility agents are unaffected.

---

## OpenCode Configuration

The file `opencode/opencode.jsonc` defines providers and agent-to-provider mappings:

### Providers

| Provider ID | Backend | Endpoint |
|------------|---------|----------|
| `local_orch` | vLLM GPU | `http://127.0.0.1:8001/v1` |
| `local_coder` | vLLM GPU | `http://127.0.0.1:8002/v1` |
| `local_reviewer` | llama.cpp CPU | `http://127.0.0.1:8011/v1` |
| `local_utility` | llama.cpp CPU | `http://127.0.0.1:8012/v1` |
| `cloud_planner` | OpenAI API | cloud |

### Agents

| Agent | Role | Default Provider |
|-------|------|-----------------|
| `primary` | Orchestrator/Planner | `local_orch:orch` |
| `coder` | Code generation | `local_coder:coder` |
| `reviewer` | Escalation review | `local_reviewer:reviewer` |
| `utility` | Quick utility tasks | `local_utility:utility` |

### Agent Token Limits

The template `.opencode/oh-my-opencode.json` caps output tokens per agent to fit local model context windows:

| Agent | maxTokens | Tier | Rationale |
|-------|-----------|------|-----------|
| sisyphus | 32,768 | Orch (64K) | 32K output + ~32K input |
| hephaestus, sisyphus-junior | 24,576 | Coder (48K) | 24K output + ~24K input |
| oracle, prometheus, metis, momus | 16,384 | Reviewer (32K) | 16K output + ~16K input |
| librarian, explore, atlas | 4,096 | Utility (8K) | 4K output + ~4K input |

Copy this file to `.opencode/` in any target project. Remove/raise limits when using cloud models.

---

## Memory Budget

### GPU Allocation (128 GB shared UMA)

| Component | Fraction | Memory | Weights (BF16) | KV Budget |
|-----------|----------|--------|----------------|-----------|
| Coder (30B MoE) | 0.60 | ~76.8 GB | ~57 GB | ~19.8 GB |
| Orchestrator (14B) | 0.35 | ~44.8 GB | ~28 GB | ~16.8 GB |
| System headroom | 0.05 | ~6.4 GB | — | — |

### KV Cache Fit

| Tier | Model | Context | KV Size | Budget | Slack |
|------|-------|---------|---------|--------|-------|
| 1 Orch | Qwen2.5-14B | 64K | ~10.0 GB | 16.8 GB | 6.8 GB |
| 2 Coder | Qwen3-Coder-30B | 48K | ~4.5 GB | 19.8 GB | 15.3 GB |

### CPU Tiers (system RAM)

| Tier | Model | GGUF Size | RAM at Load |
|------|-------|-----------|-------------|
| 3 Reviewer | Llama-3.3-70B Q4_K_M | ~40 GB | ~42 GB |
| 0 Utility | Qwen2.5-3B Q4_K_M | ~2 GB | ~3 GB |

CPU tiers share the same physical RAM. Not expected to run concurrently with both GPU tiers at full KV pressure.

> For detailed memory math, KV cache calculations, and vLLM allocation internals, see [DECISIONS.md](DECISIONS.md).

---

## oh-my-opencode Plugin

This repository uses a custom fork of [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) as a git submodule at `opencode/oh-my-opencode/`.

### Cloning with Submodules

```bash
git clone --recurse-submodules https://github.com/bjoernellens1/strix-opencode.git
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

### Replacing with Your Own Fork

```bash
# 1. Remove existing submodule
git rm opencode/oh-my-opencode
rm -rf .git/modules/opencode/oh-my-opencode

# 2. Add your fork
git submodule add https://github.com/YOUR_USER/oh-my-opencode opencode/oh-my-opencode

# 3. Commit
git add .gitmodules opencode/oh-my-opencode
git commit -m "Replace oh-my-opencode submodule with fork"
```

### Updating the Submodule

```bash
cd opencode/oh-my-opencode
git pull origin dev
cd ../..
git add opencode/oh-my-opencode
git commit -m "Update oh-my-opencode submodule"
```

---

## Distrobox Notes (Ubuntu)

**Ubuntu + distrobox** is the recommended environment. Distrobox provides seamless GPU passthrough without modifying your host system.

### Why Distrobox?

- Direct access to `/dev/kfd` and `/dev/dri` GPU devices
- Host networking by default — no port mapping needed
- User home directory shared — HuggingFace cache accessible
- Works alongside Docker on the same host

### Example

```bash
distrobox create \
  --name strix-vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri"

distrobox enter strix-vllm
```

You don't need distrobox for the compose stack — `docker compose` directly on the host works fine if Docker has GPU access. Distrobox is most useful for interactive debugging inside the toolbox images.

---

## Troubleshooting

### Models fail to download

- Check `HF_TOKEN` is set in `.env`
- Accept license terms for gated models on [huggingface.co](https://huggingface.co)
- Check disk space: GPU models need ~85 GB, GGUF reviewer needs ~40 GB

### GPU not detected

- Verify device nodes: `ls -la /dev/kfd /dev/dri`
- Check groups: `groups` should include `video` and `render`
- On Ubuntu: `sudo apt install amdgpu-dkms`

### Out of GPU memory / system freeze

- Ensure `ORCH_GPU_UTIL + CODER_GPU_UTIL <= 1.0` (default: 0.95)
- Enlarge swap file (128 GB recommended)
- Install earlyoom: `sudo apt install earlyoom`
- Reduce `*_MAX_LEN` to shrink KV cache
- **Do NOT use FP8 KV cache** (`--kv-cache-dtype fp8`) — uncalibrated on this hardware
- **Do NOT use quantized models** (AWQ, GPTQ) — they crash on gfx1151

### Coder container fails to start

- Check that orchestrator is healthy first (`docker logs vllm_orchestrator`)
- Coder waits for orchestrator via `depends_on: service_healthy`
- Both must not start simultaneously on shared-memory GPUs (staggered startup is configured)

### Port conflicts

- Check ports: `ss -tlnp | grep -E '800[12]|801[12]'`
- Change port numbers in `.env`

### Container keeps restarting

- Check logs: `docker logs vllm_orchestrator` / `docker logs vllm_coder`
- For CPU tiers: `docker logs llama_reviewer` / `docker logs llama_utility`
- Verify model name and GGUF filename are correct
- Verify GPU device permissions

### switch-orch breaks opencode.jsonc

- The script uses `sed` to replace the `"primary"` line. If malformed, restore:
  ```bash
  git checkout opencode/opencode.jsonc
  ```

---

## Hardware Constraints (gfx1151 / RDNA 3.5)

This hardware has specific limitations documented in [DECISIONS.md](DECISIONS.md):

| What | Status | Why |
|------|--------|-----|
| BF16 weights | Works | Native support |
| BF16 KV cache | Works | Native support |
| FP8 weights (`--quantization fp8`) | Crashes | `torch._scaled_mm` requires MI300+ |
| AWQ INT4 (pre-quantized) | GPU hang | Triton AWQ kernels incompatible with wave32 |
| FP8 KV cache (`--kv-cache-dtype fp8`) | Accuracy loss | Uncalibrated scales (1.0) |
| GPTQ/Marlin/bitsandbytes | N/A | CUDA-only in vLLM |

**Bottom line: ALL models must use BF16 weights + BF16 KV cache on Strix Halo gfx1151.**

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Make your changes
4. Test: `./scripts/up full && ./scripts/health`
5. Validate compose: `docker compose -f compose/vllm.yml -f compose/cpu.yml --env-file .env config`
6. Submit a pull request

### Guidelines

- Keep `.env.example` up to date with any new environment variables
- Don't commit `.env`, `secrets/`, or model files
- Test compose files with `docker compose config` before committing
- Maintain script compatibility with `bash` (`set -euo pipefail`)
- Update DECISIONS.md for any architecture changes
