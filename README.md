# strix-opencode

A reproducible **Strix Halo** dev-stack for **Ubuntu + distrobox** that runs local LLM inference servers and integrates with [OpenCode](https://opencode.ai) via a custom [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) plugin.

Optimized for a **single-user workflow** — interactive latency and stability over multi-user throughput.

Inspired by Norse mythology — each agent is named after a Norse deity whose traits match the agent's role. For detailed agent specifications, see [AGENTS.md](AGENTS.md).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Startup Modes](#startup-modes)
- [Agent Roles & Endpoints](#agent-roles--endpoints)
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

This project runs a **6-agent local AI inference stack** on a single AMD Strix Halo system (128 GB shared UMA memory):

| Agent | Role | Backend | Model | Context |
|-------|------|---------|-------|---------|
| **Thor** ⚡ | Primary Commander — planning, coordination, delegation | vLLM (GPU, BF16) | Qwen2.5-14B-Instruct | 64K |
| **Valkyrie** 🛡 | Execution Specialist — code generation, tool use | vLLM (GPU, BF16) | Qwen3-Coder-30B-A3B-Instruct | 48K |
| **Odin** 👁️ | Supreme Architect — escalation, deep review, architecture | llama.cpp (CPU) | Llama-3.3-70B Q4_K_M | 32K |
| **Heimdall** 👁 | Guardian — fast validation, monitoring, utilities | llama.cpp (CPU) | Qwen2.5-3B Q4_K_M | 8K |
| **Loki** 🧠 | Adversarial Intelligence — edge cases, creative challenges | llama.cpp (CPU) | Qwen2.5-7B Q4_K_M | 16K |
| **Frigga** 🌿 | Knowledge Curator — documentation, context compression | llama.cpp (CPU) | Qwen2.5-14B Q4_K_M | 32K |

**Key design decisions:**
- **All BF16 on GPU** — no quantization works on gfx1151 (FP8, AWQ, GPTQ all fail; see [DECISIONS.md](DECISIONS.md))
- **CPU agents via llama.cpp** — 70B Odin runs INT4 GGUF on CPU without competing for GPU memory
- **Staggered GPU startup** — prevents shared-memory race condition during vLLM memory profiling
- **0.95 total GPU utilization** — tight for single-user; swap file + earlyoom recommended
- **6 agents, 2 GPU + 4 CPU** — see [AGENTS.md](AGENTS.md) for detailed specs, escalation doctrine, and memory budget

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  OpenCode  (TUI / agent runtime)                                                    │
│  config: opencode/opencode.jsonc                                                    │
│  plugin: opencode/oh-my-opencode/                                                   │
├──────────┬──────────┬──────────┬──────────┬──────────┬────────────────────────────┤
│          │          │          │          │          │                              │
│  Thor ⚡  │ Valkyrie │ Odin 👁️  │ Heimdall │ Loki 🧠  │  Frigga 🌿                  │
│ Primary  │  🛡 Exec │ Architect│  👁 Guard│ Adversary│  Knowledge                  │
│          │          │          │          │          │                              │
│ ┌────────┤ ┌────────┤ ┌────────┤ ┌────────┤ ┌────────┤ ┌──────────────────────────┐│
│ │vLLM    │ │vLLM    │ │llama   │ │llama   │ │llama   │ │llama.cpp :8014           ││
│ │:8001   │ │:8002   │ │:8011   │ │:8012   │ │:8013   │ │14B CPU Q4_K_M            ││
│ │14B GPU │ │30B GPU │ │70B CPU │ │3B CPU  │ │7B CPU  │ │                          ││
│ │BF16    │ │BF16    │ │Q4_K_M  │ │Q4_K_M  │ │Q4_K_M  │ │                          ││
│ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └──────────────────────────┘│
└────────────────────────────────────────────────────────────────────────────────────┘
   GPU(0.35)  GPU(0.60)  CPU(16t)   CPU(8t)    CPU(8t)    CPU(12t)
        └── total: 0.95 ──┘
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
├── AGENTS.md                                  # Detailed agent specs, escalation doctrine, memory budget
├── DECISIONS.md                               # Architecture decisions, research log, quantization results
├── .env.example                               # Template for environment variables
├── .gitignore                                 # Ignores models/, .env, secrets, caches
├── strix-opencode.md                          # Original build instructions
├── compose/
│   ├── vllm.yml                               # GPU agents: Thor (14B) + Valkyrie (30B)
│   ├── cpu.yml                                # CPU agents: Odin (70B) + Heimdall (3B) + Loki (7B) + Frigga (14B)
│   ├── fallback.vulkan-radv.orch.yml          # Legacy: llama.cpp orchestrator (Vulkan RADV)
│   └── fallback.rocm-6.4.4-rocwmma.orch.yml  # Legacy: llama.cpp orchestrator (ROCm rocWMMA)
├── opencode/
│   ├── opencode.jsonc                         # OpenCode provider/agent configuration
│   └── oh-my-opencode/                        # Plugin (git submodule)
├── scripts/
│   ├── up                                     # Start services (gpu | cpu | full | hybrid-*)
│   ├── down                                   # Stop all services
│   ├── health                                 # Check all 6 endpoints
│   ├── benchmark                              # Compare Thor vs Valkyrie performance
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

# 3. Download GGUF models for CPU agents
mkdir -p models

# Odin: Llama-3.3-70B (~40 GB)
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
  llama-3.3-70b-instruct.Q4_K_M.gguf --local-dir models

# Heimdall: Qwen2.5-3B (~2 GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct.Q4_K_M.gguf --local-dir models

# Loki: Qwen2.5-7B (~4 GB)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models

# Frigga: Qwen2.5-14B (~8 GB)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF \
  qwen2.5-14b-instruct-q4_k_m.gguf --local-dir models

# 4a. Start GPU agents only (most common)
./scripts/up gpu

# 4b. Or start all 6 agents
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
| **gpu** (default) | `./scripts/up` or `./scripts/up gpu` | Thor (:8001) + Valkyrie (:8002) |
| **cpu** | `./scripts/up cpu` | Odin (:8011) + Heimdall (:8012) + Loki (:8013) + Frigga (:8014) |
| **full** | `./scripts/up full` | All 6 agents |
| **hybrid-radv** | `./scripts/up hybrid-radv` | vLLM + llama.cpp orch (Vulkan RADV, legacy) |
| **hybrid-rocm** | `./scripts/up hybrid-rocm` | vLLM + llama.cpp orch (ROCm rocWMMA, legacy) |

### Stopping

```bash
./scripts/down    # Stops all containers across all compose files
```

---

## Agent Roles & Endpoints

| Agent | Port | Default Model | Backend |
|-------|------|---------------|---------|
| Thor ⚡ | `http://127.0.0.1:8001/v1` | Qwen/Qwen2.5-14B-Instruct | vLLM (GPU) |
| Valkyrie 🛡 | `http://127.0.0.1:8002/v1` | Qwen/Qwen3-Coder-30B-A3B-Instruct | vLLM (GPU) |
| Odin 👁️ | `http://127.0.0.1:8011/v1` | Llama-3.3-70B Q4_K_M GGUF | llama.cpp (CPU) |
| Heimdall 👁 | `http://127.0.0.1:8012/v1` | Qwen2.5-3B Q4_K_M GGUF | llama.cpp (CPU) |
| Loki 🧠 | `http://127.0.0.1:8013/v1` | Qwen2.5-7B Q4_K_M GGUF | llama.cpp (CPU) |
| Frigga 🌿 | `http://127.0.0.1:8014/v1` | Qwen2.5-14B Q4_K_M GGUF | llama.cpp (CPU) |

All endpoints serve an OpenAI-compatible `/v1` API.

For detailed role descriptions, escalation paths, and inter-agent communication, see [AGENTS.md](AGENTS.md).

### Cloud Planner (Optional)

You can route the primary agent (Thor) to a cloud provider while keeping local agents:

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

### vLLM Models (GPU Agents)

vLLM pulls HuggingFace models **automatically** on first run. Models are cached in `$HF_HOME` (default: `~/.cache/huggingface`).

- Set `HF_TOKEN` in `.env` for gated models
- First download: ~28 GB (Thor 14B) + ~57 GB (Valkyrie 30B) = **~85 GB total**
- Subsequent starts use the cache

### llama.cpp GGUF Models (CPU Agents)

GGUF files must be **pre-downloaded** into `./models/`:

```bash
mkdir -p models

# Odin (Supreme Architect): Llama-3.3-70B (~40 GB)
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
  llama-3.3-70b-instruct.Q4_K_M.gguf --local-dir models

# Heimdall (Guardian): Qwen2.5-3B (~2 GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct.Q4_K_M.gguf --local-dir models

# Loki (Adversarial): Qwen2.5-7B (~4 GB)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models

# Frigga (Knowledge Curator): Qwen2.5-14B (~8 GB)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF \
  qwen2.5-14b-instruct-q4_k_m.gguf --local-dir models
```

Update the corresponding `*_GGUF` variables in `.env` if your filenames differ.

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
| `THOR_PORT` | `8001` | Thor — vLLM primary commander |
| `VALKYRIE_PORT` | `8002` | Valkyrie — vLLM execution specialist |
| `ODIN_PORT` | `8011` | Odin — llama.cpp supreme architect |
| `HEIMDALL_PORT` | `8012` | Heimdall — llama.cpp guardian |
| `LOKI_PORT` | `8013` | Loki — llama.cpp adversarial intelligence |
| `FRIGGA_PORT` | `8014` | Frigga — llama.cpp knowledge curator |

### GPU Agents (vLLM)

> **Important:** `THOR_GPU_UTIL + VALKYRIE_GPU_UTIL` must be **<= 1.0**. Both share the same physical GPU memory on Strix Halo.

| Variable | Default | Description |
|----------|---------|-------------|
| `THOR_MODEL` | `Qwen/Qwen2.5-14B-Instruct` | Thor model (14B dense) |
| `VALKYRIE_MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Valkyrie model (30B MoE) |
| `THOR_GPU_UTIL` | `0.35` | GPU memory fraction for Thor (~44.8 GB) |
| `VALKYRIE_GPU_UTIL` | `0.60` | GPU memory fraction for Valkyrie (~76.8 GB) |
| `THOR_MAX_LEN` | `65536` | Thor context window (64K tokens) |
| `VALKYRIE_MAX_LEN` | `49152` | Valkyrie context window (48K tokens) |
| `THOR_KV_CACHE_DTYPE` | `auto` | KV cache dtype (**must be `auto`** on gfx1151) |
| `VALKYRIE_KV_CACHE_DTYPE` | `auto` | KV cache dtype (**must be `auto`** on gfx1151) |
| `THOR_MAX_NUM_SEQS` | `4` | Max concurrent sequences (single-user) |
| `VALKYRIE_MAX_NUM_SEQS` | `4` | Max concurrent sequences (single-user) |

### CPU Agents (llama.cpp)

| Variable | Default | Description |
|----------|---------|-------------|
| `ODIN_GGUF` | `llama-3.3-70b-instruct.Q4_K_M.gguf` | GGUF filename for Odin |
| `ODIN_CTX` | `32768` | Odin context window (32K) |
| `ODIN_THREADS` | `16` | CPU threads for Odin |
| `HEIMDALL_GGUF` | `qwen2.5-3b-instruct.Q4_K_M.gguf` | GGUF filename for Heimdall |
| `HEIMDALL_CTX` | `8192` | Heimdall context window (8K) |
| `HEIMDALL_THREADS` | `8` | CPU threads for Heimdall |
| `LOKI_GGUF` | `qwen2.5-7b-instruct-q4_k_m.gguf` | GGUF filename for Loki |
| `LOKI_CTX` | `16384` | Loki context window (16K) |
| `LOKI_THREADS` | `8` | CPU threads for Loki |
| `FRIGGA_GGUF` | `qwen2.5-14b-instruct-q4_k_m.gguf` | GGUF filename for Frigga |
| `FRIGGA_CTX` | `32768` | Frigga context window (32K) |
| `FRIGGA_THREADS` | `12` | CPU threads for Frigga |

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
./scripts/up              # Default: GPU agents only (Thor + Valkyrie)
./scripts/up gpu          # Explicit: GPU agents only
./scripts/up cpu          # CPU agents only (Odin + Heimdall + Loki + Frigga)
./scripts/up full         # All 6 agents
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

Check all 6 endpoints (queries `/v1/models` on each port):

```bash
./scripts/health
```

Outputs the model list from each service. Services that aren't running show "(not responding)" — this is expected when not using `full` mode.

### `scripts/benchmark`

Compare Thor and Valkyrie on real-world coding tasks:

```bash
./scripts/benchmark
```

### `scripts/switch-orch`

Switch the OpenCode primary agent backend:

```bash
./scripts/switch-orch vllm   # Use local Thor (:8001)
./scripts/switch-orch cloud  # Use cloud planner (requires OPENAI_API_KEY)
```

This edits `opencode/opencode.jsonc` in-place. Other agents are unaffected.

---

## OpenCode Configuration

The file `opencode/opencode.jsonc` defines providers and agent-to-provider mappings:

### Providers

| Provider ID | Norse Agent | Backend | Endpoint |
|------------|-------------|---------|----------|
| `local_thor` | Thor ⚡ | vLLM GPU | `http://127.0.0.1:8001/v1` |
| `local_valkyrie` | Valkyrie 🛡 | vLLM GPU | `http://127.0.0.1:8002/v1` |
| `local_odin` | Odin 👁️ | llama.cpp CPU | `http://127.0.0.1:8011/v1` |
| `local_heimdall` | Heimdall 👁 | llama.cpp CPU | `http://127.0.0.1:8012/v1` |
| `local_loki` | Loki 🧠 | llama.cpp CPU | `http://127.0.0.1:8013/v1` |
| `local_frigga` | Frigga 🌿 | llama.cpp CPU | `http://127.0.0.1:8014/v1` |
| `cloud_planner` | — | OpenAI API | cloud |

### Agents

| Agent | Role | Default Provider |
|-------|------|-----------------|
| `primary` | Thor — Primary Commander | `local_thor:thor` |
| `coder` | Valkyrie — Execution Specialist | `local_valkyrie:valkyrie` |
| `reviewer` | Odin — Supreme Architect | `local_odin:odin` |
| `utility` | Heimdall — Guardian | `local_heimdall:heimdall` |

> Note: Loki and Frigga don't have standard OpenCode agent mappings yet. See [AGENTS.md — Future Considerations](AGENTS.md#future-considerations) for routing plans.

### Agent Token Limits

The template `.opencode/oh-my-opencode.json` caps output tokens per agent to fit local model context windows:

| OpenCode Agent | Norse Agent | maxTokens | Rationale |
|----------------|-------------|-----------|-----------|
| sisyphus | Thor (64K) | 32,768 | 32K output + ~32K input |
| hephaestus, sisyphus-junior | Valkyrie (48K) | 24,576 | 24K output + ~24K input |
| oracle, prometheus | Odin (32K) | 16,384 | 16K output + ~16K input |
| metis, momus | Frigga (32K) | 16,384 | 16K output + ~16K input |
| librarian, explore, atlas | Heimdall (8K) | 4,096 | 4K output + ~4K input |

Copy this file to `.opencode/` in any target project. Remove/raise limits when using cloud models.

---

## Memory Budget

### GPU Allocation (128 GB shared UMA)

| Agent | Fraction | Memory | Weights (BF16) | KV Budget |
|-------|----------|--------|----------------|-----------|
| Valkyrie (30B MoE) | 0.60 | ~76.8 GB | ~57 GB | ~19.8 GB |
| Thor (14B) | 0.35 | ~44.8 GB | ~28 GB | ~16.8 GB |
| System headroom | 0.05 | ~6.4 GB | — | — |

### KV Cache Fit

| Agent | Model | Context | KV Size | Budget | Slack |
|-------|-------|---------|---------|--------|-------|
| Thor | Qwen2.5-14B | 64K | ~10.0 GB | 16.8 GB | 6.8 GB |
| Valkyrie | Qwen3-Coder-30B | 48K | ~4.5 GB | 19.8 GB | 15.3 GB |

### CPU Agents (system RAM)

| Agent | Model | GGUF Size | RAM at Load |
|-------|-------|-----------|-------------|
| Odin | Llama-3.3-70B Q4_K_M | ~40 GB | ~42 GB |
| Frigga | Qwen2.5-14B Q4_K_M | ~8 GB | ~10 GB |
| Loki | Qwen2.5-7B Q4_K_M | ~4 GB | ~5 GB |
| Heimdall | Qwen2.5-3B Q4_K_M | ~2 GB | ~3 GB |
| **Total CPU** | | **~54 GB** | **~60 GB** |

CPU agents share physical RAM with GPU allocations. Running all 4 CPU agents simultaneously adds ~60 GB. A 128 GB swap file is essential for system stability.

> For detailed memory math, KV cache calculations, and vLLM allocation internals, see [DECISIONS.md](DECISIONS.md).
> For agent specifications, escalation doctrine, and inter-agent communication, see [AGENTS.md](AGENTS.md).

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
- Check disk space: GPU models need ~85 GB, GGUF models need ~54 GB total

### GPU not detected

- Verify device nodes: `ls -la /dev/kfd /dev/dri`
- Check groups: `groups` should include `video` and `render`
- On Ubuntu: `sudo apt install amdgpu-dkms`

### Out of GPU memory / system freeze

- Ensure `THOR_GPU_UTIL + VALKYRIE_GPU_UTIL <= 1.0` (default: 0.95)
- Enlarge swap file (128 GB recommended)
- Install earlyoom: `sudo apt install earlyoom`
- Reduce `*_MAX_LEN` to shrink KV cache
- **Do NOT use FP8 KV cache** (`--kv-cache-dtype fp8`) — uncalibrated on this hardware
- **Do NOT use quantized models** (AWQ, GPTQ) — they crash on gfx1151

### Valkyrie container fails to start

- Check that Thor is healthy first (`docker logs vllm_thor`)
- Valkyrie waits for Thor via `depends_on: service_healthy`
- Both must not start simultaneously on shared-memory GPUs (staggered startup is configured)

### Port conflicts

- Check ports: `ss -tlnp | grep -E '800[12]|801[1234]'`
- Change port numbers in `.env`

### Container keeps restarting

- Check logs: `docker logs vllm_thor` / `docker logs vllm_valkyrie`
- For CPU agents: `docker logs llama_odin` / `docker logs llama_heimdall` / `docker logs llama_loki` / `docker logs llama_frigga`
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
| BF16 weights | ✅ Works | Native support |
| BF16 KV cache | ✅ Works | Native support |
| FP8 weights (`--quantization fp8`) | ❌ Crashes | `torch._scaled_mm` requires MI300+ |
| AWQ INT4 (pre-quantized) | ❌ GPU hang | Triton AWQ kernels incompatible with wave32 |
| FP8 KV cache (`--kv-cache-dtype fp8`) | ⚠️ Accuracy loss | Uncalibrated scales (1.0) |
| GPTQ/Marlin/bitsandbytes | ❌ N/A | CUDA-only in vLLM |

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
- Update DECISIONS.md for architecture changes, AGENTS.md for agent changes
