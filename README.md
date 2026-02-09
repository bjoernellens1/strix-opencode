# strix-opencode

A reproducible **Strix Halo** dev-stack repository for **Ubuntu + distrobox** that runs local LLM inference servers and integrates with [OpenCode](https://opencode.ai) via a custom [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) plugin.

Optimized for a **single-user workflow** — interactive latency and stability over multi-user throughput.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start (vLLM Only)](#quick-start-vllm-only)
- [Hybrid Orchestrator Fallback](#hybrid-orchestrator-fallback)
- [Switch to Cloud Planner (Option C)](#switch-to-cloud-planner-option-c)
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

This project provides three deployment modes for running local AI inference alongside OpenCode:

| Mode | Orchestrator + Coder | Fast Utility |
|------|---------------------|-------------|
| **Default (vLLM)** | vLLM (shared instance) | vLLM |
| **Hybrid RADV** | llama.cpp (Vulkan RADV) + vLLM | vLLM |
| **Hybrid ROCm** | llama.cpp (ROCm 6.4.4 rocWMMA) + vLLM | vLLM |

The orchestrator and coder roles share a single vLLM instance (same model, same port) to conserve GPU memory on the Strix Halo's 128 GB shared VRAM.

A fourth option (**Cloud Planner**) lets you route the orchestrator role to a cloud provider (e.g., OpenAI) while keeping coder and fast utility local.

All modes use containers managed by Docker Compose and are built around AMD Strix Halo GPU hardware.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OpenCode  (TUI / agent runtime)                                │
│  config: opencode/opencode.jsonc                                │
│  plugin: opencode/oh-my-opencode/                               │
├────────────┬─────────────────┬──────────────────────────────────┤
│            │                 │                                    │
│  Primary   │  Coder          │  Fast Utility                     │
│  (orch)    │  (subagent)     │  (subagent)                       │
│            │                 │                                    │
│  ┌─────────┴─────────────────┴──┐  ┌────────────────────────┐   │
│  │      vLLM :8001 (shared)     │  │ vLLM :8004             │   │
│  │    OR                        │  └────────────────────────┘   │
│  │      llama.cpp :8011         │                                │
│  │    OR                        │                                │
│  │      cloud API               │                                │
│  └──────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Ubuntu** (22.04+ recommended) with AMD Strix Halo GPU
- **Docker** (with Compose v2) — `docker compose` must work
- **GPU access**: `/dev/kfd` and `/dev/dri` exposed to containers
- **Distrobox** (optional but recommended; see [Distrobox Notes](#distrobox-notes-ubuntu))
- **curl** (for the health script)
- **HuggingFace account** with token (for gated models, set `HF_TOKEN` in your environment or `~/.cache/huggingface/token`)

---

## Repository Structure

```
strix-opencode/
├── README.md                                  # This file
├── .gitignore                                 # Ignores models/, .env, caches
├── .env.example                               # Template for environment variables
├── DECISIONS.md                               # Architecture decisions and research log
├── strix-opencode.md                          # Original build instructions
├── compose/
│   ├── vllm.yml                               # vLLM services (orchestrator+coder shared + fast)
│   ├── fallback.vulkan-radv.orch.yml          # llama.cpp orchestrator (Vulkan RADV backend)
│   └── fallback.rocm-6.4.4-rocwmma.orch.yml  # llama.cpp orchestrator (ROCm rocWMMA backend)
├── opencode/
│   ├── opencode.jsonc                         # OpenCode agent/provider configuration
│   └── oh-my-opencode/                        # Plugin (git submodule or vendored fork)
└── scripts/
    ├── up                                     # Start services (vllm | hybrid-radv | hybrid-rocm)
    ├── down                                   # Stop all services
    ├── health                                 # Check endpoint health
    └── switch-orch                            # Switch orchestrator (vllm | llama | cloud)
├── .opencode/
│   └── oh-my-opencode.json                   # Template: agent maxTokens for local models
```

---

## Quick Start (vLLM Only)

The default mode runs **all three roles on vLLM** — simplest setup, single container image.

```bash
# 1. Clone and enter
git clone https://github.com/bjoernellens1/strix-opencode.git
cd strix-opencode

# 2. Set up environment
cp .env.example .env
# Edit .env if you want to change models, ports, or GPU utilization

# 3. Start all vLLM services
./scripts/up vllm

# 4. Verify health
./scripts/health

# 5. Run OpenCode pointing to the config
# (from within a project directory you want to work on)
opencode --config /path/to/strix-opencode/opencode/opencode.jsonc
```

The first start will be slow as vLLM downloads model weights from HuggingFace into `$HF_HOME`. Subsequent starts use the cache.

### Stopping

```bash
./scripts/down
```

This gracefully stops all containers across all compose files.

---

## Hybrid Orchestrator Fallback

If you want the orchestrator to run on **llama.cpp** (for GGUF models, potentially lower VRAM usage, or different performance characteristics), use a hybrid mode while keeping coder and fast utility on vLLM.

### Option A: Vulkan RADV Backend

```bash
# 1. Place your GGUF model in ./models/
mkdir -p models
# e.g., download a GGUF model into ./models/

# 2. Start hybrid stack
./scripts/up hybrid-radv

# 3. Switch OpenCode to use the llama.cpp orchestrator
./scripts/switch-orch llama

# 4. Verify
./scripts/health
```

### Option B: ROCm 6.4.4 rocWMMA Backend

The ROCm backend may offer better performance through hardware-specific matrix multiply acceleration:

```bash
# 1. Place your GGUF model in ./models/
mkdir -p models

# 2. Start hybrid stack
./scripts/up hybrid-rocm

# 3. Switch OpenCode to use the llama.cpp orchestrator
./scripts/switch-orch llama

# 4. Verify
./scripts/health
```

### Switching Back to vLLM Orchestrator

```bash
./scripts/switch-orch vllm
```

---

## Switch to Cloud Planner (Option C)

Use a cloud API (e.g., OpenAI) for the orchestrator/planner role while keeping the coder and fast utility running locally. This is useful when you want the strongest available planner without local compute constraints.

```bash
# 1. Set your cloud credentials
export OPENAI_API_KEY=sk-...
export CLOUD_PLANNER_MODEL=gpt-5

# 2. Switch OpenCode to use the cloud planner
./scripts/switch-orch cloud

# 3. Ensure local coder + fast are running
./scripts/up vllm

# 4. Run OpenCode — orchestrator goes to cloud, coder/fast stay local
opencode --config /path/to/strix-opencode/opencode/opencode.jsonc
```

### Switching Back to Local

```bash
./scripts/switch-orch vllm
```

---

## Model Roles & Endpoints

### Default (vLLM for All)

| Role | Endpoint | Default Model |
|------|----------|---------------|
| Orchestrator + Coder (shared) | `http://127.0.0.1:8001/v1` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Fast Utility (subagent) | `http://127.0.0.1:8004/v1` | `Qwen/Qwen2.5-7B-Instruct` |

### Hybrid (llama.cpp Orchestrator)

| Role | Endpoint | Default Model |
|------|----------|---------------|
| Orchestrator + Coder via llama.cpp | `http://127.0.0.1:8011/v1` | GGUF file from `./models/` |
| Fast Utility (vLLM) | `http://127.0.0.1:8004/v1` | `Qwen/Qwen2.5-7B-Instruct` |

### Model Choices (env-configurable)

All model names are set in `.env` and can be swapped:

- **Orchestrator + Coder**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (shared instance) or `meta-llama/Llama-3.3-70B-Instruct` (via GGUF fallback)
- **Fast**: `Qwen/Qwen2.5-7B-Instruct`

---

## Model Download Notes

### vLLM Models

vLLM pulls HuggingFace models **automatically** on first run. Models are cached in `$HF_HOME` (default: `~/.cache/huggingface`).

- If a model is gated (requires acceptance of license terms), make sure you have:
  1. Accepted the license on the HuggingFace model page.
  2. Set your HuggingFace token: `export HF_TOKEN=hf_...` or saved it in `~/.cache/huggingface/token`.

- First download can be **very slow** for large models (70B+ parameters). Subsequent runs use the cache.

### llama.cpp GGUF Models

llama.cpp requires **pre-downloaded GGUF files** placed into the `./models/` directory:

```bash
mkdir -p models
cd models

# Example: download a quantized model from HuggingFace
# (use huggingface-cli or wget)
huggingface-cli download TheBloke/Llama-3.3-70B-Instruct-GGUF \
  llama-3.3-70b-instruct.Q4_K_M.gguf \
  --local-dir .
```

Update `ORCH_GGUF` in `.env` to match your downloaded filename:

```bash
ORCH_GGUF=llama-3.3-70b-instruct.Q4_K_M.gguf
```

---

## Environment Variables

All configurable values are in `.env` (copied from `.env.example`). Key variables:

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `${HOME}/.cache/huggingface` | HuggingFace model cache directory |
| `VLLM_CACHE` | `${HOME}/.cache/vllm` | vLLM runtime cache |
| `LLAMA_MODELS_DIR` | `./models` | Directory containing GGUF files for llama.cpp |

### Ports

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_PORT` | `8001` | vLLM orchestrator + coder port |
| `FAST_PORT` | `8004` | vLLM fast utility port |
| `LLAMA_ORCH_PORT` | `8011` | llama.cpp orchestrator port |

### Models

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | HuggingFace model for vLLM orchestrator + coder (shared) |
| `FAST_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for vLLM fast |
| `ORCH_GGUF` | `Qwen3-30B-A3B-Instruct.Q4_K_M.gguf` | GGUF filename for llama.cpp orchestrator |

### vLLM Tuning

> **Important:** When running both vLLM services on the same GPU, their `*_GPU_UTIL` fractions must sum to **≤ 1.0** or you will hit OOM. The defaults below leave 15% (~19 GB) headroom for the system.

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_GPU_UTIL` | `0.65` | GPU memory utilization for orchestrator + coder |
| `FAST_GPU_UTIL` | `0.20` | GPU memory utilization for fast |
| `ORCH_MAX_LEN` | `32768` | Max sequence length for orchestrator (32K) |
| `FAST_MAX_LEN` | `8192` | Max sequence length for fast (8K) |
| `ORCH_KV_CACHE_DTYPE` | `fp8` | KV cache precision for orchestrator (fp8 halves KV memory) |
| `FAST_KV_CACHE_DTYPE` | `fp8` | KV cache precision for fast |
| `ORCH_MAX_NUM_SEQS` | `4` | Max concurrent sequences for orchestrator |
| `FAST_MAX_NUM_SEQS` | `4` | Max concurrent sequences for fast |

### Cloud Planner (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(unset)* | API key for cloud planner |
| `CLOUD_PLANNER_MODEL` | *(unset)* | Model name (e.g., `gpt-5`) |

---

## Scripts Reference

All scripts live in `scripts/` and are executable.

### `scripts/up`

Start the inference stack.

```bash
./scripts/up              # Default: vLLM only
./scripts/up vllm         # Explicit: vLLM only
./scripts/up hybrid-radv  # vLLM + llama.cpp orchestrator (Vulkan RADV)
./scripts/up hybrid-rocm  # vLLM + llama.cpp orchestrator (ROCm rocWMMA)
```

On first run, `.env.example` is automatically copied to `.env` if `.env` does not exist.

### `scripts/down`

Stop all services across all compose files:

```bash
./scripts/down
```

### `scripts/health`

Check whether each endpoint is responding:

```bash
./scripts/health
```

Outputs the `/v1/models` response from each port. If a service isn't running, you'll see a connection error for that port (which is expected for modes that don't start all services).

### `scripts/switch-orch`

Switch which backend the OpenCode primary agent uses:

```bash
./scripts/switch-orch vllm   # Use local vLLM orchestrator (:8001)
./scripts/switch-orch llama  # Use local llama.cpp orchestrator (:8011)
./scripts/switch-orch cloud  # Use cloud planner (requires OPENAI_API_KEY)
```

This edits `opencode/opencode.jsonc` in-place. The coder and fast utility agents are unaffected.

---

## OpenCode Configuration

The file `opencode/opencode.jsonc` defines:

- **Providers**: connection details for each inference backend
- **Agents**: role assignments mapping agents to providers
- **Plugins**: reference to the oh-my-opencode plugin

### Providers Defined

| Provider ID | Backend | Endpoint |
|------------|---------|----------|
| `local_orch_vllm` | vLLM (orch + coder shared) | `http://127.0.0.1:8001/v1` |
| `local_orch_llama` | llama.cpp | `http://127.0.0.1:8011/v1` |
| `local_fast` | vLLM | `http://127.0.0.1:8004/v1` |
| `cloud_planner` | OpenAI API | cloud |

### Agent Roles

| Agent | Role | Default Provider |
|-------|------|-----------------|
| `primary` | Orchestrator/Planner | `local_orch_vllm:orch` |
| `coder` | Code generation subagent | `local_orch_vllm:coder` |
| `utility` | Fast utility subagent | `local_fast:fast` |

---

## oh-my-opencode Plugin

This repository is designed to work with a custom fork of [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode), placed at `opencode/oh-my-opencode/`.

### Replacing with Your Own Fork as Git Submodule

If a submodule already exists (e.g., the upstream), you must remove it from Git's index first — `rm -rf` alone is not enough:

```bash
# 1. Remove the existing submodule from Git tracking
git rm opencode/oh-my-opencode

# 2. Clean up any cached submodule metadata
rm -rf .git/modules/opencode/oh-my-opencode

# 3. Add your fork as the new submodule
git submodule add https://github.com/bjoernellens1/oh-my-opencode opencode/oh-my-opencode

# 4. Commit
git add .gitmodules opencode/oh-my-opencode
git commit -m "Replace oh-my-opencode submodule with fork"
```

> **Note:** `git submodule add` will fail if the path is still tracked by Git.
> You must use `git rm` (not just `rm -rf`) to properly de-register it first.

### Cloning with Submodules

After the submodule is added, anyone cloning the repo should use:

```bash
git clone --recurse-submodules https://github.com/bjoernellens1/strix-opencode.git
```

Or if already cloned:

```bash
git submodule update --init --recursive
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

**Ubuntu + distrobox** is the recommended environment for this toolbox ecosystem. Distrobox provides seamless GPU passthrough and lets you run the specialized container images without modifying your host system.

### Why Distrobox?

- Direct access to `/dev/kfd` and `/dev/dri` GPU devices
- Host networking by default — no port mapping needed
- User home directory is shared — HuggingFace cache is accessible
- Works alongside Docker on the same host

### Example: Create a Distrobox Environment

```bash
# Create a distrobox for the vLLM toolbox image
distrobox create \
  --name strix-vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri"

# Enter the distrobox
distrobox enter strix-vllm
```

### Running with Host Docker Compose

You don't have to use distrobox for the compose stack — running `docker compose` directly on the host works fine if Docker has GPU access. Distrobox is most useful when you want an interactive shell inside the toolbox images for debugging or manual model testing.

---

## Troubleshooting

### Models fail to download

- Check your HuggingFace token is set: `echo $HF_TOKEN`
- Ensure you've accepted license terms for gated models on [huggingface.co](https://huggingface.co)
- Check disk space: large models can be 50–150 GB

### GPU not detected

- Verify device nodes exist: `ls -la /dev/kfd /dev/dri`
- Check your user is in the `video` and `render` groups: `groups`
- On Ubuntu, install the AMDGPU driver: `sudo apt install amdgpu-dkms`

### Out of GPU memory

- Ensure `*_GPU_UTIL` values in `.env` **sum to ≤ 1.0** (default: `0.65 + 0.20 = 0.85`)
- Lower individual `*_GPU_UTIL` values if needed
- Reduce `*_MAX_LEN` values to shrink KV cache per request
- Set `*_KV_CACHE_DTYPE=fp8` to halve KV cache memory
- Use a smaller model or quantized variant
- In hybrid mode, the llama.cpp orchestrator uses less VRAM for quantized GGUF models

### Port conflicts

- Check if ports are already in use: `ss -tlnp | grep -E '800[14]|8011'`
- Change port numbers in `.env`

### Container keeps restarting

- Check logs: `docker logs vllm_orchestrator`
- Ensure the model name is correct and accessible
- Verify GPU device permissions

### switch-orch breaks opencode.jsonc

- The script uses `sed` to replace the `"primary"` line. If the file is malformed, restore it:
  ```bash
  git checkout opencode/opencode.jsonc
  ```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Make your changes
4. Test with `./scripts/up` and `./scripts/health`
5. Submit a pull request

### Key Guidelines

- Keep `.env.example` up to date with any new environment variables
- Don't commit `.env` or model files
- Test compose files with `docker compose config` before committing
- Maintain script compatibility with `bash` (no bashisms beyond what `set -euo pipefail` requires)
