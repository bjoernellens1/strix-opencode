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
| **Thor** ⚡ | Primary Commander — planning, coordination, delegation | Ollama (GPU) | Qwen2.5-14B (Ollama) | 64K |
| **Valkyrie** 🛡 | Execution Specialist — code generation, tool use | Ollama (GPU) | Qwen3-Coder-30B-A3B (Ollama) | 64K |
| **Odin** 👁️ | Supreme Architect — escalation, deep review, architecture | Ollama (GPU) | Llama-3.3-70B (Ollama) | 64K |
| **Heimdall** 👁 | Guardian — fast validation, monitoring, utilities | Ollama (GPU) | Qwen2.5-3B (Ollama) | 16K |
| **Loki** 🧠 | Adversarial Intelligence — edge cases, creative challenges | Ollama (GPU) | Qwen2.5-7B (Ollama) | 32K |
| **Frigga** 🌿 | Knowledge Curator — documentation, context compression | Ollama (GPU) | Qwen2.5-14B (Ollama) | 64K |

**Key design decisions:**
- **Ollama-first Architecture** — All 6 agents run on **Ollama** using the OpenAI-compatible API.
- **Single Ollama daemon** — One server/port, minimal ops, Ollama handles loading/unloading.
- **Concurrency tuned** — `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` set for multiple simultaneous requests.
- **6 agents, all GPU** — see [AGENTS.md](AGENTS.md) for detailed specs and roles

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
│ ┌──────────────────────────────────────────────────────────────────────────────┐│
│ │                               Ollama Server                                   ││
│ │                        (GPU, OpenAI-compatible API)                           ││
│ │                                 :11434                                        ││
│ └──────────────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────────────────┘
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
├── bifrost/                                   # Bifrost scheduler source
│   ├── app.py                                 # FastAPI scheduler
│   └── Dockerfile                             # Container for Bifrost
├── compose/
│   ├── hybrid.yml                             # Phase 5: llama.cpp (Thor/Valkyrie/Odin) + vLLM (utility)
│   ├── gpu-all.yml                            # Phase 4: vLLM BF16 for all 6 agents
│   ├── vllm.yml                               # GPU agents only: Thor + Valkyrie (vLLM)
│   ├── ollama.yml                             # Ollama-first stack for all agents
│   ├── cpu.yml                                # CPU agents: Odin + Heimdall + Loki + Frigga (llama.cpp)
│   ├── fallback.vulkan-radv.orch.yml          # Legacy: llama.cpp orchestrator (Vulkan RADV)
│   └── fallback.rocm-6.4.4-rocwmma.orch.yml  # Legacy: llama.cpp orchestrator (ROCm rocWMMA)
├── ollama-modelfiles/                         # Modelfiles for 64K context variants
├── opencode/
│   ├── opencode.jsonc                         # OpenCode provider/agent configuration
│   └── oh-my-opencode/                        # Plugin (git submodule)
├── scripts/
│   ├── up                                     # Start services (hybrid | gpu-all | gpu | cpu | full)
│   ├── down                                   # Stop all services
│   ├── health                                 # Check all 6 endpoints
│   ├── benchmark                              # Compare Thor vs Valkyrie performance
│   ├── ollama-create                           # Create 64K Ollama variants
│   └── switch-orch                            # Switch orchestrator (ollama | cloud)
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
# Edit .env — set HF_TOKEN for gated model downloads (Llama-3.3, Qwen)

# 3. Start Ollama-first stack
./scripts/up ollama

# 4. Verify health
./scripts/health

# 5. Run OpenCode
OPENCODE_CONFIG=/path/to/strix-opencode/opencode/opencode.jsonc opencode
```

Ollama pulls models on first run. Models are cached in `OLLAMA_MODELS` (default: `~/.ollama`). Total weight depends on your selected tags/quantization.

---

## Startup Modes

| Mode | Command | What starts |
|------|---------|-------------|
| **ollama** (recommended) | `./scripts/up ollama` | Single Ollama server (all agents via one port) |
| **hybrid** (legacy) | `./scripts/up hybrid` | Thor + Valkyrie + Bifrost (llama.cpp GGUF + vLLM on-demand) |
| **hybrid-no-bifrost** | `./scripts/up hybrid-no-bifrost` | Thor + Valkyrie only (no Bifrost scheduler) |
| **gpu-all** | `./scripts/up gpu-all` | Phase 4: all 6 agents via vLLM BF16 + Bifrost |
| **gpu** | `./scripts/up gpu` | Legacy: Thor + Valkyrie (vLLM BF16) |
| **cpu** | `./scripts/up cpu` | Legacy: Odin + Heimdall + Loki + Frigga (llama.cpp CPU) |
| **full** | `./scripts/up full` | Legacy: GPU + CPU modes combined |

### Stopping

```bash
./scripts/down    # Stops all containers across all compose files
```

---

## Agent Roles & Endpoints

| Agent | Endpoint | Default Model (Ollama) | Backend |
|-------|----------|------------------------|---------|
| Thor ⚡ | `http://127.0.0.1:11434/v1` | qwen2.5:14b-instruct-q4_K_M-64k | Ollama (GPU) |
| Valkyrie 🛡 | `http://127.0.0.1:11434/v1` | qwen3-coder:30b-64k | Ollama (GPU) |
| Odin 👁️ | `http://127.0.0.1:11434/v1` | llama3:70b-instruct-q4_K_M-64k | Ollama (GPU) |
| Heimdall 👁 | `http://127.0.0.1:11434/v1` | qwen2.5:3b-instruct | Ollama (GPU) |
| Loki 🧠 | `http://127.0.0.1:11434/v1` | qwen2.5:7b-instruct | Ollama (GPU) |
| Frigga 🌿 | `http://127.0.0.1:11434/v1` | qwen2.5:14b-instruct-q4_K_M-64k | Ollama (GPU) |

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
./scripts/switch-orch ollama
```

---

## Model Download Notes

### Ollama Models (All Agents)

Ollama pulls models **automatically** on first run. Models are cached in `OLLAMA_MODELS` (default: `~/.ollama`).

Recommended tags (match `.env.example`):

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull qwen3-coder:30b
ollama pull llama3:70b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
```

### Context Length (64K variants)

Ollama’s OpenAI-compatible API doesn’t expose `num_ctx`, so use Modelfiles to bake context length:

```bash
./scripts/ollama-create
```

This creates `-64k` variants for the 14B, 30B, and 70B models (see `ollama-modelfiles/`).

---

## Environment Variables

All configurable values are in `.env` (copied from `.env.example`).

### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODELS` | `${HOME}/.ollama` | Ollama model store (shared by all Ollama containers) |
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated models (legacy) |
| `HF_HOME` | `${HOME}/.cache/huggingface` | HuggingFace model cache (legacy) |
| `VLLM_CACHE` | `${HOME}/.cache/vllm` | vLLM runtime cache (legacy) |
| `LLAMA_MODELS_DIR` | `./models` | Directory containing GGUF files (legacy) |

### Ports

| Variable | Default | Description |
|----------|---------|-------------|
| `THOR_PORT` | `8001` | Thor endpoint |
| `VALKYRIE_PORT` | `8002` | Valkyrie endpoint |
| `ODIN_PORT` | `8011` | Odin endpoint |
| `HEIMDALL_PORT` | `8012` | Heimdall endpoint |
| `LOKI_PORT` | `8013` | Loki endpoint |
| `FRIGGA_PORT` | `8014` | Frigga endpoint |

### Ollama Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_NUM_PARALLEL` | `2` | Concurrent requests per Ollama daemon |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Max loaded models per Ollama daemon |
| `OLLAMA_KEEP_ALIVE` | `5m` | Model keep-alive (warm cache window) |
| `THOR_OLLAMA_MODEL` | `qwen2.5:14b-instruct-q4_K_M-64k` | Thor model tag |
| `VALKYRIE_OLLAMA_MODEL` | `qwen3-coder:30b-64k` | Valkyrie model tag |
| `ODIN_OLLAMA_MODEL` | `llama3:70b-instruct-q4_K_M-64k` | Odin model tag |
| `HEIMDALL_OLLAMA_MODEL` | `qwen2.5:3b-instruct` | Heimdall model tag |
| `LOKI_OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Loki model tag |
| `FRIGGA_OLLAMA_MODEL` | `qwen2.5:14b-instruct-q4_K_M-64k` | Frigga model tag |

### GPU Agents (vLLM, legacy)

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

### CPU Agents (llama.cpp, legacy)

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
./scripts/up              # Default: Ollama-only stack
./scripts/up ollama       # Ollama-first stack (single server)
./scripts/up gpu          # Legacy: GPU agents only
./scripts/up cpu          # Legacy: CPU agents only (Odin + Heimdall + Loki + Frigga)
./scripts/up full         # Legacy: all 6 agents
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

### `scripts/ollama-create`

Create 64K context variants for the large models:

```bash
./scripts/ollama-create
```

### `scripts/benchmark`

Compare Thor and Valkyrie on real-world coding tasks:

```bash
./scripts/benchmark
```

### `scripts/switch-orch`

Switch the OpenCode primary agent backend:

```bash
./scripts/switch-orch ollama # Use local Thor (:11434)
./scripts/switch-orch cloud  # Use cloud planner (requires OPENAI_API_KEY)
```

This edits `opencode/opencode.jsonc` in-place. Other agents are unaffected.

---

## OpenCode Configuration

The file `opencode/opencode.jsonc` defines providers and agent-to-provider mappings:

### Providers

| Provider ID | Norse Agent | Backend | Endpoint |
|------------|-------------|---------|----------|
| `ollama_local` | Thor/Valkyrie/Odin/Heimdall/Loki/Frigga | Ollama GPU | `http://127.0.0.1:11434/v1` |
| `cloud_planner` | — | OpenAI API | cloud |

### Agents

| Agent | Role | Default Provider |
|-------|------|-----------------|
| `primary` | Thor — Primary Commander | `ollama_local/qwen2.5:14b-instruct-q4_K_M-64k` |
| `coder` | Valkyrie — Execution Specialist | `ollama_local/qwen3-coder:30b-64k` |
| `reviewer` | Odin — Supreme Architect | `ollama_local/llama3:70b-instruct-q4_K_M-64k` |
| `utility` | Heimdall — Guardian | `ollama_local/qwen2.5:3b-instruct` |

> Note: Loki and Frigga don't have standard OpenCode agent mappings yet. See [AGENTS.md — Future Considerations](AGENTS.md#future-considerations) for routing plans.

### Agent Token Limits

The template `.opencode/oh-my-opencode.json` caps output tokens per agent to fit local model context windows:

| OpenCode Agent | Norse Agent | maxTokens | Rationale |
|----------------|-------------|-----------|-----------|
| sisyphus | Thor (64K) | 32,768 | 32K output + ~32K input |
| hephaestus, sisyphus-junior | Valkyrie (48K) | 24,576 | 24K output + ~24K input |
| oracle, prometheus | Odin (32K) | 16,384 | 16K output + ~16K input |
| metis, momus | Frigga (16K) | 8,192 | 8K output + ~8K input |
| librarian, explore, atlas | Heimdall (8K) | 4,096 | 4K output + ~4K input |

Copy this file to `.opencode/` in any target project. Remove/raise limits when using cloud models.

---

## Memory Budget (Legacy vLLM)

### Phase 6 vLLM AWQ Allocation (128 GB shared UMA)

Legacy vLLM profile memory estimates (kept for reference).

| Profile | Agents Running | Total VRAM (est) |
|---------|----------------|------------------|
| standard | Thor (~9) + Valkyrie (~18) | ~27 GB |
| heimdall | Thor + Valkyrie + Heimdall (~3) | ~30 GB |
| loki | Thor + Valkyrie + Loki (~6) | ~33 GB |
| frigga | Thor + Valkyrie + Frigga (~9) | ~36 GB |
| odin | Thor (~9) + Odin (~42) | ~51 GB |

**Note:** Odin profile stops Valkyrie to free VRAM. Utility agents can coexist with Thor+Valkyrie.

> For detailed memory math, KV cache calculations, and phase evolution, see [DECISIONS.md](DECISIONS.md).
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

- Check logs: `docker logs llama_thor` / `docker logs llama_valkyrie` / `docker logs llama_odin`
- For vLLM agents: `docker logs vllm_heimdall` / `docker logs vllm_loki` / `docker logs vllm_frigga`
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
4. Test: `./scripts/up hybrid && ./scripts/health`
5. Validate compose: `docker compose -f compose/hybrid.yml --env-file .env config`
6. Submit a pull request

### Guidelines

- Keep `.env.example` up to date with any new environment variables
- Don't commit `.env`, `secrets/`, or model files
- Test compose files with `docker compose config` before committing
- Maintain script compatibility with `bash` (`set -euo pipefail`)
- Update DECISIONS.md for architecture changes, AGENTS.md for agent changes
