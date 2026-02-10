# AGENTS.md — Norse Agent Architecture

6-agent architecture for local AI inference on AMD Strix Halo, inspired by Norse mythology. Each agent has a dedicated model and role, optimized for single-user coding workflows.

For setup instructions and quick start see [README.md](README.md).

---

## Agent Overview

**Phase 7 — Ollama-First Architecture**: All 6 agents run on **Ollama** using the OpenAI-compatible API. A single Ollama daemon handles loading/unloading and concurrency.

| Agent | Role | Backend | Port | Model (Ollama) | Context | VRAM |
|-------|------|---------|------|-------|---------|------|
| Thor ⚡ | Primary Commander | Ollama | 11434 | Qwen2.5-14B (64K) | 64K | ~9 GB |
| Valkyrie 🛡 | Execution Specialist | Ollama | 11434 | Qwen3-Coder-30B-A3B (64K) | 64K | ~18 GB |
| Odin 👁️ | Supreme Architect | Ollama | 11434 | Llama-3.3-70B (64K) | 64K | ~42 GB |
| Heimdall 👁 | Guardian | Ollama | 11434 | Qwen2.5-3B | 16K | ~3 GB |
| Loki 🧠 | Adversarial Intelligence | Ollama | 11434 | Qwen2.5-7B | 32K | ~6 GB |
| Frigga 🌿 | Knowledge Curator | Ollama | 11434 | Qwen2.5-14B (64K) | 64K | ~9 GB |

---

## Thor ⚡ — Primary Commander

**Role**: Orchestrator and planner. Receives user requests, breaks them into tasks, delegates to specialist agents, maintains project context.

**Model**: Qwen2.5-14B (Ollama)
- 14B dense model with strong instruction following
- 32K context for project history
- Ollama OpenAI-compatible API

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8001 | `THOR_PORT` |
| GPU Util | 0.20 | `THOR_GPU_UTIL` |

---

## Valkyrie 🛡 — Execution Specialist

**Role**: Code generation, tool use, file edits. The hands-on builder.

**Model**: Qwen3-Coder-30B-A3B (Ollama)
- 30.5B total parameters (MoE)
- Purpose-built for code generation
- 32K context
- Ollama OpenAI-compatible API

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8002 | `VALKYRIE_PORT` |
| GPU Util | 0.40 | `VALKYRIE_GPU_UTIL` |

---

## Odin 👁️ — Supreme Architect

**Role**: Deep reasoning, architecture review, complexity analysis.

**Model**: Llama-3.3-70B (Ollama)
- 70B dense model — strongest reasoning capability
- Ollama OpenAI-compatible API

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8011 | `ODIN_PORT` |
| GPU Util | 0.50 | `ODIN_GPU_UTIL` |

---

## Heimdall 👁 — Guardian

**Role**: Fast validation, monitoring, simple checks.

**Model**: Qwen2.5-3B (Ollama)
- Tiny footprint (~3 GB)
- Fast response for high-frequency checks

---

## Loki 🧠 — Adversarial Intelligence

**Role**: Adversarial testing, edge case generation.

**Model**: Qwen2.5-7B (Ollama)
- Balanced capability for testing (~6 GB)

---

## Frigga 🌿 — Knowledge Curator

**Role**: Documentation, summarization, context compression.

**Model**: Qwen2.5-14B (Ollama)
- Excellent summarization (~9 GB)

---

## Escalation Doctrine (Legacy Bifrost)

The Bifrost scheduler managed resources dynamically in the vLLM era. Kept for reference.

---

## Memory Budget (128 GB UMA, legacy vLLM)

**Standard profile (Thor + Valkyrie):**
| Agent | VRAM | Notes |
|-------|------|-------|
| Thor | ~9 GB | Core |
| Valkyrie | ~18 GB | Standard |
| **Total** | **~27 GB** | |

**Odin profile (Thor + Odin):**
| Agent | VRAM | Notes |
|-------|------|-------|
| Thor | ~9 GB | Core |
| Odin | ~42 GB | Reasoning |
| **Total** | **~51 GB** | Valkyrie stopped |

**Utility Overhead:**
Adding Heimdall (+3GB) or Frigga (+9GB) to Standard profile fits easily within budget.

---

## Docker Services

| Service | Compose File |
|-------|-------------|-------------|
| `ollama` | `compose/ollama.yml` |
