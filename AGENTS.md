# AGENTS.md — Norse Agent Architecture

6-agent architecture for local AI inference on AMD Strix Halo, inspired by Norse mythology. Each agent has a dedicated model, role, and resource allocation optimized for single-user coding workflows.

For setup instructions and quick start see [README.md](README.md).

---

## Agent Overview

**Phase 6 — Full vLLM + AWQ Architecture**: All 6 agents run on **vLLM** using **AWQ 4-bit** quantization. This provides maximum performance (throughput/latency) on AMD ROCm while maintaining high memory efficiency. Agents are managed by the **Bifrost** scheduler.

| Agent | Role | Backend | Port | Model (AWQ 4-bit) | Context | VRAM |
|-------|------|---------|------|-------|---------|------|
| Thor ⚡ | Primary Commander | vLLM | 8001 | Qwen2.5-14B-Instruct | 32K | ~9 GB |
| Valkyrie 🛡 | Execution Specialist | vLLM | 8002 | Qwen3-Coder-30B-A3B | 32K | ~18 GB |
| Odin 👁️ | Supreme Architect | vLLM | 8011 | Llama-3.3-70B-Instruct | 32K | ~42 GB |
| Heimdall 👁 | Guardian | vLLM | 8012 | Qwen2.5-3B-Instruct | 32K | ~3 GB |
| Loki 🧠 | Adversarial Intelligence | vLLM | 8013 | Qwen2.5-7B-Instruct | 32K | ~6 GB |
| Frigga 🌿 | Knowledge Curator | vLLM | 8014 | Qwen2.5-14B-Instruct | 32K | ~9 GB |

---

## Thor ⚡ — Primary Commander

**Role**: Orchestrator and planner. Receives user requests, breaks them into tasks, delegates to specialist agents, maintains project context.

**Model**: Qwen2.5-14B-Instruct AWQ
- 14B dense model with strong instruction following
- 32K context for project history
- vLLM serving with prefix caching

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8001 | `THOR_PORT` |
| GPU Util | 0.20 | `THOR_GPU_UTIL` |

---

## Valkyrie 🛡 — Execution Specialist

**Role**: Code generation, tool use, file edits. The hands-on builder.

**Model**: Qwen3-Coder-30B-A3B-Instruct AWQ
- 30.5B total parameters (MoE)
- Purpose-built for code generation
- 32K context
- vLLM serving with prefix caching

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8002 | `VALKYRIE_PORT` |
| GPU Util | 0.40 | `VALKYRIE_GPU_UTIL` |

---

## Odin 👁️ — Supreme Architect

**Role**: Deep reasoning, architecture review, complexity analysis.

**Model**: Llama-3.3-70B-Instruct AWQ
- 70B dense model — strongest reasoning capability
- vLLM serving (Standard profile stops Valkyrie to run Odin)

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8011 | `ODIN_PORT` |
| GPU Util | 0.50 | `ODIN_GPU_UTIL` |

---

## Heimdall 👁 — Guardian

**Role**: Fast validation, monitoring, simple checks.

**Model**: Qwen2.5-3B-Instruct AWQ
- Tiny footprint (~3 GB)
- Fast response for high-frequency checks

---

## Loki 🧠 — Adversarial Intelligence

**Role**: Adversarial testing, edge case generation.

**Model**: Qwen2.5-7B-Instruct AWQ
- Balanced capability for testing (~6 GB)

---

## Frigga 🌿 — Knowledge Curator

**Role**: Documentation, summarization, context compression.

**Model**: Qwen2.5-14B-Instruct AWQ
- Excellent summarization (~9 GB)

---

## Escalation Doctrine (Bifrost)

The **Bifrost** scheduler manages resources dynamically.

1. **Standard Profile**: Thor + Valkyrie (Always on/Ready).
2. **Utility Profile**: Summon Heimdall, Loki, or Frigga on demand (coexists with Standard).
3. **Odin Profile**: Stops Valkyrie to free VRAM for Odin (Thor remains active).

---

## Memory Budget (128 GB UMA)

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

| Agent | Service Name | Compose File |
|-------|-------------|-------------|
| Thor | `vllm_thor` | `compose/hybrid.yml` |
| Valkyrie | `vllm_valkyrie` | `compose/hybrid.yml` |
| Odin | `vllm_odin` | `compose/hybrid.yml` |
| Heimdall | `vllm_heimdall` | `compose/hybrid.yml` |
| Loki | `vllm_loki` | `compose/hybrid.yml` |
| Frigga | `vllm_frigga` | `compose/hybrid.yml` |
| Scheduler | `bifrost` | `compose/hybrid.yml` |

