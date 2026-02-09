# AGENTS.md — Norse Agent Architecture

6-agent architecture for local AI inference on AMD Strix Halo, inspired by Norse mythology. Each agent has a dedicated model, role, and resource allocation optimized for single-user coding workflows.

For hardware research, quantization results, and memory math see [DECISIONS.md](DECISIONS.md).
For setup instructions and quick start see [README.md](README.md).

---

## Agent Overview

| Agent | Role | Backend | Port | Model | Context | Resource |
|-------|------|---------|------|-------|---------|----------|
| Thor ⚡ | Primary Commander | vLLM (GPU, BF16) | 8001 | Qwen/Qwen2.5-14B-Instruct | 64K | GPU 0.35 |
| Valkyrie 🛡 | Execution Specialist | vLLM (GPU, BF16) | 8002 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 48K | GPU 0.60 |
| Odin 👁️ | Supreme Architect | llama.cpp (CPU) | 8011 | Llama-3.3-70B-Instruct Q4_K_M | 32K | 16 threads |
| Heimdall 👁 | Guardian | llama.cpp (CPU) | 8012 | Qwen2.5-3B-Instruct Q4_K_M | 8K | 8 threads |
| Loki 🧠 | Adversarial Intelligence | llama.cpp (CPU) | 8013 | Qwen2.5-7B-Instruct Q4_K_M | 16K | 8 threads |
| Frigga 🌿 | Knowledge Curator | llama.cpp (CPU) | 8014 | Qwen2.5-14B-Instruct Q4_K_M | 32K | 12 threads |

---

## Thor ⚡ — Primary Commander

**Norse parallel**: Thor is the protector of Asgard and the strongest of the gods — the first line of action, the one who takes charge and delegates to others.

**Role**: Orchestrator and planner. Receives user requests, breaks them into tasks, delegates to specialist agents, maintains project context across the conversation.

**Model**: Qwen2.5-14B-Instruct (dense, BF16)
- 14B dense model with strong instruction following
- Long context (64K) for maintaining full project conversation history
- Not the largest model, but the most reliable for planning and delegation

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8001 | `THOR_PORT` |
| Model | Qwen/Qwen2.5-14B-Instruct | `THOR_MODEL` |
| GPU fraction | 0.35 (~44.8 GB) | `THOR_GPU_UTIL` |
| Context window | 65,536 | `THOR_MAX_LEN` |
| KV cache dtype | auto (BF16) | `THOR_KV_CACHE_DTYPE` |
| Max sequences | 4 | `THOR_MAX_NUM_SEQS` |

**OpenCode mapping**: `primary` agent (sisyphus)

---

## Valkyrie 🛡 — Execution Specialist

**Norse parallel**: Valkyries choose the worthy and execute with precision — they are the elite warriors who carry out the critical missions.

**Role**: Code generation, tool use, file edits, test writing. The hands-on builder that turns Thor's plans into working code.

**Model**: Qwen3-Coder-30B-A3B-Instruct (MoE, BF16)
- 30.5B total parameters, 3.3B active per token (Mixture of Experts)
- Purpose-built for code generation, function calling, tool use
- 48K context for large code diffs and multi-turn coding

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8002 | `VALKYRIE_PORT` |
| Model | Qwen/Qwen3-Coder-30B-A3B-Instruct | `VALKYRIE_MODEL` |
| GPU fraction | 0.60 (~76.8 GB) | `VALKYRIE_GPU_UTIL` |
| Context window | 49,152 | `VALKYRIE_MAX_LEN` |
| KV cache dtype | auto (BF16) | `VALKYRIE_KV_CACHE_DTYPE` |
| Max sequences | 4 | `VALKYRIE_MAX_NUM_SEQS` |

**OpenCode mapping**: `coder` agent (hephaestus, sisyphus-junior)

---

## Odin 👁️ — Supreme Architect

**Norse parallel**: Odin is the all-seeing father who sacrificed an eye for wisdom — he sees deeper than anyone but acts only when the stakes are highest.

**Role**: Escalation reviewer for architecture decisions, security audits, deep code review, and complex debugging. Called rarely but provides the highest-quality reasoning.

**Model**: Llama-3.3-70B-Instruct Q4_K_M GGUF
- 70B dense model — strongest reasoning capability in the fleet
- Runs on CPU via llama.cpp — does NOT compete for GPU memory
- Slow (~1-3 tok/s on CPU) but acceptable for occasional escalation tasks

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8011 | `ODIN_PORT` |
| GGUF file | llama-3.3-70b-instruct.Q4_K_M.gguf | `ODIN_GGUF` |
| Context window | 32,768 | `ODIN_CTX` |
| CPU threads | 16 | `ODIN_THREADS` |

**OpenCode mapping**: `reviewer` agent (oracle, prometheus)

---

## Heimdall 👁 — Guardian

**Norse parallel**: Heimdall guards the Bifrost bridge and can see and hear everything across the nine realms — the eternal watchman who catches what others miss.

**Role**: Fast validation, monitoring, policy enforcement, regression detection. Small model for quick checks — build verification, simple lookups, format validation.

**Model**: Qwen2.5-3B-Instruct Q4_K_M GGUF
- 3B model — negligible memory footprint (~2 GB)
- Near-instant responses on CPU
- Validation tasks are short by nature (8K context is sufficient)

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8012 | `HEIMDALL_PORT` |
| GGUF file | qwen2.5-3b-instruct.Q4_K_M.gguf | `HEIMDALL_GGUF` |
| Context window | 8,192 | `HEIMDALL_CTX` |
| CPU threads | 8 | `HEIMDALL_THREADS` |

**OpenCode mapping**: `utility` agent (librarian, explore, atlas)

---

## Loki 🧠 — Adversarial Intelligence

**Norse parallel**: Loki is the trickster god who challenges assumptions, finds weaknesses, and proposes unconventional solutions — the contrarian voice that strengthens through opposition.

**Role**: Adversarial testing of proposed solutions, edge case generation, assumption challenging, alternative strategy proposals. The agent that tries to break what others build.

**Model**: Qwen2.5-7B-Instruct Q4_K_M GGUF
- 7B model — balanced capability for adversarial tasks
- 16K context for analyzing solutions and generating counterexamples

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8013 | `LOKI_PORT` |
| GGUF file | qwen2.5-7b-instruct.Q4_K_M.gguf | `LOKI_GGUF` |
| Context window | 16,384 | `LOKI_CTX` |
| CPU threads | 8 | `LOKI_THREADS` |

**OpenCode mapping**: `loki` agent (no standard OpenCode mapping yet — future custom routing)

---

## Frigga 🌿 — Knowledge Curator

**Norse parallel**: Frigga is the queen of Asgard who knows all fates but speaks them selectively — the keeper of knowledge and long-term memory.

**Role**: Documentation generation, context compression, session summarization, long-term memory management. Processes large contexts into concise, reusable knowledge.

**Model**: Qwen2.5-14B-Instruct Q4_K_M GGUF
- 14B model — quality writing at moderate CPU speed
- 32K context for comprehensive documentation and summaries

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8014 | `FRIGGA_PORT` |
| GGUF file | qwen2.5-14b-instruct.Q4_K_M.gguf | `FRIGGA_GGUF` |
| Context window | 32,768 | `FRIGGA_CTX` |
| CPU threads | 12 | `FRIGGA_THREADS` |

**OpenCode mapping**: `frigga` agent (metis, momus)

---

## Escalation Doctrine

Inter-agent communication paths for when one agent needs another's expertise.

### Thor → Odin (Escalation)

Trigger conditions:
- Architecture uncertainty — design decisions affecting multiple systems
- Repeated failures — 2+ failed fix attempts on the same problem
- Security-critical — authentication, authorization, data handling
- Large refactors — changes spanning many files or modules

### Thor → Loki (Adversarial Review)

Trigger conditions:
- Proposed solution seems fragile or untested
- Need edge case analysis before deployment
- Want alternative approaches to a problem
- Robustness verification for critical paths

### Heimdall → Thor (Alert)

Trigger conditions:
- Policy violation detected in code changes
- Regression found in test suite
- Build anomaly or unexpected failure
- Monitoring threshold exceeded

### Thor ↔ Frigga (Knowledge)

Trigger conditions:
- Documentation needs writing or updating
- Session context exceeds model limits — needs compression
- Long-term knowledge needs archiving for future sessions
- Cross-session continuity required

---

## Memory Budget

### GPU (128 GB shared UMA)

| Agent | Fraction | Memory | Weights (BF16) | KV Cache | Slack |
|-------|----------|--------|----------------|----------|-------|
| Thor | 0.35 | 44.8 GB | ~28 GB | ~10 GB (64K) | 6.8 GB |
| Valkyrie | 0.60 | 76.8 GB | ~57 GB | ~4.5 GB (48K) | 15.3 GB |
| System | 0.05 | 6.4 GB | — | — | — |

### CPU (system RAM, Q4_K_M GGUF)

| Agent | Model Size | RAM at Load |
|-------|-----------|-------------|
| Odin | ~40 GB | ~42 GB |
| Frigga | ~8 GB | ~10 GB |
| Loki | ~4 GB | ~5 GB |
| Heimdall | ~2 GB | ~3 GB |
| **Total** | | **~60 GB** |

CPU agents share physical RAM with GPU allocations. Running all 4 CPU agents simultaneously adds ~60 GB on top. A 128 GB swap file is essential for system stability.

---

## OpenCode Agent Token Limits

The `.opencode/oh-my-opencode.json` template caps output tokens per agent to fit local model context windows:

| OpenCode Agent | Norse Agent | Context | maxTokens | Rationale |
|----------------|-------------|---------|-----------|-----------|
| sisyphus | Thor | 64K | 32,768 | 32K output + ~32K input |
| hephaestus, sisyphus-junior | Valkyrie | 48K | 24,576 | 24K output + ~24K input |
| oracle, prometheus | Odin | 32K | 16,384 | 16K output + ~16K input |
| metis, momus | Frigga | 32K | 16,384 | 16K output + ~16K input |
| librarian, explore, atlas | Heimdall | 8K | 4,096 | 4K output + ~4K input |

Copy this file to `.opencode/` in any target project. Remove or raise limits when using cloud models.

---

## Docker Services

| Agent | Service Name | Compose File |
|-------|-------------|-------------|
| Thor | `vllm_thor` | `compose/vllm.yml` |
| Valkyrie | `vllm_valkyrie` | `compose/vllm.yml` |
| Odin | `llama_odin` | `compose/cpu.yml` |
| Heimdall | `llama_heimdall` | `compose/cpu.yml` |
| Loki | `llama_loki` | `compose/cpu.yml` |
| Frigga | `llama_frigga` | `compose/cpu.yml` |

---

## Future Considerations

1. **Loki routing**: Loki doesn't have a standard OpenCode agent mapping yet. Needs custom routing or a dedicated subagent type.

2. **Frigga automation**: Context compression could be automated via cron jobs or git hooks — summarize session transcripts and store as project memory.

3. **GPU migration for CPU agents**: If future ROCm/vLLM updates bring working INT4 or FP8 on RDNA 3.5 (gfx1151), Odin/Loki/Frigga/Heimdall could move to GPU for dramatically faster inference.

4. **Speculative decoding**: Use a small drafter model alongside Valkyrie for faster code generation. vLLM supports this natively.

5. **Dynamic loading**: CPU agents don't need to run simultaneously. A launcher script could start/stop agents on demand based on escalation triggers.
