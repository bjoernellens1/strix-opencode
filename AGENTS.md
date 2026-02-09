# AGENTS.md — Norse Agent Architecture

6-agent architecture for local AI inference on AMD Strix Halo, inspired by Norse mythology. Each agent has a dedicated model, role, and resource allocation optimized for single-user coding workflows.

For hardware research, quantization results, and memory math see [DECISIONS.md](DECISIONS.md).
For setup instructions and quick start see [README.md](README.md).

---

## Agent Overview

**Phase 5 — Hybrid Architecture**: llama.cpp GGUF Q4_K_M for main agents (Thor, Valkyrie, Odin) + vLLM BF16 for utility agents (Heimdall, Loki, Frigga). All agents run on GPU.

| Agent | Role | Backend | Port | Model | Context | VRAM |
|-------|------|---------|------|-------|---------|------|
| Thor ⚡ | Primary Commander | llama.cpp (GGUF, GPU) | 8001 | Qwen2.5-14B-Instruct Q4_K_M | 64K | ~8 GB |
| Valkyrie 🛡 | Execution Specialist | llama.cpp (GGUF, GPU) | 8002 | Qwen3-Coder-30B-A3B-Instruct Q4_K_M | 48K | ~17 GB |
| Odin 👁️ | Supreme Architect | llama.cpp (GGUF, GPU) | 8011 | Llama-3.3-70B-Instruct Q4_K_M | 32K | ~40 GB |
| Heimdall 👁 | Guardian | vLLM (BF16, GPU) | 8012 | Qwen2.5-3B-Instruct | 8K | ~6 GB |
| Loki 🧠 | Adversarial Intelligence | vLLM (BF16, GPU) | 8013 | Qwen2.5-7B-Instruct | 16K | ~14 GB |
| Frigga 🌿 | Knowledge Curator | vLLM (BF16, GPU) | 8014 | Qwen2.5-14B-Instruct | 16K | ~28 GB |

---

## Thor ⚡ — Primary Commander

**Norse parallel**: Thor is the protector of Asgard and the strongest of the gods — the first line of action, the one who takes charge and delegates to others.

**Role**: Orchestrator and planner. Receives user requests, breaks them into tasks, delegates to specialist agents, maintains project context across the conversation.

**Model**: Qwen2.5-14B-Instruct Q4_K_M GGUF
- 14B dense model with strong instruction following
- Long context (64K) for maintaining full project conversation history
- Quantized to Q4_K_M (~8 GB) for efficient GPU memory usage
- llama.cpp with ROCm GPU offload (`-ngl 999 --flash-attn`)

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8001 | `THOR_PORT` |
| GGUF file | qwen2.5-14b-instruct-q4_k_m.gguf | `THOR_GGUF` |
| Context window | 65,536 | `THOR_CTX` |
| CPU threads | 8 | `THOR_THREADS` |

**OpenCode mapping**: `primary` agent (sisyphus)

---

## Valkyrie 🛡 — Execution Specialist

**Norse parallel**: Valkyries choose the worthy and execute with precision — they are the elite warriors who carry out the critical missions.

**Role**: Code generation, tool use, file edits, test writing. The hands-on builder that turns Thor's plans into working code.

**Model**: Qwen3-Coder-30B-A3B-Instruct Q4_K_M GGUF
- 30.5B total parameters, 3.3B active per token (Mixture of Experts)
- Purpose-built for code generation, function calling, tool use
- 48K context for large code diffs and multi-turn coding
- Quantized to Q4_K_M (~17 GB) for GPU memory efficiency

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8002 | `VALKYRIE_PORT` |
| GGUF file | Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf | `VALKYRIE_GGUF` |
| Context window | 49,152 | `VALKYRIE_CTX` |
| CPU threads | 8 | `VALKYRIE_THREADS` |

**OpenCode mapping**: `coder` agent (hephaestus, sisyphus-junior)

---

## Odin 👁️ — Supreme Architect

**Norse parallel**: Odin is the all-seeing father who sacrificed an eye for wisdom — he sees deeper than anyone but acts only when the stakes are highest.

**Role**: Escalation reviewer for architecture decisions, security audits, deep code review, and complex debugging. Called rarely but provides the highest-quality reasoning.

**Model**: Llama-3.3-70B-Instruct Q4_K_M GGUF
- 70B dense model — strongest reasoning capability in the fleet
- Runs on GPU via llama.cpp — does NOT compete for GPU memory with Thor/Valkyrie when active (Bifrost scheduler stops Valkyrie)
- Quantized to Q4_K_M (~40 GB) for GPU memory efficiency

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8011 | `ODIN_PORT` |
| GGUF file | llama-3.3-70b-instruct-Q4_K_M.gguf | `ODIN_GGUF` |
| Context window | 32,768 | `ODIN_CTX` |
| CPU threads | 16 | `ODIN_THREADS` |

**OpenCode mapping**: `reviewer` agent (oracle, prometheus)

---

## Heimdall 👁 — Guardian

**Norse parallel**: Heimdall guards the Bifrost bridge and can see and hear everything across the nine realms — the eternal watchman who catches what others miss.

**Role**: Fast validation, monitoring, policy enforcement, regression detection. Small model for quick checks — build verification, simple lookups, format validation.

**Model**: Qwen2.5-3B-Instruct BF16 (vLLM)
- 3B model — negligible memory footprint (~6 GB BF16)
- Fast responses on GPU
- Validation tasks are short by nature (8K context is sufficient)

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8012 | `HEIMDALL_PORT` |
| Model | Qwen/Qwen2.5-3B-Instruct | `HEIMDALL_MODEL` |
| Context window | 8,192 | `HEIMDALL_MAX_LEN` |
| GPU fraction | 0.05 | `HEIMDALL_GPU_UTIL` |

**OpenCode mapping**: `utility` agent (librarian, explore, atlas)

---

## Loki 🧠 — Adversarial Intelligence

**Norse parallel**: Loki is the trickster god who challenges assumptions, finds weaknesses, and proposes unconventional solutions — the contrarian voice that strengthens through opposition.

**Role**: Adversarial testing of proposed solutions, edge case generation, assumption challenging, alternative strategy proposals. The agent that tries to break what others build.

**Model**: Qwen2.5-7B-Instruct BF16 (vLLM)
- 7B model — balanced capability for adversarial tasks (~14 GB BF16)
- 16K context for analyzing solutions and generating counterexamples
- On-demand via Bifrost scheduler

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8013 | `LOKI_PORT` |
| Model | Qwen/Qwen2.5-7B-Instruct | `LOKI_MODEL` |
| Context window | 16,384 | `LOKI_MAX_LEN` |
| GPU fraction | 0.12 | `LOKI_GPU_UTIL` |

**OpenCode mapping**: `loki` agent (no standard OpenCode mapping yet — future custom routing)

---

## Frigga 🌿 — Knowledge Curator

**Norse parallel**: Frigga is the queen of Asgard who knows all fates but speaks them selectively — the keeper of knowledge and long-term memory.

**Role**: Documentation generation, context compression, session summarization, long-term memory management. Processes large contexts into concise, reusable knowledge.

**Model**: Qwen2.5-14B-Instruct BF16 (vLLM)
- 14B model — quality writing at GPU speed (~28 GB BF16)
- 16K context for comprehensive documentation and summaries
- On-demand via Bifrost scheduler

**Configuration**:
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Port | 8014 | `FRIGGA_PORT` |
| Model | Qwen/Qwen2.5-14B-Instruct | `FRIGGA_MODEL` |
| Context window | 16,384 | `FRIGGA_MAX_LEN` |
| GPU fraction | 0.25 | `FRIGGA_GPU_UTIL` |

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

### Phase 5 Hybrid GPU Allocation (128 GB shared UMA)

All agents run on GPU. Bifrost scheduler manages memory by stopping conflicting agents.

**Standard profile (Thor + Valkyrie):**

| Agent | Backend | VRAM | Notes |
|-------|---------|------|-------|
| Thor | llama.cpp GGUF | ~8 GB | Always running |
| Valkyrie | llama.cpp GGUF | ~17 GB | Standard profile |
| **Total** | | **~25 GB** | |

**With utility agent (Thor + Valkyrie + one utility):**

| Agent | Backend | VRAM | Notes |
|-------|---------|------|-------|
| Thor | llama.cpp GGUF | ~8 GB | Always running |
| Valkyrie | llama.cpp GGUF | ~17 GB | Standard profile |
| Heimdall | vLLM BF16 | ~6 GB | On-demand |
| **Total** | | **~31 GB** | |
| Loki | vLLM BF16 | ~14 GB | (or instead of Heimdall) |
| Frigga | vLLM BF16 | ~28 GB | (or instead of Heimdall) |

**Odin profile (stops Valkyrie and utility agents):**

| Agent | Backend | VRAM | Notes |
|-------|---------|------|-------|
| Thor | llama.cpp GGUF | ~8 GB | Always running |
| Odin | llama.cpp GGUF | ~40 GB | Stops Valkyrie |
| **Total** | | **~48 GB** | |

### Memory Budget by Profile

| Profile | Agents Running | Total VRAM |
|---------|----------------|------------|
| standard | Thor + Valkyrie | ~25 GB |
| heimdall | Thor + Valkyrie + Heimdall | ~31 GB |
| loki | Thor + Valkyrie + Loki | ~39 GB |
| frigga | Thor + Valkyrie + Frigga | ~53 GB |
| odin | Thor + Odin (stops Valkyrie) | ~48 GB |

---

## OpenCode Agent Token Limits

The `.opencode/oh-my-opencode.json` template caps output tokens per agent to fit local model context windows:

| OpenCode Agent | Norse Agent | Context | maxTokens | Rationale |
|----------------|-------------|---------|-----------|-----------|
| sisyphus | Thor | 64K | 32,768 | 32K output + ~32K input |
| hephaestus, sisyphus-junior | Valkyrie | 48K | 24,576 | 24K output + ~24K input |
| oracle, prometheus | Odin | 32K | 16,384 | 16K output + ~16K input |
| metis, momus | Frigga | 16K | 8,192 | 8K output + ~8K input |
| librarian, explore, atlas | Heimdall | 8K | 4,096 | 4K output + ~4K input |

Copy this file to `.opencode/` in any target project. Remove or raise limits when using cloud models.

---

## Docker Services

| Agent | Service Name | Compose File |
|-------|-------------|-------------|
| Thor | `llama_thor` | `compose/hybrid.yml` |
| Valkyrie | `llama_valkyrie` | `compose/hybrid.yml` |
| Odin | `llama_odin` | `compose/hybrid.yml` |
| Heimdall | `vllm_heimdall` | `compose/hybrid.yml` |
| Loki | `vllm_loki` | `compose/hybrid.yml` |
| Frigga | `vllm_frigga` | `compose/hybrid.yml` |

---

## Future Considerations

1. **Loki routing**: Loki doesn't have a standard OpenCode agent mapping yet. Needs custom routing or a dedicated subagent type.

2. **Frigga automation**: Context compression could be automated via cron jobs or git hooks — summarize session transcripts and store as project memory.

3. **GPU migration for CPU agents**: If future ROCm/vLLM updates bring working INT4 or FP8 on RDNA 3.5 (gfx1151), Odin/Loki/Frigga/Heimdall could move to GPU for dramatically faster inference.

4. **Speculative decoding**: Use a small drafter model alongside Valkyrie for faster code generation. vLLM supports this natively.

5. **Dynamic loading**: CPU agents don't need to run simultaneously. A launcher script could start/stop agents on demand based on escalation triggers.
