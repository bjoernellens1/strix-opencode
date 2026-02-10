# DECISIONS.md — Strix Halo Local LLM Stack (Norse Agent Architecture)

Technical decision log for the strix-opencode project. Documents research findings, architecture rationale, and optimization strategies for running local LLM inference on AMD Strix Halo hardware.

Last updated: 2026-02-09

---

## 1. Platform Summary

| Component | Value |
|-----------|-------|
| CPU | AMD Ryzen AI MAX+ PRO 395 (Strix Halo) — 16 cores / 32 threads |
| GPU | Integrated RDNA 3.5, 40 CUs, gfx1151 |
| VRAM | 512 MB dedicated + 128 GB GTT (shared with system RAM) |
| Memory bandwidth | ~212 GB/s (shared) |
| FP16 compute | ~59.4 TFLOPS |
| Disk | ~1.9 TB total, ~1.6 TB free |
| Container image | `docker.io/kyuz0/vllm-therock-gfx1151:latest` (Fedora 43, built 2026-02-03) |
| vLLM version | `0.16.0rc1.dev155+g61e632aea.d20260203` |
| ROCm stack | Bundled in container (gfx1151 target) |

**Key constraint**: CPU and GPU share the same 128 GB physical RAM. Every byte allocated to GPU (model weights, KV cache) is a byte unavailable to the OS, desktop, and browser. Memory budget management is critical.

---

## 2. Quantization: Testing Results on gfx1151

**Phase 6 Update: AWQ 4-bit is now fully functional and verified.**

### AWQ INT4 (Pre-quantized Checkpoint)

**Result: ✅ WORKS (with `kyuz0/vllm-therock-gfx1151`)**

Previously, AWQ resulted in GPU hangs due to Triton kernel incompatibilities with wave32 (RDNA 3.5). The current stack uses a specialized vLLM build (`kyuz0/vllm-therock-gfx1151`) that fixes these issues.

- **Verified with**: `Qwen/Qwen2.5-14B-Instruct-AWQ`, `casperhansen/llama-3.3-70b-instruct-awq`
- **Performance**: Sub-1s first-token latency, strong throughput on shared UMA memory.
- **Memory**: ~4x reduction in weight size vs BF16, enabling all 6 agents to run on GPU.

### FP8 Weight Quantization (`--quantization fp8`)

**Result: ❌ CRASH**

```
torch._scaled_mm requires MI300+ or CUDA sm_89+
```

The FP8 matrix multiply kernel (`torch._scaled_mm`) is only implemented for AMD MI300-series (CDNA 3). RDNA 3.5 (gfx1151) has no hardware FP8 compute units.

### FP8 KV Cache (`--kv-cache-dtype fp8`)

**Result: ⚠️ WORKS but with accuracy degradation**

FP8 KV cache is implemented as data packing, so it runs on any hardware. However, on gfx1151 the attention kernel uses **uncalibrated quantization scales** (hardcoded 1.0).

**Decision: Do not use.** AWQ 4-bit weights + BF16 KV cache (`--kv-cache-dtype auto`) provides the best balance of memory savings and correctness.

### Summary

| Method | Result | Status |
|--------|--------|--------|
| **AWQ INT4** | ✅ **SUCCESS** | **Default for Phase 6** |
| BF16 Weights | ✅ WORKS | Native support |
| FP8 weights | ❌ CRASH | Requires MI300+ hardware |
| FP8 KV cache | ⚠️ ACCURACY LOSS | Uncalibrated scales |
| GPTQ/Marlin | ❌ N/A | CUDA-only kernels |

**Conclusion: All agents in Phase 6 use AWQ 4-bit weights + BF16 KV cache for optimal performance on Strix Halo.**

---

## 3. Architecture Evolution

### Phase 1: Initial 3-service layout (failed)

| Service | Port | Model | GPU Util |
|---------|------|-------|----------|
| vllm_orchestrator | 8001 | openai/gpt-oss-120b | 0.45 |
| vllm_coder | 8002 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.35 |
| vllm_fast | 8004 | Qwen/Qwen2.5-7B-Instruct | 0.20 |

**Failed**: gpt-oss-120b too large. 3 weight copies. No system headroom.

### Phase 2: Consolidated 2-service layout

| Service | Port | Model | GPU Util |
|---------|------|-------|----------|
| vllm_orchestrator | 8001 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.65 |
| vllm_fast | 8004 | Qwen/Qwen2.5-7B-Instruct | 0.20 |

One model served both orch+coder roles. Worked but limited: no dedicated orchestrator intelligence, no escalation path, no adversarial testing.

### Phase 3: Current 6-agent Norse architecture

Separates concerns across 6 agents with dedicated models per role:

| Agent | Role | Backend | Port | Model | Context |
|-------|------|---------|------|-------|---------|
| Thor ⚡ | Primary Commander | vLLM (GPU) | 8001 | Qwen/Qwen2.5-14B-Instruct | 64K |
| Valkyrie 🛡 | Execution Specialist | vLLM (GPU) | 8002 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 48K |
| Odin 👁️ | Supreme Architect | llama.cpp (CPU) | 8011 | Llama-3.3-70B-Instruct Q4_K_M | 32K |
| Heimdall 👁 | Guardian | llama.cpp (CPU) | 8012 | Qwen2.5-3B-Instruct Q4_K_M | 8K |
| Loki 🧠 | Adversarial Intelligence | llama.cpp (CPU) | 8013 | Qwen2.5-7B-Instruct Q4_K_M | 16K |
| Frigga 🌿 | Knowledge Curator | llama.cpp (CPU) | 8014 | Qwen2.5-14B-Instruct Q4_K_M | 32K |

**Why 6 agents:**
- **Thor** needs long context for project memory and coordination, not raw coding skill → 14B dense model with 64K context
- **Valkyrie** needs the strongest code generation → 30B MoE with purpose-built coding capabilities
- **Odin** provides escalation for architecture decisions, deep review, security audit → 70B model via CPU (GGUF INT4, doesn't compete for GPU memory)
- **Heimdall** handles fast validation, monitoring, policy enforcement → 3B model, near-instant on CPU
- **Loki** generates adversarial challenges, edge cases, alternative approaches → 7B model for balanced capability
- **Frigga** manages documentation, context compression, long-term knowledge → 14B model for quality writing

### Phase 4: GPU-only via Bifrost scheduler

All 6 agents on GPU via vLLM BF16. Bifrost scheduler manages memory by stopping non-essential agents when Odin is summoned. However, this approach has limits:

- **Total VRAM pressure**: Running all 6 agents BF16 exceeds 120 GB budget
- **No quantization on gfx1151**: AWQ, FP8, GPTQ all fail (see Section 2)
- **Odin context limited**: QwQ-32B needs large memory, 20K context cap

### Phase 5: Hybrid Architecture (Legacy)

llama.cpp with ROCm GPU offload for main agents (Thor, Valkyrie, Odin) + vLLM BF16 for utility agents. Proved stable but lacked unified inference optimizations (vLLM prefix caching, PagedAttention).

### Phase 6: Full vLLM AWQ (Current)

**Transition to unified vLLM stack.** With the specialized `gfx1151` image, all agents now run on vLLM using AWQ 4-bit quantization.

| Agent | Backend | Model | VRAM (AWQ) | Context |
|-------|---------|-------|------------|---------|
| Thor ⚡ | vLLM (GPU) | Qwen2.5-14B-Instruct-AWQ | ~9 GB | 32K |
| Valkyrie 🛡 | vLLM (GPU) | Qwen3-Coder-30B-A3B-AWQ | ~18 GB | 32K |
| Odin 👁️ | vLLM (GPU) | Llama-3.3-70B-Instruct-AWQ | ~42 GB | 32K |
| Heimdall 👁 | vLLM (GPU) | Qwen2.5-3B-Instruct-AWQ | ~3 GB | 32K |
| Loki 🧠 | vLLM (GPU) | Qwen2.5-7B-Instruct-AWQ | ~6 GB | 32K |
| Frigga 🌿 | vLLM (GPU) | Qwen2.5-14B-Instruct-AWQ | ~9 GB | 32K |

**Why Full vLLM AWQ:**
- **Unified Stack**: Single provider backend simplifies development and deployment.
- **Prefix Caching**: vLLM's block-level caching improves multi-turn agent performance.
- **Improved VRAM Efficiency**: 4-bit weights allow 70B models to fit comfortably alongside others.
- **Lower Latency**: vLLM's continuous batching and AWQ kernels provide superior interactive speed.

**Memory profiles:**
- `standard`: Thor (~9) + Valkyrie (~18) = ~27 GB
- `heimdall`: Thor + Valkyrie + Heimdall (~3) = ~30 GB
- `loki`: Thor + Valkyrie + Loki (~6) = ~33 GB
- `frigga`: Thor + Valkyrie + Frigga (~9) = ~36 GB
- `odin`: Thor (~9) + Odin (~42) = ~51 GB (stops Valkyrie)

---

## 4. Model Selection (Norse Agents)

### Thor ⚡ — Qwen2.5-14B-Instruct

- **Dense 14B model**: ~28 GB in BF16
- **Strong instruction following**: Good at planning, delegation, maintaining project context
- **64K context**: Thor sees the full project conversation history
- **0.35 GPU util**: Fits comfortably with 16.8 GB KV headroom

### Valkyrie 🛡 — Qwen3-Coder-30B-A3B-Instruct

- **MoE architecture**: 30.5B total parameters, 3.3B active per token
- **Model weight size**: ~57 GB in BF16 (all 128 experts loaded, 8 active per forward pass)
- **Purpose-built for code**: Strong benchmarks on coding tasks, tool use, function calling
- **48K context**: Sufficient for large code diffs, file contents, and multi-turn coding conversations
- **0.60 GPU util**: Gets the largest allocation since it's the biggest model

### Odin 👁️ — Llama-3.3-70B-Instruct Q4_K_M GGUF

- **70B dense model**: ~40 GB as Q4_K_M GGUF
- **CPU-only via llama.cpp**: Does NOT compete for GPU memory with vLLM tiers
- **Strong reasoning**: 70B dense model has deeper reasoning than smaller models
- **32K context**: Sufficient for review tasks (receives summaries, not raw histories)
- **Slow but high-quality**: CPU inference is ~1-3 tok/s, acceptable for occasional escalation

### Heimdall 👁 — Qwen2.5-3B-Instruct Q4_K_M GGUF

- **3B model**: ~2 GB as Q4_K_M GGUF, negligible memory footprint
- **CPU-only**: Instant load, fast for tiny tasks
- **8K context**: Validation tasks are short by definition
- **Use cases**: Build verification, regression detection, policy enforcement, simple lookups

### Loki 🧠 — Qwen2.5-7B-Instruct Q4_K_M GGUF

- **7B model**: ~4 GB as Q4_K_M GGUF
- **CPU-only**: Balanced speed for adversarial tasks
- **16K context**: Sufficient for edge case generation and alternative approach analysis
- **Use cases**: Adversarial testing, assumption challenging, alternative strategy generation

### Frigga 🌿 — Qwen2.5-14B-Instruct Q4_K_M GGUF

- **14B model**: ~8 GB as Q4_K_M GGUF
- **CPU-only**: Quality writing at moderate speed
- **32K context**: Comprehensive summaries and documentation
- **Use cases**: Documentation generation, context compression, long-term memory management, session summaries

---

## 5. Memory Budget (Norse Agents)

### GPU Allocation (128 GB shared UMA)

| Component | Fraction | Memory | Weights | KV Budget |
|-----------|----------|--------|---------|-----------|
| Valkyrie (30B MoE) | 0.60 | ~76.8 GB | ~57 GB | ~19.8 GB |
| Thor (14B) | 0.35 | ~44.8 GB | ~28 GB | ~16.8 GB |
| System headroom | 0.05 | ~6.4 GB | — | — |

**Total GPU util: 0.95** — tight for single-user. Swap file + earlyoom strongly recommended.

### KV Cache Math (BF16 only)

Formula: `bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × 2 (BF16)`

| Model | Layers | KV Heads | Head Dim | Bytes/Token |
|-------|--------|----------|----------|-------------|
| Qwen2.5-14B-Instruct | 40 | 8 | 128 | 163,840 (~160 KB) |
| Qwen3-Coder-30B-A3B | 48 | 4 | 128 | 98,304 (~96 KB) |

### Context Window Fit

| Agent | Model | Context | KV Size | Budget | Slack |
|-------|-------|---------|---------|--------|-------|
| Thor | Qwen2.5-14B | 64K | ~10.0 GB | 16.8 GB | 6.8 GB |
| Valkyrie | Qwen3-Coder-30B | 48K | ~4.5 GB | 19.8 GB | 15.3 GB |

### CPU Agents (RAM, not GPU)

| Agent | Model | GGUF Size | RAM at Load |
|-------|-------|-----------|-------------|
| Odin | Llama-3.3-70B Q4_K_M | ~40 GB | ~42 GB with KV |
| Frigga | Qwen2.5-14B Q4_K_M | ~8 GB | ~10 GB with KV |
| Loki | Qwen2.5-7B Q4_K_M | ~4 GB | ~5 GB with KV |
| Heimdall | Qwen2.5-3B Q4_K_M | ~2 GB | ~3 GB with KV |

CPU agents share the same physical RAM. Running all 4 simultaneously adds ~60 GB on top of GPU allocation. Swap file essential.

### How vLLM Allocates Memory (from source code analysis)

Based on reading vLLM 0.16 source code inside the Docker container:

1. **Memory request** (`vllm/v1/worker/utils.py`):
   ```python
   requested_memory = math.ceil(total_memory * gpu_memory_utilization)
   ```

2. **Memory profiling** (`vllm/v1/worker/gpu_worker.py`):
   - vLLM runs a dummy forward pass to measure peak memory usage
   - `non_kv_cache_memory = model_weights + peak_activations + non_torch_overhead + 150 MB buffer`
   - `available_kv_cache_memory = requested_memory - non_kv_cache_memory`

3. **KV cache pre-allocation**: Blocks allocated at startup, assigned on demand via PagedAttention.

4. **Multiple instances supported** (from docstring): GPU memory utilization fractions can be split across instances.

5. **Auto-fit**: If `--max-model-len` exceeds KV budget, vLLM auto-reduces it.

### Shared-Memory Race Condition (Strix Halo specific)

When both vLLM containers start simultaneously on shared UMA memory, the memory profiler in each instance sees the **total GPU memory consumption from both processes**. The second instance's profiler sees the first instance's allocation and computes negative available KV cache.

**Fix**: Docker Compose `healthcheck` on the orchestrator (first to start, smaller, loads faster) + `depends_on: condition: service_healthy` on the coder. Staggered startup ensures accurate memory profiling.

---

## 6. vLLM Optimization Flags

### --enable-prefix-caching (default: true in v0.16)

Reuses KV cache blocks across requests sharing common prefixes (e.g., system prompts). Highly beneficial for agent workloads. Made explicit in compose for documentation.

### --max-num-seqs 4

Default is 256 (multi-user serving). Single-user needs 2-3 concurrent sequences max. Lower value = less activation memory overhead = more KV cache available.

### --enforce-eager (NOT enabled)

Disables HIP graph capture. Saves memory but hurts throughput. Consider if memory is extremely tight.

### --swap-space (default: 4 GiB)

vLLM's internal KV cache swap. On shared-memory Strix Halo, this reshuffles within the same RAM but lets vLLM's memory manager handle spikes. Left at default.

### --cpu-offload-gb (NOT used)

Moves model weights to CPU memory. On shared-memory systems, this adds copy overhead through the GPU memory controller without gaining real memory. Not recommended.

---

## 7. System Stability Measures

### Problem

With 128 GB shared memory and GPU claiming 95%, the system can easily run out of available memory.

### Solutions

**1. Enlarge swap file to 128 GB**
```bash
sudo swapoff /swap.img && sudo rm /swap.img
sudo fallocate -l 128G /swap.img && sudo chmod 600 /swap.img
sudo mkswap /swap.img && sudo swapon /swap.img
```

**2. Install earlyoom**
```bash
sudo apt install earlyoom && sudo systemctl enable --now earlyoom
```

**3. Increase swappiness to 80**
```bash
sudo sysctl vm.swappiness=80
echo 'vm.swappiness=80' | sudo tee /etc/sysctl.d/99-swappiness.conf
```

---

## 8. oh-my-opencode Agent Token Limits

OpenCode's built-in maxTokens defaults can exceed local model context windows. The template `.opencode/oh-my-opencode.json` caps output tokens per agent:

| Agent | maxTokens | Norse Agent | Rationale |
|-------|-----------|-------------|-----------|
| sisyphus | 32,768 | Thor (64K ctx) | 32K output leaves ~32K for input |
| hephaestus, sisyphus-junior | 24,576 | Valkyrie (48K ctx) | 24K output leaves ~24K for input |
| oracle, prometheus | 16,384 | Odin (32K ctx) | 16K output leaves ~16K for input |
| metis, momus | 16,384 | Frigga (32K ctx) | 16K output leaves ~16K for input |
| librarian, explore, atlas | 4,096 | Heimdall (8K ctx) | 4K output leaves ~4K for input |

Copy to `.opencode/` in each target project. Remove/raise limits when using cloud models.

---

## 9. Future Optimization Strategies

### Near-term

1. **Monitor KV utilization**: vLLM exposes Prometheus metrics at `/metrics`. Track whether 64K/48K limits are actually hit.

2. **Speculative decoding**: Use a small model as draft for the 30B coder. vLLM supports this natively. Could improve generation speed significantly.

3. **ROCm kernel improvements**: gfx1151 is very new. Future ROCm updates may bring working FP8/AWQ kernels for RDNA 3.5.

### Medium-term

4. **Context tuning**: If KV monitoring shows consistent underuse, reduce context to free memory. If consistently hitting limits, consider reducing one tier's GPU util to give the other more.

5. **CPU agents on GPU**: If vLLM ever supports GGUF or INT4 on ROCm, Odin/Loki/Frigga/Heimdall could move to GPU for faster inference.

### Long-term

6. **Dedicated VRAM**: Discrete GPU (MI300X, consumer RDNA with large VRAM) eliminates shared-memory bandwidth contention.

7. **Multi-node**: Split tiers across machines if workload grows beyond single-system capacity.

---

## Appendix: Key vLLM Configuration Defaults (v0.16)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_memory_utilization` | 0.9 | Fraction of GPU memory to request |
| `swap_space` | 4 GiB | CPU swap space for KV cache eviction |
| `cache_dtype` | "auto" | KV cache precision (follows model dtype) |
| `enable_prefix_caching` | true | Reuse KV blocks for common prefixes |
| `cpu_offload_gb` | 0 | Model weight offload to CPU memory |
| `block_size` | platform-dependent | PagedAttention block size |
