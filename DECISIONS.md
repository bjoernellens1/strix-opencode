# DECISIONS.md — Strix Halo Local LLM Stack

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

## 2. Quantization: Exhaustive Testing Results on gfx1151

**Every quantization method available in vLLM was tested on this hardware. None are viable.**

### FP8 Weight Quantization (`--quantization fp8`)

**Result: CRASH**

```
torch._scaled_mm requires MI300+ or CUDA sm_89+
```

The FP8 matrix multiply kernel (`torch._scaled_mm`) is only implemented for AMD MI300-series (CDNA 3) and NVIDIA Ada Lovelace+ GPUs. RDNA 3.5 (gfx1151) has no hardware FP8 compute units.

### AWQ INT4 (Pre-quantized Checkpoint)

**Result: GPU HANG — requires hard reboot**

Tested with `Qwen/Qwen2.5-14B-Instruct-AWQ`. vLLM loaded the model and began serving, but crashed on the first inference request:

```
Memory access fault by GPU node-1 (Agent handle: 0x...)
...
GPU Hang
```

Root cause: vLLM auto-enables Triton AWQ kernels on ROCm (`VLLM_USE_TRITON_AWQ`). These Triton kernels are written for CDNA (MI-series) wave64 execution. RDNA 3.5 uses wave32 — the kernels produce invalid memory accesses.

The AWQ dequantize/gemm ops exist in the build (`awq_dequantize`, `awq_gemm` verified via Python import), but the Triton dispatch path crashes.

### FP8 KV Cache (`--kv-cache-dtype fp8`)

**Result: WORKS but with accuracy degradation**

```
WARNING: Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention.
This may cause accuracy issues.
```

FP8 KV cache is implemented as pure data packing (no FP8 compute required), so it runs on any hardware. However, on gfx1151 the attention kernel uses **uncalibrated quantization scales** (hardcoded 1.0), meaning the FP8 values are not properly scaled. This causes accuracy loss, especially on longer sequences.

**Decision: Do not use.** The memory savings (2x KV) are not worth the accuracy risk for an agent coding assistant where correctness matters.

### GPTQ, Marlin, bitsandbytes, GGUF-in-vLLM

**Result: NOT AVAILABLE on ROCm**

All require CUDA-only kernel libraries (Marlin, ExLlama, cutlass, bitsandbytes CUDA ops). Not compiled for ROCm in any vLLM build.

### Summary

| Method | Result | Error |
|--------|--------|-------|
| FP8 weights (`--quantization fp8`) | ❌ CRASH | `torch._scaled_mm requires MI300+` |
| AWQ INT4 (pre-quantized) | ❌ GPU HANG | Triton AWQ kernels → memory fault |
| FP8 KV cache (`--kv-cache-dtype fp8`) | ⚠️ ACCURACY LOSS | Uncalibrated scales (1.0) |
| GPTQ/Marlin/bitsandbytes/GGUF | ❌ N/A | CUDA-only kernels |

**Conclusion: ALL vLLM models must use BF16 weights + BF16 KV cache (`--kv-cache-dtype auto`) on Strix Halo gfx1151.**

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

### Phase 3: Current 4-tier architecture

Separates concerns across 4 tiers with dedicated models per role:

| Tier | Role | Backend | Port | Model | Context |
|------|------|---------|------|-------|---------|
| 1 | Orchestrator | vLLM (GPU) | 8001 | Qwen/Qwen2.5-14B-Instruct | 64K |
| 2 | Coder | vLLM (GPU) | 8002 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 48K |
| 3 | Reviewer | llama.cpp (CPU) | 8011 | Llama-3.3-70B-Instruct Q4_K_M | 32K |
| 0 | Utility | llama.cpp (CPU) | 8012 | Qwen2.5-3B-Instruct Q4_K_M | 8K |

**Why 4 tiers:**
- **Orchestrator** needs long context for project memory and coordination, not raw coding skill → 14B dense model with 64K context
- **Coder** needs the strongest code generation → 30B MoE with purpose-built coding capabilities
- **Reviewer** provides escalation for architecture decisions, deep review, security audit → 70B model via CPU (GGUF INT4, doesn't compete for GPU memory)
- **Utility** handles fast, cheap tasks (file summaries, simple transforms) → 3B model, near-instant on CPU

---

## 4. Model Selection (4-Tier)

### Tier 1: Orchestrator — Qwen2.5-14B-Instruct

- **Dense 14B model**: ~28 GB in BF16
- **Strong instruction following**: Good at planning, delegation, maintaining project context
- **64K context**: Orchestrator sees the full project conversation history
- **0.35 GPU util**: Fits comfortably with 16.8 GB KV headroom

### Tier 2: Coder — Qwen3-Coder-30B-A3B-Instruct

- **MoE architecture**: 30.5B total parameters, 3.3B active per token
- **Model weight size**: ~57 GB in BF16 (all 128 experts loaded, 8 active per forward pass)
- **Purpose-built for code**: Strong benchmarks on coding tasks, tool use, function calling
- **48K context**: Sufficient for large code diffs, file contents, and multi-turn coding conversations
- **0.60 GPU util**: Gets the largest allocation since it's the biggest model

### Tier 3: Reviewer — Llama-3.3-70B-Instruct Q4_K_M GGUF

- **70B dense model**: ~40 GB as Q4_K_M GGUF
- **CPU-only via llama.cpp**: Does NOT compete for GPU memory with vLLM tiers
- **Strong reasoning**: 70B dense model has deeper reasoning than smaller models
- **32K context**: Sufficient for review tasks (receives summaries, not raw histories)
- **Slow but high-quality**: CPU inference is ~1-3 tok/s, acceptable for occasional escalation

### Tier 0: Utility — Qwen2.5-3B-Instruct Q4_K_M GGUF

- **3B model**: ~2 GB as Q4_K_M GGUF, negligible memory footprint
- **CPU-only**: Instant load, fast for tiny tasks
- **8K context**: Utility tasks are short by definition
- **Use cases**: File summaries, simple transforms, quick lookups, template generation

---

## 5. Memory Budget (4-Tier)

### GPU Allocation (128 GB shared UMA)

| Component | Fraction | Memory | Weights | KV Budget |
|-----------|----------|--------|---------|-----------|
| Tier 2: Coder (vLLM) | 0.60 | ~76.8 GB | ~57 GB | ~19.8 GB |
| Tier 1: Orch (vLLM) | 0.35 | ~44.8 GB | ~28 GB | ~16.8 GB |
| System headroom | 0.05 | ~6.4 GB | — | — |

**Total GPU util: 0.95** — tight for single-user. Swap file + earlyoom strongly recommended.

### KV Cache Math (BF16 only)

Formula: `bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × 2 (BF16)`

| Model | Layers | KV Heads | Head Dim | Bytes/Token |
|-------|--------|----------|----------|-------------|
| Qwen2.5-14B-Instruct | 40 | 8 | 128 | 163,840 (~160 KB) |
| Qwen3-Coder-30B-A3B | 48 | 4 | 128 | 98,304 (~96 KB) |

### Context Window Fit

| Tier | Model | Context | KV Size | Budget | Slack |
|------|-------|---------|---------|--------|-------|
| 1 Orch | Qwen2.5-14B | 64K | ~10.0 GB | 16.8 GB | 6.8 GB |
| 2 Coder | Qwen3-Coder-30B | 48K | ~4.5 GB | 19.8 GB | 15.3 GB |

### CPU Tiers (RAM, not GPU)

| Tier | Model | GGUF Size | RAM at Load |
|------|-------|-----------|-------------|
| 3 Reviewer | Llama-3.3-70B Q4_K_M | ~40 GB | ~42 GB with KV |
| 0 Utility | Qwen2.5-3B Q4_K_M | ~2 GB | ~3 GB with KV |

CPU tiers share the same physical RAM. Not expected to run concurrently with both GPU tiers at full KV pressure.

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

| Agent | maxTokens | Tier | Rationale |
|-------|-----------|------|-----------|
| sisyphus | 32,768 | Orch (64K ctx) | 32K output leaves ~32K for input |
| hephaestus, sisyphus-junior | 24,576 | Coder (48K ctx) | 24K output leaves ~24K for input |
| oracle, prometheus, metis, momus | 16,384 | Reviewer (32K ctx) | 16K output leaves ~16K for input |
| librarian, explore, atlas | 4,096 | Utility (8K ctx) | 4K output leaves ~4K for input |

Copy to `.opencode/` in each target project. Remove/raise limits when using cloud models.

---

## 9. Future Optimization Strategies

### Near-term

1. **Monitor KV utilization**: vLLM exposes Prometheus metrics at `/metrics`. Track whether 64K/48K limits are actually hit.

2. **Speculative decoding**: Use a small model as draft for the 30B coder. vLLM supports this natively. Could improve generation speed significantly.

3. **ROCm kernel improvements**: gfx1151 is very new. Future ROCm updates may bring working FP8/AWQ kernels for RDNA 3.5.

### Medium-term

4. **Context tuning**: If KV monitoring shows consistent underuse, reduce context to free memory. If consistently hitting limits, consider reducing one tier's GPU util to give the other more.

5. **Tier 3/0 on GPU**: If vLLM ever supports GGUF or INT4 on ROCm, the reviewer/utility could move to GPU for faster inference.

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
