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

## 2. Model Selection

### Why Qwen3-Coder-30B-A3B-Instruct (orchestrator + coder)

The original orchestrator was `openai/gpt-oss-120b`, which failed to load — too large for the shared memory budget when running alongside other services.

**Selection criteria:**
1. Must fit in ~83 GB GPU allocation (0.65 x 128 GB) with room for KV cache
2. Strong coding and tool-use capabilities (needed for OpenCode agent orchestration)
3. Good instruction following for planning and delegation tasks

**Why Qwen3-Coder-30B-A3B-Instruct:**
- **MoE architecture**: 30.5B total parameters, but only 3.3B active per token — fast inference
- **Model weight size**: ~60 GB in BF16 (all experts loaded, but only 8/128 active per forward pass)
- **Native context**: 262,144 tokens (256K)
- **Strong coding benchmarks**: Purpose-built for code generation and tool use
- **Fits budget**: ~60 GB weights + ~23 GB KV cache headroom within 83 GB allocation
- **Shared instance**: Same model serves both orchestrator (planning) and coder (generation) roles, saving memory vs. two separate instances

**Alternatives considered:**
- `meta-llama/Llama-3.3-70B-Instruct`: ~140 GB in BF16, won't fit in 0.65 allocation. Still available as GGUF fallback via llama.cpp.
- Smaller dense models (7B-14B): Would fit easily but lack the reasoning depth needed for orchestration.

### Why Qwen2.5-7B-Instruct (fast utility)

- **Dense 7B model**: ~14 GB in BF16, fits comfortably in 0.20 allocation (~25.6 GB)
- **Fast inference**: Small model = quick responses for utility tasks (file summaries, simple transformations)
- **8K generation cap**: Qwen2.5-7B maxes out at 8K output tokens, so large context is unnecessary

---

## 3. Architecture: 3-Service to 2-Service Consolidation

### Before (3 services)

| Service | Port | Model | GPU Util |
|---------|------|-------|----------|
| vllm_orchestrator | 8001 | openai/gpt-oss-120b | 0.45 |
| vllm_coder | 8002 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.35 |
| vllm_fast | 8004 | Qwen/Qwen2.5-7B-Instruct | 0.20 |

**Problems:**
- gpt-oss-120b too large to load
- Three services = three model weight copies in GPU memory
- 0.45 + 0.35 + 0.20 = 1.00, no headroom for system

### After (2 services)

| Service | Port | Model | GPU Util |
|---------|------|-------|----------|
| vllm_orchestrator | 8001 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.65 |
| vllm_fast | 8004 | Qwen/Qwen2.5-7B-Instruct | 0.20 |

**Benefits:**
- One model weight copy serves both orch and coder roles (same model, same port)
- 0.65 + 0.20 = 0.85, leaving 15% (~19 GB) for system
- More KV cache per service (orch gets 83 GB instead of 57 GB)
- Simpler compose configuration and health checks
- `opencode.jsonc` maps both `orch` and `coder` model aliases to the same vLLM endpoint

**Port 8002 is no longer used.** The `local_coder` provider was removed from opencode.jsonc.

---

## 4. Memory Budget

### Allocation Table (128 GB shared)

| Component | Fraction | Memory | Purpose |
|-----------|----------|--------|---------|
| vllm_orchestrator | 0.65 | ~83 GB | Model weights (~60 GB) + KV cache (~23 GB) |
| vllm_fast | 0.20 | ~25.6 GB | Model weights (~14 GB) + KV cache (~11.6 GB) |
| System headroom | 0.15 | ~19 GB | OS, desktop, browser, swap management |

### How vLLM Allocates Memory (from source code analysis)

Based on reading vLLM 0.16 source code inside the Docker container:

1. **Memory request** (`vllm/v1/worker/utils.py`):
   ```python
   requested_memory = math.ceil(total_memory * gpu_memory_utilization)
   ```
   With 0.65 util on 128 GB: `ceil(128 x 0.65)` = 84 GB requested.

2. **Memory profiling** (`vllm/v1/worker/gpu_worker.py`):
   - vLLM runs a dummy forward pass to measure peak memory usage
   - `non_kv_cache_memory = model_weights + peak_activations + non_torch_overhead + 150 MB buffer`
   - `available_kv_cache_memory = requested_memory - non_kv_cache_memory`

3. **KV cache pre-allocation**:
   - KV cache blocks are allocated **at startup** to fill all `available_kv_cache_memory`
   - Blocks are assigned to requests **on demand** via PagedAttention
   - Unused blocks don't waste bandwidth, but the memory IS reserved

4. **Multiple instances explicitly supported** (from docstring):
   > "It does not matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU memory utilization to 0.5 for each instance."

5. **Auto-fit behavior**: If `--max-model-len` requires more KV cache than available, vLLM can auto-reduce it and log a warning.

---

## 5. Context Window Sizing

### KV Cache Math

Formula: `bytes_per_token = 2 x num_layers x num_kv_heads x head_dim x dtype_bytes`

| Model | Layers | KV Heads | Head Dim | dtype | Bytes/Token | Per Token |
|-------|--------|----------|----------|-------|-------------|-----------|
| Qwen3-Coder-30B-A3B | 48 | 4 | 128 | BF16 (2B) | 98,304 | ~96 KB |
| Qwen3-Coder-30B-A3B | 48 | 4 | 128 | FP8 (1B) | 49,152 | ~48 KB |
| Qwen2.5-7B | 28 | 4 | 128 | BF16 (2B) | 57,344 | ~56 KB |
| Qwen2.5-7B | 28 | 4 | 128 | FP8 (1B) | 28,672 | ~28 KB |

### Chosen Values

| Service | max-model-len | KV per request (BF16) | KV per request (FP8) | Budget |
|---------|---------------|----------------------|---------------------|--------|
| Orchestrator | 32,768 (32K) | ~3.0 GB | ~1.5 GB | ~23 GB available |
| Fast | 8,192 (8K) | ~0.4 GB | ~0.2 GB | ~11.6 GB available |

**Rationale for 32K orchestrator:**
- Typical coding agent conversations are 10K-25K tokens (system prompt + code context + conversation)
- 32K gives comfortable headroom without over-allocating
- Native model limit is 256K, but that would consume 24 GB per request in BF16 — the entire KV budget for a single sequence
- With FP8 KV cache enabled, 32K costs only 1.5 GB per request, leaving room for 4+ concurrent sequences in the KV pool

**Rationale for 8K fast:**
- Utility tasks are short (summaries, simple transforms, quick lookups)
- Qwen2.5-7B has an 8K output generation limit anyway
- 8K context at FP8 = 0.2 GB per request, trivially small

### Why Not Larger?

`--max-model-len` doesn't directly control total memory allocation — that's governed by `gpu-memory-utilization`. However, it sets the **maximum possible context per request**. If set higher than the KV pool can support (for even one request), vLLM will auto-reduce it or fail.

The real benefit of smaller max-model-len: prevents a single runaway request from consuming all KV cache blocks, leaving nothing for other concurrent sequences.

---

## 6. KV Cache Optimization: FP8

### What It Does

`--kv-cache-dtype fp8` stores key-value cache entries in FP8 (specifically fp8_e4m3 on ROCm) instead of BF16. This halves the memory per cached token.

### ROCm Support Confirmed

From vLLM 0.16 source (`vllm/config/cache.py`):
```python
CacheDType = Literal["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla"]
# Comment: "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)"
```

### Impact

| Metric | BF16 KV | FP8 KV | Improvement |
|--------|---------|--------|-------------|
| Orch KV per token | 96 KB | 48 KB | 2x |
| Orch 32K request | 3.0 GB | 1.5 GB | 2x |
| Fast KV per token | 56 KB | 28 KB | 2x |
| Max concurrent 32K seqs in orch | ~7 | ~15 | 2x |

### Quality Impact

FP8 KV cache has negligible quality impact according to community benchmarks. The key and value tensors are less sensitive to precision than model weights because:
- Attention scores are computed in higher precision
- Softmax normalizes the result
- The slight quantization noise is absorbed by the attention mechanism

### Configuration

```env
ORCH_KV_CACHE_DTYPE=fp8
FAST_KV_CACHE_DTYPE=fp8
```

Added to `compose/vllm.yml` as `--kv-cache-dtype ${*_KV_CACHE_DTYPE:-auto}` (falls back to `auto` if env var unset).

---

## 7. Quantization Research

### vLLM on AMD ROCm: Supported Methods

Extensive research into vLLM's quantization support on AMD GPUs revealed:

| Method | AMD/ROCm Support in vLLM | Notes |
|--------|-------------------------|-------|
| **FP8 (W8A8)** | YES | The only well-supported quantization on AMD. Uses `--quantization ptpc_fp8` or pre-quantized checkpoints. |
| AWQ | NO | Requires Marlin kernels (CUDA-only) |
| GPTQ | NO | Requires Marlin/ExLlama kernels (CUDA-only) |
| Marlin | NO | CUDA-only kernel library |
| INT8 (W8A8) | NO | Requires cutlass kernels (CUDA-only) |
| bitsandbytes | NO | CUDA-only |
| GGUF in vLLM | NO | vLLM's GGUF support is CUDA-only |

### Why Quantization Helps on Strix Halo

Token generation on this platform is **memory-bandwidth-bound** (not compute-bound):
- Memory bandwidth: ~212 GB/s
- FP16 compute: ~59.4 TFLOPS
- For a 60 GB model in BF16, reading all weights once takes ~283 ms
- At 60 GB model size, maximum theoretical throughput: ~3.5 tokens/sec per sequence

Quantized models are smaller = less data to read per token = proportionally faster generation.

| Model | BF16 Size | FP8 Size | Tokens/sec (theoretical max) |
|-------|-----------|----------|------------------------------|
| Qwen3-Coder-30B-A3B | ~60 GB | ~30 GB | ~3.5 -> ~7 tok/s |
| Qwen2.5-7B | ~14 GB | ~7 GB | ~15 -> ~30 tok/s |

*Note: MoE models like Qwen3-Coder only read active expert weights per token (8/128 experts), so actual bandwidth requirement is lower than total model size suggests. Real-world throughput will be higher than these worst-case estimates.*

### FP8 Weight Quantization Status

`--quantization ptpc_fp8` is available but untested on gfx1151. This would quantize model weights from BF16 to FP8, roughly halving model weight memory. Combined with FP8 KV cache, this could significantly free up memory for larger context windows or additional services.

**Risk**: gfx1151 (RDNA 3.5) is a very new target. FP8 weight quantization kernels may not be fully optimized or may have correctness issues. Test before relying on it.

---

## 8. Additional vLLM Optimization Flags

### --enable-prefix-caching (default: true in v0.16)

Prefix caching reuses KV cache blocks across requests that share common prefixes (e.g., system prompts). Since all OpenCode agents share the same system prompt per session, this is highly beneficial.

Made explicit in compose/vllm.yml even though it's the default, for documentation clarity.

### --max-num-seqs 4

Limits the maximum number of concurrent sequences the engine will process. Default is 256, which is designed for multi-user serving.

For single-user OpenCode usage:
- Rarely more than 2-3 concurrent requests (main agent + subagent)
- Lower value = less pre-allocated activation memory overhead
- More KV cache blocks available per sequence

### --enforce-eager (NOT enabled)

Disables HIP graph capture. Would save some memory by not caching compiled graph representations, but at the cost of throughput. Not enabled by default — consider if memory is extremely tight.

### --swap-space (default: 4 GiB)

vLLM's own KV cache swap mechanism (separate from OS swap). When KV cache is full, vLLM can evict blocks to CPU memory. On Strix Halo, CPU and GPU share the same physical RAM, so this is essentially reshuffling within the same memory pool — but it lets vLLM's memory manager stay within its allocated budget while handling temporary spikes.

Left at default. Increasing may help if many long-context requests queue up.

### --cpu-offload-gb (NOT used)

Can offload model weights to CPU memory to free GPU memory for KV cache. On shared-memory systems, this moves data within the same physical RAM but adds copy overhead through the GPU's memory controller. Not recommended for Strix Halo — the indirection hurts without gaining real memory.

---

## 9. System Stability Measures

### Problem

With 128 GB shared memory and two vLLM instances claiming 85%, the system can easily run out of available memory, causing:
- Linux OOM killer terminating random processes
- System freezing due to excessive swap thrashing
- Desktop environment becoming unresponsive

### Solutions

**1. Enlarge swap file to 128 GB**

Gives the OS room to page out less-critical data when GPU memory pressure is high:
```bash
sudo swapoff /swap.img
sudo rm /swap.img
sudo fallocate -l 128G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
```

**2. Install earlyoom**

Proactive OOM killer that terminates memory-hungry processes BEFORE the kernel OOM killer activates (which often freezes the system first):
```bash
sudo apt install earlyoom
sudo systemctl enable --now earlyoom
```

earlyoom monitors available memory and swap, sending SIGTERM (then SIGKILL) to the largest process when thresholds are crossed. Much more responsive than the kernel OOM killer.

**3. Increase swappiness to 80**

Default swappiness (60) is conservative. On a system where GPU allocations dominate, we want the OS to aggressively swap out idle pages to free up physical RAM:
```bash
sudo sysctl vm.swappiness=80
echo 'vm.swappiness=80' | sudo tee /etc/sysctl.d/99-swappiness.conf
```

---

## 10. oh-my-opencode Agent Token Limits

### Problem

OpenCode's built-in maxTokens defaults (e.g., 64K for Sisyphus) exceed the local model's 32K context window. If the agent tries to generate 64K output tokens, it will exceed the vLLM `--max-model-len` limit and fail.

### Solution

A template `.opencode/oh-my-opencode.json` is included in this repo with recommended maxTokens per agent:

| Agent | maxTokens | Rationale |
|-------|-----------|-----------|
| sisyphus, hephaestus, sisyphus-junior, oracle, prometheus | 16,384 | Orch model has 32K context; 16K output leaves ~16K for input |
| metis, momus | 8,192 | Moderate output for analysis/review tasks |
| librarian, explore, atlas | 4,096 | Utility agents on fast model (8K context); 4K output leaves ~4K for input |

**Important**: This file must be copied to the `.opencode/` directory of each target project where you run OpenCode. It is NOT automatically picked up from strix-opencode itself.

If you switch to cloud models (where context windows are much larger), remove or raise these limits.

---

## 11. Future Optimization Strategies

### Near-term (test and validate)

1. **FP8 weight quantization** (`--quantization ptpc_fp8`): Would halve model weight memory. Combined with FP8 KV cache, the orchestrator would use ~30 GB for weights + ~1.5 GB per 32K request = significant headroom. Needs testing on gfx1151.

2. **Increase context to 65K or 128K**: If FP8 KV works well and FP8 weights free up memory, the orchestrator could support much larger context windows. 65K at FP8 KV = 3 GB per request; 128K = 6 GB per request.

3. **Monitor actual utilization**: vLLM exposes Prometheus metrics. Monitor KV cache utilization at `/metrics` to see if the current 32K limit is actually being hit, and whether more headroom exists.

### Medium-term (architecture changes)

4. **Speculative decoding**: Use the fast 7B model as a draft model for the 30B orchestrator. vLLM supports this natively. Could significantly improve generation speed at no quality cost.

5. **llama.cpp for orchestrator**: If vLLM's FP8 weight quantization doesn't work well on gfx1151, llama.cpp with Q4_K_M GGUF may be a better option for the orchestrator. The GGUF is already configured as a fallback.

### Long-term (hardware/software evolution)

6. **ROCm kernel improvements**: gfx1151 is very new. As AMD improves ROCm support, expect better FP8 kernels, flash attention optimizations, and potentially INT4/INT8 quantization support.

7. **Dedicated VRAM systems**: For serious multi-agent workloads, a discrete GPU (e.g., MI300X, or consumer RDNA cards with large VRAM) would eliminate the shared-memory bandwidth contention.

---

## Appendix: Key vLLM Configuration Defaults (v0.16)

From reading `vllm/config/cache.py` in the container:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_memory_utilization` | 0.9 | Fraction of GPU memory to request |
| `swap_space` | 4 GiB | CPU swap space for KV cache eviction |
| `cache_dtype` | "auto" | KV cache precision (follows model dtype) |
| `enable_prefix_caching` | true | Reuse KV blocks for common prefixes |
| `cpu_offload_gb` | 0 | Model weight offload to CPU memory |
| `block_size` | platform-dependent | PagedAttention block size |

From reading `vllm/v1/engine/core.py`:
- Auto-fit: vLLM will auto-reduce `max_model_len` if KV cache budget is insufficient
- KV blocks are pre-allocated at startup, assigned to requests on demand
- Prefix caching is enabled by default — beneficial for agent workloads with repeated system prompts
