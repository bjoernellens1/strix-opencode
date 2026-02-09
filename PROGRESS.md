# PROGRESS.md — 4-Tier Architecture Migration

**Status**: IN PROGRESS  
**Date**: 2026-02-09  
**Branch**: `main`

---

## What Changed (This Commit)

We are migrating from a **2-service layout** (shared orch+coder on one vLLM instance + fast utility) to a **4-tier architecture**. This commit captures the partially-complete migration.

### Completed

- **`.env`** — Fully rewritten for 4-tier architecture:
  - Tier 1 (GPU): Orchestrator → `Qwen/Qwen2.5-14B-Instruct` on port 8001, GPU util 0.35, 64K context
  - Tier 2 (GPU): Coder → `Qwen/Qwen3-Coder-30B-A3B-Instruct` on port 8002, GPU util 0.60, 48K context
  - Tier 3 (CPU): Reviewer → `llama-3.3-70b-instruct.Q4_K_M.gguf` on port 8011, 32K context
  - Tier 0 (CPU): Utility → `qwen2.5-3b-instruct.Q4_K_M.gguf` on port 8012, 8K context
  - All FP8 references removed — KV cache set to `auto` (BF16)
  - All quantization comments updated to reflect gfx1151 limitations

### Not Yet Updated (Still Has OLD Config)

These files still reference the **old 2-service layout** (FAST_MODEL, FAST_PORT, etc.) and need updating to match the new `.env`:

| File | What Needs Changing |
|------|-------------------|
| `.env.example` | Must mirror new `.env` (minus HF_TOKEN) |
| `compose/vllm.yml` | Rename `vllm_fast` → `vllm_coder`, swap models/ports/GPU util, update env var names |
| `compose/cpu.yml` | **NEW FILE NEEDED** — llama.cpp services for Tier 3 (reviewer) and Tier 0 (utility) |
| `opencode/opencode.jsonc` | New providers: `local_coder` (:8002), `local_reviewer` (:8011), `local_utility` (:8012). 4 agents instead of 3 |
| `.opencode/oh-my-opencode.json` | New maxTokens: orch→32K, coder→24K, reviewer→16K, utility→4K |
| `scripts/health` | Add ports 8002, 8012; rename labels to match new tiers |
| `scripts/up` | Add `cpu` and `full` modes for llama.cpp tiers |
| `scripts/down` | Add teardown for CPU compose file |
| `scripts/switch-orch` | May need rework for new provider names |
| `DECISIONS.md` | Add sections: quantization failure details, 4-tier rationale, memory recalc |
| `README.md` | Full rewrite for 4-tier architecture, new ports, new roles |

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │            OpenCode (TUI / Agent)            │
                    ├──────┬──────────┬──────────────┬────────────┤
                    │      │          │              │            │
                    │ Tier 1        Tier 2        Tier 3      Tier 0
                    │ Orchestrator  Coder         Reviewer    Utility
                    │ (GPU vLLM)   (GPU vLLM)    (CPU llama) (CPU llama)
                    │      │          │              │            │
                    │  :8001       :8002          :8011       :8012
                    │  14B BF16    30B-MoE BF16   70B Q4_K_M  3B Q4_K_M
                    │  64K ctx     48K ctx        32K ctx     8K ctx
                    │  0.35 GPU    0.60 GPU       CPU-only    CPU-only
                    └──────┴──────────┴──────────────┴────────────┘
```

## Memory Budget (128 GB Shared UMA)

| Component | Allocation | Memory | Notes |
|-----------|-----------|--------|-------|
| Tier 2: Coder (vLLM) | 0.60 | ~76.8 GB | Weights ~57 GB + KV ~4.5 GB (48K) |
| Tier 1: Orch (vLLM) | 0.35 | ~44.8 GB | Weights ~28 GB + KV ~10 GB (64K) |
| System headroom | 0.05 | ~6.4 GB | OS, desktop. Tight — swap/earlyoom recommended |
| Tier 3+0 (CPU) | shared RAM | ~42 GB peak | 70B Q4 ≈ 40 GB + 3B Q4 ≈ 2 GB (loads on demand, not concurrent with full GPU) |

**Total GPU util: 0.95** — single-user only. CPU tiers load into the same physical RAM but run when GPU tiers aren't at peak.

## Hardware Constraints (gfx1151 / RDNA 3.5)

**No quantization works in vLLM on this hardware:**

| Method | Result | Error |
|--------|--------|-------|
| FP8 weights (`--quantization fp8`) | ❌ CRASH | `torch._scaled_mm requires MI300+ or CUDA sm_89+` |
| AWQ INT4 (pre-quantized checkpoint) | ❌ GPU HANG | `Memory access fault by GPU node-1` — Triton AWQ kernels incompatible |
| FP8 KV cache (`--kv-cache-dtype fp8`) | ⚠️ BAD | Software emulation, uncalibrated scales → accuracy loss |
| GPTQ/Marlin/bitsandbytes | ❌ | CUDA-only in vLLM |

**Conclusion: ALL vLLM models must use BF16 weights + BF16 KV cache (`--kv-cache-dtype auto`).**

---

## How to Continue

Pick up from the "Not Yet Updated" table above. The `.env` is the source of truth — all other files need to be aligned to it.

Priority order:
1. `compose/vllm.yml` + new `compose/cpu.yml`
2. `opencode/opencode.jsonc`
3. `.env.example`
4. Scripts (`health`, `up`, `down`)
5. `.opencode/oh-my-opencode.json`
6. Documentation (`DECISIONS.md`, `README.md`)
