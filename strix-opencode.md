Use this as instructions for your GitHub code agent to create the repo.

## Repo name (short + clear)

**`strix-opencode`**

---

## Goal

Create a reproducible Strix Halo dev stack repo for **Ubuntu + distrobox** that runs:

* **Default (simple):** all roles via **vLLM** (OpenAI-compatible API endpoints)
* **Optional (hybrid fallback):** orchestrator via **llama.cpp** (RADV or ROCm-rocwmma), while **coder + fast utility stay on vLLM**

Integrate with a **custom fork of `oh-my-opencode`** placed in the repo and referenced from OpenCode config.

Primary user = single user (Björn). Optimize for **interactive latency + stability**, not multi-user throughput.

Sources used to shape behavior:

* vLLM Strix Halo toolbox image + distrobox recommendation.
* llama.cpp toolbox tags for RADV/ROCm stacks.

---

## Repository structure

Create the following:

```
strix-opencode/
  README.md
  .gitignore
  .env.example
  compose/
    vllm.yml
    fallback.vulkan-radv.orch.yml
    fallback.rocm-6.4.4-rocwmma.orch.yml
  opencode/
    opencode.jsonc
    oh-my-opencode/         # git submodule or vendored fork
  scripts/
    up
    down
    health
    switch-orch
```

---

## Model roles & endpoints

### Default (vLLM for all)

* Orchestrator/Planner (primary): `http://127.0.0.1:8001/v1`
* Coder (subagent): `http://127.0.0.1:8002/v1`
* Fast utility (subagent): `http://127.0.0.1:8004/v1`

### Optional hybrid (llama.cpp only for orchestrator)

* Orchestrator via llama.cpp: `http://127.0.0.1:8011/v1` (use one of the fallback compose files)
* Coder & Fast remain vLLM as above

### Model choices (env-configurable)

* Orchestrator: `openai/gpt-oss-120b` OR `meta-llama/Llama-3.3-70B-Instruct`
* Coder: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
* Fast: `Qwen/Qwen2.5-7B-Instruct`

Also support **Option C hybrid cloud planner** by letting OpenCode switch primary model to a cloud provider while keeping local coder/fast.

---

## Files to implement

### 1) `.env.example`

Create `.env.example`:

```bash
# ===== Paths =====
HF_HOME=${HOME}/.cache/huggingface
VLLM_CACHE=${HOME}/.cache/vllm
LLAMA_MODELS_DIR=./models

# ===== Ports =====
ORCH_PORT=8001
CODER_PORT=8002
FAST_PORT=8004
LLAMA_ORCH_PORT=8011

# ===== vLLM models =====
ORCH_MODEL=openai/gpt-oss-120b
# ORCH_MODEL=meta-llama/Llama-3.3-70B-Instruct
CODER_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
FAST_MODEL=Qwen/Qwen2.5-7B-Instruct

# ===== vLLM tuning (single user; fractions must sum to ≤ 1.0 for shared GPU) =====
ORCH_GPU_UTIL=0.45
CODER_GPU_UTIL=0.35
FAST_GPU_UTIL=0.20
ORCH_MAX_LEN=8192
CODER_MAX_LEN=8192
FAST_MAX_LEN=4096

# ===== llama.cpp fallback orchestrator (GGUF file inside ./models) =====
ORCH_GGUF=gpt-oss-120b.Q4_K_M.gguf
# ORCH_GGUF=llama-3.3-70b-instruct.Q4_K_M.gguf

# ===== Cloud planner (optional) =====
# OPENAI_API_KEY=...
# CLOUD_PLANNER_MODEL=gpt-5
```

Add `models/` to `.gitignore`.

---

### 2) `compose/vllm.yml` (mainline: vLLM servers)

Use the Strix Halo vLLM toolbox image and the device exposure recommended there (kfd+dri, seccomp unconfined, group_add video/render).

```yaml
services:
  vllm_orchestrator:
    image: docker.io/kyuz0/vllm-therock-gfx1151:latest
    container_name: vllm_orchestrator
    network_mode: host
    devices: ["/dev/kfd", "/dev/dri"]
    security_opt: ["seccomp=unconfined"]
    group_add: ["video", "render"]
    environment:
      - HF_HOME=${HF_HOME}
    volumes:
      - ${HF_HOME}:${HF_HOME}
      - ${VLLM_CACHE}:${VLLM_CACHE}
    command: >
      vllm serve ${ORCH_MODEL}
      --host 0.0.0.0 --port ${ORCH_PORT}
      --dtype auto
      --gpu-memory-utilization ${ORCH_GPU_UTIL}
      --max-model-len ${ORCH_MAX_LEN}

  vllm_coder:
    image: docker.io/kyuz0/vllm-therock-gfx1151:latest
    container_name: vllm_coder
    network_mode: host
    devices: ["/dev/kfd", "/dev/dri"]
    security_opt: ["seccomp=unconfined"]
    group_add: ["video", "render"]
    environment:
      - HF_HOME=${HF_HOME}
    volumes:
      - ${HF_HOME}:${HF_HOME}
      - ${VLLM_CACHE}:${VLLM_CACHE}
    command: >
      vllm serve ${CODER_MODEL}
      --host 0.0.0.0 --port ${CODER_PORT}
      --dtype auto
      --gpu-memory-utilization ${CODER_GPU_UTIL}
      --max-model-len ${CODER_MAX_LEN}

  vllm_fast:
    image: docker.io/kyuz0/vllm-therock-gfx1151:latest
    container_name: vllm_fast
    network_mode: host
    devices: ["/dev/kfd", "/dev/dri"]
    security_opt: ["seccomp=unconfined"]
    group_add: ["video", "render"]
    environment:
      - HF_HOME=${HF_HOME}
    volumes:
      - ${HF_HOME}:${HF_HOME}
      - ${VLLM_CACHE}:${VLLM_CACHE}
    command: >
      vllm serve ${FAST_MODEL}
      --host 0.0.0.0 --port ${FAST_PORT}
      --dtype auto
      --gpu-memory-utilization ${FAST_GPU_UTIL}
      --max-model-len ${FAST_MAX_LEN}
```

---

### 3) `compose/fallback.vulkan-radv.orch.yml` (llama.cpp orchestrator fallback)

This uses the llama.cpp toolbox image tag `vulkan-radv`.

```yaml
services:
  llama_orchestrator:
    image: docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv
    container_name: llama_orchestrator_vulkan
    network_mode: host
    devices: ["/dev/dri"]
    security_opt: ["seccomp=unconfined"]
    group_add: ["video"]
    volumes:
      - ${LLAMA_MODELS_DIR}:/models
    command: >
      bash -lc "
      llama-server
      --host 0.0.0.0 --port ${LLAMA_ORCH_PORT}
      -m /models/${ORCH_GGUF}
      --no-mmap -ngl 999 --flash-attn on
      "
```

---

### 4) `compose/fallback.rocm-6.4.4-rocwmma.orch.yml` (llama.cpp orchestrator perf fallback)

Uses tag `rocm-6.4.4-rocwmma`.

```yaml
services:
  llama_orchestrator:
    image: docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-6.4.4-rocwmma
    container_name: llama_orchestrator_rocm
    network_mode: host
    devices: ["/dev/dri", "/dev/kfd"]
    security_opt: ["seccomp=unconfined"]
    group_add: ["video", "render"]
    volumes:
      - ${LLAMA_MODELS_DIR}:/models
    command: >
      bash -lc "
      llama-server
      --host 0.0.0.0 --port ${LLAMA_ORCH_PORT}
      -m /models/${ORCH_GGUF}
      --no-mmap -ngl 999 --flash-attn on
      "
```

---

### 5) `opencode/opencode.jsonc`

Use OpenCode config schema, enable plugin as local path, define providers for local vLLM, local llama.cpp (optional), and cloud planner (optional).

```jsonc
{
  "$schema": "https://opencode.ai/config.json",

  "plugin": [
    "./oh-my-opencode"
  ],

  "provider": {
    "local_orch_vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strix vLLM Orchestrator",
      "options": { "baseURL": "http://127.0.0.1:8001/v1", "apiKey": "EMPTY" },
      "models": { "orch": { "name": "openai/gpt-oss-120b", "tools": true } }
    },
    "local_orch_llama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strix llama.cpp Orchestrator",
      "options": { "baseURL": "http://127.0.0.1:8011/v1", "apiKey": "EMPTY" },
      "models": { "orch": { "name": "orchestrator-gguf", "tools": true } }
    },
    "local_coder": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strix vLLM Coder",
      "options": { "baseURL": "http://127.0.0.1:8002/v1", "apiKey": "EMPTY" },
      "models": { "coder": { "name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "tools": true } }
    },
    "local_fast": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strix vLLM Fast",
      "options": { "baseURL": "http://127.0.0.1:8004/v1", "apiKey": "EMPTY" },
      "models": { "fast": { "name": "Qwen/Qwen2.5-7B-Instruct", "tools": true } }
    },
    "cloud_planner": {
      "npm": "@ai-sdk/openai",
      "name": "Cloud Planner",
      "options": { "apiKey": "${OPENAI_API_KEY}" },
      "models": { "planner": { "name": "${CLOUD_PLANNER_MODEL}", "tools": true } }
    }
  },

  "agent": {
    "primary": { "mode": "primary", "model": "local_orch_vllm:orch" },
    "coder":   { "mode": "subagent", "model": "local_coder:coder" },
    "utility": { "mode": "subagent", "model": "local_fast:fast" }
  }
}
```

**Do NOT** try to interpolate env vars inside JSON unless OpenCode supports it. Instead, implement switching by editing the file via script (below).

---

### 6) Scripts

All scripts should be executable (`chmod +x scripts/*`).

#### `scripts/up`

```bash
#!/usr/bin/env bash
set -euo pipefail
cp -n .env.example .env 2>/dev/null || true

MODE="${1:-vllm}"

case "$MODE" in
  vllm)
    docker compose -f compose/vllm.yml --env-file .env up -d
    ;;
  hybrid-radv)
    docker compose -f compose/vllm.yml -f compose/fallback.vulkan-radv.orch.yml --env-file .env up -d
    ;;
  hybrid-rocm)
    docker compose -f compose/vllm.yml -f compose/fallback.rocm-6.4.4-rocwmma.orch.yml --env-file .env up -d
    ;;
  *)
    echo "Usage: $0 {vllm|hybrid-radv|hybrid-rocm}"
    exit 1
    ;;
esac
```

#### `scripts/down`

```bash
#!/usr/bin/env bash
set -euo pipefail
docker compose -f compose/vllm.yml --env-file .env down || true
docker compose -f compose/fallback.vulkan-radv.orch.yml --env-file .env down || true
docker compose -f compose/fallback.rocm-6.4.4-rocwmma.orch.yml --env-file .env down || true
```

#### `scripts/health`

```bash
#!/usr/bin/env bash
set -uo pipefail

# Source .env for port overrides; fall back to defaults if missing
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

function check() {
  local port="$1"
  local label="$2"
  echo "== ${label} :${port} =="
  if response=$(curl -sf --connect-timeout 3 --max-time 5 "http://127.0.0.1:${port}/v1/models" 2>&1); then
    echo "$response" | head -c 400
  else
    echo "(not responding)"
  fi
  echo ""
}

check "${ORCH_PORT:-8001}" "vLLM Orchestrator"
check "${CODER_PORT:-8002}" "vLLM Coder"
check "${FAST_PORT:-8004}" "vLLM Fast"
check "${LLAMA_ORCH_PORT:-8011}" "llama.cpp Orchestrator"
```

#### `scripts/switch-orch`

Switch OpenCode primary agent between:

* local vLLM orchestrator
* local llama.cpp orchestrator
* cloud planner

This script edits `opencode/opencode.jsonc` in-place by replacing the `"primary"` agent line.

```bash
#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-vllm}" # vllm|llama|cloud

FILE="opencode/opencode.jsonc"

case "$TARGET" in
  vllm)
    sed -i 's/"primary":[^}]*}/"primary": { "mode": "primary", "model": "local_orch_vllm:orch" }/g' "$FILE"
    ;;
  llama)
    sed -i 's/"primary":[^}]*}/"primary": { "mode": "primary", "model": "local_orch_llama:orch" }/g' "$FILE"
    ;;
  cloud)
    sed -i 's/"primary":[^}]*}/"primary": { "mode": "primary", "model": "cloud_planner:planner" }/g' "$FILE"
    ;;
  *)
    echo "Usage: $0 {vllm|llama|cloud}"
    exit 1
    ;;
esac

echo "Switched primary agent to: $TARGET"
```

(Agent should ensure this `sed` pattern doesn’t break JSON; keep formatting stable.)

---

## Integrating your custom `oh-my-opencode`

Preferred approach: **git submodule** so you can pin your fork.

Agent steps:

1. If a submodule or directory already exists at the path, remove it from Git tracking first:

   * `git rm opencode/oh-my-opencode`
   * `rm -rf .git/modules/opencode/oh-my-opencode`

2. Add submodule:

   * `git submodule add https://github.com/<YOUR_ORG_OR_USER>/oh-my-opencode opencode/oh-my-opencode`
3. Ensure OpenCode plugin path points to `./oh-my-opencode` (already done in opencode.jsonc).
4. Add documentation in README on how to update submodule.

---

## README requirements

Write a practical README that includes:

### Quick start (vLLM only)

```bash
cp .env.example .env
./scripts/up vllm
./scripts/health
# run opencode pointing to ./opencode/opencode.jsonc
```

### Hybrid orchestrator fallback

```bash
./scripts/up hybrid-radv
./scripts/switch-orch llama
```

### Switch to cloud planner (Option C)

```bash
export OPENAI_API_KEY=...
export CLOUD_PLANNER_MODEL=gpt-5
./scripts/switch-orch cloud
```

### Distrobox notes (Ubuntu)

Explain that Ubuntu + distrobox is recommended for GPU access in this toolbox ecosystem.
Include a short example `distrobox create` command (don’t force it; host compose is okay).

### Model download notes

* vLLM pulls HF models automatically via cache.
* llama.cpp requires GGUF files placed into `./models`.

---

## Git hygiene

* `.gitignore`: ignore `models/`, `.env`, caches
* Add `LICENSE` only if you want (or omit)
* No secrets in repo.

---

## Deliverables checklist (for the agent)

* [ ] Repo created with structure above
* [ ] All compose files present and valid YAML
* [ ] Scripts executable and working
* [ ] OpenCode config references local plugin path
* [ ] Submodule instructions in README
* [ ] README covers vLLM-only + hybrid + cloud planner switch
* [ ] `.env.example` complete and conservative defaults

---

If you want one extra nice touch: add `make` targets (`make up`, `make hybrid-radv`, `make switch-cloud`) — but scripts are enough for now.
