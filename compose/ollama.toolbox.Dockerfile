ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/sh", "-c"]

# Install Ollama if missing.
RUN set -eux; \
    if ! command -v ollama >/dev/null 2>&1; then \
      if ! command -v curl >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then \
        if command -v dnf >/dev/null 2>&1; then \
          dnf install -y curl ca-certificates zstd tar && dnf clean all; \
        elif command -v microdnf >/dev/null 2>&1; then \
          microdnf install -y curl ca-certificates zstd tar && microdnf clean all; \
        elif command -v apt-get >/dev/null 2>&1; then \
          apt-get update && apt-get install -y curl ca-certificates zstd tar && rm -rf /var/lib/apt/lists/*; \
        else \
          echo "No supported package manager found to install curl/zstd/tar." >&2; exit 1; \
        fi; \
      fi; \
      curl -fsSL https://ollama.com/install.sh | sh; \
    fi
