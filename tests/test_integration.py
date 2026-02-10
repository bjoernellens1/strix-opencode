"""Integration tests for running services."""

import subprocess
from pathlib import Path

import httpx
import pytest

# Repo root is two levels up from tests/
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
ENV_FILE = str(Path(REPO_ROOT) / ".env.example")


def is_port_open(port: int, timeout: float = 2.0) -> bool:
    """Check if an agent is responding on given port."""
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def docker_available() -> bool:
    """Check if docker CLI is available."""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
class TestComposeFiles:
    """Test compose file validity."""

    @pytest.mark.parametrize("compose_file", [
        "compose/hybrid.yml",
        "compose/gpu-all.yml",
        "compose/vllm.yml",
        "compose/cpu.yml",
    ])
    def test_compose_config_valid(self, compose_file):
        """Verify compose files parse without errors."""
        result = subprocess.run(
            [
                "docker", "compose",
                "--project-directory", REPO_ROOT,
                "--env-file", ENV_FILE,
                "-f", compose_file,
                "config", "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Invalid compose: {result.stderr}"


@pytest.mark.integration
class TestAgentHealth:
    """Test agent health (requires running services)."""

    def test_thor_health(self, agent_ports):
        """Thor should respond on port 8001."""
        if not is_port_open(agent_ports["thor"]):
            pytest.skip("Thor not running")
        assert is_port_open(agent_ports["thor"])

    def test_valkyrie_health(self, agent_ports):
        """Valkyrie should respond on port 8002."""
        if not is_port_open(agent_ports["valkyrie"]):
            pytest.skip("Valkyrie not running")
        assert is_port_open(agent_ports["valkyrie"])

    def test_bifrost_health(self):
        """Bifrost scheduler should respond on port 8899."""
        if not is_port_open(8899):
            pytest.skip("Bifrost not running")
        resp = httpx.get("http://127.0.0.1:8899/v1/health", timeout=2.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
