"""Unit tests for Bifrost scheduler."""

import pytest
from fastapi import HTTPException

from app import (
    BifrostManager,
    container_name,
    env_int,
    LLAMA_AGENTS,
    VLLM_AGENTS,
    OLLAMA_AGENTS,
    PROFILE_ON_DEMAND,
    MANAGED_AGENTS,
)


class TestHelpers:
    """Test helper functions."""

    def test_container_name_llama(self):
        for agent in LLAMA_AGENTS:
            assert container_name(agent) == f"llama_{agent}"

    def test_container_name_vllm(self):
        for agent in VLLM_AGENTS:
            assert container_name(agent) == f"vllm_{agent}"

    def test_container_name_ollama(self, monkeypatch):
        monkeypatch.setenv("BIFROST_BACKEND", "ollama")
        for agent in OLLAMA_AGENTS:
            assert container_name(agent) == f"ollama_{agent}"

    def test_env_int_default(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        assert env_int("TEST_VAR", 42) == 42

    def test_env_int_valid(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "100")
        assert env_int("TEST_VAR", 42) == 100

    def test_env_int_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "not_a_number")
        assert env_int("TEST_VAR", 42) == 42


class TestBifrostManager:
    """Test BifrostManager logic."""

    def test_init_defaults(self):
        mgr = BifrostManager()
        assert mgr.port == 8899
        assert mgr.session_budget_odin == 3
        assert mgr.state.active_profile == "standard"

    def test_running_agents_standard(self):
        mgr = BifrostManager()
        assert mgr.running_agents() == ["thor", "valkyrie"]

    def test_running_agents_odin(self):
        mgr = BifrostManager()
        mgr.state.active_profile = "odin"
        assert mgr.running_agents() == ["thor", "odin"]

    def test_check_policy_odin_budget(self):
        mgr = BifrostManager()
        mgr.state.summon_counts["odin"] = 3
        with pytest.raises(HTTPException) as exc:
            mgr.check_policy("odin")
        assert exc.value.status_code == 429

    def test_check_policy_utility_budget(self):
        mgr = BifrostManager()
        mgr.state.summon_counts["utility"] = 10
        with pytest.raises(HTTPException) as exc:
            mgr.check_policy("heimdall")
        assert exc.value.status_code == 429

    def test_gpu_profiles_defined(self):
        mgr = BifrostManager()
        for profile in ["standard", "odin", "heimdall", "loki", "frigga"]:
            assert profile in mgr.gpu_profile_utilization


class TestConstants:
    """Test module constants."""

    def test_profile_on_demand(self):
        assert PROFILE_ON_DEMAND == {"heimdall", "loki", "frigga", "odin"}

    def test_managed_agents(self):
        assert "valkyrie" in MANAGED_AGENTS
        assert "thor" not in MANAGED_AGENTS
