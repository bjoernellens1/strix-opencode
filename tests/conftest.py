"""Pytest configuration and fixtures."""

import os
import sys

import pytest

# Add bifrost to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bifrost"))


@pytest.fixture
def agent_ports():
    """Default agent ports from .env.example."""
    return {
        "thor": 8001,
        "valkyrie": 8002,
        "odin": 8011,
        "heimdall": 8012,
        "loki": 8013,
        "frigga": 8014,
    }
