"""
Pytest configuration.
Adds the project root to sys.path and resets the ConfigLoader singleton
between tests so each test sees a fresh config.
"""

import pytest

from core.config_loader import ConfigLoader


@pytest.fixture(autouse=True)
def reset_config_loader():
    """Reset the ConfigLoader singleton before each test."""
    ConfigLoader._instance = None
    ConfigLoader._config = {}
    yield
    ConfigLoader._instance = None
    ConfigLoader._config = {}
