"""
Unit tests for the core configuration loader.
"""

import pytest

from core.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH


def test_config_loader_loads_default_config():
    """The default config file exists and loads into a dict."""
    cfg = ConfigLoader.get()
    assert isinstance(cfg._config, dict)
    assert "lmstudio" in cfg._config


def test_get_value_nested():
    """get_value traverses nested keys and returns defaults safely."""
    cfg = ConfigLoader.get()
    model = cfg.get_value("lmstudio", "model")
    assert isinstance(model, str) and model

    assert cfg.get_value("lmstudio", "does_not_exist", default=42) == 42
    assert cfg.get_value("missing_top", "nested", default="x") == "x"


def test_technique_config_accessor():
    """get_technique_config returns the per-technique section."""
    cfg = ConfigLoader.get()
    tech = cfg.get_technique_config("naive_rag")
    assert isinstance(tech, dict)


def test_config_path_missing_raises():
    """A non-existent config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.get("/no/such/config.yaml")