"""
Configuration Loader
====================
Loads and validates config.yaml. Provides typed access to all settings.
Singleton pattern ensures config is loaded once per session.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Default config path resolution (works from any working directory)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


class ConfigLoader:
    """Singleton config loader. Use ConfigLoader.get() to access settings."""

    _instance: Optional["ConfigLoader"] = None
    _config: dict[str, Any] = {}

    def __new__(cls, config_path: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            path = config_path or os.environ.get("RAG_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
            cls._instance._load(path)
        return cls._instance

    def _load(self, path: str) -> None:
        """Load YAML config from disk."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Expected at: {DEFAULT_CONFIG_PATH}"
            )
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    @classmethod
    def get(cls, config_path: str | None = None) -> "ConfigLoader":
        """Get or create the singleton instance."""
        return cls(config_path)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get_value(self, *keys: str, default: Any = None) -> Any:
        """Safely navigate nested keys. E.g., get_value('lmstudio', 'model')"""
        result = self._config
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key, default)
            else:
                return default
        return result

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def lmstudio(self) -> dict[str, Any]:
        return self._config.get("lmstudio", {})

    @property
    def lmstudio_small(self) -> dict[str, Any]:
        return self._config.get("lmstudio_small", {})

    @property
    def embeddings(self) -> dict[str, Any]:
        return self._config.get("embeddings", {})

    @property
    def vector_store(self) -> dict[str, Any]:
        return self._config.get("vector_store", {})

    @property
    def document(self) -> dict[str, Any]:
        return self._config.get("document", {})

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._config.get("retrieval", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        return self._config.get("evaluation", {})

    @property
    def rag_techniques(self) -> dict[str, Any]:
        return self._config.get("rag_techniques", {})

    def get_technique_config(self, technique_key: str) -> dict[str, Any]:
        """Get config for a specific RAG technique."""
        return self.rag_techniques.get(technique_key, {})

    def __repr__(self) -> str:
        return f"ConfigLoader(model={self.lmstudio.get('model')}, " \
               f"vector_store={self.vector_store.get('provider')})"
