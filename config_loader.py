import os
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigLoader:
    """
    Central configuration loader for the entire project.
    Picks configuration based on ENV variable (dev/preprod/prod).
    """

    VALID_ENVS = ("dev", "preprod", "prod")

    def __init__(self, env: Literal["dev", "preprod", "prod"] | None = None):
        # Determine environment
        self.env = env or os.getenv("ENV", "dev")

        if self.env not in self.VALID_ENVS:
            raise ValueError(
                f"Invalid ENV='{self.env}'. Must be one of {self.VALID_ENVS}"
            )

        # Resolve config path relative to project root
        self.project_root = Path(__file__).resolve().parent
        self.config_dir = self.project_root / "config"
        self.config_path = self.config_dir / f"{self.env}.yaml"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load snowflake config separately if needed
        self.snowflake_config_path = self.config_dir / "snowflake.yaml"
        if self.snowflake_config_path.exists():
            with open(self.snowflake_config_path, "r") as f:
                self.snowflake_config = yaml.safe_load(f)
        else:
            self.snowflake_config = None

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safely get nested config values.

        Example:
        config.get("paths", "data_raw")
        """
        d = self.config
        for key in keys:
            if not isinstance(d, dict) or key not in d:
                return default
            d = d[key]
        return d

    def get_snowflake(self, *keys: str, default: Any = None) -> Any:
        """
        Get values from snowflake.yaml
        """
        if not self.snowflake_config:
            raise ValueError("snowflake.yaml not loaded or missing")

        d = self.snowflake_config
        for key in keys:
            if not isinstance(d, dict) or key not in d:
                return default
            d = d[key]
        return d
