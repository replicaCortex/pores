"""
Module for loading and validating configuration.
"""

import json
import os
from typing import Any, Dict


class ConfigLoader:
    """Loads and validates configuration from a JSON file."""

    def __init__(self, config_path: str):
        self.config_path = config_path

    def load(self) -> Dict[str, Any]:
        """Loads the configuration from the JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self._validate_config(config)
        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates the loaded configuration dictionary."""
        required_sections = [
            "image_settings",
            "pore_settings",
            "noise_settings",
            "output_settings",
        ]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        image_settings = config["image_settings"]
        if image_settings["width"] <= 0 or image_settings["height"] <= 0:
            raise ValueError("Image dimensions (width, height) must be positive.")

        if image_settings["total_images"] <= 0:
            raise ValueError("Total images count must be positive.")
