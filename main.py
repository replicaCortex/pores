#!/usr/Bbin/env python3
"""
Main executable file for generating a dataset of pore images.
"""

import os
from datetime import datetime
from typing import Any, Dict

from tqdm import tqdm

from src.config_loader import ConfigLoader
from src.image_processor import ImageProcessor
from src.pore_generator import PoreGenerator


def create_output_directories(config: Dict[str, Any]) -> None:
    """Creates the output directories based on the configuration."""
    os.makedirs(config["output_settings"]["clean_dir"], exist_ok=True)
    os.makedirs(config["output_settings"]["noisy_dir"], exist_ok=True)


def generate_filename(config: Dict[str, Any], index: int) -> str:
    """Generates a unique filename using a timestamp and an index."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = config["output_settings"]["file_prefix"]
    return f"{prefix}_{timestamp}_{index:06d}.png"


def main() -> None:
    """Main function to run the image generation process."""
    config_loader = ConfigLoader("config.json")
    config = config_loader.load()

    create_output_directories(config)

    pore_generator = PoreGenerator(config)
    image_processor = ImageProcessor(config)

    total_images = config["image_settings"]["total_images"]
    print(f"Starting generation of {total_images} images...")

    for i in tqdm(range(total_images), desc="Generating images"):
        clean_image, pore_info = pore_generator.generate_image()

        noisy_image = image_processor.add_background_noise(clean_image)

        filename = generate_filename(config, i)

        clean_image_path = os.path.join(
            config["output_settings"]["clean_dir"], filename
        )
        noisy_image_path = os.path.join(
            config["output_settings"]["noisy_dir"], filename
        )

        image_processor.save_image(clean_image, clean_image_path)
        image_processor.save_image(noisy_image, noisy_image_path)

    print(f"\nGeneration complete! {total_images} images created.")
    print(f"Clean images saved to: {config['output_settings']['clean_dir']}")
    print(f"Noisy images saved to: {config['output_settings']['noisy_dir']}")


if __name__ == "__main__":
    main()
