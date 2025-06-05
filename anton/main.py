import logging
import pandas as pd
from pathlib import Path

from anton.core.config import Config
from anton.core.pipeline import AnalysisPipeline

logging.basicConfig(level=logging.INFO)

def main():
    """Interactive main function for Anton CMPO phenotype analysis framework."""
    print("Welcome to Anton: VLM-driven microscopy phenotype analysis framework.")
    print("Please provide the following information:")

    goal = input("Enter your analysis goal (e.g., 'Identify apoptotic cells in DAPI-stained channel 1'): ")
    image_path = input("Enter the path to your TIFF image: ")
    metadata_path = input("Enter the path to your metadata file (optional, press Enter to skip): ")
    config_path = input("Enter the path to your config file (optional, press Enter to skip): ")

    # Load configuration
    config = Config(config_path if config_path else None)
    config.set("goal", goal)
    config.set("image_path", str(image_path))
    if metadata_path:
        config.set("metadata_path", str(metadata_path))

    # Initialize pipeline
    pipeline = AnalysisPipeline(config.config)
    results = pipeline.run_pipeline(image_path)

    # Output results
    print(f"Results: {results}")
    df = pd.DataFrame([results])
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()