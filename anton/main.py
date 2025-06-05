import argparse
from pathlib import Path
import logging
import pandas as pd

from anton.core.config import Config
from anton.core.pipeline import AnalysisPipeline

logging.basicConfig(level=logging.INFO)

def main(goal, image_path, metadata_path=None, config_path=None):
    """Main function for Anton CMPO phenotype analysis framework."""
    # Load configuration
    config = Config(config_path)
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
    parser = argparse.ArgumentParser(description="Anton CMPO phenotype analysis framework")
    parser.add_argument("--goal", type=str, required=True, help="Analysis goal (e.g., 'Identify apoptotic cells in DAPI-stained channel 1')")
    parser.add_argument("--image", type=str, required=True, help="Path to TIFF image")
    parser.add_argument("--metadata", type=str, help="Path to metadata file (e.g., OME-XML)")
    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
    args = parser.parse_args()
    main(args.goal, args.image, args.metadata, args.config)