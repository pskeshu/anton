"""Basic example demonstrating Anton's core functionality."""

import asyncio
import os
from pathlib import Path
import json

from anton.core.pipeline import AntonPipeline
from anton.vlm.interface import VLMInterface
from anton.analysis.quantitative import QuantitativeAnalyzer, SegmentationStrategy
from anton.analysis.qualitative import QualitativeAnalyzer
from anton.cmpo.ontology import CMPOOntology

async def main():
    # Configuration
    config = {
        'segmentation': {
            'method': SegmentationStrategy.CELLPOSE,
            'parameters': {
                'diameter': 30,
                'batch_size': 8
            }
        },
        'analysis': {
            'channels': [0, 1],  # DAPI and GFP channels
            'batch_size': 10,
            'confidence_threshold': 0.5
        },
        'cmpo': {
            'ontology_path': 'data/cmpo.json',
            'cache_path': 'data/cmpo_cache.pkl'
        }
    }

    # Initialize components
    vlm = VLMInterface()
    ontology = CMPOOntology()
    
    # Create pipeline
    pipeline = AntonPipeline(vlm, config)

    # Example 1: Full pipeline analysis
    print("\n=== Running Full Pipeline Analysis ===")
    results = await pipeline.analyze_image(
        image_path="data/sample_images/cells.tif",
        goal="detect_apoptotic_cells"
    )

    # Print results summary
    print("\nAnalysis Results:")
    print(f"Total regions detected: {len(results['quantitative']['regions'])}")
    print("\nCMPO Phenotypes detected:")
    for phenotype, data in results['summary']['phenotype_prevalence'].items():
        print(f"- {phenotype}: {data['count']} features ({data['prevalence']:.1%})")

    # Example 2: Quantitative analysis only
    print("\n=== Running Quantitative Analysis ===")
    quant_analyzer = QuantitativeAnalyzer(config)
    quant_results = quant_analyzer.extract_quantitative_features(
        image_path="data/sample_images/cells.tif",
        channels=[0, 1]
    )

    # Print quantitative results
    print("\nQuantitative Analysis Results:")
    print(f"Regions detected: {len(quant_results['regions'])}")
    print("\nMorphological features summary:")
    print(quant_results['morphological'].describe())

    # Example 3: Qualitative analysis with CMPO mapping
    print("\n=== Running Qualitative Analysis ===")
    qual_analyzer = QualitativeAnalyzer(vlm, ontology)
    qual_results = await qual_analyzer.extract_qualitative_features(
        image_path="data/sample_images/cells.tif",
        regions=quant_results['regions'],
        config=config
    )

    # Print qualitative results
    print("\nQualitative Analysis Results:")
    print("Global context:", qual_results['global_context'])
    print("\nRegion features:", len(qual_results['region_features']))
    print("\nCMPO mappings:", len(qual_results['cmpo_mappings']))

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save full results
    with open(output_dir / "full_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save quantitative results
    quant_results['morphological'].to_csv(output_dir / "morphological_features.csv")
    quant_results['intensity'].to_csv(output_dir / "intensity_features.csv")
    quant_results['texture'].to_csv(output_dir / "texture_features.csv")

if __name__ == "__main__":
    asyncio.run(main()) 