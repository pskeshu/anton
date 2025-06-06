#!/usr/bin/env python3
"""
Anton Framework Demo - Showcase the hybrid microscopy analysis pipeline

This demo shows how Anton combines traditional computer vision with 
modern Vision Language Models for comprehensive phenotype analysis.
"""

import asyncio
import sys
from pathlib import Path
import json

# Add anton to path
sys.path.insert(0, str(Path(__file__).parent))

from anton.core.pipeline import AnalysisPipeline
from anton.analysis.quantitative import QuantitativeAnalyzer, SegmentationStrategy

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")

async def demo_anton_framework():
    """Main demo function showcasing Anton's capabilities."""
    
    print_header("🔬 ANTON FRAMEWORK DEMO")
    print("Hybrid Microscopy Phenotype Analysis using VLMs + Computer Vision")
    print("\nAnton combines:")
    print("• Traditional quantitative analysis (segmentation, morphology, texture)")
    print("• Modern Vision Language Models (Claude, Gemini, GPT-4V)")
    print("• CMPO ontology mapping for standardized phenotype terms")
    
    # Configuration
    config = {
        "vlm_provider": "claude",
        "channels": [0, 1, 2],  # RGB channels for demo
        "quantitative": {
            "min_object_area": 30,
            "max_object_area": 5000
        },
        "batch_size": 5
    }
    
    # Check for sample images
    sample_image = Path("data/sample_images/demo_images/img00.png")
    if not sample_image.exists():
        print("\n❌ Sample images not found!")
        print(f"   Expected: {sample_image}")
        print("   Please ensure demo images are available in data/sample_images/")
        return
    
    print_section("🚀 Initializing Anton Pipeline")
    
    try:
        # Initialize the analysis pipeline
        pipeline = AnalysisPipeline(config)
        print("✅ Pipeline initialized successfully!")
        print(f"   VLM Provider: {pipeline.vlm.provider}")
        print(f"   VLM Model: {pipeline.vlm.model}")
        print(f"   Available Prompts: {len(pipeline.vlm.prompts)}")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    print_section("📸 Sample Image Analysis")
    print(f"Analyzing: {sample_image.name}")
    print("Image shows fluorescence microscopy with:")
    print("• Blue nuclei (DAPI)")
    print("• Green cytoskeleton/cytoplasm")
    
    try:
        print_section("🔍 Stage 1: Global Scene Understanding")
        stage1_result = await pipeline.run_stage_1(sample_image)
        print("✅ VLM analyzed the global scene")
        print(f"   Quality Score: {stage1_result.get('quality_score', 'N/A')}")
        print(f"   Analysis: {stage1_result.get('analysis', 'N/A')}")
        
        print_section("🎯 Stage 2: Object Detection & Segmentation Guidance")
        stage2_result = await pipeline.run_stage_2()
        detected_objects = stage2_result.get('detected_objects', [])
        print(f"✅ Detected {len(detected_objects)} object types")
        for obj in detected_objects:
            print(f"   • {obj.get('type', 'Unknown')}: confidence {obj.get('confidence', 0):.2f}")
        print(f"   Segmentation guidance: {stage2_result.get('segmentation_guidance', 'N/A')}")
        
        print_section("🧬 Stage 3: Feature-Level Analysis")
        stage3_result = await pipeline.run_stage_3()
        object_analyses = stage3_result.get('object_analyses', [])
        print(f"✅ Analyzed {len(object_analyses)} objects for detailed features")
        for analysis in object_analyses[:3]:  # Show first 3
            features = analysis.get('features', [])
            print(f"   Object {analysis.get('object_id', '?')}: {', '.join(features)}")
        
        print_section("📊 Stage 4: Population-Level Insights")
        stage4_result = await pipeline.run_stage_4()
        print("✅ Generated population-level insights")
        print(f"   Summary: {stage4_result.get('population_summary', 'N/A')}")
        
        # Show full pipeline results
        print_section("🎉 Complete Pipeline Results")
        all_results = {
            "stage_1_global": stage1_result,
            "stage_2_objects": stage2_result,
            "stage_3_features": stage3_result,
            "stage_4_population": stage4_result
        }
        
        print("✅ All 4 stages completed successfully!")
        print("\nPipeline generated:")
        for stage, result in all_results.items():
            if result:
                print(f"   • {stage}: {len(result)} keys")
        
    except Exception as e:
        print(f"❌ VLM Pipeline analysis failed: {e}")
        
        if "API" in str(e) or "key" in str(e).lower():
            print_section("🔑 VLM API Key Required")
            print("To see real VLM analysis, you need to provide an API key:")
            print("")
            print("For Claude API:")
            print("  export ANTHROPIC_API_KEY='your-api-key-here'")
            print("  # Get key from: https://console.anthropic.com/")
            print("")
            print("For Gemini API:")
            print("  export GOOGLE_API_KEY='your-api-key-here'") 
            print("  # Get key from: https://makersuite.google.com/app/apikey")
            print("")
            print("For OpenAI API:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            print("  # Get key from: https://platform.openai.com/api-keys")
            print("")
            print("Set provider in config: vlm_provider='claude'|'gemini'|'openai'")
            print("Then run the demo again to see real VLM responses!")
            print("")
            print("Without API keys, Anton can still run quantitative analysis...")
        else:
            import traceback
            traceback.print_exc()
        return
    
    print_section("🔬 Quantitative Analysis Demo")
    
    try:
        # Show quantitative analysis capabilities
        quant_analyzer = QuantitativeAnalyzer(config.get("quantitative", {}))
        
        print("Available segmentation methods:")
        for strategy in SegmentationStrategy:
            print(f"   • {strategy.value}")
        
        print(f"\nTesting threshold segmentation on {sample_image.name}...")
        quant_results = quant_analyzer.extract_quantitative_features(
            image_path=sample_image,
            channels=[2],  # Blue channel for nuclei
            method=SegmentationStrategy.THRESHOLD
        )
        
        print(f"✅ Quantitative analysis completed")
        print(f"   Objects detected: {quant_results['num_objects']}")
        print(f"   Method used: {quant_results['method_used']}")
        
        if quant_results['num_objects'] > 0:
            print("   Extractable features:")
            print("     • Morphological (area, perimeter, eccentricity, etc.)")
            print("     • Intensity (mean, std, percentiles, etc.)")
            print("     • Texture (contrast, variance, LBP, etc.)")
            print("     • Spatial (neighbor distances, edge distances, etc.)")
        else:
            print("   Note: No objects detected with current parameters")
            print("         (This is normal - segmentation needs tuning for real data)")
        
    except Exception as e:
        print(f"❌ Quantitative analysis failed: {e}")
    
    print_section("🧠 Qualitative Analysis Demo")
    
    try:
        # Create mock regions for qualitative demo
        mock_regions = [
            type('MockRegion', (), {
                'label': 1, 'bbox': (10, 10, 50, 50), 'area': 1600,
                'centroid': (30, 30), 'eccentricity': 0.5, 'solidity': 0.8
            })()
        ]
        
        print("✅ Qualitative analyzer extracts:")
        print("   • Texture-based features from image patches")
        print("   • Population-level insights")
        print("   • CMPO ontology mappings")
        
        qual_results = await pipeline.qual_analyzer.extract_qualitative_features(
            image_path=sample_image,
            regions=mock_regions,
            config=config
        )
        
        print(f"✅ Qualitative analysis completed")
        print(f"   Global context: extracted")
        print(f"   Region features: {len(qual_results['region_features'])}")
        print(f"   CMPO mappings: {len(qual_results['cmpo_mappings'])}")
        
    except Exception as e:
        print(f"❌ Qualitative analysis failed: {e}")
    
    print_section("💡 Key Features Demonstrated")
    print("✅ Hybrid Analysis: Computer vision + Vision Language Models")
    print("✅ Multi-Stage Pipeline: Global → Objects → Features → Population")
    print("✅ Async/Sync Support: Efficient for batch processing")
    print("✅ Multiple VLM Providers: Claude, Gemini, OpenAI support")
    print("✅ Ontology Integration: CMPO standardized phenotype terms")
    print("✅ Comprehensive Features: Morphology, intensity, texture, spatial")
    
    print_section("🚀 Next Steps")
    print("To use Anton with real data:")
    print("1. Configure your VLM API keys (ANTHROPIC_API_KEY, etc.)")
    print("2. Adjust segmentation parameters for your image type")
    print("3. Customize prompts in prompts/ directory")
    print("4. Run: python -m anton.main for interactive analysis")
    
    print_header("✨ DEMO COMPLETE")
    print("Anton is ready for hybrid microscopy phenotype analysis!")

def demo_simple_usage():
    """Show the simplest possible usage."""
    print_header("📝 SIMPLE USAGE EXAMPLE")
    
    print("""
# Basic Anton usage:

import asyncio
from anton.core.pipeline import AnalysisPipeline

async def analyze_image():
    config = {"vlm_provider": "claude", "channels": [0, 1, 2]}
    pipeline = AnalysisPipeline(config)
    
    # Run full 4-stage analysis
    results = await pipeline.run_pipeline("path/to/image.png")
    
    # Or run synchronously
    results = pipeline.run_pipeline_sync("path/to/image.png")
    
    return results

# Run the analysis
results = asyncio.run(analyze_image())
print(f"Detected {len(results['stage_2_objects']['detected_objects'])} objects")
""")

if __name__ == "__main__":
    print("🔬 Welcome to the Anton Framework Demo!")
    print("\nChoose demo mode:")
    print("1. Full interactive demo (recommended)")
    print("2. Simple usage example")
    print("3. Run both")
    
    try:
        choice = input("\nEnter choice (1-3, or Enter for full demo): ").strip()
    except EOFError:
        # Non-interactive environment, run full demo
        choice = "1"
        print("Running in non-interactive mode - starting full demo...")
    
    if choice == "2":
        demo_simple_usage()
    elif choice == "3":
        asyncio.run(demo_anton_framework())
        demo_simple_usage()
    else:
        # Default: full demo
        asyncio.run(demo_anton_framework())