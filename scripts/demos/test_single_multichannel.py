#!/usr/bin/env python3
"""
Test Single Multi-Channel Image Analysis

Quick test to verify VLM analysis works with the existing composite image.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

async def test_single_image():
    """Test analysis of the single composite image."""
    
    print("🧬 Testing Single Multi-Channel Image Analysis")
    print("=" * 50)
    
    # Use the existing composite
    composite_path = "data/bbbc021/images/BBBC021_v1_images_Week1_22141/img.png"
    
    if not Path(composite_path).exists():
        print(f"❌ Composite image not found: {composite_path}")
        return
    
    print(f"📸 Using image: {composite_path}")
    
    if not anton_available:
        print("❌ Anton not available - cannot test VLM")
        return
    
    # Configure Anton with rich biological context
    config = {
        'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
        'biological_context': {
            'experiment_type': 'compound_screen_multichannel_fluorescence',
            'cell_line': 'MCF7_breast_cancer_cells',
            'staining_protocol': {
                'blue_channel_DAPI': 'nuclear_DNA_staining',
                'green_channel_tubulin': 'beta_tubulin_microtubules_cytoskeleton', 
                'red_channel_actin': 'F_actin_stress_fibers_cytoskeleton'
            },
            'imaging_details': {
                'microscopy_type': '3_channel_fluorescence_confocal',
                'magnification': 'high_resolution_cellular_detail',
                'composite_type': 'RGB_overlay_all_channels_visible'
            },
            'analysis_goals': {
                'primary': 'identify_cellular_phenotypes_and_morphological_changes',
                'secondary': 'assess_nuclear_and_cytoskeletal_organization',
                'quantify': 'cell_shape_size_intensity_spatial_patterns'
            },
            'expected_features': {
                'nuclei': 'blue_DAPI_round_to_irregular_shapes',
                'microtubules': 'green_tubulin_radial_or_disrupted_patterns',
                'actin': 'red_stress_fibers_or_cortical_rings',
                'cells': 'individual_cells_with_distinct_boundaries'
            }
        }
    }
    
    print("🔬 Starting VLM analysis...")
    print("   This may take 1-2 minutes...")
    
    try:
        # Run Anton analysis
        pipeline = AnalysisPipeline(config)
        anton_results = await pipeline.run_pipeline(composite_path)
        
        print("✅ VLM analysis complete!")
        
        # Display results summary
        print(f"\n📊 ANALYSIS RESULTS SUMMARY")
        print("=" * 40)
        
        for stage_name, stage_result in anton_results.items():
            if stage_result and isinstance(stage_result, str):
                preview = stage_result[:200] + "..." if len(stage_result) > 200 else stage_result
                print(f"\n🔍 {stage_name.upper()}:")
                print(f"   Length: {len(stage_result)} characters")
                print(f"   Preview: {preview}")
        
        # Check for quality indicators
        combined_text = ' '.join(str(result) for result in anton_results.values() if result).lower()
        
        quality_indicators = {
            'mentions_channels': any(word in combined_text for word in ['blue', 'green', 'red', 'channel', 'fluorescence']),
            'mentions_nuclei': any(word in combined_text for word in ['nuclei', 'nuclear', 'dapi']),
            'mentions_cytoskeleton': any(word in combined_text for word in ['cytoskeleton', 'actin', 'tubulin', 'microtubule']),
            'mentions_cells': any(word in combined_text for word in ['cell', 'cellular', 'morphology']),
            'mentions_shapes': any(word in combined_text for word in ['round', 'elongated', 'irregular', 'shape']),
            'total_length': len(combined_text)
        }
        
        print(f"\n📈 QUALITY ASSESSMENT:")
        print("=" * 30)
        for indicator, value in quality_indicators.items():
            status = "✅" if value else "❌"
            print(f"   {status} {indicator}: {value}")
        
        # Save full results
        output_file = 'test_single_multichannel_results.json'
        import json
        with open(output_file, 'w') as f:
            json.dump({
                'image_path': composite_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'anton_results': anton_results,
                'quality_indicators': quality_indicators
            }, f, indent=2, default=str)
        
        print(f"\n💾 Full results saved to: {output_file}")
        
        # Overall assessment
        quality_score = sum(1 for k, v in quality_indicators.items() if k != 'total_length' and v)
        total_checks = len(quality_indicators) - 1
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Quality Score: {quality_score}/{total_checks}")
        
        if quality_score >= 4:
            print("   🎉 EXCELLENT - VLM successfully analyzing multi-channel data!")
        elif quality_score >= 3:
            print("   ✅ GOOD - VLM recognizing most biological features")
        elif quality_score >= 2:
            print("   ⚠️  FAIR - VLM partially recognizing features")
        else:
            print("   ❌ POOR - VLM may not be processing properly")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_image())