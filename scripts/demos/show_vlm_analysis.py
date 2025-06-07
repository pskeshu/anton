#!/usr/bin/env python3
"""
Show VLM Analysis Results

Display the detailed VLM analysis of the composite image.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

async def show_vlm_analysis():
    """Show detailed VLM analysis."""
    
    print("🧬 VLM Analysis of Multi-Channel Composite")
    print("=" * 60)
    
    composite_path = "data/bbbc021/composites/Week1_150607_B02_s2_composite.png"
    
    if not anton_available:
        print("❌ Anton not available")
        return
    
    print(f"📸 Image: {composite_path}")
    print(f"🧪 Compound: DMSO (control)")
    print(f"🔬 Channels: Blue (DAPI/nuclei), Green (Tubulin), Red (Actin)")
    print()
    
    try:
        # Configure with rich context
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'biological_context': {
                'experiment_type': 'compound_screen_3channel_fluorescence_microscopy',
                'cell_line': 'MCF7_breast_cancer_cells',
                'compound_treatment': 'DMSO_vehicle_control',
                'imaging_channels': {
                    'blue_channel': 'DAPI_nuclear_DNA_staining',
                    'green_channel': 'beta_tubulin_microtubule_cytoskeleton',
                    'red_channel': 'F_actin_cytoskeleton_stress_fibers'
                },
                'analysis_objectives': [
                    'assess_cellular_morphology_and_organization',
                    'evaluate_nuclear_characteristics',
                    'analyze_cytoskeletal_architecture',
                    'identify_any_treatment_induced_phenotypes'
                ]
            }
        }
        
        print("🔄 Running detailed VLM analysis...")
        
        pipeline = AnalysisPipeline(config)
        results = await pipeline.run_pipeline(composite_path)
        
        print("✅ Analysis complete!\n")
        
        # Display each stage in detail
        for stage_name, stage_data in results.items():
            print(f"{'='*20} {stage_name.upper()} {'='*20}")
            
            if isinstance(stage_data, str):
                print(stage_data)
            elif isinstance(stage_data, dict):
                # Pretty print structured data
                if 'description' in stage_data:
                    print(stage_data['description'])
                    print()
                    
                # Show other structured elements
                for key, value in stage_data.items():
                    if key != 'description':
                        if isinstance(value, (list, dict)):
                            print(f"{key.upper()}:")
                            print(json.dumps(value, indent=2))
                        else:
                            print(f"{key}: {value}")
                        print()
            
            print() # Extra spacing between stages
        
        # Save detailed results
        output_file = 'detailed_vlm_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Complete analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(show_vlm_analysis())