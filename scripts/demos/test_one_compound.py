#!/usr/bin/env python3
"""
Test One Compound Analysis

Quick test of the fixed multi-channel analysis for just one compound.
"""

import os
import sys
import pandas as pd
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

async def test_one_compound():
    """Test analysis of one compound."""
    
    print("🧬 Testing One Compound Analysis")
    print("=" * 40)
    
    # Use the existing composite
    composite_path = "data/bbbc021/images/BBBC021_v1_images_Week1_22141/img.png"
    
    if not Path(composite_path).exists():
        print(f"❌ Composite image not found: {composite_path}")
        return
    
    if not anton_available:
        print("❌ Anton not available")
        return
    
    # Mock compound data
    compound_data = {
        'compound': 'nocodazole',
        'moa': 'Microtubule destabilizers',
        'concentration': 1.0
    }
    
    print(f"🔬 Testing: {compound_data['compound']} ({compound_data['moa']})")
    
    try:
        # Configure Anton
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'biological_context': {
                'experiment_type': 'compound_screen_multichannel',
                'cell_line': 'MCF7_breast_cancer_cells',
                'compound': compound_data['compound'],
                'mechanism_of_action': compound_data['moa'],
                'concentration': f"{compound_data['concentration']} μM",
                'staining_channels': {
                    'blue_channel': 'DAPI_nuclei_DNA',
                    'green_channel': 'beta_tubulin_microtubules', 
                    'red_channel': 'F_actin_cytoskeleton'
                }
            }
        }
        
        # Run Anton analysis
        pipeline = AnalysisPipeline(config)
        anton_results = await pipeline.run_pipeline(composite_path)
        
        print("✅ Anton analysis complete!")
        
        # Test feature extraction with fixed code
        descriptions = []
        for stage in ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']:
            if stage in anton_results and anton_results[stage]:
                stage_data = anton_results[stage]
                
                # Handle different data formats
                if isinstance(stage_data, str):
                    descriptions.append(stage_data)
                elif isinstance(stage_data, dict):
                    # Extract text from dictionary fields
                    text_parts = []
                    for key, value in stage_data.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    text_parts.append(item)
                                elif isinstance(item, dict):
                                    for sub_key, sub_value in item.items():
                                        if isinstance(sub_value, str):
                                            text_parts.append(sub_value)
                    descriptions.append(' '.join(text_parts))
        
        combined_text = ' '.join(descriptions).lower()
        
        print(f"✅ Feature extraction successful!")
        print(f"   Combined text length: {len(combined_text)} characters")
        
        # Test biological feature extraction
        features = {
            'cellular_morphology': {},
            'nuclear_phenotypes': {},
            'cytoskeletal_organization': {},
            'spatial_patterns': {},
            'treatment_effects': {}
        }
        
        # Sample pattern matching
        if 'round' in combined_text:
            features['cellular_morphology']['cell_rounding'] = 1.0
        if 'elongated' in combined_text:
            features['cellular_morphology']['cell_elongation'] = 1.0
        if 'nuclei' in combined_text or 'nuclear' in combined_text:
            features['nuclear_phenotypes']['nuclear_detection'] = 1.0
        if 'tubulin' in combined_text or 'microtubule' in combined_text:
            features['cytoskeletal_organization']['microtubule_detection'] = 1.0
        if 'actin' in combined_text:
            features['cytoskeletal_organization']['actin_detection'] = 1.0
        
        print(f"✅ Biological features extracted!")
        
        # Show extracted features
        feature_count = 0
        for category, category_features in features.items():
            if category_features:
                print(f"   {category}: {list(category_features.keys())}")
                feature_count += len(category_features)
        
        print(f"   Total features: {feature_count}")
        
        # Save results
        result = {
            'compound_id': compound_data['compound'],
            'mechanism_of_action': compound_data['moa'],
            'concentration': compound_data['concentration'],
            'image_path': composite_path,
            'semantic_features': features,
            'combined_text_length': len(combined_text),
            'feature_count': feature_count,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open('test_one_compound_result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"💾 Results saved to: test_one_compound_result.json")
        print(f"🎉 Test successful - ready for multi-compound analysis!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_one_compound())