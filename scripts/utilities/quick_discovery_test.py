#!/usr/bin/env python3
"""
Quick Discovery Test

Test connection discovery on just 3 compounds to see results faster.
"""

import os
import sys
import pandas as pd
import asyncio
import json
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

async def quick_discovery_test():
    """Quick test with 3 compounds."""
    
    print("🧬 Quick Multi-Channel Discovery Test")
    print("=" * 50)
    
    # Load dataset
    dataset_path = "data/bbbc021/multichannel_analysis_dataset.csv"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Loaded {len(df)} composite images")
    
    if not anton_available:
        print("❌ Anton not available")
        return
    
    # Select 3 different compounds for testing
    compounds_to_test = ['taxol', 'anisomycin', 'DMSO']
    test_data = []
    
    for compound in compounds_to_test:
        compound_rows = df[df['compound'] == compound]
        if not compound_rows.empty:
            test_data.append(compound_rows.iloc[0])
    
    print(f"🎯 Testing {len(test_data)} compounds:")
    for row in test_data:
        print(f"   • {row['compound']} ({row['moa']})")
    
    results = []
    
    for i, row in enumerate(test_data):
        print(f"\n🔬 Analyzing {i+1}/{len(test_data)}: {row['compound']}")
        
        result = await analyze_one_compound(row)
        if result:
            results.append(result)
            print(f"   ✅ Features found: {result['feature_count']}")
        else:
            print(f"   ❌ Analysis failed")
    
    if results:
        # Save results
        with open('quick_discovery_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Quick Test Complete!")
        show_results(results)
    else:
        print("❌ No results obtained")

async def analyze_one_compound(row):
    """Analyze one compound quickly."""
    
    try:
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'biological_context': {
                'experiment_type': 'compound_screen_3channel_fluorescence',
                'cell_line': 'MCF7_breast_cancer_cells',
                'compound': row['compound'],
                'mechanism_of_action': row['moa'],
                'imaging_channels': 'DAPI_blue_tubulin_green_actin_red',
                'analysis_goal': 'identify_phenotypic_changes_from_compound_treatment'
            }
        }
        
        pipeline = AnalysisPipeline(config)
        anton_results = await pipeline.run_pipeline(row['composite_path'])
        
        # Quick feature extraction
        combined_text = ""
        for stage_data in anton_results.values():
            if isinstance(stage_data, str):
                combined_text += stage_data.lower()
            elif isinstance(stage_data, dict):
                for value in stage_data.values():
                    if isinstance(value, str):
                        combined_text += value.lower()
        
        # Simple feature detection
        features = {}
        feature_patterns = {
            'nuclear_effects': ['nuclear', 'nuclei', 'dapi', 'chromatin'],
            'cytoskeleton_effects': ['tubulin', 'actin', 'cytoskeleton', 'microtubule'],
            'morphology_changes': ['round', 'elongated', 'shape', 'morphology'],
            'cellular_organization': ['organized', 'disrupted', 'pattern', 'structure']
        }
        
        for category, keywords in feature_patterns.items():
            detected = [kw for kw in keywords if kw in combined_text]
            if detected:
                features[category] = detected
        
        return {
            'compound': row['compound'],
            'moa': row['moa'],
            'composite_path': row['composite_path'],
            'features': features,
            'feature_count': len([f for cat in features.values() for f in cat]),
            'analysis_length': len(combined_text),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"     ❌ Error: {e}")
        return None

def show_results(results):
    """Show quick results summary."""
    
    print(f"\n📋 RESULTS SUMMARY")
    print("=" * 30)
    
    for result in results:
        print(f"\n🔬 {result['compound']} ({result['moa']})")
        print(f"   Features detected: {result['feature_count']}")
        print(f"   Analysis length: {result['analysis_length']} chars")
        
        if result['features']:
            for category, keywords in result['features'].items():
                print(f"   {category}: {', '.join(keywords)}")
        else:
            print(f"   No specific features detected")
    
    # Cross-compound comparison
    print(f"\n🔗 CROSS-COMPOUND PATTERNS")
    all_features = {}
    for result in results:
        for category, keywords in result['features'].items():
            if category not in all_features:
                all_features[category] = {}
            for keyword in keywords:
                if keyword not in all_features[category]:
                    all_features[category][keyword] = []
                all_features[category][keyword].append(result['compound'])
    
    for category, feature_dict in all_features.items():
        shared_features = {k: v for k, v in feature_dict.items() if len(v) > 1}
        if shared_features:
            print(f"\n   {category}:")
            for feature, compounds in shared_features.items():
                print(f"      '{feature}' shared by: {', '.join(compounds)}")

if __name__ == "__main__":
    asyncio.run(quick_discovery_test())