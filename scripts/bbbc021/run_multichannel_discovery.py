#!/usr/bin/env python3
"""
Run Multi-Channel Connection Discovery

Analyze the created composites for biological connection discovery.
"""

import os
import sys
import pandas as pd
import asyncio
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

async def run_discovery_analysis():
    """Run connection discovery on multi-channel composites."""
    
    print("🧬 BBBC021 Multi-Channel Connection Discovery")
    print("=" * 60)
    
    # Load analysis dataset
    dataset_path = "data/bbbc021/multichannel_analysis_dataset.csv"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Run: python create_multichannel_composites.py first")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Loaded {len(df)} composite images")
    print(f"🧪 Compounds: {df['compound'].nunique()}")
    print(f"🎯 MOAs: {df['moa'].nunique()}")
    
    if not anton_available:
        print("❌ Anton not available for VLM analysis")
        return
    
    # Group by compound for analysis
    compound_groups = df.groupby('compound')
    
    results = []
    discovery_database = defaultdict(list)
    
    for compound_name, group in compound_groups:
        moa = group.iloc[0]['moa']
        print(f"\n🔬 Analyzing {compound_name} ({moa})")
        print(f"   📸 Images: {len(group)}")
        
        # Analyze representative image from each compound
        representative = group.iloc[0]
        
        result = await analyze_composite(representative)
        if result:
            results.append(result)
            discovery_database[compound_name].append(result)
            print(f"   ✅ Analysis complete: {result['feature_count']} features")
        else:
            print(f"   ❌ Analysis failed")
    
    if results:
        # Save results
        output_file = 'bbbc021_multichannel_discovery_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Discovery Analysis Complete!")
        print(f"📊 Results saved: {output_file}")
        
        # Demonstrate discoveries
        demonstrate_discoveries(results, discovery_database)
        
    else:
        print("❌ No results obtained")

async def analyze_composite(row):
    """Analyze one composite image."""
    
    try:
        composite_path = row['composite_path']
        
        # Configure Anton with rich biological context
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'biological_context': {
                'experiment_type': '3_channel_fluorescence_compound_screen',
                'cell_line': 'MCF7_breast_cancer_cells',
                'compound': row['compound'],
                'mechanism_of_action': row['moa'],
                'concentration': f"{row['concentration']} μM",
                'well_position': row['well'],
                'site_number': row['site'],
                'imaging_channels': {
                    'blue_DAPI': 'nuclear_DNA_chromatin_organization',
                    'green_tubulin': 'microtubule_cytoskeleton_spindle_apparatus',
                    'red_actin': 'actin_cytoskeleton_stress_fibers_cell_shape'
                },
                'analysis_objectives': {
                    'primary': 'identify_compound_induced_phenotypic_changes',
                    'secondary': 'assess_nuclear_cytoskeletal_morphological_alterations',
                    'discovery': 'find_unexpected_phenotypic_similarities_across_compounds'
                }
            }
        }
        
        # Run Anton analysis
        pipeline = AnalysisPipeline(config)
        anton_results = await pipeline.run_pipeline(composite_path)
        
        # Extract features
        combined_text = extract_text_from_anton_results(anton_results)
        features = extract_biological_features(combined_text)
        
        # Create result
        result = {
            'composite_id': row['composite_id'],
            'compound': row['compound'],
            'moa': row['moa'],
            'concentration': row['concentration'],
            'well': row['well'],
            'site': row['site'],
            'composite_path': composite_path,
            'semantic_features': features,
            'feature_count': sum(len(cat) for cat in features.values()),
            'analysis_length': len(combined_text),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"     ❌ Error analyzing {row['composite_id']}: {e}")
        return None

def extract_text_from_anton_results(anton_results):
    """Extract text from Anton structured results."""
    
    descriptions = []
    for stage in ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']:
        if stage in anton_results and anton_results[stage]:
            stage_data = anton_results[stage]
            
            if isinstance(stage_data, str):
                descriptions.append(stage_data)
            elif isinstance(stage_data, dict):
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
    
    return ' '.join(descriptions).lower()

def extract_biological_features(description):
    """Extract biological features from description."""
    
    features = {
        'nuclear_phenotypes': {},
        'cytoskeletal_organization': {},
        'cellular_morphology': {},
        'treatment_effects': {},
        'spatial_organization': {}
    }
    
    # Enhanced feature patterns
    patterns = {
        'nuclear_phenotypes': {
            'fragmented|broken': 'nuclear_fragmentation',
            'condensed|compact|dense': 'nuclear_condensation',
            'enlarged|large|expanded': 'nuclear_enlargement',
            'irregular|distorted|abnormal': 'nuclear_shape_abnormalities',
            'bright|intense|strong': 'increased_nuclear_intensity',
            'dim|weak|faint': 'decreased_nuclear_intensity'
        },
        'cytoskeletal_organization': {
            'disrupted|disorganized|fragmented': 'cytoskeleton_disruption',
            'bundled|aggregated|clumped': 'cytoskeleton_bundling',
            'dense|thick|abundant': 'increased_cytoskeletal_density',
            'sparse|thin|reduced': 'decreased_cytoskeletal_density',
            'tubulin|microtubule': 'microtubule_alterations',
            'actin|filament|fiber': 'actin_alterations'
        },
        'cellular_morphology': {
            'round|circular|spherical': 'cell_rounding',
            'elongated|stretched|extended': 'cell_elongation',
            'large|enlarged|expanded': 'cell_enlargement',
            'small|shrunk|contracted': 'cell_shrinkage',
            'irregular|distorted|abnormal': 'irregular_cell_shape',
            'spread|flattened': 'cell_spreading'
        },
        'treatment_effects': {
            'apoptotic|dying|dead': 'apoptosis_induction',
            'mitotic|dividing': 'mitotic_effects',
            'arrested|blocked|stopped': 'cell_cycle_arrest',
            'stressed|damaged': 'cellular_stress',
            'proliferative|growing': 'proliferation_effects'
        },
        'spatial_organization': {
            'clustered|grouped|aggregated': 'cell_clustering',
            'scattered|dispersed|isolated': 'cell_dispersion',
            'contact|touching|adherent': 'cell_cell_contact',
            'organized|aligned|structured': 'organized_growth'
        }
    }
    
    # Extract features
    for category, pattern_dict in patterns.items():
        for pattern, feature_name in pattern_dict.items():
            if any(keyword in description for keyword in pattern.split('|')):
                features[category][feature_name] = 1.0
    
    return features

def demonstrate_discoveries(results, discovery_database):
    """Demonstrate biological discoveries."""
    
    print(f"\n🔍 BIOLOGICAL CONNECTION DISCOVERIES")
    print("=" * 60)
    
    # 1. Mechanism-specific phenotypes
    print(f"\n1. 🧬 MECHANISM-SPECIFIC PHENOTYPES")
    moa_features = defaultdict(set)
    
    for result in results:
        moa = result['moa']
        for category, features in result['semantic_features'].items():
            for feature in features.keys():
                moa_features[moa].add(f"{category}: {feature}")
    
    for moa, features in moa_features.items():
        print(f"\n   {moa}:")
        if features:
            for feature in sorted(features):
                print(f"      • {feature}")
        else:
            print(f"      • No specific features detected")
    
    # 2. Cross-compound similarities
    print(f"\n2. 🔗 CROSS-COMPOUND SIMILARITIES")
    feature_compounds = defaultdict(list)
    
    for result in results:
        compound = result['compound']
        for category, features in result['semantic_features'].items():
            for feature in features.keys():
                feature_key = f"{category}: {feature}"
                feature_compounds[feature_key].append(compound)
    
    # Find shared features
    shared_features = {k: v for k, v in feature_compounds.items() if len(v) > 1}
    
    for feature, compounds in shared_features.items():
        print(f"\n   {feature}:")
        print(f"      Shared by: {', '.join(compounds)}")
    
    # 3. Analysis statistics
    print(f"\n3. 📊 DISCOVERY STATISTICS")
    total_features = sum(result['feature_count'] for result in results)
    avg_features = total_features / len(results) if results else 0
    
    print(f"   Total compounds analyzed: {len(results)}")
    print(f"   Total features discovered: {total_features}")
    print(f"   Average features per compound: {avg_features:.1f}")
    print(f"   Shared phenotypes found: {len(shared_features)}")
    
    # 4. Compound ranking
    print(f"\n4. 🎯 COMPOUND PHENOTYPE RICHNESS")
    sorted_results = sorted(results, key=lambda x: x['feature_count'], reverse=True)
    
    for result in sorted_results:
        print(f"   {result['compound']}: {result['feature_count']} features ({result['moa']})")

if __name__ == "__main__":
    asyncio.run(run_discovery_analysis())