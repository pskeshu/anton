#!/usr/bin/env python3
"""
BBBC021 Quick Multi-Channel Analysis

Fast analysis of multiple compounds using the fixed data handling.
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

class QuickMultiChannelAnalysis:
    """Quick multi-channel analysis for BBBC021."""
    
    def __init__(self):
        self.results = []
        self.phenotype_database = defaultdict(list)
    
    async def analyze_compounds(self, max_compounds: int = 4) -> list:
        """Analyze multiple compounds quickly."""
        
        print("🧬 BBBC021 Quick Multi-Channel Analysis")
        print("=" * 50)
        
        # Use existing composite
        composite_path = "data/bbbc021/images/BBBC021_v1_images_Week1_22141/img.png"
        
        if not Path(composite_path).exists():
            print(f"❌ Composite image not found: {composite_path}")
            return []
        
        if not anton_available:
            print("❌ Anton not available")
            return []
        
        # Define compounds to test
        test_compounds = [
            {'compound': 'nocodazole', 'moa': 'Microtubule destabilizers', 'concentration': 1.0},
            {'compound': 'taxol', 'moa': 'Microtubule stabilizers', 'concentration': 2.0},
            {'compound': 'cytochalasin-d', 'moa': 'Actin disruptors', 'concentration': 0.5},
            {'compound': 'dmso', 'moa': 'Control', 'concentration': 0.0}
        ][:max_compounds]
        
        print(f"🎯 Testing {len(test_compounds)} compounds")
        
        results = []
        for i, compound_data in enumerate(test_compounds):
            print(f"\n🔬 Analyzing {i+1}/{len(test_compounds)}: {compound_data['compound']}")
            
            result = await self._analyze_one_compound(composite_path, compound_data)
            if result:
                results.append(result)
                self.phenotype_database[compound_data['compound']].append(result)
                print(f"   ✅ Analysis complete ({result['feature_count']} features)")
            else:
                print(f"   ❌ Analysis failed")
        
        return results
    
    async def _analyze_one_compound(self, image_path: str, compound_data: dict) -> dict:
        """Analyze one compound."""
        
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
                    },
                    'analysis_focus': f'phenotypic_effects_of_{compound_data["compound"]}_treatment'
                }
            }
            
            # Run Anton analysis
            pipeline = AnalysisPipeline(config)
            anton_results = await pipeline.run_pipeline(image_path)
            
            # Extract text from structured results
            combined_text = self._extract_text_from_results(anton_results)
            
            # Extract biological features
            features = self._extract_biological_features(combined_text)
            
            # Create result
            result = {
                'compound_id': compound_data['compound'],
                'mechanism_of_action': compound_data['moa'],
                'concentration': compound_data['concentration'],
                'image_path': image_path,
                'semantic_features': features,
                'combined_text_length': len(combined_text),
                'feature_count': sum(len(cat_features) for cat_features in features.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            return None
    
    def _extract_text_from_results(self, anton_results: dict) -> str:
        """Extract text from Anton results (handles structured data)."""
        
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
        
        return ' '.join(descriptions).lower()
    
    def _extract_biological_features(self, description: str) -> dict:
        """Extract biological features from text description."""
        
        features = {
            'cellular_morphology': {},
            'nuclear_phenotypes': {},
            'cytoskeletal_organization': {},
            'spatial_patterns': {},
            'treatment_effects': {}
        }
        
        # Enhanced pattern matching
        feature_patterns = {
            'cellular_morphology': {
                'round|circular|spherical': 'cell_rounding',
                'elongated|stretched|extended': 'cell_elongation',
                'large|enlarged|expanded': 'cell_enlargement',
                'small|shrunk|contracted': 'cell_shrinkage',
                'irregular|distorted|deformed': 'irregular_morphology',
                'spread|flattened': 'cell_spreading'
            },
            'nuclear_phenotypes': {
                'fragmented|broken': 'nuclear_fragmentation',
                'condensed|compact': 'nuclear_condensation',
                'bright|intense': 'increased_nuclear_signal',
                'dim|weak|faint': 'decreased_nuclear_signal',
                'irregular|distorted': 'nuclear_shape_changes',
                'nuclei|nuclear|nucleus': 'nuclear_detection'
            },
            'cytoskeletal_organization': {
                'disrupted|disorganized|fragmented': 'cytoskeleton_disruption',
                'bundled|aggregated|clumped': 'cytoskeleton_bundling',
                'dense|thick|abundant': 'increased_cytoskeletal_density',
                'sparse|thin|reduced': 'decreased_cytoskeletal_density',
                'tubulin|microtubule': 'microtubule_detection',
                'actin|filament': 'actin_detection'
            },
            'spatial_patterns': {
                'clustered|grouped|aggregated': 'cell_clustering',
                'scattered|dispersed|isolated': 'cell_dispersion',
                'contact|touching|adherent': 'cell_cell_contact'
            },
            'treatment_effects': {
                'apoptotic|dying|dead': 'cell_death',
                'mitotic|dividing': 'cell_division',
                'stressed|damaged': 'cellular_stress',
                'normal|healthy|control': 'normal_morphology',
                'arrested|blocked': 'cell_cycle_arrest'
            }
        }
        
        # Extract features using pattern matching
        for category, patterns in feature_patterns.items():
            for pattern, feature_name in patterns.items():
                if any(keyword in description for keyword in pattern.split('|')):
                    features[category][feature_name] = 1.0
        
        return features
    
    def demonstrate_discoveries(self, results: list):
        """Show biological discoveries from analysis."""
        
        print("\n🔍 BIOLOGICAL DISCOVERIES")
        print("=" * 50)
        
        # Group by mechanism
        moa_groups = defaultdict(list)
        for result in results:
            moa_groups[result['mechanism_of_action']].append(result)
        
        print("\n1. 🧬 MECHANISM-SPECIFIC PHENOTYPES")
        for moa, group_results in moa_groups.items():
            print(f"\n   {moa}:")
            
            # Collect all features for this mechanism
            all_features = set()
            for result in group_results:
                for category, features in result['semantic_features'].items():
                    for feature in features.keys():
                        all_features.add(f"{category}: {feature}")
            
            if all_features:
                for feature in sorted(all_features):
                    print(f"      • {feature}")
            else:
                print(f"      • No specific features detected")
        
        print(f"\n2. 📊 ANALYSIS STATISTICS")
        total_features = sum(result['feature_count'] for result in results)
        avg_features = total_features / len(results) if results else 0
        
        print(f"   Total compounds analyzed: {len(results)}")
        print(f"   Total features extracted: {total_features}")
        print(f"   Average features per compound: {avg_features:.1f}")
        
        print(f"\n3. 🎯 COMPOUND COMPARISON")
        for result in results:
            features_by_category = {}
            for category, features in result['semantic_features'].items():
                if features:
                    features_by_category[category] = len(features)
            
            print(f"   {result['compound_id']}: {result['feature_count']} total features")
            if features_by_category:
                category_summary = ', '.join([f"{cat}: {count}" for cat, count in features_by_category.items()])
                print(f"      Categories: {category_summary}")

async def main():
    """Main analysis workflow."""
    
    analyzer = QuickMultiChannelAnalysis()
    
    # Run analysis
    results = await analyzer.analyze_compounds(max_compounds=4)
    
    if results:
        # Save results
        output_file = 'bbbc021_quick_multichannel_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Results saved to: {output_file}")
        
        # Demonstrate discoveries
        analyzer.demonstrate_discoveries(results)
        
    else:
        print("❌ No results obtained from analysis")

if __name__ == "__main__":
    asyncio.run(main())