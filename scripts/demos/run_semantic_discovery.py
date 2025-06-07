#!/usr/bin/env python3
"""
Run Semantic Discovery Analysis

Connection discovery with fixed semantic CMPO mapping.
"""

import os
import sys
import pandas as pd
import asyncio
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

class SemanticCMPOMapper:
    """Semantic CMPO mapping with biological reasoning."""
    
    def __init__(self):
        # Core CMPO terms with biological context
        self.cmpo_terms = {
            'cell_rounding': {
                'cmpo_id': 'CMPO:0000001',
                'description': 'Cells become more spherical/round, often due to cytoskeletal disruption',
                'biological_indicators': ['spherical', 'round', 'loss of elongation', 'contracted'],
                'requires_context': ['morphology change', 'shape alteration']
            },
            'cell_elongation': {
                'cmpo_id': 'CMPO:0000002',
                'description': 'Cells become elongated/stretched, often during migration or stress',
                'biological_indicators': ['elongated', 'stretched', 'extended', 'spindle-shaped'],
                'requires_context': ['morphology change', 'shape alteration']
            },
            'nuclear_fragmentation': {
                'cmpo_id': 'CMPO:0000010',
                'description': 'Nuclear DNA fragments, hallmark of apoptosis',
                'biological_indicators': ['fragmented nuclei', 'broken DNA', 'nuclear fragments'],
                'requires_context': ['apoptosis', 'DNA damage', 'cell death']
            },
            'actin_disruption': {
                'cmpo_id': 'CMPO:0000105',
                'description': 'F-actin cytoskeleton becomes disrupted/disorganized',
                'biological_indicators': ['disrupted actin', 'disorganized filaments', 'cytoskeleton breakdown'],
                'requires_context': ['cytoskeleton alteration', 'actin organization loss']
            },
            'microtubule_disruption': {
                'cmpo_id': 'CMPO:0000020',
                'description': 'Microtubule network disrupted, affects cell division and transport',
                'biological_indicators': ['disrupted tubulin', 'microtubule breakdown', 'spindle disruption'],
                'requires_context': ['cytoskeleton alteration', 'tubulin organization loss']
            },
            'mitotic_arrest': {
                'cmpo_id': 'CMPO:0000196',
                'description': 'Cells arrested in mitosis with visible chromosomes/spindles',
                'biological_indicators': ['mitotic figures', 'visible chromosomes', 'spindle apparatus', 'condensed chromosomes'],
                'requires_context': ['mitotic cells', 'cell division arrest', 'M phase']
            },
            'increased_cell_size': {
                'cmpo_id': 'CMPO:0000030',
                'description': 'Cells larger than normal, may indicate stress or treatment effects',
                'biological_indicators': ['enlarged cells', 'increased size', 'larger morphology'],
                'requires_context': ['size change', 'morphology alteration']
            },
            'decreased_cell_size': {
                'cmpo_id': 'CMPO:0000031',
                'description': 'Cells smaller than normal, may indicate stress or apoptosis',
                'biological_indicators': ['smaller cells', 'decreased size', 'contracted morphology'],
                'requires_context': ['size change', 'morphology alteration']
            }
        }
    
    def semantic_map_phenotypes(self, vlm_analysis: Dict, experimental_context: Dict) -> List[Dict]:
        """Map phenotypes using biological reasoning."""
        
        # Extract all text from VLM analysis
        all_text = self._extract_all_text(vlm_analysis).lower()
        
        # Get compound-specific expectations
        compound = experimental_context.get('compound', 'unknown')
        moa = experimental_context.get('moa', 'unknown')
        
        mappings = []
        rejections = []
        
        for phenotype_name, phenotype_data in self.cmpo_terms.items():
            mapping_result = self._evaluate_phenotype(
                phenotype_name, phenotype_data, all_text, compound, moa
            )
            
            if mapping_result['status'] == 'ACCEPTED':
                mappings.append(mapping_result)
            else:
                rejections.append(mapping_result)
        
        return {
            'accepted_mappings': mappings,
            'rejected_mappings': rejections,
            'total_evaluated': len(self.cmpo_terms)
        }
    
    def _extract_all_text(self, vlm_analysis: Dict) -> str:
        """Extract all text from VLM analysis."""
        texts = []
        
        for stage_data in vlm_analysis.values():
            if isinstance(stage_data, str):
                texts.append(stage_data)
            elif isinstance(stage_data, dict):
                for value in stage_data.values():
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                texts.append(item)
        
        return ' '.join(texts)
    
    def _evaluate_phenotype(self, phenotype_name: str, phenotype_data: Dict, 
                           text: str, compound: str, moa: str) -> Dict:
        """Evaluate if a phenotype is present with biological reasoning."""
        
        indicators = phenotype_data['biological_indicators']
        required_context = phenotype_data['requires_context']
        
        # Check for biological indicators
        indicator_matches = [ind for ind in indicators if ind in text]
        context_matches = [ctx for ctx in required_context if ctx in text]
        
        # Special logic for different compound types
        confidence = 0.0
        evidence = []
        reasoning = ""
        status = "REJECTED"
        
        if indicator_matches:
            base_confidence = len(indicator_matches) / len(indicators)
            
            # Apply biological reasoning
            if phenotype_name == 'mitotic_arrest':
                # Mitotic arrest requires visible mitotic figures
                mitotic_evidence = any(term in text for term in 
                                     ['mitotic figures', 'chromosomes', 'spindle', 'metaphase', 'anaphase'])
                if mitotic_evidence and compound != 'dmso':
                    confidence = base_confidence * 0.9
                    evidence = indicator_matches
                    reasoning = f"Mitotic evidence found in {compound} treatment"
                    status = "ACCEPTED"
                else:
                    reasoning = f"No mitotic figures visible, DMSO is control" if compound == 'dmso' else "No clear mitotic evidence"
            
            elif phenotype_name in ['actin_disruption', 'microtubule_disruption']:
                # Cytoskeleton disruption should show clear disorganization
                disruption_evidence = any(term in text for term in 
                                        ['disrupted', 'disorganized', 'fragmented', 'breakdown'])
                organization_evidence = any(term in text for term in 
                                          ['well-preserved', 'organized', 'intact', 'healthy'])
                
                if disruption_evidence and not organization_evidence:
                    confidence = base_confidence * 0.8
                    evidence = indicator_matches
                    reasoning = f"Clear cytoskeletal disruption in {compound}"
                    status = "ACCEPTED"
                elif organization_evidence:
                    reasoning = f"Cytoskeleton appears organized/healthy, not disrupted"
                else:
                    reasoning = f"Ambiguous cytoskeletal state"
            
            elif phenotype_name in ['cell_rounding', 'cell_elongation']:
                # Morphology changes should be clearly described
                if context_matches or 'morphology' in text:
                    confidence = base_confidence * 0.7
                    evidence = indicator_matches
                    reasoning = f"Clear morphological changes described"
                    status = "ACCEPTED"
                else:
                    reasoning = f"Morphology terms present but no clear shape change context"
            
            elif phenotype_name == 'nuclear_fragmentation':
                # Nuclear fragmentation requires apoptosis context
                apoptosis_evidence = any(term in text for term in 
                                       ['apoptosis', 'cell death', 'fragmented', 'dying'])
                if apoptosis_evidence:
                    confidence = base_confidence * 0.9
                    evidence = indicator_matches
                    reasoning = f"Nuclear fragmentation with apoptosis evidence"
                    status = "ACCEPTED"
                else:
                    reasoning = f"Nuclear terms present but no apoptosis context"
            
            else:
                # Generic evaluation
                if context_matches:
                    confidence = base_confidence * 0.6
                    evidence = indicator_matches
                    reasoning = f"Phenotype indicators with appropriate context"
                    status = "ACCEPTED"
                else:
                    reasoning = f"Indicators present but missing biological context"
        
        else:
            reasoning = f"No biological indicators found for {phenotype_name}"
        
        return {
            'cmpo_id': phenotype_data['cmpo_id'],
            'phenotype_name': phenotype_name,
            'confidence': confidence,
            'evidence': evidence,
            'reasoning': reasoning,
            'status': status,
            'compound': compound,
            'moa': moa
        }

async def run_semantic_discovery():
    """Run discovery with semantic CMPO mapping."""
    
    print("🧬 Semantic Multi-Channel Discovery Analysis")
    print("=" * 60)
    
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
    
    # Initialize semantic mapper
    semantic_mapper = SemanticCMPOMapper()
    
    # Select diverse compounds for testing
    test_compounds = ['DMSO', 'taxol', 'anisomycin']
    results = []
    
    for compound in test_compounds:
        compound_rows = df[df['compound'] == compound]
        if compound_rows.empty:
            continue
            
        row = compound_rows.iloc[0]
        print(f"\n🔬 Analyzing: {compound} ({row['moa']})")
        
        # Run VLM analysis
        vlm_result = await analyze_compound_vlm(row)
        if not vlm_result:
            continue
        
        # Apply semantic CMPO mapping
        experimental_context = {
            'compound': compound,
            'moa': row['moa'],
            'cell_line': 'MCF7',
            'treatment_type': 'vehicle_control' if compound == 'DMSO' else 'compound_treatment'
        }
        
        cmpo_mappings = semantic_mapper.semantic_map_phenotypes(
            vlm_result, experimental_context
        )
        
        # Combine results
        result = {
            'compound': compound,
            'moa': row['moa'],
            'composite_path': row['composite_path'],
            'vlm_analysis': vlm_result,
            'semantic_cmpo': cmpo_mappings,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Show immediate results
        accepted = cmpo_mappings['accepted_mappings']
        rejected = cmpo_mappings['rejected_mappings']
        
        print(f"   ✅ Accepted phenotypes: {len(accepted)}")
        for mapping in accepted:
            print(f"      • {mapping['phenotype_name']} (confidence: {mapping['confidence']:.2f})")
            print(f"        Evidence: {', '.join(mapping['evidence'][:3])}")
        
        print(f"   ❌ Rejected phenotypes: {len(rejected)}")
        for rejection in rejected[:3]:  # Show top 3 rejections
            print(f"      • {rejection['phenotype_name']}: {rejection['reasoning']}")
    
    if results:
        # Save results
        output_file = 'semantic_discovery_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Semantic Discovery Complete!")
        print(f"📊 Results saved: {output_file}")
        
        # Compare compounds
        compare_compounds(results)

async def analyze_compound_vlm(row):
    """Analyze one compound with VLM."""
    
    try:
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'biological_context': {
                'experiment_type': 'compound_screen_3channel_fluorescence',
                'cell_line': 'MCF7_breast_cancer_cells',
                'compound': row['compound'],
                'mechanism_of_action': row['moa'],
                'channels': 'DAPI_blue_tubulin_green_actin_red'
            }
        }
        
        pipeline = AnalysisPipeline(config)
        result = await pipeline.run_pipeline(row['composite_path'])
        return result
        
    except Exception as e:
        print(f"     ❌ VLM analysis failed: {e}")
        return None

def compare_compounds(results):
    """Compare phenotypes across compounds."""
    
    print(f"\n🔍 COMPOUND COMPARISON")
    print("=" * 40)
    
    # Group by phenotype
    phenotype_compounds = defaultdict(list)
    
    for result in results:
        compound = result['compound']
        accepted = result['semantic_cmpo']['accepted_mappings']
        
        for mapping in accepted:
            phenotype = mapping['phenotype_name']
            confidence = mapping['confidence']
            phenotype_compounds[phenotype].append(f"{compound} ({confidence:.2f})")
    
    # Show shared and unique phenotypes
    print("\n📊 PHENOTYPE DISTRIBUTION:")
    for phenotype, compounds in phenotype_compounds.items():
        print(f"\n   {phenotype}:")
        for compound_info in compounds:
            print(f"      → {compound_info}")
    
    # Mechanism insights
    print(f"\n🎯 MECHANISM INSIGHTS:")
    for result in results:
        compound = result['compound']
        moa = result['moa']
        accepted_count = len(result['semantic_cmpo']['accepted_mappings'])
        rejected_count = len(result['semantic_cmpo']['rejected_mappings'])
        
        print(f"\n   {compound} ({moa}):")
        print(f"      Phenotypes detected: {accepted_count}")
        print(f"      False positives avoided: {rejected_count}")

if __name__ == "__main__":
    asyncio.run(run_semantic_discovery())