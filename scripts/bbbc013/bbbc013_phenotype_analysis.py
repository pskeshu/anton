#!/usr/bin/env python3
"""
BBBC013 Comprehensive Phenotype Analysis

Analyzes all BBBC013 images by experimental condition and creates CMPO phenotype networks
to visualize how Wortmannin and LY294002 treatments affect cellular phenotypes.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for enhanced visualizations
# import networkx as nx  # Optional for network analysis
from typing import Dict, List, Tuple, Any
import asyncio
from datetime import datetime

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    from anton.cmpo.ontology import CMPOOntology
    from anton.cmpo.mapping import map_to_cmpo
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")
    sys.exit(1)

class BBBC013ExperimentalDesign:
    """Parse and organize BBBC013 experimental conditions."""
    
    def __init__(self, platemap_path: str = "data/bbbc013/platemap.txt"):
        self.platemap_path = platemap_path
        self.conditions = {}
        self.dose_responses = {}
        self._parse_platemap()
    
    def _parse_platemap(self):
        """Parse the BBBC013 platemap to extract experimental conditions."""
        try:
            with open(self.platemap_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header line, read dose concentrations
            doses = [float(line.strip()) for line in lines[1:] if line.strip()]
            
            # BBBC013 experimental design:
            # Rows A-D: Wortmannin (4 replicates)
            # Rows E-H: LY294002 (4 replicates)  
            # Columns 1-2, 12: Controls (0 concentration)
            # Columns 3-11: 9-point dose curve
            
            rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            cols = list(range(1, 13))  # 1-12
            
            well_idx = 0
            for row in rows:
                for col in cols:
                    well_id = f"{row}{col:02d}"
                    
                    # Determine treatment
                    if row in ['A', 'B', 'C', 'D']:
                        treatment = "Wortmannin"
                        replicate = ord(row) - ord('A') + 1
                    else:  # E, F, G, H
                        treatment = "LY294002"
                        replicate = ord(row) - ord('E') + 1
                    
                    # Determine concentration
                    if col in [1, 2, 12]:
                        concentration = 0.0  # Control
                        condition_type = "Control"
                    else:
                        dose_idx = col - 3  # Columns 3-11 map to doses 0-8
                        if dose_idx < len(doses):
                            concentration = doses[dose_idx]
                            condition_type = "Treatment"
                        else:
                            concentration = 0.0
                            condition_type = "Control"
                    
                    self.conditions[well_id] = {
                        'treatment': treatment,
                        'concentration': concentration,
                        'replicate': replicate,
                        'condition_type': condition_type,
                        'row': row,
                        'column': col
                    }
                    
                    well_idx += 1
            
            # Group by treatment and concentration for dose-response analysis
            for well_id, condition in self.conditions.items():
                treatment = condition['treatment']
                conc = condition['concentration']
                
                if treatment not in self.dose_responses:
                    self.dose_responses[treatment] = defaultdict(list)
                
                self.dose_responses[treatment][conc].append(well_id)
            
            print(f"✅ Parsed {len(self.conditions)} experimental conditions")
            print(f"📊 Treatments: {list(self.dose_responses.keys())}")
            
        except Exception as e:
            print(f"❌ Failed to parse platemap: {e}")
            raise
    
    def get_condition_summary(self) -> pd.DataFrame:
        """Get summary of experimental conditions."""
        data = []
        for well_id, condition in self.conditions.items():
            data.append({
                'well_id': well_id,
                'treatment': condition['treatment'],
                'concentration': condition['concentration'],
                'condition_type': condition['condition_type'],
                'replicate': condition['replicate']
            })
        
        return pd.DataFrame(data)
    
    def get_controls(self) -> List[str]:
        """Get list of control wells (0 concentration)."""
        return [well_id for well_id, condition in self.conditions.items() 
                if condition['concentration'] == 0.0]
    
    def get_treatment_wells(self, treatment: str, exclude_controls: bool = True) -> List[str]:
        """Get wells for specific treatment."""
        wells = [well_id for well_id, condition in self.conditions.items() 
                if condition['treatment'] == treatment]
        
        if exclude_controls:
            wells = [well_id for well_id in wells 
                    if self.conditions[well_id]['concentration'] > 0.0]
        
        return wells

class BBBC013PhenotypeAnalyzer:
    """Analyze BBBC013 images and extract condition-specific CMPO phenotypes."""
    
    def __init__(self, images_dir: str = "data/bbbc013/BBBC013_v1_images_bmp"):
        self.images_dir = Path(images_dir)
        self.experimental_design = BBBC013ExperimentalDesign()
        self.results = {}
        self.cmpo_networks = {}
        
        # Initialize Anton pipeline
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'channels': [0],  # Single channel for now
            'biological_context': {
                'experiment_type': 'protein_translocation',
                'cell_line': 'U2OS_osteosarcoma',
                'protein': 'FKHR-GFP',
                'drugs': ['Wortmannin', 'LY294002'],
                'readout': 'nuclear_vs_cytoplasmic_localization'
            }
        }
        self.pipeline = AnalysisPipeline(config)
        self.cmpo = CMPOOntology()
    
    def get_image_path(self, well_id: str, channel: int = 1) -> Path:
        """Get path to image file for given well and channel."""
        # Convert well format: A01 -> Channel1-01-A-01.BMP
        row = well_id[0]
        col = int(well_id[1:])
        filename = f"Channel{channel}-{col:02d}-{row}-{col:02d}.BMP"
        return self.images_dir / filename
    
    async def analyze_well(self, well_id: str) -> Dict[str, Any]:
        """Analyze a single well and extract CMPO phenotypes."""
        try:
            # Get experimental condition
            condition = self.experimental_design.conditions[well_id]
            
            # Get image path
            image_path = self.get_image_path(well_id)
            if not image_path.exists():
                print(f"⚠️ Image not found: {image_path}")
                return None
            
            print(f"🔬 Analyzing {well_id}: {condition['treatment']} {condition['concentration']}µM")
            
            # Run Anton pipeline
            results = await self.pipeline.run_pipeline(str(image_path))
            
            # Extract CMPO terms from all stages
            cmpo_terms = []
            
            for stage_name in ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']:
                if stage_name in results and results[stage_name]:
                    stage_data = results[stage_name]
                    
                    # Extract text for CMPO mapping
                    stage_text = ""
                    if 'description' in stage_data:
                        stage_text = stage_data['description']
                    elif 'segmentation_guidance' in stage_data:
                        stage_text = stage_data['segmentation_guidance']
                    elif 'population_summary' in stage_data:
                        stage_text = stage_data['population_summary']
                    
                    if stage_text and len(stage_text) > 50:
                        # Map to CMPO
                        mapped_terms = map_to_cmpo(stage_text, self.cmpo, context='protein_localization')
                        
                        for term in mapped_terms:
                            term['stage'] = stage_name
                            term['well_id'] = well_id
                            term['condition'] = condition
                            cmpo_terms.append(term)
            
            return {
                'well_id': well_id,
                'condition': condition,
                'anton_results': results,
                'cmpo_terms': cmpo_terms,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Failed to analyze {well_id}: {e}")
            return None
    
    async def analyze_condition_subset(self, max_wells_per_condition: int = 3):
        """Analyze a representative subset of wells for each condition."""
        print("🚀 Starting BBBC013 phenotype analysis...")
        
        # Get representative wells for each condition
        wells_to_analyze = []
        
        # Controls (mix from both treatments)
        controls = self.experimental_design.get_controls()
        wells_to_analyze.extend(controls[:max_wells_per_condition])
        
        # Wortmannin treatments (various concentrations)
        wortmannin_wells = self.experimental_design.get_treatment_wells('Wortmannin')
        wells_to_analyze.extend(wortmannin_wells[:max_wells_per_condition])
        
        # LY294002 treatments (various concentrations)
        ly294002_wells = self.experimental_design.get_treatment_wells('LY294002')
        wells_to_analyze.extend(ly294002_wells[:max_wells_per_condition])
        
        print(f"📊 Analyzing {len(wells_to_analyze)} wells: {wells_to_analyze}")
        
        # Analyze wells
        results = []
        for well_id in wells_to_analyze:
            result = await self.analyze_well(well_id)
            if result:
                results.append(result)
                self.results[well_id] = result
        
        print(f"✅ Completed analysis of {len(results)} wells")
        return results
    
    def aggregate_cmpo_by_condition(self) -> Dict[str, Dict]:
        """Aggregate CMPO terms by experimental condition."""
        condition_cmpo = defaultdict(lambda: defaultdict(list))
        
        for well_id, result in self.results.items():
            condition = result['condition']
            condition_key = f"{condition['treatment']}_{condition['concentration']}uM"
            
            for term in result['cmpo_terms']:
                term_name = term.get('term_name', 'Unknown')
                confidence = term.get('confidence', 0)
                
                condition_cmpo[condition_key][term_name].append(confidence)
        
        # Calculate statistics for each term in each condition
        aggregated = {}
        for condition_key, terms in condition_cmpo.items():
            aggregated[condition_key] = {}
            for term_name, confidences in terms.items():
                aggregated[condition_key][term_name] = {
                    'mean_confidence': np.mean(confidences),
                    'max_confidence': np.max(confidences),
                    'frequency': len(confidences),
                    'confidences': confidences
                }
        
        return aggregated
    
    def create_cmpo_network_data(self, condition_cmpo: Dict) -> Dict:
        """Create network data structure of CMPO terms and their relationships."""
        network_data = {
            'nodes': [],
            'edges': [],
            'conditions': list(condition_cmpo.keys()),
            'terms': set()
        }
        
        # Collect all unique terms
        for condition, terms in condition_cmpo.items():
            for term_name in terms.keys():
                network_data['terms'].add(term_name)
        
        network_data['terms'] = list(network_data['terms'])
        
        # Create condition-term relationships
        for condition, terms in condition_cmpo.items():
            for term_name, stats in terms.items():
                network_data['edges'].append({
                    'condition': condition,
                    'term': term_name,
                    'confidence': stats['mean_confidence'],
                    'frequency': stats['frequency']
                })
        
        return network_data
    
    def visualize_cmpo_analysis(self, condition_cmpo: Dict):
        """Create basic visualizations of CMPO phenotype analysis."""
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Condition comparison matrix
        conditions = list(condition_cmpo.keys())
        all_terms = list(set().union(*[terms.keys() for terms in condition_cmpo.values()]))
        
        # Create confidence matrix
        matrix_data = np.zeros((len(conditions), len(all_terms)))
        for i, condition in enumerate(conditions):
            for j, term in enumerate(all_terms):
                if term in condition_cmpo[condition]:
                    matrix_data[i, j] = condition_cmpo[condition][term]['mean_confidence']
        
        im = ax1.imshow(matrix_data, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(all_terms)))
        ax1.set_xticklabels([term[:20] + '...' if len(term) > 20 else term 
                           for term in all_terms], rotation=45, ha='right')
        ax1.set_yticks(range(len(conditions)))
        ax1.set_yticklabels(conditions)
        ax1.set_title("CMPO Term Confidence by Condition", fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Confidence')
        
        # 2. Term frequency analysis
        term_frequencies = Counter()
        for condition, terms in condition_cmpo.items():
            for term_name, stats in terms.items():
                term_frequencies[term_name] += stats['frequency']
        
        if term_frequencies:
            most_common = term_frequencies.most_common(min(10, len(term_frequencies)))
            terms, freqs = zip(*most_common)
            
            y_pos = range(len(terms))
            ax2.barh(y_pos, freqs, color='steelblue')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([term[:30] + '...' if len(term) > 30 else term for term in terms])
            ax2.set_xlabel("Frequency")
            ax2.set_title("Most Common CMPO Terms", fontweight='bold')
        
        # 3. Condition-specific term analysis
        control_terms = set()
        treatment_terms = set()
        
        for condition, terms in condition_cmpo.items():
            if '0.0uM' in condition:  # Control
                control_terms.update(terms.keys())
            else:  # Treatment
                treatment_terms.update(terms.keys())
        
        unique_control = control_terms - treatment_terms
        unique_treatment = treatment_terms - control_terms
        shared_terms = control_terms & treatment_terms
        
        categories = ['Control Only', 'Treatment Only', 'Shared']
        counts = [len(unique_control), len(unique_treatment), len(shared_terms)]
        colors = ['lightcoral', 'lightgreen', 'lightblue']
        
        if sum(counts) > 0:
            ax3.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title("CMPO Term Distribution", fontweight='bold')
        
        # 4. Confidence distribution by condition type
        control_confidences = []
        treatment_confidences = []
        
        for condition, terms in condition_cmpo.items():
            for term_name, stats in terms.items():
                if '0.0uM' in condition:
                    control_confidences.extend(stats['confidences'])
                else:
                    treatment_confidences.extend(stats['confidences'])
        
        if control_confidences or treatment_confidences:
            ax4.hist([control_confidences, treatment_confidences], 
                    bins=20, alpha=0.7, label=['Control', 'Treatment'], 
                    color=['lightcoral', 'lightgreen'])
            ax4.set_xlabel("CMPO Confidence Score")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Confidence Score Distribution", fontweight='bold')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('bbbc013_cmpo_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved as 'bbbc013_cmpo_analysis.png'")
        
        # Create network data for future use
        network_data = self.create_cmpo_network_data(condition_cmpo)
        return network_data
    
    def save_results(self, filename: str = 'bbbc013_phenotype_results.json'):
        """Save analysis results to file."""
        # Prepare results for JSON serialization
        serializable_results = {}
        for well_id, result in self.results.items():
            serializable_results[well_id] = {
                'condition': result['condition'],
                'cmpo_terms': result['cmpo_terms'],
                'analysis_timestamp': result['analysis_timestamp']
                # Skip anton_results as they may contain non-serializable objects
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to {filename}")

async def main():
    """Main analysis workflow."""
    print("🧬 BBBC013 Comprehensive Phenotype Analysis")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️ No API key found - analysis will use mock responses")
    
    # Initialize analyzer
    analyzer = BBBC013PhenotypeAnalyzer()
    
    # Show experimental design summary
    summary_df = analyzer.experimental_design.get_condition_summary()
    print("\n📊 Experimental Design Summary:")
    print(summary_df.groupby(['treatment', 'condition_type']).size().unstack(fill_value=0))
    
    # Analyze subset of wells
    results = await analyzer.analyze_condition_subset(max_wells_per_condition=2)
    
    if not results:
        print("❌ No results obtained")
        return
    
    # Aggregate CMPO terms by condition
    print("\n🧬 Aggregating CMPO phenotypes by condition...")
    condition_cmpo = analyzer.aggregate_cmpo_by_condition()
    
    # Print summary
    print("\n📈 CMPO Analysis Summary:")
    for condition, terms in condition_cmpo.items():
        print(f"\n{condition}:")
        top_terms = sorted(terms.items(), key=lambda x: x[1]['mean_confidence'], reverse=True)[:3]
        for term_name, stats in top_terms:
            print(f"  • {term_name}: {stats['mean_confidence']:.2f} confidence")
    
    # Create visualizations
    print("\n📊 Creating CMPO analysis visualizations...")
    network_data = analyzer.visualize_cmpo_analysis(condition_cmpo)
    
    # Save results
    analyzer.save_results()
    
    print("\n✅ Analysis complete!")
    print(f"📋 Analyzed {len(results)} wells")
    print(f"🧬 Found {len(set().union(*[terms.keys() for terms in condition_cmpo.values()]))} unique CMPO terms")
    print("📊 Visualization saved as 'bbbc013_cmpo_analysis.png'")

if __name__ == "__main__":
    asyncio.run(main())