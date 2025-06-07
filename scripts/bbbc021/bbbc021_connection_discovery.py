#!/usr/bin/env python3
"""
BBBC021 Connection Discovery Engine

Build a biological relationship discovery system using MCF-7 compound screen data.
Focus on discovering connections between compounds, mechanisms, and phenotypes.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import asyncio
import json
from collections import defaultdict
from datetime import datetime

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    from anton.cmpo.ontology import CMPOOntology
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

class BBBC021SemanticEncoder:
    """Extract rich semantic phenotype descriptions instead of simple classifications."""
    
    def __init__(self):
        self.phenotype_dimensions = {
            'morphology': ['shape', 'size', 'aspect_ratio', 'area', 'perimeter'],
            'nuclear': ['intensity', 'texture', 'shape', 'size', 'position'],
            'cytoskeleton': ['actin_organization', 'tubulin_organization', 'fiber_density'],
            'spatial': ['cell_density', 'neighbor_distance', 'clustering'],
            'intensity': ['overall_brightness', 'contrast', 'dynamic_range']
        }
    
    async def encode_phenotype(self, image_path: str, compound_info: Dict) -> Dict:
        """Extract rich semantic phenotype encoding from image."""
        try:
            # Use Anton pipeline for analysis
            config = {
                'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
                'channels': [0, 1, 2],  # DNA, F-actin, β-tubulin
                'biological_context': {
                    'experiment_type': 'compound_screen',
                    'cell_line': 'MCF7_breast_cancer',
                    'compound': compound_info.get('compound', 'unknown'),
                    'mechanism': compound_info.get('moa', 'unknown'),
                    'concentration': compound_info.get('concentration', 0),
                    'staining': ['DAPI_DNA', 'F-actin', 'beta-tubulin'],
                    'timepoint': '24_hours'
                }
            }
            
            pipeline = AnalysisPipeline(config)
            anton_results = await pipeline.run_pipeline(str(image_path))
            
            # Extract semantic features from VLM output
            semantic_encoding = self._extract_semantic_features(anton_results, compound_info)
            
            return semantic_encoding
            
        except Exception as e:
            print(f"❌ Failed to encode {image_path}: {e}")
            return None
    
    def _extract_semantic_features(self, anton_results: Dict, compound_info: Dict) -> Dict:
        """Extract structured semantic features from Anton VLM analysis."""
        
        # Initialize semantic encoding structure
        encoding = {
            'compound_id': compound_info.get('compound', 'unknown'),
            'mechanism_of_action': compound_info.get('moa', 'unknown'),
            'concentration': compound_info.get('concentration', 0),
            'semantic_features': {
                'morphology': {},
                'nuclear': {},
                'cytoskeleton': {},
                'spatial': {},
                'intensity': {}
            },
            'raw_descriptions': {},
            'discovery_vectors': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract descriptions from each pipeline stage
        stages = ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']
        
        for stage in stages:
            if stage in anton_results:
                stage_data = anton_results[stage]
                
                # Extract description text
                description = ""
                if 'description' in stage_data:
                    description = stage_data['description']
                elif 'segmentation_guidance' in stage_data:
                    description = stage_data['segmentation_guidance']
                elif 'population_summary' in stage_data:
                    description = stage_data['population_summary']
                
                encoding['raw_descriptions'][stage] = description
                
                # Parse semantic features from description
                semantic_features = self._parse_semantic_features(description)
                
                # Merge into main encoding
                for category, features in semantic_features.items():
                    if category in encoding['semantic_features']:
                        encoding['semantic_features'][category].update(features)
        
        # Create discovery vectors for similarity search
        encoding['discovery_vectors'] = self._create_discovery_vectors(encoding)
        
        return encoding
    
    def _parse_semantic_features(self, description: str) -> Dict:
        """Parse semantic features from VLM description text."""
        features = {
            'morphology': {},
            'nuclear': {},
            'cytoskeleton': {},
            'spatial': {},
            'intensity': {}
        }
        
        description_lower = description.lower()
        
        # Morphology features
        morphology_keywords = {
            'elongated': 'cell_elongation',
            'round': 'cell_rounding', 
            'spread': 'cell_spreading',
            'contracted': 'cell_contraction',
            'large': 'increased_size',
            'small': 'decreased_size',
            'irregular': 'irregular_shape'
        }
        
        # Nuclear features
        nuclear_keywords = {
            'fragmented': 'nuclear_fragmentation',
            'condensed': 'nuclear_condensation',
            'enlarged': 'nuclear_enlargement',
            'bright': 'increased_nuclear_intensity',
            'dim': 'decreased_nuclear_intensity',
            'irregular': 'irregular_nuclear_shape'
        }
        
        # Cytoskeleton features
        cytoskeleton_keywords = {
            'disorganized': 'cytoskeleton_disruption',
            'bundled': 'cytoskeleton_bundling',
            'dense': 'increased_fiber_density',
            'sparse': 'decreased_fiber_density',
            'radial': 'radial_organization',
            'parallel': 'parallel_organization'
        }
        
        # Extract features
        for keywords_dict, feature_category in [
            (morphology_keywords, 'morphology'),
            (nuclear_keywords, 'nuclear'),
            (cytoskeleton_keywords, 'cytoskeleton')
        ]:
            for keyword, feature_name in keywords_dict.items():
                if keyword in description_lower:
                    features[feature_category][feature_name] = 1.0
        
        return features
    
    def _create_discovery_vectors(self, encoding: Dict) -> Dict:
        """Create vectors for similarity-based discovery."""
        vectors = {
            'morphology_vector': [],
            'mechanism_vector': [],
            'phenotype_vector': [],
            'combined_vector': []
        }
        
        # Morphology vector (for finding similar cellular changes)
        morphology_features = encoding['semantic_features']['morphology']
        morphology_vector = [morphology_features.get(feature, 0) for feature in [
            'cell_elongation', 'cell_rounding', 'cell_spreading', 'cell_contraction',
            'increased_size', 'decreased_size', 'irregular_shape'
        ]]
        vectors['morphology_vector'] = morphology_vector
        
        # Mechanism vector (for finding similar mechanisms)
        moa = encoding.get('mechanism_of_action', 'unknown')
        known_mechanisms = [
            'Actin disruptors', 'Aurora kinase inhibitors', 'Eg5 inhibitors',
            'Microtubule destabilizers', 'Microtubule stabilizers', 'Epithelial'
        ]
        mechanism_vector = [1.0 if moa == mechanism else 0.0 for mechanism in known_mechanisms]
        vectors['mechanism_vector'] = mechanism_vector
        
        # Combined phenotype vector
        all_features = []
        for category in ['morphology', 'nuclear', 'cytoskeleton']:
            category_features = encoding['semantic_features'][category]
            all_features.extend(category_features.values())
        vectors['phenotype_vector'] = all_features
        
        # Combined discovery vector
        vectors['combined_vector'] = morphology_vector + mechanism_vector + all_features
        
        return vectors

class BBBC021ConnectionDiscovery:
    """Discover biological connections across compounds and phenotypes."""
    
    def __init__(self):
        self.encoder = BBBC021SemanticEncoder()
        self.phenotype_database = {}
        self.similarity_index = {}
        self.connection_graph = defaultdict(list)
    
    async def analyze_real_bbbc021(self, metadata_file: str = "data/bbbc021/analysis_subset.csv"):
        """Analyze real BBBC021 images for connection discovery."""
        
        print(f"🚀 Starting REAL BBBC021 Connection Discovery Analysis")
        
        # Load metadata
        if not Path(metadata_file).exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return
        
        metadata = pd.read_csv(metadata_file)
        print(f"📋 Loaded real BBBC021 metadata: {len(metadata)} images")
        print(f"🧬 Compounds: {list(metadata['compound'].unique())}")
        print(f"🎯 Mechanisms: {list(metadata['moa'].unique())}")
        
        # Analyze all available images
        results = []
        for _, row in metadata.iterrows():
            if Path(row['image_path']).exists():
                print(f"🔬 Analyzing: {row['compound']} ({row['moa']}) - {Path(row['image_path']).name}")
                
                # Real Anton analysis
                result = await self._analyze_real_image(row)
                if result:
                    results.append(result)
                    
                    # Add to database
                    compound = row['compound']
                    if compound not in self.phenotype_database:
                        self.phenotype_database[compound] = []
                    self.phenotype_database[compound].append(result)
            else:
                print(f"⚠️ Image not found: {row['image_path']}")
        
        print(f"✅ Analyzed {len(results)} real images")
        
        # Build connection graph
        self._build_connection_graph()
        
        # Demonstrate real discoveries
        self._demonstrate_real_discoveries()
        
        return results
    
    async def _analyze_real_image(self, row: pd.Series) -> Dict:
        """Analyze real BBBC021 image using Anton pipeline."""
        try:
            image_path = row['image_path']
            
            # Configure Anton for BBBC021
            config = {
                'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
                'channels': [0, 1, 2],  # Would need multi-channel support
                'biological_context': {
                    'experiment_type': 'compound_screen',
                    'cell_line': 'MCF7_breast_cancer',
                    'compound': row['compound'],
                    'mechanism': row['moa'],
                    'concentration': row['concentration'],
                    'staining': ['DAPI_DNA', 'beta_tubulin', 'F_actin'],
                    'timepoint': '24_hours'
                }
            }
            
            pipeline = AnalysisPipeline(config)
            anton_results = await pipeline.run_pipeline(str(image_path))
            
            # Extract semantic encoding
            semantic_encoding = self._extract_real_semantic_features(anton_results, row)
            
            return semantic_encoding
            
        except Exception as e:
            print(f"❌ Failed to analyze {row['image_path']}: {e}")
            return None
    
    def _extract_real_semantic_features(self, anton_results: Dict, row: pd.Series) -> Dict:
        """Extract semantic features from real Anton VLM analysis."""
        
        encoding = {
            'compound_id': row['compound'],
            'mechanism_of_action': row['moa'],
            'concentration': row['concentration'],
            'image_path': row['image_path'],
            'well_id': row['well'],
            'semantic_features': {
                'morphology': {},
                'nuclear': {},
                'cytoskeleton': {},
                'spatial': {},
                'intensity': {}
            },
            'raw_vlm_descriptions': {},
            'discovery_vectors': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract VLM descriptions from each stage
        stages = ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']
        
        all_descriptions = []
        for stage in stages:
            if stage in anton_results:
                stage_data = anton_results[stage]
                
                # Extract description text
                description = ""
                if 'description' in stage_data:
                    description = stage_data['description']
                elif 'segmentation_guidance' in stage_data:
                    description = stage_data['segmentation_guidance']
                elif 'population_summary' in stage_data:
                    description = stage_data['population_summary']
                
                encoding['raw_vlm_descriptions'][stage] = description
                all_descriptions.append(description)
                
                # Parse semantic features from real VLM output
                semantic_features = self._parse_real_vlm_features(description)
                
                # Merge features
                for category, features in semantic_features.items():
                    if category in encoding['semantic_features']:
                        encoding['semantic_features'][category].update(features)
        
        # Create discovery vectors from real features
        encoding['discovery_vectors'] = self._create_real_discovery_vectors(encoding, all_descriptions)
        
        return encoding
    
    def _parse_real_vlm_features(self, description: str) -> Dict:
        """Parse semantic features from real VLM description."""
        features = {
            'morphology': {},
            'nuclear': {},
            'cytoskeleton': {},
            'spatial': {},
            'intensity': {}
        }
        
        if not description:
            return features
        
        description_lower = description.lower()
        
        # Enhanced keyword extraction for real VLM output
        feature_patterns = {
            'morphology': {
                'elongated|stretched|extended': 'cell_elongation',
                'round|circular|spherical': 'cell_rounding',
                'spread|flattened': 'cell_spreading',
                'contracted|shrunken': 'cell_contraction',
                'large|enlarged|bigger': 'increased_size',
                'small|tiny|reduced': 'decreased_size',
                'irregular|distorted': 'irregular_shape',
                'multinucleated|multi-nucleated': 'multinucleation'
            },
            'nuclear': {
                'fragmented|broken': 'nuclear_fragmentation',
                'condensed|compact': 'nuclear_condensation',
                'enlarged|swollen': 'nuclear_enlargement',
                'bright|intense': 'increased_nuclear_intensity',
                'dim|faint': 'decreased_nuclear_intensity',
                'irregular|distorted': 'irregular_nuclear_shape',
                'multiple nuclei|binucleated': 'multiple_nuclei'
            },
            'cytoskeleton': {
                'disorganized|disrupted|chaotic': 'cytoskeleton_disruption',
                'bundled|clustered': 'cytoskeleton_bundling',
                'dense|thick': 'increased_fiber_density',
                'sparse|thin': 'decreased_fiber_density',
                'radial|star-like': 'radial_organization',
                'parallel|aligned': 'parallel_organization',
                'microtubule|tubulin': 'microtubule_related',
                'actin|filament': 'actin_related'
            },
            'spatial': {
                'clustered|aggregated': 'cell_clustering',
                'scattered|dispersed': 'cell_scattering',
                'isolated|separated': 'cell_isolation',
                'contact|touching': 'cell_contact'
            }
        }
        
        # Extract features using pattern matching
        for category, patterns in feature_patterns.items():
            for pattern, feature_name in patterns.items():
                if any(keyword in description_lower for keyword in pattern.split('|')):
                    features[category][feature_name] = 1.0
        
        return features
    
    def _create_real_discovery_vectors(self, encoding: Dict, descriptions: List[str]) -> Dict:
        """Create discovery vectors from real VLM analysis."""
        
        # Combine all descriptions for analysis
        combined_description = ' '.join(descriptions).lower()
        
        # Morphology vector (7 dimensions)
        morphology_features = encoding['semantic_features']['morphology']
        morphology_vector = [
            morphology_features.get('cell_elongation', 0),
            morphology_features.get('cell_rounding', 0),
            morphology_features.get('cell_spreading', 0),
            morphology_features.get('increased_size', 0),
            morphology_features.get('decreased_size', 0),
            morphology_features.get('irregular_shape', 0),
            morphology_features.get('multinucleation', 0)
        ]
        
        # Cytoskeleton vector (6 dimensions)
        cytoskeleton_features = encoding['semantic_features']['cytoskeleton']
        cytoskeleton_vector = [
            cytoskeleton_features.get('cytoskeleton_disruption', 0),
            cytoskeleton_features.get('increased_fiber_density', 0),
            cytoskeleton_features.get('decreased_fiber_density', 0),
            cytoskeleton_features.get('radial_organization', 0),
            cytoskeleton_features.get('microtubule_related', 0),
            cytoskeleton_features.get('actin_related', 0)
        ]
        
        # Mechanism vector (known BBBC021 mechanisms)
        moa = encoding.get('mechanism_of_action', 'unknown')
        bbbc021_mechanisms = [
            'Microtubule destabilizers', 'Microtubule stabilizers', 'Actin disruptors',
            'Aurora kinase inhibitors', 'Eg5 inhibitors', 'Control'
        ]
        mechanism_vector = [1.0 if moa == mechanism else 0.0 for mechanism in bbbc021_mechanisms]
        
        return {
            'morphology_vector': morphology_vector,
            'cytoskeleton_vector': cytoskeleton_vector,
            'mechanism_vector': mechanism_vector,
            'combined_vector': morphology_vector + cytoskeleton_vector + mechanism_vector,
            'description_length': len(combined_description),
            'feature_richness': sum(len(features) for features in encoding['semantic_features'].values())
        }
    
    def _demonstrate_real_discoveries(self):
        """Demonstrate discoveries from real BBBC021 analysis."""
        
        print("\n🔍 REAL BIOLOGICAL DISCOVERIES")
        print("=" * 60)
        
        # Discovery 1: Morphological clustering
        print("\n1. 🧬 MORPHOLOGICAL PHENOTYPE CLUSTERING")
        morphology_clusters = defaultdict(list)
        
        for compound, results in self.phenotype_database.items():
            for result in results:
                morphology_features = result['semantic_features']['morphology']
                dominant_features = [feature for feature, value in morphology_features.items() if value > 0]
                if dominant_features:
                    cluster_key = ', '.join(sorted(dominant_features))
                    morphology_clusters[cluster_key].append(f"{compound} ({result['mechanism_of_action']})")
        
        for phenotype, compounds in morphology_clusters.items():
            if len(compounds) > 1:
                print(f"   📊 {phenotype}:")
                for compound in compounds:
                    print(f"      → {compound}")
        
        # Discovery 2: Mechanism validation
        print(f"\n2. 🎯 MECHANISM-PHENOTYPE VALIDATION")
        mechanism_phenotypes = defaultdict(set)
        
        for compound, results in self.phenotype_database.items():
            for result in results:
                moa = result['mechanism_of_action']
                all_features = []
                for category_features in result['semantic_features'].values():
                    all_features.extend(category_features.keys())
                mechanism_phenotypes[moa].update(all_features)
        
        for moa, phenotypes in mechanism_phenotypes.items():
            if phenotypes:
                print(f"   🔬 {moa}:")
                print(f"      Phenotypes: {', '.join(sorted(phenotypes))}")
        
        # Discovery 3: Unexpected connections
        print(f"\n3. 🔍 UNEXPECTED CONNECTIONS")
        for compound, connections in self.connection_graph.items():
            if connections:
                print(f"   {compound}:")
                for conn in connections:
                    other_compound = conn['connected_compound']
                    similarity = conn['similarity']
                    
                    # Get mechanisms
                    compound_moa = self.phenotype_database[compound][0]['mechanism_of_action']
                    other_moa = self.phenotype_database[other_compound][0]['mechanism_of_action']
                    
                    if compound_moa != other_moa:
                        print(f"      🔗 Unexpected similarity to {other_compound}")
                        print(f"         {compound_moa} ↔ {other_moa} (similarity: {similarity:.2f})")
        
        print(f"\n✅ Real biological discovery analysis complete!")
    
    def _select_diverse_compounds(self, metadata: pd.DataFrame, max_compounds: int) -> List[Dict]:
        """Select diverse compounds representing different mechanisms."""
        
        # Group by mechanism of action
        moa_groups = metadata.groupby('moa')
        
        selected = []
        mechanisms_covered = set()
        
        for moa, group in moa_groups:
            if len(selected) >= max_compounds:
                break
            
            if moa not in mechanisms_covered:
                # Select one compound from this mechanism
                compound_name = group['compound'].iloc[0]
                selected.append({
                    'compound': compound_name,
                    'moa': moa,
                    'sample_count': len(group)
                })
                mechanisms_covered.add(moa)
        
        return selected
    
    async def _analyze_compound(self, compound_info: Dict, metadata: pd.DataFrame, max_images: int) -> List[Dict]:
        """Analyze representative images for one compound."""
        
        compound_name = compound_info['compound']
        print(f"🔬 Analyzing compound: {compound_name} (MOA: {compound_info['moa']})")
        
        # Find images for this compound
        compound_rows = metadata[metadata['compound'] == compound_name].head(max_images)
        
        results = []
        for _, row in compound_rows.iterrows():
            # For demo, create mock analysis (would use real images in practice)
            mock_result = await self._mock_analyze_image(row, compound_info)
            if mock_result:
                results.append(mock_result)
        
        return results
    
    async def _mock_analyze_image(self, row: pd.Series, compound_info: Dict) -> Dict:
        """Mock analysis for demonstration (would use real Anton pipeline)."""
        
        # Mock semantic encoding based on known mechanism
        moa = compound_info['moa']
        
        # Create realistic mock features based on mechanism
        mock_features = self._generate_mock_features(moa)
        
        encoding = {
            'compound_id': compound_info['compound'],
            'mechanism_of_action': moa,
            'concentration': row.get('concentration', 0),
            'semantic_features': mock_features,
            'discovery_vectors': self._create_mock_vectors(mock_features, moa),
            'image_id': row.get('ImageNumber', 'unknown'),
            'well_id': row.get('well', 'unknown')
        }
        
        return encoding
    
    def _generate_mock_features(self, moa: str) -> Dict:
        """Generate realistic mock features based on mechanism of action."""
        
        # Define expected phenotypes for each mechanism
        moa_signatures = {
            'Actin disruptors': {
                'morphology': {'cell_rounding': 1.0, 'decreased_size': 0.8},
                'cytoskeleton': {'cytoskeleton_disruption': 1.0},
                'nuclear': {}
            },
            'Aurora kinase inhibitors': {
                'morphology': {'increased_size': 0.9, 'irregular_shape': 0.7},
                'nuclear': {'nuclear_fragmentation': 0.8, 'increased_nuclear_intensity': 0.6},
                'cytoskeleton': {}
            },
            'Microtubule destabilizers': {
                'morphology': {'cell_rounding': 0.9, 'cell_contraction': 0.7},
                'cytoskeleton': {'cytoskeleton_disruption': 1.0, 'decreased_fiber_density': 0.8},
                'nuclear': {}
            },
            'Eg5 inhibitors': {
                'morphology': {'increased_size': 0.8},
                'nuclear': {'nuclear_condensation': 0.9},
                'cytoskeleton': {'parallel_organization': 0.7}
            }
        }
        
        return moa_signatures.get(moa, {
            'morphology': {},
            'nuclear': {},
            'cytoskeleton': {}
        })
    
    def _create_mock_vectors(self, features: Dict, moa: str) -> Dict:
        """Create mock discovery vectors."""
        
        # Morphology vector
        morphology_features = features.get('morphology', {})
        morphology_vector = [
            morphology_features.get('cell_elongation', 0),
            morphology_features.get('cell_rounding', 0),
            morphology_features.get('cell_spreading', 0),
            morphology_features.get('increased_size', 0),
            morphology_features.get('decreased_size', 0)
        ]
        
        # Mechanism vector
        known_mechanisms = ['Actin disruptors', 'Aurora kinase inhibitors', 'Eg5 inhibitors', 'Microtubule destabilizers']
        mechanism_vector = [1.0 if moa == mechanism else 0.0 for mechanism in known_mechanisms]
        
        return {
            'morphology_vector': morphology_vector,
            'mechanism_vector': mechanism_vector,
            'combined_vector': morphology_vector + mechanism_vector
        }
    
    def _build_connection_graph(self):
        """Build connection graph for similarity-based discovery."""
        
        print("🔗 Building connection graph...")
        
        # Calculate similarities between all compound pairs
        compounds = list(self.phenotype_database.keys())
        
        for i, compound1 in enumerate(compounds):
            for j, compound2 in enumerate(compounds[i+1:], i+1):
                similarity = self._calculate_compound_similarity(compound1, compound2)
                
                if similarity > 0.5:  # Threshold for significant similarity
                    self.connection_graph[compound1].append({
                        'connected_compound': compound2,
                        'similarity': similarity,
                        'connection_type': 'phenotypic_similarity'
                    })
                    
                    self.connection_graph[compound2].append({
                        'connected_compound': compound1,
                        'similarity': similarity,
                        'connection_type': 'phenotypic_similarity'
                    })
        
        print(f"✅ Built connections for {len(self.connection_graph)} compounds")
    
    def _calculate_compound_similarity(self, compound1: str, compound2: str) -> float:
        """Calculate phenotypic similarity between two compounds."""
        
        results1 = self.phenotype_database.get(compound1, [])
        results2 = self.phenotype_database.get(compound2, [])
        
        if not results1 or not results2:
            return 0.0
        
        # Calculate average vector similarity
        similarities = []
        
        for result1 in results1:
            for result2 in results2:
                vector1 = result1['discovery_vectors']['combined_vector']
                vector2 = result2['discovery_vectors']['combined_vector']
                
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(vector1, vector2))
                magnitude1 = sum(a ** 2 for a in vector1) ** 0.5
                magnitude2 = sum(a ** 2 for a in vector2) ** 0.5
                
                if magnitude1 > 0 and magnitude2 > 0:
                    similarity = dot_product / (magnitude1 * magnitude2)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _demonstrate_discoveries(self):
        """Demonstrate the types of discoveries possible."""
        
        print("\n🔍 DEMONSTRATION: Connection Discovery Results")
        print("=" * 60)
        
        # Discovery 1: Similar phenotypes across different mechanisms
        print("\n1. 🎯 DISCOVERY: Compounds with Similar Phenotypes")
        for compound, connections in self.connection_graph.items():
            if connections:
                print(f"\n   {compound}:")
                for conn in connections[:2]:  # Top 2 connections
                    print(f"     → Similar to: {conn['connected_compound']} (similarity: {conn['similarity']:.2f})")
        
        # Discovery 2: Mechanism clustering
        print(f"\n2. 🧬 DISCOVERY: Mechanism Clustering")
        mechanism_groups = defaultdict(list)
        for compound, results in self.phenotype_database.items():
            if results:
                moa = results[0]['mechanism_of_action']
                mechanism_groups[moa].append(compound)
        
        for moa, compounds in mechanism_groups.items():
            print(f"   {moa}: {compounds}")
        
        # Discovery 3: Novel patterns
        print(f"\n3. 🔬 DISCOVERY: Potential Novel Patterns")
        isolated_compounds = [compound for compound, connections in self.connection_graph.items() if not connections]
        if isolated_compounds:
            print(f"   Compounds with unique phenotypes: {isolated_compounds}")
        else:
            print("   All compounds show phenotypic similarities to others")
        
        print(f"\n✅ Connection discovery demonstration complete!")

async def main():
    """Main workflow for REAL BBBC021 connection discovery."""
    
    print("🧬 REAL BBBC021 Connection Discovery Engine")
    print("=" * 50)
    
    # Check API availability
    if not os.getenv('GOOGLE_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️ No API key found - analysis will use mock responses")
        return
    
    # Initialize discovery engine
    discovery_engine = BBBC021ConnectionDiscovery()
    
    # Check if BBBC021 data is available
    metadata_file = "data/bbbc021/analysis_subset.csv"
    if not Path(metadata_file).exists():
        print(f"❌ BBBC021 data not found at {metadata_file}")
        print("💡 Please run the downloader first: python bbbc021_downloader.py")
        return
    
    # Run REAL connection discovery analysis
    results = await discovery_engine.analyze_real_bbbc021(metadata_file)
    
    if results:
        # Save results
        with open('bbbc021_real_discovery_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 REAL analysis complete!")
        print(f"🔬 Analyzed {len(results)} real BBBC021 images")
        print(f"🧬 Discovered connections across {len(discovery_engine.phenotype_database)} compounds")
        print("📊 Results saved to: bbbc021_real_discovery_results.json")
        
        # Summary statistics
        print(f"\n📈 DISCOVERY STATISTICS:")
        total_features = 0
        for compound, compound_results in discovery_engine.phenotype_database.items():
            for result in compound_results:
                total_features += result['discovery_vectors']['feature_richness']
        
        print(f"   • Total semantic features extracted: {total_features}")
        print(f"   • Average features per image: {total_features / len(results):.1f}")
        print(f"   • Compounds with connections: {len(discovery_engine.connection_graph)}")
    else:
        print("❌ No results obtained from analysis")

if __name__ == "__main__":
    asyncio.run(main())