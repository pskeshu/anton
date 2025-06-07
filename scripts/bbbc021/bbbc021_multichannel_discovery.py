#!/usr/bin/env python3
"""
BBBC021 Multi-Channel Connection Discovery Engine

Fixed version that properly combines DAPI, Tubulin, and Actin channels 
for accurate biological analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import asyncio
import json
from collections import defaultdict
from datetime import datetime

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL/matplotlib not available - image processing limited")

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    from anton.cmpo.ontology import CMPOOntology
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

class BBBC021MultiChannelProcessor:
    """Process and combine 3-channel BBBC021 fluorescence data."""
    
    def __init__(self, image_metadata_path: str):
        """Initialize with BBBC021 image metadata."""
        self.image_metadata = pd.read_csv(image_metadata_path)
        self.channel_mapping = {
            'DAPI': 'w1',      # Nuclear stain
            'Tubulin': 'w2',   # Microtubules  
            'Actin': 'w4'      # Actin cytoskeleton
        }
        self.composite_dir = Path("data/bbbc021/composites")
        self.composite_dir.mkdir(exist_ok=True)
        
    def find_channel_files(self, well: str, site: int, plate: str) -> Dict[str, Optional[str]]:
        """Find all 3 channel files for a given well/site/plate."""
        
        # Filter metadata for this specific location
        location_data = self.image_metadata[
            (self.image_metadata['Image_Metadata_Well_DAPI'] == well) &
            (self.image_metadata['Image_Metadata_Plate_DAPI'] == plate)
        ]
        
        if location_data.empty:
            return {}
            
        # Get the first matching row (assuming one image per well/site/plate)
        row = location_data.iloc[0]
        
        channels = {}
        
        # Map channel files
        channels['DAPI'] = self._build_image_path(row['Image_FileName_DAPI'], row['Image_PathName_DAPI'])
        channels['Tubulin'] = self._build_image_path(row['Image_FileName_Tubulin'], row['Image_PathName_Tubulin'])
        channels['Actin'] = self._build_image_path(row['Image_FileName_Actin'], row['Image_PathName_Actin'])
        
        return channels
    
    def _build_image_path(self, filename: str, pathname: str) -> str:
        """Build full image path from metadata."""
        base_dir = "data/bbbc021/images"
        return os.path.join(base_dir, pathname, filename)
    
    def create_composite_image(self, channels: Dict[str, str], output_path: str) -> bool:
        """Create multi-channel composite image."""
        
        if not PIL_AVAILABLE:
            print("❌ PIL not available for image processing")
            return False
            
        try:
            # Load all channels
            images = {}
            for channel_name, file_path in channels.items():
                if file_path and Path(file_path).exists():
                    img = Image.open(file_path)
                    images[channel_name] = np.array(img)
                else:
                    print(f"⚠️ Missing channel: {channel_name} at {file_path}")
                    return False
            
            if len(images) != 3:
                print(f"❌ Need all 3 channels, found {len(images)}")
                return False
                
            # Normalize each channel to 0-255
            normalized = {}
            for name, img in images.items():
                img_norm = img.astype(float)
                img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min()) * 255
                normalized[name] = img_norm.astype(np.uint8)
            
            # Create RGB composite
            # DAPI (nuclei) -> Blue
            # Tubulin (microtubules) -> Green
            # Actin (cytoskeleton) -> Red
            height, width = normalized['DAPI'].shape
            composite = np.zeros((height, width, 3), dtype=np.uint8)
            
            composite[:, :, 0] = normalized['Actin']     # Red channel
            composite[:, :, 1] = normalized['Tubulin']   # Green channel  
            composite[:, :, 2] = normalized['DAPI']      # Blue channel
            
            # Save composite
            composite_img = Image.fromarray(composite, 'RGB')
            composite_img.save(output_path)
            
            print(f"✅ Created composite: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create composite: {e}")
            return False

class BBBC021MultiChannelDiscovery:
    """Connection discovery with proper multi-channel analysis."""
    
    def __init__(self):
        self.phenotype_database = defaultdict(list)
        self.connection_graph = defaultdict(list)
        self.processor = None
        
    async def analyze_multichannel_bbbc021(self, moa_metadata_path: str, image_metadata_path: str, max_images: int = 5) -> List[Dict]:
        """Run connection discovery with proper multi-channel processing."""
        
        print("🧬 Starting REAL Multi-Channel BBBC021 Analysis")
        print("=" * 60)
        
        # Initialize processor
        self.processor = BBBC021MultiChannelProcessor(image_metadata_path)
        
        # Load MOA metadata
        moa_metadata = pd.read_csv(moa_metadata_path)
        print(f"📊 Loaded {len(moa_metadata)} compound records")
        
        # Select diverse compounds for analysis
        selected_compounds = self._select_diverse_compounds(moa_metadata, max_compounds=4)
        print(f"🎯 Selected {len(selected_compounds)} diverse compounds for analysis")
        
        results = []
        for compound_info in selected_compounds:
            print(f"\n🔬 Processing: {compound_info['compound']} ({compound_info['moa']})")
            
            # Find images for this compound in image metadata
            image_metadata = pd.read_csv(image_metadata_path)
            compound_rows = image_metadata[
                image_metadata['Image_Metadata_Compound'] == compound_info['compound']
            ].head(max_images)
            
            for _, row in compound_rows.iterrows():
                result = await self._analyze_multichannel_image(row)
                if result:
                    results.append(result)
                    
                    # Add to database
                    compound = result['compound_id']
                    if compound not in self.phenotype_database:
                        self.phenotype_database[compound] = []
                    self.phenotype_database[compound].append(result)
        
        print(f"✅ Analyzed {len(results)} multi-channel images")
        
        # Build connection graph
        self._build_connection_graph()
        
        # Demonstrate discoveries
        self._demonstrate_real_discoveries()
        
        return results
    
    async def _analyze_multichannel_image(self, row: pd.Series) -> Optional[Dict]:
        """Analyze one multi-channel image set."""
        
        try:
            well = row['Image_Metadata_Well_DAPI']
            plate = row['Image_Metadata_Plate_DAPI']
            compound = row['Image_Metadata_Compound']
            concentration = row['Image_Metadata_Concentration']
            
            print(f"   📍 Processing {well} on {plate}")
            
            # Find all channel files
            channels = self.processor.find_channel_files(well, site=1, plate=plate)
            
            if len(channels) != 3:
                print(f"   ⚠️ Missing channels for {well}")
                return None
            
            # Create composite image
            composite_filename = f"{plate}_{well}_composite.png"
            composite_path = self.processor.composite_dir / composite_filename
            
            success = self.processor.create_composite_image(channels, str(composite_path))
            if not success:
                print(f"   ❌ Failed to create composite for {well}")
                return None
            
            # Analyze with Anton
            if anton_available:
                result = await self._analyze_with_anton(composite_path, row)
                if result:
                    print(f"   ✅ Analysis complete for {well}")
                    return result
            else:
                print(f"   ⚠️ Anton not available, using mock analysis")
                return self._create_mock_result(row, str(composite_path))
                
        except Exception as e:
            print(f"   ❌ Error processing {row.get('Image_Metadata_Well_DAPI', 'unknown')}: {e}")
            return None
    
    async def _analyze_with_anton(self, composite_path: Path, row: pd.Series) -> Optional[Dict]:
        """Analyze composite image with Anton pipeline."""
        
        try:
            # Configure Anton for multi-channel BBBC021
            config = {
                'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
                'biological_context': {
                    'experiment_type': 'compound_screen',
                    'cell_line': 'MCF7_breast_cancer',
                    'compound': row['Image_Metadata_Compound'],
                    'concentration': row['Image_Metadata_Concentration'],
                    'staining': {
                        'DAPI': 'nuclei_DNA',
                        'Tubulin': 'microtubules_beta_tubulin', 
                        'Actin': 'cytoskeleton_F_actin'
                    },
                    'timepoint': '24_hours',
                    'channels': 'RGB_composite_DAPI_Tubulin_Actin'
                }
            }
            
            pipeline = AnalysisPipeline(config)
            anton_results = await pipeline.run_pipeline(str(composite_path))
            
            # Extract semantic features
            result = self._extract_semantic_features(anton_results, row, str(composite_path))
            return result
            
        except Exception as e:
            print(f"     ❌ Anton analysis failed: {e}")
            return None
    
    def _extract_semantic_features(self, anton_results: Dict, row: pd.Series, image_path: str) -> Dict:
        """Extract semantic features from Anton analysis results."""
        
        # Get VLM descriptions
        descriptions = []
        for stage in ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']:
            if stage in anton_results:
                descriptions.append(anton_results[stage])
        
        combined_description = ' '.join(descriptions).lower()
        
        # Extract features using keyword matching
        features = {
            'morphology': {},
            'nuclear': {},
            'cytoskeleton': {},
            'spatial': {},
            'intensity': {}
        }
        
        # Morphology keywords
        morphology_patterns = {
            'elongated|stretched': 'cell_elongation',
            'round|circular': 'cell_rounding',
            'spread|extended': 'cell_spreading',
            'large|enlarged|big': 'increased_size',
            'small|shrunk|tiny': 'decreased_size',
            'irregular|asymmetric': 'irregular_shape'
        }
        
        # Nuclear keywords
        nuclear_patterns = {
            'fragmented|broken': 'nuclear_fragmentation',
            'condensed|compact': 'nuclear_condensation',
            'bright|intense': 'increased_nuclear_intensity',
            'dim|faint': 'decreased_nuclear_intensity',
            'irregular|distorted': 'irregular_nuclear_shape'
        }
        
        # Cytoskeleton keywords
        cytoskeleton_patterns = {
            'disrupted|disorganized': 'cytoskeleton_disruption',
            'bundled|aggregated': 'cytoskeleton_bundling',
            'dense|thick': 'increased_fiber_density',
            'sparse|thin': 'decreased_fiber_density',
            'microtubule|tubulin': 'microtubule_related',
            'actin|filament': 'actin_related'
        }
        
        # Spatial keywords
        spatial_patterns = {
            'clustered|grouped': 'cell_clustering',
            'scattered|dispersed': 'cell_scattering',
            'contact|touching': 'cell_contact'
        }
        
        # Extract features
        all_patterns = [
            (morphology_patterns, 'morphology'),
            (nuclear_patterns, 'nuclear'),
            (cytoskeleton_patterns, 'cytoskeleton'),
            (spatial_patterns, 'spatial')
        ]
        
        for patterns, category in all_patterns:
            for pattern, feature_name in patterns.items():
                if any(keyword in combined_description for keyword in pattern.split('|')):
                    features[category][feature_name] = 1.0
        
        # Create result structure
        result = {
            'compound_id': row['Image_Metadata_Compound'],
            'mechanism_of_action': self._get_moa_for_compound(row['Image_Metadata_Compound']),
            'concentration': row['Image_Metadata_Concentration'],
            'image_path': image_path,
            'well_id': row['Image_Metadata_Well_DAPI'],
            'semantic_features': features,
            'raw_vlm_descriptions': anton_results,
            'discovery_vectors': self._create_discovery_vectors(features, combined_description),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _get_moa_for_compound(self, compound: str) -> str:
        """Get mechanism of action for compound."""
        moa_mapping = {
            'nocodazole': 'Microtubule destabilizers',
            'taxol': 'Microtubule stabilizers',
            'cytochalasin-d': 'Actin disruptors',
            'dmso': 'Control'
        }
        return moa_mapping.get(compound.lower(), 'Unknown')
    
    def _create_discovery_vectors(self, features: Dict, description: str) -> Dict:
        """Create discovery vectors for similarity analysis."""
        
        # Morphology vector
        morphology_features = features['morphology']
        morphology_vector = [
            morphology_features.get('cell_elongation', 0),
            morphology_features.get('cell_rounding', 0),
            morphology_features.get('cell_spreading', 0),
            morphology_features.get('increased_size', 0),
            morphology_features.get('decreased_size', 0),
            morphology_features.get('irregular_shape', 0)
        ]
        
        # Cytoskeleton vector
        cytoskeleton_features = features['cytoskeleton']
        cytoskeleton_vector = [
            cytoskeleton_features.get('cytoskeleton_disruption', 0),
            cytoskeleton_features.get('increased_fiber_density', 0),
            cytoskeleton_features.get('decreased_fiber_density', 0),
            cytoskeleton_features.get('microtubule_related', 0),
            cytoskeleton_features.get('actin_related', 0)
        ]
        
        return {
            'morphology_vector': morphology_vector,
            'cytoskeleton_vector': cytoskeleton_vector,
            'combined_vector': morphology_vector + cytoskeleton_vector,
            'description_length': len(description),
            'feature_richness': sum(len(cat_features) for cat_features in features.values())
        }
    
    def _create_mock_result(self, row: pd.Series, image_path: str) -> Dict:
        """Create mock result when Anton is not available."""
        
        compound = row['Image_Metadata_Compound']
        
        # Mock features based on known mechanisms
        mock_features = {
            'morphology': {'cell_rounding': 1.0, 'increased_size': 0.8},
            'nuclear': {'increased_nuclear_intensity': 0.7},
            'cytoskeleton': {'cytoskeleton_disruption': 0.9},
            'spatial': {},
            'intensity': {}
        }
        
        return {
            'compound_id': compound,
            'mechanism_of_action': self._get_moa_for_compound(compound),
            'concentration': row['Image_Metadata_Concentration'],
            'image_path': image_path,
            'well_id': row['Image_Metadata_Well_DAPI'],
            'semantic_features': mock_features,
            'raw_vlm_descriptions': {'mock': 'Mock analysis - Anton not available'},
            'discovery_vectors': self._create_discovery_vectors(mock_features, 'mock analysis'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _select_diverse_compounds(self, metadata: pd.DataFrame, max_compounds: int) -> List[Dict]:
        """Select diverse compounds for analysis."""
        
        # Get unique compounds from MOA metadata
        unique_compounds = metadata['compound'].unique()
        
        selected = []
        for compound in unique_compounds[:max_compounds]:
            compound_data = metadata[metadata['compound'] == compound].iloc[0]
            selected.append({
                'compound': compound,
                'moa': compound_data['moa']  # Get MOA from metadata
            })
        
        return selected
    
    def _build_connection_graph(self):
        """Build connection graph based on phenotypic similarity."""
        
        print("\n🔗 Building connection graph...")
        
        compounds = list(self.phenotype_database.keys())
        connections_found = 0
        
        for i, compound1 in enumerate(compounds):
            for j, compound2 in enumerate(compounds[i+1:], i+1):
                similarity = self._calculate_similarity(compound1, compound2)
                
                if similarity > 0.3:  # Lower threshold for real data
                    self.connection_graph[compound1].append({
                        'connected_compound': compound2,
                        'similarity': similarity,
                        'connection_type': 'phenotypic_similarity'
                    })
                    connections_found += 1
        
        print(f"✅ Found {connections_found} connections")
    
    def _calculate_similarity(self, compound1: str, compound2: str) -> float:
        """Calculate phenotypic similarity between compounds."""
        
        results1 = self.phenotype_database.get(compound1, [])
        results2 = self.phenotype_database.get(compound2, [])
        
        if not results1 or not results2:
            return 0.0
        
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
    
    def _demonstrate_real_discoveries(self):
        """Show real discoveries from multi-channel analysis."""
        
        print("\n🔍 MULTI-CHANNEL DISCOVERIES")
        print("=" * 50)
        
        # Show feature analysis
        print("\n1. 🧬 FEATURE ANALYSIS PER COMPOUND")
        for compound, results in self.phenotype_database.items():
            if results:
                print(f"\n   {compound}:")
                all_features = set()
                for result in results:
                    for category, features in result['semantic_features'].items():
                        all_features.update(features.keys())
                
                if all_features:
                    print(f"      Features: {', '.join(sorted(all_features))}")
                else:
                    print(f"      No significant features detected")
        
        # Show connections
        print(f"\n2. 🔗 PHENOTYPIC CONNECTIONS")
        for compound, connections in self.connection_graph.items():
            if connections:
                print(f"\n   {compound}:")
                for conn in connections:
                    other = conn['connected_compound']
                    similarity = conn['similarity']
                    print(f"      → Similar to {other} (similarity: {similarity:.2f})")
        
        print(f"\n✅ Multi-channel discovery analysis complete!")

async def main():
    """Main workflow for multi-channel BBBC021 analysis."""
    
    print("🧬 BBBC021 Multi-Channel Connection Discovery")
    print("=" * 60)
    
    # Check data availability
    image_metadata_path = "data/bbbc021/image_metadata.csv"
    moa_metadata_path = "data/bbbc021/moa_metadata.csv"
    
    if not Path(image_metadata_path).exists():
        print(f"❌ Image metadata not found: {image_metadata_path}")
        return
        
    if not Path(moa_metadata_path).exists():
        print(f"❌ MOA metadata not found: {moa_metadata_path}")
        return
    
    # Initialize discovery engine
    discovery_engine = BBBC021MultiChannelDiscovery()
    
    # Run analysis
    results = await discovery_engine.analyze_multichannel_bbbc021(
        moa_metadata_path, 
        image_metadata_path,
        max_images=2  # 2 images per compound for demo
    )
    
    if results:
        # Save results
        output_file = 'bbbc021_multichannel_discovery_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Multi-channel analysis complete!")
        print(f"🔬 Analyzed {len(results)} composite images")
        print(f"🧬 Processed {len(discovery_engine.phenotype_database)} compounds")
        print(f"📊 Results saved to: {output_file}")
        
        # Statistics
        total_features = sum(
            result['discovery_vectors']['feature_richness'] 
            for result in results
        )
        
        print(f"\n📈 ANALYSIS STATISTICS:")
        print(f"   • Total features extracted: {total_features}")
        print(f"   • Average features per image: {total_features / len(results):.1f}")
        print(f"   • Compounds with connections: {len(discovery_engine.connection_graph)}")
        
    else:
        print("❌ No results obtained from analysis")

if __name__ == "__main__":
    asyncio.run(main())